"""The compile check: the bridge's division machine as one exact NCA rule, checked on every question.

    ~/github/sotaku/venv/bin/python -m nca.compile               # every size, locally (about 45 minutes)
    ~/github/sotaku/venv/bin/python -m nca.compile assemble      # or: from the per-size Modal jobs (suite compile)

The rule (`grid.compiled_clauses`) becomes the NCA's own MLP (`model.NCA`): one ReLU unit per AND clause (the cell's
kind and up to three neighbour values, or a neighbour being the edge), its output weights beta on the values the
clause sets. That is an ordinary weight setting of the generic 3 x 3 rule, nothing outside it. For each size, every
(N, v) (a random sample for the biggest):
- the hard path (argmax states) must hold the machine's value (grid.machine_reference) in every channel of every
  cell after `depth` ticks, from blank scratch cells and from random ones, and still after twice as many ticks;
- the soft path (softmax states, the network as trained) must put almost all probability on the right answer;
- the same with the prompt written only at tick 0 (the rule then holds the prompt itself).
Writes nca/records/compile.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from nca import grid as G
from nca import model as M
from nca.grid import OFFS, POS

HERE = Path(__file__).resolve().parent
SIZES = [(3, 2, 1, 0), (3, 2, 2, 0), (3, 2, 3, 0), (3, 2, 4, 0), (3, 3, 2, 0), (3, 3, 3, 0), (2, 3, 3, 0), (2, 4, 4, 0),
         (4, 2, 2, 0), (4, 2, 3, 0), (5, 2, 2, 0), (4, 3, 2, 0), (5, 3, 2, 20000), (3, 4, 3, 20000)]
CHUNK = 2048


def set_compiled(net: M.NCA, B: int, beta: float = 20.0, noise: float = 0.0, gen=None, widen: int = 1,
                 task: str = "div"):
    """Write the compiled rule into `net` (hidden units beyond the clauses never fire). `widen` duplicates every
    unit that many times with its output weights divided accordingly (Net2Net). The clauses follow the net's reach
    (model.Cfg.extra): what it reads beyond the 3 x 3."""
    f = net.fmt
    NK = net.nk
    nin = NK + f.D
    reach = net.cfg.extra or "3x3"
    cl = G.compiled_clauses(B, task, reach, tuple(n for n, _ in f.chans))
    Wx = torch.zeros_like(net.Wx) if net.nextra else None
    H = net.cfg.hidden
    assert H >= widen * len(cl), f"the rule needs {widen * len(cl)} hidden units, has {H}"
    W1 = torch.zeros_like(net.W1)
    b1 = torch.full_like(net.b1, -1.0)
    W2 = torch.zeros_like(net.W2)
    for u0, (kd, conds, kconds, outs) in enumerate(cl):
        for rep in range(widen):
            u = u0 * widen + rep
            W1[u, kd, 1, 1] += 1                                       # the cell's own kind
            neg = 0
            for pos, ch, val, positive in conds:
                if pos in G.EXTRA_POS:
                    Wx[u, G.EXTRA_POS[pos] * nin + NK + f.off[ch] + val] += 1 if positive else -1
                else:
                    dr, dc = OFFS[POS[pos]]
                    W1[u, NK + f.off[ch] + val, 1 + dr, 1 + dc] += 1 if positive else -1
                neg += not positive
            for pos, kind, positive in kconds:
                dr, dc = OFFS[POS[pos]]
                W1[u, kind, 1 + dr, 1 + dc] += 1 if positive else -1
                neg += not positive
            n = 1 + len(conds) + len(kconds)
            b1[u] = -(n - 1) + neg
            for ch, val in outs.items():
                W2[f.off[ch] + val, u] = beta / widen
    with torch.no_grad():
        net.W1.copy_(W1)
        net.b1.copy_(b1)
        net.W2.copy_(W2)
        net.b2.zero_()
        if Wx is not None:
            net.Wx.copy_(Wx)
        if noise:
            for p in [net.W1, net.b1, net.W2, net.b2] + ([net.Wx] if Wx is not None else []):
                p.add_(noise * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    return len(cl) * widen


def compiled_net(B, clamp=True, dtype=torch.float64, beta=20.0, hidden=None, task="div", reach="3x3"):
    fmt = G.compiled_fmt(B, task, minimal=reach == "ports")
    nunits = len(G.compiled_clauses(B, task, reach, tuple(n for n, _ in fmt.chans)))
    net = M.NCA(fmt, M.Cfg(hidden=hidden or nunits, clamp=clamp, kindset="sq" if task == "sq" else "div",
                           extra="" if reach == "3x3" else reach), dtype=dtype)
    set_compiled(net, B, beta, task=task)
    return net


@torch.no_grad()
def check_size(B, w, k, sample=0, clamp=True, beta=20.0, task="div", reach="3x3"):
    lay = G.Layout(B, w, k, task=task)
    data = G.SqData(B, w, sample=sample, seed=1) if task == "sq" else G.Data(B, w, k, sample=sample, seed=1)
    net = compiled_net(B, clamp=clamp, beta=beta, task=task, reach=reach)
    f = net.fmt
    chans = [n for n, _ in f.chans]
    gen = torch.Generator().manual_seed(5)
    D = lay.depth
    out = {"questions": int(data.X), "cells": lay.R * lay.C, "rows": lay.R, "places": lay.C, "ticks": D,
           "hidden_units": net.cfg.hidden}
    exact = {"blank": True, "random": True, "random_2x": True}
    soft_ok = True
    minp = 1.0
    early = 0
    for i in range(0, data.X, CHUNK):
        d = data.subset(np.arange(i, min(i + CHUNK, data.X)))
        ref = G.machine_reference(lay, d, reach)
        for name, scratch, ticks in (("blank", "blank", D), ("random", "random", D), ("random_2x", "random", 2 * D)):
            S0 = net.prompt_state(lay, d.Nd, d.vd, scratch, gen)
            S, _ = net.run(lay, S0, ticks, mode="hard")
            for ch in chans:
                got = net.channel(S, ch).argmax(1).numpy()
                if not (got == ref[ch]).all():
                    exact[name] = False
        S0 = net.prompt_state(lay, d.Nd, d.vd, "random", gen)                    # one tick short, from random
        S, _ = net.run(lay, S0, D - 1, mode="hard")
        early += int((net.channel(S, "d").argmax(1).numpy() != ref["d"]).any(axis=(1, 2)).sum())
        for scratch in ("blank", "noise"):
            S0 = net.prompt_state(lay, d.Nd, d.vd, scratch, gen)
            S, _ = net.run(lay, S0, D, mode="soft")
            r, places = lay.answer
            pa = net.channel(S, "d")[:, :, r][:, :, places].permute(0, 2, 1)            # [X, w, B]
            yd = torch.from_numpy(d.yd)
            p = pa.gather(-1, yd[..., None]).squeeze(-1)
            minp = min(minp, float(p.min()))
            for ch in chans:
                if not (net.channel(S, ch).argmax(1).numpy() == ref[ch]).all():
                    soft_ok = False
        if not (ref["d"][:, lay.answer[0]][:, lay.answer[1]] == d.yd).all():
            raise AssertionError("the reference machine is wrong")
    out.update(hard_exact=exact, soft_every_cell=soft_ok, soft_min_p=minp, wrong_one_tick_short=early)
    # the fewest ticks after which every answer is right from random scratch states (the depth bounds every cell)
    d = data.subset(np.arange(min(data.X, CHUNK)))
    yd = d.yd
    t_ans = D
    for t in range(D, 0, -1):
        S0 = net.prompt_state(lay, d.Nd, d.vd, "random", torch.Generator().manual_seed(9))
        S, _ = net.run(lay, S0, t, mode="hard")
        r, places = lay.answer
        got = net.channel(S, "d")[:, :, r][:, :, places].argmax(1).numpy()
        if (got == yd).all():
            t_ans = t
        else:
            break
    out["answer_ticks"] = t_ans
    return out


SIZES_SQ = [(3, 2, 0), (3, 3, 0), (3, 4, 0), (2, 3, 0), (2, 4, 0), (2, 5, 0), (4, 2, 0), (4, 3, 0), (5, 2, 0),
            (5, 3, 0), (3, 5, 20000)]


def main():
    t0 = time.time()
    rec = {"what": "the bridge's division machine, and x^2 mod N, as one exact NCA rule each (3x3 neighbourhood, one "
                   "ReLU MLP)", "beta": 20.0, "sizes": {}, "square": {}}
    for B, w, k, sample in SIZES:
        for clamp in (True, False):
            key = f"{B},{w},{k}" + ("" if clamp else " tick0")
            t1 = time.time()
            r = check_size(B, w, k, sample, clamp=clamp)
            r["seconds"] = round(time.time() - t1, 1)
            rec["sizes"][key] = r
            print(key, r, flush=True)
    for B, w, sample in SIZES_SQ:
        key = f"{B},{w}"
        t1 = time.time()
        r = check_size(B, w, w, sample, task="sq")
        r["seconds"] = round(time.time() - t1, 1)
        rec["square"][key] = r
        print("square", key, r, flush=True)
    rec["layout_sq_3_2"] = G.Layout(3, 2, 2, task="sq").describe()
    rec["seconds"] = round(time.time() - t0, 1)
    rec["layout_3_2_1"] = G.Layout(3, 2, 1).describe()
    (HERE / "records").mkdir(exist_ok=True)
    (HERE / "records" / "compile.json").write_text(json.dumps(rec, indent=1))
    ok = all(v["hard_exact"]["blank"] and v["hard_exact"]["random"] and v["hard_exact"]["random_2x"] and
             v["soft_every_cell"] for v in list(rec["sizes"].values()) + list(rec["square"].values()))
    print("ALL EXACT" if ok else "NOT EXACT", f"{rec['seconds']}s")


def assemble():
    """records/compile.json from the per-size Modal jobs (suites.compile_all, runs/compile.jsonl)."""
    rows = [json.loads(l) for l in open(HERE / "runs" / "compile.jsonl")]
    rec = {"what": "the bridge's division machine, and x^2 mod N, as one exact NCA rule each (3x3 neighbourhood, one "
                   "ReLU MLP), and with the longer reaches ncol and ports", "beta": 20.0,
           "sizes": {r["name"]: dict(r["check"], job={k: v for k, v in r["job"].items() if k not in ("suite",)})
                     for r in sorted(rows, key=lambda r: r["name"])}}
    rec["all_exact"] = all(all(v["hard_exact"].values()) and v["soft_every_cell"] for v in rec["sizes"].values())
    rec["layout_div_3_2_1"] = G.Layout(3, 2, 1).describe()
    rec["layout_sq_3_2"] = G.Layout(3, 2, 2, task="sq").describe()
    (HERE / "records" / "compile.json").write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["assemble"]:
        r = assemble()
        print(len(r["sizes"]), "sizes;", "ALL EXACT" if r["all_exact"] else "NOT EXACT")
    else:
        main()
