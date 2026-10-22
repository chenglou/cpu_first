"""Whole-option search at free attention's width, block by block (the bounded version of census.py).

    ~/github/sotaku/venv/bin/python -m bridge.blocksearch                        # locally: a minute or two
    ~/github/sotaku/venv/bin/python -m modal run bridge/blocksearch.py           # the same on Modal, <= 16 CPUs

The menu ("free-like"): MENU's 7 rules plus every other relative offset with row -4..0 and column -3..3 that the
causal mask allows: 34 options per port, kind-agnostic, so 34^7 = 5.3e10 wirings in all, too many to enumerate. The
search enumerates whole options one block of interacting ports at a time, each block scored pass or fail by a lesson
that only reads that block (WIRING.md's way):
  stage 1  the shift and the subtraction (SHIFT.A, SUB.A, SUB.B, SUB.C: 34^4 = 1.3e6 combinations) on a grid of one
           link: the subtraction must borrow exactly when x < N, for x = each question's R0 and x = N (the
           select's ports do not reach the subtraction row, so they do not matter here)
  stage 2  for each stage-1 survivor, the select (SEL.A, SEL.B, SEL.C: 34^3 = 39,304) on the same grid: the link
           law (x stays x below N, N becomes 0), then every training answer of (3, 2, 1) on the full grid, then the
           two bigger sizes
The truth tables are given. Writes bridge/records/blocksearch.json: the counts at every stage and the wirings that
pass everything.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "grid.py", "net.py", "census.py", "blocksearch.py"):
    image = image.add_local_file(HERE / _f, f"/root/bridge/{_f}")
app = modal.App("bridge-blocksearch", image=image)


def options():
    from bridge import net as Nn
    extra = [f"rel{dr},{dc}" for dr in range(-4, 1) for dc in range(-3, 4)
             if (dr, dc) not in ((-1, 0), (-2, 0), (0, -1), (-1, -1)) and not (dr == 0 and dc >= 0)]
    return list(Nn.MENU) + extra


def setup():
    import numpy as np
    import torch
    from bridge import grid as G
    B, w, k = 3, 2, 1
    lay = G.Layout(B, w, k)
    data = G.Data(B, w, k)
    lay1 = G.Layout(B, w, 1, links=1)
    pairs = np.unique(np.stack([data.N, data.R0], 1), axis=0)
    Ns = np.concatenate([pairs[:, 0], np.unique(data.N)])
    xs = np.concatenate([pairs[:, 1], np.unique(data.N)])
    law = dict(N1=torch.from_numpy(G.digits(Ns, B, w)), v1=torch.from_numpy(G.digits(xs, B, w + 1)),
               t1=torch.from_numpy(G.digits(xs % Ns, B, w + 1)), lt1=torch.from_numpy((xs < Ns).astype(np.int64)),
               cells1=torch.tensor([lay1.cell(lay1.Rn - 1, j) for j in range(w + 1)]), top1=lay1.cell(lay1.Rn - 2, w))
    big = [(G.Layout(B, w, k + 1), G.Data(B, w, k + 1)), (G.Layout(B, w + 1, k + 1), G.Data(B, w + 1, k + 1))]
    return lay, data, lay1, law, big


def stage1(lo, hi, chunk=2048):
    """SHIFT.A, SUB.A, SUB.B, SUB.C combinations lo .. hi - 1 (base-34 digits, in that order) whose subtraction
    borrows exactly when x < N on the one-link grid. Returns the passing combinations."""
    import torch
    from bridge import census as Cs
    from bridge import net as Nn
    opts = options()
    n = len(opts)
    lay, data, lay1, law, _ = setup()
    sel_true = [Nn.MENU.index(Nn.TRUE_MENU[p]) for p in Cs.PORTS[4:]]
    place = torch.tensor([n ** (3 - j) for j in range(4)], dtype=torch.int64)
    keep = []
    for a in range(lo, hi, chunk):
        idx = torch.arange(a, min(a + chunk, hi), dtype=torch.int64)
        sub = (idx[:, None] // place[None]) % n
        combos = torch.cat([sub, torch.tensor(sel_true).expand(len(sub), 3)], 1)
        s1 = Cs.wiring_sources(lay1, opts, combos)
        _, b1 = Cs.simulate(lay1, law["N1"], law["v1"], *s1)
        ok = (b1[:, :, law["top1"]] == law["lt1"][None]).all(-1)
        keep += sub[ok].tolist()
    return keep


def stage2(subs):
    """For each stage-1 combination, every select (SEL.A, SEL.B, SEL.C): the link law on the one-link grid, then
    the training answers, then the bigger sizes. Returns counts and the wirings that pass everything."""
    import torch
    from bridge import census as Cs
    from bridge import grid as G
    from bridge import net as Nn
    opts = options()
    n = len(opts)
    lay, data, lay1, law, big = setup()
    sels = torch.tensor(list(itertools.product(range(n), repeat=3)), dtype=torch.int64)
    ans = torch.from_numpy(lay.answer)
    true = [Nn.MENU.index(Nn.TRUE_MENU[p]) for p in Cs.PORTS]
    out = dict(link=0, fit=0, cert=0, passing=[])
    for sub in subs:
        combos = torch.cat([torch.tensor(sub).expand(len(sels), 4), sels], 1)
        for a in range(0, len(combos), 4096):
            c = combos[a:a + 4096]
            d1, b1 = Cs.simulate(lay1, law["N1"], law["v1"], *Cs.wiring_sources(lay1, opts, c))
            link = (d1[:, :, law["cells1"]] == law["t1"][None]).all(-1).all(-1) & \
                   (b1[:, :, law["top1"]] == law["lt1"][None]).all(-1)
            out["link"] += int(link.sum())
            if not link.any():
                continue
            c = c[link]
            dig, _ = Cs.simulate(lay, data.Nd, data.vd, *Cs.wiring_sources(lay, opts, c))
            fit = (dig[:, :, ans] == data.yd[None]).all(-1).all(-1)
            out["fit"] += int(fit.sum())
            for i in torch.where(fit)[0].tolist():
                ci = c[i:i + 1]
                ok = True
                for L2, D2 in big:
                    d2, _ = Cs.simulate(L2, D2.Nd, D2.vd, *Cs.wiring_sources(L2, opts, ci))
                    ok &= bool((d2[:, :, torch.from_numpy(L2.answer)] == D2.yd[None]).all())
                out["cert"] += int(ok)
                out["passing"].append({"wiring": {f"{G.HEADS[h]}.{G.KINDS[kd]}": opts[o]
                                                  for (h, kd), o in zip(Cs.PORTS, ci[0].tolist())},
                                       "true": ci[0].tolist() == true, "certified": ok})
    return out


@app.function(cpu=4, memory=4096, timeout=3 * 3600, max_containers=16, retries=modal.Retries(max_retries=1))
def run_stage1(lo: int, hi: int) -> list:
    import sys
    sys.path.insert(0, "/root")
    import torch
    torch.set_num_threads(4)
    from bridge.blocksearch import stage1
    return stage1(lo, hi)


@app.function(cpu=4, memory=4096, timeout=3 * 3600, max_containers=16, retries=modal.Retries(max_retries=1))
def run_stage2(subs: list) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import torch
    torch.set_num_threads(4)
    from bridge.blocksearch import stage2
    t0 = time.time()
    r = stage2(subs)
    r["seconds"] = time.time() - t0
    return r


@app.local_entrypoint()
def main(parts: int = 16) -> None:
    import sys
    sys.path.insert(0, str(HERE.parent))
    t0 = time.time()
    n = len(options())
    total = n ** 4
    edges = [total * i // parts for i in range(parts + 1)]
    subs = [s for part in run_stage1.starmap([(edges[i], edges[i + 1]) for i in range(parts)]) for s in part]
    t1 = time.time()
    print(f"stage 1: {len(subs)} of {total} shift-and-subtraction blocks pass ({t1 - t0:.0f}s)", flush=True)
    chunks = [subs[i::parts] for i in range(parts) if subs[i::parts]]
    res = list(run_stage2.map(chunks))
    rec = {"options": n, "menu": options(), "stage1_blocks": total, "stage1_pass": len(subs),
           "stage2_select_blocks_per_survivor": n ** 3, "stage2_link_pass": sum(r["link"] for r in res),
           "stage2_fit": sum(r["fit"] for r in res), "stage2_cert": sum(r["cert"] for r in res),
           "passing": [x for r in res for x in r["passing"]], "stage1_survivors": subs,
           "seconds": time.time() - t0, "container_seconds_stage2": sum(r["seconds"] for r in res)}
    out = HERE / "records" / "blocksearch.json"
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("passing", "stage1_survivors", "menu")}),
          f"-> {out}", flush=True)
    for p in rec["passing"]:
        print(p, flush=True)


def local():
    """The same search on this machine (both stages take seconds to minutes at this size)."""
    import torch
    torch.set_num_threads(4)
    t0 = time.time()
    n = len(options())
    subs = stage1(0, n ** 4)
    t1 = time.time()
    print(f"stage 1: {len(subs)} of {n ** 4} shift-and-subtraction blocks pass ({t1 - t0:.0f}s)", flush=True)
    r = stage2(subs)
    rec = {"options": n, "menu": options(), "stage1_blocks": n ** 4, "stage1_pass": len(subs),
           "stage2_select_blocks_per_survivor": n ** 3, "stage2_link_pass": r["link"], "stage2_fit": r["fit"],
           "stage2_cert": r["cert"], "passing": r["passing"], "stage1_survivors": subs,
           "seconds_stage1": t1 - t0, "seconds": time.time() - t0}
    out = HERE / "records" / "blocksearch.json"
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("passing", "stage1_survivors", "menu")}),
          f"-> {out}", flush=True)
    for p in rec["passing"]:
        print(p, flush=True)


if __name__ == "__main__":
    local()
