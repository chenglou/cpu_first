"""Step 1: the tiny machine as one exact weight setting of the looped transformer, checked on every question.

    ~/github/sotaku/venv/bin/python -m bridge.compile

Three settings of the same generic code (`net.py`), each checked at several sizes on every (N, v):
- "fixed": the compiled attention patterns given, the rule's truth tables (the machine as it is);
- "menu": the menu heads with their logits set hard on the true options;
- "free": the most generic point: free relative-position attention whose scores are set to beta * (number of the
  compiled pattern's features the key matches), and the rule as a 2-layer ReLU MLP whose hidden units detect
  (kind, a, b, c) and whose output weights are beta_o times the truth tables.
For each: the hard path (argmax attention, snapped states) must equal the machine at every cell, and the soft path
(plain softmax attention, unsnapped states, the network as trained) must put almost all probability on the right
answer. Also checked: starting the scratch cells from random states changes nothing after D ticks.
Writes bridge/records/compile.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from bridge import grid as G
from bridge import net as Nn
from bridge.grid import NIN, SCRATCH, VIN, Z

HERE = Path(__file__).resolve().parent
# (B, w, k, sample): every (N, v) of the size, or a random sample of that many for the two biggest
SIZES = [(3, 2, 1, 0), (3, 2, 2, 0), (3, 2, 3, 0), (3, 2, 4, 0), (3, 3, 2, 0), (3, 3, 3, 0), (2, 3, 3, 0), (2, 4, 4, 0),
         (4, 2, 2, 0), (4, 2, 3, 0), (5, 2, 2, 0), (4, 3, 2, 0), (5, 3, 2, 20000), (3, 4, 3, 20000)]
CHUNK = 2048


# the compiled patterns of the input cells when the prompt is not re-injected (inject "first"): port A reads the
# cell itself (the copy rule then holds it); ports B and C read any constant zero
HOLD_FREE = {(0, kd): ((kd,), 0, 0, None) for kd in (Z, NIN, VIN)}
HOLD_FREE.update({(h, kd): ((Z,), None, None, None) for kd in (Z, NIN, VIN) for h in (1, 2)})


def _noise(t, sigma, gen):
    if sigma:
        t.add_(sigma * torch.randn(t.shape, generator=gen, dtype=t.dtype))


def set_free(net: Nn.Net, beta=20.0, noise=0.0, gen=None):
    """Scores = beta * (how many of the compiled pattern's features the key matches: its kind, row offset, column
    offset, top place), the same for every copy of a head; then Gaussian noise of std `noise` (Net2Net: the copies
    can diverge)."""
    blk = net.blocks[0]
    cfg = net.cfg
    pats = dict(Nn.TRUE_FREE)
    if cfg.inject == "first":
        pats.update(HOLD_FREE)
    cp = cfg.copies
    with torch.no_grad():
        if cfg.free == "factor":
            for t in (blk.U, blk.Rr, blk.Rc, blk.Tp):
                t.zero_()
            for (h, kq), (kinds, dr, dc, top) in pats.items():
                for j in range(cp):
                    hh = h * cp + j
                    for kk in kinds:
                        blk.U[hh, kq, kk] = beta
                    if dr is not None:
                        blk.Rr[hh, kq, dr + cfg.Rb] = beta
                    if dc is not None:
                        blk.Rc[hh, kq, dc + cfg.Cb] = beta
                    if top:
                        blk.Tp[hh, kq, 1] = beta
            for t in (blk.U, blk.Rr, blk.Rc, blk.Tp):
                _noise(t, noise, gen)
            return
        blk.Rel.zero_()
        blk.Tp.zero_()
        nb, nc = blk.Rel.shape[-2:]
        for (h, kq), (kinds, dr, dc, top) in pats.items():
            for j in range(cp):
                hh = h * cp + j
                for kk in kinds:
                    for b in range(nb):
                        for c in range(nc):
                            m = 1 + (dr is None or b == dr + cfg.Rb) + (dc is None or c == dc + cfg.Cb)
                            blk.Rel[hh, kq, kk, b, c] = beta * m
                    if top:
                        blk.Tp[hh, kq, kk, 1] = beta
        _noise(blk.Rel, noise, gen)
        _noise(blk.Tp, noise, gen)


def set_menu(net: Nn.Net, beta=30.0, noise=0.0, gen=None):
    blk = net.blocks[0]
    cp = net.cfg.copies
    with torch.no_grad():
        blk.menu.zero_()
        for (h, kq), opt in Nn.TRUE_MENU.items():
            for j in range(cp):
                blk.menu[h * cp + j, kq, Nn.MENU.index(opt)] = beta
        _noise(blk.menu, noise, gen)


def _units(net):
    """The compiled hidden units: (kind, a, b, c, output row [B + 2] in the digits layout); in inject "first" also
    the input kinds' copy units (kind, a, None, None, row)."""
    B = net.B
    T = G.truth_tables(B, net.dtype)
    out = []
    for i, kd in enumerate(SCRATCH):
        for a in range(B):
            for b in range(B):
                for c in range(2):
                    out.append((kd, a, b, c, T[i, a, b, c]))
    if net.cfg.inject == "first":
        for kd in (Z, NIN, VIN):
            for a in range(B):
                out.append((kd, a, None, None, T[0, a, 0, 0]))
    return out


def set_mlp(net: Nn.Net, beta_o=30.0, widen=1, noise=0.0, gen=None):
    """AND-detector hidden units, one per (kind, a, b, c), output weights beta_o times the truth tables (the digits
    format). widen > 1 is Net2Net's widening: every unit duplicated `widen` times with its outgoing weights divided
    by `widen`, so the function is unchanged; then Gaussian noise of std `noise` on both layers lets the copies
    diverge. Head copies read side by side ("concat") each get 1/copies of the port's input weights."""
    blk = net.blocks[0]
    B = net.B
    NK = Nn.NK
    cfg = net.cfg
    ncopy = cfg.copies if cfg.copy_mode == "concat" else 1
    per = 2 * B + 2                                                            # one copy's ports: A digit, B digit, C borrow
    units = _units(net)
    assert len(units) * widen <= blk.W1.shape[1], f"MLP needs {len(units) * widen} hidden units"
    with torch.no_grad():
        blk.W1.zero_()
        blk.b1.fill_(-1.0e3)
        blk.W2.zero_()
        blk.b2.zero_()
        u = 0
        for (kd, a, b, c, row) in units:
            for _ in range(widen):
                blk.W1[kd, u] = 1
                for j in range(ncopy):
                    base = NK + j * per
                    blk.W1[base + a, u] += 1 / ncopy
                    if b is not None:
                        blk.W1[base + B + b, u] += 1 / ncopy
                        blk.W1[base + 2 * B + c, u] += 1 / ncopy
                blk.b1[u] = -3.0 if b is not None else -1.0
                blk.W2[u] = beta_o * row / widen
                u += 1
        _noise(blk.W1, noise, gen)
        _noise(blk.W2, noise, gen)


def set_symbols(net: Nn.Net, beta=20.0, beta_o=30.0, noise=0.0, gen=None, widen=1):
    """The free-symbol format at the machine: K = 2B * lift symbols, symbol (d, b, j) = d + B b + 2B j standing for
    digit d with borrow b; the lift copies j carry the same meaning (a redundant embedding: every distribution
    spreads evenly over a meaning's copies). Inputs, the blank state, the heads' value projections, the MLP's AND
    detectors and the readout are set accordingly; the attention is set by set_free."""
    cfg = net.cfg
    B = net.B
    K = net.S
    assert K % (2 * B) == 0, "symbols format: K must be a multiple of 2B"
    lift = K // (2 * B)
    NK = Nn.NK
    syms = lambda d=None, b=None: [dd + B * bb + 2 * B * j for dd in range(B) for bb in range(2) for j in range(lift)
                                   if (d is None or dd == d) and (b is None or bb == b)]
    blk = net.blocks[0]
    units = _units(net)
    ncopy = cfg.copies if cfg.copy_mode == "concat" else 1
    with torch.no_grad():
        net.Ein.zero_()
        for ik in range(3):
            for d in range(B):
                net.Ein[ik, d, syms(d, 0)] = beta
        net.s0.zero_()
        net.s0[syms(0, 0)] = beta
        blk.Wv.copy_(beta * torch.eye(K, dtype=net.dtype).expand_as(blk.Wv))
        net.Wout.zero_()
        for d in range(B):
            net.Wout[syms(d), d] = beta
        assert len(units) * widen <= blk.W1.shape[1]
        blk.W1.zero_()
        blk.b1.fill_(-1.0e3)
        blk.W2.zero_()
        blk.b2.zero_()
        u = 0
        for (kd, a, b, c, row) in units:
            for _ in range(widen):                                             # Net2Net copies, as in set_mlp
                blk.W1[kd, u] = 1
                for j in range(ncopy):
                    base = NK + j * 3 * K
                    blk.W1[base + torch.tensor(syms(a)), u] += 1 / ncopy
                    if b is not None:
                        blk.W1[base + K + torch.tensor(syms(b)), u] += 1 / ncopy
                        blk.W1[base + 2 * K + torch.tensor(syms(None, c)), u] += 1 / ncopy
                blk.b1[u] = -3.0 if b is not None else -1.0
                od, ob = int(row[:B].argmax()), int(row[B:].argmax())
                blk.W2[u, syms(od, ob)] = beta_o / widen
                u += 1
        for t in (net.Ein, net.s0, blk.Wv, net.Wout, blk.W1, blk.W2):
            _noise(t, noise, gen)


def build(setting: str, B: int) -> Nn.Net:
    if setting == "fixed":
        cfg = Nn.Cfg(attn="fixed", rule="table", learn=())
    elif setting == "menu":
        cfg = Nn.Cfg(attn="menu", rule="table", learn=())
    elif setting == "free":
        cfg = Nn.Cfg(attn="free", rule="mlp")
    else:
        raise ValueError(setting)
    net = Nn.Net(cfg, B)
    if setting == "menu":
        set_menu(net)
    if setting == "free":
        set_free(net)
        set_mlp(net)
    return net


@torch.no_grad()
def check(net: Nn.Net, B, w, k, sample=0, rand_init=False, seed=0):
    """Every question of the size in chunks; returns the worst chunk's numbers (exact shares are pooled)."""
    lay = G.Layout(B, w, k)
    data = G.Data(B, w, k, sample=sample, seed=1)
    parts = [check_chunk(net, lay, data.subset(np.arange(i, min(i + CHUNK, data.X))), rand_init, seed + i)
             for i in range(0, data.X, CHUNK)]
    out = {}
    for path in ("hard", "soft"):
        n = [p[path]["n"] for p in parts]
        out[path] = {key: (float(sum(p[path][key] * m for p, m in zip(parts, n)) / sum(n)) if key != "min_p_answer"
                           else min(p[path][key] for p in parts)) for key in ("exact", "cells_digit", "cells_borrow",
                                                                              "min_p_answer")}
    out.update(questions=data.X, cells=lay.L, ticks=net.ticks(lay), sampled=bool(sample))
    return out


def check_chunk(net, lay, data, rand_init, seed):
    B = lay.B
    S_in = net.inputs(lay, G.onehot(data.Nd, B), G.onehot(data.vd, B))
    if rand_init:
        g = torch.Generator().manual_seed(seed)
        R = torch.rand(S_in.shape, generator=g, dtype=S_in.dtype)
        R = torch.cat([R[..., :B] / R[..., :B].sum(-1, keepdim=True), R[..., B:] / R[..., B:].sum(-1, keepdim=True)], -1)
        S_in = torch.where(torch.from_numpy(lay.is_input)[None, :, None], S_in, R)
    out = {}
    dig, bor = G.machine_reference(lay, data)
    for path in ("hard", "soft"):
        S = net(lay, S_in, hard=(path == "hard"))
        ans = net.readout(lay, S)
        exact = (ans.argmax(-1) == data.yd).all(-1).float().mean().item()
        scr = ~lay.is_input
        cell_dig = (S[..., :B].argmax(-1).numpy()[:, scr] == dig[:, scr]).mean()
        cell_bor = (S[..., B:].argmax(-1).numpy()[:, scr] == bor[:, scr]).mean()
        pmin = ans.gather(-1, data.yd[..., None]).min().item()
        out[path] = dict(exact=exact, cells_digit=float(cell_dig), cells_borrow=float(cell_bor), min_p_answer=pmin,
                         n=data.X)
    assert (dig[:, lay.answer] == data.yd.numpy()).all(), "machine reference disagrees with v mod N"
    return out


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    lay = G.Layout(3, 2, 2)
    print("The grid at (B, w, k) = (3, 2, 2): rows top to bottom, columns -1 .. P-1 (N0..: N's digits, v0..: v's)\n")
    print(lay.describe())
    print(f"\n{lay.L} cells, depth D = {lay.depth} ticks\n")
    rec = {"sizes": {}, "settings": ("fixed", "menu", "free")}
    for (B, w, k, sample) in SIZES:
        row = {}
        for setting in rec["settings"]:
            net = build(setting, B)
            r = check(net, B, w, k, sample)
            r2 = check(net, B, w, k, sample, rand_init=True, seed=B * 100 + w * 10 + k)
            r["hard_from_random_scratch"] = r2["hard"]["exact"]
            r["soft_from_random_scratch"] = r2["soft"]["exact"]
            row[setting] = r
        rec["sizes"][f"{B},{w},{k}"] = row
        f = row["free"]
        print(f"({B},{w},{k}) {f['questions']:>6} questions, {f['cells']:>3} cells, {f['ticks']:>2} ticks | "
              + " | ".join(f"{s}: hard {row[s]['hard']['exact']:.3f} cells {row[s]['hard']['cells_digit']:.3f}/"
                           f"{row[s]['hard']['cells_borrow']:.3f} soft {row[s]['soft']['exact']:.3f} "
                           f"(min p {row[s]['soft']['min_p_answer']:.6f}) rand-start {row[s]['hard_from_random_scratch']:.3f}"
                           for s in rec["settings"]), flush=True)
    net = build("free", 3)
    rec["free_param_count"] = sum(p.numel() for p in net.parameters())
    rec["mlp_hidden"] = net.blocks[0].W1.shape[1]
    rec["seconds"] = time.time() - t0
    out = HERE / "records" / "compile.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(f"\nfree setting: {rec['free_param_count']} parameters at B = 3 (MLP hidden {rec['mlp_hidden']}); "
          f"{rec['seconds']:.1f} s; wrote {out}")


if __name__ == "__main__":
    main()
