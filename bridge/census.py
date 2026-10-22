"""Every wiring of a menu, run as the hard network with the truth tables: which fit the answers, which pass the
one-link law, which certify. Is learning the menu a search problem or an identification problem?

    ~/github/sotaku/venv/bin/python -m bridge.census [extra]      # menu "rel" with `extra` extra offsets (0 = the 7 rules)

The seven ports a kind uses (SHIFT.A, SUB.A, SUB.B, SUB.C, SEL.A, SEL.B, SEL.C) each take any option of the menu;
SHIFT's B and C read a zero (the copy ignores them). Each wiring runs exactly as the network's hard path does: D
synchronous ticks from blank scratch cells, inputs re-injected, the truth rules. Counted at (3, 2, 1):
  fit    every training answer right (99 questions)
  link   the one-link law: on a grid of one link, x = R0 (below N) stays x and N becomes 0, and the link's
         subtraction borrows exactly when x < N
  both   fit and link
  cert   both, and exact on every question of (3, 2, 2) and (3, 3, 2)
Writes bridge/records/census_<options>.json with the counts and the wirings that pass both.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bridge import grid as G
from bridge import net as Nn
from bridge.grid import NIN, SEL, SHIFT, SUB, VIN, Z

HERE = Path(__file__).resolve().parent
PORTS = [(0, SHIFT), (0, SUB), (1, SUB), (2, SUB), (0, SEL), (1, SEL), (2, SEL)]


def sources(lay, opts):
    """[len(opts), L] source cell of every cell under each option (scratch cells; inputs map to themselves)."""
    out = np.tile(np.arange(lay.L), (len(opts), 1))
    for i, o in enumerate(opts):
        for p in np.where(~lay.is_input)[0]:
            out[i, p] = Nn.menu_source(lay, p, o)
    return torch.from_numpy(out)


def simulate(lay, Nd, vd, src_a, src_b, src_c, ticks=None):
    """Hard network for W wirings at once: src_* [W, L] source cells per head. Returns digits [W, X, L] and borrows
    [W, X, L] (int8)."""
    B = lay.B
    W, L = src_a.shape
    X = Nd.shape[0]
    dig = torch.zeros(W, X, L, dtype=torch.int8)
    bor = torch.zeros(W, X, L, dtype=torch.int8)
    nin = torch.from_numpy(np.where(lay.kind == NIN)[0])
    vin = torch.from_numpy(np.where(lay.kind == VIN)[0])
    inp_d = torch.zeros(X, L, dtype=torch.int8)
    inp_d[:, nin] = Nd[:, torch.from_numpy(lay.src[nin.numpy()])].to(torch.int8)
    inp_d[:, vin] = vd[:, torch.from_numpy(lay.src[vin.numpy()])].to(torch.int8)
    is_in = torch.from_numpy(lay.is_input)
    kind = torch.from_numpy(lay.kind)
    dig[:] = inp_d
    g = lambda t, s: torch.gather(t, 2, s[:, None, :].expand(W, X, L))
    for _ in range(ticks or lay.depth):
        a, b, c = g(dig, src_a), g(dig, src_b), g(bor, src_c)
        x = a - b - c
        nd = torch.where(kind == SHIFT, a, torch.where(kind == SUB, x % B, torch.where(c == 1, a, b)))
        nb = torch.where(kind == SUB, (x < 0).to(torch.int8), torch.zeros_like(x))
        dig = torch.where(is_in, inp_d, nd)
        bor = torch.where(is_in, torch.zeros_like(nb), nb)
    return dig, bor


def wiring_sources(lay, opts, combos):
    """combos [W, 7] option indices -> src_a, src_b, src_c [W, L]."""
    S = sources(lay, opts)                                                      # [n_opts, L]
    W = combos.shape[0]
    zero = torch.zeros(lay.L, dtype=torch.int64)
    kind = torch.from_numpy(lay.kind)
    src = [torch.arange(lay.L).expand(W, lay.L).clone() for _ in range(3)]
    for j, (h, kd) in enumerate(PORTS):
        m = kind == kd
        src[h][:, m] = S[combos[:, j]][:, m]
    for h in (1, 2):                                                            # SHIFT's unused ports: a zero
        src[h][:, kind == SHIFT] = 0
    return src


def setup(extra):
    """Everything a range of wirings needs: the menu, the layouts, the questions and the one-link law's rows."""
    cfg = Nn.Cfg(attn="menu", menu="rel", menu_extra=extra)
    opts = Nn.menu_options(cfg)
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
    return opts, lay, data, lay1, law, big


def run_range(extra, lo, hi, chunk=1024, log=None):
    """Counts (and the wirings that pass both) over wiring indices lo .. hi - 1, in itertools.product's order."""
    opts, lay, data, lay1, law, big = setup(extra)
    n = len(opts)
    true = [Nn.MENU.index(Nn.TRUE_MENU[p]) for p in PORTS]
    counts = dict(wirings=hi - lo, fit=0, link=0, both=0, cert=0)
    passing = []
    place = torch.tensor([n ** (len(PORTS) - 1 - j) for j in range(len(PORTS))], dtype=torch.int64)
    t0 = time.time()
    ans = torch.from_numpy(lay.answer)
    pre = torch.from_numpy(np.random.default_rng(0).choice(data.X, min(16, data.X), replace=False))
    for a in range(lo, hi, chunk):
        idx = torch.arange(a, min(a + chunk, hi), dtype=torch.int64)
        combos = (idx[:, None] // place[None]) % n
        sa, sb, sc = wiring_sources(lay, opts, combos)
        # staged: 16 questions first, every question only for the wirings that get those right (same counts)
        dig, _ = simulate(lay, data.Nd[pre], data.vd[pre], sa, sb, sc)
        fit = (dig[:, :, ans] == data.yd[pre][None]).all(-1).all(-1)
        if fit.any():
            w = torch.where(fit)[0]
            dig, _ = simulate(lay, data.Nd, data.vd, sa[w], sb[w], sc[w])
            fit[w] = (dig[:, :, ans] == data.yd[None]).all(-1).all(-1)
        s1 = wiring_sources(lay1, opts, combos)
        m = 8                                                                   # the law's first rows, then all
        d1, b1 = simulate(lay1, law["N1"][:m], law["v1"][:m], *s1)
        link = (d1[:, :, law["cells1"]] == law["t1"][:m][None]).all(-1).all(-1) & \
               (b1[:, :, law["top1"]] == law["lt1"][:m][None]).all(-1)
        if link.any():
            w = torch.where(link)[0]
            d1, b1 = simulate(lay1, law["N1"], law["v1"], *(x[w] for x in s1))
            link[w] = (d1[:, :, law["cells1"]] == law["t1"][None]).all(-1).all(-1) & \
                      (b1[:, :, law["top1"]] == law["lt1"][None]).all(-1)
        both = fit & link
        counts["fit"] += int(fit.sum())
        counts["link"] += int(link.sum())
        counts["both"] += int(both.sum())
        for i in torch.where(both)[0].tolist():
            cmb = combos[i:i + 1]
            ok = True
            for L2, D2 in big:
                d2, _ = simulate(L2, D2.Nd, D2.vd, *wiring_sources(L2, opts, cmb))
                ok &= bool((d2[:, :, torch.from_numpy(L2.answer)] == D2.yd[None]).all())
            counts["cert"] += int(ok)
            passing.append({"wiring": {f"{G.HEADS[h]}.{G.KINDS[kd]}": opts[o]
                                       for (h, kd), o in zip(PORTS, cmb[0].tolist())},
                            "true": cmb[0].tolist() == true, "certified": ok})
        if log and (a - lo) % (chunk * 200) == 0:
            log(f"{a + chunk - lo}/{hi - lo} {time.time() - t0:.0f}s {counts}")
    return {"counts": counts, "passing_both": passing, "options": n, "menu": list(opts)}


def merge(parts):
    out = dict(parts[0])
    out["counts"] = {k: sum(p["counts"][k] for p in parts) for k in parts[0]["counts"]}
    out["passing_both"] = [x for p in parts for x in p["passing_both"]]
    return out


def main():
    torch.set_num_threads(4)
    extra = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    t0 = time.time()
    n = 7 + extra
    rec = run_range(extra, 0, n ** len(PORTS), log=lambda m: print(m, flush=True))
    rec["seconds"] = time.time() - t0
    out = HERE / "records" / f"census_{n}.json"
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["counts"]), f"-> {out}")


if __name__ == "__main__":
    main()
