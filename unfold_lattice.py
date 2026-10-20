"""Analytic unfolding (UNFOLD.md section 5): integer ambiguity resolution by lattice reduction.

For a hypothesis linear in integer coefficients, V = c . phi(s), every row says c . phi(s_i) - y_i = k_i N_i with an
unknown integer wrap count k_i. The smallest c and all the k_i come out of one LLL reduction of Kannan's embedding,
the way GPS receivers resolve carrier-phase cycle counts. No curriculum, no easy rows, no gradient.

    python unfold_lattice.py --feats value --rows 2 3 5 --seeds 2
    python unfold_lattice.py --feats digits --h1_bits 16 --rows 40 --seeds 1

Uses python-flint's LLL when it is installed (fast), else sympy's (pure Python: fine up to ~60 dimensions).
Imports no torch, so it runs in a plain numpy + python-flint environment.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import h1


def lll(rows):
    try:
        import flint
        red = flint.fmpz_mat(rows).lll()
        return [[int(red[i, j]) for j in range(red.ncols())] for i in range(red.nrows())]
    except ImportError:
        from sympy import Matrix
        red = Matrix(rows).lll()
        return [[int(x) for x in red.row(i)] for i in range(red.rows)]


def lattice_solve(F, N, Y, M=2 ** 64, Q=1):
    """Smallest integer c with F c = Y (mod N) row by row. Basis rows [e_j | M F[:, j] | 0], [0 | M N_i e_i | 0],
    [0 | -M Y | Q]; a reduced vector with a zero middle block and last entry +-Q carries the solution."""
    m, p = F.shape
    rows = [[1 if k == j else 0 for k in range(p)] + [M * int(F[i, j]) for i in range(m)] + [0] for j in range(p)]
    rows += [[0] * p + [M * int(N[i]) if k == i else 0 for k in range(m)] + [0] for i in range(m)]
    rows.append([0] * p + [-M * int(Y[i]) for i in range(m)] + [Q])
    for v in lll(rows):
        if all(x == 0 for x in v[p:p + m]) and abs(v[-1]) == Q:
            sgn = 1 if v[-1] == Q else -1
            return [sgn * x for x in v[:p]]
    return None


def features(feats, X, nd):
    if feats == "value":
        return np.stack([np.ones_like(X), X, X * X], 1).astype(object)
    d = h1.digits(X, nd)
    cols = [np.ones_like(X)] + [d[:, i] for i in range(nd)] + [d[:, i] * d[:, j] for i in range(nd) for j in range(i, nd)]
    return np.stack(cols, 1).astype(object)


def target(feats, nd):
    if feats == "value":
        return [0, 0, 1]
    return [0] + [0] * nd + [10 ** (i + j) * (1 if i == j else 2) for i in range(nd) for j in range(i, nd)]


def run(feats_list, rows_list, seeds, h1_bits=0, log=None):
    """Random T=1 rows of H1's training set (optionally one modulus size); report whether c is exactly s^2."""
    tr = h1.load("train")
    t1 = tr["T"] == 1
    if h1_bits:
        t1 &= tr["bits"] == h1_bits
    Nh, Xh, Yh = tr["N"][t1], tr["X"][t1], tr["Y"][t1]
    nd = int(np.max(np.floor(np.log10(np.maximum(Xh, 1))) + 1))
    rng = np.random.default_rng(0)
    out = []
    for feats in feats_list:
        for m in rows_list:
            for trial in range(seeds):
                idx = rng.choice(len(Nh), m, replace=False)          # random rows: almost all wrap many times
                N, X, Y = Nh[idx], Xh[idx], Yh[idx]
                t0 = time.time()
                c = lattice_solve(features(feats, X, nd), N.astype(object), Y.astype(object))
                sec = time.time() - t0
                exact = None
                if c is not None:
                    pred = features(feats, Xh, nd).dot(np.array(c, dtype=object)) % Nh.astype(object)
                    exact = float(np.mean(pred.astype(np.int64) == Yh))
                r = {"feats": feats, "h1_bits": h1_bits, "rows": m, "trial": trial, "coefficients": len(target(feats, nd)),
                     "found": c is not None, "is_s2": c == target(feats, nd) if c is not None else False,
                     "exact_all_t1_rows_this_size": exact, "n_t1_rows": int(len(Nh)), "sec": sec,
                     "min_wraps_in_rows": int(min(X.astype(object) ** 2 // N.astype(object)))}
                out.append(r)
                print(f"lattice {feats:6} H1 {h1_bits or 'all'}-bit, {r['coefficients']} coefficients, rows={m:3d} "
                      f"trial {trial}: least-wrapped row wraps {r['min_wraps_in_rows']}; exactly s^2: {r['is_s2']}; "
                      f"exact on all {len(Nh)} T=1 rows: {exact}  ({sec:.1f}s)", flush=True)
                if log:
                    with open(log, "a") as f:
                        f.write(json.dumps(r) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", default=["value", "digits"])
    ap.add_argument("--rows", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--h1_bits", type=int, default=0)
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    run(a.feats, a.rows, a.seeds, a.h1_bits, a.log or None)


if __name__ == "__main__":
    main()
