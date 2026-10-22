"""How far the answers could pin a fixed modulus's digit code (LESSONS.md section 4), by brute force on a dataset's
training rows, with exact arithmetic standing in for an exact machine:

  1. the codes passing the reader's fact x < N on every row (mine_preview.code_count's partial order; each code a
     bijection digit -> label that keeps every ordered pair);
  2. of those, the codes whose machine reproduces every answer of the rows with T <= REAL_MAX_T (2): reading x and N
     through the code, squaring T times mod N' on the labels as numbers, writing the result back through the code.

If (2) leaves only the true code, the answers carry the information the facts lack, and a model would have to search
(1)'s codes as whole options to use it. A diagnostic run after the fact, not part of any model.

    python code_search.py DATA_DIR [DATA_DIR ...]
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

REAL_MAX_T = 2


def order_constraints(N: list, x: list) -> list:
    below = [set() for _ in range(10)]
    for n, v in zip(N, x):
        a = [int(c) for c in str(v)][::-1]
        b = [int(c) for c in str(n)][::-1]
        L = max(len(a), len(b))
        a, b = a + [0] * (L - len(a)), b + [0] * (L - len(b))
        top = max((j for j in range(L) if a[j] != b[j]), default=-1)
        if top >= 0:
            below[b[top]].add(a[top])
    return below


def digits(v: np.ndarray, L: int) -> np.ndarray:
    return np.stack([(v // 10 ** i) % 10 for i in range(L)], -1)


def search(data: str) -> dict:
    rows = [json.loads(l) for l in open(Path(data) / "train.jsonl")]
    N = [r["modulus"] for r in rows]
    x = [r["x"] for r in rows]
    below = order_constraints(N, x)
    ans = [r for r in rows if r["time_steps"] <= REAL_MAX_T]
    out = {"data": Path(data).name, "moduli": len(set(N)), "answered_rows": len(ans)}
    if len(set(N)) != 1:
        out["note"] = "not a fixed modulus"
        return out
    n = N[0]
    L = len(str(n))
    xs = np.array([r["x"] for r in ans], dtype=np.int64)
    Ts = np.array([r["time_steps"] for r in ans], dtype=np.int64)
    ys = np.array([r["result"] for r in ans], dtype=np.int64)
    xd, yd, nd = digits(xs, L), digits(ys, L), digits(np.array([n]), L)[0]
    pw = 10 ** np.arange(L)
    passing, fitting, true_ok = 0, 0, None
    for perm in itertools.permutations(range(10)):          # perm[d] = the label of digit d
        if any(perm[u] > perm[v] for v in range(10) for u in below[v]):
            continue
        passing += 1
        if not len(ans):
            continue
        p = np.array(perm)
        inv = np.argsort(p)
        n2 = int((p[nd] * pw).sum())
        if n2 == 0:
            continue
        y2 = (p[xd] * pw).sum(-1) % n2
        for t in range(int(Ts.max())):
            y2 = np.where(Ts > t, y2 * y2 % n2, y2)
        back = (inv[digits(y2, L)] * pw).sum(-1)              # written back through the code
        ok = bool((back == ys).all())
        fitting += ok
        if perm == tuple(range(10)):
            true_ok = ok
    out.update(codes_passing_x_lt_n=passing, codes_fitting_answers=fitting, true_code_fits=true_ok)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(search(d)), flush=True)
