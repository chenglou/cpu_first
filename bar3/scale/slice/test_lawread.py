"""The law reader's test (SLICE.md §3): on target rows built as the oracle arm builds them (the data's rows' top parts:
8 probes at widths 1 and 2, 60 acceptance rows at widths 1, 2 and 3, capacity 10^(2W)), the reading of each target.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.slice.test_lawread
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from bar3.scale.slice.lawread import constants, read

H1 = Path(__file__).resolve().parents[3].parent / "one-layer-deeper" / "data" / "generated" / \
    "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2" / "train.jsonl"
F = {"x2": lambda x, N: x * x % N, "x3": lambda x, N: pow(x, 3, N), "10x2p1": lambda x, N: (10 * x * x + 1) % N}


def top(x, N, w):
    s = str(N)
    if len(s) <= w:
        return x, N
    k = len(s) - w
    return x // 10 ** k, N // 10 ** k


def rows(seed=0):
    data = [json.loads(l) for l in open(H1)][:20000]
    rng = random.Random(seed)
    out = []
    for W, n in ((1, 8), (2, 8), (1, 60), (2, 60), (3, 60)):
        for _ in range(n):
            r = rng.choice(data)
            xt, Nt = top(int(r["x"]), int(r["modulus"]), W)
            out.append((xt, Nt, 10 ** (2 * W)))
    return out


def main():
    R = rows()
    for fn, f in F.items():
        x = np.array([a for a, _, _ in R]); N = np.array([b for _, b, _ in R]); cap = np.array([c for _, _, c in R])
        t = np.array([f(int(a), int(b)) for a, b in zip(x, N)])
        t0 = time.time()
        law, ops, d = read(x, N, t, cap, log=lambda s: print("   ", s, flush=True))
        print(f"{fn}: constants {constants(x, N, t, cap)}; law {law} (depth {d}), operations {ops}; "
              f"{time.time() - t0:.1f} s; {len(set(zip(x.tolist(), N.tolist())))} distinct rows", flush=True)


if __name__ == "__main__":
    main()
