"""Montgomery's quotient digit as a slice (theory/LARGE_NUMBERS.md §2.6 and §5.5 test 4; CARRY.md §13). Local,
seconds, no search, nothing on Modal.

Montgomery's reduction of t by N in base B (N coprime to B, k digits): k times, read t's lowest digit t0, take the
digit q with t + q·N ≡ 0 (mod B), add q·N (which clears that digit) and shift t down a place. The result is
t·B^(-k) mod N, or that plus N (one final subtraction). Three checks:
1. the one-place test, slices.py's ("the same inputs never want two answers"), on rows read off the reductions
   themselves, as the book reads a slice off its law: over random (t, N) at every base from 2 to 16, is q a function
   of (t0, N0), N's last digit? of t0 alone?
2. the lift: the reduction run from cells alone, at every base from 2 to 16, against t·B^(-k) mod N. q comes from the
   table the rows give; t + q·N from the invented multiply-accumulate's table (place by place, its carry passed up,
   H = q); then the shift; and the final subtraction from the invented subtract's and select's tables (keep t when
   t − N borrows). The three programs are the ones the learner invented (records/subgoal/), compiled by running them on
   every row of their cell (checks/carry_tables.py's way).
3. the contrast: long division's first quotient digit (t div N, with t < B·N), read off the same way: is it a function
   of t's top two digits and N's top digit, the places a division step looks at first?

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.montgomery
"""

from __future__ import annotations

import itertools
import json
from math import gcd
from pathlib import Path

import numpy as np

from library import dsl
from bar3.bridge import dcm
from bar3.bridge.dproblems import machine_for

HERE = Path(__file__).resolve().parent.parent
BASES = tuple(range(2, 17))


def digits(v, B, n):
    return [(v // B ** i) % B for i in range(n)]


def value(ds, B):
    return sum(d * B ** i for i, d in enumerate(ds))


def modulus(rng, B, k):
    """(N, k): a random N of k digits in base B, coprime to B and above 1 (k raised when no such N has k digits)."""
    while max(2, B ** (k - 1)) >= B ** k:
        k += 1
    while True:
        N = int(rng.integers(max(2, B ** (k - 1)), B ** k))
        if gcd(N, B) == 1:
            return N, k


def montgomery_rows(B, n=600, kmax=4, seed=0):
    """Every step of n random reductions at base B: (t0, N0, N's next digit, q), q the digit with t + q·N ≡ 0 mod B."""
    rng = np.random.default_rng(seed + B)
    rows = []
    for _ in range(n):
        N, k = modulus(rng, B, int(rng.integers(1, kmax + 1)))
        t = int(rng.integers(0, N * B ** k))
        for _ in range(k):
            q = next(q for q in range(B) if (t + q * N) % B == 0)
            rows.append((t % B, N % B, (N // B) % B, q))
            t = (t + q * N) // B
    return rows


def function_of(rows, key):
    """(groups, groups wanting more than one answer) when the rows are keyed by key(row)."""
    seen = {}
    for r in rows:
        seen.setdefault(key(r), set()).add(r[-1])
    return len(seen), sum(1 for v in seen.values() if len(v) > 1)


def program(run):
    rec = json.loads((HERE / "records" / "subgoal" / f"{run}.jsonl").read_text().splitlines()[-1])
    return dsl.parse(rec["general"]["program"]), {k: dsl.parse(v) for k, v in rec["lib"].items()}


def compile_cell(prog, lib, B, cs, hs, outs):
    """{(a, x, c, h): outputs} for a program run on every row a, x < B, c in cs, h in hs (A, R, C, H)."""
    rows = list(itertools.product(range(B), range(B), cs, hs))
    ins = {r: np.array([row[i] for row in rows], np.int64) for i, r in enumerate("ARCH")}
    ins["N"] = np.full(len(rows), B, np.int64)
    regs, dead = dcm.evaluate(prog, lib, ins, machine_for((B,)))
    assert not dead, "a cell ran out of steps"
    return {row: tuple(int(v) for v in o) for row, o in zip(rows, regs[:, [dcm.RI[r] for r in outs]])}


def redc(t, N, k, B, qtab, mul, sub, sel):
    """t·B^(-k) mod N from the cells: k steps of q, t + q·N and the shift, then the final subtraction."""
    P = 2 * k + 2
    T, Nd = digits(t, B, P), digits(N, B, P)
    for _ in range(k):
        q = qtab[(T[0], Nd[0])]
        out, c = [], 0
        for i in range(P):
            d, c = mul[(T[i], Nd[i], c, q)]
            out.append(d)
        assert c == 0 and out[0] == 0
        T = out[1:] + [0]
    diff, b = [], 0
    for i in range(P):
        d, b = sub[(T[i], Nd[i], b, 0)]
        diff.append(d)
    return value([sel[(diff[i], T[i], b, 0)][0] for i in range(P)], B)


def division_rows(B, n=3000, seed=0):
    """Long division's first quotient digit: (t's top two digits, N's top digit, q) for t < B·N, N of 2 to 4 digits."""
    rng = np.random.default_rng(seed + 100 + B)
    rows = []
    for _ in range(n):
        k = int(rng.integers(2, 5))
        N = int(rng.integers(B ** (k - 1), B ** k))
        t = int(rng.integers(N, B * N))
        rows.append((t // B ** (k - 1), N // B ** (k - 1), t // N))
    return rows


def main():
    print("1. the one-place test: q with t + q·N = 0 mod B, read off random reductions")
    print("   base | steps | keyed by (t0, N0): groups, mixed | by t0 alone: groups, mixed | q = -t0·N0^-1 mod B")
    for B in BASES:
        rows = montgomery_rows(B)
        g, bad = function_of(rows, lambda r: (r[0], r[1]))
        g0, bad0 = function_of(rows, lambda r: r[0])
        formula = all(q == (-t0 * pow(n0, -1, B)) % B for t0, n0, _, q in rows)
        print(f"   {B:4d} | {len(rows):5d} | {g:4d} groups, {bad} mixed | {g0:3d} groups, {bad0} mixed | {formula}")

    print("2. the lift: t·B^(-k) mod N from the q table and the invented cells (mul, sub, select)")
    pm, lm = program("slp_mul_k1")
    ps, ls = program("sct2_sbb_k1")
    pl, ll = program("slp_sel_k1")
    for B in BASES:
        mul = compile_cell(pm, lm, B, range(B), range(B), "AC")
        sub = compile_cell(ps, ls, B, range(2), [0], "AC")
        sel = compile_cell(pl, ll, B, range(2), [0], "A")
        qtab = {(t0, n0): q for t0, n0, _, q in montgomery_rows(B)}
        rng = np.random.default_rng(7 + B)
        right = total = 0
        for _ in range(300):
            N, k = modulus(rng, B, int(rng.integers(1, 5)))
            t = int(rng.integers(0, N * B ** k))
            right += redc(t, N, k, B, qtab, mul, sub, sel) == (t * pow(B, -k, N)) % N
            total += 1
        print(f"   base {B:2d}: {right} / {total} reductions right; q table {len(qtab)} rows "
              f"({sum(1 for n0 in range(B) if gcd(n0, B) == 1) * B} possible)")

    print("3. the contrast: long division's first quotient digit, keyed by t's top two digits and N's top digit")
    for B in (2, 3, 7, 10, 16):
        rows = division_rows(B)
        g, bad = function_of(rows, lambda r: (r[0], r[1]))
        spread = max((max(v) - min(v) for v in
                      [{r[2] for r in rows if (r[0], r[1]) == key} for key in {(r[0], r[1]) for r in rows}]), default=0)
        print(f"   base {B:2d}: {g} groups, {bad} mixed, quotient digits in one group differing by up to {spread}")


if __name__ == "__main__":
    main()
