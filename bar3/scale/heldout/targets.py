"""The held-out formulas (HELDOUT.md): ten targets that none of the learner's rules were written for, posed as the
learner poses x² mod N: A from x in X and N in N, on every pair x < N <= k. Data only: importing this module adds them
to bar3/breadth/targets.py's table (TG.TARGETS), which is where the learner, the search workers' N <= 40 check and the
N = 120 recheck look a target up by name. Nothing else changes, and the ten original targets are left as they are.
"""

from __future__ import annotations

import numpy as np

from bar3.breadth import targets as TG


def _i(f):
    return lambda xs, Ns: f(xs.astype(np.int64), Ns.astype(np.int64))


def _each(f):
    """f(x, N) on every pair, in Python integers (exact for any x)."""
    return lambda xs, Ns: np.array([f(x, n) for x, n in zip(xs.tolist(), Ns.tolist())], np.int64)


def _sums(start):
    """y(x) mod N for the sequence that starts at `start` and whose next term is the sum of the len(start) before it."""
    def y(x, n):
        w = [v % n for v in start]
        for _ in range(x):
            w = w[1:] + [sum(w) % n]
        return w[0]
    return _each(y)


HELDOUT = (
    TG.Target("x2x", "(x² + x) mod N", "held out", _i(lambda x, N: (x * x + x) % N)),
    TG.Target("x1sq", "(x + 1)² mod N", "held out", _i(lambda x, N: ((x + 1) * (x + 1)) % N)),
    TG.Target("three", "3·x² mod N", "held out", _i(lambda x, N: (3 * x * x) % N)),
    TG.Target("x4", "x⁴ mod N", "held out", _i(lambda x, N: (x * x * x * x) % N)),
    TG.Target("x3x", "(x³ + x) mod N", "held out", _i(lambda x, N: (x * x * x + x) % N)),
    TG.Target("pow2", "2^x mod N", "held out", _each(lambda x, n: pow(2, x, n))),
    TG.Target("lucas", "L(x) mod N, L(0) = 2, L(1) = 1, L(x + 2) = L(x) + L(x + 1)", "held out", _sums([2, 1])),
    TG.Target("trib", "T(x) mod N, T(0) = T(1) = 0, T(2) = 1, T(x + 3) = T(x) + T(x + 1) + T(x + 2)", "held out",
              _sums([0, 0, 1])),
    TG.Target("floorsq", "floor(x² / N)", "held out", _i(lambda x, N: (x * x) // N)),
    TG.Target("tri", "x·(x + 1)/2 mod N", "held out", _i(lambda x, N: (x * (x + 1) // 2) % N)),
)
NAMES = tuple(t.name for t in HELDOUT)

assert not set(NAMES) & set(TG.SEQUENCE), "a held-out name is taken"
assert all("_" not in n for n in NAMES)       # record names are <config>_<target>_<tag>
for _t in HELDOUT:
    TG.TARGETS.setdefault(_t.name, _t)
