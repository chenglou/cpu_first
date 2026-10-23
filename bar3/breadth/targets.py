"""The breadth test's targets (BREADTH.md): one round of each formula, as the learner saw x² mod N: A from x in X and
N in N, on every pair x < N <= k (Montgomery's on odd N only, where 2 has an inverse).

The learner reads its target from library/inventing/target.py (`pairs`: the instances, `answers`: the formula), via
bar3/problems.py (target_gen, the laws' data, dreams) and library/inventing/jobs.py (`general`, the N <= 40 check).
Importing this module points those two functions at the calling thread's target (x² mod N until `use(name)` says
otherwise), in this one place; nothing else changes. A search job needs the target only for `general`, so the driver
tags those jobs with the target's name and each worker calls `use` before it runs one (modal_breadth.py).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from library.inventing import target

ORIGINAL_PAIRS = target.pairs
ORIGINAL_ANSWERS = target.answers


def _fib(xs, Ns):
    """F_x mod N: (a, b) -> (b, a + b) mod N, x times from (0, 1)."""
    out = []
    for x, n in zip(xs.tolist(), Ns.tolist()):
        a, b = 0, 1
        for _ in range(x):
            a, b = b, (a + b) % n
        out.append(a % n)
    return np.array(out, np.int64)


def odd_pairs(k, lo=1):
    """Every (x, N) with lo <= N <= k, N odd, x < N, ordered by N then x."""
    xs, Ns = ORIGINAL_PAIRS(k, lo)
    keep = Ns % 2 == 1
    return xs[keep], Ns[keep]


@dataclass(frozen=True)
class Target:
    name: str
    formula: str
    family: str             # "squaring round" (x^(2^T)'s relatives) or "outside"
    answers: object         # (xs, Ns) int arrays -> int64 array
    pairs: object = ORIGINAL_PAIRS


def _i(f):
    return lambda xs, Ns: f(xs.astype(np.int64), Ns.astype(np.int64))


TARGETS = {t.name: t for t in (
    Target("h1", "x² mod N", "squaring round", _i(lambda x, N: (x * x) % N)),
    Target("rho", "(x² + 1) mod N", "squaring round", _i(lambda x, N: (x * x + 1) % N)),
    Target("minus1", "(x² − 1) mod N", "squaring round", _i(lambda x, N: (x * x - 1) % N)),
    Target("plus3", "(x² + 3) mod N", "squaring round", _i(lambda x, N: (x * x + 3) % N)),
    Target("ten", "10·x² mod N", "squaring round", _i(lambda x, N: (10 * x * x) % N)),
    Target("tenrho", "(10·x² + 1) mod N", "squaring round", _i(lambda x, N: (10 * x * x + 1) % N)),
    Target("cube", "x³ mod N (one round of x^(3^T))", "squaring round", _i(lambda x, N: (x * x * x) % N)),
    Target("mont", "x²·2⁻¹ mod N, N odd (Montgomery's squaring with B = 2)", "squaring round",
           _i(lambda x, N: (x * x * ((N + 1) // 2)) % N), odd_pairs),
    Target("affine", "(2x + 1) mod N (one step of a linear congruential map)", "outside",
           _i(lambda x, N: (2 * x + 1) % N)),
    Target("fib", "F_x mod N (a two-state recurrence: (a, b) -> (b, a + b), x times)", "outside", _fib),
)}

# the order of the carry-over sequence: the original first, then the relatives that its pieces make short, the
# easy outsider, then the targets whose shortest known programs are long, and the two-state recurrence last
SEQUENCE = ("h1", "rho", "minus1", "plus3", "affine", "ten", "tenrho", "cube", "mont", "fib")


_current = threading.local()      # the calling thread's target: one driver can run several targets at once


def _pairs(k, lo=1):
    return getattr(_current, "target", TARGETS["h1"]).pairs(k, lo)


def _answers(xs, Ns):
    return getattr(_current, "target", TARGETS["h1"]).answers(xs, Ns)


target.pairs = _pairs              # every caller of target.pairs / target.answers now asks the thread's target
target.answers = _answers


def use(name):
    """Make `name` the learner's target in this thread (library/inventing/target.py's pairs and answers)."""
    _current.target = TARGETS[name]
    return _current.target
