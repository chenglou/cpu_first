"""The target's own data: (x, N) -> x^2 mod N, smallest numbers first (INVENTING.md §2).

A size k is every pair with 1 <= N <= k and 0 <= x < N. Base 2 with up to 3 digits is k = 7, base 3 with up to 3
digits is k = 26. No Blum condition: at these sizes it would leave almost nothing (21 is the smallest Blum number with
distinct primes).
"""

from __future__ import annotations

import numpy as np

SIZES = {"b2d3": 7, "b3d2": 8, "b3d3": 26}


def pairs(k, lo=1):
    """Every (x, N) with lo <= N <= k, x < N, ordered by N then x."""
    out = [(x, n) for n in range(lo, k + 1) for x in range(n)]
    return np.array([p[0] for p in out]), np.array([p[1] for p in out])


def probes(k, n=None, seed=0):
    """The probes of size k: all pairs, or n of them (always the largest N's pairs first, then a random sample)."""
    xs, Ns = pairs(k)
    if n is None or len(xs) <= n:
        return xs, Ns
    rng = np.random.default_rng(seed)
    top = np.nonzero(Ns == k)[0]
    rest = np.setdiff1d(np.arange(len(xs)), top)
    keep = np.concatenate([top[:n // 2], rng.choice(rest, n - min(len(top), n // 2), replace=False)])
    keep.sort()
    return xs[keep], Ns[keep]


def answers(xs, Ns):
    return (xs.astype(np.int64) ** 2) % Ns


# Arm B on the bare machine: our kind of curated practice, each concept used by at least two tasks, the bare
# machine's analogues of the digit tasks (library/tasks.py): copy, +1, sum, difference (stopping at 0), comparison,
# the restoring link, the reduction, the product, the square, and the target.
CURATED = ("copy", "copyN", "inc", "incN", "add", "dbl", "monus", "monusN", "lt", "eq", "csub", "csub1", "mod",
           "modN1", "mul", "sq", "sqmod")


def curated(task, k, seed=0):
    """Inputs (x in X, N in N) and the answer A for a curated task at size k: every pair (x < N <= k), except the
    link and reduction tasks, whose x runs past N."""
    rng = np.random.default_rng(seed)
    xs, Ns = pairs(k)
    xs, Ns = xs.astype(np.int64), Ns.astype(np.int64)
    if task in ("csub", "csub1"):                    # x < 2N (csub1: x <= N, the link at its edge)
        xs = (rng.random(len(Ns)) * (2 * Ns if task == "csub" else Ns + 1)).astype(np.int64)
    elif task in ("monus", "monusN", "lt", "eq"):    # x < 2N, so both sides of N occur
        xs = (rng.random(len(Ns)) * 2 * Ns).astype(np.int64)
        if task == "eq":
            xs[::3] = Ns[::3]
    elif task in ("mod", "modN1"):                   # x < N^2, two samples
        r = np.random.default_rng(seed + (7 if task == "modN1" else 0))
        xs = (r.random(len(Ns)) * Ns * Ns).astype(np.int64)
    want = {"copy": xs, "copyN": Ns, "inc": xs + 1, "incN": Ns + 1, "add": xs + Ns, "dbl": 2 * xs,
            "monus": np.maximum(xs - Ns, 0), "monusN": np.maximum(Ns - xs, 0), "lt": (xs < Ns).astype(np.int64),
            "eq": (xs == Ns).astype(np.int64), "csub": xs % Ns, "csub1": xs % Ns, "mod": xs % Ns, "modN1": xs % Ns,
            "mul": xs * Ns, "sq": xs * xs, "sqmod": (xs * xs) % Ns}[task]
    return xs, Ns, want
