"""Shared pieces for the relaxed-rules experiments (RELAXED.md).

Small-modulus stand-ins for H1: a "b-bit modulus" is any odd N in [2^(b-1), 2^b). (Balanced Blum semiprimes, the H1
family, barely exist below 13 bits: 0 at 6 and 8 bits, 2 at 10 and 12.) Squaring mod N is the same arithmetic for any N.

- fresh_step(): unlimited single-step examples (N, s, y = s^2 mod N), s uniform in [0, N).
- h1like(): an H1-shaped finite training set at b bits: a pool of <= 430 moduli, 1,000 rows at T=1, 1,000 at T=2,
  33,333 each at T=4/8/16, x a unit; with every intermediate state, so relaxations can choose what to supervise.
- ladder(): T = 1, 2, 4, ..., 64, 768 fresh rows per rung (H1's rung size), all rungs must be 100% exact.
All numbers are LSB-first digit arrays of a fixed width W.
"""

from __future__ import annotations

import math

import numpy as np

LADDER = (1, 2, 4, 8, 16, 32, 64)
H1_COUNTS = {1: 1000, 2: 1000, 4: 33333, 8: 33333, 16: 33333}


def width(bits: int) -> int:
    return len(str((1 << bits) - 1))


def digits(v: np.ndarray, W: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64).copy()
    out = np.empty((len(v), W), dtype=np.int64)
    for i in range(W):
        out[:, i] = v % 10
        v //= 10
    return out


def undigits(d: np.ndarray) -> np.ndarray:
    return (np.asarray(d, dtype=np.int64) * (10 ** np.arange(d.shape[1], dtype=np.int64))).sum(1)


def odd_moduli(bits: int) -> np.ndarray:
    return np.arange((1 << (bits - 1)) + 1, 1 << bits, 2, dtype=np.int64)


def split_pool(bits: int, k: int = 430, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(training pool, unseen moduli). If <= k moduli exist, the pool is all of them and 'unseen' is empty."""
    allm = np.random.default_rng(seed + 1000 * bits).permutation(odd_moduli(bits))
    return np.sort(allm[:k]), np.sort(allm[k:])


def units(N: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = (rng.random(len(N)) * (N - 1)).astype(np.int64) + 1
    bad = np.gcd(x, N) != 1
    while bad.any():
        x[bad] = (rng.random(bad.sum()) * (N[bad] - 1)).astype(np.int64) + 1
        bad = np.gcd(x, N) != 1
    return x


def sqmod(s: np.ndarray, N: np.ndarray) -> np.ndarray:
    return s * s % N  # int64 is exact for N < 2^31


def power(x: np.ndarray, N: np.ndarray, T: int) -> np.ndarray:
    s = x % N
    for _ in range(T):
        s = sqmod(s, N)
    return s


def fresh_step(rng: np.random.Generator, bits: int, B: int, pool: np.ndarray | None = None):
    if pool is None:
        N = rng.integers(1 << (bits - 2), 1 << (bits - 1), B, dtype=np.int64) * 2 + 1
    else:
        N = pool[rng.integers(0, len(pool), B)]
    s = (rng.random(B) * N).astype(np.int64)
    return N, s, sqmod(s, N)


def h1like(bits: int, seed: int = 0, counts: dict[int, int] = H1_COUNTS, k: int = 430):
    """H1-shaped training rows at `bits`. Returns dict with N, X, T, Y and traj (list per row: s_0..s_T)."""
    rng = np.random.default_rng(seed + 7 * bits)
    pool, unseen = split_pool(bits, k, seed)
    Ns, Xs, Ts, trajs = [], [], [], []
    for T, c in counts.items():
        N = pool[rng.integers(0, len(pool), c)]
        x = units(N, rng)
        traj = np.empty((c, T + 1), dtype=np.int64)
        traj[:, 0] = x
        for t in range(T):
            traj[:, t + 1] = sqmod(traj[:, t], N)
        Ns.append(N), Xs.append(x), Ts.append(np.full(c, T)), trajs.extend(list(traj))
    N, X, T = np.concatenate(Ns), np.concatenate(Xs), np.concatenate(Ts)
    Y = np.array([tr[-1] for tr in trajs], dtype=np.int64)
    return {"N": N, "X": X, "T": T, "Y": Y, "traj": trajs, "pool": pool, "unseen": unseen}


def step_pairs(d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step labels: every (N, s_t, s_{t+1}) along every training trajectory."""
    N = np.concatenate([np.full(len(tr) - 1, n) for n, tr in zip(d["N"], d["traj"])])
    S = np.concatenate([tr[:-1] for tr in d["traj"]])
    Y = np.concatenate([tr[1:] for tr in d["traj"]])
    return N, S, Y


def ladder(step_fn, bits: int, moduli: np.ndarray | None = None, rows: int = 768, seed: int = 99,
           ladder_ts=LADDER, verbose: bool = True) -> dict:
    """step_fn(N int64[B], s int64[B]) -> int64[B] (one hard-snapped squaring). Certification ladder at `bits`."""
    rng = np.random.default_rng(seed + bits)
    accs, max_t, broken = {}, 0, False
    for T in ladder_ts:
        N = moduli[rng.integers(0, len(moduli), rows)] if moduli is not None else \
            rng.integers(1 << (bits - 2), 1 << (bits - 1), rows, dtype=np.int64) * 2 + 1
        x = units(N, rng)
        y = power(x, N, T)
        s = x.copy()
        for _ in range(T):
            s = step_fn(N, s)
        ok = s == y
        accs[T] = float(ok.mean())
        if not broken and ok.all():
            max_t = T
        else:
            broken = True
        if verbose:
            print(f"    ladder b={bits} T={T:<3d} exact={accs[T]:.4f}", flush=True)
    return {"max_t": max_t, "accs": accs}


def isqrt_ceil(n: int) -> int:
    r = math.isqrt(n)
    return r if r * r == n else r + 1
