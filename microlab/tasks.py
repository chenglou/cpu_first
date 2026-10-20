"""Tiny arithmetic tasks for fast hypothesis tests.

Every number is an LSB-first digit array of a fixed width. A task samples a Batch with named inputs, one target and
optional auxiliary targets ("show your work" quantities such as the raw square before the remainder). Sizes are
digits (k) for plain arithmetic and bits (b) of the modulus for modular tasks, where a b-bit modulus is any odd N in
[2^(b-1), 2^b) — the same small-modulus stand-in the relaxed-rules experiments use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def digits(v, width: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64).copy()
    out = np.empty((len(v), width), dtype=np.int64)
    for i in range(width):
        out[:, i] = v % 10
        v //= 10
    return out


def ndig(v) -> int:
    return len(str(int(v)))


@dataclass
class Batch:
    inputs: dict[str, np.ndarray]                 # name -> [B, width] digits
    target: np.ndarray                             # [B, width] digits
    aux: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Task:
    name: str
    kind: str                                      # "digits" (size = k) or "bits" (size = b)
    describe: str

    def widths(self, size: int) -> tuple[dict[str, int], int, dict[str, int]]:
        raise NotImplementedError

    def sample(self, n: int, size: int, rng: np.random.Generator) -> Batch:
        raise NotImplementedError


def _uniform_k(n, k, rng):
    lens = rng.integers(1, k + 1, n)
    return np.array([rng.integers(0, 10 ** int(L)) for L in lens], dtype=np.int64)


def _odd_moduli(n, b, rng):
    return (rng.integers(1 << (b - 2), 1 << (b - 1), n) * 2 + 1).astype(np.int64)


class Add(Task):
    def __init__(self):
        super().__init__("add", "digits", "a + b")

    def widths(self, k):
        return {"a": k, "b": k}, k + 1, {}

    def sample(self, n, k, rng):
        a, b = _uniform_k(n, k, rng), _uniform_k(n, k, rng)
        return Batch({"a": digits(a, k), "b": digits(b, k)}, digits(a + b, k + 1))


class Mul(Task):
    def __init__(self):
        super().__init__("mul", "digits", "a * b")

    def widths(self, k):
        return {"a": k, "b": k}, 2 * k, {}

    def sample(self, n, k, rng):
        a, b = _uniform_k(n, k, rng), _uniform_k(n, k, rng)
        return Batch({"a": digits(a, k), "b": digits(b, k)}, digits(a * b, 2 * k))


class Sq(Task):
    def __init__(self):
        super().__init__("sq", "digits", "x * x")

    def widths(self, k):
        return {"x": k}, 2 * k, {}

    def sample(self, n, k, rng):
        x = _uniform_k(n, k, rng)
        return Batch({"x": digits(x, k)}, digits(x * x, 2 * k))


class Mod(Task):
    def __init__(self):
        super().__init__("mod", "bits", "v mod N, v < N^2 (aux q = v // N)")

    def widths(self, b):
        w = ndig((1 << b) - 1)
        return {"N": w, "v": 2 * w}, w, {"q": w}

    def sample(self, n, b, rng):
        N = _odd_moduli(n, b, rng)
        v = (rng.random(n) * (N.astype(np.float64) ** 2)).astype(np.int64)
        v = np.minimum(v, N * N - 1)
        w = ndig((1 << b) - 1)
        return Batch({"N": digits(N, w), "v": digits(v, 2 * w)}, digits(v % N, w), {"q": digits(v // N, w)})


class SqMod(Task):
    def __init__(self):
        super().__init__("sqmod", "bits", "x^2 mod N (aux pre = x^2, q = x^2 // N)")

    def widths(self, b):
        w = ndig((1 << b) - 1)
        return {"N": w, "x": w}, w, {"pre": 2 * w, "q": w}

    def sample(self, n, b, rng):
        N = _odd_moduli(n, b, rng)
        x = (rng.random(n) * N).astype(np.int64)
        w = ndig((1 << b) - 1)
        p = x * x
        return Batch({"N": digits(N, w), "x": digits(x, w)}, digits(p % N, w),
                     {"pre": digits(p, 2 * w), "q": digits(p // N, w)})


class PowMod(Task):
    """x^(2^T) mod N for a fixed T (pass size as bits; T via PowMod(T)). aux s1..sT = every intermediate residue."""

    def __init__(self, T: int = 2):
        super().__init__(f"powmod{T}", "bits", f"x^(2^{T}) mod N (aux s1..s{T})")
        self.T = T

    def widths(self, b):
        w = ndig((1 << b) - 1)
        return {"N": w, "x": w}, w, {f"s{t}": w for t in range(1, self.T)}

    def sample(self, n, b, rng):
        N = _odd_moduli(n, b, rng)
        x = (rng.random(n) * N).astype(np.int64)
        w = ndig((1 << b) - 1)
        s, aux = x.copy(), {}
        for t in range(1, self.T + 1):
            s = s * s % N
            if t < self.T:
                aux[f"s{t}"] = digits(s, w)
        return Batch({"N": digits(N, w), "x": digits(x, w)}, digits(s, w), aux)


TASKS: dict[str, Task] = {t.name: t for t in (Add(), Mul(), Sq(), Mod(), SqMod(), PowMod(2), PowMod(4))}
