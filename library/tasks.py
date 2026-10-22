"""The tasks: tiny arithmetic sub-problems of x^2 mod N, each W- and B-generic (LIBRARY.md §2).

A task gives, for a setting (B, W) and a probe count n, the input registers (digits [n, P]) and the targets
({"A": digits [n, P]} for the accumulator, or {"c": bits [n]} for the borrow). Every unused input is left random by
dsl.fresh. Targets here are exact answers (the offline corpus); inrun.py builds its targets from laws instead.
"""

from __future__ import annotations

import numpy as np

from library.dsl import to_digits


def _num(rng, lo, hi, n):
    """n random integers in [lo, hi) (hi may be an array)."""
    hi = np.broadcast_to(np.asarray(hi, dtype=np.int64), (n,))
    lo = np.broadcast_to(np.asarray(lo, dtype=np.int64), (n,))
    return lo + (rng.random(n) * (hi - lo)).astype(np.int64)


def gen(task, B, W, n, rng):
    P = 2 * W
    top = B ** W
    M = B ** P
    X = _num(rng, 0, top, n)
    Y = _num(rng, 0, top, n)
    Nn = _num(rng, 1, top, n)
    ins, tgt = {}, {}
    if task == "copy":
        ins, tgt = {"X": X}, {"A": X}
    elif task == "shl":
        ins, tgt = {"X": X}, {"A": X * B}
    elif task == "shr":
        ins, tgt = {"X": X}, {"A": X // B}
    elif task == "sub":
        ins, tgt = {"X": X, "N": Nn}, {"A": (X - Nn) % M}
    elif task == "add":
        ins, tgt = {"X": X, "N": Nn}, {"A": X + Nn}
    elif task == "lt":
        ins, tgt = {"X": X, "N": Nn}, {"c": (X < Nn).astype(np.uint8)}
    elif task == "neg":
        ins, tgt = {"X": X}, {"A": (-X) % M}
    elif task == "dbl":
        ins, tgt = {"X": X}, {"A": 2 * X}
    elif task == "mrow":
        ins, tgt = {"X": X, "Y": Y}, {"A": X * (Y % B)}
    elif task == "mac":
        ins, tgt = {"X": X, "Y": Y, "N": Nn}, {"A": Nn + X * (Y % B)}
    elif task == "subY":
        ins, tgt = {"X": X, "Y": Y}, {"A": (X - Y) % M}
    elif task == "addY":
        ins, tgt = {"X": X, "Y": Y}, {"A": X + Y}
    elif task == "ltY":
        ins, tgt = {"X": X, "Y": Y}, {"c": (X < Y).astype(np.uint8)}
    elif task == "negN":
        ins, tgt = {"N": Nn}, {"A": (-Nn) % M}
    elif task == "mrowN":
        ins, tgt = {"X": X, "N": Nn}, {"A": Nn * (X % B)}
    elif task == "csubY":
        Yv = _num(rng, 0, 2 * Nn, n)
        ins, tgt = {"Y": Yv, "N": Nn}, {"A": Yv % Nn}
    elif task == "modBY":
        Yv = _num(rng, 0, B * Nn, n)
        ins, tgt = {"Y": Yv, "N": Nn}, {"A": Yv % Nn}
    elif task == "step0":
        Xb = _num(rng, 0, M, n)
        ins, tgt = {"X": Xb, "N": Nn}, {"A": (Xb // (M // B)) % Nn}
    elif task == "mulmod":
        Xv, Yv = _num(rng, 0, Nn, n), _num(rng, 0, Nn, n)
        ins, tgt = {"X": Xv, "Y": Yv, "N": Nn}, {"A": (Xv * Yv) % Nn}
    elif task == "csub":
        X = _num(rng, 0, 2 * Nn, n)
        ins, tgt = {"X": X, "N": Nn}, {"A": X % Nn}
    elif task == "modB":
        X = _num(rng, 0, B * Nn, n)
        ins, tgt = {"X": X, "N": Nn}, {"A": X % Nn}
    elif task == "step":
        R = _num(rng, 0, Nn, n)
        Xb = _num(rng, 0, M, n)
        ins, tgt = {"X": Xb, "Y": R, "N": Nn}, {"A": (R * B + Xb // (M // B)) % Nn}
    elif task == "reduce":
        Xb = _num(rng, 0, M, n)
        ins, tgt = {"X": Xb, "N": Nn}, {"A": Xb % Nn}
    elif task == "mul":
        ins, tgt = {"X": X, "Y": Y}, {"A": X * Y}
    elif task == "sq":
        ins, tgt = {"X": X}, {"A": X * X}
    elif task == "sqmod":
        X = _num(rng, 0, Nn, n)
        ins, tgt = {"X": X, "N": Nn}, {"A": (X * X) % Nn}
    else:
        raise ValueError(task)
    ins = {r: to_digits(v, B, P) for r, v in ins.items()}
    tgt = {r: (to_digits(v, B, P) if r == "A" else np.asarray(v, np.uint8)) for r, v in tgt.items()}
    return ins, tgt


# the first task set (record r1): one task per concept
TASKS_R1 = ("copy", "shl", "shr", "sub", "add", "lt", "neg", "dbl", "mrow", "mac", "csub", "modB", "step", "reduce",
            "mul", "sq", "sqmod")

# the corpus's tasks, roughly from smallest to largest program; every one is attempted at every level until solved.
# Each concept is used by at least two tasks (the same operation on another register, or inside a bigger task),
# since a merge has to be used more than once to pay for itself.
TASKS = ("copy", "shl", "shr", "sub", "subY", "add", "addY", "lt", "ltY", "neg", "negN", "dbl", "mrow", "mrowN", "mac",
         "csub", "csubY", "modB", "modBY", "step0", "step", "reduce", "mul", "sq", "mulmod", "sqmod")

# the registers each task's inputs live in
INPUTS = {"copy": "X", "shl": "X", "shr": "X", "sub": "XN", "subY": "XY", "add": "XN", "addY": "XY", "lt": "XN",
          "ltY": "XY", "neg": "X", "negN": "N", "dbl": "X", "mrow": "XY", "mrowN": "XN", "mac": "XYN", "csub": "XN",
          "csubY": "YN", "modB": "XN", "modBY": "YN", "step0": "XN", "step": "XYN", "reduce": "XN", "mul": "XY",
          "sq": "X", "mulmod": "XYN", "sqmod": "XN"}
