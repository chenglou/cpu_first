"""The digit law book: bar 3's law book of arithmetic on whole numbers, cut to one place of a digit machine.

A digit machine computes a law on whole numbers one place at a time: at place i it sees the operands' digits there
and what the places below passed up, and it writes the result's digit and what it passes on. That one-place step is a
function on a bounded domain (digits below the base, and a small state), the size bar 3 works at. So every law of the
book is posed here as its *slice*: the step at one place, read off the law itself on the data's numbers.

Two generic definitions make the slice from the law L and nothing else:
- what place i passes on (the state, "the carry"): for a law with a numeric value, the part of L on the low parts
  that lies above place i, floor(L(x mod B^i, y mod B^i, d) / B^i) (a borrow when it is negative: then its size);
  for a predicate (lt, eq), the predicate on the low parts, L(x mod B^i, y mod B^i). A constant in the law (the 1 of
  x + 1, the d of o + x + d) becomes the state of place 0, L(0, 0, d);
- what place i writes: digit i of L(x mod B^(i+1), y mod B^(i+1), d) (nothing for a predicate).
A slice is posed only if it is a function of what the place sees (bar 3's own test: the same inputs never want two
answers). That keeps add, sub, compare, select, multiply by a digit and multiply-accumulate, and drops min, max, the
remainder, the quotient and the full product, whose place i depends on places a step cannot see.

Registers (dcm.py): numbers' digits go to A and R (every binding), a digit operand to H, a flag operand to C, the
state to C (C carries it in and out), the base to N; the output digit is A. A law whose state is always 0 has no state
(C is then free for a flag).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SEARCH_BASES = (2, 3, 4, 5, 6, 7)      # the search size (bar 3's x < N <= 7)
CHECK_BASES = (8, 9, 10)               # accepted when right here too (bar 3's N <= 10); 10 is H1's base
GENERAL_BASES = tuple(range(2, 17))     # general: right at every base up to 16 (bar 3's N <= 40)
WIDTH = 3                               # digits per number when the slices are read off the law
SAMPLES = 900                           # operand tuples per base


# ---------------------------------------------------------------------------------------------------------------
# the law book: SELF_PRACTICE.md's 11 laws (lt, eq, min, max, red, div, add, sub, mul, a + b c, a + b + c), bar 3's
# succ and pred, and select; operand kinds: "num" (a whole number), "dig" (one digit), "flag" (0 or 1)

LAWS = {
    "lt": (("num", "num"), lambda x, y: int(x < y), "pred"),
    "eq": (("num", "num"), lambda x, y: int(x == y), "pred"),
    "min": (("num", "num"), min, "value"),
    "max": (("num", "num"), max, "value"),
    "red": (("num", "num"), lambda x, y: x % y if y else None, "value"),
    "div": (("num", "num"), lambda x, y: x // y if y else None, "value"),
    "add": (("num", "num"), lambda x, y: x + y, "value"),
    "sub": (("num", "num"), lambda x, y: x - y, "value"),
    "mul": (("num", "num"), lambda x, y: x * y, "value"),
    "muld": (("num", "dig"), lambda x, d: x * d, "value"),
    "acc.mul": (("num", "num", "dig"), lambda o, x, d: o + x * d, "value"),
    "acc.add": (("num", "num", "dig"), lambda o, x, d: o + x + d, "value"),
    "succ": (("num",), lambda x: x + 1, "value"),
    "pred": (("num",), lambda x: x - 1, "value"),
    "select": (("flag", "num", "num"), lambda f, x, y: y if f else x, "value"),
}


def bindings(kinds):
    """Every way to put a law's operands in the registers: numbers in A and R (each order), a digit in H, a flag in C."""
    nums = [i for i, k in enumerate(kinds) if k == "num"]
    out = []
    for regs in ([("A",), ("R",)] if len(nums) == 1 else [("A", "R"), ("R", "A")] if len(nums) == 2 else [()]):
        b = {}
        for i, r in zip(nums, regs):
            b[i] = r
        for i, k in enumerate(kinds):
            if k == "dig":
                b[i] = "H"
            elif k == "flag":
                b[i] = "C"
        out.append(tuple(b[i] for i in range(len(kinds))))
    return out


def _digits(v, B, W):
    return [(v // B ** i) % B for i in range(W)]


@dataclass
class Slice:
    name: str               # e.g. "sub(A,R)"
    law: str
    binding: tuple          # the register of each operand
    inputs: tuple           # input registers (in dcm.REGS order)
    outputs: tuple          # output registers
    state: bool             # does C carry a state?
    init: dict              # base -> the state of place 0
    sign: int               # -1 when the state is a borrow (stored as its size)
    rows: dict = field(default_factory=dict)       # base -> {input tuple: output tuple}
    ok: bool = True         # a function of what the place sees, at every base
    why: str = ""

    def table(self, B):
        return self.rows[B]


def slice_of(law, binding, bases, W=WIDTH, n=SAMPLES, seed=0):
    """The slice of a law in one binding, read off the law on random operands at each base."""
    kinds, fn, kind = LAWS[law]
    rng = np.random.default_rng(seed)
    name = f"{law}({','.join(binding)})"
    drop = lambda why: Slice(name, law, binding, (), (), False, {}, 1, ok=False, why=why)   # noqa: E731
    recs = {}
    for B in bases:                      # the operands, the law on the low parts below each place, the full value
        out = []
        for _ in range(n):
            ops = [int(rng.integers(0, B ** W)) if k == "num" else int(rng.integers(0, B)) if k == "dig"
                   else int(rng.integers(0, 2)) for k in kinds]
            v = fn(*ops)
            lows = [fn(*[o % B ** i if k == "num" else o for o, k in zip(ops, kinds)]) for i in range(W + 1)]
            if v is None or None in lows:
                return drop(f"undefined on the low parts at base {B} (a place cannot know it)")
            out.append((ops, lows, v))
        recs[B] = out

    def states(B, lows, i):
        if kind == "pred":
            return lows[i], lows[i + 1]
        return lows[i] // B ** i, lows[i + 1] // B ** (i + 1)

    allst = [s for B in bases for ops, lows, v in recs[B] for i in range(W) for s in states(B, lows, i)]
    has_state = any(s != 0 for s in allst)
    if min(allst) < 0 < max(allst):
        return drop("the state takes both signs")
    sign = -1 if min(allst) < 0 else 1
    inputs = tuple(r for r in "ARCH" if r in binding or r == "A" or (r == "C" and has_state)) + ("N",)
    outputs = (("A",) if kind == "value" else ()) + (("C",) if has_state else ())
    sl = Slice(name, law, binding, inputs, outputs, has_state, {}, sign)
    for B in bases:
        rows = {}
        for ops, lows, v in recs[B]:
            for i in range(W):
                ins = {r: (o // B ** i) % B if k == "num" else o for o, k, r in zip(ops, kinds, binding)}
                s_in, s_out = states(B, lows, i)
                if has_state:
                    ins["C"] = s_in * sign
                key = tuple(ins.get(r, 0) for r in "ARCH")
                out = (((v % B ** W) // B ** i) % B,) if kind == "value" else ()
                out += (s_out * sign,) if has_state else ()
                if rows.setdefault(key, out) != out:
                    return drop(f"not a function of the place at base {B}: {key} wants {rows[key]} and {out}")
        sl.rows[B] = rows
        sl.init[B] = sorted({states(B, lows, 0)[0] * sign for ops, lows, v in recs[B]})
    return sl


def book(bases=SEARCH_BASES + CHECK_BASES, laws=None):
    """Every law of the book in every binding: (posed slices, dropped slices with the reason)."""
    posed, dropped, seen = [], [], {}
    for law in (laws or LAWS):
        for b in bindings(LAWS[law][0]):
            s = slice_of(law, b, bases)
            if s.ok:
                key = (s.inputs, s.outputs, tuple(sorted((B, tuple(sorted(r.items()))) for B, r in s.rows.items())))
                if key in seen:
                    s.ok, s.why = False, f"the same slice as {seen[key]}"
                else:
                    seen[key] = s.name
            (posed if s.ok else dropped).append(s)
    return posed, dropped


# ---------------------------------------------------------------------------------------------------------------
# probes: a slice's rows as machine inputs and wanted outputs

@dataclass
class Probes:
    inputs: dict            # register -> [n]
    want: np.ndarray        # [n, len(outputs)]
    bases: np.ndarray       # [n]


def probes(sl, bases, restrict=None):
    """Every row of the slice at these bases (restrict: a predicate on the inputs dict of one row)."""
    cols = {r: [] for r in sl.inputs}
    want, bs = [], []
    for B in bases:
        for key, out in sorted(sl.rows[B].items()):
            ins = dict(zip("ARCH", key))
            ins["N"] = B
            if restrict is not None and not restrict(ins):
                continue
            for r in sl.inputs:
                cols[r].append(ins[r])
            want.append(out)
            bs.append(B)
    return Probes({r: np.array(v, np.int64) for r, v in cols.items()}, np.array(want, np.int64).reshape(len(want), -1),
                  np.array(bs))


def sample(p: Probes, n=32, seed=0):
    """At most n probes for the search: every base represented, the rest at random (bar 3 samples 32)."""
    k = len(p.bases)
    if k <= n:
        return p
    rng = np.random.default_rng(seed)
    keep = []
    for B in np.unique(p.bases):
        idx = np.nonzero(p.bases == B)[0]
        keep += rng.choice(idx, min(2, len(idx)), replace=False).tolist()
    rest = np.setdiff1d(np.arange(k), keep)
    keep += rng.choice(rest, n - len(keep), replace=False).tolist()
    keep = np.sort(np.array(keep))
    return Probes({r: v[keep] for r, v in p.inputs.items()}, p.want[keep], p.bases[keep])


def main():
    posed, dropped = book()
    print(f"posed {len(posed)}:")
    for s in posed:
        n = sum(len(s.rows[B]) for B in s.rows)
        print(f"  {s.name:24s} in {','.join(s.inputs):10s} out {','.join(s.outputs):4s} rows {n:6d} sign {s.sign:2d} "
              f"place-0 state at base 10 {s.init[10]}")
    print(f"dropped {len(dropped)}:")
    for s in dropped:
        print(f"  {s.name:24s} {s.why}")


if __name__ == "__main__":
    main()
