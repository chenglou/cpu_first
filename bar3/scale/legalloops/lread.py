"""The cuts and loops of the target's law, read legally (LEGALLOOPS.md §2).

Round 4 (bar3/scale/slice/) and round 5 (bar3/scale/loops/) read the target's law with numpy on the small target's
rows, then posed the cuts and loops of its operations with Python's answers. Here both parts are legal:

- **The reading** runs on H1's own answer rows at full size, with the learner's own routines and the book's own
  constructions, and nothing else computes a number:
  - add: the learner's addition;
  - mul: repeated addition, b additions of a (the book's mul.ladder). It is computable on a row when b, read off its
    digits as a count, is below COUNT;
  - mod: repeated conditional subtraction, the learner's link applied until its comparison says the value is below
    the modulus. It is computable on a row when that happens within COUNT links.

  Laws are left-deep chains of these over X and N, to depth 2. A law is read when it is computable on at least
  MIN_ROWS rows and gives each one's answer digit for digit. The first law that does, by depth, then inner chain,
  then operation, is the reading.
- **The cuts and loops of the read law's operations** are posed as the book's own laws on the patterns they cut to,
  answered by the book's own constructions (spcore's law_target: identities and the learner's routines):
  - the multiply's cuts: the row a + b·c on (o, x, d) and x·d on (x, d); its loop: the multiply on (x, o);
  - the reduction's cut, Horner's step: red on x + d N (the same function as red on B x + d, both below B N, but
    answered by the identity red(x + d N) = x); its loop: the reduction, red on x + o N.

  A law the book poses itself is left to the book. The reading needs the learner's addition, link and comparison,
  so it runs from the first level at which all three are routines, until it reads a law (at most TRIES levels).
"""

from __future__ import annotations

import os
import time

import torch

from integration.lib.libcore import count_of, gen_on, pad, round_settings, show

P = 16                          # the round's full-size registers
COUNT = 1000                    # a count read off digits must be below this; so must the links a reduction takes
MIN_ROWS = 4                    # rows on which a law must be computable
TRIES = 3                       # levels at which the reading is tried before it gives up
OPS = ("add", "mul", "mod")
CUTS = {                        # an operation -> its cuts, as the book's (law, pattern) and the cut's name
    "mul": (("acc.mul", "o,x,d", "row"), ("mul", "x,d", "xd")),
    "mod": (("red", "x+dn", "step"),),
    "add": (),
}
LOOP_OF = {"row": ("mul", "x,o", "mul"), "xd": ("mul", "x,o", "mul"), "step": ("red", "x+on", "red")}
READ = os.environ.get("READ", "")   # "1": every cut and loop of the read operations; else a list of their names


class Reader:
    """The learner's routines on the answer rows at full size: values are digits [n, P], each with the mask of rows
    on which it was computed."""

    def __init__(self, climb, ans):
        dev = climb.device
        self.climb, self.dev, self.n = climb, dev, int(ans[0].shape[0])
        self.st = round_settings(climb.cells, P)
        self.N, self.X, self.Y = (pad(v.to(dev), P).long() for v in ans)
        self.gen = gen_on(dev, 3)
        self.calls = 0

    def _run(self, law, operands):
        self.calls += 1
        return self.climb.ops.run(law, "pair", self.st, self.climb.lib, operands, self.gen, self.dev)

    def _small(self, v):
        return (v[:, 3:] == 0).all(1)            # a count below 1,000: its digits above the third are zero

    def add(self, a, b, m):
        out = torch.zeros_like(a)
        idx = torch.nonzero(m).flatten()
        if idx.numel():
            out[idx] = self._run("add", [a[idx], b[idx]])[0]
        return out, m

    def mul(self, a, b, m):
        m = m & self._small(b)
        out = torch.zeros_like(a)
        times = count_of(b) * m
        for k in range(int(times.max()) if times.numel() else 0):
            more = torch.nonzero(times > k).flatten()
            out[more] = self._run("add", [out[more], a[more]])[0]
        return out, m

    def mod(self, a, b, m):
        out = a.clone()
        below = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        idx = torch.nonzero(m).flatten()
        if idx.numel():
            below[idx] = self._run("lt", [out[idx], b[idx]])[1] == 1
        for _ in range(COUNT):
            idx = torch.nonzero(m & ~below).flatten()
            if not idx.numel():
                break
            out[idx] = self._run("red", [out[idx], b[idx]])[0]
            below[idx] = self._run("lt", [out[idx], b[idx]])[1] == 1
        return out, m & below

    def apply(self, op, a, b, m):
        return getattr(self, op)(a, b, m)

    def passes(self, v, m):
        return int(m.sum()) >= MIN_ROWS and bool(((v == self.Y).all(1) | ~m).all())

    def read(self):
        """(law, its operations, depth, rows it was computable on), or (None, (), None, 0)."""
        allm = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        leaves = [("X", self.X), ("N", self.N)]
        first, seen = [], set()
        for an, a in leaves:
            for bn, b in leaves:
                for op in OPS:
                    v, m = self.apply(op, a, b, allm)
                    law = f"{op}({an},{bn})"
                    if self.passes(v, m):
                        return law, (op,), 1, int(m.sum())
                    k = (tuple((v * m[:, None]).flatten().tolist()), tuple(m.tolist()))
                    if int(m.sum()) >= MIN_ROWS and k not in seen:
                        seen.add(k)
                        first.append((law, op, v, m))
        for law1, op1, c, cm in first:
            for op in OPS:
                for ln, l in leaves:
                    for side in (0, 1):
                        a, b = (c, l) if side == 0 else (l, c)
                        v, m = self.apply(op, a, b, cm)
                        law = f"{op}({law1},{ln})" if side == 0 else f"{op}({ln},{law1})"
                        if self.passes(v, m):
                            return law, tuple(dict.fromkeys((op1, op))), 2, int(m.sum())
        return None, (), None, 0


def _in_book(climb, law, pat, LAW_BOOK):
    return law in climb.laws and pat in climb.patterns and pat in LAW_BOOK[law][1]


def pose(climb, book, LAW_BOOK):
    """The book's [(pattern, laws)], with the cuts and loops of the law read legally added once it is read."""
    pairs = getattr(climb, "_lread_pairs", None)
    if pairs is None and getattr(climb, "_lread_tries", 0) < TRIES and all(
            climb.ops.has(law, "pair") for law in ("add", "red", "lt")) and getattr(climb, "answer_rows", None):
        climb._lread_tries = getattr(climb, "_lread_tries", 0) + 1
        t0 = time.time()
        r = Reader(climb, climb.answer_rows)
        law, ops, depth, rows = r.read()
        names = ({c for cs in CUTS.values() for _, _, c in cs} | {n for _, _, n in LOOP_OF.values()}
                 if READ == "1" else set(READ.split(",")))
        if law is not None:
            pairs = {}
            for op in ops:
                for lw, pat, cut in CUTS.get(op, ()):
                    if cut in names and not _in_book(climb, lw, pat, LAW_BOOK):
                        pairs[(lw, pat)] = cut
            for (lw, pat), cut in list(pairs.items()):
                if cut in LOOP_OF:
                    llw, lpat, lname = LOOP_OF[cut]
                    if lname in names and not _in_book(climb, llw, lpat, LAW_BOOK):
                        pairs.setdefault((llw, lpat), lname)
            for cut, (llw, lpat, lname) in LOOP_OF.items():      # a loop whose cut the book poses
                if lname in names and any(op in ops for op, cs in CUTS.items() if any(c == cut for _, _, c in cs)):
                    if not _in_book(climb, llw, lpat, LAW_BOOK):
                        pairs.setdefault((llw, lpat), lname)
            climb._lread_pairs = pairs
        climb.records.append(dict(task="read", level=climb.level, law=law, ops=list(ops), depth=depth, rows=rows,
                                  calls=r.calls, posed=[f"{lw}/{pat}={c}" for (lw, pat), c in (pairs or {}).items()],
                                  seconds=round(time.time() - t0, 1)))
    out = dict(book)
    for (lw, pat) in (pairs or {}):
        ls = out.setdefault(pat, [])
        if lw not in ls:
            ls.append(lw)
    return list(out.items())


def show_pairs(climb):
    return {f"{lw}/{pat}": c for (lw, pat), c in (getattr(climb, "_lread_pairs", None) or {}).items()}


__all__ = ["Reader", "pose", "show_pairs", "show"]
