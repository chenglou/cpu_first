"""Widen the pins (CARRY.md): beside a problem whose rows hold an input register at one value, pose the same problem
with that register free to hold more, the extra passed through.

A problem's rows can *pin* an input register: hold it at one value on every row. The target add restricted to R < C
holds R at 0 and C at 1 ("a carry of 1 comes in and nothing is added"). A program may then use the register as the
constant it always is there: clear it, or count it down to zero. Programs that do and programs that don't are equally
right on the rows; composed into a larger program, where the register holds more, only some of them still work. The
widening tells them apart: the problem's rows, and each row again with its pinned registers shifted up by 1 or 2
(every combination, not all 0), wanting the same outputs, an output that is a shifted register shifted by as much.
What a register holds beyond its pin is not the problem's to use: an input that is not an output must not change the
answer, and an output passes it through.

Nothing here knows a law, a carry or a register's role: the rule applies to every candidate with a pinned input (other
than the base N, the machine's size), whatever its source.

Example: add where R < C is (a, 0, 1) -> ((a + 1) mod B, [a + 1 = B]). Its widening adds (a, 0, 3) ->
((a + 1) mod B, 2 + [a + 1 = B]) and (a, 2, 1) -> ((a + 1) mod B, [a + 1 = B]), among others. `INC.A T.mod L.A[DEC.C]`
(count C down A times: to 0 unless A wrapped to 0) is right on the rows and wrong on the widening;
`INC.A T.mod Z.A[INC.C] DEC.C` (C one less, plus one if A wrapped) is right on both.
"""

from __future__ import annotations

import itertools

from bar3.bridge import dproblems as P

SHIFTS = (1, 2)


def pinned(c):
    """{register: value} for the input registers (not N) that hold one value on every row of c, at every base."""
    vals = {r: set() for r in c.inputs if r != "N"}
    for rows in c.rows.values():
        for key in rows:
            for r in vals:
                vals[r].add(key[P.KEY.index(r)])
    return {r: next(iter(v)) for r, v in vals.items() if len(v) == 1}


def widen(c):
    """c's widening (a Cand), or None when c pins nothing or the widening is not well posed."""
    regs = sorted(pinned(c), key=P.KEY.index)
    if not regs:
        return None
    rows = {}
    for B, r in c.rows.items():
        rr = dict(r)
        for key, out in r.items():
            for ks in itertools.product((0,) + SHIFTS, repeat=len(regs)):
                if not any(ks):
                    continue
                k, o = list(key), list(out)
                for reg, s in zip(regs, ks):
                    k[P.KEY.index(reg)] += s
                    if reg in c.outputs:
                        o[c.outputs.index(reg)] += s
                rr[tuple(k)] = tuple(o)
        rows[B] = rr
    w = P.Cand(f"widen/{c.name}", "widen", f"{c.name}, with {', '.join(regs)} free to hold more (passed through)",
               c.inputs, c.outputs, rows, parent=c.name)
    return w if P.well_posed(w) else None


def widenings(cands):
    """The widening of every candidate that pins an input."""
    return [w for w in (widen(c) for c in cands) if w is not None]


def conjunctions(rests, known=()):
    """The target's rows where two of its restrictions both hold, when that pins an input register neither pins alone
    (the `conj` flag): a single comparison cannot pick out, say, R = 0 and C = 1 from multiply-accumulate's rows,
    where C counts to B - 1; two can (0 < C and R + C < 2). Kept when well posed and not already a candidate."""
    seen = {c.key() for c in list(rests) + list(known)}
    out = []
    for a, b in itertools.combinations(rests, 2):
        rows = {B: {k: o for k, o in r.items() if k in b.rows.get(B, {})} for B, r in a.rows.items()}
        c = P.Cand(f"{a.name}&{b.name.rsplit('/', 1)[-1]}", "restrict",
                   f"{a.formula} and {b.formula.split(' where ')[-1]}", a.inputs, a.outputs, rows, parent=a.parent)
        if not P.well_posed(c) or c.key() in seen:
            continue
        seen.add(c.key())
        if len(pinned(c)) > max(len(pinned(a)), len(pinned(b))):
            out.append(c)
    return out
