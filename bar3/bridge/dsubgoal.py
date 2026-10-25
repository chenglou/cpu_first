"""Aim the learner's deeper search (CARRY.md §12): the subgoal study's signal (bar3/scale/subgoal/score.py), adapted to
the pieces of one place of a digit machine.

The subgoal study scores the problems posed on every one of the target's rows: does the target factor through the
problem's answer, as a function of it or one law of the book away? A digit target's pieces are mostly not posed on its
rows. They are restrictions (some of its rows) and widenings (rows it never shows), and what composes them into the
target is the counting machine's other composition, the loop. So the score here asks: does the target factor through
a loop of the candidate?

**The loop view.** A widened candidate's answers are known beyond its rows by its own rule (dwiden.py): with a pinned
register holding k more than its pin, the answers are the same, and a pinned output comes out k more. An input its
answers never depend on, on its rows, is ignored. That gives the candidate's *step* on a machine state, the registers
it does not output left as they were. The target factors through a loop of the candidate when, on every one of the
target's rows, loading a count into a register the candidate pins and applying the step that many times (the
counting machine's `L.r[...]`, r read once) gives the target's answers. The count is one of those a counting machine's
loops build: an input of the target or the base, or a sum or product of two of them, or of such a sum or product and a
third (`L.x[INC.r]` adds x to r, `L.x[L.y[INC.r]]` adds x·y), every value within the machine's cap on its row.

Example: multiply-accumulate's conjunction `0 < C and R + C < 2`, widened, steps as "A + 1 mod B; C one less, plus one
if A wrapped", with R and H ignored. Loading C with C + R·H and stepping C times gives (a + r·h + c) mod B and its
carry on every row, so multiply-accumulate factors through a loop of it.

**The subgoal** is the candidate with an exact loop, when every candidate with one steps the same way on the states the
loops visit (two posings of one piece); of those, the one posed on the most rows. Nothing here knows a law, a carry or
a register's role: only widened candidates have a step beyond their rows, and only the registers they pin can count.
"""

from __future__ import annotations

import itertools

import numpy as np

from bar3.bridge import dproblems as P, slices as S

KEY = P.KEY


class Step:
    """A widened candidate's step on a state {register: value} at base B: the next state, or None where unknown."""

    def __init__(self, c):
        self.c, self.pins, self.outs = c, dict(c.pins), tuple(c.outputs)
        ins = [r for r in c.inputs if r != "N"]
        base = {B: {k: o for k, o in rows.items() if all(k[KEY.index(r)] == v for r, v in self.pins.items())}
                for B, rows in c.rows.items()}
        self.drop = []
        for j in ins:
            if j in self.pins or len({k[KEY.index(j)] for rows in base.values() for k in rows}) < 2:
                continue
            seen, ok = {}, True
            for B, rows in base.items():
                for k, o in rows.items():
                    if seen.setdefault((B,) + tuple(v for i, v in enumerate(k) if KEY[i] != j), o) != o:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                self.drop.append(j)
        self.keep = [r for r in ins if r not in self.pins and r not in self.drop]
        self.table = {(B,) + tuple(k[KEY.index(r)] for r in self.keep): o for B, rows in base.items()
                      for k, o in rows.items()}

    def __call__(self, B, s):
        shift = {}
        for r, v in self.pins.items():
            if s[r] < v:
                return None
            shift[r] = s[r] - v
        o = self.table.get((B,) + tuple(s[r] for r in self.keep))
        if o is None:
            return None
        t = dict(s)
        for reg, val in zip(self.outs, o):
            t[reg] = val + shift.get(reg, 0)
        return t


def target_rows(tgt):
    """The target's rows at the search and check bases: [(B, state, answer, cap)]."""
    out = []
    for bases in (S.SEARCH_BASES, S.CHECK_BASES):
        cap = P.machine_for(bases).cap
        for B in bases:
            for key, ans in sorted(tgt.rows.get(B, {}).items()):
                out.append((B, dict(zip(KEY, key)), tuple(ans), cap))
    return out


def counts(tgt, rows):
    """{name: values on the rows}: the counts a loop prefix builds from the target's inputs and the base, each within
    the machine's cap on every row, one name per distinct column of values."""
    leaves = {r: np.array([B if r == "N" else s[r] for B, s, _, _ in rows], np.int64) for r in tgt.inputs}
    cap = np.array([c for *_, c in rows], np.int64)
    terms = dict(leaves)
    one = {}
    for (an, a), (bn, b) in itertools.combinations_with_replacement(leaves.items(), 2):
        one[f"{an}+{bn}"], one[f"{an}·{bn}"] = a + b, a * b
    terms.update(one)
    for (sn, s), (cn, c) in itertools.product(one.items(), leaves.items()):
        terms[f"({sn})+{cn}"], terms[f"({sn})·{cn}"] = s + c, s * c
    out, seen = {}, set()
    for name, v in terms.items():
        sig = v.tobytes()
        if sig not in seen and ((v >= 0) & (v < cap)).all():
            seen.add(sig)
            out[name] = v
    return out


def loops(step, z, n, rows, outs, order):
    """Does loading n into z and stepping n times give the target's answers (its registers `outs`) on every row of
    `order` (a sample of rows first, so that most failures are cheap)? A signature of the states visited, or None."""
    trail = []
    for i in order:
        B, s, ans, _ = rows[i]
        s = dict(s)
        s[z] = int(n[i])
        for _ in range(int(n[i])):
            s = step(B, s)
            if s is None:
                return None
            trail.append((B, tuple(s[r] for r in KEY)))
        if tuple(s[o] for o in outs) != ans:
            return None
    return hash(tuple(trail))


def score_pool(pool, tgt, sample=24):
    """{candidate name: dict(loop=its loop, rows=its rows at the search bases, sig=its trail, drop=inputs it
    ignores)} for every widened candidate through a loop of which the target factors."""
    rows = target_rows(tgt)
    cnt = counts(tgt, rows)
    first = np.random.default_rng(0).choice(len(rows), min(sample, len(rows)), replace=False).tolist()
    order = first + sorted(set(range(len(rows))) - set(first))
    out = {}
    for c in pool:
        if not getattr(c, "pins", None):
            continue
        step = Step(c)
        for z, (name, n) in itertools.product(sorted(step.pins, key=KEY.index), cnt.items()):
            if loops(step, z, n, rows, tgt.outputs, first) is None:
                continue
            sig = loops(step, z, n, rows, tgt.outputs, order)
            if sig is not None:
                out[c.name] = dict(loop=f"{z} := {name}, then {z} times the step", rows=c.n(S.SEARCH_BASES), sig=sig,
                                   drop=step.drop)
                break
    return out


def subgoal_of(pool, tgt):
    """(the pool's loop scores, the subgoal's name or None)."""
    sc = score_pool(pool, tgt)
    if not sc or len({v["sig"] for v in sc.values()}) > 1:
        return sc, None
    names = [c.name for c in pool if c.name in sc]
    return sc, max(names, key=lambda n: (sc[n]["rows"], -names.index(n)))
