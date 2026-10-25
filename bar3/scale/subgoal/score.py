"""A signal that sees a subgoal before it is solved (SUBGOAL.md §1): how much of the target's answers a candidate's
answers determine, read off the rows alone.

A candidate g is scored when it is posed on every one of the target's instances (X = x and N, nothing else set, its
answer in A, at both sizes): a problem posed on other inputs, or on part of them, is not a step on the way to this
target. Two views:
- table: group the target's rows by (g's answer, N), x dropped; by (g's answer, x), N dropped; or by g's answer
  alone. When every class wants one answer, the target is a function of that grouping and needs one entry per
  class: the view's value is the rows it saves (rows − classes), from the best grouping that determines the target;
  0 when none does (a grouping that leaves some class mixed gets no credit: the rows it merges where the target is
  constant anyway, e.g. 10·x² mod N at N = 5 and 10, are the target's own structure, not g's). 10·x² mod N and
  (10·x² + 1) mod N are functions of (x² mod N, N), so x drops out; x³ mod N is not one (x and N − x share
  x² mod N, not x³).
- law: the target is one law of the book away from g: t = op(a, b) or op(op'(a, b), c), for op, op' among the law
  book's operations (bar3/problems.OPS2, OPS1) and a, b, c among g, X, N and the constants the data shows
  (sources.constants), g among them, every value on the way below the machine's cap. x³ mod N = (x² mod N)·X mod N.
  Exact on every row, or nothing. Off for a target that is such a law of X, N and the constants alone (then every
  copy of X passes).
A candidate that equals the target on all its rows is the target posed again (a near miss's remainder can be, when
its value permutes x: X = N − x on a target symmetric in x and N − x), not a step towards it: it is left out.
`score_pool` orders by the law first, then by the table's rows saved.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from library.inventing.jobs import machine_for
from bar3 import problems as P
from bar3.scale import sources as S

BAD = -1                         # an undefined value (division by 0); every check below rejects it

OPS2 = {                         # bar3/problems.OPS2 on arrays (checked against it in `_check_ops`)
    "add": lambda a, b: a + b,
    "monus": lambda a, b: np.maximum(a - b, 0),
    "absdiff": lambda a, b: np.abs(a - b),
    "mul": lambda a, b: a * b,
    "div": lambda a, b: np.where(b != 0, a // np.where(b != 0, b, 1), BAD),
    "mod": lambda a, b: np.where(b != 0, a % np.where(b != 0, b, 1), BAD),
    "min": np.minimum,
    "max": np.maximum,
    "lt": lambda a, b: (a < b).astype(np.int64),
    "le": lambda a, b: (a <= b).astype(np.int64),
    "eq": lambda a, b: (a == b).astype(np.int64),
    "gcd": np.gcd,
}
OPS1 = {"succ": lambda a: a + 1, "pred": lambda a: np.maximum(a - 1, 0)}


def _check_ops():
    vals = np.arange(0, 9)
    for name, f in OPS2.items():
        ref, _ = P.OPS2[name]
        for a, b in itertools.product(vals, vals):
            want = ref(int(a), int(b))
            got = int(f(np.array([a]), np.array([b]))[0])
            assert (want is None and got == BAD) or got == want, (name, a, b, want, got)
    for name, f in OPS1.items():
        ref, _ = P.OPS1[name]
        assert all(int(f(np.array([a]))[0]) == ref(int(a)) for a in vals), name


@dataclass
class Rows:
    """The target's rows at both sizes (each (x, N) once) and the machine's cap on each row."""
    keys: list
    x: np.ndarray
    N: np.ndarray
    t: np.ndarray
    cap: np.ndarray
    consts: list


def target_rows(tgt, k, k2=P.K2, gen=P.target_gen):
    keys, t, cap = [], [], []
    seen = set()
    for xs, Ns, w, size in ((tgt.xs, tgt.Ns, tgt.want, k), (tgt.cx, tgt.cN, tgt.cw, k2)):
        c = machine_for(size).cap
        for key, v in zip(zip(xs, Ns), w):
            if key not in seen:
                seen.add(key)
                keys.append(key)
                t.append(v)
                cap.append(c)
    consts = sorted({c.want[0] for c in S.constants(gen, k, k2, regs=("A",))})
    return Rows(keys, np.array([a for a, _ in keys], np.int64), np.array([b for _, b in keys], np.int64),
                np.array(t, np.int64), np.array(cap, np.int64), consts)


def on_target(c, rows):
    """g's answers on every one of the target's rows, or None when c is not posed on all of them with X and N."""
    if getattr(c, "init", None) or getattr(c, "every", ()) or tuple(getattr(c, "outs", ("A",))) != ("A",):
        return None
    d = {}
    for xs, Ns, w in ((c.xs, c.Ns, c.want), (c.cx, c.cN, c.cw)):
        d.update(zip(zip(xs, Ns), w))
    if any(key not in d for key in rows.keys):
        return None
    return np.array([d[key] for key in rows.keys], np.int64)


def table_view(g, rows):
    """(rows saved, grouping, rows in one-answer classes) for the best grouping of the target's rows that determines
    the target, by (g, N), (g, x) or g alone; (0, "", most rows in one-answer classes) when none does."""
    best, how, pure_rows = 0, "", 0
    for name, other in (("g,N", rows.N), ("g,x", rows.x), ("g", None)):
        cls = {}
        for i in range(len(rows.t)):
            key = (int(g[i]), int(other[i])) if other is not None else int(g[i])
            n, ans = cls.get(key, (0, set()))
            ans.add(int(rows.t[i]))
            cls[key] = (n + 1, ans)
        pure = sum(n for n, ans in cls.values() if len(ans) == 1)
        pure_rows = max(pure_rows, pure)
        if pure == len(rows.t) and len(rows.t) - len(cls) > best:
            best, how = len(rows.t) - len(cls), name
    return best, how, pure_rows


def _laws(leaves, rows, need):
    """The first law of depth <= 2 over `leaves` ({name: values}) that is exact on the target's rows, every value on
    the way within [0, cap), and uses a leaf of `need` (None: any); its description, else None."""
    t, cap = rows.t, rows.cap
    ok = lambda v: ((v >= 0) & (v < cap)).all(axis=-1)            # noqa: E731
    names, vals = [], []
    for (an, a), (bn, b) in itertools.product(leaves.items(), repeat=2):
        for op, f in OPS2.items():
            names.append(f"{op}({an},{bn})")
            vals.append(f(a, b))
    for an, a in leaves.items():
        for op, f in OPS1.items():
            names.append(f"{op}({an})")
            vals.append(f(a))
    V = np.stack(vals)
    valid = ok(V)
    uses = np.array([need is None or any(f"({n}," in s or f",{n})" in s or f"({n})" in s for n in need)
                     for s in names])
    hit = valid & uses & (V == t).all(axis=1)
    if hit.any():
        return names[int(np.argmax(hit))]
    V, names, uses = V[valid], [s for s, v in zip(names, valid) if v], uses[valid]
    for op, f in OPS2.items():
        for bn, b in leaves.items():
            for side in (0, 1):
                W = f(V, b[None, :]) if side == 0 else f(b[None, :], V)
                hit = ok(W) & (W == t).all(axis=1) & (uses | (need is None) | (bn in (need or ())))
                if hit.any():
                    s = names[int(np.argmax(hit))]
                    return f"{op}({s},{bn})" if side == 0 else f"{op}({bn},{s})"
    for op, f in OPS1.items():
        W = f(V)
        hit = ok(W) & (W == t).all(axis=1) & uses
        if hit.any():
            return f"{op}({names[int(np.argmax(hit))]})"
    return None


def inputs_law(rows):
    """Is the target itself a law of depth <= 2 over X, N and the data's constants? (Then the law view is off.)"""
    leaves = {"X": rows.x, "N": rows.N, **{str(c): np.full_like(rows.x, c) for c in rows.consts}}
    return _laws(leaves, rows, None)


def law_view(g, rows):
    leaves = {"g": g, "X": rows.x, "N": rows.N, **{str(c): np.full_like(rows.x, c) for c in rows.consts}}
    return _laws(leaves, rows, ("g",))


def score_pool(pool, tgt, k, k2=P.K2):
    """{candidate name: dict(law, table, how, pure, key)} for every candidate posed on all the target's instances
    (except the target posed again); key orders them, larger first: (a law is exact, rows the table saves)."""
    rows = target_rows(tgt, k, k2)
    law_on = inputs_law(rows) is None
    out = {}
    for c in pool:
        g = on_target(c, rows)
        if g is None or (g == rows.t).all():
            continue
        saved, how, pure = table_view(g, rows)
        law = law_view(g, rows) if law_on else None
        out[c.name] = dict(law=law, table=saved, how=how, pure=pure, key=(int(law is not None), saved))
    return out


if __name__ == "__main__":
    _check_ops()
    print("OPS2 and OPS1 agree with bar3/problems")
