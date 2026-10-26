"""Bar 3's subgoal score (bar3/scale/subgoal/score.py, SUBGOAL.md §1) on the digit machine's rows (COMPOSE2.md §2.2).

score.py's operations, table view and law view are copied here unchanged in what they compute (score.py imports
library/inventing, which the runs' image leaves out; `check_subgoal.py` checks the copies against score.py on random
rows). One change, the reason this study ports the score: score.py turns the whole law view off for a target that is
itself a law of depth <= 2 of X, N and the constants, and x^2 mod N = mod(mul(X, X), N) is one. Here the rule is
applied per depth: depth-1 laws are on unless the target is a depth-1 law of the inputs alone, depth-2 laws unless it
is one of depth <= 2. For x^2 mod N, depth 1 stays on (no single law of X and N is it), and mod(g, N) for g = x^2 is
one such law.

Rows are (x', N', t, cap) per row: the target's (or a rest's) probes and acceptance rows, cap the register's capacity
(10^P at the row's setting). A candidate's answers on those rows are its g.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

BAD = -1
OPS2 = {
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


@dataclass
class Rows:
    keys: list
    x: np.ndarray
    N: np.ndarray
    t: np.ndarray
    cap: np.ndarray
    consts: list


def rows_from(xs, Ns, ts, caps):
    """Rows, each (x, N) once (the first), as score.target_rows keeps them."""
    keys, t, cap, seen = [], [], [], set()
    for key, v, c in zip(zip(xs, Ns), ts, caps):
        if key not in seen:
            seen.add(key)
            keys.append(key)
            t.append(v)
            cap.append(c)
    return Rows(keys, np.array([a for a, _ in keys], np.int64), np.array([b for _, b in keys], np.int64),
                np.array(t, np.int64), np.array(cap, np.int64), [])


def first_rows(keys, g):
    """g's answers on each key's first row (the Rows' keys order), from per-row answers over (x, N) keys."""
    d = {}
    for key, v in zip(keys, g):
        d.setdefault(key, v)
    return d


def table_view(g, rows):
    """score.table_view."""
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


def laws(leaves, rows, need, depth=2):
    """score._laws, stopped after depth 1 when depth is 1."""
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
    if depth < 2:
        return None
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


def law_depths(rows):
    """The depths the law view is on at: depth d is off when the target is a law of depth <= d of X and N alone."""
    leaves = {"X": rows.x, "N": rows.N}
    return [d for d in (1, 2) if laws(leaves, rows, None, d) is None]


def law_view(g, rows, depths):
    leaves = {"g": g, "X": rows.x, "N": rows.N}
    for d in depths:
        law = laws(leaves, rows, ("g",), d)
        if law is not None:
            return law
    return None


def score(cands, rows):
    """{name: dict(law, table, how, pure, key)} for candidates {name: g on rows' keys}, except the target posed again;
    key (a law is exact, rows the table saves), larger first (score.score_pool's)."""
    depths = law_depths(rows)
    out = {}
    for name, g in cands.items():
        g = np.asarray(g, np.int64)
        if (g == rows.t).all():
            continue
        saved, how, pure = table_view(g, rows)
        law = law_view(g, rows, depths) if depths else None
        out[name] = dict(law=law, table=saved, how=how, pure=pure, key=(int(law is not None), saved))
    return out


def singled_out(sc):
    """The candidate with the best positive key that no other candidate shares, else None (SUBGOAL.md's subgoal)."""
    keys = sorted((v["key"] for v in sc.values()), reverse=True)
    if keys and keys[0] > (0, 0) and (len(keys) == 1 or keys[1] < keys[0]):
        return next(n for n, v in sc.items() if v["key"] == keys[0])
    return None
