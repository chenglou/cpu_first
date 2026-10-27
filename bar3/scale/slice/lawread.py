"""The slicing source's law reader (SLICE.md §2.1): the target as a law over X, N and the data's constants, the reading
the subgoal score's law view makes at depth <= 2 (bar3/scale/compose2/subgoal_c2.py, score.py's operations), taken to
depth 4 so that x^3 mod N and 10 x^2 + 1 mod N are read too.

- **Laws** are left-deep chains: an operation of score.py's on two leaves, then on the chain so far and a leaf (either
  side), or succ and pred of the chain, up to `depth` operations.
- **Leaves** are X, N and the data's constants: for x = 0 and for x = 1, the one c >= 1 below the registers' capacity
  with c mod N equal to the answer on every such row (bar3/scale/sources.py's constants), and the difference of the
  two.
- **A value is kept** only when it is in [0, the row's capacity) on every row, and only the first chain for each value
  on the rows (observational equivalence). The first chain, by depth and then in a fixed order, whose value is the
  answer on every row is the reading.

Pure numpy: the same arithmetic the law view uses, on the target's own rows.
"""

from __future__ import annotations

import numpy as np

from bar3.scale.compose2.subgoal_c2 import OPS1, OPS2


def constants(x, N, t, cap):
    """The data's constants: for x = 0 and x = 1 (at least 3 rows each), the smallest c >= 1 below every capacity with
    c mod N = t on those rows; and their difference when it is >= 1."""
    x, N, t, cap = (np.asarray(v, np.int64) for v in (x, N, t, cap))
    out = []
    for x0 in (0, 1):
        m = x == x0
        if m.sum() < 3:
            out.append(None)
            continue
        hi = int(cap[m].min())
        c = next((c for c in range(1, min(hi, 100_000)) if (c % N[m] == t[m]).all()), None)
        out.append(c)
    cs = [c for c in out if c is not None]
    if None not in out and out[1] - out[0] >= 1:
        cs.append(out[1] - out[0])
    return list(dict.fromkeys(cs))


def read(x, N, t, cap, depth=4, consts=None, log=None):
    """(the law as a string, its operations, its depth), or (None, [], None)."""
    x, N, t, cap = (np.asarray(v, np.int64) for v in (x, N, t, cap))
    consts = constants(x, N, t, cap) if consts is None else consts
    leaves = {"X": x, "N": N}
    for c in consts:
        leaves[str(c)] = np.full_like(x, c)
    seen = set()

    def keep(v):
        if not ((v >= 0) & (v < cap)).all():
            return False
        k = v.tobytes()
        if k in seen:
            return False
        seen.add(k)
        return True

    hit = lambda v: bool((v == t).all())                                  # noqa: E731
    for v in leaves.values():
        seen.add(v.tobytes())
    frontier = []                                                         # [(law, ops, value)]
    for (an, a), (bn, b) in ((p, q) for p in leaves.items() for q in leaves.items()):
        for op, f in OPS2.items():
            v = f(a, b)
            if keep(v):
                frontier.append((f"{op}({an},{bn})", (op,), v))
                if hit(v):
                    return frontier[-1][0], [op], 1
    for an, a in leaves.items():
        for op, f in OPS1.items():
            v = f(a)
            if keep(v):
                frontier.append((f"{op}({an})", (op,), v))
                if hit(v):
                    return frontier[-1][0], [op], 1
    for d in range(2, depth + 1):
        if log:
            log(f"depth {d}: {len(frontier)} chains")
        nxt = []
        for s, ops, a in frontier:
            for op, f in OPS2.items():
                for bn, b in leaves.items():
                    for side in (0, 1):
                        v = f(a, b) if side == 0 else f(b, a)
                        if keep(v):
                            law = f"{op}({s},{bn})" if side == 0 else f"{op}({bn},{s})"
                            nxt.append((law, ops + (op,), v))
                            if hit(v):
                                return law, list(dict.fromkeys(ops + (op,))), d
            for op, f in OPS1.items():
                v = f(a)
                if keep(v):
                    nxt.append((f"{op}({s})", ops + (op,), v))
                    if hit(v):
                        return nxt[-1][0], list(dict.fromkeys(ops + (op,))), d
        frontier = nxt
    return None, [], None
