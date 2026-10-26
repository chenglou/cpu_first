"""Two more practice sources for the bar-3 learner (NEWTON.md §2): the first two moves of Newton's forward
differences, taking away the constant term and differencing, whose inverse is the running sum. Each is generic: it
reads the data's table of answers, or a posed problem's rows, never the formula.

- `running_sum`: the answer at x + 1 from the answer at x and x itself, y(x + 1) = f(y(x), x, N): the step of a
  running sum, y(x + 1) = y(x) + Δy(x). bar3/scale/sources.recurrences poses y(x + 1) from y(x) alone, when the data
  says that is a function; with x beside it the rows are always one, so it is posed whenever the data has rows. The
  triangular numbers' step is (Y + X + 1) mod N. Posed in every placement: y(x) in A, U or T, updated in place, and x
  in X, U or T.
- `offsets`: a rest of the way's answer less its answer at X = 0 (for the same N), mod N, as SUBGOAL.md §8 defined it
  (bar3/scale/subgoal/rank_rest.offset), for each rest of the way posed in the pool: sources.routine_remainders'
  problems and the near misses' remainders, the target's answer from X = what a routine or a near miss leaves in A.
  The rest of the way from x² mod N for 10·x² + 1, (10X + 1) mod N, gives 10X mod N. Not posed when the answer at
  X = 0 is 0 for every N: the offset is then the problem itself. Not posed for the target itself: its offset,
  10·x² mod N for 10·x² + 1, is one law (add 1, reduce) from the target, so the subgoal's score puts it above
  x² mod N and takes the subgoal's deeper probe from the piece the rest of the way starts from (NEWTON.md §3.1).
"""

from __future__ import annotations

import numpy as np

from bar3 import problems as P
from bar3.scale import sources as S

STATE = ("A", "U", "T")
ARG = ("X", "U", "T")
REST = ("rrem", "remainder")       # the sources whose problems are a rest of the way


def running_sum(gen, k, k2=P.K2, parent="target"):
    """y(x + 1) from y(x) and x on the data's table (every x < N for the N it holds), in every placement."""
    def rows(size):
        return [((a, x, n), b) for n, ys in S.table(gen, size).items() for x, (a, b) in enumerate(zip(ys, ys[1:]))]

    if len(rows(k)) < 4:
        return []
    out = []
    for s in STATE:
        for a in ARG:
            if a == s:
                continue

            def g(size, s=s, a=a):
                rs = rows(size)
                ys, xs = [r[0][0] for r in rs], [r[0][1] for r in rs]
                init = {s: ys} if a == "X" else {s: ys, a: xs}
                return xs if a == "X" else [0] * len(rs), [r[0][2] for r in rs], init, [r[1] for r in rs]
            c = S.make(f"rsum/{s}.{a}", "rsum", f"y(x + 1) from y(x) in {s} and x in {a}", g, k, k2, outs=(s,),
                       parent=parent)
            if c:
                out.append(c)
    return out


def offset(c, k, k2=P.K2):
    """c's answer less its answer at X = 0 for the same N, mod N, on c's instances whose N has X = 0 among them."""
    def gen(size):
        xs, Ns, w = ((c.xs, c.Ns, c.want) if size == k else (c.cx, c.cN, c.cw))
        at0 = {n: v for a, n, v in zip(xs, Ns, w) if a == 0}
        if not any(at0.values()):
            return None
        keep = [i for i, n in enumerate(Ns) if n in at0]
        return ([xs[i] for i in keep], [Ns[i] for i in keep],
                np.array([(w[i] - at0[Ns[i]]) % Ns[i] for i in keep]))
    return P.make(f"offset/{c.name}", "offset", f"{c.name}: its answer less its answer at X = 0, mod N", gen, k, k2,
                  parent=c.name)


def offsets(pool, k, k2=P.K2):
    """The offset of each rest of the way in the pool (plain problems only: X and N in, the answer in A)."""
    out = []
    for c in [c for c in pool if c.source in REST and not getattr(c, "init", None)]:
        o = offset(c, k, k2)
        if o:
            out.append(o)
    return out
