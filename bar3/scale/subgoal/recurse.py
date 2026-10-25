"""The recursive subgoal (SUBGOAL.md §8): a subgoal that its deeper probe leaves unsolved is treated as a target in
its own right. Its instances get the sources the target's get, posed from its rows instead of the target's: the near
misses of its own probe (the values they hold and the rest of the way from each), its restrictions, projections,
recurrences and constants, and the rest of the way from each routine. The laws and the dreams are posed on the
data's pairs, which hold every instance of the subgoal (its X is below N), so they are scored as the pool already
has them. The whole lot is scored against the subgoal by score.py's rule, with the subgoal's rows and the constants
its data shows, and the candidate the score singles out (the best key, held by no other) is the subgoal's subgoal.

Nothing here reads the formula: the subgoal's rows, the routines and the data's pairs are all it sees. score.py is
used as it is (its functions take the rows); only the rows and the constants are the subgoal's.
"""

from __future__ import annotations

import numpy as np

from bar3 import problems as P
from bar3.scale import sources as S
from bar3.scale.subgoal import score as SC


def posable(c):
    """Can c be treated as a target: x in X and N in N, nothing else set, one answer in A?"""
    return not (getattr(c, "init", None) or getattr(c, "every", ()) or tuple(getattr(c, "outs", ("A",))) != ("A",))


def gen_of(c, k):
    """c's instances as a generator, size -> (xs, Ns, want): its search instances at k, its check instances else."""
    def gen(size):
        if size == k:
            return np.array(c.xs, np.int64), np.array(c.Ns, np.int64), np.array(c.want, np.int64)
        return np.array(c.cx, np.int64), np.array(c.cN, np.int64), np.array(c.cw, np.int64)
    return gen


def pose_on(c, lib, near, k, k2, src):
    """The sources of learn.run's step 2 that read the target's rows, posed from c's rows instead (`near`: the near
    misses of c's own probe), each renamed `sub[<c>]/<name>` so that it cannot take a pool candidate's name."""
    gen, tag = gen_of(c, k), c.name
    out = []
    if "restrict" in src:
        out += P.restrictions(gen, k, k2, parent=tag)
    if "project" in src:
        out += P.projections(gen, k, k2, parent=tag)
    if "value" in src or "remainder" in src:
        out += [d for d in P.from_near_misses(near, lib, k, k2, gen=gen, parent=tag)
                if d.source in src or (d.source == "restrict" and "remainder" in src)]
    if "recur" in src:
        out += S.recurrences(gen, k, k2, parent=tag)
    if "const" in src:
        out += S.constants(gen, k, k2, parent=tag)
    if "rrem" in src:
        out += S.routine_remainders(lib, k, k2, gen=gen)
    for d in out:
        d.name = f"sub[{tag}]/{d.name}"
    return out


def score_against(pool, c, k, k2):
    """score.score_pool with c as the target: {name: dict(law, table, how, pure, key)} for every candidate posed on
    all of c's instances, except c posed again."""
    rows = SC.target_rows(c, k, k2, gen=gen_of(c, k))
    law_on = SC.inputs_law(rows) is None
    out = {}
    for d in pool:
        g = SC.on_target(d, rows)
        if g is None or (g == rows.t).all():
            continue
        saved, how, pure = SC.table_view(g, rows)
        law = SC.law_view(g, rows) if law_on else None
        out[d.name] = dict(law=law, table=saved, how=how, pure=pure, key=(int(law is not None), saved))
    return out


def singled_out(scores):
    """The candidate with the best key, when that key is positive and no other candidate shares it; else None."""
    keys = sorted((v["key"] for v in scores.values()), reverse=True)
    if keys and keys[0] > (0, 0) and (len(keys) == 1 or keys[1] < keys[0]):
        return next(n for n, v in scores.items() if v["key"] == keys[0])
    return None
