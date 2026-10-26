"""Three changes to how the learner picks practice (PICKS.md §2), each behind its own flag:

- `tiescore`: hardest first breaks its ties by the subgoal score on the rest of the way. Once a subgoal has been
  practised, the rest of the way from its routine is the problem left (the "rrem" problem: the target's answer from X
  = what the routine leaves in A, under that name or another candidate's that posed the same problem). The pool is
  scored against that rest with score.py's rule, as SUBGOAL.md §8's own-subgoal rule scores it. The candidate the
  score singles out (a positive key no other candidate shares) wins hardest first's ties at its cost, as the subgoal
  does. The latest practised subgoal's rest first.
- `twin`: a subgoal that is a duplicate of a learnable problem passes its priority to that twin, as a job blocking
  higher-priority jobs inherits their priority (Sha, Rajkumar and Lehoczky 1990). The twin wins ties as the subgoal
  would, and it is practised as soon as it is learnable, in the last pick's place.
- `bigx_score`: for a target with a check at big x (the digits study's), the description length that anchored pairs
  scores looks by (bar3/learner.mdl) also charges every level without an accepted program for the big-x answers it
  does not give: n instances, each costing which instance and its answer below `top`. A level whose program the
  target accepts is charged nothing more, so a look that finds such a program gains that charge. Where no record has
  an accepted level the charge cancels and every score is the frozen one.
"""

from __future__ import annotations

import math

from bar3.learner import mdl as frozen_mdl
from bar3.scale.subgoal import score as SC


def rest_scores(pool, chained, k, k2):
    """The candidate the subgoal score singles out against the rest of the way from the latest practised subgoal's
    routine that the pool poses ({} when none does, or when the score singles nothing out)."""
    for tok in reversed(list(chained)):
        rest = next((c for c in pool if f"rrem/{tok}" in (c.name, *c.also)), None)
        if rest is None:
            continue
        sc = SC.score_pool(pool, rest, k, k2)
        keys = sorted((v["key"] for v in sc.values()), reverse=True)
        if keys and keys[0] > (0, 0) and (len(keys) == 1 or keys[1] < keys[0]):
            name = next(n for n, v in sc.items() if v["key"] == keys[0])
            return dict(rest=rest.name, after=tok, name=name, key=list(keys[0]))
        return dict(rest=rest.name, after=tok, name=None)
    return {}


def twins(subs, status, learnable):
    """The learnable problems that the subgoal is a duplicate of."""
    names = {c.name for c in learnable}
    return {status[s]["of"] for s in subs if status.get(s, {}).get("status") == "duplicate"
            and status[s].get("of") in names}


def mdl(cfg, rec, n_symbols, k):
    """bar3/learner.mdl, and with cfg["bigx_score"] = {n, top}, the big-x charge on every level without an accepted
    program."""
    bx = cfg.get("bigx_score")
    if not bx:
        return frozen_mdl(rec, n_symbols, k)
    NP = rec["NP"]
    tok, exc = math.log2(n_symbols), math.log2(NP) + math.log2(k)
    big = bx["n"] * (math.log2(bx["n"]) + math.log2(bx["top"]))
    best = math.inf
    for d, f in enumerate(rec["best_fit"]):
        if rec.get("found_cost") == d:
            best = min(best, d * tok + (NP - f) * exc)
            continue
        if f == NP:
            f = NP - 1
        best = min(best, d * tok + (NP - f) * exc + big)
    return best


def record(cfg, twin, tie):
    """The round record's `picks` field: what the flags did; nothing when every flag is off."""
    if not (cfg.get("tiescore") or cfg.get("twin") or cfg.get("bigx_score")):
        return {}
    return dict(picks=dict(twin=sorted(twin), rest=tie))
