"""This study's configs (REACH.md §2): the earlier studies' configs, for the replays with every flag off, and each with
one of this study's changes:
- `_deep`: the deeper look at the digits source's steps; on a digits config it comes with the picks study's big-x
  score (`_bigx_deep`), since the deeper look is there for that score to act on;
- `_twin5`: the subgoal's twin practised as a fifth pick.
The base configs are the held-out test's frozen ones (`<rule>_deep_sub`, no source on), the picks study's
10·x² + 1 config (`lp_clean_deep_sub_ns_rc_tie`: both sources, the row check and the tie-break), both run by
learn_newton.py, and the digits study's base-2 configs with the source (`<rule>_b_src`), run by learn_digits.py.
LEARNER[config] names the copy."""

from __future__ import annotations

from bar3.scale.picks.configs import BIGX, CONFIGS as PK

CONFIGS, LEARNER = {}, {}
for _base, _l in [(f"{r}_deep_sub", "newton") for r in ("lp_clean", "ct2a2_clean")] + \
                 [("lp_clean_deep_sub_ns_rc_tie", "newton")] + \
                 [(f"{r}_b_src", "digits") for r in ("lp_clean", "ct2a2_clean")]:
    _c = PK[_base]
    variants = {"": _c, "_twin5": dict(_c, twin5=True)}
    if _l == "digits":
        variants["_bigx_deep"] = dict(_c, bigx_score=BIGX, deeplook=True)
    else:
        variants["_deep"] = dict(_c, deeplook=True)
    for _v, _cv in variants.items():
        CONFIGS[_base + _v], LEARNER[_base + _v] = _cv, _l
