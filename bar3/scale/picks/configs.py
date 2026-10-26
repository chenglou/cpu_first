"""This study's configs (PICKS.md §2). The earlier studies' configs, for the replays with every flag off, and each
with one of this study's changes:
- `_rc_tie`: the row check and the tie-break by the subgoal score (the tie-break needs the row check: without it
  10X mod N is never learnable);
- `_twin`: the subgoal's priority passed to its twin;
- `_bigx`: anchored pairs' score charging for the big-x check, on the digits configs only (the held-out targets
  have no such check).
Which learner copy runs a config is LEARNER[config]: learn_newton.py runs the held-out test's frozen configs
(`<rule>_deep_sub`, no source on) and the two-source study's (`_ns`); learn_digits.py runs the digits study's base-2
configs with the source (`<rule>_b_src`)."""

from __future__ import annotations

from bar3.scale.digits import targets as DT
from bar3.scale.digits.configs import CONFIGS as DIG
from bar3.scale.newton.configs import CONFIGS as NEW

BIGX = dict(n=DT.BIG_N * DT.BIG_PER_N, top=DT.BIG_N)        # 200 instances, answers below N <= 10
CONFIGS, LEARNER = {}, {}
for _r in ("lp_clean", "ct2a2_clean"):
    for _name, _c, _l in ((f"{_r}_deep_sub", NEW[f"{_r}_deep_sub"], "newton"),
                          (f"{_r}_deep_sub_ns", NEW[f"{_r}_deep_sub_ns"], "newton"),
                          (f"{_r}_b_src", DIG[f"{_r}_b_src"], "digits")):
        variants = {"": _c, "_rc_tie": dict(_c, rowcheck=True, tiescore=True), "_twin": dict(_c, twin=True)}
        if _l == "digits":
            variants["_bigx"] = dict(_c, bigx_score=BIGX)
        for _v, _cv in variants.items():
            CONFIGS[_name + _v], LEARNER[_name + _v] = _cv, _l
