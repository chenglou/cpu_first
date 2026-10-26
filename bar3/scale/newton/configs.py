"""The two-source study's configs (NEWTON.md §2): the held-out test's frozen configs (bar3/scale/subgoal/configs.py:
lp_clean_deep_sub, hardest first, and ct2a2_clean_deep_sub, anchored pairs) with both sources on, `_ns`. Without
them the configs are the frozen ones, whose earlier records are the runs without the sources."""

from __future__ import annotations

from bar3.scale.subgoal.configs import CONFIGS as SUB

CONFIGS = {}
for _r in ("lp_clean", "ct2a2_clean"):
    _c = SUB[f"{_r}_deep_sub"]
    CONFIGS[f"{_r}_deep_sub"] = _c
    CONFIGS[f"{_r}_deep_sub_ns"] = dict(_c, sources=list(_c["sources"]) + ["rsum", "offset"])
