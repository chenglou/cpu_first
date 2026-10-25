"""The subgoal study's configs (SUBGOAL.md §3): the scale study's `<rule>_deep` (every lever and the depth-2 law book,
bar3/scale/configs.py) with the subgoal signal (learn.py): `_sub`, the subgoal's deeper probe and its ties;
`_subpick`, those and the subgoal practised as soon as it is learnable; `_subchain`, `_sub` and, once a subgoal is
practised, the rest of the way from its routine as the next subgoal; `_subnoties`, `_sub` with pool order for the
subgoal's ties too (the deeper probe alone). For rule in lp_clean (hardest first) and
ct2a2_clean (pairs with one member among the two hardest)."""

from __future__ import annotations

from bar3 import learner as L
from bar3.scale.configs import CONFIGS as SCALE

DEEP = dict(L.PROBE_A, max_states=10_000_000, time_limit=3600)   # streams one level beyond the 1M-state probe's
CONFIGS = {}
for _r in ("lp_clean", "ct2a2_clean"):
    CONFIGS[f"{_r}_deep"] = SCALE[f"{_r}_deep"]                  # the control: no subgoal (= bar3/scale/learn.run)
    CONFIGS[f"{_r}_deep_sub"] = dict(SCALE[f"{_r}_deep"], subgoal=dict(deep=DEEP))
    CONFIGS[f"{_r}_deep_subpick"] = dict(SCALE[f"{_r}_deep"], subgoal=dict(deep=DEEP, pick=True))
    CONFIGS[f"{_r}_deep_subchain"] = dict(SCALE[f"{_r}_deep"], subgoal=dict(deep=DEEP, chain=True))
    CONFIGS[f"{_r}_deep_subnoties"] = dict(SCALE[f"{_r}_deep"], subgoal=dict(deep=DEEP, ties=False))
