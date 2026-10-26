"""The digits study's configs (DIGITS.md §2): the frozen learner's (bar3/scale/subgoal/configs.py: lp_clean_deep_sub,
hardest first, and ct2a2_clean_deep_sub, anchored pairs), each with x in one representation (cfg "base": 1 unary,
2 binary, 10 decimal, the digit block of machine.py) and with the one-more-digit source (`_src`) or without it, the
control (`_ctl`). Names: <rule>_<u|b|d>_<src|ctl>."""

from __future__ import annotations

from bar3.scale.subgoal.configs import CONFIGS as SUB

REPS = {"u": 1, "b": 2, "d": 10}
CONFIGS = {}
for _r in ("lp_clean", "ct2a2_clean"):
    _c = SUB[f"{_r}_deep_sub"]
    for _rep, _B in REPS.items():
        CONFIGS[f"{_r}_{_rep}_src"] = dict(_c, base=_B, sources=list(_c["sources"]) + ["digit"])
        CONFIGS[f"{_r}_{_rep}_ctl"] = dict(_c, base=_B)
