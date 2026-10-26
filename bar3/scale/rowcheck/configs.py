"""This study's configs (ROWCHECK.md §2): the earlier studies' configs, each also with the duplicate check on a
problem's own rows, `_rc`. Which learner copy runs a config is LEARNER[config].
- the held-out test's frozen configs (`<rule>_deep_sub`), for its wins; learn_newton.py runs them, with no source on;
- the two-source study's (`<rule>_deep_sub_ns`), for 10·x² + 1; learn_newton.py;
- the digits study's base-2 configs with the source (`<rule>_b_src`), for x² mod N and binary x mod N;
  learn_digits.py."""

from __future__ import annotations

from bar3.scale.digits.configs import CONFIGS as DIG
from bar3.scale.newton.configs import CONFIGS as NEW

CONFIGS, LEARNER = {}, {}
for _r in ("lp_clean", "ct2a2_clean"):
    for _name, _c, _l in ((f"{_r}_deep_sub", NEW[f"{_r}_deep_sub"], "newton"),
                          (f"{_r}_deep_sub_ns", NEW[f"{_r}_deep_sub_ns"], "newton"),
                          (f"{_r}_b_src", DIG[f"{_r}_b_src"], "digits")):
        CONFIGS[_name], CONFIGS[f"{_name}_rc"] = _c, dict(_c, rowcheck=True)
        LEARNER[_name] = LEARNER[f"{_name}_rc"] = _l
