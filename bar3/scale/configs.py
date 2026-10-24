"""The scale study's configs: the breadth study's (bar3/breadth/drive.py: the learner's, all_once, ct2a2_clean), plus
those of the levers (SCALE.md §2), all run by learn.py:
- `<rule>_new`: the learner's sources plus "recur" (recurrences: one- and two-number steps) and "const" (constants);
- `<rule>_rrem`: those plus "rrem" (the rest of the way from each routine's output);
- `<rule>_guided`: `<rule>_new` with a second target search guided by learned costs when the plain one fails;
- `<rule>_all3`: every lever;
- `<rule>_deep`: every lever and PRACTICE.md's depth-2 law book (one law applied to another's result: it poses
  x² mod N itself, which 10·x² needs as a piece).
for rule in lp_clean (hardest first) and ct2a2_clean (pairs with one member among the two hardest).
- `<config>_k26`, `<config>_k50`: every pair x < N <= 26 (checked at N <= 30) or <= 50 (checked at N <= 56), with
  PRACTICE.md's bigger probe at N <= 26 and a program called general when right on every pair up to twice the size
  (at least 40, the breadth study's bound).
"""

from __future__ import annotations

from bar3.breadth.drive import CONFIGS as BREADTH
from bar3.learner import BIG26, SOURCES

CONFIGS = dict(BREADTH)
for _r in ("lp_clean", "ct2a2_clean"):
    _b = BREADTH[_r]
    CONFIGS[f"{_r}_new"] = dict(_b, sources=SOURCES + ("recur", "const"))
    CONFIGS[f"{_r}_rrem"] = dict(_b, sources=SOURCES + ("recur", "const", "rrem"))
    CONFIGS[f"{_r}_guided"] = dict(CONFIGS[f"{_r}_new"], guided=True)
    CONFIGS[f"{_r}_all3"] = dict(CONFIGS[f"{_r}_rrem"], guided=True)
    CONFIGS[f"{_r}_deep"] = dict(CONFIGS[f"{_r}_all3"], sources=SOURCES + ("law2", "recur", "const", "rrem"))
for _c in [c for c in CONFIGS if c.endswith(("_clean", "_new", "_rrem", "_guided", "_all3", "_deep"))]:
    CONFIGS[f"{_c}_k26"] = dict(CONFIGS[_c], k=26, k2=30, budgets=BIG26, general_upto=52)
    CONFIGS[f"{_c}_k50"] = dict(CONFIGS[_c], k=50, k2=56, budgets=BIG26, general_upto=100)
