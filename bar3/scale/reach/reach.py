"""Two more changes to the learner's picks (REACH.md §2), each behind its own flag, beside the earlier studies'
(bar3/scale/rowcheck/rowcheck.py, bar3/scale/picks/picks.py):

- `deeplook`: anchored pairs' look at a step the digits source posed gets a deeper budget, as the subgoal gets a
  deeper probe. The 30k-state look ends with level 4, one short of the five-token program that passes at big x on
  x² mod N (`D.2[step] MOV.X.A CLR.A T.law10`, PICKS.md §8). The deeper look's state budget lets it store level 4
  and stream level 5, its time limit lets level 5 finish, and it checks more of the programs it finds: at level 4
  alone, 602 counting programs fit every row and fail at big x. Only singles get it. A pair look is of two routines,
  and the program needs the step alone.
- `twin5`: the subgoal's twin (bar3/scale/picks/picks.twins: the learnable problem the subgoal is a duplicate of) is
  practised as an extra pick, a fifth slot, as soon as it is learnable. Unlike picks.py's `twin` it neither replaces
  one of the four picks nor joins the subgoal's ties, so the four picks are the frozen learner's. It joins the
  subgoal set, so its routine counts as a practised subgoal's.
"""

from __future__ import annotations

from bar3 import learner as L

DEEP_LOOK = dict(L.LOOK, max_states=1_000_000, time_limit=3600, limit=20_000)


def look_budget(cfg, budget, c):
    """The look's budget for candidate c: the deeper one for a step the digits source posed, with cfg["deeplook"]."""
    return DEEP_LOOK if cfg.get("deeplook") and c.source == "digit" else budget
