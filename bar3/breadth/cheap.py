"""Cheaper pairs signals (BREADTH.md §5): bar3.learner.run with one change to the ct2 rule. Instead of every pair of
learnable problems (L(L−1)/2 lookaheads), only the pairs with at least one member among
- the `anchor` best single problems (by the single lookahead, which ct2 computes anyway), or
- the `anchor_cost` hardest problems (the lp rule's order: the costliest program first, ties in pool order):
about k·L lookaheads. With neither set it is learner.run exactly. As in carry.py, the code that runs is learner.run's
own source with PATCH applied once.
"""

from __future__ import annotations

import inspect

from bar3 import learner

PATCH = (
    "                combos = list(itertools.combinations(range(len(learnable)), 2))\n",
    "                combos = list(itertools.combinations(range(len(learnable)), 2))\n"
    "                if cfg.get(\"anchor\") or cfg.get(\"anchor_cost\"):   # pairs with an anchored member only\n"
    "                    if cfg.get(\"anchor\"):\n"
    "                        top = sorted(learnable, key=lambda c: -scores.get(c.name, -1e9))[:cfg[\"anchor\"]]\n"
    "                    else:\n"
    "                        top = sorted(learnable, key=lambda c: -status[c.name][\"cost\"])[:cfg[\"anchor_cost\"]]\n"
    "                    top = {c.name for c in top}\n"
    "                    combos = [(a, b) for a, b in combos if learnable[a].name in top or learnable[b].name in top]\n",
)


def _build():
    src = inspect.getsource(learner.run)
    old, new = PATCH
    assert src.count(old) == 1, "learner.run changed"
    ns = dict(vars(learner))
    exec(compile(src.replace(old, new), learner.__file__ + " (anchor)", "exec"), ns)
    return ns["run"]


run = _build()
