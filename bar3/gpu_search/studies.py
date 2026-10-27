"""The bar-3 studies whose learners modal_learn.py runs (GPU_SEARCH.md §8): for each, where its configs are and, per
config, the learner copy that runs it and the CPU searcher its own Modal runner gave that copy. The two kinds of copy
need different workers: a "digits" copy's searches need the D.B block and the digits jobs (bar3/scale/digits/
searcher.py), a "newton" copy's the held-out targets registered (bar3/scale/heldout/searcher.py; the subgoal study's
runner used pool.Searcher itself, which differs only in not knowing the held-out targets).

A new study joins by an entry here: its configs module and its learner module (or, with a LEARNER table in its
configs, its learn_newton.py and learn_digits.py copies). Nothing here imports a study until it is asked for.
"""

from __future__ import annotations

import importlib

SEARCHER = {"newton": "bar3.scale.heldout.searcher", "digits": "bar3.scale.digits.searcher"}
STUDIES = {
    # study: (configs module, learner module or None for LEARNER's copies, the copy when there is one learner)
    "subgoal": ("bar3.scale.subgoal.configs", "bar3.scale.subgoal.learn", "newton"),
    "heldout": ("bar3.scale.subgoal.configs", "bar3.scale.subgoal.learn", "newton"),
    "newton": ("bar3.scale.newton.configs", "bar3.scale.newton.learn", "newton"),
    "digits": ("bar3.scale.digits.configs", "bar3.scale.digits.learn", "digits"),
    "rowcheck": ("bar3.scale.rowcheck.configs", None, None),
    "picks": ("bar3.scale.picks.configs", None, None),
    "reach": ("bar3.scale.reach.configs", None, None),
    "consolidate": ("bar3.scale.consolidate.configs", None, None),
}


def configs(study):
    return importlib.import_module(STUDIES[study][0]).CONFIGS


def copy_of(study, config):
    """"digits" or "newton": which workers the config's searches need."""
    mod, learn, copy = STUDIES[study]
    return copy if learn else importlib.import_module(mod).LEARNER[config]


def learner(study, config):
    """The learner module that runs the config (its `fresh(name, cfg, target, search_many, emit)`)."""
    mod, learn, _ = STUDIES[study]
    return importlib.import_module(learn or f"bar3.scale.{study}.learn_{copy_of(study, config)}")


def cpu_searcher(copy):
    """The CPU searcher class the studies' own runners used for this kind of copy."""
    return importlib.import_module(SEARCHER[copy]).Searcher
