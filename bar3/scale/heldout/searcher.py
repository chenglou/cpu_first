"""bar3/scale/pool.Searcher with workers that know the held-out targets (targets.py): the N <= 40 check of a target
search runs in a worker, which looks its target up by name, so each worker registers the held-out targets before it
warms up. Everything else is pool.Searcher's.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp

from bar3.scale import pool


def _warm():
    from bar3.scale.heldout import targets  # noqa: F401  (adds the held-out targets to this worker's table)
    pool.warm()


class Searcher(pool.Searcher):
    def __init__(self, workers, *args, **kw):
        super().__init__(workers, *args, **kw)
        self.pool.shutdown(wait=True)       # nothing was submitted to it: its workers were never started
        self.pool = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_warm)
