"""bar3/scale/pool.Searcher whose workers run the digits study's jobs (jobs.py: the digit block, the study's targets,
the target's machines and the check at big x). Everything else is pool.Searcher's.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp

from bar3.scale import pool
from bar3.scale.digits import jobs  # noqa: F401  (in this process too: same_search keeps the target's jobs apart)


def _warm():
    from bar3.scale.digits import jobs  # noqa: F401, F811  (installs everything in the worker)
    pool.warm()


class Searcher(pool.Searcher):
    def __init__(self, workers, *args, **kw):
        super().__init__(workers, *args, **kw)
        self.pool.shutdown(wait=True)       # nothing was submitted to it: its workers were never started
        self.pool = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_warm)
