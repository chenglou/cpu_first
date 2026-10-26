"""With cfg["alloc"] off, this study's learner copies are reach's (CONSOLIDATE.md §3): bar3/scale/reach/check.py's
replays, with its learner swapped for this study's copies. `off` replays earlier records from their stores and says
whether each reproduces exactly.

    cd cpu_first && <python with numba> -m bar3.scale.consolidate.check off [--digits] STORE ... -- RECORD ...
"""

from __future__ import annotations

from bar3.scale.reach import check as RC
from bar3.scale.reach.configs import LEARNER


def learner(config):
    if LEARNER[config] == "digits":
        from bar3.scale.digits import jobs as DJ     # installs the block and the digits targets and jobs
        from bar3.scale.consolidate import learn_digits as L
        return L, DJ.search_job, L.DigitsTap
    from bar3.scale.jobs import search_job
    from bar3.scale.consolidate import learn_newton as L
    return L, search_job, RC.Tap


RC.learner = learner

if __name__ == "__main__":
    RC.main()
