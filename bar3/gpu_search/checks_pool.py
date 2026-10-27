"""bar3/scale/jobs.search_job and search_jobs with each level's found programs checked on a worker pool (GPU_SEARCH.md
§10), for the jobs digits_accept.py does not take (every job but those on the digits target).

A search checks the programs it reads back at a level (up to `limit`) before it goes on: each is run on the problem's
instances at both sizes (jobs._posed's verify, brm's interpreter) and, for a target, on every pair to N = 40. On the
CPU containers that ran inside each search's own worker; on a GPU host it runs in the process that drives the GPU,
which waits meanwhile. Here a level's programs not yet checked are sent at once to the pool, where each worker builds
the job's own verify with jobs._posed and returns the row it computes for the program; the rows are kept in this
process under the same keys, and the search reads them as it would its own (a program the pool did not see is checked
here, by jobs._posed's verify). Batches under MIN programs are checked here. The record is the original's.

install(pool) points bar3/scale/jobs.search_job and search_jobs here (and the digits module's originals, which its
own search_job and search_jobs call for the jobs that are not on the target).
"""

from __future__ import annotations

import sys
import time

from library import dsl
from bar3.scale import jobs as J

MIN = 4                      # fewer programs than this are checked here (a round trip costs more)
POOL = None


def init(digits):
    """A check worker: the digits block and jobs, or the held-out targets (a target's N <= 40 check reads its target
    by name)."""
    if digits:
        from bar3.scale.digits import jobs  # noqa: F401
    else:
        from bar3.scale.heldout import targets  # noqa: F401


def row(p, job, lib):
    """jobs._posed's verify(p) for the job, in a worker: the row it stores."""
    from library.inventing.jobs import machine_for
    if job.get("breadth_target"):
        from bar3.breadth import targets as TG
        TG.use(job["breadth_target"])
    _, verify = J._posed(job, lib, machine_for(job["k"]))
    return verify(p)


def _checked(job, lib, m):
    """The job's problem, a verify that reads the rows the pool computed, and the acceptance that fills them."""
    prob, verify0 = J._posed(job, lib, m)
    rows = {}

    def verify(p):
        key = dsl.show(p)
        if key not in rows:
            rows[key] = verify0(p)
        return rows[key]

    def accept(ps):
        todo = {}
        for p in ps:
            key = dsl.show(p)
            if key not in rows and key not in todo:
                todo[key] = p
        if len(todo) >= MIN:
            n = len(todo)
            for key, r in zip(todo, POOL.map(row, todo.values(), [job] * n, [lib] * n, chunksize=max(1, n // 32))):
                rows[key] = r
        return [verify(p)["verified"] for p in ps]
    return prob, verify, accept


def search_job(job):
    """jobs.search_job, line for line, but for the acceptance (`_checked`)."""
    if job.get("engine") == "py":
        from bar3.jobs import search_job as original
        return original({k: v for k, v in job.items() if k != "engine"})
    t0 = time.time()
    lib, m, classes, n_tokens, codes = J._vocabulary(job)
    t_items = time.time() - t0
    prob, verify, accept = _checked(job, lib, m)
    res = J.engine.bfs(prob, classes, lib, m, max_cost=job.get("max_cost", 10),
                       max_states=job.get("max_states", 200_000), time_limit=job.get("time_limit", 300),
                       accept=accept, limit=job.get("limit", 200), codes=codes)
    rec = J._record(job, res, prob, verify, classes, n_tokens, lib, t_items)
    rec["wall"] = round(time.time() - t0, 2)
    return rec


def search_jobs(jobs):
    """jobs.search_jobs, line for line, but for the acceptances (`_checked`)."""
    if len(jobs) == 1 or any(j.get("near") or j.get("engine") == "py" for j in jobs):
        return [search_job(j) for j in jobs]
    t0 = time.time()
    lib, m, classes, n_tokens, codes = J._vocabulary(jobs[0])
    t_items = time.time() - t0
    posed = [_checked(j, lib, m) for j in jobs]
    j0 = jobs[0]
    results = J.engine.bfs_multi([p for p, _, _ in posed], classes, lib, m, max_cost=j0.get("max_cost", 10),
                                 max_states=j0.get("max_states", 200_000), time_limit=j0.get("time_limit", 300),
                                 accepts=[a for _, _, a in posed], limit=j0.get("limit", 200), codes=codes)
    recs = [J._record(j, r, p, v, classes, n_tokens, lib, t_items / len(jobs))
            for j, r, (p, v, _) in zip(jobs, results, posed)]
    share = round((time.time() - t0) / len(jobs), 2)
    for r in recs:
        r["wall"] = share
        r["seconds"] = share
        r["shared"] = len(jobs)
    return recs


def install(pool):
    """Before the digits module's install (it wraps these), or after it: its originals are pointed here as well."""
    global POOL
    POOL = pool
    DJ = sys.modules.get("bar3.scale.digits.jobs")
    if DJ is not None:
        DJ._ORIGINAL["search_job"], DJ._ORIGINAL["search_jobs"] = search_job, search_jobs
    else:
        J.search_job, J.search_jobs = search_job, search_jobs
