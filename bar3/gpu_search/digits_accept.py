"""bar3/scale/digits/jobs.search_job with a level's found programs checked on a worker pool (GPU_SEARCH.md §10).

A job on the digits target accepts a program only when it is right on the target's rows at both sizes and at big x
(targets.bigx_share: 200 instances of up to 100,000 steps each), and the search checks each level's programs (up to
`limit`) before it goes on. On the CPU containers those checks ran inside the search's own worker, beside fifteen other
searches; on a GPU host they would run one after another in the process that drives the GPU, which waits meanwhile
(a lookahead whose level holds 433 programs right on the rows only spent 17-20 s on them). Here the programs of a
level not yet checked are checked at once on the pool, each by the same code (machine.evaluate, bigx_share), and
their rows are filled in the order the original fills them, so `rows`, `right_on_rows_only` and the record are the
original's. Every other job runs as bar3/scale/digits/jobs.py runs it.

install(pool) points bar3/scale/jobs.search_job and the digits module's search_job (which its search_jobs calls) here.
"""

from __future__ import annotations

import time

from library import dsl
from bar3.scale import jobs as J
from bar3.scale.digits import jobs as DJ
from bar3.scale.digits import machine as M
from bar3.scale.digits import targets as DT

POOL = None


def init():
    """A check worker: the digit block installed, the digits targets known."""
    from bar3.scale.digits import jobs  # noqa: F401, F811


def check(p, lib, every, chk, target):
    """DJ._posed's verify for one program, but the row it would store: (right on the rows at both sizes, big-x
    share)."""
    (ex, eN, ew, k), (cx, cN, cw, k2) = every, chk
    got, dead = M.evaluate(p, lib, ex, eN, DT.machine(k))
    ok = (not dead) and bool((got == ew).all())
    if ok:
        got, dead = M.evaluate(p, lib, cx, cN, DT.machine(k2))
        ok = (not dead) and bool((got == cw).all())
    return ok, (DT.bigx_share(p, lib, target) if ok else 0.0)


def _accept(job, lib, rows):
    import numpy as np
    every = tuple(np.array(v) for v in job["every"]) + (job["k"],)
    chk = tuple(np.array(v) for v in job["check"][1:]) + (job["check"][0],)

    def accept(ps):
        keys = [dsl.show(p) for p in ps]
        todo = {}
        for key, p in zip(keys, ps):
            if key not in rows and key not in todo:
                todo[key] = p
        if todo:
            n = len(todo)
            outs = POOL.map(check, todo.values(), [lib] * n, [every] * n, [chk] * n, [job["digits_target"]] * n,
                            chunksize=max(1, n // 32))
            for (key, p), (ok, share) in zip(todo.items(), outs):
                prim = dsl.expand(p, lib)
                rows[key] = dict(program=key, primitive=dsl.show(prim), primitive_cost=dsl.cost(prim),
                                 verified=ok and share == 1.0, rows=ok, general=share)
        return [rows[key]["verified"] for key in keys]
    return accept


def search_job(job):
    """DJ.search_job, line for line, but for the acceptance (`_accept`)."""
    if not job.get("digits_target"):
        return DJ._ORIGINAL["search_job"](job)
    t0 = time.time()
    lib, _, classes, n_tokens, codes = J._vocabulary(job)
    t_items = time.time() - t0
    m, m2 = DT.machine(job["k"]), DT.machine(job["check"][0])
    prob, verify, rows = DJ._posed(job, lib, m, m2)
    res = DJ.engine.bfs(prob, classes, lib, m, max_cost=job.get("max_cost", 10),
                        max_states=job.get("max_states", 200_000), time_limit=job.get("time_limit", 300),
                        accept=_accept(job, lib, rows), limit=job.get("limit", 200), codes=codes)
    rec = J._record(job, res, prob, verify, classes, n_tokens, lib, t_items)
    rejected = [r for r in rows.values() if r["rows"] and not r["verified"]]
    rec["right_on_rows_only"] = dict(n=len(rejected), first=rejected[:10])      # right on the rows, not at big x
    rec["wall"] = round(time.time() - t0, 2)
    return rec


def install(pool):
    global POOL
    POOL = pool
    J.search_job = DJ.search_job = search_job
