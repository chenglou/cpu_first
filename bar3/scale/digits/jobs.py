"""The search jobs of the digits study (DIGITS.md §2): bar3/scale/jobs.py, but for the jobs on the target.

Importing this module installs the digit block (machine.py), adds the study's targets (targets.py), and points
bar3/scale/jobs's search_job, search_jobs and same_search at the versions here, in this process. A job tagged
`digits_target` (learn.DigitsTap: every job on the target) runs on the target's machines (targets.machine: the
caps as before, 20,000 steps at the search size, 100,000 at the check size), and a program for it is accepted only
when it is right on all the target's rows at both sizes and also passes the check at big x; its "general" is its
share right at big x. Every other job runs exactly as bar3/scale/jobs runs it.
"""

from __future__ import annotations

import json
import time

import numpy as np

from library import dsl
from bar3 import search
from bar3.scale import engine
from bar3.scale import jobs as J
from bar3.scale.digits import machine as M
from bar3.scale.digits import targets as DT

_ORIGINAL = dict(search_job=J.search_job, search_jobs=J.search_jobs, same_search=J.same_search)


def _posed(job, lib, m, m2):
    """The target's problem and its acceptance test: right on every row at both sizes (on the compiled kernel,
    machine.evaluate, which equals brm.evaluate) and right at big x."""
    prob = search.Problem(np.array(job["xs"]), np.array(job["Ns"]), np.array(job["want"]))
    ex, eN, ew = (np.array(v) for v in job["every"])
    cx, cN, cw = (np.array(v) for v in job["check"][1:])
    rows = {}

    def verify(p):
        key = dsl.show(p)
        if key not in rows:
            got, dead = M.evaluate(p, lib, ex, eN, m)
            ok = (not dead) and bool((got == ew).all())
            if ok:
                got, dead = M.evaluate(p, lib, cx, cN, m2)
                ok = (not dead) and bool((got == cw).all())
            share = DT.bigx_share(p, lib, job["digits_target"]) if ok else 0.0
            prim = dsl.expand(p, lib)
            rows[key] = dict(program=key, primitive=dsl.show(prim), primitive_cost=dsl.cost(prim),
                             verified=ok and share == 1.0, rows=ok, general=share)
        return rows[key]

    return prob, verify, rows


def search_job(job):
    if not job.get("digits_target"):
        return _ORIGINAL["search_job"](job)
    t0 = time.time()
    lib, _, classes, n_tokens, codes = J._vocabulary(job)
    t_items = time.time() - t0
    m, m2 = DT.machine(job["k"]), DT.machine(job["check"][0])
    prob, verify, rows = _posed(job, lib, m, m2)
    res = engine.bfs(prob, classes, lib, m, max_cost=job.get("max_cost", 10),
                     max_states=job.get("max_states", 200_000), time_limit=job.get("time_limit", 300),
                     accept=lambda ps: [verify(p)["verified"] for p in ps], limit=job.get("limit", 200), codes=codes)
    rec = J._record(job, res, prob, verify, classes, n_tokens, lib, t_items)
    rejected = [r for r in rows.values() if r["rows"] and not r["verified"]]
    rec["right_on_rows_only"] = dict(n=len(rejected), first=rejected[:10])      # right on the rows, not at big x
    rec["wall"] = round(time.time() - t0, 2)
    return rec


def search_jobs(jobs):
    if any(j.get("digits_target") for j in jobs):
        return [search_job(j) for j in jobs]
    return _ORIGINAL["search_jobs"](jobs)


def same_search(job):
    """bar3/scale/jobs.same_search, with the target's tag: a job on the target never shares a search with another."""
    return json.dumps([_ORIGINAL["same_search"](job), job.get("digits_target")])


J.search_job, J.search_jobs, J.same_search = search_job, search_jobs, same_search
