"""One bare-machine search job -> a record (plain dicts, so it can run on Modal). Digit jobs (machine "digit") go
to digit_jobs.py.

A job: name, task, k (the size), xs / Ns / want (the probes and their targets), check (all pairs the program must
get right to be accepted: the whole size for the target, the probes for a dream), lib, kinds, budget (max_cost,
max_states, time_limit, max_body), machine (cap, limit).
"""

from __future__ import annotations

import math
import time

import numpy as np

from library import dsl
from library.inventing import bfs, brm, target


def machine_for(k):
    """Values saturate at 2k^2 (x^2 fits), and a machine may run 50 k^2 + 1000 steps (the modular counter needs
    about 6 k^2; square-then-subtract-repeatedly needs about k^4 and is dropped above small k)."""
    return brm.Machine(cap=max(64, 2 * k * k), limit=50 * k * k + 1000)


def general(prog, lib, upto=40):
    """The share of every pair with N <= upto that prog gets right, on a machine big enough for the true algorithm."""
    xs, Ns = target.pairs(upto)
    got, dead = brm.evaluate(prog, lib, xs, Ns, brm.Machine(cap=4 * upto * upto, limit=10 ** 6))
    return 0.0 if dead else float(np.mean(got == target.answers(xs, Ns)))


def search_job(job):
    if job.get("machine") == "digit":
        from library.inventing.digit_jobs import digit_search_job
        return digit_search_job(job)
    t0 = time.time()
    lib = {k: dsl.parse(v) for k, v in job.get("lib", {}).items()}
    kinds = list(job.get("kinds", brm.LOOP_KINDS))
    tokens = list(brm.PRIMS) + [k for k in lib if k not in kinds]
    m = machine_for(job["k"])
    classes = bfs.enumerate_items(tokens, lib, kinds, m, max_body=job.get("max_body", 2))
    t_items = time.time() - t0
    prob = bfs.Problem(np.array(job["xs"]), np.array(job["Ns"]), np.array(job["want"]))
    cx, cN, cw = np.array(job["check"][0]), np.array(job["check"][1]), np.array(job["check"][2])
    rows = {}

    def verify(p):
        key = dsl.show(p)
        if key not in rows:
            got, dead = brm.evaluate(p, lib, cx, cN, m)
            ok = (not dead) and bool((got == cw).all())
            rows[key] = dict(program=key, primitive=dsl.show(dsl.expand(p, lib)),
                             primitive_cost=dsl.cost(dsl.expand(p, lib)), verified=ok)
            if ok and job["task"] == "target":
                rows[key]["general"] = general(p, lib)
        return rows[key]

    res = bfs.bfs(prob, classes, lib, m, max_cost=job.get("max_cost", 10), max_states=job.get("max_states", 3_000_000),
                  time_limit=job.get("time_limit", 1800), accept=lambda ps: [verify(p)["verified"] for p in ps],
                  limit=job.get("limit", 200))
    rec = dict(name=job.get("name"), task=job["task"], k=job["k"], found_cost=res.found_cost,
               level_sizes=res.level_sizes, seconds=round(res.seconds, 2), item_seconds=round(t_items, 2),
               stop=res.stop, expanded=int(res.expanded), n_tokens=len(tokens), n_classes=len(classes),
               lib_size=len(lib), aliases=res.aliases)
    reached = res.found_cost if res.found_cost is not None else len(res.level_sizes) - 1
    rec["raw_log2"] = round(math.log2(max(1, sum(dsl.raw_count(len(tokens), len(kinds), L, 3)
                                                 for L in range(1, reached + 1)))), 2)
    if res.found_cost is not None:
        out = [verify(p) for p in bfs.programs(res, classes, limit=job.get("limit", 200))]
        rec["programs"] = out
        rec["n_programs"] = len(out)
        rec["n_verified"] = sum(r["verified"] for r in out)
    rec["wall"] = round(time.time() - t0, 2)
    return rec
