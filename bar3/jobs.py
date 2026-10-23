"""One search job -> a record of plain dicts (so it can run on Modal): a probe of a candidate problem, a practice
search, a target attempt, or a lookahead (the target searched with extra routines).

A job: name, kind, k, xs / Ns / want (the probes), every (all instances at size k, as [xs, Ns, want]), check (the
instances at size k2, [k2, xs, Ns, want]), lib, kinds, budget (max_cost, max_states, time_limit, max_body, limit),
near (how many near misses to return), general (check a found program on every pair with N <= 40, for the target).
A program is accepted when it is right on every instance at size k and at size k2.
"""

from __future__ import annotations

import json
import time

import numpy as np

from library import dsl
from library.inventing import brm
from library.inventing.jobs import general, machine_for
from bar3 import search

_CLASSES = {}


def classes_for(lib, kinds, k, max_body):
    """The search items of a vocabulary, cached per container (probes of one round share the library)."""
    key = (json.dumps({t: dsl.show(v) for t, v in sorted(lib.items())}), tuple(kinds), k, max_body)
    if key not in _CLASSES:
        if len(_CLASSES) > 8:
            _CLASSES.clear()
        tokens = list(brm.PRIMS) + [t for t in lib if t not in kinds]
        _CLASSES[key] = (search.enumerate_items(tokens, lib, kinds, machine_for(k), max_body=max_body), len(tokens))
    return _CLASSES[key]


def search_job(job):
    t0 = time.time()
    lib = {t: dsl.parse(v) for t, v in job.get("lib", {}).items()}
    kinds = list(job.get("kinds", brm.LOOP_KINDS))
    k = job["k"]
    m = machine_for(k)
    classes, n_tokens = classes_for(lib, kinds, k, job.get("max_body", 2))
    t_items = time.time() - t0
    prob = search.Problem(np.array(job["xs"]), np.array(job["Ns"]), np.array(job["want"]))
    ex, eN, ew = (np.array(v) for v in job["every"])
    k2, cx, cN, cw = job["check"][0], *(np.array(v) for v in job["check"][1:])
    m2 = machine_for(k2)
    rows = {}

    def verify(p):
        key = dsl.show(p)
        if key not in rows:
            got, dead = brm.evaluate(p, lib, ex, eN, m)
            ok = (not dead) and bool((got == ew).all())
            if ok:
                got2, dead2 = brm.evaluate(p, lib, cx, cN, m2)
                ok = (not dead2) and bool((got2 == cw).all())
            prim = dsl.expand(p, lib)
            rows[key] = dict(program=key, primitive=dsl.show(prim), primitive_cost=dsl.cost(prim), verified=ok)
            if ok and job.get("general"):
                rows[key]["general"] = general(p, lib)
        return rows[key]

    res = search.bfs(prob, classes, lib, m, max_cost=job.get("max_cost", 10),
                     max_states=job.get("max_states", 200_000), time_limit=job.get("time_limit", 300),
                     accept=lambda ps: [verify(p)["verified"] for p in ps], limit=job.get("limit", 200))
    rec = dict(name=job.get("name"), kind=job.get("kind"), k=k, found_cost=res.found_cost, level_sizes=res.level_sizes,
               best_fit=res.best_fit, NP=len(job["xs"]), seconds=round(res.seconds, 2), item_seconds=round(t_items, 2),
               stop=res.stop, n_tokens=n_tokens, n_classes=len(classes), lib_size=len(lib),
               aliases=[(d, ps[:3]) for d, ps in res.aliases])
    if res.found_cost is not None:
        out = [verify(p) for p in search.programs(res, classes, limit=job.get("limit", 200))]
        rec["programs"] = [r for r in out if r["verified"]]
    if job.get("near"):
        rec["near"] = [(fit, d, dsl.show(p)) for fit, d, p, _ in search.near_misses(res, prob, classes, n=job["near"])]
    rec["wall"] = round(time.time() - t0, 2)
    return rec
