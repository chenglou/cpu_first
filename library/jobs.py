"""One search job: a task, a vocabulary, a budget -> a record (what was found, how big the search was).

A job is a plain dict so it can travel to a Modal container:
    task, settings [(B, W)...], lib {name: program text}, kinds [loop kinds incl. abstractions], max_cost,
    max_states, time_limit, NP, seed, max_body, max_arity, limit (programs kept), verify [(B, W)...]
"""

from __future__ import annotations

import math
import time

from library import dsl, search

VERIFY = ((2, 2), (2, 4), (3, 3), (4, 2), (5, 2), (7, 2), (10, 2), (10, 3))


def parse_lib(lib_txt):
    return {k: dsl.parse(v) for k, v in lib_txt.items()}


def vocabulary(lib, kinds, prims=None):
    """Tokens of the search: the primitives and every library macro (abstractions are loop kinds instead)."""
    return list(prims or dsl.PRIMS) + [k for k in lib if k not in kinds]


def cells_for(job, B):
    """The cells of base B: exact, or the job's learned tables {"B": .., "sbb_d": [B*B*2], "sbb_c": [..]} (flat over
    (a, b, borrow in), anchors' order) where the base matches."""
    c = dsl.Cells(B)
    lc = job.get("cells")
    if lc and lc["B"] == B:
        import numpy as np
        c.sbb_d = np.asarray(lc["sbb_d"], np.uint8).reshape(B, B, 2)
        c.sbb_c = np.asarray(lc["sbb_c"], np.uint8).reshape(B, B, 2)
    return c


def search_job(job):
    t0 = time.time()
    lib = parse_lib(job.get("lib", {}))
    kinds = list(job.get("kinds", dsl.LOOP_KINDS))
    tokens = vocabulary(lib, kinds, job.get("prims"))
    settings = [dsl.Setting(B, W, cells=cells_for(job, B)) for B, W in job.get("settings", ((2, 3), (3, 2)))]
    classes = search.enumerate_items(tokens, lib, settings, kinds=kinds, max_body=job.get("max_body", 2),
                                     max_arity=job.get("max_arity", 3))
    t_items = time.time() - t0
    if job.get("inrun"):
        from library import inrun
        rows = [tuple(r) for r in job["rows"]]
        helpers = {k: dsl.parse(v) for k, v in job.get("helpers", {}).items()}
        prob = inrun.LawProblem(job["task"], settings, job.get("NP", 8), job.get("seed", 0), rows, helpers, lib)
    else:
        prob = search.Problem(job["task"], settings, NP=job.get("NP", 8), seed=job.get("seed", 0))
    vs = [dsl.Setting(B, W, cells=cells_for(job, B)) for B, W in job.get("verify", VERIFY)]
    exact = [dsl.Setting(B, W) for B, W in job.get("verify", VERIFY)]
    rows = {}

    def verify(p):
        key = dsl.show(p)
        if key in rows:
            return rows[key]
        if job.get("inrun"):      # in-run: the task's laws on fresh rows (and the labels for sqmod)
            r = inrun.accept(job["task"], p, lib, helpers, [tuple(x) for x in job["verify_rows"]], vs,
                             n=job.get("verify_n", 60), label_rows=[tuple(x) for x in job.get("label_rows", [])])
            r = {k: float(v) for k, v in r.items()}
        else:
            r = search.check(job["task"], p, lib, vs, n=job.get("verify_n", 200))
        row = dict(program=key, primitive=dsl.show(dsl.expand(p, lib)), primitive_cost=dsl.cost(dsl.expand(p, lib)),
                   verified=min(r.values()) == 1.0, worst=min(r.values()))
        if job.get("cells"):          # does the program also compute the right thing with the true tables?
            row["exact_worst"] = min(search.check(job["task"], p, lib, exact, n=job.get("verify_n", 200)).values())
        if job.get("inrun"):          # for the report only: is it the offline task's right program?
            off = inrun.OFFLINE[job["task"]]
            row["offline_worst"] = min(search.check(off, p, lib, [dsl.Setting(B, W) for B, W in VERIFY], n=100)
                                       .values())
        rows[key] = row
        return row

    res = search.bfs(prob, classes, lib, max_cost=job.get("max_cost", 6), max_states=job.get("max_states", 2_000_000),
                     time_limit=job.get("time_limit", 1800), verbose=job.get("verbose", False),
                     accept=lambda progs: [verify(p)["verified"] for p in progs], limit=job.get("limit", 200))
    fallback = None
    if res.found_cost is None and job.get("fallback_primitive") and len(tokens) > len(job.get("prims") or dsl.PRIMS):
        # a bigger vocabulary makes programs that do not use it harder to reach: try the primitives alone too
        fb_classes = search.enumerate_items(list(job.get("prims") or dsl.PRIMS), {}, settings, kinds=dsl.LOOP_KINDS,
                                            max_body=job.get("max_body", 2), max_arity=job.get("max_arity", 3))
        fb = search.bfs(prob, fb_classes, lib, max_cost=job.get("max_cost", 6),
                        max_states=job.get("max_states", 2_000_000), time_limit=job.get("time_limit", 1800),
                        accept=lambda progs: [verify(p)["verified"] for p in progs], limit=job.get("limit", 200))
        fallback = dict(found_cost=fb.found_cost, level_sizes=fb.level_sizes, seconds=round(fb.seconds, 2),
                        stop=fb.stop)
        if fb.found_cost is not None:
            res, classes = fb, fb_classes
    rec = dict(name=job.get("name", job["task"]), task=job["task"], found_cost=res.found_cost,
               level_sizes=res.level_sizes, seconds=round(res.seconds, 2), item_seconds=round(t_items, 2),
               stop=res.stop, expanded=int(res.expanded), n_tokens=len(tokens), n_kinds=len(kinds),
               n_classes=len(classes), classes_by_cost={c: sum(1 for x in classes if x.cost == c)
                                                       for c in sorted({x.cost for x in classes})},
               job={k: v for k, v in job.items() if k not in ("lib", "rows", "verify_rows", "label_rows")},
               lib_size=len(lib), aliases=res.aliases, fallback=fallback)
    reached = res.found_cost if res.found_cost is not None else len(res.level_sizes) - 1
    rec["raw_log2"] = round(math.log2(max(1, sum(dsl.raw_count(len(tokens), len(kinds), L, job.get("max_arity", 3))
                                                 for L in range(1, reached + 1)))), 2)
    if res.found_cost is not None:
        progs = search.programs(res, classes, limit=job.get("limit", 200))
        out = [verify(p) for p in progs]
        rec["programs"] = out
        rec["n_programs"] = len(out)
        rec["n_verified"] = sum(r["verified"] for r in out)
    rec["wall"] = round(time.time() - t0, 2)
    return rec
