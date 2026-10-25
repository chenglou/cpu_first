"""One search job on the counting machine -> a record of plain dicts (bar3/jobs.py with several inputs and outputs).

A job: name, kind, regs (the problem's machine), outputs, probe (inputs {reg: [n]}, want [[...]]: the search's
sampled rows), every (all rows at the search bases), check (all rows at the check bases), general (optional: all rows
at bases up to 16), lib, budget (max_cost, max_states, time_limit, max_body, limit), near (near misses to return).
A program is accepted when it is right on every row at the search bases and at the check bases. With `clean_win`
(CARRY.md §12, off by default), a right program that changes an input it does not output wins only when no right
program within one level of it leaves those inputs as they were: the search looks one level further, and keeps the
first right programs if none turns up.
"""

from __future__ import annotations

import json
import time

import numpy as np

from library import dsl
from bar3.bridge import dcm, dsearch, slices as S
from bar3.bridge.dproblems import machine_for

_CLASSES = {}


def vocabulary(lib, regs):
    """The tokens of a problem's machine: the primitives over its registers, and every library token (routine or
    merge) that names only those registers."""
    pl = {t: dsl.parse(v) for t, v in lib.items()}
    toks = dcm.prims(regs) + [t for t in lib if dcm.regs_of(pl[t], pl) <= set(regs)]
    return toks, dcm.loop_kinds(regs)


def classes_for(lib, regs, max_body):
    key = (json.dumps(sorted(lib.items())), tuple(regs), max_body)
    if key not in _CLASSES:
        if len(_CLASSES) > 8:
            _CLASSES.clear()
        toks, kinds = vocabulary(lib, regs)
        pl = {t: dsl.parse(v) for t, v in lib.items()}
        _CLASSES[key] = (dsearch.enumerate_items(toks, pl, kinds, machine_for(S.SEARCH_BASES), max_body=max_body),
                         len(toks), len(kinds))
    return _CLASSES[key]


def right(prog, lib, inputs, want, outputs, bases):
    """Is prog right on every row (inputs {reg: list}, want [[...]])?"""
    if not want:
        return True
    regs, dead = dcm.evaluate(prog, lib, {r: np.array(v, np.int64) for r, v in inputs.items()}, machine_for(bases))
    got = regs[:, [dcm.RI[o] for o in outputs]]
    return (not dead) and bool((got == np.array(want, np.int64)).all())


def search_job(job):
    t0 = time.time()
    lib = {t: dsl.parse(v) for t, v in job.get("lib", {}).items()}
    classes, n_tokens, n_kinds = classes_for(job.get("lib", {}), job["regs"], job.get("max_body", 2))
    t_items = time.time() - t0
    outs = tuple(job["outputs"])
    prob = dsearch.Problem({r: np.array(v, np.int64) for r, v in job["probe"][0].items()}, outs,
                           np.array(job["probe"][1], np.int64).reshape(-1, len(outs)))
    rows = {}

    def verify(p):
        key = dsl.show(p)
        if key not in rows:
            ok = right(p, lib, *job["every"], outs, S.SEARCH_BASES) and right(p, lib, *job["check"], outs,
                                                                            S.CHECK_BASES)
            prim = dsl.expand(p, lib)
            rows[key] = dict(program=key, primitive=dsl.show(prim), primitive_cost=dsl.cost(prim), verified=ok)
            if ok and job.get("general"):
                rows[key]["general"] = right(p, lib, *job["general"], outs, S.GENERAL_BASES)
        return rows[key]

    held = {}

    def clean(p):
        """Does p leave the inputs it does not output as they were, on every row at the search bases?"""
        ins = {r: np.array(v, np.int64) for r, v in job["every"][0].items()}
        regs, dead = dcm.evaluate(p, lib, ins, machine_for(S.SEARCH_BASES))
        return not dead and all((regs[:, dcm.RI[r]] == v).all() for r, v in ins.items() if r not in outs)

    def accept(ps):
        ok = [verify(p)["verified"] for p in ps]
        if not job.get("clean_win"):
            return ok
        good = [o and clean(p) for p, o in zip(ps, ok)]
        if any(good):
            return good
        if any(ok) and not held:
            held["progs"] = [p for p, o in zip(ps, ok) if o]
            return [False] * len(ps), "defer"
        return [False] * len(ps)

    res = dsearch.bfs(prob, classes, lib, machine_for(S.SEARCH_BASES), max_cost=job.get("max_cost", 10),
                      max_states=job.get("max_states", 200_000), time_limit=job.get("time_limit", 300),
                      accept=accept, limit=job.get("limit", 200))
    if res.found_cost is None and held:                # nothing clean within one level: the first right programs
        res.found_cost = dsl.cost(held["progs"][0])
    rec = dict(name=job.get("name"), kind=job.get("kind"), found_cost=res.found_cost, level_sizes=res.level_sizes,
               best_fit=res.best_fit, NP=prob.NP, seconds=round(res.seconds, 2), item_seconds=round(t_items, 2),
               stop=res.stop, n_tokens=n_tokens, n_kinds=n_kinds, n_classes=len(classes), lib_size=len(lib),
               aliases=[(d, ps[:3]) for d, ps in res.aliases])
    if res.found_cost is not None:
        if held and res.accepted is None:
            found = held["progs"]
        else:
            found = dsearch.programs(res, classes, limit=job.get("limit", 200))
        out = [verify(p) for p in found]
        rec["programs"] = [r for r in out if r["verified"]]
        if job.get("clean_win"):
            rec["clean"] = any(clean(dsl.parse(r["program"])) for r in rec["programs"])
            if held:
                rec["dirty_first"] = [dsl.show(p) for p in held["progs"][:5]]
    if job.get("near"):
        rec["near"] = [(fit, d, dsl.show(p)) for fit, d, p in dsearch.near_misses(res, prob, classes, n=job["near"])]
    rec["wall"] = round(time.time() - t0, 2)
    return rec
