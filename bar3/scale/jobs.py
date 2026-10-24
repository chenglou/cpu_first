"""bar3/jobs.py's search_job on the compiled engine (engine.py): the same job in, the same record out (check.py), but
for the seconds. A job may set "engine": "py" to run the original (bar3/jobs.py) instead, for comparisons.

A job may also pose a problem of another shape (sources.py): "init" sets registers beyond X and N before the
program runs (per probe; "every_init" and "check_init" for the acceptance instances), and "outs" names the registers
that hold the answer (want[i] then lists one value per register).
"""

from __future__ import annotations

import json
import time

import numpy as np

from library import dsl
from library.inventing import brm
from library.inventing.jobs import general, machine_for
from bar3 import search
from bar3.scale import engine

_CLASSES = {}


class RProblem:
    """bar3/search.Problem with registers set beyond X and N, and an answer in any registers."""

    def __init__(self, xs, Ns, want, init, outs):
        self.xs, self.Ns = xs, Ns
        self.want = np.asarray(want, np.int64).reshape(len(xs), len(outs))
        self.outs = [brm.RI[r] for r in outs]
        self.root = start(xs, Ns, init)

    def fits(self, b):
        return (b.regs[:, :, self.outs] == self.want[None]).all(axis=2).sum(axis=1) * ~b.dead

    def goal(self, b):
        return self.fits(b) == len(self.want)


def start(xs, Ns, init):
    b = brm.fresh(np.asarray(xs), np.asarray(Ns))
    for r, v in (init or {}).items():
        b.regs[0, :, brm.RI[r]] = v
    return b


def run_on(prog, lib, xs, Ns, init, m):
    """The registers after prog on each instance, and whether some instance ran out of steps (brm.evaluate's run)."""
    b = start(xs, Ns, init)
    brm.run_items(m, b, prog, lib)
    return b.regs[0], bool(b.dead[0])


def classes_for(lib, kinds, k, max_body, costs=None):
    """The search items of a vocabulary and their bytecode, cached per container (probes of one round share the
    library), as bar3/jobs.classes_for. With `costs` (token or loop kind -> bits), each item costs the bits of its
    tokens and loop kinds instead of its token count, and a class is searched as its cheapest member: the search is
    then by description length under that model (learn.learned_costs), cheapest first."""
    key = (json.dumps({t: dsl.show(v) for t, v in sorted(lib.items())}), tuple(kinds), k, max_body,
           json.dumps(costs, sort_keys=True) if costs else "")
    if key not in _CLASSES:
        if len(_CLASSES) > 8:
            _CLASSES.clear()
        tokens = list(brm.PRIMS) + [t for t in lib if t not in kinds]
        classes = engine.enumerate_items(tokens, lib, kinds, machine_for(k), max_body=max_body)
        if costs:
            def bits(items):
                return sum(costs[it[1]] + bits(it[2]) if isinstance(it, tuple) else costs[it] for it in items)
            priced = []
            for cl in classes:
                rep = min(cl.members, key=lambda it: bits((it,)))
                priced.append(engine.ItemClass(rep, bits((rep,)), list(cl.members)))
            classes = sorted(priced, key=lambda cl: cl.cost)
        _CLASSES[key] = (classes, len(tokens), engine.Codes(classes, lib))
    return _CLASSES[key]


def _posed(job, lib, m):
    """The job's problem and its acceptance test (verify(program) -> its row: verified, primitive form, general)."""
    shaped = "outs" in job or "init" in job
    if shaped:
        outs = job.get("outs", ["A"])
        prob = RProblem(np.array(job["xs"]), np.array(job["Ns"]), job["want"], job.get("init"), outs)
        oi = [brm.RI[r] for r in outs]
    else:
        prob = search.Problem(np.array(job["xs"]), np.array(job["Ns"]), np.array(job["want"]))
    ex, eN, ew = (np.array(v) for v in job["every"])
    k2, cx, cN, cw = job["check"][0], *(np.array(v) for v in job["check"][1:])
    m2 = machine_for(k2)
    rows = {}

    def right(p, xs, Ns, init, want, mm):
        if not shaped:
            got, dead = brm.evaluate(p, lib, xs, Ns, mm)
            return (not dead) and bool((got == want).all())
        regs, dead = run_on(p, lib, xs, Ns, init, mm)
        return (not dead) and bool((regs[:, oi] == np.asarray(want).reshape(len(xs), len(oi))).all())

    def verify(p):
        key = dsl.show(p)
        if key not in rows:
            ok = right(p, ex, eN, job.get("every_init"), ew, m)
            if ok:
                ok = right(p, cx, cN, job.get("check_init"), cw, m2)
            prim = dsl.expand(p, lib)
            rows[key] = dict(program=key, primitive=dsl.show(prim), primitive_cost=dsl.cost(prim), verified=ok)
            if ok and job.get("general"):
                rows[key]["general"] = general(p, lib, upto=job.get("general_upto", 40))
        return rows[key]

    return prob, verify


def _record(job, res, prob, verify, classes, n_tokens, lib, t_items):
    rec = dict(name=job.get("name"), kind=job.get("kind"), k=job["k"], found_cost=res.found_cost,
               level_sizes=res.level_sizes, best_fit=res.best_fit, NP=len(job["xs"]), seconds=round(res.seconds, 2),
               item_seconds=round(t_items, 2), stop=res.stop, n_tokens=n_tokens, n_classes=len(classes),
               lib_size=len(lib), aliases=[(d, ps[:3]) for d, ps in res.aliases])
    if res.found_cost is not None:
        out = [verify(p) for p in search.programs(res, classes, limit=job.get("limit", 200))]
        rec["programs"] = [r for r in out if r["verified"]]
    if job.get("near"):
        rec["near"] = [(fit, d, dsl.show(p)) for fit, d, p, _ in search.near_misses(res, prob, classes, n=job["near"])]
    return rec


def _vocabulary(job):
    lib = {t: dsl.parse(v) for t, v in job.get("lib", {}).items()}
    kinds = list(job.get("kinds", brm.LOOP_KINDS))
    m = machine_for(job["k"])
    return (lib, m) + classes_for(lib, kinds, job["k"], job.get("max_body", 2), job.get("costs"))


def search_job(job):
    if job.get("engine") == "py":
        from bar3.jobs import search_job as original
        return original({k: v for k, v in job.items() if k != "engine"})
    t0 = time.time()
    lib, m, classes, n_tokens, codes = _vocabulary(job)
    t_items = time.time() - t0
    prob, verify = _posed(job, lib, m)
    res = engine.bfs(prob, classes, lib, m, max_cost=job.get("max_cost", 10),
                     max_states=job.get("max_states", 200_000), time_limit=job.get("time_limit", 300),
                     accept=lambda ps: [verify(p)["verified"] for p in ps], limit=job.get("limit", 200), codes=codes)
    rec = _record(job, res, prob, verify, classes, n_tokens, lib, t_items)
    rec["wall"] = round(time.time() - t0, 2)
    return rec


SEARCH_SPACE = ("k", "xs", "Ns", "init", "lib", "kinds", "max_cost", "max_states", "time_limit", "max_body", "limit",
                "costs")


def same_search(job):
    """What fixes a job's search, apart from its goal: jobs with the same key build the same levels."""
    return json.dumps([job.get(f) for f in SEARCH_SPACE], sort_keys=True)


def search_jobs(jobs):
    """Jobs with the same search space (same_search) in one search (engine.bfs_multi): each gets the record its own
    search_job would give, but for the seconds, which are the shared search's, split evenly."""
    if len(jobs) == 1 or any(j.get("near") or j.get("engine") == "py" for j in jobs):
        return [search_job(j) for j in jobs]
    t0 = time.time()
    lib, m, classes, n_tokens, codes = _vocabulary(jobs[0])
    t_items = time.time() - t0
    posed = [_posed(j, lib, m) for j in jobs]
    j0 = jobs[0]
    accepts = [lambda ps, v=v: [v(p)["verified"] for p in ps] for _, v in posed]
    results = engine.bfs_multi([p for p, _ in posed], classes, lib, m, max_cost=j0.get("max_cost", 10),
                               max_states=j0.get("max_states", 200_000), time_limit=j0.get("time_limit", 300),
                               accepts=accepts, limit=j0.get("limit", 200), codes=codes)
    recs = [_record(j, r, p, v, classes, n_tokens, lib, t_items / len(jobs))
            for j, r, (p, v) in zip(jobs, results, posed)]
    share = round((time.time() - t0) / len(jobs), 2)
    for r in recs:
        r["wall"] = share
        r["seconds"] = share
        r["shared"] = len(jobs)
    return recs
