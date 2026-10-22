"""Arms C and D on the digit instructions (the library study's dsl.py): the target only at growing sizes, plus
self-made practice. Sizes are digit counts: settings (B, W) in bases 2 and 3.

A digit job is either the target (x^2 mod N on random pairs at its settings, search.Problem) or a dream (inputs and
the answers a random program gave, DreamProblem). A target program is accepted when it is right on 200 fresh pairs at
every setting of its size; `general` is its worst share over the library study's eight verification settings.
"""

from __future__ import annotations

import time

import numpy as np

from library import dsl, search
from library.jobs import VERIFY
from library.tasks import gen


class DreamProblem:
    """search.Problem's interface for a task given as inputs and answers per setting."""

    def __init__(self, settings, inputs, answers, seed=0):
        rng = np.random.default_rng(seed)
        self.task, self.settings = "dream", settings
        self.roots, self.targets = [], []
        for st, ins, ans in zip(settings, inputs, answers):
            ins = {r: np.asarray(v, np.uint8) for r, v in ins.items()}
            n = len(next(iter(ins.values())))
            self.roots.append(dsl.fresh(st, ins, rng, n))
            self.targets.append({"A": np.asarray(ans, np.uint8).T})

    def goal(self, batches):
        ok = np.ones(batches[0].S, bool)
        for b, t in zip(batches, self.targets):
            ok &= (b.regs[:, dsl.A] == t["A"][None]).all(axis=(1, 2))
        return ok


def run_on(prog, lib, st, ins, seed):
    """A's digits after prog on inputs {reg: [n, P] digits}, from a random start drawn with seed."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(ins.values())))
    b = dsl.fresh(st, {r: np.asarray(v, np.uint8) for r, v in ins.items()}, rng, n)
    dsl.run_items(st, b, prog, lib)
    return b.regs[0, dsl.A].T.copy()


def target_inputs(st, n, seed):
    ins, tgt = gen("sqmod", st.B, st.W, n, np.random.default_rng(seed))
    return ins, tgt["A"]


def digit_search_job(job):
    t0 = time.time()
    lib = {k: dsl.parse(v) for k, v in job.get("lib", {}).items()}
    kinds = list(job.get("kinds", dsl.LOOP_KINDS))
    tokens = list(dsl.PRIMS) + [k for k in lib if k not in kinds]
    settings = [dsl.Setting(B, W) for B, W in job["settings"]]
    classes = search.enumerate_items(tokens, lib, settings, kinds=kinds, max_body=job.get("max_body", 2),
                                     max_arity=3)
    if job["task"] == "target":
        prob = search.Problem("sqmod", settings, NP=job.get("NP", 8), seed=job.get("seed", 0))
    else:
        prob = DreamProblem(settings, job["inputs"], job["answers"])
    rows = {}

    def verify(p):
        key = dsl.show(p)
        if key not in rows:
            if job["task"] == "target":
                ok = min(search.check("sqmod", p, lib, settings, n=200).values()) == 1.0
            else:
                ok = all((run_on(p, lib, st, ins, s) == np.asarray(ans)).all()
                         for st, ins, ans in zip(settings, job["check_inputs"], job["check_answers"]) for s in (1, 2))
            rows[key] = dict(program=key, primitive=dsl.show(dsl.expand(p, lib)),
                             primitive_cost=dsl.cost(dsl.expand(p, lib)), verified=bool(ok))
            if ok and job["task"] == "target":
                rows[key]["general"] = min(search.check("sqmod", p, lib, [dsl.Setting(B, W) for B, W in VERIFY],
                                                        n=200).values())
        return rows[key]

    res = search.bfs(prob, classes, lib, max_cost=job.get("max_cost", 10), max_states=job.get("max_states", 3_000_000),
                     time_limit=job.get("time_limit", 1800), accept=lambda ps: [verify(p)["verified"] for p in ps],
                     limit=job.get("limit", 200))
    rec = dict(name=job.get("name"), task=job["task"], k=job["k"], found_cost=res.found_cost,
               level_sizes=res.level_sizes, seconds=round(res.seconds, 2), stop=res.stop, expanded=int(res.expanded),
               n_tokens=len(tokens), n_classes=len(classes), lib_size=len(lib), aliases=res.aliases)
    if res.found_cost is not None:
        out = [verify(p) for p in search.programs(res, classes, limit=job.get("limit", 200))]
        rec["programs"] = out
        rec["n_programs"] = len(out)
        rec["n_verified"] = sum(r["verified"] for r in out)
    rec["wall"] = round(time.time() - t0, 2)
    return rec
