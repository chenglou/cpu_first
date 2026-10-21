"""Job lists for Modal (modal_tinyrouting.py) and the executor every container runs.

A job is a dict: {"name", "kind": "train" | "impostors" | "saddle" | ..., and the kind's arguments}. Train jobs carry
Cfg fields; their rows keep the curve every `every` steps.
"""

from __future__ import annotations

import itertools

ALL = ("vread", "nread", "shift", "a", "b", "keep", "read")
L4 = ("shift", "a", "b", "keep")

PARAMS = {
    "mix": dict(param="mix"),
    "fmix": dict(param="fmix"),
    "soft": dict(param="soft"),
    "st": dict(param="st"),
    "gumbel": dict(param="gumbel"),
    "reinforce": dict(param="reinforce", K=16),
    "hybrid": dict(param="hybrid", coarse=("vread", "nread", "shift", "keep")),
    "given": dict(param="given"),
}
RECIPES = {
    "ans": dict(),
    "cell": dict(checks={"cell": 1.0}),
    "cellown": dict(checks={"cellown": 1.0}),
    "laws": dict(checks={"laws": 1.0, "range": 1.0}),
    "cell+laws": dict(checks={"cell": 1.0, "laws": 1.0, "range": 1.0}),
}


def held_for(B, w, k):
    return ((B, w, k + 1), (B, w + 1, k + 1))


def n_examples(B, w, k):
    return sum(N * B ** k for N in range(max(2, B ** (w - 1)), B ** w))


def train_job(name, B, w, k, param, recipe, seed, learn=ALL, steps=2000, **kw):
    """One training run; sizes with more than 300 examples train on batches of 128."""
    cfg = dict(B=B, w=w, k=k, learn=learn, seed=seed, steps=steps, held=held_for(B, w, k),
               batch=128 if n_examples(B, w, k) > 300 else 0)
    cfg.update(PARAMS[param])
    cfg.update(RECIPES[recipe])
    cfg.update(kw)
    return {"name": name, "kind": "train", "cfg": cfg}


SIZES = [(3, 2, 1), (3, 2, 2), (4, 2, 1), (3, 3, 1), (4, 2, 2), (5, 2, 1), (3, 3, 2)]


def phase():
    """The phase map: parametrization x recipe x size, 7 muxes (1,152 wirings), 4 seeds. The whole-program mixtures
    from answers alone (slow: every wiring every step) only at the three smallest sizes, 2 seeds beyond (3, 2, 1),
    batch 64."""
    jobs = []
    for (B, w, k), param, recipe, seed in itertools.product(
            SIZES, ["mix", "fmix", "soft", "st", "hybrid"], ["ans", "cell"], range(4)):
        kw = {}
        if param in ("mix", "fmix") and recipe == "ans" and (B, w, k) != (3, 2, 1):
            if (B, w, k) not in ((3, 2, 2), (4, 2, 1)) or seed >= 2:
                continue
            kw = dict(batch=64, steps=1500)
        if param in ("mix", "fmix") and recipe == "cell" and n_examples(B, w, k) > 300:
            kw = dict(batch=64)
        jobs.append(train_job(f"{B}{w}{k}_{param}_{recipe}_s{seed}", B, w, k, param, recipe, seed, **kw))
    return jobs


def alone():
    """The wiring alone: the table given exact, 7 muxes; every parametrization; bigger sizes too."""
    jobs = []
    sizes = SIZES + [(5, 2, 2), (3, 3, 3), (2, 3, 3), (2, 4, 3)]
    for (B, w, k), param, seed in itertools.product(
            sizes, ["mix", "fmix", "soft", "st", "gumbel", "reinforce", "hybrid"], range(4)):
        if param in ("gumbel", "reinforce") and (B, w, k) not in ((3, 2, 1), (3, 2, 2), (4, 2, 1)):
            continue
        kw = dict(batch=64, steps=1000) if param in ("mix", "fmix") and n_examples(B, w, k) > 99 else {}
        jobs.append(train_job(f"{B}{w}{k}_{param}_s{seed}", B, w, k, param, "ans", seed, table="exact", **kw))
    return jobs


TRICKS_ANS = {
    "none": dict(),
    "anchor": dict(checks={"anchor": 1.0}),
    "laws": dict(checks={"laws": 1.0, "range": 1.0}),
    "anchor+laws": dict(checks={"anchor": 1.0, "laws": 1.0, "range": 1.0}),
    "tablefirst": dict(route_freeze=200),
    "init1": dict(init_route=1.0),
    "sym": dict(sym_init=True),
    "search": dict(search={"every": 50}),
}
TRICKS_CELL = {
    "none": dict(),
    "ownpath": dict(own_path=True),
    "search": dict(search={"every": 50}),
    "init1": dict(init_route=1.0),
    "tablefirst": dict(route_freeze=50),
    "laws": dict(checks={"cell": 1.0, "laws": 1.0, "range": 1.0}),
    "sym": dict(sym_init=True),
}


def tricks():
    """The known tricks on the routing problem, 7 muxes: from answers alone (every parametrization fails
    there), and with the cell check (where the hard-path routing fails). 4 seeds."""
    jobs = []
    for (B, w, k), seed in itertools.product([(3, 2, 1), (3, 2, 2)], range(4)):
        for param in ("fmix", "soft", "st"):
            for t, kw in TRICKS_ANS.items():
                if t == "search" and param == "fmix" or param == "fmix" and (B, w, k) != (3, 2, 1):
                    continue
                jobs.append(train_job(f"ans_{t}_{B}{w}{k}_{param}_s{seed}", B, w, k, param, "ans", seed, **kw))
            for t, kw in TRICKS_CELL.items():
                if t in ("ownpath", "search") and param == "fmix":
                    continue
                kw = dict(kw)
                kw.setdefault("checks", {"cell": 1.0})
                jobs.append(train_job(f"cell_{t}_{B}{w}{k}_{param}_s{seed}", B, w, k, param, "cell", seed, **kw))
    return jobs


def tricks2():
    """The tricks on the joint whole-program mixture from answers alone at (3, 2, 1): the one parametrization that
    keeps every wiring's own weight alive. 4 seeds."""
    jobs = []
    for seed in range(4):
        for t, kw in TRICKS_ANS.items():
            if t == "search":
                continue
            jobs.append(train_job(f"ans_{t}_321_mix_s{seed}", 3, 2, 1, "mix", "ans", seed, **kw))
    return jobs


def stages():
    """One quotient digit at a time vs all at once, with the cell check and from answers alone."""
    jobs = []
    for (B, w, k), seed in itertools.product([(3, 2, 3), (4, 2, 2), (2, 4, 3)], range(4)):
        for param in ("fmix", "soft", "st"):
            for recipe in ("ans", "cell"):
                if param == "fmix" and recipe == "ans":
                    continue
                jobs.append(train_job(f"joint_{recipe}_{B}{w}{k}_{param}_s{seed}", B, w, k, param, recipe, seed))
                jobs.append(train_job(f"staged_{recipe}_{B}{w}{k}_{param}_s{seed}", B, w, k, param, recipe, seed,
                                      stages=tuple(range(1, k + 1)), stage_steps=500))
    return jobs


def untie():
    """Tied (one mux per kind of connection) vs untied (one per step and link) routing, joint vs one step at a
    time; the table given or taught by the cell check. 4 seeds."""
    jobs = []
    for (B, w, k), seed in itertools.product([(3, 2, 2), (3, 2, 3), (4, 2, 2), (2, 3, 3), (3, 3, 2)], range(4)):
        for table in ("exact", "cell"):
            tkw = dict(table="exact") if table == "exact" else dict(checks={"cell": 1.0})
            for param, ut in [("soft", ()), ("soft", ("shift", "keep")), ("soft", ("a", "b")),
                              ("soft", ("shift", "a", "b", "keep")), ("st", ("shift", "a", "b", "keep"))]:
                for staged in (False, True):
                    if not ut and staged:
                        continue
                    kw = dict(tkw, untie=ut)
                    if staged:
                        kw.update(stages=tuple(range(1, k + 1)), stage_steps=600)
                    name = f"{table}_{param}_{'+'.join(ut) or 'tied'}_{'staged' if staged else 'joint'}_{B}{w}{k}_s{seed}"
                    jobs.append(train_job(name, B, w, k, param, "ans", seed, **kw))
    return jobs


def school():
    """The schoolbook core: tied vs untied column routing, labels on the product or only after mod N."""
    jobs = []
    for (B, w), seed in itertools.product([(3, 2), (3, 3), (4, 2)], range(4)):
        for route in ("tied", "rel", "abs"):
            jobs.append({"name": f"rladder_none_{route}_soft_{B}{w}_s{seed}", "kind": "school",
                         "cfg": dict(B=B, w=w, route=route, param="soft", labels="mod", seed=seed, steps=2000,
                                     answers=0.0, rladder=1.0, table="learn", ladder=1.0)})
        for labels in ("product", "mod"):
            for route, param in [("tied", "mix"), ("tied", "soft"), ("tied", "st"), ("rel", "soft"), ("rel", "st"),
                                 ("abs", "soft")]:
                jobs.append({"name": f"exact_{labels}_{route}_{param}_{B}{w}_s{seed}", "kind": "school",
                             "cfg": dict(B=B, w=w, route=route, param=param, labels=labels, seed=seed, steps=2000)})
            for route in ("tied", "rel"):
                jobs.append({"name": f"ladder_{labels}_{route}_soft_{B}{w}_s{seed}", "kind": "school",
                             "cfg": dict(B=B, w=w, route=route, param="soft", labels=labels, seed=seed, steps=2000,
                                         table="learn", ladder=1.0)})
    return jobs


COMPOSITE = {
    "laws": dict(table="learn", ladder=1.0, rladder=1.0, div_table="learn", cell=1.0),
    "noroute": dict(table="learn", ladder=1.0, div_table="learn", cell=1.0),
    "answers": dict(table="learn", div_table="learn"),
}


def composite(sizes=((3, 2), (4, 2)), seeds=None):
    """H1's step in miniature, multiply then reduce, every connection and both tables learned: the placement (tied or
    per digit) and the division's 7 switches, graded only on x*y mod N. "hybrid": the tied placement with the
    division's reading, shift and readout enumerated (32 sub-wirings), the rest soft. 6 seeds at (3, 2), 4 beyond."""
    ALLD = ("vread", "nread", "shift", "a", "b", "keep", "read")
    jobs = []
    for B, w in sizes:
        for seed in range(seeds or (6 if (B, w) == (3, 2) else 4)):
            for recipe, kw in COMPOSITE.items():
                for route, param in [("tied", "soft"), ("rel", "soft"), ("tied", "st")]:
                    if recipe != "laws" and param == "st":
                        continue
                    jobs.append({"name": f"{recipe}_{route}_{param}_{B}{w}_s{seed}", "kind": "school",
                                 "cfg": dict(B=B, w=w, route=route, param=param, labels="mod", seed=seed, steps=2000,
                                             div_learn=ALLD, **kw)})
            if seed < 4:
                jobs.append({"name": f"hybrid_tied_soft_{B}{w}_s{seed}", "kind": "school",
                             "cfg": dict(B=B, w=w, route="tied", param="soft", labels="mod", seed=seed, steps=2000,
                                         div_learn=ALLD, div_coarse=("vread", "nread", "shift", "read"),
                                         **COMPOSITE["laws"])})
    return jobs


def census():
    """Impostor census over all 1,152 wirings, each with its own best-fit table (answers alone; answers + cell)."""
    jobs = []
    for size, cell, chunk in itertools.product([(3, 2, 1)], (0.0, 1.0), range(16)):
        jobs.append({"name": f"census_{''.join(map(str, size))}_cell{int(cell)}_c{chunk}", "kind": "census",
                     "args": dict(size=size, chunk=chunk, nchunks=16, cell=cell)})
    return jobs


def depth():
    """How the flat stretch grows with the number of quotient digits: tied soft routing and the factorized mixture,
    the table given or taught by the cell check, all at once vs one quotient digit at a time. 4 seeds."""
    jobs = []
    sizes = [(3, 2, k) for k in range(1, 6)] + [(2, 3, k) for k in range(1, 7)]
    for (B, w, k), seed in itertools.product(sizes, range(4)):
        for param in ("soft", "fmix"):
            if param == "fmix" and (B, w, k) not in ((3, 2, 1), (3, 2, 2), (3, 2, 3), (2, 3, 1), (2, 3, 2),
                                                     (2, 3, 3), (2, 3, 4)):
                continue
            for table in ("exact", "cell"):
                tkw = dict(table="exact") if table == "exact" else dict(checks={"cell": 1.0})
                if param == "fmix":
                    tkw.update(batch=64, steps=1500)
                for staged in (False, True):
                    if staged and k == 1:
                        continue
                    kw = dict(tkw, steps=4000) if param == "soft" else dict(tkw)
                    if staged:
                        kw.update(stages=tuple(range(1, k + 1)), stage_steps=600)
                    name = f"{table}_{param}_{'staged' if staged else 'joint'}_{B}{w}{k}_s{seed}"
                    jobs.append(train_job(name, B, w, k, param, "ans", seed, **kw))
    return jobs


SUITES = {"phase": phase, "alone": alone, "tricks": tricks, "stages": stages, "untie": untie, "school": school,
          "census": census, "depth": depth, "tricks2": tricks2, "composite": composite}


def execute(job):
    import torch
    kind = job["kind"]
    if kind == "train":
        from tinyrouting.train import Cfg, Trainer
        cfg = dict(job["cfg"])
        for f in ("learn", "coarse", "stages"):
            if f in cfg:
                cfg[f] = tuple(cfg[f])
        if "held" in cfg:
            cfg["held"] = tuple(tuple(h) for h in cfg["held"])
        r = Trainer(Cfg(**cfg)).train()
        r["name"] = job["name"]
        return r
    if kind == "school":
        from tinyrouting.school import School, SCfg
        r = School(SCfg(**job["cfg"])).train()
        r["name"] = job["name"]
        return r
    if kind == "census":
        from tinyrouting.exact import census_chunk
        args = dict(job["args"])
        args["size"] = tuple(args["size"])
        return {"name": job["name"], "args": job["args"], "rows": census_chunk(**args), "hit": None,
                "certified": None}
    if kind == "exact":
        import io
        import contextlib
        from tinyrouting import exact
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            getattr(exact, job["fn"])(**job.get("args", {}))
        return {"name": job["name"], "text": buf.getvalue()}
    raise ValueError(kind)
