"""A search job on the gated engine (gengine.py): bar3/scale/jobs.search_job with three more fields in the job,
- "gate": a gate (model.Gate as JSON, or a path to one); none runs the plain search;
- "budget": "level" (the original's rule, the default) or "total" (the running total; gengine.bfs);
- "max_level": the priced levels' bound with a gate (default max_cost * (1 + D));
- "prune": never expand dead states (gengine.dead_states);
and more in the record: `first_goal`, `first_accept` and `generated` (gengine.bfs: candidates generated when the
first state fitting every probe appeared, when the first with an accepted program appeared, and in all), `states` (the
sum of level_sizes, as the studies count a search) and `delays` (each kept level's histogram of delays).
"""

from __future__ import annotations

import base64
import time

import numpy as np

from library import dsl
from library.inventing import brm
from library.inventing.jobs import machine_for
from library.search import hash_rows
from bar3.scale import jobs as J
from bar3.scale.gate import gengine
from bar3.scale.gate.model import Gate

_GATES = {}


def gate_of(spec):
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec not in _GATES:
            _GATES[spec] = Gate.load(spec)
        return _GATES[spec]
    return Gate.from_json(spec)


def search_job(job, harvest=0):
    """The record; with harvest = n (> 0), also the training states of a solved search (harvest_states)."""
    t0 = time.time()
    lib, m, classes, n_tokens, codes = J._vocabulary(job)
    t_items = time.time() - t0
    prob, verify = J._posed(job, lib, m)
    gate = gate_of(job.get("gate"))
    delay = gate.for_problem(prob) if gate is not None else None
    max_cost = job.get("max_cost", 10)
    res = gengine.bfs(prob, classes, lib, m, delay=delay, max_cost=max_cost,
                      max_states=job.get("max_states", 200_000), time_limit=job.get("time_limit", 300),
                      accept=lambda ps: [verify(p)["verified"] for p in ps], limit=job.get("limit", 200), codes=codes,
                      budget=job.get("budget", "level"),
                      max_level=job.get("max_level") or (max_cost * (1 + gate.D) if gate is not None else None),
                      prune=bool(job.get("prune")))
    rec = J._record(job, res, prob, verify, classes, n_tokens, lib, t_items)
    if gate is not None and rec.get("programs"):
        # A gated level counts the delays, and the learner reads found_cost as the program's price (hardest-first
        # picks, priors, gain thresholds): give the cheapest accepted program's price and those programs, as the plain
        # search's record does (every program there has the price of its level).
        price = [sum(dsl.item_cost(it) for it in dsl.parse(p["program"])) for p in rec["programs"]]
        rec["gated_level"] = rec["found_cost"]
        rec["found_cost"] = min(price)
        rec["programs"] = [p for p, c in zip(rec["programs"], price) if c == min(price)]
    rec["wall"] = round(time.time() - t0, 2)
    rec["first_goal"] = res.first_goal
    rec["first_accept"] = res.first_accept
    rec["generated"] = res.generated
    rec["states"] = int(sum(res.level_sizes))
    rec["delays"] = res.delays if gate is not None else None
    rec["dead"] = res.dead if job.get("prune") else None
    if harvest:
        return rec, harvest_states(job, rec, res, lib, harvest)
    return rec


def pack(a):
    a = np.ascontiguousarray(a, np.int16)
    return [list(a.shape), base64.b64encode(a.tobytes()).decode()]


def unpack(p):
    return np.frombuffer(base64.b64decode(p[1]), np.int16).reshape(p[0])


def harvest_states(job, rec, res, lib, n_neg, seed=0):
    """GATE.md §3, the gate's training states from one solved search: the states the accepted programs pass through
    after each item but the last (positives, with their levels), and up to n_neg states of each kept level below the
    solution's that are not positives (negatives, drawn at random). Registers on the search's probes, as int16."""
    progs = rec.get("programs") or []
    if res.found_cost is None or not progs:
        return None
    m = machine_for(job["k"])
    pos, plev, keys = [], [], set()
    for p in progs:
        items = dsl.parse(p["program"])
        b = J.start(np.array(job["xs"]), np.array(job["Ns"]), job.get("init"))
        cost = 0
        for it in items[:-1]:
            brm.run_items(m, b, (it,), lib)
            cost += dsl.item_cost(it)                  # the plain search's price (harvest runs on plain searches)
            k = int(hash_rows(b.features())[0])
            if k not in keys:
                keys.add(k)
                pos.append(b.regs[0].copy())
                plev.append(cost)
    rng = np.random.default_rng(seed)
    neg, nlev = [], []
    for d in range(1, res.found_cost):
        lv = res.levels[d] if d < len(res.levels) else None
        if lv is None or not lv.S:
            continue
        take = rng.choice(lv.S, min(n_neg, lv.S), replace=False)
        ks = hash_rows(lv.take(take).features())
        for i, k in zip(take.tolist(), ks.tolist()):
            if int(k) not in keys:
                neg.append(lv.regs[i])
                nlev.append(d)
    out = dict(found=res.found_cost, n_pos=len(pos), n_neg=len(neg),
               pos=pack(np.array(pos)) if pos else None, plev=plev,
               neg=pack(np.array(neg)) if neg else None, nlev=nlev)
    return out
