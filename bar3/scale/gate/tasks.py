"""The arms' tasks (GATE.md §4, PREREG.md "Arms and measures"): every test problem of each fold, from the captures,
as search jobs under each arm, written to a task file for modal_gate.evaluate.

A problem is the learner's own posed one (its instances, vocabulary and acceptance check), identified across its two
probes (30k, then 1M states). The plain arm is the capture itself: the first probe that solved it (its states to
solution, `first_accept`) or, unsolved, the states its 1M probe spent. Every other arm runs the 1M probe's job with a
hard cap on states:
- solved by the plain search: 20 times its states to solution (at least 100k, at most 5M), so a slower arm is still
  measured;
- unsolved: the states the plain search spent on it, at most 5M (the gated engine keeps every level it expands);
and the targets themselves (the target probe's job) at 3M under every arm, the plain one included.
Arms: gate (the other fold's gate, full features), gate_prune (with dead states never expanded), prune (that alone),
the ablations' gates (on the solved problems and a sample of 50 unsolved), and guided (SCALE.md §4.4's learned costs,
on a carried library only).

    cd cpu_first && <python with numba> -m bar3.scale.gate.tasks CORPUS MODELS_DIR OUT.jsonl.gz CAPTURE.jsonl.gz ...
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys

import numpy as np

from bar3.scale.gate.train import FOLDS, SHARED, target_of

SAME = ("kind", "max_states", "time_limit", "near")
ABLATIONS = ("noprob", "fitonly", "noanswer", "nodet")


def problem_id(job):
    return hashlib.sha1(json.dumps({k: v for k, v in job.items() if k not in SAME}, sort_keys=True).encode()
                        ).hexdigest()[:16]


def load(paths):
    """problem id -> dict(job (the last probe's), runs, source, a (probe_a record), b (probe_b record), t (target))."""
    jobs, probs = {}, {}
    for p in paths:
        for line in gzip.open(p, "rt"):
            run, key, job, rec, _ = json.loads(line)
            if job is not None:
                jobs[key] = job
            job = jobs[key]
            pid = problem_id(job)
            d = probs.setdefault(pid, dict(id=pid, job=job, runs=set(), source=job["name"].split("/")[0]))
            d["runs"].add(run)
            kind = job["kind"]
            if kind == "probe_a":
                d["a"] = rec
            elif kind == "probe_b":
                d["b"], d["job"] = rec, job
            elif kind == "target_probe":
                d["t"] = rec
    return probs


def plain_outcome(d):
    for r in (d.get("a"), d.get("b")):
        if r and "error" not in r and r.get("found_cost") is not None and r.get("programs"):
            return dict(solved=True, cost=r["found_cost"], states=r["first_accept"])
    r = d.get("b") or d.get("a")
    return dict(solved=False, cost=None, states=(r or {}).get("generated"))


def build(corpus, models, paths, b_cap=5_000_000, seed=0):
    from library.inventing import brm
    from bar3.scale import learn
    probs = load(paths)
    rng = np.random.default_rng(seed)
    tasks = []
    for fold, other in (("A", "B"), ("B", "A")):
        gate = f"{models}/{corpus}_{fold}_full.json"
        test = [d for d in probs.values()
                if d["source"] not in SHARED and {target_of(r) for r in d["runs"]} <= set(FOLDS[other])]
        unsolved = [d for d in test if d["source"] != "target" and not plain_outcome(d)["solved"]]
        sample = {d["id"] for d in rng.permutation(np.array(unsolved, dtype=object))[:50]} if unsolved else set()
        for d in test:
            job = dict(d["job"])
            job.pop("near", None)
            lib = job.get("lib", {})
            if d["source"] == "target":
                job.update(kind="target", time_limit=3600, general=False)
                cap = 3_000_000
                arms = ["plain", "gate", "gate_prune", "prune"] + (["guided"] if lib else [])
            else:
                po = plain_outcome(d)
                if po["solved"]:
                    if po["cost"] < 2:
                        continue
                    cap = int(min(5_000_000, max(100_000, 20 * po["states"])))
                    arms = ["gate", "gate_prune", "prune", *ABLATIONS] + (["guided"] if lib else [])
                else:
                    cap = int(min(b_cap, po["states"] or b_cap))
                    arms = ["gate", "gate_prune", "prune"] + (["guided"] if lib else [])
                    if d["id"] in sample:
                        arms += list(ABLATIONS)
                job["time_limit"] = 3600
            base = dict(job, budget="hard", max_states=cap)
            for arm in arms:
                j = dict(base)
                if arm in ("gate", "gate_prune"):
                    j["gate"] = gate
                if arm in ABLATIONS:
                    j["gate"] = f"{models}/{corpus}_{fold}_{arm}.json"
                if arm in ("prune", "gate_prune"):
                    j["prune"] = True
                if arm == "guided":
                    j["costs"] = learn.learned_costs(lib, job.get("kinds", list(brm.LOOP_KINDS)))
                    j["max_cost"] = 80
                tasks.append(dict(id=d["id"], arm=arm, fold=other, corpus=corpus, job=j))
    return tasks, probs


def main():
    corpus, models, out = sys.argv[1:4]
    tasks, probs = build(corpus, models, sys.argv[4:])
    with gzip.open(out, "wt") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    by = {}
    for t in tasks:
        by[t["arm"]] = by.get(t["arm"], 0) + 1
    print(f"{corpus}: {len(probs)} problems, {len(tasks)} tasks: {by}")


if __name__ == "__main__":
    main()
