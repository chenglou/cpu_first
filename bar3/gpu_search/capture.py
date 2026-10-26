"""The stored searches as jobs (GPU_SEARCH.md §3): a store holds (job key, record) only, so the jobs are recovered by
replaying a learner run from its stores, as bar3/scale/newton/run.py does: every search the run asks for is looked up
by pool.py's key (the sha1 of the job's JSON); a job no store holds runs here on the compiled CPU engine. Every job is
written once, with its key, where its record came from ("store" or "ran") and the record:

    [key, "store" | "ran", job, record]        one line each, gzipped

The replay is the learner's own: "newton" for the two-source study's (bar3/scale/newton/learn.py), "subgoal" for the
frozen learner of the subgoal and held-out studies, "picks" and "rowcheck" for those studies' copies (their
check.learner picks the copy, and the digits search for digits configs), each with its configs.py.

    cd cpu_first && <python with numba> -m bar3.gpu_search.capture LEARNER CONFIG TARGET TAG OUT.jsonl.gz STORE ...
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import sys
import time

from bar3.breadth import targets as TG
from bar3.breadth.drive import Tap
from bar3.scale.heldout import targets as _ho  # noqa: F401  (registers the held-out targets)
from bar3.scale.subgoal.attribute import load_stores


def key_of(job):
    return hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()


def learner_of(spec, config):
    """(learn module, search_job, Tap class, the config) for a learner spec, as that study's run.py builds them."""
    configs = importlib.import_module(f"bar3.scale.{spec}.configs")
    cfg = json.loads(json.dumps(configs.CONFIGS[config]))
    if spec in ("picks", "rowcheck"):
        L, search_job, tap = importlib.import_module(f"bar3.scale.{spec}.check").learner(config)
        return L, search_job, tap, cfg, configs.LEARNER[config] == "digits"
    from bar3.scale.jobs import search_job
    return importlib.import_module(f"bar3.scale.{spec}.learn"), search_job, Tap, cfg, False


def main():
    spec, config, tname, tag, out = sys.argv[1:6]
    stores = sys.argv[6:]
    learn, search_job, TapClass, cfg, digits = learner_of(spec, config)
    name = f"{config}_{tname}_{tag}"
    cache = load_stores(stores)
    t0 = time.time()
    written = set()
    count = dict(store=0, ran=0)
    fh = gzip.open(out, "wt")

    def search_many(jobs):
        recs = []
        for j in jobs:
            key = key_of(j)
            r = cache.get(key)
            src = "store"
            if r is None:
                r = json.loads(json.dumps(search_job(j)))
                cache[key] = r
                src = "ran"
                print(f"  ran {j['kind']} {j['name']}: cost {r.get('found_cost')}, {r.get('wall', 0):.0f}s", flush=True)
            if key not in written:
                written.add(key)
                count[src] += 1
                fh.write(json.dumps([key, src, j, r]) + "\n")
            recs.append(r)
        return recs

    rounds = []
    tap = TapClass(search_many)
    TG.use(tname)
    tap.target = tname
    if digits:
        cfg = dict(cfg, target=tname)
    learn.run(name, cfg, tap.mapper, tap.emit_with_looks(lambda r: rounds.append(r)), log=print)
    fh.close()
    with open(out.replace(".jsonl.gz", ".rounds.jsonl"), "w") as f:
        for r in rounds:
            f.write(json.dumps(dict(r, target_name=tname)) + "\n")
    print(f"{name}: {len(written)} jobs ({count['store']} from the stores, {count['ran']} run here), "
          f"{len(rounds)} rounds, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
