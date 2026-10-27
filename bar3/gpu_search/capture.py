"""The stored searches as jobs (GPU_SEARCH.md §3): a store holds (job key, record) only, so the jobs are recovered by
replaying a learner run from its stores, as bar3/scale/newton/run.py does: every search the run asks for is looked up
by pool.py's key (the sha1 of the job's JSON); a job no store holds runs here on the compiled CPU engine. Every job is
written once, with its key, where its record came from ("store" or "ran") and the record:

    [key, "store" | "ran", job, record]        one line each, gzipped

The replay is the learner's own: "newton" for the two-source study's (bar3/scale/newton/learn.py), "subgoal" for the
frozen learner of the subgoal and held-out studies, "digits" for the digits study's (bar3/scale/digits/learn.py, its
jobs and the D.B block), "picks", "rowcheck", "reach" and "consolidate" for those studies' copies (their
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
    if spec == "consolidate":       # its check.learner reads reach's LEARNER table: the copy is picked here
        L = importlib.import_module(f"bar3.scale.consolidate.learn_{configs.LEARNER[config]}")
        if configs.LEARNER[config] == "digits":
            from bar3.scale.digits import jobs as DJ
            return L, DJ.search_job, L.DigitsTap, cfg, True
        from bar3.scale.jobs import search_job
        return L, search_job, Tap, cfg, False
    if spec in ("picks", "rowcheck", "reach"):
        L, search_job, tap = importlib.import_module(f"bar3.scale.{spec}.check").learner(config)
        return L, search_job, tap, cfg, configs.LEARNER[config] == "digits"
    if spec == "digits":
        from bar3.scale.digits import jobs as DJ     # the D.B block, the digits targets and jobs
        from bar3.scale.digits import learn as L
        return L, DJ.search_job, L.DigitsTap, cfg, True
    from bar3.scale.jobs import search_job
    return importlib.import_module(f"bar3.scale.{spec}.learn"), search_job, Tap, cfg, False


def main():
    capture_run(*sys.argv[1:6], sys.argv[6:])


class Missing(Exception):
    pass


def capture_run(spec, config, tname, tag, out, stores, stored_only=False):
    """With stored_only, the replay ends at the first job no store holds (a run cut short, or a failed job): the jobs
    captured are then exactly the ones the run asked for, and nothing is searched here."""
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
            if r is None and stored_only:
                count["missing"] = count.get("missing", 0) + 1
                print(f"  not stored: {j['kind']} {j['name']}: the replay ends here", flush=True)
                raise Missing
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
    try:
        learn.run(name, cfg, tap.mapper, tap.emit_with_looks(lambda r: rounds.append(r)), log=print)
    except Missing:
        pass
    fh.close()
    with open(out.replace(".jsonl.gz", ".rounds.jsonl"), "w") as f:
        for r in rounds:
            f.write(json.dumps(dict(r, target_name=tname)) + "\n")
    print(f"{name}: {len(written)} jobs ({count['store']} from the stores, {count['ran']} run here"
          + (", then one not stored: stopped" if count.get("missing") else "")
          + f"), {len(rounds)} rounds, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
