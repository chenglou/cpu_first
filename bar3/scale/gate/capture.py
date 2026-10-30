"""The corpus (GATE.md §3): the learner's round 0, every search it asks for run here and written out, with the training
states of every solved one (gjobs.harvest_states). The learner is the consolidation study's copy (learn_newton, the
allocator with anchored pairs and the uniform probe: `alloc_deep_ct2a2_probe`, the arm that kept 13 of the 16 earlier
wins), so the problems are exactly those it poses. The target probe and both probes (30k states, then 1M for the
unsolved) are searched for real; the 10M deep probe is answered "unsolved" without a search; at the first look the
round is stopped (nothing after the probes is needed).

Output, one gzipped JSON line per search: [run, key, job or None (when the same key was written before), record,
harvest or None]. Identical jobs (the law book is the same for every target) run once per container.

    cd cpu_first && <python with numba> -m bar3.scale.gate.capture OUT.jsonl.gz CONFIG TARGET[@STATE] ... [--tiny]
"""

from __future__ import annotations

import concurrent.futures as cf
import gzip
import hashlib
import json
import multiprocessing as mp
import sys
import threading
import time
import traceback

RUN = ("target_probe", "probe_a", "probe_b")
STUB = ("probe_deep",)


class Stop(Exception):
    pass


def key_of(job):
    return hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()


def _warm():
    from bar3.scale.heldout import targets  # noqa: F401  (the held-out targets, in this worker)
    from bar3.scale.pool import warm
    warm()


def _work(job, harvest):
    from bar3.breadth import targets as TG
    from bar3.scale.gate import gjobs
    if job.get("breadth_target"):
        TG.use(job["breadth_target"])
    return gjobs.search_job(job, harvest=harvest)


class Runner:
    """[job] -> [(key, record, harvest)] on a process pool; identical jobs run once (in flight or done)."""

    def __init__(self, workers, harvest=64):
        self.pool = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_warm)
        self.harvest = harvest
        self.lock = threading.Lock()
        self.cache = {}

    def __call__(self, jobs):
        keyed = [(key_of(j), j) for j in jobs]
        with self.lock:
            for key, j in keyed:
                if key not in self.cache:
                    self.cache[key] = self.pool.submit(_work, j, self.harvest)
            futs = [self.cache[key] for key, _ in keyed]
        out = []
        for (key, _), f in zip(keyed, futs):
            try:
                rec, harv = f.result()
                out.append((key, rec, harv))
            except Exception as e:  # noqa: BLE001  (the learner treats a failed job as failed)
                out.append((key, e, None))
        return out


def stub(j):
    return dict(name=j.get("name"), kind=j["kind"], k=j["k"], found_cost=None, level_sizes=[1], best_fit=[0],
                NP=len(j["xs"]), seconds=0.0, item_seconds=0.0, stop="stub (capture)", n_tokens=0, n_classes=0,
                lib_size=len(j.get("lib", {})), aliases=[], wall=0.0)


class Capture:
    """The learner's mapper for one run: RUN kinds searched, STUB kinds answered unsolved, anything else stops."""

    def __init__(self, run_name, runner, writer):
        self.run_name, self.runner, self.writer = run_name, runner, writer

    def __call__(self, jobs):
        kinds = {j["kind"] for j in jobs}
        if kinds - set(RUN) - set(STUB):
            raise Stop(sorted(kinds))
        real = [j for j in jobs if j["kind"] in RUN]
        got = iter(self.runner(real) if real else [])
        out = []
        for j in jobs:
            if j["kind"] in STUB:
                out.append(stub(j))
                continue
            key, rec, harv = next(got)
            self.writer(self.run_name, key, j, rec, harv)
            out.append(rec)
        return out


class Writer:
    def __init__(self, path):
        self.f = gzip.open(path, "wt")
        self.lock = threading.Lock()
        self.written = set()

    def __call__(self, run, key, job, rec, harv):
        if not isinstance(rec, dict):
            rec = dict(error=repr(rec)[:300])
        with self.lock:
            first = key not in self.written
            self.written.add(key)
            self.f.write(json.dumps([run, key, job if first else None, rec, harv if first else None]) + "\n")

    def close(self):
        self.f.close()


def capture_run(name, config, tname, runner, writer, state=None, tiny=False, log=print):
    """Round 0 of one run: the consolidation copy, `config` (consolidate/configs.py), on target tname, from the carried
    state file `state` (a sequence's *.state.*.json) or from scratch."""
    from bar3 import learner as BL
    from bar3.breadth import targets as TG
    from bar3.breadth.drive import Tap
    from bar3.scale.consolidate import learn_newton as LN
    from bar3.scale.consolidate.configs import CONFIGS
    cfg = json.loads(json.dumps(CONFIGS[config]))
    cfg["rounds"] = 1
    if tiny:
        cfg["budgets"] = json.loads(json.dumps(BL.TINY))
    from bar3.scale.learn import from_json
    init = from_json(json.load(open(state))["state"]) if state else None
    TG.use(tname)
    tap = Tap(Capture(name, runner, writer))
    tap.target = tname
    t0 = time.time()
    try:
        LN.run(name, cfg, tap.mapper, lambda r: None, log=log, init=init)
        log(f"[{name}] round 0 finished without a look ({time.time() - t0:.0f}s)")
    except Stop as e:
        log(f"[{name}] stopped at {e} ({time.time() - t0:.0f}s)")


def capture_many(out, specs, workers, tiny=False, log=print):
    """specs [(run name, config, target, state file or None)], a thread each, one pool."""
    from bar3.scale.heldout import targets  # noqa: F401
    runner = Runner(workers)
    writer = Writer(out)

    def guard(*a):
        try:
            capture_run(*a, tiny=tiny, log=log)
        except Exception:
            traceback.print_exc()

    threads = [threading.Thread(target=guard, args=(n, c, t, runner, writer, s)) for n, c, t, s in specs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    writer.close()
    runner.pool.shutdown(wait=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out, config, targets = args[0], args[1], args[2:]
    specs = []
    for t in targets:
        tname, _, state = t.partition("@")
        specs.append((f"{config}_{tname}", config, tname, state or None))
    capture_many(out, specs, workers=2, tiny="--tiny" in sys.argv)


if __name__ == "__main__":
    main()
