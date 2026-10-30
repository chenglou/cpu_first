"""Whole learner runs with the gate (GATE.md §6; PREREG.md measure 4, PREREG_ADDENDUM1.md): the consolidation arm
(consolidate/learn_newton, `alloc_deep_ct2a2_probe` or `alloc_ns_ct2a2_probe`) unchanged, its searches on the study's
own plain searcher (heldout/searcher.Searcher: shared searches, as the consolidation runs ran), and one change in the
mapper: wherever a 1M probe, the 10M deep probe or the round's target search comes back with nothing, the gate searches
the same job again, on the gated engine with a hard cap of the states the plain search spent (at most `cap`), and a
program it finds (verified, as any) replaces the empty record. Nothing else changes, so a run differs from its plain
twin only where the gate found what the plain search did not.

The gate is cross-fitted: a run on a formula of fold A (SCALE.md's ten) uses fold B's gates, and the other way round;
C0's gate on an empty library, C1's over one. Each run writes its rounds (<name>.jsonl, the learner's records) and every
gated search (<name>.gate.jsonl: kind, problem, the plain search's states, what the gate found).

Every finished search, plain and gated, is also appended to the container's stores (store_plain.jsonl,
store_gate.jsonl), as the consolidation runs did: Modal preempts CPU containers, and a restarted container replays its
runs from the stores instead of searching again.

With no gates (`gates` empty), a run is the plain twin: the same arm on the same searcher, every worker on the plain
search (for the formulas whose twins are missing or cut short in consolidate/records).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import threading
import time
import traceback

from bar3.scale.gate.capture import _warm, key_of
from bar3.scale.gate.train import FOLDS
from bar3.scale.pool import Done

GATED = ("probe_b", "probe_deep", "target")


def _gwork(job):
    from bar3.breadth import targets as TG
    from bar3.scale.gate import gjobs
    if job.get("breadth_target"):
        TG.use(job["breadth_target"])
    return gjobs.search_job(job)


class GatedSearcher:
    """One run's mapper. `cache` (shared by the runs of a container): identical gated jobs run once, as the plain
    searcher runs identical plain jobs once (the law book's probes are the same in every run's round 0); `store`: the
    container's file of finished gated searches, read into `cache` at the start (load_store)."""

    def __init__(self, plain, pool, gates, cap, stats, cache, lock, store=None, commit=None):
        self.plain, self.pool, self.gates, self.cap, self.stats = plain, pool, gates, cap, stats
        self.cache, self.lock, self.store, self.commit = cache, lock, store, commit

    def __call__(self, jobs):
        recs = self.plain(jobs)
        redo = []
        for i, (j, r) in enumerate(zip(jobs, recs)):
            if j["kind"] in GATED and isinstance(r, dict) and r.get("found_cost") is None:
                spent = int(sum(r["level_sizes"]))
                gj = dict(j, gate=self.gates["lib" if j.get("lib") else "empty"], budget="hard",
                          max_states=int(min(self.cap, max(spent, j.get("max_states", 0)))),
                          time_limit=max(j.get("time_limit", 600), 1800))
                redo.append((i, gj, spent))
        futs = []
        with self.lock:
            for i, gj, spent in redo:
                key = key_of(gj)
                if key not in self.cache:
                    self.cache[key] = self.pool.submit(_gwork, gj)
                futs.append((i, gj, spent, key, self.cache[key]))
        for i, gj, spent, key, f in futs:
            try:
                g = f.result()
            except Exception as e:  # noqa: BLE001  (the plain record stands)
                self.stats(dict(kind=gj["kind"], name=gj["name"], error=repr(e)[:300]))
                continue
            if self.store:
                with self.lock:
                    if not isinstance(self.cache[key], Done):
                        with open(self.store, "a") as fh:
                            fh.write(json.dumps([key, g]) + "\n")
                        self.cache[key] = Done(g)
                        if self.commit:
                            self.commit()
            found = g["found_cost"] is not None and bool(g.get("programs"))
            self.stats(dict(kind=gj["kind"], name=gj["name"], lib=len(gj.get("lib", {})), plain_states=spent,
                            cap=gj["max_states"], found=g["found_cost"] if found else None,
                            first_accept=g.get("first_accept"), generated=g.get("generated"),
                            seconds=g.get("seconds"), programs=[p["program"] for p in g.get("programs") or []][:3]))
            if found:
                recs[i] = dict(g, gated=True, plain_states=spent)
        return recs


def load_store(path):
    """key -> Done(record) for every gated search a store holds (a line cut off by a preemption is skipped)."""
    cache = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                key, rec = json.loads(line)
            except ValueError:
                continue
            cache[key] = Done(rec)
    return cache


def run_many(runs, gates, out, workers, commit=None, cap=4_000_000, log=print):
    """runs [[name, config, target]], a thread each; gates {"A": {"empty": path, "lib": path}, "B": {...}}: the gates
    trained on each fold ({}: plain twins)."""
    from bar3.scale.heldout import targets  # noqa: F401  (the held-out targets, in this process too)
    from bar3.scale.heldout.searcher import Searcher
    from bar3.scale.consolidate import learn_newton as LN
    from bar3.scale.consolidate.configs import CONFIGS
    plain = Searcher(max(2, workers // 2) if gates else workers, "nb", True, store=f"{out}/store_plain.jsonl",
                     commit=commit)
    pool = cf.ProcessPoolExecutor(max(2, workers - workers // 2), mp_context=mp.get_context("spawn"),
                                  initializer=_warm) if gates else None
    lock = threading.Lock()
    gstore = f"{out}/store_gate.jsonl"
    cache, clock = load_store(gstore), threading.Lock()
    launch = f"{time.time():.0f}"
    t0 = time.time()

    def writer(path):
        open(path, "w").close()

        def w(rec):
            with lock:
                with open(path, "a") as f:
                    f.write(json.dumps(dict(rec, launch=launch, elapsed=round(time.time() - t0, 1))) + "\n")
                if commit:
                    commit()
        return w

    def one(name, config, tname):
        try:
            gs = plain                                          # a plain twin
            if gates:
                fold = "B" if tname in FOLDS["A"] else "A"      # the other fold's gates
                gs = GatedSearcher(plain, pool, gates[fold], cap, writer(f"{out}/{name}.gate.jsonl"), cache, clock,
                                   store=gstore, commit=commit)
            cfg = json.loads(json.dumps(CONFIGS[config]))
            LN.fresh(name, cfg, tname, gs, writer(f"{out}/{name}.jsonl"), log=log)
        except Exception:
            traceback.print_exc()

    threads = [threading.Thread(target=one, args=tuple(r)) for r in runs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    plain.pool.shutdown(wait=True)
    if pool:
        pool.shutdown(wait=True)
