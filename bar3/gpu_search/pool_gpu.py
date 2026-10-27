"""bar3/scale/pool.Searcher on one GPU (GPU_SEARCH.md §8): the learner's mapper, [job] -> [record or exception], with
pool.Searcher's arguments and behaviour, but every search on this process's GPU, and on pools of CPU worker processes
only the vocabularies (the item enumeration, which stays on the CPU) and the checks of found programs.

As pool.Searcher: identical jobs run once (in flight or done); with share=True, the jobs of one call that pose
problems on the same search space (jobs.same_search) run as one search (jobs.search_jobs), each with the record it
would get alone; with `store` (a path; `commit` makes it durable on a Volume), every finished search is appended there
as (key, record), and a new Searcher on the same path starts with them, so a container restarted after preemption
replays its runs. A job that fails returns its exception, which is never stored.

Here: several learner threads may call at once (a study's Modal group runs a thread per run). Each call runs its own
searches in its own thread, one search at a time on the GPU (a lock), each search's vocabulary prepared ahead on the
worker pool by jobs.classes_for itself and put in jobs.py's cache under its own key just before the search (as
searcher.py does). A shared search whose jobs need a target registered for their N <= 40 check (`breadth_target`,
which pool.work sets for a job run alone) runs with it set, and is counted in stats["general_shared"].

A search's found programs are checked on a second pool of `workers` processes (checks_pool.py, and digits_accept.py
for the jobs on the digits target, whose check at big x is the costly part): the same rows, so the same records.

`digits=True` serves the digits study's learner copies: the D.B block, the digits targets and the digits jobs
(bar3/scale/digits/jobs.py) are installed in this process and in every worker, as bar3/scale/digits/searcher.py's
workers have them; otherwise the held-out targets are registered, as bar3/scale/heldout/searcher.py's workers do.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import threading
import time

from bar3.gpu_search.searcher import prepare, vocab_args
from bar3.scale.pool import Done

CACHE = 64                   # vocabularies kept (each is reused by every job of the same library and machine)


def setup(digits):
    """This process's study modules, then the GPU engine (engine.use also points the digits jobs at it)."""
    if digits:
        from bar3.scale.digits import jobs  # noqa: F401  (the D.B block, the digits targets and jobs)
    else:
        from bar3.scale.heldout import targets  # noqa: F401  (the held-out targets)
    from bar3.gpu_search import engine as G
    G.use(True)


class Searcher:
    def __init__(self, workers, engine="gpu", share=False, store=None, commit=None, digits=False):
        assert engine == "gpu", engine
        setup(digits)
        self.pool = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=setup,
                                           initargs=(digits,))
        from bar3.gpu_search import checks_pool as CP
        self.checks = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=CP.init,
                                             initargs=(digits,))
        if digits:
            from bar3.gpu_search import digits_accept as DA
            DA.install(self.checks)
        CP.install(self.checks)
        self.ahead = 2 * workers            # vocabularies prepared ahead of the GPU, per call
        self.share = share
        self.lock = threading.Lock()        # the caches
        self.gpu = threading.Lock()         # one search at a time
        self.cache = {}                     # job key -> (future, index in its shared search or None)
        self.vocab = collections.OrderedDict()      # vocab_args -> future of (classes_for key, entry)
        self.stats = collections.Counter()
        self.store, self.commit = store, commit
        if store and os.path.exists(store):
            for line in open(store):
                try:
                    key, rec = json.loads(line)
                except ValueError:          # a line cut off by the preemption
                    continue
                self.cache[key] = (Done(rec), None)

    def _vocab(self, job):
        a = vocab_args(job)
        with self.lock:
            if a not in self.vocab:
                self.vocab[a] = self.pool.submit(prepare, a)
                self.stats["vocabularies"] += 1
                while len(self.vocab) > CACHE:
                    self.vocab.popitem(last=False)
            else:
                self.vocab.move_to_end(a)
            return self.vocab[a]

    def _search(self, js, vf):
        from bar3.breadth import targets as TG
        from bar3.scale import jobs as J
        t = time.time()
        vkey, entry = vf.result()
        self.stats["wait_vocab_s"] += time.time() - t
        tagged = next((j["breadth_target"] for j in js if j.get("breadth_target")), None)
        with self.gpu:
            t = time.time()
            prev = TG._current.__dict__.get("target")         # this (learner) thread's target, put back after
            J._CLASSES.clear()
            J._CLASSES[vkey] = entry
            if tagged:
                TG.use(tagged)
            try:
                recs = J.search_jobs(js) if len(js) > 1 else [J.search_job(js[0])]
            except Exception as e:          # noqa: BLE001  (the learner logs a failed job and goes on)
                recs = [e] * len(js)
            finally:
                if tagged and prev is None:
                    del TG._current.target
                elif tagged:
                    TG._current.target = prev
            self.stats["vocab_misses"] += list(J._CLASSES) != [vkey]     # a vocabulary enumerated here instead
            J._CLASSES.clear()
            self.stats["search_s"] += time.time() - t
            self.stats["searches"] += 1
            self.stats["general_shared"] += len(js) > 1 and tagged is not None
        return recs

    def __call__(self, jobs):
        from bar3.scale import jobs as J
        keyed = [(hashlib.sha1(json.dumps(j, sort_keys=True).encode()).hexdigest(), j) for j in jobs]
        own = []
        with self.lock:
            groups, mine = collections.OrderedDict(), set()
            for key, j in keyed:
                if key in self.cache or key in mine:
                    continue
                mine.add(key)
                g = J.same_search(j) if self.share and not j.get("near") else key
                groups.setdefault(g, []).append((key, j))
            for members in groups.values():
                f = cf.Future()
                for i, (key, _) in enumerate(members):
                    self.cache[key] = (f, None if len(members) == 1 else i)
                own.append((members, f))
            slots = [self.cache[key] for key, _ in keyed]
        vfs = [None] * len(own)
        for q, (members, f) in enumerate(own):
            for a in range(q, min(len(own), q + self.ahead)):       # vocabularies ahead of this search
                if vfs[a] is None:
                    vfs[a] = self._vocab(own[a][0][0][1])
            recs = self._search([j for _, j in members], vfs[q])
            f.set_result(recs[0] if len(members) == 1 else recs)
            vfs[q] = None
        self.stats["jobs"] += len(jobs)
        out, seen, fresh = [], set(), []
        for (key, _), (f, i) in zip(keyed, slots):
            hit = key in seen or isinstance(f, Done)
            seen.add(key)
            r = f.result()
            r = r if i is None else r[i]
            if isinstance(r, Exception):
                out.append(r)
                continue
            out.append(dict(r, cached=True) if hit else r)
            if not hit:
                fresh.append((key, r))
        if self.store and fresh:
            with self.lock:
                with open(self.store, "a") as fh:
                    for key, r in fresh:
                        fh.write(json.dumps([key, r]) + "\n")
                        self.cache[key] = (Done(r), None)
            if self.commit:
                self.commit()
        return out

    def close(self):
        self.pool.shutdown(wait=True)
        self.checks.shutdown(wait=True)
