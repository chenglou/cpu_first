"""A learner's searches on one GPU (GPU_SEARCH.md §5): the learner's mapper, [job] -> [record or exception], as
bar3/scale/pool.Searcher is on the CPU containers, but with every search on this process's GPU, one at a time, and
each batch's vocabularies (the item enumeration, which stays on the CPU) prepared ahead on a pool of CPU processes, so
the GPU does not wait for a lookahead's new routines to be enumerated. A vocabulary is jobs.classes_for's result,
computed by that same function in a worker and put in its cache here under its own key; the search then finds it
there. Jobs of one batch that share a search space run as one search (share=True, jobs.search_jobs), as on the CPU
containers; identical jobs run once.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp

CACHE = 64                   # vocabularies kept here


def _init():
    from bar3.scale.heldout import targets as _ho  # noqa: F401  (the held-out targets)
    from bar3.gpu_search import engine as G
    G.use(True)              # classes_for builds this engine's Codes (their GPU copy is made where they are used)


def vocab_args(job):
    """A job's vocabulary, as jobs._vocabulary reads it. The library keeps its order (pairs, not a sorted dict): the item
    enumeration depends on it (it picks each class's representative), though classes_for's cache key does not."""
    return json.dumps([list(job.get("lib", {}).items()), list(job.get("kinds", [])) if "kinds" in job else None,
                       job["k"], job.get("max_body", 2), job.get("costs")])


def prepare(args):
    """jobs.classes_for for one vocabulary, in a worker: (its cache key, its cache entry)."""
    from library import dsl
    from library.inventing import brm
    from bar3.scale import jobs as J
    lib, kinds, k, max_body, costs = json.loads(args)
    J._CLASSES.clear()
    J.classes_for({t: dsl.parse(v) for t, v in lib}, list(brm.LOOP_KINDS) if kinds is None else kinds, k, max_body,
                  costs)
    (key, entry), = J._CLASSES.items()
    return key, entry


class Searcher:
    def __init__(self, workers, share=True):
        self.pool = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init)
        self.share = share
        self.vocab = collections.OrderedDict()      # vocab_args -> future of (key, entry)
        self.stats = collections.Counter()

    def _future(self, job):
        a = vocab_args(job)
        if a not in self.vocab:
            self.vocab[a] = self.pool.submit(prepare, a)
            while len(self.vocab) > CACHE:
                self.vocab.popitem(last=False)
        else:
            self.vocab.move_to_end(a)
        return self.vocab[a]

    def __call__(self, jobs):
        import time
        from bar3.breadth import targets as TG
        from bar3.scale import jobs as J
        keyed = [(hashlib.sha1(json.dumps(j, sort_keys=True).encode()).hexdigest(), j) for j in jobs]
        groups = collections.OrderedDict()
        for key, j in keyed:
            if any(key in g for g in groups.values()):
                continue
            g = J.same_search(j) if self.share and not j.get("near") and not j.get("general") else key
            groups.setdefault(g, collections.OrderedDict())[key] = j
        futs = {g: self._future(next(iter(m.values()))) for g, m in groups.items()}
        done = {}
        for g, members in groups.items():
            js = list(members.values())
            t = time.time()
            vkey, entry = futs[g].result()
            self.stats["wait_vocab_s"] += time.time() - t
            J._CLASSES.clear()
            J._CLASSES[vkey] = entry
            if js[0].get("breadth_target"):
                TG.use(js[0]["breadth_target"])
            t = time.time()
            try:
                recs = J.search_jobs(js) if len(js) > 1 else [J.search_job(js[0])]
            except Exception as e:                  # noqa: BLE001  (the learner logs a failed job and goes on)
                recs = [e] * len(js)
            self.stats["search_s"] += time.time() - t
            self.stats["searches"] += 1
            for key, r in zip(members, recs):
                done[key] = json.loads(json.dumps(r)) if isinstance(r, dict) else r
        self.stats["jobs"] += len(jobs)
        return [done[key] for key, _ in keyed]

    def close(self):
        self.pool.shutdown(wait=True)
