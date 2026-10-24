"""Search jobs on a process pool (SCALE.md): `Searcher` maps [job] -> [record or exception] like the breadth study's
Modal fan-out, but on local worker processes, with identical jobs run once (in flight or done). A cached record
still counts its seconds in every run that uses it, as in the breadth study. With share=True, the jobs of one call
that pose problems on the same search space (jobs.same_search: the same instances and vocabulary) run as one search
(jobs.search_jobs), each with the record it would get alone and an even share of the seconds.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import threading


def work(job):
    """One search in a worker (a target job sets its target first, for the N <= 40 check)."""
    from bar3.breadth import targets as TG
    from bar3.scale.jobs import search_job
    if job.get("breadth_target"):
        TG.use(job["breadth_target"])
    return search_job(job)


def work_many(jobs):
    from bar3.scale.jobs import search_jobs
    return search_jobs(jobs)


def warm():
    """Compile the kernels in each worker before it times a job."""
    from library.inventing import brm
    from bar3 import learner, problems as P
    from bar3.scale.jobs import search_job
    search_job(learner.job(P.the_target(5), {}, list(brm.LOOP_KINDS), dict(learner.PROBE_A, max_states=500),
                           "probe_a", 5))


class Done:
    """A finished search read back from the store: a future that is already resolved."""

    def __init__(self, rec):
        self.rec = rec

    def result(self):
        return self.rec


class Searcher:
    """With `store` (a path; on Modal, on the Volume, with `commit` to make it durable), every finished search is
    appended there as (key, record), and a new Searcher on the same path starts with them: a container restarted
    after preemption replays its runs from the store instead of searching again (the records are the originals,
    seconds included, so the replay's accounting is the first run's)."""

    def __init__(self, workers, engine="nb", share=False, store=None, commit=None):
        self.pool = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=warm)
        self.engine = engine            # "nb": the compiled engine; "py": bar3/jobs.py's original search
        self.share = share and engine == "nb"
        self.lock = threading.Lock()
        self.cache = {}                 # job key -> (future, index in its shared search or None)
        self.store, self.commit = store, commit
        if store:
            import os
            if os.path.exists(store):
                for line in open(store):
                    try:
                        key, rec = json.loads(line)
                    except ValueError:  # a line cut off by the preemption
                        continue
                    self.cache[key] = (Done(rec), None)

    def __call__(self, jobs):
        from bar3.scale.jobs import same_search
        keyed = []
        for j in jobs:
            j = dict(j, engine=self.engine) if self.engine != "nb" else j
            keyed.append((hashlib.sha1(json.dumps(j, sort_keys=True).encode()).hexdigest(), j))
        slots = []
        with self.lock:
            groups = {}
            for key, j in keyed:
                if key in self.cache or key in groups:
                    continue
                g = same_search(j) if self.share and not j.get("near") else key
                groups.setdefault(g, []).append((key, j))
            for members in groups.values():
                if len(members) == 1:
                    self.cache[members[0][0]] = (self.pool.submit(work, members[0][1]), None)
                else:
                    f = self.pool.submit(work_many, [j for _, j in members])
                    for i, (key, _) in enumerate(members):
                        self.cache[key] = (f, i)
            for key, j in keyed:
                slots.append(self.cache[key])
        out, seen, fresh = [], set(), []
        for (key, _), (f, i) in zip(keyed, slots):
            hit = key in seen or isinstance(f, Done)
            seen.add(key)
            try:
                r = f.result()
                r = r if i is None else r[i]
                out.append(dict(r, cached=True) if hit else r)
                if not hit:
                    fresh.append((key, r))
            except Exception as e:  # noqa: BLE001  (the learner logs a failed job and goes on)
                out.append(e)
        if self.store and fresh:
            with self.lock:
                with open(self.store, "a") as fh:
                    for key, r in fresh:
                        fh.write(json.dumps([key, r]) + "\n")
                        self.cache[key] = (Done(r), None)
                if self.commit:
                    self.commit()
        return out
