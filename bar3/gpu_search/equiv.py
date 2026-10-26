"""The stored searches again, on the GPU (GPU_SEARCH.md §3): captured jobs (capture.py) through bar3/scale/jobs.py on
the GPU engine, each record compared with the stored one, every field but the seconds: the programs and their order,
the level sizes, the best fits, the stop, the aliases, the near misses.

A stored search that stopped on its time limit is not expected to match (the CPU's clock cut it; the GPU gets further
in the same time): it is run and reported apart. Jobs that share a search space can run as one search
(jobs.search_jobs, --share), as the held-out and newton runs did; each record is then compared with its own.

    cd cpu_first && <python with numba, torch, a GPU> -m bar3.gpu_search.equiv OUT.jsonl JOBS.jsonl.gz ...
        [--kinds=probe_b,target] [--per=N] [--seed=S] [--share] [--skip=KEYS.txt]

Each job's line in OUT: key, kind, name, the stored and the GPU seconds, level sizes, stop, and the differing fields.
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import random
import sys
import time

TIMING = ("seconds", "item_seconds", "wall", "shared", "cached")


def diff(a, b):
    """Differing fields but the times; both as stored (JSON has lists where a fresh record has tuples)."""
    a, b = json.loads(json.dumps(a)), json.loads(json.dumps(b))
    return [f for f in sorted(set(a) | set(b)) if f not in TIMING and a.get(f) != b.get(f)]


def load(paths, kinds=None):
    """Captured jobs, each key once (the first file that has it)."""
    out, seen = [], set()
    for p in paths:
        with gzip.open(p, "rt") as fh:
            for line in fh:
                key, src, job, rec = json.loads(line)
                if key in seen or (kinds and job["kind"] not in kinds):
                    continue
                seen.add(key)
                out.append((key, src, job, rec, os.path.basename(p)))
    return out


def select(jobs, per, seed):
    """Up to `per` jobs of each kind, drawn at random (seeded); all of them when per is 0."""
    if not per:
        return jobs
    by = collections.defaultdict(list)
    for j in jobs:
        by[j[2]["kind"]].append(j)
    rng = random.Random(seed)
    out = []
    for k in sorted(by):
        out += by[k] if len(by[k]) <= per else rng.sample(by[k], per)
    return out


def vocabulary(job):
    """What fixes a job's items (jobs.classes_for's cache key, in effect): jobs with the same one share them."""
    return json.dumps([job.get("lib", {}), job.get("kinds"), job["k"], job.get("max_body", 2), job.get("costs")],
                      sort_keys=True)


def part(jobs, i, n):
    """Part i of n: whole vocabularies (each enumerated once per container), every n-th one in key order."""
    by = collections.defaultdict(list)
    for j in jobs:
        by[vocabulary(j[2])].append(j)
    return [j for q, v in enumerate(sorted(by)) if q % n == i for j in by[v]]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    out_path, paths = args[0], args[1:]
    kinds = opt["kinds"].split(",") if opt.get("kinds") else None
    jobs = select(load(paths, kinds), int(opt.get("per", 0)), int(opt.get("seed", 0)))
    if opt.get("skip"):
        skip = set(open(opt["skip"]).read().split())
        jobs = [j for j in jobs if j[0] not in skip]
    if opt.get("part"):
        i, n = (int(x) for x in opt["part"].split("/"))
        jobs = part(jobs, i, n)
    else:
        jobs = sorted(jobs, key=lambda j: vocabulary(j[2]))
    from bar3.breadth import targets as TG
    from bar3.scale.heldout import targets as _ho  # noqa: F401  (the held-out targets, for their N <= 40 checks)
    from bar3.scale import jobs as J
    from bar3.gpu_search import engine as G
    G.use(True)
    print(f"{len(jobs)} jobs on {G.DEVICE} ({opt})", flush=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fh = open(out_path, "a")
    tally = collections.Counter()
    t_cpu = collections.Counter()
    t_gpu = collections.Counter()

    def report(key, job, stored, rec, src, wall, shared=1):
        d = diff(stored, rec)
        timed = stored.get("stop", "").startswith("time")
        status = "cpu_timed" if timed else ("same" if not d else "DIFF")
        tally[(job["kind"], status)] += 1
        t_cpu[job["kind"]] += stored.get("wall", stored["seconds"])
        t_gpu[job["kind"]] += wall
        fh.write(json.dumps(dict(key=key, file=src, kind=job["kind"], name=job["name"], status=status, diff=d,
                                 cpu_s=stored.get("wall", stored["seconds"]), gpu_s=round(wall, 4), shared=shared,
                                 levels=stored["level_sizes"], gpu_levels=rec["level_sizes"], stop=stored["stop"],
                                 gpu_stop=rec["stop"])) + "\n")
        fh.flush()
        if status != "same":
            print(f"  {status} {job['kind']} {job['name']}: {d} stored {stored['level_sizes']} {stored['stop']!r} "
                  f"gpu {rec['level_sizes']} {rec['stop']!r}", flush=True)

    def run_one(job):
        if job.get("breadth_target"):
            TG.use(job["breadth_target"])
        return J.search_job(job)

    t0 = time.time()
    done = 0
    if opt.get("workers"):              # searcher.py: vocabularies prepared on CPU processes, batches of 200
        from bar3.gpu_search.searcher import Searcher
        s = Searcher(int(opt["workers"]), share=bool(opt.get("share")))
        for b in range(0, len(jobs), 200):
            part_ = jobs[b:b + 200]
            t = time.time()
            recs = s([j[2] for j in part_])
            wall = (time.time() - t) / len(part_)
            for (key, _, job, stored, src), rec in zip(part_, recs):
                if isinstance(rec, dict):
                    report(key, job, stored, rec, src, wall)
                else:
                    tally[(job["kind"], "failed")] += 1
                    print(f"  failed {job['kind']} {job['name']}: {rec!r}", flush=True)
            done += len(part_)
            print(f"  {done}/{len(jobs)} {time.time() - t0:.0f}s {dict(tally)}", flush=True)
        s.close()
    elif opt.get("share"):
        groups = collections.defaultdict(list)
        for j in jobs:      # a job with a general check runs alone, with its own target registered
            alone = j[2].get("near") or j[2].get("general")
            groups[J.same_search(j[2]) if not alone else j[0]].append(j)
        for g in groups.values():
            t = time.time()
            if g[0][2].get("breadth_target"):
                TG.use(g[0][2]["breadth_target"])
            recs = J.search_jobs([j[2] for j in g])
            wall = (time.time() - t) / len(g)
            for (key, _, job, stored, src), rec in zip(g, recs):
                report(key, job, stored, rec, src, wall, len(g))
            done += len(g)
            if done % 200 < len(g):
                print(f"  {done}/{len(jobs)} {time.time() - t0:.0f}s {dict(tally)}", flush=True)
    else:
        for key, _, job, stored, src in jobs:
            t = time.time()
            rec = run_one(job)
            report(key, job, stored, rec, src, time.time() - t)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} {time.time() - t0:.0f}s {dict(tally)}", flush=True)
    fh.close()
    print(f"done in {time.time() - t0:.0f}s")
    for k in sorted(t_cpu):
        st = {s: tally[(k, s)] for s in ("same", "DIFF", "cpu_timed") if tally[(k, s)]}
        print(f"  {k:14s} {st}  stored cpu {t_cpu[k]:9.1f}s  gpu {t_gpu[k]:8.1f}s  ({t_cpu[k] / max(t_gpu[k], 1e-9):.1f}x)")
    bad = sum(v for (k, s), v in tally.items() if s == "DIFF")
    print("PASS" if not bad else f"FAIL: {bad} differing", flush=True)


if __name__ == "__main__":
    main()
