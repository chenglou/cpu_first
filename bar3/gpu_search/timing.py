"""Seconds per level, CPU against GPU (GPU_SEARCH.md §4): the same jobs through bar3/scale/jobs.py on either engine,
with the clock read at each level's end. The CPU engine's clock comes from a copy of engine.bfs written from its own
source with two asserted insertions (the clock list, and a reading after each level's best fit), so the copy is the
search line for line; every record is checked equal to the stored one.

    cd cpu_first && <python with numba, torch> -m bar3.gpu_search.timing ENGINE OUT.jsonl JOBS.jsonl.gz ...
        [--kinds=...] [--per=N] [--seed=S] [--min_states=M] [--repeat=R] [--workers=W] [--prof]

ENGINE: "cpu" (the compiled engine; with --workers, W searches at once on a process pool, as the Modal CPU containers
ran them) or "gpu" (one search at a time). The same options select the same jobs for both. Each job's line: kind,
name, level sizes, the clock at each level's end (seconds from the search's start), the search's seconds (bfs), the
wall time (with the vocabulary's item enumeration and the record), and whether the record equals the stored one.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time
import types

from bar3.scale import engine as E

_STATE = {}


def timed_cpu_engine():
    """A module like bar3/scale/engine.py whose bfs reads the clock at each level's end (res.level_seconds)."""
    src = inspect.getsource(E.bfs)
    for old, new in (("    res = Result()\n", "    res = Result()\n    res.level_seconds = []\n"),
                     ("        res.best_fit.append(best)\n",
                      "        res.best_fit.append(best)\n        res.level_seconds.append(round(time.time() - t0, 4))\n")):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    ns = dict(vars(E))
    exec(compile(src, E.__file__, "exec"), ns)
    mod = types.ModuleType("bar3.gpu_search.cpu_timed")
    mod.__dict__.update(vars(E))
    mod.bfs = ns["bfs"]
    return mod


def setup(which):
    """Point bar3/scale/jobs.py at the engine, with a clock on its bfs (once per process)."""
    if _STATE.get("which") == which:
        return
    from bar3.scale.heldout import targets as _ho  # noqa: F401  (the held-out targets)
    from bar3.scale import jobs as J
    from bar3.gpu_search import engine as G
    G.use(which == "gpu")
    if which != "gpu":
        J.engine = timed_cpu_engine()
    inner = J.engine.bfs

    def bfs(*a, **kw):
        res = inner(*a, **kw)
        _STATE["clock"] = getattr(res, "level_seconds", None)
        return res

    J.engine.bfs = bfs
    _STATE["which"] = which


def time_job(which, key, job, stored, prof=False, rep=0):
    from bar3.breadth import targets as TG
    from bar3.scale import jobs as J
    from bar3.gpu_search import engine as G
    from bar3.gpu_search import equiv as Q
    setup(which)
    if job.get("breadth_target"):
        TG.use(job["breadth_target"])
    if prof and which == "gpu":
        G.PROF = G.Prof()
    t = time.time()
    rec = J.search_job(job)
    wall = time.time() - t
    phases = {k: round(v, 3) for k, v in G.PROF.t.items()} if G.PROF is not None else None
    G.PROF = None
    d = Q.diff(stored, rec)
    return dict(key=key, kind=job["kind"], name=job["name"], engine=which, rep=rep, levels=rec["level_sizes"],
                clock=_STATE.get("clock"), wall=round(wall, 4), search_s=rec["seconds"],
                stored_search_s=stored["seconds"], stored_wall=stored.get("wall", stored["seconds"]),
                same=not d, diff=d, stop=rec["stop"], stored_stop=stored["stop"], lib_size=rec["lib_size"],
                n_classes=rec["n_classes"], NP=rec["NP"], phases=phases)


def _work(a):
    return time_job(*a)


def main():
    from bar3.gpu_search import equiv as Q
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    which, out_path, paths = args[0], args[1], args[2:]
    kinds = opt["kinds"].split(",") if opt.get("kinds") else None
    jobs = Q.load(paths, kinds)
    jobs = [j for j in jobs if sum(j[3]["level_sizes"]) >= int(opt.get("min_states", 0))
            and not j[3]["stop"].startswith("time")]
    jobs = Q.select(jobs, int(opt.get("per", 0)), int(opt.get("seed", 0)))
    reps = int(opt.get("repeat", 1))
    tasks = [(which, key, job, stored, bool(opt.get("prof")), rep) for key, _, job, stored, _ in jobs
             for rep in range(reps)]
    workers = int(opt.get("workers", 1))
    print(f"{len(jobs)} jobs x {reps}, engine {which}, {workers} at once", flush=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if workers > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        import bar3.gpu_search.timing as T        # by name: this file may run as __main__, whose functions don't pickle
        pool = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
        results = pool.map(T._work, tasks)
    else:
        results = map(_work, tasks)
    t0 = time.time()
    with open(out_path, "a") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"  {which} {r['kind']:12s} search {r['search_s']:8.2f}s wall {r['wall']:8.2f}s (stored search "
                  f"{r['stored_search_s']:8.2f}s) {'same' if r['same'] else 'DIFF ' + str(r['diff'])} "
                  f"{r['levels']} {r['clock']}" + (f" phases {r['phases']}" if r["phases"] else ""), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
