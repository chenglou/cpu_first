"""Is the GPU search exact? bar3/scale/check.py's jobs (the h1 target probe, a spread of round-0 candidates of every
source, unsolved ones at 150k states, lookaheads with routines, a target probe over the 34-routine library, one at
N <= 26) and its shared searches (the h1 round-0 pool's probes grouped by search space), each through bar3/scale/jobs.py
on the compiled CPU engine and on the GPU engine (engine.use), every record field compared but the seconds.

    cd cpu_first && <python with numba and torch> -m bar3.gpu_search.check [--big]

Without a GPU the GPU engine runs its bookkeeping in torch on the CPU and the CPU kernel in place of Triton (a test of
everything but the kernels). --big adds probes at 1M states and a 3M-state target search.
"""

from __future__ import annotations

import collections
import sys
import time

from library.inventing import brm
from bar3 import learner
from bar3.scale import check as SC
from bar3.scale import jobs as J
from bar3.gpu_search import engine as G

TIMING = ("seconds", "item_seconds", "wall", "shared")


def big_jobs():
    tgt, pool, _ = SC.pool_for("h1")
    kinds = list(brm.LOOP_KINDS)
    out = []
    for name in ("law/mod(X,N)/x+QN", "law/gcd(X,N)/x", "restrict/target/X<N·N"):
        c = next((c for c in pool if c.name == name), None)
        if c:
            out.append((f"probe_b {c.name}", learner.job(c, {}, kinds, learner.PROBE_B, "probe_b", 7)))
    out.append(("h1 target 3M", learner.job(tgt, {}, kinds, learner.FULL, "target", 7, general=True)))
    return out


def diff(a, b):
    return [f for f in sorted(set(a) | set(b)) if f not in TIMING and a.get(f) != b.get(f)]


def run(js, on):
    G.use(on)
    out = []
    for label, j in js:
        t = time.time()
        r = J.search_job(j)
        out.append((r, time.time() - t))
    return out


def main():
    js = SC.jobs() + (big_jobs() if "--big" in sys.argv else [])
    run(js[:1], False)
    run(js[:1], True)                    # compile both before timing
    cpu = run(js, False)
    gpu = run(js, True)
    bad = 0
    for (label, _), (a, ta), (b, tb) in zip(js, cpu, gpu):
        d = diff(a, b)
        bad += bool(d)
        print(f"{'DIFF ' + str(d) if d else 'same':<20} cpu {ta:7.2f}s  gpu {tb:7.2f}s  found {a['found_cost']}  "
              f"levels {a['level_sizes']}  {label}", flush=True)
    print(f"{bad} differing records of {len(js)}; cpu {sum(t for _, t in cpu):.1f}s, gpu {sum(t for _, t in gpu):.1f}s "
          f"on {G.DEVICE}", flush=True)
    bad += multi()
    print("PASS" if not bad else f"FAIL: {bad}")
    sys.exit(1 if bad else 0)


def multi():
    """Shared searches: each group's records from GPU bfs_multi against the CPU's bfs_multi and the CPU's own bfs."""
    tgt, pool, _ = SC.pool_for("h1")
    kinds = list(brm.LOOP_KINDS)
    js = [learner.job(c, {}, kinds, learner.PROBE_A, "probe_a", 7) for c in pool]
    groups = collections.defaultdict(list)
    for j in js:
        groups[J.same_search(j)].append(j)
    groups = [g for g in groups.values() if len(g) > 1]
    G.use(False)
    alone = [[J.search_job(j) for j in g] for g in groups]
    shared_cpu = [J.search_jobs(g) for g in groups]
    G.use(True)
    t = time.time()
    shared_gpu = [J.search_jobs(g) for g in groups]
    t_gpu = time.time() - t
    bad = 0
    for g, a1, a2, b in zip(groups, alone, shared_cpu, shared_gpu):
        for x, y, z in zip(a1, a2, b):
            d = diff(x, z) + diff(y, z)
            bad += bool(d)
            if d:
                print("DIFF", x["name"], d)
    print(f"multi: {sum(len(g) for g in groups)} jobs in {len(groups)} shared searches, {bad} differing; "
          f"gpu shared {t_gpu:.1f}s", flush=True)
    return bad


if __name__ == "__main__":
    main()
