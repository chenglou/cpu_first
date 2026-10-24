"""Is the compiled engine exact? The same search jobs through bar3/jobs.py (the original) and bar3/scale/jobs.py, and
every record field compared but the seconds. Jobs: the h1 target probe, a spread of round-0 candidates (every
source) at the probe-A budget, a few unsolved ones at a bigger budget, lookaheads with two routines, a target probe
over the 34-routine library carried after 2x + 1 (BREADTH.md §4), and one at N <= 26. Then the shared search
(engine.bfs_multi): the h1 round-0 pool's probes grouped by search space, each record against its own search's.

    cd cpu_first && <python with numba> -m bar3.scale.check            # about seven minutes, under 1 GB
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from library.inventing import brm
from bar3 import learner, problems as P
from bar3.breadth import targets as TG
from bar3.jobs import search_job as original
from bar3.scale.jobs import search_job as compiled

ROOT = Path(__file__).resolve().parents[2]
TIMING = ("seconds", "item_seconds", "wall")


def pool_for(tname, k=7):
    TG.use(tname)
    rec = json.loads(open(ROOT / f"bar3/breadth/records/lp_clean_{tname}_b1.jsonl").readline())
    near = [(f, d, p) for f, d, p in rec["target_probe"]["near"] if p]
    tgt = P.the_target(k)
    pool = P.laws(k) + P.restrictions(P.target_gen, k) + P.projections(P.target_gen, k)
    pool += P.from_near_misses(near, {}, k)
    pool += P.dreams({}, list(brm.LOOP_KINDS), k, np.random.default_rng(0), {tuple(tgt.want)}, n_want=16)
    return tgt, [c for c in P.dedupe(pool) if c.key() != tgt.key()], rec


def jobs():
    kinds = list(brm.LOOP_KINDS)
    out = []
    tgt, pool, rec = pool_for("h1")
    out.append(("h1 target probe", learner.job(tgt, {}, kinds, learner.LOOK, "target_probe", 7, near=12)))
    for c in pool[::6]:
        out.append((f"probe_a {c.name}", learner.job(c, {}, kinds, learner.PROBE_A, "probe_a", 7)))
    big = dict(learner.PROBE_B, max_states=150_000)
    for name in ("law/mod(X,N)/x+QN", "restrict/target/X<N·N"):
        c = next((c for c in pool if c.name == name), None)
        if c:
            out.append((f"probe_b(150k) {c.name}", learner.job(c, {}, kinds, big, "probe_b", 7)))
    st = rec["status"]
    link, sq = st["law/mod(X,N)/x+qN"]["best"], st["law/mul(X,X)/x"]["best"]
    for lb in ({"T.look0": link}, {"T.look0": sq, "T.look1": link}):
        out.append((f"lookahead {list(lb)}", learner.job(tgt, lb, kinds, learner.LOOK, "look", 7)))
    state = json.loads(open(ROOT / "bar3/breadth/records/seq_lp_clean_targets_b1.state.affine.json").read())["state"]
    TG.use("cube")
    ct = P.the_target(7)
    out.append(("cube target probe, 34-routine library",
                learner.job(ct, state["lib"], state["kinds"], learner.LOOK, "target_probe", 7, near=12)))
    TG.use("h1")
    t26 = P.the_target(26, 30)
    out.append(("h1 target probe at N <= 26", learner.job(t26, {}, kinds, learner.LOOK, "target_probe", 26, 30,
                                                          near=12)))
    return out


def main():
    compiled(jobs()[0][1])             # compile the kernels before timing
    bad, t_py, t_nb = 0, 0.0, 0.0
    for label, j in jobs():
        t = time.time()
        a = original(j)
        t_py += time.time() - t
        t = time.time()
        b = compiled(j)
        t_nb += time.time() - t
        diff = [f for f in sorted(set(a) | set(b)) if f not in TIMING and a.get(f) != b.get(f)]
        bad += bool(diff)
        print(f"{'DIFF ' + str(diff) if diff else 'same':<20} py {a['wall']:6.2f}s  nb {b['wall']:6.2f}s  "
              f"found {a['found_cost']}  levels {a['level_sizes']}  {label}")
    print(f"{bad} differing records; original {t_py:.1f}s, compiled {t_nb:.1f}s ({t_py / max(t_nb, 1e-9):.1f}x)")
    bad += check_multi()
    sys.exit(1 if bad else 0)


def check_multi():
    """bfs_multi against bfs: the h1 round-0 pool's probes grouped by search space, one search a group; every record
    field but the seconds must equal the single search's."""
    import collections
    from bar3.scale.jobs import same_search, search_jobs
    tgt, pool, rec = pool_for("h1")
    kinds = list(brm.LOOP_KINDS)
    js = [learner.job(c, {}, kinds, learner.PROBE_A, "probe_a", 7) for c in pool]
    groups = collections.defaultdict(list)
    for j in js:
        groups[same_search(j)].append(j)
    bad = 0
    t_one = t_many = 0.0
    for g in groups.values():
        if len(g) < 2:
            continue
        t = time.time()
        one = [compiled(j) for j in g]
        t_one += time.time() - t
        t = time.time()
        many = search_jobs(g)
        t_many += time.time() - t
        for a, b in zip(one, many):
            diff = [f for f in sorted(set(a) | set(b)) if f not in TIMING + ("shared",) and a.get(f) != b.get(f)]
            bad += bool(diff)
            if diff:
                print("DIFF", a["name"], diff)
    print(f"multi: {sum(len(g) for g in groups.values() if len(g) > 1)} jobs in "
          f"{sum(1 for g in groups.values() if len(g) > 1)} shared searches, {bad} differing; "
          f"alone {t_one:.1f}s, shared {t_many:.1f}s")
    return bad


if __name__ == "__main__":
    main()
