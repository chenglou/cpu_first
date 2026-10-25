"""Checks for the subgoal learner (SUBGOAL.md §3), local, tiny budgets (k = 5, a few seconds):
1. `same`: learn.py's copy without "subgoal" asks for exactly the jobs bar3/scale/learn.run asks for and writes the
   same records: learn.run runs first with every search kept by its job; the copy then runs on those searches only
   (a job learn.run never asked for is an error) and its records must equal the original's.
2. `smoke`: with "subgoal" on, two rounds on each named target: the subgoal, its deeper probe, the record's fields.

    cd cpu_first && <python with numba> -m bar3.scale.subgoal.check [same|smoke] [target ...]
"""

from __future__ import annotations

import hashlib
import json
import sys

from bar3 import learner as L
from bar3.breadth import targets as TG
from bar3.scale import learn as original
from bar3.scale.configs import CONFIGS as SCALE
from bar3.scale.jobs import search_job
from bar3.scale.subgoal import learn as sub

TINY = dict(SCALE["lp_clean_deep"], k=5, rounds=2, budgets=L.TINY)
TINY_CT = dict(SCALE["ct2a2_clean_deep"], k=5, rounds=2, budgets=L.TINY)


def key(j):
    return hashlib.sha1(json.dumps(j, sort_keys=True).encode()).hexdigest()


def strip(rec):
    return json.loads(json.dumps(rec))


def same(tname, cfg):
    kept = {}

    def record(jobs):
        out = []
        for j in jobs:
            r = kept.get(key(j))
            if r is None:
                r = kept[key(j)] = search_job(j)
            out.append(r)
        return out

    def replay(jobs):
        return [kept[key(j)] for j in jobs]          # KeyError: the copy asked for a job the original did not

    a, b = [], []
    TG.use(tname)
    original.run("check", cfg, record, lambda r: a.append(strip(r)), log=lambda *_: None)
    TG.use(tname)
    sub.run("check", cfg, replay, lambda r: b.append(strip(r)), log=lambda *_: None)
    for ra in a:
        ra.pop("subgoal", None)
    for rb in b:
        rb.pop("subgoal", None)
    assert a == b, f"{tname}: the copy's records differ"
    print(f"same {tname} {cfg['rule']}: {len(a)} rounds, {len(kept)} searches, records equal")


def smoke(tname, cfg):
    recs = []
    TG.use(tname)
    sub.run("smoke", dict(cfg, subgoal=dict(deep=dict(L.TINY["probe_b"], max_states=20_000))),
            lambda jobs: [search_job(j) for j in jobs], recs.append)
    for r in recs:
        s = r["subgoal"]
        top = list(s["scores"].items())[:3]
        print(f"smoke {tname} round {r['round']}: subgoal {s['name']}, {len(s['deep'])} deep probes, "
              f"top {[(n, v['key']) for n, v in top]}, picked {r['picked']}")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "same"
    targets = sys.argv[2:] or ["ten", "h1"]
    for t in targets:
        for cfg in (TINY, TINY_CT):
            (same if what == "same" else smoke)(t, cfg)


if __name__ == "__main__":
    main()
