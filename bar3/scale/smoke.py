"""End to end on the plumbing: tiny configs (k = 5, tiny budgets) on two targets, fresh and carried, through the
process pool, twice; the round records must agree in every field but the seconds.
- `--engine`: the original learner (bar3/breadth/drive.py) with the original search against the compiled one;
- `--share`: this folder's learner with and without shared searches (pool.py);
- default: the original learner against this folder's (learn.py), both on the compiled search.

    cd cpu_first && <python with numba> -m bar3.scale.smoke [--engine | --share]
"""

from __future__ import annotations

import json
import sys

from bar3 import learner
from bar3.breadth import drive as old_drive
from bar3.scale import drive as new_drive
from bar3.scale.pool import Searcher

TIMING = {"seconds", "item_seconds", "wall", "totals", "cached", "shared"}


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, list):
        out = [strip(v) for v in x]
        return out
    return x


TINY = {"tiny": learner.CONFIGS["tiny"], "tiny_lp": dict(learner.CONFIGS["tiny"], rule="lp"),
        "tiny_a2": dict(learner.CONFIGS["tiny"], pair_pool=None, anchor_cost=2)}


def runs(engine, drive, share=False):
    recs = []
    search = Searcher(4, engine, share)
    for name, cfg in TINY.items():
        for t in ("rho", "affine"):
            drive.fresh(f"{name}_{t}", cfg, t, search, recs.append, log=lambda *a: None)
        if "anchor_cost" not in cfg:             # the original sequence (carry.py) has no anchored pairs
            drive.carried(f"{name}_seq", cfg, ["rho", "affine"], search, recs.append, log=lambda *a: None)
    search.pool.shutdown()
    return recs


def main():
    if "--engine" in sys.argv:
        a, b = runs("py", old_drive), runs("nb", old_drive)
    elif "--share" in sys.argv:
        a, b = runs("nb", new_drive), runs("nb", new_drive, share=True)
    else:
        a, b = runs("nb", old_drive), runs("nb", new_drive)
    looks = lambda r: [l[:5] + l[6:] for l in r.get("looks", [])]   # noqa: E731  (a look's seconds are field 5)
    bad = [i for i, (x, y) in enumerate(zip(a, b))
           if strip({**x, "looks": looks(x)}) != strip({**y, "looks": looks(y)})]
    tot = lambda rs: sum(v["seconds"] for r in rs if "round" in r for v in r["totals"].values())   # noqa: E731
    print(len(a), len(b), "records;", "differ at", bad, f"; search seconds py {tot(a):.1f}, nb {tot(b):.1f}")
    sys.exit(1 if bad or len(a) != len(b) else 0)


if __name__ == "__main__":
    main()
