"""tdsl (torch) against library/search.py (numpy): the same task, settings and probes -> the same found cost and the
same programs. Local, seconds.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m integration.lib.test_tdsl
"""
import time

import numpy as np
import torch

from library import dsl, search
from integration.lib import tdsl


def to_torch_problem(prob, settings_t):
    roots = [tdsl.Batch(*(torch.as_tensor(getattr(b, f)) for f in ("regs", "c", "h", "k", "p"))) for b in prob.roots]
    targets = []
    for t in prob.targets:
        targets.append({k: torch.as_tensor(v) for k, v in t.items()})
    return tdsl.Problem(prob.task, settings_t, roots, targets)


def main():
    for task, budget in (("sub", 40_000), ("lt", 40_000), ("csub", 3_000_000), ("shl", 1000)):
        sets = [dsl.Setting(2, 3), dsl.Setting(3, 2)]
        sets_t = [tdsl.Setting(2, 3, tdsl.Cells(2)), tdsl.Setting(3, 2, tdsl.Cells(3))]
        prob = search.Problem(task, sets, NP=8, seed=0)
        cls = search.enumerate_items(list(dsl.PRIMS), {}, sets)
        t0 = time.time()
        r1 = search.bfs(prob, cls, {}, max_cost=6, max_states=budget, time_limit=120)
        t1 = time.time()
        cls_t = tdsl.enumerate_items(list(dsl.PRIMS), {}, sets_t)
        t2 = time.time()
        r2 = tdsl.bfs(to_torch_problem(prob, sets_t), cls_t, {}, max_cost=6, max_states=budget, time_limit=120)
        t3 = time.time()
        p1 = sorted(dsl.show(p) for p in search.programs(r1, cls)) if r1.found_cost else []
        p2 = sorted(dsl.show(p) for p in tdsl.programs(r2, cls_t)) if r2.found_cost else []
        print(f"{task}: numpy cost {r1.found_cost} levels {r1.level_sizes} {t1 - t0:.1f}s | torch cost {r2.found_cost} "
              f"levels {r2.level_sizes} {t3 - t2:.1f}s (+items {t2 - t1:.1f}s) classes {len(cls)}/{len(cls_t)} "
              f"same programs: {p1 == p2} ({len(p1)}, {len(p2)}) e.g. {p2[:2]}", flush=True)


if __name__ == "__main__":
    main()
