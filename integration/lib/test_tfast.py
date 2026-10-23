"""tfast (batched items) against tdsl (one item at a time): the same classes and the same searches. Local, seconds.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m integration.lib.test_tfast
"""
import time

import torch

from library import dsl, search
from integration.lib import tdsl, tfast
from integration.lib.test_tdsl import to_torch_problem


def main():
    sets_t = [tdsl.Setting(2, 3, tdsl.Cells(2)), tdsl.Setting(3, 2, tdsl.Cells(3))]
    t0 = time.time()
    c1 = tdsl.enumerate_items(list(dsl.PRIMS), {}, sets_t)
    t1 = time.time()
    c2 = tfast.enumerate_items(list(dsl.PRIMS), {}, sets_t)
    t2 = time.time()
    same = [dsl.show((a.rep,)) for a in c1] == [dsl.show((b.rep,)) for b in c2]
    print(f"classes: tdsl {len(c1)} in {t1 - t0:.1f}s, tfast {len(c2)} in {t2 - t1:.1f}s, same reps in order: {same}")
    for task, budget in (("sub", 3_000_000), ("csub", 3_000_000), ("shl", 1000)):
        sets = [dsl.Setting(2, 3), dsl.Setting(3, 2)]
        prob = search.Problem(task, sets, NP=8, seed=0)
        tp = to_torch_problem(prob, sets_t)
        t0 = time.time()
        r1 = tdsl.bfs(tp, c1, {}, max_cost=6, max_states=budget, time_limit=600)
        t1 = time.time()
        r2 = tfast.bfs(to_torch_problem(prob, sets_t), c2, {}, max_cost=6, max_states=budget)
        t2 = time.time()
        p1 = sorted(dsl.show(p) for p in tdsl.programs(r1, c1)) if r1.found_cost else []
        p2 = sorted(dsl.show(p) for p in tdsl.programs(r2, c2)) if r2.found_cost else []
        print(f"{task}: tdsl cost {r1.found_cost} {r1.level_sizes} {t1 - t0:.1f}s | tfast cost {r2.found_cost} "
              f"{r2.level_sizes} {t2 - t1:.1f}s | same programs {p1 == p2} ({len(p1)}) {p2[:2]}", flush=True)


if __name__ == "__main__":
    main()
