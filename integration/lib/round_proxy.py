"""The acceptance tasks' deciding step, alone (INTEGRATION.md §3): on H1's prompts everything stage 3 does before the
round (the reader, the cells, the practice problems, the climb) is the same for every task, since only the answers
differ. So the round's search with a given library, on a task's T = 1 answers, says which formulas a stage-3 run can
learn. This runs libcore's round search (routines + straight-line primitives at full size, goal = 8 answer rows,
accepted on all of them) for each task's data with a library (JSON {name: program text}).

    python -m integration.lib.round_proxy LIB.json DATA_ROOT TASK [TASK ...]      (DATA_ROOT holds litmus_<task>/)
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

from integration.lib import libcore as L


def rows_T1(path, n=400, seed=3):
    rows = [json.loads(l) for l in open(os.path.join(path, "train.jsonl"))]
    rows = [r for r in rows if r["time_steps"] == 1]
    g = torch.Generator().manual_seed(seed)
    rows = [rows[i] for i in torch.randperm(len(rows), generator=g)[:n].tolist()]

    def dig(s):
        s = str(s)
        return [int(ch) for ch in reversed(s)][:8] + [0] * (8 - min(8, len(s)))
    return tuple(torch.tensor([dig(r[k]) for r in rows]) for k in ("modulus", "x", "result"))


def main(lib_path, root, tasks):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lib = {k: L.parse(v) for k, v in json.load(open(lib_path))["lib"].items()}
    lib = {k: v for k, v in lib.items() if k not in ("T.sq", "T.sqmod", "T.mulmod")}   # the round is what is searched
    routines = [k for k in lib if k.startswith("T.")]
    cells = L.Cells(10, dev)
    for task in tasks:
        path = os.path.join(root, "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2" if task == "h1"
                            else f"litmus_{task}")
        ans = tuple(t.to(dev) for t in rows_T1(path))
        t0 = time.time()
        prob = L.round_problem(cells, ans, 8, 16, dev)
        classes = L.enumerate_items(list(L.PRIMS) + routines, lib, prob.settings, dev, max_body=0)
        res = L.bfs(prob, classes, lib, max_cost=int(os.environ.get("ROUND_COST", 5)),
                    max_states=int(os.environ.get("ROUND_STATES", 4_000_000)),
                    accept=lambda ps: [L.round_fit(p, lib, cells, ans, 16, dev) == 1.0 for p in ps], chunk=1 << 19)
        print("ROUND " + json.dumps({"task": task, "found_cost": res.found_cost, "levels": res.level_sizes,
                                      "stop": res.stop, "seconds": round(time.time() - t0, 1),
                                      "programs": [L.show(p) for p in (res.accepted or [])][:6],
                                      "aliases": res.aliases[:1]}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
