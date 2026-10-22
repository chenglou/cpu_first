"""Learned digit cells (LIBRARY.md §7): the subtraction table learned from the order of the numbers, then library
learning on top of it.

    H1_DATA=<H1 folder> ~/github/sotaku/venv/bin/python -m library.learned train     # ~1 min per seed, < 0.5 GB

`train` runs the anchors' certified recipe (anchors/core.py, R2: x < N on every top part, then the column and row
order laws and both bijectivities; 400 steps of Adam at batch 64 on random H1 training rows, base 10) and saves the
snapped table every 25 steps to records/learned_tables.json, with how many of its 200 entries are right.

The wake-sleep configs `lc_*` (wake_sleep.CONFIGS) then search the subtraction half of the task set in base 10
(settings (10, 1) and (10, 2)) with only the subtraction cell: no ADC, MUL or LDH instructions, so every digit of
arithmetic goes through the learned table. They differ only in the table: exact, or a snapshot.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLES = HERE / "records" / "learned_tables.json"

R2 = {"pref": 1, "colall": 1, "rowall": 1, "bij": 1, "bijr": 1}

# the subtraction half: every task whose program needs only the subtraction cell
SUB_TASKS = ("copy", "shl", "shr", "sub", "subY", "lt", "ltY", "neg", "negN", "csub", "csubY", "modB", "modBY",
             "step0", "step", "reduce")
SUB_PRIMS = [p for p in __import__("library.dsl", fromlist=["PRIMS"]).PRIMS
             if not p.startswith(("ADC", "MUL", "LDH")) and p != "CLK"]


def train(seeds=(1, 2), steps=400, every=25):
    sys.path.insert(0, str(HERE.parent / "anchors"))
    import torch
    import core
    torch.set_num_threads(2)
    out = []
    for seed in seeds:
        t0 = time.time()
        lab = core.Lab(10, 8, core.h1_rows(4096, seed=seed), R2, seed=seed, gate=1.0, batch=64)
        td, tb = lab.mc.true_tables()
        for s in range(0, steps + 1):
            if s > 0:
                lab.train_step()
            if s % every == 0:
                Td, Tb = lab.tables()
                Td, Tb = Td.flatten(), Tb.flatten()
                right = int(((Td == td) & (Tb == tb)).sum())
                out.append(dict(seed=seed, step=s, right=right, digits_right=int((Td == td).sum()),
                                borrows_right=int((Tb == tb).sum()), gate=lab.gated_on,
                                sbb_d=Td.tolist(), sbb_c=Tb.tolist()))
                print(f"seed {seed} step {s:3d}: {right:3d}/200 entries right (digits {int((Td == td).sum())}, "
                      f"borrows {int((Tb == tb).sum())}), gate {lab.gated_on}, {time.time() - t0:.0f}s", flush=True)
    TABLES.write_text(json.dumps(out))
    print("wrote", TABLES)


def snapshot(seed, right_at_least=None, step=None):
    """The first snapshot of a seed with at least `right_at_least` entries right (or the one at `step`)."""
    rows = [r for r in json.loads(TABLES.read_text()) if r["seed"] == seed]
    for r in rows:
        if (step is not None and r["step"] == step) or (right_at_least is not None and r["right"] >= right_at_least):
            return r
    return None


if __name__ == "__main__":
    if sys.argv[1:2] == ["train"]:
        train()
