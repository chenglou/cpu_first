"""Replays the evaluator's training batch order (data/factory.py: a DataLoader over train.jsonl in file order, batch
512, shuffle, drop_last, a CPU generator seeded with the run's seed) and prints, per step, the batch's smallest T and
how many rows have it: which batches gave ans_smallest_t's pool rows with T > 1 (ABLATION_S3.md §3). Checked against
the control's log: the answer pool's size at steps 650, 700 and 750. Local: seconds, about 300 MB.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/batch_t.py [SEED=74] [FROM=600] [TO=800]
"""

import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

DATA = os.path.expanduser("~/github/one-layer/one-layer-deeper/data/generated/"
                          "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2/train.jsonl")
args = dict(a.split("=") for a in sys.argv[1:])
seed, lo, hi = int(args.get("SEED", 74)), int(args.get("FROM", 600)), int(args.get("TO", 800))
T = torch.tensor([json.loads(line)["time_steps"] for line in open(DATA)])


class Index(Dataset):
    def __len__(self):
        return len(T)

    def __getitem__(self, i):
        return i


loader = DataLoader(Index(), batch_size=512, shuffle=True, drop_last=True,
                    generator=torch.Generator(device="cpu").manual_seed(seed), collate_fn=torch.tensor)
it, step, pool = iter(loader), 0, 0
print(f"{len(T)} rows, T=1: {int((T == 1).sum())}, {len(loader)} batches an epoch")
while step < hi:
    try:
        idx = next(it)
    except StopIteration:                        # the evaluator's _next_batch: a new iterator each epoch
        it = iter(loader)
        idx = next(it)
    step += 1
    t = T[idx]
    tmin = int(t.min())
    n = int((t == tmin).sum())
    kept = min(n, max(1, 96 // (tmin + 1)))      # answer_rows: at most ANS_WORK // (T + 1) rows
    if step >= lo:
        pool += kept if tmin == 1 else 0          # s3i keeps T = 1 rows only
        if tmin != 1 or step % 50 == 0:
            print(f"step {step}: smallest T {tmin} ({n} rows, {kept} kept){'  <-- no T = 1 row' if tmin != 1 else ''}"
                  f"  [T = 1 rows kept since step {lo}: {pool}]")
