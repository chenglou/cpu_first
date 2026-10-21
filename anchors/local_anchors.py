"""Train a submission file locally the way the evaluator does (zero_grad, backward, clip 1, Adam step) on random H1
training rows, printing its snapped table against true subtraction. Small runs only (well under 1 GB and a minute).

    python local_anchors.py [FILE] [STEPS] [BATCH] [SEED] [CONSTANT=value ...]
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "learned_remainder", "chain_local", "stub"), HERE]
import torch
from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch

import core


def module(path, name="sub"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows(n, seed=0):
    with open(os.path.join(core.H1_DATA, "train.jsonl")) as f:
        lines = f.readlines()
    g = torch.Generator().manual_seed(seed)
    return [json.loads(lines[i]) for i in torch.randperm(len(lines), generator=g)[:n].tolist()]


def batch(rs):
    L = max(len(r["input_ids"]) for r in rs)
    ids = torch.zeros(len(rs), L, dtype=torch.long)
    lab = torch.full((len(rs), 8), -100, dtype=torch.long)
    for i, r in enumerate(rs):
        ids[i, :len(r["input_ids"])] = torch.tensor(r["input_ids"])
        lab[i, :len(r["labels"])] = torch.tensor(r["labels"])
    return ids, ids != 0, lab


def train(mod, steps, B, seed, data, every=25, quiet=False):
    torch.manual_seed(seed)
    m = mod.build_model(ModelSpec(vocab_size=17, max_seq_len=21, maximum_model_state_elements=500_000_000)).train()
    bundle = mod.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    g = torch.Generator().manual_seed(seed)
    fam = core.families(10)
    out = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(data), (B,), generator=g)
        ids, am, lab = batch([data[i] for i in idx])
        logits, aux = m(ids, am)
        loss = mod.token_training_loss(TokenLossBatch(logits, lab, lab != -100, None, aux))
        bundle.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        bundle.optimizer.step()
        if bundle.scheduler is not None:
            bundle.scheduler.step()
        if step % every == 0 or step == steps:
            Td, Tb = m.tables()
            st = core.describe(Td, Tb, fam)
            out.append((step, float(loss.detach()), st))
            if not quiet:
                print(f"step {step:4d} loss {float(loss.detach()):.4f} digits {st['digits']} borrows {st['borrows']} "
                      f"rows {st['rows']} | {st['closest']}", flush=True)
    return m, out


def main(argv):
    path = argv[0] if argv else os.path.join(HERE, "submission_anchors.py")
    steps = int(argv[1]) if len(argv) > 1 else 300
    B = int(argv[2]) if len(argv) > 2 else 64
    seed = int(argv[3]) if len(argv) > 3 else 1
    mod = module(path)
    for kv in argv[4:]:
        k, v = kv.split("=", 1)
        setattr(mod, k, eval(v))
    t0 = time.time()
    m, _ = train(mod, steps, B, seed, rows(4096))
    print("wrong:", core.table_str(*m.tables(), 10)[:400], f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1:])
