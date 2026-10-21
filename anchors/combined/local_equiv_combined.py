"""How close anchors/submission_combined_anchors.py with TABLE_TEACH = "adder" comes to the certified
learned_remainder/submission_combined.py (batch 16 from a random start, as the evaluator steps). It prints, per step,
whether the losses and every parameter are identical, up to the first step where a parameter differs.

What it shows (ANCHORS.md §6): the losses and every parameter are identical through step 7; at step 8 (I/O stage 2)
the parameters start to drift (a, b, u, wc). The likely cause is the zero-valued term the order branch adds to the loss
(it touches every parameter): Adam then steps parameters that no lesson has reached yet, with zero gradients, and the
gradient sums are accumulated in a different order. "adder" mode is the certified recipe's lessons and schedule, not
the certified run bit for bit.

    python local_equiv_combined.py [STEPS]
"""

import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LR_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "learned_remainder")
sys.path[:0] = [os.path.join(LR_DIR, "chain_local", "stub"), HERE]
import torch
from benchmark import TokenLossBatch
import local_combined_anchors as lc


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(mod, steps, B, rows, sets=None):
    for k, v in (sets or {}).items():
        setattr(mod, k, v)
    torch.manual_seed(1)
    m = mod.build_model(lc.spec()).train()
    bundle = mod.build_optimizer(m, lc.OptimizerSpec(3600.0, "cpu"))
    g = torch.Generator().manual_seed(1)
    trace = []
    for _ in range(steps):
        idx = torch.randint(0, len(rows), (B,), generator=g)
        ids, am, lab, tpos = lc.batch([rows[i] for i in idx])
        logits, aux = m(ids, am)
        loss = mod.token_training_loss(TokenLossBatch(logits, lab, lab != -100, tpos, aux))
        bundle.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        bundle.optimizer.step()
        bundle.scheduler.step()
        trace.append((float(loss.detach()), {n: p.detach().clone() for n, p in m.named_parameters()}, m.ctl.io))
    return trace


def main(steps):
    rows = lc.load("train", 20000)
    t0 = time.time()
    a = run(module(os.path.join(LR_DIR, "submission_combined.py"), "orig"), steps, 16, rows, {"LOG_EVERY": 10 ** 9})
    b = run(module(os.path.join(os.path.dirname(HERE), "submission_combined_anchors.py"), "anch"), steps, 16, rows,
            {"LOG_EVERY": 10 ** 9, "TABLE_TEACH": "adder"})
    for s, (x, y) in enumerate(zip(a, b), 1):
        diff = [n for n in x[1] if not torch.equal(x[1][n], y[1][n])]
        print(f"step {s}: losses identical {x[0] == y[0]}, I/O stage {x[2]}, parameters differing {diff}", flush=True)
        if diff:
            break
    print(f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
