"""self_practice/smoke_phase_c.py for this folder's builds (GENERIC_OPS.md). Its docstring: a local check of the variant
file's phase C (SELF_PRACTICE.md): the reader and the cells at the truth (MODE =
"designed"), the schedule moved to phase C, then evaluator-style training steps (forward, token_training_loss,
backward, Adam, the scheduler) so that the self-practising climb runs inside the file, one unit per step, on the
batches' read rows. Tiny budgets; local, seconds, well under 1 GB.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/generic_ops/smoke_g.py [STEPS=30] [B=64] [FILE=name] [K=V ...]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CPU = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(CPU, "wider_menus", "local")]
import torch  # noqa: E402
import local_wide as LW  # noqa: E402
from benchmark import OptimizerSpec, TokenLossBatch  # noqa: E402

argv = sys.argv[1:]
steps = next((int(a[6:]) for a in argv if a.startswith("STEPS=")), 30)
B = next((int(a[2:]) for a in argv if a.startswith("B=")), 64)
name = next((a[5:] for a in argv if a.startswith("FILE=")), "submission_stage3_g.py")
sets = [a for a in argv if "=" in a and not a.startswith(("STEPS=", "B=", "FILE="))]
base = ["MODE=\"designed\"", "LOG_EVERY=10", "SP_PROBE=30000", "ROUND_STATES=20000", "SEARCH_S=20", "SP_LEVELS=2",
        "ANS_MIN=8"]
sc = LW.sc = LW.load_module(os.path.join("..", "integration", "generic_ops", name), base + sets)
torch.manual_seed(0)
m = sc.build_model(LW.spec())
m.ctl.phase, m.ctl.io, m.ctl.gated_on = "C", 5, -1000
m.ctl.cells_on = True
m.train()
bundle = sc.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
rows = LW.load("train")
g = torch.Generator().manual_seed(0)
for step in range(1, steps + 1):
    t0 = time.time()
    idx = torch.randint(0, len(rows), (B,), generator=g)
    ids, am, lab, tpos = LW.batch([rows[i] for i in idx])
    logits, aux = m(ids, am)
    loss = sc.token_training_loss(TokenLossBatch(logits, lab, lab != -100, tpos, aux))
    bundle.optimizer.zero_grad()
    loss.backward()
    bundle.optimizer.step()
    bundle.scheduler.step()
    print(f"STEP {step} loss {float(loss):.4f} phase {m.ctl.phase} answers t={m.ctl.ans_t} "
          f"{sum(a[0].shape[0] for a in m.ctl.ans_pool)} {time.time() - t0:.2f}s", flush=True)
c = m.ctl.climb
print("CLIMB level", c.level if c else None, "lib", {k: sc.show(v) for k, v in (c.lib if c else {}).items()})
