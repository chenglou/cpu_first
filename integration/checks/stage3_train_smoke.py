"""A few evaluator-style training steps of the stage-3 file (forward, token_training_loss, backward, Adam, the
scheduler) on real H1 rows: catches errors in the training path. Local: small batch, seconds.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/checks/stage3_train_smoke.py [STEPS=5] [B=32] [K=V ...]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INTEG = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.dirname(INTEG), "wider_menus", "local")]
import torch
import local_wide as LW
from benchmark import OptimizerSpec, TokenLossBatch

argv = sys.argv[1:]
steps = next((int(a[6:]) for a in argv if a.startswith("STEPS=")), 5)
B = next((int(a[2:]) for a in argv if a.startswith("B=")), 32)
sets = [a for a in argv if "=" in a and not a.startswith(("STEPS=", "B="))]
sc = LW.sc = LW.load_module(os.path.join(INTEG, "submission_stage3.py"), ["LOG_EVERY=1"] + sets)
torch.manual_seed(0)
m = sc.build_model(LW.spec())
if os.environ.get("FROM_B"):                         # the reader and the cells at the truth, the schedule at phase B
    m.ctl.phase, m.ctl.io, m.ctl.gated_on = "B", 4, -1000
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
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    bundle.optimizer.step()
    bundle.scheduler.step()
    print(f"STEP {step} loss {float(loss):.4f} phase {m.ctl.phase} io {m.ctl.io} {time.time() - t0:.2f}s", flush=True)
