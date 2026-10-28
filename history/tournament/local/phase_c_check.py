"""The H1 file's training step from phase C on (seconds to minutes on a laptop): every table and the reader at the
truth (MODE "designed"), the division's four tournament blocks and the gate table re-randomized, the schedule set to phase C; then the
evaluator's step (forward, token_training_loss, backward, Adam, the scheduler) on batches of the real H1 training
rows, until the model freezes or STEPS. Exercises the chain's tournament, the I/O lessons, the answers and the freeze.

    ~/github/sotaku/venv/bin/python tournament/local/phase_c_check.py [STEPS=60] [SEED=0] [B=512] [K=V ...]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "wider_menus", "local")]
import torch
import local_wide as LW
from benchmark import OptimizerSpec, TokenLossBatch

argv = sys.argv[1:]
steps = next((int(a[6:]) for a in argv if a.startswith("STEPS=")), 60)
seed = next((int(a[5:]) for a in argv if a.startswith("SEED=")), 0)
B = next((int(a[2:]) for a in argv if a.startswith("B=")), 512)
sets = [a for a in argv if "=" in a and not a.startswith(("STEPS=", "SEED=", "B="))]
sc = LW.sc = LW.load_module(os.path.join(os.path.dirname(HERE), "submission_tournament.py"),
                            ["MODE=\"designed\"", "LOG_EVERY=5"] + sets)
torch.manual_seed(seed)
m = sc.build_model(LW.spec())
with torch.no_grad():
    for p in list(m.blk.values()) + [m.g]:                                 # the blocks and the gate table
        p.copy_(sc.INIT * torch.randn_like(p))
c = m.ctl
c.phase, c.io, c.ladder, c.wire_io, c.gated_on = "C", 5, True, False, -1000
m.train()
bundle = sc.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
rows = LW.load("train")
g = torch.Generator().manual_seed(seed)
t0 = time.time()
for step in range(1, steps + 1):
    ts = time.time()
    idx = torch.randint(0, len(rows), (B,), generator=g)
    ids, am, lab, tpos = LW.batch([rows[i] for i in idx])
    logits, aux = m(ids, am)
    loss = sc.token_training_loss(TokenLossBatch(logits, lab, lab != -100, tpos, aux))
    bundle.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    bundle.optimizer.step()
    bundle.scheduler.step()
    print(f"STEP {step} loss {float(loss):.4f} phase {c.phase} wire_io {c.wire_io} streak {c.streak} "
          f"chain {c.chain_info} wiring {m.wiring_names()} chain={sc.classify(m.names())['chain']} "
          f"{time.time() - ts:.1f}s", flush=True)
    if c.phase == "frozen":
        break
print(f"history {c.history} {time.time() - t0:.0f}s", flush=True)
