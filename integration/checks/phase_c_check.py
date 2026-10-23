"""An integration file's training step from phase C on (tournament/local/phase_c_check.py, made per lesson set): every
table and the reader at the truth (MODE "designed"), every lesson set's tournament blocks, the gate table, the
selector and the round's switches re-randomized, the schedule set to phase C; then the evaluator's step (forward,
token_training_loss, backward, Adam, the scheduler) on batches of the real H1 training rows, until the model freezes or
STEPS. Exercises each set's chain tournament, its I/O lessons, the selector and round by the answers, and the freeze.

    ~/github/sotaku/venv/bin/python integration/checks/phase_c_check.py [FILE=submission_stage1.py] [STEPS=60] [SEED=0]
        [B=512] [K=V ...]
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
steps = next((int(a[6:]) for a in argv if a.startswith("STEPS=")), 60)
seed = next((int(a[5:]) for a in argv if a.startswith("SEED=")), 0)
B = next((int(a[2:]) for a in argv if a.startswith("B=")), 512)
fname = next((a[5:] for a in argv if a.startswith("FILE=")), "submission_stage1.py")
sets = [a for a in argv if "=" in a and not a.startswith(("STEPS=", "SEED=", "B=", "FILE="))]
sc = LW.sc = LW.load_module(os.path.join(INTEG, fname), ["MODE=\"designed\"", "LOG_EVERY=5"] + sets)
torch.manual_seed(seed)
m = sc.build_model(LW.spec())
with torch.no_grad():
    rand = [p for bd in m.blks for p in bd.values()] + [m.g, m.lsel] + list(m.rnd.values())
    for p in rand:                                                          # every set's blocks, the gate, the selector
        p.copy_(sc.INIT * torch.randn_like(p))
c = m.ctl
c.phase, c.io, c.ladder, c.gated_on = "C", 5, True, -1000
c.wire_ios = [False] * len(sc.PREDICATES)
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
    per = " | ".join(f"{p}: {m.wiring_names(i)} chain={sc.classify(m.names(i))['chain']}"
                     for i, p in enumerate(sc.PREDICATES))
    sel = ",".join(f"{float(v):.3f}" for v in m.lsel.float().softmax(-1).tolist())
    print(f"STEP {step} loss {float(loss):.4f} phase {c.phase} wire_ios {c.wire_ios} streak {c.streak} "
          f"selector {sel} round {m.round_names()} chain {c.chain_info} active "
          f"{'/'.join(','.join(f'{n}={int(m.active(n, i).numel())}' for n in sc.BLOCKS) for i in range(len(sc.PREDICATES)))} "
          f"wiring {per} {time.time() - ts:.1f}s", flush=True)
    if c.phase == "frozen":
        break
print(f"history {c.history} {time.time() - t0:.0f}s", flush=True)
