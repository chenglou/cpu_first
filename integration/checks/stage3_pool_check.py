"""Diagnostic (INTEGRATION.md §3): the stage-3 file trained as the evaluator does (learned reader and cells, real H1
rows, batch 512) until phase C has filled the round's answer pool, then every pooled row (N, x, y as read) checked
against y = x^2 mod N, and the misread rows printed with the true values. The climb is cut short (LEVELS=0) since only
the pool matters. Not a lesson: a check from outside the model.

    python integration/checks/stage3_pool_check.py [STEPS=760] [K=V ...]
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
steps = next((int(a[6:]) for a in argv if a.startswith("STEPS=")), 760)
seed = next((int(a[5:]) for a in argv if a.startswith("SEED=")), 0)
sets = [a for a in argv if "=" in a and not a.startswith(("STEPS=", "SEED=", "FILE="))]
sc = LW.sc = LW.load_module(os.path.join(INTEG, "submission_stage3.py"), ["LOG_EVERY=100", "LEVELS=0"] + sets)
torch.manual_seed(seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = sc.build_model(LW.spec()).to(dev)
m.train()
bundle = sc.build_optimizer(m, OptimizerSpec(3600.0, dev))
rows = LW.load("train")
g = torch.Generator().manual_seed(seed)
t0 = time.time()
for step in range(1, steps + 1):
    idx = torch.randint(0, len(rows), (512,), generator=g)
    ids, am, lab, tpos = (t.to(dev) for t in LW.batch([rows[i] for i in idx]))
    logits, aux = m(ids, am)
    loss = sc.token_training_loss(TokenLossBatch(logits, lab, lab != -100, tpos, aux))
    bundle.optimizer.zero_grad()
    loss.backward()
    bundle.optimizer.step()
    bundle.scheduler.step()
v = lambda d: int("".join(str(int(t)) for t in reversed(d.tolist())) or 0)
pool = m.ctl.ans_pool
print(f"after {steps} steps ({time.time() - t0:.0f}s): phase {m.ctl.phase}, pool {len(pool)} batches, "
      f"{sum(a[0].shape[0] for a in pool)} rows", flush=True)
bad = tot = 0
for b, (Nd, Xd, Yd) in enumerate(pool):
    for i in range(Nd.shape[0]):
        N, x, y = v(Nd[i].cpu()), v(Xd[i].cpu()), v(Yd[i].cpu())
        tot += 1
        if N == 0 or x * x % N != y:
            bad += 1
            if bad <= 12:
                print(f"POOL batch {b} row {i}: N {N} x {x} y(read) {y}; x^2 mod N = {x * x % N if N else None}", flush=True)
print(f"POOL rows {tot} inconsistent {bad}", flush=True)
# the round's search as the in-run climb ran it at level 6 (s3d_relaxed_h100): its library (lib/s3d_sleeps.txt), this
# model's own answer pool and cells
lib = {}
for line in open(os.path.join(INTEG, "lib", "s3d_sleeps.txt")):
    body = line.split("; ", 1)[1].rsplit(" (", 1)[0]
    for part in body.split("; "):
        if "=" in part:
            k, val = part.split("=", 1)
            lib[k.strip()] = sc.parse(val.strip())
ans = tuple(torch.cat([a[i] for a in pool]) for i in range(3))
c = sc.Climb(m.snapped_cells(), dev, sc.PR, **{**sc.CLIMB, "levels": 12})
c.lib = dict(lib)
t1 = time.time()
progs, rec = c.search_round(ans)
print("ROUND on the model's own pool:", rec["found_cost"], rec["level_sizes"], rec["stop"], rec["programs"][:4],
      f"{time.time() - t1:.1f}s", flush=True)
good = sc.parse("LDA.X STA.Y T.mul~ STA.X T.reduce")
print("fit of LDA.X STA.Y T.mul~ STA.X T.reduce on the pool:", sc.round_fit(good, lib, m.snapped_cells(), ans, sc.PR, dev),
      flush=True)
