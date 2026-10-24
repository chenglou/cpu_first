"""The learned-loop file's training path after the round (LEARNED_LOOP.md §4): the reader and the cells at the truth
(MODE = "designed"), T's direction not yet resolved (io 4), the climb replaced by a finished one that holds sp2's
official run's two rounds and the routines they call (a replay, as integration/lib/debug_patch.py's REPLAY does),
then evaluator-style training steps on real H1 rows: the count pool fills, the answer program is searched inside the
file (one unit), phase D's tournament and T's direction lesson run through the answer program, and the model
freezes. Small budgets; local, CPU.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/learned_loop/checks/smoke_replay.py [STEPS=60] [B=64] [K=V ...]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LL = os.path.dirname(HERE)
CPU = os.path.dirname(os.path.dirname(LL))
sys.path[:0] = [os.path.join(CPU, "wider_menus", "local")]
import torch  # noqa: E402
import local_wide as LW  # noqa: E402
from benchmark import OptimizerSpec, TokenLossBatch  # noqa: E402

argv = sys.argv[1:]
steps = next((int(a[6:]) for a in argv if a.startswith("STEPS=")), 60)
B = next((int(a[2:]) for a in argv if a.startswith("B=")), 64)
sets = [a for a in argv if "=" in a and not a.startswith(("STEPS=", "B="))]
base = ["MODE=\"designed\"", "LOG_EVERY=10", "FREEZE_WIN=20", "T_AFTER=3", "OUTER_STATES=300000"]
sc = LW.sc = LW.load_module(os.path.join(LL, "submission_stage3_ll.py"), base + sets)
torch.manual_seed(0)
m = sc.build_model(LW.spec())
m.ctl.phase, m.ctl.io, m.ctl.gated_on = "C", 4, -1000
m.ctl.cells_on = True
m.train()
bundle = sc.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
full = {k: sc.parse(v) for k, v in json.load(open(os.path.join(HERE, "sp2_official_lib.json"))).items()}
rounds = [sc.parse("LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA"),
          sc.parse("LDA.X STA.Y T.mul.x,o.YX STA.X LDA.N T.red.x+on.XA")]


def names(items, out):
    for it in items:
        if isinstance(it, tuple):
            if it[1] in full:
                names(full[it[1]], out)
                out.add(it[1])
            names(it[2], out)
        elif it in full and it not in out:
            out.add(it)
            names(full[it], out)
    return out


keep = set()
for r in rounds:
    names(r, keep)
m.ctl.cells = m.snapped_cells()
c = m.ctl.climb = sc.SelfClimb(m.ctl.cells, "cpu", sc.PR, chunk=1 << 18, **sc.SP)
c.lib = {k: v for k, v in full.items() if k in keep}
c.rounds, c.done = rounds, True
print("replayed library:", {k: sc.show(v) for k, v in c.lib.items()}, flush=True)
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
    ctl = m.ctl
    print(f"STEP {step} loss {float(loss):.4f} phase {ctl.phase} io {ctl.io} counts "
          f"{ {k: v[0].shape[0] for k, v in sorted(ctl.count_pool.items())} } ans_ok {ctl.ans_ok} "
          f"streak {ctl.streak} progs {[sc.show(p) for p in ctl.progs]} {time.time() - t0:.2f}s", flush=True)
    if ctl.phase == "frozen":
        break
print("FROZEN" if m.ctl.phase == "frozen" else "NOT FROZEN", m.ctl.history)
