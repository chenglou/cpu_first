"""Does the diagnostic copy (make_gate_replay.py) train as sp2 does? sp2's file, the diagnostic copy and the learned-loop
file, each from the same seed on the same batches, with the evaluator's step (forward, loss, backward, the gradient's
norm clipped at 1, Adam, the scheduler), until each reaches phase C (the climb) or STEPS; prints the first step where
each file's loss differs from sp2's (exact float equality) and each file's schedule. Local, CPU, minutes.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/learned_loop/diag/replay_check.py [STEPS=900] [B=256] [SEED=0] [TWICE=1]
    (TWICE=1: sp2's file against itself, to see how far a CPU run repeats exactly)
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CPU = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [os.path.join(CPU, "wider_menus", "local")]
import torch  # noqa: E402
import local_wide as LW  # noqa: E402
from benchmark import OptimizerSpec, TokenLossBatch  # noqa: E402

argv = sys.argv[1:]
arg = lambda k, d: next((a.split("=", 1)[1] for a in argv if a.startswith(k + "=")), d)
steps, B, seed = int(arg("STEPS", 900)), int(arg("B", 256)), int(arg("SEED", 0))
FILES = {"sp2": os.path.join(CPU, "integration", "self_practice", "submission_stage3_sp2.py"),
         "diag": os.path.join(HERE, "submission_stage3_ll_gate.py"),
         "ll": os.path.join(CPU, "integration", "learned_loop", "submission_stage3_ll.py")}
if arg("TWICE", ""):                       # a file against itself: is the training deterministic here?
    FILES = {"sp2": FILES["sp2"], "sp2_again": FILES["sp2"]}
rows = LW.load("train")
runs = {}
for name, path in FILES.items():
    t0 = time.time()
    sc = LW.sc = LW.load_module(path, ["LOG_EVERY=100000"])
    torch.manual_seed(seed)
    m = sc.build_model(LW.spec())
    m.train()
    bundle = sc.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    g = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(rows), (B,), generator=g)
        ids, am, lab, tpos = LW.batch([rows[i] for i in idx])
        logits, aux = m(ids, am)
        loss = sc.token_training_loss(TokenLossBatch(logits, lab, lab != -100, tpos, aux))
        bundle.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        bundle.optimizer.step()
        bundle.scheduler.step()
        losses.append(float(loss))
        if m.ctl.phase == "C":
            break
    runs[name] = (losses, list(m.ctl.history))
    print(f"{name}: {len(losses)} steps to phase {m.ctl.phase}, schedule {m.ctl.history} ({time.time() - t0:.0f}s)",
          flush=True)
ref = runs["sp2"][0]
for name in [n for n in runs if n != "sp2"]:
    ls = runs[name][0]
    n = min(len(ref), len(ls))
    first = next((i + 1 for i in range(n) if ls[i] != ref[i]), None)
    print(f"{name} against sp2: {'identical losses at all ' + str(n) + ' steps' if first is None else 'first differs at step ' + str(first)}"
          f"{'' if first is None else f' ({ref[first - 1]!r} against {ls[first - 1]!r})'}; same schedule: "
          f"{runs[name][1] == runs['sp2'][1]}", flush=True)
