"""How T's field width decides the reading (LEARNED_LOOP.md §6): the file's phase A (the reader and the subtraction
table from random) for STEPS steps, with T's field K slots wide (the file's K = 2, or K = 8, as wide as N's and x's),
on several seeds, until the reading is chosen (phase B); prints when each reader stage came and which reading (N's
run and direction, x's run and direction) the reader's pointers make then. Local, CPU, a minute a run.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/learned_loop/checks/reader_k.py [STEPS=1200] [B=256] [K=8] [SEEDS=0,1,2]
"""
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
arg = lambda k, d: next((a.split("=", 1)[1] for a in argv if a.startswith(k + "=")), d)
steps, B, K = int(arg("STEPS", 1200)), int(arg("B", 256)), arg("K", "8")
sc = LW.sc = LW.load_module(os.path.join(LL, "submission_stage3_ll.py"), [f"K={K}", "LOG_EVERY=100000"])
rows = LW.load("train")
for seed in [int(s) for s in arg("SEEDS", "0,1,2").split(",")]:
    t0 = time.time()
    torch.manual_seed(seed)
    m = sc.build_model(LW.spec())
    m.train()
    bundle = sc.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    g = torch.Generator().manual_seed(seed)
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
        if m.ctl.phase != "A":                      # the reading is chosen (phase B begins only once it is)
            break
    with torch.no_grad():
        e, end = m.ext(ids, am)
        k = m.selected(e, end)
        hyp = [sc.HYP[int(h)] if h >= 0 else None for h in k.tolist()]
    top = max(set(hyp), key=hyp.count)
    print(f"K={K} seed={seed}: {m.ctl.history[:6]} phase {m.ctl.phase} io {m.ctl.io}; the pointers' reading "
          f"(N run, N dir, x run, x dir) {top} on {hyp.count(top)}/{len(hyp)} rows "
          f"({'x from the second run: right' if top and top[2] == 1 else 'x from the third run (T): wrong'}) "
          f"{time.time() - t0:.0f}s", flush=True)
