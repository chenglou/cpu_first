"""The H1 file's chain tournament in isolation (seconds to minutes on a laptop): every table at the truth (MODE
"designed"), the four blocks' logits and the gate table re-randomized, then Adam on the chain lesson alone (link law + pairs) for a few
steps on batches of the real H1 training rows. Prints, per step, the active and passing counts, the argmax chain, and
the time.

    ~/github/sotaku/venv/bin/python tournament/local/chain_check.py [STEPS=30] [SEED=0] [K=V ...]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "wider_menus", "local")]
import torch
import local_wide as LW

argv = sys.argv[1:]
steps = next((int(a[6:]) for a in argv if a.startswith("STEPS=")), 30)
seed = next((int(a[5:]) for a in argv if a.startswith("SEED=")), 0)
sets = [a for a in argv if "=" in a and not a.startswith(("STEPS=", "SEED="))]
sc = LW.sc = LW.load_module(os.path.join(os.path.dirname(HERE), "submission_tournament.py"), ["MODE=\"designed\""] + sets)
torch.manual_seed(seed)
m = sc.build_model(LW.spec())
with torch.no_grad():
    for p in list(m.blk.values()) + [m.g]:                                 # the blocks and the gate table
        p.copy_(sc.INIT * torch.randn_like(p))
m.ctl.phase = "C"
opt = torch.optim.Adam([{"params": list(m.blk.parameters()), "lr": sc.LR_WIRE},
                        {"params": [m.g], "lr": sc.LR}])
rows = LW.load("train")
g = torch.Generator().manual_seed(seed)
tabs = m.tables()
for step in range(1, steps + 1):
    t0 = time.time()
    idx = torch.randint(0, len(rows), (512,), generator=g)
    ids, am, lab, tpos = LW.batch([rows[i] for i in idx])
    e, end = m.ext(ids, am)
    k = m.selected(e, end)
    code = m.ranks(k)
    Nd, _ = m.read(0, e, end, code)
    Xd, _ = m.read(1, e, end, code)
    X = Xd.shape[0]
    lad_loss, lad_ok, verified, lad_rows = m.ladder(torch.cat([Xd, Nd]), tabs, False)
    Cd, _ = m.read(2, e, end, code)
    tabs = m.tables()                                                    # the gate table moves
    loss, div_ok, regs = m.chain_lesson(Xd, Nd, lad_rows[X:, 0], verified[X:, 0], tabs, True, Cd)
    opt.zero_grad()
    if loss is not None:
        loss.backward()
        opt.step()
    print(f"step {step} loss {None if loss is None else round(float(loss), 4)} div_ok {div_ok:.3f} "
          f"{m.ctl.chain_info} active {[int(m.active(n).numel()) for n in sc.BLOCKS]} wiring {m.wiring_names()} "
          f"chain={sc.classify(m.names())['chain']} {time.time() - t0:.1f}s", flush=True)
