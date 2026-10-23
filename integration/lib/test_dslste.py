"""dslste (one-hot, straight-through) against tdsl (hard digits) on LIBRARY.md's x^2 mod N at H1's size, and against
the rows' answers; then the gradient from the output to the tables. Local, seconds, small.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m integration.lib.test_dslste
"""
import json
import os
import time

import torch

from library import dsl as pdsl
from integration.lib import tdsl, inclimb, dslste

here = os.path.dirname(os.path.abspath(__file__))
lib = {k: pdsl.parse(v) for k, v in json.load(open(os.path.join(here, "r11_lib.json")))["lib"].items()}
prog = lib["T.sqmod"]
rows = inclimb.load_rows(8, seed=3, T=1)
Nd, Xd, Yd = inclimb.digits_of_rows(rows)
cells = tdsl.Cells(10)
st = tdsl.Setting(10, 8, cells)
A, _ = inclimb.run_prog(st, prog, lib, {"X": inclimb.pad(Xd, 16), "N": inclimb.pad(Nd, 16)}, torch.Generator().manual_seed(0))
print("hard machine vs answers:", float((A[:, :8] == Yd).all(1).float().mean()))
SBB, ADC, MUL = dslste.tables_onehot(cells)
SBB.requires_grad_(), ADC.requires_grad_(), MUL.requires_grad_()
m = dslste.SteMachine(SBB, ADC, MUL, P=16)
regs, c, h, k = dslste.start(8, 16, {"X": Xd, "N": Nd})
t0 = time.time()
regs, c, h, k, p = m.run(regs, c, h, k, prog, lib)
out = regs[:, 0].argmax(-1)
print(f"one-hot machine vs hard machine: {bool((out == A.long()).all())}, vs answers "
      f"{float((out[:, :8] == Yd).all(1).float().mean())}, exactly one-hot {bool(((regs == 0) | (regs == 1)).all())}, "
      f"{time.time() - t0:.1f}s")
loss = -(regs[:, 0, :8].clamp_min(1e-9).log() * torch.nn.functional.one_hot(Yd, 10)).sum()
loss.backward()
print("gradient reaches SBB/ADC/MUL:", [bool(t.grad is not None and t.grad.abs().sum() > 0) for t in (SBB, ADC, MUL)])
