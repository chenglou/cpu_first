"""'Generalised Hard': the recurrence is unknown. CPU-first answer: one self-calibrated ALU, a menu of complete
candidate programs, and labels that select (speculation over whole machines + an exact all-or-nothing test).

Hidden tasks reuse H1's (N, x, T) rows and recompute y under a different recurrence s_{t+1} = f(s_t).
Menu: s' = s^e * x0^f + a, e in {1,2,3}, f in {0,1}, a in {0, 1, 2, x0}  (24 programs), jointly with N' per class.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

import h1
from machine import ALU, K, ninv_class, run

TASKS = {
    "square (H1)": lambda s, x0, n: s * s % n,
    "cube": lambda s, x0, n: s * s * s % n,
    "square+1": lambda s, x0, n: (s * s + 1) % n,
    "square*x0": lambda s, x0, n: s * s * x0 % n,
    "square+x0": lambda s, x0, n: (s * s + x0) % n,
    "cube+2": lambda s, x0, n: (s * s * s + 2) % n,
}


def relabel(d, f):
    ys = []
    for n, x, t in zip(d["N"].tolist(), d["X"].tolist(), d["T"].tolist()):
        s = x % n
        for _ in range(t):
            s = f(s, x, n)
        ys.append(s)
    return dict(d, Y=np.array(ys, dtype=np.int64))


def make_program(e, f, a):
    def prog(alu, s, ctx):
        mN, ni = ctx["multN"], ctx["ninv"]
        v = s
        for _ in range(e - 1):
            v = alu.monpro(v, s, mN, ni)
        if f:
            v = alu.monpro(v, ctx["x0"], mN, ni)
        if a == "x0":
            v = alu.modadd(v, ctx["x0"], ctx["Nd"])
        elif a:
            v = alu.modadd(v, ctx[f"c{a}"], ctx["Nd"])
        return v
    prog.name = f"s^{e}" + ("*x0" if f else "") + (f"+{a}" if a else "")
    return prog


MENU = [make_program(e, f, a) for e, f, a in itertools.product((1, 2, 3), (0, 1), (0, 1, 2, "x0"))]


def run_prog(alu, prog, d, ninv_of_class):
    # constants in Montgomery form (c*R mod N) come from the same entry microcode
    def wrapped(alu_, s, ctx):
        if "c1" not in ctx:
            for c in (1, 2):
                cd = torch.zeros_like(s)
                cd[:, 0] = c
                ctx[f"c{c}"] = alu_.entry(cd, ctx["multN"])
        return prog(alu_, s, ctx)
    return run(alu, wrapped, d["N"], d["X"], d["T"], ninv_of_class).numpy()


alu = ALU.learn(seed=0)
print("ALU from self-tests:", alu.exactness())
train = h1.load("train")
train = {k: v[train["T"] <= 2] for k, v in train.items()}
cert = {prof: {t: h1.load(f"{prof}_{t}") for t in h1.LADDER} for prof in ("depth_t", "depth_ood_n_t")}

for task, f in TASKS.items():
    t0 = time.time()
    tr = relabel(train, f)
    sub = {k: v[:400] for k, v in tr.items()}  # 400 T<=2 rows suffice: a wrong machine matches ~1/N of rows
    match = np.zeros((len(MENU), 10, len(sub["Y"])), dtype=np.float32)
    with torch.no_grad():
        for pi, prog in enumerate(MENU):
            for c in range(10):
                match[pi, c] = h1.exact(run_prog(alu, prog, sub, torch.full((4,), c, dtype=torch.long)), sub["Y"])
    match = torch.from_numpy(match)
    cls = ninv_class(sub["N"])
    gp = torch.zeros(len(MENU), requires_grad=True)   # gate over programs
    gn = torch.zeros(4, 10, requires_grad=True)       # gate over N' per N mod 10 class
    opt = torch.optim.Adam([gp, gn], lr=0.2)
    for _ in range(300):
        pn = F.softmax(gn, -1)[cls]                   # [rows, 10]
        pm = (F.softmax(gp, -1)[:, None, None] * pn.T[None] * match).sum((0, 1))
        loss = -torch.log(pm + 1e-6).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    best = MENU[gp.argmax().item()]
    ninv = gn.argmax(-1)
    rates = match.amax(1).mean(-1)
    top = sorted(((rates[i].item(), MENU[i].name) for i in range(len(MENU))), reverse=True)[:3]
    res = {}
    for prof, rungs in cert.items():
        max_t = 0
        for t in h1.LADDER:
            d = relabel(rungs[t], f)
            ok = h1.exact(run_prog(alu, best, d, ninv), d["Y"]).all()
            if not ok:
                break
            max_t = t
        res[prof] = max_t
    print(f"{task:12s} -> selected {best.name:10s} N'={ninv.tolist()}  certified Max T {res}  "
          f"(top match rates {[(round(r, 3), n) for r, n in top]}) [{time.time()-t0:.0f}s]", flush=True)
