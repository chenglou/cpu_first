"""READING.md §5.4: how often a record's learned make-1 routines give 1 on dirty machines, at the practice widths and
at full size: N' the data's top parts as the n pattern draws them, every other register and the carry random.

    cd cpu_first && SUBLAW=1 CONST=1 ~/github/sotaku/venv/bin/python -m bar3.scale.reading.const_robust records/r6_10x2p1_nobook.json
"""
import json, sys
import torch
from integration.lib import libcore as L
from bar3.scale.reading.core_r import digits, load_rows
from bar3.scale.reading.oracle_r import top_parts
rec = json.load(open(sys.argv[1]))
lib = {k: L.parse(v) for k, v in rec["lib"].items()}
rows = load_rows(4096, seed=0)
Nd, Xd = digits(rows, "modulus"), digits(rows, "x")
gen = L.gen_on("cpu", 5)
for name, reg in (("T.const.1.n.N", "N"), ("T.const.1.n.Y", "Y")):
    for W in (1, 2, 3, 8):
        st = L.Setting(10, W, L.Cells(10, "cpu"))
        P = st.P
        pick = torch.randint(0, Nd.shape[0], (4000,), generator=gen)
        Nt, _ = top_parts(Nd[pick], Xd[pick], P)
        Nt = Nt[(Nt != 0).any(1)]
        b = L.fresh(st, {reg: L.pad(Nt, P)}, gen, Nt.shape[0], "cpu")
        L.run_items(st, b, L.parse(name), lib)
        A = b.regs[0, 0].T.long()
        one = torch.zeros_like(A); one[:, 0] = 1
        ok = (A == one).all(1)
        low0 = (Nt[:, 0] == 0)
        print(f"{name} ({rec['lib'][name]}) at W={W}, P={P}: {int(ok.sum())}/{len(ok)} right; rows with N'[0] = 0: {int(low0.sum())}, of them right {int(ok[low0].sum())}")
