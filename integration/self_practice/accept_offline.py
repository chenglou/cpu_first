"""INTEGRATION.md §1.3's other formulas against the round's search, offline (SELF_PRACTICE.md §4.5): the routines the
certified H1 run discovered (submission_stage3_sp_official's library at its round, inlined below), each formula's
answers on H1's T = 1 rows (computed here with Python's integers: a study check, never part of a training file), and
the round's search as the file runs it (the flat search to cost 5, then by calls to 2) and with one call more (3).

    run on Modal (a GPU; the full-size search is heavy): modal_sp.py --envs '[{"TAG": "accept_offline",
        "MODULE": "integration.self_practice.accept_offline"}]'
"""
import time

import torch

from integration.lib import libcore as L
from integration.self_practice import spcore as S
from integration.self_practice.core import digits, load_rows

dev = "cuda" if torch.cuda.is_available() else "cpu"
LIB = {
 'B0.1': 'SBB.Y INC',
 'B0.2': 'P[B0.1]',
 'B0.3': 'CLC B0.2',
 'T.lt.pair.AX': 'CLC P[SBB.X INC]',
 'T.lt.pair.AY': 'B0.3',
 'T.lt.pair.AN': 'CLC P[SBB.N INC]',
 'T.sub.pair.AY': 'CLC P[INC SBB.Y]',
 'B1.1': 'STA.S T.lt.pair.AY',
 'B1.2': 'STA.S T.lt.pair.AX',
 'B1.3': 'STA.S T.lt.pair.AN',
 'T.min.pair.AX': 'B1.2 LDA.X SEL.S',
 'T.min.pair.AY': 'B1.1 LDA.Y SEL.S',
 'T.min.pair.AN': 'B1.3 LDA.N SEL.S',
 'T.red.pair.AX': 'B1.2 SEL.S',
 'T.red.pair.AY': 'B1.1 SEL.S',
 'T.red.pair.AN': 'B1.3 SEL.S',
 'T.add.pair.AX': 'CLC P[ADC.X INC]',
 'T.add.pair.AY': 'CLC P[ADC.Y INC]',
 'T.add.pair.AN': 'CLC P[ADC.N INC]',
 'B3.1': 'B[T.red.pair.AX]',
 'B3.2': 'B[T.red.pair.AN]',
 'B3.3': 'MUL.N INC',
 'B3.4': 'P[B3.3]',
 'B3.5': 'CLK B3.4',
 'B3.6': 'CLK LDH.Y',
 'B3.7': 'CLK LDH.X',
 'T.acc.mul.o,x,d.AXY': 'LDH.Y CLK P[MUL.X INC]',
 'T.acc.mul.o,x,d.AYX': 'LDH.X CLK P[MUL.Y INC]',
 'T.acc.mul.o,x,d.ANX': 'LDH.X B3.5',
 'T.acc.mul.o,x,d.ANY': 'LDH.Y B3.5',
 'T.red.x+n.AX': 'P[T.red.pair.AX]',
 'T.red.x+n.AN': 'P[T.red.pair.AN]',
 'T.sub.x+n.XY': 'LDA.X T.lt.pair.AY',
 'T.sub.x+n.XN': 'LDA.X T.lt.pair.AN',
 'T.sub.x+n.YX': 'LDA.Y T.lt.pair.AX',
 'T.sub.x+n.YN': 'LDA.Y T.lt.pair.AN',
 'T.sub.x+n.NX': 'LDA.N T.lt.pair.AX',
 'T.sub.x+n.NY': 'LDA.N T.lt.pair.AY',
 'T.red.x+dn.AX': 'B3.1',
 'T.red.x+dn.AY': 'B[T.red.pair.AY]',
 'T.red.x+dn.AN': 'B3.2',
 'T.mul.x,o.XY': 'CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]',
 'T.mul.x,o.YX': 'CLA W[T.acc.mul.o,x,d.AYX INC SHL.Y]',
 'T.red.x+on.XA': 'STA.Y P[BRG.X T.red.x+dn.AY]',
 'T.red.Bx+d.XY': 'LDA.X T.red.x+dn.AY',
 'T.red.Bx+d.XN': 'LDA.X T.red.x+dn.AN',
 'T.red.Bx+d.YX': 'LDA.Y T.red.x+dn.AX',
 'T.red.Bx+d.YN': 'LDA.Y T.red.x+dn.AN',
 'T.red.Bx+d.NX': 'LDA.N T.red.x+dn.AX',
 'T.red.Bx+d.NY': 'LDA.N T.red.x+dn.AY',
 'T.mul.x,d.XY': 'CLA T.acc.mul.o,x,d.AXY',
 'T.mul.x,d.YX': 'CLA T.acc.mul.o,x,d.AYX',
 'T.mul.x,d.NX': 'CLA T.acc.mul.o,x,d.ANX',
 'T.mul.x,d.NY': 'CLA T.acc.mul.o,x,d.ANY'
}
STEP = {"h1": lambda y, n: y * y % n, "rho": lambda y, n: (y * y + 1) % n, "ten": lambda y, n: 10 * y * y % n,
        "cube": lambda y, n: y ** 3 % n, "mont": lambda y, n: y * y * pow(10 ** 8, -1, n) % n,
        "tenrho": lambda y, n: (10 * y * y + 1) % n, "plus3": lambda y, n: (y * y + 3) % n,
        "minus1": lambda y, n: (y * y - 1) % n}


def climb():
    c = S.SelfClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16, fuse=4)
    c.lib = {k: L.parse(v) for k, v in LIB.items()}
    for tok in LIB:
        if tok.startswith("T."):
            *law, pattern, binding = tok[2:].split(".")
            c.ops.add(".".join(law), pattern, tok, tuple(binding))
    return c


rows = load_rows(64, seed=1, T=1)
for task, step in STEP.items():
    t0 = time.time()
    rws = [dict(r, result=step(int(r["x"]), int(r["modulus"]))) for r in rows]
    ans = tuple(digits(rws, k).to(dev) for k in ("modulus", "x", "result"))
    c = climb()
    progs, rec = c.search_round(ans, t=1)
    line = f"{task}: the file's search: cost {rec['found_cost']} {rec['stop']} {rec['programs'][:2]}"
    if not progs:
        prob = L.round_problem(c.cells, ans, 8, 16, dev)
        res = L.bfs(prob, c.macro_classes(c.lib), c.lib, max_cost=3, max_states=4_000_000,
                    accept=lambda ps: [L.round_ok(p, c.lib, c.cells, ans, 16, dev) for p in ps], chunk=1 << 19, fuse=4)
        line += (f"; by calls to 3: cost {res.found_cost} levels {res.level_sizes} {res.stop} "
                 f"{[L.show(S.unwrap(p)) for p in (res.accepted or [])][:2]}")
    print(f"{line} ({time.time() - t0:.0f} s)", flush=True)
