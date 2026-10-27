"""The windows' check (FAMILY.md §3): the window rule on the family's readings, then windows posed as the climb poses
them (spcore_f.Root and Acceptor, the answers by hooks.subterm_target), on the data's rows, against programs whose
function is known: right programs must pass the law's probes and its acceptor at every verify width, wrong ones must
fail. Writes logs/known_lib.json, the known routines, for the routes' check on both ladders (verify7.py).

    cd cpu_first && SUBTERMS=1 CONST=1 bar3/scale/rowcheck/memcap.sh 8 ~/github/sotaku/venv/bin/python -m bar3.scale.family.family_check
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

assert os.environ.get("SUBTERMS") and os.environ.get("CONST"), "the windows' and the constants' patterns need the flags"
from integration.lib import libcore as L                               # noqa: E402
from bar3.scale.family import hooks as H                               # noqa: E402
from bar3.scale.family import spcore_f as S                            # noqa: E402
from bar3.scale.family.core_f import digits, load_rows                 # noqa: E402

READINGS = {   # a reading -> its windows' shapes, smallest first (two holes at most, three operations at most)
    "mod(mul(X,X),N)": ["mod(mul(a,b),N)"],
    "mod(mul(mod(mul(X,X),N),X),N)": ["mod(mul(a,b),N)", "mul(mod(a,N),b)", "mod(mul(mod(a,N),b),N)"],
    "mod(add(mul(mod(mul(X,X),N),10),1),N)": ["mod(add(a,1),N)", "add(mul(a,10),1)", "mul(mod(a,N),10)",
                                              "mod(mul(a,b),N)", "mod(add(mul(a,10),1),N)",
                                              "add(mul(mod(a,N),10),1)", "mul(mod(mul(a,b),N),10)"],
}
LIB = {"T.lt.pair.AN": "CLC P[SBB.N INC]", "T.lt.pair.AX": "CLC P[SBB.X INC]",
       "T.red.pair.AN": "STA.S T.lt.pair.AN SEL.S", "T.red.Bx+d.AN": "B[T.red.pair.AN]",
       "T.red.pair.AX": "STA.S T.lt.pair.AX SEL.S",
       "T.acc.mul.o,x,d.AXY": "LDH.Y CLK P[MUL.X INC]", "T.mul.x,o.XY": "CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]",
       "T.red.x+on.XN": "LDA.N P[BRG.X T.red.Bx+d.AN]",
       "T.add.pair.AX": "CLC P[ADC.X INC]", "T.add.pair.AY": "CLC P[ADC.Y INC]",
       "T.const.1.n.N": "CLA STA.S T.lt.pair.AN CLA ADC.S",
       "T.add.1.x,n.AN": "STA.X T.const.1.n.N T.add.pair.AX",
       "T.add.3.x,n.AN": "T.add.1.x,n.AN T.add.1.x,n.AN T.add.1.x,n.AN",
       "T.pred.A": "STA.Y T.const.1.n.N STA.X LDA.Y T.lt.pair.AX",
       "T.w.mod.mul.w2.XYN": "T.mul.x,o.XY STA.X T.red.x+on.XN",
       "T.w.mod.mul10.w1.AN": "SHL.A STA.X T.red.x+on.XN",
       "T.w.mod.add1.w1.AN": "T.add.1.x,n.AN STA.X T.red.x+on.XN",
       "T.w.mod.add3.w1.AN": "T.add.3.x,n.AN STA.X T.red.x+on.XN",
       "T.w.mod.add1.mul10.w1.AN": "SHL.A T.add.1.x,n.AN STA.X T.red.x+on.XN",
       "T.w.mod.pred.add.w1.AN": "STA.Y LDA.N T.add.pair.AY T.pred.A STA.X T.red.x+on.XN"}
WINDOWS = {"w.mod.mul": "mod(mul(a,b),N)", "w.mod.mul10": "mod(mul(a,10),N)", "w.mod.add1": "mod(add(a,1),N)",
           "w.mod.add3": "mod(add(a,3),N)", "w.mod.add1.mul10": "mod(add(mul(a,10),1),N)",
           "w.mod.add1.mul": "mod(add(mul(a,b),1),N)", "w.mod.pred.add": "mod(pred(add(a,N)),N)"}
CASES = [   # (law, pattern, binding, program, should pass)
    ("w.mod.mul", "w2", "XYN", "T.mul.x,o.XY STA.X T.red.x+on.XN", True),             # (a b) mod N
    ("w.mod.mul", "w2", "XYN", "T.mul.x,o.XY", False),                                #   unreduced
    ("w.mod.mul10", "w1", "AN", "SHL.A STA.X T.red.x+on.XN", True),                   # 10 a mod N
    ("w.mod.mul10", "w1", "AN", "SHL.A", False),
    ("w.mod.add1", "w1", "AN", "T.add.1.x,n.AN STA.X T.red.x+on.XN", True),           # (a + 1) mod N
    ("w.mod.add1", "w1", "AN", "T.add.1.x,n.AN", False),                              #   wrong where a + 1 >= N
    ("w.mod.add3", "w1", "AN", "T.add.3.x,n.AN STA.X T.red.x+on.XN", True),           # (a + 3) mod N
    ("w.mod.add1.mul10", "w1", "AN", "SHL.A T.add.1.x,n.AN STA.X T.red.x+on.XN", True),   # (10 a + 1) mod N
    ("w.mod.add1.mul10", "w1", "AN", "SHL.A STA.X T.red.x+on.XN", False),
    ("w.mod.add1.mul", "w2", "XYN", "T.mul.x,o.XY T.add.1.x,n.AN STA.X T.red.x+on.XN", True),   # (a b + 1) mod N
    ("w.mod.pred.add", "w1", "AN", "STA.Y LDA.N T.add.pair.AY T.pred.A STA.X T.red.x+on.XN", True),   # (a + N - 1) mod N
    ("w.mod.pred.add", "w1", "AN", "STA.X T.red.x+on.XN", False),
    # addendum 1's fixes. CONST1: pred, the reader's max(a - 1, 0), over make 1 and the link (one conditional
    # subtraction of 1); the plain subtraction is wrong at 0
    ("pred", "x,n", "AN", "STA.Y T.const.1.n.N STA.X LDA.Y T.red.pair.AX", True),
    ("pred", "x,n", "AN", "STA.Y T.const.1.n.N STA.X LDA.Y T.lt.pair.AX", False),
    # HOLES: with the hole below N (w1), one conditional subtraction passes for the reduction of a + N - 1; with the
    # hole from another row (w1o) it fails, and only the whole reduction passes
    ("w.mod.pred.add", "w1", "AN", "STA.Y LDA.N T.add.pair.AY T.pred.A T.red.pair.AN", True),
    ("w.mod.pred.add", "w1o", "AN", "STA.Y LDA.N T.add.pair.AY T.pred.A T.red.pair.AN", False),
    ("w.mod.pred.add", "w1o", "AN", "STA.Y LDA.N T.add.pair.AY T.pred.A STA.X T.red.x+on.XN", True),
    ("w.mod.mul", "w2o", "XYN", "T.mul.x,o.XY STA.X T.red.x+on.XN", True),
    ("w.mod.mul", "w2o", "XYN", "T.mul.x,o.XY STA.X T.red.pair.AN", False),
]


def main():
    bad = 0
    for law, want in READINGS.items():
        got = [s for s, _, _ in H.windows(H._tree(law), 3, 2)]
        res = "ok" if got == want else "WRONG"
        bad += res != "ok"
        print(f"windows of {law}: {got}: {res}", flush=True)
    dev = "cpu"
    rows = load_rows(4096, seed=0)
    Nd, Xd = digits(rows, "modulus").to(dev), digits(rows, "x").to(dev)
    settings = [L.Setting(10, w, L.Cells(10, dev)) for w in (1, 2)]
    verify = [L.Setting(10, w, L.Cells(10, dev)) for w in (1, 2, 3)]
    lib = {k: L.parse(v) for k, v in LIB.items()}
    for name, shape in WINDOWS.items():
        H.SUBTERM_TREES[name] = H._tree(shape)
    for law, pat, *_ in CASES:
        H.SLICE_PAIRS[(law, pat)] = "check"
        H.READING_PAIRS.add((law, pat))
    ops = S.Ops()
    one = torch.zeros(1, dtype=torch.long)
    for law, pat, bd, prog, want in CASES:
        root = S.Root(pat, tuple(bd), settings, 8, (Nd, Xd), ops, lib, dev, 0)
        g = root.goals(ops, lib, [law])
        acc = S.Acceptor(law, root, verify, (Nd, Xd), ops, lib, dev)
        p = L.parse(prog)
        runs = [L.apply_items(st, b.take(one), one, [("REP", L.ONE, tuple(p))], lib) for st, b in zip(settings, root.roots)]
        probe = bool(g) and bool(g[0].test(runs, lib)[0])
        ok = acc.ok and acc.many([p], lib)[0]
        res = "ok" if ok == want and (probe == want or not want) else "WRONG"
        bad += res != "ok"
        print(f"{law}/{pat}@{bd}: {prog!r}: probes {probe}, accepted {ok}, expected {want}: {res}", flush=True)
    out = Path(__file__).resolve().parent / "logs" / "known_lib.json"
    out.write_text(json.dumps({"lib": LIB}, indent=1))
    print("ALL OK" if not bad else f"{bad} WRONG")


if __name__ == "__main__":
    main()
