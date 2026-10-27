"""The reading source's check (READING.md §3): the reading's parts (the sub-laws it holds, its constants) on the three
targets' readings, then each new law posed as the climb poses it (spcore_r.Root and Acceptor, the answers by
hooks.reading_target), on the data's rows, against programs whose function is known: right programs must pass the
law's probes and its acceptor at every verify width, wrong ones must fail. Writes logs/known_lib.json, the known
routines, for the routes' check on both ladders (bar3/scale/compose3/verify3.py).

    cd cpu_first && SUBLAW=1 CONST=1 bar3/scale/rowcheck/memcap.sh 8 ~/github/sotaku/venv/bin/python -m bar3.scale.reading.reading_check
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

assert os.environ.get("SUBLAW") and os.environ.get("CONST"), "the new patterns need SUBLAW or CONST on"
from integration.lib import libcore as L                               # noqa: E402
from bar3.scale.reading import hooks as H                              # noqa: E402
from bar3.scale.reading import spcore_r as S                           # noqa: E402
from bar3.scale.reading.core_r import digits, load_rows                # noqa: E402

READINGS = {   # the readings round 4's law reader made (SLICE.md §2.1), and what this study's source finds in them
    "x2": ("mod(mul(X,X),N)", {("mod", "mul"): 1}, {}),
    "x3": ("mod(mul(mod(mul(X,X),N),X),N)", {("mod", "mul"): 2}, {}),
    "10x2p1": ("mod(add(mul(mod(mul(X,X),N),10),1),N)", {("mod", "add"): 1, ("mod", "mul"): 1},
               {10: ["mul"], 1: ["add"]}),
}
LIB = {"T.lt.pair.AN": "CLC P[SBB.N INC]", "T.lt.pair.AX": "CLC P[SBB.X INC]",
       "T.red.pair.AN": "STA.S T.lt.pair.AN SEL.S", "T.red.pair.AX": "STA.S T.lt.pair.AX SEL.S",
       "T.red.Bx+d.AN": "B[T.red.pair.AN]",
       "T.acc.mul.o,x,d.AXY": "LDH.Y CLK P[MUL.X INC]", "T.mul.x,o.XY": "CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]",
       "T.red.x+on.XN": "LDA.N P[BRG.X T.red.Bx+d.AN]", "T.add.pair.AX": "CLC P[ADC.X INC]",
       "T.const.1.n.N": "CLA STA.S T.lt.pair.AN CLA ADC.S",
       "T.mod.mul.x,o,n.XYN": "T.mul.x,o.XY STA.X T.red.x+on.XN",
       "T.add.1.x,n.AN": "STA.X T.const.1.n.N T.add.pair.AX"}
CASES = [   # (law, pattern, binding, program, should pass)
    ("lt", "pair", "AX", "CLC P[SBB.X INC]", True),                              # the comparison: a - b's borrow
    ("lt", "pair", "AX", "CLC P[ADC.X INC]", False),
    ("red", "pair", "AX", "STA.S T.lt.pair.AX SEL.S", True),                      # the link: one conditional subtraction
    ("red", "pair", "AX", "STA.S CLC P[SBB.X INC] SEL.S", True),                  #   over the primitives
    ("red", "pair", "AX", "T.lt.pair.AX", False),                                 #   the subtraction, unselected
    ("red", "pair", "AX", "CLA", False),
    ("red", "Bx+d", "AN", "B[T.red.pair.AN]", True),                             # the step over the link (round 4's cut)
    ("mod.mul", "x,o,n", "XYN", "T.mul.x,o.XY STA.X T.red.x+on.XN", True),        # the sub-law: (x o) mod N
    ("mod.mul", "x,o,n", "XYN", "T.mul.x,o.XY", False),                           #   unreduced
    ("mod.mul", "x,o,n", "XYN", "T.mul.x,o.XY STA.X CLA P[BRG.X T.red.pair.AN]", False),   # one subtraction a digit
    ("const.1", "n", "N", "CLA STA.S T.lt.pair.AN CLA ADC.S", True),              # 1: the borrow of 0 - N, added to 0
    ("const.1", "n", "N", "CLA STA.S CLC P[SBB.N INC] CLA ADC.S", True),          #   over the primitives
    ("const.1", "n", "N", "CLA ADC.S", False),                                    #   a dirty S and carry
    ("const.1", "n", "N", "CLA", False),
    ("const.10", "n", "N", "T.const.1.n.N SHL.A", True),
    ("add.1", "x,n", "AN", "STA.X T.const.1.n.N T.add.pair.AX", True),            # x + 1
    ("add.1", "x,n", "AN", "T.add.pair.AX", False),
    ("add.1", "x,n", "AN", "STA.X T.const.1.n.N", False),
    ("mul.10", "x,n", "AN", "SHL.A", True),                                       # x 10, at base 10
    ("mul.10", "x,n", "AN", "CLA", False),
]


def main():
    bad = 0
    for tf, (law, subs, consts) in READINGS.items():
        t = H._tree(law)
        got_s, got_c = H.sublaws(t), H.constant_ops(t)
        res = "ok" if got_s == subs and got_c == consts else "WRONG"
        bad += res != "ok"
        print(f"reading {tf}: {law}: sub-laws {got_s}, constants {got_c}: {res}", flush=True)
    dev = "cpu"
    rows = load_rows(4096, seed=0)
    Nd, Xd = digits(rows, "modulus").to(dev), digits(rows, "x").to(dev)
    settings = [L.Setting(10, w, L.Cells(10, dev)) for w in (1, 2)]
    verify = [L.Setting(10, w, L.Cells(10, dev)) for w in (1, 2, 3)]
    lib = {k: L.parse(v) for k, v in LIB.items()}
    for law, pat, *_ in CASES:
        H.SLICE_PAIRS[(law, pat)] = "check"
        if (law, pat) != ("red", "Bx+d"):
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
