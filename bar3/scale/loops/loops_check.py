"""The loops' check (LOOPS.md §3): each loop posed as the climb poses it (spcore_l.Root and Acceptor, the instances and
answers by hooks.xon_term and slice_target), on the data's rows, against programs whose function is known. The
multiply over the row and the reduction over Horner's step (and the addition) must pass their loops' acceptors at
every verify width; wrong programs must fail.

    cd cpu_first && bar3/scale/rowcheck/memcap.sh 8 ~/github/sotaku/venv/bin/python -m bar3.scale.loops.loops_check
"""

from __future__ import annotations

import torch

from integration.lib import libcore as L
from bar3.scale.loops import hooks as H
from bar3.scale.loops import spcore_l as S
from bar3.scale.loops.core_l import digits, load_rows

LIB = {"T.lt.pair.AN": "CLC P[SBB.N INC]", "T.lt.pair.AY": "CLC P[SBB.Y INC]",
       "T.red.pair.AN": "STA.S T.lt.pair.AN SEL.S", "T.red.pair.AY": "STA.S T.lt.pair.AY SEL.S",
       "T.red.Bx+d.AN": "B[T.red.pair.AN]", "T.red.Bx+d.AY": "B[T.red.pair.AY]",
       "T.acc.mul.o,x,d.AXY": "LDH.Y CLK P[MUL.X INC]"}
CASES = [   # (law, pattern, binding, program, should pass)
    ("mul", "x,o", "XY", "CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]", True),          # the row over the multiplier
    ("mul", "x,o", "XY", "CLA T.acc.mul.o,x,d.AXY", False),                      # its low digit only
    ("mul", "x,o", "XY", "W[T.acc.mul.o,x,d.AXY INC SHL.X]", False),             # A not cleared (a dirty start)
    ("red", "x+on", "XA", "STA.Y P[BRG.X T.red.Bx+d.AY]", True),                 # Horner over the step
    ("red", "x+on", "XN", "CLA P[BRG.X T.red.Bx+d.AN]", True),
    ("red", "x+on", "XN", "LDA.N P[BRG.X T.red.Bx+d.AN]", True),
    ("red", "x+on", "XN", "CLA P[BRG.X T.red.pair.AN]", False),                  # one subtraction a digit
    ("red", "x+on", "XN", "LDA.X", False),
    ("add", "pair", "AY", "CLC P[ADC.Y INC]", True),
    ("add", "pair", "AY", "P[ADC.Y INC]", False),
]


def main():
    dev = "cpu"
    rows = load_rows(4096, seed=0)
    Nd, Xd = digits(rows, "modulus").to(dev), digits(rows, "x").to(dev)
    settings = [L.Setting(10, w, L.Cells(10, dev)) for w in (1, 2)]
    verify = [L.Setting(10, w, L.Cells(10, dev)) for w in (1, 2, 3)]
    lib = {k: L.parse(v) for k, v in LIB.items()}
    for law, pat, *_ in CASES:
        H.SLICE_PAIRS[(law, pat)] = "check"
    ops = S.Ops()
    bad = 0
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
    print("ALL OK" if not bad else f"{bad} WRONG")


if __name__ == "__main__":
    main()
