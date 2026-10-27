"""The cuts' check (SLICE.md §3): each cut posed as the climb poses it (spcore_s.Root and Acceptor, answers by
hooks.slice_target), on the data's rows, against programs whose function is known. The row, x d, x + d and Horner's
step (over the link routine) must pass their cuts' acceptors at every verify width; wrong programs must fail. Then the
law reader on the oracle arm's own target rows (the target's probes and acceptance rows, TARGET_FN's answers).

    cd cpu_first && bar3/scale/rowcheck/memcap.sh 8 ~/github/sotaku/venv/bin/python -m bar3.scale.slice.slice_check
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import torch

from integration.lib import libcore as L
from bar3.scale.slice import hooks as H
from bar3.scale.slice import spcore_s as S
from bar3.scale.slice.core_s import digits, load_rows

LIB = {"T.lt.pair.AN": "CLC P[SBB.N INC]", "T.lt.pair.AY": "CLC P[SBB.Y INC]",
       "T.red.pair.AN": "STA.S T.lt.pair.AN SEL.S", "T.red.pair.AY": "STA.S T.lt.pair.AY SEL.S"}
CASES = [   # (law, pattern, binding, program, should pass)
    ("acc.mul", "o,x,d", "AXY", "LDH.Y CLK P[MUL.X INC]", True),
    ("acc.mul", "o,x,d", "AYX", "LDH.X CLK P[MUL.Y INC]", True),
    ("acc.mul", "o,x,d", "AXY", "LDH.Y P[MUL.X INC]", False),          # the carry not cleared (a dirty start)
    ("acc.mul", "o,x,d", "AXY", "CLA LDH.Y CLK P[MUL.X INC]", False),  # x d, the accumulator dropped
    ("mul", "x,d", "XY", "CLA LDH.Y CLK P[MUL.X INC]", True),
    ("mul", "x,d", "XY", "LDH.Y CLK P[MUL.X INC]", False),             # adds A's dirty start
    ("add", "x,d", "XY", "LDA.X CLC P[ADC.Y INC]", True),
    ("add", "x,d", "XY", "LDA.X P[ADC.Y INC]", False),                 # the carry not cleared
    ("red", "Bx+d", "AN", "B[T.red.pair.AN]", True),
    ("red", "Bx+d", "XN", "LDA.X B[T.red.pair.AN]", True),
    ("red", "Bx+d", "XN", "LDA.X T.red.pair.AN", False),               # one conditional subtraction only
    ("red", "Bx+d", "XN", "LDA.X", False),
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
    for law, pat, bd, prog, want in CASES:
        root = S.Root(pat, tuple(bd), settings, 8, (Nd, Xd), ops, lib, dev, 0)
        g = root.goals(ops, lib, [law])
        acc = S.Acceptor(law, root, verify, (Nd, Xd), ops, lib, dev)
        p = L.parse(prog)
        one = torch.zeros(1, dtype=torch.long)
        runs = [L.apply_items(st, b.take(one), one, [("REP", L.ONE, tuple(p))], lib)
                for st, b in zip(settings, root.roots)]
        probe = bool(g) and bool(g[0].test(runs, lib)[0])
        ok = acc.ok and acc.many([p], lib)[0]
        res = "ok" if ok == want and (probe == want or not want) else "WRONG"
        bad += res != "ok"
        print(f"{law}/{pat}@{bd}: {prog!r}: probes {probe}, accepted {ok}, expected {want}: {res}", flush=True)
    # the law reader on the oracle arm's own target rows, for each target function
    from bar3.scale.slice import lawread
    for fn in ("x2", "x3", "10x2p1"):
        os.environ["TARGET_FN"] = fn
        H.TARGET_F = {"x2": lambda x, N: x * x % N, "x3": lambda x, N: pow(x, 3, N),
                      "10x2p1": lambda x, N: (10 * x * x + 1) % N}[fn]
        from bar3.scale.slice.oracle_s import SmallTarget
        t = SmallTarget()
        climb = SimpleNamespace(settings=settings, verify=verify, NP=8, device=dev, seed=0, target=t, records=[],
                                level=0, laws=["lt", "red", "add", "mul"], patterns=["pair", "x,o"])
        t0 = time.time()
        xs, Ns, ys, caps = H._target_rows(climb, (Nd, Xd))
        law, ops_, depth = lawread.read(xs, Ns, ys, caps, depth=H.SLICE_DEPTH)
        print(f"{fn}: {len(set(zip(xs, Ns)))} distinct rows; constants {lawread.constants(xs, Ns, ys, caps)}; "
              f"law {law} (depth {depth}), operations {ops_}; {time.time() - t0:.1f} s", flush=True)
    print("ALL OK" if not bad else f"{bad} WRONG")


if __name__ == "__main__":
    main()
