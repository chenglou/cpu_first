"""Plumbing check: hand-written reference programs for every task run exactly at many settings.

    ~/github/sotaku/venv/bin/python -m library.selftest

These references are only for checking the machine and for counting how long each task's program is in primitives;
the search never sees them.
"""

from __future__ import annotations

from library import dsl
from library.search import check

RIPPLE = "CLC P[SBB.N INC]"
LINK = f"STA.S {RIPPLE} SEL.S"
ROW = "CLK P[MUL.X INC]"
MUL = f"CLA W[LDH.Y {ROW} SHL.X SHR.Y]"
REDUCE = f"CLA P[BRG.X B[{LINK}]]"

REFERENCE = {
    "copy": "LDA.X",
    "shl": "LDA.X SHL.A",
    "shr": "LDA.X SHR.A",
    "sub": f"LDA.X {RIPPLE}",
    "add": "LDA.X CLC P[ADC.N INC]",
    "lt": f"LDA.X {RIPPLE}",
    "neg": "CLA CLC P[SBB.X INC]",
    "dbl": "LDA.X CLC P[ADC.X INC]",
    "mrow": f"CLA LDH.Y {ROW}",
    "mac": f"LDA.N LDH.Y {ROW}",
    "csub": f"LDA.X {LINK}",
    "modB": f"LDA.X B[{LINK}]",
    "step": f"LDA.Y BRG.X B[{LINK}]",
    "reduce": REDUCE,
    "mul": MUL,
    "sq": f"LDA.X STA.Y {MUL}",
    "sqmod": f"LDA.X STA.Y {MUL} STA.X {REDUCE}",
    "subY": "LDA.X CLC P[SBB.Y INC]",
    "addY": "LDA.X CLC P[ADC.Y INC]",
    "ltY": "LDA.X CLC P[SBB.Y INC]",
    "negN": f"CLA {RIPPLE}",
    "mrowN": "CLA LDH.X CLK P[MUL.N INC]",
    "csubY": f"LDA.Y {LINK}",
    "modBY": f"LDA.Y B[{LINK}]",
    "step0": f"CLA BRG.X B[{LINK}]",
    "mulmod": f"{MUL} STA.X {REDUCE}",
}

SETTINGS = [dsl.Setting(B, W) for B, W in ((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (4, 2), (5, 2), (7, 2),
                                             (10, 2), (10, 3), (10, 4))]


def main():
    bad = 0
    for task, src in REFERENCE.items():
        prog = dsl.parse(src)
        assert dsl.show(prog) == src, (dsl.show(prog), src)
        r = check(task, prog, {}, SETTINGS, n=200)
        worst = min(r.values())
        bad += worst < 1
        print(f"{task:7s} cost {dsl.cost(prog):2d}  worst {worst:.3f}  {src}")
    h1 = dsl.Setting(10, 8)
    r = check("sqmod", dsl.parse(REFERENCE["sqmod"]), {}, [h1], n=100)
    print("sqmod at (10, 8):", r)
    bad += min(r.values()) < 1
    print("raw programs of cost 5 / 7 / 10 over 32 tokens and 3 loop kinds:",
          [f"{dsl.raw_count(32, 3, L, 3):.3g}" for L in (5, 7, 10)])
    print("ALL OK" if not bad else f"{bad} FAILED")


if __name__ == "__main__":
    main()
