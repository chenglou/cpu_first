"""Do the invented add-with-carry and multiply-accumulate compute the place functions of the certified variant's ADC
and MUL cells (CARRY.md, end to end)? Each program, compiled to a table by running it on every row of the cell (ADC:
a, x and a carry of 0 or 1; MUL: a, x, h and a carry digit), against the variant's own `Cells` class (its source taken from submission_stage3_bridge.py, md5
asserted, run here without the rest of the file): at base 10 (H1's base) and at every base from 2 to 16. Also the
place loop of each table on random whole numbers (lift.lift).

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_tables records/carry/<run>.jsonl ...

A record's last line names the program it invented (`general`) and the library it was found with (`lib`). With no
records, the programs built on the hand-written counting piece are checked instead.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import itertools

import numpy as np
import torch

from library import dsl
from bar3.bridge import dcm, lift, slices as S
from bar3.bridge.dproblems import machine_for

HERE = Path(__file__).resolve().parent.parent
VARIANT = HERE / "submission_stage3_bridge.py"
MD5 = "85ff74c5faf902c1ede7a9f07d059026"
CELL = {"add(A,R)": ("adc_d", "adc_c"), "acc.mul(A,R,H)": ("mul_lo", "mul_hi")}


def variant_cells():
    src = VARIANT.read_text()
    assert hashlib.md5(src.encode()).hexdigest() == MD5, "the certified variant changed"
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Cells")
    ns = {"torch": torch, "CELLS": []}
    exec(compile(ast.Module([node], []), str(VARIANT), "exec"), ns)
    cls = ns["Cells"]
    cls._flat = lambda self: None          # the machine's joined lookup needs the rest of the file; the tables do not
    return cls


def table(prog, lib, name, B):
    """The program run on every row of the variant's cell at base B: {(a, x, carry, h): (digit, carry out)}, None if
    some row runs out of steps. ADC's rows are (a, x, carry 0 or 1), MUL's (a, x, h, carry digit)."""
    cs = range(2) if name == "add(A,R)" else range(B)
    hs = [0] if name == "add(A,R)" else range(B)
    rows = list(itertools.product(range(B), range(B), cs, hs))
    ins = {r: np.array([row[i] for row in rows], np.int64) for i, r in enumerate("ARCH")}
    ins["N"] = np.full(len(rows), B, np.int64)
    regs, dead = dcm.evaluate(prog, lib, ins, machine_for((B,)))
    return None if dead else {row: (int(o[0]), int(o[1])) for row, o in zip(rows, regs[:, [dcm.RI["A"], dcm.RI["C"]]])}


def check(name, prog, lib, Cells, by):
    sl = by[name]
    pl = {k: dsl.parse(v) for k, v in lib.items()}
    p = dsl.parse(prog)
    lo, hi = CELL[name]
    out = {}
    for B in S.GENERAL_BASES:
        tab = table(p, pl, name, B)
        cells = Cells(B)
        want_lo, want_hi = getattr(cells, lo), getattr(cells, hi)
        eq = 0
        for (a, r, c, h), got in (tab or {}).items():
            ix = (a, r, c) if name == "add(A,R)" else (a, r, h, c)
            w = (want_lo[ix], want_hi[ix])
            eq += got == (int(w[0]), int(w[1]))
        out[B] = (eq, len(tab or {}), want_lo.numel())
    share, bad = lift.lift(sl, table(p, pl, name, 10))
    ok = all(e == r == f for e, r, f in out.values())
    bad_b = {B: v for B, v in out.items() if not v[0] == v[1] == v[2]}
    print(f"{name}: `{prog}`  (primitive `{dsl.show(dsl.expand(p, pl))}`)")
    print(f"  base 10: {out[10][0]:,} of the variant's {out[10][2]:,} rows equal; bases 2-16 all equal: {ok}; "
          f"place loop on random numbers {share:.3f}" + (f", e.g. wrong {bad}" if bad else ""))
    if bad_b:
        print(f"  differing bases (equal, compiled rows, the variant's rows): {bad_b}")
    return ok and share == 1.0


def main(paths):
    posed, _ = S.book(S.SEARCH_BASES + S.CHECK_BASES)
    by = {s.name: s for s in posed}
    Cells = variant_cells()
    if not paths:
        lib = {"T.mod": "MOV.T.N L.A[DEC.T] Z.T[L.N[DEC.A]]", "T.cnt": "INC.A T.mod Z.A[INC.C] DEC.C"}
        todo = [("add(A,R)", "L.R[INC.C] L.C[T.cnt]", lib), ("acc.mul(A,R,H)", "L.H[L.R[INC.C]] L.C[T.cnt]", lib)]
    else:
        todo = []
        for p in paths:
            rec = json.loads(Path(p).read_text().splitlines()[-1])
            g = rec.get("general")
            if g is None:
                print(f"{p}: invented nothing")
                continue
            todo.append((rec["target"], g["program"], rec["lib"]))
    ok = all([check(name, prog, lib, Cells, by) for name, prog, lib in todo])
    print("all match" if ok else "SOME DIFFER")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
