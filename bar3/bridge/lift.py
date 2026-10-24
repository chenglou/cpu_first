"""From an invented counting program to a digit instruction: compile it to a table, and check that the place loop of
that table computes its law on whole numbers.

- compile_cell: run the program on every input of the place at base B (A, R digits; C every state the slice uses;
  H a digit) and keep its outputs: the table a digit machine looks up, the tensor form of stage 3's cells
  (integration/lib/libcore.py Cells: rows over (a, x, c, h, k) -> (digit, flag or carry)).
- lift: the digit machine's own loop over places with that table (the state of place 0 being the law's L(0, 0, d),
  then passed from place to place) on random whole numbers, against the law on those numbers.
- against_s3i: the table against stage 3's exact cells (libcore.Cells(10)) and SEL.S's per-place select.
"""

from __future__ import annotations

import itertools

import numpy as np

from library import dsl
from bar3.bridge import dcm, slices as S
from bar3.bridge.dproblems import machine_for


def domain(sl, B):
    """Every input of the place at base B, as rows (A, R, C, H)."""
    states = sorted({k[2] for b in sl.rows for k in sl.rows[b]}) if "C" in sl.inputs else [0]
    states = [s for s in states if s < max(2, B)] if sl.law != "acc.add" else list(range(B))
    rng = {"A": range(B), "R": range(B) if "R" in sl.inputs else [0], "C": states if "C" in sl.inputs else [0],
           "H": range(B) if "H" in sl.inputs else [0]}
    return list(itertools.product(rng["A"], rng["R"], rng["C"], rng["H"]))


def compile_cell(prog, lib, sl, B=10):
    """{(a, r, c, h): outputs} at base B, by running the program on every input of the place."""
    rows = domain(sl, B)
    ins = {r: np.array([row["ARCH".index(r)] for row in rows], np.int64) for r in sl.inputs if r != "N"}
    ins["N"] = np.full(len(rows), B, np.int64)
    regs, dead = dcm.evaluate(prog, lib, ins, machine_for((B,)))
    if dead:
        return None
    outs = regs[:, [dcm.RI[o] for o in sl.outputs]]
    return {row: tuple(int(v) for v in o) for row, o in zip(rows, outs)}


def lift(sl, table, B=10, W=4, n=3000, seed=1):
    """The place loop of the table, on n random operand tuples of W-digit numbers: (share right, an example wrong)."""
    kinds, fn, kind = S.LAWS[sl.law]
    rng = np.random.default_rng(seed)
    right, bad = 0, None
    for _ in range(n):
        ops = [int(rng.integers(0, B ** W)) if k == "num" else int(rng.integers(0, B)) if k == "dig"
               else int(rng.integers(0, 2)) for k in kinds]
        v = fn(*ops)
        lo0 = fn(*[0 if k == "num" else o for o, k in zip(ops, kinds)])
        state = (lo0 if kind == "pred" else lo0) * sl.sign if sl.state else 0
        digits = []
        for i in range(W):
            ins = {r: (o // B ** i) % B if k == "num" else o for o, k, r in zip(ops, kinds, sl.binding)}
            if sl.state:
                ins["C"] = state
            key = tuple(ins.get(r, 0) for r in "ARCH")
            if key not in table:
                digits = None
                break
            out = dict(zip(sl.outputs, table[key]))
            if "A" in out:
                digits.append(out["A"])
            if sl.state:
                state = out["C"]
        if digits is None:
            continue
        if kind == "pred":
            ok = state == v
        else:
            got = sum(d * B ** i for i, d in enumerate(digits))
            ok = got == v % B ** W
        right += ok
        if not ok and bad is None:
            bad = dict(operands=ops, law=v, digits=digits, state=state)
    return right / n, bad


def against_s3i(sl, table, B=10):
    """How the table compares with stage 3's exact cells: rows equal / rows compared, per output."""
    a, r, c, h = (np.arange(B),) * 4
    out = {}
    if sl.name in ("sub(A,R)", "add(A,R)"):
        for (aa, rr, cc, hh), o in table.items():
            if cc > 1:
                continue
            d = aa - rr - cc if sl.law == "sub" else aa + rr + cc
            want = (d % B, int(d < 0) if sl.law == "sub" else int(d >= B))
            out.setdefault("rows", 0)
            out["rows"] += 1
            out["equal"] = out.get("equal", 0) + (o == want)
    elif sl.name == "acc.mul(A,R,H)":
        for (aa, rr, cc, hh), o in table.items():
            t = aa + rr * hh + cc
            out["rows"] = out.get("rows", 0) + 1
            out["equal"] = out.get("equal", 0) + (o == (t % B, t // B))
    elif sl.name == "select(C,A,R)":
        for (aa, rr, cc, hh), o in table.items():
            out["rows"] = out.get("rows", 0) + 1
            out["equal"] = out.get("equal", 0) + (o == ((rr if cc else aa),))
    return out


def report(name, prog, lib=None):
    """Compile one slice's program at base 10, lift it, compare it with s3i's cell."""
    posed, _ = S.book(S.SEARCH_BASES + S.CHECK_BASES)
    sl = next(s for s in posed if s.name == name)
    pl = {t: dsl.parse(v) for t, v in (lib or {}).items()}
    tab = compile_cell(dsl.parse(prog) if isinstance(prog, str) else prog, pl, sl)
    if tab is None:
        return dict(slice=name, compiled=False)
    share, bad = lift(sl, tab)
    return dict(slice=name, compiled=True, rows=len(tab), lift=share, lift_bad=bad, s3i=against_s3i(sl, tab))


if __name__ == "__main__":
    V = "Z.A[MOV.A.N INC.C] DEC.A"
    W = "INC.A MOV.T.N L.A[DEC.T] Z.T[CLR.A INC.C]"
    for nm, p, lb in (("sub(A,R)", "L.C[INC.R] CLR.C L.R[T.v]", {"T.v": V}),
                      ("add(A,R)", "L.C[INC.R] CLR.C L.R[T.w]", {"T.w": W}),
                      ("acc.mul(A,R,H)", "MOV.T.C CLR.C L.T[T.w] L.H[L.R[T.w]]", {"T.w": W}),
                      ("select(C,A,R)", "L.C[MOV.A.R]", {}),
                      ("lt(A,R)", "L.C[INC.R] CLR.C L.R[T.v]", {"T.v": V})):
        print(report(nm, p, lb))
