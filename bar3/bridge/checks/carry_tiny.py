"""The tiny check of CARRY.md's rule, local and in seconds: does widening the pins tell the counting add piece from the
clearing one, and do add-with-carry and multiply-accumulate follow from the counting one?

1. the pieces the digit link allows (T.mod = A mod N for A < 2N, the learner's own round-0 program for law/mod(A,N)/a+qN),
   run on the restriction add where R < C and on its widening;
2. the restriction searched with T.mod (a 30,000-state search, as the learner's first probe): its cheapest programs,
   each run on the widening;
3. the compositions with the counting piece as a routine, on the target's rows at bases 2 to 16, compiled at base 10
   against stage 3's tables (lift.py);
4. how many candidates of each target's round-0 pool pin an input, and so are widened.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_tiny
"""

from __future__ import annotations

import time

import numpy as np

from library import dsl
from bar3.bridge import dcm, djobs, dlearner as L, dproblems as P, dwiden as W, lift, slices as S

MOD = "MOV.T.N L.A[DEC.T] Z.T[L.N[DEC.A]]"
CLEAR = "INC.A T.mod L.A[DEC.C]"
COUNT = "INC.A T.mod Z.A[INC.C] DEC.C"


def right(prog, lib, c, bases):
    """(rows right, rows) of a program on a candidate's rows at these bases."""
    ins, want, _ = P.table(c, bases)
    regs, dead = dcm.evaluate(dsl.parse(prog), {k: dsl.parse(v) for k, v in lib.items()}, ins,
                              P.machine_for(bases))
    if dead:
        return 0, len(want)
    got = regs[:, [dcm.RI[o] for o in c.outputs]]
    return int((got == want).all(1).sum()), len(want)


def outputs(prog, lib, c, bases=S.SEARCH_BASES + S.CHECK_BASES):
    """A program's output registers on every row of c at these bases."""
    ins, _, _ = P.table(c, bases)
    regs, _ = dcm.evaluate(dsl.parse(prog), {k: dsl.parse(v) for k, v in lib.items()}, ins, P.machine_for(bases))
    return regs[:, [dcm.RI[o] for o in c.outputs]]


def show(label, prog, lib, c, bases=S.SEARCH_BASES + S.CHECK_BASES):
    k, n = right(prog, lib, c, bases)
    print(f"  {label:34s} {prog:44s} {k:5d} / {n:5d}")
    return k == n


def main():
    posed, _, gen = L.the_book()
    by = {s.name: s for s in posed}
    add = P.from_slice(by["add(A,R)"], "target")
    rs = {c.name: c for c in P.restrictions(add)}
    piece = rs["restrict/add(A,R)/R<C"]
    wide = W.widen(piece)
    lib = {"T.mod": MOD}
    print(f"add where R < C: {piece.n(S.SEARCH_BASES)} rows at bases 2-7, pinned {W.pinned(piece)}; "
          f"its widening {wide.n(S.SEARCH_BASES)} rows")
    print("1. the pieces (rows right / rows, bases 2-10)")
    for label, prog in (("clearing, on the restriction", CLEAR), ("clearing, on the widening", CLEAR),
                        ("counting, on the restriction", COUNT), ("counting, on the widening", COUNT)):
        show(label, prog, lib, piece if "restriction" in label else wide)

    print("2. the restriction searched with T.mod (30,000 states), its cheapest programs on the widening")
    t0 = time.time()
    r = djobs.search_job(L.job(piece, lib, L.PROBE_A, "probe_a"))
    print(f"  found cost {r['found_cost']}, levels {r['level_sizes']}, {time.time() - t0:.0f} s")
    progs = [p["program"] for p in r.get("programs", [])]
    ok = [p for p in progs if right(p, lib, wide, S.SEARCH_BASES + S.CHECK_BASES)[0]
          == wide.n(S.SEARCH_BASES) + wide.n(S.CHECK_BASES)]
    print(f"  {len(progs)} programs of cost {r['found_cost']}; right on the widening: {len(ok)}; "
          f"the learner keeps {L.choose(progs, lib, piece)[0] if progs else None}")
    for p in progs[:6]:
        show("", p, lib, wide)
    beh = {}
    for p in progs:
        beh.setdefault(outputs(p, lib, wide).tobytes(), []).append(p)
    print(f"  distinct behaviours on the widening among them: {len(beh)} "
          f"({', '.join(str(len(v)) for v in beh.values())} programs each); the counting piece costs 5")

    print("3. compositions with the counting piece (T.cnt), every row at bases 2-16, and at base 10 against s3i")
    lib2 = {"T.mod": MOD, "T.cnt": COUNT}
    for name, prog in (("add(A,R)", "L.R[INC.C] L.C[T.cnt]"), ("acc.mul(A,R,H)", "L.H[L.R[INC.C]] L.C[T.cnt]")):
        tgt = P.from_slice(gen[name], "target")
        show(f"{name} ({dsl.cost(dsl.parse(prog))} instructions)", prog, lib2, tgt, S.GENERAL_BASES)
        rep = lift.report(name, prog, lib2)
        print(f"    base 10: {rep['rows']} rows, place loop {rep['lift']:.3f}, s3i's table {rep['s3i']}")
    for name, prog in (("add(A,R)", "L.R[INC.C] L.C[T.clr]"), ("acc.mul(A,R,H)", "L.H[L.R[INC.C]] L.C[T.clr]")):
        tgt = P.from_slice(gen[name], "target")
        show(f"{name}, the clearing piece", prog, {"T.mod": MOD, "T.clr": CLEAR}, tgt, S.GENERAL_BASES)

    print("5. multiply-accumulate's own pool: the conjunction that pins R = 0 and C = 1, widened (rows right / rows)")
    mac = P.from_slice(by["acc.mul(A,R,H)"], "target")
    conj = {c.name: c for c in W.conjunctions(P.restrictions(mac))}
    cp = conj["restrict/acc.mul(A,R,H)/0<C&R+C<2"]
    cw = W.widen(cp)
    print(f"  {len(conj)} conjunctions pin more than either part; {cp.name} pins {W.pinned(cp)}, "
          f"{cp.n(S.SEARCH_BASES)} rows at bases 2-7, widened {cw.n(S.SEARCH_BASES)}")
    for label, prog, c in (("clearing, on the widened conjunction", CLEAR, cw), ("counting, on it", COUNT, cw)):
        show(label, prog, lib, c)
    rmac = {c.name: c for c in P.restrictions(mac)}["restrict/acc.mul(A,R,H)/R<1"]
    lib3 = {"T.mod": MOD, "T.cnt": COUNT, "T.addC": "L.C[T.cnt]"}
    show("add C into A (its R < 1)", "L.C[T.cnt]", lib3, rmac)
    tgt = P.from_slice(gen["acc.mul(A,R,H)"], "target")
    show("acc.mul(A,R,H), 4 instructions", "L.H[L.R[INC.C]] T.addC", lib3, tgt, S.GENERAL_BASES)

    print("4. candidates that pin an input, in each target's round-0 pool")
    for t, sname in L.TARGETS.items():
        tgt = P.from_slice(by[sname], "target")
        pool = P.dedupe(P.digit_laws(tgt) + [P.from_slice(s, "slice") for s in posed if s.name != sname]
                        + P.restrictions(tgt) + P.projections(tgt))
        ws = W.widenings(pool)
        print(f"  {t}: pool {len(pool)} (without near misses), widened {len(ws)}: "
              f"{sorted(set(tuple(sorted(W.pinned(c))) for c in pool if W.pinned(c)))}")


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    main()
