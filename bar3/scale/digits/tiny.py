"""The digits study's tiny local tests (DIGITS.md §3), no Modal:

    cd cpu_first && <python with numba> -m bar3.scale.digits.tiny table      # the pose rule's truth table, seconds
    cd cpu_first && <python with numba> -m bar3.scale.digits.tiny programs   # the digit block and the budget, a minute
    cd cpu_first && <python with numba> -m bar3.scale.digits.tiny pose       # what the source poses, seconds
    cd cpu_first && <python with numba> -m bar3.scale.digits.tiny probe      # the probe on the posed steps, minutes

`table` and `programs` run no search; they fixed the design and the budget before PREREG.md. `pose` and `probe` came
after it.
"""

from __future__ import annotations

import sys

import numpy as np

from library import dsl
from library.inventing import brm
from library.inventing.jobs import machine_for
from bar3.scale.digits import machine as M
from bar3.scale.digits import source as SRC
from bar3.scale.digits import targets as DT

MOD = "MOV.A.X L.N[MOV.T.A DEC.A] Z.T[MOV.A.X]"          # "subtract N if you can", from X into A (bar 3's link)
LIB = {"T.mod": MOD}
TEN = " ".join(["INC.X"] * 10)
DEC_STEP = f"MOV.T.A L.T[{TEN}] T.mod L.A[MOV.X.A T.mod]"   # X = d + 10·A, then reduce fully into A
SQUARE = "MOV.X.A CLR.A L.X[L.X[INC.A]] L.A[MOV.X.A T.mod]"  # A <- A² mod N
PROGRAMS = {                                              # hand-written, to show the block and size the budget
    "x mod N": {
        "counting (L.X)": "L.X[INC.A MOV.X.A T.mod]",
        "unary digits (D.1)": "D.1[INC.A MOV.X.A T.mod]",
        "binary digits (D.2)": "D.2[L.A[INC.X] L.A[INC.X] T.mod]",
        "decimal digits (D.10)": f"D.10[{DEC_STEP}]",
    },
    "x² mod N": {
        "counting (L.X)": f"L.X[INC.A MOV.X.A T.mod] {SQUARE}",
        "binary digits (D.2)": f"D.2[L.A[INC.X] L.A[INC.X] T.mod] {SQUARE}",
        "decimal digits (D.10)": f"D.10[{DEC_STEP}] {SQUARE}",
    },
    "2^x mod N": {
        "counting (L.X)": "INC.A L.X[MOV.X.A L.A[INC.X] T.mod] MOV.X.A T.mod",
        "unary digits (D.1)": "INC.A D.1[MOV.X.A L.A[INC.X] T.mod] MOV.X.A T.mod",
        "binary digits (D.2)": f"INC.A D.2[MOV.U.X {SQUARE} L.U[MOV.X.A L.A[INC.X] T.mod]] MOV.X.A T.mod",
    },
}
TARGET_OF = {"x mod N": "dxmod", "x² mod N": "dxsq", "2^x mod N": "dpow2"}


def table():
    rows = SRC.truth_table()
    print("Is the answer its own state over digits? (rows (answer(x), d, N) -> answer(x·B + d) a function)")
    print("learner = on the learner's own rows (x < 70 at N <= 7 and x < 100 at N <= 10); theory = every x < 2,000 at"
          " N <= 7, as LARGE_NUMBERS.md §5.2 checked")
    print(f"{'target':30s} {'§5.2':5s} | {'unary':16s} | {'base 2':16s} | {'base 10':16s}")
    bad = 0
    for r in rows:
        cells = []
        for B in (1, 2, 10):
            v = r[B]
            cells.append(f"learner {'yes' if v['learner'] else 'no '} theory {'yes' if v['theory_rows'] else 'no '}")
            bad += (v["learner"] != r["theory"]) + (v["theory_rows"] != r["theory"])
        print(f"{r['target']:30s} {'yes' if r['theory'] else 'no':5s} | " + " | ".join(cells))
    print(f"disagreements with §5.2: {bad}")


def programs():
    lib = {t: dsl.parse(v) for t, v in LIB.items()}
    print(f"The check at big x: N <= 10, 20 values of x with 6–8 digits per N; cap 10⁹; budget {DT.BIG_STEPS:,} steps."
          f" The rows at k = 7 and k = 10 on the target's machines ({DT.LIMITS[7]:,} and {DT.LIMITS[10]:,} steps)")
    for tgt, progs in PROGRAMS.items():
        name = TARGET_OF[tgt]
        print(f"\n{tgt}")
        for label, src in progs.items():
            prog = dsl.parse(src)
            right = []
            for size in (7, 10):
                xs, Ns, want = DT.rows(name, size)
                got, dead = brm.evaluate(prog, lib, xs, Ns, DT.machine(size))
                old = brm.evaluate(prog, lib, xs, Ns, machine_for(size))[1]
                right.append(("dead" if dead else f"{int((got == want).sum())}/{len(want)}")
                             + (" (dead at machine_for's steps)" if old else ""))
            share = DT.bigx_share(prog, lib, name)
            bx, bN, bw = DT.bigx(name)
            steps = []
            for x, n, w in zip(bx.tolist(), bN.tolist(), bw.tolist()):
                s, a = DT.steps_on(prog, lib, x, n, limit=10 ** 7)
                steps.append(s)
                if s is not None:
                    assert a == w, (tgt, label, x, n)
            fit = [s for s in steps if s is not None]
            at = {}
            for x in (10 ** 3, 10 ** 4, 10 ** 5):
                at[x] = DT.steps_on(prog, lib, x, 7, limit=10 ** 8)[0]
            big = (f"max {max(fit):,} steps" if len(fit) == len(steps) else
                   f"{len(steps) - len(fit)} of {len(steps)} over 10⁷ steps")
            print(f"  {label:22s} rows k=7 {right[0]:8s} k=10 {right[1]:10s} | big x: {share:.0%} right within the"
                  f" budget; {big} | steps at x = 10³, 10⁴, 10⁵ (N = 7): "
                  + ", ".join("–" if v is None else f"{v:,}" for v in at.values()))
            print(f"      {src}")


def pose():
    for B in (1, 2, 10):
        for name in DT.NAMES:
            gen = DT.gen_for(name)
            got = SRC.digit_steps(gen, B, 7, 10, lib=LIB)
            print(f"base {B:2d} {name:7s}: {len(got)} problems " + ", ".join(c.name for c in got))
            for c in got[:1]:
                print(f"    e.g. {c.name}: {len(c.xs)} instances at k = 7, {len(c.cx)} at k = 10; outs {c.outs};"
                      f" init {sorted(c.init)}")


XMOD_ROWS = "MOV.A.X L.A[MOV.X.A T.mod]"                   # x mod N on the rows, by repeated subtraction


def probe():
    """The learner's two probes (30k then 1M states, the frozen budgets) on each posed step, with the library the
    learner has after round 0 in every held-out run: "subtract N if you can". For x² mod N, also x mod N on the rows
    as a routine, so that the source poses its step. Then, for each step found, the digit loop over it: right on the
    target's rows and at big x?"""
    from bar3.scale import learn as L          # its job() carries a shaped problem's registers (sources.RCand)
    from bar3.scale.digits import jobs as DJ
    from bar3.scale.subgoal.configs import CONFIGS
    B_ = CONFIGS["lp_clean_deep_sub"]["budgets"]
    for B in (1, 2, 10):
        kinds = list(brm.LOOP_KINDS) + [M.KINDS[B]]
        for name in DT.NAMES:
            lib = dict(LIB, **({"T.xm": XMOD_ROWS} if name == "dxsq" else {}))
            posed = [c for c in SRC.digit_steps(DT.gen_for(name), B, 7, 10, lib=lib) if c.name.endswith("/A.X")]
            for c in posed:
                out = None
                for kind in ("probe_a", "probe_b"):
                    r = DJ.search_job(L.job(c, lib, kinds, B_[kind], kind, 7, 10))
                    out = r
                    if r["found_cost"] is not None:
                        break
                best = out["programs"][0]["program"] if out.get("programs") else None
                line = (f"base {B:2d} {name:6s} {c.name:24s} {kind}: cost {out['found_cost']}, levels"
                        f" {out['level_sizes']}, {out['seconds']:.0f}s, stop '{out['stop']}'")
                if best:
                    pl = {t: dsl.parse(v) for t, v in dict(lib, **{"T.step": best}).items()}
                    tprog = dsl.parse(f"D.{B}[T.step]" if name == "dxmod" else
                                      f"INC.A D.{B}[T.step] MOV.X.A T.mod" if name == "dpow2" else
                                      f"D.{B}[T.step] {SQUARE}")
                    xs, Ns, want = DT.rows(name, 10)
                    got, dead = M.evaluate(tprog, pl, xs, Ns, DT.machine(10))
                    line += (f"\n      step: {best}\n      {dsl.show(tprog)}: rows k=10 "
                             f"{'dead' if dead else int((got == want).sum())}/{len(want)}, big x "
                             f"{DT.bigx_share(tprog, pl, name):.0%}")
                print(line, flush=True)


if __name__ == "__main__":
    {"table": table, "programs": programs, "pose": pose, "probe": probe}[sys.argv[1]]()
