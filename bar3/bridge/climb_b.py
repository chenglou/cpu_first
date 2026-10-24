"""Stage 3's climb (s3i's libcore.Climb with s3i's settings) on another instruction set: the frame plus cells from a
list (libcore_b.set_instructions), the cells' tables compiled from counting programs (lift.compile_cell), on H1's rows,
the round searched against T = 1 answers whenever new routines appear (integration/lib/profile_climb.py's harness, with
s3i's CLIMB constants). Not a training run: the tables are exact here, so this asks only whether the climb finds the
round, and how fast, with these instructions.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.climb_b --set inv4 --seconds 600

Sets (SETS): s3i (libcore_b's defaults: s3i's own instructions, the control), and invented cells by slice name with
the program that computes them; `sel` keeps s3i's SEL.S (keep or discard on the borrow) in the frame.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from library import dsl
from bar3.bridge import lift, slices as S
from bar3.bridge import libcore_b as L

HERE = Path(__file__).resolve().parent
V = "Z.A[MOV.A.N INC.C] DEC.A"                          # the hand programs of lift.py's check (the same tables
W = "INC.A MOV.T.N L.A[DEC.T] Z.T[CLR.A INC.C]"         # as s3i's cells); runs replace them with invented ones
HAND = {"sub(A,R)": ("L.C[INC.R] CLR.C L.R[T.v]", {"T.v": V}),
        "add(A,R)": ("L.C[INC.R] CLR.C L.R[T.w]", {"T.w": W}),
        "acc.mul(A,R,H)": ("MOV.T.C CLR.C L.T[T.w] L.H[L.R[T.w]]", {"T.w": W}),
        "select(C,A,R)": ("L.C[MOV.A.R]", {}),
        "lt(A,R)": ("L.C[INC.R] CLR.C L.R[T.v]", {"T.v": V})}
def view_tables(B=10):
    """Every slice of the law book that passes the lift (lift.py), as a table at base 10 built from s3i's three exact
    tables and the select by moving their axes (a view, no new arithmetic), checked against the slice's own rows
    (read off the law, slices.py) before use. name -> (CellSpec reads, writes, table [rows, 2])."""
    import itertools
    from integration.lib.libcore import Cells as C0
    c = C0(B)
    sbb, mul = c.sbb.view(B, B, 2, 2), c.mul.view(B, B, B, B, 2)
    out = {}
    def tab(reads, f):
        rad = {"a": B, "x": B, "c": 2, "h": B, "k": B}
        return torch.tensor([list(f(**dict(zip(reads, v)))) for v in itertools.product(*[range(rad[r]) for r in reads])])
    out["sub(A,R)"] = ("axc", ("A", "c"), tab("axc", lambda a, x, c: sbb[a, x, c].tolist()))
    out["sub(R,A)"] = ("axc", ("A", "c"), tab("axc", lambda a, x, c: sbb[x, a, c].tolist()))
    out["lt(A,R)"] = ("axc", ("c",), tab("axc", lambda a, x, c: [0, int(sbb[a, x, c, 1])]))
    out["lt(R,A)"] = ("axc", ("c",), tab("axc", lambda a, x, c: [0, int(sbb[x, a, c, 1])]))
    out["add(A,R)"] = ("axc", ("A", "c"), tab("axc", lambda a, x, c: c_.adc.view(B, B, 2, 2)[a, x, c].tolist()))
    out["muld(A,H)"] = ("ahk", ("A", "k"), tab("ahk", lambda a, h, k: mul[0, a, h, k].tolist()))
    out["muld(R,H)"] = ("axhk", ("A", "k"), tab("axhk", lambda a, x, h, k: mul[0, x, h, k].tolist()))
    out["acc.mul(A,R,H)"] = ("axhk", ("A", "k"), tab("axhk", lambda a, x, h, k: mul[a, x, h, k].tolist()))
    out["acc.mul(R,A,H)"] = ("axhk", ("A", "k"), tab("axhk", lambda a, x, h, k: mul[x, a, h, k].tolist()))
    out["select(C,A,R)"] = ("axc", ("A",), tab("axc", lambda a, x, c: [x if c else a, 0]))
    out["select(C,R,A)"] = ("axc", ("A",), tab("axc", lambda a, x, c: [a if c else x, 0]))
    return out


c_ = None


def build_views(names):
    """CellSpecs for slices from view_tables, each checked against every row of its slice at base 10."""
    global c_
    from integration.lib.libcore import Cells as C0
    c_ = C0(10)
    posed, _ = S.book(S.SEARCH_BASES + S.CHECK_BASES)
    by = {s.name: s for s in posed}
    views = view_tables()
    out, info = [], []
    for name in names:
        reads, writes, t = views[name]
        sl = by[name]
        mult = []
        rad = {"a": 10, "x": 10, "c": 2, "h": 10, "k": 10}
        m = 1
        for r in reversed(reads):
            mult.insert(0, m)
            m *= rad[r]
        bad = 0
        for key, o in sl.rows[10].items():
            v = {"a": key[0], "x": key[1], "c": key[2], "h": key[3], "k": key[2]}
            row = t[sum(v[r] * mm for r, mm in zip(reads, mult))].tolist()
            got = ((row[0],) if "A" in sl.outputs else ()) + ((row[1],) if "C" in sl.outputs else ())
            bad += got != o
        assert bad == 0, (name, bad)
        cs = L.CellSpec(cell_name(sl), reads, writes, t)
        out.append(cs)
        info.append(dict(slice=name, cell=repr(cs), view=True, rows_checked=len(sl.rows[10])))
    return out, info


SETS = {
    "s3i": None,
    "lifted": dict(cells=["sub(A,R)", "sub(R,A)", "lt(A,R)", "lt(R,A)", "add(A,R)", "muld(A,H)", "muld(R,H)",
                          "acc.mul(A,R,H)", "acc.mul(R,A,H)", "select(C,A,R)", "select(C,R,A)"], sel=False,
                   views=True),
    "inv3_sel": dict(cells=["sub(A,R)", "add(A,R)", "acc.mul(A,R,H)"], sel=True),
    "inv4": dict(cells=["sub(A,R)", "add(A,R)", "acc.mul(A,R,H)", "select(C,A,R)"], sel=False),
}
CLIMB = dict(budget=5_800_000, fallback_budget=27_000_000, max_body=3, max_cost=7, min_count=2.0, max_merges=12,
             levels=12, NP=8, round_budget=4_000_000, round_cost=5, comp_first=True, comp_body=3, skip_twins=True,
             stall_after=2, search_s=900, listing="goals", occam_steps=True, merged_routines=True, round_retry=10,
             fuse=4)


def strong_modx():
    """modx's probes also with N < x' < 2N (a third of them: x' = x + N, the target still x). s3i's modx probes are
    x < N (keep x) and x = N (0) only, which a program that zeroes x = N and keeps everything else passes; with all 61
    lifted instructions the climb finds such a program (`T.lt P[BRG.X RSELC.X]`, 0 of 12 right on N < x < 2N), and the
    addition, checked by undoing it with modx, then has no solution. A diagnostic: x + N is computed here, digit by
    digit, as no training run may."""
    orig = L.build

    def build(task, st, n, gen, rows, helpers, lib, device):
        ins, outs, inv = orig(task, st, n, gen, rows, helpers, lib, device)
        if task.rstrip("~") != "modx":
            return ins, outs, inv
        xr = "Y" if task.endswith("~") else "X"
        X, N = ins[xr].long(), ins["N"].long()
        s, c = torch.zeros_like(X), torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        for j in range(X.shape[1]):
            t = X[:, j] + N[:, j] + c
            s[:, j], c = t % st.B, t // st.B
        use = (torch.arange(X.shape[0], device=X.device) % 3 == 1) & (c == 0) & ~(X == N).all(1)
        return dict(ins, **{xr: torch.where(use[:, None], s, X).to(ins[xr].dtype)}), outs, inv
    L.build = build


def cell_name(sl):
    law = sl.law.replace(".", "").upper()
    law = {"ACCMUL": "MAC", "SELECT": "SELC", "ACCADD": "ADD3"}.get(law, law)
    nums = [r for r in sl.binding if r in "AR"]
    return ("R" if nums and nums[0] == "R" else "") + law


def cell_spec(sl, table, B=10):
    """A slice's compiled table as libcore_b's CellSpec: reads in the order a, x, c, h, k (the state goes to c when it
    is a bit, to k when it is a digit), writes A[p] and the state's register, the table in that mixed radix."""
    states = sorted({k[2] for k in table}) if "C" in sl.inputs else []
    sreg = "c" if states and max(states) <= 1 else "k"
    reads = ["a"] + (["x"] if "R" in sl.inputs else []) + (["c"] if "C" in sl.inputs and sreg == "c" else []) + \
        (["h"] if "H" in sl.inputs else []) + (["k"] if "C" in sl.inputs and sreg == "k" else [])
    writes = (["A"] if "A" in sl.outputs else []) + ([sreg] if "C" in sl.outputs else [])
    rad = {"a": B, "x": B, "c": 2, "h": B, "k": B}
    import itertools
    rows = []
    for vals in itertools.product(*[range(rad[r]) for r in reads]):
        v = dict(zip(reads, vals))
        key = (v["a"], v.get("x", 0), v.get("c", v.get("k", 0)), v.get("h", 0))
        out = dict(zip(sl.outputs, table.get(key, (0,) * len(sl.outputs))))
        rows.append([out.get("A", 0), out.get("C", 0)])
    return L.CellSpec(cell_name(sl), reads, writes, torch.tensor(rows, dtype=torch.long))


def build(cells, progs=None):
    """CellSpecs for slice names, each compiled from its program (progs: name -> (program, lib); default HAND)."""
    posed, _ = S.book(S.SEARCH_BASES + S.CHECK_BASES)
    by = {s.name: s for s in posed}
    out, info = [], []
    for name in cells:
        prog, lib = (progs or HAND)[name]
        pl = {t: dsl.parse(v) for t, v in lib.items()}
        tab = lift.compile_cell(dsl.parse(prog), pl, by[name])
        share, bad = lift.lift(by[name], tab)
        cs = cell_spec(by[name], tab)
        out.append(cs)
        info.append(dict(slice=name, cell=repr(cs), program=prog, lib=lib, lift=share, s3i=lift.against_s3i(by[name], tab)))
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="inv4")
    ap.add_argument("--progs", default="", help="a JSON file {slice: [program, lib]} of invented programs")
    ap.add_argument("--seconds", type=float, default=3600)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--out", default="")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplies the climb's state budgets and its seconds per search (a bigger instruction set "
                         "needs more states for the same reach)")
    ap.add_argument("--modx-above", action="store_true", help="modx's probes also with N < x < 2N (strong_modx)")
    a = ap.parse_args()
    cfg = dict(CLIMB, **{k: int(CLIMB[k] * a.scale) for k in ("budget", "fallback_budget", "round_budget", "search_s")})
    if a.modx_above:
        strong_modx()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = SETS[a.set] if a.set in SETS else json.loads(Path(a.set).read_text())
    info = []
    if spec is not None:
        progs = {k: tuple(v) for k, v in json.loads(Path(a.progs).read_text()).items()} if a.progs else None
        cells, info = build_views(spec["cells"]) if spec.get("views") else build(spec["cells"], progs)
        L.set_instructions(cells, sel=spec.get("sel", False))
    L.MAC_ADDEND = "A"
    L.NEEDS.update(mul=("add", "mac"), reduce=("add", "modB"))
    from integration.lib import inclimb
    rows = inclimb.load_rows(a.rows, seed=0)
    Nd, Xd, _ = (t.to(dev) for t in inclimb.digits_of_rows(rows))
    ans = tuple(t.to(dev) for t in inclimb.digits_of_rows(inclimb.load_rows(400, seed=1, T=1)))
    climb = L.Climb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 18, **cfg)
    t0 = time.time()
    deadline = t0 + a.seconds
    print(f"device {torch.cuda.get_device_name() if dev == 'cuda' else 'cpu'}; set {a.set}: {len(L.PRIMS)} "
          f"instructions {L.PRIMS}", flush=True)
    for i in info:
        print("cell", i, flush=True)
    log = []
    while True:
        t1 = time.time()
        line = climb.step((Nd, Xd), answers=ans, deadline=deadline - 60)
        if line is None:
            break
        log.append(dict(t=round(time.time() - t0, 1), dt=round(time.time() - t1, 1), line=line))
        print(f"UNIT {time.time() - t0:7.1f}s +{time.time() - t1:6.1f}s {line[:220]}", flush=True)
        if climb.rounds or time.time() > deadline:
            break
    rec = dict(set=a.set, spec=spec, instructions=list(L.PRIMS), cells=info, device=dev, climb=cfg,
               modx_above=a.modx_above,
               seconds=round(time.time() - t0, 1), rounds=[L.show(p) for p in climb.rounds],
               rounds_primitive=[L.show(L.expand(p, climb.lib)) for p in climb.rounds], log=log,
               records=climb.records, history=climb.history, lib={k: L.show(v) for k, v in climb.lib.items()},
               solved_at=climb.solved_at)
    print("ROUNDS", rec["rounds"][:4], "after", rec["seconds"], "s", flush=True)
    out = Path(a.out) if a.out else HERE / "records" / f"climb_{Path(a.set).stem}_{dev}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
