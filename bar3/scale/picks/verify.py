"""PICKS.md §5: the runs' outcomes, read from the records (records/*_pk1.jsonl), each program rechecked. For each run:
whether and in which round its target came, its program, its size in primitive instructions, and its check.
- A held-out, two-source or breadth target is checked on every pair x < N <= 120 (the compiled engine, each pair
  alone), on machine_for(120) and with 10^9 steps, so that a slow program is told from a wrong one.
- A digits target is checked on its rows at both sizes, at big x, and on 1,000 x below 10^9 at N <= 10
  (bar3/scale/digits/verify.py's checks).
Then, against the earlier run without the change: the rounds whose picks, library and target search are the same,
and what the change did in each round (the record's `picks` field). No search runs.

    cd cpu_first && <python with numba> -m bar3.scale.picks.verify [records/*_pk1.jsonl ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EARLIER = {("lp_clean_deep_sub_ns", "tenrho"): "newton/records/lp_clean_deep_sub_ns_tenrho_nw1.jsonl",
           ("lp_clean_deep_sub_ns", "tri"): "newton/records/lp_clean_deep_sub_ns_tri_nw1.jsonl",
           ("lp_clean_b_src", "dxsq"): "digits/records/lp_clean_b_src_dxsq_dg1.jsonl",
           ("ct2a2_clean_b_src", "dxsq"): "digits/records/ct2a2_clean_b_src_dxsq_dg1.jsonl",
           ("lp_clean_b_src", "dxmod"): "digits/records/lp_clean_b_src_dxmod_dg1.jsonl",
           ("ct2a2_clean_b_src", "dxmod"): "digits/records/ct2a2_clean_b_src_dxmod_dg1.jsonl"}
DIGITS = ("dxmod", "dxsq", "dpow2")


def read(path):
    return [r for r in (json.loads(line) for line in open(path) if line.strip()) if "round" in r]


def decided(rec):
    tr = rec["target"] or {}
    return (rec["picked"], rec["lib"], tr.get("found_cost"), [p["program"] for p in tr.get("programs", [])],
            (rec.get("general") or {}).get("program"))


def check(g, tname):
    from library import dsl
    lib = {t: dsl.parse(v) for t, v in g["lib"].items()}
    prog = dsl.parse(g["general"]["program"])
    prim = dsl.cost(dsl.expand(prog, lib))
    if tname in DIGITS:
        from library.inventing import brm
        from bar3.scale.digits import targets as DT
        from bar3.scale.digits import verify as DV      # installs the digit block
        rows = [DV.right(prog, lib, DT.rows(tname, size), DT.machine(size)) for size in (7, 10)]
        longer = DV.right(prog, lib, DV.wide(tname, 10), brm.Machine(cap=10 ** 9, limit=DT.BIG_STEPS))
        bigger = DV.right(prog, lib, DV.wide(tname, 40), brm.Machine(cap=10 ** 9, limit=10 ** 7))
        return (f"{prim} primitive; rows {rows[0]} and {rows[1]}, big x {DT.bigx_share(prog, lib, tname):.0%}; "
                f"x < 10^9 at N <= 10 {longer}, at N <= 40 {bigger}")
    from library.inventing import brm
    from library.inventing.jobs import machine_for
    from bar3.scale.newton.verify import check as on_pairs
    return (f"{prim} primitive; every pair x < N <= 120: {on_pairs(prog, lib, tname, machine_for(120))} on "
            f"machine_for(120), {on_pairs(prog, lib, tname, brm.Machine(cap=10 ** 9, limit=10 ** 9))} with 10^9 steps")


def main():
    from bar3.scale.heldout import targets as _ho  # noqa: F401  (registers the held-out targets)
    paths = sys.argv[1:] or sorted(str(p) for p in (HERE / "records").glob("*_pk1.jsonl"))
    for path in paths:
        recs = read(path)
        tname = recs[0]["target_name"]
        config = Path(path).name.removesuffix("_pk1.jsonl").removesuffix(f"_{tname}")
        base = next(b for (b, t) in EARLIER if t == tname and config.startswith(b + "_"))
        old = read(HERE.parent / EARLIER[(base, tname)])
        same = [r["round"] for r, o in zip(recs, old) if decided(r) == decided(o)]
        g = next((r for r in recs if r.get("general")), None)
        came = next((o["round"] for o in old if o.get("general")), None)
        print(f"{Path(path).name}: {len(recs)} rounds; the earlier run ({base}): "
              + (f"came in round {came}" if came is not None else "did not come")
              + f"; rounds with its picks, library and target search: {same}; "
              + (f"came in round {g['round']}: {g['general']['program']}; {check(g, tname)}" if g else "did not come"),
              flush=True)
        for r in recs:
            if (r.get("picks") or {}).get("twin") or (r.get("picks") or {}).get("rest"):
                print(f"  round {r['round']}: {r['picks']}; picked {r['picked']}", flush=True)


if __name__ == "__main__":
    main()
