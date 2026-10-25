"""The searches behind CARRY.md §12's tiny check, on Modal CPU containers through the shared cache:
1. multiply-accumulate's subgoal (the widened conjunction `0<C & R+C<2`, dsubgoal.py) given the deeper probe (10M
   states, an hour) with clean_win, under hardest first's and the pairs signal's round-1 libraries (their round-0
   records' libraries, records/carry/);
2. with the clean counting piece a routine, multiply-accumulate's target searched at the usual budget (3M) and at the
   deeper one (10M), with hardest first's round-2 library;
3. the loop's count, C := C + R·H on the target's rows, as a problem, at the probes' budgets, same library.

    cd cpu_first && BRIDGE_APP=bar3-bridge-subgoal PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.subgoal_jobs

Writes records/carry/subgoal_jobs.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from bar3.bridge import dlearner as L, dproblems as P, dwiden as W, modal_bridge as MB

HERE = Path(__file__).resolve().parent.parent
DEEP = L.PROBE_DEEP
COUNT = "INC.A T.lawmod(A,N)aqN Z.A[INC.C] DEC.C"


def lib_after(run, rnd):
    recs = [json.loads(line) for line in (HERE / "records" / "carry" / f"{run}.jsonl").read_text().splitlines()]
    return recs[rnd]["lib"]


def jobs():
    posed, _, gen = L.the_book()
    by = {s.name: s for s in posed}
    mac = P.from_slice(by["acc.mul(A,R,H)"], "target")
    piece = W.widen({c.name: c for c in W.conjunctions(P.restrictions(mac))}["restrict/acc.mul(A,R,H)/0<C&R+C<2"])
    lp1, ct1, lp2 = lib_after("lp_mul_k1", 0), lib_after("ct2_mul_k1", 0), lib_after("lp_mul_k1", 1)
    out = []
    for run, lib in (("lp_mul", lp1), ("ct2_mul", ct1)):
        out.append((f"the piece, {run}'s round-1 library, deeper probe with clean_win",
                    dict(L.job(piece, lib, DEEP, "probe_deep"), clean_win=True)))
    lib = dict(lp2, **{"T.cnt": COUNT})
    tgt_gen = P.from_slice(gen["acc.mul(A,R,H)"], "target")
    for label, budget in (("3M", L.FULL), ("10M", DEEP)):
        out.append((f"acc.mul, lp_mul's round-2 library + the clean piece, target search {label}",
                    L.job(mac, lib, budget, "target", general=tgt_gen)))
    rows = {B: {k: (k[2] + k[1] * k[3],) for k in r} for B, r in mac.rows.items()}
    count = P.Cand("count/C:=(R·H)+C", "count", "C + R·H", mac.inputs, ("C",), rows, parent=mac.name)
    for kind in ("probe_a", "probe_b"):
        out.append((f"the count C := C + R·H, lp_mul's round-2 library, {kind}", L.job(count, lp2, L.BIG[kind], kind)))
    return out


def main():
    todo = jobs()
    with MB.app.run():
        recs = list(MB.search_remote.map([j for _, j in todo], return_exceptions=True, wrap_returned_exceptions=False))
    out = []
    for (label, j), r in zip(todo, recs):
        if not isinstance(r, dict):
            print(f"{label}: failed {r!r}"[:300])
            continue
        progs = [p["program"] + (" (general)" if p.get("general") else "") for p in r.get("programs", [])]
        print(f"{label}: found {r['found_cost']} levels {r['level_sizes']} {r['wall']:.0f} s {r['stop']}; "
              f"clean {r.get('clean')}; dirty first {r.get('dirty_first')}")
        for p in progs[:6]:
            print(f"      {p}")
        out.append(dict(label=label, lib=j["lib"], record=r))
    dest = HERE / "records" / "carry" / "subgoal_jobs.json"
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
