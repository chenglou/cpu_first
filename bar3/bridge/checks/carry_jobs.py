"""The searches behind CARRY.md's tiny check, on Modal CPU containers (bar3/bridge/modal_bridge.py's search_remote,
through the shared cache): the widened add piece searched with the digit link, alone and with each rule's round-1
library (the library its round-0 record left), and the targets searched with the counting piece as a routine.

    cd cpu_first && BRIDGE_APP=bar3-bridge-carry ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_jobs

Writes records/carry/carry_jobs.json (every job's record, the libraries' programs dropped from the copy).
"""

from __future__ import annotations

import json
from pathlib import Path

from bar3.bridge import dlearner as L, dproblems as P, dwiden as W, modal_bridge as MB, report as R

HERE = Path(__file__).resolve().parent.parent
MOD = "MOV.T.N L.A[DEC.T] Z.T[L.N[DEC.A]]"
COUNT = "INC.A T.mod Z.A[INC.C] DEC.C"


def jobs():
    posed, _, gen = L.the_book()
    by = {s.name: s for s in posed}
    add = P.from_slice(by["add(A,R)"], "target")
    piece = {c.name: c for c in P.restrictions(add)}["restrict/add(A,R)/R<C"]
    wide = W.widen(piece)
    out = []
    for kind in ("probe_a", "probe_b"):
        out.append((f"widened piece, T.mod only, {kind}", L.job(wide, {"T.mod": MOD}, L.BIG[kind], kind)))
        for run in ("lp_adc", "ct2_adc", "all_adc"):
            lib = R.load(f"{run}_r1")[0]["lib"]
            out.append((f"widened piece, {run}'s round-1 library, {kind}", L.job(wide, lib, L.BIG[kind], kind)))
    for name in ("add(A,R)", "acc.mul(A,R,H)"):
        tgt = P.from_slice(by[name], "target")
        out.append((f"{name} with T.mod and the counting piece", L.job(
            tgt, {"T.mod": MOD, "T.cnt": COUNT}, L.FULL, "target", general=P.from_slice(gen[name], "target"))))
    return out


def main():
    todo = jobs()
    with MB.app.run():
        recs = list(MB.search_remote.map([j for _, j in todo], return_exceptions=True,
                                         wrap_returned_exceptions=False))
    out = []
    for (label, j), r in zip(todo, recs):
        if not isinstance(r, dict):
            print(f"{label}: failed {r!r}"[:300])
            continue
        progs = [p["program"] + (" (general)" if p.get("general") else "") for p in r.get("programs", [])]
        print(f"{label}: found {r['found_cost']} levels {r['level_sizes']} {r['wall']:.0f} s {r['stop']}")
        for p in progs[:6]:
            print(f"      {p}")
        out.append(dict(label=label, lib=j["lib"], record=r))
    dest = HERE / "records" / "carry" / "carry_jobs.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
