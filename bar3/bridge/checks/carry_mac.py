"""Multiply-accumulate's own path under the pose rules (CARRY.md), step by step with the learners' real libraries, on
Modal CPU containers through the shared cache:
1. the conjunction `0<C & R+C<2` of its restrictions (R = 0, C = 1 pinned: the add piece inside its rows), widened,
   searched as the learner's probes would (30,000 then 1M states) with hardest first's and the pairs signal's round-1
   libraries (the libraries their round-0 records left);
2. its restriction R < 1 ("add C into A") with the counting piece added, at the probes' budgets;
3. the target with hardest first's round-2 library (its round-1 record's) plus the counting piece and `L.C[T.cnt]`,
   at the full budget (3M states, 2,400 s).

    cd cpu_first && BRIDGE_APP=bar3-bridge-carry PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_mac

Writes records/carry/carry_mac.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from bar3.bridge import dlearner as L, dproblems as P, dwiden as W, modal_bridge as MB

HERE = Path(__file__).resolve().parent.parent
COUNT = "INC.A T.lawmod(A,N)aqN Z.A[INC.C] DEC.C"


def lib_after(run, rnd):
    recs = [json.loads(line) for line in (HERE / "records" / "carry" / f"{run}.jsonl").read_text().splitlines()]
    return recs[rnd]["lib"]


def jobs():
    posed, _, gen = L.the_book()
    by = {s.name: s for s in posed}
    mac = P.from_slice(by["acc.mul(A,R,H)"], "target")
    conj = {c.name: c for c in W.conjunctions(P.restrictions(mac))}
    wide = W.widen(conj["restrict/acc.mul(A,R,H)/0<C&R+C<2"])
    addc = {c.name: c for c in P.restrictions(mac)}["restrict/acc.mul(A,R,H)/R<1"]
    lp1, ct1, lp2 = lib_after("lp_mul_k1", 0), lib_after("ct2_mul_k1", 0), lib_after("lp_mul_k1", 1)
    out = []
    for kind in ("probe_a", "probe_b"):
        out.append((f"widened conjunction, lp_mul's round-1 library, {kind}", L.job(wide, lp1, L.BIG[kind], kind)))
        out.append((f"widened conjunction, ct2_mul's round-1 library, {kind}", L.job(wide, ct1, L.BIG[kind], kind)))
        out.append((f"add C into A, lp_mul's round-1 library + the counting piece, {kind}",
                    L.job(addc, dict(lp1, **{"T.cnt": COUNT}), L.BIG[kind], kind)))
    lib = dict(lp2, **{"T.cnt": COUNT, "T.addC": "L.C[T.cnt]"})
    out.append(("acc.mul(A,R,H), lp_mul's round-2 library + the counting piece and add-C-into-A",
                L.job(P.from_slice(by["acc.mul(A,R,H)"], "target"), lib, L.FULL, "target",
                      general=P.from_slice(gen["acc.mul(A,R,H)"], "target"))))
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
    dest = HERE / "records" / "carry" / "carry_mac.json"
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
