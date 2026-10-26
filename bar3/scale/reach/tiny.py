"""REACH.md §3: the tiny test, no Modal. Round 3 of the earlier anchored-pairs run on x² mod N in base 2 (bar3/scale/
digits/records/ct2a2_clean_b_src_dxsq_dg1.jsonl): with that round's library and the twin's step digit/T.mod(X,N)@x+QN/A.X
as T.look0, the target's look at the frozen budget (30k states, 200 programs a level) and at the deeper one
(reach.DEEP_LOOK), each a digits target job (rows and the check at big x). Does the deeper look find a program the
target accepts?

    cd cpu_first && bar3/scale/rowcheck/memcap.sh 10 <python with numba> -m bar3.scale.reach.tiny
"""

from __future__ import annotations

import json
from pathlib import Path

from bar3 import problems as P
from bar3.scale.digits import jobs as DJ
from bar3.scale.digits import machine as M
from bar3.scale.digits import targets as DT
from bar3.scale.learn import job
from bar3.scale.reach.reach import DEEP_LOOK
from bar3.scale.subgoal.configs import CONFIGS
from library.inventing import brm

HERE = Path(__file__).resolve().parent


def main():
    path = HERE.parent / "digits" / "records" / "ct2a2_clean_b_src_dxsq_dg1.jsonl"
    recs = [r for r in (json.loads(line) for line in open(path) if line.strip()) if "round" in r]
    lib = dict(recs[2]["lib"])
    lib["T.look0"] = recs[3]["status"]["digit/T.mod(X,N)@x+QN/A.X"]["best"]
    tgt = P.make("target", "target", DT.FORMULAS["dxsq"][0], DT.gen_for("dxsq"), 7, 10)
    kinds = list(brm.LOOP_KINDS) + [M.KINDS[2]]
    print(f"T.look0 = {lib['T.look0']}; T.law10 = {lib['T.law10']}", flush=True)
    for label, budget in (("the frozen look", CONFIGS["ct2a2_clean_deep_sub"]["budgets"]["look"]),
                          ("the deeper look", DEEP_LOOK)):
        r = DJ.search_job(dict(job(tgt, lib, kinds, budget, "look", 7, 10), digits_target="dxsq"))
        print(f"{label}: found_cost {r['found_cost']}, levels {r['level_sizes']}, best_fit {r['best_fit']}, stop "
              f"'{r['stop']}', {r.get('wall', 0):.0f}s; programs {[p['program'] for p in r.get('programs', [])[:3]]}; "
              f"right on the rows only: {(r.get('right_on_rows_only') or {}).get('n')}", flush=True)


if __name__ == "__main__":
    main()
