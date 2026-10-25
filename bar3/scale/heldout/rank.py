"""HELDOUT.md §2, the early check (after the pre-registration, before the runs): round 0 of the frozen learner's
`*_deep_sub` pool on each held-out target, scored by the subgoal score (bar3/scale/subgoal/score.py), as SUBGOAL.md §2
did for the ten training targets. The round-0 target probe (the learner's own job: 30k states, 12 near misses) is run
here, so the pool is the one the learner builds in round 0 (empty library, nothing practised, dreams from seed 0).

    cd cpu_first && <python with numba> -m bar3.scale.heldout.rank [target ...]   # writes records/rank0.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from library.inventing import brm
from bar3 import problems as P
from bar3.breadth import targets as TG
from bar3.scale.heldout import targets as H
from bar3.scale.jobs import search_job
from bar3.scale.learn import job
from bar3.scale.subgoal import rank as R0
from bar3.scale.subgoal import score as SC
from bar3.scale.subgoal.configs import CONFIGS

HERE = Path(__file__).resolve().parent
K, K2 = 7, P.K2


def rank(tname):
    TG.use(tname)
    tgt = P.the_target(K, K2)
    tp = search_job(job(tgt, {}, list(brm.LOOP_KINDS), CONFIGS["lp_clean_deep_sub"]["budgets"]["target_probe"],
                        "target_probe", K, K2, near=12))
    near = [(f, d, p) for f, d, p in tp.get("near", []) if p]
    _, pool = R0.round0_pool(near)
    sc = SC.score_pool(pool, tgt, K, K2)
    rows = SC.target_rows(tgt, K, K2)
    order = sorted(sc, key=lambda n: sc[n]["key"], reverse=True)
    keys = [sc[n]["key"] for n in order]
    sub = order[0] if keys and keys[0] > (0, 0) and (len(keys) == 1 or keys[1] < keys[0]) else None
    sq = (rows.x * rows.x) % rows.N
    sq_names = [c.name for c in pool if (lambda g: g is not None and (g == sq).all())(SC.on_target(c, rows))]
    return dict(target=tname, formula=TG.TARGETS[tname].formula, pool=len(pool), scored=len(sc), rows=len(rows.t),
                consts=rows.consts, inputs_law=SC.inputs_law(rows), target_probe=dict(best_fit=tp["best_fit"],
                NP=tp["NP"], found_cost=tp["found_cost"]), near=near, subgoal=sub,
                tied_at_top=sum(1 for k in keys if keys and k == keys[0]),
                sq=[dict(name=n, rank=1 + sum(1 for m in sc if sc[m]["key"] > sc[n]["key"]), **sc[n])
                    for n in sq_names if n in sc],
                top=[dict(name=n, **{a: b for a, b in sc[n].items() if a != "key"}) for n in order[:12]])


def main():
    SC._check_ops()
    out = []
    for tname in sys.argv[1:] or H.NAMES:
        t = time.time()
        r = rank(tname)
        out.append(r)
        print(f"== {tname} ({r['formula']}): pool {r['pool']}, scored {r['scored']}, rows {r['rows']}, consts "
              f"{r['consts']}, a law of the inputs: {r['inputs_law']}; target probe best fit "
              f"{max(r['target_probe']['best_fit'])} of {r['target_probe']['NP']}  ({time.time() - t:.0f}s)")
        print(f"   subgoal: {r['subgoal']}" + ("" if r["subgoal"] else f" ({r['tied_at_top']} tied at the top)"))
        print("   x² mod N: " + (", ".join(f"rank {s['rank']} law {s['law']} table {s['table']}" for s in r["sq"])
                                 or "not scored"))
        for i, c in enumerate(r["top"][:6]):
            print(f"   {i + 1:2d}. {c['name'][:60]:<60} law {c['law']!s:<26} table {c['table']:3d} {c['how']}")
    (HERE / "records").mkdir(exist_ok=True)
    (HERE / "records" / "rank0.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
