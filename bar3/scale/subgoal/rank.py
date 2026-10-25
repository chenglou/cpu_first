"""SUBGOAL.md §2, the tiny test: round 0's candidates, with the depth-2 law book on (the `*_deep` configs' sources),
ranked by the subgoal score (score.py), for each of the ten targets; where x² mod N lands.

The pool is learn.run's round-0 pool (empty library), built here from the target probe's near misses as the
records hold them (`*_deep_<target>_L1` for the four missed targets, `lp_clean_new_<target>_L1` for the rest: the
round-0 target probe is the same in every config). Its size is checked against the record's.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.subgoal.rank [target ...]   # writes records/rank0.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from library.inventing import brm
from bar3 import problems as P
from bar3.breadth import targets as TG
from bar3.scale import sources as S
from bar3.scale.subgoal import score as SC

HERE = Path(__file__).resolve().parent
SCALE_RECORDS = HERE.parent / "records"
OUT = HERE / "records"
K, K2 = 7, P.K2
SOURCES = ("law", "restrict", "project", "value", "remainder", "dream", "law2", "recur", "const", "rrem")
MISSED = ("ten", "tenrho", "cube", "mont")


def round0_pool(near, k=K, k2=K2, src=SOURCES, n_dreams=16, seed=0):
    """learn.run's step 2 in round 0 (no library, nothing practised), for the current target (TG.use)."""
    tgt = P.the_target(k, k2)
    rng = np.random.default_rng(seed)
    pool = []
    if "law" in src:
        pool += P.laws(k, k2)
    if "law2" in src:
        pool += P.laws2(k, k2)
    if "restrict" in src:
        pool += P.restrictions(P.target_gen, k, k2)
    if "project" in src:
        pool += P.projections(P.target_gen, k, k2)
    if "value" in src or "remainder" in src:
        pool += [c for c in P.from_near_misses(near, {}, k, k2)
                 if c.source in src or (c.source == "restrict" and "remainder" in src)]
    if "dream" in src:
        pool += P.dreams({}, list(brm.LOOP_KINDS), k, rng, {tuple(tgt.want)}, n_want=n_dreams, k2=k2)
    if "recur" in src:
        pool += S.recurrences(P.target_gen, k, k2)
    if "const" in src:
        pool += S.constants(P.target_gen, k, k2)
    if "rrem" in src:
        pool += S.routine_remainders({}, k, k2)
    return tgt, [c for c in P.dedupe(pool) if c.key() != tgt.key()]


def record_round0(tname):
    f = SCALE_RECORDS / (f"lp_clean_deep_{tname}_L1.jsonl" if tname in MISSED else f"lp_clean_new_{tname}_L1.jsonl")
    with open(f) as fh:
        return json.loads(fh.readline()), f.name


def rank(tname):
    TG.use(tname)
    rec, fname = record_round0(tname)
    near = [(f, d, p) for f, d, p in rec["target_probe"].get("near", []) if p]
    tgt, pool = round0_pool(near)
    sc = SC.score_pool(pool, tgt, K, K2)
    rows = SC.target_rows(tgt, K, K2)
    sq = (rows.x * rows.x) % rows.N
    sq_names = [c.name for c in pool if (lambda g: g is not None and (g == sq).all())(SC.on_target(c, rows))]
    order = sorted(sc, key=lambda n: sc[n]["key"], reverse=True)
    st = rec["status"]

    def place(name):
        key = sc[name]["key"]
        return 1 + sum(1 for n in sc if sc[n]["key"] > key), sum(1 for n in sc if sc[n]["key"] == key)

    top = [dict(name=n, **{k: v for k, v in sc[n].items() if k != "key"}, status=st.get(n, {}).get("status"),
                cost=st.get(n, {}).get("cost")) for n in order[:15]]
    sq_place = {n: place(n) for n in sq_names if n in sc}
    return dict(target=tname, record=fname, pool=len(pool), record_pool=len(rec["pool"]), scored=len(sc),
                inputs_law=SC.inputs_law(rows), consts=rows.consts, rows=len(rows.t),
                sq=[dict(name=n, rank=sq_place[n][0], tied=sq_place[n][1], **{k: v for k, v in sc[n].items()
                                                                                if k != "key"},
                         status=st.get(n, {}).get("status")) for n in sq_place],
                top=top)


def main():
    SC._check_ops()
    names = sys.argv[1:] or list(TG.SEQUENCE)
    out = []
    for tname in names:
        t = time.time()
        r = rank(tname)
        out.append(r)
        print(f"== {tname}: pool {r['pool']} (record {r['record_pool']}), scored {r['scored']}, rows {r['rows']}, "
              f"consts {r['consts']}, target a law of the inputs: {r['inputs_law']}  ({time.time() - t:.1f}s)")
        print("   x² mod N:", [(s["name"], f"rank {s['rank']} (tied {s['tied']})", s["law"], s["table"], s["how"],
                                 s["status"]) for s in r["sq"]] or "not posed on every instance")
        for i, c in enumerate(r["top"][:10]):
            print(f"   {i + 1:2d}. {c['name']:<44} law {c['law']!s:<24} table {c['table']:3d} {c['how']:<4} "
                  f"pure {c['pure']:3d}  {c['status']} {c['cost']}")
    OUT.mkdir(exist_ok=True)
    (OUT / "rank0.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
