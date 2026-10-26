"""The two sources' tiny local tests (NEWTON.md §3), no Modal:

    cd cpu_first && <python with numba> -m bar3.scale.newton.tiny pose     # what each source poses, and the score, seconds
    cd cpu_first && <python with numba> -m bar3.scale.newton.tiny probe    # the offset's piece and the alias check, minutes

`pose`: on the triangular numbers (tri) and 10·x² + 1 (tenrho), with the library each earlier run had at the start
of a round (read from its record: the held-out test's for tri, the subgoal study's for tenrho), what each source
poses, with its rows, and what the subgoal's score makes of the target's new candidates in that round's pool.
`probe` (after PREREG.md): the frozen probes (30k, then 1M states) on 10X mod N, the offset of the rest of the way
from x² mod N, with hardest first's libraries at rounds 2 and 3; and the learner's alias check (bar3/learner
Signatures: 24 random machine states) on the program found and on 3X's, the problem the learner called it a duplicate
of. No learner run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from library import dsl
from library.inventing import brm
from bar3 import problems as P
from bar3.breadth import targets as TG
from bar3.scale import sources as S
from bar3.scale.heldout import targets as _ho  # noqa: F401  (registers tri)
from bar3.scale.newton import sources as NS
from bar3.scale.subgoal import score as SC

ROOT = Path(__file__).resolve().parent.parent
RECORDS = {"tri": ROOT / "heldout" / "records" / "{rule}_deep_sub_tri_ho1.jsonl",
           "tenrho": ROOT / "subgoal" / "records" / "{rule}_deep_sub_tenrho_s1.jsonl"}
K, K2 = 7, P.K2


def read(name, rule):
    path = str(RECORDS[name]).format(rule=rule)
    return [r for r in (json.loads(line) for line in open(path) if line.strip()) if "round" in r]


def lib_at(recs, rnd):
    """The library at the start of round rnd (the record of round rnd − 1 holds the library after it)."""
    return dict(recs[rnd - 1]["lib"]) if rnd else {}


def show_rows(c, n=4):
    init = getattr(c, "init", {})
    return "; ".join(f"X={c.xs[i]} N={c.Ns[i]}" + "".join(f" {r}={v[i]}" for r, v in init.items()) + f" -> {c.want[i]}"
                     for i in range(min(n, len(c.xs))))


def posed(name, lib):
    """What the two sources pose for target `name` with library `lib`: the running sum's steps and the offset of each
    rest of the way from the library's routines (sources.routine_remainders); and, to show why the source leaves it
    out, the target's own offset (scored with the rest)."""
    TG.use(name)
    tgt = P.the_target(K, K2)
    rs = NS.running_sum(P.target_gen, K, K2)
    rest = S.routine_remainders(lib, K, K2) if lib else []
    return tgt, rs, rest, NS.offsets(rest, K, K2) + [o for o in [NS.offset(tgt, K, K2)] if o]


def pose():
    for name in ("tri", "tenrho"):
        for rule in ("lp_clean", "ct2a2_clean"):
            recs = read(name, rule)
            for rnd in (0, 1, 2):
                lib = lib_at(recs, rnd)
                tgt, rs, rest, offs = posed(name, lib)
                print(f"{name} {rule} round {rnd} (library of {len(lib)}): running sum {len(rs)} placements, "
                      f"{len(rest)} rests of the way, {len(offs)} offsets", flush=True)
                for c in rs[:1] + offs:
                    print(f"  {c.name}: {len(c.xs)} rows at k = {K}, {len(c.cx)} at k = {K2}; {show_rows(c)}")
                sc = SC.score_pool(offs + rest, tgt, K, K2)
                for n, v in sorted(sc.items(), key=lambda kv: kv[1]["key"], reverse=True)[:4]:
                    print(f"  score {n}: {v}")
                old = recs[rnd].get("subgoal") or {}
                best = sorted(((tuple(v["key"]), n) for n, v in old.get("scores", {}).items()), reverse=True)[:2]
                print(f"  the record's pool: subgoal {old.get('name')}, its best keys {best}")


def probe():
    from bar3 import learner as L
    from bar3.learner import Signatures
    from bar3.scale.jobs import search_job
    from bar3.scale.subgoal.configs import CONFIGS
    from library.inventing.jobs import machine_for
    B = CONFIGS["lp_clean_deep_sub"]["budgets"]
    recs = read("tenrho", "lp_clean")
    for rnd in (2, 3):
        lib = lib_at(recs, rnd)
        _, _, rest, _ = posed("tenrho", lib)
        for c in NS.offsets(rest, K, K2):
            if c.name != "offset/rrem/T.law5":
                continue
            print(f"round {rnd} (library of {len(lib)}): {c.name}; rows {show_rows(c, 6)}", flush=True)
            for kind in ("probe_a", "probe_b"):
                r = search_job(L.job(c, lib, list(brm.LOOP_KINDS), B[kind], kind, K, K2))
                print(f"  {kind}: cost {r['found_cost']}, levels {r['level_sizes']}, {r['wall']:.0f}s, stop "
                      f"'{r['stop']}'; programs {[p['program'] for p in r.get('programs', [])[:3]]}", flush=True)
                if r["found_cost"] is not None:
                    break
    lib = lib_at(recs, 2)
    three = recs[2]["status"]["law2/add(add(X,X),X)"]["best"]
    found = "T.con2 MOV.A.X L.X[MOV.X.U M1.4]"
    sg = Signatures(K)
    pl = {t: dsl.parse(v) for t, v in lib.items()}
    xs, Ns = np.array([1, 2, 3, 4, 5, 6]), np.array([7] * 6)
    print(f"the alias check: {found} (10X mod N) and {three} (3X, law2/add(add(X,X),X)): same signature "
          f"{sg.of(dsl.parse(found), lib) == sg.of(dsl.parse(three), lib)}")
    for prog in (found, three):
        b = sg.tests.copy()
        brm.run_items(sg.m, b, tuple(dsl.parse(prog)), pl)
        v, _ = brm.evaluate(tuple(dsl.parse(prog)), pl, xs, Ns, machine_for(K))
        print(f"  {prog}: dies on {int(b.dead.sum())} of {b.dead.size} test states (cap {sg.m.cap}, "
              f"{sg.m.limit} steps); at N = 7, x = 1..6: {v.tolist()}")


if __name__ == "__main__":
    {"pose": pose, "probe": probe}[sys.argv[1]]()
