"""SUBGOAL.md §8, the tiny test: the chain's subgoal (the rest of the way from x² mod N's routine, which no probe
solved in the chain runs) treated as a target in its own right (recurse.py), in the round where it first became the
subgoal. Each round's pool is rebuilt from the run's record (its libraries, the target probes' near misses, its
picks) and checked against the record's; the subgoal's near misses come from one search of it with the 1M-state
probe's budget (the levels the 10M-state probe holds too), run here. Then every candidate posed on its instances is
scored against it, and the one the score singles out is printed with the rest of the top. Also, for x² mod N as the
subgoal in round 0 (unsolved after its deeper probe in the squaring runs), whether the rule would act there at all.

For the write-up's named pieces (PIECES: not seen by the score): where they land when posed, and where they would
land if they were (posed on the data's pairs and scored with the rest). And one source the rule does not have
(`offset`: the answer less its answer at X = 0, mod N): whether it poses the piece, and the usual probes on it.

    cd cpu_first && <python with numba> -m bar3.scale.subgoal.rank_rest [record:round ...]   # writes records/rank_rest.json
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
from bar3.scale.jobs import search_job
from bar3.scale.learn import job
from bar3.scale.subgoal import recurse as R
from bar3.scale.subgoal import score as SC
from bar3.scale.subgoal.configs import CONFIGS

HERE = Path(__file__).resolve().parent
RUNS = ("lp_clean_deep_subchain_tenrho_s1:2", "lp_clean_deep_subchain_mont_s1:2",
        "ct2a2_clean_deep_subchain_mont_s1:2", "lp_clean_deep_sub_tenrho_s1:0", "lp_clean_deep_sub_cube_s1:0")
PIECES = {                     # the pieces the write-up names, to say where they land (not used by the score)
    "tenrho": {"10X mod N": lambda X, N: (10 * X) % N},
    "mont": {"X + N if X is odd": lambda X, N: X + N * (X % 2), "X·2⁻¹ mod N": lambda X, N: (X * (N + 1) // 2) % N},
}


def offset(c, k, k2):
    """A source the rule does not have: c's answer less its answer at X = 0 (for the same N), mod N, on c's instances
    whose N has X = 0 among them."""
    def gen(size):
        xs, Ns, w = R.gen_of(c, k)(size)
        at0 = {int(n): int(v) for a, n, v in zip(xs, Ns, w) if a == 0}
        keep = [i for i in range(len(xs)) if int(Ns[i]) in at0]
        return xs[keep], Ns[keep], np.array([(int(w[i]) - at0[int(Ns[i])]) % int(Ns[i]) for i in keep])
    return P.make(f"offset/{c.name}", "offset", f"{c.name}: its answer less its answer at X = 0, mod N", gen, k, k2)


def read(name):
    with open(HERE / "records" / f"{name}.jsonl") as fh:
        return [r for r in (json.loads(line) for line in fh if line.strip()) if "round" in r]


def pools(recs, cfg, upto):
    """learn.run's step 2 in rounds 0..upto, rebuilt from the record, for the current target (TG.use): [(library at
    the round's start, pool, keys practised before it)], each pool's names checked against the record's."""
    k, k2, src = cfg["k"], cfg.get("k2", P.K2), cfg["sources"]
    tgt = P.the_target(k, k2)
    rng = np.random.default_rng(cfg.get("seed", 0))
    known, kinds, practiced, out = {tuple(tgt.want)}, list(brm.LOOP_KINDS), {}, []
    for r in range(upto + 1):
        lib = recs[r - 1]["lib"] if r else {}
        near = [(f, d, p) for f, d, p in recs[r]["target_probe"].get("near", []) if p]
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
            pool += [c for c in P.from_near_misses(near, lib, k, k2)
                     if c.source in src or (c.source == "restrict" and "remainder" in src)]
        if "dream" in src:
            pool += P.dreams(lib, kinds, k, rng, known, n_want=cfg.get("dreams", 16), k2=k2)
        if "recur" in src:
            pool += S.recurrences(P.target_gen, k, k2)
        if "const" in src:
            pool += S.constants(P.target_gen, k, k2)
        if "rrem" in src:
            pool += S.routine_remainders(lib, k, k2)
        pool = [c for c in P.dedupe(pool) if c.key() not in practiced and c.key() != tgt.key()]
        assert [c.name for c in pool] == [c["name"] for c in recs[r]["pool"]], f"round {r}: the pool differs"
        out.append((lib, pool, dict(practiced)))
        for c in pool:
            if c.name in recs[r]["picked"]:
                practiced.setdefault(c.key(), c.name)
    return out


def rank(spec):
    name, rnd = spec.rsplit(":", 1)
    rnd = int(rnd)
    recs = read(name)
    tname = recs[0]["target_name"]
    cfg = CONFIGS[name.rsplit("_", 2)[0]]
    k, k2 = cfg["k"], cfg.get("k2", P.K2)
    TG.use(tname)
    lib, pool, practiced = pools(recs, cfg, rnd)[rnd]
    tgt = P.the_target(k, k2)
    sub = recs[rnd]["subgoal"]["name"]
    c = next(d for d in pool if d.name == sub)
    t0 = time.time()
    nj = search_job(job(c, lib, list(brm.LOOP_KINDS), cfg["budgets"]["probe_b"], "sub_near", k, k2, near=12))
    near = [(f, d, p) for f, d, p in nj.get("near", []) if p]
    new = R.pose_on(c, lib, near, k, k2, cfg["sources"])
    both = [d for d in P.dedupe(pool + new) if d.key() not in practiced and d.key() not in (tgt.key(), c.key())]
    sc = R.score_against(both, c, k, k2)
    rows = SC.target_rows(c, k, k2, gen=R.gen_of(c, k))
    order = sorted(sc, key=lambda n: sc[n]["key"], reverse=True)

    def place(n):
        return 1 + sum(1 for m in sc if sc[m]["key"] > sc[n]["key"]), sum(1 for m in sc if sc[m]["key"] == sc[n]["key"])

    pieces, offs = {}, None
    for label, f in (PIECES.get(tname, {}).items() if c.source != "law2" else ()):
        want = f(rows.x, rows.N)
        hits = [d.name for d in both if (lambda g: g is not None and (g == want).all())(SC.on_target(d, rows))]
        hyp = P.make(f"hyp/{label}", "hyp", label, lambda size, f=f: (lambda xs, Ns: (xs, Ns, f(xs, Ns)))(
            *(v.astype(np.int64) for v in P.target.pairs(size))), k, k2)
        h = R.score_against([hyp], c, k, k2).get(hyp.name)
        pieces[label] = dict(posed=[dict(name=n, rank=place(n)[0], tied=place(n)[1],
                                         **{a: b for a, b in sc[n].items() if a != "key"}) for n in hits if n in sc],
                             if_posed=h and dict(rank=1 + sum(1 for m in sc if sc[m]["key"] > h["key"]),
                                                 tied=sum(1 for m in sc if sc[m]["key"] == h["key"]),
                                                 **{a: b for a, b in h.items() if a != "key"}))
        o = offset(c, k, k2)
        g = o and SC.on_target(o, rows)
        if g is not None and (g == want).all() and (g == rows.t).all():
            offs = dict(piece=label, itself=True)           # the answer at X = 0 is 0: the subgoal posed again
        elif g is not None and (g == want).all():
            probes = []
            for b in ("probe_a", "probe_b"):
                pr = search_job(job(o, lib, list(brm.LOOP_KINDS), cfg["budgets"][b], b, k, k2))
                probes.append(dict(budget=b, found_cost=pr["found_cost"], best_fit=pr["best_fit"], NP=pr["NP"],
                                   seconds=pr["wall"], programs=[p["program"] for p in pr.get("programs", [])][:3]))
                if pr["found_cost"] is not None:
                    break
            offs = dict(piece=label, probes=probes)
    return dict(record=name, round=rnd, target=tname, subgoal=sub, formula=c.formula, rows=len(rows.t),
                consts=rows.consts, inputs_law=SC.inputs_law(rows), pool=len(pool), new=len(new),
                new_by_source={s: sum(1 for d in new if d.source == s) for s in sorted({d.source for d in new})},
                scored=len(sc), scored_new=sum(1 for n in sc if n.startswith("sub[")),
                near_job=dict(best_fit=nj["best_fit"], NP=nj["NP"], level_sizes=nj["level_sizes"], seconds=nj["wall"],
                              found_cost=nj["found_cost"]),
                near=near, singled_out=R.singled_out(sc), pieces=pieces, offset=offs,
                top=[dict(name=n, **{a: b for a, b in sc[n].items() if a != "key"}) for n in order[:15]])


def main():
    SC._check_ops()
    out = []
    for spec in sys.argv[1:] or RUNS:
        t = time.time()
        r = rank(spec)
        out.append(r)
        print(f"== {r['record']} round {r['round']}: subgoal {r['subgoal']} ({r['formula']}); {r['rows']} rows, "
              f"consts {r['consts']}; pool {r['pool']} + {r['new']} posed on it {r['new_by_source']}; scored "
              f"{r['scored']} ({r['scored_new']} new)  ({time.time() - t:.0f}s)")
        print(f"   its near-miss search: best fit {r['near_job']['best_fit']} of {r['near_job']['NP']}, levels "
              f"{r['near_job']['level_sizes']}, {r['near_job']['seconds']}s; near misses {[n[:2] for n in r['near']]}")
        print(f"   singled out: {r['singled_out']}")
        for label, pc in r["pieces"].items():
            h = pc["if_posed"]
            print(f"   {label}: " + (", ".join(f"{p['name']} rank {p['rank']} (tied {p['tied']}) law {p['law']} "
                                           f"table {p['table']}" for p in pc["posed"]) or "not posed") +
                  (f"; if it were: rank {h['rank']} (tied {h['tied']}), law {h['law']}, table {h['table']}" if h
                   else "; if it were: not scored"))
        if r["offset"] and r["offset"].get("itself"):
            print(f"   the offset source poses {r['offset']['piece']}, which is the subgoal itself")
        elif r["offset"]:
            print(f"   the offset source poses {r['offset']['piece']}; the usual probes on it: " +
                  "; ".join(f"{p['budget']} cost {p['found_cost']} (fit {max(p['best_fit'])} of {p['NP']}, "
                            f"{p['seconds']}s) {p['programs'][:1]}" for p in r["offset"]["probes"]))
        for i, c in enumerate(r["top"][:12]):
            print(f"   {i + 1:2d}. {c['name'][:70]:<70} law {c['law']!s:<26} table {c['table']:3d} {c['how']:<4}")
    (HERE / "records" / "rank_rest.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
