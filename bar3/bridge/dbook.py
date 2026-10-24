"""The digit law book with no target: bar 3's learner practising everything it can (its rule `all`), with every slice
of the book as a problem of its own, until each slice is solved or a round solves nothing new. What comes out is the
instruction set a learner invents without being told which instructions a digit machine needs: every slice it solves
becomes a candidate cell (lift.py then checks that its place loop computes its law on whole numbers).

    ~/github/sotaku/venv/bin/python -m bar3.bridge.dbook --config tiny_book        # a local check, tiny budgets

Each round, as dlearner.run with every unsolved slice in the target's place:
1. pose: every unsolved slice; bar 3's law book on a place's values (every operation on A, R, C, H and the base, in
   every binding, A built by the patterns a, a + N, a + qN); each unsolved slice's restrictions, projections, and the
   values and remainders of its near misses (from its own probe); minus what was practised;
2. probe all (a small search, then a bigger one for the unsolved), and sort them;
3. a slice solved by an accepted program that is right at every base up to 16 is invented;
4. practise every learnable problem: the corpus, MDL merges, a routine per problem (bar 3's v2, without side effects).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from library import dsl, merge
from bar3.bridge import dcm, dproblems as P, slices as S
from bar3.bridge import dlearner as DL
from bar3.bridge.djobs import search_job

HERE = Path(__file__).resolve().parent
CONFIGS = {
    "tiny_book": dict(rounds=2, budgets=DL.TINY, sources=DL.SOURCES, laws=("lt", "add", "sub", "select")),
    "book": dict(rounds=5, budgets=DL.BIG, sources=DL.SOURCES),
    "book_noslice": dict(rounds=5, budgets=DL.BIG, sources=tuple(s for s in DL.SOURCES if s != "slice")),
}


def universe(bases=S.SEARCH_BASES + S.CHECK_BASES):
    """Every place: A, R, H digits and C a bit, at every base (the rows bar 3's law book is applied to)."""
    rows = {B: {k: (0,) for k in itertools.product(range(B), range(B), range(2), range(B))} for B in bases}
    return P.Cand("place", "universe", "every place", ("A", "R", "C", "H", "N"), ("A",), rows)


def run(name, cfg, mapper, emit, log=print):
    Bud = cfg["budgets"]
    posed, dropped, gen = DL.the_book()
    if cfg.get("laws"):
        posed = [s for s in posed if s.law in cfg["laws"]]
    targets = {s.name: P.from_slice(s, "slice") for s in posed}
    gens = {s.name: P.from_slice(gen[s.name], "slice") for s in posed if s.name in gen}
    laws = P.dedupe(P.digit_laws(universe())) if "law" in cfg["sources"] else []
    lib, corpus, practiced, stems = {}, {}, {}, {}
    invented = {}
    sigs = DL.Signatures()
    totals = {}

    def account(kind, recs):
        t = totals.setdefault(kind, dict(jobs=0, states=0, seconds=0.0))
        for r in recs:
            if isinstance(r, dict):
                t["jobs"] += 1
                t["states"] += sum(r["level_sizes"])
                t["seconds"] += r.get("wall", r["seconds"])

    def run_jobs(kind, jobs):
        recs = mapper(jobs) if jobs else []
        for j, r in zip(jobs, recs):
            if not isinstance(r, dict):
                log(f"  job failed: {j['name']}: {r!r}"[:300])
        account(kind, recs)
        return [r if isinstance(r, dict) else None for r in recs]

    for rnd in range(cfg["rounds"]):
        open_ = [t for n, t in targets.items() if n not in invented]
        if not open_:
            break
        tp = run_jobs("slice_probe", [DL.job(t, lib, Bud["target_probe"], "slice_probe", general=gens.get(t.name),
                                            near=8) for t in open_])
        pool = []
        src = cfg["sources"]
        for t, r in zip(open_, tp):
            if "restrict" in src:
                pool += P.restrictions(t)
            if "project" in src:
                pool += P.projections(t)
            if r is not None and ("value" in src or "remainder" in src):
                near = [(f, d, p) for f, d, p in r.get("near", []) if p]
                pool += [c for c in P.from_near_misses(near, lib, t)
                         if c.source in src or (c.source == "restrict" and "remainder" in src)]
        pool = [t for t in open_] + laws + pool
        pool = [c for c in P.dedupe(pool) if c.key() not in practiced]
        log(f"[{name}] round {rnd}: {len(open_)} slices open; pool {len(pool)}")
        status = {}
        ra = run_jobs("probe_a", [DL.job(c, lib, Bud["probe_a"], "probe_a", general=gens.get(c.name)
                                         if c.source == "slice" else None) for c in pool])
        todo = [i for i, r in enumerate(ra) if r is not None and r["found_cost"] is None]
        rb = run_jobs("probe_b", [DL.job(pool[i], lib, Bud["probe_b"], "probe_b", general=gens.get(pool[i].name)
                                         if pool[i].source == "slice" else None) for i in todo])
        res = list(ra)
        for i, r in zip(todo, rb):
            res[i] = r
        have = {sigs.of((t,), lib) for t in dcm.prims(dcm.REGS) + list(lib)}
        learn_sig = {}
        for c, r in zip(pool, res):
            if r is None:
                status[c.name] = dict(status="failed")
                continue
            probe = dict(states=sum(r["level_sizes"]), seconds=r["wall"], best_fit=r["best_fit"], NP=r["NP"],
                         stop=r["stop"])
            if r["found_cost"] is None or not r.get("programs"):
                status[c.name] = dict(status="unsolved", fit=max(r["best_fit"]) / r["NP"], cost=None, **probe)
                continue
            progs = DL.choose([p["program"] for p in r["programs"]], lib, c)
            st = dict(status="trivial" if r["found_cost"] <= 1 else "learnable", cost=r["found_cost"],
                      programs=progs[:20], best=progs[0], **probe)
            if c.source == "slice":
                gp = [p for p in r["programs"] if p.get("general")]
                st["general"] = bool(gp)
                if gp:
                    first = next(p for p in r["programs"] if p["program"] == progs[0])
                    invented[c.name] = dict(round=rnd, cost=r["found_cost"], program=progs[0],
                                            primitive=first["primitive"], general=bool(first.get("general")),
                                            lib=dict(lib), states=probe["states"], seconds=probe["seconds"])
            if st["status"] == "learnable":
                s = sigs.of(dsl.parse(progs[0]), lib)
                if s in have:
                    st["status"] = "alias"
                elif s in learn_sig:
                    st["status"], st["of"] = "duplicate", learn_sig[s]
                else:
                    learn_sig[s] = c.name
            status[c.name] = st
        pick = [c for c in pool if status.get(c.name, {}).get("status") == "learnable"]
        log("  probes: " + ", ".join(f"{s} {sum(1 for v in status.values() if v['status'] == s)}"
                                     for s in ("trivial", "learnable", "alias", "duplicate", "unsolved")))
        log(f"  invented so far: {sorted(invented)}")
        for c in pick:
            practiced.setdefault(c.key(), c.name)
            stems[c.name] = DL.stem(c, len(stems) + 1)
        pcorpus = {t: [dsl.parse(p) for p in ps] for t, ps in corpus.items()}
        for c in pick:
            pcorpus[stems[c.name]] = [dsl.parse(p) for p in status[c.name]["programs"]]
        plib = {t: dsl.parse(v) for t, v in lib.items()}
        mlog = []
        pcorpus, plib, _ = merge.mdl(pcorpus, plib, prefix=f"M{rnd}.", log=mlog, kinds=[], lib_weight=0.5,
                                     n_prims=DL.N_PRIMS, n_kinds=DL.N_KINDS)
        pcorpus, plib, plog = DL.promote_first(pcorpus, plib, [stems[c.name] for c in pick])
        for e in mlog + plog:
            e["expands_to"] = dsl.show(dsl.expand(plib[e["name"]], plib)) if e["name"] in plib else ""
        lib = {t: dsl.show(v) for t, v in plib.items()}
        corpus = {t: [dsl.show(p) for p in ps[:20]] for t, ps in pcorpus.items()}
        emit(dict(config=name, round=rnd, pool=[c.to_dict() for c in pool], status=status,
                  picked=[c.name for c in pick], stems=dict(stems), merges=mlog, promoted=plog, lib=lib,
                  corpus=corpus, invented=invented, open=[t.name for t in open_], totals=json.loads(json.dumps(totals)),
                  dropped=[(s.name, s.why) for s in dropped]))
        if not pick:
            log("  nothing learnable: stop")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny_book")
    a = ap.parse_args()
    (HERE / "records").mkdir(exist_ok=True)
    out = HERE / "records" / f"{a.config}_local.jsonl"
    out.write_text("")

    def emit(rec):
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    run(a.config, CONFIGS[a.config], lambda jobs: [search_job(j) for j in jobs], emit)
    print("wrote", out)


if __name__ == "__main__":
    main()
