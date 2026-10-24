"""Bar 3's learner (bar3/learner.py) on one place of a digit machine: it invents a digit instruction of stage 3 from
the counting machine's primitives, choosing its own practice from the digit law book (slices.py, dproblems.py).

    ~/github/sotaku/venv/bin/python -m bar3.bridge.dlearner --config tiny_sub        # a local check, tiny budgets

The loop is bar 3's, unchanged in what it does each round:
1. probe the target (small budget): how close the library gets, and its near misses;
2. pose the candidates: bar 3's law book on the target's own rows (law), every other slice of the digit law book
   (slice), the target's restrictions and projections, and the values and remainders of its near misses; minus those
   already practised;
3. probe every candidate (a tiny search, then a bigger one for those still unsolved) and sort them into trivial
   (cost <= 1), learnable (cost >= 2 and not an alias of a token or a duplicate of another candidate), unsolved;
4. pick by the rule; 5. the picks' programs join the corpus, MDL merges run over it (half-price library), and every
   pick's program becomes a routine T.<stem>, the one without side effects on its inputs (bar 3's v2, "clean");
6. search the target with the full budget; stop when a program is right at every base up to 16.

Rules: lp (hardest learnable first), ct2 with anchor 2 (BREADTH.md's cheaper pairs signal: the target's description
shortened by a pair of routines, over pairs with one member among the 2 hardest), all (every learnable candidate),
ct1 (bar 3's single-candidate signal; ct1d gives each lookahead the 1M-state probe's budget).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np

from library import dsl, merge
from bar3.bridge import dcm, dproblems as P, dsearch, slices as S
from bar3.bridge.djobs import search_job

HERE = Path(__file__).resolve().parent

PROBE_A = dict(max_cost=10, max_states=30_000, time_limit=120, max_body=2, limit=200)
PROBE_B = dict(PROBE_A, max_states=1_000_000, time_limit=600)
FULL = dict(PROBE_A, max_states=3_000_000, time_limit=2400)
LOOK = dict(PROBE_A, max_states=30_000, time_limit=120)
BIG = dict(probe_a=PROBE_A, probe_b=PROBE_B, practice=FULL, target=FULL, look=LOOK, target_probe=LOOK)
TINY = dict(probe_a=dict(PROBE_A, max_states=2_000, time_limit=20), probe_b=dict(PROBE_A, max_states=10_000,
            time_limit=30), practice=dict(PROBE_A, max_states=20_000, time_limit=60),
            target=dict(PROBE_A, max_states=20_000, time_limit=60), look=dict(PROBE_A, max_states=2_000, time_limit=20),
            target_probe=dict(PROBE_A, max_states=2_000, time_limit=20))
SOURCES = ("law", "slice", "restrict", "project", "value", "remainder")

# the digit functions of stage 3's instructions (s3i's cells and SEL.S's per-place select), as slices of the book
TARGETS = {"sbb": "sub(A,R)", "adc": "add(A,R)", "mul": "acc.mul(A,R,H)", "sel": "select(C,A,R)"}
CONFIGS = {}
for _t, _s in TARGETS.items():
    CONFIGS[f"tiny_{_t}"] = dict(target=_s, rule="lp", K=4, rounds=2, budgets=TINY, sources=SOURCES)
    for _r in ("lp", "ct2", "all"):
        CONFIGS[f"{_r}_{_t}"] = dict(target=_s, rule=_r, K=4, rounds=5 if _r == "lp" else 3, budgets=BIG,
                                     sources=SOURCES, anchor=2)
DEEP = dict(BIG, look=PROBE_B)          # ct1d: each candidate's lookahead with the 1M-state probe's budget
for _t, _s in TARGETS.items():
    CONFIGS[f"ct1d_{_t}"] = dict(target=_s, rule="ct1", K=4, rounds=3, budgets=DEEP, sources=SOURCES)
N_PRIMS, N_KINDS = len(dcm.prims(dcm.REGS)), len(dcm.loop_kinds(dcm.REGS))


def mdl(rec, n_symbols, n_out):
    """bar3.learner.mdl: bits to describe the target's outputs with the best program of each level (its tokens), plus,
    for every probe it still gets wrong, which probe and its outputs."""
    NP = rec["NP"]
    tok, exc = math.log2(n_symbols), math.log2(NP) + n_out * math.log2(max(S.SEARCH_BASES))
    best = math.inf
    for d, f in enumerate(rec["best_fit"]):
        if f == NP and rec.get("found_cost") != d:
            f = NP - 1
        best = min(best, d * tok + (NP - f) * exc)
    return best


def job(c, lib, budget, kind, general=None, **extra):
    ins, want = P.sample(c)
    every = P.table(c, S.SEARCH_BASES)
    check = P.table(c, S.CHECK_BASES)
    lst = lambda t: [{r: v.tolist() for r, v in t[0].items()}, t[1].tolist()]   # noqa: E731
    out = dict(name=c.name, kind=kind, regs=list(c.regs()), outputs=list(c.outputs),
               probe=[{r: v.tolist() for r, v in ins.items()}, want.tolist()], every=lst(every), check=lst(check),
               lib=dict(lib), **budget, **extra)
    if general is not None:
        out["general"] = lst(P.table(general, S.GENERAL_BASES))
    return out


class Signatures:
    """What a program does to random machines (the search's own test for equal items), to spot aliases."""

    def __init__(self, seed=7):
        self.m = P.machine_for(S.SEARCH_BASES)
        self.tests = dcm.random_states(24, 6, np.random.default_rng(seed))

    def of(self, items, lib):
        return dsearch._sig(self.m, self.tests, tuple(items), {t: dsl.parse(v) for t, v in lib.items()})


def stem(c, i):
    """A readable routine name: a slice's own name, or the source and a number."""
    if c.source in ("law", "slice"):
        return re.sub(r"[^A-Za-z0-9.(),]", "", c.name)
    return f"{c.source[:3]}{i}"


def side_effects(prog, lib, c):
    """Does prog leave the inputs that are not outputs as they were, and how many registers does it change?"""
    ins, _, _ = P.table(c, S.SEARCH_BASES)
    b = dcm.fresh(ins)
    start = b.regs.copy()
    dcm.run_items(P.machine_for(S.SEARCH_BASES), b, prog, lib)
    changed = (b.regs[0] != start[0]).any(axis=0)
    keep = [dcm.RI[r] for r in c.inputs if r not in c.outputs]
    return bool(not changed[keep].any()), int(changed.sum())


def choose(progs, lib, c):
    """bar 3's v2 ("clean"): among the minimal programs, one that leaves its inputs as they were and changes the
    fewest registers, then the shortest expansion."""
    pl = {t: dsl.parse(v) for t, v in lib.items()}

    def key(p):
        keeps, touched = side_effects(p, pl, c)
        return (not keeps, touched, dsl.cost(dsl.expand(p, pl)))
    parsed = [dsl.parse(p) for p in progs]
    first = dsl.show(min(parsed, key=key))
    return [first] + [p for p in progs if p != first]


def promote_first(corpus, lib, tasks):
    """bar3.learner.promote_first: each task's first program becomes a routine T.<task>."""
    lib, log = dict(lib), []
    for t in tasks:
        prog = corpus[t][0]
        name = f"T.{t}"
        if dsl.cost(prog) < 2 or name in lib:
            continue
        lib[name] = prog
        fn = merge._apply_pattern(prog, name, False)
        corpus = {u: [merge.rewrite(p, fn) for p in ps] for u, ps in corpus.items()}
        log.append({"name": name, "def": dsl.show(prog), "routine": True})
    return corpus, lib, log


def the_book():
    posed, dropped = S.book(S.SEARCH_BASES + S.CHECK_BASES)
    gen, _ = S.book(S.GENERAL_BASES)
    return posed, dropped, {s.name: s for s in gen}


def run(name, cfg, mapper, emit, log=print):
    B, rule, K = cfg["budgets"], cfg["rule"], cfg.get("K", 4)
    rng = np.random.default_rng(cfg.get("seed", 0))
    posed, dropped, gen = the_book()
    tsl = next(s for s in posed if s.name == cfg["target"])
    tgt = P.from_slice(tsl, "target")
    tgt_gen = P.from_slice(gen[tsl.name], "target")
    laws = [P.from_slice(s, "slice") for s in posed if s.name != tsl.name]
    lib, corpus, practiced, stems = {}, {}, {}, {}
    sigs = Signatures()
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

    n_sym = lambda lb: N_PRIMS + len(lb) + N_KINDS   # noqa: E731
    for rnd in range(cfg["rounds"]):
        tp = run_jobs("target_probe", [job(tgt, lib, B["target_probe"], "target_probe", near=12)])[0]
        mdl0 = mdl(tp, n_sym(lib), len(tgt.outputs))
        near = [(f, d, p) for f, d, p in tp.get("near", []) if p]
        src = cfg["sources"]
        pool = []
        if "law" in src:
            pool += P.digit_laws(tgt)
        if "slice" in src:
            pool += laws
        if "restrict" in src:
            pool += P.restrictions(tgt)
        if "project" in src:
            pool += P.projections(tgt)
        if "value" in src or "remainder" in src:
            pool += [c for c in P.from_near_misses(near, lib, tgt)
                     if c.source in src or (c.source == "restrict" and "remainder" in src)]
        pool = [c for c in P.dedupe(pool) if c.key() not in practiced and c.key() != tgt.key()]
        log(f"[{name}] round {rnd}: target probe mdl {mdl0:.1f} best {tp['best_fit']}; pool {len(pool)}")
        status = {}
        ra = run_jobs("probe_a", [job(c, lib, B["probe_a"], "probe_a") for c in pool])
        todo = [i for i, r in enumerate(ra) if r is not None and r["found_cost"] is None]
        rb = run_jobs("probe_b", [job(pool[i], lib, B["probe_b"], "probe_b") for i in todo])
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
            progs = choose([p["program"] for p in r["programs"]], lib, c)
            st = dict(status="trivial" if r["found_cost"] <= 1 else "learnable", cost=r["found_cost"],
                      programs=progs[:20], best=progs[0], **probe)
            if st["status"] == "learnable":
                s = sigs.of(dsl.parse(progs[0]), lib)
                if s in have:
                    st["status"] = "alias"
                elif s in learn_sig:
                    st["status"], st["of"] = "duplicate", learn_sig[s]
                else:
                    learn_sig[s] = c.name
            status[c.name] = st
        learnable = [c for c in pool if status.get(c.name, {}).get("status") == "learnable"]
        log("  probes: " + ", ".join(f"{s} {sum(1 for v in status.values() if v['status'] == s)}"
                                     for s in ("trivial", "learnable", "alias", "duplicate", "unsolved")))
        scores, pairs = {}, []
        if rule == "all":
            pick = learnable
        elif rule == "lp":
            pick = sorted(learnable, key=lambda c: -status[c.name]["cost"])[:K]
        elif rule == "ct1":
            # bar 3's ct1 (one candidate at a time), with the lookahead's budget from the config (ct1d: the 1M-state
            # probe's, deep enough to reach a target that one routine makes 5 instructions long)
            rl = run_jobs("look", [job(tgt, dict(lib, **{"T.look0": status[c.name]["best"]}), B["look"], "look")
                                   for c in learnable])
            for c, r in zip(learnable, rl):
                if r is not None:
                    scores[c.name] = mdl0 - mdl(r, n_sym(lib) + 1, len(tgt.outputs))
            pick = sorted(learnable, key=lambda c: -scores.get(c.name, -1e9))[:K]
        elif rule == "ct2":
            def with_routines(cs):
                lb = dict(lib)
                for j, c in enumerate(cs):
                    lb[f"T.look{j}"] = status[c.name]["best"]
                return lb
            rl = run_jobs("look", [job(tgt, with_routines([c]), B["look"], "look") for c in learnable])
            for c, r in zip(learnable, rl):
                if r is not None:
                    scores[c.name] = mdl0 - mdl(r, n_sym(lib) + 1, len(tgt.outputs))
            hard = sorted(learnable, key=lambda c: -status[c.name]["cost"])[:cfg.get("anchor") or len(learnable)]
            combos = sorted({tuple(sorted((a, b))) for a in range(len(learnable)) for b in range(len(learnable))
                             if a != b and (learnable[a] in hard or learnable[b] in hard)})
            rl2 = run_jobs("look", [job(tgt, with_routines([learnable[a], learnable[b]]), B["look"], "look")
                                    for a, b in combos])
            for (a, b), r in zip(combos, rl2):
                if r is not None:
                    pairs.append((mdl0 - mdl(r, n_sym(lib) + 2, len(tgt.outputs)), learnable[a].name,
                                  learnable[b].name, r["found_cost"]))
            pairs.sort(key=lambda p: -p[0])
            chosen = []
            for g, a, b, _ in pairs:
                for nm in (a, b):
                    if nm not in chosen and len(chosen) < K:
                        chosen.append(nm)
                if len(chosen) >= K:
                    break
            for c in sorted(learnable, key=lambda c: -scores.get(c.name, -1e9)):
                if len(chosen) < K and c.name not in chosen:
                    chosen.append(c.name)
            pick = sorted([c for c in learnable if c.name in chosen], key=lambda c: chosen.index(c.name))
        else:
            raise ValueError(rule)
        for c in pick:
            practiced.setdefault(c.key(), c.name)
        for c in pick:
            stems[c.name] = stem(c, len(stems) + 1)
        pcorpus = {t: [dsl.parse(p) for p in ps] for t, ps in corpus.items()}
        for c in pick:
            pcorpus[stems[c.name]] = [dsl.parse(p) for p in status[c.name]["programs"]]
        plib = {t: dsl.parse(v) for t, v in lib.items()}
        mlog = []
        pcorpus, plib, _ = merge.mdl(pcorpus, plib, prefix=f"M{rnd}.", log=mlog, kinds=[], lib_weight=0.5,
                                     n_prims=N_PRIMS, n_kinds=N_KINDS)
        pcorpus, plib, plog = promote_first(pcorpus, plib, [stems[c.name] for c in pick])
        for e in mlog + plog:
            e["expands_to"] = dsl.show(dsl.expand(plib[e["name"]], plib)) if e["name"] in plib else ""
        lib = {t: dsl.show(v) for t, v in plib.items()}
        corpus = {t: [dsl.show(p) for p in ps[:20]] for t, ps in pcorpus.items()}
        tr = run_jobs("target", [job(tgt, lib, B["target"], "target", general=tgt_gen)])[0]
        found = None
        if tr is not None and tr["found_cost"] is not None and tr.get("programs"):
            gens = [p for p in tr["programs"] if p.get("general")]
            found = gens[0] if gens else None
        log(f"  picked: {[c.name for c in pick]}")
        for e in mlog + plog:
            log(f"    + {e['name']} = {e['def']}    ({e['expands_to']})")
        log(f"  target: cost {tr and tr['found_cost']} levels {tr and tr['level_sizes']} "
            f"{tr and [p['program'] for p in tr.get('programs', [])][:3]}")
        emit(dict(config=name, round=rnd, target=cfg["target"], rule=rule, pool=[c.to_dict() for c in pool],
                  status=status, scores=scores, pairs=pairs[:200], picked=[c.name for c in pick], stems=dict(stems),
                  merges=mlog, promoted=plog, lib=lib, corpus=corpus, target_probe=tp, target_rec=tr, general=found,
                  mdl0=mdl0, totals=json.loads(json.dumps(totals)), dropped=[(s.name, s.why) for s in dropped]))
        if found is not None:
            log(f"  invented {cfg['target']} in round {rnd}: {found['program']} = {found['primitive']}")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny_sub")
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
