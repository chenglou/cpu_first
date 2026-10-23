"""A learner that picks its own practice (PRACTICE.md): the inventing study's loop on the bare machine, with the
curated task list replaced by generic candidates (problems.py) and a selection rule.

    ~/github/sotaku/venv/bin/python -m bar3.learner --config tiny        # a local check (k = 5, tiny budgets)

Each round:
1. probe the target (small budget): how close the library gets, and its near misses;
2. pose the candidates from the configured sources (problems.py), minus those already practiced;
3. probe every candidate: a tiny search, then a bigger one for those still unsolved. Trivial = solved at cost <= 1
   (nothing to learn), learnable = solved at cost >= 2 by a program that is not an alias of an existing token,
   unsolved = the rest (with the most probes any program got right);
4. select up to K by the rule (below); unsolved picks get a full practice search;
5. the picked problems' programs join the corpus, MDL merges run over it (half-price library, as Bbare_mdl_rt), and
   every picked solved problem's simplest program becomes a routine T.<id> (wake_sleep.promote);
6. search the target with the full budget; stop when a program is right on every pair with N <= 40.

Rules (the learner's own signals, and the references):
- ct2: target compression, pairs. For every pair of learnable candidates, the target is probed with both their
  programs as routines; the score is how many bits that saves on the target's description (program bits plus bits for
  each probe still wrong, as `mdl`). The best pairs' members are picked.
- ct1: the same with one candidate at a time.
- lp: learning progress without the target: candidates neither solved before nor hopeless now (learnable at the
  probe budget), the hardest first (largest cost).
- nov: novelty: learnable candidates whose program's outputs on the target's pairs differ most from every token's.
- random_ok: K learnable at random (pays for the probes, ignores the signals).
- random: K candidates at random, no probes; each gets a full practice search.
- all: every learnable candidate.
- curated: arm B's list (target.CURATED), every unsolved task searched with the full budget each round (Bbare's loop).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from library import dsl, merge
from library.inventing import brm
from library.inventing.bfs import _sig
from library.inventing.jobs import machine_for
from library.wake_sleep import promote, simplest
from bar3 import problems as P
from bar3.jobs import search_job

HERE = Path(__file__).resolve().parent

PROBE_A = dict(max_cost=10, max_states=30_000, time_limit=120, max_body=2, limit=200)
PROBE_B = dict(PROBE_A, max_states=1_000_000, time_limit=600)
FULL = dict(PROBE_A, max_states=3_000_000, time_limit=2400)
LOOK = dict(PROBE_A, max_states=30_000, time_limit=120)
SOURCES = ("law", "restrict", "project", "value", "remainder", "dream")
BIG = dict(probe_a=PROBE_A, probe_b=PROBE_B, practice=FULL, target=FULL, look=LOOK, target_probe=LOOK)
TINY = dict(probe_a=dict(PROBE_A, max_states=1_000, time_limit=20), probe_b=dict(PROBE_A, max_states=5_000,
            time_limit=30), practice=dict(PROBE_A, max_states=10_000, time_limit=30),
            target=dict(PROBE_A, max_states=10_000, time_limit=30), look=dict(PROBE_A, max_states=3_000, time_limit=20),
            target_probe=dict(PROBE_A, max_states=3_000, time_limit=20))

CONFIGS = {
    "tiny": dict(rule="ct2", k=5, K=4, rounds=2, budgets=TINY, sources=SOURCES, pair_pool=3, pool_cap=12,
                 pool_keep=("law/mul(X,X)/x", "law/mod(X,N)/x+qN")),
    "tiny_curated": dict(rule="curated", k=5, rounds=2, budgets=TINY),
}
for _r in ("ct2", "ct1", "lp", "nov", "random_ok", "random", "all"):
    CONFIGS[_r] = dict(rule=_r, k=7, K=4, rounds=5, budgets=BIG, sources=SOURCES)
for _r in ("random_ok", "random"):
    for _s in (1, 2):
        CONFIGS[f"{_r}_s{_s}"] = dict(CONFIGS[_r], seed=_s)
CONFIGS["curated"] = dict(rule="curated", k=7, rounds=5, budgets=BIG)
ABLATE = {"law": ("law",), "nolaw": ("restrict", "project", "value", "remainder", "dream"),
          "near": ("value", "remainder"), "trans": ("restrict", "project"), "dream": ("dream",)}
for _a, _src in ABLATE.items():
    for _r in ("ct2", "lp", "all"):
        CONFIGS[f"{_r}_{_a}"] = dict(CONFIGS[_r], sources=_src)
CONFIGS["ct2_law2"] = dict(CONFIGS["ct2"], sources=SOURCES + ("law2",))   # + one law applied to another's result
# one size up: every N up to 26 (base 3, three digits; INVENTING.md's S2), checked up to 30
BIG26 = dict(BIG, probe_b=dict(PROBE_B, max_states=2_000_000))
for _r in ("ct2", "lp", "all", "curated"):
    CONFIGS[f"{_r}_k26"] = dict(CONFIGS[_r], k=26, k2=30, budgets=BIG26)
for _r in ("lp", "all"):
    CONFIGS[f"{_r}_law2"] = dict(CONFIGS[_r], sources=SOURCES + ("law2",))
for _c in [c for c in CONFIGS if not c.startswith("tiny")]:               # v2: routines without side effects
    CONFIGS[f"{_c}_clean"] = dict(CONFIGS[_c], routine="clean")


def mdl(rec, n_symbols, k):
    """Bits to describe the target's answers with the best program of each level: its tokens, then for every probe it
    still gets wrong, which probe and the right answer. A level whose only fits are aliases counts one probe wrong."""
    NP = rec["NP"]
    tok, exc = math.log2(n_symbols), math.log2(NP) + math.log2(k)
    best = math.inf
    for d, f in enumerate(rec["best_fit"]):
        if f == NP and rec.get("found_cost") != d:
            f = NP - 1
        best = min(best, d * tok + (NP - f) * exc)
    return best


def job(c, lib, kinds, budget, kind, k, k2=P.K2, **extra):
    probes = list(range(len(c.xs)))
    if len(probes) > 32:
        probes = sorted(np.random.default_rng(len(c.xs)).choice(len(c.xs), 32, replace=False).tolist())
    return dict(name=c.name, kind=kind, k=k, xs=[c.xs[i] for i in probes], Ns=[c.Ns[i] for i in probes],
                want=[c.want[i] for i in probes], every=[c.xs, c.Ns, c.want], check=[k2, c.cx, c.cN, c.cw],
                lib=dict(lib), kinds=list(kinds), **budget, **extra)


class Signatures:
    """What a program does to random machines (the search's own test for equal items), to spot aliases."""

    def __init__(self, k, seed=7):
        self.m = machine_for(k)
        self.tests = brm.random_states(24, 6, np.random.default_rng(seed))

    def of(self, items, lib):
        return _sig(self.m, self.tests, tuple(items), {t: dsl.parse(v) for t, v in lib.items()})


def stem(c, i):
    """A readable routine name (the token is T.<stem>): no spaces or brackets (dsl.parse splits on those)."""
    if c.source == "law":
        _, op, pat = c.name.split("/")
        return f"{op}{'' if pat == 'x' else '@' + pat}"
    if c.source == "curated":
        return c.name.split("/")[1]
    return f"{c.source[:3]}{i}"


def side_effects(prog, lib, xs, Ns, k):
    """Does prog leave the inputs X and N as they were, and how many registers does it change (on these instances)?"""
    b = brm.fresh(np.asarray(xs), np.asarray(Ns))
    start = b.regs.copy()
    brm.run_items(machine_for(k), b, prog, lib)
    changed = (b.regs[0] != start[0]).any(axis=0)
    return bool(not changed[brm.X] and not changed[brm.N]), int(changed.sum())


def choose(progs, lib, c, k, how):
    """The program a solved problem keeps as its routine, first in the returned list. "simplest": the shortest
    expansion (wake_sleep.simplest; the inventing study's rule). "clean": among the minimal programs, one that leaves
    its inputs as they were and changes the fewest registers, then the shortest expansion: a routine without side
    effects can be reused inside a loop (a generic convention for subroutines, not a fact about the target)."""
    pl = {t: dsl.parse(v) for t, v in lib.items()}
    parsed = [dsl.parse(p) for p in progs]
    if how == "simplest":
        best = simplest(parsed, pl)
    else:
        def key(p):
            keeps, touched = side_effects(p, pl, c.xs, c.Ns, k)
            return (not keeps, touched, dsl.cost(dsl.expand(p, pl)))
        best = min(parsed, key=key)
    first = dsl.show(best)
    return [first] + [p for p in progs if p != first]


def promote_first(corpus, lib, tasks):
    """wake_sleep.promote, but each task's routine is its first program (the one `choose` put first)."""
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


def run(name, cfg, mapper, emit, log=print):
    k, k2, B, rule = cfg["k"], cfg.get("k2", P.K2), cfg["budgets"], cfg["rule"]
    how = cfg.get("routine", "simplest")
    rng = np.random.default_rng(cfg.get("seed", 0))
    lib, kinds = {}, list(brm.LOOP_KINDS)        # lib: token -> program text
    corpus, practiced, stems = {}, {}, {}         # corpus: stem -> [program text]; practiced: key -> candidate name
    sigs = Signatures(k)
    tgt = P.the_target(k, k2)
    known_dreams = {tuple(tgt.want)}             # behaviours already posed (the target's own, as arm D)
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

    n_sym = lambda lb: len(brm.PRIMS) + len(lb) + len(kinds)   # noqa: E731
    for rnd in range(cfg["rounds"]):
        # 1. the target probe
        tp = run_jobs("target_probe", [job(tgt, lib, kinds, B["target_probe"], "target_probe", k, k2, near=12)])[0]
        mdl0 = mdl(tp, n_sym(lib), k)
        near = [(f, d, p) for f, d, p in tp.get("near", []) if p]
        # 2. the candidates
        if rule == "curated":
            pool = [c for c in P.curated(k, k2) if c.name not in stems]
        else:
            pool = []
            src = cfg["sources"]
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
                pool += P.dreams(lib, kinds, k, rng, known_dreams, n_want=cfg.get("dreams", 16), k2=k2)
            pool = [c for c in P.dedupe(pool) if c.key() not in practiced and c.key() != tgt.key()]
            if cfg.get("pool_cap"):              # local checks only: a random part of the pool
                keep = set(rng.choice(len(pool), min(cfg["pool_cap"], len(pool)), replace=False).tolist())
                keep |= {i for i, c in enumerate(pool) if c.name in cfg.get("pool_keep", ())}
                pool = [c for i, c in enumerate(pool) if i in keep]
        log(f"[{name}] round {rnd}: target probe mdl {mdl0:.1f} best {tp['best_fit']}; pool {len(pool)}")
        # 3. probes
        status = {}
        if rule in ("random", "curated"):
            ra = rb = []
        else:
            ra = run_jobs("probe_a", [job(c, lib, kinds, B["probe_a"], "probe_a", k, k2) for c in pool])
            todo = [i for i, r in enumerate(ra) if r is not None and r["found_cost"] is None]
            rb = run_jobs("probe_b", [job(pool[i], lib, kinds, B["probe_b"], "probe_b", k, k2) for i in todo])
            res = list(ra)
            for i, r in zip(todo, rb):
                res[i] = r
            have = {sigs.of((t,), lib) for t in list(brm.PRIMS) + list(lib)}
            learn_sig = {}
            for c, r in zip(pool, res):
                if r is None:
                    status[c.name] = dict(status="failed")
                    continue
                probe = dict(states=sum(r["level_sizes"]), seconds=r["wall"], best_fit=r["best_fit"], NP=r["NP"],
                             stop=r["stop"])
                if r["found_cost"] is None:
                    status[c.name] = dict(status="unsolved", fit=max(r["best_fit"]) / r["NP"], cost=None, **probe)
                    continue
                progs = choose([p["program"] for p in r["programs"]], lib, c, k, how)
                best = progs[0]
                st = dict(status="trivial" if r["found_cost"] <= 1 else "learnable", cost=r["found_cost"],
                          programs=progs[:20], best=best, **probe)
                if st["status"] == "learnable":
                    s = sigs.of(dsl.parse(best), lib)
                    if s in have:
                        st["status"] = "alias"
                    elif s in learn_sig:
                        st["status"] = "duplicate"
                        st["of"] = learn_sig[s]
                    else:
                        learn_sig[s] = c.name
                status[c.name] = st
        learnable = [c for c in pool if status.get(c.name, {}).get("status") == "learnable"]
        log(f"  probes: " + ", ".join(f"{s} {sum(1 for v in status.values() if v['status'] == s)}"
                                      for s in ("trivial", "learnable", "alias", "duplicate", "unsolved")))
        # 4. selection
        K = cfg.get("K", 4)
        scores, pairs = {}, []
        if rule == "curated":
            pick = list(pool)
        elif rule == "random":
            pick = [pool[i] for i in sorted(rng.choice(len(pool), min(K, len(pool)), replace=False))]
        elif rule == "random_ok":
            pick = [learnable[i] for i in sorted(rng.choice(len(learnable), min(K, len(learnable)), replace=False))]
        elif rule == "all":
            pick = learnable
        elif rule == "lp":
            pick = sorted(learnable, key=lambda c: -status[c.name]["cost"])[:K]
        elif rule == "nov":
            # novelty: how many of the target's pairs the candidate's program computes differently from every token
            # the library already has (each run alone from a fresh machine), the most different first
            pl = {t: dsl.parse(v) for t, v in lib.items()}
            outs = [brm.evaluate((t,), pl, tgt.xs, tgt.Ns, machine_for(k))[0] for t in list(brm.PRIMS) + list(lib)]
            for c in learnable:
                got = brm.evaluate(dsl.parse(status[c.name]["best"]), pl, tgt.xs, tgt.Ns, machine_for(k))[0]
                scores[c.name] = float(min(int((got != o).sum()) for o in outs))
            pick = sorted(learnable, key=lambda c: (-scores[c.name], status[c.name]["cost"]))[:K]
        elif rule in ("ct1", "ct2"):
            def with_routines(cs):
                lb = dict(lib)
                for j, c in enumerate(cs):
                    lb[f"T.look{j}"] = status[c.name]["best"]
                return lb
            looks = [job(tgt, with_routines([c]), kinds, B["look"], "look", k, k2) for c in learnable]
            rl = run_jobs("look", looks)
            for c, r in zip(learnable, rl):
                if r is not None:
                    scores[c.name] = mdl0 - mdl(r, n_sym(lib) + 1, k)
            if rule == "ct1":
                pick = sorted(learnable, key=lambda c: -scores.get(c.name, -1e9))[:K]
            else:
                cand = learnable
                if cfg.get("pair_pool"):            # pairs among the best singles only (local checks)
                    cand = sorted(learnable, key=lambda c: -scores.get(c.name, -1e9))[:cfg["pair_pool"]]
                    learnable = cand
                combos = list(itertools.combinations(range(len(learnable)), 2))
                rl2 = run_jobs("look", [job(tgt, with_routines([learnable[a], learnable[b]]), kinds, B["look"], "look",
                                            k, k2) for a, b in combos])
                for (a, b), r in zip(combos, rl2):
                    if r is not None:
                        pairs.append((mdl0 - mdl(r, n_sym(lib) + 2, k), learnable[a].name, learnable[b].name,
                                      r["found_cost"]))
                pairs.sort(key=lambda p: -p[0])
                chosen = []
                for g, a, b, _ in pairs:
                    for nm in (a, b):
                        if nm not in chosen and len(chosen) < K:
                            chosen.append(nm)
                    if len(chosen) >= K:
                        break
                for c in sorted(learnable, key=lambda c: -scores.get(c.name, -1e9)):   # < 2 learnable: singles
                    if len(chosen) < K and c.name not in chosen:
                        chosen.append(c.name)
                pick = [c for c in learnable if c.name in chosen]
                pick.sort(key=lambda c: chosen.index(c.name))
        else:
            raise ValueError(rule)
        # 5. practice: unsolved picks get the full budget
        need = [c for c in pick if status.get(c.name, {}).get("status") not in ("learnable", "trivial", "alias",
                                                                             "duplicate")]
        rp = run_jobs("practice", [job(c, lib, kinds, B["practice"], "practice", k, k2) for c in need])
        for c, r in zip(need, rp):
            if r is not None and r["found_cost"] is not None:
                progs = choose([p["program"] for p in r["programs"]], lib, c, k, how)
                status[c.name] = dict(status="solved in practice", cost=r["found_cost"], programs=progs[:20],
                                      best=progs[0])
            else:
                status.setdefault(c.name, {})["practice"] = "not solved" if r is not None else "failed"
        solved_now = [c for c in pick if "best" in status.get(c.name, {})]
        for c in pick:                      # a pick is not posed again, solved or not
            practiced.setdefault(c.key(), c.name)
        for c in solved_now:
            stems[c.name] = stem(c, len(stems) + 1)
        pcorpus = {t: [dsl.parse(p) for p in ps] for t, ps in corpus.items()}
        for c in solved_now:
            pcorpus[stems[c.name]] = [dsl.parse(p) for p in status[c.name]["programs"]]
        plib = {t: dsl.parse(v) for t, v in lib.items()}
        mlog = []
        pcorpus, plib, _ = merge.mdl(pcorpus, plib, prefix=f"M{rnd}.", log=mlog, kinds=kinds, lib_weight=0.5,
                                     n_prims=len(brm.PRIMS), n_kinds=len(brm.LOOP_KINDS))
        pcorpus, plib, plog = (promote if how == "simplest" else promote_first)(
            pcorpus, plib, [stems[c.name] for c in solved_now])                              # routines T.<stem>
        for e in mlog + plog:
            e["expands_to"] = dsl.show(dsl.expand(plib[e["name"]], plib)) if e["name"] in plib else ""
        lib = {t: dsl.show(v) for t, v in plib.items()}
        corpus = {t: [dsl.show(p) for p in ps[:20]] for t, ps in pcorpus.items()}
        # 6. the target
        tr = run_jobs("target", [job(tgt, lib, kinds, B["target"], "target", k, k2, general=True)])[0]
        gen = None
        if tr is not None and tr["found_cost"] is not None:
            best = max(tr["programs"], key=lambda p: p.get("general", 0))
            gen = best if best.get("general") == 1.0 else None
        log(f"  picked: {[c.name for c in pick]}")
        for e in mlog + plog:
            log(f"    + {e['name']} = {e['def']}    ({e['expands_to']})")
        log(f"  target: cost {tr and tr['found_cost']} levels {tr and tr['level_sizes']} "
            f"{tr and [p['program'] for p in tr.get('programs', [])][:3]}")
        emit(dict(config=name, round=rnd, k=k, rule=rule, pool=[c.to_dict() for c in pool], status=status,
                  scores=scores, pairs=pairs[:200], picked=[c.name for c in pick],
                  solved=[c.name for c in solved_now], stems=dict(stems), merges=mlog,
                  promoted=plog, lib=lib, corpus=corpus, target_probe=tp, target=tr, general=gen,
                  mdl0=mdl0, totals=json.loads(json.dumps(totals)),
                  records=[r for r in list(ra) + list(rb) + list(rp) if r is not None and r.get("kind") == "practice"]))
        if gen is not None:
            log(f"  general x^2 mod N in round {rnd}: {gen['program']} = {gen['primitive']}")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny")
    a = ap.parse_args()
    out = HERE / "records" / f"{a.config}_local.jsonl"
    out.write_text("")

    def emit(rec):
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    run(a.config, CONFIGS[a.config], lambda jobs: [search_job(j) for j in jobs], emit)
    print("wrote", out)


if __name__ == "__main__":
    main()
