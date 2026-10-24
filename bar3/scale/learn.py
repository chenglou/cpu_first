"""The bar-3 learner with the scale study's levers (SCALE.md §2): bar3/learner.run, line for line, plus
- the carry-over of bar3/breadth/carry.py (start from a state, return it) and the cheaper pairs signal of
  bar3/breadth/cheap.py (`anchor_cost`: pairs with one member among the k hardest), built in;
- two more practice sources (sources.py): "recur" (the answer from the answers before it, as one number or as the
  step that moves two numbers at once) and "const" (the constants the answers hold at a fixed x, as "make c");
- problems of other shapes (registers set beyond X and N, answers in several registers), which the jobs and the
  routine choice handle.
A config without the new options runs exactly as bar3/learner.run (smoke.py checks it against the original).
"""

from __future__ import annotations

import itertools
import json

import numpy as np

from library import dsl, merge
from library.inventing import brm
from library.inventing.jobs import machine_for
from library.wake_sleep import promote
from bar3 import learner as L
from bar3 import problems as P
from bar3.breadth import targets as TG
from bar3.scale import sources as S
from bar3.scale.jobs import start

mdl, Signatures, stem, promote_first = L.mdl, L.Signatures, L.stem, L.promote_first


def job(c, lib, kinds, budget, kind, k, k2=P.K2, **extra):
    """bar3/learner.job, and for a shaped problem (sources.RCand) its registers, answer registers and the instances
    it is accepted on."""
    if not isinstance(c, S.RCand):
        return L.job(c, lib, kinds, budget, kind, k, k2, **extra)
    probes = list(range(len(c.xs)))
    if len(probes) > 32:
        probes = sorted(np.random.default_rng(len(c.xs)).choice(len(c.xs), 32, replace=False).tolist())
    ev = c.every or (c.xs, c.Ns, c.init, c.want)
    return dict(name=c.name, kind=kind, k=k, xs=[c.xs[i] for i in probes], Ns=[c.Ns[i] for i in probes],
                want=[c.want[i] for i in probes], init={r: [v[i] for i in probes] for r, v in c.init.items()},
                outs=list(c.outs), every=[ev[0], ev[1], ev[3]], every_init=ev[2], check=[k2, c.cx, c.cN, c.cw],
                check_init=c.cinit, lib=dict(lib), kinds=list(kinds), **budget, **extra)


def side_effects(prog, lib, c, k):
    """bar3/learner.side_effects for any problem: does prog leave its inputs as they were (X, N and the registers
    set before it runs, except those that hold the answer), and how many registers does it change?"""
    init = getattr(c, "init", {})
    b = start(c.xs, c.Ns, init)
    before = b.regs.copy()
    brm.run_items(machine_for(k), b, prog, lib)
    changed = (b.regs[0] != before[0]).any(axis=0)
    inputs = [r for r in ["X", "N"] + list(init) if r not in getattr(c, "outs", ("A",))]
    return bool(not any(changed[brm.RI[r]] for r in inputs)), int(changed.sum())


def choose(progs, lib, c, k, how):
    """bar3/learner.choose with side_effects above (the same for the learner's own problems)."""
    pl = {t: dsl.parse(v) for t, v in lib.items()}
    parsed = [dsl.parse(p) for p in progs]
    if how == "simplest":
        from library.wake_sleep import simplest
        best = simplest(parsed, pl)
    else:
        def key(p):
            keeps, touched = side_effects(p, pl, c, k)
            return (not keeps, touched, dsl.cost(dsl.expand(p, pl)))
        best = min(parsed, key=key)
    first = dsl.show(best)
    return [first] + [p for p in progs if p != first]


def learned_costs(lib, kinds, alpha=0.5):
    """Bits per token and loop kind under what has been learned: a unigram model over the library's definitions
    (every routine's and merge's program, inside its loops too), plus one count for each routine or merge token, with
    add-alpha smoothing over the whole vocabulary; cost = round(-log2 p), at least 1. With an empty library every
    token costs the same, and the guided search is the plain one."""
    import collections
    import math
    counts = collections.Counter()

    def walk(items):
        for it in items:
            if isinstance(it, tuple):
                counts[it[1]] += 1
                walk(it[2])
            else:
                counts[it] += 1

    for t, v in lib.items():
        counts[t] += 1
        walk(dsl.parse(v))
    vocab = list(brm.PRIMS) + [t for t in lib if t not in kinds] + list(kinds)
    total = sum(counts[t] for t in vocab) + alpha * len(vocab)
    return {t: max(1, round(-math.log2((counts[t] + alpha) / total))) for t in vocab}


def run(name, cfg, mapper, emit, log=print, init=None):
    init = init or {}
    k, k2, B, rule = cfg["k"], cfg.get("k2", P.K2), cfg["budgets"], cfg["rule"]
    how = cfg.get("routine", "simplest")
    rng = np.random.default_rng(cfg.get("seed", 0))
    lib, kinds = dict(init.get("lib", {})), list(init.get("kinds", brm.LOOP_KINDS))
    corpus, practiced, stems = (dict(init.get("corpus", {})), dict(init.get("practiced", {})),
                                dict(init.get("stems", {})))
    sigs = Signatures(k)
    tgt = P.the_target(k, k2)
    known_dreams = {tuple(tgt.want)}
    totals = {}
    gen = None

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
            if "recur" in src:                                                        # scale: new sources
                pool += S.recurrences(P.target_gen, k, k2)
            if "const" in src:
                pool += S.constants(P.target_gen, k, k2)
            if "rrem" in src:
                pool += S.routine_remainders(lib, k, k2)
            pool = [c for c in P.dedupe(pool) if c.key() not in practiced and c.key() != tgt.key()]
            if cfg.get("pool_cap"):
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
                if cfg.get("pair_pool"):
                    cand = sorted(learnable, key=lambda c: -scores.get(c.name, -1e9))[:cfg["pair_pool"]]
                    learnable = cand
                combos = list(itertools.combinations(range(len(learnable)), 2))
                if cfg.get("anchor") or cfg.get("anchor_cost"):          # scale: cheap.py's patch, built in
                    if cfg.get("anchor"):
                        top = sorted(learnable, key=lambda c: -scores.get(c.name, -1e9))[:cfg["anchor"]]
                    else:
                        top = sorted(learnable, key=lambda c: -status[c.name]["cost"])[:cfg["anchor_cost"]]
                    top = {c.name for c in top}
                    combos = [(a, b) for a, b in combos if learnable[a].name in top or learnable[b].name in top]
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
                for c in sorted(learnable, key=lambda c: -scores.get(c.name, -1e9)):
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
        for c in pick:
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
            pcorpus, plib, [stems[c.name] for c in solved_now])
        for e in mlog + plog:
            e["expands_to"] = dsl.show(dsl.expand(plib[e["name"]], plib)) if e["name"] in plib else ""
        lib = {t: dsl.show(v) for t, v in plib.items()}
        corpus = {t: [dsl.show(p) for p in ps[:20]] for t, ps in pcorpus.items()}
        # 6. the target (with cfg "guided": a second search, cheapest first under learned_costs, when the plain one
        #    finds no general program)
        gup = {"general_upto": cfg["general_upto"]} if cfg.get("general_upto") else {}     # scale: bigger N
        tr = run_jobs("target", [job(tgt, lib, kinds, B["target"], "target", k, k2, general=True, **gup)])[0]
        gen = None
        if tr is not None and tr["found_cost"] is not None:
            best = max(tr["programs"], key=lambda p: p.get("general", 0))
            gen = best if best.get("general") == 1.0 else None
        guided = None
        if cfg.get("guided") and gen is None and lib:
            costs = learned_costs(lib, kinds)
            guided = run_jobs("guided", [job(tgt, lib, kinds, dict(B["target"], max_cost=cfg.get("guided_max", 80)),
                                             "guided", k, k2, general=True, costs=costs, **gup)])[0]
            if guided is not None and guided["found_cost"] is not None:
                best = max(guided["programs"], key=lambda p: p.get("general", 0))
                gen = best if best.get("general") == 1.0 else None
            log(f"  guided target: cost {guided and guided['found_cost']} bits, levels "
                f"{guided and len(guided['level_sizes'])} {guided and [p['program'] for p in guided.get('programs', [])][:3]}")
        log(f"  picked: {[c.name for c in pick]}")
        for e in mlog + plog:
            log(f"    + {e['name']} = {e['def']}    ({e['expands_to']})")
        log(f"  target: cost {tr and tr['found_cost']} levels {tr and tr['level_sizes']} "
            f"{tr and [p['program'] for p in tr.get('programs', [])][:3]}")
        emit(dict(config=name, round=rnd, k=k, rule=rule, pool=[c.to_dict() for c in pool], status=status,
                  scores=scores, pairs=pairs[:200], picked=[c.name for c in pick],
                  solved=[c.name for c in solved_now], stems=dict(stems), merges=mlog,
                  promoted=plog, lib=lib, corpus=corpus, target_probe=tp, target=tr, general=gen,
                  **({"guided": guided} if cfg.get("guided") else {}),
                  mdl0=mdl0, totals=json.loads(json.dumps(totals)),
                  records=[r for r in list(ra) + list(rb) + list(rp) if r is not None and r.get("kind") == "practice"]))
        if gen is not None:
            log(f"  general in round {rnd}: {gen['program']} = {gen['primitive']}")
            break
    return dict(lib=lib, kinds=kinds, corpus=corpus, practiced=practiced, stems=stems, general=gen)


def from_json(d):
    """bar3/breadth/carry.from_json for this learner's problems too: a practised problem's key comes back as tuples
    all the way down (a shaped problem's key nests its registers, which carry.from_json leaves as lists)."""
    def tup(v):
        return tuple(tup(x) for x in v) if isinstance(v, list) else v
    return dict(d, practiced={tup(key): nm for key, nm in d["practiced"]})


def sequence(name, cfg, order, mapper, emit, carry="targets", log=print, on_target=None, state=None, save=None):
    """bar3/breadth/carry.sequence with this run: each target of `order` in turn, the state carried over."""
    from bar3.breadth.carry import add_target
    state = state or {}
    for tname in order:
        TG.use(tname)
        if on_target:
            on_target(tname)
        last = {}

        def emit_t(rec):
            rec = dict(rec, target_name=tname, carry=carry, carried_lib=len(state.get("lib", {})))
            last["rec"] = rec
            emit(rec)

        before = dict(state.get("stems", {}))
        state = run(f"{name}/{tname}", cfg, mapper, emit_t, log=log, init=state)
        new = {n: s for n, s in state["stems"].items() if n not in before}
        state["stems"] = {**before, **{f"{tname}::{n}": s for n, s in new.items()}}
        if carry == "targets" and state["general"] is not None:
            state, plog = add_target(state, tname, last["rec"]["target"], cfg["k"], cfg.get("k2", P.K2))
            emit(dict(config=f"{name}/{tname}", target_name=tname, carry=carry, kind="carried_target",
                      promoted=plog, lib=state["lib"]))
            log(f"  carried target {tname}: {plog}")
        if save:
            save(tname, state)
    return state
