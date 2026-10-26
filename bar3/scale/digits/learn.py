"""The digits study's learner (DIGITS.md §2), written by make_learner.py: bar3/scale/subgoal/learn.py line for line
(the learner frozen for the held-out test), with these changes, marked "# digits:" below:
- the loop kinds include the digit block of the representation, cfg["base"] (machine.py: D.1, D.2 or D.10);
- the target is posed on its rows (targets.py: x < 10·k), and so is everything the learner poses from the target's
  rows: restrictions, projections, the near misses' values and remainders, constants and routine remainders. The
  recurrence source reads the target on the pairs x < N, as before (its table needs every x < N and no more);
- the law book is posed on the target's rows as well (source.row_laws), besides the pairs x < N;
- with "digit" in cfg["sources"], the one-more-digit source (source.digit_steps) poses its steps every round;
- every job on the target is tagged with its name (DigitsTap), so that a worker runs it on the target's machines and
  accepts a program only when it also passes the check at big x (jobs.py).
Everything else, the subgoal included, is the frozen learner's.
"""

from __future__ import annotations

import itertools
import json

import numpy as np

from library import dsl, merge
from library.inventing import brm
from library.inventing.jobs import machine_for
from library.wake_sleep import promote
from bar3 import problems as P
from bar3.breadth import targets as TG
from bar3.breadth.drive import Tap
from bar3.scale import sources as S
from bar3.scale.learn import choose, job, learned_costs, mdl, promote_first, Signatures, stem
from bar3.scale.subgoal import score as SC
from bar3.scale.digits import machine as M
from bar3.scale.digits import source as DS
from bar3.scale.digits import targets as DT


def subgoal_of(pool, tgt, k, k2, chained=()):
    """The pool's scores (score.score_pool) and the subgoal. After a subgoal has been practised (its routine in
    `chained`), the rest of the way from that routine, when the pool poses it: the "rrem" problem (the target's answer
    from X = what the routine leaves in A), under that name or another candidate's that posed the same problem; the
    latest routine first. Otherwise the candidate with the best key, when that key is positive and no other candidate
    shares it."""
    sc = SC.score_pool(pool, tgt, k, k2)
    for tok in reversed(list(chained)):
        rest = next((c.name for c in pool if f"rrem/{tok}" in (c.name, *c.also)), None)
        if rest:
            return dict(scores=sc, name=rest, after=tok)
    keys = sorted((v["key"] for v in sc.values()), reverse=True)
    name = None
    if keys and keys[0] > (0, 0) and (len(keys) == 1 or keys[1] < keys[0]):
        name = next(n for n, v in sc.items() if v["key"] == keys[0])
    return dict(scores=sc, name=name)


def run(name, cfg, mapper, emit, log=print, init=None):
    init = init or {}
    k, k2, B, rule = cfg["k"], cfg.get("k2", P.K2), cfg["budgets"], cfg["rule"]
    how = cfg.get("routine", "simplest")
    rng = np.random.default_rng(cfg.get("seed", 0))
    lib, kinds = dict(init.get("lib", {})), list(init.get("kinds", brm.LOOP_KINDS + (M.KINDS[cfg["base"]],)))  # digits
    corpus, practiced, stems = (dict(init.get("corpus", {})), dict(init.get("practiced", {})),
                                dict(init.get("stems", {})))
    sigs = Signatures(k)
    tgen, small = DT.gen_for(cfg["target"]), DT.small_gen(cfg["target"])          # digits: the target's rows
    tgt = P.make("target", "target", DT.FORMULAS[cfg["target"]][0], tgen, k, k2)
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
    chained = []                                            # subgoal: routines of the subgoals practised so far
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
                pool += DS.row_laws(tgen, k, k2)                                         # digits: on the rows
            if "law2" in src:
                pool += P.laws2(k, k2)
            if "restrict" in src:
                pool += P.restrictions(tgen, k, k2)                                       # digits
            if "project" in src:
                pool += P.projections(tgen, k, k2)                                        # digits
            if "value" in src or "remainder" in src:
                pool += [c for c in P.from_near_misses(near, lib, k, k2, gen=tgen)             # digits
                         if c.source in src or (c.source == "restrict" and "remainder" in src)]
            if "dream" in src:
                pool += P.dreams(lib, kinds, k, rng, known_dreams, n_want=cfg.get("dreams", 16), k2=k2)
            if "recur" in src:                                                        # scale: new sources
                pool += S.recurrences(small, k, k2)                                       # digits: x < N
            if "const" in src:
                pool += S.constants(tgen, k, k2)                                          # digits
            if "rrem" in src:
                pool += S.routine_remainders(lib, k, k2, gen=tgen)                        # digits
            if "digit" in src:                                                        # digits: the new source
                pool += DS.digit_steps(tgen, cfg["base"], k, k2, lib=lib)
            pool = [c for c in P.dedupe(pool) if c.key() not in practiced and c.key() != tgt.key()]
            if cfg.get("pool_cap"):
                keep = set(rng.choice(len(pool), min(cfg["pool_cap"], len(pool)), replace=False).tolist())
                keep |= {i for i, c in enumerate(pool) if c.name in cfg.get("pool_keep", ())}
                pool = [c for i, c in enumerate(pool) if i in keep]
        log(f"[{name}] round {rnd}: target probe mdl {mdl0:.1f} best {tp['best_fit']}; pool {len(pool)}")
        sub = (subgoal_of(pool, tgt, k, k2, chained if cfg["subgoal"].get("chain") else ())
               if cfg.get("subgoal") else dict(scores={}, name=None))                                   # subgoal
        subs = {sub["name"]} - {None}
        tsubs = subs if cfg.get("subgoal", {}).get("ties", True) else set()   # the subgoal's ties (off: pool order)
        if subs:
            log(f"  subgoal: {sub['name']} " + (f"(the rest of the way from {sub['after']})" if sub.get("after")
                                                else str(sub["scores"][sub["name"]])))
        # 3. probes
        status, deep = {}, []
        if rule in ("random", "curated"):
            ra = rb = []
        else:
            ra = run_jobs("probe_a", [job(c, lib, kinds, B["probe_a"], "probe_a", k, k2) for c in pool])
            todo = [i for i, r in enumerate(ra) if r is not None and r["found_cost"] is None]
            rb = run_jobs("probe_b", [job(pool[i], lib, kinds, B["probe_b"], "probe_b", k, k2) for i in todo])
            res = list(ra)
            for i, r in zip(todo, rb):
                res[i] = r
            todo = [i for i, r in enumerate(res) if pool[i].name in subs and r is not None and r["found_cost"] is None]
            deep = run_jobs("probe_deep", [job(pool[i], lib, kinds, cfg["subgoal"]["deep"], "probe_deep", k, k2)
                                           for i in todo]) if todo else []    # subgoal: a deeper probe
            for i, r in zip(todo, deep):
                if r is not None:
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
            pick = sorted(learnable, key=lambda c: (-status[c.name]["cost"], c.name not in tsubs))[:K]
        elif rule == "nov":
            pl = {t: dsl.parse(v) for t, v in lib.items()}
            outs = [brm.evaluate((t,), pl, tgt.xs, tgt.Ns, machine_for(k))[0] for t in list(brm.PRIMS) + list(lib)]
            for c in learnable:
                got = brm.evaluate(dsl.parse(status[c.name]["best"]), pl, tgt.xs, tgt.Ns, machine_for(k))[0]
                scores[c.name] = float(min(int((got != o).sum()) for o in outs))
            pick = sorted(learnable, key=lambda c: (-scores[c.name], status[c.name]["cost"], c.name not in tsubs))[:K]
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
                pick = sorted(learnable, key=lambda c: (-scores.get(c.name, -1e9), c.name not in tsubs))[:K]
            else:
                cand = learnable
                if cfg.get("pair_pool"):
                    cand = sorted(learnable, key=lambda c: (-scores.get(c.name, -1e9), c.name not in tsubs))[:cfg["pair_pool"]]
                    learnable = cand
                combos = list(itertools.combinations(range(len(learnable)), 2))
                if cfg.get("anchor") or cfg.get("anchor_cost"):          # scale: cheap.py's patch, built in
                    if cfg.get("anchor"):
                        top = sorted(learnable, key=lambda c: (-scores.get(c.name, -1e9), c.name not in tsubs))[:cfg["anchor"]]
                    else:
                        top = sorted(learnable, key=lambda c: (-status[c.name]["cost"], c.name not in tsubs))[:cfg["anchor_cost"]]
                    top = {c.name for c in top}
                    combos = [(a, b) for a, b in combos if learnable[a].name in top or learnable[b].name in top]
                rl2 = run_jobs("look", [job(tgt, with_routines([learnable[a], learnable[b]]), kinds, B["look"], "look",
                                            k, k2) for a, b in combos])
                for (a, b), r in zip(combos, rl2):
                    if r is not None:
                        pairs.append((mdl0 - mdl(r, n_sym(lib) + 2, k), learnable[a].name, learnable[b].name,
                                      r["found_cost"]))
                pairs.sort(key=lambda p: (-p[0], not tsubs & {p[1], p[2]}))
                chosen = []
                for g, a, b, _ in pairs:
                    for nm in (a, b):
                        if nm not in chosen and len(chosen) < K:
                            chosen.append(nm)
                    if len(chosen) >= K:
                        break
                for c in sorted(learnable, key=lambda c: (-scores.get(c.name, -1e9), c.name not in tsubs)):
                    if len(chosen) < K and c.name not in chosen:
                        chosen.append(c.name)
                pick = [c for c in learnable if c.name in chosen]
                pick.sort(key=lambda c: chosen.index(c.name))
        else:
            raise ValueError(rule)
        if cfg.get("subgoal", {}).get("pick") and subs and subs & {c.name for c in learnable} - {c.name for c in pick}:
            pick = pick[:K - 1] + [c for c in learnable if c.name in subs]   # subgoal: practised when learnable
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
        chained += [f"T.{stems[c.name]}" for c in solved_now if c.name in subs and f"T.{stems[c.name]}" in lib]
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
                  subgoal=dict(name=sub["name"], after=sub.get("after"),
                               scores=dict(sorted(sub["scores"].items(), key=lambda kv: kv[1]["key"], reverse=True)[:20]),
                               deep=[r for r in deep if r is not None]),
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


def fresh(name, cfg, tname, search_many, emit, log=print):
    """bar3/scale/drive.fresh with this run."""
    cfg = dict(cfg, target=tname)                                                    # digits
    tap = DigitsTap(search_many)
    TG.use(tname)
    tap.target = tname
    run(name, cfg, tap.mapper, tap.emit_with_looks(lambda r: emit(dict(r, target_name=tname))), log=log)


class DigitsTap(Tap):
    """bar3/breadth/drive.Tap, with every job on the target (all are named "target") tagged with the target's name:
    jobs.py runs those on the target's machines, with the check at big x."""

    def mapper(self, jobs):
        return super().mapper([dict(j, digits_target=self.target) if j["name"] == "target" else j for j in jobs])
