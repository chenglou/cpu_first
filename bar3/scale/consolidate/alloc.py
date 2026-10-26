"""One allocator for a round's search and practice (CONSOLIDATE.md §2; PREREG.md). Called by the learner copies
(make_learners.py) in place of the two fixed probes, the subgoal's deeper probe, the choosing rules and their
patches (four picks, ties, the twin, the twin as a fifth pick, the rest's tie-break, the deeper look), when
cfg["alloc"] is on.

1. The census: every candidate's 30k-state probe (the old first probe, the same job).
2. The round's budget R, matched to the old rules: 1M states a candidate the census left unsolved, plus 10M, plus 30k a
   learnable one.
3. Actions: an unsolved candidate's next probe rung (1M, then 10M: the old second probe's and the subgoal's deeper
   probe's jobs), and a learnable one's next look (the target with its routine: 30k, then 1M, the old look's and
   reach's deeper look's jobs). So every search the old runs made is the same job, read from their stores.
4. Values, in bits (tok = log2 of the symbols, exc = one exception's bits):
   - learnable and looked: the measured gain, mdl0 - mdl(its deepest look);
   - learnable, not looked: (cost - 1)·tok, or the subgoal score's bits when more;
   - unsolved: the subgoal score's bits (a law: mdl0 - 3·tok; else the rows a table saves × exc), else
     (the probe's last level - 1)·tok × its best fit's share;
   - candidates whose best programs share a signature are one group: the first in pool order is looked at, with the
     group's best value (a duplicate's value passes to the problem it duplicates).
5. Best first, 32 actions at a time by value per state (ties: the cheaper, then pool order), until R is spent or no
   action has a positive value.
6. Practice: every learnable candidate (the row check's classification, in pool order) whose measured gain exceeds
   its library cost, cost·tok·0.5 (the merge's library weight). No fixed count.

The ablation (PREREG_ADDENDUM1.md) adds old rules back, each behind its own flag, all off by default:
- cfg["alloc_probe_all"]: the old uniform second probe, every candidate the census left unsolved probed at 1M before
  the best-first loop, its states outside R;
- cfg["alloc_pick"] = "lp": the old hardest first in place of step 6, the K costliest learnable candidates (ties: the
  subgoal's, then pool order); = "k": the K learnable candidates of highest value, with no threshold; = "kmax"
  (PREREG_ADDENDUM2.md): the K of highest max(measured gain, prior), so that a null look cannot erase the prior.
"""

from __future__ import annotations

import math

from library import dsl
from library.inventing import brm
from bar3.scale.learn import choose, job
from bar3.scale.rowcheck import rowcheck as RC
from bar3.scale.picks import picks as PK
from bar3.scale.reach.reach import DEEP_LOOK
from bar3.scale.subgoal.configs import DEEP

BATCH = 32
LIB_WEIGHT = 0.5


def _probe_status(c, r, lib, k, how):
    """The old learner's status for a probe result (its classification before the alias and duplicate checks)."""
    probe = dict(states=sum(r["level_sizes"]), seconds=r["wall"], best_fit=r["best_fit"], NP=r["NP"], stop=r["stop"])
    if r["found_cost"] is None:
        return dict(status="unsolved", fit=max(r["best_fit"]) / r["NP"], cost=None, **probe)
    progs = choose([p["program"] for p in r["programs"]], lib, c, k, how)
    return dict(status="trivial" if r["found_cost"] <= 1 else "learnable", cost=r["found_cost"], programs=progs[:20],
                best=progs[0], **probe)


def probe_and_choose(cfg, pool, lib, kinds, B, tgt, tp, k, k2, run_jobs, sigs, sub, mdl0, n_sym, how, log):
    """(census results, second-rung results, third-rung results, status, the allocator's record with its picks)."""
    tok = math.log2(n_sym(lib))
    exc = math.log2(tp["NP"]) + math.log2(k)
    deep_b = (cfg.get("subgoal") or {}).get("deep", DEEP)
    rungs = [("probe_a", B["probe_a"]), ("probe_b", B["probe_b"]), ("probe_deep", deep_b)]
    looks = [B["look"], cfg.get("alloc_look2", DEEP_LOOK)]          # the second look rung (a tiny test may shrink it)
    scores = sub.get("scores", {})

    def sbits(name):
        v = scores.get(name)
        if not v:
            return 0.0
        return max(mdl0 - 3 * tok, 0.0) if v.get("law") else float(v.get("table", 0)) * exc

    ra = run_jobs("probe_a", [job(c, lib, kinds, B["probe_a"], "probe_a", k, k2) for c in pool])
    st = {c.name: (_probe_status(c, r, lib, k, how) if r is not None else dict(status="failed")) for c, r in zip(pool, ra)}
    rung = {c.name: 0 for c in pool}                      # the probe rung done
    lrung, gain, lrec = {}, {}, {}                        # the look rung done, the measured gain, the deepest look
    n_uns = sum(1 for s in st.values() if s["status"] == "unsolved")
    n_lrn = sum(1 for s in st.values() if s["status"] == "learnable")
    R = n_uns * B["probe_b"]["max_states"] + deep_b["max_states"] + n_lrn * B["look"]["max_states"]
    spent, steps, rb, rdeep = 0, [], [], []
    if cfg.get("alloc_probe_all"):                        # ablation: the old uniform second probe, outside R
        todo = [c for c in pool if st[c.name]["status"] == "unsolved"]
        res = run_jobs("probe_b", [job(c, lib, kinds, B["probe_b"], "probe_b", k, k2) for c in todo])
        for c, r in zip(todo, res):
            rung[c.name] = 1
            if r is None:
                continue
            rb.append(r)
            st[c.name] = _probe_status(c, r, lib, k, how)
            steps.append(dict(kind="probe_b", name=c.name, value=None, states=sum(r["level_sizes"]),
                              found=r["found_cost"], uniform=True))

    def groups():
        """{signature: [names in pool order]} of the learnable candidates."""
        out = {}
        for c in pool:
            s = st[c.name]
            if s["status"] == "learnable":
                s.setdefault("sig", sigs.of(dsl.parse(s["best"]), lib))
                out.setdefault(s["sig"], []).append(c.name)
        return out

    def value(name):
        s = st[name]
        if s["status"] == "unsolved":
            b = sbits(name)
            if b > 0:
                return b
            last = len(s.get("best_fit", [])) - 1
            return max(last - 1, 0) * tok * s.get("fit", 0.0)
        if s["status"] == "learnable":
            if name in gain:
                return gain[name]
            return max((s["cost"] - 1) * tok, sbits(name))
        return 0.0

    by_name = {c.name: c for c in pool}
    while spent < R:
        acts = []
        for i, c in enumerate(pool):
            s = st[c.name]
            if s["status"] == "unsolved" and rung[c.name] + 1 < len(rungs):
                kind, b = rungs[rung[c.name] + 1]
                acts.append((value(c.name) / b["max_states"], b["max_states"], i, c.name, "probe", kind, b))
        for sig, names in groups().items():
            head = names[0]
            j = lrung.get(head, 0)
            if j < len(looks):
                v = max(value(n) for n in names)
                acts.append((v / looks[j]["max_states"], looks[j]["max_states"], pool.index(by_name[head]), head,
                             "look", "look", looks[j]))
        acts = [a for a in acts if a[0] > 0]
        if not acts:
            break
        acts.sort(key=lambda a: (-a[0], a[1], a[2]))
        take, room = [], R - spent
        for a in acts:
            if take and a[1] > room:
                break
            take.append(a)
            room -= a[1]
            if len(take) >= BATCH:
                break
        pj = [(a, job(by_name[a[3]], lib, kinds, a[6], a[5], k, k2)) for a in take if a[4] == "probe"]
        lj = []
        for a in take:
            if a[4] == "look":
                lb = dict(lib)
                lb["T.look0"] = st[a[3]]["best"]
                lj.append((a, job(tgt, lb, kinds, a[6], "look", k, k2)))
        for kind in ("probe_b", "probe_deep"):
            sel = [(a, j) for a, j in pj if a[5] == kind]
            if not sel:
                continue
            res = run_jobs(kind, [j for _, j in sel])
            for (a, _), r in zip(sel, res):
                name = a[3]
                rung[name] += 1
                if r is None:
                    continue
                spent += sum(r["level_sizes"])
                (rb if kind == "probe_b" else rdeep).append(r)
                st[name] = _probe_status(by_name[name], r, lib, k, how)
                steps.append(dict(kind=kind, name=name, value=round(a[0] * a[1], 2), states=sum(r["level_sizes"]),
                                  found=r["found_cost"]))
        if lj:
            res = run_jobs("look", [j for _, j in lj])
            for (a, _), r in zip(lj, res):
                name = a[3]
                lrung[name] = lrung.get(name, 0) + 1
                if r is None:
                    continue
                spent += sum(r["level_sizes"])
                gain[name] = mdl0 - PK.mdl(cfg, r, n_sym(lib) + 1, k)
                lrec[name] = dict(found_cost=r["found_cost"], best_fit=r["best_fit"], budget=a[6]["max_states"])
                steps.append(dict(kind="look", name=name, rung=lrung[name], value=round(a[0] * a[1], 2),
                                  states=sum(r["level_sizes"]), gain=round(gain[name], 2), found=r["found_cost"]))
    # the old learner's alias and duplicate checks, in pool order, on the final results
    status, learn_sig = {}, {}
    have = {sigs.of((t,), lib) for t in list(brm.PRIMS) + list(lib)}
    sig_tok, learn_rc = (RC.tokens_by_sig(sigs, lib) if cfg.get("rowcheck") else {}), {}
    for c in pool:
        s = dict(st[c.name])
        s.pop("sig", None)
        if s["status"] == "learnable":
            sg = sigs.of(dsl.parse(s["best"]), lib)
            if cfg.get("rowcheck"):
                s["status"], of = RC.classify(sg, c, lib, sig_tok, learn_rc, status, k, k2)
                if of:
                    s["of"] = of
            elif sg in have:
                s["status"] = "alias"
            elif sg in learn_sig:
                s["status"], s["of"] = "duplicate", learn_sig[sg]
            else:
                learn_sig[sg] = c.name
        if c.name in gain:
            s["gain"], s["look"] = round(gain[c.name], 2), lrec[c.name]
        s["value"] = round(value(c.name), 2)
        status[c.name] = s
    for name, s in status.items():                    # a duplicate's measured gain passes to its learnable original
        if s.get("status") == "duplicate" and "gain" in s and s.get("of") in status:
            o = status[s["of"]]
            if o.get("status") == "learnable" and s["gain"] > o.get("gain", -math.inf):
                o["gain"], o["look"] = s["gain"], s["look"]
    pick = []
    K, lrn = cfg.get("K", 4), [c for c in pool if status[c.name]["status"] == "learnable"]
    if cfg.get("alloc_pick") == "lp":                     # ablation: the old hardest first
        subs = {sub.get("name")} - {None}
        pick = sorted(lrn, key=lambda c: (-status[c.name]["cost"], c.name not in subs))[:K]
    elif cfg.get("alloc_pick") == "k":                    # ablation: the fixed count, by value
        pick = sorted(lrn, key=lambda c: -status[c.name].get("gain", value(c.name)))[:K]
    elif cfg.get("alloc_pick") == "kmax":                 # addendum 2: the fixed count, by max(gain, prior)
        prior = lambda n: max((status[n]["cost"] - 1) * tok, sbits(n))   # noqa: E731
        pick = sorted(lrn, key=lambda c: -max(status[c.name].get("gain", -math.inf), prior(c.name)))[:K]
    else:
        assert not cfg.get("alloc_pick"), cfg["alloc_pick"]
        for c in pool:
            s = status[c.name]
            if s["status"] == "learnable" and "gain" in s and s["gain"] > s["cost"] * tok * LIB_WEIGHT:
                pick.append(c)
        pick.sort(key=lambda c: -status[c.name]["gain"])
    log(f"  alloc: budget {R / 1e6:.1f}M, spent {spent / 1e6:.1f}M in {len(steps)} searches; picks "
        f"{[(c.name, status[c.name].get('gain')) for c in pick]}")
    rec = dict(budget=R, spent=spent, steps=steps, tok=round(tok, 3), exc=round(exc, 3),
               picks=[c.name for c in pick])
    return ra, rb, rdeep, status, dict(pick=pick, scores={n: g for n, g in gain.items()}, record=rec)
