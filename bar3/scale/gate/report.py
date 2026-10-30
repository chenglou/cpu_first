"""The measures of PREREG.md from the results (GATE.md's tables): states to solution, reach (slope and the plain-
unsolved), the targets at 3M, per corpus, fold and arm.

    cd cpu_first && python3 -m bar3.scale.gate.report CORPUS RESULTS_DIR CAPTURE.jsonl.gz ...

States to solution: the first state with an accepted program, counted as candidates generated (gengine). An arm that
finds nothing within its cap is censored at the cap: its ratio plain/arm is at most plain/cap, counted as that bound
(so the geometric means are upper bounds for arms that miss). The slope: least squares of log10(states to solution)
on the plain search's cost (its program's tokens), over problems of cost 3 or more where the arm found it.
"""

from __future__ import annotations

import collections
import glob
import json
import math
import sys

import numpy as np

from bar3.scale.gate.tasks import load, plain_outcome
from bar3.scale.gate.train import FOLDS, SHARED, target_of


def results(dirpath, corpus):
    out = {}
    for p in glob.glob(f"{dirpath}/*.jsonl"):
        for line in open(p):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            out[(d["id"], d["arm"])] = d
    return out


def fold_of(d):
    ts = {target_of(r) for r in d["runs"]}
    for f in ("A", "B"):
        if ts <= set(FOLDS[f]):
            return f
    return None


def summarize(corpus, probs, res, log=print):
    arms = sorted({a for _, a in res})
    rows = []
    for f in ("A", "B"):
        test = [d for d in probs.values() if d["source"] not in SHARED and d["source"] != "target" and fold_of(d) == f]
        solved = [d for d in test if plain_outcome(d)["solved"] and plain_outcome(d)["cost"] >= 3]
        unsolved = [d for d in test if not plain_outcome(d)["solved"]]
        log(f"\n{corpus} test fold {f} (gate trained on the other): {len(test)} problems, {len(solved)} solved by plain "
            f"at cost >= 3, {len(unsolved)} unsolved")
        for arm in arms:
            ratios, faster, n, hits = [], 0, 0, collections.defaultdict(list)
            for d in solved:
                r = res.get((d["id"], arm))
                if r is None or "error" in r:
                    continue
                po = plain_outcome(d)
                n += 1
                cap = r["level_sizes"] and r.get("generated")
                st = r.get("first_accept") if r.get("found_cost") is not None else None
                arm_states = st if st else max(cap or 1, 1)
                ratios.append(po["states"] / arm_states)
                faster += bool(st and st < po["states"])
                if st:
                    hits[po["cost"]].append(math.log10(st))
            if not n:
                continue
            gm = math.exp(np.mean(np.log(ratios)))
            found = sum(len(v) for v in hits.values())
            slope = _slope(hits)
            un = [res.get((d["id"], arm)) for d in unsolved]
            un = [r for r in un if r and "error" not in r]
            un_hit = sum(1 for r in un if r.get("found_cost") is not None)
            rows.append(dict(corpus=corpus, fold=f, arm=arm, n=n, found=found, gmean=gm, faster=faster / n,
                             slope=slope, per_tenfold=(1 / slope if slope else None), n_un=len(un), un_hit=un_hit))
            log(f"  {arm:>10}: solved-by-plain {n}: found {found}, geo-mean plain/arm {gm:6.2f}, faster on "
                f"{faster / n:5.1%}; slope {slope if slope is None else round(slope, 3)} decades/instruction"
                f" ({'-' if not slope else round(1 / slope, 2)} instructions per tenfold); plain-unsolved: "
                f"{un_hit}/{len(un)} found")
        ps = {c: [math.log10(plain_outcome(d)["states"]) for d in solved if plain_outcome(d)["cost"] == c]
              for c in sorted({plain_outcome(d)["cost"] for d in solved})}
        sl = _slope(ps)
        log(f"  {'plain':>10}: slope {sl if sl is None else round(sl, 3)} decades/instruction "
            f"({'-' if not sl else round(1 / sl, 2)} per tenfold); by cost: "
            + ", ".join(f"{c}: {len(v)} median {10 ** np.median(v):,.0f}" for c, v in ps.items()))
    return rows


def matched(corpus, probs, res, arms=None, min_cost=2, cap=5_000_000, seed=0, log=print):
    """Every arm on the same problems: the test problems of each fold solved by the plain search (cost >= min_cost)
    with a result for every arm. Per arm: the geometric mean of plain/arm and the share faster; the same by the plain
    program's cost; and the reach, measure 3 on these problems: the least-squares slope of log10(states to solution)
    on the plain program's cost for the plain search and for the arm (1/slope: instructions per tenfold of states),
    and beta, the arm's slope over the plain search's, with a 90% bootstrap interval over problems. beta < 1: the arm's
    advantage grows by a constant factor per instruction, so per tenfold of states it reaches 1/beta of the plain
    search's instructions; beta = 1: a constant factor. An arm that misses within its cap (20 times the plain search's
    states) counts at the cap, which flatters it; the misses are counted.

    The cost is on the x axis, not log(plain states): the plain count also carries where the solution sits inside its
    level (up to a factor of b), and that noise, independent of the arm's, would pull a log-log slope below 1 for a
    mere constant factor.

    Only problems whose plain states to solution are within the arms' cap (5M) are compared: a 1M probe streams its
    last level, so the plain search's own count reaches tens of millions, beyond any arm's cap, where a miss says
    nothing. Those are reported apart: the share each arm solved within the cap (so with at least plain/cap times fewer
    states)."""
    arms = arms or sorted({a for _, a in res} - {"plain"})
    rng = np.random.default_rng(seed)
    rows = []
    for f in ("A", "B", None):
        test = [d for d in probs.values() if d["source"] not in SHARED and d["source"] != "target"
                and fold_of(d) and (f is None or fold_of(d) == f)
                and plain_outcome(d)["solved"] and plain_outcome(d)["cost"] >= min_cost]
        test = [d for d in test if all("error" not in res.get((d["id"], a), {"error": 1}) for a in arms)]
        hard = [d for d in test if plain_outcome(d)["states"] > cap]
        test = [d for d in test if plain_outcome(d)["states"] <= cap]
        if not test:
            continue
        cost = np.array([plain_outcome(d)["cost"] for d in test], float)
        yp = np.array([math.log10(plain_outcome(d)["states"]) for d in test])
        sp = _slope_xy(cost, yp)
        log(f"\n{corpus} matched, fold {f or 'A+B'}: {len(test)} test problems solved by plain (cost >= {min_cost}), "
            f"every arm on each; plain slope {sp:.3f} decades per instruction ({1 / sp:.2f} instructions per tenfold)")
        costs = sorted(set(cost.tolist()))
        for arm in arms:
            ya, miss, faster = [], 0, 0
            for d in test:
                po, r = plain_outcome(d), res[(d["id"], arm)]
                st = r.get("first_accept") if r.get("found_cost") is not None else None
                miss += not st
                faster += bool(st and st < po["states"])
                ya.append(math.log10(st or max(r.get("generated") or 1, 1)))
            ya = np.array(ya)
            lr = yp - ya
            sa = _slope_xy(cost, ya)
            boot = []
            for _ in range(2000):
                i = rng.integers(0, len(test), len(test))
                if np.ptp(cost[i]) > 0:
                    boot.append(_slope_xy(cost[i], ya[i]) / _slope_xy(cost[i], yp[i]))
            lo, hi = np.percentile(boot, [5, 95])
            row = dict(corpus=corpus, fold=f or "A+B", arm=arm, n=len(test), missed=miss, gmean=10 ** lr.mean(),
                       faster=faster / len(test), slope_plain=sp, slope=sa, beta=sa / sp, beta90=(lo, hi),
                       by_cost={c: (int((cost == c).sum()), 10 ** lr[cost == c].mean()) for c in costs})
            rows.append(row)
            log(f"  {arm:>10}: geo-mean plain/arm {row['gmean']:6.2f}, faster on {row['faster']:5.1%}, missed {miss}; "
                f"slope {sa:.3f} ({1 / sa:.2f} instructions per tenfold), beta {sa / sp:.3f} [{lo:.3f}, {hi:.3f}]; "
                f"by cost: " + ", ".join(f"{c:.0f}: {n} x{g:.2f}" for c, (n, g) in row["by_cost"].items()))
        if hard:
            hc = collections.Counter(plain_outcome(d)["cost"] for d in hard)
            log(f"  beyond the cap: {len(hard)} problems the plain search needed more than {cap:,} states for (costs "
                f"{dict(sorted(hc.items()))}, median {np.median([plain_outcome(d)['states'] for d in hard]):,.0f}); "
                f"solved within the cap: " + ", ".join(
                    f"{a} {sum(res[(d['id'], a)].get('found_cost') is not None for d in hard)}" for a in arms))
            for row in rows[-len(arms):]:
                row["hard"] = (len(hard), sum(res[(d["id"], row["arm"])].get("found_cost") is not None for d in hard))
    return rows


def _slope_xy(x, y):
    return float(np.polyfit(x, y, 1)[0])


def reach_table(corpus, probs, res, budgets=(30_000, 300_000, 3_000_000), log=print):
    """Reach at a fixed budget: for each arm, the share of test problems of each plain cost solved within B states, and
    the largest cost up to which at least half of each cost's problems are solved (the plain search's own states for
    the plain arm)."""
    arms = sorted({a for _, a in res} | {"plain"})
    test = [d for d in probs.values() if d["source"] not in SHARED and d["source"] != "target" and fold_of(d)
            and plain_outcome(d)["solved"] and plain_outcome(d)["cost"] >= 2]
    costs = sorted({plain_outcome(d)["cost"] for d in test})
    log(f"\n{corpus} reach at a fixed budget (test problems of both folds solved by plain, by plain cost; share solved "
        f"within B):")
    for B in budgets:
        line = []
        for arm in arms:
            cells, reach = [], None
            for c in costs:
                ds = [d for d in test if plain_outcome(d)["cost"] == c]
                hit = n = 0
                for d in ds:
                    if arm == "plain":
                        st = plain_outcome(d)["states"]
                    else:
                        r = res.get((d["id"], arm))
                        if r is None or "error" in r:
                            continue
                        st = r.get("first_accept") if r.get("found_cost") is not None else None
                    n += 1
                    hit += bool(st and st <= B)
                if n:
                    cells.append(f"{c}:{hit}/{n}")
                    if hit * 2 >= n and reach == (costs[costs.index(c) - 1] if costs.index(c) else None):
                        reach = c
            line.append(f"{arm} reach {reach} [{' '.join(cells)}]")
        log(f"  B={B:,}: " + "; ".join(line))


def _slope(by_cost):
    xs, ys = [], []
    for c, v in by_cost.items():
        for y in v:
            xs.append(c)
            ys.append(y)
    if len(set(xs)) < 2:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def beyond(corpus, probs, res, log=print):
    """Programs an arm found where the plain search found none: their length in the search's tokens (the plain
    search's cost), against the states the arm spent, by arm."""
    from library import dsl
    log(f"\n{corpus} found where the plain search found nothing (the test problems its 1M probe left unsolved; tokens of "
        f"the found program):")
    for arm in sorted({a for _, a in res} - {"plain"}):
        hits = []
        for d in probs.values():
            if d["source"] in SHARED or d["source"] == "target" or not fold_of(d) or plain_outcome(d)["solved"]:
                continue
            r = res.get((d["id"], arm))
            if r and "error" not in r and r.get("found_cost") is not None and r.get("programs"):
                tok = min(dsl.cost(dsl.parse(p)) for p in r["programs"])
                hits.append((tok, r["first_accept"], d["job"]["name"], r["programs"][0]))
        if hits:
            c = collections.Counter(t for t, *_ in hits)
            log(f"  {arm:>10}: {len(hits)} found; tokens {dict(sorted(c.items()))}; e.g. "
                + "; ".join(f"{n} = {p} ({t} tokens, {s:,} states)" for t, s, n, p in sorted(hits, reverse=True)[:3]))


def price(programs):
    """The plain search's price of the cheapest program (an arm's found_cost is its own level: with a gate, the delays
    are in it; guided's is in bits)."""
    from library import dsl
    return min(sum(dsl.item_cost(it) for it in dsl.parse(p)) for p in programs)


def targets(corpus, probs, res, log=print):
    log(f"\n{corpus} targets at 3M states (the found program's price, and the states to it):")
    for d in sorted((d for d in probs.values() if d["source"] == "target"), key=lambda d: sorted(d["runs"])[0]):
        t = target_of(sorted(d["runs"])[0])
        parts = []
        for arm in ("plain", "gate", "gate_prune", "prune", "guided"):
            r = res.get((d["id"], arm))
            if r is None:
                continue
            found = r.get("found_cost") is not None and r.get("programs")
            parts.append(f"{arm} " + (f"{price(r['programs'])} at {r['first_accept']:,}" if found else
                                      f"no ({r.get('generated') or 0:,})"))
        log(f"  {t:>8}: " + "; ".join(parts))


def main():
    corpus, rdir = sys.argv[1:3]
    probs = load(sys.argv[3:])
    res = results(rdir, corpus)
    summarize(corpus, probs, res)
    matched(corpus, probs, res)
    reach_table(corpus, probs, res)
    beyond(corpus, probs, res)
    targets(corpus, probs, res)


if __name__ == "__main__":
    main()
