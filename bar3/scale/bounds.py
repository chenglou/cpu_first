"""Bounds that might shrink the bar-3 search space (SCALE.md §8), measured locally on the compiled engine (each part
under a minute and 1 GB). The search already caps values (2k², 98 at N <= 7), drops a machine after 50k² + 1000
steps, merges items that do the same to 24 random machines, and merges programs that reach the same state. What is
left to bound:

1. `probes`: how the level sizes fall when the search looks at fewer instances (the premise of counterexample-
   guided search): x² mod N's target search on 28, 14, 7, 4 and 2 of its instances.
2. `cegis`: counterexample-guided search on the learner's own problems (every third of x² mod N's round-0 pool at the
   30k-state probe, and a fifth of those it leaves unsolved at 150k): search on 4 instances, check each program
   found on every instance of both sizes, add an instance where the first rejected program fails, and search again;
   against the plain search on the same problems: the cost found, the routine kept (learn.choose), states, seconds.
3. `merges`: on the plain search's levels, how many states remain when states that differ only by swapping the two
   scratch registers T and U are merged, and how many distinct answers (register A) a level holds.
4. `inverse`: for meet in the middle, how many of the search's items are one to one on a level's states.
5. `cap`: the level sizes with a lower value cap (not a sound bound: it changes what programs compute).

    cd cpu_first && <python with numba> -m bar3.scale.bounds [part ...]      # writes records/bounds_<part>.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from library import dsl
from library.inventing import brm
from library.inventing.jobs import machine_for
from bar3 import learner as L, problems as P, search
from bar3.scale import engine, learn
from bar3.scale.jobs import search_job

HERE = Path(__file__).resolve().parent
K = 7
KINDS = list(brm.LOOP_KINDS)


def base_classes(m):
    classes = engine.enumerate_items(list(brm.PRIMS), {}, KINDS, m)
    return classes, engine.Codes(classes, {})


def spread(n, k):
    """k instance indices spread over 0..n-1, the last (largest N) included."""
    return sorted({int(round(i)) for i in np.linspace(n - 1, 0, k)})


def part_probes():
    tgt = P.the_target(K)
    m = machine_for(K)
    classes, codes = base_classes(m)
    xs, Ns, want = (np.array(v) for v in (tgt.xs, tgt.Ns, tgt.want))
    out = []
    for n in (28, 14, 7, 4, 2):
        idx = spread(len(xs), n)
        prob = search.Problem(xs[idx], Ns[idx], want[idx])
        t = time.time()
        res = engine.bfs(prob, classes, {}, m, max_cost=10, max_states=300_000, time_limit=50, codes=codes)
        out.append(dict(probes=n, levels=res.level_sizes, stop=res.stop, seconds=round(time.time() - t, 2)))
        print(out[-1])
    return out


def run_cegis(cand, budget, start=4, rounds=12):
    """Search on `start` of cand's instances; each rejected program's first failing instance (of every instance
    at size K, then at the check size) joins the probes, until a program passes or the budget stops the search."""
    xs, Ns, want = cand.xs, cand.Ns, cand.want
    probes = spread(len(xs), min(start, len(xs)))
    total_states, total_s, history = 0, 0.0, []
    pl = {}
    for r in range(rounds):
        j = L.job(cand, {}, KINDS, budget, "probe", K)
        j.update(xs=[xs[i] for i in probes], Ns=[Ns[i] for i in probes], want=[want[i] for i in probes])
        rec = search_job(j)
        total_states += sum(rec["level_sizes"])
        total_s += rec["wall"]
        history.append(dict(probes=len(probes), found=rec["found_cost"], levels=rec["level_sizes"]))
        if rec["found_cost"] is not None or not rec["aliases"]:
            return rec, total_states, total_s, history
        bad = dsl.parse(rec["aliases"][-1][1][0])
        got, _ = brm.evaluate(bad, pl, np.array(xs), np.array(Ns), machine_for(K))
        miss = [i for i in range(len(xs)) if got[i] != want[i] and i not in probes]
        if not miss:      # wrong only at the check size: the search cannot see that instance, give up
            return rec, total_states, total_s, history
        probes = sorted(probes + [miss[0]])
    return rec, total_states, total_s, history


def part_cegis():
    from bar3.scale.check import pool_for
    tgt, pool, _ = pool_for("h1")
    rows = []
    for budget_name, budget, cands in (("probe A, 30k states", L.PROBE_A, pool[::3]),
                                       ("150k states", dict(L.PROBE_B, max_states=150_000), None)):
        if cands is None:   # the problems the plain 30k-state probe left unsolved, a sample of them
            cands = [c for c in pool if c.name in {r["name"] for r in rows if r["plain_found"] is None}][::5]
        for c in cands:
            plain = search_job(L.job(c, {}, KINDS, budget, "probe", K))
            cg, states, secs, hist = run_cegis(c, budget)
            keep = lambda r: (learn.choose([p["program"] for p in r["programs"]], {}, c, K, "clean")[0]   # noqa: E731
                              if r.get("programs") else None)
            rows.append(dict(budget=budget_name, name=c.name, plain_found=plain["found_cost"],
                             plain_states=sum(plain["level_sizes"]), plain_s=plain["wall"],
                             cegis_found=cg["found_cost"], cegis_states=states, cegis_s=round(secs, 2),
                             restarts=len(hist) - 1, final_probes=hist[-1]["probes"],
                             same_routine=keep(plain) == keep(cg), plain_routine=keep(plain), cegis_routine=keep(cg)))
        sub = [r for r in rows if r["budget"] == budget_name]
        both = [r for r in sub if r["plain_found"] is not None and r["cegis_found"] is not None]
        print(budget_name, len(sub), "problems; plain found", sum(r["plain_found"] is not None for r in sub),
              "; cegis found", sum(r["cegis_found"] is not None for r in sub),
              "; same cost", sum(r["plain_found"] == r["cegis_found"] for r in both), "of", len(both),
              "; same routine", sum(r["same_routine"] for r in both),
              "; states", sum(r["plain_states"] for r in sub), "->", sum(r["cegis_states"] for r in sub),
              "; seconds", round(sum(r["plain_s"] for r in sub), 1), "->", round(sum(r["cegis_s"] for r in sub), 1))
    return rows


def plain_levels(budget=300_000):
    tgt = P.the_target(K)
    m = machine_for(K)
    classes, codes = base_classes(m)
    prob = search.Problem(np.array(tgt.xs), np.array(tgt.Ns), np.array(tgt.want))
    res = engine.bfs(prob, classes, {}, m, max_cost=10, max_states=budget, time_limit=50, codes=codes)
    return res, classes, codes, m


def part_merges():
    res, _, _, _ = plain_levels()
    out = []
    for d, b in enumerate(res.levels):
        if b is None:
            continue
        regs = b.regs
        swapped = regs[:, :, [0, 1, 2, 4, 3]]
        a = regs.reshape(len(regs), -1)
        s = swapped.reshape(len(regs), -1)
        first_diff = np.argmax(a != s, axis=1)          # canonical: the lexicographically smaller of state and swap
        a_smaller = a[np.arange(len(a)), first_diff] <= s[np.arange(len(a)), first_diff]
        canon = np.where(a_smaller[:, None], a, s)
        n_tu = len(np.unique(canon, axis=0))
        n_a = len(np.unique(regs[:, :, brm.A], axis=0))
        n_axn = len(np.unique(regs[:, :, [brm.X, brm.N, brm.A]].reshape(len(regs), -1), axis=0))
        out.append(dict(level=d, states=int(b.S), tu_merged=int(n_tu), distinct_A=int(n_a), distinct_XNA=int(n_axn)))
        print(out[-1])
    return out


def part_inverse():
    res, classes, codes, m = plain_levels(30_000)
    src = res.levels[3]
    rows = []
    for ci, cl in enumerate(classes):
        regs, drop, keys = engine.apply_classes(codes, [ci], src.regs, m)
        alive = ~drop | (regs == src.regs).all(axis=(1, 2))        # an unchanged state is an output too
        outs = regs[alive].reshape(int(alive.sum()), -1)
        n_out = len(np.unique(outs, axis=0)) if len(outs) else 0
        rows.append((cl.cost, int(alive.sum()), n_out))
    one_to_one = sum(1 for _, n, u in rows if n == u)
    by_cost = {c: (sum(1 for cc, n, u in rows if cc == c and n == u), sum(1 for cc, _, _ in rows if cc == c))
               for c in sorted({r[0] for r in rows})}
    collapse = np.mean([n / max(u, 1) for _, n, u in rows])
    out = dict(level=3, states=int(src.S), classes=len(rows), one_to_one=one_to_one, by_cost=by_cost,
               mean_states_per_output=round(float(collapse), 2))
    print(out)
    return out


def part_cap():
    tgt = P.the_target(K)
    out = []
    for cap in (98, 49, 20):
        m = brm.Machine(cap=cap, limit=machine_for(K).limit)
        classes, codes = base_classes(m)
        prob = search.Problem(np.array(tgt.xs), np.array(tgt.Ns), np.array(tgt.want))
        t = time.time()
        res = engine.bfs(prob, classes, {}, m, max_cost=10, max_states=300_000, time_limit=50, codes=codes)
        out.append(dict(cap=cap, classes=len(classes), levels=res.level_sizes, seconds=round(time.time() - t, 2)))
        print(out[-1])
    return out


PARTS = dict(probes=part_probes, cegis=part_cegis, merges=part_merges, inverse=part_inverse, cap=part_cap)


def main():
    for name in sys.argv[1:] or list(PARTS):
        t = time.time()
        res = PARTS[name]()
        (HERE / "records" / f"bounds_{name}.json").write_text(json.dumps(res, indent=1))
        print(f"== {name}: {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
