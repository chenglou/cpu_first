"""The inventing study's breadth-first search (library/inventing/bfs.py), plus what a learner needs to judge its own
practice: how close each level came to the goal, and the near misses themselves.

Level d holds every distinct machine state (over all probes) that some program of cost d reaches and no cheaper one
does. On top of bfs.bfs this records, per level, the most probes any state there gets right (`best_fit`), and keeps
the held levels so that the best near misses (states whose A is right on many probes) can be read back as programs.
A level that would pass the budget is streamed (goal checks and fits only) and the search stops after it, as in bfs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from library import dsl
from library.inventing import brm
from library.inventing.bfs import enumerate_items, keys_of, programs  # noqa: F401  (re-exported for callers)
from library.search import _fresh


@dataclass
class Problem:
    xs: np.ndarray
    Ns: np.ndarray
    want: np.ndarray

    def __post_init__(self):
        self.root = brm.fresh(self.xs, self.Ns)

    def fits(self, b):
        return (b.regs[:, :, brm.A] == self.want[None]).sum(axis=1) * ~b.dead

    def goal(self, b):
        return self.fits(b) == len(self.want)


@dataclass
class Result:
    found_cost: int = None
    goals: list = None
    level_sizes: list = field(default_factory=list)
    best_fit: list = field(default_factory=list)       # per level: most probes right by any state of that level
    seconds: float = 0.0
    stop: str = ""
    parents: list = None
    levels: list = None
    expanded: int = 0
    aliases: list = field(default_factory=list)
    accepted: list = None


def bfs(prob: Problem, classes, lib, m, max_cost=10, max_states=1_000_000, time_limit=600.0, accept=None, limit=200,
        slice_size=16384):
    t0 = time.time()
    levels = [prob.root]
    seen = np.sort(keys_of(prob.root))
    parents = [None]
    res = Result()
    res.level_sizes.append(1)
    res.best_fit.append(int(prob.fits(prob.root)[0]))
    if prob.goal(prob.root)[0]:
        res.found_cost, res.goals, res.parents, res.levels = 0, [0], parents, levels
        return res
    for d in range(1, max_cost + 1):
        last = d == max_cost
        ck, cc, cs, ci_, chunks, gk = [], [], [], [], [], []
        n_cand = n_fresh = 0
        best = 0
        for ci, cl in enumerate(classes):
            sl = d - cl.cost
            if sl < 0 or levels[sl] is None:
                continue
            for lo in range(0, levels[sl].S, slice_size):
                src = levels[sl].take(slice(lo, lo + slice_size))
                out = brm.apply(m, src, cl.rep, lib)
                res.expanded += src.S
                alive = np.nonzero(~out.dead)[0]
                if not len(alive):
                    continue
                out = out.take(alive)
                keys = keys_of(out)
                fm = _fresh(seen, keys)
                if not fm.any():
                    continue
                idx = np.nonzero(fm)[0]
                n_fresh += len(idx)
                sub = out.take(idx)
                f = prob.fits(sub)
                best = max(best, int(f.max()))
                g = f == len(prob.want)
                if g.any():
                    gk.append(keys[idx][g])
                if last:
                    if not g.any():
                        continue
                    idx, sub = idx[g], sub.take(np.nonzero(g)[0])
                ck.append(keys[idx])
                cc.append(np.full(len(idx), ci, np.int32))
                cs.append(np.full(len(idx), sl, np.int16))
                ci_.append(alive[idx].astype(np.int64) + lo)
                chunks.append(sub)
                n_cand += len(idx)
                if not last and n_cand > 1.5 * max_states:
                    last = True
                    res.stop = f"budget: level {d} streamed (over {max_states} states)"
                    keep = [(j, np.nonzero(prob.goal(chunks[j]))[0]) for j in range(len(chunks))]
                    keep = [(j, gm) for j, gm in keep if len(gm)]
                    ck = [ck[j][gm] for j, gm in keep]
                    cc = [cc[j][gm] for j, gm in keep]
                    cs = [cs[j][gm] for j, gm in keep]
                    ci_ = [ci_[j][gm] for j, gm in keep]
                    chunks = [chunks[j].take(gm) for j, gm in keep]
            if time.time() - t0 > time_limit:
                res.stop = f"time: {time_limit}s at level {d}"
                break
        if ck:
            allk = np.concatenate(ck)
            uk, first, inv = np.unique(allk, return_index=True, return_inverse=True)
            sfirst = np.sort(first)
            rank = np.searchsorted(sfirst, first)
            child = rank[inv]
            order = np.argsort(child, kind="stable")
            parents.append((child[order], np.concatenate(cc)[order], np.concatenate(cs)[order],
                            np.concatenate(ci_)[order]))
            if not last and not res.stop:
                parts, off = [], 0
                for ch in chunks:
                    n = ch.S
                    a, b_ = np.searchsorted(sfirst, off), np.searchsorted(sfirst, off + n)
                    parts.append(ch.take(sfirst[a:b_] - off))
                    off += n
                levels.append(brm.Batch.concat(parts))
            else:
                levels.append(None)
        else:
            uk = np.zeros(0, np.uint64)
            rank = np.zeros(0, np.int64)
            parents.append(None)
            levels.append(None)
        res.level_sizes.append(int(len(uk)) if not last else int(n_fresh))
        res.best_fit.append(best)
        if gk:
            g = np.unique(np.concatenate(gk))
            res.found_cost = d
            res.goals = rank[np.searchsorted(uk, g)].tolist()
            res.parents = parents
            if accept is None:
                break
            progs = programs(res, classes, limit=limit)
            ok = accept(progs)
            if any(ok):
                res.accepted = [p for p, o in zip(progs, ok) if o]
                break
            res.aliases.append((d, [dsl.show(p) for p in progs[:20]]))
            res.found_cost, res.goals = None, None
        if res.stop:
            break
        seen = np.union1d(seen, uk)
    res.seconds = time.time() - t0
    if res.found_cost is None and not res.stop:
        res.stop = f"exhausted cost {max_cost}"
    res.parents = parents
    res.levels = levels
    return res


def programs_at(res, classes, level, idx, limit=1):
    """Programs reaching state idx of a held level (the same read-back as bfs.programs, for any state)."""
    saved = res.found_cost, res.goals
    res.found_cost, res.goals = level, [int(idx)]
    try:
        return programs(res, classes, limit=limit)
    finally:
        res.found_cost, res.goals = saved


def near_misses(res, prob, classes, n=12, min_fit=1):
    """The best near misses of the held levels: one program per distinct output (A over the probes), most probes right
    first, then the cheapest. Returns [(fit, cost, program, outputs)]."""
    cands = []
    for d, b in enumerate(res.levels or []):
        if b is None:
            continue
        f = prob.fits(b)
        if not len(f):
            continue
        top = np.argsort(-f, kind="stable")[: 50 * n]
        for i in top:
            if f[i] < min_fit:
                break
            cands.append((int(f[i]), d, int(i)))
    cands.sort(key=lambda c: (-c[0], c[1]))
    out, seen = [], set()
    for fit, d, i in cands:
        outs = tuple(res.levels[d].regs[i, :, brm.A].tolist())
        if outs in seen:
            continue
        seen.add(outs)
        progs = programs_at(res, classes, d, i, limit=1)
        if progs:
            out.append((fit, d, progs[0], outs))
        if len(out) == n:
            break
    return out
