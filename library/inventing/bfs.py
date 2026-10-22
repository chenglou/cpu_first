"""Breadth-first program search by cost on the bare machine (the library study's search.py, one probe set).

Level d holds every distinct machine state (over all probes) that some program of cost d reaches and no cheaper one
does; level d is built by applying every item of cost c to level d - c. Every minimal-cost parent is kept, so all
minimal programs can be read back. A level that would pass the budget is finished streaming (goal checks only) and the
search stops after it. `accept(programs)` filters goal programs (aliases are noted and the search goes on).
Items are the tokens plus blocks over bodies of cost <= max_body, merged by what they do to random machines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from library import dsl
from library.inventing import brm
from library.search import hash_rows, _fresh



def keys_of(b):
    return hash_rows(b.features())


@dataclass
class ItemClass:
    rep: object
    cost: int
    members: list = field(default_factory=list)


def _sig(m, tests, seq, lib):
    b = tests.copy()
    brm.run_items(m, b, seq, lib)
    rows = keys_of(b)
    rows = np.where(b.dead, np.uint64(7), rows)
    return int(hash_rows(rows.view(np.uint8).reshape(1, -1))[0])


def enumerate_items(tokens, lib, kinds, m, max_body=2, max_arity=3, seed=7, S_test=24, NP_test=6):
    rng = np.random.default_rng(seed)
    tests = brm.random_states(S_test, NP_test, rng)
    ident = _sig(m, tests, (), lib)
    classes, by_sig = [], {ident: None}

    def add(item, c):
        s = _sig(m, tests, (item,), lib)
        if s in by_sig:
            cl = by_sig[s]
            if cl is not None and cl.cost == c:
                cl.members.append(item)
            return
        cl = ItemClass(item, c, [item])
        by_sig[s] = cl
        classes.append(cl)

    for t in tokens:
        add(t, 1)
    body_sig = {ident}
    for bc in range(1, max_body + 1):
        found = []
        reps = list(classes)

        def extend(prefix, pc, n):
            if pc == bc:
                s = _sig(m, tests, prefix, lib)
                if s not in body_sig:
                    body_sig.add(s)
                    found.append(prefix)
                return
            if n == max_arity:
                return
            for cl in reps:
                if pc + cl.cost <= bc:
                    extend(prefix + (cl.rep,), pc + cl.cost, n + 1)

        extend((), 0, 0)
        for body in found:
            for kind in kinds:
                if kind in lib and (len(body) != 1 or isinstance(body[0], tuple)):
                    continue
                add(("REP", kind, body), bc + 1)
    return classes


@dataclass
class Problem:
    xs: np.ndarray
    Ns: np.ndarray
    want: np.ndarray

    def __post_init__(self):
        self.root = brm.fresh(self.xs, self.Ns)

    def goal(self, b):
        return (b.regs[:, :, brm.A] == self.want[None]).all(axis=1) & ~b.dead


@dataclass
class Result:
    found_cost: int = None
    goals: list = None
    level_sizes: list = field(default_factory=list)
    seconds: float = 0.0
    stop: str = ""
    parents: list = None
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
    if prob.goal(prob.root)[0]:
        res.found_cost, res.goals, res.parents = 0, [0], parents
        return res
    for d in range(1, max_cost + 1):
        last = d == max_cost
        ck, cc, cs, ci_, chunks, gk = [], [], [], [], [], []
        n_cand = n_fresh = 0
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
                g = prob.goal(sub)
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
    return res


def programs(res, classes, limit=200):
    memo = {}

    def paths(level, idx):
        if level == 0:
            return [()]
        key = (level, idx)
        if key in memo:
            return memo[key]
        out = []
        child, pcl, psl, psi = res.parents[level]
        lo, hi = np.searchsorted(child, idx), np.searchsorted(child, idx, side="right")
        for ci, sl, si in zip(pcl[lo:hi].tolist(), psl[lo:hi].tolist(), psi[lo:hi].tolist()):
            for p in paths(sl, si):
                out.append(p + (classes[ci].rep,))
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        memo[key] = out
        return out

    progs = []
    for g in res.goals:
        progs.extend(paths(res.found_cost, g)[:limit - len(progs)])
        if len(progs) >= limit:
            break
    return progs
