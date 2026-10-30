"""The gated search (GATE.md §2): bar3/scale/engine.bfs, line for line, with one change. Each state a level keeps gets
a delay from the gate (a function of the state's registers on the problem's rows), and it is expanded `delay` levels
later than it would be: a state of level g with delay t sources level d through an item of cost k when d = g + t + k.
A level still holds the states whose cheapest program, so priced, costs d; the read-back, the near misses and the
acceptance test are the original's. Delays are finite, so every state is still expanded: the search is reordered,
never pruned. With no gate (delay None) every delay is 0 and the search is the original (check.py: the same levels,
programs and records).

With `prune`, a state that has become dead is kept in its level (for the read-back) but never expanded: two probes with
the same five registers but different wanted answers. Each probe runs as its own machine and its future depends on its
registers alone, so no program can tell those two apart again: nothing after a dead state can fit every probe. This
drops only states that cannot lead to a solution, so the search stays complete (GATE.md §5).

Three budget rules:
- "level" (the original's): a level whose candidates pass 1.5 * max_states is streamed, and the search stops after it;
- "total": the same for the running total of states generated (every level so far, this one's candidates included);
- "hard": the search stops as soon as the running total passes max_states, inside a level if need be (after the
  chunk that passes it, whose goals are still checked), so that every arm is cut at the same count.

Extra fields on the result, counted the same way with or without a gate (candidates: new states as generated, before
a level's own duplicates are merged):
- `first_goal`: candidates generated when the first state fitting every probe appeared, accepted or not;
- `first_accept`: candidates generated when the first state with an accepted program appeared (the measure of
  "states to solution": the read-back is checked goal state by goal state, in the order they were generated);
- `generated`: candidates generated in all.
"""

from __future__ import annotations

import time

import numpy as np

from library import dsl
from library.inventing import brm
from library.search import _fresh, hash_rows
from bar3.search import Result, programs, programs_at
from bar3.scale import engine as E


def _chunks(classes, size, d):
    """engine._chunks over the source buckets: (class indices, source bucket, lo, hi, whether the chunk ends its
    classes), in the original's order."""
    ci = 0
    while ci < len(classes):
        sl = d - classes[ci].cost
        S = size(sl)
        if not S:
            ci += 1
            continue
        if S >= E.SLICE:
            for lo in range(0, S, E.SLICE):
                yield [ci], sl, lo, min(lo + E.SLICE, S), lo + E.SLICE >= S
            ci += 1
            continue
        group = [ci]
        while (ci + len(group) < len(classes) and classes[ci + len(group)].cost == classes[ci].cost
               and (len(group) + 1) * S <= E.SLICE):
            group.append(ci + len(group))
        yield group, sl, 0, S, True
        ci += len(group)


class _Buckets:
    """States waiting to be expanded, by the level they source: e -> [(level g, indices into level g)], assembled into
    one batch (and the (g, index) of each row) when first read."""

    def __init__(self, levels):
        self.levels, self.parts, self.cache = levels, {}, {}

    def add(self, e, g, idx):
        if len(idx):
            self.parts.setdefault(e, []).append((g, idx))

    def size(self, e):
        return sum(len(i) for _, i in self.parts.get(e, ()))

    def get(self, e):
        if e not in self.cache:
            ps = self.parts[e]
            if len(ps) == 1 and len(ps[0][1]) == self.levels[ps[0][0]].S:     # a whole level, in order: no copy
                g, i = ps[0]
                self.cache[e] = (self.levels[g].regs, np.full(len(i), g, np.int32), i.astype(np.int64))
                return self.cache[e]
            regs = np.concatenate([self.levels[g].regs[i] for g, i in ps])
            gs = np.concatenate([np.full(len(i), g, np.int32) for g, i in ps])
            ix = np.concatenate([i for _, i in ps]).astype(np.int64)
            self.cache[e] = (regs, gs, ix)
        return self.cache[e]

    def drop_below(self, e):
        for k in [k for k in self.cache if k < e]:
            del self.cache[k]

    def last(self):
        return max(self.parts) if self.parts else -1


def bfs(prob, classes, lib, m, delay=None, max_cost=10, max_states=1_000_000, time_limit=600.0, accept=None,
        limit=200, codes=None, budget="level", max_level=None, prune=False):
    """engine.bfs with delays (see the module). max_cost bounds the plain search's levels; with a gate the levels are
    priced costs, and `max_level` (default max_cost * 4) bounds them instead."""
    t0 = time.time()
    codes = codes or E.Codes(classes, lib)
    top = max_cost if delay is None else (max_level or max_cost * 4)
    maxc = max(cl.cost for cl in classes) if classes else 1
    levels = [prob.root]
    seen = np.sort(hash_rows(prob.root.features()))
    parents = [None]
    res = Result()
    res.first_goal = res.first_accept = None
    res.level_sizes.append(1)
    res.best_fit.append(int(prob.fits(prob.root)[0]))
    if prob.goal(prob.root)[0]:
        if accept is None or any(accept([()])):
            res.found_cost, res.goals, res.parents, res.levels = 0, [0], parents, levels
            res.first_goal = res.first_accept = res.generated = 1
            return res
        res.aliases.append((0, [""]))
    buckets = _Buckets(levels)
    buckets.add(0, 0, np.zeros(1, np.int64))              # the start is never delayed
    gen = 1                                                # candidates generated so far, the start included
    res.delays, res.dead = [], []
    for d in range(1, top + 1):
        if delay is not None and d > buckets.last() + maxc:
            break                                          # nothing left to expand: exhausted
        last = d == top
        ck, cc, cs, ci_, chunks, gk, gpos = [], [], [], [], [], [], []
        n_cand = n_fresh = 0
        best = 0
        for group, sl, lo, hi, ends in _chunks(classes, buckets.size, d):
            src, gsrc, isrc = buckets.get(sl)
            S = hi - lo
            regs, drop, keys = E.apply_classes(codes, group, src[lo:hi], m)
            res.expanded += S * len(group)
            alive = np.nonzero(~drop)[0]
            if len(alive):
                keys = keys[alive]
                fm = _fresh(seen, keys)
                if fm.any():
                    idx = np.nonzero(fm)[0]
                    sub = brm.Batch(regs[alive[idx]], np.zeros(len(idx), bool))
                    f = prob.fits(sub)
                    best = max(best, int(f.max()))
                    g = f == len(prob.want)
                    if g.any():
                        gk.append(keys[idx][g])
                        gpos.append(gen + n_fresh + 1 + np.nonzero(g)[0])
                        if res.first_goal is None:
                            res.first_goal = gen + n_fresh + 1 + int(np.argmax(g))
                    n_fresh += len(idx)
                    flat = alive[idx]
                    if last:
                        if g.any():
                            idx, sub, flat = idx[g], sub.take(np.nonzero(g)[0]), flat[g]
                        else:
                            idx = idx[:0]
                    if len(idx):
                        ck.append(keys[idx])
                        cc.append(np.asarray(group, np.int32)[flat // S])
                        pos = (flat % S).astype(np.int64) + lo
                        cs.append(gsrc[pos])
                        ci_.append(isrc[pos])
                        chunks.append(sub)
                        n_cand += len(idx)
                    over = (n_cand > 1.5 * max_states if budget == "level" else gen + n_cand > max_states)
                    if not last and over:
                        last = True
                        res.stop = f"budget: level {d} streamed (over {max_states} states)"
                        keep = [(j, np.nonzero(prob.goal(chunks[j]))[0]) for j in range(len(chunks))]
                        keep = [(j, gm) for j, gm in keep if len(gm)]
                        ck = [ck[j][gm] for j, gm in keep]
                        cc = [cc[j][gm] for j, gm in keep]
                        cs = [cs[j][gm] for j, gm in keep]
                        ci_ = [ci_[j][gm] for j, gm in keep]
                        chunks = [chunks[j].take(gm) for j, gm in keep]
            if budget == "hard" and gen + n_fresh > max_states:
                res.stop = f"budget: cut in level {d} at {max_states} states"
                last = True
                break
            if ends and time.time() - t0 > time_limit:
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
                lv = brm.Batch.concat(parts)
                levels.append(lv)
                t = np.zeros(lv.S, np.int64) if delay is None else np.asarray(delay(lv), np.int64)
                res.delays.append(np.bincount(t).tolist())
                live = ~dead_states(lv.regs, prob) if prune else np.ones(lv.S, bool)
                res.dead.append(int((~live).sum()))
                for v in np.unique(t).tolist():
                    buckets.add(d + v, d, np.nonzero((t == v) & live)[0])
            else:
                levels.append(None)
                res.delays.append([])
        else:
            uk = np.zeros(0, np.uint64)
            rank = np.zeros(0, np.int64)
            parents.append(None)
            levels.append(None)
            res.delays.append([])
        res.level_sizes.append(int(len(uk)) if not last else int(n_fresh))
        res.best_fit.append(best)
        gen += n_fresh
        buckets.drop_below(d + 1 - maxc)
        if gk:
            g = np.unique(np.concatenate(gk))
            res.found_cost = d
            res.goals = rank[np.searchsorted(uk, g)].tolist()
            res.parents = parents
            if accept is None:
                res.first_accept = res.first_goal
                break
            progs = programs(res, classes, limit=limit)
            ok = accept(progs)
            if any(ok):
                res.accepted = [p for p, o in zip(progs, ok) if o]
                res.first_accept = _first_accepted(res, classes, d, uk, rank, gk, gpos, accept)
                break
            res.aliases.append((d, [dsl.show(p) for p in progs[:20]]))
            res.found_cost, res.goals = None, None
        if res.stop:
            break
        seen = np.union1d(seen, uk)
    res.generated = gen
    res.seconds = time.time() - t0
    if res.found_cost is None and not res.stop:
        res.stop = f"exhausted cost {top}"
    res.parents = parents
    res.levels = levels
    return res


def dead_states(regs, prob):
    """[S] bool: some two probes hold the same five registers but want different answers."""
    S, NP, _ = regs.shape
    want = np.asarray(prob.want).reshape(NP, -1)
    w = np.zeros(NP, np.int64)
    for c in range(want.shape[1]):
        w = w * 65536 + want[:, c].astype(np.int64)
    out = np.zeros(S, bool)
    assert regs.max(initial=0) < 4096, "registers packed 12 bits each"
    for lo in range(0, S, 1 << 16):
        r = regs[lo:lo + (1 << 16)].astype(np.int64)
        code = np.zeros(r.shape[:2], np.int64)
        for k in range(5):
            code = code * 4096 + r[:, :, k]
        order = np.argsort(code, axis=1, kind="stable")
        c = np.take_along_axis(code, order, 1)
        ww = w[order]
        out[lo:lo + len(r)] = ((c[:, 1:] == c[:, :-1]) & (ww[:, 1:] != ww[:, :-1])).any(axis=1)
    return out


def _first_accepted(res, classes, d, uk, rank, gk, gpos, accept):
    """The candidate count at the first goal state (in generation order) with an accepted program."""
    keys = np.concatenate(gk)
    pos = np.concatenate(gpos)
    order = np.argsort(pos, kind="stable")
    done = set()
    for i in order.tolist():
        k = keys[i]
        if k in done:
            continue
        done.add(k)
        idx = int(rank[np.searchsorted(uk, k)])
        if any(accept(programs_at(res, classes, d, idx, limit=20))):
            return int(pos[i])
    return None
