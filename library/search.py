"""Exhaustive program search by cost, with states merged when they agree on every probe (LIBRARY.md §2).

A problem is a task on a few settings (B, W), NP probes each. The search is breadth-first by cost: level d holds
every distinct machine state (over all probes of all settings at once) that some program of cost d reaches and no
cheaper program does. Level d is built by applying every item of cost c to every state of level d - c. Two
programs that reach the same state are interchangeable from then on, so only one copy is expanded, but every
minimal-cost parent of a state is kept: the corpus gets every minimal program, not one.

Items are the vocabulary's tokens (primitives and library macros, cost 1) and loops over bodies of up to
`max_body` cost. Items are first merged by what they do to 24 random machines per setting (random registers,
flags and pointer): items that agree there are taken to be the same function, and only the cheapest is searched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from library import dsl
from library.tasks import gen, INPUTS

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def hash_rows(F):
    """A 64-bit hash of every row of a uint8 matrix."""
    S, n = F.shape
    pad = (-n) % 8
    if pad:
        F = np.concatenate([F, np.zeros((S, pad), np.uint8)], 1)
    Wd = np.ascontiguousarray(F).view(np.uint64)
    h = np.full(S, np.uint64(0x9E3779B97F4A7C15))
    with np.errstate(over="ignore"):
        for i in range(Wd.shape[1]):
            h = (h ^ Wd[:, i]) * _M1
            h ^= h >> np.uint64(31)
        h = h * _M2
        h ^= h >> np.uint64(29)
    return h


def keys_of(batches):
    return hash_rows(np.concatenate([b.features() for b in batches], axis=1))


# ---------------------------------------------------------------------------------------------------------------
# problems

@dataclass
class Problem:
    task: str
    settings: list
    NP: int = 12
    seed: int = 0
    targets: dict = None   # per setting index: {"A": [P, NP]} or {"c": [NP]}
    roots: list = None     # per setting: Batch with S = 1

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.roots, self.targets = [], []
        for st in self.settings:
            ins, tgt = gen(self.task, st.B, st.W, self.NP, rng)
            self.roots.append(dsl.fresh(st, ins, rng, self.NP))
            self.targets.append({k: (v.T if k == "A" else v) for k, v in tgt.items()})

    def goal(self, batches):
        ok = np.ones(batches[0].S, bool)
        for b, t in zip(batches, self.targets):
            if "A" in t:
                ok &= (b.regs[:, dsl.A] == t["A"][None]).all(axis=(1, 2))
            if "c" in t:
                ok &= (b.c == t["c"][None]).all(axis=1)
        return ok


def check(task, program, lib, settings, n=300, seed=1, cells=None):
    """Is program right on n fresh probes of every setting? Returns {setting: fraction right}."""
    rng = np.random.default_rng(seed)
    out = {}
    for st in settings:
        ins, tgt = gen(task, st.B, st.W, n, rng)
        b = dsl.fresh(st, ins, rng, n)
        dsl.run_items(st, b, program, lib)
        ok = np.ones(n, bool)
        if "A" in tgt:
            ok &= (b.regs[0, dsl.A].T == tgt["A"]).all(axis=1)
        if "c" in tgt:
            ok &= b.c[0] == tgt["c"]
        out[repr(st)] = float(ok.mean())
    return out


# ---------------------------------------------------------------------------------------------------------------
# items

@dataclass
class ItemClass:
    rep: object
    cost: int
    members: list = field(default_factory=list)


def _sig(settings, tests, seq, lib):
    outs = []
    for st, t in zip(settings, tests):
        b = t.copy()
        dsl.run_items(st, b, seq, lib)
        outs.append(b)
    rows = keys_of(outs)
    return int(hash_rows(rows.view(np.uint8).reshape(1, -1))[0])


def enumerate_items(tokens, lib, settings, kinds=dsl.LOOP_KINDS, max_body=2, max_arity=3, seed=7, S_test=24,
                    NP_test=6):
    """Every distinct item (tokens, then loops with bodies of cost <= max_body), cheapest spelling first."""
    rng = np.random.default_rng(seed)
    tests = [dsl.random_states(st, S_test, NP_test, rng) for st in settings]
    ident = _sig(settings, tests, (), lib)
    classes, by_sig = [], {ident: None}

    def add(item, c):
        s = _sig(settings, tests, (item,), lib)
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
    # distinct straight-line bodies by cost, built from the distinct items found so far
    bodies = {}          # cost -> list of body tuples (distinct as functions)
    body_sig = {ident}
    for bc in range(1, max_body + 1):
        found = []
        reps = [cl for cl in classes]

        def extend(prefix, pc, n):
            if pc == bc:
                s = _sig(settings, tests, prefix, lib)
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
        bodies[bc] = found
        for body in found:
            for kind in kinds:
                if kind in lib and (len(body) != 1 or isinstance(body[0], tuple)):
                    continue        # an abstraction takes one token as its argument
                add(("REP", kind, body), bc + 1)
    return classes


# ---------------------------------------------------------------------------------------------------------------
# the search

@dataclass
class Result:
    task: str
    found_cost: int = None
    goals: list = None          # indices into the found level
    level_sizes: list = field(default_factory=list)
    seconds: float = 0.0
    stop: str = ""
    parents: list = None        # per level: (child, class index, src level, src index) arrays sorted by child
    expanded: int = 0           # item applications x source states
    aliases: list = field(default_factory=list)   # (cost, programs) of goal costs whose programs all failed accept
    accepted: list = None


def _fresh(seen, keys):
    if len(seen) == 0:
        return np.ones(len(keys), bool)
    pos = np.searchsorted(seen, keys)
    pos[pos == len(seen)] = 0
    return seen[pos] != keys


def bfs(prob: Problem, classes, lib, max_cost=8, max_states=400_000, time_limit=600.0, verbose=False,
        slice_size=32768, accept=None, limit=200):
    """Breadth-first by cost. Stops at the first cost with a goal state (all its minimal programs are kept), when a
    level would pass max_states, or after time_limit seconds. The last level (max_cost) is never stored.
    accept(programs) -> list of bools: when given, a cost whose goal programs all fail it (aliases that fit the
    probes only) does not stop the search; they are noted in res.aliases and the search goes on."""
    t0 = time.time()
    settings = prob.settings
    levels = [prob.roots]
    seen = np.sort(keys_of(prob.roots))
    parents = [None]
    res = Result(prob.task)
    res.level_sizes.append(1)
    if prob.goal(prob.roots)[0]:
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
            for lo in range(0, levels[sl][0].S, slice_size):
                src = [b.take(slice(lo, lo + slice_size)) for b in levels[sl]]
                outs = [dsl.apply(st, b, cl.rep, lib) for st, b in zip(settings, src)]
                res.expanded += src[0].S
                keys = keys_of(outs)
                m = _fresh(seen, keys)
                if not m.any():
                    continue
                idx = np.nonzero(m)[0]
                n_fresh += len(idx)
                sub = [b.take(idx) for b in outs]
                del outs
                g = prob.goal(sub)
                if g.any():
                    gk.append(keys[idx][g])
                if last:   # keep only what leads to goals
                    if not g.any():
                        continue
                    idx, sub = idx[g], [b.take(np.nonzero(g)[0]) for b in sub]
                ck.append(keys[idx])
                cc.append(np.full(len(idx), ci, np.int32))
                cs.append(np.full(len(idx), sl, np.int16))
                ci_.append(idx.astype(np.int64) + lo)
                chunks.append(sub)
                n_cand += len(idx)
                if not last and n_cand > 1.5 * max_states:
                    # too big to store: finish this level streaming (goal states only), then stop
                    last = True
                    res.stop = f"budget: level {d} streamed (over {max_states} states)"
                    new_ck, new_cc, new_cs, new_ci, new_ch = [], [], [], [], []
                    for j in range(len(chunks)):
                        gm = np.nonzero(prob.goal(chunks[j]))[0]
                        if len(gm):
                            new_ck.append(ck[j][gm])
                            new_cc.append(cc[j][gm])
                            new_cs.append(cs[j][gm])
                            new_ci.append(ci_[j][gm])
                            new_ch.append([b.take(gm) for b in chunks[j]])
                    ck, cc, cs, ci_, chunks = new_ck, new_cc, new_cs, new_ci, new_ch
            if time.time() - t0 > time_limit:
                res.stop = f"time: {time_limit}s at level {d}"
                break
        if ck:
            # the level keeps each state's first occurrence, in the order the candidates came
            allk = np.concatenate(ck)
            uk, first, inv = np.unique(allk, return_index=True, return_inverse=True)
            sfirst = np.sort(first)
            rank = np.searchsorted(sfirst, first)          # level position of uk[j]
            child = rank[inv]
            order = np.argsort(child, kind="stable")
            parents.append((child[order], np.concatenate(cc)[order], np.concatenate(cs)[order],
                            np.concatenate(ci_)[order]))
            del allk, inv, child, order, ck, cc, cs, ci_
            if not last and not res.stop:
                parts, off = [[] for _ in settings], 0
                for j in range(len(chunks)):
                    ch = chunks[j]
                    n = ch[0].S
                    lo_, hi_ = np.searchsorted(sfirst, off), np.searchsorted(sfirst, off + n)
                    rows = sfirst[lo_:hi_] - off
                    for si in range(len(settings)):
                        parts[si].append(ch[si].take(rows))
                    chunks[j] = None
                    off += n
                levels.append([dsl.Batch.concat(p) for p in parts])
                del parts
            else:
                levels.append(None)
            chunks = None
        else:
            uk = np.zeros(0, np.uint64)
            rank = np.zeros(0, np.int64)
            parents.append(None)
            levels.append(None)
        res.level_sizes.append(int(len(uk)) if not last else int(n_fresh))   # a streamed level: candidates
        if len(uk) > max_states and not res.stop:
            res.stop = f"budget: level {d} has {len(uk)} > {max_states} states"
        if verbose:
            print(f"  level {d}: {len(uk)} states, {time.time() - t0:.1f}s", flush=True)
        if gk:
            g = np.unique(np.concatenate(gk))
            res.found_cost = d
            res.goals = rank[np.searchsorted(uk, g)].tolist()
            if accept is None:
                break
            res.parents = parents
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


def programs(res: Result, classes, limit=200, rng=None):
    """Up to `limit` minimal programs (tuples of items) reaching the goal states."""
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
        plist = list(zip(pcl[lo:hi].tolist(), psl[lo:hi].tolist(), psi[lo:hi].tolist()))
        if rng is not None:
            plist = list(plist)
            rng.shuffle(plist)
        for ci, sl, si in plist:
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
