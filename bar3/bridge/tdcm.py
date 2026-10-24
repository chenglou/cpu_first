"""The counting machine (dcm.py) and its breadth-first search (dsearch.py) in torch, for a GPU: the same programs, the
same states, the same levels. A step toward running the digit-level learner inside a training run (rule 9: the
computation stays on the GPU). Torch only, so that a submission can inline it.

Differences of form, not of result: every probe of every state runs at once with masks (dcm runs index subsets), and a
loop runs as many iterations as its largest count (one host sync per loop); a machine that passes the step limit is
dead, and a state with a dead probe is dropped, exactly as in dcm.
"""

from __future__ import annotations

import time

import torch

REGS = ("A", "R", "C", "H", "N", "T")
RI = {r: i for i, r in enumerate(REGS)}
_M1, _M2 = -0x40A7B892E31B1A47, -0x6B2FB644ECCEEE15    # libcore's hash constants as signed 64-bit


def is_loop(it):
    return isinstance(it, tuple)


def fill(pattern, arg):
    out = []
    for it in pattern:
        if is_loop(it):
            out.append(("REP", it[1], fill(it[2], arg)))
        else:
            out.append(arg if it == "_" else it)
    return tuple(out)


class Run:
    """One execution over flat machines M [m, 6] (int32), with per-machine step counts and a dead flag."""

    def __init__(self, cap, limit, M, lib):
        self.cap, self.limit, self.M, self.lib = cap, limit, M, lib
        self.steps = torch.zeros(M.shape[0], dtype=torch.int32, device=M.device)
        self.dead = torch.zeros(M.shape[0], dtype=torch.bool, device=M.device)

    def items(self, items, mask):
        M = self.M
        for it in items:
            if is_loop(it):
                kind, body = it[1], it[2]
                if kind in self.lib:
                    self.items(fill(self.lib[kind], body[0]), mask)
                    continue
                r = RI[kind[2:]]
                if kind[0] == "Z":
                    self.items(body, mask & (M[:, r] == 0))
                    continue
                cnt = torch.where(mask, M[:, r], 0)
                n = int(cnt.max()) if cnt.numel() else 0
                for t in range(n):
                    live = mask & (cnt > t)
                    self.steps += live
                    over = live & (self.steps > self.limit)
                    self.dead |= over
                    self.items(body, live & ~over)
                continue
            if it in self.lib:
                self.items(self.lib[it], mask)
                continue
            op, _, arg = it.partition(".")
            self.steps += mask
            if op == "MOV":
                d, s = arg.split(".")
                M[:, RI[d]] = torch.where(mask, M[:, RI[s]], M[:, RI[d]])
                continue
            r = RI[arg]
            if op == "INC":
                v = (M[:, r] + 1).clamp(max=self.cap)
            elif op == "DEC":
                v = (M[:, r] - 1).clamp(min=0)
            elif op == "CLR":
                v = torch.zeros_like(M[:, r])
            else:
                raise ValueError(it)
            M[:, r] = torch.where(mask, v, M[:, r])


def apply(cap, limit, regs, items, lib):
    """regs [S, NP, 6] -> (new regs, dead [S]): items run on every probe of every state."""
    S, NP, nr = regs.shape
    M = regs.reshape(S * NP, nr).clone()
    run = Run(cap, limit, M, lib)
    run.items(tuple(items), torch.ones(S * NP, dtype=torch.bool, device=regs.device))
    return M.view(S, NP, nr), run.dead.view(S, NP).any(1)


def hash_rows(F):
    """A 64-bit hash of every row of an int matrix [S, n] (values below 2^16)."""
    h = torch.full((F.shape[0],), 0x1E3779B97F4A7C15, dtype=torch.int64, device=F.device)
    for i in range(F.shape[1]):
        h = (h ^ F[:, i].long()) * _M1
        h = h ^ ((h >> 31) & ((1 << 33) - 1))
    h = h * _M2
    return h ^ ((h >> 29) & ((1 << 35) - 1))


def keys_of(regs):
    return hash_rows(regs.reshape(regs.shape[0], -1))


class ItemClass:
    def __init__(self, rep, cost):
        self.rep, self.cost = rep, cost


def _sig(cap, limit, tests, seq, lib):
    regs, dead = apply(cap, limit, tests, seq, lib)
    k = torch.where(dead, torch.full_like(dead, 7, dtype=torch.int64), keys_of(regs))
    return int(hash_rows(k[None])[0])


def enumerate_items(tokens, lib, kinds, cap, limit, device, max_body=2, max_arity=3, seed=7, S_test=24, NP_test=6):
    """dsearch.enumerate_items: the tokens, then loops over bodies of cost up to max_body, merged by what they do to
    random machines (here the test machines come from torch's generator, so the merged classes can differ from the
    numpy search's in which spelling represents a class, never in what is reachable)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    tests = torch.randint(0, 12, (S_test, NP_test, len(REGS)), generator=g, dtype=torch.int32).to(device)
    ident = _sig(cap, limit, tests, (), lib)
    classes, by_sig = [], {ident}

    def add(item, c):
        s = _sig(cap, limit, tests, (item,), lib)
        if s in by_sig:
            return
        by_sig.add(s)
        classes.append(ItemClass(item, c))

    for t in tokens:
        add(t, 1)
    body_sig = {ident}
    for bc in range(1, max_body + 1):
        found, reps = [], list(classes)

        def extend(prefix, pc, n):
            if pc == bc:
                s = _sig(cap, limit, tests, prefix, lib)
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
                if kind in lib and (len(body) != 1 or is_loop(body[0])):
                    continue
                add(("REP", kind, body), bc + 1)
    return classes


class Result:
    def __init__(self):
        self.found_cost, self.goals, self.level_sizes, self.best_fit = None, None, [], []
        self.seconds, self.stop, self.parents, self.levels, self.aliases, self.accepted = 0.0, "", None, None, [], None


def bfs(root, outputs, want, classes, lib, cap, limit, max_cost=10, max_states=1_000_000, time_limit=600.0,
        accept=None, limit_progs=200, slice_size=1 << 15):
    """dsearch.bfs: root [1, NP, 6], outputs (register indices), want [NP, len(outputs)]."""
    t0 = time.time()
    dev = root.device
    NP = want.shape[0]
    oi = torch.tensor(outputs, dtype=torch.long, device=dev)

    def fits(regs, dead):
        return (regs[:, :, oi] == want[None]).all(2).sum(1) * ~dead

    levels = [root]
    seen = torch.sort(keys_of(root)).values
    parents = [None]
    res = Result()
    res.level_sizes.append(1)
    f0 = fits(root, torch.zeros(1, dtype=torch.bool, device=dev))
    res.best_fit.append(int(f0[0]))
    if int(f0[0]) == NP:
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
            src_all = levels[sl]
            for lo in range(0, src_all.shape[0], slice_size):
                src = src_all[lo:lo + slice_size]
                out, dead = apply(cap, limit, src, (cl.rep,), lib)
                alive = torch.nonzero(~dead).flatten()
                if not alive.numel():
                    continue
                out = out[alive]
                keys = keys_of(out)
                pos = torch.searchsorted(seen, keys).clamp(max=max(seen.numel() - 1, 0))
                fm = (seen[pos] != keys) if seen.numel() else torch.ones_like(keys, dtype=torch.bool)
                if not bool(fm.any()):
                    continue
                idx = torch.nonzero(fm).flatten()
                n_fresh += idx.numel()
                sub = out[idx]
                f = fits(sub, torch.zeros(sub.shape[0], dtype=torch.bool, device=dev))
                best = max(best, int(f.max()))
                g = f == NP
                if bool(g.any()):
                    gk.append(keys[idx][g])
                if last:
                    if not bool(g.any()):
                        continue
                    gi = torch.nonzero(g).flatten()
                    idx, sub = idx[gi], sub[gi]
                ck.append(keys[idx])
                cc.append(torch.full((idx.numel(),), ci, dtype=torch.int32, device=dev))
                cs.append(torch.full((idx.numel(),), sl, dtype=torch.int16, device=dev))
                ci_.append(alive[idx].long() + lo)
                chunks.append(sub)
                n_cand += idx.numel()
                if not last and n_cand > 1.5 * max_states:
                    last = True
                    res.stop = f"budget: level {d} streamed (over {max_states} states)"
                    keep = []
                    for j in range(len(chunks)):
                        gm = torch.nonzero(fits(chunks[j], torch.zeros(chunks[j].shape[0], dtype=torch.bool,
                                                                       device=dev)) == NP).flatten()
                        if gm.numel():
                            keep.append((j, gm))
                    ck = [ck[j][gm] for j, gm in keep]
                    cc = [cc[j][gm] for j, gm in keep]
                    cs = [cs[j][gm] for j, gm in keep]
                    ci_ = [ci_[j][gm] for j, gm in keep]
                    chunks = [chunks[j][gm] for j, gm in keep]
            if time.time() - t0 > time_limit:
                res.stop = f"time: {time_limit}s at level {d}"
                break
        if ck:
            allk = torch.cat(ck)
            uk, inv = torch.unique(allk, sorted=True, return_inverse=True)
            first = torch.full((uk.numel(),), allk.numel(), dtype=torch.long, device=dev)
            first.scatter_reduce_(0, inv, torch.arange(allk.numel(), device=dev), reduce="amin")
            sfirst = torch.sort(first).values
            rank = torch.searchsorted(sfirst, first)
            child = rank[inv]
            order = torch.sort(child, stable=True).indices
            parents.append((child[order], torch.cat(cc)[order], torch.cat(cs)[order], torch.cat(ci_)[order]))
            if not last and not res.stop:
                levels.append(torch.cat(chunks)[sfirst])
            else:
                levels.append(None)
        else:
            uk = torch.zeros(0, dtype=torch.int64, device=dev)
            rank = torch.zeros(0, dtype=torch.long, device=dev)
            parents.append(None)
            levels.append(None)
        res.level_sizes.append(int(uk.numel()) if not last else int(n_fresh))
        res.best_fit.append(best)
        if gk:
            g = torch.unique(torch.cat(gk))
            res.found_cost = d
            res.goals = rank[torch.searchsorted(uk, g)].tolist()
            res.parents = parents
            if accept is None:
                break
            progs = programs(res, classes, limit=limit_progs)
            ok = accept(progs)
            if any(ok):
                res.accepted = [p for p, o in zip(progs, ok) if o]
                break
            res.aliases.append((d, progs[:20]))
            res.found_cost, res.goals = None, None
        if res.stop:
            break
        seen = torch.sort(torch.cat([seen, uk])).values
    res.seconds = time.time() - t0
    if res.found_cost is None and not res.stop:
        res.stop = f"exhausted cost {max_cost}"
    res.parents = parents
    res.levels = levels
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
        lo = int(torch.searchsorted(child, torch.tensor([idx], device=child.device))[0])
        hi = int(torch.searchsorted(child, torch.tensor([idx], device=child.device), right=True)[0])
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
