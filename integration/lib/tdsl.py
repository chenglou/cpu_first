"""The library's digit machine and its program search (library/dsl.py, library/search.py) on torch tensors, so that
the search can run inside a training run on the GPU (INTEGRATION.md, stage 2). Same machine, same instructions, same
breadth-first search by cost with states merged over every probe; the cells are tensors that learned tables can
replace, and the probes of a problem can be built from digit tensors.

Programs, their costs and their text forms are library/dsl.py's (pure Python, imported); this module only runs them.
Batches: regs [S, 5, P, NP] uint8, c / h / k [S, NP] uint8, p [S] int64, on one device.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from library import dsl as pdsl
from library.dsl import PRIMS, LOOP_KINDS, REGS, cost, expand, fill, show, parse, is_loop  # noqa: F401

RI = {r: i for i, r in enumerate(REGS)}
A, X, Y, N, S = range(5)


# ---------------------------------------------------------------------------------------------------------------
# cells

class Cells:
    """The digit cells as uint8 tensors on a device: sbb [B, B, 2] -> digit, borrow; adc [B, B, 2] -> digit, carry;
    mul [B, B, B, B] (a, x, h, k) -> low, high. Exact by default; learned tables replace them (`set`)."""

    def __init__(self, B: int, device="cpu"):
        self.B, self.device = B, device
        a, b, c = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
        d = a - b - c
        self.sbb_d, self.sbb_c = (d % B).to(torch.uint8), (d < 0).to(torch.uint8)
        d = a + b + c
        self.adc_d, self.adc_c = (d % B).to(torch.uint8), (d >= B).to(torch.uint8)
        a, x, h, k = torch.meshgrid(*(torch.arange(B),) * 4, indexing="ij")
        t = a + x * h + k
        self.mul_lo, self.mul_hi = (t % B).to(torch.uint8), (t // B).to(torch.uint8)
        self.to(device)

    def to(self, device):
        self.device = device
        for n in ("sbb_d", "sbb_c", "adc_d", "adc_c", "mul_lo", "mul_hi"):
            setattr(self, n, getattr(self, n).to(device))
        self._flat()
        return self

    def set(self, **tabs):
        """Replace some tables (uint8 or long tensors of the tables' shapes)."""
        for n, t in tabs.items():
            setattr(self, n, torch.as_tensor(t).to(torch.uint8).to(self.device).reshape(getattr(self, n).shape))
        self._flat()
        return self

    def _flat(self):
        B = self.B
        self.sbb = torch.stack([self.sbb_d, self.sbb_c], -1).reshape(B * B * 2, 2).long()
        self.adc = torch.stack([self.adc_d, self.adc_c], -1).reshape(B * B * 2, 2).long()
        self.mul = torch.stack([self.mul_lo, self.mul_hi], -1).reshape(B ** 4, 2).long()

    def exact(self) -> bool:
        ref = Cells(self.B, self.device)
        return all(torch.equal(getattr(self, n), getattr(ref, n))
                   for n in ("sbb_d", "sbb_c", "adc_d", "adc_c", "mul_lo", "mul_hi"))


@dataclass
class Setting:
    B: int
    W: int
    cells: Cells = None

    def __post_init__(self):
        self.P = 2 * self.W

    def count(self, kind):
        return {"P": self.P, "B": self.B, "W": self.W}[kind]

    def __repr__(self):
        return f"({self.B},{self.W})"


@dataclass
class Batch:
    regs: torch.Tensor   # [S, 5, P, NP] uint8
    c: torch.Tensor      # [S, NP]
    h: torch.Tensor
    k: torch.Tensor
    p: torch.Tensor      # [S] int64

    @property
    def S(self):
        return self.regs.shape[0]

    def copy(self):
        return Batch(self.regs.clone(), self.c.clone(), self.h.clone(), self.k.clone(), self.p.clone())

    def take(self, idx):
        return Batch(self.regs[idx], self.c[idx], self.h[idx], self.k[idx], self.p[idx])

    def repeat(self, n):
        return Batch(self.regs.repeat_interleave(n, 0), self.c.repeat_interleave(n, 0), self.h.repeat_interleave(n, 0),
                     self.k.repeat_interleave(n, 0), self.p.repeat_interleave(n, 0))

    @staticmethod
    def concat(bs):
        return Batch(*(torch.cat([getattr(b, f) for b in bs], 0) for f in ("regs", "c", "h", "k", "p")))

    def features(self):
        S_ = self.S
        return torch.cat([self.regs.reshape(S_, -1), self.c, self.h, self.k, self.p.to(torch.uint8)[:, None]], 1)


def fresh(st: Setting, inputs: dict, gen: torch.Generator, NP: int | None = None, device=None):
    """One fresh machine (S = 1) over NP probes: inputs {reg: digits [NP, P]}; everything else random (dirty start)."""
    device = device or st.cells.device
    B, P = st.B, st.P
    if NP is None:
        NP = next(iter(inputs.values())).shape[0]
    regs = torch.randint(0, B, (1, 5, P, NP), generator=gen, dtype=torch.uint8).to(device)
    for r, v in inputs.items():
        regs[0, RI[r]] = torch.as_tensor(v).to(device=device, dtype=torch.uint8).T
    c = torch.randint(0, 2, (1, NP), generator=gen, dtype=torch.uint8).to(device)
    h = torch.randint(0, B, (1, NP), generator=gen, dtype=torch.uint8).to(device)
    k = torch.randint(0, B, (1, NP), generator=gen, dtype=torch.uint8).to(device)
    return Batch(regs, c, h, k, torch.zeros(1, dtype=torch.long, device=device))


def random_states(st: Setting, S_: int, NP: int, gen: torch.Generator, device=None):
    device = device or st.cells.device
    B, P = st.B, st.P
    return Batch(torch.randint(0, B, (S_, 5, P, NP), generator=gen, dtype=torch.uint8).to(device),
                 torch.randint(0, 2, (S_, NP), generator=gen, dtype=torch.uint8).to(device),
                 torch.randint(0, B, (S_, NP), generator=gen, dtype=torch.uint8).to(device),
                 torch.randint(0, B, (S_, NP), generator=gen, dtype=torch.uint8).to(device),
                 torch.randint(0, P, (S_,), generator=gen).to(device))


# ---------------------------------------------------------------------------------------------------------------
# the instructions (in place on a batch)

def _at(b: Batch, reg: int):
    """The digits at each state's pointer of register reg [S, NP] (a view index for writing back)."""
    ar = torch.arange(b.S, device=b.regs.device)
    return ar, b.regs[ar, reg, b.p]


def _shl(b, r):
    b.regs[:, r, 1:] = b.regs[:, r, :-1].clone()
    b.regs[:, r, 0] = 0


def _shr(b, r):
    b.regs[:, r, :-1] = b.regs[:, r, 1:].clone()
    b.regs[:, r, -1] = 0


def step(st: Setting, b: Batch, tok: str):
    """Run one primitive on batch b in place."""
    cl = st.cells
    B = st.B
    op, _, arg = tok.partition(".")
    if op == "SBB" or op == "ADC":
        ar, a = _at(b, A)
        x = b.regs[ar, RI[arg], b.p]
        idx = (a.long() * B + x.long()) * 2 + b.c.long()
        out = (cl.sbb if op == "SBB" else cl.adc)[idx]
        b.regs[ar, A, b.p] = out[..., 0].to(torch.uint8)
        b.c = out[..., 1].to(torch.uint8)
    elif op == "MUL":
        ar, a = _at(b, A)
        x = b.regs[ar, RI[arg], b.p]
        idx = ((a.long() * B + x.long()) * B + b.h.long()) * B + b.k.long()
        out = cl.mul[idx]
        b.regs[ar, A, b.p] = out[..., 0].to(torch.uint8)
        b.k = out[..., 1].to(torch.uint8)
    elif op == "SEL":
        b.regs[:, A] = torch.where(b.c[:, None, :].bool(), b.regs[:, S], b.regs[:, A])
    elif op == "LDA":
        b.regs[:, A] = b.regs[:, RI[arg]]
    elif op == "STA":
        b.regs[:, RI[arg]] = b.regs[:, A]
    elif op == "CLA":
        b.regs[:, A] = 0
    elif op == "CLC":
        b.c = torch.zeros_like(b.c)
    elif op == "CLK":
        b.k = torch.zeros_like(b.k)
    elif op == "LDH":
        ar = torch.arange(b.S, device=b.regs.device)
        b.h = b.regs[ar, RI[arg], b.p].clone()
    elif op == "INC":
        b.p = (b.p + 1) % st.P
    elif op == "SHL":
        _shl(b, RI[arg])
    elif op == "SHR":
        _shr(b, RI[arg])
    elif op == "BRG":
        top = b.regs[:, X, -1].clone()
        _shl(b, A)
        b.regs[:, A, 0] = top
        _shl(b, X)
    else:
        raise ValueError(tok)


def run_items(st, b, items, lib):
    """Run a sequence of items on batch b in place (library/dsl.py's semantics: a routine T.<task> restores the
    pointer it was called with)."""
    for it in items:
        if isinstance(it, tuple):
            _, kind, body = it
            if kind in lib:
                run_items(st, b, fill(lib[kind], body[0]), lib)
            else:
                for _ in range(st.count(kind)):
                    run_items(st, b, body, lib)
        elif it in lib:
            if it.startswith("T."):
                p = b.p.clone()
                run_items(st, b, lib[it], lib)
                b.p = p
            else:
                run_items(st, b, lib[it], lib)
        else:
            step(st, b, it)


def apply(st, b, item, lib):
    out = b.copy()
    run_items(st, out, (item,), lib)
    return out


# ---------------------------------------------------------------------------------------------------------------
# hashing (64-bit, wrapping int64 arithmetic)

def _s64(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


_M1, _M2, _H0 = _s64(0xBF58476D1CE4E5B9), _s64(0x94D049BB133111EB), _s64(0x9E3779B97F4A7C15)


def _lsr(h: torch.Tensor, s: int) -> torch.Tensor:
    return (h >> s) & ((1 << (64 - s)) - 1)


def hash_rows(F: torch.Tensor) -> torch.Tensor:
    """A 64-bit hash (int64) of every row of a uint8 matrix [S, n]."""
    S_, n = F.shape
    pad = (-n) % 8
    if pad:
        F = torch.cat([F, torch.zeros(S_, pad, dtype=torch.uint8, device=F.device)], 1)
    Wd = F.contiguous().view(torch.int64)
    h = torch.full((S_,), _H0, dtype=torch.int64, device=F.device)
    for i in range(Wd.shape[1]):
        h = (h ^ Wd[:, i]) * _M1
        h = h ^ _lsr(h, 31)
    h = h * _M2
    return h ^ _lsr(h, 29)


def keys_of(batches):
    return hash_rows(torch.cat([b.features() for b in batches], 1))


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
        run_items(st, b, seq, lib)
        outs.append(b)
    rows = keys_of(outs)
    return int(hash_rows(rows.view(torch.uint8).reshape(1, -1))[0])


def enumerate_items(tokens, lib, settings, kinds=LOOP_KINDS, max_body=2, max_arity=3, seed=7, S_test=24, NP_test=6):
    """Every distinct item (tokens, then loops with bodies of cost <= max_body), cheapest spelling first: items that
    agree on S_test random machines per setting are taken to be the same function (library/search.py)."""
    gen = torch.Generator().manual_seed(seed)
    tests = [random_states(st, S_test, NP_test, gen) for st in settings]
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
    body_sig = {ident}
    for bc in range(1, max_body + 1):
        found = []
        reps = list(classes)

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
        for body in found:
            for kind in kinds:
                if kind in lib and (len(body) != 1 or isinstance(body[0], tuple)):
                    continue
                add(("REP", kind, body), bc + 1)
    return classes


# ---------------------------------------------------------------------------------------------------------------
# problems

class Problem:
    """A task on a few settings: roots (one fresh machine per setting) and targets per setting ({"A": digits [P, NP]}
    and/or {"c": bits [NP]}), and an optional extra check `inverse` per setting (a function of the A values of
    candidate states [S, NP, P] -> bool [S])."""

    def __init__(self, task, settings, roots, targets, inverse=None):
        self.task, self.settings, self.roots, self.targets = task, settings, roots, targets
        self.inverse = inverse or [None] * len(settings)

    def goal(self, batches):
        ok = torch.ones(batches[0].S, dtype=torch.bool, device=batches[0].regs.device)
        for b, t, inv in zip(batches, self.targets, self.inverse):
            if "A" in t:
                ok &= (b.regs[:, A] == t["A"][None]).all(-1).all(-1)
            if "c" in t:
                ok &= (b.c == t["c"][None]).all(-1)
            if inv is not None and bool(ok.any()):
                idx = torch.nonzero(ok).flatten()
                sub = inv(b.regs[idx, A].permute(0, 2, 1))
                ok[idx[~sub]] = False
        return ok


# ---------------------------------------------------------------------------------------------------------------
# the search

@dataclass
class Result:
    task: str
    found_cost: int = None
    goals: list = None
    level_sizes: list = field(default_factory=list)
    seconds: float = 0.0
    stop: str = ""
    parents: list = None
    expanded: int = 0
    aliases: list = field(default_factory=list)
    accepted: list = None


def _fresh(seen, keys):
    if seen.numel() == 0:
        return torch.ones(keys.numel(), dtype=torch.bool, device=keys.device)
    pos = torch.searchsorted(seen, keys)
    pos[pos == seen.numel()] = 0
    return seen[pos] != keys


def bfs(prob, classes, lib, max_cost=8, max_states=400_000, time_limit=600.0, verbose=False, slice_size=1 << 18,
        accept=None, limit=200, deadline=None):
    """library/search.py's bfs on tensors: breadth-first by cost, states merged over every probe of every setting,
    every minimal-cost parent kept. Stops at the first cost whose goal programs pass `accept`, when a level would
    pass max_states (that level is finished streaming goal states only), or after time_limit seconds."""
    t0 = time.time()
    settings = prob.settings
    dev = prob.roots[0].regs.device
    levels = [prob.roots]
    seen = torch.sort(keys_of(prob.roots)).values
    parents = [None]
    res = Result(prob.task)
    res.level_sizes.append(1)
    if bool(prob.goal(prob.roots)[0]):
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
                outs = [apply(st, b, cl.rep, lib) for st, b in zip(settings, src)]
                res.expanded += src[0].S
                keys = keys_of(outs)
                m = _fresh(seen, keys)
                if not bool(m.any()):
                    continue
                idx = torch.nonzero(m).flatten()
                n_fresh += idx.numel()
                sub = [b.take(idx) for b in outs]
                del outs
                g = prob.goal(sub)
                if bool(g.any()):
                    gk.append(keys[idx][g])
                if last:
                    if not bool(g.any()):
                        continue
                    gi = torch.nonzero(g).flatten()
                    idx, sub = idx[gi], [b.take(gi) for b in sub]
                ck.append(keys[idx])
                cc.append(torch.full((idx.numel(),), ci, dtype=torch.int32, device=dev))
                cs.append(torch.full((idx.numel(),), sl, dtype=torch.int16, device=dev))
                ci_.append(idx + lo)
                chunks.append(sub)
                n_cand += idx.numel()
                if not last and n_cand > 1.5 * max_states:
                    last = True
                    res.stop = f"budget: level {d} streamed (over {max_states} states)"
                    new = [[], [], [], [], []]
                    for j in range(len(chunks)):
                        gm = torch.nonzero(prob.goal(chunks[j])).flatten()
                        if gm.numel():
                            new[0].append(ck[j][gm])
                            new[1].append(cc[j][gm])
                            new[2].append(cs[j][gm])
                            new[3].append(ci_[j][gm])
                            new[4].append([b.take(gm) for b in chunks[j]])
                    ck, cc, cs, ci_, chunks = new
            if time.time() - t0 > time_limit or (deadline is not None and time.time() > deadline):
                res.stop = f"time: {time_limit}s at level {d}"
                break
        if ck:
            allk = torch.cat(ck)
            uk, inv = torch.unique(allk, sorted=True, return_inverse=True)
            pos = torch.arange(allk.numel(), device=dev)
            first = torch.full((uk.numel(),), allk.numel(), dtype=torch.long, device=dev).scatter_reduce(
                0, inv, pos, reduce="amin")
            sfirst = torch.sort(first).values
            rank = torch.searchsorted(sfirst, first)
            child = rank[inv]
            order = torch.sort(child, stable=True).indices
            parents.append(tuple(t.cpu().numpy() for t in (child[order], torch.cat(cc)[order],
                                                            torch.cat(cs)[order], torch.cat(ci_)[order])))
            del allk, inv, child, order, ck, cc, cs, ci_
            if not last and not res.stop:
                parts, off = [[] for _ in settings], 0
                for j in range(len(chunks)):
                    ch = chunks[j]
                    n = ch[0].S
                    lo_, hi_ = torch.searchsorted(sfirst, off), torch.searchsorted(sfirst, off + n)
                    rows = sfirst[int(lo_):int(hi_)] - off
                    for si in range(len(settings)):
                        parts[si].append(ch[si].take(rows))
                    chunks[j] = None
                    off += n
                levels.append([Batch.concat(p) for p in parts])
                del parts
            else:
                levels.append(None)
            chunks = None
        else:
            uk = torch.zeros(0, dtype=torch.int64, device=dev)
            rank = torch.zeros(0, dtype=torch.long, device=dev)
            parents.append(None)
            levels.append(None)
        res.level_sizes.append(int(uk.numel()) if not last else int(n_fresh))
        if uk.numel() > max_states and not res.stop:
            res.stop = f"budget: level {d} has {uk.numel()} > {max_states} states"
        if verbose:
            print(f"  level {d}: {uk.numel()} states, {time.time() - t0:.1f}s", flush=True)
        if gk:
            g = torch.unique(torch.cat(gk))
            res.found_cost = d
            res.goals = rank[torch.searchsorted(uk, g)].tolist()
            if accept is None:
                break
            res.parents = parents
            progs = programs(res, classes, limit=limit)
            ok = accept(progs)
            if any(ok):
                res.accepted = [p for p, o in zip(progs, ok) if o]
                break
            res.aliases.append((d, [show(p) for p in progs[:20]]))
            res.found_cost, res.goals = None, None
        if res.stop:
            break
        seen = torch.sort(torch.cat([seen, uk])).values
    res.seconds = time.time() - t0
    if res.found_cost is None and not res.stop:
        res.stop = f"exhausted cost {max_cost}"
    res.parents = parents
    return res


def programs(res: Result, classes, limit=200):
    """Up to `limit` minimal programs reaching the goal states (the parents are numpy arrays)."""
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


# ---------------------------------------------------------------------------------------------------------------
# digits

def to_digits(v, B, P):
    """Python ints -> digits [n, P] (LSB first) as a uint8 tensor (for the offline checks only)."""
    return torch.as_tensor(pdsl.to_digits(np.asarray(v, dtype=object), B, P))


def from_digits(d, B):
    return pdsl.from_digits(np.asarray(d.cpu()), B)
