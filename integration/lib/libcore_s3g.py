"""LIBRARY CORE (INTEGRATION.md, stages 2-3): everything the in-training library learning needs, in one torch-only
module that the stage-3 submission inlines (make_stage3.py copies this file's body). It consolidates, unchanged in what
they compute:
  * programs: library/dsl.py's items, costs, expansion, text form;
  * BPE: library/merge.py's tree-BPE (pair counts, merges, routines' rewriting);
  * the machine: tdsl.py (library/dsl.py's machine on torch tensors: 5 registers, a shared pointer, a borrow/carry, a
    multiplier digit and a multiply carry; 32 primitive instructions; loops by P, B, W; routines that restore the
    pointer), with the cells as tensors that the learned tables replace;
  * the search: tfast.py (breadth-first by cost, states merged over every probe, every minimal parent kept, many items
    applied at once);
  * the practice tasks from the law library and the rows' own digits, with the X <-> Y register variants (inclimb.py),
    and the climb (search, BPE, Occam, routines) as a state machine a training loop drives one unit at a time;
  * the round: found by composing the routines at the task's full size, its goal the answers (round_search.py);
  * the one-hot machine of the prediction path (dslste.py).
Rows are digit tensors (LSB first), as the model's reader produces them; no number is computed from a value.
"""

from __future__ import annotations

import time
from collections import Counter
import torch

# ===================================================================================================== programs
REGS = ("A", "X", "Y", "N", "S")
RI = {r: i for i, r in enumerate(REGS)}
A_, X_, Y_, N_, S_ = range(5)
PRIMS = ([f"SBB.{r}" for r in "XYNS"] + [f"ADC.{r}" for r in "XYNS"] + [f"MUL.{r}" for r in "XYN"]
         + ["SEL.S"] + [f"LDA.{r}" for r in "XYNS"] + [f"STA.{r}" for r in "XYS"] + ["CLA", "CLC", "CLK"]
         + [f"LDH.{r}" for r in "XY"] + ["INC"] + [f"SHL.{r}" for r in "AXY"] + [f"SHR.{r}" for r in "AXY"]
         + ["BRG.X"])
LOOP_KINDS = ("P", "B", "W")
ONE = "ONE"                                          # a pseudo loop kind run once: a straight-line body


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


def cost(items):
    return sum(1 + cost(it[2]) if is_loop(it) else 1 for it in items)


def expand(items, lib):
    out = []
    for it in items:
        if is_loop(it) and it[1] in lib:
            out.extend(expand(fill(lib[it[1]], it[2][0]), lib))
        elif is_loop(it):
            out.append(("REP", it[1], tuple(expand(it[2], lib))))
        elif it in lib:
            out.extend(expand(lib[it], lib))
        else:
            out.append(it)
    return tuple(out)


def show(items):
    return " ".join(f"{it[1]}[{show(it[2])}]" if is_loop(it) else it for it in items)


def parse(s):
    toks = s.replace("[", " [ ").replace("]", " ] ").split()
    pos = 0

    def seq():
        nonlocal pos
        out = []
        while pos < len(toks) and toks[pos] != "]":
            t = toks[pos]
            if pos + 1 < len(toks) and toks[pos + 1] == "[":
                pos += 2
                body = seq()
                pos += 1
                out.append(("REP", t, tuple(body)))
            else:
                out.append(t)
                pos += 1
        return out
    return tuple(seq())


# ===================================================================================================== BPE
def seqs(items):
    yield items
    for it in items:
        if is_loop(it):
            yield from seqs(it[2])


def rewrite(items, fn):
    out = []
    for it in items:
        out.append(("REP", it[1], rewrite(it[2], fn)) if is_loop(it) else it)
    return tuple(fn(tuple(out)))


def _name(lib, prefix):
    i = 1
    while f"{prefix}{i}" in lib:
        i += 1
    return f"{prefix}{i}"


def _pairs(seq):
    out, i = [], 0
    while i < len(seq):
        a = seq[i]
        if is_loop(a):
            if len(a[2]) == 1 and not is_loop(a[2][0]):
                out.append(("LOOP", a[1], a[2][0]))
            i += 1
            continue
        if i + 1 < len(seq) and not is_loop(seq[i + 1]):
            out.append(("PAIR", a, seq[i + 1]))
            if a == seq[i + 1]:
                i += 2
                continue
        i += 1
    return out


def pair_counts(corpus):
    """A task counts a pair as often as its best program uses it (library/merge.py, weighting "max")."""
    cnt = Counter()
    for progs in corpus.values():
        if not progs:
            continue
        per = []
        for p in progs:
            c = Counter()
            for s in seqs(p):
                c.update(_pairs(s))
            per.append(c)
        for pr in set().union(*per):
            cnt[pr] += max(c[pr] for c in per)
    return cnt


def _apply_pair(pr, name):
    kind = pr[0]

    def fn(seq):
        out, i = [], 0
        while i < len(seq):
            it = seq[i]
            if kind == "LOOP" and is_loop(it) and it[1] == pr[1] and it[2] == (pr[2],):
                out.append(name)
                i += 1
            elif kind == "PAIR" and i + 1 < len(seq) and it == pr[1] and seq[i + 1] == pr[2]:
                out.append(name)
                i += 2
            else:
                out.append(it)
                i += 1
        return out
    return fn


def _pair_size(pr, lib):
    items = (("REP", pr[1], (pr[2],)),) if pr[0] == "LOOP" else (pr[1], pr[2])
    return cost(expand(items, lib))


def bpe(corpus, lib, max_merges=12, min_count=2.0, prefix="B", log=None):
    """Greedy pair merges, ties toward the pair standing for the most primitives (library/merge.py's "largest")."""
    lib, new = dict(lib), []
    for _ in range(max_merges):
        cnt = pair_counts(corpus)
        if not cnt:
            break
        pr, w = max(cnt.items(), key=lambda kv: (round(kv[1], 9), _pair_size(kv[0], lib), str(kv[0])))
        if w < min_count:
            break
        name = _name(lib, prefix)
        lib[name] = (("REP", pr[1], (pr[2],)),) if pr[0] == "LOOP" else (pr[1], pr[2])
        fn = _apply_pair(pr, name)
        corpus = {t: [rewrite(p, fn) for p in progs] for t, progs in corpus.items()}
        new.append(name)
        if log is not None:
            log.append({"name": name, "def": show(lib[name]), "weight": round(w, 3)})
    return corpus, lib, new


def _apply_routine(pat, name):
    def fn(seq):
        out, i = [], 0
        while i < len(seq):
            if tuple(seq[i:i + len(pat)]) == tuple(pat):
                out.append(name)
                i += len(pat)
            else:
                out.append(seq[i])
                i += 1
        return out
    return fn


# ===================================================================================================== the machine
class Cells:
    """The digit cells as tensors: sbb, adc [B, B, 2] -> digit, flag; mul [B, B, B, B] (a, x, h, k) -> low, high.
    Exact by default (for checks); the model's learned tables replace them (set)."""

    def __init__(self, B=10, device="cpu"):
        self.B = B
        a, b, c = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
        d = a - b - c
        self.sbb_d, self.sbb_c = d % B, (d < 0).long()
        d = a + b + c
        self.adc_d, self.adc_c = d % B, (d >= B).long()
        a4, x4, h4, k4 = torch.meshgrid(*(torch.arange(B),) * 4, indexing="ij")
        t = a4 + x4 * h4 + k4
        self.mul_lo, self.mul_hi = t % B, t // B
        self.to(device)

    def to(self, device):
        self.device = device
        for n in ("sbb_d", "sbb_c", "adc_d", "adc_c", "mul_lo", "mul_hi"):
            setattr(self, n, getattr(self, n).to(device))
        self._flat()
        return self

    def set(self, **tabs):
        for n, t in tabs.items():
            setattr(self, n, t.long().to(self.device).reshape(getattr(self, n).shape))
        self._flat()
        return self

    def _flat(self):
        B = self.B
        self.sbb = torch.stack([self.sbb_d, self.sbb_c], -1).reshape(B * B * 2, 2)
        self.adc = torch.stack([self.adc_d, self.adc_c], -1).reshape(B * B * 2, 2)
        self.mul = torch.stack([self.mul_lo, self.mul_hi], -1).reshape(B ** 4, 2)


class Setting:
    """A machine size: base B, operands of W digits, registers of P places (2 W unless given)."""

    def __init__(self, B, W, cells=None, P=0):
        self.B, self.W, self.cells = B, W, cells
        self.P = P or 2 * W

    def count(self, kind):
        return 1 if kind == ONE else {"P": self.P, "B": self.B, "W": self.W}[kind]

    def __repr__(self):
        return f"({self.B},{self.W})"


class Batch:
    """S machine states over NP probes: regs [S, 5, P, NP] uint8, c / h / k [S, NP], p [S]."""

    def __init__(self, regs, c, h, k, p):
        self.regs, self.c, self.h, self.k, self.p = regs, c, h, k, p

    @property
    def S(self):
        return self.regs.shape[0]

    def copy(self):
        return Batch(self.regs.clone(), self.c.clone(), self.h.clone(), self.k.clone(), self.p.clone())

    def take(self, idx):
        return Batch(self.regs[idx], self.c[idx], self.h[idx], self.k[idx], self.p[idx])

    @staticmethod
    def concat(bs):
        return Batch(*(torch.cat([getattr(b, f) for b in bs], 0) for f in ("regs", "c", "h", "k", "p")))

    def features(self):
        return torch.cat([self.regs.reshape(self.S, -1), self.c, self.h, self.k, self.p.to(torch.uint8)[:, None]], 1)


def gen_on(device, seed):
    """A random generator on the device (the machines' dirty starts and the probes' picks stay on it)."""
    return torch.Generator(device=device).manual_seed(seed)


def fresh(st, inputs, gen, NP, device):
    """One fresh machine (S = 1) over NP probes: inputs {reg: digits [NP, P]}; everything else random (a dirty
    start, so that every found program sets up what it reads)."""
    B, P = st.B, st.P
    rnd = lambda hi, shape: torch.randint(0, hi, shape, generator=gen, device=device, dtype=torch.uint8)
    regs = rnd(B, (1, 5, P, NP))
    for r, v in inputs.items():
        regs[0, RI[r]] = v.to(device=device, dtype=torch.uint8).T
    return Batch(regs, rnd(2, (1, NP)), rnd(B, (1, NP)), rnd(B, (1, NP)), torch.zeros(1, dtype=torch.long,
                                                                                         device=device))


def random_states(st, S, NP, gen, device):
    B, P = st.B, st.P
    rnd = lambda hi, shape: torch.randint(0, hi, shape, generator=gen, device=device, dtype=torch.uint8)
    return Batch(rnd(B, (S, 5, P, NP)), rnd(2, (S, NP)), rnd(B, (S, NP)), rnd(B, (S, NP)),
                 torch.randint(0, P, (S,), generator=gen, device=device))


def _shl(b, r):
    b.regs[:, r, 1:] = b.regs[:, r, :-1].clone()
    b.regs[:, r, 0] = 0


def _shr(b, r):
    b.regs[:, r, :-1] = b.regs[:, r, 1:].clone()
    b.regs[:, r, -1] = 0


def step(st, b, tok):
    """One primitive on batch b in place."""
    cl, B = st.cells, st.B
    op, _, arg = tok.partition(".")
    if op in ("SBB", "ADC", "MUL"):
        ar = torch.arange(b.S, device=b.regs.device)
        a, x = b.regs[ar, A_, b.p].long(), b.regs[ar, RI[arg], b.p].long()
        if op == "MUL":
            out = cl.mul[((a * B + x) * B + b.h.long()) * B + b.k.long()]
            b.k = out[..., 1].to(torch.uint8)
        else:
            out = (cl.sbb if op == "SBB" else cl.adc)[(a * B + x) * 2 + b.c.long()]
            b.c = out[..., 1].to(torch.uint8)
        b.regs[ar, A_, b.p] = out[..., 0].to(torch.uint8)
    elif op == "SEL":
        b.regs[:, A_] = torch.where(b.c[:, None, :].bool(), b.regs[:, S_], b.regs[:, A_])
    elif op == "LDA":
        b.regs[:, A_] = b.regs[:, RI[arg]]
    elif op == "STA":
        b.regs[:, RI[arg]] = b.regs[:, A_]
    elif op == "CLA":
        b.regs[:, A_] = 0
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
        top = b.regs[:, X_, -1].clone()
        _shl(b, A_)
        b.regs[:, A_, 0] = top
        _shl(b, X_)
    else:
        raise ValueError(tok)


def run_items(st, b, items, lib):
    for it in items:
        if is_loop(it):
            _, kind, body = it
            if kind in lib:
                run_items(st, b, fill(lib[kind], body[0]), lib)
            else:
                for _ in range(st.count(kind)):
                    run_items(st, b, body, lib)
        elif it in lib:
            if it.startswith("T."):                      # a routine restores the pointer it was called with
                p = b.p.clone()
                run_items(st, b, lib[it], lib)
                b.p = p
            else:
                run_items(st, b, lib[it], lib)
        else:
            step(st, b, it)


# ===================================================================================================== batched search
_GEN = torch.Generator().manual_seed(12345)
_MULT = torch.randint(-(1 << 62), 1 << 62, (8192,), generator=_GEN, dtype=torch.int64) * 2 + 1
_MULT_ON = {}


def _s64(v):
    return v - (1 << 64) if v >= 1 << 63 else v


_M2 = _s64(0x94D049BB133111EB)


def _lsr(h, s):
    return (h >> s) & ((1 << (64 - s)) - 1)


def hash_rows(F):
    """A 64-bit hash (int64, wrapping arithmetic) of every row of a uint8 matrix: a random odd linear combination of
    its 64-bit words, then a mix."""
    S, n = F.shape
    pad = (-n) % 8
    if pad:
        F = torch.cat([F, torch.zeros(S, pad, dtype=torch.uint8, device=F.device)], 1)
    Wd = F.contiguous().view(torch.int64)
    if F.device not in _MULT_ON:
        _MULT_ON[F.device] = _MULT.to(F.device)
    h = (Wd * _MULT_ON[F.device][:Wd.shape[1]][None]).sum(1)
    h = h ^ _lsr(h, 31)
    h = h * _M2
    return h ^ _lsr(h, 29)


def keys_of(batches):
    return hash_rows(torch.cat([b.features() for b in batches], 1))


def _put(out, idx, sub):
    out.regs[idx] = sub.regs
    out.c[idx] = sub.c
    out.h[idx] = sub.h
    out.k[idx] = sub.k
    out.p[idx] = sub.p


def _groups(ids):
    keep = torch.nonzero(ids >= 0).flatten()
    if keep.numel() == 0:
        return []
    order = keep[torch.sort(ids[keep], stable=True).indices]
    uniq, counts = torch.unique_consecutive(ids[order], return_counts=True)
    out, start = [], 0
    for u, c in zip(uniq.tolist(), counts.tolist()):
        out.append((u, order[start:start + c]))
        start += c
    return out


def apply_items(st, b, item_of_row, items, lib):
    """A new batch: row r of b with items[item_of_row[r]] applied; tokens run once per distinct token, loops run their
    iterations together with the body's elements grouped position by position."""
    out = b.copy()
    dev = b.regs.device
    loop = [is_loop(it) and it[1] not in lib for it in items]
    is_l = torch.tensor(loop, dtype=torch.bool, device=dev)[item_of_row]
    for u, idx in _groups(torch.where(~is_l, item_of_row, torch.full_like(item_of_row, -1))):
        sub = out.take(idx)
        run_items(st, sub, (items[u],), lib)
        _put(out, idx, sub)
    if not any(loop):
        return out
    elems, eid = {}, []
    L = max((len(it[2]) for it, lp in zip(items, loop) if lp), default=0)
    cnt = torch.zeros(len(items), dtype=torch.long)
    table = torch.full((len(items), max(L, 1)), -1, dtype=torch.long)
    for i, (it, lp) in enumerate(zip(items, loop)):
        if lp:
            cnt[i] = st.count(it[1])
            for j, e in enumerate(it[2]):
                if e not in elems:
                    elems[e] = len(elems)
                    eid.append(e)
                table[i, j] = elems[e]
    rcnt = torch.where(is_l, cnt.to(dev)[item_of_row], torch.zeros_like(item_of_row))
    rtab = table.to(dev)[item_of_row]
    for itn in range(int(rcnt.max())):
        live = rcnt > itn
        for j in range(L):
            for u, idx in _groups(torch.where(live, rtab[:, j], torch.full_like(item_of_row, -1))):
                sub = out.take(idx)
                run_items(st, sub, (eid[u],), lib)
                _put(out, idx, sub)
    return out


class ItemClass:
    def __init__(self, rep, cost):
        self.rep, self.cost = rep, cost


class Deadline(Exception):
    """The training's deadline passed inside a climb unit (the unit returns at once; nothing is cached)."""


def _late(deadline):
    return deadline is not None and time.time() > deadline


def _signatures(settings, tests, items, lib, chunk=1 << 21, deadline=None):
    S = tests[0].S
    out = []
    step_ = max(1, chunk // S)
    for i0 in range(0, len(items), step_):
        if _late(deadline):
            raise Deadline()
        part = items[i0:i0 + step_]
        k = len(part)
        dev = tests[0].regs.device
        ri = torch.arange(k, device=dev).repeat_interleave(S)
        rs = torch.arange(S, device=dev).repeat(k)
        outs = [apply_items(st, t.take(rs), ri, part, lib) for st, t in zip(settings, tests)]
        keys = keys_of(outs).view(k, S)
        out.append(hash_rows(keys.contiguous().view(torch.uint8).view(k, S * 8)))
    return torch.cat(out).tolist() if out else []


def enumerate_items(tokens, lib, settings, device, max_body=2, max_arity=3, seed=7, S_test=24, NP_test=6,
                    deadline=None, cache=None):
    """Every distinct item (tokens, then loops over bodies of cost up to max_body built from the distinct items so
    far), the cheapest spelling first: items that agree on S_test random machines of every setting are one item.
    Raises Deadline if the deadline passes. cache: {item: signature} kept across calls with the same settings (the
    test machines are the same every call and an item's meaning never changes, so only new items are run)."""
    gen = gen_on(device, seed)
    tests = [random_states(st, S_test, NP_test, gen, device) for st in settings]

    def sigs(items):
        if cache is None:
            return _signatures(settings, tests, items, lib, deadline=deadline)
        miss = list(dict.fromkeys(it for it in items if it not in cache))
        if miss:
            for it, sg in zip(miss, _signatures(settings, tests, miss, lib, deadline=deadline)):
                cache[it] = sg
        return [cache[it] for it in items]
    ident = sigs([("REP", ONE, ())])[0]
    classes, by_sig = [], {ident}

    def add_all(cands, c):
        for item, s in zip(cands, sigs(cands)):
            if s not in by_sig:
                by_sig.add(s)
                classes.append(ItemClass(item, c))

    add_all(list(tokens), 1)
    body_sig = {ident}
    for bc in range(1, max_body + 1):
        reps, cands = list(classes), []

        def extend(prefix, pc, n):
            if pc == bc:
                cands.append(prefix)
                return
            if n == max_arity:
                return
            for cl in reps:
                if pc + cl.cost <= bc:
                    extend(prefix + (cl.rep,), pc + cl.cost, n + 1)
        extend((), 0, 0)
        found = []
        for body, s in zip(cands, sigs([("REP", ONE, b) for b in cands])):
            if s not in body_sig:
                body_sig.add(s)
                found.append(body)
        add_all([("REP", kind, body) for body in found for kind in LOOP_KINDS], bc + 1)
    return classes


class Problem:
    """A task on some settings: one fresh machine per setting, the targets ({"A": digits [P, NP]} and/or {"c": [NP]})
    and an optional extra check per setting (a function of the candidates' A digits [S, NP, P] -> bool [S])."""

    def __init__(self, task, settings, roots, targets, inverse=None):
        self.task, self.settings, self.roots, self.targets = task, settings, roots, targets
        self.inverse = inverse or [None] * len(settings)

    def goal(self, batches):
        ok = torch.ones(batches[0].S, dtype=torch.bool, device=batches[0].regs.device)
        for b, t, inv in zip(batches, self.targets, self.inverse):
            if "A" in t:
                ok &= (b.regs[:, A_] == t["A"][None]).all(-1).all(-1)
            if "c" in t:
                ok &= (b.c == t["c"][None]).all(-1)
            if inv is not None and bool(ok.any()):
                idx = torch.nonzero(ok).flatten()
                ok[idx[~inv(b.regs[idx, A_].permute(0, 2, 1))]] = False
        return ok


class Result:
    def __init__(self, task):
        self.task, self.found_cost, self.goals, self.level_sizes, self.stop = task, None, None, [], ""
        self.parents, self.aliases, self.accepted = None, [], None


def _fresh(seen, keys):
    if seen.numel() == 0:
        return torch.ones(keys.numel(), dtype=torch.bool, device=keys.device)
    pos = torch.searchsorted(seen, keys)
    pos[pos == seen.numel()] = 0
    return seen[pos] != keys


def bfs(prob, classes, lib, max_cost=8, max_states=400_000, accept=None, limit=200, deadline=None, chunk=1 << 21,
        listing="spellings", max_goals=50_000, spell=8, batch=256):
    """Breadth-first by cost, states merged over every probe of every setting, every minimal-cost parent kept. Stops
    at the first cost whose goal programs pass `accept`, when a level would pass max_states (that level is finished
    streaming goal states only), or at the deadline. The goal programs tested: the first `limit` spellings
    (listing "spellings"), or one spelling per goal state (listing "goals"), tested in batches, fewest top-level items
    first (then fewest primitives), until a batch has one that passes; the accepted goal states then give up to `spell`
    spellings each that pass."""
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
    by_cost = {}
    for ci, cl in enumerate(classes):
        by_cost.setdefault(cl.cost, []).append(ci)
    by_cost = {c: torch.tensor(v, dtype=torch.long, device=dev) for c, v in by_cost.items()}
    for d in range(1, max_cost + 1):
        last = d == max_cost
        ck, cc, cs, ci_, chunks, gk = [], [], [], [], [], []
        n_cand = n_fresh = 0
        late = False
        for c, cis in by_cost.items():
            sl = d - c
            if sl < 0 or levels[sl] is None or late:
                continue
            S_src = levels[sl][0].S
            for s0 in range(0, S_src, chunk):
                if late:
                    break
                srcs = [b.take(slice(s0, s0 + chunk)) for b in levels[sl]]
                ns = srcs[0].S
                i_step = max(1, chunk // ns)
                for i0 in range(0, cis.numel(), i_step):
                    if _late(deadline):
                        late = True
                        break
                    ii = cis[i0:i0 + i_step]
                    k = ii.numel()
                    ri = torch.arange(k, device=dev).repeat_interleave(ns)
                    rs = torch.arange(ns, device=dev).repeat(k)
                    part = [classes[i].rep for i in ii.tolist()]
                    outs = [apply_items(st, b.take(rs), ri, part, lib) for st, b in zip(settings, srcs)]
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
                    cc.append(ii[ri[idx]])
                    cs.append(torch.full((idx.numel(),), sl, dtype=torch.long, device=dev))
                    ci_.append(rs[idx] + s0)
                    chunks.append(sub)
                    n_cand += idx.numel()
                    if not last and n_cand > 1.5 * max_states:
                        last = True
                        res.stop = f"budget: level {d} streamed (over {max_states} states)"
                        new = [[], [], [], [], []]
                        for j in range(len(chunks)):
                            gm = torch.nonzero(prob.goal(chunks[j])).flatten()
                            if gm.numel():
                                for lst, t in zip(new, (ck[j], cc[j], cs[j], ci_[j])):
                                    lst.append(t[gm])
                                new[4].append([b.take(gm) for b in chunks[j]])
                        ck, cc, cs, ci_, chunks = new
            if late or _late(deadline):
                res.stop = f"deadline at level {d}"
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
            parents.append((child[order], torch.cat(cc)[order], torch.cat(cs)[order], torch.cat(ci_)[order]))
            if not last and not res.stop:
                parts, off = [[] for _ in settings], 0
                for j in range(len(chunks)):
                    ch, n = chunks[j], chunks[j][0].S
                    lo_, hi_ = int(torch.searchsorted(sfirst, off)), int(torch.searchsorted(sfirst, off + n))
                    rows = sfirst[lo_:hi_] - off
                    for si in range(len(settings)):
                        parts[si].append(ch[si].take(rows))
                    off += n
                levels.append([Batch.concat(p) for p in parts])
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
        if gk:
            g = torch.unique(torch.cat(gk))
            res.found_cost = d
            res.goals = rank[torch.searchsorted(uk, g)].tolist()
            res.parents = parents
            if listing == "goals":
                cand = []
                for gi, g in enumerate(res.goals[:max_goals]):
                    if gi % 256 == 0 and _late(deadline):
                        res.stop = res.stop or f"deadline at level {d}"
                        break
                    f = programs(_OneGoal(res, g), classes, limit=1)
                    if f:
                        cand.append((len(f[0]), cost(expand(f[0], lib)), f[0], g))
                cand.sort(key=lambda c: (c[0], c[1]))
                progs = [c[2] for c in cand]
                if accept is None:
                    res.accepted = progs
                    break
                good = []
                for i0 in range(0, len(cand), batch):
                    if _late(deadline):
                        res.stop = res.stop or f"deadline at level {d}"
                        break
                    good = [c for c, o in zip(cand[i0:i0 + batch], accept(progs[i0:i0 + batch])) if o]
                    if good:
                        break
                if good:
                    res.accepted = [c[2] for c in good]
                    if spell > 1:                              # more spellings of the accepted goal states, for BPE
                        more = [q for c in good for q in programs(_OneGoal(res, c[3]), classes, limit=spell)[1:]]
                        res.accepted += [q for q, o in zip(more, accept(more) if more else []) if o]
                    break
                res.aliases.append((d, [show(p) for p in progs[:20]]))
                res.found_cost, res.goals = None, None
                if res.stop:
                    break
                seen = torch.sort(torch.cat([seen, uk])).values
                continue
            progs = programs(res, classes, limit=limit)
            if accept is None:
                res.accepted = progs
                break
            ok = accept(progs)
            if any(ok):
                res.accepted = [p for p, o in zip(progs, ok) if o]
                break
            res.aliases.append((d, [show(p) for p in progs[:20]]))
            res.found_cost, res.goals = None, None
        if res.stop:
            break
        seen = torch.sort(torch.cat([seen, uk])).values
    if res.found_cost is None and not res.stop:
        res.stop = f"exhausted cost {max_cost}"
    res.parents = parents
    return res


class _OneGoal:
    def __init__(self, res, g):
        self.parents, self.found_cost, self.goals = res.parents, res.found_cost, [g]


def programs(res, classes, limit=200):
    """Up to `limit` minimal programs reaching the goal states."""
    memo = {}

    def paths(level, idx):
        if level == 0:
            return [()]
        if (level, idx) in memo:
            return memo[(level, idx)]
        out = []
        child, pcl, psl, psi = res.parents[level]
        key = torch.tensor([idx], device=child.device)
        lo, hi = int(torch.searchsorted(child, key)), int(torch.searchsorted(child, key, right=True))
        for ci, sl, si in zip(pcl[lo:hi].tolist(), psl[lo:hi].tolist(), psi[lo:hi].tolist()):
            for p in paths(sl, si):
                out.append(p + (classes[ci].rep,))
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        memo[(level, idx)] = out
        return out

    progs = []
    for g in res.goals:
        progs.extend(paths(res.found_cost, g)[:limit - len(progs)])
        if len(progs) >= limit:
            break
    return progs


# ===================================================================================================== practice tasks
NEEDS = {"lt": (), "modx": (), "add": ("lt", "modx"), "modB": ("add",), "mrow": ("add",), "mac": ("mrow", "add"),
         "mul": ("add",), "step": ("modB",), "reduce": ("add",)}
SWAP = {"X": "Y", "Y": "X", "N": "N"}
MAC_ADDEND = "N"           # mac's addend: N (x d + N) or the running sum in A (A + x d, the A/B's variant C)


def task_names(variants=True):
    out = []
    for t in NEEDS:
        out.append(t)
        if variants:
            out.append(t + "~")
    return out


def needs(task):
    return tuple(n + ("~" if task.endswith("~") else "") for n in NEEDS[task.rstrip("~")])


def top_parts(Nd, Xd, w):
    """N's top w digits N' and x's digits at the same places x' [n, w] (digit strings only)."""
    L = Nd.shape[1]
    top = torch.where(Nd != 0, torch.arange(L, device=Nd.device)[None], -1).max(1).values
    lo = (top - w + 1).clamp(min=0)
    idx = lo[:, None] + torch.arange(w, device=Nd.device)[None]
    valid = idx <= top[:, None]
    idx = idx.clamp(max=L - 1)
    return Nd.gather(1, idx) * valid, Xd.gather(1, idx) * valid


def pad(d, P):
    if d.shape[1] >= P:
        return d[:, :P]
    return torch.cat([d, torch.zeros(d.shape[0], P - d.shape[1], dtype=d.dtype, device=d.device)], 1)


def width(task, st):
    base = task.rstrip("~")
    if base in ("lt", "modx"):
        return st.P
    if base in ("add", "modB", "mrow", "step"):
        return st.P - 1
    return st.W


def run_prog(st, prog, lib, ins, gen, device):
    """A found program on inputs {reg: digits [n, P]}: (A digits [n, P], c bits [n]), by the machine's cells."""
    n = next(iter(ins.values())).shape[0]
    b = fresh(st, ins, gen, n, device)
    run_items(st, b, prog, lib)
    return b.regs[0, A_].T.long(), b.c[0].long()


def count_of(d):
    """A digit string's count (how many times to repeat an addition): read off its digits."""
    v = torch.zeros(d.shape[0], dtype=torch.long, device=d.device)
    for j in reversed(range(d.shape[1])):
        v = v * 10 + d[:, j]
    return v


def build(task, st, n, gen, rows, helpers, lib, device):
    """Probes of one task at one setting from the rows' own digits (rows: (Nd, Xd) [R, L]): inputs, targets, check."""
    base, swap = task.rstrip("~"), task.endswith("~")
    reg = (lambda r: SWAP[r]) if swap else (lambda r: r)
    P, B = st.P, st.B
    Nd, Xd = rows
    pick = torch.randint(0, Nd.shape[0], (4 * n,), generator=gen, device=device)
    Nt, xt = top_parts(Nd[pick], Xd[pick], width(task, st))
    Nt, xt, other = pad(Nt[:n], P), pad(xt[:n], P), pad(xt[n:2 * n], P)
    eq = (Nt == xt).all(1)
    zero = torch.zeros_like(Nt)
    run = lambda name, ins: run_prog(st, helpers[name], lib, {reg(k): v for k, v in ins.items()}, gen, device)
    qd = torch.arange(n, device=device) % B
    qdig = pad(qd[:, None], P)

    def repeat_add(D, Nv, times):
        D = D.clone()
        for k in range(int(times.max()) if times.numel() else 0):
            more = times > k
            if bool(more.any()):
                D[more] = run("add", {"X": D[more], "N": Nv[more]})[0]
        return D
    if base == "lt":
        kind = torch.arange(n, device=device) % 4
        Xv = torch.where((kind == 2)[:, None], Nt, xt)
        Nv = torch.where((kind < 2)[:, None], Nt, xt)
        return {reg("X"): Xv, "N": Nv}, {"c": torch.where(kind < 2, (~eq).long(), 0)}, None
    if base == "modx":
        kind = torch.arange(n, device=device) % 3
        Xv = torch.where((kind < 2)[:, None], xt, Nt)
        return {reg("X"): Xv, "N": Nt}, {"A": torch.where(((kind < 2) & ~eq)[:, None], xt, zero)}, None
    if base == "add":
        keep = torch.nonzero(~eq).flatten()
        Nk, xk = Nt[keep], xt[keep]
        kind = torch.arange(keep.numel(), device=device) % 4
        Nv = torch.where((kind < 3)[:, None], Nk, torch.zeros_like(Nk))

        def inverse(A_out):
            S = A_out.shape[0]
            A_out = A_out.long()
            sel = torch.nonzero(kind < 3).flatten()
            flat = A_out[:, sel].reshape(-1, P)
            Ns = Nk[sel].repeat(S, 1)
            ok = (run("modx", {"X": flat, "N": Ns})[0].view(S, -1, P) == xk[sel][None]).all(-1).all(-1)
            ok &= (run("lt", {"X": flat, "N": Ns})[1].view(S, -1) == 0).all(-1)
            sel0 = torch.nonzero(kind == 3).flatten()
            return ok & (A_out[:, sel0] == xk[sel0][None]).all(-1).all(-1)
        return {reg("X"): xk, "N": Nv}, {}, inverse
    if base == "modB":
        return {reg("X"): repeat_add(xt, Nt, qd), "N": Nt}, {"A": torch.where(eq[:, None], zero, xt)}, None
    if base == "mrow":
        return {reg("X"): xt, reg("Y"): qdig}, {"A": repeat_add(zero, xt, qd)}, None
    if base == "mac":
        R = run("mrow", {"X": xt, "Y": qdig})[0]
        if MAC_ADDEND == "A":
            return {reg("X"): xt, reg("Y"): qdig, "A": other}, {"A": run("add", {"X": R, "N": other})[0]}, None
        return {reg("X"): xt, reg("Y"): qdig, "N": Nt}, {"A": run("add", {"X": R, "N": Nt})[0]}, None
    if base == "mul":
        return {reg("X"): xt, reg("Y"): other}, {"A": repeat_add(zero, xt, count_of(other))}, None
    if base == "step":
        keep = torch.nonzero(~eq).flatten()
        Nk, xk = Nt[keep], xt[keep]
        d = torch.arange(keep.numel(), device=device) % B
        Xv = torch.zeros_like(Nk)
        Xv[:, P - 1] = d                                                     # the digit d at X's top place
        v = torch.cat([d[:, None], xk[:, :P - 1]], 1)                        # r B + d: r's digits moved up, d below
        return {reg("X"): Xv, reg("Y"): xk, "N": Nk}, {"A": run("modB", {"X": v, "N": Nk})[0]}, None
    if base == "reduce":
        return ({reg("X"): repeat_add(xt, Nt, count_of(other)), "N": Nt},
                {"A": torch.where(eq[:, None], zero, xt)}, None)
    raise ValueError(task)


class LawProblem(Problem):
    def __init__(self, task, settings, NP, seed, rows, helpers, lib, device):
        gen = gen_on(device, seed)
        roots, targets, inverse = [], [], []
        for st in settings:
            ins, tgt, inv = build(task, st, 2 * NP if task.rstrip("~") == "lt" else NP, gen, rows, helpers, lib,
                                  device)
            n = next(iter(ins.values())).shape[0]
            roots.append(fresh(st, ins, gen, n, device))
            t = {}
            if "A" in tgt:
                t["A"] = tgt["A"].T.to(torch.uint8)
            if "c" in tgt:
                t["c"] = tgt["c"].to(torch.uint8)
            targets.append(t)
            inverse.append(inv)
        super().__init__(task, settings, roots, targets, inverse)


class Acceptor:
    """A task's acceptance probes built once per search (accept() builds the same ones, from the same seed, for every
    program): a program passes if it reaches the goal at every setting."""

    def __init__(self, task, lib, helpers, rows, settings, device, n=60, seed=11):
        self.lib, self.cache = lib, {}
        self.probs = [(st, LawProblem(task, [st], n, seed, rows, helpers, lib, device)) for st in settings]

    def __call__(self, prog):
        k = show(prog)
        if k not in self.cache:
            ok = True
            for st, prob in self.probs:
                b = prob.roots[0].copy()
                run_items(st, b, prog, self.lib)
                if not bool(prob.goal([b])[0]):
                    ok = False
                    break
            self.cache[k] = ok
        return self.cache[k]

    def many(self, progs):
        """The same verdicts for many programs at once: one machine per program, the programs run in lockstep over
        their top-level items (each step one apply_items over every machine), setting by setting, a program leaving at
        its first failing setting."""
        keys = [show(p) for p in progs]
        live = sorted({i for i, k in enumerate(keys) if k not in self.cache}, key=lambda i: keys[i])
        live = [i for j, i in enumerate(live) if j == 0 or keys[i] != keys[live[j - 1]]]
        for st, prob in self.probs:
            if not live:
                break
            dev = prob.roots[0].regs.device
            b = prob.roots[0].take(torch.zeros(len(live), dtype=torch.long, device=dev))
            ps = [progs[i] for i in live]
            for t in range(max(len(p) for p in ps)):
                step_items = [p[t] if len(p) > t else ("REP", ONE, ()) for p in ps]
                index = {}
                for it in step_items:
                    index.setdefault(it, len(index))
                ids = torch.tensor([index[it] for it in step_items], dtype=torch.long, device=dev)
                b = apply_items(st, b, ids, list(index), self.lib)
            ok = prob.goal([b]).tolist()
            for i, o in zip(live, ok):
                if not o:
                    self.cache[keys[i]] = False
            live = [i for i, o in zip(live, ok) if o]
        for i in live:
            self.cache[keys[i]] = True
        return [self.cache[k] for k in keys]


def accept(task, prog, lib, helpers, rows, settings, device, n=60, seed=11):
    """Does prog pass its task's laws on fresh rows at every setting?"""
    for st in settings:
        prob = LawProblem(task, [st], n, seed, rows, helpers, lib, device)
        b = prob.roots[0].copy()
        run_items(st, b, prog, lib)
        if not bool(prob.goal([b])[0]):
            return False
    return True


# ===================================================================================================== the round
def round_settings(cells, P):
    return Setting(10, P // 2, cells, P)


def round_problem(cells, rows, NP, P, device, seed=0):
    """Probes at the task's size: x in X, N in N, the target the row's answer y in A (rows: Nd, Xd, Yd)."""
    Nd, Xd, Yd = rows
    st = round_settings(cells, P)
    gen = gen_on(device, seed)
    root = fresh(st, {"X": pad(Xd[:NP], P), "N": pad(Nd[:NP], P)}, gen, min(NP, Nd.shape[0]), device)
    return Problem("round", [st], [root], [{"A": pad(Yd[:NP], P).T.to(torch.uint8)}])


def round_fit(prog, lib, cells, rows, P, device, seed=1):
    """The share of answer rows (Nd, Xd, Yd) the program reproduces at the task's size."""
    Nd, Xd, Yd = rows
    st = round_settings(cells, P)
    A, _ = run_prog(st, prog, lib, {"X": pad(Xd, P), "N": pad(Nd, P)}, gen_on(device, seed), device)
    return float((A[:, :Yd.shape[1]] == Yd).all(1).float().mean())


# ===================================================================================================== the climb
def steps(items, lib, P=4, B=10, W=2):
    """How many primitive instructions a program executes at a setting (base B, W-digit operands, P places)."""
    n = 0
    for it in items:
        if is_loop(it) and it[1] in lib:
            n += steps(fill(lib[it[1]], it[2][0]), lib, P, B, W)
        elif is_loop(it):
            n += {"P": P, "B": B, "W": W, ONE: 1}[it[1]] * steps(it[2], lib, P, B, W)
        elif it in lib:
            n += steps(lib[it], lib, P, B, W)
        else:
            n += 1
    return n


def simplest(progs, lib, by_steps=False):
    """Occam: the program with the fewest primitives written; with by_steps, ties go to the one that executes fewer
    (Levin's time term), else to the first found."""
    if by_steps:
        return min(progs, key=lambda p: (cost(expand(p, lib)), steps(p, lib)))
    return min(progs, key=lambda p: cost(expand(p, lib)))


class Climb:
    """The wake-sleep loop as a state machine a training loop drives (`step`: one unit of work): every task whose
    helpers are found is searched over the primitives plus the library; at the end of a level the corpus is merged
    (BPE), each task keeps its simplest program (Occam), newly solved tasks become routines (T.<task>, which restore
    the pointer), and the round is tried by composition against the answers whenever new routines appeared.

    The A/B's rules (INTEGRATION.md §2.1; all off by default, which is variant A):
      comp_first: a task is first searched over the solved routines and the primitives, with loops whose bodies (up to
        comp_body) are made of those, then over the library, then the primitives alone;
      skip_twins: a twin (task~) whose original is solved and whose own search used up its cap is not searched
        again (an original is always searched: the other originals need it as their helper);
      stall_after: a task whose searches used up their caps this many times is a stall: from then on only the
        composition search runs for it, and only when new routines appeared since its last one;
      search_s: one search's time limit (its stop counts as a used-up cap);
      listing="goals": every goal state is tested (bfs), with the acceptance probes built once per search;
      occam_steps: Occam's ties go to the program that executes fewer steps;
      merged_routines: a solved task whose whole program BPE merged into one item still becomes a routine (by its
        expanded size; else the multiply can end up a merge the round's search does not see)."""

    def __init__(self, cells, device, P_round, budget=3_000_000, fallback_budget=16_000_000, max_body=3, max_cost=7,
                 min_count=2.0, max_merges=12, levels=12, NP=8, round_budget=4_000_000, round_cost=5,
                 settings=((10, 1), (10, 2)), verify=((10, 1), (10, 2), (10, 3)), chunk=1 << 21,
                 comp_first=False, comp_body=3, skip_twins=False, stall_after=0, search_s=None, listing="spellings",
                 occam_steps=False, merged_routines=False):
        self.cells, self.device, self.P_round = cells, device, P_round
        self.tasks = task_names()
        self.budget, self.fallback_budget, self.max_body, self.max_cost = budget, fallback_budget, max_body, max_cost
        self.min_count, self.max_merges, self.levels, self.NP = min_count, max_merges, levels, NP
        self.round_budget, self.round_cost, self.chunk = round_budget, round_cost, chunk
        self.settings = [Setting(B, W, cells) for B, W in settings]
        self.verify = [Setting(B, W, cells) for B, W in verify]
        self.corpus, self.lib, self.solved_at, self.level = {}, {}, {}, 0
        self.queue, self.records, self.history = None, [], []
        self.done = False
        self.rounds = []                      # accepted round programs
        self.fallback_failed = set()          # tasks whose primitives-alone search failed (it never changes)
        self.round_tried = None               # the routines the last round search had
        self._classes = None
        self.comp_first, self.comp_body, self.skip_twins = comp_first, comp_body, skip_twins
        self.stall_after, self.search_s, self.listing = stall_after, search_s, listing
        self.occam_steps, self.merged_routines = occam_steps, merged_routines
        self._sig_cache = {}                  # item -> signature on the settings' test machines (enumerate_items)
        self.used_up = Counter()              # task -> searches that used up their caps without a program
        self.stalled, self.skipped = {}, {}   # task -> the level from which it is not searched (fully)
        self.comp_seen = {}                   # task -> the routines its last composition search had
        self.events, self._comp = [], None

    def helpers_for(self, task):
        return {n.rstrip("~"): simplest(self.corpus[n], self.lib, self.occam_steps) for n in needs(task)}

    def routines(self):
        return tuple(sorted(k for k in self.lib if k.startswith("T.")))

    def todo(self):
        out = []
        for t in self.tasks:
            if t in self.corpus or not all(n in self.corpus for n in needs(t)) or t in self.skipped:
                continue
            if (self.skip_twins and t.endswith("~") and t not in self.stalled and t[:-1] in self.corpus
                    and self.used_up[t] >= 1):
                self.skipped[t] = self.level
                self.events.append(f"L{self.level} TWIN {t}: not searched again (its original {t[:-1]} is solved; "
                                   f"its own search used up its cap)")
                continue
            if self.stall_after and t not in self.stalled and self.used_up[t] >= self.stall_after:
                self.stalled[t] = self.level
                self.events.append(f"L{self.level} STALL {t}: its searches used up their caps {self.used_up[t]} "
                                   f"times; from now on only compositions, when new routines appear")
            if t in self.stalled and self.comp_seen.get(t) == self.routines():
                continue
            out.append(t)
        return out

    def comp_classes(self, deadline=None):
        """The composition search's vocabulary: the routines and the primitives, and loops whose bodies (cost <=
        comp_body) are made of those; the merges appear only inside the routines."""
        routines = self.routines()
        if self._comp is None or self._comp[0] != routines:
            self._comp = (routines, enumerate_items(list(PRIMS) + list(routines), self.lib, self.settings,
                                                    self.device, max_body=self.comp_body, deadline=deadline,
                                                    cache=self._sig_cache))
        return self._comp[1]

    def classes(self, prims_only=False, deadline=None):
        key = ("prims" if prims_only else "lib", tuple(sorted(self.lib)))
        if self._classes is None or self._classes[0] != key:
            tokens = list(PRIMS) + ([] if prims_only else [k for k in self.lib])
            self._classes = (key, enumerate_items(tokens, {} if prims_only else self.lib, self.settings, self.device,
                                                  max_body=self.max_body, deadline=deadline, cache=self._sig_cache))
        return self._classes[1]

    def search(self, task, rows, deadline=None):
        t0 = time.time()
        climb_deadline = deadline
        if self.search_s:
            deadline = min(deadline or float("inf"), t0 + self.search_s)
        helpers = self.helpers_for(task)
        prob = LawProblem(task, self.settings, self.NP, 0, rows, helpers, self.lib, self.device)
        if self.listing == "goals":
            ok = Acceptor(task, self.lib, helpers, rows, self.verify, self.device)
        else:
            cache = {}

            def ok(p):
                k = show(p)
                if k not in cache:
                    cache[k] = accept(task, p, self.lib, helpers, rows, self.verify, self.device)
                return cache[k]
        runs = []

        def run(how, classes, lib, cap):
            t1 = time.time()
            r = bfs(prob, classes, lib, max_cost=self.max_cost, max_states=cap,
                    accept=ok.many if self.listing == "goals" else (lambda ps: [ok(p) for p in ps]),
                    deadline=deadline, chunk=self.chunk, listing=self.listing)
            runs.append(dict(how=how, found_cost=r.found_cost, level_sizes=r.level_sizes, stop=r.stop,
                             seconds=round(time.time() - t1, 1)))
            return r
        res, how = None, "library"
        try:
            if self.comp_first and self.routines():
                res, how = run("routines", self.comp_classes(deadline), self.lib, self.budget), "routines"
                self.comp_seen[task] = self.routines()
            if res is None or (res.found_cost is None and task not in self.stalled):
                res, how = run("library", self.classes(deadline=deadline), self.lib, self.budget), "library"
            if res.found_cost is None and self.lib and task not in self.fallback_failed and task not in self.stalled:
                # a program using no merge: the primitives alone, a bigger budget (the same search at every level, so
                # a task whose fallback failed once is not tried again)
                fb = run("primitives", self.classes(prims_only=True, deadline=deadline), {}, self.fallback_budget)
                if fb.found_cost is not None:
                    res, how = fb, "primitives"
                else:
                    self.fallback_failed.add(task)
        except Deadline:                       # the deadline passed while a vocabulary was being built
            if res is None:
                res = Result(task)
            res.stop = res.stop or "deadline (building the vocabulary)"
        if res.found_cost is None and not (climb_deadline is not None and time.time() > climb_deadline):
            self.used_up[task] += 1
        rec = dict(task=task, level=self.level, found_cost=res.found_cost, level_sizes=res.level_sizes,
                   seconds=round(time.time() - t0, 1), stop=res.stop, how=how,
                   programs=[show(p) for p in (res.accepted or [])][:8], runs=runs)
        return res.accepted or [], rec

    def search_round(self, answers, deadline=None):
        """The round by composition: the routines and the straight-line primitives at the task's size, the goal the
        answers (answers: Nd, Xd, Yd of the answer rows; the first NP are the probes, all of them accept)."""
        t0 = time.time()
        routines = [k for k in self.lib if k.startswith("T.")]
        prob = round_problem(self.cells, answers, self.NP, self.P_round, self.device)
        try:
            classes = enumerate_items(list(PRIMS) + routines, self.lib, prob.settings, self.device, max_body=0,
                                      deadline=deadline)
            res = bfs(prob, classes, self.lib, max_cost=self.round_cost, max_states=self.round_budget,
                      accept=lambda ps: [round_fit(p, self.lib, self.cells, answers, self.P_round, self.device) == 1.0
                                         for p in ps], deadline=deadline, chunk=max(1, self.chunk // 4))
        except Deadline:
            res = Result("round")
            res.stop = "deadline (building the vocabulary)"
        rec = dict(task="round", level=self.level, found_cost=res.found_cost, level_sizes=res.level_sizes,
                   seconds=round(time.time() - t0, 1), stop=res.stop, routines=len(routines),
                   programs=[show(p) for p in (res.accepted or [])][:8])
        return res.accepted or [], rec

    def step(self, rows, answers=None, deadline=None):
        """One unit of work: the next search of this level, the level's sleep phase, or (after a sleep that made new
        routines, with answers given) the round. Returns a log line (a rule's event first, if one is pending), or None
        when there is nothing left."""
        if not self.events:
            line = self._step(rows, answers, deadline)
            if line is not None:
                self.events.append(line)
        return self.events.pop(0) if self.events else None

    def _step(self, rows, answers=None, deadline=None):
        if self.done:
            return None
        if self.queue is None:
            routines = tuple(sorted(k for k in self.lib if k.startswith("T.")))
            if answers is not None and not self.rounds and routines and routines != self.round_tried:
                self.round_tried = routines
                progs, rec = self.search_round(answers, deadline)
                self.records.append(rec)
                self.rounds = progs
                return (f"L{self.level} round: cost {rec['found_cost']} levels {rec['level_sizes']} {rec['seconds']}s "
                        f"{rec['stop']} {rec['programs'][:3]}")
            self.queue = self.todo()
            self.solved_now = []
            if not self.queue or self.level >= self.levels:
                self.done = True
                self.queue = []
                return f"climb done at level {self.level}: solved {sorted(self.corpus)}"
        if self.queue:
            task = self.queue.pop(0)
            progs, rec = self.search(task, rows, deadline)
            self.records.append(rec)
            if progs:
                self.corpus[task] = progs
                self.solved_at[task] = self.level
                self.solved_now.append(task)
            return (f"L{self.level} {task:8s} cost {rec['found_cost']} ({rec['how']}) levels {rec['level_sizes']} "
                    f"{rec['seconds']}s {rec['stop']} {rec['programs'][:2]}")
        mlog = []
        self.corpus, self.lib, new = bpe(self.corpus, self.lib, max_merges=self.max_merges, min_count=self.min_count,
                                         prefix=f"B{self.level}.", log=mlog)
        self.corpus = {t: [simplest(ps, self.lib, self.occam_steps)] for t, ps in self.corpus.items()}
        for t in self.solved_now:                    # every newly solved task's simplest program becomes a routine
            prog, name = simplest(self.corpus[t], self.lib, self.occam_steps), f"T.{t}"
            if (cost(expand(prog, self.lib)) if self.merged_routines else cost(prog)) >= 2 and name not in self.lib:
                self.lib[name] = prog
                fn = _apply_routine(prog, name)
                self.corpus = {u: [rewrite(p, fn) for p in ps] for u, ps in self.corpus.items()}
                mlog.append({"name": name, "def": show(prog), "routine": True})
        self.history.append(dict(level=self.level, solved=list(self.solved_now), merges=mlog,
                                 lib={k: show(v) for k, v in self.lib.items()}))
        line = f"L{self.level} sleep: solved {self.solved_now}; " + "; ".join(f"{m['name']}={m['def']}" for m in mlog)
        if not mlog and not self.solved_now:
            self.done = True
        self.level += 1
        self.queue = None
        return line


# ===================================================================================================== the one-hot machine
class SteMachine:
    """A found program on one-hot digits (the prediction path): every cell a straight-through table (rows (a, b, c) ->
    digit 10 + flag 2; (a, x, h, k) -> low 10 + high 10), the pointer a Python int, registers immutable tensors (so
    autograd keeps every earlier value); every cell's and gate's output passes `damp` (its gradient scaled)."""

    def __init__(self, SBB, ADC, MUL, P, damp, B=10):
        self.SBB, self.ADC, self.MUL, self.P, self.B, self.damp = SBB, ADC, MUL, P, B, damp

    def count(self, kind):
        return {"P": self.P, "B": self.B, "W": self.P // 2}[kind]

    def run(self, regs, c, h, k, items, lib, p=0):
        st = [list(regs), c, h, k, p]
        self._items(st, items, lib)
        return st

    def _items(self, st, items, lib):
        for it in items:
            if is_loop(it):
                _, kind, body = it
                if kind in lib:
                    self._items(st, fill(lib[kind], body[0]), lib)
                else:
                    for _ in range(self.count(kind)):
                        self._items(st, body, lib)
            elif it in lib:
                if it.startswith("T."):
                    p = st[4]
                    self._items(st, lib[it], lib)
                    st[4] = p
                else:
                    self._items(st, lib[it], lib)
            else:
                self._step(st, it)

    @staticmethod
    def _joint(a, flags):
        j = a
        for f in flags:
            j = (j[:, :, None] * f[:, None, :]).reshape(a.shape[0], -1)
        return j

    def _step(self, st, tok):
        regs, c, h, k, p = st
        B = self.B
        op, _, arg = tok.partition(".")
        regs = list(regs)
        zero = torch.zeros_like(regs[0][:, :1])
        zero[..., 0] = 1
        put = lambda R, d: torch.cat([R[:, :p], d[:, None], R[:, p + 1:]], 1)
        if op in ("SBB", "ADC"):
            out = self.damp(self._joint(regs[0][:, p], (regs[RI[arg]][:, p], c)) @ (self.SBB if op == "SBB" else self.ADC))
            regs[0] = put(regs[0], out[:, :B])
            st[0], st[1] = regs, out[:, B:]
        elif op == "MUL":
            out = self.damp(self._joint(regs[0][:, p], (regs[RI[arg]][:, p], h, k)) @ self.MUL)
            regs[0] = put(regs[0], out[:, :B])
            st[0], st[3] = regs, out[:, B:]
        elif op == "SEL":
            sel = c[:, 1][:, None, None]
            regs[0] = self.damp(sel * regs[4] + (1 - sel) * regs[0])
            st[0] = regs
        elif op == "LDA":
            regs[0] = regs[RI[arg]]
            st[0] = regs
        elif op == "STA":
            regs[RI[arg]] = regs[0]
            st[0] = regs
        elif op == "CLA":
            regs[0] = zero.expand_as(regs[0])
            st[0] = regs
        elif op == "CLC":
            st[1] = torch.zeros_like(c)
            st[1][:, 0] = 1
        elif op == "CLK":
            st[3] = torch.zeros_like(k)
            st[3][:, 0] = 1
        elif op == "LDH":
            st[2] = regs[RI[arg]][:, p]
        elif op == "INC":
            st[4] = (p + 1) % self.P
        elif op == "SHL":
            regs[RI[arg]] = torch.cat([zero, regs[RI[arg]][:, :-1]], 1)
            st[0] = regs
        elif op == "SHR":
            regs[RI[arg]] = torch.cat([regs[RI[arg]][:, 1:], zero], 1)
            st[0] = regs
        elif op == "BRG":
            top = regs[1][:, -1:]
            regs[0] = torch.cat([top, regs[0][:, :-1]], 1)
            regs[1] = torch.cat([zero, regs[1][:, :-1]], 1)
            st[0] = regs
        else:
            raise ValueError(tok)
