"""LIBRARY CORE (INTEGRATION.md, stages 2-3), learned-loop copy (integration/learned_loop/LEARNED_LOOP.md):
integration/lib/libcore.py (s3i's) with one instruction kind added, the count loop, and the parts the self-practising
file never runs left out (s3i's practice tasks and its climb, libcore.Climb: spcore replaces both). It holds:
  * programs: library/dsl.py's items, costs, expansion, text form;
  * BPE: library/merge.py's tree-BPE (pair counts, merges, routines' rewriting);
  * the machine: tdsl.py (library/dsl.py's machine on torch tensors: 5 registers, a shared pointer, a borrow/carry, a
    multiplier digit and a multiply carry; 32 primitive instructions; loops by P, B, W; routines that restore the
    pointer), with the cells as tensors that the learned tables replace; and the count loop C.r[body]: the body run
    as a routine as many times as register r holds, counted down by the subtraction cell (the only loop whose count
    is data, so the only one each probe runs its own number of times);
  * the search: tfast.py (breadth-first by cost, states merged over every probe, every minimal parent kept, many items
    applied at once);
  * the round: found by composing the routines at the task's full size, its goal the answers (round_search.py); and
    the answer's program, found by composing the round with whole-register moves and count loops (outer_*);
  * the one-hot machine of the prediction path (dslste.py).
Rows are digit tensors (LSB first), as the model's reader produces them; no number is computed from a value.
"""

from __future__ import annotations

import time
from array import array
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
COUNT_KINDS = tuple(f"C.{r}" for r in REGS)          # count loops: the body runs as many times as register r holds
LOOP_CAP = 100                                       # a count loop runs its body at most this many times; a larger
                                                     #   count is an error that clears the probe's machine


def is_loop(it):
    return isinstance(it, tuple)


def is_count(it):
    return isinstance(it, tuple) and it[1] in COUNT_KINDS


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
        self.alu = torch.cat([self.sbb, self.adc, self.mul]).to(torch.uint8)    # the lockstep machine's one lookup


class Setting:
    """A machine size: base B, operands of W digits, registers of P places (2 W unless given)."""

    def __init__(self, B, W, cells=None, P=0):
        self.B, self.W, self.cells = B, W, cells
        self.P = P or 2 * W

    def count(self, kind):
        assert kind not in COUNT_KINDS, "a count loop's count is data, not a size"
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


def count_down(cells, C):
    """How many times a count loop runs on each count C [..., P] (digits, LSB first): C counted down by one with the
    subtraction cell (the digit 1 at place 0, the borrows rippling up) until a count down borrows, at most LOOP_CAP
    times. Returns the counts [...] and ok [...]: the count is at most LOOP_CAP (else the loop's error)."""
    D, B = C.long(), cells.B
    cnt = torch.zeros(D.shape[:-1], dtype=torch.long, device=D.device)
    done = torch.zeros_like(cnt, dtype=torch.bool)
    for _ in range(LOOP_CAP + 1):
        c, out = torch.zeros_like(cnt), []
        for j in range(D.shape[-1]):
            o = cells.sbb[(D[..., j] * B + int(j == 0)) * 2 + c]
            out.append(o[..., 0])
            c = o[..., 1]
        go = ~done & (c == 0)
        D = torch.where(go[..., None], torch.stack(out, -1), D)
        cnt, done = cnt + go.long(), done | (c == 1)
        if bool(done.all()):
            break
    return cnt, done


def _merge(b, nb, live):
    """b's probes where live [S, NP] take nb's registers and flags (in place)."""
    b.regs = torch.where(live[:, None, None, :], nb.regs, b.regs)
    b.c, b.h, b.k = (torch.where(live, getattr(nb, f), getattr(b, f)) for f in ("c", "h", "k"))


def _clear(b, bad):
    """The count loop's error: the probes where bad [S, NP] get every register and flag 0 (in place)."""
    b.regs = b.regs.masked_fill(bad[:, None, None, :], 0)
    b.c, b.h, b.k = (getattr(b, f).masked_fill(bad, 0) for f in ("c", "h", "k"))


def _run_count(st, b, it, lib):
    """A count loop on batch b in place: each probe runs the body as many times as its count register held when the
    loop began (count_down), the body as a routine (it starts at the loop's pointer, which is restored after it)."""
    _, kind, body = it
    cnt, ok = count_down(st.cells, b.regs[:, RI[kind[2:]]].transpose(1, 2))
    p = b.p.clone()
    for i in range(int(torch.where(ok, cnt, 0).max()) if cnt.numel() else 0):
        nb = b.copy()
        run_items(st, nb, body, lib)
        _merge(b, nb, ok & (cnt > i))
        b.p = p.clone()
    _clear(b, ~ok)


def run_items(st, b, items, lib):
    for it in items:
        if is_count(it):
            _run_count(st, b, it, lib)
        elif is_loop(it):
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


def _apply_grouped(st, b, item_of_row, items, lib):
    """apply_items item by item: tokens run once per distinct token, loops run their iterations together with the
    body's elements grouped position by position."""
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


# the lockstep machine: every row runs its own primitive instructions, one per step, all rows in the same kernels
OPCODE = {t: i for i, t in enumerate(PRIMS)}
NOP = len(PRIMS)                 # no instruction (a sequence shorter than the others it runs with)
SAVE = NOP + 1                   # a routine's entry at depth d (SAVE + 2 d) keeps the pointer, its exit (SAVE + 2 d + 1)
MAX_DEPTH = 32                   #   restores it
_TRACES = {}                     # (P, B, W, id(lib)) -> (lib, {item: (instructions, depth)})
_OPS = {}                        # (P, device) -> _Ops
# apply_items' time model (microseconds; measured on an H100, s3h): a lockstep step, plus per row; on the item-by-item
# path an instruction, a group's rows (taken and put back once per group: an item, or each element of a loop body at
# each iteration), and a group's split and merge
LOCK_STEP, LOCK_ROW, PRIM_STEP, PRIM_ROW, GROUP_FIX = 200.0, 0.0025, 25.0, 0.004, 60.0
LOCK_COST = 0.0                  # (tests: a large negative value forces the lockstep on every bucket)
LOCK_ROWS = 1 << 22              # a lockstep runs at most this many rows at once (its step's temporaries: ~1 KB a row)


def trace(st, items, lib, out, depth=0):
    """The primitive instructions run_items executes for items at setting st, in order, as opcodes (a routine between
    its pointer's save and restore); returns the deepest routine nesting."""
    deep = depth
    for it in items:
        if is_loop(it):
            _, kind, body = it
            if kind in lib:
                deep = max(deep, trace(st, fill(lib[kind], body[0]), lib, out, depth))
            else:
                for _ in range(st.count(kind)):
                    deep = max(deep, trace(st, body, lib, out, depth))
        elif it in lib:
            if it.startswith("T."):
                assert depth < MAX_DEPTH, it
                out.append(SAVE + 2 * depth)
                deep = max(deep, trace(st, lib[it], lib, out, depth + 1))
                out.append(SAVE + 2 * depth + 1)
            else:
                deep = max(deep, trace(st, lib[it], lib, out, depth))
        else:
            out.append(OPCODE[it])
    return deep


TRACE_KEEP = 50_000_000         # instructions the trace memo holds before it starts over (int16: 100 MB)


def _traces(st, lib):
    """The memo of traces at st's sizes under lib, kept while lib is (a lib only ever gains names; a name's definition
    never changes): {(item, depth): (instructions as array('h'), deepest nesting)}."""
    key = (st.P, st.B, st.W, id(lib) if lib else 0)
    ent = _TRACES.get(key)
    if ent is None or (lib and ent[0] is not lib) or ent[2][0] > TRACE_KEEP:
        while len(_TRACES) >= 6:
            _TRACES.pop(next(iter(_TRACES)))
        ent = _TRACES[key] = (lib, {}, [0])
    return ent


def _trace_of(st, it, lib, depth, ent):
    """trace() of one item at a routine depth, memoized: array('h') of opcodes and the deepest nesting."""
    memo = ent[1]
    got = memo.get((it, depth))
    if got is not None:
        return got
    if is_loop(it):
        _, kind, body = it
        if kind in lib:
            got = _trace_seq(st, fill(lib[kind], body[0]), lib, depth, ent)
        else:
            ops, deep = _trace_seq(st, body, lib, depth, ent)
            got = (ops * st.count(kind), deep)
    elif it in lib:
        if it.startswith("T."):
            assert depth < MAX_DEPTH, it
            ops, deep = _trace_seq(st, lib[it], lib, depth + 1, ent)
            got = (array("h", [SAVE + 2 * depth]) + ops + array("h", [SAVE + 2 * depth + 1]), deep)
        else:
            got = _trace_seq(st, lib[it], lib, depth, ent)
    else:
        got = (array("h", [OPCODE[it]]), depth)
    memo[(it, depth)] = got
    ent[2][0] += len(got[0])
    return got


def _trace_seq(st, items, lib, depth, ent):
    ops, deep = array("h"), depth
    for it in items:
        o, d = _trace_of(st, it, lib, depth, ent)
        ops += o
        deep = max(deep, d)
    return ops, deep


class _Ops:
    """Every opcode's effect as table rows, for one register size P: where each register place comes from (rm; 5 P is
    a zero), the cell family and its register (SBB 1, ADC 2, MUL 3), and the other instructions' flags."""

    def __init__(self, P, device):
        n, Z = SAVE + 2 * MAX_DEPTH, 5 * P
        ar = torch.arange(P)
        rm = torch.arange(5 * P).repeat(n, 1)
        fam, ralu, rldh, dep = (torch.zeros(n, dtype=torch.long) for _ in range(4))
        flag = {f: torch.zeros(n, dtype=torch.bool) for f in ("sel", "clc", "clk", "ldh", "inc", "save", "rest")}
        for tok, o in OPCODE.items():
            op, _, arg = tok.partition(".")
            r = RI.get(arg, 0)
            if op in ("SBB", "ADC", "MUL"):
                fam[o], ralu[o] = {"SBB": 1, "ADC": 2, "MUL": 3}[op], r
            elif op == "LDA":
                rm[o, :P] = r * P + ar
            elif op == "STA":
                rm[o, r * P:(r + 1) * P] = ar
            elif op == "CLA":
                rm[o, :P] = Z
            elif op == "LDH":
                flag["ldh"][o], rldh[o] = True, r
            elif op == "SHL":
                rm[o, r * P] = Z
                rm[o, r * P + 1:(r + 1) * P] = r * P + ar[:-1]
            elif op == "SHR":
                rm[o, (r + 1) * P - 1] = Z
                rm[o, r * P:(r + 1) * P - 1] = r * P + ar[1:]
            elif op == "BRG":
                rm[o, 0] = X_ * P + P - 1
                rm[o, 1:P] = ar[:-1]
                rm[o, X_ * P] = Z
                rm[o, X_ * P + 1:(X_ + 1) * P] = X_ * P + ar[:-1]
            else:
                flag[{"SEL": "sel", "CLC": "clc", "CLK": "clk", "INC": "inc"}[op]][o] = True
        for d in range(MAX_DEPTH):
            flag["save"][SAVE + 2 * d], flag["rest"][SAVE + 2 * d + 1] = True, True
            dep[SAVE + 2 * d] = dep[SAVE + 2 * d + 1] = d
        self.P, self.Z = P, Z
        self.rm = rm.to(device)
        self.ri = torch.stack([ralu * P, rldh * P], 1).to(device)       # the register a cell or LDH reads, as a place
        self.dep = dep.to(device)
        self.flags = torch.stack([flag[f] for f in ("sel", "clc", "clk", "ldh", "inc", "save", "rest")]
                                 + [fam > 0, (fam == 1) | (fam == 2), fam == 3], 1).to(device)
        # the cell table's row (Cells.alu: the subtraction's 2 B^2 rows, the addition's, then the multiply's B^4) as
        # off + a ca + x cx + c cc + h ch + k ck over (a, x, c, h, k)
        self.B = None

    def coefs(self, B):
        if self.B != B:
            n = self.flags.shape[0]
            co = torch.zeros(n, 6, dtype=torch.int32)
            fam = torch.zeros(n, dtype=torch.long)
            for tok, o in OPCODE.items():
                op = tok.partition(".")[0]
                if op in ("SBB", "ADC"):
                    co[o] = torch.tensor([2 * B, 2, 1, 0, 0, 0 if op == "SBB" else 2 * B * B], dtype=torch.int32)
                elif op == "MUL":
                    co[o] = torch.tensor([B ** 3, B * B, 0, B, 1, 4 * B * B], dtype=torch.int32)
            self.co, self.B = co.to(self.rm.device), B
        return self.co


def _ops(P, device):
    key = (P, str(device))
    if key not in _OPS:
        _OPS[key] = _Ops(P, device)
    return _OPS[key]


def _ustep(o, alu, co, regs, c, h, k, p, ps, op, ar, ad):
    """One instruction on every machine, each its own (op [S]), exactly as step() runs it: regs [S, 5 P, NP] uint8
    (flat, contiguous), c, h, k [S, NP] uint8, p [S], ps [S, D] the routines' saved pointers (ar, ad: aranges of S and
    D). Returns the new state. Register moves go by words of the probes' bytes (8, 4 or 2 at a time, as NP allows);
    the three cells by one table (Cells.alu) at one row index per probe."""
    S, P = op.shape[0], o.P
    NP = regs.shape[2]
    fl = o.flags[op]
    m = o.rm[op]
    wd = torch.int64 if NP % 8 == 0 else torch.int32 if NP % 4 == 0 else torch.int16 if NP % 2 == 0 else torch.uint8
    w = regs.view(wd)
    g = w.gather(1, m.clamp(max=o.Z - 1)[:, :, None].expand(S, o.Z, w.shape[2]))
    g = g.masked_fill_((m == o.Z)[:, :, None], 0).view(torch.uint8)
    sel = fl[:, 0:1] & (c > 0)
    g[:, :P] = torch.where(sel[:, None, :], g[:, 4 * P:], g[:, :P])
    ri = o.ri[op]
    a = g[ar, p]
    v = torch.stack([a, g[ar, ri[:, 0] + p], c, h, k], -1).int()
    cf = co[op]
    out = alu[((v * cf[:, None, :5]).sum(-1) + cf[:, 5:6]).long()]
    g[ar, p] = torch.where(fl[:, 7:8], out[..., 0], a)
    c = torch.where(fl[:, 8:9], out[..., 1], c).masked_fill_(fl[:, 1:2], 0)
    k = torch.where(fl[:, 9:10], out[..., 1], k).masked_fill_(fl[:, 2:3], 0)
    h = torch.where(fl[:, 3:4], g[ar, ri[:, 1] + p], h)
    if ps.shape[1]:
        d = o.dep[op].clamp(max=ps.shape[1] - 1)[:, None]
        ps = torch.where(fl[:, 5:6] & (ad == d), p[:, None], ps)
        p = torch.where(fl[:, 6], ps.gather(1, d)[:, 0], p)
    p = torch.where(fl[:, 4], (p + 1) % P, p)
    return g, c, h, k, p, ps


def _columns(arrs, n, dev, block=512):
    """The instructions of the sequences arrs (array('h') each, at most n long), NOP-padded, as columns on the device:
    [n, len(arrs)] (built block by block of steps, so that no index tensor is larger than block x len(arrs))."""
    flat = array("h")
    offs = []
    for a in arrs:
        offs.append(len(flat))
        flat += a
    flat.append(NOP)
    f = torch.frombuffer(flat, dtype=torch.int16).to(dev)
    off = torch.tensor(offs, dtype=torch.long, device=dev)
    ln = torch.tensor([len(a) for a in arrs], dtype=torch.long, device=dev)
    out = torch.empty(n, len(arrs), dtype=torch.long, device=dev)
    for t0 in range(0, n, block):
        t = torch.arange(t0, min(n, t0 + block), device=dev)[:, None]
        out[t0:t0 + block] = f[torch.where(t < ln[None], off[None] + t, len(flat) - 1)]
    return out


GRAPH_STEPS = 32                 # on a GPU, a long lockstep replays its steps from a captured CUDA graph of this many
GRAPH_MIN = 192                  #   steps (0: never), from this many steps on (the same kernels, one launch per graph)


def _lockstep(st, b, cols, lr, depth):
    """b's rows (a Batch, in place) run their sequences, one step per column of cols [n, items] (opcodes, NOP-padded;
    lr [S]: each row's column), all rows at once."""
    S, _, P, NP = b.regs.shape
    dev = b.regs.device
    o = _ops(P, dev)
    co = o.coefs(st.B)
    state = [b.regs.reshape(S, 5 * P, NP).contiguous(), b.c, b.h, b.k, b.p,
             torch.zeros(S, depth, dtype=torch.long, device=dev)]
    ar, ad = torch.arange(S, device=dev), torch.arange(depth, device=dev)[None]
    n = cols.shape[0]
    t0 = 0
    if dev.type == "cuda" and GRAPH_STEPS and n >= GRAPH_MIN:
        state, t0 = _graph_steps(o, st.cells.alu, co, state, cols, lr, ar, ad)
    for t in range(t0, n):
        state = list(_ustep(o, st.cells.alu, co, *state, cols[t][lr], ar, ad))
    regs, c, h, k, p, _ = state
    b.regs, b.c, b.h, b.k, b.p = regs.view(S, 5, P, NP), c, h, k, p


def _graph_steps(o, alu, co, state, cols, lr, ar, ad):
    """As many whole blocks of GRAPH_STEPS steps as cols has, by one CUDA graph: the first block runs eagerly on static
    buffers (it also warms up), the second is captured (capture records, it does not run), then the graph is replayed
    for the rest. Returns the state and the steps done."""
    with torch.no_grad():
        return _graph_steps_(o, alu, co, state, cols, lr, ar, ad)


def _graph_steps_(o, alu, co, state, cols, lr, ar, ad):
    K = GRAPH_STEPS
    nb = cols.shape[0] // K
    buf = [x.clone() for x in state]
    t = torch.zeros(1, dtype=torch.long, device=cols.device)

    def block():
        cur = buf
        for _ in range(K):
            cur = _ustep(o, alu, co, *cur, cols.index_select(0, t)[0][lr], ar, ad)
            t.add_(1)
        for x, y in zip(buf, cur):
            x.copy_(y)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        block()
    torch.cuda.current_stream().wait_stream(side)
    if nb > 1:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            block()
        for _ in range(nb - 1):
            g.replay()
        del g
    return buf, nb * K


def apply_items(st, b, item_of_row, items, lib):
    """A new batch: row r of b with items[item_of_row[r]] applied. Each distinct item is traced into the instructions
    run_items would execute (trace); the rows then run their own instructions in lockstep, one per step, every row in
    the same kernels (_ustep), grouped by the length of their sequences (powers of 2) so that short ones do not wait
    for long ones. When few distinct items share the steps, item by item (_apply_grouped) launches fewer kernels and
    is used instead. Both give exactly run_items' result. Rows whose item is a count loop go to _apply_counts."""
    if any(is_count(it) for it in items):
        return _apply_counts(st, b, item_of_row, items, lib)
    ent = _traces(st, lib)
    seqs = [_trace_of(st, it, lib, 0, ent) for it in items]
    lens = [len(s[0]) for s in seqs]
    bk = [n.bit_length() if n else 0 for n in lens]                   # 0; 1; 2-3; 4-7; ...: one lockstep per bucket
    nb = max(bk, default=0) + 1
    mx, per = [0] * nb, [[] for _ in range(nb)]
    loc = [0] * len(items)
    for i, (j, n) in enumerate(zip(bk, lens)):
        mx[j] = max(mx[j], n)
        loc[i] = len(per[j])
        per[j].append(i)
    S, k = b.S, max(1, len(items))
    rows_per = S / k
    groups = [st.count(it[1]) * len(it[2]) if is_loop(it) and it[1] not in lib else 1 for it in items]
    lock = {j for j in range(1, nb) if per[j] and LOCK_COST + mx[j] * (LOCK_STEP + len(per[j]) * rows_per * LOCK_ROW)
            < sum(lens[i] * PRIM_STEP + groups[i] * rows_per * PRIM_ROW + GROUP_FIX for i in per[j])}
    if b.S == 0 or not lock:
        return _apply_grouped(st, b, item_of_row, items, lib)
    out = b.copy()
    dev = b.regs.device
    loc_row = torch.tensor(loc, dtype=torch.long, device=dev)[item_of_row]
    rest = [j for j in range(1, nb) if per[j] and j not in lock]
    if len(lock) == 1 and not rest and not per[0]:
        parts = [(next(iter(lock)), None)]
    else:
        bucket = torch.tensor(bk, dtype=torch.long, device=dev)[item_of_row]
        order = torch.sort(bucket, stable=True).indices
        counts = torch.bincount(bucket, minlength=nb).tolist()
        starts = [sum(counts[:j]) for j in range(nb)]
        parts = [(j, order[starts[j]:starts[j] + counts[j]]) for j in range(1, nb) if per[j] and counts[j]]
    for j, idx in parts:
        sub = out if idx is None else out.take(idx)
        lr = loc_row if idx is None else loc_row[idx]
        if j in lock:
            cols = _columns([seqs[i][0] for i in per[j]], mx[j], dev)
            depth = max(seqs[i][1] for i in per[j])
            if sub.S <= LOCK_ROWS:
                _lockstep(st, sub, cols, lr, depth)
            else:                                    # in slices of rows: a step's temporaries stay bounded
                for r0 in range(0, sub.S, LOCK_ROWS):
                    part = sub.take(slice(r0, r0 + LOCK_ROWS)).copy()
                    _lockstep(st, part, cols, lr[r0:r0 + LOCK_ROWS], depth)
                    _put(sub, slice(r0, r0 + LOCK_ROWS), part)
        else:
            sub = _apply_grouped(st, sub, lr, [items[i] for i in per[j]], lib)
        if idx is not None:
            _put(out, idx, sub)
    return out


def _apply_counts(st, b, item_of_row, items, lib):
    """apply_items with count loops among the items: the other rows as apply_items runs them; the count loops' rows run
    their bodies together, one apply_items call per iteration (each body one item, as a routine), every probe keeping
    the result while its own count lasts; a probe whose count is above LOOP_CAP ends cleared. run_items' result."""
    dev = b.regs.device
    flag = torch.tensor([is_count(it) for it in items], dtype=torch.bool, device=dev)[item_of_row]
    out = b.copy()
    rest = torch.nonzero(~flag).flatten()
    if rest.numel():
        _put(out, rest, apply_items(st, b.take(rest), item_of_row[rest],
                                    [("REP", ONE, ()) if is_count(it) else it for it in items], lib))
    rows = torch.nonzero(flag).flatten()
    if rows.numel() == 0:
        return out
    sub, ior = b.take(rows), item_of_row[rows]
    reg = torch.tensor([RI[it[1][2:]] if is_count(it) else 0 for it in items], dtype=torch.long, device=dev)[ior]
    cnt, ok = count_down(st.cells, sub.regs[torch.arange(rows.numel(), device=dev), reg].transpose(1, 2))
    bodies = [("REP", ONE, it[2] if is_count(it) else ()) for it in items]
    for i in range(int(torch.where(ok, cnt, 0).max())):
        live = ok & (cnt > i)
        act = torch.nonzero(live.any(1)).flatten()
        part = sub.take(act)
        new = apply_items(st, part, ior[act], bodies, lib)
        _merge(part, new, live[act])
        _put(sub, act, part)
    _clear(sub, ~ok)
    _put(out, rows, sub)
    return out


# the round machine (s3i): one program on many rows from fresh machines, as run_prog runs it. Control never depends on
# data, so the pointer is resolved once on the host (INC and the routines' saves and restores cost nothing) and every
# other instruction becomes one or two fixed slice operations on one state tensor; on a GPU the whole program is one
# CUDA graph, captured once and replayed
ROUND_ROWS = 128                 # a round machine's rows: a multiple of this (fewer shapes, so fewer graphs)
ROUND_KEEP = 8                   # round machines kept (each with its graph)
ROUND_GRAPH = True               # on a GPU, replay the program from a CUDA graph (False: eagerly, as on a CPU)
_ROUND = {}                      # (instructions, P, B, rows, device) -> RoundMachine


class RoundMachine:
    """One program's instructions (a trace) on `rows` machines at once: state St [5 P + 4, rows] int64, the registers'
    places (A, X, Y, N, S), then c, h, k and a row that stays 0; the cells as tables [2 (digit, flag), inputs...]."""

    def __init__(self, st, ops, rows, device, share=None, p0=0):
        P, B = st.P, st.B
        self.st, self.P, self.dev = Setting(B, P // 2, None, P), P, torch.device(device)
        if share is None:
            St = self.St = torch.zeros(5 * P + 4, rows, dtype=torch.long, device=device)
            self.sbb = torch.zeros(2, B, B, 2, dtype=torch.long, device=device)
            self.adc = torch.zeros_like(self.sbb)
            self.mul = torch.zeros(2, B, B, B, B, dtype=torch.long, device=device)
        else:                             # a stretch of a longer program (OuterMachine): its state and its tables
            St = self.St = share.St
            self.sbb, self.adc, self.mul = share.sbb, share.adc, share.mul
        c, h, k, Z = St[5 * P], St[5 * P + 1], St[5 * P + 2], 5 * P + 3
        A, S = St[:P], St[S_ * P:(S_ + 1) * P]
        idx = {}

        def rows_of(*ix):
            if ix not in idx:
                idx[ix] = torch.tensor(ix, dtype=torch.long, device=device)
            return idx[ix]
        fns, p, saved = [], p0, [0] * MAX_DEPTH
        for o in ops:
            if o >= SAVE:                                                   # a routine's entry or exit
                d = (o - SAVE) // 2
                if (o - SAVE) % 2 == 0:
                    saved[d] = p
                else:
                    p = saved[d]
                continue
            op, _, arg = PRIMS[o].partition(".")
            r = RI.get(arg, 0)
            R = St[r * P:(r + 1) * P]
            if op == "INC":
                p = (p + 1) % P
            elif op in ("SBB", "ADC"):                                      # A[p], c = the cell at (A[p], R[p], c)
                fns.append(lambda tab=(self.sbb if op == "SBB" else self.adc), a=A[p], x=R[p], to=rows_of(p, 5 * P):
                           St.index_copy_(0, to, tab[:, a, x, c]))
            elif op == "MUL":                                               # A[p], k = the cell at (A[p], R[p], h, k)
                fns.append(lambda a=A[p], x=R[p], to=rows_of(p, 5 * P + 2):
                           St.index_copy_(0, to, self.mul[:, a, x, h, k]))
            elif op == "SEL":
                fns.append(lambda: torch.where(c != 0, S, A, out=A))
            elif op == "LDA":
                fns.append(lambda R=R: A.copy_(R))
            elif op == "STA":
                fns.append(lambda R=R: R.copy_(A))
            elif op in ("CLA", "CLC", "CLK"):
                fns.append({"CLA": A, "CLC": c, "CLK": k}[op].zero_)
            elif op == "LDH":
                fns.append(lambda x=R[p]: h.copy_(x))
            else:                                    # SHL, SHR, BRG: places moved within registers (Z: a 0 place)
                q = list(range(r * P, (r + 1) * P))
                src = {"SHL": [Z] + q[:-1], "SHR": q[1:] + [Z],
                       "BRG": [X_ * P + P - 1] + list(range(P - 1)) + [Z] + list(range(X_ * P, X_ * P + P - 1))}[op]
                lo = 0 if op == "BRG" else r * P
                fns.append(lambda dst=St[lo:lo + len(src)], src=rows_of(*src): dst.copy_(St.index_select(0, src)))
        self.fns, self.graph, self.p_end = fns, None, p

    def cells(self, cells):
        """The cells the program reads (copied into the machine's tables, which a graph reads in place)."""
        B = self.st.B
        self.sbb.copy_(cells.sbb.T.reshape(2, B, B, 2))
        self.adc.copy_(cells.adc.T.reshape(2, B, B, 2))
        self.mul.copy_(cells.mul.T.reshape(2, B, B, B, B))

    def run(self, Xd, Nd, go, n, gen):
        """One run from fresh machines: x digits Xd in X and Nd in N (rows m <= rows [m, <= P]); everything else fresh's
        random draws for the n rows of go [m] alone, in their order, exactly as run_prog draws them for those rows.
        Every row runs; returns A's digits [m, P] (a view, until the next run; rows outside go: anything)."""
        P, St, m = self.P, self.St, Xd.shape[0]
        b = fresh(self.st, {}, gen, n, self.dev)
        at = (go.long().cumsum(0) - 1).clamp(min=0)                          # each row's place among go's rows
        St[:5 * P, :m] = b.regs[0].reshape(5 * P, n)[:, at]
        St[X_ * P:(X_ + 1) * P, :m] = pad(Xd, P).T
        St[N_ * P:(N_ + 1) * P, :m] = pad(Nd, P).T
        St[5 * P:5 * P + 3, :m] = torch.cat([b.c, b.h, b.k])[:, at]
        with torch.no_grad():
            self._program()
        return St[:P, :m].T

    def _program(self):
        if self.dev.type != "cuda" or not ROUND_GRAPH:
            for f in self.fns:
                f()
        elif self.graph is None:                     # the first run eagerly on a side stream (it also warms up), then
            side = torch.cuda.Stream()               # the program captured (capture records, it does not run)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for f in self.fns:
                    f()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                for f in self.fns:
                    f()
        else:
            self.graph.replay()


def round_machine(st, prog, lib, rows, device):
    """prog's RoundMachine at setting st (its cells loaded) for rows machines (rounded up to a multiple of ROUND_ROWS),
    made once per instruction sequence and size and kept (ROUND_KEEP at most)."""
    ops, _ = _trace_seq(st, prog, lib, 0, _traces(st, lib))
    R = -(-max(rows, 1) // ROUND_ROWS) * ROUND_ROWS
    key = (ops.tobytes(), st.P, st.B, R, str(device))
    m = _ROUND.get(key)
    if m is None:
        while len(_ROUND) >= ROUND_KEEP:
            _ROUND.pop(next(iter(_ROUND)))
        m = _ROUND[key] = RoundMachine(st, ops, R, device)
    m.cells(st.cells)
    return m


class OuterMachine:
    """An answer's program (count loops at its top level, plain items between them) on `rows` machines at once: each
    stretch of plain items and each count loop's body is a RoundMachine on one shared state (the body's from the loop's
    pointer); a count loop replays its body's machine while any row's count lasts, each row keeping the result while
    its own does (count_down on the count register's digits, as the loop began). Returns run_items' result."""

    def __init__(self, st, prog, lib, rows, device):
        ent, self.parts, self.base, p, run = _traces(st, lib), [], None, 0, []
        for it in tuple(prog) + (None,):
            if it is not None and not is_count(it):
                run.append(it)
                continue
            if run or self.base is None:
                m = RoundMachine(st, _trace_seq(st, tuple(run), lib, 0, ent)[0], rows, device, self.base, p)
                self.base, p = self.base or m, m.p_end
                if run:
                    self.parts.append((None, m))
                run = []
            if it is not None:
                body = RoundMachine(st, _trace_seq(st, it[2], lib, 0, ent)[0], rows, device, self.base, p)
                self.parts.append((RI[it[1][2:]], body))
        self.P, self.st, self.dev = st.P, self.base.st, self.base.dev

    def cells(self, cells):
        self.base.cells(cells)
        self.cl = cells

    def run(self, Xd, Nd, Cd, gen):
        """From fresh machines (fresh's draws for all m rows, in order): x digits Xd in X, Nd in N and the count's Cd
        in S ([m, <= P] each). Returns A's digits [m, P] (a view, until the next run)."""
        P, St, m = self.P, self.base.St, Xd.shape[0]
        b = fresh(self.st, {}, gen, m, self.dev)
        St.zero_()
        St[:5 * P, :m] = b.regs[0].reshape(5 * P, m)
        for r, d in ((X_, Xd), (N_, Nd), (S_, Cd)):
            St[r * P:(r + 1) * P, :m] = pad(d, P).T
        St[5 * P:5 * P + 3, :m] = torch.cat([b.c, b.h, b.k])
        with torch.no_grad():
            for r, mach in self.parts:
                if r is None:
                    mach._program()
                    continue
                cnt, ok = count_down(self.cl, St[r * P:(r + 1) * P].T)
                for i in range(int(torch.where(ok, cnt, 0).max())):
                    old = St.clone()
                    mach._program()
                    St.copy_(torch.where((ok & (cnt > i))[None], St, old))
                St[:5 * P + 3] *= ok[None]
        return St[:P, :m].T


_OUTER = {}                      # (instructions, P, B, rows, device) -> OuterMachine


def outer_machine(st, prog, lib, rows, device):
    """prog's OuterMachine (rows rounded up as round_machine's), its cells loaded; made once per program and size."""
    ent = _traces(st, lib)
    key = tuple((it[1], _trace_seq(st, it[2], lib, 0, ent)[0].tobytes()) if is_count(it)
                else _trace_of(st, it, lib, 0, ent)[0].tobytes() for it in prog)
    R = -(-max(rows, 1) // ROUND_ROWS) * ROUND_ROWS
    key = (key, st.P, st.B, R, str(device))
    m = _OUTER.get(key)
    if m is None:
        while len(_OUTER) >= ROUND_KEEP:
            _OUTER.pop(next(iter(_OUTER)))
        m = _OUTER[key] = OuterMachine(st, prog, lib, R, device)
    m.cells(st.cells)
    return m


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
                    deadline=None, cache=None, chunk=1 << 21):
    """Every distinct item (tokens, then loops over bodies of cost up to max_body built from the distinct items so
    far), the cheapest spelling first: items that agree on S_test random machines of every setting are one item.
    Raises Deadline if the deadline passes. cache: {item: signature} kept across calls with the same settings (the
    test machines are the same every call and an item's meaning never changes, so only new items are run)."""
    gen = gen_on(device, seed)
    tests = [random_states(st, S_test, NP_test, gen, device) for st in settings]

    def sigs(items):
        if cache is None:
            return _signatures(settings, tests, items, lib, chunk=chunk, deadline=deadline)
        miss = list(dict.fromkeys(it for it in items if it not in cache))
        if miss:
            for it, sg in zip(miss, _signatures(settings, tests, miss, lib, chunk=chunk, deadline=deadline)):
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
        listing="spellings", max_goals=50_000, spell=8, batch=256, fuse=1):
    """Breadth-first by cost, states merged over every probe of every setting, every minimal-cost parent kept. Stops
    at the first cost whose goal programs pass `accept`, when a level would pass max_states (that level is finished
    streaming goal states only), or at the deadline. The goal programs tested: the first `limit` spellings
    (listing "spellings"), or one spelling per goal state (listing "goals"), tested in batches, fewest top-level items
    first (then fewest primitives), until a batch has one that passes; the accepted goal states then give up to `spell`
    spellings each that pass. A level's work is blocks of (items, source states) of at most `chunk` rows, taken in
    order; up to `fuse` chunks' worth of consecutive blocks run in one apply_items call (fewer lockstep steps per row;
    every block is then treated as alone, so the result is the same for any fuse)."""
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
        def blocks(d=d):
            for c, cis in by_cost.items():
                sl = d - c
                if sl < 0 or levels[sl] is None:
                    continue
                for s0 in range(0, levels[sl][0].S, chunk):
                    srcs = [b.take(slice(s0, s0 + chunk)) for b in levels[sl]]
                    i_step = max(1, chunk // srcs[0].S)
                    for i0 in range(0, cis.numel(), i_step):
                        yield cis[i0:i0 + i_step], sl, s0, srcs
        work = blocks()
        nxt = next(work, None)
        while nxt is not None:
            if _late(deadline):
                late = True
                break
            group, n_rows = [], 0
            while nxt is not None and (not group or n_rows + nxt[0].numel() * nxt[3][0].S <= fuse * chunk):
                group.append(nxt)
                n_rows += nxt[0].numel() * nxt[3][0].S
                nxt = next(work, None)
            parts, its, srcrows = [], [], [[] for _ in settings]
            for ii, sl, s0, srcs in group:
                ns, k = srcs[0].S, ii.numel()
                its.append(torch.arange(k, device=dev).repeat_interleave(ns) + len(parts))
                parts.extend(classes[i].rep for i in ii.tolist())
                rs = torch.arange(ns, device=dev).repeat(k)
                for si, b in enumerate(srcs):
                    srcrows[si].append(b.take(rs))
            ri_all = its[0] if len(its) == 1 else torch.cat(its)
            outs_all = [apply_items(st, r[0] if len(r) == 1 else Batch.concat(r), ri_all, parts, lib)
                        for st, r in zip(settings, srcrows)]
            del srcrows
            off = 0
            for ii, sl, s0, srcs in group:
                ns, k = srcs[0].S, ii.numel()
                outs = outs_all if len(group) == 1 else [o.take(slice(off, off + k * ns)) for o in outs_all]
                off += k * ns
                ri = torch.arange(k, device=dev).repeat_interleave(ns)
                rs = torch.arange(ns, device=dev).repeat(k)
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
            del outs_all
        if late or _late(deadline):
            res.stop = f"deadline at level {d}"
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


# ===================================================================================================== the practice problems' helpers
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
    """The share of answer rows (Nd, Xd, Yd) the program reproduces at the task's size (a report: on a GPU the mean is
    a float32 sum times 1/n, below 1.0 for some n although every row is right; round_ok is the exact test)."""
    Nd, Xd, Yd = rows
    st = round_settings(cells, P)
    A, _ = run_prog(st, prog, lib, {"X": pad(Xd, P), "N": pad(Nd, P)}, gen_on(device, seed), device)
    return float((A[:, :Yd.shape[1]] == Yd).all(1).float().mean())


def round_ok(prog, lib, cells, rows, P, device, seed=1):
    """Does the program reproduce every answer row (Nd, Xd, Yd) at the task's size? (round_fit's machine and seed.)"""
    Nd, Xd, Yd = rows
    st = round_settings(cells, P)
    A, _ = run_prog(st, prog, lib, {"X": pad(Xd, P), "N": pad(Nd, P)}, gen_on(device, seed), device)
    return bool((A[:, :Yd.shape[1]] == Yd).all())


# ===================================================================================================== Occam's step count
def steps(items, lib, P=4, B=10, W=2):
    """How many primitive instructions a program executes at a setting (base B, W-digit operands, P places)."""
    n = 0
    for it in items:
        if is_count(it):                  # its count is data: the body once
            n += steps(it[2], lib, P, B, W)
        elif is_loop(it) and it[1] in lib:
            n += steps(fill(lib[it[1]], it[2][0]), lib, P, B, W)
        elif is_loop(it):
            n += {"P": P, "B": B, "W": W, ONE: 1}[it[1]] * steps(it[2], lib, P, B, W)
        elif it in lib:
            n += steps(lib[it], lib, P, B, W)
        else:
            n += 1
    return n


# ===================================================================================================== the one-hot machine
class SteMachine:
    """A found program on one-hot digits (the prediction path): every cell a straight-through table (rows (a, b, c) ->
    digit 10 + flag 2; (a, x, h, k) -> low 10 + high 10), the pointer a Python int, registers immutable tensors (so
    autograd keeps every earlier value); every cell's output and every count loop's blend passes `damp` (its gradient
    scaled)."""

    def __init__(self, SBB, ADC, MUL, P, damp, B=10, used=None):
        self.SBB, self.ADC, self.MUL, self.P, self.B, self.damp = SBB, ADC, MUL, P, B, damp
        self.used = used                 # the subtraction's rows (a, b, c) the count loops' count downs meet (a tally)

    def count(self, kind):
        return {"P": self.P, "B": self.B, "W": self.P // 2}[kind]

    def run(self, regs, c, h, k, items, lib, p=0):
        st = [list(regs), c, h, k, p]
        self._items(st, items, lib)
        return st

    def _count(self, st, it, lib):
        """A count loop: the count (register r as the loop begins) counted down by the subtraction cell, place by
        place; each round the rows whose count down did not borrow run the body (as a routine, from the loop's
        pointer) and take its state, the others keep theirs (a straight-through blend, as stage 1's countdown); rows
        whose count is above LOOP_CAP end cleared."""
        _, kind, body = it
        C, p, B = st[0][RI[kind[2:]]], st[4], self.B
        one = torch.zeros_like(C)
        one[..., 0] = 1
        one[:, 0] = 0
        one[:, 0, 1] = 1
        for i in range(LOOP_CAP + 1):
            c = torch.zeros_like(C[:, 0, :2])
            c[:, 0] = 1
            out = []
            for j in range(C.shape[1]):
                jt = self._joint(C[:, j], (one[:, j], c))
                if self.used is not None:
                    self.used += jt.detach().sum(0)
                o = self.damp(jt @ self.SBB)
                out.append(o[:, :B])
                c = o[:, B:]
            go = c[:, 0]                 # the count down did not borrow: the count lasts
            if not bool((go > 0.5).any()):
                break
            g3, g2 = go[:, None, None], go[:, None]
            if i == LOOP_CAP:            # the error: a count above LOOP_CAP clears the row's machine
                zero = torch.zeros_like(C)
                zero[..., 0] = 1
                st[0] = [g3 * zero + (1 - g3) * a for a in st[0]]
                st[1], st[2], st[3] = (g2 * zero[:, 0, :a.shape[1]] + (1 - g2) * a for a in st[1:4])
                break
            new = [list(st[0]), st[1], st[2], st[3], p]
            self._items(new, body, lib)
            st[0] = [self.damp(g3 * a + (1 - g3) * b) for a, b in zip(new[0], st[0])]
            st[1], st[2], st[3] = (self.damp(g2 * a + (1 - g2) * b) for a, b in zip(new[1:4], st[1:4]))
            C = self.damp(g3 * torch.stack(out, 1) + (1 - g3) * C)
        st[4] = p

    def _items(self, st, items, lib):
        for it in items:
            if is_count(it):
                self._count(st, it, lib)
            elif is_loop(it):
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
