"""The bare machine on the GPU (GPU_SEARCH.md §1): one Triton kernel runs every probe of every (class, state) pair of a
chunk as its own machine, a second hashes each output state with brm's hash, bit for bit.

The machine is bar3/scale/engine._run_block's, step for step: values saturate at cap; DEC stops at 0; every primitive
counts one step and so does every loop iteration as it starts; a machine dies when its step count passes the limit at
an iteration's start (and only there), and a state with a dead probe is dropped; L.r reads r once, on entry; Z.r runs
its body when r = 0; the step count starts at 0 for each applied item.

Bytecode. engine.compile_items gives each class a flat program (library tokens and abstractions inlined) of
(op, a, b) rows: INC/DEC/CLR a, MOV a b, LOOP r e (e: its END), END, ZERO r e (e: past the body). Inlined, every loop
has a fixed nesting depth, so a machine needs no stack: one remaining count per depth. `pack` writes each row as one
int32, op | a << 3 | b << 6 | depth << 9 | target << 13, where an END's target is its LOOP (the jump back goes to the
body's first row) and its depth the LOOP's, and appends a HALT (op 7) to every class.

Without a GPU (or Triton), `apply_chunk` runs the compiled CPU kernel (engine._apply_many) instead, so the search
around it can be tested anywhere.
"""

from __future__ import annotations

import numpy as np
import torch

from bar3.scale import engine as E

INC, DEC, CLR, MOV, LOOP, END, ZERO, HALT = range(8)
DMAX = 8                     # loop nesting depths the kernel holds (a class nested deeper is refused, not run wrong)
H0, M1, M2 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
FLIP = 1 << 63               # keys are kept as int64 with the top bit flipped, so int64 order is the uint64 order


def pack(code):
    """One class's engine bytecode [n, 3] -> packed int32 [n + 1] (with the HALT), and its deepest loop depth."""
    out, stack, deepest = [], [], 0
    for pc, (op, a, b) in enumerate(code.tolist()):
        if op == LOOP:
            dep = len(stack)
            stack.append(pc)
            deepest = max(deepest, dep + 1)
            out.append((LOOP, a, 0, dep, b))
        elif op == END:
            start = stack.pop()
            out.append((END, 0, 0, len(stack), start))
        elif op == ZERO:
            out.append((ZERO, a, 0, len(stack), b))
        elif op == MOV:
            out.append((MOV, a, b, 0, 0))
        else:
            out.append((op, a, 0, 0, 0))
    assert not stack
    out.append((HALT, 0, 0, 0, 0))
    words = []
    for op, a, b, dep, tgt in out:
        assert 0 <= tgt < (1 << 18) and dep < 16
        words.append(op | a << 3 | b << 6 | dep << 9 | tgt << 13)
    return np.array(words, np.int32), deepest


def pack_all(flat, off):
    """pack() on every class at once: engine.Codes' flat bytecode [n, 3] and class starts off [C + 1] -> the packed
    words of all classes, each with its HALT, their starts [C], and the deepest loop depth."""
    flat = np.asarray(flat, np.int64).reshape(-1, 3)
    off = np.asarray(off, np.int64)
    n, C = len(flat), len(off) - 1
    op, a, b = flat[:, 0], flat[:, 1], flat[:, 2]
    delta = (op == LOOP).astype(np.int64) - (op == END)
    before = np.cumsum(delta) - delta                   # loops open before each row (each class balances to 0)
    cls = np.repeat(np.arange(C), np.diff(off))
    local = np.arange(n) - off[cls]
    dep = np.where(op == END, before - 1, before)
    tgt = np.where((op == LOOP) | (op == ZERO), b, 0)
    loops = np.nonzero(op == LOOP)[0]
    tgt[off[cls[loops]] + b[loops]] = local[loops]       # an END's target is its LOOP
    ra = np.where(op <= ZERO, a, 0)
    ra = np.where(op == END, 0, ra)
    rb = np.where(op == MOV, b, 0)
    dep = np.where((op == LOOP) | (op == END) | (op == ZERO), dep, 0)
    assert (tgt < (1 << 18)).all() and (dep < 16).all()
    words = (op | ra << 3 | rb << 6 | dep << 9 | tgt << 13).astype(np.int32)
    out = np.insert(words, off[1:], HALT)                 # a HALT after each class's last row
    starts = (off[:-1] + np.arange(C)).astype(np.int32)
    deepest = int(before[op == LOOP].max()) + 1 if (op == LOOP).any() else 0
    return out, starts, deepest


class GCodes:
    """Every class's packed bytecode, flat, with each class's start, on a device (from engine.Codes' bytecode)."""

    def __init__(self, codes, device):
        flat, starts, self.depth = pack_all(codes.flat, codes.off)
        self.flat = torch.from_numpy(flat).to(device)
        self.off = torch.from_numpy(starts).to(device)
        self.n_code = len(flat)


def row_len(NP):
    """int16 entries in one state's row: its NP * 5 registers, zero-padded to whole 8-byte words (the hash's padding)."""
    return -(-NP * 5 // 4) * 4


# ---------------------------------------------------------------------------------------------------------------
# Triton kernels

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:          # no Triton here: the CPU kernel stands in
    HAVE_TRITON = False

if HAVE_TRITON:
    @triton.jit(do_not_specialize=["n_lanes", "NP", "c0", "S", "lo", "cap", "limit", "ROW"])
    def _interp_k(CODE, OFF, SRC, OUT, DEAD, n_lanes, NP, c0, S, lo, cap, limit, ROW,
                  BLOCK: tl.constexpr, D: tl.constexpr, UNROLL: tl.constexpr):
        """Lane = (candidate c, probe p); candidate c is item c0 + (lo + c) // S on source state (lo + c) % S."""
        pid = tl.program_id(0)
        lane = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ok = lane < n_lanes
        c = lane // NP
        p = lane % NP
        g = lo + c
        ci = c0 + g // S
        s = g % S
        base = tl.load(OFF + ci, mask=ok, other=0)
        sp = SRC + s * ROW + p * 5
        r0 = tl.load(sp + 0, mask=ok, other=0).to(tl.int32)
        r1 = tl.load(sp + 1, mask=ok, other=0).to(tl.int32)
        r2 = tl.load(sp + 2, mask=ok, other=0).to(tl.int32)
        r3 = tl.load(sp + 3, mask=ok, other=0).to(tl.int32)
        r4 = tl.load(sp + 4, mask=ok, other=0).to(tl.int32)
        zero = tl.zeros([BLOCK], dtype=tl.int32)
        pc = zero
        steps = zero
        m0 = zero
        m1 = zero
        m2 = zero
        m3 = zero
        m4 = zero
        m5 = zero
        m6 = zero
        m7 = zero
        run = ok
        dead = lane < 0
        while tl.max(run.to(tl.int32), axis=0) > 0:
            for _u in tl.static_range(UNROLL):      # UNROLL steps between checks of the loop condition
                ins = tl.load(CODE + base + pc, mask=run, other=7)
                op = ins & 7
                ra = (ins >> 3) & 7
                rb = (ins >> 6) & 7
                dp = (ins >> 9) & 15
                tg = ins >> 13
                va = tl.where(ra == 0, r0, tl.where(ra == 1, r1, tl.where(ra == 2, r2, tl.where(ra == 3, r3, r4))))
                vb = tl.where(rb == 0, r0, tl.where(rb == 1, r1, tl.where(rb == 2, r2, tl.where(rb == 3, r3, r4))))
                isp = run & (op < 4)
                nv = tl.where(op == 0, tl.minimum(va + 1, cap), tl.where(op == 1, tl.maximum(va - 1, 0),
                                                                           tl.where(op == 2, 0, vb)))
                r0 = tl.where(isp & (ra == 0), nv, r0)
                r1 = tl.where(isp & (ra == 1), nv, r1)
                r2 = tl.where(isp & (ra == 2), nv, r2)
                r3 = tl.where(isp & (ra == 3), nv, r3)
                r4 = tl.where(isp & (ra == 4), nv, r4)
                isl = run & (op == 4)
                enter = isl & (va > 0)
                ise = run & (op == 5)
                mv = m0
                if D > 1:
                    mv = tl.where(dp == 1, m1, mv)
                if D > 2:
                    mv = tl.where(dp == 2, m2, mv)
                if D > 3:
                    mv = tl.where(dp == 3, m3, mv)
                if D > 4:
                    mv = tl.where(dp == 4, m4, mv)
                if D > 5:
                    mv = tl.where(dp == 5, m5, mv)
                if D > 6:
                    mv = tl.where(dp == 6, m6, mv)
                if D > 7:
                    mv = tl.where(dp == 7, m7, mv)
                again = ise & (mv > 0)
                up = enter | again
                nm = tl.where(enter, va - 1, mv - 1)
                m0 = tl.where(up & (dp == 0), nm, m0)
                if D > 1:
                    m1 = tl.where(up & (dp == 1), nm, m1)
                if D > 2:
                    m2 = tl.where(up & (dp == 2), nm, m2)
                if D > 3:
                    m3 = tl.where(up & (dp == 3), nm, m3)
                if D > 4:
                    m4 = tl.where(up & (dp == 4), nm, m4)
                if D > 5:
                    m5 = tl.where(up & (dp == 5), nm, m5)
                if D > 6:
                    m6 = tl.where(up & (dp == 6), nm, m6)
                if D > 7:
                    m7 = tl.where(up & (dp == 7), nm, m7)
                steps = steps + (isp | up).to(tl.int32)
                died = up & (steps > limit)
                dead = dead | died
                isz = run & (op == 6)
                pc = tl.where((isl & ~enter) | again, tg + 1, tl.where(isz & (va != 0), tg, pc + 1))
                run = run & (op != 7) & ~died
        op_ = OUT + c * ROW + p * 5
        tl.store(op_ + 0, r0.to(tl.int16), mask=ok)
        tl.store(op_ + 1, r1.to(tl.int16), mask=ok)
        tl.store(op_ + 2, r2.to(tl.int16), mask=ok)
        tl.store(op_ + 3, r3.to(tl.int16), mask=ok)
        tl.store(op_ + 4, r4.to(tl.int16), mask=ok)
        tl.store(DEAD + lane, dead.to(tl.int8), mask=ok)

    @triton.jit(do_not_specialize=["n", "NP", "lo", "S", "W"])
    def _hash_k(OUT, SRC, DEAD, KEY, DROP, n, NP, lo, S, W, h0, k1, k2, flip, BLOCK: tl.constexpr):
        """Per candidate: library/search.hash_rows of its row (W little-endian words of four int16), whether it equals
        its source row, and whether a probe died. KEY gets the hash with its top bit flipped, as int64."""
        pid = tl.program_id(0)
        c = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ok = c < n
        s = (lo + c) % S
        ow = OUT + c * W
        sw = SRC + s * W
        h = tl.full([BLOCK], h0, tl.uint64)
        same = ok
        for i in range(W):
            w = tl.load(ow + i, mask=ok, other=0)
            v = tl.load(sw + i, mask=ok, other=0)
            same = same & (w == v)
            h = (h ^ w.to(tl.uint64, bitcast=True)) * k1
            h = h ^ (h >> 31)
        h = h * k2
        h = h ^ (h >> 29)
        dead = c < 0
        for j in range(NP):
            dead = dead | (tl.load(DEAD + c * NP + j, mask=ok, other=0) != 0)
        tl.store(KEY + c, (h ^ flip).to(tl.int64, bitcast=True), mask=ok)
        tl.store(DROP + c, (dead | same).to(tl.int8), mask=ok)


BLOCK_INTERP = 32            # lanes per program of the interpreter: one warp (kernel_bench.py, GPU_SEARCH.md §4)
WARPS_INTERP = 1
UNROLL = 2                   # machine steps between checks of the program's loop condition
BLOCK_HASH = 128


class Scratch:
    """The chunk's output buffers, kept between calls (their padding columns stay zero: the kernel never writes them)."""

    def __init__(self):
        self.key = None

    def get(self, n, NP, ROW, device):
        key = (NP, ROW, str(device))
        if self.key != key or self.out.shape[0] < n:
            self.out = torch.zeros((max(n, 1), ROW), dtype=torch.int16, device=device)
            self.dead = torch.empty(max(n, 1) * NP, dtype=torch.int8, device=device)
            self.keys = torch.empty(max(n, 1), dtype=torch.int64, device=device)
            self.drop = torch.empty(max(n, 1), dtype=torch.int8, device=device)
            self.key = key
        return self.out[:n], self.dead[:n * NP], self.keys[:n], self.drop[:n]


SCRATCH = Scratch()


def apply_chunk(gc, codes, c0, S, lo, n, src, NP, cap, limit):
    """Candidates lo .. lo + n - 1 of a segment (items c0, c0 + 1, ... each on every state of src [S, ROW] int16, item
    major): their rows [n, ROW] int16, keys [n] int64 (flipped) and drop [n] bool (a probe died, or the item left the
    state as it was). The rows are a view of a reused buffer: valid until the next call."""
    ROW = src.shape[1]
    W = ROW // 4
    if src.is_cuda and HAVE_TRITON:
        assert gc.depth <= DMAX, f"loop nesting {gc.depth} > {DMAX}"
        out, dead, keys, drop = SCRATCH.get(n, NP, ROW, src.device)
        lanes = n * NP
        _interp_k[(triton.cdiv(lanes, BLOCK_INTERP),)](
            gc.flat, gc.off, src, out, dead, lanes, NP, c0, S, lo, cap, limit, ROW,
            BLOCK=BLOCK_INTERP, D=max(gc.depth, 1), UNROLL=UNROLL, num_warps=WARPS_INTERP)
        _hash_k[(triton.cdiv(n, BLOCK_HASH),)](
            out.view(torch.int64), src.view(torch.int64), dead, keys, drop, n, NP, lo, S, W, H0, M1, M2, FLIP,
            BLOCK=BLOCK_HASH, num_warps=4)
        return out, keys, drop.bool()
    return _apply_chunk_cpu(codes, c0, S, lo, n, src, NP, cap, limit)


def _apply_chunk_cpu(codes, c0, S, lo, n, src, NP, cap, limit):
    """The same through the compiled CPU kernel (engine._apply_many), one item's slice at a time."""
    ROW = src.shape[1]
    s_np = src[:, :NP * 5].cpu().numpy().astype(np.int32).reshape(-1, NP, 5)
    rows, keys, drops = [], [], []
    g = lo
    while g < lo + n:
        ci, s = c0 + g // S, g % S
        e = min(S, s + (lo + n - g))
        regs, drop, k = E._apply_many(codes.flat, codes.off, np.array([ci], np.int64),
                                      np.ascontiguousarray(s_np[s:e]), cap, limit, codes.depth, E.MODE)
        rows.append(regs)
        keys.append(k)
        drops.append(drop)
        g += e - s
    regs = np.concatenate(rows)
    out = np.zeros((n, ROW), np.int16)
    out[:, :NP * 5] = regs.reshape(n, NP * 5).astype(np.int16)
    k = (np.concatenate(keys) ^ np.uint64(FLIP)).view(np.int64)
    return (torch.from_numpy(out).to(src.device), torch.from_numpy(k).to(src.device),
            torch.from_numpy(np.concatenate(drops)).to(src.device))
