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

The digit loop. With bar3/scale/digits/machine.py installed, compile_items also writes DLOOP B e (op 7) for a D.B
block: read X once on entry, run the body once per digit of that value in base B, from the top, with the digit in X
(B = 1: v ones). It is packed as a LOOP on X whose b field (and its END's) holds the base's index + 1 in the
vocabulary's bases (`pack_all`'s fourth result); its iterations count steps as a LOOP's do. The kernel then also keeps
the value read on entry per depth, and takes the number of digits and the place values from small tables
(`digit_tables`). Its digit code is compiled only for a vocabulary with digit loops (the DIG switch), so a
vocabulary without them runs the kernel as it was.

Without a GPU (or Triton), `apply_chunk` runs the compiled CPU kernel (engine._apply_many) instead, so the search
around it can be tested anywhere.
"""

from __future__ import annotations

import numpy as np
import torch

from bar3.scale import engine as E

INC, DEC, CLR, MOV, LOOP, END, ZERO, HALT = range(8)
DLOOP = 7                    # the digit loop in engine bytecode (bar3/scale/digits/machine.py); packed as a LOOP
DMAX = 8                     # loop nesting depths the kernel holds (a class nested deeper is refused, not run wrong)
NBASE = 7                    # digit bases a vocabulary may hold (their index + 1 fills the packed b field's 3 bits)
VMAX = 1 << 15               # register values are int16: the digit tables cover 0 .. VMAX - 1
PMAX = 16                    # places of a value below VMAX in any base >= 2
H0, M1, M2 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
FLIP = 1 << 63               # keys are kept as int64 with the top bit flipped, so int64 order is the uint64 order


def pack(code, bases=None):
    """One class's engine bytecode [n, 3] -> packed int32 [n + 1] (with the HALT), its deepest loop depth and its digit
    bases (`bases`: a vocabulary's, else the class's own). Row by row, as a reference for pack_all."""
    rows = code.tolist()
    bases = sorted({a for op, a, _ in rows if op == DLOOP}) if bases is None else list(bases)
    out, stack, deepest = [], [], 0
    for pc, (op, a, b) in enumerate(rows):
        if op in (LOOP, DLOOP):
            dep = len(stack)
            bi = bases.index(a) + 1 if op == DLOOP else 0
            stack.append((pc, bi))
            deepest = max(deepest, dep + 1)
            out.append((LOOP, a if op == LOOP else 0, bi, dep, b))
        elif op == END:
            start, bi = stack.pop()
            out.append((END, 0, bi, len(stack), start))
        elif op == ZERO:
            out.append((ZERO, a, 0, len(stack), b))
        elif op == MOV:
            out.append((MOV, a, b, 0, 0))
        else:
            assert op in (INC, DEC, CLR), f"unknown opcode {op}"
            out.append((op, a, 0, 0, 0))
    assert not stack
    out.append((HALT, 0, 0, 0, 0))
    words = []
    for op, a, b, dep, tgt in out:
        assert 0 <= tgt < (1 << 18) and dep < 16 and b < 8
        words.append(op | a << 3 | b << 6 | dep << 9 | tgt << 13)
    return np.array(words, np.int32), deepest, bases


def pack_all(flat, off):
    """pack() on every class at once: engine.Codes' flat bytecode [n, 3] and class starts off [C + 1] -> the packed
    words of all classes, each with its HALT, their starts [C], the deepest loop depth, and the digit loops' bases
    (sorted; a digit loop's packed b field is its base's index + 1)."""
    flat = np.asarray(flat, np.int64).reshape(-1, 3)
    off = np.asarray(off, np.int64)
    n, C = len(flat), len(off) - 1
    op, a, b = flat[:, 0], flat[:, 1], flat[:, 2]
    assert ((op >= INC) & (op <= DLOOP)).all(), f"unknown opcodes {sorted(set(op.tolist()) - set(range(8)))}"
    opens = (op == LOOP) | (op == DLOOP)
    delta = opens.astype(np.int64) - (op == END)
    before = np.cumsum(delta) - delta                   # loops open before each row (each class balances to 0)
    cls = np.repeat(np.arange(C), np.diff(off))
    local = np.arange(n) - off[cls]
    dep = np.where(op == END, before - 1, before)
    tgt = np.where(opens | (op == ZERO), b, 0)
    loops = np.nonzero(opens)[0]
    tgt[off[cls[loops]] + b[loops]] = local[loops]       # an END's target is its LOOP
    ra = np.where(op <= ZERO, a, 0)                      # a DLOOP reads X (register 0)
    ra = np.where(op == END, 0, ra)
    rb = np.where(op == MOV, b, 0)
    dig = np.nonzero(op == DLOOP)[0]
    bases = sorted(set(a[dig].tolist()))
    assert len(bases) <= NBASE and all(B >= 1 for B in bases), f"digit bases {bases}"
    if len(dig):
        bi = np.searchsorted(bases, a[dig]) + 1
        rb[dig] = bi
        rb[off[cls[dig]] + b[dig]] = bi                  # and its END
    dep = np.where(opens | (op == END) | (op == ZERO), dep, 0)
    assert (tgt < (1 << 18)).all() and (dep < 16).all()
    words = (np.where(op == DLOOP, LOOP, op) | ra << 3 | rb << 6 | dep << 9 | tgt << 13).astype(np.int32)
    out = np.insert(words, off[1:], HALT)                 # a HALT after each class's last row
    starts = (off[:-1] + np.arange(C)).astype(np.int32)
    deepest = int(before[opens].max()) + 1 if opens.any() else 0
    return out, starts, deepest, bases


def digit_tables(bases):
    """For each digit base (packed index i + 1): the base, the number of digits of every value below VMAX in it (unary:
    the value itself), and its place values B^p for p < PMAX (capped at 2^31 - 1, beyond any value's places). With no
    base, one-entry stand-ins (the kernel is then compiled without its digit code and reads none)."""
    if not bases:
        return np.ones(1, np.int32), np.zeros(1, np.int32), np.ones(1, np.int32)
    v = np.arange(VMAX, dtype=np.int64)
    base = np.array(bases, np.int32)
    ndig = np.zeros((len(bases), VMAX), np.int32)
    place = np.ones((len(bases), PMAX), np.int32)
    for i, B in enumerate(bases):
        if B == 1:
            ndig[i] = v
            continue
        w = v.copy()
        while (w > 0).any():
            ndig[i] += w > 0
            w //= B
        place[i] = [min(B ** p, 2 ** 31 - 1) for p in range(PMAX)]
    return base, ndig.reshape(-1), place.reshape(-1)


class GCodes:
    """Every class's packed bytecode, flat, with each class's start, on a device (from engine.Codes' bytecode), and the
    digit tables of its digit loops' bases."""

    def __init__(self, codes, device):
        flat, starts, self.depth, self.bases = pack_all(codes.flat, codes.off)
        self.flat = torch.from_numpy(flat).to(device)
        self.off = torch.from_numpy(starts).to(device)
        self.n_code = len(flat)
        self.dbase, self.dndig, self.dplace = (torch.from_numpy(t).to(device) for t in digit_tables(self.bases))


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
    def _interp_k(CODE, OFF, SRC, OUT, DEAD, DBASE, DNDIG, DPLACE, n_lanes, NP, c0, S, lo, cap, limit, ROW,
                  BLOCK: tl.constexpr, D: tl.constexpr, UNROLL: tl.constexpr, DIG: tl.constexpr,
                  VM: tl.constexpr, PM: tl.constexpr):
        """Lane = (candidate c, probe p); candidate c is item c0 + (lo + c) // S on source state (lo + c) % S. With DIG,
        digit loops too (DBASE, DNDIG, DPLACE: digit_tables, VM and PM their row lengths)."""
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
        v0 = zero                   # a digit loop's value read on entry, per depth (DIG only)
        v1 = zero
        v2 = zero
        v3 = zero
        v4 = zero
        v5 = zero
        v6 = zero
        v7 = zero
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
                ise = run & (op == 5)
                cnt = va                    # a LOOP's iterations: its register's value
                if DIG:
                    dl = (isl | ise) & (rb > 0)          # a digit loop's LOOP or END (b: its base's index + 1)
                    bi = tl.maximum(rb - 1, 0)
                    bse = tl.load(DBASE + bi, mask=dl, other=1)
                    nd = tl.load(DNDIG + bi * VM + va, mask=dl & isl & (bse > 1), other=0)
                    cnt = tl.where(dl & isl & (bse > 1), nd, va)     # X's digits (unary: X itself)
                enter = isl & (cnt > 0)
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
                nm = tl.where(enter, cnt - 1, mv - 1)
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
                if DIG:
                    # an iteration starts (on entry, or at its END): X gets the digit at place nm (the iterations
                    # left after this one) of the value read on entry; in unary, 1
                    vv = v0
                    if D > 1:
                        vv = tl.where(dp == 1, v1, vv)
                    if D > 2:
                        vv = tl.where(dp == 2, v2, vv)
                    if D > 3:
                        vv = tl.where(dp == 3, v3, vv)
                    if D > 4:
                        vv = tl.where(dp == 4, v4, vv)
                    if D > 5:
                        vv = tl.where(dp == 5, v5, vv)
                    if D > 6:
                        vv = tl.where(dp == 6, v6, vv)
                    if D > 7:
                        vv = tl.where(dp == 7, v7, vv)
                    val = tl.where(enter, va, vv)
                    pl = tl.load(DPLACE + bi * PM + nm, mask=dl & up & (bse > 1), other=1)
                    r0 = tl.where(dl & up, tl.where(bse > 1, (val // pl) % bse, 1), r0)
                    ent = dl & enter
                    v0 = tl.where(ent & (dp == 0), va, v0)
                    if D > 1:
                        v1 = tl.where(ent & (dp == 1), va, v1)
                    if D > 2:
                        v2 = tl.where(ent & (dp == 2), va, v2)
                    if D > 3:
                        v3 = tl.where(ent & (dp == 3), va, v3)
                    if D > 4:
                        v4 = tl.where(ent & (dp == 4), va, v4)
                    if D > 5:
                        v5 = tl.where(ent & (dp == 5), va, v5)
                    if D > 6:
                        v6 = tl.where(ent & (dp == 6), va, v6)
                    if D > 7:
                        v7 = tl.where(ent & (dp == 7), va, v7)
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
            gc.flat, gc.off, src, out, dead, gc.dbase, gc.dndig, gc.dplace, lanes, NP, c0, S, lo, cap, limit, ROW,
            BLOCK=BLOCK_INTERP, D=max(gc.depth, 1), UNROLL=UNROLL, DIG=bool(gc.bases), VM=VMAX, PM=PMAX,
            num_warps=WARPS_INTERP)
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
