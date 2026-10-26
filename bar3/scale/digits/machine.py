"""The digit loop (DIGITS.md §2): one new block for the bare machine (library/inventing/brm.py), in every base.

    D.B[...]   read X once on entry, as L.X does; run the body once per digit of that value in base B, from the top,
               with the digit in X. B = 1 is unary: v ones. The value 0 has no digits.

It knows nothing of remainders or of arithmetic: it walks a number's digits, the way L.X walks its units. Every
iteration counts one step as it starts, as L.r's do, so a machine dies when its step count passes the limit at an
iteration's start. In unary, D.1 runs as many iterations as L.X.

Importing this module adds the block to brm's interpreter (brm._Run.items) and to the compiled search
(bar3/scale/engine.py: its bytecode compiler, the per-machine kernel and the signature kernel), in this process
only. Programs without a D block run exactly as before; `check()` tests that, and that the interpreter and the
kernel agree on programs with D blocks.

    cd cpu_first && <python with numba> -m bar3.scale.digits.machine        # the check, a minute
"""

from __future__ import annotations

import numba as nb
import numpy as np

from library.inventing import brm
from bar3.scale import engine

DLOOP = 7                                 # the new opcode: DLOOP B e, run to the END at e once per digit
KINDS = {1: "D.1", 2: "D.2", 10: "D.10"}  # the block in each representation of x: unary, binary, decimal


def base_of(kind):
    """The base of a digit block's kind ("D.10" -> 10), or None for any other kind."""
    return int(kind[2:]) if kind.startswith("D.") else None


def digits(v, B):
    """v's digits in base B, from the top (B = 1: v ones); 0 has none."""
    v = int(v)
    if B == 1:
        return [1] * v
    out = []
    while v > 0:
        out.append(v % B)
        v //= B
    return out[::-1]


def _count(v, B):
    """The number of digits of each value of v (an int array), in base B."""
    v = v.astype(np.int64)
    if B == 1:
        return v.copy()
    n = np.zeros(len(v), np.int64)
    w = v.copy()
    while (w > 0).any():
        n += w > 0
        w //= B
    return n


# ---------------------------------------------------------------------------------------------------------------
# the interpreter: brm._Run.items with the block

_ORIGINAL_ITEMS = brm._Run.items


def _items(self, items, idx):
    R = self.R
    rest = []
    for j, it in enumerate(items):
        if isinstance(it, tuple) and it[1] not in self.lib and base_of(it[1]) is not None:
            if rest:
                _ORIGINAL_ITEMS(self, tuple(rest), idx)
                rest = []
            if len(idx) == 0:
                return
            B = base_of(it[1])
            v = R[idx, brm.X].astype(np.int64)                 # read once, on entry
            cnt = _count(v, B)
            cur, t = idx, 0
            while True:
                keep = cnt > t
                if not keep.all():
                    cur, cnt, v = cur[keep], cnt[keep], v[keep]
                if len(cur) == 0:
                    break
                self.steps[cur] += 1
                over = self.steps[cur] > self.m.limit
                if over.any():
                    self.dead[cur[over]] = True
                    cur, cnt, v = cur[~over], cnt[~over], v[~over]
                    if len(cur) == 0:
                        break
                R[cur, brm.X] = 1 if B == 1 else (v // B ** (cnt - 1 - t)) % B
                self.items(it[2], cur)
                t += 1
            continue
        rest.append(it)
    if rest:
        _ORIGINAL_ITEMS(self, tuple(rest), idx)


# ---------------------------------------------------------------------------------------------------------------
# the compiled search: engine.py's kernels with the DLOOP opcode (the rest of engine.py is unchanged and calls these)

_ORIGINAL_COMPILE = engine.compile_items


def compile_items(items, lib):
    """engine.compile_items, line for line, plus DLOOP B e for a digit block (its body, then END, as LOOP)."""
    out, depth = [], [0]

    def emit(items, d):
        depth[0] = max(depth[0], d)
        for it in items:
            if isinstance(it, tuple):
                kind, body = it[1], it[2]
                if kind in lib:
                    emit(brm.fill(lib[kind], body[0]), d)
                    continue
                pos = len(out)
                if base_of(kind) is not None:
                    out.append([DLOOP, base_of(kind), 0])
                    emit(body, d + 1)
                    out.append([engine.END, 0, 0])
                    out[pos][2] = len(out) - 1
                    continue
                r = brm.RI[kind[2:]]
                if kind[0] == "Z":
                    out.append([engine.ZERO, r, 0])
                    emit(body, d)
                    out[pos][2] = len(out)
                else:
                    out.append([engine.LOOP, r, 0])
                    emit(body, d + 1)
                    out.append([engine.END, 0, 0])
                    out[pos][2] = len(out) - 1
                continue
            if it in lib:
                emit(lib[it], d)
                continue
            op, _, arg = it.partition(".")
            if op == "MOV":
                a, b = arg.split(".")
                out.append([engine.MOV, brm.RI[a], brm.RI[b]])
            else:
                out.append([{"INC": engine.INC, "DEC": engine.DEC, "CLR": engine.CLR}[op], brm.RI[arg], 0])

    emit(tuple(items), 0)
    return np.array(out, np.int32).reshape(-1, 3), depth[0]


@nb.njit(cache=True)
def _place(v, B, p):
    """The digit of v at place p (0 = units) in base B; 1 in unary."""
    if B == 1:
        return 1
    w = v
    for _ in range(p):
        w //= B
    return w % B


@nb.njit(cache=True)
def _run_block(code, lo, hi, regs, steps, cap, limit, spc, srem, sval, sbase):
    """engine._run_block with DLOOP: srem holds the iterations left after the current one, which is also the
    current digit's place; sval and sbase hold the value read on entry and the base (0 for an L block)."""
    n = hi
    pc = lo
    sp = 0
    while pc < n:
        op = code[pc, 0]
        if op == 0:
            a = code[pc, 1]
            v = regs[a] + 1
            regs[a] = cap if v > cap else v
            steps += 1
            pc += 1
        elif op == 1:
            a = code[pc, 1]
            v = regs[a] - 1
            regs[a] = 0 if v < 0 else v
            steps += 1
            pc += 1
        elif op == 2:
            regs[code[pc, 1]] = 0
            steps += 1
            pc += 1
        elif op == 3:
            regs[code[pc, 1]] = regs[code[pc, 2]]
            steps += 1
            pc += 1
        elif op == 4:
            cnt = regs[code[pc, 1]]
            if cnt <= 0:
                pc = code[pc, 2] + 1
            else:
                steps += 1
                if steps > limit:
                    return -1
                spc[sp] = pc
                srem[sp] = cnt - 1
                sbase[sp] = 0
                sp += 1
                pc += 1
        elif op == 5:
            t = sp - 1
            if srem[t] > 0:
                srem[t] -= 1
                steps += 1
                if steps > limit:
                    return -1
                if sbase[t] > 0:
                    regs[0] = _place(sval[t], sbase[t], srem[t])
                pc = spc[t] + 1
            else:
                sp -= 1
                pc += 1
        elif op == 6:
            if regs[code[pc, 1]] == 0:
                pc += 1
            else:
                pc = code[pc, 2]
        else:
            B = code[pc, 1]
            v = np.int64(regs[0])
            cnt = v
            if B > 1:
                cnt = 0
                w = v
                while w > 0:
                    cnt += 1
                    w //= B
            if cnt <= 0:
                pc = code[pc, 2] + 1
            else:
                steps += 1
                if steps > limit:
                    return -1
                spc[sp] = pc
                srem[sp] = cnt - 1
                sval[sp] = v
                sbase[sp] = B
                sp += 1
                regs[0] = _place(v, B, cnt - 1)
                pc += 1
    return steps


@nb.njit(cache=True)
def _run(code, regs, cap, limit, spc, srem, sval, sbase):
    return _run_block(code, 0, code.shape[0], regs, 0, cap, limit, spc, srem, sval, sbase) < 0


@nb.njit(cache=True)
def _run_state(code, st, steps, cnt, live, cap, limit, spc, srem, sval, sbase, mode):
    """engine._run_state; a top-level DLOOP runs one probe at a time, as any block with a block inside."""
    n = code.shape[0]
    NP = st.shape[0]
    if mode == 0:
        for p in range(NP):
            if _run(code, st[p], cap, limit, spc, srem, sval, sbase):
                return True
        return False
    steps[:] = 0
    live[:] = True
    pc = 0
    while pc < n:
        op = code[pc, 0]
        a = code[pc, 1]
        if op <= 3:
            engine._masked(code, pc, st, live, cap)
            steps += 1
            pc += 1
            continue
        end = code[pc, 2] + 1 if op == 4 or op == 7 else code[pc, 2]
        last = end - 1 if op == 4 or op == 7 else end
        straight = mode == 2 and op != 7
        for q in range(pc + 1, last):
            if code[q, 0] >= 4:
                straight = False
                break
        if not straight:
            for p in range(NP):
                r = _run_block(code, pc, end, st[p], steps[p], cap, limit, spc, srem, sval, sbase)
                if r < 0:
                    return True
                steps[p] = r
        elif op == 6:
            for p in range(NP):
                live[p] = st[p, a] == 0
            for q in range(pc + 1, last):
                engine._masked(code, q, st, live, cap)
            for p in range(NP):
                if live[p]:
                    steps[p] += last - pc - 1
            live[:] = True
        else:
            top = 0
            for p in range(NP):
                cnt[p] = st[p, a]
                if cnt[p] > top:
                    top = cnt[p]
            for t in range(top):
                for p in range(NP):
                    live[p] = cnt[p] > t
                    if live[p]:
                        steps[p] += 1
                        if steps[p] > limit:
                            return True
                for q in range(pc + 1, last):
                    engine._masked(code, q, st, live, cap)
                for p in range(NP):
                    if live[p]:
                        steps[p] += last - pc - 1
            live[:] = True
        pc = end
    return False


@nb.njit(cache=True)
def _apply_many(flat, off, cls, src, cap, limit, depth, mode):
    """engine._apply_many on this module's kernel."""
    S, NP = src.shape[0], src.shape[1]
    n = cls.shape[0] * S
    out = np.empty((n, NP, 5), np.int32)
    drop = np.zeros(n, np.bool_)
    keys = np.zeros(n, np.uint64)
    spc = np.empty(depth + 1, np.int64)
    srem = np.empty(depth + 1, np.int64)
    sval = np.empty(depth + 1, np.int64)
    sbase = np.empty(depth + 1, np.int64)
    steps = np.empty(NP, np.int64)
    cnt = np.empty(NP, np.int64)
    live = np.empty(NP, np.bool_)
    for j in range(cls.shape[0]):
        code = flat[off[cls[j]]:off[cls[j] + 1]]
        for s in range(S):
            o = j * S + s
            out[o] = src[s]
            dead = _run_state(code, out[o], steps, cnt, live, cap, limit, spc, srem, sval, sbase, mode)
            if dead:
                drop[o] = True
                continue
            same = True
            for p in range(NP):
                for r in range(5):
                    if out[o, p, r] != src[s, p, r]:
                        same = False
                        break
                if not same:
                    break
            if same:
                drop[o] = True
            else:
                keys[o] = engine._hash_state(out[o])
    return out, drop, keys


@nb.njit(cache=True)
def _sig_kernel(code, tests, cap, limit, depth):
    """engine._sig_kernel on this module's kernel."""
    S = tests.shape[0]
    out = tests.copy()
    spc = np.empty(depth + 1, np.int64)
    srem = np.empty(depth + 1, np.int64)
    sval = np.empty(depth + 1, np.int64)
    sbase = np.empty(depth + 1, np.int64)
    rows = np.empty(S, np.uint64)
    for s in range(S):
        d = False
        for p in range(tests.shape[1]):
            if _run(code, out[s, p], cap, limit, spc, srem, sval, sbase):
                d = True
                break
        rows[s] = np.uint64(7) if d else engine._hash_state(out[s])
    h = np.uint64(0x9E3779B97F4A7C15)
    m1 = np.uint64(0xBF58476D1CE4E5B9)
    m2 = np.uint64(0x94D049BB133111EB)
    for i in range(S):
        h = (h ^ rows[i]) * m1
        h ^= h >> np.uint64(31)
    h = h * m2
    h ^= h >> np.uint64(29)
    return h


@nb.njit(cache=True)
def _eval_many(code, xs, Ns, cap, limit, depth):
    """A after code on fresh machines (x, N), one at a time, and which ran out of steps."""
    n = xs.shape[0]
    out = np.zeros(n, np.int64)
    dead = np.zeros(n, np.bool_)
    spc = np.empty(depth + 1, np.int64)
    srem = np.empty(depth + 1, np.int64)
    sval = np.empty(depth + 1, np.int64)
    sbase = np.empty(depth + 1, np.int64)
    regs = np.zeros(5, np.int32)
    for i in range(n):
        regs[:] = 0
        regs[0] = xs[i]
        regs[1] = Ns[i]
        dead[i] = _run_block(code, 0, code.shape[0], regs, 0, cap, limit, spc, srem, sval, sbase) < 0
        out[i] = regs[2]
    return out, dead


def evaluate(prog, lib, xs, Ns, m):
    """brm.evaluate on the compiled kernel (the same values; check() tests the kernel against the interpreter): A's
    value for each (x, N), and whether any ran out of steps. lib: parsed items."""
    code, depth = compile_items(tuple(prog), lib)
    got, dead = _eval_many(code, np.asarray(xs, np.int32), np.asarray(Ns, np.int32), m.cap, m.limit, depth)
    return got, bool(dead.any())


_ORIGINAL = dict(compile_items=engine.compile_items, _apply_many=engine._apply_many, _sig_kernel=engine._sig_kernel)


def install():
    """Add the block to brm's interpreter and to the compiled search, in this process."""
    brm._Run.items = _items
    engine.compile_items, engine._apply_many, engine._sig_kernel = compile_items, _apply_many, _sig_kernel


install()


# ---------------------------------------------------------------------------------------------------------------
# the check

def _random_program(rng, kinds, depth=0):
    out = []
    for _ in range(rng.integers(1, 4)):
        if depth < 2 and rng.random() < 0.3:
            out.append(("REP", kinds[rng.integers(len(kinds))], _random_program(rng, kinds, depth + 1)))
        else:
            out.append(brm.PRIMS[rng.integers(len(brm.PRIMS))])
    return tuple(out)


def check(n=3000, seed=0):
    """(1) Without D blocks, the kernel and the interpreter here give what the original kernel gives: the same
    states, deaths and hash keys. (2) With D blocks, the kernel and the interpreter agree on states and deaths."""
    rng = np.random.default_rng(seed)
    m = brm.Machine(cap=200, limit=2000)
    src = brm.random_states(64, 6, rng, top=40)
    plain = list(brm.LOOP_KINDS)
    for i in range(n):
        with_d = i % 2 == 1
        kinds = plain + (list(KINDS.values()) if with_d else [])
        prog = _random_program(rng, kinds)
        code, depth = compile_items(prog, {})
        flat, off, cls = code, np.array([0, len(code)], np.int64), np.array([0], np.int64)
        out, drop, keys = _apply_many(flat, off, cls, src.regs, m.cap, m.limit, depth, engine.MODE)
        b = src.copy()
        brm.run_items(m, b, prog, {})
        for s in range(src.S):
            if b.dead[s]:
                assert drop[s], (prog, s)
            else:
                assert (out[s] == b.regs[s]).all(), (prog, s)
                assert drop[s] == (out[s] == src.regs[s]).all(), (prog, s)
        if not with_d:
            oc, od = _ORIGINAL_COMPILE(prog, {})
            assert (oc == code).all() and od == depth, prog
            o2, d2, k2 = _ORIGINAL["_apply_many"](flat, off, cls, src.regs, m.cap, m.limit, depth, engine.MODE)
            assert (o2 == out).all() and (d2 == drop).all() and (k2 == keys).all(), prog
    big = brm.Machine(cap=10 ** 9, limit=10 ** 6)
    for B in KINDS:
        for v in (0, 1, 5, 9, 10, 99, 100, 12345):
            b = brm.fresh(np.array([v]), np.array([7]))
            brm.run_items(big, b, (("REP", KINDS[B], ("INC.A", "MOV.T.X", "INC.U")),), {})
            ds = digits(v, B)
            assert b.regs[0, 0, brm.A] == len(ds) and b.regs[0, 0, brm.U] == len(ds), (B, v)
            assert b.regs[0, 0, brm.T] == (ds[-1] if ds else 0), (B, v)
    print(f"machine check: {n} random programs, half with D blocks: kernel and interpreter agree; without D blocks"
          " the kernel is the original's")


if __name__ == "__main__":
    check()
