"""The bar-3 search (bar3/search.py over library/inventing/brm.py) with the bare machine compiled, exact: the same
states, the same hash keys, the same order, so every record field but the seconds comes out the same (check.py).

Where the time went (a 250k-state probe, profiled): three quarters in brm's interpreter, which walks every item as
Python recursion over numpy index arrays (1.65M calls), and most of the rest in hashing states row by row. Here
- every item is compiled once per job into a flat bytecode (library tokens and abstractions inlined, as brm runs
  them), and one numba kernel runs it on every probe of every state (straight-line code and loops over it on all of
  a state's probes in lockstep, `MODE` 2) and hashes each new state with brm's own hash (library/search.hash_rows
  over the state's int16 bytes), in the same pass; an item that leaves a state as it was is not hashed, since that
  state is already seen;
- the search keeps bar3/search.py's bookkeeping line for line, but feeds the kernel many classes at once when their
  source level is small (chunk boundaries change no outcome: the level's candidates come in the same order);
- `bfs_multi` serves several problems posed on the same instances with one search;
- the item enumeration (library/inventing/bfs.enumerate_items) is the same code with the signature computed by the
  kernel.

Semantics kept from brm._Run: values saturate at cap; DEC stops at 0; every primitive counts one step and so does
every loop iteration, as it starts; a machine whose step count passes the limit at an iteration's start is dead, and a
state with a dead probe is dropped; L.r reads r once, on entry; Z.r runs its body when r = 0. Steps restart at 0 for
each applied item (each brm.apply is its own run). Checking a found program on all instances stays brm.evaluate.
"""

from __future__ import annotations

import time

import numba as nb
import numpy as np

from library import dsl
from library.inventing import brm
from library.inventing.bfs import ItemClass
from library.search import _fresh, hash_rows
from bar3.search import Problem, Result, programs, programs_at, near_misses  # noqa: F401  (re-exported)

INC, DEC, CLR, MOV, LOOP, END, ZERO = range(7)
SLICE = 16384


def compile_items(items, lib):
    """Items -> (code [n, 3] int32, loop depth). LOOP r e: run to the END at e, r times; ZERO r e: skip to e unless
    r = 0. Library tokens and abstractions (a loop kind in lib) are inlined, as brm._Run does."""
    out, depth = [], [0]

    def emit(items, d):
        depth[0] = max(depth[0], d)
        for it in items:
            if isinstance(it, tuple):
                kind, body = it[1], it[2]
                if kind in lib:
                    emit(brm.fill(lib[kind], body[0]), d)
                    continue
                r = brm.RI[kind[2:]]
                pos = len(out)
                if kind[0] == "Z":
                    out.append([ZERO, r, 0])
                    emit(body, d)
                    out[pos][2] = len(out)
                else:
                    out.append([LOOP, r, 0])
                    emit(body, d + 1)
                    out.append([END, 0, 0])
                    out[pos][2] = len(out) - 1
                continue
            if it in lib:
                emit(lib[it], d)
                continue
            op, _, arg = it.partition(".")
            if op == "MOV":
                a, b = arg.split(".")
                out.append([MOV, brm.RI[a], brm.RI[b]])
            else:
                out.append([{"INC": INC, "DEC": DEC, "CLR": CLR}[op], brm.RI[arg], 0])

    emit(tuple(items), 0)
    return np.array(out, np.int32).reshape(-1, 3), depth[0]


@nb.njit(cache=True)
def _run(code, regs, cap, limit, spc, srem):
    """One machine (regs: int32[5], in place). Returns True when it dies (its step count passes the limit)."""
    return _run_block(code, 0, code.shape[0], regs, 0, cap, limit, spc, srem) < 0


@nb.njit(cache=True)
def _run_block(code, lo, hi, regs, steps, cap, limit, spc, srem):
    """code[lo:hi] on one machine, its step count starting at `steps`: the count after, or -1 if it dies."""
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
                sp += 1
                pc += 1
        elif op == 5:
            t = sp - 1
            if srem[t] > 0:
                srem[t] -= 1
                steps += 1
                if steps > limit:
                    return -1
                pc = spc[t] + 1
            else:
                sp -= 1
                pc += 1
        else:
            if regs[code[pc, 1]] == 0:
                pc += 1
            else:
                pc = code[pc, 2]
    return steps


@nb.njit(cache=True)
def _masked(code, q, st, live, cap):
    """Instruction q on every probe p of st with live[p]."""
    op, a, b = code[q, 0], code[q, 1], code[q, 2]
    for p in range(st.shape[0]):
        if live[p]:
            if op == 0:
                v = st[p, a] + 1
                st[p, a] = cap if v > cap else v
            elif op == 1:
                v = st[p, a] - 1
                st[p, a] = 0 if v < 0 else v
            elif op == 2:
                st[p, a] = 0
            else:
                st[p, a] = st[p, b]


@nb.njit(cache=True)
def _run_state(code, st, steps, cnt, live, cap, limit, spc, srem, mode):
    """Every probe of one state (st: int32[NP, 5], in place), with the outcome of _run on each probe. mode 0: each
    probe alone; 1: straight-line instructions on all probes at once, blocks one probe at a time; 2: as 1, and a
    top-level loop or zero test over straight-line code in lockstep (each probe with its own count, read on entry, and
    its own step count). Returns True when some probe dies."""
    n = code.shape[0]
    NP = st.shape[0]
    if mode == 0:
        for p in range(NP):
            if _run(code, st[p], cap, limit, spc, srem):
                return True
        return False
    steps[:] = 0
    live[:] = True
    pc = 0
    while pc < n:
        op = code[pc, 0]
        a = code[pc, 1]
        if op <= 3:
            _masked(code, pc, st, live, cap)
            steps += 1
            pc += 1
            continue
        end = code[pc, 2] + 1 if op == 4 else code[pc, 2]       # past the loop's END, or past the zero test's body
        last = end - 1 if op == 4 else end                     # the body is code[pc + 1:last]
        straight = mode == 2
        for q in range(pc + 1, last):
            if code[q, 0] >= 4:
                straight = False
                break
        if not straight:
            for p in range(NP):
                r = _run_block(code, pc, end, st[p], steps[p], cap, limit, spc, srem)
                if r < 0:
                    return True
                steps[p] = r
        elif op == 6:
            for p in range(NP):
                live[p] = st[p, a] == 0
            for q in range(pc + 1, last):
                _masked(code, q, st, live, cap)
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
                    _masked(code, q, st, live, cap)
                for p in range(NP):
                    if live[p]:
                        steps[p] += last - pc - 1
            live[:] = True
        pc = end
    return False


@nb.njit(cache=True)
def _hash_state(st):
    """library/search.hash_rows of one state's features (its registers as int16, little-endian, zero-padded)."""
    flat = st.ravel()
    n16 = flat.size
    h = np.uint64(0x9E3779B97F4A7C15)
    m1 = np.uint64(0xBF58476D1CE4E5B9)
    m2 = np.uint64(0x94D049BB133111EB)
    full = n16 // 4
    for i in range(full):
        q = 4 * i
        w = (np.uint64(flat[q] & 0xFFFF) | (np.uint64(flat[q + 1] & 0xFFFF) << np.uint64(16))
             | (np.uint64(flat[q + 2] & 0xFFFF) << np.uint64(32)) | (np.uint64(flat[q + 3] & 0xFFFF) << np.uint64(48)))
        h = (h ^ w) * m1
        h ^= h >> np.uint64(31)
    if n16 > 4 * full:
        w = np.uint64(0)
        for j in range(n16 - 4 * full):
            w |= np.uint64(flat[4 * full + j] & 0xFFFF) << np.uint64(16 * j)
        h = (h ^ w) * m1
        h ^= h >> np.uint64(31)
    h = h * m2
    h ^= h >> np.uint64(29)
    return h


@nb.njit(cache=True)
def _apply_many(flat, off, cls, src, cap, limit, depth, mode):
    """Classes cls (each code flat[off[c]:off[c+1]]) on every state of src [S, NP, 5]: outputs in class-major order,
    [len(cls) * S, NP, 5], with hash keys, and `drop`: the state died, or the item left it as it was. An unchanged
    state is its source, which an earlier level already holds, so the search would find it stale anyway: it is not
    hashed (the outcome is the same, minus the hash)."""
    S, NP = src.shape[0], src.shape[1]
    n = cls.shape[0] * S
    out = np.empty((n, NP, 5), np.int32)
    drop = np.zeros(n, np.bool_)
    keys = np.zeros(n, np.uint64)
    spc = np.empty(depth + 1, np.int64)
    srem = np.empty(depth + 1, np.int64)
    steps = np.empty(NP, np.int64)
    cnt = np.empty(NP, np.int64)
    live = np.empty(NP, np.bool_)
    for j in range(cls.shape[0]):
        code = flat[off[cls[j]]:off[cls[j] + 1]]
        for s in range(S):
            o = j * S + s
            out[o] = src[s]
            dead = _run_state(code, out[o], steps, cnt, live, cap, limit, spc, srem, mode)
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
                keys[o] = _hash_state(out[o])
    return out, drop, keys


@nb.njit(cache=True)
def _sig_kernel(code, tests, cap, limit, depth):
    """bfs._sig: the item run on random test states; a dead state's row is 7; the rows' keys hashed together."""
    S = tests.shape[0]
    out = tests.copy()
    spc = np.empty(depth + 1, np.int64)
    srem = np.empty(depth + 1, np.int64)
    rows = np.empty(S, np.uint64)
    for s in range(S):
        d = False
        for p in range(tests.shape[1]):
            if _run(code, out[s, p], cap, limit, spc, srem):
                d = True
                break
        rows[s] = np.uint64(7) if d else _hash_state(out[s])
    h = np.uint64(0x9E3779B97F4A7C15)
    m1 = np.uint64(0xBF58476D1CE4E5B9)
    m2 = np.uint64(0x94D049BB133111EB)
    for i in range(S):
        h = (h ^ rows[i]) * m1
        h ^= h >> np.uint64(31)
    h = h * m2
    h ^= h >> np.uint64(29)
    return h


class Codes:
    """Every class's bytecode, flat, for one job."""

    def __init__(self, classes, lib):
        parts, offs, self.depth = [], [0], 0
        for cl in classes:
            c, d = compile_items((cl.rep,), lib)
            parts.append(c)
            offs.append(offs[-1] + len(c))
            self.depth = max(self.depth, d)
        self.flat = np.concatenate(parts) if parts else np.zeros((0, 3), np.int32)
        self.off = np.array(offs, np.int64)


MODE = 2          # the kernel variant (_run_state): 2 was the fastest on Modal's hosts (SCALE.md §1)


def apply_classes(codes, cls, src_regs, m):
    return _apply_many(codes.flat, codes.off, np.asarray(cls, np.int64), np.ascontiguousarray(src_regs),
                       m.cap, m.limit, codes.depth, MODE)


def sig(m, tests, seq, lib):
    code, depth = compile_items(seq, lib)
    return int(_sig_kernel(code, tests.regs, m.cap, m.limit, depth))


def enumerate_items(tokens, lib, kinds, m, max_body=2, max_arity=3, seed=7, S_test=24, NP_test=6):
    """library/inventing/bfs.enumerate_items, line for line, with the kernel's signature."""
    rng = np.random.default_rng(seed)
    tests = brm.random_states(S_test, NP_test, rng)
    ident = sig(m, tests, (), lib)
    classes, by_sig = [], {ident: None}

    def add(item, c):
        s = sig(m, tests, (item,), lib)
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
                s = sig(m, tests, prefix, lib)
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


def _chunks(classes, levels, d):
    """bar3/search.bfs's (class, slice) loop in the same order, packed: (class indices, source level, lo, hi, whether
    the chunk ends its classes). A small source level takes several classes at once (up to SLICE outputs); a big one
    goes a slice at a time."""
    ci = 0
    while ci < len(classes):
        sl = d - classes[ci].cost
        if sl < 0 or levels[sl] is None:
            ci += 1
            continue
        S = levels[sl].S
        if S >= SLICE:
            for lo in range(0, S, SLICE):
                yield [ci], sl, lo, min(lo + SLICE, S), lo + SLICE >= S
            ci += 1
            continue
        group = [ci]
        while (ci + len(group) < len(classes) and classes[ci + len(group)].cost == classes[ci].cost
               and (len(group) + 1) * S <= SLICE):
            group.append(ci + len(group))
        yield group, sl, 0, S, True
        ci += len(group)


def bfs(prob: Problem, classes, lib, m, max_cost=10, max_states=1_000_000, time_limit=600.0, accept=None, limit=200,
        codes=None):
    """bar3/search.bfs with the compiled machine. The level logic is the original's; only the inner (class, slice)
    loop runs packed chunks through the kernel (see _chunks)."""
    t0 = time.time()
    codes = codes or Codes(classes, lib)
    levels = [prob.root]
    seen = np.sort(hash_rows(prob.root.features()))
    parents = [None]
    res = Result()
    res.level_sizes.append(1)
    res.best_fit.append(int(prob.fits(prob.root)[0]))
    if prob.goal(prob.root)[0]:
        # the start already fits the probes: the empty program, accepted only if right on every instance (the
        # original returned it unchecked, which at N <= 26, where the probes are a sample, crashed the learner)
        if accept is None or any(accept([()])):
            res.found_cost, res.goals, res.parents, res.levels = 0, [0], parents, levels
            return res
        res.aliases.append((0, [""]))
    for d in range(1, max_cost + 1):
        last = d == max_cost
        ck, cc, cs, ci_, chunks, gk = [], [], [], [], [], []
        n_cand = n_fresh = 0
        best = 0
        for group, sl, lo, hi, ends in _chunks(classes, levels, d):
            S = hi - lo
            regs, drop, keys = apply_classes(codes, group, levels[sl].regs[lo:hi], m)
            res.expanded += S * len(group)
            alive = np.nonzero(~drop)[0]
            if len(alive):
                keys = keys[alive]
                fm = _fresh(seen, keys)
                if fm.any():
                    idx = np.nonzero(fm)[0]
                    n_fresh += len(idx)
                    sub = brm.Batch(regs[alive[idx]], np.zeros(len(idx), bool))
                    f = prob.fits(sub)
                    best = max(best, int(f.max()))
                    g = f == len(prob.want)
                    if g.any():
                        gk.append(keys[idx][g])
                    flat = alive[idx]
                    if last:
                        if g.any():
                            idx, sub, flat = idx[g], sub.take(np.nonzero(g)[0]), flat[g]
                        else:
                            idx = idx[:0]
                    if len(idx):
                        ck.append(keys[idx])
                        cc.append(np.asarray(group, np.int32)[flat // S])
                        cs.append(np.full(len(idx), sl, np.int16))
                        ci_.append((flat % S).astype(np.int64) + lo)
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
            if ends and time.time() - t0 > time_limit:      # after a class, as the original checks
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
        res.best_fit.append(best)
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
    res.levels = levels
    return res


def bfs_multi(probs, classes, lib, m, max_cost=10, max_states=1_000_000, time_limit=600.0, accepts=None, limit=200,
              codes=None):
    """bfs for several problems posed on the same instances (the same root state): their searches build the same
    levels, and only the goal test differs, so one search serves them all. Each problem gets the Result its own bfs
    would give: its levels up to where it stops, its best fits, its goals, programs and aliases (a problem found at
    level d stops there, as alone; the others go on). The streamed level keeps every live problem's goal states, and
    a problem's goals are read back through the same parents, so its programs come out in the same order."""
    t0 = time.time()
    codes = codes or Codes(classes, lib)
    root = probs[0].root
    levels = [root]
    seen = np.sort(hash_rows(root.features()))
    parents = [None]
    out = [Result() for _ in probs]
    live = []
    for j, (pr, res) in enumerate(zip(probs, out)):
        res.level_sizes.append(1)
        res.best_fit.append(int(pr.fits(root)[0]))
        if pr.goal(root)[0] and (accepts is None or any(accepts[j]([()]))):      # as bfs
            res.found_cost, res.goals, res.parents, res.levels = 0, [0], parents, levels
        else:
            if pr.goal(root)[0]:
                res.aliases.append((0, [""]))
            live.append(j)
    stop = ""
    for d in range(1, max_cost + 1):
        if not live:
            break
        last = d == max_cost
        ck, cc, cs, ci_, chunks = [], [], [], [], []
        gk = {j: [] for j in live}
        best = {j: 0 for j in live}
        n_cand = n_fresh = 0

        def goal_any(b):
            g = np.zeros(b.S, bool)
            for j in live:
                g |= probs[j].goal(b)
            return g

        for group, sl, lo, hi, ends in _chunks(classes, levels, d):
            S = hi - lo
            regs, drop, keys = apply_classes(codes, group, levels[sl].regs[lo:hi], m)
            alive = np.nonzero(~drop)[0]
            if len(alive):
                keys = keys[alive]
                fm = _fresh(seen, keys)
                if fm.any():
                    idx = np.nonzero(fm)[0]
                    n_fresh += len(idx)
                    sub = brm.Batch(regs[alive[idx]], np.zeros(len(idx), bool))
                    g_any = np.zeros(len(idx), bool)
                    for j in live:
                        f = probs[j].fits(sub)
                        best[j] = max(best[j], int(f.max()))
                        g = f == len(probs[j].want)
                        if g.any():
                            gk[j].append(keys[idx][g])
                            g_any |= g
                    flat = alive[idx]
                    if last:
                        if g_any.any():
                            idx, sub, flat = idx[g_any], sub.take(np.nonzero(g_any)[0]), flat[g_any]
                        else:
                            idx = idx[:0]
                    if len(idx):
                        ck.append(keys[idx])
                        cc.append(np.asarray(group, np.int32)[flat // S])
                        cs.append(np.full(len(idx), sl, np.int16))
                        ci_.append((flat % S).astype(np.int64) + lo)
                        chunks.append(sub)
                        n_cand += len(idx)
                    if not last and n_cand > 1.5 * max_states:
                        last = True
                        stop = f"budget: level {d} streamed (over {max_states} states)"
                        keep = [(q, np.nonzero(goal_any(chunks[q]))[0]) for q in range(len(chunks))]
                        keep = [(q, gm) for q, gm in keep if len(gm)]
                        ck = [ck[q][gm] for q, gm in keep]
                        cc = [cc[q][gm] for q, gm in keep]
                        cs = [cs[q][gm] for q, gm in keep]
                        ci_ = [ci_[q][gm] for q, gm in keep]
                        chunks = [chunks[q].take(gm) for q, gm in keep]
            if ends and time.time() - t0 > time_limit:
                stop = f"time: {time_limit}s at level {d}"
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
            if not last and not stop:
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
        still = []
        for j in live:
            res = out[j]
            res.level_sizes.append(int(len(uk)) if not last else int(n_fresh))
            res.best_fit.append(best[j])
            res.stop = stop
            done = False
            if gk[j]:
                g = np.unique(np.concatenate(gk[j]))
                res.found_cost = d
                res.goals = rank[np.searchsorted(uk, g)].tolist()
                res.parents = parents
                if accepts is None:
                    done = True
                else:
                    progs = programs(res, classes, limit=limit)
                    ok = accepts[j](progs)
                    if any(ok):
                        res.accepted = [p for p, o in zip(progs, ok) if o]
                        done = True
                    else:
                        res.aliases.append((d, [dsl.show(p) for p in progs[:20]]))
                        res.found_cost, res.goals = None, None
            if not done and not stop:
                still.append(j)
        live = still
        if stop:
            break
        seen = np.union1d(seen, uk)
    for res in out:
        res.seconds = time.time() - t0
        if res.found_cost is None and not res.stop:
            res.stop = f"exhausted cost {max_cost}"
        res.parents = parents
        res.levels = levels
    return out
