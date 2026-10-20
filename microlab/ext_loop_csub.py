"""Loop bodies over the bits of x with no fused modular add: how far does speculation reach?

ext_exact_loop.py found Blakley's binary modular multiplication as a 3-instruction body over {add, sub, mul, addm}, where
addm(a, b) = a + b - N[a + b >= N] fuses an add with the conditional subtract. Here the only reduction block is
    csub(a) = a - N [a >= N]          (unary: one conditional subtract, no fused add)
so the shortest correct body is 5 generic instructions (Blakley):
    t0=add(s,s); t1=mul(x,d); t2=add(t0,t1); t3=csub(t2); t4=csub(t3)
The loop scans the 24 bits of x MSB-first from s = 0 and applies s <- body(s, x, N, d); the registers are
[s, x, N, d, 0, 1, 2] plus earlier results. Everything is exact int64 (wraparound like a 64-bit ALU).

Search:
  1. Bodies are enumerated bottom-up. Partial bodies whose results agree on K random probe states (s < 2N, d in {0, 1},
     x < N, odd N of 12-24 bits) are merged. An instruction that recomputes an existing register, and a body with an
     unused result, is dropped: it is a shorter body, searched at a smaller L.
  2. A complete body is reduced to its function: its outputs on the probes. Bodies with the same function share one
     loop run (dedup within each work chunk; the global number of distinct functions is counted exactly when small and
     by HyperLogLog otherwise).
  3. The loop runs on the first labelled row for every function, and on the next rows only for those that pass.
Options: --brute merges nothing (every non-redundant body runs the loop on its own: the check that merging loses
nothing); --scan R also runs every function on all of the first R labelled rows, so survivor curves can be replayed
for random label orders; --isa addm re-runs ext_exact_loop's fused grammar with the same machinery, as the control.
Results: FINDINGS_loop_csub.md.

    python -m microlab.ext_loop_csub search --L 4 --workers 3
    python -m microlab.ext_loop_csub search --isa addm --L 3 --scan 32
    python -m microlab.ext_loop_csub certify --body "add,0,0;mul,1,3;add,7,8;csub,9,9;csub,10,10"
    ~/github/tips-for-running-modal/venv/bin/python -m modal run microlab/ext_loop_csub_modal.py --length 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "results_loop_csub.jsonl")

OPS = ("add", "sub", "mul", "csub", "addm")    # addm(a, b) = a + b - N[a + b >= N]: only for the comparison grammar
ISAS = {"csub": (0, 1, 2, 3), "addm": (0, 1, 2, 4)}  # "addm" is ext_exact_loop.py's grammar, re-run with this machinery
CSUB = ISAS["csub"]
R0 = 7  # s, x, N, d, 0, 1, 2
NAMES = ("s", "x", "N", "d", "0", "1", "2")
BITS = 24
BLAKLEY = "add,0,0;mul,1,3;add,7,8;csub,9,9;csub,10,10"


# ---------------------------------------------------------------- instructions and bodies

def instrs(R, isa=CSUB):
    """Every instruction over R registers: add/mul/addm on a <= b, sub on ordered pairs, csub on one operand (b = a)."""
    a2, b2 = np.triu_indices(R)
    a3, b3 = np.divmod(np.arange(R * R), R)
    a1 = np.arange(R)
    every = {0: (a2, b2), 1: (a3, b3), 2: (a2, b2), 3: (a1, a1), 4: (a2, b2)}
    return [(op, *every[op]) for op in isa]


def body_space(L, isa=CSUB):
    """Raw number of bodies of exactly L instructions (only the commutative a <= b symmetry removed)."""
    return math.prod(sum(len(a) for _, a, _ in instrs(R0 + i, isa)) for i in range(L))


def apply(op, A, B, N):
    if op == 0:
        return A + B
    if op == 1:
        return A - B
    if op == 2:
        return A * B
    if op == 4:
        A = A + B
    return np.where(A >= N, A - N, A)


def show(body):
    names = list(NAMES)
    out = []
    for i, (o, a, b) in enumerate(body):
        o = int(o)
        args = names[a] if o == 3 else f"{names[a]},{names[b]}"
        out.append(f"t{i}={OPS[o]}({args})")
        names.append(f"t{i}")
    return "; ".join(out)


def encode(body):
    return ";".join(f"{OPS[int(o)]},{int(a)},{int(b)}" for o, a, b in body)


def parse(text):
    return [(OPS.index(t.split(",")[0]), int(t.split(",")[1]), int(t.split(",")[2])) for t in text.split(";")]


# ---------------------------------------------------------------- hashing (uint64, wrapping)

_rng_h = np.random.default_rng(20261019)
_HK = (_rng_h.integers(1, 2 ** 62, size=512, dtype=np.int64) * 2 + 1).astype(np.uint64)
_USED = np.uint64(0x9E3779B97F4A7C15)


def _mix(z):
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(0xBF58476D1CE4E5B9)
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def vec_hash(v):
    """v: int64 [..., K] -> uint64 [...]: hash of the value vector (one register's values on the K probes)."""
    with np.errstate(all="ignore"):
        u = v.view(np.uint64)
        u = u ^ (u >> np.uint64(29))
        return _mix((u * _HK[:v.shape[-1]]).sum(-1, dtype=np.uint64))


def _state_key(H, U):
    """Multiset of (value hash, used-later flag) over a state's intermediates -> one uint64 per state."""
    if H.shape[1] == 0:
        return np.zeros(len(H), np.uint64)
    with np.errstate(all="ignore"):
        z = np.sort(H ^ (U.astype(np.uint64) * _USED), 1)
        return _mix((z * _HK[256:256 + z.shape[1]]).sum(1, dtype=np.uint64))


# ---------------------------------------------------------------- probe states and labelled rows

def make_probes(K=32, seed=1):
    """K probe states (s, x, N, d): odd N of 12-24 bits, x < N, s < 2N (two probes have s = 0), d alternating."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(12, 25, K)
    N = np.array([int(rng.integers(1 << (b - 2), 1 << (b - 1))) * 2 + 1 for b in bits], np.int64)
    x = (rng.random(K) * N).astype(np.int64)
    s = (rng.random(K) * 2 * N).astype(np.int64)
    s[:2] = 0
    d = np.arange(K, dtype=np.int64) % 2
    return np.stack([s, x, N, d, np.zeros(K, np.int64), np.ones(K, np.int64), np.full(K, 2, np.int64)])


def h1_label_rows(seed=0):
    """H1 T=1 training rows, in ext_exact_loop's order (permutation seed 0 puts the same row first)."""
    sys.path.insert(0, os.path.dirname(HERE))
    import h1
    d = h1.load("train")
    m = d["T"] == 1
    rows = {"N": d["N"][m], "x": d["X"][m], "y": d["Y"][m], "bits": d["bits"][m]}
    perm = np.random.default_rng(seed).permutation(len(rows["N"]))
    return {k: v[perm] for k, v in rows.items()}


def fresh_rows(n, bits, rng):
    """b-bit odd N in [2^(b-1), 2^b) (microlab's stand-in), x uniform in [0, N), y = x^2 mod N."""
    N = (rng.integers(1 << (bits - 2), 1 << (bits - 1), n) * 2 + 1).astype(np.int64)
    x = (rng.random(n) * N).astype(np.int64)
    return {"N": N, "x": x, "y": x * x % N}


# ---------------------------------------------------------------- bottom-up enumeration

def _root(probes):
    V = probes[None].copy()                               # [1, R0, K]
    H = vec_hash(probes)[None]                            # [1, R0]
    return V, H, np.zeros((1, 0), bool), np.zeros((1, 0, 3), np.int16), np.ones(1)


def expand(V, H, U, P, M, L, isa=CSUB):
    """Every one-instruction extension of depth-d states that is not redundant (equal to an existing register on all
    probes) and can still end without an unused result."""
    p, R, K = V.shape
    d = R - R0
    Nv = V[0, 2]
    left = L - d - 1                                      # instructions after this one
    jidx = R0 + np.arange(d)
    parts = []
    for op, A, B in instrs(R, isa):
        with np.errstate(all="ignore"):
            out = apply(op, V[:, A, :], V[:, B, :], Nv)   # [p, n, K]
        h = vec_hash(out)                                 # [p, n]
        dup = (h[:, :, None] == H[:, None, :]).any(-1)
        hit = (A[:, None] == jidx) | (B[:, None] == jidx)  # [n, d]
        Un = U[:, None, :] | hit[None]                    # [p, n, d]
        ok = ~dup & ((~Un).sum(-1) <= left)               # unused after this one: (~Un).sum + 1 <= left + 1
        si, pi = np.nonzero(ok)
        if len(si):
            ins = np.stack([np.full(len(si), op), A[pi], B[pi]], 1).astype(np.int16)
            parts.append((si, out[si, pi], h[si, pi], Un[si, pi], ins))
    if not parts:
        return (np.zeros((0, R + 1, K), np.int64), np.zeros((0, R + 1), np.uint64), np.zeros((0, d + 1), bool),
                np.zeros((0, d + 1, 3), np.int16), np.zeros(0))
    si = np.concatenate([q[0] for q in parts])
    V2 = np.concatenate([V[si], np.concatenate([q[1] for q in parts])[:, None, :]], 1)
    H2 = np.concatenate([H[si], np.concatenate([q[2] for q in parts])[:, None]], 1)
    U2 = np.concatenate([np.concatenate([q[3] for q in parts]), np.zeros((len(si), 1), bool)], 1)
    P2 = np.concatenate([P[si], np.concatenate([q[4] for q in parts])[:, None, :]], 1)
    return V2, H2, U2, P2, M[si]


def dedup(V, H, U, P, M):
    """Merge partial bodies whose intermediates are the same multiset of (probe values, used-later flag)."""
    if len(V) == 0:
        return V, H, U, P, M
    key = _state_key(H[:, R0:], U)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return V[first], H[first], U[first], P[first], np.bincount(inv.ravel(), weights=M, minlength=len(first))


def finals(V, H, U, P, M, isa=CSUB):
    """Every non-redundant last instruction that reads all still-unused results.
    Returns (function key, state index, instruction) per complete body."""
    p, R, K = V.shape
    d = R - R0
    Nv = V[0, 2] if p else None
    pat = (U.astype(np.int64) << np.arange(d)).sum(1) if d else np.zeros(p, np.int64)
    upat, pinv = np.unique(pat, return_inverse=True)
    keys, sis, inss = [], [], []
    for op, A, B in instrs(R, isa):
        table = np.ones((len(upat), len(A)), bool)
        for t, pv in enumerate(upat):
            for j in range(d):
                if not (pv >> j) & 1:
                    table[t] &= (A == R0 + j) | (B == R0 + j)
        si, pi = np.nonzero(table[pinv.ravel()])
        if len(si) == 0:
            continue
        with np.errstate(all="ignore"):
            out = apply(op, V[si, A[pi], :], V[si, B[pi], :], Nv)   # [m, K]
        h = vec_hash(out)
        keep = ~(h[:, None] == H[si]).any(-1)
        keys.append(h[keep])
        sis.append(si[keep])
        inss.append(np.stack([np.full(keep.sum(), op), A[pi][keep], B[pi][keep]], 1).astype(np.int16))
    if not keys:
        return np.zeros(0, np.uint64), np.zeros(0, np.int64), np.zeros((0, 3), np.int16)
    return np.concatenate(keys), np.concatenate(sis), np.concatenate(inss)


# ---------------------------------------------------------------- the loop

def loop_batch(progs, N, x, bits=BITS):
    """Final s of the loop on one row (N, x), for every body in progs [P, L, 3] at once."""
    P, L, _ = progs.shape
    regs = np.zeros((R0 + L, P), np.int64)
    regs[1], regs[2], regs[5], regs[6] = x, N, 1, 2
    flat = regs.reshape(-1)
    ar = np.arange(P, dtype=np.int64)
    ia = [progs[:, i, 1].astype(np.int64) * P + ar for i in range(L)]
    ib = [progs[:, i, 2].astype(np.int64) * P + ar for i in range(L)]
    is_op = [[progs[:, i, 0] == o for o in range(3)] for i in range(L)]
    addm = [progs[:, i, 0] == 4 if (progs[:, i, 0] == 4).any() else None for i in range(L)]
    with np.errstate(all="ignore"):
        for k in range(bits - 1, -1, -1):
            regs[3] = (int(x) >> k) & 1
            for i in range(L):
                A, B = flat.take(ia[i]), flat.take(ib[i])
                m0, m1, m2 = is_op[i]
                if addm[i] is not None:
                    A = np.where(addm[i], A + B, A)
                regs[R0 + i] = np.where(m0, A + B, np.where(m1, A - B, np.where(m2, A * B,
                                                                                np.where(A >= N, A - N, A))))
            regs[0] = regs[R0 + L - 1]
    return regs[0].copy()


def run_body_rows(body, N, x, bits=BITS):
    """One body, many rows (vectorised over rows): the loop's x*x mod N."""
    N = np.asarray(N, np.int64)
    x = np.asarray(x, np.int64)
    s = np.zeros_like(x)
    const = [np.zeros_like(x), np.ones_like(x), np.full_like(x, 2)]
    with np.errstate(all="ignore"):
        for k in range(bits - 1, -1, -1):
            regs = [s, x, N, (x >> k) & 1] + const
            for o, a, b in body:
                regs.append(apply(int(o), regs[a], regs[b], N))
            s = regs[-1]
    return s


# ---------------------------------------------------------------- HyperLogLog (distinct functions across chunks)

HLL_P = 16


def hll_add(reg, keys):
    idx = (keys >> np.uint64(64 - HLL_P)).astype(np.int64)
    rest = (keys & np.uint64((1 << (64 - HLL_P)) - 1)).astype(np.float64)
    bl = np.frexp(rest)[1]                                 # bit length (0 for rest = 0)
    rank = (64 - HLL_P + 1 - bl).astype(np.uint8)
    np.maximum.at(reg, idx, rank)


def hll_count(reg):
    m = len(reg)
    est = 0.7213 / (1 + 1.079 / m) * m * m / np.sum(2.0 ** -reg.astype(np.float64))
    zeros = int((reg == 0).sum())
    if est <= 2.5 * m and zeros:
        est = m * math.log(m / zeros)
    return float(est)


# ---------------------------------------------------------------- the search

def _score(progs, mult, labels):
    """Sequential labels: every body is run on label 0, survivors on label 1, ... Returns (passed[i] = number of
    leading labels body i reproduces, curve_raw[k] = raw bodies consistent with the first k labels)."""
    passed = np.zeros(len(progs), np.int16)
    curve = np.zeros(len(labels[0]) + 1)
    curve[0] = mult.sum()
    alive = np.arange(len(progs))
    for k, (N, x, y) in enumerate(zip(*labels)):
        if len(alive) == 0:
            break
        outs = [loop_batch(progs[alive[j:j + 65536]], int(N), int(x)) for j in range(0, len(alive), 65536)]
        alive = alive[np.concatenate(outs) == y]
        passed[alive] += 1
        curve[k + 1] = mult[alive].sum()
    return passed, curve


def _scan(progs, mult, scan, K_lab):
    """Every body on every one of R scan rows (no early exit): mask bit r = reproduces row r. The scan rows start with
    the K_lab sequential labels, so the sequential result is read off the mask."""
    mask = np.zeros(len(progs), np.int64)
    for r, (N, x, y) in enumerate(zip(*scan)):
        out = np.concatenate([loop_batch(progs[j:j + 65536], int(N), int(x)) for j in range(0, len(progs), 65536)])
        mask |= (out == y).astype(np.int64) << r
    lead = np.zeros(len(progs), np.int16)
    alive = np.ones(len(progs), bool)
    curve = np.zeros(K_lab + 1)
    curve[0] = mult.sum()
    for k in range(K_lab):
        alive &= ((mask >> k) & 1).astype(bool)
        lead += alive
        curve[k + 1] = mult[alive].sum()
    return mask, lead, curve


def values(P, probes):
    """Recompute every register of a batch of partial bodies on the probes (deep levels are stored as programs only)."""
    n, d, _ = P.shape
    K = probes.shape[1]
    V = np.empty((n, R0 + d, K), np.int64)
    V[:, :R0] = probes
    Nv = probes[2]
    ar = np.arange(n)
    with np.errstate(all="ignore"):
        for i in range(d):
            A, B = V[ar, P[:, i, 1]], V[ar, P[:, i, 2]]
            op = P[:, i, 0][:, None]
            A = np.where(op == 4, A + B, A)
            V[:, R0 + i] = np.where(op == 0, A + B, np.where(op == 1, A - B, np.where(op == 2, A * B,
                                                                                      np.where(A >= Nv, A - Nv, A))))
    return V, vec_hash(V)


def _grow(args):
    """Worker: one level deeper for a chunk of states given as programs (chunk-local merge); returns programs + keys."""
    P, U, M, L, probes, merge, isa = args
    V, H = values(P, probes)
    V, H, U, P, M = expand(V, H, U, P, M, L, isa)
    if merge:
        V, H, U, P, M = dedup(V, H, U, P, M)
    return _state_key(H[:, R0:], U), P, U, M


def _last_two(args):
    """Worker: expand a chunk of depth L-2 states (chunk-local merge), enumerate the last instruction, merge bodies by
    function, run the loop on the labels. Returns survivors of the first label and the counts. merge=False runs every
    body on its own (no merging of partial bodies or of functions): the brute-force check."""
    V, H, U, P, M, L, labels, sub, want_keys, merge, probes, scan, isa = args
    t0 = time.time()
    if V is None:
        V, H = values(P, probes)
    if L >= 2:
        V, H, U, P, M = expand(V, H, U, P, M, L, isa)
        if merge:
            V, H, U, P, M = dedup(V, H, U, P, M)
    n_states = len(V)
    keys, reps, mults = [], [], []
    for s in range(0, len(V), sub):
        k, si, ins = finals(V[s:s + sub], H[s:s + sub], U[s:s + sub], P[s:s + sub], M[s:s + sub], isa)
        keys.append(k)
        reps.append((si + s, ins))
        mults.append(M[s:s + sub][si])
    key = np.concatenate(keys) if keys else np.zeros(0, np.uint64)
    si = np.concatenate([r[0] for r in reps]) if reps else np.zeros(0, np.int64)
    ins = np.concatenate([r[1] for r in reps]) if reps else np.zeros((0, 3), np.int16)
    mult = np.concatenate(mults) if mults else np.zeros(0)
    uk = np.unique(key)
    n_cand = len(key)
    if merge:
        key, first, inv = np.unique(key, return_index=True, return_inverse=True)
        mult = np.bincount(inv.ravel(), weights=mult, minlength=len(key))
        si, ins = si[first], ins[first]
    progs = np.concatenate([P[si], ins[:, None, :]], 1)
    t1 = time.time()
    if scan is None:
        passed, curve = _score(progs, mult, labels)
        sc = None
    else:
        mask, passed, curve = _scan(progs, mult, scan, len(labels[0]))
        hit = mask != 0
        sc = {"keys": key[hit], "progs": progs[hit], "mult": mult[hit], "mask": mask[hit]}
    hll = np.zeros(1 << HLL_P, np.uint8)
    hll_add(hll, uk)
    surv = passed > 0
    return {"curve": curve, "raw": float(mult.sum()), "bodies": n_cand, "loop_runs": len(progs),
            "states": n_states, "functions": len(uk), "hll": hll, "keys": uk if want_keys else None,
            "surv_keys": key[surv], "surv_progs": progs[surv], "surv_mult": mult[surv], "surv_passed": passed[surv],
            "scan": sc, "t_enum": t1 - t0, "t_loop": time.time() - t1}


def speculate(L, labels, K=32, probe_seed=1, chunk=256, sub=4096, workers=1, exact_count=True, verbose=True,
              job_slice=None, merge=True, serial_depth=3, grow_chunk=1024, progress=None, scan=None, isa=CSUB):
    """Every body of exactly L instructions (non-redundant, no unused result), merged by function on the probes; the
    loop is run on the labelled rows (N, x, y arrays; the first row for all, later rows for survivors).
    Depths 1..serial_depth are built in this process with their probe values; deeper stored levels (up to L-2) are
    built by the workers and kept as programs only (values are recomputed per chunk), merged globally by state key."""
    probes = make_probes(K, probe_seed)
    labels = tuple(np.asarray(v, np.int64) for v in labels)
    if scan is not None:
        scan = tuple(np.asarray(v, np.int64) for v in scan)
        assert len(scan[0]) <= 63 and all((scan[i][:len(labels[0])] == labels[i]).all() for i in range(3))
    t0 = time.time()
    V, H, U, P, M = _root(probes)
    level = []
    ex = None
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(workers)
    pmap = (lambda f, jobs: ex.map(f, jobs, chunksize=1)) if ex else (lambda f, jobs: map(f, jobs))
    try:
        for d in range(max(L - 2, 0)):
            if d + 1 <= serial_depth:
                parts = [expand(V[s:s + chunk], H[s:s + chunk], U[s:s + chunk], P[s:s + chunk], M[s:s + chunk], L,
                                isa) for s in range(0, len(V), chunk)]
                cat = tuple(np.concatenate([q[i] for q in parts]) for i in range(5))
                V, H, U, P, M = dedup(*cat) if merge else cat
                n_partial = len(cat[0])
            else:
                jobs = [(P[s:s + grow_chunk], U[s:s + grow_chunk], M[s:s + grow_chunk], L, probes, merge, isa)
                        for s in range(0, len(P), grow_chunk)]
                res = list(pmap(_grow, jobs))
                key = np.concatenate([r[0] for r in res])
                P, U, M = (np.concatenate([r[i] for r in res]) for i in (1, 2, 3))
                n_partial = len(key)
                if merge:
                    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
                    P, U, M = P[first], U[first], np.bincount(inv.ravel(), weights=M, minlength=len(first))
                V = H = None
                del res, key
            level.append({"depth": d + 1, "partial": n_partial, "states": len(P), "raw": float(M.sum())})
            if verbose:
                print(f"    depth {d + 1}: {n_partial:,} partial bodies -> {len(P):,} "
                      f"{'distinct states' if merge else 'bodies (no merging)'} ({time.time() - t0:.1f}s)", flush=True)
        cut = lambda a, s: None if a is None else a[s:s + chunk]
        jobs = [(cut(V, s), cut(H, s), U[s:s + chunk], P[s:s + chunk], M[s:s + chunk], L, labels, sub, exact_count,
                 merge, probes, scan, isa) for s in range(0, len(P), chunk)]
        n_jobs_total = len(jobs)
        if job_slice is not None:                           # timing a sample of the jobs (to extrapolate)
            jobs = jobs[job_slice[0]:job_slice[1]]
        t_mat = time.time() - t0
        acc = {"curve": np.zeros(len(labels[0]) + 1), "raw": 0.0, "bodies": 0, "loop_runs": 0, "states": 0,
               "functions": 0, "t_enum": 0.0, "t_loop": 0.0}
        hll = np.zeros(1 << HLL_P, np.uint8)
        keys, sk, sp, sm, spass, scans = [], [], [], [], [], []
        done, last = 0, time.time()
        for r in pmap(_last_two, jobs):
            for k in ("curve", "raw", "bodies", "loop_runs", "states", "functions", "t_enum", "t_loop"):
                acc[k] = acc[k] + r[k]
            np.maximum(hll, r["hll"], out=hll)
            if r["keys"] is not None:
                keys.append(r["keys"])
            sk.append(r["surv_keys"]); sp.append(r["surv_progs"]); sm.append(r["surv_mult"])
            spass.append(r["surv_passed"])
            if r["scan"] is not None:
                scans.append(r["scan"])
            done += 1
            if verbose and (done % max(1, len(jobs) // 20) == 0 or done == len(jobs) or time.time() - last > 60):
                el = time.time() - t0
                line = (f"    jobs {done}/{len(jobs)}: {acc['bodies']:,} candidates, {acc['loop_runs']:,} loop "
                        f"runs, label-1 survivors {int(acc['curve'][1])}, {el:.1f}s "
                        f"(eta {(el - t_mat) / done * (len(jobs) - done):.0f}s)")
                print(line, flush=True)
                if progress:
                    progress(line)
                last = time.time()
    finally:
        if ex:
            ex.shutdown()
    secs = time.time() - t0
    # survivors of the first label, merged by function across chunks (raw bodies kept too when not merging)
    sk = np.concatenate(sk) if sk else np.zeros(0, np.uint64)
    sp = np.concatenate(sp) if sp else np.zeros((0, L, 3), np.int16)
    sm = np.concatenate(sm) if sm else np.zeros(0)
    spass = np.concatenate(spass) if spass else np.zeros(0, np.int16)
    uk, first, inv = np.unique(sk, return_index=True, return_inverse=True)
    surv = {"keys": uk, "progs": sp[first], "mult": np.bincount(inv.ravel(), weights=sm, minlength=len(uk)),
            "passed": spass[first]}
    n_exact = int(len(np.unique(np.concatenate(keys)))) if keys else None
    curve_fn = [n_exact if n_exact is not None else hll_count(hll)] + \
               [int((surv["passed"] >= k).sum()) for k in range(1, len(labels[0]) + 1)]
    out = {"L": L, "K": K, "isa": [OPS[o] for o in isa], "merge": merge, "space": body_space(L, isa), "levels": level,
           "states_last": acc["states"],
           "bodies": acc["bodies"], "raw": acc["raw"], "functions_chunk_local": acc["functions"],
           "loop_runs": acc["loop_runs"],
           "functions_exact": n_exact, "functions_hll": hll_count(hll), "curve_raw": acc["curve"].tolist(),
           "curve_fn": curve_fn, "survivors": surv, "seconds": secs, "t_materialise": t_mat,
           "t_enum_cpu": acc["t_enum"], "t_loop_cpu": acc["t_loop"], "jobs": len(jobs), "jobs_total": n_jobs_total}
    if not merge:
        out["raw_survivors"] = {"keys": sk, "progs": sp, "mult": sm, "passed": spass}
    if scan is not None:                              # functions reproducing at least one scan row, merged by key
        k_ = np.concatenate([q["keys"] for q in scans]) if scans else np.zeros(0, np.uint64)
        uk, first, inv = np.unique(k_, return_index=True, return_inverse=True)
        cat = {f: np.concatenate([q[f] for q in scans]) if scans else np.zeros(0) for f in ("progs", "mult", "mask")}
        out["scan"] = {"keys": uk, "progs": cat["progs"][first], "mask": cat["mask"][first],
                       "mult": np.bincount(inv.ravel(), weights=cat["mult"], minlength=len(uk)), "R": len(scan[0])}
    return out


# ---------------------------------------------------------------- checking survivors, certification

def check_survivors(surv, rows, held, n_small=200):
    """Every surviving function: how many of the labelled rows it reproduces in order (all 1,000 = fits the training
    set), and its accuracy on fresh rows at each size (2,000 rows per size if it fits the training set, else 200)."""
    progs = surv["progs"]
    rows_passed, _ = _score(progs, np.ones(len(progs)), (rows["N"], rows["x"], rows["y"]))
    out = []
    for i in range(len(progs)):
        body = [tuple(int(v) for v in t) for t in progs[i]]
        train = int(rows_passed[i]) == len(rows["N"])
        n = None if train else n_small
        acc = {b: float((run_body_rows(body, r["N"][:n], r["x"][:n]) == r["y"][:n]).mean()) for b, r in held.items()}
        out.append({"body": encode(body), "show": show(body), "mult": float(surv["mult"][i]),
                    "passed": int(surv["passed"][i]), "rows_passed": int(rows_passed[i]), "train_all": train,
                    "fresh": acc, "correct": train and all(a == 1.0 for a in acc.values())})
    return out


def order_stats(sc, rows, held, n_orders=2000, seed=0, ks=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32)):
    """Label-order robustness from a scan: each function's pass mask over the R scan rows, its pass rate on all
    labelled rows, and whether it is correct. Over random orders of the R rows: how many functions are left after k
    labels, and after how many labels only correct ones are left. E[left after k] = sum_f p_f^k uses the pass rates on
    all labelled rows (k labels drawn independently)."""
    R = int(sc["R"])
    F = len(sc["keys"])
    rate, correct, info = np.zeros(F), np.zeros(F, bool), []
    for i in range(F):
        body = [tuple(int(v) for v in t) for t in sc["progs"][i]]
        ok = run_body_rows(body, rows["N"], rows["x"]) == rows["y"]
        rate[i] = ok.mean()
        if ok.all():
            correct[i] = all((run_body_rows(body, r["N"], r["x"]) == r["y"]).all() for r in held.values())
        info.append({"show": show(body), "body": encode(body), "mult": float(sc["mult"][i]), "rate": float(rate[i]),
                     "correct": bool(correct[i]), "scan_hits": int(bin(int(sc["mask"][i])).count("1"))})
    rng = np.random.default_rng(seed)
    left = np.zeros((n_orders, R), np.int64)
    wrong = np.zeros((n_orders, R), np.int64)
    masks = sc["mask"].astype(np.int64)
    for o in range(n_orders):
        order = rng.permutation(R)
        pm = np.cumsum(np.int64(1) << order.astype(np.int64))              # prefix masks (distinct bits: sum = or)
        alive = (masks[None, :] & pm[:, None]) == pm[:, None]               # [R, F]
        left[o] = alive.sum(1)
        wrong[o] = (alive & ~correct[None]).sum(1)
    need = np.where((wrong == 0).any(1), (wrong == 0).argmax(1) + 1, R + 1)  # labels until only correct ones are left
    tab = []
    for k in ks:
        if k > R:
            continue
        lk = left[:, k - 1]
        tab.append({"k": k, "median": float(np.median(lk)), "p90": float(np.percentile(lk, 90)), "max": int(lk.max()),
                    "only_correct": float((wrong[:, k - 1] == 0).mean()),
                    "expected_iid": float((rate ** k).sum())})
    info.sort(key=lambda c: (-c["rate"], -c["mult"]))
    return {"R": R, "functions_hit": F, "functions_correct": int(correct.sum()), "orders": n_orders, "table": tab,
            "labels_needed": {"median": float(np.median(need)), "p90": float(np.percentile(need, 90)),
                              "p99": float(np.percentile(need, 99)), "max": int(need.max()),
                              "over_R": float((need > R).mean())},
            "expected_iid_more": {str(k): float((rate ** k).sum()) for k in (48, 64, 100, 200)},
            "functions": info[:60]}


def held_out(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    return {b: fresh_rows(n, b, rng) for b in (13, 16, 20, 24)}


def survivor_table(checked, K_lab, n_rows=1000):
    """Per k: distinct functions and raw bodies consistent with the first k labels, and how many are correct
    (k = n_rows: the whole labelled training set)."""
    tab = []
    for k in list(range(1, K_lab + 1)) + [n_rows]:
        alive = [c for c in checked if c["rows_passed"] >= k]
        tab.append({"k": k, "functions": len(alive), "raw": sum(c["mult"] for c in alive),
                    "correct_functions": sum(c["correct"] for c in alive),
                    "correct_raw": sum(c["mult"] for c in alive if c["correct"])})
    return tab


def certify_body(body, verbose=True):
    """Loop the found squaring (the loop over the bits of s) T times on both H1 depth profiles."""
    sys.path.insert(0, os.path.dirname(HERE))
    import h1

    def predict(N, X, T):
        s = X.astype(np.int64) % N
        for t in range(int(T.max())):
            s = np.where(T > t, run_body_rows(body, N, s), s)
        return h1.digits(s)

    return {prof: h1.certify(predict, prof, verbose=verbose)["max_t"] for prof in ("depth_t", "depth_ood_n_t")}


def report(res, rows, K_lab, certify=True, verbose=True):
    """Check every survivor of the first label on all labelled rows and fresh rows; certify the best correct one."""
    t0 = time.time()
    checked = check_survivors(res["survivors"], rows, held_out())
    checked.sort(key=lambda c: (-c["rows_passed"], not c["correct"], -c["mult"]))
    tab = survivor_table(checked, K_lab)
    res = dict(res)
    res.pop("survivors")
    res["survivor_table"] = tab
    res["checked"] = checked[:200]
    res["n_checked"] = len(checked)
    if "scan" in res:
        res["order_stats"] = order_stats(res.pop("scan"), rows, held_out())
        if verbose:
            o = res["order_stats"]
            print(f"  scan of {o['R']} rows: {o['functions_hit']} functions reproduce at least one, "
                  f"{o['functions_correct']} correct; over {o['orders']} random orders:", flush=True)
            for t in o["table"]:
                print(f"    after {t['k']:2d} labels: left median {t['median']:.0f}, p90 {t['p90']:.0f}, "
                      f"max {t['max']}; only correct ones left in {t['only_correct']:.1%} of orders; "
                      f"E[left] for iid labels {t['expected_iid']:.2f}", flush=True)
            print(f"    labels until only correct ones are left: {o['labels_needed']}", flush=True)
            for c in o["functions"][:10]:
                print(f"      rate {c['rate']:.3f} x{c['mult']:<5.0f} {'CORRECT' if c['correct'] else '       '} "
                      f"{c['show']}", flush=True)
    if "raw_survivors" in res:                        # brute force: every surviving raw body checked on its own
        raw = check_survivors(res.pop("raw_survivors"), rows, held_out())
        raw.sort(key=lambda c: (-c["rows_passed"], not c["correct"]))
        res["raw_table"] = survivor_table(raw, K_lab)
        res["checked_raw"] = raw[:500]
    if verbose:
        f = res["functions_exact"]
        print(f"  L={res['L']}{'' if res['merge'] else ' (brute force, nothing merged)'}: {res['space']:,} raw "
              f"bodies (2^{math.log2(res['space']):.1f}); "
              f"{res['raw']:,.0f} non-redundant; distinct functions on {res['K']} probes: "
              + (f"{f:,} (exact)" if f is not None else f"~{res['functions_hll']:,.0f} (HyperLogLog)")
              + f"; {res['bodies']:,} (state, last instruction) candidates, {res['loop_runs']:,} loop runs; "
              f"{res['seconds']:.1f}s wall "
              f"(cpu: enumerate {res['t_enum_cpu']:.1f}s, loop {res['t_loop_cpu']:.1f}s)", flush=True)
        print("  consistent with the first k labels (raw bodies): "
              + ", ".join(f"k={k}:{c:,.0f}" for k, c in enumerate(res["curve_raw"])), flush=True)
        print("  distinct functions: "
              + ", ".join(f"k={k}:{c:,.0f}" for k, c in enumerate(res["curve_fn"])), flush=True)
        for t in tab:
            print(f"    after {t['k']} labels: {t['functions']} functions ({t['raw']:,.0f} bodies), "
                  f"{t['correct_functions']} correct on fresh 13/16/20/24-bit rows "
                  f"({t['correct_raw']:,.0f} bodies)", flush=True)
        for c in checked[:12]:
            tag = "CORRECT" if c["correct"] else ("train-all" if c["train_all"] else f"rows {c['rows_passed']}")
            print(f"    {tag:10s} x{c['mult']:<6.0f} {c['show']}  fresh24={c['fresh'][24]:.3f}", flush=True)
        for c in res.get("checked_raw", [])[:40]:
            tag = "CORRECT" if c["correct"] else ("train-all" if c["train_all"] else f"rows {c['rows_passed']}")
            print(f"    raw body {tag:10s} {c['show']}  fresh24={c['fresh'][24]:.3f}", flush=True)
    good = [c for c in checked if c["correct"]]
    if certify and good:
        best = max(good, key=lambda c: c["mult"])
        cert = certify_body(parse(best["body"]), verbose=verbose)
        res["certified"] = {"body": best["show"], **cert}
        if verbose:
            print(f"  certified {best['show']}: {cert}", flush=True)
    res["t_check"] = time.time() - t0
    return res


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def log(rec):
    with open(LOG, "a") as fh:
        fh.write(json.dumps(_jsonable(rec)) + "\n")


def cmd_search(a):
    rows = h1_label_rows(a.seed)
    labels = (rows["N"][:a.K_lab], rows["x"][:a.K_lab], rows["y"][:a.K_lab])
    for L in range(a.Lmin, a.L + 1):
        isa = ISAS[a.isa]
        print(f"[{a.isa} loop, L={L}] {body_space(L, isa):,} raw bodies (2^{math.log2(body_space(L, isa)):.1f}), "
              f"K={a.K} probes, {a.workers} workers", flush=True)
        scan = (rows["N"][:a.scan], rows["x"][:a.scan], rows["y"][:a.scan]) if a.scan else None
        res = speculate(L, labels, K=a.K, probe_seed=a.probe_seed, chunk=a.chunk, sub=a.sub, workers=a.workers,
                        exact_count=not a.hll, merge=not a.brute, serial_depth=a.serial_depth, scan=scan, isa=isa)
        res = report(res, rows, a.K_lab, certify=not a.no_certify)
        log({"exp": "loop_csub", "where": "local", "isa_name": a.isa, "workers": a.workers, "probe_seed": a.probe_seed,
             "label_seed": a.seed, "K_lab": a.K_lab, **res})


def cmd_certify(a):
    body = parse(a.body)
    print("certifying", show(body), flush=True)
    print(certify_body(body))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--L", type=int, default=4)
    s.add_argument("--isa", default="csub", choices=sorted(ISAS))
    s.add_argument("--Lmin", type=int, default=None)
    s.add_argument("--K", type=int, default=32, help="probe states for merging bodies by function")
    s.add_argument("--K_lab", type=int, default=8, help="labelled rows used inside the search")
    s.add_argument("--seed", type=int, default=0, help="permutation of the H1 rows (0 = ext_exact_loop's order)")
    s.add_argument("--probe_seed", type=int, default=1)
    s.add_argument("--chunk", type=int, default=256)
    s.add_argument("--sub", type=int, default=4096)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--hll", action="store_true", help="count distinct functions by HyperLogLog only")
    s.add_argument("--no_certify", action="store_true")
    s.add_argument("--brute", action="store_true", help="merge nothing: run the loop on every body on its own")
    s.add_argument("--scan", type=int, default=0,
                   help="also run every function on the first R labelled rows (R <= 63) for label-order statistics")
    s.add_argument("--serial_depth", type=int, default=3,
                   help="deepest level built in the main process; deeper stored levels are built by the workers")
    c = sub.add_parser("certify")
    c.add_argument("--body", default=BLAKLEY)
    a = p.parse_args()
    if a.cmd == "search" and a.Lmin is None:
        a.Lmin = a.L
    {"search": cmd_search, "certify": cmd_certify}[a.cmd](a)


if __name__ == "__main__":
    main()
