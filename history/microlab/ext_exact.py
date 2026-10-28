"""Exact arithmetic as fixed blocks; only the control is learned. How little must be learned for x^2 mod N?

The blocks are whole-number operations on int64 registers, exact (two's-complement wraparound on overflow, like a
64-bit ALU; floor division; x/0 = 0 and x%0 = x as in RISC-V). The learned part is a straight-line program: L
instructions, each an (op, a, b) over the registers [N, x, 0, 1, 2, earlier results]; the answer is the last result.

Learners:
  spec   speculation: enumerate every program of length L (exact, bottom-up, observational-equivalence dedup of
         partial programs on K probe rows), keep the ones that reproduce the labelled rows.
  soft   soft selection that hardens: per-slot softmax over ops and operands, values mixed (DARTS-style), Adam,
         temperature annealed, argmax at the end. Also the output-level mixture over complete programs and a
         REINFORCE sampler (see ext_exact_soft.py).

    python -m microlab.ext_exact spec --isa basic --target sq --L 4
    python -m microlab.ext_exact certify            # loop the found step T times on the H1 depth profiles
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
sys.path.insert(0, os.path.dirname(HERE))  # cpu_first/, for h1.py

LOG = os.path.join(HERE, "results_exact.jsonl")


# ---------------------------------------------------------------- exact blocks (int64, numpy)

def _div(a, b):
    with np.errstate(all="ignore"):
        return np.where(b == 0, 0, a // np.where(b == 0, 1, b))


def _mod(a, b):
    with np.errstate(all="ignore"):
        return np.where(b == 0, a, a % np.where(b == 0, 1, b))


def _shl(a, b):
    return np.left_shift(a, b & 63)


def _shr(a, b):
    return np.right_shift(a, b & 63)


OPS = {  # name -> (fn, commutative)
    "add": (np.add, True),
    "sub": (np.subtract, False),
    "mul": (np.multiply, True),
    "div": (_div, False),
    "mod": (_mod, False),
    "min": (np.minimum, True),
    "max": (np.maximum, True),
    "lt": (lambda a, b: (a < b).astype(np.int64), False),
    "and": (np.bitwise_and, True),
    "or": (np.bitwise_or, True),
    "xor": (np.bitwise_xor, True),
    "shl": (_shl, False),
    "shr": (_shr, False),
}

ISAS = {
    "tiny": ("mul", "mod"),
    "ring": ("add", "sub", "mul"),
    "basic": ("add", "sub", "mul", "div", "mod"),
    "nomod": ("add", "sub", "mul", "div"),
    "full": ("add", "sub", "mul", "div", "mod", "min", "max", "lt", "and", "or", "xor", "shl", "shr"),
    "full_nomod": ("add", "sub", "mul", "div", "min", "max", "lt", "and", "or", "xor", "shl", "shr"),
}


def run_prog(prog, inputs, consts=(0, 1, 2)):
    """prog: list of (opname, a, b). inputs: list of int64 arrays (N, x, ...)."""
    regs = [np.asarray(v, np.int64) for v in inputs]
    regs += [np.full_like(regs[0], c) for c in consts]
    with np.errstate(all="ignore"):
        for o, a, b in prog:
            regs.append(OPS[o][0](regs[a], regs[b]))
    return regs[-1]


def show(prog, names):
    names = list(names)
    out = []
    for i, (o, a, b) in enumerate(prog):
        out.append(f"t{i}={o}({names[a]},{names[b]})")
        names.append(f"t{i}")
    return "; ".join(out)


# ---------------------------------------------------------------- data

def free_rows(n, bits, rng, T=1, power=None):
    """b-bit odd N (microlab's stand-in), x uniform in [0, N), y = x^(2^T) mod N."""
    from .tasks import _odd_moduli
    N = _odd_moduli(n, bits, rng)
    x = (rng.random(n) * N).astype(np.int64)
    y = x.copy()
    for _ in range(T):
        y = y * y % N
    return {"N": N, "x": x, "y": y}


def h1_rows(T=1, split="train"):
    import h1
    d = h1.load(split)
    m = d["T"] == T
    return {"N": d["N"][m], "x": d["X"][m], "y": d["Y"][m], "bits": d["bits"][m]}


def reverse_digits(v):
    return np.array([int(str(int(a))[::-1]) for a in v], np.int64)


def add_readings(rows):
    """The prompt writes N's digits LSB-first and x's MSB-first. A fixed exact atoi that reads every string the same
    way sees N and reverse(x); the other way it sees reverse(N) and x. Offer both readings of both."""
    rows = dict(rows)
    rows["Nr"], rows["xr"] = reverse_digits(rows["N"]), reverse_digits(rows["x"])
    return rows


def target_fn(name):
    """Reference semantics of each target, for checking survivors on held-out rows."""
    def sq(N, x):
        return x * x % N
    def p4(N, x):
        s = x * x % N
        return s * s % N
    def p8(N, x):
        s = x
        for _ in range(3):
            s = s * s % N
        return s
    def sqp1(N, x):
        return (x * x + 1) % N
    return {"sq": sq, "p4": p4, "p8": p8, "sqp1": sqp1}[name]


# ---------------------------------------------------------------- speculation

_rng_h = np.random.default_rng(12345)
_HASH = _rng_h.integers(1, 2 ** 62, size=64, dtype=np.int64) * 2 + 1


def _reg_hash(v, used):
    """v: [..., K] int64 values, used: [...] bool -> [...] int64 hash (wrapping arithmetic)."""
    K = v.shape[-1]
    with np.errstate(all="ignore"):
        h = (v * _HASH[:K]).sum(-1)
        h = h * _HASH[K] + used.astype(np.int64) * _HASH[K + 1]
        h ^= (h >> 29)
        h = h * _HASH[K + 2]
    return h


def _pairs(R, comm):
    a, b = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
    a, b = a.ravel(), b.ravel()
    if comm:
        keep = a <= b
        a, b = a[keep], b[keep]
    return a, b


def space_size(isa, L, R0):
    """Raw number of straight-line programs (only the commutative a<=b symmetry removed)."""
    tot = 1
    for d in range(L):
        R = R0 + d
        tot *= sum(R * (R + 1) // 2 if OPS[o][1] else R * R for o in isa)
    return tot


def _expand(V, U, P, M, d, L, isa, R0):
    """All one-instruction extensions of a chunk of depth-d states that are not redundant (value equal to an existing
    register) and can still end without dead code."""
    R = R0 + d
    new = ([], [], [], [])
    for oi, o in enumerate(isa):
        fn, comm = OPS[o]
        A, B = _pairs(R, comm)
        with np.errstate(all="ignore"):
            out = fn(V[:, A, :], V[:, B, :])                       # [p, pairs, K]
        dup = (out[:, :, None, :] == V[:, None, :, :]).all(-1).any(-1)
        Unew = np.repeat(U[:, None, :], len(A), 1)                 # [p, pairs, d]
        for j in range(d):
            Unew[:, :, j] |= ((A == R0 + j) | (B == R0 + j))[None]
        n_unused = (~Unew).sum(-1) + 1
        ok = ~dup & (n_unused <= (L - d - 1) + 1)
        si, pi = np.nonzero(ok)
        if len(si) == 0:
            continue
        new[0].append(np.concatenate([V[si], out[si, pi][:, None, :]], 1))
        new[1].append(np.concatenate([Unew[si, pi], np.zeros((len(si), 1), bool)], 1))
        ins = np.stack([np.full(len(si), oi), A[pi], B[pi]], 1).astype(np.int16)[:, None, :]
        new[2].append(np.concatenate([P[si], ins], 1))
        new[3].append(M[si])
    if not new[0]:
        K = V.shape[2]
        return (np.zeros((0, R + 1, K), np.int64), np.zeros((0, d + 1), bool), np.zeros((0, d + 1, 3), np.int16),
                np.zeros(0))
    return tuple(np.concatenate(x) for x in new)


def _dedup(vals, used, prog, mult, R0):
    """Merge partial programs whose intermediates are the same multiset of (probe values, used-later flag)."""
    if len(vals) == 0:
        return vals, used, prog, mult
    h = np.sort(_reg_hash(vals[:, R0:, :], used), 1)
    with np.errstate(all="ignore"):
        key = (h * _HASH[40:40 + h.shape[1]]).sum(1)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    m2 = np.bincount(inv.ravel(), weights=mult, minlength=len(first))
    return vals[first], used[first], prog[first], m2


def _final(V, U, P, M, d, isa, R0, y):
    """Evaluate every last instruction on a chunk of depth-d states. Probe 0 first (almost everything fails it), the
    other probes only for the few that pass. Returns (curve[k] = raw programs matching the first k probes, survivors)."""
    K = len(y)
    R = R0 + d
    curve = np.zeros(K + 1)
    survivors = []
    n_unused = (~U).sum(1)
    for oi, o in enumerate(isa):
        fn, comm = OPS[o]
        A, B = _pairs(R, comm)
        if d > 0:
            # dead code: every still-unused intermediate must be an operand of the last instruction; this depends
            # only on the state's unused pattern, so build one mask per pattern
            pat = (U.astype(np.int64) << np.arange(d)).sum(1)
            upat, pinv = np.unique(pat, return_inverse=True)
            table = np.zeros((len(upat), len(A)), bool)
            for t, pv in enumerate(upat):
                need = [R0 + j for j in range(d) if not (pv >> j) & 1]
                cov = sum(((A == r) | (B == r)).astype(np.int64) for r in need) if need else 0
                table[t] = cov == len(need)
            ok = table[pinv.ravel()]
        else:
            ok = np.ones((len(V), len(A)), bool)
        curve[0] += (M * ok.sum(1)).sum()
        with np.errstate(all="ignore"):
            out0 = fn(V[:, A, 0], V[:, B, 0])                       # [p, pairs]
        si, pi = np.nonzero(ok & (out0 == y[0]))
        if len(si) == 0:
            continue
        with np.errstate(all="ignore"):
            out = fn(V[si, A[pi], :], V[si, B[pi], :])              # [m, K]
        lead = np.cumprod(out == y[None], axis=1).sum(1)
        for k in range(1, K + 1):
            curve[k] += M[si][lead >= k].sum()
        for s, p in zip(si[lead == K], pi[lead == K]):
            pr = [(isa[int(q[0])], int(q[1]), int(q[2])) for q in P[s]] + [(o, int(A[p]), int(B[p]))]
            survivors.append((pr, float(M[s])))
    return curve, survivors


def _last_two(args):
    """Worker: expand a chunk of depth L-2 states by one instruction (local dedup), then evaluate the last one."""
    V, U, P, M, d, L, isa, R0, y = args
    V2, U2, P2, M2 = _dedup(*_expand(V, U, P, M, d, L, isa, R0), R0)
    curve, surv = _final(V2, U2, P2, M2, d + 1, isa, R0, y)
    return curve, surv, len(V2)


def speculate(rows, isa, L, K=8, consts=(0, 1, 2), chunk=2048, verbose=True, workers=1, inputs=("N", "x")):
    """Every program of exactly L instructions, bottom-up with dedup of equivalent partial programs on the K probe
    rows. Returns survivors (programs matching all K probes, with how many raw programs each represents), the
    label curve (non-redundant programs consistent with the first k probes, k = 0..K) and timing."""
    isa = tuple(isa)
    base = [rows[k][:K] for k in inputs] + [np.full(K, c, np.int64) for c in consts]
    base = np.stack(base).astype(np.int64)
    y = rows["y"][:K].astype(np.int64)
    R0 = base.shape[0]
    vals, used = base[None], np.zeros((1, 0), bool)
    prog, mult = np.zeros((1, 0, 3), np.int16), np.ones(1, np.float64)
    t0 = time.time()
    level_states = []
    # materialise depths 1 .. L-2 with global dedup
    for d in range(max(L - 2, 0)):
        parts = [_expand(vals[s:s + chunk], used[s:s + chunk], prog[s:s + chunk], mult[s:s + chunk], d, L, isa, R0)
                 for s in range(0, len(vals), chunk)]
        cat = tuple(np.concatenate([p[i] for p in parts]) for i in range(4))
        raw = len(cat[0])
        vals, used, prog, mult = _dedup(*cat, R0)
        level_states.append(len(vals))
        if verbose:
            print(f"    depth {d + 1}: {raw:,} partial programs -> {len(vals):,} distinct states "
                  f"({time.time() - t0:.1f}s)", flush=True)
    curve, survivors = np.zeros(K + 1), []
    if L == 1:
        c, s = _final(vals, used, prog, mult, 0, isa, R0, y)
        curve += c; survivors += s
    else:
        d = L - 2
        jobs = [(vals[s:s + chunk], used[s:s + chunk], prog[s:s + chunk], mult[s:s + chunk], d, L, isa, R0, y)
                for s in range(0, len(vals), chunk)]
        n_states = 0
        if workers > 1 and len(jobs) > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers) as ex:
                results = ex.map(_last_two, jobs, chunksize=1)
                for i, (c, s, n) in enumerate(results):
                    curve += c; survivors += s; n_states += n
        else:
            for job in jobs:
                c, s, n = _last_two(job)
                curve += c; survivors += s; n_states += n
        level_states.append(n_states)
        if verbose:
            print(f"    depth {L - 1}: {n_states:,} states (chunk-local dedup), last instruction evaluated "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return {"survivors": survivors, "curve": curve.tolist(), "states": level_states,
            "seconds": time.time() - t0, "space": space_size(isa, L, R0)}


def check_survivors(survivors, target, held, consts=(0, 1, 2), inputs=("N", "x")):
    """Group survivors by their behaviour on held-out rows; mark which equal the target there."""
    ref = target_fn(target)
    groups = {}
    for pr, m in survivors:
        sig_parts, correct = [], True
        for name, r in held.items():
            out = run_prog(pr, [r[k] for k in inputs], consts)
            good = out == ref(r["N"], r["x"])
            sig_parts.append(float(good.mean()))
            correct &= bool(good.all())
        out24 = run_prog(pr, [held["24"][k] for k in inputs], consts) if "24" in held else None
        key = (correct, tuple(round(s, 4) for s in sig_parts),
               None if out24 is None else hash(out24[:64].tobytes()))
        g = groups.setdefault(key, {"correct": correct, "acc": sig_parts, "raw": 0.0, "n": 0, "example": pr})
        g["raw"] += m
        g["n"] += 1
    return sorted(groups.values(), key=lambda g: -g["raw"])


def held_out_sets(rng, n=2000, bits_list=(13, 16, 20, 24), T=1):
    return {str(b): free_rows(n, b, rng, T=T) for b in bits_list}


def cmd_spec(a):
    isa = ISAS[a.isa]
    T = {"sq": 1, "p4": 2, "p8": 3, "sqp1": 1}[a.target]
    if a.data == "h1" and a.target in ("sq", "p4"):
        rows = h1_rows(T=T)
        perm = np.random.default_rng(a.seed).permutation(len(rows["N"]))
        rows = {k: v[perm] for k, v in rows.items()}
    else:
        rows = free_rows(1000, a.bits, np.random.default_rng(a.seed), T=T)
        if a.target == "sqp1":
            rows["y"] = (rows["x"] * rows["x"] + 1) % rows["N"]
    inputs = ("N", "x") if a.io == "plain" else ("Nr", "N", "xr", "x")
    if a.io == "prompt":
        rows = add_readings(rows)
    names = list(inputs) + ["0", "1", "2"]
    held = held_out_sets(np.random.default_rng(999), T=T)
    if a.target == "sqp1":
        for r in held.values():
            r["y"] = (r["x"] * r["x"] + 1) % r["N"]
    for L in range(a.Lmin, a.L + 1):
        sp = space_size(isa, L, len(inputs) + 3)
        print(f"[{a.isa} isa={len(isa)} ops, target {a.target}, L={L}] raw program space {sp:,} "
              f"(2^{math.log2(sp):.1f})", flush=True)
        res = speculate(rows, isa, L, K=a.K, verbose=True, workers=a.workers, inputs=inputs)
        surv = res["survivors"]
        # every labelled row, then held-out
        full = [(pr, m) for pr, m in surv
                if (run_prog(pr, [rows[k] for k in inputs]) == rows["y"]).all()]
        if a.io == "prompt":
            held = {k: add_readings(v) for k, v in held.items()}
        groups = check_survivors(full, a.target, held, inputs=inputs) if full else []
        curve = res["curve"]
        print(f"  {res['seconds']:.1f}s; raw programs consistent with first k labels: "
              + ", ".join(f"k={k}:{curve[k]:.3g}" for k in (0, 1, 2, 3, 4, a.K) if k <= a.K), flush=True)
        print(f"  survivors on {a.K} probes: {len(surv)} states; on all {len(rows['N'])} labelled rows: "
              f"{len(full)} states ({sum(m for _, m in full):.0f} raw programs) in {len(groups)} behaviours",
              flush=True)
        for g in groups[:6]:
            print(f"    {'CORRECT' if g['correct'] else 'alias  '} raw={g['raw']:.0f} states={g['n']} "
                  f"held-out acc {[round(x, 4) for x in g['acc']]}  e.g. {show(g['example'], names)}")
        rec = {"exp": "spec", "isa": a.isa, "n_ops": len(isa), "target": a.target, "data": a.data, "L": L,
               "K": a.K, "space": sp, "log2_space": round(math.log2(sp), 2), "states": res["states"],
               "seconds": round(res["seconds"], 2), "curve": curve,
               "survivors_probe": len(surv), "survivors_all_rows": len(full),
               "raw_all_rows": sum(m for _, m in full),
               "groups": [{"correct": g["correct"], "raw": g["raw"], "n": g["n"], "acc": g["acc"],
                           "example": show(g["example"], names)} for g in groups[:20]],
               "seed": a.seed}
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if groups and any(g["correct"] for g in groups) and not a.all_L:
            break


# ---------------------------------------------------------------- certification of the learned step

def certify_step(prog, consts=(0, 1, 2), verbose=True):
    """Loop the learned step s <- prog(N, s) T times and certify on both H1 depth profiles."""
    import h1

    def predict(N, X, T):
        s = X.astype(np.int64) % N
        out = s.copy()
        for t in range(int(T.max())):
            nxt = run_prog(prog, [N, out], consts)
            out = np.where(T > t, nxt, out)
        return h1.digits(out)

    res = {}
    for prof in ("depth_t", "depth_ood_n_t"):
        res[prof] = h1.certify(predict, prof, verbose=verbose)
    return res


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec")
    s.add_argument("--isa", default="basic", choices=sorted(ISAS))
    s.add_argument("--target", default="sq", choices=["sq", "p4", "p8", "sqp1"])
    s.add_argument("--L", type=int, default=4, help="search lengths 1..L (stops at the first L with a correct one)")
    s.add_argument("--K", type=int, default=8, help="probe rows used inside the search")
    s.add_argument("--data", default="h1", choices=["h1", "free"])
    s.add_argument("--bits", type=int, default=16)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--all_L", action="store_true")
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--io", default="plain", choices=["plain", "prompt"],
                   help="prompt: both LSB/MSB readings of N's and x's digit strings are offered as inputs")
    s.add_argument("--Lmin", type=int, default=1)
    c = sub.add_parser("certify")
    c.add_argument("--prog", default="mul,1,1;mod,5,0", help="op,a,b;... over registers N,x,0,1,2,t0,...")
    a = p.parse_args()
    if a.cmd == "spec":
        cmd_spec(a)
    elif a.cmd == "certify":
        prog = [(t.split(",")[0], int(t.split(",")[1]), int(t.split(",")[2])) for t in a.prog.split(";")]
        print("certifying", show(prog, ["N", "s", "0", "1", "2"]))
        r = certify_step(prog)
        print({k: v["max_t"] for k, v in r.items()})


if __name__ == "__main__":
    main()
