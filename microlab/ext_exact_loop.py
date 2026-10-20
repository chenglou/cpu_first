"""Small exact blocks + a loop: which representation gives the shortest learnable control?

Without a whole-number remainder block, x*x mod N needs a loop. The loop scans the digits of x in some base from the
top (MSB first) and applies one learned body per digit: s <- body(s, x, N, d), s = 0 at the start, answer = s. The
body is a straight-line program over registers [s, x, N, d, 0, 1, 2] and exact small blocks:
    add, sub, mul, and addm(a, b) = a + b - N [a + b >= N]   (one conditional subtract: the cmov step)
Speculation enumerates every body of L instructions and runs the whole loop for each, on a few labelled rows.

In base 2 Blakley's interleaved multiplication is a 3-instruction body (s <- 2s + d x, reduced twice by addm); in
base 10 the digit product d*x reaches 9N and 10 s reaches 10N, which one conditional subtract cannot reduce.

    python -m microlab.ext_exact_loop search --base 2 --L 3
    python -m microlab.ext_exact_loop search --base 10 --L 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import numpy as np

from . import ext_exact as E

OPS = ("add", "sub", "mul", "addm")
COMM = {"add": True, "sub": False, "mul": True, "addm": True}
R0 = 7  # s, x, N, d, 0, 1, 2


def n_digits(base, bits=24):
    return math.ceil(bits / math.log2(base)) if base > 2 else bits


def body_space(L):
    tot = 1
    for i in range(L):
        R = R0 + i
        tot *= sum(R * (R + 1) // 2 if COMM[o] else R * R for o in OPS)
    return tot


def enumerate_bodies(L):
    """All bodies of exactly L instructions (commutative a<=b), last instruction must read the one before it."""
    per_slot = []
    for i in range(L):
        R = R0 + i
        slot = []
        for oi, o in enumerate(OPS):
            for a in range(R):
                for b in range(R):
                    if COMM[o] and b < a:
                        continue
                    slot.append((oi, a, b))
        per_slot.append(np.array(slot, np.int16))
    return per_slot


def run_loop(prog_arr, N, x, base, width):
    """prog_arr: [P, L, 3] int16 (op, a, b). Runs the loop for one row (scalars N, x) on every program at once."""
    P, L, _ = prog_arr.shape
    s = np.zeros(P, np.int64)
    digs = []
    v = int(x)
    for _ in range(width):
        digs.append(v % base)
        v //= base
    with np.errstate(all="ignore"):
        for d in reversed(digs):                           # most significant digit first
            regs = [s, np.full(P, x, np.int64), np.full(P, N, np.int64), np.full(P, d, np.int64),
                    np.zeros(P, np.int64), np.ones(P, np.int64), np.full(P, 2, np.int64)]
            for i in range(L):
                R = np.stack(regs)                         # [R, P]
                A = R[prog_arr[:, i, 1], np.arange(P)]
                B = R[prog_arr[:, i, 2], np.arange(P)]
                op = prog_arr[:, i, 0]
                ssum = A + B
                out = np.where(op == 0, ssum,
                      np.where(op == 1, A - B,
                      np.where(op == 2, A * B, ssum - N * (ssum >= N))))
                regs.append(out)
            s = regs[-1]
    return s


def cmd_search(a):
    width = n_digits(a.base)
    rows = E.h1_rows(T=1)
    perm = np.random.default_rng(a.seed).permutation(len(rows["N"]))
    rows = {k: v[perm] for k, v in rows.items()}
    space = body_space(a.L)
    print(f"[loop base {a.base}, {width} digits per x, body L={a.L}] {space:,} bodies (2^{math.log2(space):.1f})",
          flush=True)
    t0 = time.time()
    slots = enumerate_bodies(a.L)
    # every combination, streamed in blocks over the first slot
    survivors, evaluated = [], 0
    rest = list(itertools.product(*[range(len(s)) for s in slots[1:]]))
    rest = np.array(rest, np.int64) if rest else np.zeros((1, 0), np.int64)
    curve = np.zeros(a.K + 1)
    for i0 in range(len(slots[0])):
        P = len(rest)
        prog = np.zeros((P, a.L, 3), np.int16)
        prog[:, 0] = slots[0][i0]
        for j in range(1, a.L):
            prog[:, j] = slots[j][rest[:, j - 1]]
        # the last instruction must read the one before it (else the body is a shorter one in disguise)
        if a.L > 1:
            last = R0 + a.L - 2
            keep = (prog[:, -1, 1] == last) | (prog[:, -1, 2] == last)
            prog = prog[keep]
        evaluated += len(prog)
        curve[0] += len(prog)
        alive = prog
        for k in range(a.K):
            if len(alive) == 0:
                break
            out = run_loop(alive, int(rows["N"][k]), int(rows["x"][k]), a.base, width)
            alive = alive[out == rows["y"][k]]
            curve[k + 1] += len(alive)
        survivors += [p.copy() for p in alive]
    secs = time.time() - t0
    names = ["s", "x", "N", "d", "0", "1", "2"]
    show = lambda p: "; ".join(f"t{i}={OPS[o]}({(names + [f't{j}' for j in range(i)])[x_]},"
                                f"{(names + [f't{j}' for j in range(i)])[y_]})" for i, (o, x_, y_) in enumerate(p))
    # verify on all 1000 rows and on fresh 24-bit moduli
    held = E.free_rows(300, 24, np.random.default_rng(7))
    final = []
    for p in survivors:
        P1 = p[None]
        ok_all = all(run_loop(P1, int(n), int(x_), a.base, width)[0] == y for n, x_, y in
                     zip(rows["N"][:200], rows["x"][:200], rows["y"][:200]))
        ok24 = ok_all and all(run_loop(P1, int(n), int(x_), a.base, width)[0] == y for n, x_, y in
                              zip(held["N"], held["x"], held["y"]))
        final.append((p, ok_all, ok24))
    print(f"  {evaluated:,} bodies evaluated in {secs:.1f}s; consistent with the first k labels: "
          + ", ".join(f"k={k}:{int(c):,}" for k, c in enumerate(curve)), flush=True)
    for p, ok_all, ok24 in final[:12]:
        print(f"    {'CORRECT' if ok24 else ('train-only' if ok_all else 'probe-only')}  {show(p)}")
    rec = {"exp": "loop", "base": a.base, "L": a.L, "space": space, "evaluated": evaluated,
           "seconds": round(secs, 1), "curve": curve.tolist(),
           "survivors": [{"body": show(p), "train200": ok_all, "fresh24": ok24} for p, ok_all, ok24 in final]}
    with open(E.LOG, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return final


def run_body_rows(body, N, x, base, width):
    """One body, many rows (vectorised over rows): the loop's x*x mod N."""
    N = np.asarray(N, np.int64)
    x = np.asarray(x, np.int64)
    s = np.zeros_like(x)
    digs, v = [], x.copy()
    for _ in range(width):
        digs.append(v % base)
        v = v // base
    with np.errstate(all="ignore"):
        for d in reversed(digs):
            regs = [s, x, N, d, np.zeros_like(x), np.ones_like(x), np.full_like(x, 2)]
            for o, i, j in body:
                A, B = regs[i], regs[j]
                op = OPS[o] if not isinstance(o, str) else o
                if op == "add":
                    out = A + B
                elif op == "sub":
                    out = A - B
                elif op == "mul":
                    out = A * B
                else:
                    t = A + B
                    out = t - N * (t >= N)
                regs.append(out)
            s = regs[-1]
    return s


def cmd_certify(a):
    """Loop the found squaring (the loop over x's digits) T times on the H1 depth profiles."""
    import h1
    names = ["s", "x", "N", "d", "0", "1", "2"]
    body = [tuple(int(v) if i else OPS.index(v) for i, v in enumerate(t.split(","))) for t in a.body.split(";")]
    prog = np.array(body, np.int16)[None]
    width = n_digits(a.base)

    def square(N, s):
        return run_body_rows(body, N, s, a.base, width)

    def predict(N, X, T):
        s = X.astype(np.int64) % N
        for t in range(int(T.max())):
            s = np.where(T > t, square(N, s), s)
        return h1.digits(s)

    for prof in ("depth_t", "depth_ood_n_t"):
        r = h1.certify(predict, prof, verbose=True)
        print(prof, "max certified T =", r["max_t"], flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--base", type=int, default=2)
    s.add_argument("--L", type=int, default=3)
    s.add_argument("--K", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("certify")
    c.add_argument("--base", type=int, default=2)
    c.add_argument("--body", default="addm,0,0;mul,3,1;addm,7,8")
    a = p.parse_args()
    {"search": cmd_search, "certify": cmd_certify}[a.cmd](a)


if __name__ == "__main__":
    main()
