"""Is the Triton kernel the CPU kernel? kernel.apply_chunk on a GPU against engine._apply_many (through
kernel._apply_chunk_cpu) on the same chunks: the drop flags equal, and every kept candidate's row and hash key equal.

Programs: random items over every instruction (INC/DEC/CLR/MOV on every register, L and Z blocks on every register,
nested to depth 8, bodies of 1-4 items), on random states with values from 0 to the cap (and some at 0 and the cap),
at several probe counts, caps and step limits (some low, so machines die); then real vocabularies: every class of the
35 primitives' item enumeration, and of the 34-routine library carried after 2x + 1, on random states of real sizes,
in chunks that start and end mid-class.

    cd cpu_first && <python with numba, torch and a GPU> -m bar3.gpu_search.kernel_check
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from library import dsl
from library.inventing import brm
from library.inventing.jobs import machine_for
from bar3.scale import engine as E
from bar3.gpu_search import kernel as K

ROOT = Path(__file__).resolve().parents[2]
REGS = brm.REGS


def rand_item(rng, depth, maxdep):
    if depth < maxdep and rng.random() < 0.45:
        kind = rng.choice(["L", "Z"]) + "." + rng.choice(REGS)
        body = tuple(rand_item(rng, depth + 1, maxdep) for _ in range(rng.integers(1, 5)))
        return ("REP", kind, body)
    op = rng.choice(["INC", "DEC", "CLR", "MOV"])
    if op == "MOV":
        a, b = rng.choice(5, 2, replace=False)
        return f"MOV.{REGS[a]}.{REGS[b]}"
    return f"{op}.{rng.choice(REGS)}"


def states(rng, S, NP, cap):
    regs = rng.integers(0, min(cap, 12) + 1, size=(S, NP, 5))
    big = rng.random((S, NP, 5)) < 0.1
    regs[big] = rng.integers(0, cap + 1, size=big.sum())
    regs[rng.random((S, NP, 5)) < 0.05] = cap
    regs[rng.random((S, NP, 5)) < 0.1] = 0
    return regs.astype(np.int32)


def rows_of(regs):
    S, NP, _ = regs.shape
    ROW = K.row_len(NP)
    r = np.zeros((S, ROW), np.int16)
    r[:, :NP * 5] = regs.reshape(S, -1).astype(np.int16)
    return r


def compare(classes, lib, regs, cap, limit, pieces, dev):
    """Every (c0, lo, n) piece of the (classes x states) candidates, both kernels. Returns (pieces checked, bad)."""
    codes = E.Codes(classes, lib)
    gc = K.GCodes(codes, dev)
    NP = regs.shape[1]
    src = rows_of(regs)
    sg, sc = torch.from_numpy(src).to(dev), torch.from_numpy(src)
    S = len(regs)
    bad = 0
    for c0, c1, lo, n in pieces:
        n = min(n, (c1 - c0) * S - lo)
        if n <= 0:
            continue
        og, kg, dg = (x.cpu().numpy().copy() for x in K.apply_chunk(gc, codes, c0, S, lo, n, sg, NP, cap, limit))
        oc, kc, dc = (x.numpy() for x in K._apply_chunk_cpu(codes, c0, S, lo, n, sc, NP, cap, limit))
        keep = ~dc
        ok = (dg == dc).all() and (og[keep] == oc[keep]).all() and (kg[keep] == kc[keep]).all()
        if not ok:
            bad += 1
            i = int(np.nonzero((dg != dc) | (keep & ((og != oc).any(1) | (kg != kc))))[0][0])
            print(f"  DIFF piece c0={c0} lo={lo} n={n}: first at {i}: drop {dg[i]} vs {dc[i]}; "
                  f"class {c0 + (lo + i) // S} {dsl.show((classes[c0 + (lo + i) // S].rep,))}", flush=True)
    return len(pieces), bad


def main():
    dev = torch.device("cuda")
    rng = np.random.default_rng(1)
    bad = total = 0
    t = time.time()
    # random programs
    for trial in range(40):
        maxdep = int(rng.integers(1, 9))
        items = [rand_item(rng, 0, maxdep) for _ in range(int(rng.integers(5, 60)))]
        classes = [E.ItemClass(it, 1, [it]) for it in items]
        NP = int(rng.choice([1, 3, 5, 21, 28, 32]))
        k = int(rng.choice([3, 5, 7, 26]))
        m = machine_for(k)
        limit = int(rng.choice([m.limit, 40, 200]))
        regs = states(rng, int(rng.integers(1, 300)), NP, m.cap)
        S = len(regs)
        tot = len(classes) * S
        pieces = [(0, len(classes), 0, tot)]
        for _ in range(3):
            lo = int(rng.integers(0, tot))
            pieces.append((0, len(classes), lo, int(rng.integers(1, tot - lo + 1))))
        c0 = int(rng.integers(0, len(classes)))
        pieces.append((c0, len(classes), 0, (len(classes) - c0) * S))
        n, b = compare(classes, {}, regs, m.cap, limit, pieces, dev)
        total += n
        bad += b
    print(f"random programs: {total} pieces, {bad} differing ({time.time() - t:.0f}s)", flush=True)
    # real vocabularies
    state = json.loads(open(ROOT / "bar3/breadth/records/seq_lp_clean_targets_b1.state.affine.json").read())["state"]
    for name, lib, kinds, k, NP, S in (("primitives k=7", {}, list(brm.LOOP_KINDS), 7, 28, 600),
                                       ("primitives k=26", {}, list(brm.LOOP_KINDS), 26, 32, 200),
                                       ("34-routine library k=7", state["lib"], state["kinds"], 7, 28, 60)):
        t = time.time()
        pl = {t_: dsl.parse(v) for t_, v in lib.items()}
        m = machine_for(k)
        tokens = list(brm.PRIMS) + [t_ for t_ in pl if t_ not in kinds]
        classes = E.enumerate_items(tokens, pl, kinds, m, max_body=2)
        regs = states(rng, S, NP, m.cap)
        regs[:, :, brm.N] = rng.integers(1, k + 1, size=(S, NP))
        tot = len(classes) * S
        pieces = [(0, len(classes), lo, 250_000) for lo in range(0, tot, 250_000)]
        pieces += [(int(c0), len(classes), int(rng.integers(0, S)), 9_999) for c0 in rng.integers(0, len(classes), 6)]
        n, b = compare(classes, pl, regs, m.cap, m.limit, pieces, dev)
        total += n
        bad += b
        print(f"{name}: {len(classes)} classes x {S} states, depth {K.GCodes(E.Codes(classes, pl), dev).depth}: "
              f"{n} pieces, {b} differing ({time.time() - t:.0f}s)", flush=True)
    print("PASS" if not bad else f"FAIL: {bad} of {total} pieces differ", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
