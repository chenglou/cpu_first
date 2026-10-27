"""Is the Triton kernel the CPU kernel on digit loops? kernel_check.py with bar3/scale/digits/machine.py's D.B block
installed: kernel.apply_chunk on a GPU against the digits study's compiled CPU kernel (machine._apply_many, which
engine._apply_many is once the block is installed) on the same chunks: the drop flags equal, and every kept
candidate's row and hash key equal.

- The packing: pack_all (all classes at once) against pack (row by row) on the random programs below.
- Random programs: items over every instruction and L, Z and D blocks (D in bases 1, 2, 3, 10), nested to depth 8
  (a D block nests as an L block does), bodies of 1-4 items, on random states with values from 0 to the cap, at
  several probe counts, at caps up to 32,000 (values of up to 15 binary digits) and step limits from 40 (machines die)
  to the digits study's 100,000.
- Real vocabularies: every class of the digits study's item enumeration (the primitives and D.1, D.2 or D.10, at
  k = 7 and 10, on the target's machines: 20,000 and 100,000 steps), and of a digits run's last library (27 routines,
  two with D.2 blocks; lp_clean_b_src_dxmod_dg1, round 3), in chunks that start and end mid-class.

    cd cpu_first && <python with numba, torch and a GPU> -m bar3.gpu_search.kernel_check_digits
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
from bar3.scale.digits import machine as M          # installs the D.B block (interpreter and CPU kernel)
from bar3.scale.digits import targets as DT
from bar3.gpu_search import kernel as K
from bar3.gpu_search.kernel_check import compare, states

ROOT = Path(__file__).resolve().parents[2]
REGS = brm.REGS
BASES = (1, 2, 3, 10)


def rand_item(rng, depth, maxdep):
    if depth < maxdep and rng.random() < 0.45:
        r = rng.random()
        kind = (f"D.{int(rng.choice(BASES))}" if r < 0.4 else
                rng.choice(["L", "Z"]) + "." + rng.choice(REGS))
        body = tuple(rand_item(rng, depth + 1, maxdep) for _ in range(rng.integers(1, 5)))
        return ("REP", kind, body)
    op = rng.choice(["INC", "DEC", "CLR", "MOV"])
    if op == "MOV":
        a, b = rng.choice(5, 2, replace=False)
        return f"MOV.{REGS[a]}.{REGS[b]}"
    return f"{op}.{rng.choice(REGS)}"


def check_packing(rng, n=400):
    """pack_all on many classes at once equals pack class by class (with the vocabulary's bases)."""
    bad = 0
    for _ in range(n // 20):
        items = [rand_item(rng, 0, int(rng.integers(1, 9))) for _ in range(20)]
        codes = E.Codes([E.ItemClass(it, 1, [it]) for it in items], {})
        flat, starts, deepest, bases = K.pack_all(codes.flat, codes.off)
        ref, d_ref = [], 0
        for c in range(len(items)):
            w, d, _ = K.pack(codes.flat[codes.off[c]:codes.off[c + 1]], bases)
            ref.append(w)
            d_ref = max(d_ref, d)
        ref = np.concatenate(ref)
        st = np.cumsum([0] + [len(codes.flat[codes.off[c]:codes.off[c + 1]]) + 1 for c in range(len(items))])[:-1]
        bad += not ((flat == ref).all() and (starts == st).all() and deepest == d_ref == codes.depth)
    return n // 20, bad


def main():
    dev = torch.device("cuda")
    rng = np.random.default_rng(2)
    t = time.time()
    n, bad = check_packing(rng)
    total = n
    print(f"packing: {n} vocabularies of 20 random classes, {bad} differing from pack ({time.time() - t:.0f}s)",
          flush=True)
    # random programs
    t = time.time()
    n_d = 0
    for trial in range(60):
        maxdep = int(rng.integers(1, 9))
        items = [rand_item(rng, 0, maxdep) for _ in range(int(rng.integers(5, 60)))]
        classes = [E.ItemClass(it, 1, [it]) for it in items]
        n_d += sum("D." in dsl.show((it,)) for it in items)
        NP = int(rng.choice([1, 3, 5, 21, 28, 32]))
        k = int(rng.choice([3, 5, 7, 10, 26]))
        m = machine_for(k)
        cap = int(rng.choice([m.cap, m.cap, 1000, 32000]))
        limit = int(rng.choice([m.limit, 40, 200, 20_000, 100_000]))
        regs = states(rng, int(rng.integers(1, 300)), NP, cap)
        S = len(regs)
        tot = len(classes) * S
        pieces = [(0, len(classes), 0, tot)]
        for _ in range(3):
            lo = int(rng.integers(0, tot))
            pieces.append((0, len(classes), lo, int(rng.integers(1, tot - lo + 1))))
        c0 = int(rng.integers(0, len(classes)))
        pieces.append((c0, len(classes), 0, (len(classes) - c0) * S))
        n, b = compare(classes, {}, regs, cap, limit, pieces, dev)
        total += n
        bad += b
    print(f"random programs: {total} pieces ({n_d} classes with a D block), {bad} differing ({time.time() - t:.0f}s)",
          flush=True)
    # real vocabularies
    rec = [json.loads(line) for line in open(ROOT / "bar3/scale/digits/records/lp_clean_b_src_dxmod_dg1.jsonl")]
    lib3 = rec[-1]["lib"]
    for name, lib, B, k, NP, S in (("primitives + D.2, k=7", {}, 2, 7, 28, 400),
                                   ("primitives + D.10, k=7", {}, 10, 7, 28, 400),
                                   ("primitives + D.1, k=7", {}, 1, 7, 28, 200),
                                   ("primitives + D.2, k=10", {}, 2, 10, 32, 200),
                                   ("27-routine digits library + D.2, k=7", lib3, 2, 7, 28, 60)):
        t = time.time()
        pl = {t_: dsl.parse(v) for t_, v in lib.items()}
        kinds = list(brm.LOOP_KINDS) + [M.KINDS[B]]
        m = DT.machine(k)
        tokens = list(brm.PRIMS) + [t_ for t_ in pl if t_ not in kinds]
        classes = E.enumerate_items(tokens, pl, kinds, machine_for(k), max_body=2)
        n_dc = sum(f"D.{B}[" in dsl.show((cl.rep,)) for cl in classes)
        regs = states(rng, S, NP, m.cap)
        regs[:, :, brm.N] = rng.integers(1, k + 1, size=(S, NP))
        regs[:, :, brm.X] = rng.integers(0, 10 * k, size=(S, NP))       # the target's rows: x < 10k
        tot = len(classes) * S
        pieces = [(0, len(classes), lo, 250_000) for lo in range(0, tot, 250_000)]
        pieces += [(int(c0), len(classes), int(rng.integers(0, S)), 9_999) for c0 in rng.integers(0, len(classes), 6)]
        n, b = compare(classes, pl, regs, m.cap, m.limit, pieces, dev)
        total += n
        bad += b
        gc = K.GCodes(E.Codes(classes, pl), dev)
        print(f"{name}: {len(classes)} classes ({n_dc} with a D block) x {S} states, depth {gc.depth}, bases "
              f"{gc.bases}: {n} pieces, {b} differing ({time.time() - t:.0f}s)", flush=True)
    print("PASS" if not bad else f"FAIL: {bad} of {total} pieces differ", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
