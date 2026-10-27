"""Checks after the run, not pre-registered (LEGAL_TEST.md §5), local on the CPU (about a minute, under 250 MB):

1. **The spellings.** Q2 accepted four spellings of the reduction on the 400 answer rows. Each one as the rest of the
   round (x², STA.X, the spelling) on both of H1's ladders, where the failures fall (N's bits and digits), and the key
   the climb's own choose_routine sorts them by: (keeps its inputs, registers touched, size, steps executed).
2. **An identity check.** Each spelling as a reduction on instances whose answer is known by construction,
   red(r + q N') = r, with N' the training moduli as they are, and moved up one digit (N' = 10 N + d: 8 digits from 7,
   a digit move). The instances are built here with Python's integers, only to ask whether such a check tells the
   spellings apart; in a run the book's own constructions would build them (the digit move, the learner's addition).

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 ~/github/sotaku/venv/bin/python -m bar3.scale.legal.posthoc RECORD.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter

import torch

from integration.lib import libcore as L
from bar3.scale.legal.offline import H1_DIR, X2
from bar3.scale.legal.rowtarget import P, RowTarget, answer_rows, digits, h1_rows
from bar3.scale.slice.spcore_s import side_effects

RUNGS = (1, 2, 4, 8, 16, 32, 64)


def num(vals):
    return torch.tensor([[int(c) for c in reversed(str(v))] + [0] * (P - len(str(v))) for v in vals])


def main():
    rec = json.load(open(sys.argv[1]))
    arm = rec["arms"]["full"]
    lib = {k: L.parse(v) for k, v in arm["lib"].items()}
    found = next(r for r in arm["q2"]["runs"] if r["found_cost"] is not None)
    spellings = list(dict.fromkeys(found["programs"]))
    st = L.round_settings(L.Cells(10, "cpu"), P)
    print("1. the accepted spellings, in the order bfs_roots lists them:", spellings)
    rt = RowTarget(answer_rows(H1_DIR, 400, 1), "cpu")
    root, _ = rt.rest_root(rt.value(L.parse(X2), lib), torch.arange(8))
    for s in spellings:
        p = L.parse(s)
        keeps, touched = side_effects(p, lib, st, root, ("X", "N"))
        print(f"  choose_routine's key, {s}: keeps its inputs {keeps}, registers touched {touched}, "
              f"size {L.cost(L.expand(p, lib))}, steps {L.steps(p, lib)}")
    for s in spellings:
        prog = L.parse(X2) + ("STA.X",) + L.parse(s)
        out = []
        for name in ("depth_t", "depth_ood_n_t"):
            for T in RUNGS:
                rows = [json.loads(line) for line in open(os.path.join(H1_DIR, f"{name}_{T}.jsonl"))]
                X, Nd, Y = (digits(rows, k) for k in ("x", "modulus", "result"))
                for _ in range(T):
                    X, _c = L.run_prog(st, prog, lib, {"X": X, "N": Nd}, L.gen_on("cpu", 1), "cpu")
                ok = (X == Y).all(1)
                out.append(int(ok.sum()))
                if T == 1 and not bool(ok.all()):
                    bad = [rows[i] for i in torch.nonzero(~ok).flatten().tolist()]
                    print(f"  {s} on {name}_1, wrong rows: N's bits {dict(Counter(r['modulus_bits'] for r in bad))}, "
                          f"N's digits {dict(Counter(len(str(r['modulus'])) for r in bad))}")
        print(f"  ladders, {s}: seen moduli {out[:7]}, unseen {out[7:]} (of 768)")
    print("2. the identity check, red(r + q N') = r, 1,000 instances from the T = 1 rows' moduli:")
    rows, rng = h1_rows(H1_DIR), random.Random(3)
    for widen in (0, 1):
        Ns = [r["modulus"] * 10 ** widen + (rng.randrange(10) if widen else 0) for r in rows]
        rs = [rng.randrange(N) for N in Ns]
        qs = [rng.randrange(min(N, 10 ** 15 // N)) for N in Ns]
        Xs = [r + q * N for r, q, N in zip(rs, qs, Ns)]
        for s in spellings:
            A, _ = L.run_prog(st, L.parse(s), lib, {"X": num(Xs), "N": num(Ns)}, L.gen_on("cpu", 1), "cpu")
            ok = (A == num(rs)).all(1)
            print(f"  N' {'moved up one digit' if widen else 'as in the data'}, {s}: {int(ok.sum())}/{len(ok)} right")


if __name__ == "__main__":
    main()
