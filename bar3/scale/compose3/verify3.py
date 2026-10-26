"""A found program checked as the round is (COMPOSE3.md §5), for any TARGET_FN: H1's two depth ladders' (x, N) (the
seen moduli, depth_t_<T>, and the unseen ones, depth_ood_n_t_<T>; T = 1 to 64, 768 rows a rung), the program applied
T times at full size by the machine's exact cells, against the target function applied T times by Python's integers.
At x2 that is the ladders' own answers (round 1's verify.py ladder).

    cd cpu_first && TARGET_FN=x3 ~/github/sotaku/venv/bin/python -m bar3.scale.compose3.verify3 RECORD.json "PROGRAM"
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

from integration.lib import libcore as L
from bar3.scale.compose3 import hooks as H

H1_DIR = os.environ.get("H1_DATA", os.path.expanduser(
    "~/github/one-layer/one-layer-deeper/data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
P = 16
RUNGS = (1, 2, 4, 8, 16, 32, 64)


def num(vals):
    return torch.tensor([[int(ch) for ch in reversed(str(v))] + [0] * (P - len(str(v))) for v in vals])


def main():
    rec = json.load(open(sys.argv[1]))
    lib = {k: L.parse(v) for k, v in rec["lib"].items()}
    prog = L.parse(sys.argv[2])
    st = L.round_settings(L.Cells(10, "cpu"), P)
    print(f"{sys.argv[1]}: {sys.argv[2]} (TARGET_FN {H.TARGET_FN})", flush=True)
    for name in ("depth_t", "depth_ood_n_t"):
        for T in RUNGS:
            t0 = time.time()
            rows = [json.loads(l) for l in open(os.path.join(H1_DIR, f"{name}_{T}.jsonl"))]
            xs, Ns = [int(r["x"]) for r in rows], [int(r["modulus"]) for r in rows]
            want = []
            for x, N in zip(xs, Ns):
                for _ in range(T):
                    x = H.TARGET_F(x, N)
                want.append(x)
            if H.TARGET_FN == "x2":
                assert want == [int(r["result"]) for r in rows]
            X = num(xs)
            for _ in range(T):
                X, _c = L.run_prog(st, prog, lib, {"X": X, "N": num(Ns)}, L.gen_on("cpu", 1), "cpu")
            ok = (X == num(want)).all(1)
            print(f"  {name}_{T}: {int(ok.sum())}/{len(ok)} right ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
