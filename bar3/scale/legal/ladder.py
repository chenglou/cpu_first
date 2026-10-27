"""A round from a record of offline.py, checked as the round is (compose3/verify3.py's ladder): H1's two depth ladders
(the seen moduli, depth_t_<T>; the unseen ones, depth_ood_n_t_<T>; T = 1 to 64, 768 rows a rung), the round applied T
times at full size by the machine's exact cells, against each row's own answer, read. Local, on the CPU, in seconds.

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 ~/github/sotaku/venv/bin/python -m bar3.scale.legal.ladder RECORD.json ARM
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

from integration.lib import libcore as L
from bar3.scale.legal.offline import H1_DIR
from bar3.scale.legal.rowtarget import P, digits

RUNGS = (1, 2, 4, 8, 16, 32, 64)


def main():
    rec = json.load(open(sys.argv[1]))
    arm = rec["arms"][sys.argv[2]]
    lib = {k: L.parse(v) for k, v in arm["lib"].items()}
    prog = L.parse(arm["q3"]["round"])
    st = L.round_settings(L.Cells(10, "cpu"), P)
    print(f"{sys.argv[1]} [{sys.argv[2]}]: {arm['q3']['round']}", flush=True)
    for name in ("depth_t", "depth_ood_n_t"):
        for T in RUNGS:
            t0 = time.time()
            rows = [json.loads(line) for line in open(os.path.join(H1_DIR, f"{name}_{T}.jsonl"))]
            X, Nd, Y = (digits(rows, k) for k in ("x", "modulus", "result"))
            for _ in range(T):
                X, _c = L.run_prog(st, prog, lib, {"X": X, "N": Nd}, L.gen_on("cpu", 1), "cpu")
            ok = (X == Y).all(1)
            print(f"  {name}_{T}: {int(ok.sum())}/{len(ok)} right ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
