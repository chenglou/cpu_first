"""FAMILY.md §5: each found round on inputs the ladders do not hold: one round at full size on every modulus of H1's
ladders with x drawn uniformly below it (x = 0 included), against the task's function, by the machine's exact cells.

    cd cpu_first && TARGET_FN=rho ~/github/sotaku/venv/bin/python -m bar3.scale.family.general_check RECORD.json "PROGRAM"
"""

from __future__ import annotations

import json
import os
import random
import sys

import torch

from integration.lib import libcore as L
from bar3.scale.family import hooks as H
from bar3.scale.family.verify7 import H1_DIR, P, num


def main():
    rec = json.load(open(sys.argv[1]))
    lib = {k: L.parse(v) for k, v in rec["lib"].items()}
    prog = L.parse(sys.argv[2])
    st = L.round_settings(L.Cells(10, "cpu"), P)
    Ns = sorted({int(json.loads(l)["modulus"]) for name in ("depth_t", "depth_ood_n_t")
                 for l in open(os.path.join(H1_DIR, f"{name}_1.jsonl"))})
    rng = random.Random(7)
    xs = [0] * len(Ns) + [rng.randrange(N) for N in Ns] + [N - 1 for N in Ns]
    Nall = Ns * 3
    want = [H.TARGET_F(x, N) for x, N in zip(xs, Nall)]
    X, _c = L.run_prog(st, prog, lib, {"X": num(xs), "N": num(Nall)}, L.gen_on("cpu", 3), "cpu")
    ok = (X == num(want)).all(1).tolist()
    k = len(Ns)
    print(f"{sys.argv[2]} (TARGET_FN {H.TARGET_FN}) on {k} moduli: x = 0 {sum(ok[:k])}/{k}, x random "
          f"{sum(ok[k:2 * k])}/{k}, x = N - 1 {sum(ok[2 * k:])}/{k}", flush=True)


if __name__ == "__main__":
    main()
