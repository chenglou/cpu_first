"""Diagnostic (INTEGRATION.md §3, s3h): the round's search as the in-run climb ran it at level 6, with the library
rebuilt from a run's sleep lines (lib/<LIB>_sleeps.txt) and DRAWS random sets of the data's T = 1 rows as the answers
(the first 8 of each set are the probes), under the current state hash (libcore.hash_rows) or one that mixes every
64-bit word first. The current hash keeps only 8 bits of the last byte of each word (the last of the 8 probes), so
states that differ only there can be merged as one. Not a lesson: a reproduction of a miss.

    env: LIB (s3d | s3g5 | s3g6), HASH (weak | mixed), DRAWS (e.g. 0-11), ROWS (512)
"""
import json
import os
import time

import torch

from integration.lib import libcore as L, inclimb

dev = "cuda" if torch.cuda.is_available() else "cpu"
here = os.path.dirname(os.path.abspath(__file__))
C1, C2 = L._s64(0xBF58476D1CE4E5B9), L._s64(0x94D049BB133111EB)


def mix64(x):
    x = x ^ L._lsr(x, 30)
    x = x * C1
    x = x ^ L._lsr(x, 27)
    x = x * C2
    return x ^ L._lsr(x, 31)


weak = L.hash_rows


def mixed(F):
    S, n = F.shape
    pad = (-n) % 8
    if pad:
        F = torch.cat([F, torch.zeros(S, pad, dtype=torch.uint8, device=F.device)], 1)
    Wd = mix64(F.contiguous().view(torch.int64))
    if F.device not in L._MULT_ON:
        L._MULT_ON[F.device] = L._MULT.to(F.device)
    return mix64((Wd * L._MULT_ON[F.device][:Wd.shape[1]][None]).sum(1))


def lib_of(name):
    lib = {}
    for line in open(os.path.join(here, f"{name}_sleeps.txt")):
        body = line.split("; ", 1)[1].rsplit(" (", 1)[0]
        for part in body.split("; "):
            if "=" in part:
                k, v = part.split("=", 1)
                lib[k.strip()] = L.parse(v.strip())
    return lib


def main():
    name, hname = os.environ.get("LIB", "s3d"), os.environ.get("HASH", "weak")
    a, b = (int(v) for v in os.environ.get("DRAWS", "0-3").split("-"))
    lib = lib_of(name)
    L.hash_rows = {"weak": weak, "mixed": mixed}[hname]
    cells = L.Cells(10, dev)
    mul = "T.mul" if "T.mul" in lib else "T.mul~"
    good = L.parse(f"LDA.X STA.Y {mul} STA.X T.reduce")
    print("LIB", name, "HASH", hname, "device", torch.cuda.get_device_name() if dev == "cuda" else "cpu",
          {k: L.show(v) for k, v in lib.items() if k.startswith("T.")}, flush=True)
    for draw in range(a, b + 1):
        rows = inclimb.load_rows(int(os.environ.get("ROWS", 512)), seed=draw, T=1)
        ans = tuple(t.to(dev) for t in inclimb.digits_of_rows(rows))
        c = L.Climb(cells, dev, 16)
        c.lib = dict(lib)
        t0 = time.time()
        progs, rec = c.search_round(ans)
        fit = L.round_fit(good, lib, cells, ans, 16, dev)
        print("DRAW", json.dumps({"lib": name, "hash": hname, "draw": draw, "found_cost": rec["found_cost"],
                                  "levels": rec["level_sizes"], "stop": rec["stop"], "programs": rec["programs"][:3],
                                  "good_fit": fit, "s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
