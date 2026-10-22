"""What the miner's grammar finds in a dataset when every field is read right (LESSONS.md section 2): the relations
of submission_general.py's `relations` (order, equality, constant, nonzero, range, divisibility by 2, 3, 5) among N, x,
T and the answer y, evaluated on every training row from the generator's own fields (modulus, x, time_steps, result).
A diagnostic for the write-up, run after the fact; the model itself mines inside training on the fields as it reads
them, and its own list is in each run's log (the MINED line). Mirrors `relations` and `Miner.facts` on integers.

    python mine_preview.py DATA_DIR [DATA_DIR ...]       # prints one JSON line per dataset
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path


def relations(v: dict, W: int = 9) -> dict:
    out = {}
    names = list(v)
    for a, b in itertools.permutations(names, 2):
        out[f"{a} < {b}"] = all(p < q for p, q in zip(v[a], v[b]))
    for a, b in itertools.combinations(names, 2):
        out[f"{a} = {b}"] = all(p == q for p, q in zip(v[a], v[b]))
    for a in names:
        col = v[a]
        out[f"{a} constant"] = all(p == col[0] for p in col)
        out[f"{a} != 0"] = all(p != 0 for p in col)
        for k in range(1, W):
            out[f"{a} < 10^{k}"] = all(p < 10 ** k for p in col)
        for d in (2, 3, 5):
            out[f"{a} mod {d} != 0"] = all(p % d != 0 for p in col)
            for r in range(d):
                out[f"{a} mod {d} = {r}"] = all(p % d == r for p in col)
    return out


def prune(f: set) -> list:
    out = []
    for k in sorted(f):
        if " < 10^" in k:
            a, e = k.split(" < 10^")
            if f"{a} < 10^{int(e) - 1}" in f:
                continue
        if k.endswith(" != 0") and " mod " in k:
            a, d = k.split(" mod ")[0], k.split(" mod ")[1].split()[0]
            if any(f"{a} mod {d} = {r}" in f for r in range(1, 5)):
                continue
        out.append(k)
    return out


def code_count(N: list, x: list, reverse_n: bool = False) -> dict:
    """How many digit codes pass x < N on every row, the reader's fact: at the highest place where x's and N's digits
    differ, x's digit ranks below N's (and where x is shorter, N's digit there ranks above 0's). The codes are the
    linear extensions of that partial order on the ten digits (a DP over subsets); 1 means the facts pin the code.
    With reverse_n, N's digits are read the other way round (does that reading also pass, and with how many codes?)."""
    below = [set() for _ in range(10)]                 # below[v]: digits that must rank below v
    for n, v in zip(N, x):
        a = [int(c) for c in str(v)][::-1]
        b = [int(c) for c in str(n)]
        b = b if reverse_n else b[::-1]
        L = max(len(a), len(b))
        a, b = a + [0] * (L - len(a)), b + [0] * (L - len(b))
        top = max((j for j in range(L) if a[j] != b[j]), default=-1)
        if top >= 0:
            below[b[top]].add(a[top])
    # a cycle means no code passes (the reading is ruled out)
    f = {0: 1}
    for mask in range(1 << 10):
        if mask not in f:
            continue
        for v in range(10):
            if not mask >> v & 1 and all(mask >> u & 1 for u in below[v]):
                f[mask | 1 << v] = f.get(mask | 1 << v, 0) + f[mask]
    pairs = sum(len(s) for s in below)
    adjacent = sum(1 for k in range(9) if k in below[k + 1])
    return {"codes": f.get((1 << 10) - 1, 0), "ordered_pairs": pairs, "adjacent_pinned": adjacent}


def preview(data: str) -> dict:
    v = {"N": [], "x": [], "T": [], "y": []}
    with open(Path(data) / "train.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            v["N"].append(r["modulus"])
            v["x"].append(r["x"])
            v["T"].append(r["time_steps"])
            v["y"].append(r["result"])
    facts = {k for k, ok in relations(v).items() if ok}
    return {"data": Path(data).name, "rows": len(v["N"]), "moduli": len(set(v["N"])), "facts": prune(facts),
            "codes": code_count(v["N"], v["x"]), "codes_n_reversed": code_count(v["N"], v["x"], reverse_n=True)}


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(preview(d)), flush=True)
