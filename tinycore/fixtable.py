"""Tables of the fix comparisons: per size, fix and choice, seeds solved (exact >= 0.99), median steps to solve,
median final exact, and a typical final slot map.

    python -m tinycore.fixtable runs/fixes.jsonl runs/constructs.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    return rows


def table(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["B"], r["w"], r["k"], r.get("label", r["fix"]), r["choice"])].append(r)
    out = []
    for key, rs in sorted(g.items()):
        hits = [r["hit"] for r in rs if r["hit"] is not None]
        finals = sorted(r["final"] for r in rs)
        fail = [r["slots"] for r in rs if r["hit"] is None]
        extra = ""
        if rs[0].get("multipliers"):
            extra = " mult " + str([m for m, _ in rs[0]["multipliers"]]) + f" modN {np.median([r['exact_mod_N'] for r in rs]):.2f}"
        out.append((key, len(rs), len(hits), int(np.median(hits)) if hits else None, float(np.median(finals)),
                    fail[0] if fail else None, extra))
    return out


if __name__ == "__main__":
    print("B w k | fix | choice | solved | median steps | median final | a failed run's slot map")
    for key, n, nh, mh, mf, fs, extra in table(load(sys.argv[1:])):
        B, w, k, fix, ch = key
        print(f"{B} {w} {k} | {fix:14s} | {ch:6s} | {nh}/{n} | {mh} | {mf:.2f} | {fs}{extra}")
