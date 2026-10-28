"""Tables from sweep rows: per size and variant, how many seeds reach exact >= 0.99, when, and how runs look
(easy: hit within 300 steps; plateau then take-off: a flat stretch first; never: no hit).

    python -m tinycore.summarize runs/phase.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np


def shape(r, flat_tol=0.02, flat_len=300):
    """'easy' | 'plateau->takeoff' | 'never', and the length of the longest flat stretch before the hit."""
    c = r["curve"]
    hit = r["hit"]
    if hit is None:
        return "never", None
    ex = [e for _, _, e in c]
    st = [s for s, _, _ in c]
    # longest stretch (in steps) over which exact moved less than flat_tol, before the hit
    best, i = 0, 0
    for j in range(len(ex)):
        while ex[j] - min(ex[i:j + 1]) > flat_tol or max(ex[i:j + 1]) - ex[j] > flat_tol:
            i += 1
        if ex[i] > 0.05:                                       # flat at a level above the start
            best = max(best, st[j] - st[i])
    return ("plateau->takeoff" if best >= flat_len else "easy"), best


def table(rows, key=lambda r: (r["B"], r["w"], r["k"], r.get("cell", "slot"), r["choice"])):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    lines = []
    for k, rs in sorted(groups.items()):
        hits = [r["hit"] for r in rs if r["hit"] is not None]
        shapes = [shape(r)[0] for r in rs]
        finals = [r["final"] for r in rs]
        lines.append((k, len(rs), len(hits), (int(np.median(hits)) if hits else None),
                      shapes.count("easy"), shapes.count("plateau->takeoff"), shapes.count("never"),
                      float(np.median(finals)), rs[0]["X"]))
    return lines


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]
    print("size (B w k cell choice) | X | solved/seeds | median hit step | easy / plateau->takeoff / never | median final exact")
    for k, n, nh, mh, e, p, nv, mf, X in table(rows):
        print(f"{k} | {X} | {nh}/{n} | {mh} | {e}/{p}/{nv} | {mf:.2f}")
