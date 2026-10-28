"""Table of the answers-only route at base 10 (suites b10d, b10e, b10f): per bits and estimator window, the
estimator phase (exact, and exact with the true digits given, i.e. whether the shared subtractor is right), then the
cell's own borrow choice (seeds reaching exact >= 0.99, and median steps after the switch), then the same tables run
at other widths.

    python -m tinycore.switchtable runs/b10d.jsonl runs/b10e.jsonl runs/b10f.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np

rows = []
for p in sys.argv[1:]:
    try:
        rows += [json.loads(l) for l in open(p)]
    except FileNotFoundError:
        pass
g = defaultdict(list)
for r in rows:
    win = "2/1" if "est21" in r["name"] else "3/2"
    g[(r["bits"], r["cell"], win, "xfer" in r["name"])].append(r)
print("bits | cell | window | seeds | estimator phase: exact / exact given true digits | own borrow: solved, median steps | same tables at 6 / 7 / 10 bits")
for key, rs in sorted(g.items()):
    bits, cell, win, xf = key
    est = np.median([r["final"] for r in rs])
    giv = np.median([r["exact_given"] for r in rs])
    sw = [r["switched"] for r in rs if r.get("switched")]
    solved = [s for s in sw if s["final"] >= 0.99]
    steps = [s["curve"][-1][0] for s in solved]
    tr = defaultdict(list)
    for r in rs:
        for b, v in (r.get("transfer") or {}).items():
            tr[int(b)].append(v)
    trs = " / ".join(f"{np.median(tr[b]):.2f}" for b in (6, 7, 10) if b in tr) or "·"
    print(f"{bits} | {cell} | {win} | {len(rs)} | {est:.2f} / {giv:.2f} | {len(solved)}/{len(sw)}, "
          f"{int(np.median(steps)) if steps else None} | {trs}")
