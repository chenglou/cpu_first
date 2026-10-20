"""Table of the base-10 runs: per bits and variant, seeds solved (exact >= 0.99), median steps, median final exact,
exact with the true digits given, and a failed run's slot map.

    python -m tinycore.b10table runs/b10.jsonl runs/b10b.jsonl runs/b10c.jsonl
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
    key = (r["bits"], r["cell"], r["choice"], r["fix"], r["start"], r.get("mods", "microlab"), r.get("score_top", False))
    g[key].append(r)
print("bits | cell | choice | fix | start | moduli | top digit scored | solved | median steps | median final | median exact given | a failed run's slots")
for key, rs in sorted(g.items()):
    hits = [r["hit"] for r in rs if r["hit"] is not None]
    fails = [r["slots"] for r in rs if r["hit"] is None]
    print(" | ".join(str(x) for x in key) + f" | {len(hits)}/{len(rs)} | {int(np.median(hits)) if hits else None} | "
          f"{np.median([r['final'] for r in rs]):.2f} | {np.median([r['exact_given'] for r in rs]):.2f} | "
          f"{fails[0] if fails else ''}")
