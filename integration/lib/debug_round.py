"""Diagnostic (INTEGRATION.md §3): the round's search as the in-run climb ran it at level 6 (s3d_relaxed_h100), with the
library rebuilt from that run's sleep lines (lib/s3d_sleeps.txt) and the data file's T = 1 rows as the answers; the
goal states and the programs listed at each cost. Not a lesson: a reproduction of a miss.

    env: ROWS (512), LISTING ("spellings" as the climb, or "goals")
"""
import os
import time

import torch

from integration.lib import libcore as L, inclimb

dev = "cuda" if torch.cuda.is_available() else "cpu"
here = os.path.dirname(os.path.abspath(__file__))
lib = {}
for line in open(os.path.join(here, "s3d_sleeps.txt")):
    body = line.split("; ", 1)[1].rsplit(" (", 1)[0]
    for part in body.split("; "):
        if "=" in part:
            k, v = part.split("=", 1)
            lib[k.strip()] = L.parse(v.strip())
rows = inclimb.load_rows(int(os.environ.get("ROWS", 512)), seed=3, T=1)
ans = tuple(t.to(dev) for t in inclimb.digits_of_rows(rows))
cells = L.Cells(10, dev)
_programs = L.programs


def programs(res, classes, limit=200):
    out = _programs(res, classes, limit)
    print("LISTED", {"cost": res.found_cost, "goal_states": len(res.goals), "listed": len(out),
                     "first": [L.show(p) for p in out[:12]]}, flush=True)
    return out


L.programs = programs
for listing in os.environ.get("LISTING", "spellings,goals").split(","):
    c = L.Climb(cells, dev, 16, listing=listing)
    c.lib = dict(lib)
    t0 = time.time()
    progs, rec = c.search_round(ans)
    print("ROUND", listing, rec["found_cost"], rec["level_sizes"], rec["stop"], rec["programs"][:6],
          f"{time.time() - t0:.1f}s", flush=True)
    good = L.parse("LDA.X STA.Y T.mul~ STA.X T.reduce")
    print("fit of LDA.X STA.Y T.mul~ STA.X T.reduce on all rows:", L.round_fit(good, lib, cells, ans, 16, dev), flush=True)
