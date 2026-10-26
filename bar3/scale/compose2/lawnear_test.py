"""A tiny test of LAWNEAR and SUBGOAL where the multiply exists (COMPOSE2.md §3): round 1's A2c final library
(book_min less its reduction, clean values; it has the multiply), on round 2's copies, locally (CPU draws). The look
(the target's search at the look budget, 12 near misses by fit) with LAWNEAR on; then the pose and the subgoal score.
Passes when x^2 (`LDA.X STA.Y T.mul...`) is among the near misses, its value is posed, the subgoal score singles it
out with the law mod(g,N), and the rest of the way from it is posed.

    cd cpu_first && H1_DATA=... ~/github/sotaku/venv/bin/python -m bar3.scale.compose2.lawnear_test
"""

from __future__ import annotations

import json
import os

import torch

from integration.lib import libcore as L
from bar3.scale.compose2 import climb_c2 as CC
from bar3.scale.compose2 import hooks as H
from bar3.scale.compose2.clean_c2 import CleanTarget
from bar3.scale.compose2.core_c2 import digits, load_rows

REC = "bar3/scale/compose/records/cp_bar3_minus_red_clean.json"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows(4096, seed=0)
    R = (digits(rows, "modulus").to(dev), digits(rows, "x").to(dev))
    t = CleanTarget(sources=("restrict", "project", "value", "rest"))
    t.lawnear = True
    c = CC.ComposeClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16, target=t,
                        rule="lpca", NP=8, fuse=4 if dev == "cuda" else 1,
                        look_budget=int(os.environ.get("LOOK", 300_000)))
    c.subgoal_on = True
    c.lib = {k: L.parse(v) for k, v in json.load(open(REC))["lib"].items()}
    _, rec, near = t.search(c, R, c.look_budget, None, near=12)
    law = [q for q in c.records if q.get("task") == "lawnear"]
    print("1. the look's near misses:", rec["near"], flush=True)
    print("   LAWNEAR's:", law[-1]["near"] if law else None, flush=True)
    out = t.pose(c, R, near)
    roots = []
    for r, goals in out:
        r.goal_list = goals
        roots.append(r)
    vals = {nm: L.show(p) for nm, (p, _) in t.values.items()}
    print("2. the values posed:", vals, flush=True)
    print("   the rests posed:", [r.key for r in roots if r.kind == "rest"], flush=True)
    H.subgoal_scores(c, R, roots)
    sg = [q for q in c.records if q.get("task") == "subgoal"]
    print("3. the subgoal score:", sg[-1] if sg else None, flush=True)
    ok = (t.subgoal is not None and "T.mul" in vals.get(t.subgoal, "") and sg[-1]["top"][0][2] == "mod(g,N)"
          and any(r.key == f"rest.{t.subgoal[4:]}@XN" for r in roots))
    print("PASS" if ok else "FAIL", flush=True)
    if os.environ.get("REST_DEEP") == "1" and ok:          # REST_DEEP end to end: the subgoal solved, its rest open
        c.rest_deep, c._level_roots, c.level = True, roots, 0
        c.results[f"{t.subgoal}/tgt@XN"] = dict(found_cost=1, accepted=[], aliases=[], how="routines", root=None)
        c._accs = {}
        line = H.rest_deep(c, R, None)
        got = c.results.get(f"target/rest.{t.subgoal[4:]}@XN", {})
        print("4.", line, flush=True)
        print("REST_DEEP PASS" if got.get("found_cost") is not None else "REST_DEEP FAIL", flush=True)


if __name__ == "__main__":
    main()
