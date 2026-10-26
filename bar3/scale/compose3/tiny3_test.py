"""Round 3's tiny tests of its two flags (COMPOSE3.md §3), locally:
- TARGET_FN: the oracle's draws' answers are the function of (x, N) the env names, by Python's integers (run once a
  function);
- TIEFOLLOW (TARGET_FN x2 only): on round 2's B2rd library, where x²'s routine `T.val.126.tgt.XN` exists, the look
  (LAWNEAR on, a small budget) poses x² again under a new name, and `hooks.follow` finds the rest of the way from it
  for the practised subgoal `val.126`. Without TIEFOLLOW the tie-break looks for `rest.126@XN` and finds nothing.

    cd cpu_first && H1_DATA=... TARGET_FN=x3 ~/github/sotaku/venv/bin/python -m bar3.scale.compose3.tiny3_test
"""

from __future__ import annotations

import json

from integration.lib import libcore as L
from bar3.scale.compose3 import climb_c3 as CC
from bar3.scale.compose3 import hooks as H
from bar3.scale.compose3.clean_c3 import CleanTarget
from bar3.scale.compose3.core_c3 import digits, load_rows

REC = "bar3/scale/compose2/records/c2_b2_rd.json"
FN = {"x2": lambda x, N: x * x % N, "x3": lambda x, N: x ** 3 % N, "10x2p1": lambda x, N: (10 * x * x + 1) % N}


def main():
    rows = load_rows(4096, seed=0)
    R = (digits(rows, "modulus"), digits(rows, "x"))
    t = CleanTarget(sources=("restrict", "project", "value", "rest"))
    t.lawnear = True
    c = CC.ComposeClimb(L.Cells(10, "cpu"), "cpu", 16, chunk=1 << 16, target=t, rule="lpca", NP=8, look_budget=20_000)
    tr = t.root(c, R)
    ok = all(d.ys == [FN[H.TARGET_FN](x, N) for x, N in zip(d.xs, d.Ns)] for d in tr.draws)
    print(f"1. TARGET_FN {H.TARGET_FN}: the draws' answers are the function on all {sum(d.n for d in tr.draws)} "
          f"probes: {ok}; e.g. {[(x, N, y) for x, N, y in zip(tr.draws[-1].xs, tr.draws[-1].Ns, tr.draws[-1].ys)][:4]}")
    if H.TARGET_FN != "x2":
        print("PASS" if ok else "FAIL")
        return
    c.lib = {k: L.parse(v) for k, v in json.load(open(REC))["lib"].items()}
    t.values["val.126"] = (L.parse("LDA.X STA.Y T.mul.x,o.XY"), (2, "A"))
    t.subgoals_seen.append("val.126")
    c.practiced["val.126/tgt@XN"] = 5
    _, rec, near = t.search(c, R, c.look_budget, None, near=12)
    out = t.pose(c, R, near)
    roots = []
    for r, goals in out:
        r.goal_list = goals
        roots.append(r)
    c._level_roots = roots
    rests = [r.key for r in roots if r.kind == "rest"]
    hit = H.follow(c, "val.126")
    by_name = next((x for x in roots if x.key == "rest.126@XN"), None)
    print(f"2. the look's near misses: {[p for _, _, p in rec['near']][-4:]}; the rests posed: {rests}")
    print(f"   by name (round 2's tie-break): {by_name and by_name.key}; by value (TIEFOLLOW): {hit and hit.key}, "
          f"from {L.show(t.values['val.' + hit.key.split('@')[0][5:]][0]) if hit else None}")
    print("PASS" if ok and hit is not None and by_name is None else "FAIL")


if __name__ == "__main__":
    main()
