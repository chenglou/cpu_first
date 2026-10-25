"""The subgoal runs' outcomes, checked from their records (SUBGOAL.md): for each record, the round its target was
invented in (a general program found: right on every pair up to the learner's bound, N <= 40), that program
re-checked on every pair up to N = 120 (7,260 pairs; 3,600 for Montgomery's, odd N), its routines, and the round's
subgoal and what became of it; or how far a run got without one.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.subgoal.verify [records/*.jsonl ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from library import dsl
from library.inventing import brm, target
from bar3.breadth import targets as TG

HERE = Path(__file__).resolve().parent
UPTO = 120


def check(prog, lib, tname, upto=UPTO):
    """(pairs right, pairs) for prog on every pair with N <= upto, on a machine big enough for the true algorithm."""
    TG.use(tname)
    pl = {t: dsl.parse(v) for t, v in lib.items()}
    xs, Ns = target.pairs(upto)
    got, dead = brm.evaluate(dsl.parse(prog), pl, xs, Ns, brm.Machine(cap=4 * upto * upto, limit=10 ** 7))
    return (0 if dead else int((got == target.answers(xs, Ns)).sum())), len(xs)


def read(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summary(path):
    recs = [r for r in read(path) if "round" in r]
    if not recs:
        return dict(record=Path(path).name, rounds=0)
    tname = recs[0]["target_name"]
    out = dict(record=Path(path).name, target=tname, rounds=len(recs), invented=None,
               hours=round(sum(v["seconds"] for v in recs[-1]["totals"].values()) / 3600, 2),
               subgoals=[(r["round"], (r.get("subgoal") or {}).get("name"),
                          r["status"].get((r.get("subgoal") or {}).get("name") or "", {}).get("status"),
                          r["status"].get((r.get("subgoal") or {}).get("name") or "", {}).get("cost"),
                          (r.get("subgoal") or {}).get("name") in r["picked"]) for r in recs])
    for r in recs:
        if r.get("general"):
            g = r["general"]
            right, n = check(g["program"], r["lib"], tname)
            out.update(invented=r["round"], program=g["program"], primitive=g["primitive"],
                       primitive_cost=dsl.cost(dsl.parse(g["primitive"])), right=right, pairs=n,
                       routines={t: r["lib"][t] for t in dsl.show(dsl.parse(g["program"])).split()
                                 if t in r["lib"]})
            break
    return out


def main():
    paths = sys.argv[1:] or sorted(str(p) for p in (HERE / "records").glob("*.jsonl"))
    for p in paths:
        s = summary(p)
        if not s["rounds"]:
            print(f"{s['record']}: no rounds")
            continue
        head = f"{s['record']}: {s['rounds']} rounds, {s['hours']} search-hours"
        if s["invented"] is None:
            print(f"{head}; not invented")
        else:
            ok = "right on all" if s["right"] == s["pairs"] else "WRONG on some of"
            print(f"{head}; invented in round {s['invented']}: {s['program']} ({s['primitive_cost']} primitive "
                  f"instructions), {ok} {s['pairs']} pairs up to N = {UPTO} ({s['right']} right)")
            for t, v in s["routines"].items():
                print(f"    {t} = {v}")
        for rnd, name, st, cost, picked in s["subgoals"]:
            if name:
                print(f"    round {rnd}: subgoal {name}: {st} {cost}{', picked' if picked else ''}")


if __name__ == "__main__":
    main()
