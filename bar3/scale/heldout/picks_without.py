"""HELDOUT.md §5.2: what the learner picks without the subgoal in the round where a run first differs from its replay
without it (attribute.py). The replay is attribute.py's, on the same stores, with one change: a target or guided
search that no store holds finds nothing, so the round gets to its picks and logs them; any other search no store
holds ends the replay. The picks are the round's; nothing after them is (the target search was not run). No search
runs here.

    cd cpu_first && <python with numba> -m bar3.scale.heldout.picks_without STORE.jsonl ... -- RUN:ROUND ...

RUN is a record's name (`lp_clean_deep_sub_x4_ho1`), ROUND the first round that differs.
"""

from __future__ import annotations

import hashlib
import json
import sys

from bar3.breadth import targets as TG
from bar3.breadth.drive import Tap
from bar3.scale.heldout import targets  # noqa: F401  (the held-out targets, for TG.use)
from bar3.scale.subgoal import attribute as A
from bar3.scale.subgoal import learn
from bar3.scale.subgoal.configs import CONFIGS

NOTHING = dict(found_cost=None, programs=[], level_sizes=[0], seconds=0.0, wall=0.0, best_fit=[0], NP=1,
               stop="not stored")


def picks(cfg, tname, cache, rnd):
    """The learner's log lines from round rnd on, under cfg, up to the first search no store holds that is not a
    target or guided search."""
    def search_many(jobs):
        out = []
        for j in jobs:
            r = cache.get(hashlib.sha1(json.dumps(j, sort_keys=True).encode()).hexdigest())
            if r is None and j["kind"] not in ("target", "guided"):
                raise A.NotStored(j["name"])
            out.append(NOTHING if r is None else r)
        return out

    lines, recs = [], []
    tap = Tap(search_many)
    TG.use(tname)
    tap.target = tname
    try:
        learn.run("replay", json.loads(json.dumps(cfg)), tap.mapper, tap.emit_with_looks(recs.append),
                  log=lambda *a: lines.append(" ".join(map(str, a))))
    except A.NotStored as e:
        lines.append(f"  (no store holds {e}: the replay ends)")
    start = next((i for i, line in enumerate(lines) if f"round {rnd}:" in line), len(lines))
    return lines[start:]


def main():
    args = sys.argv[1:]
    cut = args.index("--")
    cache = A.load_stores(args[:cut])
    for spec in args[cut + 1:]:
        name, rnd = spec.split(":")
        config, tname, _ = name.rsplit("_", 2)
        for label, cfg in (("with the subgoal", CONFIGS[config]),
                           ("without", {k: v for k, v in CONFIGS[config].items() if k != "subgoal"})):
            shown = [line for line in picks(cfg, tname, cache, int(rnd))
                     if any(w in line for w in ("subgoal:", "picked:", "no store holds"))]
            upto = next((i for i, line in enumerate(shown) if "picked:" in line), len(shown) - 1)
            print(f"{name}, round {rnd}, {label}:")
            for line in shown[:upto + 1]:
                print("   ", line.strip()[:240])


if __name__ == "__main__":
    main()
