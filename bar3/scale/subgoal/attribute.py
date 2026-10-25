"""Which rounds the subgoal changed (SUBGOAL.md §5): each subgoal run replayed on the searches its container stored
(the Volume's store_*.jsonl, fetched), first as it ran (a check of the replay: its records must come back), then
without the subgoal: the same learner and config with no score, no deeper probe and pool order for every tie, which is
the scale study's `<rule>_deep`. A round is the same when its picks, library, target search and every probe outcome
except the subgoal's own are the same as in the subgoal run; the first round that differs is where the subgoal first
changed the run. From there the run without it asks for searches no store holds; the first such search ends its replay,
and its round counts as different. No search runs here.

    cd cpu_first && <python with numba> -m bar3.scale.subgoal.attribute STORE.jsonl ... -- RECORD.jsonl ... [--as SUFFIX]

With --as SUFFIX (e.g. `chain`), each run is also replayed under its config + SUFFIX (`lp_clean_deep_subchain`): the
round from which that config would have run differently, if any.
"""

from __future__ import annotations

import hashlib
import json
import sys

from bar3.breadth import targets as TG
from bar3.breadth.drive import Tap
from bar3.scale.subgoal import learn
from bar3.scale.subgoal.configs import CONFIGS


class NotStored(Exception):
    pass


def load_stores(paths):
    cache = {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                try:
                    key, rec = json.loads(line)
                except ValueError:
                    continue
                cache[key] = rec
    return cache


def replay(cfg, tname, cache):
    """The learner's round records for cfg on tname, every search read from the stores (pool.py's job key), and the
    round in which it asked for a search none of them holds (None if it never did)."""
    def search_many(jobs):
        out = []
        for j in jobs:
            r = cache.get(hashlib.sha1(json.dumps(j, sort_keys=True).encode()).hexdigest())
            if r is None:
                raise NotStored(j["name"])
            out.append(r)
        return out

    recs = []
    tap = Tap(search_many)
    TG.use(tname)
    tap.target = tname
    try:
        learn.run("replay", json.loads(json.dumps(cfg)), tap.mapper, tap.emit_with_looks(recs.append),
                  log=lambda *_: None)
        stop = None
    except NotStored:
        stop = sum(1 for r in recs if "round" in r)
    return [r for r in recs if "round" in r], stop


def trajectory(rec, sub):
    """What a round decides: picks, library, the target search, and every probe outcome but the subgoal's (sub: the
    subgoal run's subgoal in that round, left out of both runs' outcomes)."""
    status = {n: {k: v.get(k) for k in ("status", "cost", "best", "practice")} for n, v in rec["status"].items()
              if n != sub}
    tr = rec["target"] or {}
    return dict(picked=rec["picked"], solved=rec["solved"], lib=rec["lib"], status=status,
                target=(tr.get("found_cost"), [p["program"] for p in tr.get("programs", [])]),
                general=(rec.get("general") or {}).get("program"))


def first_difference(orig, recs, stop):
    """The first round in which recs differ from orig, or None."""
    for i, (x, y) in enumerate(zip(orig, recs)):
        sub = (x.get("subgoal") or {}).get("name")
        if trajectory(x, sub) != trajectory(y, sub):
            return i
    if stop is not None:
        return stop
    return None if len(orig) == len(recs) else min(len(orig), len(recs))


def main():
    args = sys.argv[1:]
    alt = None
    if "--as" in args:
        i = args.index("--as")
        alt = args[i + 1]
        args = args[:i] + args[i + 2:]
    cut = args.index("--")
    cache = load_stores(args[:cut])
    print(f"{len(cache)} stored searches")
    for path in args[cut + 1:]:
        with open(path) as fh:
            orig = [json.loads(line) for line in fh if line.strip()]
        orig = [r for r in orig if "round" in r]
        name = path.rsplit("/", 1)[-1].removesuffix(".jsonl")
        config = name.rsplit("_", 2)[0]                     # <config>_<target>_<tag>
        tname = orig[0]["target_name"]
        check = first_difference(orig, *replay(CONFIGS[config], tname, cache))
        base_cfg = {k: v for k, v in CONFIGS[config].items() if k != "subgoal"}
        diff = first_difference(orig, *replay(base_cfg, tname, cache))
        print(f"{name}: replay {'reproduces the record' if check is None else f'DIFFERS from round {check}'}; "
              "without the subgoal: " + ("the same in every round" if diff is None else
                                         f"different from round {diff}" + (f" (rounds 0-{diff - 1} the same)"
                                                                           if diff else "")))
        if alt:
            other = first_difference(orig, *replay(CONFIGS[config + alt], tname, cache))
            print(f"    as {config + alt}: " + ("the same in every round" if other is None else
                                                f"different from round {other}"))


if __name__ == "__main__":
    main()
