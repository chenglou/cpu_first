"""DIGITS.md §3: with this study's code installed (machine.py's block in the interpreter and the compiled engine,
jobs.py's search jobs), committed records still reproduce. Held-out runs (bar3/scale/heldout, committed) are replayed
on their stores, as bar3/scale/subgoal/attribute.py replays them, and one in every `every` of their cheap searches
(the 30k-state probes, looks and target probes) is run again on the patched engine and compared with the stored
record, field by field but the seconds. No other search runs.

    cd cpu_first && <python with numba> -m bar3.scale.digits.check STORE.jsonl ... -- RECORD.jsonl ...
"""

from __future__ import annotations

import hashlib
import json
import sys

from bar3.scale.digits import jobs as DJ          # installs the block and the jobs
from bar3.scale.heldout import targets  # noqa: F401  (the held-out targets)
from bar3.breadth import targets as TG
from bar3.breadth.drive import Tap
from bar3.scale.subgoal import attribute as A
from bar3.scale.subgoal import learn
from bar3.scale.subgoal.configs import CONFIGS

FIELDS = ("found_cost", "level_sizes", "best_fit", "NP", "stop", "n_tokens", "n_classes", "lib_size", "aliases",
          "programs", "near")
CHEAP = ("probe_a", "look", "target_probe")


def replay(cfg, tname, cache, every):
    count = dict(searches=0, rerun=0, differ=[])

    def search_many(jobs):
        out = []
        for j in jobs:
            r = cache.get(hashlib.sha1(json.dumps(j, sort_keys=True).encode()).hexdigest())
            if r is None:
                raise A.NotStored(j["name"])
            count["searches"] += 1
            if j["kind"] in CHEAP and count["searches"] % every == 0:
                fresh = json.loads(json.dumps(DJ.search_job(j)))
                count["rerun"] += 1
                bad = [f for f in FIELDS if fresh.get(f) != r.get(f)]
                if bad:
                    count["differ"].append((j["kind"], j["name"], bad))
            out.append(r)
        return out

    recs = []
    tap = Tap(search_many)
    TG.use(tname)
    tap.target = tname
    learn.run("replay", json.loads(json.dumps(cfg)), tap.mapper, tap.emit_with_looks(recs.append),
              log=lambda *a: None)
    return recs, count


def main():
    args = sys.argv[1:]
    every = 40
    if args and args[0].startswith("--every="):
        every = int(args.pop(0).split("=")[1])
    cut = args.index("--")
    cache = A.load_stores(args[:cut])
    for path in args[cut + 1:]:
        name = path.rsplit("/", 1)[-1].removesuffix(".jsonl")
        config = name.rsplit("_", 2)[0]                     # <config>_<target>_<tag>
        orig = [r for r in (json.loads(line) for line in open(path) if line.strip()) if "round" in r]
        tname = orig[0]["target_name"]
        recs, count = replay(CONFIGS[config], tname, cache, every)
        diff = A.first_difference(orig, recs, None)
        print(f"{name}: replay {'reproduces the record' if diff is None else f'differs from round {diff}'}; "
              f"{count['searches']} searches, {count['rerun']} run again on the patched engine, "
              f"{len(count['differ'])} differ" + (f": {count['differ'][:3]}" if count["differ"] else ""), flush=True)


if __name__ == "__main__":
    main()
