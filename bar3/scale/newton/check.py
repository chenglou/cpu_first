"""NEWTON.md §3: replays on the earlier runs' stores (bar3/scale/subgoal/attribute.py's replay, with this study's
learner), no Modal.
- `off`: with neither source (the frozen configs), each committed record replays exactly from its store: the copy
  is the frozen learner.
- `on`: with both sources (the `_ns` configs), each run is replayed up to a round (default: the round its target came
  in, else every round), every search read from the stores. A search none of them holds is run here when it is cheap
  (the 30k- and 1M-state probes, the looks, the target probe); the replay stops at the first other one, and names it:
  what a Modal run would have to search. For each round replayed: what the two sources posed, which of it was
  learnable and picked, and whether the round's picks, library and target search match the record's.

    cd cpu_first && <python with numba> -m bar3.scale.newton.check off STORE ... -- RECORD ...
    cd cpu_first && <python with numba> -m bar3.scale.newton.check on [--upto=R] [--run=KIND,...] STORE ... -- RECORD ...

`--run` widens the kinds run here (default: the cheap ones), e.g. `--run=probe_a,probe_b,look,target_probe,target,
practice,guided` to replay a round whose picks changed.
"""

from __future__ import annotations

import hashlib
import json
import sys

from bar3.breadth import targets as TG
from bar3.breadth.drive import Tap
from bar3.scale.heldout import targets as _ho  # noqa: F401  (registers the held-out targets)
from bar3.scale.jobs import search_job
from bar3.scale.newton import learn
from bar3.scale.newton.configs import CONFIGS
from bar3.scale.subgoal import attribute as A

CHEAP = ("probe_a", "probe_b", "look", "target_probe")
NEW = ("rsum", "offset")


class Stop(Exception):
    pass


def replay(cfg, tname, cache, run_cheap, upto=None, kinds=CHEAP):
    """The round records, and (kind, name) of the first search neither stored nor cheap (None if none was needed)."""
    ran, need = [], []

    def search_many(jobs):
        out = []
        for j in jobs:
            r = cache.get(hashlib.sha1(json.dumps(j, sort_keys=True).encode()).hexdigest())
            if r is None:
                if not (run_cheap and j["kind"] in kinds):
                    need.append((j["kind"], j["name"]))
                    raise Stop
                r = json.loads(json.dumps(search_job(j)))
                ran.append((j["kind"], j["name"], r.get("found_cost")))
            out.append(r)
        return out

    recs = []
    tap = Tap(search_many)
    TG.use(tname)
    tap.target = tname

    def emit(r):
        recs.append(r)
        if upto is not None and "round" in r and r["round"] >= upto:
            raise Stop

    try:
        learn.run("replay", json.loads(json.dumps(cfg)), tap.mapper, tap.emit_with_looks(emit), log=lambda *_: None)
    except Stop:
        pass
    return [r for r in recs if "round" in r], (need[0] if need else None), ran


def decided(rec):
    tr = rec["target"] or {}
    return dict(picked=rec["picked"], lib=rec["lib"], target=(tr.get("found_cost"), [p["program"] for p in
                tr.get("programs", [])]), general=(rec.get("general") or {}).get("program"))


def read(path):
    return [r for r in (json.loads(line) for line in open(path) if line.strip()) if "round" in r]


def name_of(path):
    return path.rsplit("/", 1)[-1].removesuffix(".jsonl").rsplit("_", 2)[0]     # <config>_<target>_<tag>


def off(cache, paths):
    for path in paths:
        orig = read(path)
        recs, need, _ = replay(CONFIGS[name_of(path)], orig[0]["target_name"], cache, False)
        diff = A.first_difference(orig, recs, None)
        print(f"{path.rsplit('/', 1)[-1]}: " + ("reproduces the record" if diff is None and need is None else
                                                 f"differs from round {diff}, missing {need}"), flush=True)


def on(cache, paths, upto, kinds=CHEAP):
    for path in paths:
        orig = read(path)
        came = next((r["round"] for r in orig if r.get("general")), None)
        last = upto if upto is not None else (came if came is not None else len(orig) - 1)
        recs, need, ran = replay(CONFIGS[name_of(path) + "_ns"], orig[0]["target_name"], cache, True, last, kinds)
        print(f"{path.rsplit('/', 1)[-1]} (came in round {came}): replayed rounds 0–{len(recs) - 1} of 0–{last}; "
              f"{len(ran)} searches run here; " + (f"stopped at a {need[0]} search of {need[1]}" if need else
                                                   "no other search needed"), flush=True)
        for r, o in zip(recs, orig):
            new = [c for c in r["pool"] if c["source"] in NEW]
            learnable = {n: v.get("cost") for n, v in r["status"].items() if n.split("/", 1)[0] in NEW
                         and v.get("status") in ("learnable", "solved in practice")}
            other = {n: (v.get("status"), v.get("cost"), v.get("of")) for n, v in r["status"].items()
                     if n.split("/", 1)[0] in NEW and n not in learnable and v.get("cost") is not None}
            picked = [n for n in r["picked"] if n.split("/", 1)[0] in NEW]
            same = decided(r) == decided(o)
            print(f"  round {r['round']}: posed {len(new)} new ({', '.join(c['name'] for c in new[:8])}); learnable "
                  f"{learnable}; solved but not learnable {other}; picked {picked}; picks, library and target search "
                  + ("as in the record" if same else f"DIFFER: picked {r['picked']} vs {o['picked']}, general "
                     f"{(r.get('general') or {}).get('program')} vs {(o.get('general') or {}).get('program')}"),
                  flush=True)


def main():
    args = sys.argv[1:]
    mode = args.pop(0)
    upto, kinds = None, CHEAP
    while args and args[0].startswith("--"):
        a = args.pop(0)
        if a.startswith("--upto="):
            upto = int(a.split("=")[1])
        elif a.startswith("--run="):
            kinds = tuple(a.split("=")[1].split(","))
    cut = args.index("--")
    cache = A.load_stores(args[:cut])
    print(f"{len(cache)} stored searches", flush=True)
    (off(cache, args[cut + 1:]) if mode == "off" else on(cache, args[cut + 1:], upto, kinds))


if __name__ == "__main__":
    main()
