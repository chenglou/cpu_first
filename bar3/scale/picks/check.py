"""PICKS.md §3–5: replays on the earlier runs' stores (bar3/scale/subgoal/attribute.py's replay, with this study's
learner copies), no Modal. A record's config is read from its file name (<config>_<target>_<tag>.jsonl).
- `off`: with every flag off, each record replays exactly from its stores: the copy is that study's learner.
- `on --variant=V`: with one change on (the config's variant V: `_rc_tie`, `_twin` or `_bigx`), each run is replayed
  up to a round (default: the round its target came in, else every round), every search read from the stores. A
  search none of them holds is run here when it is of a kind given (default: the cheap ones: the 30k- and 1M-state
  probes, the looks, the target probe); the replay stops at the first other one and names it. For each round: what
  the change did (its `picks` field), the problems whose status changed, and whether the round's picks, library and
  target search are the record's.
Digits records need --digits (the digit block and the digits jobs are installed in this process only then).

    cd cpu_first && <python with numba> -m bar3.scale.picks.check off [--digits] STORE ... -- RECORD ...
    cd cpu_first && <python with numba> -m bar3.scale.picks.check on --variant=V [--digits] [--upto=R] [--run=KIND,...] STORE ... -- RECORD ...
"""

from __future__ import annotations

import hashlib
import json
import sys

from bar3.breadth import targets as TG
from bar3.breadth.drive import Tap
from bar3.scale.heldout import targets as _ho  # noqa: F401  (registers the held-out targets)
from bar3.scale.picks.configs import CONFIGS, LEARNER
from bar3.scale.subgoal import attribute as A

CHEAP = ("probe_a", "probe_b", "look", "target_probe")


class Stop(Exception):
    pass


def learner(config):
    if LEARNER[config] == "digits":
        from bar3.scale.digits import jobs as DJ     # installs the block and the digits targets and jobs
        from bar3.scale.picks import learn_digits as L
        return L, DJ.search_job, L.DigitsTap
    from bar3.scale.jobs import search_job
    from bar3.scale.picks import learn_newton as L
    return L, search_job, Tap


def replay(config, tname, cache, run_kinds=(), upto=None):
    """The round records, (kind, name) of the first search neither stored nor runnable here, and the searches run."""
    L, search_job, TapClass = learner(config)
    ran, need = [], []

    def search_many(jobs):
        out = []
        for j in jobs:
            r = cache.get(hashlib.sha1(json.dumps(j, sort_keys=True).encode()).hexdigest())
            if r is None:
                if j["kind"] not in run_kinds:
                    need.append((j["kind"], j["name"]))
                    raise Stop
                r = json.loads(json.dumps(search_job(j)))
                ran.append((j["kind"], j["name"], r.get("found_cost")))
            out.append(r)
        return out

    recs = []
    tap = TapClass(search_many)
    TG.use(tname)
    tap.target = tname

    def emit(r):
        recs.append(r)
        if upto is not None and "round" in r and r["round"] >= upto:
            raise Stop

    cfg = json.loads(json.dumps(CONFIGS[config]))
    if LEARNER[config] == "digits":
        cfg = dict(cfg, target=tname)
    try:
        L.run("replay", cfg, tap.mapper, tap.emit_with_looks(emit), log=lambda *_: None)
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
        recs, need, _ = replay(name_of(path), orig[0]["target_name"], cache)
        diff = A.first_difference(orig, recs, None)
        print(f"{path.rsplit('/', 1)[-1]}: " + ("reproduces the record" if diff is None and need is None else
                                                 f"differs from round {diff}, missing {need}"), flush=True)


def changed(r, o):
    """{problem: (status in the record, status now)} for the problems both rounds probed whose status differs."""
    return {n: (o["status"][n].get("status"), v.get("status"), v.get("cost"), v.get("of"))
            for n, v in r["status"].items() if n in o["status"] and o["status"][n].get("status") != v.get("status")}


def on(cache, paths, upto, kinds, variant):
    for path in paths:
        orig = read(path)
        came = next((r["round"] for r in orig if r.get("general")), None)
        last = upto if upto is not None else (came if came is not None else len(orig) - 1)
        recs, need, ran = replay(name_of(path) + variant, orig[0]["target_name"], cache, kinds, last)
        g = next((r for r in recs if r.get("general")), None)
        print(f"{path.rsplit('/', 1)[-1]} (came in round {came}): replayed rounds 0–{len(recs) - 1} of 0–{last}; "
              f"{len(ran)} searches run here; " + (f"stopped at a {need[0]} search of {need[1]}" if need else
                                                   "no other search needed")
              + (f"; came in round {g['round']}: {g['general']['program']}" if g else ""), flush=True)
        for r, o in zip(recs, orig):
            ch = changed(r, o)
            same = decided(r) == decided(o)
            print(f"  round {r['round']}: {r.get('picks')}; {len(ch)} statuses changed {dict(list(ch.items())[:4])}; "
                  "picks, library and "
                  "target search " + ("as in the record" if same else f"DIFFER: picked {r['picked']} vs "
                                      f"{o['picked']}"), flush=True)


def main():
    args = sys.argv[1:]
    mode = args.pop(0)
    upto, kinds, variant = None, CHEAP, None
    while args and args[0].startswith("--"):
        a = args.pop(0)
        if a.startswith("--upto="):
            upto = int(a.split("=")[1])
        elif a.startswith("--run="):
            kinds = tuple(a.split("=")[1].split(","))
        elif a.startswith("--variant="):
            variant = a.split("=")[1]
        elif a == "--digits":
            from bar3.scale.digits import jobs  # noqa: F401  (installs the block before any replay)
    cut = args.index("--")
    cache = A.load_stores(args[:cut])
    print(f"{len(cache)} stored searches", flush=True)
    (off(cache, args[cut + 1:]) if mode == "off" else on(cache, args[cut + 1:], upto, kinds, variant))


if __name__ == "__main__":
    main()
