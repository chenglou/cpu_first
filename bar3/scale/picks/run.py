"""PICKS.md §4: a run of this study's learner copies here, from the earlier runs' stores (as
bar3/scale/newton/run.py): every search those runs made is read back, every other one is run in this process and
appended to this run's own store (logs/store_<run>.jsonl), so the run replays from the stores alone. It writes the
run's round records (records/<config>_<target>_<tag>.jsonl) and stops where the learner stops (5 rounds, or when the
target comes). Run it under memcap.sh.

    cd cpu_first && bar3/scale/rowcheck/memcap.sh 10 <python with numba> -m bar3.scale.picks.run CONFIG TARGET TAG STORE ...
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from bar3.breadth import targets as TG
from bar3.scale.heldout import targets as _ho  # noqa: F401  (registers the held-out targets)
from bar3.scale.picks.check import learner
from bar3.scale.picks.configs import CONFIGS, LEARNER
from bar3.scale.subgoal import attribute as A

HERE = Path(__file__).resolve().parent


def main():
    config, tname, tag, stores = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    L, search_job, TapClass = learner(config)
    name = f"{config}_{tname}_{tag}"
    own = HERE / "logs" / f"store_{name}.jsonl"
    cache = A.load_stores(stores + ([str(own)] if own.exists() else []))
    out = HERE / "records" / f"{name}.jsonl"
    out.write_text("")
    t0 = time.time()
    count = dict(read=0, ran=0)

    def search_many(jobs):
        recs = []
        for j in jobs:
            key = hashlib.sha1(json.dumps(j, sort_keys=True).encode()).hexdigest()
            r = cache.get(key)
            if r is None:
                r = json.loads(json.dumps(search_job(j)))
                cache[key] = r
                with open(own, "a") as fh:
                    fh.write(json.dumps([key, r]) + "\n")
                count["ran"] += 1
                print(f"  ran {j['kind']} {j['name']}: cost {r.get('found_cost')}, {r.get('wall', 0):.0f}s",
                      flush=True)
            else:
                count["read"] += 1
            recs.append(r)
        return recs

    def emit(rec):
        with open(out, "a") as fh:
            fh.write(json.dumps(dict(rec, elapsed=round(time.time() - t0, 1))) + "\n")

    tap = TapClass(search_many)
    TG.use(tname)
    tap.target = tname
    cfg = json.loads(json.dumps(CONFIGS[config]))
    if LEARNER[config] == "digits":
        cfg = dict(cfg, target=tname)
    L.run(name, cfg, tap.mapper, tap.emit_with_looks(lambda r: emit(dict(r, target_name=tname))), log=print)
    print(f"{name}: done in {time.time() - t0:.0f}s; {count['read']} searches read from the stores, "
          f"{count['ran']} run here", flush=True)


if __name__ == "__main__":
    main()
