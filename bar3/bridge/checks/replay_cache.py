"""Replay a learner run from the shared search cache alone (dlearner.run with a mapper that only reads the cache): what
each round picks and keeps, as far as the cache reaches. A job the cache does not hold counts as failed, so the round
where the cache runs out is shown partly (its target search, typically, as failed). Launches nothing.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.replay_cache lp_adc 2 [widen]
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import modal

from bar3.bridge import dlearner as L

cache = modal.Dict.from_name("bar3-bridge-cache")


def key(job):
    return "v1" + hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()


def replay(name, rounds, flags=(), log=print):
    """(the run's records, the jobs the cache did not hold) for `rounds` rounds of config `name`."""
    cfg = dict(L.CONFIGS[name], rounds=int(rounds), **{f: True for f in flags})
    missing = []

    def mapper(jobs):
        with ThreadPoolExecutor(32) as ex:
            got = list(ex.map(lambda j: cache.get(key(j)), jobs))
        out = []
        for j, r in zip(jobs, got):
            if r is None:
                missing.append((j["kind"], j["name"]))
                out.append(KeyError("not cached"))
            else:
                out.append(r)
        return out

    recs = []
    L.run(name, cfg, mapper, recs.append, log=lambda s: None if s.startswith("  job failed") else log(s))
    return recs, missing


def main(name, rounds, *flags):
    recs, missing = replay(name, rounds, flags)
    print("not cached:", len(missing), {k: sum(1 for m in missing if m[0] == k) for k in {m[0] for m in missing}})
    for r in recs:
        st = r["status"]
        print(f"round {r['round']}: picked {r['picked']}")
        for nm in r["picked"]:
            print(f"    {nm}: cost {st[nm]['cost']} {st[nm]['best']}")
        for nm, v in sorted(st.items()):
            if "R<C" in nm and "add" in nm or "mod(A,N)" in nm or nm.startswith("widen/"):
                print(f"  {nm:60s} {v['status']:10s} cost {v.get('cost')} best {v.get('best')}")


if __name__ == "__main__":
    main(*sys.argv[1:])
