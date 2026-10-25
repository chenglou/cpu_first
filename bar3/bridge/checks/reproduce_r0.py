"""The learner's records still reproduce with CARRY.md's flags off: round 0 of each run, replayed from the shared search
cache with the current code (checks/replay_cache.py), against its stored record, field by field (every field but the
run's name and launch time, and the `cached` flag search_remote adds to a job the cache answered: the target probe of
an empty library is one job for the three rules, so two of them were served it). A job whose description changed would
miss the cache, so the replay also checks that the jobs are the ones the records were made with. Launches nothing.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.reproduce_r0 lp_adc ct2_mul all_sel
"""

from __future__ import annotations

import json
import sys

from bar3.bridge import report as R
from bar3.bridge.checks.replay_cache import replay


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k != "cached"}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def main(names):
    bad = 0
    for name in names:
        recs, missing = replay(name, 1, log=lambda s: None)
        old = strip(R.load(f"{name}_r1")[0])
        new = strip(json.loads(json.dumps(recs[0])))
        diff = sorted(k for k in set(old) | set(new) if k not in ("config", "launch") and old.get(k) != new.get(k))
        print(f"{name}: jobs not cached {len(missing)}; fields {len(new)}; differing: {diff or 'none'}")
        bad += bool(diff or missing)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
