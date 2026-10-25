"""Watch CARRY.md's learner runs: read their records straight from the Volume `bar3-bridge` (v1/carry/*.jsonl; no
container), and print one line per new round (what it picked, the target search, and the program when invented). With
--once, print every round so far and stop.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_watch [--once]
"""

from __future__ import annotations

import json
import sys
import time

import modal

vol = modal.Volume.from_name("bar3-bridge")


def rounds():
    out = {}
    for e in vol.listdir("v1/carry"):
        if not e.path.endswith(".jsonl"):
            continue
        txt = b"".join(vol.read_file(e.path)).decode()
        out[e.path.split("/")[-1][:-6]] = [json.loads(line) for line in txt.splitlines() if line.strip()]
    return out


def line(name, r):
    tr = r.get("target_rec") or {}
    hours = sum(t["seconds"] for t in r["totals"].values()) / 3600
    g = r.get("general")
    inv = f"INVENTED `{g['program']}`" if g else f"target cost {tr.get('found_cost')} levels {tr.get('level_sizes')}"
    return (f"{name} round {r['round']}: picked {[p[:40] for p in r['picked'][:6]]}{'...' if len(r['picked']) > 6 else ''}"
            f" ({len(r['picked'])}); {inv}; lib {len(r['lib'])}; {hours:.1f} search-hours")


def main(once):
    seen = {}
    while True:
        try:
            cur = rounds()
        except Exception as e:                  # a transient Volume error must not end the watch
            print(f"watch: read failed {e!r}"[:200], flush=True)
            cur = {}
        for name, recs in sorted(cur.items()):
            for r in recs[seen.get(name, 0):]:
                print(line(name, r), flush=True)
            seen[name] = max(seen.get(name, 0), len(recs))
        if once:
            return
        time.sleep(120)


if __name__ == "__main__":
    main("--once" in sys.argv)
