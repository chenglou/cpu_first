"""SLICE.md §5: each run's climb in short, from its record (records/<TAG>.json): the reading and cuts (the "slice"
record), each level's picks and routines, the subgoal and LAWNEAR records, and the round or how the run ended.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.slice.summary records/s4_x2.json.gz ...

A record is read as .json or .json.gz.
"""

from __future__ import annotations

import gzip
import json
import re
import sys


def main():
    for path in sys.argv[1:]:
        r = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
        cfg = r["config"]
        print(f"== {r['tag']}: {r['seconds']} s, final {r['final']}; rule {cfg.get('rule')}, laws {cfg.get('laws')}, "
              f"patterns {cfg.get('patterns')}, slice {cfg.get('slice')}, target_fn {cfg.get('target_fn', 'x2')}")
        for x in r["records"]:
            if x.get("task") == "slice":
                print(f"   reading: {x['law']} (depth {x['depth']}, constants {x['constants']}, {x['seconds']} s); "
                      f"cuts {x['cuts']}")
        for u in r["units"]:
            line = u["line"]
            if re.search(r"L\d+ (sleep|choose)|subgoal|rest deep|general|round|climb done|DEADLINE", line):
                print(f"   {u['t']:7.1f}s {line[:260]}")
        print(f"   rounds: {r['rounds'][:3]}")
        print(f"   routines: {[k for k in r['lib'] if k.startswith('T.')]}")


if __name__ == "__main__":
    main()
