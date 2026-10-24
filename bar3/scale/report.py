"""Tables for SCALE.md from the scale records.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.report --matrix bar3/scale/records/*_L1.jsonl
    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.report --seq bar3/scale/records/seq_*_L1.jsonl
    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.report --detail bar3/scale/records/X.jsonl

--matrix: target × config, the round a general program came in (or "no, R rounds") and the search-hours.
--seq: a sequence target by target. --detail: each round's picks (and their sources), the pieces the general
program uses, and the target and guided searches' reach.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

from bar3.breadth.report import load, rounds_of


def hours(rec):
    return sum(v["seconds"] for v in rec["totals"].values()) / 3600


def outcome(rounds):
    for r in rounds:
        if r["general"]:
            return f"round {r['round']}"
    return f"no, {len(rounds)} rounds"


def split(path):
    """(config, target, tag) from <config>_<target>_<tag>.jsonl."""
    m = re.match(r"(.+)_(h1|rho|minus1|plus3|ten|tenrho|cube|mont|affine|fib)_([A-Za-z0-9]+)$", Path(path).stem)
    return m.groups() if m else (Path(path).stem, "", "")


def matrix(paths):
    cells, targets, configs = {}, [], []
    for p in paths:
        cfg, t, _ = split(p)
        if not t:
            continue
        rs = rounds_of(load(p))
        if not rs:
            continue
        cells[(t, cfg)] = f"{outcome(rs)} ({hours(rs[-1]):.2f} h)"
        targets += [t] if t not in targets else []
        configs += [cfg] if cfg not in configs else []
    print("| target | " + " | ".join(configs) + " |")
    print("|---|" + "---|" * len(configs))
    for t in targets:
        print(f"| {t} | " + " | ".join(cells.get((t, c), "–") for c in configs) + " |")


def seq(paths):
    for p in paths:
        recs = load(p)
        by = collections.OrderedDict()
        for r in recs:
            if "round" in r:
                by.setdefault(r["target_name"], []).append(r)
        print(Path(p).stem)
        for t, rs in by.items():
            g = next((r["general"] for r in rs if r["general"]), None)
            print(f"  {t:7s} {outcome(rs):14s} {sum(hours(r) for r in rs[-1:]):6.2f} h  lib {rs[0].get('carried_lib')}"
                  f"  {g['program'] if g else ''}")


def detail(path):
    for r in rounds_of(load(path)):
        src = {c["name"]: c["source"] for c in r["pool"]}
        picks = [f"{n} ({src.get(n, '?')}, cost {r['status'].get(n, {}).get('cost')})" for n in r["picked"]]
        t = r.get("target") or {}
        g = r.get("guided") or {}
        print(f"round {r['round']}: pool {len(r['pool'])}, learnable "
              f"{sum(1 for v in r['status'].values() if v.get('status') == 'learnable')}; {hours(r):.2f} h so far")
        print("   picked: " + "; ".join(picks))
        print(f"   target: found {t.get('found_cost')} levels {t.get('level_sizes')} best {t.get('best_fit')}")
        if g:
            print(f"   guided: found {g.get('found_cost')} bits, {len(g.get('level_sizes', []))} levels, "
                  f"{sum(g.get('level_sizes', []))} states")
        if r["general"]:
            print(f"   GENERAL: {r['general']['program']}  =  {r['general']['primitive']}")


def main():
    mode, paths = sys.argv[1], [p for p in sys.argv[2:] if not Path(p).name.startswith("store_")]   # not records
    {"--matrix": matrix, "--seq": seq}.get(mode, lambda ps: [detail(p) for p in ps])(paths)


if __name__ == "__main__":
    main()
