"""Print an inventing-learner record level by level.

    ~/github/sotaku/venv/bin/python -m library.inventing.summarize library/inventing/records/C_mdl_r1.jsonl
    ~/github/sotaku/venv/bin/python -m library.inventing.summarize --table library/inventing/records/*_r[0-9].jsonl
"""

from __future__ import annotations

import json
import sys


def table(paths):
    """One line per run: how far it got, whether x^2 mod N came out general, states, search-hours."""
    print("| run | levels | sizes | x² mod N | states | search-hours |\n|---|---|---|---|---|---|")
    for path in paths:
        recs = [json.loads(l) for l in open(path)]
        if not recs:
            continue
        rs = [r for rec in recs for r in rec["records"] if isinstance(r, dict)]
        states = sum(sum(r["level_sizes"]) for r in rs)
        hours = sum(r.get("wall", r["seconds"]) for r in rs) / 3600
        gen = next((rec["general"] for rec in recs if rec.get("general")), None)
        found = (f"general at level {next(rec['level'] for rec in recs if rec.get('general'))}: {gen['program']}"
                 if gen else "no")
        print(f"| {recs[0]['config']} | {len(recs)} | {', '.join(str(rec['k']) for rec in recs)} | {found} | "
              f"{states / 1e6:,.1f}M | {hours:.2f} |")


def main():
    if sys.argv[1] == "--table":
        return table(sys.argv[2:])
    for path in sys.argv[1:]:
        recs = [json.loads(l) for l in open(path)]
        if not recs:
            continue
        print(f"### {recs[0]['config']}")
        states = secs = 0
        for rec in recs:
            t = next((r for r in rec["records"] if isinstance(r, dict) and r["task"] == "target"), None)
            dr = [r for r in rec["records"] if isinstance(r, dict) and r["task"] == "dream"]
            for r in rec["records"]:
                if isinstance(r, dict):
                    states += sum(r["level_sizes"])
                    secs += r.get("wall", r["seconds"])
            if t:
                good =[p for p in t.get("programs", []) if p["verified"]]
                best = max(good, key=lambda p: p.get("general", 0)) if good else None
                print(f"L{rec['level']} k={rec['k']}: target cost {t['found_cost']} states {sum(t['level_sizes']):,} "
                      f"{t['seconds']:.0f}s {t['stop']}"
                      + (f" | {best['program']} = {best['primitive']} (general {best.get('general')})" if best else ""))
            for r in rec["records"]:
                if isinstance(r, dict) and r["task"] == "curated":
                    good = [p for p in r.get("programs", []) if p["verified"]]
                    if good:
                        p = min(good, key=lambda p: p["primitive_cost"])
                        print(f"   {r['name'].rsplit('-', 1)[1]:7s} cost {r['found_cost']}: {p['program']} = {p['primitive']}")
                    else:
                        print(f"   {r['name'].rsplit('-', 1)[1]:7s} not found ({sum(r['level_sizes']):,} states)")
            if dr:
                solved = sum(1 for r in dr if any(p["verified"] for p in r.get("programs", [])))
                print(f"   dreams: {len(dr)} posed, {solved} solved; e.g. {[d['program'] for d in rec['dreams'][:4]]}")
            for e in rec["merges"]:
                print(f"   + {e['name']} = {e['def']}   ({e.get('expands_to', '')})")
            if rec.get("general"):
                print(f"   GENERAL: {rec['general']['program']} = {rec['general']['primitive']}")
        print(f"   total: {states:,} distinct states, {secs / 3600:.2f} search-hours\n")


if __name__ == "__main__":
    main()
