"""A scale run against the breadth run it repeats (SCALE.md §1): round by round, are the pool, every probe's outcome,
the picks, the library and the target's programs the same, and what did each cost in search-hours?

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compare bar3/scale/records/ct2a2_clean_h1_s1.jsonl ...

Each new run `<config>_<target>_<tag>` is matched with bar3/breadth/records/<config>_<target>_b1.jsonl (or `--old`
for one pair). A probe outcome is (status, cost, the routine it keeps); a probe whose original search stopped on its
time limit is listed apart, since a faster search can get further within the same state budget.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from bar3.breadth.report import load, rounds_of

ROOT = Path(__file__).resolve().parents[2]


def hours(rec):
    return sum(v["seconds"] for v in rec["totals"].values()) / 3600


def step_hours(rounds, i):
    """A round's own search-hours: totals run on within one target's run and start again with the next target."""
    r = rounds[i]
    prev = rounds[i - 1] if i else None
    same_run = prev is not None and prev.get("target_name") == r.get("target_name") and prev["round"] < r["round"]
    return hours(r) - (hours(prev) if same_run else 0)


def outcome(st):
    return (st.get("status"), st.get("cost"), st.get("best"))


def target_progs(rec):
    t = rec.get("target") or {}
    return t.get("found_cost"), sorted(p["program"] for p in t.get("programs", []))


def compare(new_path, old_path):
    new, old = rounds_of(load(new_path)), rounds_of(load(old_path))
    rows = []
    for i, (a, b) in enumerate(zip(new, old)):
        diffs = []
        if [c["name"] for c in a["pool"]] != [c["name"] for c in b["pool"]]:
            diffs.append("pool")
        timed = {n for n, s in b["status"].items() if str(s.get("stop", "")).startswith("time")}
        changed = {n for n in set(a["status"]) | set(b["status"])
                   if outcome(a["status"].get(n, {})) != outcome(b["status"].get(n, {}))}
        if changed - timed:
            diffs.append(f"{len(changed - timed)} probe outcomes")
        for f in ("picked", "solved", "lib"):
            if a[f] != b[f]:
                diffs.append(f)
        if target_progs(a) != target_progs(b):
            diffs.append("target")
        if (a["general"] or {}).get("program") != (b["general"] or {}).get("program"):
            diffs.append("general")
        rows.append(dict(round=i, same=not diffs, diffs=diffs, timed=len(timed), timed_changed=len(changed & timed),
                         new_h=step_hours(new, i), old_h=step_hours(old, i), picked=a["picked"]))
    return rows, len(new), len(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--old", default="")
    a = ap.parse_args()
    tot_new = tot_old = 0.0
    for path in [p for p in a.runs if not Path(p).name.startswith("store_")]:      # a search store, not a run
        name = Path(path).stem
        base = re.sub(r"_[a-z0-9]+$", "", name)
        old = a.old or str(ROOT / "bar3" / "breadth" / "records" / f"{base}_b1.jsonl")
        if not Path(old).exists():
            print(f"{name}: no breadth run {old}")
            continue
        rows, n_new, n_old = compare(path, old)
        hn, ho = sum(r["new_h"] for r in rows), sum(r["old_h"] for r in rows)
        tot_new, tot_old = tot_new + hn, tot_old + ho
        verdict = "same" if all(r["same"] for r in rows) and n_new == n_old else "DIFFERS"
        print(f"{name}: {verdict}; rounds {n_new} vs {n_old}; search-hours {hn:.3f} vs {ho:.2f} "
              f"({ho / max(hn, 1e-9):.0f}x)")
        for r in rows:
            if not r["same"] or r["timed"]:
                print(f"   round {r['round']}: {r['diffs'] or 'same'}; {r['timed']} probes hit the time limit in "
                      f"the original, {r['timed_changed']} of them came out differently")
    print(f"total search-hours {tot_new:.2f} vs {tot_old:.2f} ({tot_old / max(tot_new, 1e-9):.1f}x)")


if __name__ == "__main__":
    main()
