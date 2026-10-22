"""Primitive-only search against search with the learned library, task by task (LIBRARY.md §5.6).

    ~/github/sotaku/venv/bin/python -m library.compare records/bpe_r3.jsonl records/bpe_big_r5.jsonl ...

Level 0 of a wake-sleep record is primitive-only search at that record's budget (no library yet). Later levels search
with the library. For each task: the cost found and the states the search stored or streamed, for primitives alone
(level 0) and for the level that solved it, with the solving program and its length in primitive tokens.
"""

from __future__ import annotations

import json
import sys

from library.tasks import TASKS


def load(path):
    return [json.loads(l) for l in open(path)]


def by_task(recs):
    """task -> (level, record) of the level that solved it; and task -> level-0 record."""
    solved, first = {}, {}
    for rec in recs:
        for r in rec["records"]:
            if not isinstance(r, dict):
                continue
            if rec["level"] == 0:
                first[r["task"]] = r
            b = next((p for p in r.get("programs", []) if p["verified"]), None)
            if b and r["task"] not in solved:
                solved[r["task"]] = (rec["level"], r, b)
    return solved, first


def fmt_states(r):
    n = sum(r["level_sizes"])
    return f"{n / 1e6:.1f}M" if n >= 1e5 else f"{n:,}"


def summary(paths):
    """One line per run: tasks solved, the longest program found in primitives, library size, search effort."""
    print("| run | levels | tasks solved (of 26; 17 in r1) | longest program found (primitives) | library entries | distinct states searched | search hours (2-CPU containers) |")
    print("|---|---|---|---|---|---|---|")
    for p in paths:
        recs = load(p)
        solved, _ = by_task(recs)
        longest = max(((b["primitive_cost"], t) for t, (lv, r, b) in solved.items()), default=(0, ""))
        rs = [r for rec in recs for r in rec["records"] if isinstance(r, dict)]
        states = sum(sum(r["level_sizes"]) for r in rs)
        hours = sum(r.get("wall", r["seconds"]) for r in rs) / 3600
        name = p.split("/")[-1].replace(".jsonl", "")
        print(f"| {name} | {len(recs)} | {len(solved)} | {longest[0]} ({longest[1]}) | {len(recs[-1]['lib'])} | "
              f"{states / 1e9:.2f}G | {hours:.1f} |")


def main():
    if sys.argv[1] == "--summary":
        return summary(sys.argv[2:])
    paths = sys.argv[1:]
    runs = [(p.split("/")[-1].replace(".jsonl", ""), *by_task(load(p))) for p in paths]
    head = "| task |" + "".join(f" {n}: primitives alone | {n}: with the library |" for n, _, _ in runs)
    print(head)
    print("|---|" + "---|---|" * len(runs))
    for t in TASKS:
        row = f"| {t} |"
        for name, solved, first in runs:
            f = first.get(t)
            if f is None:
                row += " | |"
                continue
            if f.get("found_cost") is not None and any(p["verified"] for p in f.get("programs", [])):
                row += f" cost {f['found_cost']}, {fmt_states(f)} states, {f['seconds']:.0f} s | (level 0) |"
                continue
            row += f" not found ({fmt_states(f)} states, {f['seconds']:.0f} s) |"
            if t in solved:
                lv, r, b = solved[t]
                row += (f" level {lv}: `{b['program']}`, cost {r['found_cost']} ({b['primitive_cost']} primitive), "
                        f"{fmt_states(r)} states, {r['seconds']:.0f} s |")
            else:
                row += " not found |"
        print(row)


if __name__ == "__main__":
    main()
