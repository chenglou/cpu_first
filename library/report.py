"""Tables for LIBRARY.md from the wake-sleep records.

    ~/github/sotaku/venv/bin/python -m library.report records/bpe_r3.jsonl [more records ...]

For each record: per level, the tasks solved, each one's program in the vocabulary of the moment and its length in
primitives, the search's distinct states and seconds; the merges with their expansions; and a summary line: tasks
solved, the longest program reached (in primitive tokens), the library's size.
"""

from __future__ import annotations

import json
import sys


def best(r):
    return next((p for p in r.get("programs", []) if p["verified"]), None)


def summarize(path):
    recs = [json.loads(l) for l in open(path)]
    if not recs:
        return None
    name = recs[0]["config"]
    solved, longest, states, secs = {}, (0, ""), 0, 0.0
    lines = [f"### {name}", "", "| level | task | program | primitives | states searched | s |", "|---|---|---|---|---|---|"]
    for rec in recs:
        for r in rec["records"]:
            if not isinstance(r, dict):
                continue
            states += sum(r["level_sizes"])
            secs += r["seconds"]
            b = best(r)
            if b:
                solved[r["task"]] = rec["level"]
                if b["primitive_cost"] > longest[0]:
                    longest = (b["primitive_cost"], r["task"])
                lines.append(f"| {rec['level']} | {r['task']} | `{b['program']}` | {b['primitive_cost']} | "
                             f"{sum(r['level_sizes']):,} | {r['seconds']:.0f} |")
        for m in rec["merges"]:
            lines.append(f"| {rec['level']} | + **{m['name']}** | `{m['def']}` | = `{m.get('expands_to', '')}` | | |")
    last = recs[-1]
    lines.append("")
    lines.append(f"{len(solved)} tasks solved, longest program {longest[0]} primitive tokens ({longest[1]}), "
                 f"library {len(last['lib'])} entries, {states:,} distinct states and {secs / 60:.0f} search-minutes "
                 f"over {len(recs)} levels.")
    unsolved = [r["task"] for r in recs[-1]["records"] if isinstance(r, dict) and not best(r)]
    if unsolved:
        lines.append(f"Unsolved at the last level: {', '.join(unsolved)}.")
    return dict(name=name, solved=solved, longest=longest, lib=last["lib"], text="\n".join(lines))


def main():
    for p in sys.argv[1:]:
        s = summarize(p)
        if s:
            print(s["text"])
            print()


if __name__ == "__main__":
    main()
