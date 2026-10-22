"""The per-task table of LIBRARY.md §5.6: primitive search at both budgets against the cheapest library search.

    ~/github/sotaku/venv/bin/python -m library.table_tasks
"""

from __future__ import annotations

import json
from pathlib import Path

from library.compare import by_task, load
from library.tasks import TASKS

REC = Path(__file__).resolve().parent / "records"
SMALL, BIG = "bpe_r3", "bpe_big_r5"
LIBRARY_RUNS = ("bpe_r3", "mdl_half_r6", "mdlh_half_r6", "bpe_big_r5", "bpe_big_mb3_r8", "bpe_big_routines_r11")


def fmt(n):
    return f"{n / 1e6:.1f}M" if n >= 1e5 else f"{n:,}"


def main():
    runs = {}
    for name in LIBRARY_RUNS:
        p = REC / f"{name}.jsonl"
        if p.exists():
            runs[name] = by_task(load(p))
    print("| task | shortest program found (primitives) | primitives alone, 3M | primitives alone, 16M | "
          "cheapest search with a library: run, level, program, states, s |")
    print("|---|---|---|---|---|")
    for t in TASKS:
        cells = [t]
        lens = [b["primitive_cost"] for s, _ in runs.values() if t in s for b in [s[t][2]]]
        cells.append(str(min(lens)) if lens else "—")
        for name in (SMALL, BIG):
            f = runs[name][1].get(t) if name in runs else None
            if f and f.get("found_cost") is not None and any(p["verified"] for p in f.get("programs", [])):
                cells.append(f"cost {f['found_cost']}, {fmt(sum(f['level_sizes']))}, {f['seconds']:.0f} s")
            elif f:
                cells.append(f"no ({fmt(sum(f['level_sizes']))}, {f['seconds']:.0f} s)")
            else:
                cells.append("")
        best = None
        for name, (solved, _) in runs.items():
            if t in solved and solved[t][0] > 0:
                lv, r, b = solved[t]
                key = (sum(r["level_sizes"]), r["seconds"])
                if best is None or key < best[0]:
                    best = (key, f"{name} L{lv}: `{b['program']}` (cost {r['found_cost']}), "
                                 f"{fmt(sum(r['level_sizes']))}, {r['seconds']:.0f} s")
        cells.append(best[1] if best else ("(level 0)" if lens else "not found"))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
