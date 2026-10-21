"""MENUS.md's tables from the tiny-core rows (runs/<suite>.jsonl) and the exact counts (runs/exact/*.json).

    ~/github/sotaku/venv/bin/python -m wider_menus.summarize            # prints every table (markdown)
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def rows(suite):
    p = RUNS / f"{suite}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def sci(v):
    e = len(str(int(v))) - 1
    return f"{v / 10 ** e:.1f}e{e}"


def span(v):
    v = sorted(v)
    if not v:
        return "-"
    return f"{statistics.median(v):.0f} ({v[0]}-{v[-1]})" if len(v) > 1 else str(v[0])


def cell_table(rs, key, title):
    cells = collections.defaultdict(list)
    for r in rs:
        cells[key(r)].append(r)
    out = [f"| {title} | menu: division / multiply / total | certified | steps to the final wiring, median (range) | "
           f"layout found at | chain lesson passes at | division chain chosen | multiply grouping chosen | "
           f"failures |", "|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(cells):
        c = cells[k]
        s = c[0]["sizes"]
        ok = [r for r in c if r["certified"]]
        chains = collections.Counter(r["classes"]["chain"] for r in c)
        nums = collections.Counter(r["classes"]["numbers"] for r in c)
        bad = [r for r in c if not r["certified"]]
        fails = "; ".join(fail_label(r) for r in bad[:2]) + (f"; {len(bad) - 2} more alike" if len(bad) > 2 else "")
        out.append(f"| {k} | {sci(s['div'])} / {sci(s['mul'])} / {sci(s['total'])} | {len(ok)}/{len(c)} | "
                   f"{span([r['settled_at'] for r in ok])} | {span([r['events'].get('layout', 0) for r in c])} | "
                   f"{span([r['events']['chain'] for r in c if 'chain' in r['events']])} | "
                   f"{', '.join(f'{n} {v}' for v, n in chains.most_common())} | "
                   f"{len(nums)} kind{'s' if len(nums) != 1 else ''} ({', '.join(f'{n} {v}' for v, n in nums.most_common(3))}) | "
                   f"{fails or '-'} |")
    return "\n".join(out)


def fail_label(r):
    w = r["wiring"]
    io = f"start {w['start']}, bring {w['bring']}, final {w['final']}"
    return (f"{r['name']}: {r['pool_exact']:.2f} of training rows right at step {r['steps']} "
            f"(chain {r['classes']['chain']}; {io})")


def main():
    g = rows("grow")
    print("## grow\n")
    print(cell_table(g, lambda r: f"D{r['cfg']['div']} M{r['cfg']['mul']}", "levels"))
    for suite in ("io", "laws", "w4"):
        rs = rows(suite)
        if rs:
            print(f"\n## {suite}\n")
            print(cell_table(rs, lambda r: r["name"].rsplit("_s", 1)[0], "variant"))
    print("\n## exact\n")
    for f in sorted((RUNS / "exact").glob("*.json")):
        d = json.loads(f.read_text())
        print(f"### {f.stem} ({d['seconds']} s)")
        for part in ("chain", "io_truth", "io_trial", "joint", "mul"):
            if part in d:
                p = {k: v for k, v in d[part].items() if k not in ("passing", "log")}
                print(f"- {part}: {json.dumps(p)}")


if __name__ == "__main__":
    main()
