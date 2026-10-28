"""Markdown tables for LADDER.md from microlab/ladder_results.jsonl.

    python -m microlab.ext_ladder_report --tags S3k,S12k

- breaking points: largest size whose best seed reaches 99% exact, climbing from the smallest size without a gap;
  later budgets (e.g. S12k, which restarts each ladder at the S3k breaking point) are shown after an arrow.
- the breaking cell: per-digit accuracy (lo->hi, mean of seeds) at the first size that fails, against the
  majority-class baseline, with the share of test inputs that also occurred in training and exact match on the rest.
"""

from __future__ import annotations

import argparse

import numpy as np

from .ext_ladder import PASS, SIZES, load
from .tasks import TASKS

TASK_ORDER = ["add", "mul", "sq", "mod", "sqmod", "powmod2", "powmod4"]
COLS = [(m, d) for m in ("mlp", "xf", "loop") for d in ("fresh", "pool1000")]


def by_cell(rows):
    out = {}
    for r in rows:
        out.setdefault((r["task"], r["model"], r["data"]), {}).setdefault(r["size"], []).append(r)
    return out


def top_size(by_size, sizes):
    """(largest passing size of the unbroken run, first failing size). A ladder that starts above the smallest size
    (a later budget restarted at an earlier breaking point) counts the sizes below its start as passed."""
    lo = sizes.index(min(by_size))
    top = sizes[lo - 1] if lo else None
    for s in sizes[lo:]:
        if s not in by_size:
            return top, None
        if max(r["exact"] for r in by_size[s]) < PASS:
            return top, s
        top = s
    return top, None


def fmt_top(top, fail, sizes):
    if top is None:
        return f"<{sizes[0]}"
    return f"≥{top}" if fail is None and top == sizes[-1] else str(top)


def breaking_table(tags):
    cells = [by_cell(load(tag=t)) for t in tags]
    lines = ["| task (size) | " + " | ".join(f"{m} {d.replace('pool1000', 'pool 1k')}" for m, d in COLS) + " |",
             "|---|" + "---|" * len(COLS)]
    for task in TASK_ORDER:
        unit = "digits" if TASKS[task].kind == "digits" else "bits"
        sizes = SIZES[TASKS[task].kind]
        row = []
        for m, d in COLS:
            parts = [fmt_top(*top_size(c[(task, m, d)], sizes), sizes) for c in cells if (task, m, d) in c]
            row.append(" → ".join(parts) or "·")
        lines.append(f"| {task} ({unit}) | " + " | ".join(row) + " |")
    return "\n".join(lines)


def breaking_cells(tag, models=("mlp", "xf", "loop"), data=("fresh", "pool1000")):
    c = by_cell(load(tag=tag))
    lines = ["| task | model | data | size | exact | seen | exact unseen | digits lo→hi (mean of seeds) | majority baseline |",
             "|---|---|---|---|---|---|---|---|---|"]
    for task in TASK_ORDER:
        sizes = SIZES[TASKS[task].kind]
        for m in models:
            for d in data:
                if (task, m, d) not in c:
                    continue
                top, fail = top_size(c[(task, m, d)], sizes)
                if fail is None:
                    continue
                rs = c[(task, m, d)][fail]
                ex = max(r["exact"] for r in rs)
                seen = np.mean([r["seen_frac"] for r in rs])
                un = [r["exact_unseen"] for r in rs if r["exact_unseen"] is not None]
                dg = np.mean([r["per_digit"] for r in rs], 0)
                lines.append(f"| {task} | {m} | {d} | {fail} | {ex:.3f} | {seen:.2f} | "
                             f"{'—' if not un else f'{np.mean(un):.3f}'} | {' '.join(f'{v:.2f}' for v in dg)} | "
                             f"{' '.join(f'{v:.2f}' for v in rs[0]['baseline'])} |")
    return "\n".join(lines)


def unseen_table(tags, min_unseen=0.25, models=("mlp", "xf", "loop")):
    """Best exact match on test inputs that never occurred in training, over every cell (any size, data regime or
    budget in `tags`) whose test set is at least `min_unseen` unseen, with the cell it came from; the chance column
    is the product of the per-digit majority baselines of that cell."""
    rows = [r for t in tags for r in load(tag=t) if r["exact_unseen"] is not None and 1 - r["seen_frac"] >= min_unseen]
    lines = ["| task | " + " | ".join(f"{m}: best exact on unseen inputs" for m in models) + " |",
             "|---|" + "---|" * len(models)]
    for task in TASK_ORDER:
        out = []
        for m in models:
            rs = [r for r in rows if r["task"] == task and r["model"] == m]
            if not rs:
                out.append("·")
                continue
            r = max(rs, key=lambda r: (r["exact_unseen"], -r["size"]))
            big = max(rs, key=lambda r: (r["size"], r["exact_unseen"]))
            chance = float(np.prod(r["baseline"]))
            out.append(f"{r['exact_unseen']:.2f} at {r['size']} {r['data'].replace('pool1000', 'pool')} "
                       f"(chance {chance:.3f}); at {big['size']}: {big['exact_unseen']:.2f}")
        lines.append(f"| {task} | " + " | ".join(out) + " |")
    return "\n".join(lines)


def beyond_memory(tag="S3k", model="xf", data="pool1000", min_unseen=0.7):
    """Per task, the smallest size whose test inputs are mostly (>= min_unseen) absent from training (else the size
    with the most unseen inputs): exact match and per-digit accuracy on those unseen inputs (mean of seeds) against
    the per-digit majority baseline."""
    c = by_cell(load(tag=tag))
    lines = ["| task@size | test inputs unseen | exact on unseen | unseen digits lo→hi | majority baseline |",
             "|---|---|---|---|---|"]
    for task in TASK_ORDER:
        by = c.get((task, model, data), {})
        unseen = {s: np.mean([1 - r["seen_frac"] for r in rs]) for s, rs in by.items()}
        if not unseen or max(unseen.values()) == 0:
            continue
        ok = [s for s in sorted(by) if unseen[s] >= min_unseen] or [max(unseen, key=unseen.get)]
        rs = by[ok[0]]
        dg = np.mean([r["per_digit_unseen"] for r in rs], 0)
        lines.append(f"| {task}@{ok[0]} | {np.mean([1 - r['seen_frac'] for r in rs]):.2f} | "
                     f"{np.mean([r['exact_unseen'] for r in rs]):.3f} | {' '.join(f'{v:.2f}' for v in dg)} | "
                     f"{' '.join(f'{v:.2f}' for v in rs[0]['baseline'])} |")
    return "\n".join(lines)


def probe_table(task, tags):
    """Every ladder of `task` in `tags`, one row per (tag, model, data, model options): best-seed exact per size,
    with mean exact on unseen inputs after a slash where at least a quarter of the test inputs are unseen."""
    rows = [r for t in tags for r in load(tag=t) if r["task"] == task]
    groups = {}
    for r in rows:
        kw = ",".join(f"{k}={v}" for k, v in sorted(r["model_kw"].items()))
        groups.setdefault((r["tag"], r["model"], kw, r["data"], r["steps"]), {}).setdefault(r["size"], []).append(r)
    sizes = sorted({r["size"] for r in rows})
    lines = ["| run | model | data | steps | " + " | ".join(f"{task}@{s}" for s in sizes) + " |",
             "|---|---|---|---|" + "---|" * len(sizes)]
    for (tag, m, kw, d, st), by in sorted(groups.items(), key=lambda g: (g[0][1], g[0][4], g[0][3], g[0][2])):
        cells = []
        for s in sizes:
            if s not in by:
                cells.append("")
                continue
            rs = by[s]
            un = [r["exact_unseen"] for r in rs if r["exact_unseen"] is not None and 1 - r["seen_frac"] >= 0.25]
            cells.append(f"{max(r['exact'] for r in rs):.2f}" + (f" / {np.mean(un):.2f}" if un else ""))
        lines.append(f"| {tag} | {m}{' ' + kw if kw else ''} | {d} | {st // 1000}k | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ledger(tags):
    """Every ladder, every size: best-seed exact / mean seen share / mean exact on unseen inputs."""
    for t in tags:
        c = by_cell(load(tag=t))
        print(f"## {t}")
        for task in TASK_ORDER:
            for m, d in COLS:
                if (task, m, d) not in c:
                    continue
                parts = []
                for s, rs in sorted(c[(task, m, d)].items()):
                    un = [r["exact_unseen"] for r in rs if r["exact_unseen"] is not None]
                    parts.append(f"{s}:{max(r['exact'] for r in rs):.2f}/{np.mean([r['seen_frac'] for r in rs]):.2f}"
                                 + (f"/{np.mean(un):.2f}" if un else ""))
                print(f"{task:8} {m:4} {d:8} " + "  ".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", default="S3k")
    p.add_argument("--cells", default=None, help="tag whose breaking cells to list (default: first tag)")
    p.add_argument("--ledger", action="store_true", help="print exact/seen/unseen for every cell instead")
    a = p.parse_args()
    tags = a.tags.split(",")
    if a.ledger:
        return ledger(tags)
    print(breaking_table(tags))
    print()
    print(breaking_cells(a.cells or tags[0]))
    print()
    print(unseen_table(tags))


if __name__ == "__main__":
    main()
