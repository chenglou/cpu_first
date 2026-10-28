"""Tables from the Modal rows (tinyrouting/runs/<suite>.jsonl).

    python -m tinyrouting.summarize phase | alone | tricks | stages | untie | all

A cell reads "c/n (h)": c of n seeds certified (the argmax wiring with the snapped table exact on every training
answer and on both held-out sizes), h the median step at which the training answers became exact (over the certified
seeds). "f" counts seeds exact on the training answers that failed a held-out size.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"


def load(suite):
    p = RUNS / f"{suite}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def cell(rows):
    if not rows:
        return "·"
    n = len(rows)
    c = sum(r["certified"] for r in rows)
    f = sum(1 for r in rows if r["hit"] is not None and not r["certified"])
    hits = [r["hit"] for r in rows if r["certified"]]
    s = f"{c}/{n}"
    if hits:
        s += f" ({int(np.median(hits))})"
    if f:
        s += f" f{f}"
    return s


def table(rows, rkey, ckey, rorder=None, corder=None, title=""):
    grid = defaultdict(list)
    for r in rows:
        grid[(rkey(r), ckey(r))].append(r)
    rs = rorder or sorted({k[0] for k in grid}, key=str)
    cs = corder or sorted({k[1] for k in grid}, key=str)
    if title:
        print(f"\n### {title}\n")
    print("| | " + " | ".join(str(c) for c in cs) + " |")
    print("|---" * (len(cs) + 1) + "|")
    for rk in rs:
        print(f"| {rk} | " + " | ".join(cell(grid.get((rk, ck), [])) for ck in cs) + " |")


def size(r):
    c = r["cfg"]
    return f"({c['B']}, {c['w']}, {c['k']})"


def size_order(rows):
    return sorted({size(r) for r in rows}, key=lambda s: tuple(int(x) for x in s.strip("()").split(", ")))


def phase():
    rows = load("phase")
    params = ["mix", "fmix", "hybrid", "soft", "st"]
    for recipe, title in [("cell", "With the cell check (answers + (a - b) + b = a)"), ("ans", "Answers only")]:
        sel = [r for r in rows if (bool(r["cfg"]["checks"]) == (recipe == "cell"))]
        table(sel, size, lambda r: r["cfg"]["param"], size_order(sel), params, f"phase map: {title}")
    wrong_progs(rows)


def wrong_progs(rows):
    """What the failed runs settled on: the most common wrong options."""
    true = {"vread": "given", "nread": "given", "shift": "up", "a": "prev", "b": "N", "keep": "stop", "read": "low"}
    count = defaultdict(int)
    n = 0
    for r in rows:
        if r["certified"]:
            continue
        n += 1
        for m, v in r["final"]["prog"].items():
            if v != true[m.split("@")[0]]:
                count[f"{m}={v}"] += 1
    if n:
        print(f"\nwrong options in the {n} uncertified runs: " +
              ", ".join(f"{k} {v}" for k, v in sorted(count.items(), key=lambda kv: -kv[1])[:14]))


def alone():
    rows = load("alone")
    params = ["mix", "fmix", "hybrid", "soft", "st", "gumbel", "reinforce"]
    table(rows, size, lambda r: r["cfg"]["param"], size_order(rows), params,
          "the wiring alone (table given exact), 7 muxes")
    wrong_progs(rows)


def tricks():
    rows = load("tricks")
    for regime in ("ans", "cell"):
        sel = [r for r in rows if r["name"].startswith(regime + "_")]
        trick = lambda r: r["name"].split("_")[1]
        col = lambda r: f"{size(r)} {r['cfg']['param']}"
        cols = sorted({col(r) for r in sel})
        table(sel, trick, col, None, cols, f"tricks, {'answers only' if regime == 'ans' else 'with the cell check'}")


def tricks2():
    rows = load("tricks2")
    table(rows, lambda r: r["name"].split("_")[1], lambda r: f"{size(r)} {r['cfg']['param']}", None, None,
          "tricks on the whole-wiring mixture, answers only")


def stages():
    rows = load("stages")
    for recipe in ("ans", "cell"):
        sel = [r for r in rows if f"_{recipe}_" in r["name"]]
        table(sel, lambda r: f"{size(r)} {r['cfg']['param']}", lambda r: r["name"].split("_")[0], None,
              ["joint", "staged"], f"one quotient digit at a time ({recipe})")


def depth():
    rows = load("depth")
    for table_kind in ("exact", "cell"):
        sel = [r for r in rows if r["name"].startswith(table_kind + "_")]
        table(sel, size, lambda r: f"{r['cfg']['param']} {r['name'].split('_')[2]}", size_order(sel),
              ["soft joint", "soft staged", "fmix joint", "fmix staged"],
              f"steps to the wiring against depth, table {'given' if table_kind == 'exact' else 'taught by the cell check'}")


def census():
    rows = load("census")
    for cell in (0, 1):
        sel = [x for r in rows if r["args"]["cell"] == cell for x in r["rows"]]
        if not sel:
            continue
        fit = [x for x in sel if x["train"] == 1.0]
        cert = [x for x in fit if all(h == 1.0 for h in x["held"])]
        tr = np.array([x["train"] for x in sel])
        print(f"\ncensus {'answers + cell' if cell else 'answers only'}: {len(sel)} wirings, {len(fit)} fit every "
              f"training answer with their own table, {len(cert)} also every held-out one; train-exact quantiles "
              f"50/90/99%: {np.quantile(tr, [0.5, 0.9, 0.99]).round(2).tolist()}")
        for x in sorted(fit, key=lambda x: -min(x["held"]))[:12]:
            print(f"   {x['prog']}: held {x['held']}, with the true table {x['true_table']}")


def untie():
    rows = load("untie")
    table(rows, lambda r: f"{size(r)} {r['cfg']['table']}",
          lambda r: f"{r['cfg']['param']} {'+'.join(r['cfg']['untie']) or 'tied'} {'staged' if r['cfg']['stages'] else 'joint'}",
          title="tied vs untied routing")


def school():
    rows = load("school")
    def cellsch(rs):
        if not rs:
            return "·"
        n = len(rs)
        ok = sum(1 for r in rs if r["exact"] == 1.0 and r["true_wiring"])
        alias = sum(1 for r in rs if r["exact"] == 1.0 and not r["true_wiring"])
        hits = [r["hit"] for r in rs if r["exact"] == 1.0 and r["true_wiring"] and r["hit"] is not None]
        out = f"{ok}/{n}"
        if hits:
            out += f" ({int(np.median(hits))})"
        if alias:
            out += f" a{alias}"
        if ok < n:
            bad = [r for r in rs if not (r["exact"] == 1.0 and r["true_wiring"])]
            out += f" [{min(r['exact'] for r in bad):.2f}-{max(r['exact'] for r in bad):.2f}]"
        return out
    grid = defaultdict(list)
    for r in rows:
        c = r["cfg"]
        kind = r["name"].split("_")[0]
        col = f"{kind} {c['labels'] if c['answers'] else 'no labels'}"
        grid[(f"({c['B']}, {c['w']}) {c['route']} {c['param']}", col)].append(r)
    rs_ = sorted({k[0] for k in grid})
    cs = sorted({k[1] for k in grid})
    print("\n### schoolbook core\n")
    print("| | " + " | ".join(cs) + " |")
    print("|---" * (len(cs) + 1) + "|")
    for rk in rs_:
        print(f"| {rk} | " + " | ".join(cellsch(grid.get((rk, ck), [])) for ck in cs) + " |")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"phase": phase, "alone": alone, "tricks": tricks, "tricks2": tricks2, "stages": stages, "untie": untie,
           "depth": depth, "census": census, "school": school}
    for name, f in fns.items():
        if what in (name, "all"):
            f()
