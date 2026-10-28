"""Tables for NCA.md from the run records (nca/runs/<suite>.jsonl).

    ~/github/sotaku/venv/bin/python -m nca.summarize explore ladder2 square1

Per arm (the record's tag): certified / runs (median step at which the training answers first all came out
right), then the medians of the snapped and soft training shares and of the two bigger sizes' shares, and the probes'
medians: the subtraction table's right entries (of 2 B^2 digits and 2 B^2 borrows), the share of training answers
equal to 0, and the share of scratch cells constant over every question.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(suite):
    p = HERE / "runs" / f"{suite}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def fmt(x, d=2):
    return "·" if x is None else f"{x:.{d}f}"


def table(rows, order=None):
    by = defaultdict(list)
    for r in rows:
        by[r["cfg"]["tag"]].append(r)
    tags = order or sorted(by)
    out = ["| arm | certified | training, snapped / soft | bigger sizes, snapped | sub table | answers 0 | constant "
           "cells |", "|---|---|---|---|---|---|---|"]
    for t in tags:
        rs = by.get(t, [])
        if not rs:
            continue
        cert = sum(bool(r.get("certified")) for r in rs)
        hits = [r["hit"] for r in rs if r.get("certified")]
        c = f"{cert}/{len(rs)}" + (f" ({int(med(hits))})" if hits else "")
        tr = f"{fmt(med([r['train_hard'] for r in rs]))} / {fmt(med([r['train_soft'] for r in rs]))}"
        keys = list(rs[0].get("bigger", {}))
        big = " / ".join(fmt(med([r.get("bigger", {}).get(k) for r in rs])) for k in keys)
        p = [r["probe"] for r in rs]
        sub = ("·" if p[0].get("sub_digits_right") is None else
               f"{fmt(med([x['sub_digits_right'] for x in p]), 0)} + {fmt(med([x['sub_borrows_right'] for x in p]), 0)}")
        out.append(f"| {t} | {c} | {tr} | {big} | {sub} | {fmt(med([x['answers_0'] for x in p]))} | "
                   f"{fmt(med([x['const_scratch_cells'] for x in p]))} |")
    return "\n".join(out)


def main(argv):
    for suite in argv or ["explore"]:
        rows = load(suite)
        print(f"## {suite} ({len(rows)} records)\n")
        print(table(rows))
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
