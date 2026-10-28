"""Tables of the tiny suites (runs/<suite>.jsonl) for WIRING.md.

    ~/github/sotaku/venv/bin/python -m learned_wiring.tiny.summarize_tiny s1b s2 s3
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def rows(suite):
    p = RUNS / f"{suite}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def group(rs):
    by = collections.defaultdict(list)
    for r in rs:
        by[r["name"].rsplit("_s", 1)[0]].append(r)
    return by


def span(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return "-"
    return f"{vals[0]}" if vals[0] == vals[-1] else f"{vals[0]}-{vals[-1]}"


def s1b():
    print("| variant | certified | true wiring | table exact at step | chain switches right after (steps) | "
          "I/O switches right after (steps) | failures (table entries right / 400) |")
    print("|---|---|---|---|---|---|---|")
    for name, rs in group(rows("s1b")).items():
        cert = sum(r["certified"] for r in rs)
        tw = sum(r["true_wiring"] for r in rs)
        h = [dict((p, s) for s, p in r["hist"]) for r in rs]
        chain = [d.get("chain") for d in h]
        io = [d["io"] - d["chain"] if "io" in d and "chain" in d else None for d in h]
        done = [d["done"] - d["io"] if "done" in d and "io" in d else None for d in h]
        fails = [str(r["table"]) for r in rs if not r["certified"]]
        print(f"| {name} | {cert}/{len(rs)} | {tw}/{len(rs)} | {span(chain)} | {span(io)} | {span(done)} | "
              f"{', '.join(fails) or '-'} |")


def s2():
    print("| layout | W | cell ladder | certified (square exact) | true layout | stopped at step |")
    print("|---|---|---|---|---|---|")
    for name, rs in sorted(group(rows("s2")).items()):
        c = rs[0]["cfg"]
        cert = sum(r["certified"] for r in rs)
        tl = sum(r["true_layout"] for r in rs)
        print(f"| {c['route']}{' (one row per number per step)' if c.get('rows_per_step') else ''} | {c['W']} | "
              f"{'yes' if c['cell_ladder'] else 'no'} | {cert}/{len(rs)} | {tl}/{len(rs)} | "
              f"{span([r['stop'] for r in rs])} |")


def s2b():
    print("| layout | trained on numbers below | certified (W = 4 squares exact) | true layout | stopped at step | "
          "square exact (failures) |")
    print("|---|---|---|---|---|---|")
    for name, rs in sorted(group(rows("s2b")).items()):
        c = rs[0]["cfg"]
        fails = [str(round(r["square_exact"], 3)) for r in rs if not r["certified"]]
        print(f"| {c['route']} | {10 ** c['train_digits'] if c['train_digits'] else 10 ** c['W']} | "
              f"{sum(r['certified'] for r in rs)}/{len(rs)} | {sum(r['true_layout'] for r in rs)}/{len(rs)} | "
              f"{span([r['stop'] for r in rs])} | {', '.join(fails) or '-'} |")


def s3b():
    s3("s3b")


def s3(suite="s3"):
    print("| shapes offered | certified (every field exact on the T = 1, 16, 32, 64 rungs, both profiles) | "
          "shapes chosen (N, x, T, writer) | N exact | x exact | T exact | writer exact | every field exact on the "
          "8-digit-N prompts | stopped at step |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, rs in group(rows(suite)).items():
        cert = sum(r["certified"] for r in rs)
        ch = collections.Counter("/".join(r["shapes"]) for r in rs)
        f = lambda k: span([round(r[k], 3) for r in rs])
        print(f"| {rs[0]['cfg']['shapes']} | {cert}/{len(rs)} | "
              f"{'; '.join(f'{k} x{v}' for k, v in ch.items())} | {f('N')} | {f('x')} | {f('T')} | "
              f"{f('writer')} | {f('N_8digit')} | {span([r['stop'] for r in rs])} |")


if __name__ == "__main__":
    for s in sys.argv[1:] or ["s1b", "s2", "s2b", "s3", "s3b"]:
        print(f"\n## {s}\n")
        globals()[s]()
