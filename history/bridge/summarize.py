"""Tables from the sweep records (bridge/runs/<suite>.jsonl).

    ~/github/sotaku/venv/bin/python -m bridge.summarize ladder

A cell is "certified / runs (median step at which every training answer first came out right, over the certified
runs)", and in brackets the median training share of the runs that did not certify. "fit" counts runs exact on
training but wrong on a bigger size.
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


def cell(rs):
    if not rs:
        return "·"
    cert = [r for r in rs if r.get("certified")]
    fit = [r for r in rs if not r.get("certified") and r.get("train_hard") == 1.0]
    rest = [r for r in rs if not r.get("certified")]
    s = f"{len(cert)}/{len(rs)}"
    if cert:
        s += f" ({int(np.median([r['hit'] for r in cert]))})"
    if fit:
        s += f" fit{len(fit)}"
    if rest:
        s += f" [{np.median([r['train_hard'] for r in rest]):.2f}]"
    return s


def table(rows, notches, tiers, sizes):
    by = defaultdict(list)
    for r in rows:
        by[(r["notch"], r["tier"], r["size"])].append(r)
    head = "| notch | lessons | " + " | ".join(sizes) + " |"
    out = [head, "|" + "---|" * (len(sizes) + 2)]
    for n in notches:
        for t in tiers.get(n, ()):
            cells = [cell(by[(n, t, s)]) for s in sizes]
            if all(c == "·" for c in cells):
                continue
            out.append(f"| {n} | {t} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def tier_of(r):
    t = r["tier"]
    cfg = r["cfg"]
    if cfg.get("route_freeze") and "+f" not in t:
        t += "+f"
    if cfg.get("link_stages") and "+s" not in t:
        t += "+s"
    return t


def main():
    suite = sys.argv[1] if len(sys.argv) > 1 else "ladder"
    rows = load(suite)
    for r in rows:
        r["tier"] = tier_of(r)
    notches = list(dict.fromkeys(r["notch"] for r in rows))
    tiers = defaultdict(list)
    for r in rows:
        if r["tier"] not in tiers[r["notch"]]:
            tiers[r["notch"]].append(r["tier"])
    sizes = sorted({r["size"] for r in rows})
    print(f"{len(rows)} records\n")
    print(table(rows, notches, tiers, sizes))


if __name__ == "__main__":
    main()
