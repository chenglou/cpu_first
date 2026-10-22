"""Tables from the sweeps' records (runs/<suite>.jsonl), in the attention bridge's format.

    python -m handwired_bridge.summarize ladder width ...

A cell reads "certified/runs (median step at which the training answers first all came out right)"; "fitN" counts runs
exact on training but wrong on a bigger size; the bracket is the median training share of the runs that did not
certify.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def rows(suite):
    path = HERE / "runs" / f"{suite}.jsonl"
    if not path.exists():
        return []
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            out[r["name"]] = r
    return list(out.values())


def cell(rs):
    if not rs:
        return "·"
    cert = [r for r in rs if r.get("certified")]
    fit = [r for r in rs if not r.get("certified") and r.get("train_hard") == 1.0]
    s = f"{len(cert)}/{len(rs)}"
    if cert:
        s += f" ({int(np.median([r['hit'] for r in cert]))})"
    if fit:
        s += f" fit{len(fit)}"
    rest = [r["train_hard"] for r in rs if not r.get("certified")]
    if rest:
        s += f" [{np.median(rest):.2f}]"
    return s


def table(rs, key=lambda r: (r["notch"], r["tier"]), col=lambda r: r["size"]):
    groups = defaultdict(list)
    cols = []
    for r in rs:
        groups[(key(r), col(r))].append(r)
        if col(r) not in cols:
            cols.append(col(r))
    keys = []
    for r in rs:
        if key(r) not in keys:
            keys.append(key(r))
    lines = ["| " + " | ".join(["notch", "lessons"] + [str(c) for c in cols]) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for k in keys:
        lines.append("| " + " | ".join(list(k) + [cell(groups[(k, c)]) for c in cols]) + " |")
    return "\n".join(lines)


def width_table(rs):
    """Multiplexer x options per port, for each tier."""
    out = []
    for tier in ("A", "AL"):
        sub = [r for r in rs if r["tier"] == tier]
        mux = lambda r: r["notch"].split("_given_")[1]
        n = lambda r: "7" if r["notch"].startswith("H3_menu_given") else r["notch"].split("_")[1][3:]
        out.append(f"**{tier}**\n\n" + table(sub, key=lambda r: (mux(r), tier), col=n))
    return "\n\n".join(out)


if __name__ == "__main__":
    for suite in sys.argv[1:] or ["ladder", "width"]:
        rs = rows(suite)
        print(f"## {suite} ({len(rs)} records)\n")
        print(width_table(rs) if suite == "width" else table(rs))
        print()
