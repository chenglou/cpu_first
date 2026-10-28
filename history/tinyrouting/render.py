"""Print the tables of TINYROUTING.md from the rows in runs/ (to paste over the placeholders).

    python -m tinyrouting.render
"""

from __future__ import annotations

import contextlib
import io
import json
from collections import defaultdict

import numpy as np

from tinyrouting import summarize as S


def capture(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **kw)
    return buf.getvalue()


def tables(text):
    """The markdown tables in a block of text, in order."""
    out, cur = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            out.append("\n".join(cur))
            cur = []
    if cur:
        out.append("\n".join(cur))
    return out


def composite(path="runs/composite_local.jsonl"):
    rows = [json.loads(l) for l in open(S.RUNS / path.split("/")[-1])]
    d = defaultdict(list)
    for r in rows:
        c = r["cfg"]
        d[("_".join(r["name"].split("_")[:3]), (c["B"], c["w"]))].append(r)
    out = {}
    for (key, size), rs in d.items():
        ok = [r for r in rs if r["exact"] == 1.0 and r["true_wiring"]]
        hits = sorted(r["hit"] for r in ok)
        bad = [r["exact"] for r in rs if not (r["exact"] == 1.0 and r["true_wiring"])]
        cell = f"{len(ok)}/{len(rs)}"
        if hits:
            cell += f" ({hits[0]}–{hits[-1]})" if hits[0] != hits[-1] else f" ({hits[0]})"
        if bad:
            cell += f"; others {min(bad):.2f}–{max(bad):.2f} right" if len(bad) > 1 else f"; other {bad[0]:.2f} right"
        out[(key, size)] = cell
    return out


def main():
    print("PHASE (cell, ans):")
    for t in tables(capture(S.phase)):
        print(t, "\n")
    print("ALONE:")
    for t in tables(capture(S.alone)):
        print(t, "\n")
    print("TRICKS (ans, cell):")
    for t in tables(capture(S.tricks)):
        print(t, "\n")
    for name in ("tricks2", "stages", "untie", "depth", "school"):
        print(name.upper() + ":")
        try:
            for t in tables(capture(getattr(S, name))):
                print(t, "\n")
        except Exception as e:                                                  # suite not run yet
            print("  (none)", repr(e)[:100])
    print("CENSUS:")
    print(capture(S.census))
    print("COMPOSITE:")
    for k, v in sorted(composite().items()):
        print(k, v)


if __name__ == "__main__":
    main()
