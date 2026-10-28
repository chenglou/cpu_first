"""What the tiny tournament's checks let through (TOURNAMENT.md section 3): every combination that passes block
"sub"'s law (the subtraction borrows exactly when x < N on one link) and every pair that passes the one-link law, at
free attention's width, with the truth tables; each classified by what its wrong ports read on the training grid
(tiny.Tiny.trap: an input cell the machine does not read, a bypass; a constant zero; another computed cell), and
whether it is exact on the training answers and on the two bigger sizes.

    ~/github/sotaku/venv/bin/python -m tournament.traps            # about a minute; writes records/traps.json
"""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import torch

from tournament import tiny as T

HERE = Path(__file__).resolve().parent


def kind(trap: dict) -> str:
    """One label per wiring: "true" (no wrong port), "bypass" if some wrong port reads an input cell on some cells,
    else "constant" if some wrong port reads only zeros, else "impostor" (wrong ports read other computed cells)."""
    if not trap:
        return "true"
    if any(t["input"] > 0 for t in trap.values()):
        return "bypass"
    if any(t["zero"] == 1.0 for t in trap.values()):
        return "constant"
    return "impostor"


def main():
    t0 = time.time()
    m = T.Tiny(T.Cfg(tables="given"))
    tabs = m.tables()
    law = m.law_rows(m.data)
    n = m.n
    allc = torch.arange(n ** 4)
    v1 = m.v_sub(allc, law, tabs)
    surv = allc[v1]
    a2 = torch.arange(n ** 3)
    c1, c2 = surv.repeat_interleave(len(a2)), a2.repeat(len(surv))
    m.cfg.answers = False
    vl = m.v_pair(c1, c2, law, tabs, m.data)
    rec = {"options": n, "sub_blocks": n ** 4, "sub_pass": len(surv), "link_pairs": len(c1), "link_pass": int(vl.sum())}
    # block "sub" survivors, each with the true select (their subtraction row is all the law reads)
    sub_kinds = collections.Counter()
    for c in surv.tolist():
        w = T.decode(torch.tensor([c]), n, 4)[0].tolist() + m.true[4:]
        sub_kinds[kind({k: v for k, v in m.trap(w).items() if k.endswith(".SUB") or k == "A.SHIFT"})] += 1
    rec["sub_kinds"] = dict(sub_kinds)
    # the one-link law's survivors: kind, and whether exact on the training answers and the bigger sizes
    link = []
    ok_pairs = torch.stack([c1[vl], c2[vl]], 1)
    for a, b in ok_pairs.tolist():
        w = T.decode(torch.tensor([a]), n, 4)[0].tolist() + T.decode(torch.tensor([b]), n, 3)[0].tolist()
        link.append(w)
    kinds = collections.Counter()
    fit = collections.Counter()
    for w in link:
        kd = kind(m.trap(w))
        kinds[kd] += 1
        ex = m.exact(m.lay, m.data, w, tabs, m.wires)
        fit[(kd, ex == 1.0)] += 1
    rec["link_kinds"] = dict(kinds)
    rec["link_fit"] = {f"{k}|{'fits' if f else 'no'}": v for (k, f), v in fit.items()}
    rec["seconds"] = time.time() - t0
    out = HERE / "records" / "traps.json"
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1), f"-> {out}")


if __name__ == "__main__":
    main()
