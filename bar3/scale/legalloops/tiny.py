"""The tiny tests (LEGALLOOPS.md §3).

- MODE=read (local, CPU): test 3's g1 climb (bar3/scale/legalclimb/records/g1.json) at the start of its level 3, when
  its routines are the comparison, the link and the addition. The legal reader on the 780 answer rows: which law
  it reads, on how many rows, at what cost in routine calls; and which cuts and loops it would pose for each book.
- MODE=pose (local, CPU): the same state, the pose phase with READ=1 on book lt;red;add (pair): the roots and laws
  posed, and each new law's answers on one draw against Python's integers (a check of the book's constructions).

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 H1_DATA=... READ=1 MODE=read ~/github/sotaku/venv/bin/python -m bar3.scale.legalloops.tiny
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("SPELL", "16")
os.environ.setdefault("READ", "1")

import torch  # noqa: E402

from integration.lib import libcore as L  # noqa: E402
from bar3.scale.legalloops import core_k as C  # noqa: E402
from bar3.scale.legalloops import hooks as H  # noqa: E402
from bar3.scale.legalloops import lread  # noqa: E402
from bar3.scale.legalloops.lclimb import LegalClimb  # noqa: E402
from bar3.scale.legalloops.spcore_k import LAW_BOOK  # noqa: E402

HERE = Path(__file__).resolve().parent
REC = HERE.parent / "legalclimb" / "records" / "g1.json"
env = os.environ.get


def climb_at_level3(dev, ans, laws, patterns):
    rec = json.load(open(REC))
    c = LegalClimb(L.Cells(10, dev), dev, 16, chunk=1 << 16, target=None, rule="lpca", K=4, NP=8, laws=laws,
                   patterns=patterns, deep_by="bindings", deep_focus="closure")
    c.lib = {k: L.parse(v) for k, v in rec["history"][2]["lib"].items()}
    for key, toks in rec["ops"].items():
        law, pat = key.split("@")
        for tok, bd in toks:
            if tok in c.lib:
                c.ops.add(law, pat, tok, tuple(bd))
    for h in rec["history"][:3]:
        for n in h["picks"]:
            c.practiced[n] = h["level"]
    c.level, c.answer_rows = 3, ans
    return c


def main():
    dev = "cpu"
    arows = C.load_rows(int(env("ANSWERS", 780)), seed=1, T=1)
    ans = tuple(C.digits(arows, k) for k in ("modulus", "x", "result"))
    t0 = time.time()
    if env("MODE", "read") == "read":
        c = climb_at_level3(dev, ans, ["lt", "red", "add"], ["pair"])
        print("routines:", sorted(k for k in c.lib if k.startswith("T.")))
        r = lread.Reader(c, ans)
        law, ops, depth, rows = r.read()
        print(f"read: {law} (operations {ops}, depth {depth}) on {rows} rows; {r.calls} routine calls; "
              f"{time.time() - t0:.0f} s")
        for name, laws, pats in (("book less its reduction, row and multiply", ["lt", "red", "add"], ["pair"]),
                                 ("book less its reduction and row", ["lt", "red", "add", "mul"], ["pair", "x,o"])):
            c2 = climb_at_level3(dev, ans, laws, pats)
            c2._lread_tries = 0
            book = [(p, [lw for lw in c2.laws if p in LAW_BOOK[lw][1]]) for p in c2.patterns]
            out = lread.pose(c2, book, LAW_BOOK)
            print(f"{name}: posed {out}; pairs {lread.show_pairs(c2)}; {c2.records[-1]['seconds']} s")
    else:
        rows = C.load_rows(512, seed=0)
        rows = (C.digits(rows, "modulus"), C.digits(rows, "x"))
        c = climb_at_level3(dev, ans, ["lt", "red", "add"], ["pair"])
        pool, roots = c.pose(rows)
        print("pool:", pool)
        print("records:", [x for x in c.records if x.get("task") == "read"])
        val = lambda t: [sum(int(v) * 10 ** i for i, v in enumerate(r_)) for r_ in t.tolist()]   # noqa: E731
        for r_ in roots:
            for g in r_.goal_list:
                law, pat = g.name, r_.pattern
                if (law, pat) not in (("acc.mul", "o,x,d"), ("mul", "x,d"), ("red", "x+dn"), ("mul", "x,o"),
                                      ("red", "x+on")):
                    continue
                d, t = r_.draws[-1], g.targets[-1]
                ins = [val(o) for o in d.operands(c.ops, c.lib)]
                got = val(t["A"])
                if law == "acc.mul":
                    want = [o + x * e for o, x, e in zip(*ins)]
                elif law == "mul":
                    want = [a * b for a, b in zip(*ins)]
                else:
                    want = [a % n for a, n in zip(*ins)]
                m = t["Am"].tolist()
                ok = all(w == g_ for w, g_, k in zip(want, got, m) if k)
                print(f"  {law} on {pat} @{''.join(r_.binding)}: {sum(m)} rows answered; the book's construction "
                      f"{'agrees' if ok else 'DISAGREES'} with Python's integers")
                break
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
