"""Fill (or refresh) the data tables of TINYROUTING.md from runs/. A placeholder is a line holding only its name
(e.g. PHASE_CELL); once filled, the table sits between <!-- NAME --> and <!-- /NAME --> and is replaced on refill.

    python -m tinyrouting.fill
"""

from __future__ import annotations

import re
from pathlib import Path

from tinyrouting import render as R
from tinyrouting import summarize as S

DOC = Path(__file__).resolve().parent / "TINYROUTING.md"


def tables_of(fn):
    return R.tables(R.capture(fn))


def put(doc, name, body):
    block = f"<!-- {name} -->\n{body}\n<!-- /{name} -->"
    pat = re.compile(rf"<!-- {name} -->.*?<!-- /{name} -->", re.S)
    if pat.search(doc):
        return pat.sub(lambda _: block, doc)
    return re.sub(rf"^{name}$", lambda _: block, doc, flags=re.M)


def main():
    doc = DOC.read_text()
    phase = tables_of(S.phase)
    if len(phase) >= 2:
        doc = put(doc, "PHASE_CELL", phase[0])
        doc = put(doc, "PHASE_ANS", phase[1])
    alone = tables_of(S.alone)
    if alone:
        doc = put(doc, "ALONE", alone[0])
    tricks = tables_of(S.tricks)
    if len(tricks) >= 2:
        doc = put(doc, "TRICKS_ANS", tricks[0])
        doc = put(doc, "TRICKS_CELL", tricks[1])
    for name, fn in (("UNTIE", S.untie), ("SCHOOL", S.school)):
        t = tables_of(fn)
        if t and t[0].count("\n") > 2:
            doc = put(doc, name, t[0])
    depth = tables_of(S.depth)
    if len(depth) >= 2 and depth[0].count("\n") > 2:
        doc = put(doc, "DEPTH", "Table given:\n\n" + depth[0] + "\n\nTable taught by the cell check:\n\n" + depth[1])
    st = tables_of(S.stages)
    if len(st) >= 2 and st[0].count("\n") > 2:
        doc = put(doc, "STAGES", "From the answers alone:\n\n" + st[0] + "\n\nWith the cell check:\n\n" + st[1])
    t2 = tables_of(S.tricks2)
    if t2 and t2[0].count("\n") > 2:
        doc = put(doc, "TRICKS2", "**The whole-wiring mixture from the answers alone** (`tricks2`, (3, 2, 1)):\n\n" + t2[0])
    DOC.write_text(doc)
    left = re.findall(r"^[A-Z_0-9]{4,}$", doc, flags=re.M)
    print("filled; placeholders left:", left)


if __name__ == "__main__":
    main()
