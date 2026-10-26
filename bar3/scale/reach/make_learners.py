"""Writes learn_newton.py and learn_digits.py (REACH.md §2): bar3/scale/subgoal/learn.py, the learner frozen at 97f30b9,
with the two-source study's replacements (bar3/scale/newton/make_learner.py) or the digits study's
(bar3/scale/digits/make_learner.py, and its DigitsTap), then the row check's (bar3/scale/rowcheck/make_learners.py),
the picks study's (bar3/scale/picks/make_learners.py), and this study's two flags (reach.py): `deeplook` and `twin5`.
Every replacement is asserted to match once, so no copy can drift from the original silently. With every flag off
each copy is that study's learner (check.py replays their records).

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/reach/make_learners.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SRC = HERE.parent / "subgoal" / "learn.py"
sys.path.insert(0, str(ROOT))

from bar3.scale.digits import make_learner as MD  # noqa: E402  (their replacements, read, not run)
from bar3.scale.newton import make_learner as MN  # noqa: E402
from bar3.scale.picks import make_learners as MP  # noqa: E402
from bar3.scale.rowcheck import make_learners as MR  # noqa: E402

REACH = [
    ("from bar3.scale.picks import picks as PK\n",
     "from bar3.scale.picks import picks as PK\n"
     "from bar3.scale.reach import reach as RH\n"),
    ('            looks = [job(tgt, with_routines([c]), kinds, B["look"], "look", k, k2) for c in learnable]\n',
     '            looks = [job(tgt, with_routines([c]), kinds, RH.look_budget(cfg, B["look"], c), "look", k, k2)\n'
     "                     for c in learnable]                                      # reach: deeplook\n"),
    ("            pick = pick[:K - 1] + [c for c in learnable if c.name in twin]\n",
     "            pick = pick[:K - 1] + [c for c in learnable if c.name in twin]\n"
     '        twin5 = PK.twins(subs, status, learnable) if cfg.get("twin5") else set()   # reach: the twin, a fifth pick\n'
     "        subs = subs | twin5\n"
     "        pick = pick + [c for c in learnable if c.name in twin5 - {p.name for p in pick}]\n"),
    ("                  **PK.record(cfg, twin, tie),                                         # picks\n",
     "                  **PK.record(cfg, twin, tie),                                         # picks\n"
     '                  **({"reach": dict(twin5=sorted(twin5))} if cfg.get("twin5") else {}),   # reach\n'),
]

NOTE = '''- with cfg["deeplook"] or cfg["twin5"], this study's two changes (bar3/scale/reach/reach.py), marked "# reach:" below.
'''


def build(doc, repl, tail, out):
    text = SRC.read_text()
    body = text[text.index("from __future__"):]
    for old, new in repl + MR.ROWCHECK + MP.PICKS + REACH:
        n = body.count(old)
        assert n == 1, (n, old)
        body = body.replace(old, new)
    marker = 'Everything else'
    assert doc.count(marker) == 1, "the doc's last line"
    doc = doc.replace(marker, MR.NOTE + MP.NOTE + NOTE + marker)
    doc = doc.replace('"""', '"""REACH.md §2: written by bar3/scale/reach/make_learners.py, from the following.\n\n', 1)
    out.write_text(doc + body.rstrip() + "\n" + tail)
    print(f"wrote {out}: {len(repl)} + {len(MR.ROWCHECK)} + {len(MP.PICKS)} + {len(REACH)} replacements")


def main():
    build(MN.DOC, MN.REPL, "", HERE / "learn_newton.py")
    build(MD.DOC, MD.REPL, MD.TAIL, HERE / "learn_digits.py")


if __name__ == "__main__":
    main()
