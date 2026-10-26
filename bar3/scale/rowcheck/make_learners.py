"""Writes learn_newton.py and learn_digits.py (ROWCHECK.md §2): bar3/scale/subgoal/learn.py, the learner frozen at
97f30b9, with the two-source study's replacements (bar3/scale/newton/make_learner.py) or the digits study's
(bar3/scale/digits/make_learner.py, and its DigitsTap), then this study's: the duplicate check on a problem's own rows
behind cfg["rowcheck"]. Every replacement is asserted to match once, so no copy can drift from the original silently.
With the flag off each copy is that study's learner (check.py replays their records).

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/rowcheck/make_learners.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SRC = HERE.parent / "subgoal" / "learn.py"
sys.path.insert(0, str(ROOT))

from bar3.scale.digits import make_learner as MD  # noqa: E402  (its replacements and DigitsTap, read, not run)
from bar3.scale.newton import make_learner as MN  # noqa: E402

ROWCHECK = [
    ("from bar3.scale.subgoal import score as SC\n",
     "from bar3.scale.subgoal import score as SC\n"
     "from bar3.scale.rowcheck import rowcheck as RC\n"),
    ("            have = {sigs.of((t,), lib) for t in list(brm.PRIMS) + list(lib)}\n"
     "            learn_sig = {}\n",
     "            have = {sigs.of((t,), lib) for t in list(brm.PRIMS) + list(lib)}\n"
     "            learn_sig = {}\n"
     '            sig_tok, learn_rc = (RC.tokens_by_sig(sigs, lib) if cfg.get("rowcheck") else {}), {}   # rowcheck\n'),
    ("                    s = sigs.of(dsl.parse(best), lib)\n"
     "                    if s in have:\n",
     "                    s = sigs.of(dsl.parse(best), lib)\n"
     '                    if cfg.get("rowcheck"):                                  # rowcheck: on its own rows\n'
     '                        st["status"], of = RC.classify(s, c, lib, sig_tok, learn_rc, status, k, k2)\n'
     "                        if of:\n"
     '                            st["of"] = of\n'
     "                    elif s in have:\n"),
]

NOTE = '''- with cfg["rowcheck"], the alias and duplicate check runs on each problem's own rows (bar3/scale/rowcheck/
  rowcheck.py): a shared signature on random machine states only proposes, and a problem is an alias or a duplicate
  when the routine or the earlier problem's best program is right on its rows. Marked "# rowcheck:" below.
'''


def build(doc, repl, tail, out):
    text = SRC.read_text()
    body = text[text.index("from __future__"):]
    for old, new in repl + ROWCHECK:
        n = body.count(old)
        assert n == 1, (n, old)
        body = body.replace(old, new)
    marker = 'Everything else'
    assert doc.count(marker) == 1, "the doc's last line"
    doc = doc.replace(marker, NOTE + marker)
    doc = doc.replace('"""', '"""ROWCHECK.md §2: written by bar3/scale/rowcheck/make_learners.py, from the following.\n\n', 1)
    out.write_text(doc + body.rstrip() + "\n" + tail)
    print(f"wrote {out}: {len(repl)} + {len(ROWCHECK)} replacements")


def main():
    build(MN.DOC, MN.REPL, "", HERE / "learn_newton.py")
    build(MD.DOC, MD.REPL, MD.TAIL, HERE / "learn_digits.py")


if __name__ == "__main__":
    main()
