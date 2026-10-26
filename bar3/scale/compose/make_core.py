"""Writes core_compose.py (COMPOSE.md §2): integration/self_practice/core.py, the tiny core's driver, copied line for
line with two changes, each replacement asserted to match once:
- the PATTERNS and LAWS lists are split on ";" instead of ",", since three of the law book's pattern names hold commas
  ("x,d", "x,o", "o,x,d"), which the "," split breaks;
- the climb is climb.ComposeClimb, which is spcore.SelfClimb unless ROUND_TOO=1 (then the oracle arm's target phase
  also runs the round's own search);
- with CLEAN_VALUES=1, the oracle arm's target is clean.CleanTarget (near misses' values posed only when they are
  functions of x' and N'), and the record's config says so. Without it the file runs as the five arms of PREREG.md
  ran it.
Nothing else changes: the climb, its rules and the oracle arm's sources (oracle.SmallTarget) are imported from the
committed files as they are.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/compose/make_core.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent.parent / "integration" / "self_practice" / "core.py"

REPL = [
    ('               patterns=env("PATTERNS").split(",") if env("PATTERNS") else None,\n',
     '               patterns=env("PATTERNS").split(";") if env("PATTERNS") else None,      # compose: ";"\n'),
    ('               laws=env("LAWS").split(",") if env("LAWS") else None)\n',
     '               laws=env("LAWS").split(";") if env("LAWS") else None)                  # compose: ";"\n'),
    ("         PATTERNS / LAWS (comma lists: a subset of the law book, for smoke tests), OUT (the records folder),\n",
     "         PATTERNS / LAWS (\";\" lists: a subset of the law book), OUT (the records folder),\n"),
    ("from integration.self_practice import spcore as S\n",
     "from integration.self_practice import spcore as S\n"
     "from bar3.scale.compose import climb as CC\n"),
    ("    c = S.SelfClimb(L.Cells(10, dev), dev, 16,",
     "    c = CC.ComposeClimb(L.Cells(10, dev), dev, 16,"),
    ("    cfg = cfg_rec\n",
     "    c.round_too = env(\"ROUND_TOO\", \"0\") == \"1\"                    # compose: the round's search too\n"
     "    cfg = dict(cfg_rec, round_too=c.round_too,\n"
     "               **({\"clean_values\": True} if env(\"CLEAN_VALUES\", \"0\") == \"1\" else {}))   # compose: clean.py\n"),
    ("        target = SmallTarget(sources=tuple(x for x in sources if x != \"law\"))\n",
     "        if env(\"CLEAN_VALUES\", \"0\") == \"1\":                    # compose: clean near-miss values only\n"
     "            from bar3.scale.compose.clean import CleanTarget as SmallTarget\n"
     "        target = SmallTarget(sources=tuple(x for x in sources if x != \"law\"))\n"),
]


def main():
    text = SRC.read_text()
    for old, new in REPL:
        n = text.count(old)
        assert n == 1, (n, old)
        text = text.replace(old, new)
    head = ('"""COMPOSE.md §2: written by bar3/scale/compose/make_core.py from integration/self_practice/core.py: the PATTERNS\n'
            'and LAWS lists split on ";", the climb climb.ComposeClimb (ROUND_TOO), and clean.CleanTarget (CLEAN_VALUES).\n'
            'The rest of this docstring is the original\'s.\n\n')
    assert text.startswith('"""')
    (HERE / "core_compose.py").write_text(head + text[3:])
    print(f"wrote {HERE / 'core_compose.py'}: {len(REPL)} replacements")


if __name__ == "__main__":
    main()
