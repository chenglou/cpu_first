"""Writes this study's copies (LEGALCLIMB.md §2): round 4's committed copies (bar3/scale/slice/, every earlier round's
replacements already in them, their sha256 asserted), module paths moved to this folder, then this study's
replacements, each asserted to match exactly once:
- hooks.py: SUBGOAL's score, for a full-size target (TARGET=rows), is the target's own law view by routines
  (rowclimb.RowClimbTarget.subgoal_scores) instead of the small target's numpy score;
- core_l.py: TARGET=rows builds rowclimb.RowClimbTarget on the answer rows; the climb is lclimb.LegalClimb; the flags
  LOOK_COST, IDCHECK and FULL_BODY in the run's config; SEED, when set, is the climb's seed (ADDENDUM1.md);
  MIN_ROWS, the law view's floor (ADDENDUM2.md).
spcore_l.py, climb_l.py, oracle_l.py and clean_l.py are round 4's with the paths moved. LegalClimb is ComposeClimb
whenever no root is full-size, so with TARGET small or round the copies run as round 4's ran.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.legalclimb.make_copies5
"""

from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLICE = HERE.parent / "slice"
SOURCES = {                                       # round 4's copies at commit 05810dc, unchanged since
    "spcore_s.py": ("spcore_l.py", "2c724f4ff86049bbbc0f945ef42efa32bc5a4def3b55acbfce9644faa046eef3"),
    "climb_s.py": ("climb_l.py", "67927104fd452f5d8740e6f0c388f090d3908fbea4cd4e8f9fe12b4b035daf82"),
    "core_s.py": ("core_l.py", "5dacdb681758fcbf35e290a0f430e0d1f2757674af75eb5b296e8faf85781dd4"),
    "hooks.py": ("hooks.py", "fd463e1e9921b690049571ea448cc91389432a024e7bd7f7f133660b58905efb"),
    "oracle_s.py": ("oracle_l.py", "67d86b5f43dd049ae686591417e2814824925b14e46f1d5d1a4ba808e25d1e78"),
    "clean_s.py": ("clean_l.py", "b9feb974899c38f4f5e8e7645943928721b2141690de0ccab528bd3fad863843"),
}


def move(s):
    """Round 4's module paths, moved to this folder."""
    for a, b in (("bar3.scale.slice.", "bar3.scale.legalclimb."), ("bar3.scale.slice import", "bar3.scale.legalclimb import"),
                 ("spcore_s", "spcore_l"), ("oracle_s", "oracle_l"), ("climb_s", "climb_l"), ("clean_s", "clean_l"),
                 ("core_s", "core_l")):
        s = s.replace(a, b)
    return s


HOOKS5 = [
    ("    t = climb.target\n"
     "    if not getattr(climb, \"subgoal_on\", False) or t is None:\n"
     "        return\n"
     "    from bar3.scale.legalclimb.oracle_l import prefix_values\n",
     "    t = climb.target\n"
     "    if not getattr(climb, \"subgoal_on\", False) or t is None:\n"
     "        return\n"
     "    if getattr(t, \"full\", False):                    # legalclimb: the full-size target's law view by routines\n"
     "        return t.subgoal_scores(climb, rows, roots)\n"
     "    from bar3.scale.legalclimb.oracle_l import prefix_values\n"),
]
CORE5 = [
    ("from bar3.scale.legalclimb import climb_l as CC\n",
     "from bar3.scale.legalclimb import climb_l as CC\n"
     "from bar3.scale.legalclimb import lclimb as LC                # legalclimb: the climb with full-size roots\n"),
    ("        target.restrict_rows = env(\"RESTRICT_ROWS\", \"0\") == \"1\"  # compose2: RESTRICT_ROWS\n",
     "        target.restrict_rows = env(\"RESTRICT_ROWS\", \"0\") == \"1\"  # compose2: RESTRICT_ROWS\n"
     "    if env(\"TARGET\", \"round\") == \"rows\":                   # legalclimb: bar 3's sources on the answer rows\n"
     "        from bar3.scale.legalclimb.rowclimb import RowClimbTarget\n"
     "        target = RowClimbTarget(ans, dev, sources=tuple(x for x in sources if x != \"law\"),\n"
     "                                lawnear=env(\"LAWNEAR\", \"0\") == \"1\", clean=env(\"CLEAN_VALUES\", \"0\") == \"1\",\n"
     "                                idcheck=env(\"IDCHECK\", \"0\") == \"1\", look_cost=int(env(\"LOOK_COST\", 3)),\n"
     "                                full_body=int(env(\"FULL_BODY\", 0)) or None,\n"
     "                                min_rows=int(env(\"MIN_ROWS\", 0)) or None)\n"),
    ("    sources = env(\"SOURCES\", \"law\").split(\",\")\n",
     "    if env(\"SEED\"):                                             # legalclimb: the climb's draws (SelfClimb's seed)\n"
     "        cfg[\"seed\"] = int(env(\"SEED\"))\n"
     "    sources = env(\"SOURCES\", \"law\").split(\",\")\n"),
    ("    c = CC.ComposeClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == \"cuda\" else 1 << 16, target=target, **cfg)\n",
     "    c = LC.LegalClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == \"cuda\" else 1 << 16, target=target, **cfg)\n"),
    ("    if H.SLICE:                                                   # slice: the cuts posed\n"
     "        flags[\"slice\"] = H.SLICE\n",
     "    if H.SLICE:                                                   # slice: the cuts posed\n"
     "        flags[\"slice\"] = H.SLICE\n"
     "    if env(\"TARGET\", \"round\") == \"rows\":                   # legalclimb: the look's cost, the identity check\n"
     "        flags.update(look_cost=target.look_cost, idcheck=target.idcheck, full_body=target.full_body,\n"
     "                     min_rows=target.min_rows)\n"),
]
EDITS = {"hooks.py": HOOKS5, "core_l.py": CORE5}


def main():
    for src, (dst, want) in SOURCES.items():
        text = (SLICE / src).read_text()
        got = hashlib.sha256(text.encode()).hexdigest()
        assert got == want, (src, got)
        text = move(text)
        for a, b in EDITS.get(dst, []):
            assert text.count(a) == 1, (dst, a[:60], text.count(a))
            text = text.replace(a, b)
        first, rest = text.split("\n", 1)
        head = f'"""LEGALCLIMB.md §2: written by bar3/scale/legalclimb/make_copies5.py from bar3/scale/slice/{src}, ' \
               f'{len(EDITS.get(dst, []))} asserted replacements\n(this study\'s, marked "legalclimb"). The rest of ' \
               f'this docstring is the copy\'s.\n\n'
        assert first.startswith('"""'), (src, first[:40])
        (HERE / dst).write_text(head + first[3:] + "\n" + rest)
        print("wrote", dst, len(EDITS.get(dst, [])), "replacements")


if __name__ == "__main__":
    main()
