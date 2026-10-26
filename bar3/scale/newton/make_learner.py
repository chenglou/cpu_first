"""Writes learn.py (NEWTON.md §2): bar3/scale/subgoal/learn.py, the learner frozen at 97f30b9 (the held-out test's),
copied line for line with the two sources added as exact replacements, each asserted to match once, so the copy cannot
drift from the original silently. Both are behind flags that are off unless a config names them.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/newton/make_learner.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "subgoal" / "learn.py"

DOC = '''"""The two-source study's learner (NEWTON.md §2), written by make_learner.py: bar3/scale/subgoal/learn.py line for
line (the learner frozen at 97f30b9 for the held-out test), with two sources added, marked "# newton:" below, each
posed only when cfg["sources"] names it:
- "rsum": the running sum's step, y(x + 1) from y(x) and x (sources.running_sum), right after the recurrences;
- "offset": each rest of the way's answer less its answer at X = 0, mod N (sources.offsets), right after the routine
  remainders, from the rests of the way posed by then.
Everything else is the frozen learner's. With neither flag it is the frozen learner (check.py).
"""

'''

REPL = [
    ("from bar3.scale.subgoal import score as SC\n",
     "from bar3.scale.subgoal import score as SC\n"
     "from bar3.scale.newton import sources as NS\n"),
    ("                pool += S.recurrences(P.target_gen, k, k2)\n",
     "                pool += S.recurrences(P.target_gen, k, k2)\n"
     '            if "rsum" in src:                                                         # newton: the running sum\n'
     "                pool += NS.running_sum(P.target_gen, k, k2)\n"),
    ("                pool += S.routine_remainders(lib, k, k2)\n",
     "                pool += S.routine_remainders(lib, k, k2)\n"
     '            if "offset" in src:                                                       # newton: the offsets\n'
     "                pool += NS.offsets(pool, k, k2)\n"),
]


def main():
    text = SRC.read_text()
    body = text[text.index("from __future__"):]
    for old, new in REPL:
        n = body.count(old)
        assert n == 1, (n, old)
        body = body.replace(old, new)
    (HERE / "learn.py").write_text(DOC + body.rstrip() + "\n")
    print(f"wrote {HERE / 'learn.py'}: {len(REPL)} replacements")


if __name__ == "__main__":
    main()
