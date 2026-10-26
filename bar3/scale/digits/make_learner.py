"""Writes learn.py (DIGITS.md §2): bar3/scale/subgoal/learn.py, the learner frozen for the held-out test, copied line
for line with the digits study's changes applied as exact replacements, each asserted to match once, so the copy
cannot drift from the original silently.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/digits/make_learner.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "subgoal" / "learn.py"

DOC = '''"""The digits study's learner (DIGITS.md §2), written by make_learner.py: bar3/scale/subgoal/learn.py line for line
(the learner frozen for the held-out test), with these changes, marked "# digits:" below:
- the loop kinds include the digit block of the representation, cfg["base"] (machine.py: D.1, D.2 or D.10);
- the target is posed on its rows (targets.py: x < 10·k), and so is everything the learner poses from the target's
  rows: restrictions, projections, the near misses' values and remainders, constants and routine remainders. The
  recurrence source reads the target on the pairs x < N, as before (its table needs every x < N and no more);
- the law book is posed on the target's rows as well (source.row_laws), besides the pairs x < N;
- with "digit" in cfg["sources"], the one-more-digit source (source.digit_steps) poses its steps every round;
- every job on the target is tagged with its name (DigitsTap), so that a worker runs it on the target's machines and
  accepts a program only when it also passes the check at big x (jobs.py).
Everything else, the subgoal included, is the frozen learner's.
"""

'''

REPL = [
    ("from bar3.scale.subgoal import score as SC\n",
     "from bar3.scale.subgoal import score as SC\n"
     "from bar3.scale.digits import machine as M\n"
     "from bar3.scale.digits import source as DS\n"
     "from bar3.scale.digits import targets as DT\n"),
    ('    lib, kinds = dict(init.get("lib", {})), list(init.get("kinds", brm.LOOP_KINDS))\n',
     '    lib, kinds = dict(init.get("lib", {})), list(init.get("kinds", brm.LOOP_KINDS + (M.KINDS[cfg["base"]],)))'
     '  # digits\n'),
    ("    tgt = P.the_target(k, k2)\n",
     '    tgen, small = DT.gen_for(cfg["target"]), DT.small_gen(cfg["target"])          # digits: the target\'s rows\n'
     '    tgt = P.make("target", "target", DT.FORMULAS[cfg["target"]][0], tgen, k, k2)\n'),
    ('            if "law" in src:\n'
     '                pool += P.laws(k, k2)\n',
     '            if "law" in src:\n'
     '                pool += P.laws(k, k2)\n'
     '                pool += DS.row_laws(tgen, k, k2)                                         # digits: on the rows\n'),
    ("                pool += P.restrictions(P.target_gen, k, k2)\n",
     "                pool += P.restrictions(tgen, k, k2)                                       # digits\n"),
    ("                pool += P.projections(P.target_gen, k, k2)\n",
     "                pool += P.projections(tgen, k, k2)                                        # digits\n"),
    ("                pool += [c for c in P.from_near_misses(near, lib, k, k2)\n",
     "                pool += [c for c in P.from_near_misses(near, lib, k, k2, gen=tgen)             # digits\n"),
    ("                pool += S.recurrences(P.target_gen, k, k2)\n",
     "                pool += S.recurrences(small, k, k2)                                       # digits: x < N\n"),
    ("                pool += S.constants(P.target_gen, k, k2)\n",
     "                pool += S.constants(tgen, k, k2)                                          # digits\n"),
    ("                pool += S.routine_remainders(lib, k, k2)\n",
     "                pool += S.routine_remainders(lib, k, k2, gen=tgen)                        # digits\n"
     '            if "digit" in src:                                                        # digits: the new source\n'
     '                pool += DS.digit_steps(tgen, cfg["base"], k, k2, lib=lib)\n'),
    ('    """bar3/scale/drive.fresh with this run."""\n'
     "    tap = Tap(search_many)\n",
     '    """bar3/scale/drive.fresh with this run."""\n'
     "    cfg = dict(cfg, target=tname)                                                    # digits\n"
     "    tap = DigitsTap(search_many)\n"),
]

TAIL = '''

class DigitsTap(Tap):
    """bar3/breadth/drive.Tap, with every job on the target (all are named "target") tagged with the target's name:
    jobs.py runs those on the target's machines, with the check at big x."""

    def mapper(self, jobs):
        return super().mapper([dict(j, digits_target=self.target) if j["name"] == "target" else j for j in jobs])
'''


def main():
    text = SRC.read_text()
    body = text[text.index("from __future__"):]
    for old, new in REPL:
        n = body.count(old)
        assert n == 1, (n, old)
        body = body.replace(old, new)
    (HERE / "learn.py").write_text(DOC + body.rstrip() + "\n" + TAIL)
    print(f"wrote {HERE / 'learn.py'}: {len(REPL)} replacements")


if __name__ == "__main__":
    main()
