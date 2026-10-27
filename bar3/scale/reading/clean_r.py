"""READING.md §2: written by bar3/scale/reading/make_copies6.py from bar3/scale/compose/clean.py, 1 + 0 + 0 + 0
asserted replacements (rounds 2 and 3's, marked "compose2" and "compose3", round 4's, marked "slice", round 5's,
marked "loops", then this study's, marked "reading"). The rest of this docstring is the original's.

The oracle arm's sources with one change behind a flag, CLEAN_VALUES (COMPOSE.md §2.4): a near miss's value is posed
(as a value problem on the target's root, and as the rest of the way from it) only when it is a function of x' and
N': the same under a second dirty start of the target's root. Bar 3's own machine starts clean (library/inventing/
brm.py: A, T and U are 0), so there every value is one. The digit machine starts dirty (library/dsl.py: A, S, the
carry and the multiply digits random), and the ported source (integration/self_practice/oracle.py) poses whatever a
near miss's prefix leaves in a register, whatever it read. The projections and restrictions are the target's own
functions, unchanged.
"""

from __future__ import annotations

from integration.lib.libcore import fresh, gen_on
from bar3.scale.reading.oracle_r import SmallTarget, prefix_values      # compose2: the copy


class CleanTarget(SmallTarget):
    dirty = frozenset()

    def pose(self, climb, rows, near):
        before = set(self.values)
        out = super().pose(climb, rows, near)
        tr = self.root(climb, rows)
        if not hasattr(self, "_alt"):                   # the target's root again: the same x', N', another dirty start
            gen = gen_on(climb.device, 4242)
            self._alt = [fresh(st, {"X": d.x, "N": d.N}, gen, d.n, climb.device)
                         for st, d in zip(climb.settings, tr.draws)]
        dirty = set()
        for nm in sorted(set(self.values) - before):
            prog, ir = self.values[nm]
            for st, rt, alt in zip(climb.settings, tr.roots, self._alt):
                if not bool((prefix_values(prog, st, rt, climb.lib)[ir] ==
                             prefix_values(prog, st, alt, climb.lib)[ir]).all()):
                    dirty.add(nm)
                    break
        self.dirty = self.dirty | dirty
        self.posed_clean = sorted(set(self.values) - before - dirty)
        kept = []
        for r, goals in out:
            if r.kind == "rest" and "val." + r.key.split("@")[0][len("rest."):] in self.dirty:
                continue
            if r is tr:
                goals = [g for g in goals if g.name not in self.dirty]
                if not goals:
                    continue
            kept.append((r, goals))
        return kept
