"""LEGALLOOPS.md §2: written by bar3/scale/legalloops/make_copies6.py from bar3/scale/legalclimb/lclimb.py, 0 asserted replacements
(this study's, marked "legalloops"). The rest of this docstring is the copy's.

The climb with full-size roots (LEGALCLIMB.md §2): climb_k.ComposeClimb, where a full-size root (rowclimb.FullRoot,
a target-derived problem on the answer rows) is probed on its own setting and over the full-size vocabulary.

SelfClimb already groups roots by their probes' shapes, so full-size roots are searched apart from the practice
roots. What changes for a full-size group:
- `probe`: its setting, and the target's cached full-size vocabulary (RowClimbTarget.classes); no primitives-alone
  probe; a deep probe (REST_DEEP's, SUBGOAL's) over every routine, the focus off;
- `deep_roots`: the regular deep phase leaves full-size roots out (their deep probe is REST_DEEP's alone);
- `choose_routine`: the climb's own rule (inputs kept, registers touched, size, steps executed), on the root's own
  setting.
With no full-size root (TARGET small or round) every method is ComposeClimb's.
"""

from __future__ import annotations

from integration.lib.libcore import cost, expand, steps
from bar3.scale.legalloops.climb_k import ComposeClimb
from bar3.scale.legalloops.spcore_k import side_effects


class LegalClimb(ComposeClimb):

    def probe(self, group, rows, deadline=None, deep=False):
        if not getattr(group[0], "full", False):
            return super().probe(group, rows, deadline, deep)
        saved = self.settings, self.fallback_budget, self.deep_focus
        self.settings, self.fallback_budget, self.deep_focus = group[0].settings, 0, False
        self.comp_classes = lambda deadline=None: self.target.classes(self, deadline)
        try:
            return super().probe(group, rows, deadline, deep)
        finally:
            self.settings, self.fallback_budget, self.deep_focus = saved
            del self.comp_classes

    def deep_roots(self, roots):
        return super().deep_roots([r for r in roots if not getattr(r, "full", False)])

    def choose_routine(self, progs, lib, root):
        if not getattr(root, "full", False):
            return super().choose_routine(progs, lib, root)
        st, r0 = root.settings[-1], root.roots[-1]

        def key(p):
            keeps, touched = side_effects(p, lib, st, r0, root.binding)
            return (not keeps, touched, cost(expand(p, lib)), steps(p, lib))
        best = min(progs, key=key)
        return [best] + [p for p in progs if p != best]
