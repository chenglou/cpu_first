"""LEGALCLIMB.md §2: written by bar3/scale/legalclimb/make_copies5.py from bar3/scale/slice/climb_s.py, 0 asserted replacements
(this study's, marked "legalclimb"). The rest of this docstring is the copy's.

SLICE.md §2: written by bar3/scale/slice/make_copies4.py from bar3/scale/compose/climb.py, 1 + 0 asserted
replacements (rounds 2 and 3's, marked "compose2" and "compose3", then this study's, marked "slice"). The rest of
this docstring is the original's.

The climb of COMPOSE.md §2: spcore.SelfClimb as it is, with one change behind a flag (`round_too`).

With the oracle arm's target (oracle.SmallTarget: x'^2 mod N' at the practice widths), SelfClimb's target phase searches
only that small target, by composition (the routines, the primitives, their loops) at the round's budget. The round's
own search (search_round: straight-line primitives and the routines at full size, then, past 12 routines, by calls)
never runs. With `round_too`, when the small target is not found, the target phase also runs the round's search at
full size against the answers. So an arm whose sources need the small target and an arm with the law book alone face
the same round search. Without the flag this class is SelfClimb.
"""

from __future__ import annotations

from integration.lib.libcore import round_ok
from bar3.scale.legalclimb.spcore_l import SelfClimb                    # compose2: the copy


class ComposeClimb(SelfClimb):
    round_too = False

    def _step(self, rows, answers, deadline):
        if not (self.round_too and self.phase == "target" and self.target is not None and not self.done):
            return super()._step(rows, answers, deadline)
        self.phase = "pose"
        lvl = self.level
        self.level += 1
        progs, rec, _ = self.target.search(self, rows, self.round_budget, deadline)
        if progs and answers is not None:           # is the small target's program the round at full size?
            rec["full_size"] = [bool(round_ok(p, self.lib, self.cells, answers, self.P_round, self.device))
                                for p in progs[:8]]
        self.records.append(rec)
        line = (f"L{lvl} target: cost {rec['found_cost']} levels {rec['level_sizes']} fits {rec['fits']} "
                f"{rec['seconds']}s {rec['stop']} {rec['programs'][:3]} full size {rec.get('full_size')}")
        if progs:
            self.rounds = progs
            self.done = True
            return line
        if answers is not None and self.routines():
            rprogs, rrec = self.search_round(answers, deadline, t=self.t)
            self.records.append(rrec)
            line += (f" | round: cost {rrec['found_cost']} levels {rrec['level_sizes']} {rrec['seconds']}s "
                     f"{rrec['stop']} {rrec['programs'][:3]}")
            if rprogs:
                self.rounds = rprogs
                self.done = True
        return line
