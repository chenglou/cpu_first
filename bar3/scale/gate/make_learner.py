"""Writes learn_anchor.py (GATE.md §9, PREREG_ADDENDUM2.md): the consolidation learner (consolidate/learn_newton.py)
with one asserted replacement, marked "gate": under cfg["anchor_plain"], anchored pairs anchor only on problems the
plain search solved. A problem the gate's second search solved (its probe record has `gated`, learnrun.py) stays
learnable: it can still be a pair's other member, a single, a pick. With cfg["anchor_plain"] off, the copy is the
consolidation's learner.

    cd cpu_first && python3 -m bar3.scale.gate.make_learner
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "consolidate" / "learn_newton.py"

OLD = ('                        top = sorted(learnable, key=lambda c: (-status[c.name]["cost"], c.name not in tsubs))'
       '[:cfg["anchor_cost"]]\n')
NEW = ('                        gated = ({r["name"] for r in list(rb) + list(deep) if r is not None and r.get("gated")}\n'
       '                                 if cfg.get("anchor_plain") else set())        # gate: the second search\'s finds\n'
       '                        top = sorted([c for c in learnable if c.name not in gated],\n'
       '                                     key=lambda c: (-status[c.name]["cost"], c.name not in tsubs))'
       '[:cfg["anchor_cost"]]\n')


def main():
    text = SRC.read_text()
    assert text.count(OLD) == 1, text.count(OLD)
    text = text.replace(OLD, NEW)
    head = ('"""GATE.md §9: written by bar3/scale/gate/make_learner.py from bar3/scale/consolidate/learn_newton.py, with 1\n'
            'asserted replacement (marked "gate"): under cfg["anchor_plain"], anchored pairs anchor only on problems the\n'
            'plain search solved. The rest of this docstring is the original\'s.\n\n')
    assert text.startswith('"""')
    (HERE / "learn_anchor.py").write_text(head + text[3:])
    print("wrote learn_anchor.py: 1 replacement")


if __name__ == "__main__":
    main()
