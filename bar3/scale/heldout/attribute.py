"""HELDOUT.md §5: bar3/scale/subgoal/attribute.py on the held-out records, with the held-out targets registered: each
run replayed on its container's stored searches, first as it ran (its records must come back), then without the
subgoal (no score, no deeper probe, pool order for every tie). No search runs.

    cd cpu_first && <python with numba> -m bar3.scale.heldout.attribute STORE.jsonl ... -- RECORD.jsonl ...
"""

from __future__ import annotations

from bar3.scale.heldout import targets  # noqa: F401  (the held-out targets, for TG.use)
from bar3.scale.subgoal import attribute

if __name__ == "__main__":
    attribute.main()
