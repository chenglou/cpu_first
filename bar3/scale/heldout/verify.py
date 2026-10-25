"""The held-out runs' outcomes (HELDOUT.md §4): bar3/scale/subgoal/verify.py on this folder's records, with the
held-out targets registered: the round each target came in, its program rechecked on every pair up to N = 120 (7,260
pairs), its routines, and every round's subgoal.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.heldout.verify [records/*.jsonl ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from bar3.scale.heldout import targets  # noqa: F401  (the held-out targets, for TG.use)
from bar3.scale.subgoal import verify as V

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    sys.argv[1:] = sys.argv[1:] or sorted(str(p) for p in (HERE / "records").glob("*_ho1.jsonl"))
    V.main()
