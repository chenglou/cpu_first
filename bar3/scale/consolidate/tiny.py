"""The allocator's tiny test (CONSOLIDATE.md §3), locally: the newton copy with `alloc_deep` on one formula at the
learner's TINY budgets (the deeper rungs shrunk to match), at most 2 rounds, its searches in a store under logs/. It
shows the code path end to end: the census, the best-first actions, the gain threshold's picks, the record's `alloc`.

    cd cpu_first && bar3/scale/rowcheck/memcap.sh 10 <python with numba> -m bar3.scale.consolidate.tiny h1 [FLAG=JSON ...]

With flags (the ablation's, PREREG_ADDENDUM1.md), each is set in the config and the record's name carries them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from bar3 import learner as BL
from bar3.scale.heldout import targets  # noqa: F401  (registers the held-out formulas)
from bar3.scale.heldout.searcher import Searcher
from bar3.scale.consolidate import learn_newton as L
from bar3.scale.consolidate.configs import CONFIGS

HERE = Path(__file__).resolve().parent


def main():
    tname = sys.argv[1] if len(sys.argv) > 1 else "h1"
    flags = dict((a.split("=", 1)[0], json.loads(a.split("=", 1)[1])) for a in sys.argv[2:])
    tag = "".join(f"_{f}-{v}" for f, v in flags.items())
    cfg = json.loads(json.dumps(CONFIGS["alloc_deep"]))
    cfg.update(budgets=BL.TINY, rounds=2)
    cfg["subgoal"] = dict(deep=dict(BL.PROBE_A, max_states=20_000, time_limit=60))
    cfg["alloc_look2"] = dict(BL.PROBE_A, max_states=10_000, time_limit=60)
    cfg.update(flags)
    search = Searcher(4, "nb", True, store=str(HERE / "logs" / f"store_tiny_{tname}.jsonl"))
    out = []
    try:
        L.fresh(f"tiny_alloc_{tname}{tag}", cfg, tname, search, out.append)
    finally:
        search.pool.shutdown(wait=True)
    (HERE / "logs" / f"tiny_alloc_{tname}{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    for r in out:
        a = r.get("alloc", {})
        print(f"round {r['round']}: pool {len(r['pool'])}, budget {a.get('budget')}, spent {a.get('spent')}, "
              f"searches {len(a.get('steps', []))}, picked {r['picked']}, general {bool(r['general'])}")


if __name__ == "__main__":
    main()
