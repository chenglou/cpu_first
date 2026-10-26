"""A local smoke run of the digits learner (DIGITS.md §3): the plumbing, not a result. Each named config with tiny
budgets (a tenth of the states, at least 3,000, and 120 s a search) for 2 rounds on one target, on a local pool of 4
workers, records to logs/smoke_<config>_<target>.jsonl.

    cd cpu_first && <python with numba> -m bar3.scale.digits.smoke lp_clean_b_src,ct2a2_clean_d_ctl dxmod
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

from bar3.scale.digits import learn
from bar3.scale.digits.configs import CONFIGS
from bar3.scale.digits.searcher import Searcher

HERE = Path(__file__).resolve().parent


def tiny(cfg):
    small = {k: dict(v, max_states=max(3000, v["max_states"] // 10), time_limit=120) for k, v in cfg["budgets"].items()}
    return dict(cfg, rounds=2, budgets=small,
                subgoal=dict(cfg["subgoal"], deep=dict(cfg["subgoal"]["deep"], max_states=1_000_000, time_limit=120)))


def main():
    names, target = sys.argv[1].split(","), sys.argv[2]
    search = Searcher(4, "nb", True)
    threads = []
    for name in names:
        path = HERE / "logs" / f"smoke_{name}_{target}.jsonl"
        path.write_text("")

        def emit(rec, path=path):
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        threads.append(threading.Thread(target=learn.fresh, args=(name, tiny(CONFIGS[name]), target, search, emit)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    search.pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
