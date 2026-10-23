"""The breadth runs (BREADTH.md), local or on Modal: the learner unchanged on one target (`fresh`), or on targets in
sequence with its library carried (`carried`, carry.py). Both go through a Tap between the learner and the search
workers, which
- tags the jobs that need the target (a target search's N <= 40 check) with its name, so a worker can `use` it;
- keeps every lookahead's result, so that the pair choice can be replayed in full afterwards (a round record keeps
  only its best 200 pairs): each round record gains `looks`, one [routine 0, routine 1, found cost, best fit per
  level, probes, seconds, vocabulary size with the routines] per lookahead.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.breadth.drive --targets rho,fib     # tiny, local, seconds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from library.inventing import brm
from bar3 import learner
from bar3.breadth import carry as C
from bar3.breadth import cheap
from bar3.breadth import targets as TG

HERE = Path(__file__).resolve().parent

# the learner's configs, plus "practice everything" for one round only: its round 1 would probe ~150 new candidates
# with a ~90-token vocabulary, about 25 search-hours a target
CONFIGS = dict(learner.CONFIGS, all_once=dict(learner.CONFIGS["all_clean"], rounds=1),
               ct2a2_clean=dict(learner.CONFIGS["ct2_clean"], anchor_cost=2))   # the cheaper pairs signal (cheap.py)


class Tap:
    def __init__(self, search_many):
        self.search_many = search_many      # [job] -> [record or exception]
        self.target = None
        self.looks = []

    def mapper(self, jobs):
        tagged = [dict(j, breadth_target=self.target) if j.get("general") else j for j in jobs]
        recs = self.search_many(tagged)
        for j, r in zip(jobs, recs):
            if j["kind"] == "look" and isinstance(r, dict):
                self.looks.append([j["lib"].get("T.look0"), j["lib"].get("T.look1"), r["found_cost"], r["best_fit"],
                                   r["NP"], r.get("wall", r["seconds"]),
                                   len(brm.PRIMS) + len(j["lib"]) + len(j["kinds"])])
        return recs

    def emit_with_looks(self, emit):
        def e(rec):
            emit(dict(rec, looks=self.looks) if "round" in rec else rec)
            self.looks = []
        return e


def fresh(name, cfg, tname, search_many, emit, log=print):
    """The learner on one target; a config with `anchor` or `anchor_cost` runs a cheaper pairs signal (cheap.py)."""
    tap = Tap(search_many)
    TG.use(tname)
    tap.target = tname
    run = cheap.run if cfg.get("anchor") or cfg.get("anchor_cost") else learner.run
    run(name, cfg, tap.mapper, tap.emit_with_looks(lambda r: emit(dict(r, target_name=tname))), log=log)


def carried(name, cfg, order, search_many, emit, carry="targets", log=print, state=None, save=None):
    tap = Tap(search_many)

    def on_target(t):
        tap.target = t

    return C.sequence(name, cfg, order, tap.mapper, tap.emit_with_looks(emit), carry=carry, log=log,
                      on_target=on_target, state=state, save=save)


def main():
    """A local check of the plumbing: the tiny config (k = 5, tiny budgets, 12 candidates) on each target, then the
    same targets in sequence with the library carried."""
    from bar3.jobs import search_job
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="rho,fib")
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--carry", default="targets")
    a = ap.parse_args()
    cfg = learner.CONFIGS[a.config]
    order = a.targets.split(",")
    out = HERE / "records" / f"local_{a.config}.jsonl"
    out.write_text("")

    def emit(rec):
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def search_many(jobs):
        recs = []
        for j in jobs:
            if j.get("breadth_target"):
                TG.use(j["breadth_target"])
            recs.append(search_job(j))
        return recs

    for t in order:
        fresh(f"{a.config}_{t}", cfg, t, search_many, emit)
    carried(f"{a.config}_seq", cfg, order, search_many, emit, carry=a.carry)
    print("wrote", out)


if __name__ == "__main__":
    main()
