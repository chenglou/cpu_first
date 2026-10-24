"""The round's search offline on a library a run left (LEARNED_LOOP.md §5): the library rebuilt from a record's sleep
lines (checks/ll_seed2_lib.json: the learned-loop file's B200 seed 2, whose climb ended without a round), the
routines' bindings read from their names (T.<law>.<pattern>.<binding>), exact cells, the answers of 64 T = 1 training
rows; the round's search as the file runs it (by composition, then by calls); if a round is found, the answer
program's search after it. Does the run's library hold a round, so that the run lacked only time?

    env: LIB (checks/ll_seed2_lib.json), TAG, OUT
    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m integration.learned_loop.round_offline
"""
import json
import os
import time
from pathlib import Path

import torch

from integration.learned_loop import libcore as L
from integration.learned_loop import spcore as S
from integration.learned_loop.rows import digits, load_rows

HERE = Path(__file__).resolve().parent


def main():
    env = os.environ.get
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = env("TAG", "round_seed2")
    out = Path(env("OUT", str(HERE / "records"))) / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    lib = {k: L.parse(v) for k, v in json.load(open(HERE / env("LIB", "checks/ll_seed2_lib.json"))).items()}
    c = S.SelfClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16,
                    fuse=4 if dev == "cuda" else 1, macro_after=12, macro_cost=2)
    c.lib = dict(lib)
    for name in lib:                     # each routine's binding from its name: T.<law>.<pattern>.<binding>
        if name.startswith("T."):
            parts = name[2:].split(".")
            c.ops.add(".".join(parts[:-2]), parts[-2], name, tuple(parts[-1]))
    rows = load_rows(64, seed=1, T=1)
    ans = tuple(digits(rows, k).to(dev) for k in ("modulus", "x", "result"))
    t0 = time.time()
    progs, rec = c.search_round(ans)
    rec.update(tag=tag, device=dev, routines=sum(1 for k in lib if k.startswith("T.")), wall=round(time.time() - t0, 1))
    print("ROUND", json.dumps(rec), flush=True)
    if progs:
        mrows = [r for t in (1, 2, 4, 8) for r in load_rows(64, seed=t, T=t)]
        pool = tuple(digits(mrows, k).to(dev) for k in ("modulus", "x")) + \
            (digits(mrows, "time_steps", 2).to(dev), digits(mrows, "result").to(dev))
        outer, _, orec = c.search_outer(pool, progs[:4])
        rec["outer"] = orec
        print("OUTER", json.dumps(orec), flush=True)
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
