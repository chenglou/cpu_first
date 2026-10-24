"""The answer program's search offline (LEARNED_LOOP.md §3): sp2's official run's recorded library and its two round
programs (the answers of T = 1, found in that run), exact cells, the full-size machine (16 places), and H1 training
rows of several T as the goal. Does the search find the count loop over the round (C.S[T.round<i> STA.X]), and what
does it cost? The record goes to OUT/<TAG>.json.

    env: VOCAB (all: every routine of the library; rounds: the rounds only), MAX_COST (3), MAX_BODY (2), BUDGET (1e6),
         NP (8: probes), ROWS (64 a T), TS (1,2,4,8: the T values of the rows), ROUNDS (both | one), TAG, OUT
    cd cpu_first && PYTHONPATH=. VOCAB=rounds ~/github/sotaku/venv/bin/python -m integration.learned_loop.outer_offline
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
ROUNDS = ["LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA", "LDA.X STA.Y T.mul.x,o.YX STA.X LDA.N T.red.x+on.XA"]


def main():
    env = os.environ.get
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = env("TAG", f"outer_{env('VOCAB', 'all')}")
    out = Path(env("OUT", str(HERE / "records"))) / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    lib = {k: L.parse(v) for k, v in json.load(open(HERE / "checks" / "sp2_official_lib.json")).items()}
    rounds = [L.parse(r) for r in ROUNDS[:1 if env("ROUNDS", "both") == "one" else 2]]
    ts = [int(t) for t in env("TS", "1,2,4,8").split(",")]
    rows = [r for t in ts for r in load_rows(int(env("ROWS", 64)), seed=t, T=t)]
    pool = tuple(digits(rows, k).to(dev) for k in ("modulus", "x")) + \
        (digits(rows, "time_steps", 2).to(dev), digits(rows, "result").to(dev))
    c = S.SelfClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16,
                    fuse=4 if dev == "cuda" else 1)
    c.lib = dict(lib)
    t0 = time.time()
    progs, lib2, rec = c.search_outer(pool, rounds, budget=int(float(env("BUDGET", 1e6))),
                                      max_cost=int(env("MAX_COST", 3)), max_body=int(env("MAX_BODY", 2)),
                                      NP=int(env("NP", 8)), vocab=env("VOCAB", "all"))
    rec.update(tag=tag, device=dev, wall=round(time.time() - t0, 1), ts=ts,
               gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu")
    print(json.dumps(rec), flush=True)
    # every accepted program, checked again on fresh rows of every T up to 64 that training never shows
    test = [r for t in (1, 2, 4, 8, 16) for r in load_rows(32, seed=100 + t, T=t)]
    test += [r for t in (32, 64) for r in load_rows(32, seed=100 + t, T=t, split="depth_t_%d" % t)]
    tp = tuple(digits(test, k).to(dev) for k in ("modulus", "x")) + \
        (digits(test, "time_steps", 2).to(dev), digits(test, "result").to(dev))
    st = L.round_settings(c.cells, 16)
    rec["unseen"] = {}
    for p in progs:
        A = L.outer_machine(st, p, lib2, tp[0].shape[0], dev).run(tp[1], tp[0], tp[2], L.gen_on(dev, 5))
        rec["unseen"][L.show(p)] = float((A[:, :8] == tp[3]).all(1).float().mean())
    rec["unseen_rows"] = len(test)
    print("UNSEEN", rec["unseen"], len(test), flush=True)
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
