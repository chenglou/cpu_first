"""Train the tiny core over a grid of sizes, choices and seeds; one JSON row per run.

    python -m tinycore.sweep --B 2 --w 2,3,4,5 --k 1,2,3 --choice est,borrow --seeds 4 --steps 3000 --out sizes.jsonl

Each row has the curve (step, loss, exact) every `every` steps, the first step with exact >= 0.99 (`hit`), the final
slot map and the share of cell rows right. Runs go to a small process pool (one thread each).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(job):
    import torch
    torch.set_num_threads(1)
    from .core import Config, Core, Data, train
    cfg_kw, tr_kw, seed, data_kw = job["cfg"], job["train"], job["seed"], job.get("data", {})
    cfg = Config(**cfg_kw)
    data = Data(cfg.B, cfg.w, cfg.k, odd=cfg.odd, start=cfg.start, **data_kw)
    m = Core(cfg, seed=seed)
    t = time.time()
    curve, hit = train(m, data, seed=seed, stop_at=tr_kw.pop("stop_at", 0.99), **tr_kw)
    smap, raw, mapped = m.slot_map(data)
    return {**cfg_kw, **data_kw, "seed": seed, "train": job["train_orig"], "X": data.X, "hit": hit,
            "final": curve[-1][2], "loss": curve[-1][1], "curve": curve, "slots": smap, "raw": raw,
            "mapped": mapped, "cell_right": m.cell_right(), "seconds": round(time.time() - t, 2)}


def run_jobs(jobs, out, workers=6):
    rows = []
    with ProcessPoolExecutor(workers) as ex, open(out, "a") as f:
        for r in ex.map(run_one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            rows.append(r)
            print(f"B{r['B']} w{r['w']} k{r['k']} {r['cell']}/{r['choice']} init {r['init']} seed {r['seed']}: "
                  f"hit {r['hit']} final {r['final']:.3f} slots {r['slots']} ({r['seconds']}s)", flush=True)
    return rows


def grid(B, w, k, choice, cell="slot", seeds=3, steps=3000, lr=0.05, every=25, init=0.1, extra=None, data=None,
         opt="adam", batch=0):
    jobs = []
    for B_, w_, k_, ch, ce, i_ in itertools.product(B, w, k, choice, cell, init):
        for s in range(seeds):
            cfg = {"B": B_, "w": w_, "k": k_, "choice": ch, "cell": ce, "init": i_, **(extra or {})}
            tr = {"steps": steps, "lr": lr, "every": every, "opt": opt, "batch": batch}
            jobs.append({"cfg": cfg, "train": dict(tr), "train_orig": dict(tr), "seed": s, "data": data or {}})
    return jobs


def ints(s):
    return [int(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", default="2")
    p.add_argument("--w", default="3")
    p.add_argument("--k", default="2")
    p.add_argument("--choice", default="est")
    p.add_argument("--cell", default="slot")
    p.add_argument("--init", default="0.1")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--opt", default="adam")
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--extra", default="{}")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    jobs = grid(ints(a.B), ints(a.w), ints(a.k), a.choice.split(","), a.cell.split(","), a.seeds, a.steps, a.lr,
                a.every, [float(x) for x in a.init.split(",")], json.loads(a.extra), opt=a.opt, batch=a.batch)
    run_jobs(jobs, os.path.join(HERE, "runs", a.out), a.workers)


if __name__ == "__main__":
    main()
