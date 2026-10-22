"""Tiny tournament sweeps on Modal: one CPU container per run (jobs in `suite()`), at most 20 at once.

    ~/github/sotaku/venv/bin/python -m modal run --detach tournament/modal_tiny.py --suite main
    ~/github/sotaku/venv/bin/python -m modal run tournament/modal_tiny.py --suite main --fetch

Each run writes its record to the Volume `tournament` under tiny/<suite>/<name>.json; the entrypoint appends finished
records to tournament/runs/<suite>.jsonl, and `--fetch` adds the Volume's records that are not there yet.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "grid.py", "net.py", "train.py", "census.py", "notches.py", "compile.py"):
    image = image.add_local_file(ROOT / "bridge" / _f, f"/root/bridge/{_f}")
for _f in ("__init__.py", "tiny.py", "modal_tiny.py"):
    image = image.add_local_file(HERE / _f, f"/root/tournament/{_f}")
vol = modal.Volume.from_name("tournament", create_if_missing=True)
app = modal.App("tournament-tiny", image=image)

SEEDS = range(8)


def job(name, **cfg):
    return {"name": name, "cfg": cfg}


def suite(name: str) -> list:
    """The sweeps (TOURNAMENT.md section 2). Every run: free attention's width (34 options per port, 5.3e10
    wirings) unless the name says otherwise, the MLP subtraction cell learned, 8 seeds."""
    jobs = []
    if name == "main":
        variants = {
            "main": {},                                            # the recipe
            "given": {"tables": "given"},                          # the truth tables: the wiring alone
            "nolabel": {"answers": False},                         # label-free lessons only
            "lab10": {"label_frac": 0.10},                         # 10 of the 99 questions carry an answer
            "lab02": {"label_frac": 0.02},                         # 2 of 99 (H1: 2% of the rows)
            "nogate": {"gate": False},                             # the wiring's lessons from step 1
            "mlptab": {"cell_form": "mlptab", "hidden": 256},      # the H1 file's cell form (notch 1e)
            "mb16": {"batch": 16},                                 # 16 questions per step
            "lr02": {"lr_wire": 0.2},
            "lr005": {"lr_wire": 0.05, "steps": 900},              # the H1 file's switch rate
            "k2": {"k": 2},                                        # (3, 2, 2): two quotient digits
            "b4": {"B": 4},                                        # (4, 2, 1)
        }
        for (v, kw), s in itertools.product(variants.items(), SEEDS):
            cfg = {"seed": s, "lr_wire": 0.5, "tag": v, **kw}
            jobs.append(job(f"{v}.s{s}", **cfg))
    elif name == "slow":
        variants = {
            "row": {"verdict": "row"},                             # per-row mixture verdicts (WIRING.md's form)
            "b5": {"B": 5},                                        # (5, 2, 1)
            "h1like": {"batch": 32, "label_frac": 0.05},           # minibatches, 5% labelled
        }
        for (v, kw), s in itertools.product(variants.items(), SEEDS):
            jobs.append(job(f"{v}.s{s}", seed=s, lr_wire=0.5, tag=v, **kw))
    elif name == "wide":
        variants = {
            "w52": {"rows": 5, "cols": 4, "pair_budget": 100_000_000},   # 168 x 140,608 pairs at first
            "w61": {"rows": 6, "cols": 4, "pair_budget": 200_000_000},
        }
        for (v, kw), s in itertools.product(variants.items(), SEEDS):
            if v == "w61" and s >= 4:
                continue                                           # 4 seeds: about an hour each
            jobs.append(job(f"{v}.s{s}", seed=s, lr_wire=0.5, tag=v, **kw))
    else:
        raise ValueError(name)
    return jobs


@app.function(cpu=4, memory=8192, timeout=6 * 3600, volumes={"/out": vol}, retries=modal.Retries(max_retries=1),
              max_containers=int(os.environ.get("TOURNAMENT_CONTAINERS", "20")))
def run_job(j: dict) -> dict:
    import os
    import sys
    import time
    sys.path.insert(0, "/root")
    import torch
    torch.set_num_threads(4)
    from tournament import tiny as T
    t0 = time.time()
    rec = T.run(T.Cfg(**j["cfg"]), grad=True)
    rec.update(name=j["name"], suite=j["suite"], wall=time.time() - t0)
    os.makedirs(f"/out/tiny/{j['suite']}", exist_ok=True)
    with open(f"/out/tiny/{j['suite']}/{j['name']}.json", "w") as f:
        f.write(json.dumps(rec) + "\n")
    vol.commit()
    return rec


def rows_path(s):
    return HERE / "runs" / f"{s}.jsonl"


def have_names(s):
    p = rows_path(s)
    return {json.loads(l)["name"] for l in open(p)} if p.exists() else set()


@app.local_entrypoint()
def main(suite: str = "main", fetch: bool = False, only: str = "", limit: int = 0) -> None:
    suites = suite.split(",")
    rows_path(suites[0]).parent.mkdir(exist_ok=True)
    have = {s: have_names(s) for s in suites}
    if fetch:
        for s in suites:
            n = 0
            try:
                entries = list(vol.listdir(f"tiny/{s}"))
            except Exception as e:
                print(s, "nothing on the Volume yet:", repr(e)[:200])
                continue
            for entry in entries:
                for row in b"".join(vol.read_file(entry.path)).decode().splitlines():
                    if row.strip() and json.loads(row)["name"] not in have[s]:
                        with open(rows_path(s), "a") as f:
                            f.write(row + "\n")
                        have[s].add(json.loads(row)["name"])
                        n += 1
            print(f"fetched {n} new records from {s}/ -> {rows_path(s)}")
        return
    jobs = [dict(j, suite=s) for s in suites for j in suite_jobs(s) if j["name"] not in have[s]]
    if only:
        keys = only.split(",")
        jobs = [j for j in jobs if any(k_ in j["name"] for k_ in keys)]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} jobs", flush=True)
    for r in run_job.map(jobs, order_outputs=False, return_exceptions=True, wrap_returned_exceptions=False):
        if isinstance(r, Exception):
            print("job failed:", repr(r)[:300], flush=True)
            continue
        s = r["suite"]
        if r["name"] not in have[s]:
            with open(rows_path(s), "a") as f:
                f.write(json.dumps(r) + "\n")
            have[s].add(r["name"])
        print(s, r["name"], "hit", r.get("hit"), "certified", r.get("certified"), "true", r.get("true_wiring"),
              f"{r.get('wall', 0):.0f}s", flush=True)


suite_jobs = suite
