"""Lab runs of the anchors study on Modal CPU containers (app `anchors-lab`, at most 20 containers).

A plan is a JSON list of jobs. Each job names a kind and its arguments:
  {"name", "kind": "tiny", "B", "W", "steps", "seed", "w": {lesson: weight}, "kw": {...}}   tiny.Tiny, every input
  {"name", "kind": "core", "B", "W", "steps", "seed", "w": {...}, "kw": {...}, "rows": n}   core.run on H1 rows
  {"name", "kind": "exact", "B", "W", "lessons": [...], "borrows_true": bool, "limit": k} exact.search
The client waits for every job and writes one JSON line per job to runs/<plan stem>.jsonl.

    ~/github/sotaku/venv/bin/python -m modal run cpu_first/anchors/modal_lab.py --plan cpu_first/anchors/plans/<plan>.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
H1 = str(HERE.parent.parent / "one-layer-deeper" / "data" / "generated" / "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "numpy", index_url="https://download.pytorch.org/whl/cpu")
    .env({"H1_DATA": "/data/h1", "PYTHONPATH": "/lab"})
    .add_local_file(f"{H1}/train.jsonl", "/data/h1/train.jsonl")
    .add_local_file(str(HERE / "core.py"), "/lab/core.py")
    .add_local_file(str(HERE / "tiny.py"), "/lab/tiny.py")
    .add_local_file(str(HERE / "exact.py"), "/lab/exact.py")
)
app = modal.App("anchors-lab", image=image)


@app.function(cpu=2, memory=4096, timeout=3 * 3600, max_containers=8)
def run_job(job: dict) -> dict:
    import torch
    torch.set_num_threads(2)
    t0 = time.time()
    out = {"job": job}
    kind = job["kind"]
    if kind == "tiny":
        import tiny
        import core
        t = tiny.Tiny(job["B"], job["W"], job["w"], job["seed"], **job.get("kw", {}))
        traj = []
        every = job.get("log_every", 50)
        for s in range(1, job["steps"] + 1):
            L = t.step()
            if s % every == 0 or s == job["steps"]:
                st = t.state()
                traj.append({"step": s, "loss": L, "gated_on": t.gated_on, **st})
        Td, Tb = t.tables()
        out.update(final=traj[-1], traj=traj, wrong=core.table_str(Td, Tb, job["B"]),
                   digit="".join(map(str, Td.tolist())), borrow="".join(map(str, Tb.tolist())))
    elif kind == "core":
        import core
        if "tiny_data" in job:                                          # [B, wmin, wmax, n]: made-up rows in base B
            Bt, wmin, wmax, n = job["tiny_data"]
            data = core.tiny_rows(Bt, wmin, wmax, n, seed=job.get("data_seed", 0))
        else:
            data = core.h1_rows(job.get("rows", 4096), seed=job.get("data_seed", 0))
        if "answer_share" in job:                                       # a share of the rows with T <= 2 (answers)
            import torch
            allr = core.h1_rows(None)
            small = {k: v[allr["T"] <= 2] for k, v in allr.items()}
            big = {k: v[allr["T"] > 2] for k, v in allr.items()}
            n = job.get("rows", 4096)
            k = int(round(job["answer_share"] * n))
            g = torch.Generator().manual_seed(job.get("data_seed", 0))
            i1 = torch.randperm(len(small["N"]), generator=g)[:k]
            i2 = torch.randperm(len(big["N"]), generator=g)[:n - k]
            data = {key: torch.cat([small[key][i1], big[key][i2]]) for key in allr}
        lab, traj = core.run(job["B"], job["W"], data, job["w"], steps=job["steps"], seed=job["seed"],
                             log_every=job.get("log_every", 25), verbose=False, **job.get("kw", {}))
        Td, Tb = lab.tables()
        out.update(final=traj[-1], traj=traj, wrong=core.table_str(Td, Tb, job["B"]),
                   digit="".join(map(str, Td.tolist())), borrow="".join(map(str, Tb.tolist())))
    elif kind == "exact":
        import exact
        rows, links = exact.tiny_inputs(job["B"], job["W"])
        inst = [f for nm in job["lessons"] for f in exact.lesson_instances(nm, job["B"], job["W"], rows, links)]
        td, tb = exact.true_table(job["B"])
        sols, count = exact.search(job["B"], inst, fixed_borrows=tb if job.get("borrows_true") else None,
                                   limit=job.get("limit", 50), seconds=job.get("seconds"))
        out.update(groups=len(sols), tables=count, capped=len(sols) >= job.get("limit", 50) or exact.search.timed_out,
                   timed_out=exact.search.timed_out, sols=[exact.show(job["B"], d, b) for d, b in sols])
    out["seconds"] = time.time() - t0
    return out


@app.local_entrypoint()
def main(plan: str):
    jobs = json.loads(Path(plan).read_text())
    dest = HERE / "runs" / (Path(plan).stem + ".jsonl")
    dest.parent.mkdir(exist_ok=True)
    done = set()
    if dest.exists():
        done = {json.loads(l)["job"]["name"] for l in dest.read_text().splitlines() if l.strip()}
    todo = [j for j in jobs if j["name"] not in done]
    print(f"{len(todo)} jobs ({len(done)} already in {dest.name})", flush=True)
    with open(dest, "a") as f:
        for res in run_job.map(todo, return_exceptions=True, order_outputs=False):
            if isinstance(res, Exception):
                print("ERROR", repr(res)[:300], flush=True)
                continue
            f.write(json.dumps(res) + "\n")
            f.flush()
            fin = res.get("final", {})
            print(res["job"]["name"], {k: fin.get(k) for k in ("rows", "digits", "borrows", "closest", "gated_on")}
                  if fin else (res.get("groups"), res.get("tables")), f"{res['seconds']:.0f}s", flush=True)
