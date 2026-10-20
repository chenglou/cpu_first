"""Run ext_quotient cells on Modal: one CPU container per cell (model x size x steps x seed x holdout).

    modal run --detach microlab/ext_quotient_modal.py --models longdiv_est,longdiv_estc --sizes 6,7 --steps 3000 --tag Q3k
    modal run microlab/ext_quotient_modal.py --fetch --tag Q3k        # copy finished cells from the Volume instead

Each cell trains with 4 threads on a 4-CPU container (the scan models are sequential small ops), writes its row to
the Volume `microlab-quotient` under <tag>/, and the entrypoint appends it to microlab/results_quotient.jsonl
(`--fetch` adds the Volume's rows that are not there yet). `--extra` passes cell options as JSON on top of
longdiv's lr 3e-3, e.g. '{"kr": 4, "kn": 3, "name": "_kr4kn3"}'; `--holdout 0,4` runs both.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "models.py", "tasks.py", "run.py", "ext_ladder.py", "ext_matched.py", "ext_quotient.py"):
    image = image.add_local_file(HERE / _f, f"/root/microlab/{_f}")
vol = modal.Volume.from_name("microlab-quotient", create_if_missing=True)
app = modal.App("microlab-quotient", image=image)
OUT = HERE / "results_quotient.jsonl"


def _key(r):
    return (r["tag"], r["task"], r.get("train_task"), r["model"], r["size"], r["steps"], r["seed"], r.get("holdout", 0),
            json.dumps(r["model_kw"], sort_keys=True))


@app.function(cpu=4, timeout=5 * 3600, volumes={"/out": vol}, retries=modal.Retries(max_retries=1))
def run_cell(spec: dict) -> dict:
    import os
    import sys
    sys.path.insert(0, "/root")
    from microlab.ext_quotient import cell
    spec = dict(spec)
    name = spec.pop("name", "")
    os.makedirs(f"/out/{spec['tag']}", exist_ok=True)
    r = cell(log=None, threads=4, **spec)
    path = (f"/out/{spec['tag']}/{spec['task_name']}_{spec['model']}_{spec['size']}_{spec['steps']}_s{spec['seed']}"
            f"_ho{spec.get('holdout', 0)}{name}.json")
    with open(path, "w") as f:                       # a retry rewrites its own file
        f.write(json.dumps(r) + "\n")
    vol.commit()
    return r


@app.function(cpu=4, timeout=3600)
def fit_check(steps: int, lr: float) -> dict:
    """Fit the digit cell to its truth table; report the share of entries right and longdiv_tr's exact with it."""
    import sys
    import time
    sys.path.insert(0, "/root")
    import numpy as np
    import torch
    from microlab.ext_quotient import MODELS
    from microlab.run import _tensors
    from microlab.tasks import TASKS
    torch.set_num_threads(4)
    torch.manual_seed(0)
    t0 = time.time()
    in_w, out_w, _ = TASKS["mod"].widths(6)
    net = MODELS["longdiv_tr"](in_w, out_w, {}, fit_cell=steps, fit_lr=lr, freeze_cell=True)
    out = {"steps": steps, "lr": lr, "cell_fit": round(net.cell_fit, 5), "seconds": round(time.time() - t0, 1)}
    for b in (6, 7, 10, 14):
        bt = TASKS["mod"].sample(1000, b, np.random.default_rng(1))
        x, y, _ = _tensors(bt, "cpu")
        with torch.no_grad():
            out[f"exact@{b}"] = round((net(x)["target"].argmax(-1) == y).all(1).float().mean().item(), 4)
    return out


@app.local_entrypoint()
def main(models: str = "longdiv_estc", sizes: str = "6,7", steps: str = "3000", seeds: int = 2, holdout: str = "0",
         task: str = "mod", tag: str = "", extra: str = "{}", fetch: bool = False, fitcheck: str = "") -> None:
    import sys
    sys.path.insert(0, str(HERE.parent))
    if fitcheck:                                     # e.g. "3000:1e-2,6000:3e-3"
        for r in fit_check.starmap([(int(a), float(b)) for a, b in (s.split(":") for s in fitcheck.split(","))]):
            print(json.dumps(r), flush=True)
        return
    have = {_key(json.loads(l)) for l in open(OUT)} if OUT.exists() else set()
    if fetch:
        n = 0
        for entry in vol.listdir(tag):
            for row in b"".join(vol.read_file(entry.path)).decode().splitlines():
                if row.strip() and _key(json.loads(row)) not in have:
                    with open(OUT, "a") as f:
                        f.write(row + "\n")
                    n += 1
        print(f"fetched {n} new cells from {tag}/ -> {OUT}")
        return
    from microlab.ext_quotient import show_q
    specs = [dict(task_name=task, size=int(s), model=m, steps=int(st), seed=sd, holdout=int(h), tag=tag,
                  **{"lr": 3e-3, **json.loads(extra)})
             for m in models.split(",") for s in sizes.split(",") for st in steps.split(",")
             for h in holdout.split(",") for sd in range(seeds)]
    print(f"{len(specs)} cells", flush=True)
    for r in run_cell.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("cell failed:", repr(r)[:300], flush=True)
            continue
        if _key(r) not in have:
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
        print(show_q(r) + f"  ho{r.get('holdout', 0)} {r['steps']}", flush=True)
