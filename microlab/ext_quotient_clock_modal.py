"""Run ext_quotient_clock cells on Modal: one CPU container per cell (variant x size x steps x seed x holdout).

    modal run --detach microlab/ext_quotient_clock_modal.py --task modq --sizes 6,7 --steps 3000 --tag C1 \
        --extra '[{"pick": "given", "score": "ce"}, {"pick": "given", "score": "clock"}]'
    modal run microlab/ext_quotient_clock_modal.py --fetch --tag C1        # copy finished cells from the Volume instead
    modal run --detach microlab/ext_quotient_clock_modal.py --grad --sizes 6,7,10 --pre 0,3000 --tag C3g   # run 3

Each cell trains with 4 threads on a 4-CPU container (as ext_quotient_modal.py), writes its row to the Volume
`microlab-quotient-clock` under <tag>/, and the entrypoint appends it to microlab/results_quotient_clock.jsonl
(`--fetch` adds the Volume's rows that are not there yet). `--extra` is one JSON object of cell options, or a list of
them (one variant each), on top of longdiv's lr 3e-3; `name` only names the Volume file. Each cell also saves its
trained weights next to its row (<file>.pt). `--grad` runs ext_quotient_clock.gradcheck instead (one container per
size x pre-training x seed; rows to microlab/results_quotient_clock_grad.jsonl).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "models.py", "tasks.py", "run.py", "ext_ladder.py", "ext_matched.py", "ext_quotient.py",
           "ext_quotient_report.py", "ext_quotient_clock.py"):
    image = image.add_local_file(HERE / _f, f"/root/microlab/{_f}")
vol = modal.Volume.from_name("microlab-quotient-clock", create_if_missing=True)
app = modal.App("microlab-quotient-clock", image=image)
OUT = HERE / "results_quotient_clock.jsonl"
GRAD = HERE / "results_quotient_clock_grad.jsonl"


def _key(r):
    return (r["tag"], r["task"], r.get("train_task"), r["model"], r["size"], r["steps"], r["seed"], r.get("holdout", 0),
            json.dumps(r["model_kw"], sort_keys=True))


def _file(spec, name):
    kw = "_".join(f"{k}{v}" for k, v in sorted(spec.items()) if k in ("pick", "score", "noise", "train_on"))
    return (f"{spec['task_name']}_{kw}_{spec['size']}_{spec['steps']}_s{spec['seed']}_ho{spec.get('holdout', 0)}"
            f"{name}.json")


@app.function(cpu=4, timeout=6 * 3600, volumes={"/out": vol}, retries=modal.Retries(max_retries=1))
def run_cell(spec: dict) -> dict:
    import os
    import sys
    sys.path.insert(0, "/root")
    from microlab.ext_quotient_clock import cell
    spec = dict(spec)
    name = spec.pop("name", "")
    os.makedirs(f"/out/{spec['tag']}", exist_ok=True)
    path = f"/out/{spec['tag']}/{_file(spec, name)}"
    r = cell(log=None, threads=4, save=path.removesuffix(".json") + ".pt", **spec)
    with open(path, "w") as f:                       # a retry rewrites its own file
        f.write(json.dumps(r) + "\n")
    vol.commit()
    return r


@app.function(cpu=4, timeout=6 * 3600, volumes={"/out": vol}, retries=modal.Retries(max_retries=1))
def grad_cell(spec: dict) -> dict:
    import os
    import sys
    sys.path.insert(0, "/root")
    from microlab.ext_quotient_clock import gradcheck
    spec = dict(spec)
    tag = spec.pop("tag")
    os.makedirs(f"/out/{tag}", exist_ok=True)
    r = gradcheck(threads=4, **spec) | {"tag": tag}
    with open(f"/out/{tag}/grad_{spec['size']}_pre{spec['pre_steps']}_s{spec['seed']}.json", "w") as f:
        f.write(json.dumps(r) + "\n")
    vol.commit()
    return r


@app.local_entrypoint()
def main(sizes: str = "6,7", steps: str = "3000", seeds: int = 2, holdout: str = "0", task: str = "mod",
         tag: str = "", extra: str = "{}", fetch: bool = False, grad: bool = False, pre: str = "0") -> None:
    import sys
    sys.path.insert(0, str(HERE.parent))
    have = {_key(json.loads(l)) for l in open(OUT)} if OUT.exists() else set()
    if fetch:
        n = 0
        for entry in vol.listdir(tag):
            for row in b"".join(vol.read_file(entry.path)).decode().splitlines():
                if row.strip() and _key(json.loads(row)) not in have:
                    with open(OUT, "a") as f:
                        f.write(row + "\n")
                    have.add(_key(json.loads(row)))
                    n += 1
        print(f"fetched {n} new cells from {tag}/ -> {OUT}")
        return
    if grad:
        specs = [dict(size=int(s), pre_steps=int(p), seed=sd, tag=tag) for s in sizes.split(",") for p in pre.split(",")
                 for sd in range(seeds)]
        print(f"{len(specs)} gradient checks", flush=True)
        for r in grad_cell.map(specs, order_outputs=False, return_exceptions=True):
            if isinstance(r, Exception):
                print("check failed:", repr(r)[:300], flush=True)
                continue
            with open(GRAD, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"grad {r['size']} bits pre {r['pre_steps']} s{r['seed']} losses {r['losses']} "
                  + " | ".join(f"B={b['batch']}: clock {b['clock_vs_true']:+.2f} ce {b['ce_vs_true']:+.2f}"
                               for b in r["by_batch"]), flush=True)
        return
    from microlab.ext_quotient_clock import name, show_clock
    variants = json.loads(extra)
    specs = [dict(task_name=task, size=int(s), model="longdiv_clock", steps=int(st), seed=sd, holdout=int(h), tag=tag,
                  **{"lr": 3e-3, **x})
             for x in (variants if isinstance(variants, list) else [variants]) for st in steps.split(",")
             for s in sizes.split(",") for h in holdout.split(",") for sd in range(seeds)]
    print(f"{len(specs)} cells", flush=True)
    for r in run_cell.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("cell failed:", repr(r)[:300], flush=True)
            continue
        if _key(r) not in have:
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
            have.add(_key(r))
        print(name(r) + " | " + show_clock(r), flush=True)
