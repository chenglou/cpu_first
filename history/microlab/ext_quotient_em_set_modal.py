"""Run ext_quotient_em_set cells on Modal: one CPU container per cell (variant x size x steps x seed x holdout).

    modal run --detach microlab/ext_quotient_em_set_modal.py --models longdiv_em_set,longdiv_em --sizes 6,7 --steps 3000,12000 \
        --tag S1
    modal run microlab/ext_quotient_em_set_modal.py --fetch --tag S1        # copy finished cells from the Volume instead

Each cell trains with 4 threads on a 4-CPU container (as ext_quotient_modal.py), writes its row to the Volume
`microlab-quotient-em-set` under <tag>/, and the entrypoint appends it to microlab/results_quotient_em_set.jsonl (`--fetch`
adds the Volume's rows that are not there yet). `--extra` is one JSON object of cell options, or a list of them (one
variant each), on top of longdiv's lr 3e-3; `name` only names the Volume file.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "models.py", "tasks.py", "run.py", "ext_ladder.py", "ext_matched.py", "ext_quotient.py",
           "ext_quotient_report.py", "ext_quotient_em.py", "ext_quotient_em_set.py"):
    image = image.add_local_file(HERE / _f, f"/root/microlab/{_f}")
vol = modal.Volume.from_name("microlab-quotient-em-set", create_if_missing=True)
app = modal.App("microlab-quotient-em-set", image=image)
OUT = HERE / "results_quotient_em_set.jsonl"


def _key(r):
    return (r["tag"], r["task"], r.get("train_task"), r["model"], r["size"], r["steps"], r["seed"], r.get("holdout", 0),
            json.dumps(r["model_kw"], sort_keys=True))


@app.function(cpu=4, timeout=6 * 3600, volumes={"/out": vol}, retries=modal.Retries(max_retries=1))
def run_cell(spec: dict) -> dict:
    import os
    import sys
    sys.path.insert(0, "/root")
    from microlab.ext_quotient_em_set import cell
    spec = dict(spec)
    name = spec.pop("name", "")
    os.makedirs(f"/out/{spec['tag']}", exist_ok=True)
    path = (f"/out/{spec['tag']}/{spec['task_name']}_{spec['model']}_{spec['size']}_{spec['steps']}_s{spec['seed']}"
            f"_ho{spec.get('holdout', 0)}{name}.json")
    r = cell(log=None, threads=4, save=path.removesuffix(".json") + ".pt", **spec)
    with open(path, "w") as f:                       # a retry rewrites its own file
        f.write(json.dumps(r) + "\n")
    vol.commit()
    return r


@app.local_entrypoint()
def main(models: str = "longdiv_em_set,longdiv_em", sizes: str = "6,7", steps: str = "3000", seeds: int = 2, holdout: str = "0",
         task: str = "mod", tag: str = "", extra: str = "{}", fetch: bool = False) -> None:
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
    from microlab.ext_quotient_em import show_q
    variants = json.loads(extra)
    specs = [dict(task_name=task, size=int(s), model=m, steps=int(st), seed=sd, holdout=int(h), tag=tag,
                  **{"lr": 3e-3, **x})
             for x in (variants if isinstance(variants, list) else [variants]) for m in models.split(",")
             for st in steps.split(",") for s in sizes.split(",") for h in holdout.split(",") for sd in range(seeds)]
    print(f"{len(specs)} cells", flush=True)
    for r in run_cell.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("cell failed:", repr(r)[:300], flush=True)
            continue
        if _key(r) not in have:
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
            have.add(_key(r))
        print(show_q(r) + f"  ho{r.get('holdout', 0)} {r['steps']}", flush=True)
