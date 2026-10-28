"""Run ext_matched ladders on Modal: one container per ladder (task x model x data), seeds in parallel processes.

    modal run --detach microlab/ext_matched_modal.py --tasks add,mul,sq,mod,sqmod --models colsum,abacus --data fresh --tag M3k
    modal run microlab/ext_matched_modal.py --fetch --tag M3k        # copy finished cells from the Volume instead

Only valid task x model pairs run (ext_matched.FOR). abacus models run on an L4 (on 16 CPUs, 8 threads per seed, with
--cpu-only, when the account's GPU quota is taken); the scan models (colsum, longdiv, colsum_longdiv) run on 8 CPUs,
4 threads per seed (sequential small ops). Cells go to the Volume `microlab-ladder`
under <tag>/ after each size and to microlab/ladder_matched_results.jsonl from the entrypoint. `--extra` passes cell
options as JSON (e.g. '{"lr": 2e-3, "holdout": 4, "sizes": [6, 7]}'); `--start` gives first sizes per ladder.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "models.py", "tasks.py", "run.py", "ext_ladder.py", "ext_matched.py"):
    image = image.add_local_file(HERE / _f, f"/root/microlab/{_f}")
vol = modal.Volume.from_name("microlab-ladder", create_if_missing=True)
app = modal.App("microlab-matched", image=image)
GPU_MODELS = {"abacus", "abacus_loop"}


def _climb(task, model, data, seeds, steps, tag, extra, device):
    import sys
    sys.path.insert(0, "/root")
    from microlab.ext_matched import climb
    os.makedirs(f"/out/{tag}", exist_ok=True)
    extra = dict(extra)
    name = extra.pop("name", "")
    log = f"/out/{tag}/{task}_{model}_{data}{name}.jsonl"
    if os.path.exists(log):                       # a retry starts the ladder over
        os.remove(log)
    sizes = extra.pop("sizes", None)
    threads = extra.pop("threads", 2 if device == "cuda" else 4)
    return climb(task, model=model, data=data, seeds=seeds, steps=steps, sizes=sizes, workers=seeds, threads=threads,
                 device=device, log=log, tag=tag, after=lambda rs: vol.commit(), **extra)


@app.function(gpu="L4", cpu=4, timeout=6 * 3600, volumes={"/out": vol}, retries=modal.Retries(max_retries=1))
def ladder(task: str, model: str, data: str, seeds: int, steps: int, tag: str, extra: dict) -> list[dict]:
    return _climb(task, model, data, seeds, steps, tag, extra, "cuda")


@app.function(cpu=8, timeout=6 * 3600, volumes={"/out": vol}, retries=modal.Retries(max_retries=1))
def ladder_cpu(task: str, model: str, data: str, seeds: int, steps: int, tag: str, extra: dict) -> list[dict]:
    return _climb(task, model, data, seeds, steps, tag, extra, "cpu")


@app.function(cpu=16, timeout=8 * 3600, volumes={"/out": vol}, retries=modal.Retries(max_retries=1))
def ladder_cpu16(task: str, model: str, data: str, seeds: int, steps: int, tag: str, extra: dict) -> list[dict]:
    return _climb(task, model, data, seeds, steps, tag, {"threads": 8, **extra}, "cpu")


@app.local_entrypoint()
def main(tasks: str = "mul", models: str = "colsum", data: str = "fresh", seeds: int = 2, steps: int = 3000,
         tag: str = "", extra: str = "{}", start: str = "{}", fetch: bool = False, cpu_only: bool = False) -> None:
    out = HERE / "ladder_matched_results.jsonl"
    if fetch:
        n = 0
        for entry in vol.listdir(tag):
            rows = b"".join(vol.read_file(entry.path)).decode().splitlines()
            with open(out, "a") as f:
                f.writelines(r + "\n" for r in rows if r.strip())
            n += len(rows)
        print(f"fetched {n} cells from {tag}/ -> {out}")
        return
    import sys
    sys.path.insert(0, str(HERE.parent))
    from microlab.ext_matched import FOR, SIZES
    from microlab.tasks import TASKS
    first = json.loads(start)
    jobs = []
    for t in tasks.split(","):
        for m in models.split(","):
            if t not in FOR[m]:
                continue
            for d in data.split(","):
                kw = json.loads(extra)
                if f"{t},{m},{d}" in first:
                    kw["sizes"] = [s for s in SIZES[TASKS[t].kind] if s >= first[f"{t},{m},{d}"]]
                elif first:
                    continue
                jobs.append((t, m, d, seeds, steps, tag, kw))
    q: queue.Queue = queue.Queue()

    def drain(fn, js):
        for rows in fn.starmap(js, order_outputs=False, return_exceptions=True):
            q.put(rows)
        q.put(None)

    gpu_fn = ladder_cpu16 if cpu_only else ladder                 # --cpu-only: transformers on 16 CPUs instead
    groups = [(gpu_fn, [j for j in jobs if j[1] in GPU_MODELS]), (ladder_cpu, [j for j in jobs if j[1] not in GPU_MODELS])]
    threads = [threading.Thread(target=drain, args=g, daemon=True) for g in groups if g[1]]
    for th in threads:
        th.start()
    total, finished = 0, 0
    while finished < len(threads):
        rows = q.get()
        if rows is None:
            finished += 1
            continue
        if isinstance(rows, Exception):
            print("ladder failed:", repr(rows)[:300], flush=True)
            continue
        with open(out, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        total += len(rows)
        r = rows[-1]
        print(f"done {r['task']} {r['model']} {r['data']}: sizes "
              + " ".join(f"{x['size']}:{x['exact']:.2f}" for x in rows), flush=True)
    print(f"{total} cells appended to {out}")
