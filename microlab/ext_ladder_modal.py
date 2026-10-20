"""Run ext_ladder climbs on Modal: one container per ladder (task x model x data), seeds in parallel processes.

    modal run --detach microlab/ext_ladder_modal.py --tasks add,mul --models mlp,xf,loop --data fresh,pool1000 --tag S3k
    modal run microlab/ext_ladder_modal.py --fetch --tag S3k        # copy finished cells from the Volume instead
    modal run microlab/ext_ladder_modal.py --quotients '[{"task_name": "sqmod", "size": 7, "pool": 1000}]'

xf and loop ladders run on an L4; mlp ladders (and everything with --cpu-only) run on 8 CPUs, 4 threads per seed.
Every cell is written to the Volume `microlab-ladder` under <tag>/ as soon as its size finishes, and the entrypoint
appends all returned cells to microlab/ladder_results.jsonl (several launchers at once can interleave there; the
Volume's per-ladder files are the durable copy). `--extra` passes cell options as JSON, e.g. '{"d": 256, "depth": 6}',
'{"holdout": 4}' or '{"sizes": [8, 9]}'; `--start` gives per-ladder first sizes as JSON {"task,model,data": size}.
Only the five .py files a cell imports are uploaded, so other sessions' files in microlab/ never enter the image.
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
for _f in ("__init__.py", "models.py", "tasks.py", "run.py", "ext_ladder.py"):   # only what a cell imports
    image = image.add_local_file(HERE / _f, f"/root/microlab/{_f}")
vol = modal.Volume.from_name("microlab-ladder", create_if_missing=True)
app = modal.App("microlab-ladder", image=image)


def _climb(task, model, data, seeds, steps, tag, extra, device):
    import sys
    sys.path.insert(0, "/root")
    from microlab.ext_ladder import climb
    os.makedirs(f"/out/{tag}", exist_ok=True)
    log = f"/out/{tag}/{task}_{model}_{data}.jsonl"
    if os.path.exists(log):                       # a retry starts the ladder over
        os.remove(log)
    extra = dict(extra)
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


@app.function(cpu=4, timeout=2 * 3600, volumes={"/out": vol})
def quotient(kw: dict) -> dict:
    """ext_ladder.by_quotient for one configuration, saved under quotient/ on the Volume."""
    import sys
    sys.path.insert(0, "/root")
    from microlab.ext_ladder import by_quotient
    res = {"args": kw, **by_quotient(**kw)}
    os.makedirs("/out/quotient", exist_ok=True)
    with open("/out/quotient/results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    vol.commit()
    return res


@app.local_entrypoint()
def main(tasks: str = "add", models: str = "xf", data: str = "fresh", seeds: int = 2, steps: int = 3000,
         tag: str = "", extra: str = "{}", start: str = "{}", fetch: bool = False, quotients: str = "",
         cpu_only: bool = False) -> None:
    out = HERE / "ladder_results.jsonl"
    if quotients:                                  # JSON list of by_quotient keyword dicts
        with open(HERE / "ladder_quotient.jsonl", "a") as f:
            for res in quotient.map(json.loads(quotients), order_outputs=False):
                print(json.dumps(res), flush=True)
                f.write(json.dumps(res) + "\n")
        return
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
    from microlab.ext_ladder import SIZES
    from microlab.tasks import TASKS
    first = json.loads(start)
    jobs = []
    for t in tasks.split(","):
        for m in models.split(","):
            for d in data.split(","):
                kw = json.loads(extra)
                if f"{t},{m},{d}" in first:
                    kw["sizes"] = [s for s in SIZES[TASKS[t].kind] if s >= first[f"{t},{m},{d}"]]
                elif first:
                    continue                     # --start lists exactly the ladders to run
                jobs.append((t, m, d, seeds, steps, tag, kw))
    q: queue.Queue = queue.Queue()

    def drain(fn, js):
        for rows in fn.starmap(js, order_outputs=False, return_exceptions=True):
            q.put(rows)
        q.put(None)

    on_cpu = lambda j: cpu_only or j[1] == "mlp"
    groups = [(ladder_cpu, [j for j in jobs if on_cpu(j)]), (ladder, [j for j in jobs if not on_cpu(j)])]
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
