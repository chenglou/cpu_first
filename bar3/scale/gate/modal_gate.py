"""The gate study on Modal (GATE.md): CPU containers, the compiled CPU engine, its own app (`bar3-gate`) and Volume
(`bar3-gate`, under v1/). Three kinds of call:
- `capture`: round 0 of the learner for several runs in one container (capture.py), the corpus file on the Volume;
- `evaluate`: search jobs under several arms (plain, gated, pruned, guided; gjobs.search_job), from a task file on the
  Volume, results back to the Volume;
- `learn`: whole learner runs with the gate as a second search where the plain one finds nothing (learnrun.py).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/gate/modal_gate.py --capture c0a --specs alloc_deep_ct2a2_probe:h1,alloc_deep_ct2a2_probe:rho
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal volume get bar3-gate v1/capture_c0a.jsonl.gz bar3/scale/gate/records/
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal volume put bar3-gate TASKS.jsonl.gz v1/tasks/NAME.jsonl.gz
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/gate/modal_gate.py --evaluate NAME --shards 4

Limits: every search has its job's state and time budget (the arms use the hard cap); a container has a timeout
(capture and evaluate 6 h, learn 12 h) and a hard memory limit of 1.5 times its request; workers per container are
fewer than its cores, for the 1M-state searches' memory; at most 8 containers at once (max_containers).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
DIRS = ("library", "library/inventing", "bar3", "bar3/breadth", "bar3/scale", "bar3/scale/subgoal",
        "bar3/scale/heldout", "bar3/scale/newton", "bar3/scale/digits", "bar3/scale/rowcheck", "bar3/scale/picks",
        "bar3/scale/reach", "bar3/scale/consolidate", "bar3/scale/gate")


def _with_files(image):
    for d in DIRS:
        for f in sorted((ROOT / d).glob("*.py")):
            image = image.add_local_file(f, f"/root/{d}/{f.name}")
    for f in sorted((ROOT / "bar3/scale/records").glob("seq_ct2a2_clean_new_targets_L1.state.*.json")):
        image = image.add_local_file(f, f"/root/bar3/scale/records/{f.name}")
    for f in sorted((HERE / "models").glob("*.json")) if (HERE / "models").exists() else []:
        image = image.add_local_file(f, f"/root/bar3/scale/gate/models/{f.name}")
    return image


image = _with_files(modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.4", "numba==0.68.0"))
vol = modal.Volume.from_name("bar3-gate", create_if_missing=True)
app = modal.App("bar3-gate", image=image)
V = "/out/v1"
CPU = int(os.environ.get("GATE_CPU", "16"))
MEM = int(os.environ.get("GATE_MEM", "65536"))
WORKERS = int(os.environ.get("GATE_WORKERS", "12"))


@app.function(cpu=CPU, memory=(MEM, MEM * 3 // 2), timeout=6 * 3600, volumes={"/out": vol}, max_containers=8)
def capture(label: str, specs: list, workers: int = WORKERS) -> dict:
    import sys
    import time
    sys.path.insert(0, "/root")
    os.chdir("/root")
    from bar3.scale.gate import capture as C
    os.makedirs(V, exist_ok=True)
    out = f"{V}/capture_{label}.jsonl.gz"
    t0 = time.time()
    C.capture_many(out, [tuple(s) for s in specs], workers)
    vol.commit()
    return dict(label=label, runs=[s[0] for s in specs], seconds=round(time.time() - t0))


def _eval_one(task):
    from bar3.breadth import targets as TG
    from bar3.scale.gate import gjobs
    job = task["job"]
    if job.get("breadth_target"):
        TG.use(job["breadth_target"])
    out = dict(id=task["id"], arm=task["arm"])
    try:
        r = gjobs.search_job(job)
        out.update({k: r.get(k) for k in ("found_cost", "level_sizes", "stop", "first_goal", "first_accept",
                                          "generated", "states", "seconds", "wall", "dead", "delays", "n_classes")})
        out["programs"] = [p["program"] for p in (r.get("programs") or [])][:5]
        out["primitive_cost"] = min((p["primitive_cost"] for p in (r.get("programs") or [])), default=None)
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:300]
    return out


@app.function(cpu=CPU, memory=(MEM, MEM * 3 // 2), timeout=6 * 3600, volumes={"/out": vol}, max_containers=8)
def evaluate(name: str, shard: int, shards: int, workers: int = WORKERS) -> dict:
    """Tasks shard::shards of v1/tasks/<name>.jsonl.gz (one line: dict(id, arm, job)), results to
    v1/results/<name>_<shard>.jsonl, appended as they finish (a rerun skips the ones there)."""
    import concurrent.futures as cf
    import gzip
    import multiprocessing as mp
    import sys
    import time
    sys.path.insert(0, "/root")
    os.chdir("/root")
    from bar3.scale.gate.capture import _warm
    tasks = [json.loads(line) for i, line in enumerate(gzip.open(f"{V}/tasks/{name}.jsonl.gz", "rt"))
             if i % shards == shard]
    os.makedirs(f"{V}/results", exist_ok=True)
    path = f"{V}/results/{name}_{shard}.jsonl"
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line)
                done.add((d["id"], d["arm"]))
            except ValueError:
                pass
    todo = [t for t in tasks if (t["id"], t["arm"]) not in done]
    t0 = time.time()
    last = t0
    with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_warm) as ex:
        futs = [ex.submit(_eval_one, t) for t in todo]
        with open(path, "a") as f:
            for fu in cf.as_completed(futs):
                f.write(json.dumps(fu.result()) + "\n")
                if time.time() - last > 60:
                    f.flush()
                    vol.commit()
                    last = time.time()
    vol.commit()
    return dict(name=name, shard=shard, tasks=len(tasks), ran=len(todo), seconds=round(time.time() - t0))


@app.function(cpu=8, memory=(65536, 98304), timeout=4 * 3600, volumes={"/out": vol}, max_containers=4)
def train(corpus: str, labels: list, b: float, groups: list) -> dict:
    """The gates of one corpus (train.train_all) from the capture files v1/capture_<label>.jsonl.gz, to v1/models/."""
    import sys
    sys.path.insert(0, "/root")
    os.chdir("/root")
    from bar3.scale.gate import train as T
    vol.reload()
    os.makedirs(f"{V}/models", exist_ok=True)
    out = T.train_all([f"{V}/capture_{x}.jsonl.gz" for x in labels], corpus, b, groups, f"{V}/models")
    with open(f"{V}/models/judge_{corpus}.json", "w") as f:
        json.dump(out, f, indent=1)
    vol.commit()
    return out


@app.function(cpu=CPU, memory=(MEM, MEM * 3 // 2), timeout=12 * 3600, volumes={"/out": vol}, max_containers=8)
def learn(label: str, runs: list, gates: dict, workers: int = WORKERS, anchor_plain: bool = False) -> dict:
    """Whole learner runs (learnrun.py), a thread each, records to v1/learn/<label>/; anchor_plain: GATE.md §9."""
    import sys
    import time
    sys.path.insert(0, "/root")
    os.chdir("/root")
    from bar3.scale.gate import learnrun
    out = f"{V}/learn/{label}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    learnrun.run_many(runs, gates, out, workers, commit=vol.commit, anchor_plain=anchor_plain)
    vol.commit()
    return dict(label=label, runs=[r[0] for r in runs], seconds=round(time.time() - t0))


@app.local_entrypoint()
def main(capture_label: str = "", specs: str = "", evaluate_name: str = "", shards: int = 1, workers: int = WORKERS,
         learn_label: str = "", runs: str = "", gates: str = "", train_corpus: str = "", labels: str = "",
         b: float = 10.0, groups: str = "full,noprob,fitonly,noanswer,nodet", anchor_plain: int = 0):
    """--capture-label L --specs CONFIG:TARGET[@STATE],...   one container, one corpus file;
    --evaluate-name NAME --shards K                          K containers over the task file;
    --learn-label L --runs NAME:CONFIG:TARGET,... --gates JSON  one container, whole runs (--anchor-plain 1: §9)."""
    if capture_label:
        sp = []
        for s in specs.split(","):
            config, rest = s.split(":")
            tname, _, state = rest.partition("@")
            st = f"bar3/scale/records/{state}" if state else None
            sp.append([f"{config}_{tname}", config, tname, st])                # one corpus per label: C0, C1
        call = capture.spawn(capture_label, sp, workers)
        print("spawned capture", call.object_id, [s[0] for s in sp])
    if evaluate_name:
        for k in range(shards):
            call = evaluate.spawn(evaluate_name, k, shards, workers)
            print("spawned evaluate", evaluate_name, k, call.object_id)
    if train_corpus:
        call = train.spawn(train_corpus, labels.split(","), b, groups.split(","))
        print("spawned train", train_corpus, call.object_id)
    if learn_label:
        rs = [r.split(":") for r in runs.split(",")]
        call = learn.spawn(learn_label, rs, json.loads(gates), workers, bool(anchor_plain))
        print("spawned learn", learn_label, call.object_id, [r[0] for r in rs], "anchor_plain" if anchor_plain else "")
