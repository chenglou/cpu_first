"""Run NCA sweeps on Modal: one container per job (job lists in suites.py): CPU (at most 32 at once), or an L4 GPU
for jobs marked "gpu" (at most 11 at once; jobs with the same "pack" share one, each in its own process).

    ~/github/sotaku/venv/bin/python -m modal run --detach nca/modal_nca.py --suite explore
    ~/github/sotaku/venv/bin/python -m modal run nca/modal_nca.py --suite explore --fetch
    ~/github/sotaku/venv/bin/python -m modal run nca/modal_nca.py --suite trim1 --peek

The launch spawns one driver inside the app and returns; the driver spawns the jobs (so the sweep does not depend on
the laptop staying connected). Each job writes its record to the Volume `nca` under <VERSION>/<suite>/<name>.json;
`--fetch` appends the Volume's records that are not in nca/runs/<suite>.jsonl yet. A launch skips jobs whose record
is already on the Volume or in the local file. Modal restarts a killed container's input on its own (out of memory,
preemption); a job that starts twice in a row (per launch) without writing a new checkpoint is reported failed
instead of starting again: the starts are counted on the Volume. A run with a curve (`curve_every`) saves itself at every curve point under
<VERSION>/.ckpt/<suite>/ and resumes from there when restarted; `--peek` prints each such run's latest curve point.
Modal hands a container that has finished one job the next, and a container sees the Volume as it was when it
started, so every job first reloads the Volume: without that, a restarted run that landed on such a container could
miss its own checkpoint and start over (one rest1 run did, on 2026-10-24).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "numpy==2.0.2")
for _f in ("__init__.py", "grid.py", "model.py", "train.py", "compile.py", "suites.py", "logic.py", "verify.py",
           "hyper.py"):
    if (HERE / _f).exists():
        image = image.add_local_file(HERE / _f, f"/root/nca/{_f}")
vol = modal.Volume.from_name("nca", create_if_missing=True)
VERSION = "v1"
app = modal.App("nca", image=image)


def _prepare(job: dict, device: str):
    """In the container's own process: skip a finished job, count this start (committed at once, so that a container
    that dies still counts it; a new checkpoint since the last start resets the count), give a train job its device
    and checkpoint path. Returns (job, None) to run, or (None, result) when there is nothing to run."""
    import os
    try:
        vol.reload()                      # the Volume as it is now, not as it was when this container started
    except Exception as e:                # (it refuses while a file is open; the job then sees the older view)
        print("Volume reload failed:", repr(e)[:200], flush=True)
    job = dict(job)
    launch = job.pop("launch", "")
    job.pop("gpu", None)
    job.pop("pack", None)
    if os.path.exists(f"/out/{VERSION}/{job['suite']}/{job['name']}.json"):   # a restarted input never redoes a job
        return None, {"name": job["name"], "suite": job["suite"], "skipped": True}
    tries = f"/out/{VERSION}/.tries/{job['suite']}/{job['name']}.{launch}"
    peek = f"/out/{VERSION}/.ckpt/{job['suite']}/{job['name']}.pt.json"
    step = json.load(open(peek))["step"] if os.path.exists(peek) else 0
    last = json.load(open(tries)) if os.path.exists(tries) else {"n": 0, "step": -1}
    n = 1 if step > last["step"] else last["n"] + 1           # starts in a row without a new checkpoint
    os.makedirs(os.path.dirname(tries), exist_ok=True)
    with open(tries, "w") as f:
        json.dump({"n": n, "step": step}, f)
    vol.commit()
    if n > 2:
        return None, {"name": job["name"], "suite": job["suite"], "failed": f"{n} starts without a new checkpoint"}
    if job.get("kind", "train") == "train":
        if device != "cpu":
            job["device"] = device
        if job.get("curve_every"):
            job["ckpt"] = f"/out/{VERSION}/.ckpt/{job['suite']}/{job['name']}.pt"
    return job, None


def _run(job: dict, device: str) -> dict:
    """One job in this process."""
    import sys
    sys.path.insert(0, "/root")
    job, done = _prepare(job, device)
    if done:
        return done
    from nca import suites, train as T
    T.CKPT_HOOK = vol.commit
    out = suites.run_record(job, f"/out/{VERSION}/{job['suite']}/{job['name']}.json", threads=4)
    vol.commit()
    return out


def _run_many(jobs: list, device: str) -> list:
    """Several jobs in one container, each in its own process (on a GPU a job's Python is the bottleneck, not the
    GPU); the children die with this process. They never commit the Volume: this process commits every 5 minutes
    (their checkpoints) and when they are done."""
    import multiprocessing as mp
    import sys
    import threading
    sys.path.insert(0, "/root")
    from nca import suites
    todo, out = [], []
    for j in jobs:
        job, done = _prepare(j, device)
        (out.append(done) if done else todo.append(job))
    if not todo:
        return out
    stop = threading.Event()

    def commits():
        while not stop.wait(300):
            vol.commit()
    threading.Thread(target=commits, daemon=True).start()
    try:
        with mp.get_context("spawn").Pool(len(todo)) as pool:
            calls = [pool.apply_async(suites.run_record, (j, f"/out/{VERSION}/{j['suite']}/{j['name']}.json", 2))
                     for j in todo]
            for j, call in zip(todo, calls):
                try:
                    out.append(call.get())
                except Exception as e:
                    out.append({"name": j["name"], "suite": j["suite"], "failed": repr(e)[:300]})
    finally:
        stop.set()
        vol.commit()
    return out


@app.function(cpu=4, memory=(2048, 8192), timeout=20 * 3600, volumes={"/out": vol},
              retries=modal.Retries(max_retries=1), max_containers=32)
def run_job(job: dict) -> dict:
    return _run(job, "cpu")


@app.function(gpu="L4", cpu=4, memory=(4096, 24576), timeout=20 * 3600, volumes={"/out": vol},
              retries=modal.Retries(max_retries=1), max_containers=11)
def run_jobs_gpu(jobs: list) -> list:
    """The jobs of one pack (the same "pack" key) on one GPU."""
    return [_run(jobs[0], "cuda")] if len(jobs) == 1 else _run_many(jobs, "cuda")


@app.function(cpu=0.25, memory=512, timeout=24 * 3600, nonpreemptible=True)
def driver(jobs: list) -> None:
    packs = {}
    for j in jobs:
        if j.get("gpu"):
            packs.setdefault(j.get("pack") or j["name"], []).append(j)
    calls = ([(j["name"], run_job.spawn(j)) for j in jobs if not j.get("gpu")] +
             [("+".join(x["name"] for x in pk), run_jobs_gpu.spawn(pk)) for pk in packs.values()])
    for n, (name, call) in enumerate(calls, 1):
        try:
            print(n, call.get(), flush=True)
        except Exception as e:
            print("job failed:", name, repr(e)[:300], flush=True)


def rows_path(suite):
    return HERE / "runs" / f"{suite}.jsonl"


def have_names(suite):
    out = rows_path(suite)
    return {json.loads(l)["name"] for l in open(out)} if out.exists() else set()


def volume_names(suite):
    try:
        return {Path(e.path).stem for e in vol.listdir(f"{VERSION}/{suite}")}
    except Exception:
        return set()


@app.local_entrypoint()
def main(suite: str = "explore", fetch: bool = False, limit: int = 0, only: str = "", dry: bool = False,
         peek: bool = False) -> None:
    """--suite takes one suite or several separated by commas. --only keeps jobs whose name contains one of the
    comma-separated strings."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    suites = suite.split(",")
    if peek:
        for s in suites:
            try:
                entries = sorted(e.path for e in vol.listdir(f"{VERSION}/.ckpt/{s}") if e.path.endswith(".json"))
            except Exception as e:
                print(s, "no checkpoints yet:", repr(e)[:200])
                continue
            for path in entries:
                pk = json.loads(b"".join(vol.read_file(path)))
                last = dict(pk["curve"][-1]) if pk["curve"] else {}
                last.pop("step", None)
                print(f"{Path(path).name[:-8]:24s} step {pk['step']:>6}/{pk['steps']} {pk['t'] / 3600:5.1f} h", last)
        return
    rows_path(suites[0]).parent.mkdir(exist_ok=True)
    have = {s: have_names(s) for s in suites}
    if fetch:
        for s in suites:
            n = 0
            try:
                entries = list(vol.listdir(f"{VERSION}/{s}"))
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
            print(f"fetched {n} new records from {s}/ -> {rows_path(s)} ({len(entries)} on the Volume)")
        return
    from nca.suites import SUITES
    done = {s: have[s] | volume_names(s) for s in suites}
    jobs = [dict(j, suite=s) for s in suites for j in SUITES[s]() if j["name"] not in done[s]]
    if only:
        keys = only.split(",")
        jobs = [j for j in jobs if any(k_ in j["name"] for k_ in keys)]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} jobs ({sum(bool(j.get('gpu')) for j in jobs)} on GPU)", flush=True)
    if dry or not jobs:
        return
    import time
    launch = time.strftime("%Y%m%d-%H%M%S")
    call = driver.spawn([dict(j, launch=launch) for j in jobs])
    print("driver", call.object_id, flush=True)
