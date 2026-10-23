"""Run NCA sweeps on Modal: one CPU container per job (job lists in suites.py), at most 20 at once.

    ~/github/sotaku/venv/bin/python -m modal run --detach nca/modal_nca.py --suite explore
    ~/github/sotaku/venv/bin/python -m modal run nca/modal_nca.py --suite explore --fetch

The launch spawns one driver inside the app and returns; the driver maps the jobs (so the sweep does not depend on
the laptop staying connected). Each job writes its record to the Volume `nca` under <VERSION>/<suite>/<name>.json;
`--fetch` appends the Volume's records that are not in nca/runs/<suite>.jsonl yet. A launch skips jobs whose record
is already on the Volume or in the local file.
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


@app.function(cpu=4, memory=(2048, 8192), timeout=4 * 3600, volumes={"/out": vol},
              retries=modal.Retries(max_retries=1), max_containers=20)
def run_job(job: dict) -> dict:
    import os
    import sys
    import time
    sys.path.insert(0, "/root")
    import torch
    torch.set_num_threads(4)
    path = f"/out/{VERSION}/{job['suite']}/{job['name']}.json"
    if os.path.exists(path):                                   # a restarted input never redoes a finished job
        return {"name": job["name"], "suite": job["suite"], "skipped": True}
    from nca.suites import execute
    t = time.time()
    r = execute(job)
    r["wall"] = time.time() - t
    r["suite"] = job["suite"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(r) + "\n")
    vol.commit()
    return {"name": r["name"], "suite": r["suite"], "hit": r.get("hit"), "certified": r.get("certified")}


@app.function(cpu=0.25, memory=512, timeout=12 * 3600, nonpreemptible=True)
def driver(jobs: list) -> None:
    n = 0
    for r in run_job.map(jobs, order_outputs=False, return_exceptions=True, wrap_returned_exceptions=False):
        n += 1
        if isinstance(r, Exception):
            print("job failed:", repr(r)[:300], flush=True)
        else:
            print(n, r, flush=True)


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
def main(suite: str = "explore", fetch: bool = False, limit: int = 0, only: str = "", dry: bool = False) -> None:
    """--suite takes one suite or several separated by commas. --only keeps jobs whose name contains one of the
    comma-separated strings."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    suites = suite.split(",")
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
    print(f"{len(jobs)} jobs", flush=True)
    if dry or not jobs:
        return
    call = driver.spawn(jobs)
    print("driver", call.object_id, flush=True)
