"""Run tinycore jobs on Modal: one CPU container per job (see suites.py for the job lists).

    ~/github/sotaku/venv/bin/python -m modal run --detach tinycore/modal_tinycore.py --suite b10
    ~/github/sotaku/venv/bin/python -m modal run tinycore/modal_tinycore.py --suite b10 --fetch

Each job writes its row to the Volume `tinycore` under <suite>/<name>.json; the entrypoint appends finished rows to
tinycore/runs/<suite>.jsonl, and `--fetch` adds the Volume's rows that are not there yet (for detached runs).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "core.py", "fixes.py", "fixsweep.py", "exact.py", "b10.py", "suites.py", "lesson.py",
           "paths.py"):
    image = image.add_local_file(HERE / _f, f"/root/tinycore/{_f}")
vol = modal.Volume.from_name("tinycore", create_if_missing=True)
app = modal.App("tinycore", image=image)


@app.function(cpu=2, memory=4096, timeout=4 * 3600, volumes={"/out": vol}, retries=modal.Retries(max_retries=1))
def run_job(job: dict) -> dict:
    import os
    import sys
    sys.path.insert(0, "/root")
    import torch
    torch.set_num_threads(2)
    from tinycore.suites import execute
    r = execute(job)
    r["suite"] = job["suite"]
    os.makedirs(f"/out/{job['suite']}", exist_ok=True)
    with open(f"/out/{job['suite']}/{job['name']}.json", "w") as f:      # a retry rewrites its own file
        f.write(json.dumps(r) + "\n")
    vol.commit()
    return r


@app.local_entrypoint()
def main(suite: str = "b10", fetch: bool = False, limit: int = 0) -> None:
    import sys
    sys.path.insert(0, str(HERE.parent))
    from tinycore.suites import SUITES
    out = HERE / "runs" / f"{suite}.jsonl"
    have = {json.loads(l)["name"] for l in open(out)} if out.exists() else set()
    if fetch:
        n = 0
        for entry in vol.listdir(suite):
            for row in b"".join(vol.read_file(entry.path)).decode().splitlines():
                if row.strip() and json.loads(row)["name"] not in have:
                    with open(out, "a") as f:
                        f.write(row + "\n")
                    have.add(json.loads(row)["name"])
                    n += 1
        print(f"fetched {n} new rows from {suite}/ -> {out}")
        return
    jobs = [dict(j, suite=suite) for j in SUITES[suite]() if j["name"] not in have]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} jobs", flush=True)
    for r in run_job.map(jobs, order_outputs=False, return_exceptions=True, wrap_returned_exceptions=False):
        if isinstance(r, Exception):
            print("job failed:", repr(r)[:300], flush=True)
            continue
        if r["name"] not in have:
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
        print(r["name"], "hit", r.get("hit"), "final", r.get("final"), flush=True)
