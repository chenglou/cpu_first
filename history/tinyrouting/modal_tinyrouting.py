"""Run tinyrouting jobs on Modal: one CPU container per job (job lists in suites.py), at most 20 at once.

    ~/github/sotaku/venv/bin/python -m modal run --detach tinyrouting/modal_tinyrouting.py --suite phase
    ~/github/sotaku/venv/bin/python -m modal run tinyrouting/modal_tinyrouting.py --suite phase --fetch

Each job writes its row to the Volume `tinyrouting` under <suite>/<name>.json; the entrypoint appends finished rows
to tinyrouting/runs/<suite>.jsonl, and `--fetch` adds the Volume's rows that are not there yet (for detached runs).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "machine.py", "train.py", "exact.py", "suites.py", "school.py"):
    image = image.add_local_file(HERE / _f, f"/root/tinyrouting/{_f}")
vol = modal.Volume.from_name("tinyrouting", create_if_missing=True)
VERSION = "v1"                     # Volume folder of this code version (v0: stopped at the first exact step)
app = modal.App("tinyrouting", image=image)


@app.function(cpu=4, memory=8192, timeout=6 * 3600, volumes={"/out": vol}, retries=modal.Retries(max_retries=1),
              max_containers=20)
def run_job(job: dict) -> dict:
    import os
    import sys
    sys.path.insert(0, "/root")
    import torch
    torch.set_num_threads(4)
    from tinyrouting.suites import execute
    r = execute(job)
    r["suite"] = job["suite"]
    os.makedirs(f"/out/{VERSION}/{job['suite']}", exist_ok=True)
    with open(f"/out/{VERSION}/{job['suite']}/{job['name']}.json", "w") as f:   # a retry rewrites its own file
        f.write(json.dumps(r) + "\n")
    vol.commit()
    return r


def rows_path(suite):
    return HERE / "runs" / f"{suite}.jsonl"


def have_names(suite):
    out = rows_path(suite)
    return {json.loads(l)["name"] for l in open(out)} if out.exists() else set()


@app.local_entrypoint()
def main(suite: str = "phase", fetch: bool = False, limit: int = 0) -> None:
    """--suite takes one suite or several separated by commas (one map, so one 20-container cap)."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    from tinyrouting.suites import SUITES
    suites = suite.split(",")
    have = {s: have_names(s) for s in suites}
    if fetch:
        for s in suites:
            n = 0
            try:
                entries = list(vol.listdir(f"{VERSION}/{s}"))
            except Exception as e:                                              # nothing written yet
                print(s, "nothing on the Volume yet:", repr(e)[:200])
                continue
            for entry in entries:
                for row in b"".join(vol.read_file(entry.path)).decode().splitlines():
                    if row.strip() and json.loads(row)["name"] not in have[s]:
                        with open(rows_path(s), "a") as f:
                            f.write(row + "\n")
                        have[s].add(json.loads(row)["name"])
                        n += 1
            print(f"fetched {n} new rows from {s}/ -> {rows_path(s)}")
        return
    jobs = [dict(j, suite=s) for s in suites for j in SUITES[s]() if j["name"] not in have[s]]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} jobs", flush=True)
    for r in run_job.map(jobs, order_outputs=False, return_exceptions=True, wrap_returned_exceptions=False):
        if isinstance(r, Exception):
            print("job failed:", repr(r)[:300], flush=True)
            continue
        s = r["suite"]
        if r["name"] not in have[s]:
            with open(rows_path(s), "a") as f:
                f.write(json.dumps(r) + "\n")
            have[s].add(r["name"])
        print(s, r["name"], "hit", r.get("hit"), "certified", r.get("certified"), r.get("secs"), flush=True)
