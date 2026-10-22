"""Run the handwired bridge's sweeps on Modal: one CPU container per job (job lists in suites.py), at most 20 at once.

    ~/github/sotaku/venv/bin/python -m modal run --detach handwired_bridge/modal_hb.py --suite ladder
    ~/github/sotaku/venv/bin/python -m modal run handwired_bridge/modal_hb.py --suite ladder --fetch

Run from cpu_first/. Each job writes its record to the Volume `handwired-bridge` under <VERSION>/<suite>/<name>.json;
the entrypoint appends finished records to handwired_bridge/runs/<suite>.jsonl, and `--fetch` adds the Volume's
records that are not there yet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "grid.py", "net.py", "train.py", "notches.py", "probe.py", "compile.py"):
    image = image.add_local_file(ROOT / "bridge" / _f, f"/root/bridge/{_f}")
for _f in ("__init__.py", "load.py", "hnet.py", "train.py", "notches.py", "suites.py", "probe.py", "updown.py"):
    image = image.add_local_file(HERE / _f, f"/root/handwired_bridge/{_f}")
for _f in ("B2w3k3.json", "B3w2k1.json", "B4w2k1.json", "B5w2k1.json"):
    image = image.add_local_file(HERE / "records" / "compiled" / _f, f"/root/handwired_bridge/records/compiled/{_f}")
vol = modal.Volume.from_name("handwired-bridge", create_if_missing=True)
VERSION = "v1"
app = modal.App("handwired-bridge", image=image)


@app.function(cpu=4, memory=4096, timeout=3 * 3600, volumes={"/out": vol}, retries=modal.Retries(max_retries=1),
              max_containers=int(os.environ.get("HB_MAX_CONTAINERS", "20")))
def run_job(job: dict) -> dict:
    import os
    import sys
    sys.path.insert(0, "/root")
    import torch
    torch.set_num_threads(4)
    from handwired_bridge.suites import execute
    r = execute(job)
    r["suite"] = job["suite"]
    os.makedirs(f"/out/{VERSION}/{job['suite']}", exist_ok=True)
    with open(f"/out/{VERSION}/{job['suite']}/{job['name']}.json", "w") as f:
        f.write(json.dumps(r) + "\n")
    vol.commit()
    return r


def rows_path(suite):
    return HERE / "runs" / f"{suite}.jsonl"


def have_names(suite):
    out = rows_path(suite)
    return {json.loads(l)["name"] for l in open(out)} if out.exists() else set()


@app.local_entrypoint()
def main(suite: str = "ladder", fetch: bool = False, limit: int = 0, only: str = "") -> None:
    """--suite takes one suite or several separated by commas (one map, so one 20-container cap). --only keeps jobs
    whose name contains one of the comma-separated strings."""
    import sys
    sys.path.insert(0, str(ROOT))
    from handwired_bridge.suites import SUITES
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
            print(f"fetched {n} new records from {s}/ -> {rows_path(s)}")
        return
    jobs = [dict(j, suite=s) for s in suites for j in SUITES[s]() if j["name"] not in have[s]]
    if only:
        keys = only.split(",")
        jobs = [j for j in jobs if any(k_ in j["name"] for k_ in keys)]
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
        print(s, r["name"], "hit", r.get("hit"), "certified", r.get("certified"), f"{r.get('wall', 0):.0f}s",
              flush=True)
