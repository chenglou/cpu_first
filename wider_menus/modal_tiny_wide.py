"""Run the tiny-core suites (suites.py) on Modal: one CPU container per job, at most 12 at once (the workspace's
100-container cap is shared with other sessions).

    cd ~/github/one-layer/cpu_first
    ~/github/sotaku/venv/bin/python -m modal run --detach wider_menus/modal_tiny_wide.py --suite grow
    ~/github/sotaku/venv/bin/python -m modal run wider_menus/modal_tiny_wide.py --suite grow --fetch

Each job writes its row (the RESULT plus the log's tail) to the Volume `wider-menus` under tiny/<suite>/<name>.json;
the entrypoint appends finished rows to wider_menus/runs/<suite>.jsonl, and --fetch adds the Volume's rows that are
not there yet (for detached runs).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = 4
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "wide.py", "tiny_wide.py", "suites.py"):
    image = image.add_local_file(HERE / _f, f"/root/wider_menus/{_f}")
vol = modal.Volume.from_name("wider-menus", create_if_missing=True)
app = modal.App("wider-menus-tiny", image=image)


@app.function(cpu=CPU, memory=8192, timeout=3 * 3600, volumes={"/out": vol}, max_containers=12)
def run_job(job: dict) -> dict:
    import contextlib
    import io
    import os
    import sys
    sys.path.insert(0, "/root")
    import torch
    torch.set_num_threads(CPU)
    from wider_menus.tiny_wide import CFG, run
    cfg = dict(CFG)
    for k, v in job["cfg"].items():
        cfg[k] = tuple(v) if isinstance(CFG[k], tuple) else v
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = run(cfg)
    r["suite"], r["name"], r["log"] = job["suite"], job["name"], buf.getvalue()[-20000:]
    os.makedirs(f"/out/tiny/{job['suite']}", exist_ok=True)
    with open(f"/out/tiny/{job['suite']}/{job['name']}.json", "w") as f:
        f.write(json.dumps(r, default=str) + "\n")
    vol.commit()
    return r


def rows_path(suite):
    return HERE / "runs" / f"{suite}.jsonl"


def have_names(suite):
    out = rows_path(suite)
    return {json.loads(l)["name"] for l in open(out)} if out.exists() else set()


def short(r):
    keys = ("certified", "true_wiring", "pool_exact", "cert_exact", "stopped", "settled_at", "seconds")
    return " ".join(f"{k}={r.get(k)}" for k in keys) + f" {r.get('classes')}"


@app.local_entrypoint()
def main(suite: str = "grow", fetch: bool = False, limit: int = 0, only: str = "") -> None:
    import sys
    sys.path.insert(0, str(HERE.parent))
    from wider_menus.suites import SUITES
    (HERE / "runs").mkdir(exist_ok=True)
    suites = suite.split(",")
    have = {s: have_names(s) for s in suites}
    if fetch:
        for s in suites:
            n = 0
            try:
                entries = list(vol.listdir(f"tiny/{s}"))
            except Exception as e:
                print(s, "nothing on the Volume yet:", repr(e)[:200])
                continue
            for entry in entries:
                row = b"".join(vol.read_file(entry.path)).decode().strip()
                r = json.loads(row)
                if r["name"] not in have[s]:
                    with open(rows_path(s), "a") as f:
                        f.write(row + "\n")
                    have[s].add(r["name"])
                    n += 1
                    print(s, r["name"], short(r))
            print(f"fetched {n} new rows from tiny/{s}/ -> {rows_path(s)}")
        return
    jobs = [dict(j, suite=s) for s in suites for j in SUITES[s]() if j["name"] not in have[s]]
    if only:
        jobs = [j for j in jobs if any(o in j["name"] for o in only.split(","))]
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
                f.write(json.dumps(r, default=str) + "\n")
            have[s].add(r["name"])
        print(s, r["name"], short(r), flush=True)
