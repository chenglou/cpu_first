"""Run tiny-core jobs (tiny_div.py, tiny_sq.py) on Modal: one CPU container per job, at most 20 at once.

    ~/github/sotaku/venv/bin/python -m modal run --detach learned_wiring/tiny/modal_tiny.py --suite s1
    ~/github/sotaku/venv/bin/python -m modal run learned_wiring/tiny/modal_tiny.py --suite s1 --fetch

Each job writes its row (the RESULT line plus the log) to the Volume `learned-wiring` under tiny/<suite>/<name>.json;
the entrypoint appends finished rows to learned_wiring/tiny/runs/<suite>.jsonl, and --fetch adds the Volume's rows
that are not there yet (for detached runs). Suites are in tiny_suites.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
H1_DIR = Path.home() / "github/one-layer/one-layer-deeper/data/generated" / \
    "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .env({"H1_DATA": "/root/h1"}))
if H1_DIR.exists():                                         # the reader lab (tiny_io.py) reads H1's prompts
    image = image.add_local_dir(H1_DIR, "/root/h1")
for _f in ("__init__.py", "tiny_div.py", "tiny_suites.py", "tiny_sq.py", "tiny_io.py"):
    if (HERE / _f).exists():
        image = image.add_local_file(HERE / _f, f"/root/learned_wiring/tiny/{_f}")
image = image.add_local_file(HERE.parent / "__init__.py", "/root/learned_wiring/__init__.py")
vol = modal.Volume.from_name("learned-wiring", create_if_missing=True)
app = modal.App("learned-wiring-tiny", image=image)


@app.function(cpu=2, memory=4096, timeout=3 * 3600, volumes={"/out": vol}, max_containers=20)
def run_job(job: dict) -> dict:
    import contextlib
    import io
    import os
    import sys
    sys.path.insert(0, "/root")
    import torch
    torch.set_num_threads(2)
    from learned_wiring.tiny.tiny_suites import execute
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = execute(job)
    r["suite"], r["name"], r["log"] = job["suite"], job["name"], buf.getvalue()[-60000:]
    os.makedirs(f"/out/tiny/{job['suite']}", exist_ok=True)
    with open(f"/out/tiny/{job['suite']}/{job['name']}.json", "w") as f:
        f.write(json.dumps(r) + "\n")
    vol.commit()
    return r


def rows_path(suite):
    return HERE / "runs" / f"{suite}.jsonl"


def have_names(suite):
    out = rows_path(suite)
    return {json.loads(l)["name"] for l in open(out)} if out.exists() else set()


def short(r):
    keys = ("table", "train", "full", "certified", "stop", "secs", "true_wiring")
    return " ".join(f"{k}={r.get(k)}" for k in keys)


@app.local_entrypoint()
def main(suite: str = "s1", fetch: bool = False, limit: int = 0) -> None:
    import sys
    sys.path.insert(0, str(HERE.parent.parent))
    from learned_wiring.tiny.tiny_suites import SUITES
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
        print(s, r["name"], short(r), flush=True)
