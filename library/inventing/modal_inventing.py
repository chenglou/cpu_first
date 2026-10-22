"""Run the inventing-learner arms on Modal: one driver container per arm, searches fanned out to CPU containers.

    LIB_MAX=10 ~/github/sotaku/venv/bin/python -m modal run --detach library/inventing/modal_inventing.py --configs A_s1,C_mdl --tag r1
    ~/github/sotaku/venv/bin/python -m modal run library/inventing/modal_inventing.py --wait library/inventing/records/calls_r1.json

Records go to the Volume `inventing` as v1/<name>.jsonl; --wait blocks on the spawned drivers (run it as a background
task) and copies every record file to library/inventing/records/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LIBDIR = HERE.parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.4")
for _f in sorted(LIBDIR.glob("*.py")):
    image = image.add_local_file(_f, f"/root/library/{_f.name}")
for _f in sorted(HERE.glob("*.py")):
    image = image.add_local_file(_f, f"/root/library/inventing/{_f.name}")
vol = modal.Volume.from_name("inventing", create_if_missing=True)
VERSION = "v1"
app = modal.App("inventing", image=image)
MAX = int(os.environ.get("LIB_MAX", "10"))
MAX_BIG = int(os.environ.get("LIB_MAX_BIG", "2"))


@app.function(cpu=2, memory=16384, timeout=4 * 3600, max_containers=MAX, retries=modal.Retries(max_retries=1))
def search_remote(job: dict) -> dict:
    import sys
    sys.path.insert(0, "/root")
    from library.inventing.jobs import search_job
    return search_job(job)


@app.function(cpu=2, memory=32768, timeout=6 * 3600, max_containers=MAX_BIG, retries=modal.Retries(max_retries=1))
def search_big(job: dict) -> dict:
    import sys
    sys.path.insert(0, "/root")
    from library.inventing.jobs import search_job
    return search_job(job)


@app.function(cpu=1, memory=4096, timeout=20 * 3600, volumes={"/out": vol})
def driver(name: str, cfg: dict) -> str:
    import sys
    sys.path.insert(0, "/root")
    from library.inventing import loop
    os.makedirs(f"/out/{VERSION}", exist_ok=True)
    path = f"/out/{VERSION}/{name}.jsonl"
    open(path, "w").close()

    def emit(rec):
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        vol.commit()

    fn = search_big if cfg.get("big") else search_remote

    def mapper(jobs):
        return list(fn.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))

    (loop.run_curated if cfg["arm"] == "Bbare" else loop.run)(name, cfg, mapper, emit)
    return path


@app.function(cpu=1, memory=1024, timeout=600, volumes={"/out": vol})
def list_records() -> dict:
    vol.reload()
    d = f"/out/{VERSION}"
    return {f: open(os.path.join(d, f)).read() for f in sorted(os.listdir(d))} if os.path.isdir(d) else {}


@app.local_entrypoint()
def main(configs: str = "", wait: str = "", fetch: bool = False, tag: str = "") -> None:
    import sys
    sys.path.insert(0, str(LIBDIR.parent))
    if configs:
        from library.inventing.loop import CONFIGS
        calls = {}
        for c in configs.split(","):
            name = c + (f"_{tag}" if tag else "")
            calls[name] = driver.spawn(name, json.loads(json.dumps(CONFIGS[c]))).object_id
        print("spawned", calls)
        ids = HERE / "records" / f"calls_{tag or 'default'}.json"
        ids.write_text(json.dumps(calls, indent=1))
        print("call ids ->", ids)
        return
    if wait:
        for name, cid in json.loads(Path(wait).read_text()).items():
            try:
                print(name, "->", modal.FunctionCall.from_id(cid).get())
            except Exception as e:
                print(name, "failed:", repr(e)[:300])
        fetch = True
    if fetch:
        for f, txt in list_records.remote().items():
            (HERE / "records" / f).write_text(txt)
            print("fetched", f, len(txt.splitlines()), "levels")
