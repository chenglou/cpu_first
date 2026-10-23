"""Run the practice-choosing learner on Modal: one non-preemptible driver per config, searches fanned out to CPU
containers (library/inventing/modal_inventing.py's pattern).

    cd cpu_first && BAR3_MAX=24 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/modal_practice.py --configs ct2,curated --tag r1
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/modal_practice.py --wait bar3/records/calls_r1.json
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/modal_practice.py --fetch

Records go to the Volume `bar3-practice` as v1/<config>_<tag>.jsonl, one line per round, committed as each round ends;
--wait blocks on the drivers and then fetches, --fetch copies every record file to bar3/records/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.4")
for _f in sorted((ROOT / "library").glob("*.py")):
    image = image.add_local_file(_f, f"/root/library/{_f.name}")
for _f in sorted((ROOT / "library" / "inventing").glob("*.py")):
    image = image.add_local_file(_f, f"/root/library/inventing/{_f.name}")
for _f in sorted(HERE.glob("*.py")):
    image = image.add_local_file(_f, f"/root/bar3/{_f.name}")
vol = modal.Volume.from_name("bar3-practice", create_if_missing=True)
VERSION = "v1"
app = modal.App("bar3-practice", image=image)
MAX = int(os.environ.get("BAR3_MAX", "24"))


cache = modal.Dict.from_name("bar3-practice-cache", create_if_missing=True)


@app.function(cpu=2, memory=(16384, 32768), timeout=3 * 3600, max_containers=MAX,   # (request, hard limit) in MiB:
              retries=modal.Retries(max_retries=1))                                    # a bare memory=N sets no limit
def search_remote(job: dict) -> dict:
    """One search; identical jobs (the same probe posed by two arms) are answered from a shared cache. A cached
    record still counts its states and seconds in the arm that asked: they are what the search costs."""
    import hashlib
    import sys
    import time
    sys.path.insert(0, "/root")
    from bar3.jobs import search_job
    key = VERSION + hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()
    try:
        hit = cache.get(key)
        if hit is None and not cache.put(key + ":claim", time.time(), skip_if_exists=True):
            t0 = time.time()                # another container is computing it: wait for its record
            while hit is None and time.time() - t0 < job.get("time_limit", 600) + 600:
                time.sleep(10)
                hit = cache.get(key)
    except Exception:
        hit = None
    if hit is not None:
        return dict(hit, cached=True)
    rec = search_job(job)
    try:
        cache[key] = rec
    except Exception:
        pass
    return rec


@app.function(cpu=1, memory=(4096, 8192), timeout=24 * 3600, volumes={"/out": vol}, nonpreemptible=True)
def driver(name: str, cfg: dict) -> str:
    import sys
    sys.path.insert(0, "/root")
    from bar3 import learner
    os.makedirs(f"/out/{VERSION}", exist_ok=True)
    path = f"/out/{VERSION}/{name}.jsonl"
    open(path, "w").close()

    def emit(rec):
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        vol.commit()

    def mapper(jobs):
        return list(search_remote.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))

    learner.run(name, cfg, mapper, emit)
    return path


@app.function(cpu=1, memory=1024, timeout=600, volumes={"/out": vol})
def list_records() -> dict:
    vol.reload()
    d = f"/out/{VERSION}"
    return {f: open(os.path.join(d, f)).read() for f in sorted(os.listdir(d))} if os.path.isdir(d) else {}


@app.local_entrypoint()
def main(configs: str = "", wait: str = "", fetch: bool = False, tag: str = "", only: str = "") -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    if configs:
        from bar3.learner import CONFIGS
        calls = {}
        for c in configs.split(","):
            name = c + (f"_{tag}" if tag else "")
            calls[name] = driver.spawn(name, json.loads(json.dumps(CONFIGS[c]))).object_id
        print("spawned", calls)
        ids = HERE / "records" / f"calls_{tag or 'default'}.json"
        old = json.loads(ids.read_text()) if ids.exists() else {}
        ids.write_text(json.dumps({**old, **calls}, indent=1))
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
            if only and only not in f:
                continue
            (HERE / "records" / f).write_text(txt)
            print("fetched", f, len(txt.splitlines()), "rounds")
