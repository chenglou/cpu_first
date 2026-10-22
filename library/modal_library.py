"""Run library-learning experiments on Modal: one driver container per experiment, searches fanned out to CPU
containers (at most 16 at once, shared by all drivers).

    ~/github/sotaku/venv/bin/python -m modal run --detach library/modal_library.py --configs bpe,mdl
    ~/github/sotaku/venv/bin/python -m modal run library/modal_library.py --wait <call ids file>   # waits, fetches

Every driver writes its level records to the Volume `library` as <VERSION>/<name>.jsonl; --wait blocks on the
spawned calls (run it as a background task) and then copies every record file to library/records/.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.4")
for _f in sorted(HERE.glob("*.py")):
    image = image.add_local_file(_f, f"/root/library/{_f.name}")
vol = modal.Volume.from_name("library", create_if_missing=True)
VERSION = "v1"
app = modal.App("library", image=image)
import os as _os
# container caps of this launch (the 20-container budget is shared by every app running at once)
MAX_SMALL = int(_os.environ.get("LIB_MAX_SMALL", "16"))
MAX_BIG = int(_os.environ.get("LIB_MAX_BIG", "16"))


@app.function(cpu=2, memory=16384, timeout=4 * 3600, max_containers=MAX_SMALL, retries=modal.Retries(max_retries=1))
def search_remote(job: dict) -> dict:
    import sys
    sys.path.insert(0, "/root")
    from library.jobs import search_job
    return search_job(job)


def _cached(job):
    """A job's record from the Volume's cache if the same job (name aside) already ran, else run it and cache it."""
    import hashlib
    import os
    import sys
    sys.path.insert(0, "/root")
    from library.jobs import search_job
    key = hashlib.md5(json.dumps({k: v for k, v in job.items() if k != "name"}, sort_keys=True).encode()).hexdigest()
    path = f"/out/cache/{key}.json"
    vol.reload()
    if os.path.exists(path):
        rec = json.loads(open(path).read())
        rec["name"], rec["cached"] = job.get("name", rec["name"]), True
        return rec
    rec = search_job(job)
    os.makedirs("/out/cache", exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(rec))
    vol.commit()
    return rec


@app.function(cpu=2, memory=32768, timeout=6 * 3600, max_containers=MAX_BIG, retries=modal.Retries(max_retries=1),
              volumes={"/out": vol})
def search_big(job: dict) -> dict:
    return _cached(job)


@app.function(cpu=1, memory=4096, timeout=20 * 3600, volumes={"/out": vol})
def driver_remote(name: str, cfg: dict) -> str:
    import os
    import sys
    sys.path.insert(0, "/root")
    from library import wake_sleep
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

    start = None
    if cfg.get("resume"):                       # continue from another run's last level record
        vol.reload()
        prev = [json.loads(l) for l in open(f"/out/{VERSION}/{cfg['resume']}.jsonl")]
        start = prev[-1]
        for rec in prev:                        # its levels, copied first so this record file is whole
            emit(rec)
    wake_sleep.run(name, cfg, mapper, emit, start=start)
    return path


@app.function(cpu=2, memory=8192, timeout=20 * 3600, volumes={"/out": vol})
def driver_multi(runs: list) -> list:
    """Several experiments in one container, one thread each (they share the search functions' container caps)."""
    import threading
    out = []

    def one(name, cfg):
        try:
            out.append(driver_remote.local(name, cfg))
        except Exception as e:
            out.append(f"{name} failed: {e!r}")

    ts = [threading.Thread(target=one, args=(n, c)) for n, c in runs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


@app.function(cpu=1, memory=1024, timeout=600, volumes={"/out": vol})
def list_records() -> dict:
    import os
    vol.reload()
    out = {}
    d = f"/out/{VERSION}"
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        out[f] = open(os.path.join(d, f)).read()
    return out


@app.local_entrypoint()
def main(configs: str = "", wait: str = "", fetch: bool = False, levels: int = 0, max_cost: int = 0,
         max_states: int = 0, tag: str = "", multi: bool = False) -> None:
    import sys
    sys.path.insert(0, str(HERE.parent))
    if configs:
        from library.wake_sleep import CONFIGS
        calls, runs = {}, []
        for c in configs.split(","):
            cfg = json.loads(json.dumps(CONFIGS[c]))
            if levels:
                cfg["levels"] = levels
            if max_cost:
                cfg["budget"]["max_cost"] = max_cost
            if max_states:
                cfg["budget"]["max_states"] = max_states
            runs.append((c + (f"_{tag}" if tag else ""), cfg))
        if multi:                               # one driver container for all of them, one thread each
            calls["+".join(n for n, _ in runs)] = driver_multi.spawn(runs).object_id
        else:
            for n, cfg in runs:
                calls[n] = driver_remote.spawn(n, cfg).object_id
        print("spawned", calls)
        ids = HERE / "records" / f"calls_{tag or 'default'}.json"
        ids.write_text(json.dumps(calls, indent=1))
        print("call ids ->", ids)
        return
    if wait:
        calls = json.loads(Path(wait).read_text())
        for name, cid in calls.items():
            try:
                print(name, "->", modal.FunctionCall.from_id(cid).get())
            except Exception as e:
                print(name, "failed:", repr(e)[:500])
        fetch = True
    if fetch:
        for f, txt in list_records.remote().items():
            (HERE / "records" / f).write_text(txt)
            print("fetched", f, len(txt.splitlines()), "levels")
