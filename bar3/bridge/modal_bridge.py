"""Run the digit-level learner on Modal: bar3/modal_practice.py's pattern (one non-preemptible driver per config,
searches fanned out to CPU containers, identical jobs shared through a modal.Dict with in-flight claims).

    cd cpu_first && BRIDGE_MAX=18 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_bridge.py --multi --configs lp_sbb,lp_adc --tag r1
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/bridge/modal_bridge.py --fetch

Records go to the Volume `bar3-bridge` as v1/<config>_<tag>.jsonl, one line per round; --fetch copies them to
bar3/bridge/records/. Limits: every search has its own state and time budget; search containers have a 3-hour
timeout and a 32 GiB hard memory limit; drivers 24 hours; a cancelled driver's inputs are cancelled with it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.4")
for _f in sorted((ROOT / "library").glob("*.py")):
    image = image.add_local_file(_f, f"/root/library/{_f.name}")
image = image.add_local_file(ROOT / "bar3" / "__init__.py", "/root/bar3/__init__.py")
for _f in sorted(HERE.glob("*.py")):
    image = image.add_local_file(_f, f"/root/bar3/bridge/{_f.name}")
vol = modal.Volume.from_name("bar3-bridge", create_if_missing=True)
VERSION = "v1"
app = modal.App("bar3-bridge", image=image)
MAX = int(os.environ.get("BRIDGE_MAX", "20"))
cache = modal.Dict.from_name("bar3-bridge-cache", create_if_missing=True)


@app.function(cpu=2, memory=(8192, 32768), timeout=3 * 3600, max_containers=MAX, retries=modal.Retries(max_retries=1))
def search_remote(job: dict) -> dict:
    """One search; an identical job (the same probe posed by two configs) is answered from the shared cache, and its
    states and seconds still count in the config that asked."""
    import hashlib
    import sys
    import time
    sys.path.insert(0, "/root")
    from bar3.bridge.djobs import search_job
    key = VERSION + hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()
    try:
        hit = cache.get(key)
        if hit is None and not cache.put(key + ":claim", time.time(), skip_if_exists=True):
            t0 = time.time()
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
    from bar3.bridge import dlearner
    os.makedirs(f"/out/{VERSION}", exist_ok=True)
    path = f"/out/{VERSION}/{name}.jsonl"
    open(path, "w").close()

    def emit(rec):
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        vol.commit()

    def mapper(jobs):
        return list(search_remote.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))

    dlearner.run(name, cfg, mapper, emit)
    return path


@app.function(cpu=1, memory=(4096, 8192), timeout=24 * 3600, volumes={"/out": vol}, nonpreemptible=True)
def driver_multi(runs: list) -> list:
    """Several configs in one container, a thread each ([[name, cfg]]): a driver mostly waits on its searches."""
    import sys
    import threading
    import time
    import traceback
    sys.path.insert(0, "/root")
    from bar3.bridge import dlearner
    os.makedirs(f"/out/{VERSION}", exist_ok=True)
    lock = threading.Lock()
    launch = f"{time.time():.0f}"

    def emitter(path):
        open(path, "w").close()

        def emit(rec):
            with lock:
                with open(path, "a") as f:
                    f.write(json.dumps(dict(rec, launch=launch)) + "\n")
                vol.commit()
        return emit

    def mapper(jobs):
        return list(search_remote.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))

    def guard(name, cfg):
        from bar3.bridge import dbook
        try:
            (dbook.run if cfg.get("book") else dlearner.run)(name, cfg, mapper, emitter(f"/out/{VERSION}/{name}.jsonl"))
        except Exception:
            traceback.print_exc()

    threads = [threading.Thread(target=guard, args=(name, cfg)) for name, cfg in runs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [name for name, _ in runs]


@app.function(cpu=1, memory=1024, timeout=600, volumes={"/out": vol})
def list_records() -> dict:
    vol.reload()
    d = f"/out/{VERSION}"
    return {f: open(os.path.join(d, f)).read() for f in sorted(os.listdir(d))} if os.path.isdir(d) else {}


@app.local_entrypoint()
def main(configs: str = "", fetch: bool = False, tag: str = "", only: str = "", multi: bool = False) -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    if configs:
        from bar3.bridge.dlearner import CONFIGS
        from bar3.bridge.dbook import CONFIGS as BOOK
        allc = {**CONFIGS, **{k: dict(v, book=True) for k, v in BOOK.items()}}
        calls = {}
        names = [(c + (f"_{tag}" if tag else ""), json.loads(json.dumps(allc[c]))) for c in configs.split(",")]
        if multi:
            calls["multi:" + ",".join(n for n, _ in names)] = driver_multi.spawn([[n, c] for n, c in names]).object_id
        for name, cfg in ([] if multi else names):
            calls[name] = driver.spawn(name, cfg).object_id
        print("spawned", calls)
        (HERE / "records").mkdir(exist_ok=True)
        ids = HERE / "records" / f"calls_{tag or 'default'}.json"
        old = json.loads(ids.read_text()) if ids.exists() else {}
        ids.write_text(json.dumps({**old, **calls}, indent=1))
        print("call ids ->", ids)
        return
    if fetch:
        (HERE / "records").mkdir(exist_ok=True)
        for f, txt in list_records.remote().items():
            if only and only not in f:
                continue
            (HERE / "records" / f).write_text(txt)
            print("fetched", f, len(txt.splitlines()), "rounds")
