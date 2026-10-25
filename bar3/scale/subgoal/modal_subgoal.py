"""The subgoal runs on Modal (SUBGOAL.md): bar3/scale/modal_scale.py's `group` for runs from scratch, on this
folder's learner (learn.py here), with its own app and Volume (bar3-scale's are untouched). One container per group,
a thread per run, the searches on a process pool in the container (the jobs of a batch that share a search space
run as one search, as `--share` did: each gets the record it would alone); every finished search goes to a store on
the Volume, so a container restarted after preemption replays its runs instead of searching again.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/subgoal/modal_subgoal.py --fresh lp_clean_deep_sub --targets ten,tenrho,cube --tag s1
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/subgoal/modal_subgoal.py --fetch

Limits: every search has its job's state and time budget; the container has a timeout and a hard memory limit; a job
that fails returns its exception, which the learner logs and treats as a failed job.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.4", "numba==0.68.0")
for _d, _dst in ((ROOT / "library", "library"), (ROOT / "library" / "inventing", "library/inventing"),
                 (ROOT / "bar3", "bar3"), (ROOT / "bar3" / "breadth", "bar3/breadth"),
                 (ROOT / "bar3" / "scale", "bar3/scale"), (HERE, "bar3/scale/subgoal")):
    for _f in sorted(_d.glob("*.py")):
        image = image.add_local_file(_f, f"/root/{_dst}/{_f.name}")
vol = modal.Volume.from_name("bar3-subgoal", create_if_missing=True)
VERSION = "v1"
app = modal.App("bar3-subgoal", image=image)
CPU = int(os.environ.get("SUB_CPU", "16"))
MEM = int(os.environ.get("SUB_MEM", "65536"))            # MiB requested; the hard limit is 1.5 times that


@app.function(cpu=CPU, memory=(MEM, MEM * 3 // 2), timeout=24 * 3600, volumes={"/out": vol},
              retries=modal.Retries(max_retries=1))      # preemptible: a preempted group replays from its store
def group(fresh_runs: list, workers: int = CPU, store: str = "") -> dict:
    """fresh_runs [[name, cfg, target]], a thread each, their searches on a pool of `workers` processes; `store`: the
    Volume's store to read and extend (default: a new one named after the first run)."""
    import sys
    import threading
    import time
    import traceback
    sys.path.insert(0, "/root")
    from bar3.scale.pool import Searcher
    from bar3.scale.subgoal import learn
    os.makedirs(f"/out/{VERSION}", exist_ok=True)
    search = Searcher(workers, "nb", True, store=f"/out/{VERSION}/{store or f'store_{fresh_runs[0][0]}.jsonl'}",
                      commit=vol.commit)                 # shared searches (pool.py): the same records, faster
    lock = threading.Lock()
    launch = f"{time.time():.0f}"
    t0 = time.time()

    def emitter(path):
        open(path, "w").close()

        def emit(rec):
            with lock:
                with open(path, "a") as f:
                    f.write(json.dumps(dict(rec, launch=launch, elapsed=round(time.time() - t0, 1))) + "\n")
                vol.commit()
        return emit

    def guard(fn, *a, **kw):
        try:
            fn(*a, **kw)
        except Exception:
            traceback.print_exc()

    threads = [threading.Thread(target=guard, args=(learn.fresh, name, cfg, tname, search,
                                                    emitter(f"/out/{VERSION}/{name}.jsonl")))
               for name, cfg, tname in fresh_runs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    import resource
    search.pool.shutdown(wait=True)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    print(f"group done in {time.time() - t0:.0f}s on {os.cpu_count()} cores; largest worker {peak:.1f} GB")
    return dict(runs=[r[0] for r in fresh_runs], seconds=round(time.time() - t0), worker_peak_gb=round(peak, 2))


@app.function(cpu=1, memory=1024, timeout=600, volumes={"/out": vol})
def list_records() -> dict:
    vol.reload()
    d = f"/out/{VERSION}"
    return ({f: open(os.path.join(d, f)).read() for f in sorted(os.listdir(d)) if not f.startswith("store_")}
            if os.path.isdir(d) else {})


@app.local_entrypoint()
def main(fresh: str = "", targets: str = "", tag: str = "", overrides: str = "", groups: int = 1,
         workers: int = CPU, fetch: bool = False, only: str = "", store: str = "") -> None:
    """--fresh CONFIGS --targets T,...: each (config, target) from scratch, in `groups` containers. --overrides: JSON
    merged into every config. --store: an existing store on the Volume to start from (searches are keyed by their
    job, so runs that ask for the same searches replay them). --fetch: the records (not the stores) into records/."""
    import sys
    sys.path.insert(0, str(ROOT))
    out = HERE / "records"
    if fetch:
        out.mkdir(exist_ok=True)
        for f, txt in list_records.remote().items():
            if only and only not in f:
                continue
            (out / f).write_text(txt)
            print("fetched", f, len(txt.splitlines()), "lines")
        return
    from bar3.scale.subgoal.configs import CONFIGS
    extra = json.loads(overrides) if overrides else {}
    sfx = f"_{tag}" if tag else ""
    fr = [[f"{c}_{t}{sfx}", json.loads(json.dumps({**CONFIGS[c], **extra})), t]
          for c in fresh.split(",") for t in targets.split(",")]
    calls = {}
    for g in range(groups):
        part = fr[g::groups]
        if part:
            call = group.spawn(part, workers, store).object_id
            calls.update({r[0]: call for r in part})
    print("spawned", calls)
    out.mkdir(exist_ok=True)
    ids = out / f"calls_{tag or 'default'}.json"
    old = json.loads(ids.read_text()) if ids.exists() else {}
    ids.write_text(json.dumps({**old, **calls}, indent=1))
