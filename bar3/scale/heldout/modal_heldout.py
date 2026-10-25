"""The held-out runs on Modal (HELDOUT.md): bar3/scale/subgoal/modal_subgoal.py's `group`, unchanged in what it runs
(the frozen learner, bar3/scale/subgoal/learn.py, under its own configs), on the held-out targets (targets.py), with
its own app and Volume. One container per group, a thread per run, the searches on a process pool in the container
whose workers know the held-out targets (searcher.py); jobs of a batch that share a search space run as one search,
each with the record it would get alone; every finished search goes to a store on the Volume, so a container
restarted after preemption replays its runs instead of searching again.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/heldout/modal_heldout.py --fresh lp_clean_deep_sub,ct2a2_clean_deep_sub --targets x4,tri --tag ho1
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/heldout/modal_heldout.py --fetch

Limits: every search has its job's state and time budget; at most 5 rounds a run (the config's); the container has a
timeout and a hard memory limit; a job that fails returns its exception, which the learner logs and treats as a
failed job. Only the frozen configs and the held-out targets are accepted; with --control (HELDOUT.md §5, after the
test), also the same configs without the subgoal, checked to be exactly the frozen ones less their "subgoal":

    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/heldout/modal_heldout.py --control --fresh lp_clean_deep --targets x2x,three,x4,lucas --tag ctl1 --store store_ctl_lp_ho1.jsonl
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
FROZEN = ("lp_clean_deep_sub", "ct2a2_clean_deep_sub")
CONTROLS = ("lp_clean_deep", "ct2a2_clean_deep")          # the frozen configs without the subgoal (--control only)
image = modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.4", "numba==0.68.0")
for _d, _dst in ((ROOT / "library", "library"), (ROOT / "library" / "inventing", "library/inventing"),
                 (ROOT / "bar3", "bar3"), (ROOT / "bar3" / "breadth", "bar3/breadth"),
                 (ROOT / "bar3" / "scale", "bar3/scale"), (ROOT / "bar3" / "scale" / "subgoal", "bar3/scale/subgoal"),
                 (HERE, "bar3/scale/heldout")):
    for _f in sorted(_d.glob("*.py")):
        image = image.add_local_file(_f, f"/root/{_dst}/{_f.name}")
vol = modal.Volume.from_name("bar3-heldout", create_if_missing=True)
VERSION = "v1"
app = modal.App("bar3-heldout", image=image)
CPU = int(os.environ.get("HO_CPU", "16"))
MEM = int(os.environ.get("HO_MEM", "65536"))             # MiB requested; the hard limit is 1.5 times that


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
    from bar3.scale.heldout import targets  # noqa: F401  (the held-out targets, in this process)
    from bar3.scale.heldout.searcher import Searcher
    from bar3.scale.subgoal import learn
    os.makedirs(f"/out/{VERSION}", exist_ok=True)
    search = Searcher(workers, "nb", True, store=f"/out/{VERSION}/{store or f'store_{fresh_runs[0][0]}.jsonl'}",
                      commit=vol.commit)
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
def main(fresh: str = "", targets: str = "", tag: str = "", groups: int = 1, workers: int = CPU,
         fetch: bool = False, only: str = "", store: str = "", control: bool = False) -> None:
    """--fresh CONFIGS --targets T,...: each (config, target) from scratch, in `groups` containers, interleaved.
    --store: an existing store on the Volume to start from. --fetch: the records (not the stores) into records/.
    --control: the frozen configs without the subgoal are accepted too."""
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
    from bar3.scale.heldout.targets import NAMES
    from bar3.scale.subgoal.configs import CONFIGS
    cs, ts = fresh.split(","), targets.split(",")
    assert set(cs) <= set(FROZEN + (CONTROLS if control else ())), f"only the frozen configs: {FROZEN}"
    for c in CONTROLS if control else ():
        less = {k: v for k, v in CONFIGS[f"{c}_sub"].items() if k != "subgoal"}
        assert json.dumps(CONFIGS[c], sort_keys=True) == json.dumps(less, sort_keys=True), c
    assert set(ts) <= set(NAMES), f"only the held-out targets: {NAMES}"
    sfx = f"_{tag}" if tag else ""
    fr = [[f"{c}_{t}{sfx}", json.loads(json.dumps(CONFIGS[c])), t] for t in ts for c in cs]
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
