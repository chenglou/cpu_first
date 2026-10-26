"""This study's runs on Modal (PICKS.md §4), for the arms that leave the stored searches: bar3/scale/heldout/
modal_heldout.py's `group` with this study's learner copies and configs, on its own app and Volume. One container per
group, a thread per run, the searches on a process pool in the container. The workers know the held-out targets
(bar3/scale/heldout/searcher.py) for the two-source study's copy, or run the digits jobs (bar3/scale/digits/
searcher.py) for the digits copy. A group reads and extends one store on the Volume, uploaded before the launch: the
earlier runs' stores joined. A search those runs made is read back, not repeated, and a container restarted after
preemption replays its runs.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal volume put bar3-picks STORE.jsonl v1/STORE.jsonl
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/picks/modal_picks.py --fresh lp_clean_deep_sub_ns_rc_tie --targets tenrho --tag pk1 --store STORE.jsonl
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/picks/modal_picks.py --fetch

Limits: every search has its job's state and time budget; at most 5 rounds a run (the config's), and a run stops when
its target comes; the container has a timeout and a hard memory limit; a job that fails returns its exception, which
the learner logs and treats as a failed job (never stored). Only this study's flagged configs are accepted, and one
group runs one learner copy.
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
                 (ROOT / "bar3" / "scale", "bar3/scale"), (ROOT / "bar3" / "scale" / "subgoal", "bar3/scale/subgoal"),
                 (ROOT / "bar3" / "scale" / "heldout", "bar3/scale/heldout"),
                 (ROOT / "bar3" / "scale" / "newton", "bar3/scale/newton"),
                 (ROOT / "bar3" / "scale" / "digits", "bar3/scale/digits"),
                 (ROOT / "bar3" / "scale" / "rowcheck", "bar3/scale/rowcheck"), (HERE, "bar3/scale/picks")):
    for _f in sorted(_d.glob("*.py")):
        image = image.add_local_file(_f, f"/root/{_dst}/{_f.name}")
vol = modal.Volume.from_name("bar3-picks", create_if_missing=True)
VERSION = "v1"
app = modal.App("bar3-picks", image=image)
CPU = int(os.environ.get("PK_CPU", "16"))
MEM = int(os.environ.get("PK_MEM", "65536"))             # MiB requested; the hard limit is 1.5 times that


@app.function(cpu=CPU, memory=(MEM, MEM * 3 // 2), timeout=24 * 3600, volumes={"/out": vol},
              retries=modal.Retries(max_retries=1))      # preemptible: a preempted group replays from its store
def group(fresh_runs: list, copy: str, workers: int = CPU, store: str = "") -> dict:
    """fresh_runs [[name, cfg, target]], a thread each, all on one learner copy ("newton" or "digits"), their searches
    on a pool of `workers` processes; `store`: the Volume's store to read and extend."""
    import sys
    import threading
    import time
    import traceback
    sys.path.insert(0, "/root")
    if copy == "digits":
        from bar3.scale.digits.searcher import Searcher
        from bar3.scale.picks import learn_digits as learn
    else:
        from bar3.scale.heldout import targets  # noqa: F401  (the held-out targets, in this process)
        from bar3.scale.heldout.searcher import Searcher
        from bar3.scale.picks import learn_newton as learn
    os.makedirs(f"/out/{VERSION}", exist_ok=True)
    search = Searcher(workers, "nb", True, store=f"/out/{VERSION}/{store}", commit=vol.commit)
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
def main(fresh: str = "", targets: str = "", tag: str = "", workers: int = CPU, fetch: bool = False,
         only: str = "", store: str = "") -> None:
    """--fresh CONFIGS --targets T,... --store S: each (config, target) in one container, on the learner copy the
    configs share, reading and extending the Volume's store S. --fetch: the records (not the stores) into records/."""
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
    from bar3.scale.picks.configs import CONFIGS, LEARNER
    cs, ts = fresh.split(","), targets.split(",")
    assert all(c in CONFIGS and c.endswith(("_rc_tie", "_twin", "_bigx")) for c in cs), "only this study's flags"
    copies = {LEARNER[c] for c in cs}
    assert len(copies) == 1, "one learner copy per group"
    assert store, "--store: the joined earlier stores, uploaded to the Volume"
    sfx = f"_{tag}" if tag else ""
    fr = [[f"{c}_{t}{sfx}", json.loads(json.dumps(CONFIGS[c])), t] for t in ts for c in cs]
    call = group.spawn(fr, copies.pop(), workers, store).object_id
    print("spawned", call, [r[0] for r in fr])
    out.mkdir(exist_ok=True)
    ids = out / f"calls_{tag or 'default'}.json"
    old = json.loads(ids.read_text()) if ids.exists() else {}
    ids.write_text(json.dumps({**old, **{r[0]: call for r in fr}}, indent=1))
