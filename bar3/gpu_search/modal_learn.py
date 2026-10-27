"""Any bar-3 study's learner runs on Modal, its searches on the GPU or on the CPU (GPU_SEARCH.md §8): the studies' own
`group` (bar3/scale/heldout/modal_heldout.py and its copies: one container per group, a thread per run, every finished
search appended to the group's store on the Volume, so a container restarted after preemption replays its runs), with
the flag `--engine`:
- `gpu` (the default): one GPU container per group, every search on its GPU (pool_gpu.Searcher; vocabularies
  prepared on the container's other cores);
- `cpu`: one 16-core CPU container per group, the searches on the process pool the study's own runner used
  (studies.cpu_searcher), as before.
The study (studies.py) gives the configs and, per config, the learner copy; a group runs one copy. Runs are named
<config>_<target>_<tag>, as the studies name them, so a rerun's round records can be compared with the stored ones
field by field (compare_runs.py); each engine writes under its own folder, learn/<study>/<engine>/ on the Volume
(--folder: another, e.g. for a rerun from a store of pinned searches, compare_runs.py pins).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/gpu_search/modal_learn.py --study digits --fresh lp_clean_b_src --targets dxmod --tag dg1
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/gpu_search/modal_learn.py --fetch --study digits --engine gpu
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/gpu_search/modal_learn.py --results --study digits --engine gpu --tag dg1

--store NAME starts each group from a copy of the Volume's store NAME under learn/<study>/ (the earlier runs'
searches, uploaded with `modal volume put bar3-learn STORE learn/<study>/NAME`): a search they made is read back, not
repeated. Without it every search is run. --lib-order turns on bar3/scale/jobs.py's LIB_ORDER (vocabularies and shared
searches keyed by the library's order, GPU_SEARCH.md §9) in the container and its workers; off, the keys are as the
studies had them.

Limits: every search has its job's state and time budget; at most the config's rounds; each container has a timeout
(GPU 4 h, CPU 24 h) and a hard memory limit (1.5 times the request); at most 10 GPU containers and 8 CPU containers at
once (max_containers); a job that fails returns its exception, which the learner logs and treats as a failed job.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DIRS = ("library", "library/inventing", "bar3", "bar3/breadth", "bar3/scale", "bar3/scale/subgoal",
        "bar3/scale/heldout", "bar3/scale/newton", "bar3/scale/digits", "bar3/scale/rowcheck", "bar3/scale/picks",
        "bar3/scale/reach", "bar3/scale/consolidate", "bar3/gpu_search")


def _with_files(image):
    for d in DIRS:
        for f in sorted((ROOT / d).glob("*.py")):
            image = image.add_local_file(f, f"/root/{d}/{f.name}")
    return image


cpu_image = _with_files(modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.4", "numba==0.68.0"))
gpu_image = _with_files(modal.Image.debian_slim(python_version="3.13").apt_install("gcc")
                        .pip_install("numpy==2.5.4", "numba==0.68.0", "torch==2.14.1"))
vol = modal.Volume.from_name("bar3-learn", create_if_missing=True)
app = modal.App("bar3-learn")
GPU = os.environ.get("LEARN_GPU", "L40S")
GCPU = int(os.environ.get("LEARN_GCPU", "16"))
GMEM = int(os.environ.get("LEARN_GMEM", "65536"))        # MiB requested; the hard limit is 1.5 times that
CPU = int(os.environ.get("LEARN_CPU", "16"))
MEM = int(os.environ.get("LEARN_MEM", "65536"))


def _group(engine, study, fresh_runs, workers, store, lib_order, folder):
    """fresh_runs [[name, cfg, target, config]], a thread each, on one learner copy; returns the group's times and its
    searcher's counts."""
    import resource
    import shutil
    import sys
    import threading
    import time
    import traceback
    sys.path.insert(0, "/root")
    os.environ["BAR3_LIB_ORDER"] = "1" if lib_order else ""     # read by jobs.py here and in every worker
    if "bar3.scale.jobs" in sys.modules:                        # a container kept warm from another call
        sys.modules["bar3.scale.jobs"].LIB_ORDER = lib_order
    from bar3.gpu_search import studies as ST
    copies = {ST.copy_of(study, c) for _, _, _, c in fresh_runs}
    assert len(copies) == 1, f"one learner copy per group: {copies}"
    copy = copies.pop()
    if copy == "newton":
        from bar3.scale.heldout import targets  # noqa: F401  (the held-out targets, in this process)
    out = f"/out/learn/{study}/{folder or engine}"
    os.makedirs(out, exist_ok=True)
    own = f"{out}/store_{fresh_runs[0][0]}.jsonl"           # each group its own store (a restart reads it back)
    if store and not os.path.exists(own):
        shutil.copy(f"/out/learn/{study}/{store}", own)
        vol.commit()
    if engine == "gpu":
        from bar3.gpu_search.pool_gpu import Searcher
        search = Searcher(workers, "gpu", True, store=own, commit=vol.commit, digits=copy == "digits")
    else:
        search = ST.cpu_searcher(copy)(workers, "nb", True, store=own, commit=vol.commit)
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

    threads = [threading.Thread(target=guard, args=(ST.learner(study, c).fresh, name, cfg, tname, search,
                                                    emitter(f"{out}/{name}.jsonl")))
               for name, cfg, tname, c in fresh_runs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = dict(getattr(search, "stats", {}))
    (search.close() if engine == "gpu" else search.pool.shutdown(wait=True))
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    gpu_peak = 0.0
    if engine == "gpu":
        import torch
        gpu_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"group done in {time.time() - t0:.0f}s on {engine}; largest worker {peak:.1f} GB, GPU {gpu_peak:.1f} GB; "
          f"{stats}")
    return dict(runs=[r[0] for r in fresh_runs], engine=engine, lib_order=lib_order, seconds=round(time.time() - t0),
                worker_peak_gb=round(peak, 2), gpu_peak_gb=round(gpu_peak, 2), stats=stats)


@app.function(image=gpu_image, gpu=GPU, cpu=GCPU, memory=(GMEM, GMEM * 3 // 2), timeout=4 * 3600,
              volumes={"/out": vol}, retries=modal.Retries(max_retries=1), max_containers=10)
def group_gpu(study: str, fresh_runs: list, workers: int = GCPU - 1, store: str = "", lib_order: bool = False,
              folder: str = "") -> dict:
    return _group("gpu", study, fresh_runs, workers, store, lib_order, folder)


@app.function(image=cpu_image, cpu=CPU, memory=(MEM, MEM * 3 // 2), timeout=24 * 3600, volumes={"/out": vol},
              retries=modal.Retries(max_retries=1), max_containers=8)
def group_cpu(study: str, fresh_runs: list, workers: int = CPU, store: str = "", lib_order: bool = False,
              folder: str = "") -> dict:
    return _group("cpu", study, fresh_runs, workers, store, lib_order, folder)


@app.function(image=cpu_image, cpu=1, memory=1024, timeout=600, volumes={"/out": vol})
def list_records(study: str, folder: str) -> dict:
    vol.reload()
    d = f"/out/learn/{study}/{folder}"
    return ({f: open(os.path.join(d, f)).read() for f in sorted(os.listdir(d)) if not f.startswith("store_")}
            if os.path.isdir(d) else {})


@app.local_entrypoint()
def main(study: str = "", fresh: str = "", targets: str = "", tag: str = "", engine: str = "gpu", groups: int = 1,
         workers: int = 0, store: str = "", fetch: bool = False, only: str = "", results: bool = False,
         lib_order: bool = False, folder: str = "") -> None:
    """--study S --fresh CONFIGS --targets T,... --tag TAG: each (config, target) from scratch (or from --store), in
    `groups` containers, interleaved. --fetch: the engine's records into records/learn/<study>/<engine>/. --results:
    the finished groups' returns (times, searcher counts) for a tag."""
    import sys
    sys.path.insert(0, str(ROOT))
    out = HERE / "records" / "learn" / study / (folder or engine)
    out.mkdir(parents=True, exist_ok=True)
    ids = out / f"calls_{tag or 'default'}.json"
    if fetch:
        for f, txt in list_records.remote(study, folder or engine).items():
            if only and only not in f:
                continue
            (out / f).write_text(txt)
            print("fetched", f, len(txt.splitlines()), "lines")
        return
    if results:
        calls = json.loads(ids.read_text())
        got = {}
        for call in sorted(set(calls.values())):
            try:
                got[call] = modal.FunctionCall.from_id(call).get(timeout=5)
            except Exception as e:      # noqa: BLE001  (still running, or failed)
                got[call] = repr(e)
            print(call, json.dumps(got[call])[:600])
        (out / f"groups_{tag or 'default'}.json").write_text(json.dumps(got, indent=1))
        return
    from bar3.gpu_search import studies as ST
    assert engine in ("gpu", "cpu"), engine
    CONFIGS = ST.configs(study)
    cs, ts = fresh.split(","), targets.split(",")
    assert all(c in CONFIGS for c in cs), f"only {study}'s configs"
    sfx = f"_{tag}" if tag else ""
    fr = [[f"{c}_{t}{sfx}", json.loads(json.dumps(CONFIGS[c])), t, c] for t in ts for c in cs]
    fn = group_gpu if engine == "gpu" else group_cpu
    calls = {}
    for g in range(groups):
        part = fr[g::groups]
        if part:
            call = fn.spawn(study, part, workers or (GCPU - 1 if engine == "gpu" else CPU), store, lib_order,
                            folder).object_id
            calls.update({r[0]: call for r in part})
    print("spawned", engine, calls)
    old = json.loads(ids.read_text()) if ids.exists() else {}
    ids.write_text(json.dumps({**old, **calls}, indent=1))
