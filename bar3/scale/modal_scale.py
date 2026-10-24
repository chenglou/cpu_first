"""The scale runs on Modal (SCALE.md): learner runs in one container each group, a thread per run, and their search
jobs on a process pool in the same container, so a job costs no network round trip (the compiled engine makes most
jobs take milliseconds, where bar3/breadth/modal_breadth.py's per-job fan-out and Dict claims would dominate).
Identical jobs within a container are run once (an in-memory cache with in-flight sharing); a cached record still
counts its seconds in every run that uses it, as in the breadth study. Its own app and Volume; bar3's are untouched.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/modal_scale.py --fresh ct2a2_clean --targets h1,rho --tag s1
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/modal_scale.py --seq lp_clean --order h1,rho,minus1,plus3,affine --tag s1
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/modal_scale.py --fetch

--engine py runs bar3/jobs.py's original search instead of the compiled one (the same-hardware comparison);
--share runs the jobs of one batch that share a search space as one search (pool.py).
Limits: every search has its job's state and time budget; the container has a timeout and a hard memory limit; a
job that fails returns its exception, which the learner logs and treats as a failed job.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.4", "numba==0.68.0")
for _d, _dst in ((ROOT / "library", "library"), (ROOT / "library" / "inventing", "library/inventing"),
                 (ROOT / "bar3", "bar3"), (ROOT / "bar3" / "breadth", "bar3/breadth"), (HERE, "bar3/scale")):
    for _f in sorted(_d.glob("*.py")):
        image = image.add_local_file(_f, f"/root/{_dst}/{_f.name}")
vol = modal.Volume.from_name("bar3-scale", create_if_missing=True)
VERSION = "v1"
app = modal.App("bar3-scale", image=image)
CPU = int(os.environ.get("SCALE_CPU", "16"))
MEM = int(os.environ.get("SCALE_MEM", "65536"))          # MiB requested; the hard limit is 1.5 times that


@app.function(cpu=CPU, memory=(MEM, MEM * 3 // 2), timeout=24 * 3600, volumes={"/out": vol},
              retries=modal.Retries(max_retries=1))      # preemptible: a preempted group restarts from scratch
def group(fresh_runs: list, seq_runs: list, engine: str = "nb", workers: int = CPU, share: bool = False) -> dict:
    """Runs in one container: fresh_runs [[name, cfg, target]], seq_runs [[name, cfg, order, carry, state or None]],
    a thread each, their searches on a pool of `workers` processes (the cores requested: os.cpu_count() inside the
    container is the host's, 24 for an 8-core request)."""
    import sys
    import threading
    import time
    import traceback
    sys.path.insert(0, "/root")
    from bar3.breadth import carry as C
    from bar3.scale import drive, learn                  # learn.py: the original learner plus the levers
    from bar3.scale.pool import Searcher
    os.makedirs(f"/out/{VERSION}", exist_ok=True)
    store = f"/out/{VERSION}/store_{fresh_runs[0][0] if fresh_runs else seq_runs[0][0]}.jsonl"
    search = Searcher(workers, engine, share, store=store, commit=vol.commit)   # survives a preemption
    lock = threading.Lock()
    launch = f"{time.time():.0f}"
    t0 = time.time()

    def emitter(path):
        open(path, "w").close()

        def emit(rec):
            with lock:
                with open(path, "a") as f:
                    f.write(json.dumps(dict(rec, launch=launch, engine=engine,
                                            elapsed=round(time.time() - t0, 1))) + "\n")
                vol.commit()
        return emit

    def guard(fn, *a, **kw):
        try:
            fn(*a, **kw)
        except Exception:
            traceback.print_exc()

    threads = []
    for name, cfg, tname in fresh_runs:
        threads.append(threading.Thread(target=guard, args=(drive.fresh, name, cfg, tname, search,
                                                             emitter(f"/out/{VERSION}/{name}.jsonl"))))
    for name, cfg, order, carry, start in seq_runs:
        spath = f"/out/{VERSION}/{name}.state.json"

        def save(tname, st, spath=spath):
            with lock:
                for p in (spath, spath.replace(".state.json", f".state.{tname}.json")):
                    json.dump(dict(after=tname, state=C.to_json(st)), open(p, "w"))
                vol.commit()

        threads.append(threading.Thread(target=guard, args=(drive.carried, name, cfg, order, search,
                                                             emitter(f"/out/{VERSION}/{name}.jsonl")),
                                        kwargs=dict(carry=carry, save=save,
                                                    state=learn.from_json(start) if start else None)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    import resource
    search.pool.shutdown(wait=True)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    print(f"group done in {time.time() - t0:.0f}s on {os.cpu_count()} cores; largest worker {peak:.1f} GB")
    return dict(runs=[r[0] for r in fresh_runs + seq_runs], seconds=round(time.time() - t0), cores=os.cpu_count(),
                worker_peak_gb=round(peak, 2))


@app.function(cpu=1, memory=1024, timeout=600, volumes={"/out": vol})
def list_records() -> dict:
    vol.reload()
    d = f"/out/{VERSION}"
    return ({f: open(os.path.join(d, f)).read() for f in sorted(os.listdir(d)) if not f.startswith("store_")}
            if os.path.isdir(d) else {})


@app.local_entrypoint()
def main(fresh: str = "", targets: str = "", seq: str = "", order: str = "", carry: str = "targets",
         branch: str = "", branch_config: str = "lp_clean", tag: str = "", suffix: str = "", overrides: str = "",
         engine: str = "nb", groups: int = 1, workers: int = CPU, fetch: bool = False, only: str = "",
         share: bool = False) -> None:
    """--fresh CONFIGS --targets T,...: each (config, target) from scratch. --seq CONFIGS --order T,...: each config
    on the targets in turn, the library carried. --branch STATE.json --targets T,...: each target alone from a saved
    sequence state, with --branch-config. All of them in `groups` containers. --overrides: JSON merged into every
    config; --suffix: added to the run names."""
    import sys
    sys.path.insert(0, str(ROOT))
    if fetch:
        for f, txt in list_records.remote().items():
            if (only and only not in f) or f.startswith("store_"):     # stores are the Volume's search caches
                continue
            (HERE / "records" / f).write_text(txt)
            print("fetched", f, len(txt.splitlines()), "lines")
        return
    from bar3.scale.configs import CONFIGS
    extra = json.loads(overrides) if overrides else {}
    sfx = (f"_{suffix}" if suffix else "") + (f"_{tag}" if tag else "")

    def cfg_of(c):
        return json.loads(json.dumps({**CONFIGS[c], **extra}))

    fr = [[f"{c}_{t}{sfx}", cfg_of(c), t] for c in (fresh.split(",") if fresh else ()) for t in targets.split(",")]
    sq = [[f"seq_{c}_{carry}{sfx}", cfg_of(c), order.split(","), carry, None] for c in (seq.split(",") if seq else ())]
    if branch:
        saved = json.loads(Path(branch).read_text())
        base = Path(branch).name.split(".state")[0].removeprefix("seq_")
        sq += [[f"branch_{base}_after_{saved['after']}_{t}{sfx}", cfg_of(branch_config), [t], carry, saved["state"]]
               for t in targets.split(",")]
    calls = {}
    for g in range(groups):
        part_f, part_s = fr[g::groups], sq[g::groups]
        if part_f or part_s:
            call = group.spawn(part_f, part_s, engine, workers, share).object_id
            calls.update({r[0]: call for r in part_f + part_s})
    print("spawned", calls)
    ids = HERE / "records" / f"calls_{tag or 'default'}.json"
    old = json.loads(ids.read_text()) if ids.exists() else {}
    ids.write_text(json.dumps({**old, **calls}, indent=1))


@app.function(cpu=2, memory=(8192, 12288), timeout=1800)
def profile(names: list, budget: dict, engines: list) -> str:
    """cProfile of round-0 h1 candidates' searches (by name) on this hardware, with each engine: where the time goes."""
    import cProfile
    import io
    import pstats
    import sys
    import time
    sys.path.insert(0, "/root")
    import numpy as np
    from library.inventing import brm
    from bar3 import learner, problems as P
    from bar3.scale.jobs import search_job
    from bar3.scale.pool import warm
    warm()
    pool = {c.name: c for c in P.laws(7) + P.restrictions(P.target_gen, 7)}
    out = io.StringIO()
    from bar3.scale import engine
    for n in names:
        j = learner.job(pool[n], {}, list(brm.LOOP_KINDS), dict(learner.PROBE_B, **budget), "probe_b", 7)
        for e in engines:
            if e.startswith("nb"):          # "nb0", "nb1", "nb2": the kernel variant
                engine.MODE = int(e[2:] or 1)
                search_job(dict(j, max_states=300, name="warm"))
            pr = cProfile.Profile()
            t = time.time()
            pr.enable()
            r = search_job(dict(j, engine="nb" if e.startswith("nb") else e))
            pr.disable()
            print(f"== {n} {e}: {time.time() - t:.1f}s levels {r['level_sizes']}", file=out)
            pstats.Stats(pr, stream=out).sort_stats("tottime").print_stats(8)
    return out.getvalue()


@app.function(cpu=2, memory=(65536, 98304), timeout=6 * 3600, volumes={"/out": vol})
def reach(label: str, tname: str, state_file: str, budgets: list, guided: bool = False) -> list:
    """How far the target search gets for its state budget (SCALE.md §1): the target alone (no practice) searched
    with each budget, on the base vocabulary or a carried library (a sequence's saved state, read from the Volume or
    passed in), plainly or guided. One record per budget: level sizes, the cost reached, seconds."""
    import sys
    import time
    sys.path.insert(0, "/root")
    from library.inventing import brm
    from bar3 import learner as L, problems as P
    from bar3.breadth import targets as TG
    from bar3.scale import learn
    from bar3.scale.jobs import search_job
    from bar3.scale.pool import warm
    warm()
    lib, kinds = {}, list(brm.LOOP_KINDS)
    if state_file:
        st = json.loads(state_file)["state"]
        lib, kinds = st["lib"], st["kinds"]
    TG.use(tname)
    tgt = P.the_target(7)
    out = []
    for b in budgets:
        extra = dict(costs=learn.learned_costs(lib, kinds)) if guided else {}
        j = learn.job(tgt, lib, kinds, dict(L.FULL, max_states=b, time_limit=5 * 3600, max_cost=80 if guided else 10),
                      "target", 7, general=True, **extra)
        j["breadth_target"] = tname
        t = time.time()
        r = search_job(j)
        out.append(dict(label=label, target=tname, budget=b, guided=guided, lib=len(lib), found=r["found_cost"],
                        levels=r["level_sizes"], stop=r["stop"], seconds=round(time.time() - t, 1),
                        programs=[p["program"] for p in r.get("programs", [])][:5]))
        print(out[-1])
    with open(f"/out/{VERSION}/reach_{label}.json", "w") as f:
        json.dump(out, f)
    vol.commit()
    return out
