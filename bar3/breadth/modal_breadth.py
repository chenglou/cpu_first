"""The breadth runs on Modal (BREADTH.md), bar3/modal_practice.py's pattern: non-preemptible drivers, searches
fanned out to CPU containers, identical jobs shared through a modal.Dict with in-flight claims (the law probes are the
same for every target with the same pairs). Its own app, Volume and Dict; bar3's are untouched.

    cd cpu_first && BREADTH_MAX=40 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/breadth/modal_breadth.py --multi --groups 2 --fresh ct2_clean,lp_clean --targets rho,ten --tag b1
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/breadth/modal_breadth.py --multi --seq lp_clean --order h1,rho,minus1 --carry targets --tag b1
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/breadth/modal_breadth.py --multi --branch bar3/breadth/records/seq_lp_clean_targets_b1.state.affine.json --targets cube,mont
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/breadth/modal_breadth.py --stop lp_clean_ten_b1=2
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/breadth/modal_breadth.py --fetch

Drivers: one per run (driver_fresh, driver_seq), or with --multi many runs in one container, a thread each
(driver_multi: the target is per thread, targets.py), which saves containers since a driver mostly waits. Records:
Volume `bar3-breadth`, v1/<run>.jsonl, one line per round (committed as each round ends); a sequence also saves its
carried state after each target (v1/<run>.state.<target>.json), for --branch (each target alone from that state) or
--resume (driver_seq). --stop NAME=R ends a multi-driver run once it has finished R rounds. Limits: every search has
its own state and time budget (the learner's), search containers a 3-hour timeout and a 32 GiB hard memory limit,
drivers 24 hours; cancelling a driver cancels its inputs.
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
for _f in sorted((ROOT / "library" / "inventing").glob("*.py")):
    image = image.add_local_file(_f, f"/root/library/inventing/{_f.name}")
for _f in sorted((ROOT / "bar3").glob("*.py")):
    image = image.add_local_file(_f, f"/root/bar3/{_f.name}")
for _f in sorted(HERE.glob("*.py")):
    image = image.add_local_file(_f, f"/root/bar3/breadth/{_f.name}")
vol = modal.Volume.from_name("bar3-breadth", create_if_missing=True)
cache = modal.Dict.from_name("bar3-breadth-cache", create_if_missing=True)
VERSION = "v1"
app = modal.App("bar3-breadth", image=image)
MAX = int(os.environ.get("BREADTH_MAX", "32"))


@app.function(cpu=2, memory=(16384, 32768), timeout=3 * 3600, max_containers=MAX,   # (request, hard limit) in MiB
              retries=modal.Retries(max_retries=1))
def search_remote(job: dict) -> dict:
    """One search. A job tagged with a target (a target search, for its N <= 40 check) sets it first. Identical jobs
    are answered from the shared cache; a cached record still counts its states and seconds where it is used."""
    import hashlib
    import sys
    import time
    sys.path.insert(0, "/root")
    from bar3.breadth import targets as TG
    from bar3.jobs import search_job
    key = VERSION + hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()
    patience = job.get("time_limit", 600) + 300
    try:
        hit = cache.get(key)
        if hit is None and not cache.put(key + ":claim", time.time(), skip_if_exists=True):
            # another container claimed it: wait for its record, unless the claim is older than the job can take
            # (its container was cancelled), in which case claim it afresh and compute it here
            while hit is None and time.time() - (cache.get(key + ":claim") or 0) < patience:
                time.sleep(10)
                hit = cache.get(key)
            if hit is None:
                cache[key + ":claim"] = time.time()
    except Exception:
        hit = None
    if hit is not None:
        return dict(hit, cached=True)
    if job.get("breadth_target"):
        TG.use(job["breadth_target"])
    rec = search_job(job)
    try:
        cache[key] = rec
    except Exception:
        pass
    return rec


def _searcher():
    def search_many(jobs):
        return list(search_remote.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    return search_many


def _emitter(path):
    def emit(rec):
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        vol.commit()
    return emit


@app.function(cpu=1, memory=(4096, 8192), timeout=24 * 3600, volumes={"/out": vol}, nonpreemptible=True)
def driver_fresh(name: str, cfg: dict, tname: str) -> str:
    import sys
    sys.path.insert(0, "/root")
    from bar3.breadth import drive
    os.makedirs(f"/out/{VERSION}", exist_ok=True)
    path = f"/out/{VERSION}/{name}.jsonl"
    open(path, "w").close()
    drive.fresh(name, cfg, tname, _searcher(), _emitter(path))
    return path


@app.function(cpu=1, memory=(4096, 8192), timeout=24 * 3600, volumes={"/out": vol}, nonpreemptible=True)
def driver_seq(name: str, cfg: dict, order: list, carry: str, resume: bool = False) -> str:
    import sys
    sys.path.insert(0, "/root")
    from bar3.breadth import carry as C
    from bar3.breadth import drive
    os.makedirs(f"/out/{VERSION}", exist_ok=True)
    path, spath = f"/out/{VERSION}/{name}.jsonl", f"/out/{VERSION}/{name}.state.json"
    state = None
    if resume and os.path.exists(spath):
        saved = json.load(open(spath))
        state = C.from_json(saved["state"])
        order = order[order.index(saved["after"]) + 1:]
        print("resuming after", saved["after"], "->", order)
    else:
        open(path, "w").close()

    def save(tname, st):
        for p in (spath, spath.replace(".state.json", f".state.{tname}.json")):
            json.dump(dict(after=tname, state=C.to_json(st)), open(p, "w"))
        vol.commit()

    drive.carried(name, cfg, order, _searcher(), _emitter(path), carry=carry, state=state, save=save)
    return path


@app.function(cpu=2, memory=(16384, 32768), timeout=24 * 3600, volumes={"/out": vol}, nonpreemptible=True)
def driver_multi(fresh_runs: list, seq_runs: list) -> list:
    """Several runs in one container, one thread each (the target is per thread, targets.py): fresh_runs
    [[name, cfg, target]], seq_runs [[name, cfg, order, carry]]. Saves containers: a driver mostly waits."""
    import sys
    import threading
    import traceback
    sys.path.insert(0, "/root")
    from bar3.breadth import carry as C
    from bar3.breadth import drive
    import time
    os.makedirs(f"/out/{VERSION}", exist_ok=True)
    lock = threading.Lock()
    launch = f"{time.time():.0f}"   # each record says which launch wrote it: a cancelled driver can still append one

    class Stop(Exception):
        pass

    def emitter(path, name):
        """Write each record; then end the run if `stop:<name>` in the Dict says it may not start another round
        (its value is how many rounds the run may finish: --stop on the local entrypoint sets it)."""
        open(path, "w").close()

        def emit(rec):
            with lock:
                with open(path, "a") as f:
                    f.write(json.dumps(dict(rec, launch=launch)) + "\n")
                vol.commit()
            limit = cache.get(f"stop:{name}")
            if limit is not None and "round" in rec and rec["round"] + 1 >= limit:
                raise Stop(f"{name}: stopped after round {rec['round']} (limit {limit})")
        return emit

    def guard(fn, *a, **kw):
        try:
            fn(*a, **kw)
        except Stop as e:
            print(e)
        except Exception:
            traceback.print_exc()

    threads = []
    for name, cfg, tname in fresh_runs:
        threads.append(threading.Thread(target=guard, args=(drive.fresh, name, cfg, tname, _searcher(),
                                                             emitter(f"/out/{VERSION}/{name}.jsonl", name))))
    for name, cfg, order, carry, start in seq_runs:      # start: a saved state to begin from (a branch), or None
        spath = f"/out/{VERSION}/{name}.state.json"

        def save(tname, st, spath=spath):
            with lock:
                for p in (spath, spath.replace(".state.json", f".state.{tname}.json")):
                    json.dump(dict(after=tname, state=C.to_json(st)), open(p, "w"))
                vol.commit()

        threads.append(threading.Thread(target=guard, args=(drive.carried, name, cfg, order, _searcher(),
                                                             emitter(f"/out/{VERSION}/{name}.jsonl", name)),
                                        kwargs=dict(carry=carry, save=save,
                                                    state=C.from_json(start) if start else None)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r[0] for r in fresh_runs + seq_runs]


@app.function(cpu=1, memory=1024, timeout=600, volumes={"/out": vol})
def list_records() -> dict:
    vol.reload()
    d = f"/out/{VERSION}"
    return {f: open(os.path.join(d, f)).read() for f in sorted(os.listdir(d))} if os.path.isdir(d) else {}


@app.local_entrypoint()
def main(fresh: str = "", targets: str = "", seq: str = "", order: str = "", carry: str = "targets",
         resume: bool = False, fetch: bool = False, tag: str = "", only: str = "", overrides: str = "",
         multi: bool = False, suffix: str = "", groups: int = 1, skip: str = "", branch: str = "",
         branch_config: str = "lp_clean", stop: str = "") -> None:
    """--fresh CONFIGS --targets T,...: one driver per (config, target). --seq CONFIGS --order T,...: one driver per
    config, each running the whole sequence. --multi: all of them in `groups` driver containers instead, a thread
    each; with --branch STATE.json (a fetched sequence state), also each of --targets alone from that state, with
    --branch-config. --overrides: JSON merged into each config (the scale tests' anchor, for instance), --suffix
    added to its names, --skip: run names to leave out (finished ones). --stop NAME=R,...: a multi-run driver ends
    run NAME once it has finished R rounds (R = 0: at its next round's end; a sequence counts each target's rounds)."""
    import sys
    sys.path.insert(0, str(ROOT))
    if stop:
        for item in stop.split(","):
            name, limit = item.split("=")
            cache[f"stop:{name}"] = int(limit)
            print("stop", name, "after", limit, "rounds")
        return
    from bar3.breadth.drive import CONFIGS
    extra = json.loads(overrides) if overrides else {}
    calls = {}
    sfx = (f"_{suffix}" if suffix else "") + (f"_{tag}" if tag else "")
    if multi:
        fr = [[f"{c}_{t}{sfx}", json.loads(json.dumps({**CONFIGS[c], **extra})), t]
              for c in (fresh.split(",") if fresh else ()) for t in targets.split(",")]
        sq = [[f"seq_{c}_{carry}{sfx}", json.loads(json.dumps({**CONFIGS[c], **extra})), order.split(","), carry,
               None] for c in (seq.split(",") if seq else ())]
        if branch:                  # each target alone, from a sequence's saved state (a local copy of it)
            saved = json.loads(Path(branch).read_text())
            base = Path(branch).name.split(".state")[0].removeprefix("seq_")
            cfg = json.loads(json.dumps({**CONFIGS[branch_config], **extra}))
            sq += [[f"branch_{base}_after_{saved['after']}_{t}{sfx}", cfg, [t], carry, saved["state"]]
                   for t in targets.split(",")]
        fr = [r for r in fr if r[0] not in skip.split(",")]
        for g in range(groups):
            part_f, part_s = fr[g::groups], sq[g::groups]
            if part_f or part_s:
                call = driver_multi.spawn(part_f, part_s).object_id
                calls.update({r[0]: call for r in part_f + part_s})
    if fresh and not multi:
        for c in fresh.split(","):
            for t in targets.split(","):
                name = f"{c}_{t}{sfx}"
                cfg = json.loads(json.dumps({**CONFIGS[c], **extra}))
                calls[name] = driver_fresh.spawn(name, cfg, t).object_id
    for c in seq.split(",") if seq and not multi else ():
        name = f"seq_{c}_{carry}{sfx}"
        cfg = json.loads(json.dumps({**CONFIGS[c], **extra}))
        calls[name] = driver_seq.spawn(name, cfg, order.split(","), carry, resume).object_id
    if calls:
        print("spawned", calls)
        ids = HERE / "records" / f"calls_{tag or 'default'}.json"
        old = json.loads(ids.read_text()) if ids.exists() else {}
        ids.write_text(json.dumps({**old, **calls}, indent=1))
        print("call ids ->", ids)
        return
    if fetch:
        for f, txt in list_records.remote().items():
            if only and only not in f:
                continue
            (HERE / "records" / f).write_text(txt)
            print("fetched", f, len(txt.splitlines()), "lines")
