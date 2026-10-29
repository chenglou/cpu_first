"""The shared Modal runner (tools/README.md): any submission file, from any folder, through the unmodified One Layer
Deeper evaluator on Modal, safe against preemption and capped in containers. It replaces the per-folder copies of
litmus/modal_litmus.py; those stay as they are.

    PY=~/github/sotaku/venv/bin/python   R=cpu_first/tools/modal_runner.py   # Modal client 1.2.6; any directory
    $PY -m modal run --detach $R --mode official --file learned_remainder/submission.py --out X/records   # H100 only
    $PY -m modal run --detach $R --mode seeds --file X/submission.py --seeds 4 --out X/records   # on b200
    $PY -m modal run --detach $R --mode plan --plan X/plans/P.json --out X/records [--hardware cpu] [--cap 20]
    $PY -m modal run --detach $R --mode resume --run TAG     # a new driver for a run whose driver ended
    $PY $R --mode peek --run TAG [--app-logs 15]             # early check-ins: latest lines and memory per live job
    $PY $R --mode status [--run TAG | --out X/records]
    $PY $R --mode fetch --out X/records [--run TAG] [--wait]
    $PY $R --mode cost --run TAG                             # the run's billed cost (Modal billing report)
    $PY $R --mode compare REF.json NEW.json ...

The launches follow ~/github/tips-for-running-modal: one spawn (the driver) per detached launch, refused under
`modal run` without --detach; `$PY $R` with the same arguments does the same in a detached app of its own. Paths are
relative to the current directory or to cpu_first/. Every job has a timeout and a memory cap (--timeout, --memory-gb;
plan keys timeout_s, memory_gb), its evaluator runs in its own session, and every way out kills that whole session.
The image is the competition's, built layer for layer
as general_lessons/modal_general.py and integration/modal_integration.py build it (Modal reuses the cached build):
every dataset of scripts/generate_datasets.sh by the official generator with its official arguments (Easy e1-e10,
Medium m1-m10 and H1), then the litmus and held-out tasks (litmus/tasks.py, general_lessons/tasks_general.py).

Tasks and manifests, as in litmus/modal_litmus.py: e1..e10, m1..m10 and h1 run the task's official manifest file
untouched when a job has no overrides (an override equal to the official value counts as none); with overrides, a
copy with those keys replaced. The litmus tasks run H1's manifest with data.data_root (and the name) replaced. A CPU
job also sets runtime.device to cpu; a GPU job keeps the official cuda:0.

Each launch is its own detached app `cpu-first-runner`, tagged run=TAG, with one non-preemptible driver. The driver
spawns the jobs, at most --cap (default 20) unfinished at a time, and keeps every call id on the Volume
`cpu-first-runner` under runs/TAG/calls/. A driver that Modal restarts re-attaches to those calls instead of spawning
the plan again. A worker that starts after its record already exists returns at once. Records go to
runs/TAG/records/NAME.json with the keys of the litmus / general_lessons / integration records (name, task, gpu,
returncode, wall_seconds, manifest_file, manifest, stdout, stderr, result, submission), followed by submission_md5,
submission_file, file_md5, set, hardware, runner (run tag, call id, attempt, data md5, timeout) and usage (CPU
seconds, peak memory). Workers copy their log's tail and memory use to runs/TAG/live/ every minute for --mode peek
and status. Hardware: b200 (the default for seeds and plan jobs, the fastest measured), h200 (replays the official
H100 runs exactly, faster), h100 ("H100!", the only hardware for official runs), h100_sealed (h100 without network or other Modal resources, for
other people's files; records say whether the network was blocked) and cpu. Seed policy: tools/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent  # cpu_first/tools (in a container: /root, and nothing below uses it)
CPU_FIRST = HERE.parent
OLD = Path(os.environ.get("ONE_LAYER_DEEPER", str(CPU_FIRST.parent / "one-layer-deeper")))  # the competition clone
REMOTE = "/workspace/one-layer-benchmark"
OFFICIAL = "benchmark/manifests/h100_hard_h1.json"
TIER = {**{f"e{i}": f"benchmark/manifests/h100_easy_e{i}.json" for i in range(1, 11)},
        **{f"m{i}": f"benchmark/manifests/h100_medium_m{i}.json" for i in range(1, 11)}, "h1": OFFICIAL}
LITMUS = ("xfirst", "mont", "ten", "rho", "cube", "xy", "tenrho", "plus3", "minus1")
SEED_ORDER = (74, 1, 2, 3, 4, 5, 6, 7)  # every session's robustness seeds: the official 74, then 1-7
NAME = "cpu-first-runner"  # the app and the Volume
EVAL_FRACTION = 0.5  # benchmark.runner: each seed's evaluation budget is half its training budget
MEMORY_GB = 24  # default cap on the summed RSS of a job's evaluator processes (plan key memory_gb, --memory-gb)
CONTAINER_MIB = 65536  # every worker container's hard memory limit, the cap's backstop (a bare memory=N sets none)
LIVE_EVERY = 60  # seconds between copies of a running job's log tail to the Volume
DEFAULT_HARDWARE = "b200"  # seeds and plan jobs: the fastest measured (tools/README.md); official runs: always h100

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.13")
    .apt_install("git", "build-essential")
    .pip_install("uv==0.11.3")
    .env({"PYTHONPATH": REMOTE})
    .run_commands(f"mkdir -p {REMOTE}")
    .add_local_file(str(OLD / "pyproject.toml"), f"{REMOTE}/pyproject.toml", copy=True)
    .add_local_file(str(OLD / "uv.lock"), f"{REMOTE}/uv.lock", copy=True)
    .add_local_file(str(OLD / "submission_validation.py"), f"{REMOTE}/submission_validation.py", copy=True)
    .run_commands(f"cd {REMOTE} && uv sync --frozen --no-install-project")
    .add_local_dir(str(OLD / "benchmark"), f"{REMOTE}/benchmark", copy=True, ignore=["__pycache__"])
    .add_local_dir(str(OLD / "data"), f"{REMOTE}/data", copy=True, ignore=["__pycache__", "generated"])
    .add_local_file(str(OLD / "scripts" / "generate_datasets.sh"), f"{REMOTE}/scripts/generate_datasets.sh", copy=True)
    .run_commands(f"cd {REMOTE} && PATH={REMOTE}/.venv/bin:$PATH bash scripts/generate_datasets.sh")
    .add_local_file(str(CPU_FIRST / "litmus" / "tasks.py"), f"{REMOTE}/litmus_tasks.py", copy=True)
    .add_local_file(str(CPU_FIRST / "general_lessons" / "tasks_general.py"), f"{REMOTE}/tasks_general.py", copy=True)
    .run_commands(f"cd {REMOTE} && .venv/bin/python -m litmus_tasks data/generated")
    .run_commands(f"cd {REMOTE} && .venv/bin/python -m tasks_general data/generated")
)
app = modal.App(NAME, image=image)
vol = modal.Volume.from_name(NAME, create_if_missing=True)


# ---------------------------------------------------------------------------------------------------------------
# Remote side: the worker (one evaluation) and the driver (one run's scheduler)
# ---------------------------------------------------------------------------------------------------------------

_DATA_MD5: dict = {}


def _data_md5(root: Path) -> str:
    """md5 over the generated dataset's files (relative path and bytes, in sorted order): the data the run read."""
    if str(root) not in _DATA_MD5:
        h = hashlib.md5()
        for p in sorted(q for q in root.rglob("*") if q.is_file()):
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes())
        _DATA_MD5[str(root)] = h.hexdigest()
    return _DATA_MD5[str(root)]


def _manifest(task: str, overrides: dict | None, work: Path) -> str:
    if task in TIER and overrides is None:
        return TIER[task]                                                   # the official file, untouched
    m = json.loads(Path(REMOTE, TIER.get(task, OFFICIAL)).read_text())
    if task in LITMUS:
        m["name"] = f"litmus-{task}"
        m["data"]["data_root"] = f"data/generated/litmus_{task}"
    for section, values in (overrides or {}).items():
        m[section].update(values)
    path = work / "manifest.json"
    path.write_text(json.dumps(m, indent=1))
    return str(path)


def _cgroup() -> dict:
    out = {}
    try:
        out["cpu_usec"] = int(next(l.split()[1] for l in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines()
                                   if l.startswith("usage_usec")))
    except Exception:
        pass
    for key, path in (("memory_peak", "/sys/fs/cgroup/memory.peak"), ("cpu_max", "/sys/fs/cgroup/cpu.max")):
        try:
            out[key] = Path(path).read_text().strip()
        except Exception:
            pass
    return out


def _usage(ru0, cg0: dict, wall: float) -> dict:
    import resource
    ru, cg = resource.getrusage(resource.RUSAGE_CHILDREN), _cgroup()
    cpu = ru.ru_utime - ru0.ru_utime + ru.ru_stime - ru0.ru_stime
    out = {"child_cpu_seconds": round(cpu, 1), "child_cpu_cores_mean": round(cpu / max(wall, 1e-9), 2),
           "child_max_rss_mb": round(ru.ru_maxrss / 1024), "nproc": os.cpu_count(),
           "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
           "cpu_model": next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                              if l.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else None}
    if "cpu_usec" in cg and "cpu_usec" in cg0:
        out["container_cpu_seconds"] = round((cg["cpu_usec"] - cg0["cpu_usec"]) / 1e6, 1)
    if cg.get("memory_peak", "").isdigit():
        out["container_memory_peak_mb"] = round(int(cg["memory_peak"]) / 2**20)   # the container's lifetime peak
    if "cpu_max" in cg:
        out["cgroup_cpu_max"] = cg["cpu_max"]
    return out


def _session(sid: int) -> list:
    """(pid, RSS bytes) of the live processes in session `sid`: the evaluator and its DataLoader workers."""
    page, out = os.sysconf("SC_PAGE_SIZE"), []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                stat = Path(f"/proc/{d}/stat").read_text()
            except OSError:
                continue
            rest = stat[stat.rindex(")") + 2:].split()   # fields 3.. of proc(5): state, ppid, pgrp, session, ...
            if int(rest[3]) == sid and rest[0] != "Z":
                out.append((int(d), int(rest[21]) * page))
    return out


def _kill_session(proc) -> int:
    """SIGKILL the evaluator's whole session, children included, and reap it; returns the processes still alive."""
    import signal
    for _ in range(3):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        for pid, _ in _session(proc.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        proc.wait()
        time.sleep(0.5)
        if not _session(proc.pid):
            return 0
    return len(_session(proc.pid))


def _live(name: str, dest: Path, log: Path, err: Path, elapsed: float, rss: int, peak: int) -> None:
    """The log's last lines and the memory in use, to the Volume (runs/TAG/live/NAME.log) for --mode peek and
    status, and one progress line to the container log: `modal container exec` does not see /tmp on every container."""
    out, errs = log.read_text().splitlines(), err.read_text().splitlines()
    step = next((l for l in reversed(out) if l.startswith("step=")), "")
    head = (f"# {time.time():.0f} {time.strftime('%Y-%m-%d %H:%M:%S %Z')} elapsed {elapsed:.0f}s, "
            f"rss {rss / 2**30:.1f} GiB (peak {peak / 2**30:.1f})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join([head, *out[-30:], "# stderr (last 8 lines)", *errs[-8:]]) + "\n")
    vol.commit()
    print(f"[{name}] {elapsed:.0f}s rss {rss / 2**30:.1f} GiB peak {peak / 2**30:.1f} GiB | {step[:100]}", flush=True)


def _evaluate(tag: str, job: dict) -> dict:
    import resource
    name, task, src = job["name"], job["task"], job["source"]
    run = Path("/out/runs") / tag
    record = run / "records" / f"{name}.json"
    vol.reload()
    if record.exists():                                     # a restarted or duplicate input: the work is done
        return {"name": name, "skipped": "record exists"}
    me = modal.current_function_call_id()
    started = run / "started" / f"{name}.json"
    attempt = 1
    if started.exists():
        prev = json.loads(started.read_text())
        attempt = prev.get("attempt", 1) + 1 if prev.get("call_id") == me else 1
    started.parent.mkdir(parents=True, exist_ok=True)
    started.write_text(json.dumps({"call_id": me, "container": os.environ.get("MODAL_TASK_ID"), "attempt": attempt,
                                   "started_at": time.time()}))
    vol.commit()

    work = Path(f"/tmp/{name}")
    work.mkdir(parents=True, exist_ok=True)
    (work / "submission.py").write_bytes(src.encode("utf-8"))
    md5 = hashlib.md5((work / "submission.py").read_bytes()).hexdigest()
    if md5 != job["submission_md5"]:
        raise RuntimeError(f"{name}: the submission's bytes changed on the way ({md5} != {job['submission_md5']})")
    manifest = _manifest(task, job["overrides"], work)
    m = json.loads(Path(REMOTE, manifest).read_text())
    limit = job.get("timeout_s") or int(m["runtime"]["total_training_time_seconds"] * (1 + EVAL_FRACTION)
                                        * len(m["runtime"]["seeds"]) + 1800)
    cap = int(float(job.get("memory_gb") or MEMORY_GB) * 2**30)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() if os.path.exists("/usr/bin/nvidia-smi") else ""
    data_md5 = _data_md5(Path(REMOTE, m["data"]["data_root"]))
    network = None
    if job["hardware"].endswith("_sealed"):                 # record that the seal holds, not just that it was asked for
        import socket
        try:
            socket.create_connection(("1.1.1.1", 443), timeout=3).close()
            network = "open"
        except OSError as exc:
            network = f"blocked ({type(exc).__name__})"
    ru0, cg0, t0 = resource.getrusage(resource.RUSAGE_CHILDREN), _cgroup(), time.time()
    log, err = work / "stdout.log", work / "stderr.log"
    killed, peak, copied = None, 0, t0
    with open(log, "w") as fo, open(err, "w") as fe:
        proc = subprocess.Popen([f"{REMOTE}/.venv/bin/python", "-m", "benchmark.runner", "--manifest", manifest,
                                 "--submission-file", str(work / "submission.py"), "--include-structured-metrics"],
                                cwd=REMOTE, stdout=fo, stderr=fe, text=True, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                                start_new_session=True)       # its own session: the kill below reaches every child
        try:
            while True:
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    pass
                rss = sum(r for _, r in _session(proc.pid))
                peak, now = max(peak, rss), time.time()
                if now - t0 > limit:
                    killed = f"timeout: {limit} s"
                elif rss > cap:
                    killed = f"memory: {rss / 2**30:.1f} GiB > cap {cap / 2**30:g} GiB"
                if killed:
                    print(f"[{name}] killed, {killed}", flush=True)
                    break
                if now - copied >= LIVE_EVERY:
                    copied = now
                    try:
                        _live(name, run / "live" / f"{name}.log", log, err, now - t0, rss, peak)
                    except Exception as exc:
                        print(f"[{name}] live copy failed: {exc!r}", flush=True)
        finally:                                              # every way out, a cancelled input included
            leftover = _kill_session(proc)
    returncode = -9 if killed else proc.returncode
    wall = time.time() - t0
    stdout, stderr = log.read_text(), err.read_text()
    try:
        result = next((json.loads(l.removeprefix("RESULT_JSON=")) for l in reversed(stdout.splitlines())
                       if l.startswith("RESULT_JSON=")), None)
    except ValueError:                                        # a cut-off line: keep the record, without a result
        result = None
    rec = {"name": name, "task": task, "gpu": gpu, "returncode": returncode, "wall_seconds": wall,
           "manifest_file": manifest if manifest.startswith("benchmark/") else None, "manifest": m,
           "stdout": stdout, "stderr": stderr[-200000:], "result": result, "submission": src,
           "submission_md5": md5, "submission_file": job["file"], "file_md5": job["file_md5"], "set": job["set"],
           "hardware": job["hardware"],
           "runner": {"tool": "cpu_first/tools/modal_runner.py", "tool_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
                      "run": tag, "call_id": me, "container": os.environ.get("MODAL_TASK_ID"), "attempt": attempt,
                      "data_md5": data_md5, "started_at": t0, "timeout_seconds": limit,
                      "timed_out": bool(killed and killed.startswith("timeout")), "memory_cap_gb": cap / 2**30,
                      "killed": killed, "leftover_processes": leftover, "network": network},
           "usage": dict(_usage(ru0, cg0, wall), peak_session_rss_mb=round(peak / 2**20))}
    record.parent.mkdir(parents=True, exist_ok=True)
    part = record.with_name(f".{name}.json.part")
    part.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(part, record)                                  # whole records only
    vol.commit()
    return {"name": name, "returncode": returncode}


_WORKER = dict(timeout=24 * 3600, volumes={"/out": vol}, max_containers=20)   # 20: a backstop under the driver's cap
_CPU_MEMORY, _GPU_MEMORY = (16384, CONTAINER_MIB), (4096, CONTAINER_MIB)   # (request, hard limit) in MiB


@app.function(gpu="B200", memory=_GPU_MEMORY, **_WORKER)   # the default: the fastest measured (tools/README.md)
def run_b200(tag: str, job: dict) -> dict:
    return _evaluate(tag, job)


@app.function(gpu="H200", memory=_GPU_MEMORY, **_WORKER)   # replays the official H100 runs exactly, faster
def run_h200(tag: str, job: dict) -> dict:
    return _evaluate(tag, job)


@app.function(gpu="H100!", memory=_GPU_MEMORY, **_WORKER)   # "!": never an H200. The official hardware
def run_h100(tag: str, job: dict) -> dict:
    return _evaluate(tag, job)


@app.function(cpu=8, memory=_CPU_MEMORY, **_WORKER)   # the CPU container of every earlier runner's robustness seeds
def run_cpu(tag: str, job: dict) -> dict:
    return _evaluate(tag, job)


@app.function(gpu="H100!", memory=_GPU_MEMORY, block_network=True, restrict_modal_access=True, **_WORKER)
def run_h100_sealed(tag: str, job: dict) -> dict:   # other people's files: no network, no other Modal resources
    return _evaluate(tag, job)


HARDWARE = {"b200": run_b200, "h200": run_h200, "h100": run_h100, "cpu": run_cpu, "h100_sealed": run_h100_sealed}
GPUS = {"b200", "h200", "h100", "h100_sealed"}


def _poll(call) -> object:
    """None while the call is queued or running, else its outcome. A failed read of a live call is not an outcome:
    the call graph decides (tips-for-running-modal: never replace a call over a connection error)."""
    try:
        return call.get(timeout=0) or "done"
    except TimeoutError:
        return None
    except modal.exception.OutputExpiredError:
        return "output expired (the record on the Volume is the result)"
    except Exception as exc:
        try:
            graph = call.get_call_graph()
            if not graph or graph[0].status == modal.call_graph.InputStatus.PENDING:
                return None
        except Exception:
            return None
        return f"failed: {exc!r}"[:500]


@app.function(cpu=0.25, memory=512, timeout=24 * 3600, volumes={"/out": vol}, nonpreemptible=True)
def driver(tag: str) -> dict:
    """One run's scheduler: spawns its jobs in plan order, at most `cap` unfinished at a time. Every call id is
    committed to runs/TAG/calls/ as soon as it is spawned, so a restarted driver (same input) re-attaches to the calls
    it made instead of spawning them again."""
    run = Path("/out/runs") / tag
    vol.reload()
    jobs = json.loads((run / "jobs.json").read_text())
    meta = json.loads((run / "run.json").read_text())
    cap = int(meta["cap"])
    (run / "calls").mkdir(parents=True, exist_ok=True)
    live, done = {}, {}
    for j in jobs:
        f = run / "calls" / f"{j['name']}.id"
        if f.exists():
            live[j["name"]] = modal.FunctionCall.from_id(f.read_text().split()[0])   # older runs add a spawn time
    pending = [j for j in jobs if j["name"] not in live]
    print(f"driver {tag}: {len(live)} calls re-attached, {len(pending)} to spawn, cap {cap}", flush=True)
    status, last = {}, None
    while pending or live:
        vol.reload()
        while pending and len(live) < cap:
            j = pending.pop(0)
            if (run / "records" / f"{j['name']}.json").exists():
                done[j["name"]] = "record exists"
                continue
            call = HARDWARE[j["hardware"]].spawn(tag, j)
            (run / "calls" / f"{j['name']}.id").write_text(call.object_id)
            vol.commit()
            live[j["name"]] = call
            print(f"spawned {j['name']} on {j['hardware']}: {call.object_id}", flush=True)
        for name, call in list(live.items()):
            outcome = _poll(call)
            if outcome is not None:
                done[name] = outcome
                del live[name]
                print(f"finished {name}: {outcome}", flush=True)
        status = {"live": sorted(live), "pending": [j["name"] for j in pending], "done": done}
        if status != last:
            (run / "status.json").write_text(json.dumps(dict(status, updated=time.time()), indent=1, default=str))
            vol.commit()
            last = json.loads(json.dumps(status, default=str))
        if pending or live:
            time.sleep(15)
    return done


# ---------------------------------------------------------------------------------------------------------------
# Local side: jobs, launch, fetch, status, resume, cost, compare
# ---------------------------------------------------------------------------------------------------------------

def _resolve(p: str | Path) -> Path:
    p = Path(p).expanduser()
    for cand in ([p] if p.is_absolute() else [Path.cwd() / p, CPU_FIRST / p]):
        if cand.exists():
            return cand.resolve()
    raise SystemExit(f"not found (relative to {Path.cwd()} or {CPU_FIRST}): {p}")


def _out_dir(p: str | Path) -> Path:
    """A records folder, which may not exist yet: relative to the current directory if its parent is there, else to
    cpu_first/."""
    p = Path(p).expanduser()
    if p.is_absolute():
        return p.resolve()
    for base in (Path.cwd(), CPU_FIRST):
        if (base / p).exists() or (base / p).parent.exists():
            return (base / p).resolve()
    raise SystemExit(f"no parent folder for {p} (relative to {Path.cwd()} or {CPU_FIRST})")


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(CPU_FIRST)) if CPU_FIRST in p.parents or p == CPU_FIRST else str(p)


def _variant(src: str, sets: dict) -> str:
    for key, value in sets.items():
        src, count = re.subn(rf"^{key} = .*$", f"{key} = {value}", src, count=1, flags=re.M)
        if count != 1:
            raise KeyError(key)
    return src


def _official(task: str) -> dict:
    return json.loads((OLD / TIER.get(task, OFFICIAL)).read_text())


def _overrides_of_record(rec: dict) -> dict | None:
    """The overrides that reproduce a stored record's manifest: none if it ran an untouched official file."""
    m = rec.get("manifest")
    if rec.get("manifest_file") or isinstance(m, str) or not m:
        return None
    return {s: dict(v) for s, v in m.items() if isinstance(v, dict)}


def _canonical(task: str, overrides: dict | None, hardware: str) -> dict | None:
    """Merge the device rule into a job's overrides and drop every key equal to the official manifest's value, so a
    job whose overrides change nothing runs the untouched official file (for example an H100 job with seed 74)."""
    over = {s: dict(v) for s, v in (overrides or {}).items()}
    if hardware in GPUS:
        over.get("runtime", {}).pop("device", None)      # a GPU job keeps the official cuda:0 (even from a CPU record)
    else:
        over.setdefault("runtime", {})["device"] = "cpu"
    base = _official(task)
    if task in LITMUS:
        base["name"], base["data"]["data_root"] = f"litmus-{task}", f"data/generated/litmus_{task}"
    out = {}
    for s, values in over.items():
        if s not in base:
            raise SystemExit(f"unknown manifest section {s!r}")
        changed = {k: v for k, v in values.items() if base[s].get(k) != v}
        if changed:
            out[s] = changed
    return out or None


def _job(j: dict, default_file: str | None = None, default_hardware: str | None = None,
         memory_gb: float | None = None, timeout_s: int | None = None) -> dict:
    """A launchable job from a plan entry. Hardware: the entry's `hardware`; else the old `gpu` flag; else cpu if the
    entry's own manifest sets runtime.device to cpu (the old plans' way to ask for CPU); else the run's --hardware,
    else DEFAULT_HARDWARE. Limits: the entry's memory_gb / timeout_s, else the run's --memory-gb / --timeout."""
    if j.get("hardware"):
        hardware = j["hardware"]
    elif "gpu" in j:
        hardware = "h100" if j["gpu"] else "cpu"
    elif ((j.get("manifest") or {}).get("runtime") or {}).get("device") == "cpu":
        hardware = "cpu"
    else:
        hardware = default_hardware or DEFAULT_HARDWARE
    if hardware not in HARDWARE:
        raise SystemExit(f"{j.get('name')}: hardware {hardware!r} is not one of {sorted(HARDWARE)}")
    mem = float(j.get("memory_gb") or memory_gb or MEMORY_GB)
    if not 1 <= mem <= CONTAINER_MIB / 1024 - 4:
        raise SystemExit(f"{j.get('name')}: memory_gb {mem:g} must be within 1-{CONTAINER_MIB // 1024 - 4}"
                         f" (the container's hard limit is {CONTAINER_MIB // 1024} GiB)")
    if j.get("record"):                                     # the exact source (and manifest) of a stored record
        path = _resolve(j["record"])
        rec = json.loads(path.read_text())
        raw, task = rec["submission"].encode("utf-8"), j.get("task", rec.get("task", "h1"))
        overrides = j["manifest"] if "manifest" in j else _overrides_of_record(rec)
        origin = f"record:{_rel(path)}"
    else:
        if not (j.get("file") or default_file):
            raise SystemExit(f"{j.get('name')}: no file (give the job a file or pass --file)")
        path = _resolve(j.get("file") or default_file)
        raw, task, overrides, origin = path.read_bytes(), j.get("task", "h1"), j.get("manifest"), _rel(path)
    if task not in TIER and task not in LITMUS:
        raise SystemExit(f"{j.get('name')}: unknown task {task!r}")
    if j.get("seed") is not None:
        overrides = {s: dict(v) for s, v in (overrides or {}).items()}
        overrides.setdefault("runtime", {})["seeds"] = [int(j["seed"])]
    src = _variant(raw.decode("utf-8"), j.get("set", {}))
    return {"name": j["name"], "task": task, "file": origin, "file_md5": hashlib.md5(raw).hexdigest(),
            "set": j.get("set", {}), "hardware": hardware, "overrides": _canonical(task, overrides, hardware),
            "memory_gb": mem, "timeout_s": int(j.get("timeout_s") or timeout_s or 0) or None,
            "source": src, "submission_md5": hashlib.md5(src.encode("utf-8")).hexdigest()}


def _read(path: str) -> bytes:
    return b"".join(vol.read_file(path))


def _listdir(path: str) -> list:
    try:
        return vol.listdir(path)
    except Exception:
        return []


def _put(files: dict) -> None:
    with vol.batch_upload(force=True) as b:
        for remote, data in files.items():
            b.put_file(io.BytesIO(data), remote)


def _runs(prefix: str = "") -> dict:
    """run.json of every run (whose tag starts with `prefix`), read in parallel: each Volume read takes ~0.5 s."""
    from concurrent.futures import ThreadPoolExecutor

    def one(tag):
        try:
            return tag, json.loads(_read(f"runs/{tag}/run.json"))
        except Exception:
            return tag, None

    tags = [Path(e.path).name for e in _listdir("runs") if Path(e.path).name.startswith(prefix)]
    with ThreadPoolExecutor(16) as pool:
        return {t: m for t, m in pool.map(one, tags) if m is not None}


def _select(tag: str | None, out: str | None, latest: int = 0) -> list:
    if tag:
        try:
            return [(tag, json.loads(_read(f"runs/{tag}/run.json")))]     # an exact tag: one read
        except Exception:
            pass
    runs = _runs(tag or "")
    if tag:
        hits = [t for t in runs if t == tag] or [t for t in runs if t.startswith(tag)]
        if len(hits) != 1:
            raise SystemExit(f"run {tag!r}: {'no match' if not hits else 'ambiguous: ' + ', '.join(sorted(hits))}")
        return [(hits[0], runs[hits[0]])]
    sel = sorted(runs.items(), key=lambda kv: kv[1].get("created", 0))
    if out:
        sel = [(t, m) for t, m in sel if m.get("out") == _rel(_out_dir(out))]
    return sel[-latest:] if latest else sel


def _status_of(call_id: str) -> str:
    """A call's state from its result (the call graph can show a stopped attempt's FAILURE for a while)."""
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
        return "SUCCESS"
    except TimeoutError:
        return "PENDING"
    except modal.exception.OutputExpiredError:
        return "ENDED (output expired)"
    except (modal.exception.ConnectionError, modal.exception.InternalFailure) as exc:
        return f"unknown ({type(exc).__name__})"
    except Exception as exc:
        return f"FAILURE ({type(exc).__name__}: {str(exc)[:80]})"


_MODAL_RUN = {"active": False, "detached": False}   # set by `entry`, the `modal run` entrypoint at the end


def _check_launch_mode() -> None:
    if _MODAL_RUN["active"] and not _MODAL_RUN["detached"]:
        raise SystemExit("launch with `modal run --detach ...` (without --detach the app stops when this entrypoint "
                         "returns, and the driver with it), or run this file with python")


def _spawn_driver(tag: str):
    """The launch's one spawn (tips-for-running-modal: a detached entrypoint keeps only the last call it spawned
    alive). Under `modal run --detach`, into the running app; under python, into a detached app of its own."""
    def spawn():
        try:
            app.set_tags({"run": tag[:60]})
        except Exception as exc:
            print(f"warning: app tags not set ({exc!r})", flush=True)
        return app.app_id, driver.spawn(tag)
    if _MODAL_RUN["active"]:
        return spawn()
    with modal.enable_output():
        with app.run(detach=True):
            return spawn()


def _launch(jobs: list, out: Path, cap: int, tag: str, label: str, force: bool) -> None:
    _check_launch_mode()
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate job names: {sorted({n for n in names if names.count(n) > 1})}")
    clash = [n for n in names if (out / f"{n}.json").exists()]
    for t, m in _runs().items():
        if t == tag:
            raise SystemExit(f"run {tag} exists already; pick another --run or use --mode resume")
        if m.get("out") == _rel(out):
            clash += [n for n in names if n in m.get("names", [])]
    if clash and not force:
        raise SystemExit(f"record names already used for {_rel(out)} (a local record or an earlier run): "
                         f"{sorted(set(clash))}; rename the jobs or pass --force")
    meta = {"tag": tag, "label": label, "created": time.time(), "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "out": _rel(out), "cap": cap, "names": names,
            "jobs": [{k: v for k, v in j.items() if k != "source"} for j in jobs]}
    _put({f"runs/{tag}/jobs.json": json.dumps(jobs).encode(), f"runs/{tag}/run.json": json.dumps(meta, indent=1).encode()})
    app_id, call = _spawn_driver(tag)
    meta.update(app_id=app_id, driver_calls=[call.object_id])
    _put({f"runs/{tag}/run.json": json.dumps(meta, indent=1).encode()})
    me = f"{sys.executable} {os.path.relpath(__file__)}"
    print(f"\nrun {tag}: {len(jobs)} jobs, cap {cap}, app {app_id}, driver {call.object_id}\n"
          f"  check in: {me} --mode peek --run {tag}     (every few minutes at first)\n"
          f"  fetch:    {me} --mode fetch --run {tag} --wait     (records -> {_rel(out)})", flush=True)


def summary(rec: dict) -> str:
    r = rec.get("result") or {}
    dp, s = r.get("depth_profile") or {}, (r.get("seeds") or [{}])[0]
    frozen = re.search(r"^LR frozen step=(\d+)", rec.get("stdout", ""), re.M)
    splits = " ".join(f"{k}={v.get('correct_examples')}/{v.get('example_count')}"
                      for k, v in sorted((s.get("evaluation") or {}).items()))
    where = rec.get("hardware") or ("h100" if "H100" in rec.get("gpu", "") else "cpu")
    return (f"{rec['name']} [{where}] rc={rec.get('returncode')} T={dp.get('max_certified_time_steps')}/"
            f"{dp.get('ood_n_max_certified_time_steps')} steps={s.get('completed_training_steps')} "
            f"frozen={frozen.group(1) if frozen else '-'} train={s.get('training_seconds') or 0:.0f}s "
            f"eval={s.get('evaluation_seconds') or 0:.0f}s wall={rec.get('wall_seconds') or 0:.0f}s {splits} "
            f"md5={hashlib.md5(rec['submission'].encode()).hexdigest()[:8]}")


def _fetch(sel: list, out: str | None, wait: bool, force: bool) -> None:
    seen: set = set()
    while True:
        missing, live = 0, 0
        for tag, meta in sel:
            dest_dir = _out_dir(out) if out else CPU_FIRST / meta["out"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            have = {Path(e.path).name: e for e in _listdir(f"runs/{tag}/records")}
            for name in meta["names"]:
                e = have.get(f"{name}.json")
                if e is None:
                    missing += 1
                    continue
                dest = dest_dir / f"{name}.json"
                if dest.exists() and dest.stat().st_size == e.size:
                    if (tag, name) not in seen and not wait:
                        print("have   ", name, flush=True)
                    seen.add((tag, name))
                    continue
                if dest.exists() and not force:
                    try:
                        mine = json.loads(dest.read_text()).get("runner", {}).get("run") == tag
                    except Exception:
                        mine = False
                    if not mine:
                        print(f"skip    {dest}: a different record is there (pass --force to replace it)", flush=True)
                        seen.add((tag, name))
                        continue
                data = _read(e.path)
                part = dest.with_name(f".{dest.name}.{os.getpid()}.part")   # one per process: fetches may overlap
                part.write_bytes(data)
                part.replace(dest)
                seen.add((tag, name))
                print("fetched", summary(json.loads(data)), flush=True)
            live += any(_status_of(c).startswith(("PENDING", "unknown")) for c in meta.get("driver_calls", []))
        if not wait or not missing:
            break
        if not live:
            print(f"{missing} records missing and no driver is running: see --mode status (or --mode resume)")
            break
        time.sleep(60)


def _status(sel: list) -> None:
    for tag, meta in sel:
        drivers = [_status_of(c) for c in meta.get("driver_calls", [])]
        have = {Path(e.path).stem for e in _listdir(f"runs/{tag}/records")}
        print(f"== {tag} ({meta.get('created_at')}, {meta.get('label') or ''}) -> {meta['out']}: "
              f"{len(have & set(meta['names']))}/{len(meta['names'])} records; cap {meta.get('cap')}; "
              f"driver {', '.join(drivers) or '-'}; app {meta.get('app_id')}")
        if len(sel) > 1 and len(have & set(meta["names"])) == len(meta["names"]):
            continue
        calls = {Path(e.path).stem for e in _listdir(f"runs/{tag}/calls")}
        for name in meta["names"]:
            if name in have:
                print("  done    ", summary(json.loads(_read(f"runs/{tag}/records/{name}.json"))))
            elif name in calls:
                try:
                    st = json.loads(_read(f"runs/{tag}/started/{name}.json"))
                except Exception:
                    print(f"  queued   {name}")
                    continue
                try:
                    lines = _read(f"runs/{tag}/live/{name}.log").decode().splitlines()
                    last = next((l for l in reversed(lines) if l.startswith("step=")), lines[-1] if lines else "")
                except Exception:
                    last = "(no log copied yet)"
                print(f"  running  {name}: since {time.strftime('%H:%M', time.localtime(st['started_at']))}, "
                      f"attempt {st['attempt']}, container {st['container']}; {last[:90]}")
            else:
                print(f"  pending  {name} (not spawned)")


def _peek(sel: list, lines: int, app_seconds: int) -> None:
    """Early check-ins: each unfinished job's latest log lines and memory, from the copy its worker makes every
    LIVE_EVERY seconds; with app_seconds, also the run's app log (driver and progress lines), read for that long."""
    for tag, meta in sel:
        have = {Path(e.path).stem for e in _listdir(f"runs/{tag}/records")}
        live = {Path(e.path).stem for e in _listdir(f"runs/{tag}/live")}
        todo = [n for n in meta["names"] if n not in have]
        print(f"== {tag}: {len(meta['names']) - len(todo)}/{len(meta['names'])} records in")
        for name in todo:
            if name not in live:
                print(f"-- {name}: no log copied yet (queued, or started less than {LIVE_EVERY} s ago)")
                continue
            text = _read(f"runs/{tag}/live/{name}.log").decode().splitlines()
            stamp = re.match(r"# (\d+) ", text[0]) if text else None
            age = time.time() - float(stamp.group(1)) if stamp else None
            stale = age is not None and age > 3 * LIVE_EVERY
            print(f"-- {name}: copied {age:.0f} s ago{' (STALE: stalled, preempted or dead?)' if stale else ''}"
                  if age is not None else f"-- {name}")
            body = [l for l in text[1:] if not l.startswith("# stderr")]
            print("\n".join([text[0]] + body[-lines:]) if text else "")
        if app_seconds:
            for app_id in [meta.get("app_id"), *meta.get("resumed_apps", [])][-1:]:
                try:
                    p = subprocess.run([sys.executable, "-m", "modal", "app", "logs", app_id], capture_output=True,
                                       text=True, timeout=app_seconds)
                    out = p.stdout
                except subprocess.TimeoutExpired as exc:
                    out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                keep = [l for l in out.splitlines() if l.strip() and not re.search(r"CUDA|NVIDIA|====|license|toolkit", l, re.I)]
                print(f"-- app {app_id} log, last {lines} lines read within {app_seconds} s:")
                print("\n".join(keep[-lines:]) or "(nothing yet)")


def _resume(tag: str, force: bool) -> None:
    _check_launch_mode()
    [(tag, meta)] = _select(tag, None)
    alive = [c for c in meta.get("driver_calls", []) if _status_of(c).startswith(("PENDING", "unknown"))]
    if alive and not force:
        raise SystemExit(f"run {tag}: driver {alive[0]} is still running; pass --force to start a second one")
    app_id, call = _spawn_driver(tag)
    meta.setdefault("resumed_apps", []).append(app_id)
    meta["driver_calls"] = meta.get("driver_calls", []) + [call.object_id]
    _put({f"runs/{tag}/run.json": json.dumps(meta, indent=1).encode()})
    print(f"run {tag}: new driver {call.object_id}", flush=True)


PRICE = {"cpu_core": 0.0000131, "memory_gib": 0.00000222, "b200": 0.001736, "h200": 0.001261, "h100": 0.001097,
         "h100_sealed": 0.001097,
         "a10": 0.000306, "l4": 0.000222, "t4": 0.000164}   # $/s, modal.com/pricing on 2026-10-22; CPU per physical
                                                             # core; CPU and memory billed at max(request, use)
REQUEST = {"cpu": (8, 16), "cpu4": (4, 16)}   # (cores, GiB) of run_cpu and the retired run_cpu4; GPU functions: 4 GiB


def estimate(rec: dict) -> float:
    """List-price cost of a record's evaluation: GPU seconds, plus CPU and memory at max(request, mean use)."""
    hw, wall, u = rec.get("hardware"), rec.get("wall_seconds") or 0, rec.get("usage") or {}
    if hw == "fast":                                          # older records: priced as the GPU it got
        hw = next((k for k in ("b200", "h200", "h100") if k.upper() in (rec.get("gpu") or "")), "b200")
    cores, gib = REQUEST.get(hw, (0.125, 4))
    cores = max(cores, (u.get("container_cpu_seconds") or u.get("child_cpu_seconds") or 0) / max(wall, 1e-9))
    gib = max(gib, (u.get("container_memory_peak_mb") or u.get("child_max_rss_mb") or 0) / 1024)
    return wall * (PRICE.get(hw, 0) + cores * PRICE["cpu_core"] + gib * PRICE["memory_gib"])


def _cost(sel: list) -> None:
    from datetime import datetime, timedelta, timezone
    from modal.billing import workspace_billing_report
    for tag, meta in sel:
        apps = {meta.get("app_id"), *meta.get("resumed_apps", [])} - {None}
        start = datetime.fromtimestamp(meta["created"], timezone.utc).replace(minute=0, second=0, microsecond=0)
        rows = workspace_billing_report(start=start - timedelta(hours=1), resolution="h")
        billed = sum(float(r["cost"]) for r in rows if r["object_id"] in apps)
        last = max((r["interval_start"] for r in rows if r["object_id"] in apps), default=None)
        recs = [json.loads(_read(e.path)) for e in _listdir(f"runs/{tag}/records")]
        est = {r["name"]: estimate(r) for r in recs}
        print(f"== {tag}: billed ${billed:.3f} for app(s) {sorted(apps)} (complete hours only; last {last}); "
              f"list-price estimate of the {len(recs)} evaluations ${sum(est.values()):.3f}")
        for r in sorted(recs, key=lambda r: r["name"]):
            u = r.get("usage") or {}
            print(f"  {r['name']} [{r.get('hardware')}] wall {r['wall_seconds']:.0f}s est ${est[r['name']]:.3f}: "
                  f"{u.get('child_cpu_cores_mean')} cores busy on average, peak RSS {u.get('child_max_rss_mb')} MB, "
                  f"{(r.get('gpu') or 'no GPU').split(',')[0]}")


def _facts(rec: dict) -> dict:
    r = rec.get("result") or {}
    dp, s = r.get("depth_profile") or {}, (r.get("seeds") or [{}])[0]
    sdp = s.get("depth_profile") or {}
    trace = [(m["step"], m.get("loss"), m.get("exact_accuracy")) for m in r.get("structured_metrics") or []
             if m.get("type") == "train"]
    log = [l for l in rec.get("stdout", "").splitlines() if not re.search(r"elapsed=|seconds|RESULT_JSON=", l)]
    frozen = re.search(r"^LR frozen step=(\d+)", rec.get("stdout", ""), re.M)
    m = rec.get("manifest")
    untouched = bool(rec.get("manifest_file")) or isinstance(m, str)   # older records keep only the official path
    return {
        "submission md5": hashlib.md5(rec["submission"].encode()).hexdigest(),
        "manifest": rec.get("manifest_file") or (m if isinstance(m, str) else "a copy with overrides"),
        "manifest runtime": "(the official file)" if untouched else (m or {}).get("runtime"),
        "returncode": rec.get("returncode"),
        "certified T (seen N, unseen N)": (dp.get("max_certified_time_steps"), dp.get("ood_n_max_certified_time_steps")),
        "rungs, seen N": [(g["time_steps"], g["correct_examples"], g["example_count"]) for g in sdp.get("rungs", [])],
        "rungs, unseen N": [(g["time_steps"], g["correct_examples"], g["example_count"]) for g in sdp.get("ood_n_rungs", [])],
        "splits": {k: (v.get("correct_examples"), v.get("example_count")) for k, v in (s.get("evaluation") or {}).items()},
        "split losses": {k: v.get("loss") for k, v in (s.get("evaluation") or {}).items()},
        "steps": s.get("completed_training_steps"),
        "frozen step": int(frozen.group(1)) if frozen else None,
        "final train loss": s.get("final_train_loss"),
        "state elements (model, optimizer)": (s.get("model_state_elements"), s.get("optimizer_state_elements_after_first_step")),
        "train trace (step, loss, accuracy)": trace,
        "log without timings": log,
    }


def _compare(paths: list) -> None:
    recs = [json.loads(_resolve(p).read_text()) for p in paths]
    ref = _facts(recs[0])
    print(f"reference: {paths[0]}\n  {summary(recs[0])}")
    for p, rec in zip(paths[1:], recs[1:]):
        print(f"\n{p}\n  {summary(rec)}")
        facts = _facts(rec)
        for key, a in ref.items():
            b = facts[key]
            if a == b:
                n = f" ({len(a)} entries)" if isinstance(a, list) else ""
                print(f"  same     {key}{n}")
            elif isinstance(a, list) and isinstance(b, list):
                i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                print(f"  DIFFERS  {key}: {len(a)} vs {len(b)} entries, first difference at #{i}: "
                      f"{a[i] if i < len(a) else '-'} vs {b[i] if i < len(b) else '-'}")
            else:
                print(f"  DIFFERS  {key}: {a} vs {b}")
        t = lambda r: ((r.get("result") or {}).get("seeds") or [{}])[0]
        print(f"  timing   train {t(recs[0]).get('training_seconds') or 0:.1f}s -> {t(rec).get('training_seconds') or 0:.1f}s, "
              f"eval {t(recs[0]).get('evaluation_seconds') or 0:.1f}s -> {t(rec).get('evaluation_seconds') or 0:.1f}s, "
              f"wall {recs[0].get('wall_seconds') or 0:.0f}s -> {rec.get('wall_seconds') or 0:.0f}s")


def _seeds(spec: str) -> list:
    if spec.isdigit() and int(spec) <= len(SEED_ORDER) and "," not in spec:
        return list(SEED_ORDER[:int(spec)])
    return [int(x) for x in spec.split(",") if x.strip()]


def main(argv: list) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["official", "seeds", "plan", "fetch", "status", "peek", "resume",
                                                      "cost", "compare"])
    ap.add_argument("--file", help="the submission (official, seeds; a plan's default file)")
    ap.add_argument("--plan", help="plan JSON: a list of jobs (tools/README.md)")
    ap.add_argument("--out", help="records folder (default for official/seeds: <file's folder>/records)")
    ap.add_argument("--run", help="run tag (launch: default <time>-<label>; fetch/status/resume/cost: tag or prefix)")
    ap.add_argument("--name", help="official: the record name (default <file stem>_official); seeds: the name prefix")
    ap.add_argument("--task", default="h1")
    ap.add_argument("--set", default="{}", help='JSON {CONSTANT: python source}: rewrites `CONSTANT = ...` lines')
    ap.add_argument("--manifest", default="null", help='seeds: extra overrides, JSON {section: {key: value}}')
    ap.add_argument("--seeds", default="4", help="seeds: a count (first N of 74,1..7) or a list like 74,3")
    ap.add_argument("--hardware", choices=sorted(HARDWARE), help=f"seeds, and plan jobs that name none: where they run "
                    f"(default {DEFAULT_HARDWARE}, the fastest measured); official: always h100, the contest's hardware")
    ap.add_argument("--memory-gb", type=float, help=f"cap on a job's evaluator processes' summed RSS (default {MEMORY_GB})")
    ap.add_argument("--timeout", type=int, help="seconds before a job is killed (default 1.5 x budget per seed + 1800)")
    ap.add_argument("--cap", type=int, default=20, help="most unfinished calls (containers, GPUs) of this run at a time")
    ap.add_argument("--wait", action="store_true", help="fetch: poll until every record of the run(s) is in")
    ap.add_argument("--latest", type=int, default=0, help="status, peek: only the latest N runs")
    ap.add_argument("--lines", type=int, default=12, help="peek: log lines per job")
    ap.add_argument("--app-logs", type=int, default=0, help="peek: also read the run's app log for this many seconds")
    ap.add_argument("--dry", action="store_true", help="official/seeds/plan: print the jobs, launch nothing")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("records", nargs="*", help="compare: the reference record, then the others")
    a = ap.parse_args(argv)
    sets = json.loads(a.set)
    if a.mode in ("official", "seeds", "plan"):
        if a.mode == "plan":
            if not a.plan or not a.out:
                raise SystemExit("--mode plan needs --plan and --out")
            plan = _resolve(a.plan)
            jobs = [_job(j, a.file, a.hardware, a.memory_gb, a.timeout) for j in json.loads(plan.read_text())]
            label = plan.stem
        else:
            if not a.file:
                raise SystemExit(f"--mode {a.mode} needs --file")
            path = _resolve(a.file)
            if a.mode == "official":
                if a.task not in TIER:
                    raise SystemExit(f"--mode official runs a tier's official manifest; {a.task!r} has none (use seeds)")
                if a.hardware not in (None, "h100"):
                    raise SystemExit("--mode official runs on the H100 only, like the contest; other hardware: seeds")
                jobs = [_job({"name": a.name or f"{path.stem}_official", "task": a.task, "set": sets, "hardware": "h100"},
                             str(path), None, a.memory_gb, a.timeout)]
                if jobs[0]["overrides"] is not None:
                    raise SystemExit("internal: an official job must run the untouched manifest")
            else:
                hw, prefix = a.hardware or DEFAULT_HARDWARE, a.name or path.stem
                jobs = [_job({"name": f"{prefix}_{hw}_seed{s}", "task": a.task, "set": sets, "seed": s, "hardware": hw,
                              "manifest": json.loads(a.manifest)}, str(path), None, a.memory_gb, a.timeout)
                        for s in _seeds(a.seeds)]
            label = jobs[0]["name"] if a.mode == "official" else f"{a.name or path.stem}-{a.hardware or DEFAULT_HARDWARE}"
        out = _out_dir(a.out) if a.out else _resolve(a.file).parent / "records"
        tag = a.run or f"{time.strftime('%Y%m%d-%H%M%S')}-{re.sub(r'[^A-Za-z0-9_.-]+', '-', label)[:40]}"
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,63}", tag):
            raise SystemExit(f"run tag {tag!r}: use 1-63 letters, digits, '-', '_' or '.'")
        for j in jobs:
            if j["task"] in TIER:
                desc = "official file, untouched" if j["overrides"] is None else json.dumps(j["overrides"])
            else:
                desc = f"H1's, data_root litmus_{j['task']}" + (f" + {json.dumps(j['overrides'])}" if j["overrides"] else "")
            print(f"job {j['name']}: {j['file']} md5 {j['submission_md5'][:8]} task {j['task']} on {j['hardware']}, "
                  f"manifest {desc}; limits {j['memory_gb']:g} GiB, {j['timeout_s'] or 'budget-based'} s")
        if a.dry:
            print(f"(dry) run {tag} -> {_rel(out)}, cap {a.cap}: nothing launched")
            return
        _launch(jobs, out, a.cap, tag, label, a.force)
    elif a.mode == "fetch":
        sel = _select(a.run, a.out if not a.run else None)
        if not sel:
            raise SystemExit("no runs selected (give --run or the --out a run was launched with)")
        _fetch(sel, a.out, a.wait, a.force)
    elif a.mode == "status":
        sel = _select(a.run, a.out, a.latest or (0 if a.run or a.out else 10))
        _status(sel)
    elif a.mode == "peek":
        sel = _select(a.run, a.out, a.latest or (0 if a.run or a.out else 3))
        _peek(sel, a.lines, a.app_logs)
    elif a.mode == "resume":
        if not a.run:
            raise SystemExit("--mode resume needs --run")
        _resume(a.run, a.force)
    elif a.mode == "cost":
        _cost(_select(a.run, a.out))
    elif a.mode == "compare":
        if len(a.records) < 2:
            raise SystemExit("--mode compare needs a reference record and at least one other")
        _compare(a.records)


@app.local_entrypoint()
def entry(*argv: str) -> None:
    """`modal run --detach cpu_first/tools/modal_runner.py --mode ...`, the tips-for-running-modal launch form, with the
    same arguments as `python cpu_first/tools/modal_runner.py --mode ...`."""
    import click
    ctx = click.get_current_context(silent=True)
    _MODAL_RUN.update(active=True, detached=bool(ctx is not None and (ctx.obj or {}).get("detach")))
    main(list(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
