"""General-lessons runs (LESSONS.md): a submission, by default general_lessons/submission_general.py byte for byte,
through the unmodified One Layer Deeper evaluator on Modal, on H1 and on the litmus tasks.

A copy of litmus/modal_litmus.py with its own app (`general-lessons`), Volume (`general-lessons`, folder records/) and
records folder (cpu_first/general_lessons/records/); the image also generates tasks_general.py's held-out tasks
(tenrho, plus3, minus1). The image is the competition's, with every dataset of scripts/generate_datasets.sh
(Easy e1-e10, Medium m1-m10 and H1, by the official generator and its official arguments) and the litmus tasks
(tasks.py: H1's generator with another transition or tokenizer).

Tasks and manifests:
  * e1..e10, m1..m10, h1: the task's official manifest file, untouched when the job gives no overrides (`"manifest":
    null`); with overrides, a copy of it with those keys replaced (for example a relaxed training budget);
  * xfirst, mont, ten, rho, cube, xy: H1's manifest with data.data_root (and the name) replaced: research only.
A CPU job (`"gpu": false`) also sets runtime.device to cpu. Every run writes its record (RESULT_JSON, the full log,
the exact submission source, the manifest) to the Volume, so detached runs survive the client; wait_fetch_litmus.py
downloads them. Run from the one-layer-deeper repo root:

    modal run --detach ../cpu_first/litmus/modal_litmus.py --mode plan --plan ../cpu_first/litmus/plans/P.json
    modal run ../cpu_first/litmus/modal_litmus.py --mode audit          # data facts of every task -> litmus/audit.json
    modal run ../cpu_first/litmus/modal_litmus.py --mode agreement      # share of answers equal to H1's -> agreement.json
    modal run ../cpu_first/litmus/modal_litmus.py --mode fetch          # Volume -> litmus/records

A plan is a JSON list of {"name", "task", "file" (relative to cpu_first/, default --file), "set": {CONSTANT: python
source} (rewrites `CONSTANT = ...` lines of the submission), "manifest": {section: {key: value}} or null, "gpu"}.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECORDS = HERE / "records"  # general_lessons/records
REMOTE = "/workspace/one-layer-benchmark"
OFFICIAL = "benchmark/manifests/h100_hard_h1.json"
TIER = {**{f"e{i}": f"benchmark/manifests/h100_easy_e{i}.json" for i in range(1, 11)},
        **{f"m{i}": f"benchmark/manifests/h100_medium_m{i}.json" for i in range(1, 11)}, "h1": OFFICIAL}
LITMUS = ("xfirst", "mont", "ten", "rho", "cube", "xy", "tenrho", "plus3", "minus1")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.13")
    .apt_install("git", "build-essential")
    .pip_install("uv==0.11.3")
    .env({"PYTHONPATH": REMOTE})
    .run_commands(f"mkdir -p {REMOTE}")
    .add_local_file("pyproject.toml", f"{REMOTE}/pyproject.toml", copy=True)
    .add_local_file("uv.lock", f"{REMOTE}/uv.lock", copy=True)
    .add_local_file("submission_validation.py", f"{REMOTE}/submission_validation.py", copy=True)
    .run_commands(f"cd {REMOTE} && uv sync --frozen --no-install-project")
    .add_local_dir("benchmark", f"{REMOTE}/benchmark", copy=True, ignore=["__pycache__"])
    .add_local_dir("data", f"{REMOTE}/data", copy=True, ignore=["__pycache__", "generated"])
    .add_local_file("scripts/generate_datasets.sh", f"{REMOTE}/scripts/generate_datasets.sh", copy=True)
    .run_commands(f"cd {REMOTE} && PATH={REMOTE}/.venv/bin:$PATH bash scripts/generate_datasets.sh")
    .add_local_file(str(ROOT / "litmus" / "tasks.py"), f"{REMOTE}/litmus_tasks.py", copy=True)
    .add_local_file(str(HERE / "tasks_general.py"), f"{REMOTE}/tasks_general.py", copy=True)
    .run_commands(f"cd {REMOTE} && .venv/bin/python -m litmus_tasks data/generated")
    .run_commands(f"cd {REMOTE} && .venv/bin/python -m tasks_general data/generated")
    .add_local_file(str(ROOT / "litmus" / "audit_data.py"), f"{REMOTE}/litmus_audit.py", copy=True)
    .add_local_file(str(HERE / "mine_preview.py"), f"{REMOTE}/mine_preview.py", copy=True)
    .add_local_file(str(HERE / "code_search.py"), f"{REMOTE}/code_search.py", copy=True)
)
app = modal.App("general-lessons", image=image)
vol = modal.Volume.from_name("general-lessons", create_if_missing=True)


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


def _evaluate(name: str, task: str, src: str, overrides: dict | None) -> dict:
    work = Path(f"/tmp/{name}")
    work.mkdir(parents=True, exist_ok=True)
    (work / "submission.py").write_text(src)
    manifest = _manifest(task, overrides, work)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() if os.path.exists("/usr/bin/nvidia-smi") else ""
    t0 = time.time()
    log, err = work / "stdout.log", work / "stderr.log"      # live: modal container exec <id> -- tail /tmp/<name>/stdout.log
    with open(log, "w") as fo, open(err, "w") as fe:
        proc = subprocess.run([f"{REMOTE}/.venv/bin/python", "-m", "benchmark.runner", "--manifest", manifest,
                               "--submission-file", str(work / "submission.py"), "--include-structured-metrics"],
                              cwd=REMOTE, stdout=fo, stderr=fe, text=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    stdout, stderr = log.read_text(), err.read_text()
    result = next((json.loads(l.removeprefix("RESULT_JSON=")) for l in reversed(stdout.splitlines())
                   if l.startswith("RESULT_JSON=")), None)
    rec = {"name": name, "task": task, "gpu": gpu, "returncode": proc.returncode, "wall_seconds": time.time() - t0,
           "manifest_file": manifest if manifest.startswith("benchmark/") else None,
           "manifest": json.loads(Path(REMOTE, manifest).read_text()),
           "stdout": stdout, "stderr": stderr[-200000:], "result": result, "submission": src}
    Path("/out/records").mkdir(parents=True, exist_ok=True)
    Path(f"/out/records/{name}.json").write_text(json.dumps(rec, indent=1, default=str))
    vol.commit()
    return {"name": name, "returncode": proc.returncode}


@app.function(cpu=8, memory=16384, timeout=6 * 3600, volumes={"/out": vol}, max_containers=5)
def run_cpu(name: str, task: str, src: str, overrides: dict | None) -> dict:
    return _evaluate(name, task, src, overrides)


@app.function(gpu="H100", timeout=4 * 3600, volumes={"/out": vol}, max_containers=3)
def run_h100(name: str, task: str, src: str, overrides: dict | None) -> dict:
    return _evaluate(name, task, src, overrides)


@app.function(cpu=1, memory=1024, timeout=12 * 3600, volumes={"/out": vol})
def driver(cpu_jobs: list, gpu_jobs: list, tag: str = "") -> list:
    """Holds a plan's calls remotely, so the plan survives the local client (one spawn per detached launch). The CPU
    and the H100 calls are all spawned at once; each function's max_containers queues the rest. A preempted driver is
    restarted with the same input: it re-attaches to the calls it spawned (their ids kept on the Volume under the
    launch's tag) instead of spawning the whole plan again, which is what waves 3 and 4 did at 04:40."""
    ids = Path("/out/calls") / tag
    ids.mkdir(parents=True, exist_ok=True)
    calls = []
    for fn, jobs in ((run_cpu, cpu_jobs), (run_h100, gpu_jobs)):
        for j in jobs:
            f = ids / f"{j[0]}.id"
            if tag and f.exists():
                calls.append(modal.FunctionCall.from_id(f.read_text()))
                continue
            calls.append(fn.spawn(*j))
            f.write_text(calls[-1].object_id)
            vol.commit()
    out = []
    for c in calls:
        try:
            out.append(c.get())
        except Exception as exc:  # a failed run still leaves the others' records
            out.append(repr(exc)[:500])
    return out


@app.function(cpu=4, memory=16384, timeout=3600)
def audit() -> dict:
    """The data facts of every task (audit_data.py), from the generated files."""
    import importlib, sys
    sys.path.insert(0, REMOTE)
    mod = importlib.import_module("litmus_audit")
    roots = {t: str(Path(REMOTE, json.loads(Path(REMOTE, f).read_text())["data"]["data_root"])) for t, f in TIER.items()}
    roots.update({t: f"{REMOTE}/data/generated/litmus_{t}" for t in LITMUS})
    return {t: mod.facts(r) for t, r in roots.items()}


@app.function(cpu=4, memory=16384, timeout=3600)
def agreement() -> dict:
    """audit_data.h1_agreement on every litmus task (and H1, where it must be 1)."""
    import importlib, sys
    sys.path.insert(0, REMOTE)
    mod = importlib.import_module("litmus_audit")
    out = {t: mod.h1_agreement(f"{REMOTE}/data/generated/litmus_{t}") for t in LITMUS}
    out["h1"] = mod.h1_agreement(str(Path(REMOTE, json.loads(Path(REMOTE, OFFICIAL).read_text())["data"]["data_root"])))
    return out


@app.function(cpu=4, memory=16384, timeout=3600)
def preview() -> dict:
    """mine_preview.py on every task's training rows (the facts the miner's grammar finds with every field read right,
    and how many digit codes pass x < N)."""
    import importlib, sys
    sys.path.insert(0, REMOTE)
    mod = importlib.import_module("mine_preview")
    roots = {t: str(Path(REMOTE, json.loads(Path(REMOTE, f).read_text())["data"]["data_root"])) for t, f in TIER.items()}
    roots.update({t: f"{REMOTE}/data/generated/litmus_{t}" for t in LITMUS})
    return {t: mod.preview(r) for t, r in roots.items()}


@app.function(cpu=8, memory=16384, timeout=3600)
def codes() -> dict:
    """code_search.py on the 14 fixed-modulus tiers: the codes passing x < N, and those also fitting the answers."""
    out = {}
    for t, f in TIER.items():
        root = str(Path(REMOTE, json.loads(Path(REMOTE, f).read_text())["data"]["data_root"]))
        p = subprocess.run([f"{REMOTE}/.venv/bin/python", f"{REMOTE}/code_search.py", root], capture_output=True, text=True)
        r = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"error": p.stderr[-2000:]}
        if r.get("moduli") == 1 or "error" in r:
            out[t] = r
    return out


def _variant(src: str, sets: dict) -> str:
    for key, value in sets.items():
        src, count = re.subn(rf"^{key} = .*$", f"{key} = {value}", src, count=1, flags=re.M)
        if count != 1:
            raise KeyError(key)
    return src


@app.local_entrypoint()
def main(mode: str = "plan", plan: str = "", file: str = "general_lessons/submission_general.py") -> None:
    RECORDS.mkdir(parents=True, exist_ok=True)
    if mode == "fetch":
        for entry in vol.listdir("records"):
            dest = RECORDS / Path(entry.path).name
            if not dest.exists() or dest.stat().st_size != entry.size:
                with open(dest, "wb") as f:
                    for chunk in vol.read_file(entry.path):
                        f.write(chunk)
                print("fetched", dest.name, flush=True)
        return
    if mode == "agreement":
        out = agreement.remote()
        (HERE / "agreement.json").write_text(json.dumps(out, indent=1, sort_keys=True))
        for t, f in out.items():
            print(t, {k: v for k, v in f.items() if not k.endswith("=")}, flush=True)
        return
    if mode == "codes":
        out = codes.remote()
        (HERE / "codes.json").write_text(json.dumps(out, indent=1, sort_keys=True))
        for t, f in out.items():
            print(t, f, flush=True)
        return
    if mode == "preview":
        out = preview.remote()
        (HERE / "preview.json").write_text(json.dumps(out, indent=1, sort_keys=True))
        for t, f in out.items():
            print(t, f["moduli"], f["codes"], f["codes_n_reversed"], f["facts"], flush=True)
        return
    if mode == "audit":
        out = audit.remote()
        (HERE / "audit.json").write_text(json.dumps(out, indent=1, sort_keys=True))
        for t, f in out.items():
            print(t, {k: v for k, v in f.items() if not isinstance(v, (list, dict))}, flush=True)
        return
    jobs = json.loads(Path(plan).read_text())
    cpu_jobs, gpu_jobs = [], []
    for j in jobs:
        src = _variant((ROOT / j.get("file", file)).read_text(), j.get("set", {}))
        over = j.get("manifest")
        if not j.get("gpu"):
            over = dict(over or {})
            over["runtime"] = dict(over.get("runtime", {}), device="cpu")
        (gpu_jobs if j.get("gpu") else cpu_jobs).append((j["name"], j["task"], src, over))
    call = driver.spawn(cpu_jobs, gpu_jobs, time.strftime("%Y%m%d-%H%M%S"))
    print("spawned plan", [j["name"] for j in jobs], call.object_id, flush=True)
