"""A tournament H1 submission (cpu_first/history/tournament/submission_*.py, --file) through the unmodified One Layer
Deeper evaluator on Modal.

A copy of bridge/h1/modal_h1.py (itself a copy of wider_menus/modal_wide.py) with its own app (`tournament-h1`),
Volume (`tournament-h1`, folder records/) and records folder (cpu_first/history/tournament/h1_records/). Same competition image (H1 is regenerated inside it with
the official generator arguments, so its build is cached). Every run writes its record (RESULT_JSON, the full log, the
exact submission source, the manifest) to the Volume, so detached runs survive the client and wait_fetch.py fetches
them. Run from the one-layer-deeper repo root:

    modal run --detach /path/to/modal_wiring.py --file submission_div.py --mode official --name div_official
    modal run --detach /path/to/modal_wiring.py --file submission_div.py --mode official --name div_s0 \\
        --set '{"MODE": "\\"designed\\""}'
    modal run --detach /path/to/modal_wiring.py --file submission_div.py --mode plan --plan p.json
    modal run /path/to/modal_wiring.py --mode fetch                                    # Volume -> runs/learned_wiring

`--set` rewrites `CONSTANT = ...` lines of the submission (JSON {CONSTANT: python source}); with --mode official the
manifest file stays untouched, so a set run is the official evaluation of that exact source (saved in the record).
A plan is a JSON list of {"name", "set": {...}, "manifest": {section: {key: value}} or null (null = the official
manifest file, untouched), "gpu": true/false}.
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
RUNS = HERE / "h1_records"
REMOTE = "/workspace/one-layer-benchmark"
H1_DIR = "data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
H1_ARGS = (
    f"--output_dir {H1_DIR} --modulus_bits '[13,16,20]' --time_steps '[4,8,16]' --ood_time_steps '[32]' "
    "--ood_examples_per_setting 3334 --examples_per_setting 12222 "
    "--extra_train_examples_by_time_step '{\"1\":1000,\"2\":1000}' "
    "--depth_evaluation_time_steps '[1,2,4,8,16,32,64]' --depth_evaluation_examples_per_setting 256 "
    "--ood_n_depth_evaluation_modulus_bits '[14,18,22,24]' --ood_n_depth_evaluation_examples_per_setting 192 "
    "--train_fraction 0.9090909090909091 --test_fraction 0.09090909090909091 "
    "--split_group modulus --factor_modulus 4 --factor_remainder 3 "
    "--separate_ood_splits true --seed 45 --separate_input_output true"
)
OFFICIAL = "benchmark/manifests/h100_hard_h1.json"

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
    .run_commands(f"cd {REMOTE} && .venv/bin/python -m data.reversed_n_squaring_mod {H1_ARGS}")
)
app = modal.App("tournament-h1", image=image)
vol = modal.Volume.from_name("tournament-h1", create_if_missing=True)


def _evaluate(name: str, src: str, overrides: dict | None) -> dict:
    work = Path(f"/tmp/{name}")
    work.mkdir(parents=True, exist_ok=True)
    (work / "submission.py").write_text(src)
    if overrides is None:
        manifest = OFFICIAL
    else:
        m = json.loads(Path(REMOTE, OFFICIAL).read_text())
        for section, values in overrides.items():
            m[section].update(values)
        manifest = str(work / "manifest.json")
        Path(manifest).write_text(json.dumps(m, indent=1))
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
    rec = {"name": name, "gpu": gpu, "returncode": proc.returncode, "wall_seconds": time.time() - t0,
           "manifest": manifest if overrides is None else json.loads(Path(manifest).read_text()),
           "stdout": stdout, "stderr": stderr[-200000:], "result": result, "submission": src}
    Path("/out/records").mkdir(parents=True, exist_ok=True)
    Path(f"/out/records/{name}.json").write_text(json.dumps(rec, indent=1, default=str))
    vol.commit()
    return {"name": name, "returncode": proc.returncode}


@app.function(cpu=8, memory=16384, timeout=4 * 3600, volumes={"/out": vol}, max_containers=8)
def run_cpu(name: str, src: str, overrides: dict | None) -> dict:
    return _evaluate(name, src, overrides)


@app.function(gpu="H100", timeout=4 * 3600, volumes={"/out": vol}, max_containers=3)
def run_h100(name: str, src: str, overrides: dict | None) -> dict:
    return _evaluate(name, src, overrides)


@app.function(cpu=1, memory=1024, timeout=8 * 3600)
def driver(cpu_jobs: list, gpu_jobs: list) -> list:
    """Holds a plan's calls remotely, so the plan survives the local client (one spawn per detached launch)."""
    out = []
    if cpu_jobs:
        out += list(run_cpu.starmap(cpu_jobs, order_outputs=False, return_exceptions=True))
    if gpu_jobs:
        out += list(run_h100.starmap(gpu_jobs, order_outputs=False, return_exceptions=True))
    return [o if isinstance(o, dict) else repr(o)[:500] for o in out]


def _variant(src: str, sets: dict) -> str:
    for key, value in sets.items():
        src, count = re.subn(rf"^{key} = .*$", f"{key} = {value}", src, count=1, flags=re.M)
        if count != 1:
            raise KeyError(key)
    return src


@app.local_entrypoint()
def main(mode: str = "official", plan: str = "", name: str = "official_h100", set: str = "{}",
         file: str = "submission_tournament.py") -> None:
    src = (HERE / file).read_text()
    RUNS.mkdir(parents=True, exist_ok=True)
    if mode == "fetch":
        for entry in vol.listdir("records"):
            dest = RUNS / Path(entry.path).name
            if not dest.exists() or dest.stat().st_size != entry.size:
                with open(dest, "wb") as f:
                    for chunk in vol.read_file(entry.path):
                        f.write(chunk)
                print("fetched", dest.name, flush=True)
        return
    if mode == "official":
        call = run_h100.spawn(name, _variant(src, json.loads(set)), None)
        print("spawned", name, call.object_id, flush=True)
        return
    jobs = json.loads(Path(plan).read_text())
    cpu_jobs = [(j["name"], _variant(src, j.get("set", {})), j.get("manifest", {"runtime": {"device": "cpu"}}))
                for j in jobs if not j.get("gpu")]
    gpu_jobs = [(j["name"], _variant(src, j.get("set", {})), j.get("manifest")) for j in jobs if j.get("gpu")]
    call = driver.spawn(cpu_jobs, gpu_jobs)
    print("spawned plan", [j["name"] for j in jobs], call.object_id, flush=True)
