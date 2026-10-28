"""The ablation runs (cpu_first/history/ablation/ABLATION.md) through the unmodified One Layer Deeper evaluator on
Modal.

A copy of cpu_first/learned_remainder/modal_chain.py (the runner every certified piece used: same competition image,
H1 regenerated inside it with the official generator arguments, the same `benchmark.runner` call and CPU/H100
containers), with three changes:
  * its own app `ablation` and Volume `ablation`;
  * it is deployed once, so every plan's calls share one container cap: `run_cpu` at most 20 containers (the
    workspace's 100 are shared with other sessions), `run_h100` at most 2;
  * jobs are spawned on the deployed functions by launch.py, and fetch.py downloads the records.
Deploy from the one-layer-deeper repo root (the image is built from it):

    ~/github/sotaku/venv/bin/python -m modal deploy /path/to/cpu_first/history/ablation/modal_ablate.py

Every run writes its record to the Volume: RESULT_JSON, the full log, the exact submission source, the manifest
(the path of the untouched official manifest when no overrides are given) and the job's metadata.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

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
app = modal.App("ablation", image=image)
vol = modal.Volume.from_name("ablation", create_if_missing=True)


def _evaluate(name: str, src: str, overrides: dict | None, meta: dict | None) -> dict:
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
           "meta": meta or {}, "stdout": stdout, "stderr": stderr[-200000:], "result": result, "submission": src}
    Path("/out").mkdir(exist_ok=True)
    Path(f"/out/{name}.json").write_text(json.dumps(rec, indent=1, default=str))
    vol.commit()
    return {"name": name, "returncode": proc.returncode}


@app.function(cpu=8, memory=16384, timeout=4 * 3600, volumes={"/out": vol}, max_containers=20)
def run_cpu(name: str, src: str, overrides: dict | None, meta: dict | None = None) -> dict:
    return _evaluate(name, src, overrides, meta)


@app.function(gpu="H100", timeout=4 * 3600, volumes={"/out": vol}, max_containers=2)
def run_h100(name: str, src: str, overrides: dict | None, meta: dict | None = None) -> dict:
    return _evaluate(name, src, overrides, meta)
