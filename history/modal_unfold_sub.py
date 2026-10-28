"""cpu_first/history/unfold_sub/submission.py (UnfoldB) through the unmodified One Layer Deeper evaluator on Modal.

Run from the one-layer-deeper repo root. The image is modal_h100.py's (CUDA 12.9 base, Python 3.13, `uv sync --frozen`,
H1 generated inside the image), so its build is cached:

    modal run /path/to/cpu_first/history/modal_unfold_sub.py --mode official            # h100_hard_h1.json untouched, H100
    modal run /path/to/cpu_first/history/modal_unfold_sub.py --mode plan --plan p.json  # variants, one container each
    modal run /path/to/cpu_first/history/modal_unfold_sub.py --mode check               # closed-form Jacobian vs autograd

A plan is a JSON list of {"name", "set": {CONSTANT: python source}, "manifest": {section: {key: value}} or null
(null = the official manifest file, untouched), "gpu": true/false}. "set" rewrites `CONSTANT = ...` lines of the
submission. Results: cpu_first/history/runs/unfold_sub/<name>.json (log tail, RESULT_JSON, the exact submission source).
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
app = modal.App("unfold-sub", image=image)


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
    proc = subprocess.run([f"{REMOTE}/.venv/bin/python", "-m", "benchmark.runner", "--manifest", manifest,
                           "--submission-file", str(work / "submission.py"), "--include-structured-metrics"],
                          cwd=REMOTE, capture_output=True, text=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    out = proc.stdout + proc.stderr
    result = next((json.loads(l.removeprefix("RESULT_JSON=")) for l in reversed(proc.stdout.splitlines())
                   if l.startswith("RESULT_JSON=")), None)
    return {"name": name, "gpu": gpu, "returncode": proc.returncode, "wall_seconds": time.time() - t0,
            "manifest": manifest if overrides is None else json.loads(Path(manifest).read_text()),
            "log": out[-60000:], "result": result, "submission": src}


@app.function(cpu=8, memory=16384, timeout=3 * 3600)
def run_cpu(name: str, src: str, overrides: dict | None) -> dict:
    return _evaluate(name, src, overrides)


@app.function(gpu="H100", timeout=3 * 3600)
def run_h100(name: str, src: str, overrides: dict | None) -> dict:
    return _evaluate(name, src, overrides)


@app.function(cpu=4, memory=16384, timeout=1800)
def check(src: str, script: str) -> str:
    """unfold_sub/check_grad.py: closed-form residual Jacobian and the scaled-gradient channel vs autograd."""
    Path("/tmp/check").mkdir(exist_ok=True)
    Path("/tmp/check/submission.py").write_text(src)
    Path("/tmp/check/check_grad.py").write_text(script)
    proc = subprocess.run([f"{REMOTE}/.venv/bin/python", "/tmp/check/check_grad.py", "/tmp/check/submission.py"],
                          cwd=REMOTE, capture_output=True, text=True)
    return proc.stdout + proc.stderr[-5000:]


@app.function(cpu=2, memory=4096, timeout=3600)
def mini(src: str, script: str, grid: list) -> str:
    """unfold_sub/mini.py: stage dynamics on the 1,000 T=1 rows, without the evaluator (one grid slice)."""
    Path("/tmp/mini").mkdir(exist_ok=True)
    Path("/tmp/mini/submission.py").write_text(src)
    Path("/tmp/mini/mini.py").write_text(script)
    proc = subprocess.run([f"{REMOTE}/.venv/bin/python", "/tmp/mini/mini.py", "/tmp/mini/submission.py",
                           json.dumps(grid)], cwd=REMOTE, capture_output=True, text=True,
                          env=dict(os.environ, OMP_NUM_THREADS="2"))
    return proc.stdout + proc.stderr[-3000:]


def _variant(src: str, sets: dict) -> str:
    for key, value in sets.items():
        src, count = re.subn(rf"^{key} = .*$", f"{key} = {value}", src, count=1, flags=re.M)
        if count != 1:
            raise KeyError(key)
    return src


@app.local_entrypoint()
def main(mode: str = "plan", plan: str = "", name: str = "official_h100") -> None:
    src = (HERE / "unfold_sub" / "submission.py").read_text()
    out_dir = HERE / "runs" / "unfold_sub"
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "check":
        print(check.remote(src, (HERE / "unfold_sub" / "check_grad.py").read_text()))
        return
    if mode == "mini":   # plan = a JSON grid for mini.py; one container per 6 runs
        grid = json.loads(Path(plan).read_text())
        script = (HERE / "unfold_sub" / "mini.py").read_text()
        chunks = [grid[i:i + 6] for i in range(0, len(grid), 6)]
        lines = []
        for out in mini.map([src] * len(chunks), [script] * len(chunks), chunks, order_outputs=False):
            print(out.strip(), flush=True)
            lines += [l for l in out.splitlines() if l.startswith("{")]
        (out_dir / (Path(plan).stem + ".jsonl")).write_text("\n".join(lines) + "\n")
        return
    if mode == "official":
        jobs = [{"name": name, "set": {}, "manifest": None, "gpu": True}]
    else:
        jobs = json.loads(Path(plan).read_text())
    cpu_jobs = [(j["name"], _variant(src, j.get("set", {})), j.get("manifest", {"runtime": {"device": "cpu"}}))
                for j in jobs if not j.get("gpu")]
    gpu_jobs = [(j["name"], _variant(src, j.get("set", {})), j.get("manifest"))
                for j in jobs if j.get("gpu")]
    calls = []
    if cpu_jobs:
        calls.append(run_cpu.starmap(cpu_jobs, order_outputs=False, return_exceptions=True))
    if gpu_jobs:
        calls.append(run_h100.starmap(gpu_jobs, order_outputs=False, return_exceptions=True))
    for results in calls:
        for res in results:
            if isinstance(res, Exception):
                print("failed:", repr(res)[:500], flush=True)
                continue
            (out_dir / f"{res['name']}.json").write_text(json.dumps(res, indent=1, default=str))
            r = res["result"] or {}
            dp = r.get("depth_profile", {})
            seed = (r.get("seeds") or [{}])[0]
            ev = {k: round(v["exact_accuracy"], 4) for k, v in seed.get("evaluation", {}).items()}
            print(f"done {res['name']} rc={res['returncode']} wall={res['wall_seconds']:.0f}s {res['gpu']} "
                  f"maxT={dp.get('max_certified_time_steps')} oodN_maxT={dp.get('ood_n_max_certified_time_steps')} "
                  f"train_s={seed.get('training_seconds', 0):.0f} steps={seed.get('completed_training_steps')} "
                  f"eval={ev}", flush=True)
            if res["returncode"] != 0:
                print(res["log"][-3000:], flush=True)
