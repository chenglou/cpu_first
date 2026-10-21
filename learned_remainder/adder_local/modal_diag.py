"""Run adder_local/local_adder.py variants in parallel on Modal CPU containers (the same cached competition image as
modal_chain.py, whose build regenerates H1). Run from the one-layer-deeper repo root:

    modal run /path/to/adder_local/modal_diag.py --plan plan.json

A plan is a JSON list of {"name": ..., "argv": ["train", "1500", "64", "1", "K=V", ...], "env": {...}, "gpu": bool}
(env: harness-only diagnostics such as FIX=b; gpu: an L4 instead of a CPU container). Each job's stdout lands in
adder_local/logs/<name>.log when the call returns (the local client waits: launch it as a background command).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LR = HERE.parent
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
app = modal.App("learned-adder-diag", image=image)

FILES = ["submission_adder.py", "submission_chain.py", "adder_local/local_adder.py", "adder_local/landscape.py",
         "chain_local/stub/benchmark/__init__.py"]


def _diag(name: str, files: dict, argv: list, env_extra: dict) -> tuple:
    work = Path(f"/tmp/{name}/learned_remainder")
    for rel, src in files.items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text(src)
    env = dict(os.environ, H1_DATA=f"{REMOTE}/{H1_DIR}", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", **env_extra)
    proc = subprocess.run([f"{REMOTE}/.venv/bin/python", "-W", "ignore", str(work / "adder_local/local_adder.py"), *argv],
                          cwd=str(work / "adder_local"), capture_output=True, text=True, env=env)
    return name, proc.stdout + ("\n--- stderr ---\n" + proc.stderr[-20000:] if proc.returncode else "")


@app.function(cpu=4, memory=8192, timeout=3 * 3600)
def diag(name: str, files: dict, argv: list, env_extra: dict) -> tuple:
    return _diag(name, files, argv, env_extra)


@app.function(gpu="L4", timeout=3 * 3600)
def diag_gpu(name: str, files: dict, argv: list, env_extra: dict) -> tuple:
    return _diag(name, files, argv, dict(env_extra, DEVICE="cuda"))


@app.local_entrypoint()
def main(plan: str) -> None:
    jobs = json.loads(Path(plan).read_text())
    files = {rel: (LR / rel).read_text() for rel in FILES}
    (HERE / "logs").mkdir(exist_ok=True)
    calls = [(j["name"], (diag_gpu if j.get("gpu") else diag).spawn(j["name"], files, j["argv"], j.get("env", {})))
             for j in jobs]
    while calls:                                            # collect in completion order
        pending = []
        for job, call in calls:
            try:
                name, out = call.get(timeout=0)
            except TimeoutError:
                pending.append((job, call))
                continue
            except Exception as exc:
                print("FAILED call:", job, repr(exc)[:2000], flush=True)
                continue
            (HERE / "logs" / f"{name}.log").write_text(out)
            tail = [l for l in out.splitlines() if l.startswith("step")][-1:] or out.splitlines()[-3:]
            print(name, "|", " ".join(tail)[:300], flush=True)
        calls = pending
        if calls:
            time.sleep(20)
