"""SpecBit through the unmodified H1 evaluator, and the speculation benchmarks, on a Modal H100.

Run from the one-layer-deeper repo root (the image mirrors the competition's own modal_runner.py: CUDA 12.9 base,
Python 3.13, `uv sync --frozen`, H1 generated inside the image):

    modal run /path/to/cpu_first/modal_h100.py --task h1        # benchmark/manifests/h100_hard_h1.json, untouched
    modal run /path/to/cpu_first/modal_h100.py --task bench     # spec_grammar.py --bench on cuda
    modal run /path/to/cpu_first/modal_h100.py --task search    # the staged grammar search + aliasing audit on cuda
    add --extra "--compile ABC" --tag _compiled to fuse the bulk passes with torch.compile

Results land in cpu_first/runs/. Long searches: `modal run --detach ... --task search --spawn --tag _x` returns at once;
the worker writes h100_search_x.{log,json} to the `specbit-runs` Volume (read it while it runs with
`modal container exec <id> -- tail /runs/h100_search_x.log`, fetch it with `modal volume get specbit-runs ...`).
"""

from __future__ import annotations

import json
import os
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
    .add_local_file(str(HERE / "submission.py"), "/root/cpu_first/submission.py")
    .add_local_file(str(HERE / "spec_grammar.py"), "/root/cpu_first/spec_grammar.py")
)
app = modal.App("specbit-h100", image=image)
runs_vol = modal.Volume.from_name("specbit-runs", create_if_missing=True)


def _run(cmd: list[str]) -> tuple[int, str, float]:
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REMOTE, capture_output=True, text=True,
                          env=dict(os.environ, PYTHONUNBUFFERED="1", H1_DATA=f"{REMOTE}/{H1_DIR}"))
    print(proc.stdout[-20000:], proc.stderr[-5000:], flush=True)
    return proc.returncode, proc.stdout + proc.stderr, time.time() - t0


@app.function(gpu="H100", timeout=3600)
def h1() -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    code, out, secs = _run([f"{REMOTE}/.venv/bin/python", "-m", "benchmark.runner",
                            "--manifest", "benchmark/manifests/h100_hard_h1.json",
                            "--submission-file", "/root/cpu_first/submission.py", "--include-structured-metrics"])
    result = next((json.loads(l.removeprefix("RESULT_JSON=")) for l in reversed(out.splitlines())
                   if l.startswith("RESULT_JSON=")), None)
    return {"gpu": gpu, "returncode": code, "wall_seconds": secs, "log": out[-30000:], "result": result}


@app.function(gpu="H100", cpu=16, timeout=4 * 3600, volumes={"/runs": runs_vol})  # cpu: Triton builds are CPU-bound
def grammar(args: list[str], name: str = "grammar") -> dict:
    """spec_grammar.py on cuda; its log and JSON also go to the specbit-runs Volume, so a detached run survives the
    client."""
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    log_path, out_path = f"/runs/{name}.log", f"/runs/{name}.out.json"
    t0 = time.time()
    with open(log_path, "w") as fh:
        code = subprocess.run([f"{REMOTE}/.venv/bin/python", "-u", "/root/cpu_first/spec_grammar.py", "--device", "cuda",
                               "--data", f"{REMOTE}/{H1_DIR}", "--out", out_path, *args], cwd=REMOTE, stdout=fh,
                              stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1")).returncode
    secs = time.time() - t0
    out = open(log_path).read()
    print(out[-20000:], flush=True)
    payload = json.load(open(out_path)) if os.path.exists(out_path) else None
    res = {"gpu": gpu, "returncode": code, "wall_seconds": secs, "log": out[-30000:], "out": payload}
    with open(f"/runs/{name}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    runs_vol.commit()
    return res


@app.local_entrypoint()
def main(task: str = "h1", chunk: int = 1 << 22, extra: str = "", tag: str = "", spawn: bool = False) -> None:
    runs = HERE / "runs"
    runs.mkdir(exist_ok=True)
    name = f"h100_{task}{tag}"
    if task == "h1":
        res = h1.remote()
    else:
        args = (["--bench"] if task == "bench" else []) + ["--chunk", str(chunk), *extra.split()]
        if spawn:
            call = grammar.spawn(args, name)
            print(f"{task}: spawned {call.object_id}; results -> Volume specbit-runs: {name}.json, {name}.log")
            return
        res = grammar.remote(args, name)
    path = runs / f"{name}.json"
    path.write_text(json.dumps(res, indent=1, default=str))
    print(f"{task}: returncode={res['returncode']} wall={res['wall_seconds']:.1f}s gpu={res['gpu']} -> {path}")
