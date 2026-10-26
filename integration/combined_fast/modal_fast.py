"""This folder's tiny cores and profiles on Modal, as combined_sp2/modal_c.py runs core_c.py: a script of this folder
(SCRIPT: core_c.py, the tiny core, or prof_c.py, the profile) run with env vars on one GPU container, its output
streamed to the app log, its log and JSON record saved to the Volume `combined-fast` (runs/<TAG>.log, .json, _prof.json)
every minute. Limits: the container's timeout and hard memory limit, the script's own SECONDS (a wall-clock guard below
the timeout), and the script and its children killed with the call. The official runs go through the shared runner.

    cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach \
        integration/combined_fast/modal_fast.py --envs '[{"TAG": "prof_base", "SCRIPT": "prof_c.py"}]'
    ~/github/sotaku/venv/bin/python -m modal run integration/combined_fast/modal_fast.py --mode fetch
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
TIMEOUT_S = 3 * 3600                # one container at most (the script's SECONDS is set below this)
MEMORY = (16384, 65536)             # request, hard limit (MiB)

image = (modal.Image.debian_slim(python_version="3.13").apt_install("gcc")      # (gcc: Triton builds its launcher)
         .pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "learned_remainder" / "chain_local" / "stub",
                        "/root/cpu_first/learned_remainder/chain_local/stub", ignore=["__pycache__"])
         .add_local_dir(HERE, "/root/cpu_first/integration/combined_fast",        # the scripts, the files, research builds
                        ignore=["records", "checks/logs", "__pycache__", "*.md"])
         .add_local_file(HERE.parent / "combined_sp2" / "records" / "core_combined2_measure.json",
                         "/root/cpu_first/integration/combined_sp2/records/core_combined2_measure.json")
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
OLD, REMOTE = CPU.parent / "one-layer-deeper", "/workspace/one-layer-benchmark"
official_image = (                   # the evaluator's environment, as tools/modal_runner.py builds it (its first layers:
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.13")   # torch 2.12.1, its Triton)
    .apt_install("git", "build-essential")
    .pip_install("uv==0.11.3")
    .env({"PYTHONPATH": REMOTE})
    .run_commands(f"mkdir -p {REMOTE}")
    .add_local_file(str(OLD / "pyproject.toml"), f"{REMOTE}/pyproject.toml", copy=True)
    .add_local_file(str(OLD / "uv.lock"), f"{REMOTE}/uv.lock", copy=True)
    .add_local_file(str(OLD / "submission_validation.py"), f"{REMOTE}/submission_validation.py", copy=True)
    .run_commands(f"cd {REMOTE} && uv sync --frozen --no-install-project")
    .add_local_dir(CPU / "learned_remainder" / "chain_local" / "stub",
                   "/root/cpu_first/learned_remainder/chain_local/stub", ignore=["__pycache__"])
    .add_local_dir(HERE, "/root/cpu_first/integration/combined_fast", ignore=["records", "checks/logs", "__pycache__",
                                                                              "*.md"])
    .add_local_file(HERE.parent / "combined_sp2" / "records" / "core_combined2_measure.json",
                    "/root/cpu_first/integration/combined_sp2/records/core_combined2_measure.json"))
app = modal.App("combined-fast", image=image)
vol = modal.Volume.from_name("combined-fast", create_if_missing=True)


def _run(env: dict, python: str = "python") -> str:
    import os
    import signal
    import subprocess
    tag = env.get("TAG", "run")
    Path("/out/runs").mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, H1_DATA=f"/root/data/{DATA}", OUT="/out/runs", PYTHONPATH="/root/cpu_first",
             **{k: str(v) for k, v in env.items()})
    e.setdefault("SECONDS", str(TIMEOUT_S - 1200))
    script = f"/root/cpu_first/integration/combined_fast/{env.get('SCRIPT', 'core_c.py')}"
    proc = subprocess.Popen([python, "-W", "ignore", "-u", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=e, cwd="/root/cpu_first", start_new_session=True)
    lines, last = [], time.time()
    try:
        for line in proc.stdout:                         # streamed to the container's log as it comes (watchable)
            print(line, end="", flush=True)
            lines.append(line)
            if time.time() - last > 60:                  # the log so far on the Volume every minute
                Path(f"/out/runs/{tag}.log").write_text("".join(lines[-20000:]))
                vol.commit()
                last = time.time()
        proc.wait()
    finally:                                             # the script and its children never outlive this call
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)      # (children such as the GPU sampler)
            except ProcessLookupError:
                pass
    out = "".join(lines[-20000:]) + f"\nEXIT {proc.returncode}\n"
    Path(f"/out/runs/{tag}.log").write_text(out)
    vol.commit()
    return out[-2000:]


@app.function(gpu="B200", timeout=TIMEOUT_S, memory=MEMORY, volumes={"/out": vol}, max_containers=4)
def run_b200(env: dict) -> str:
    return _run(env)


@app.function(gpu="H100!", timeout=TIMEOUT_S, memory=MEMORY, volumes={"/out": vol}, max_containers=2)
def run_h100(env: dict) -> str:
    return _run(env)


@app.function(gpu="H100!", image=official_image, timeout=3600, memory=MEMORY, volumes={"/out": vol}, max_containers=1)
def run_h100_official_env(env: dict) -> str:
    """A check script in the evaluator's own environment (its venv's python): no data, so checks only."""
    return _run(env, f"{REMOTE}/.venv/bin/python")


@app.local_entrypoint()
def main(mode: str = "run", envs: str = "[]") -> None:
    if mode == "fetch":
        dest = HERE / "records"
        dest.mkdir(parents=True, exist_ok=True)
        for entry in vol.listdir("runs"):
            with open(dest / Path(entry.path).name, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
            print("fetched", Path(entry.path).name)
        return
    for e in json.loads(envs):
        fn = {"h100": run_h100, "h100_official_env": run_h100_official_env}.get(e.get("GPU"), run_b200)
        call = fn.spawn(e)
        print("spawned", e.get("TAG"), "on", e.get("GPU", "b200"), call.object_id, time.strftime("%H:%M:%S"),
              flush=True)
