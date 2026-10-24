"""Research runs of the self-practice study (SELF_PRACTICE.md) on Modal: a module of integration/self_practice/ (the
tiny core, core.py, by default) run with env vars on a GPU or CPU container, its output streamed to the app log, its
log and JSON record saved to the Volume `self-practice` (runs/<TAG>.log, runs/<TAG>.json) every minute. Submission
files go through the shared runner (tools/modal_runner.py) instead. Every run's limits are here: the container's
timeout and hard memory limit, and the module's own deadline (SECONDS) and per-search limit (SEARCH_S) in its env.

    cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach \
        integration/self_practice/modal_sp.py --envs '[{"TAG": "core_lp", "RULE": "lp"}]'
    ~/github/sotaku/venv/bin/python -m modal run integration/self_practice/modal_sp.py --mode fetch
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
TIMEOUT_S = 12 * 3600               # one container at most (the module's own SECONDS is set below this)
MEMORY = (16384, 65536)             # request, hard limit (MiB)

image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "library", "/root/cpu_first/library", ignore=["records", "inventing", "__pycache__"])
         .add_local_dir(CPU / "anchors", "/root/cpu_first/anchors", ignore=["records", "__pycache__", "*.md"])
         .add_local_file(CPU / "integration" / "__init__.py", "/root/cpu_first/integration/__init__.py")
         .add_local_dir(CPU / "integration" / "lib", "/root/cpu_first/integration/lib",
                        ignore=["records", "__pycache__", "logs", "*.out", "*.json", "*.txt"])
         .add_local_dir(HERE, "/root/cpu_first/integration/self_practice", ignore=["records", "__pycache__", "*.md"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("self-practice", image=image)
vol = modal.Volume.from_name("self-practice", create_if_missing=True)


def _run(env: dict) -> str:
    import os
    import signal
    import subprocess
    tag = env.get("TAG", "run")
    Path("/out/runs").mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, H1_DATA=f"/root/data/{DATA}", PYTHONPATH="/root/cpu_first", OUT="/out/runs",
             **{k: str(v) for k, v in env.items()})
    e.setdefault("SECONDS", str(TIMEOUT_S - 1800))
    proc = subprocess.Popen(["python", "-W", "ignore", "-u", "-m", env.get("MODULE", "integration.self_practice.core")],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e, cwd="/root/cpu_first",
                            start_new_session=True)
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
    finally:                                             # the module and its children never outlive this call
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    out = "".join(lines[-20000:]) + f"\nEXIT {proc.returncode}\n"
    Path(f"/out/runs/{tag}.log").write_text(out)
    vol.commit()
    return out[-2000:]


@app.function(gpu="B200", timeout=TIMEOUT_S, memory=MEMORY, volumes={"/out": vol}, max_containers=10)
def run_b200(env: dict) -> str:
    return _run(env)


@app.function(gpu="H100!", timeout=TIMEOUT_S, memory=MEMORY, volumes={"/out": vol}, max_containers=4)
def run_h100(env: dict) -> str:
    return _run(env)


@app.function(cpu=8.0, timeout=TIMEOUT_S, memory=MEMORY, volumes={"/out": vol}, max_containers=8)
def run_cpu(env: dict) -> str:
    return _run(env)


@app.local_entrypoint()
def main(mode: str = "run", envs: str = "[]", gpu: str = "B200") -> None:
    if mode == "fetch":
        dest = HERE / "records"
        dest.mkdir(parents=True, exist_ok=True)
        for entry in vol.listdir("runs"):
            with open(dest / Path(entry.path).name, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
            print("fetched", Path(entry.path).name)
        return
    fn = {"B200": run_b200, "H100": run_h100, "CPU": run_cpu}[gpu]
    for e in json.loads(envs):
        call = fn.spawn(e)
        print("spawned", e.get("TAG"), "on", gpu, call.object_id, time.strftime("%H:%M:%S"), flush=True)
