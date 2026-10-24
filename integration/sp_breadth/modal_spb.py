"""Offline research runs of this study (SP_BREADTH.md) on Modal, as integration/self_practice/modal_sp.py runs its
own: a module (integration/sp_breadth/offline.py by default) with env vars on a GPU or CPU container, its output
streamed to the app log, its log saved to the Volume `sp-breadth` (runs/<TAG>.log) every minute. Submission files go
through the shared runner (tools/modal_runner.py). Limits: the container's timeout and hard memory limit here, the
module's process group killed with the call.

    cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach \
        integration/sp_breadth/modal_spb.py --envs '[{"TAG": "off_g1", "LIB": "integration/sp_breadth/libs/h1_l4.json"}]'
    ~/github/sotaku/venv/bin/python -m modal run integration/sp_breadth/modal_spb.py --mode fetch
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
TIMEOUT_S = 3 * 3600                # one container at most
MEMORY = (16384, 65536)             # request, hard limit (MiB)

image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "library", "/root/cpu_first/library", ignore=["records", "inventing", "__pycache__"])
         .add_local_dir(CPU / "anchors", "/root/cpu_first/anchors", ignore=["records", "__pycache__", "*.md"])
         .add_local_file(CPU / "integration" / "__init__.py", "/root/cpu_first/integration/__init__.py")
         .add_local_dir(CPU / "integration" / "lib", "/root/cpu_first/integration/lib",
                        ignore=["records", "__pycache__", "logs", "*.out", "*.json", "*.txt"])
         .add_local_dir(CPU / "integration" / "self_practice", "/root/cpu_first/integration/self_practice",
                        ignore=["records", "__pycache__", "*.md", "submission_*.py"])
         .add_local_dir(HERE, "/root/cpu_first/integration/sp_breadth", ignore=["records", "__pycache__", "*.md",
                                                                                "submission_*.py"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("sp-breadth", image=image)
vol = modal.Volume.from_name("sp-breadth", create_if_missing=True)


def _run(env: dict) -> str:
    import os
    import signal
    import subprocess
    tag = env.get("TAG", "run")
    Path("/out/runs").mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, H1_DATA=f"/root/data/{DATA}", PYTHONPATH="/root/cpu_first", OUT="/out/runs",
             **{k: str(v) for k, v in env.items()})
    e.setdefault("SECONDS", str(TIMEOUT_S - 1800))
    proc = subprocess.Popen(["python", "-W", "ignore", "-u", "-m", env.get("MODULE", "integration.sp_breadth.offline")],
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


@app.function(gpu="B200", timeout=TIMEOUT_S, memory=MEMORY, volumes={"/out": vol}, max_containers=8)
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
        dest = HERE / "offline"
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
