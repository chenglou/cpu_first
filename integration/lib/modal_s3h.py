"""Research runs for stage 3's s3h (INTEGRATION.md §3): a lib/ module run on a Modal GPU with env vars, its output
streamed to the app log and saved to the Volume `integration` (s3h/<TAG>.log); --mode fetch copies the logs to
integration/lib/records/s3h/. Same image as modal_ab.py (Modal's cache). Every run's limits are here.

    cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run --detach \
        ../cpu_first/integration/lib/modal_s3h.py --mode run --env '{"MODULE": "integration.lib.round_hash_diag"}'
    ~/github/sotaku/venv/bin/python -m modal run ../cpu_first/integration/lib/modal_s3h.py --mode fetch
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
TIMEOUT_S = 3 * 3600                 # one diagnostic's container at most
MEMORY_MB = 64 * 1024

image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "library", "/root/cpu_first/library", ignore=["records", "inventing", "__pycache__"])
         .add_local_dir(CPU / "anchors", "/root/cpu_first/anchors", ignore=["records", "__pycache__", "*.md"])
         .add_local_file(CPU / "integration" / "__init__.py", "/root/cpu_first/integration/__init__.py")
         .add_local_dir(HERE, "/root/cpu_first/integration/lib", ignore=["records", "__pycache__", "logs", "*.out"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("integration-s3h", image=image)
vol = modal.Volume.from_name("integration", create_if_missing=True)


def _run(env: dict) -> str:
    import os
    import subprocess
    e = dict(os.environ, H1_DATA=f"/root/data/{DATA}", PYTHONPATH="/root/cpu_first",
             **{k: str(v) for k, v in env.items()})
    proc = subprocess.Popen(["python", "-W", "ignore", "-u", "-m", env["MODULE"]], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=e, cwd="/root/cpu_first")
    lines = []
    tag = env.get("TAG", env["MODULE"].rsplit(".", 1)[-1])
    last = time.time()
    for line in proc.stdout:                             # streamed to the container's log as it comes (watchable)
        print(line, end="", flush=True)
        lines.append(line)
        if time.time() - last > 60:                      # the log so far on the Volume every minute
            Path("/out/s3h").mkdir(parents=True, exist_ok=True)
            Path(f"/out/s3h/{tag}.log").write_text("".join(lines))
            vol.commit()
            last = time.time()
    proc.wait()
    out = "".join(lines) + f"\nEXIT {proc.returncode}\n"
    Path("/out/s3h").mkdir(parents=True, exist_ok=True)
    Path(f"/out/s3h/{tag}.log").write_text(out)
    vol.commit()
    return out


@app.function(gpu="B200", timeout=TIMEOUT_S, memory=(16384, MEMORY_MB), volumes={"/out": vol}, max_containers=12)
def run_b200(env: dict) -> str:
    return _run(env)


@app.function(gpu="H100!", timeout=TIMEOUT_S, memory=(16384, MEMORY_MB), volumes={"/out": vol}, max_containers=4)
def run_h100(env: dict) -> str:
    return _run(env)


@app.function(cpu=8.0, timeout=TIMEOUT_S, memory=(16384, MEMORY_MB), volumes={"/out": vol}, max_containers=8)
def run_cpu(env: dict) -> str:
    return _run(env)


@app.local_entrypoint()
def main(mode: str = "run", env: str = "{}", envs: str = "", gpu: str = "B200") -> None:
    if mode == "fetch":
        dest_dir = HERE / "records" / "s3h"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for entry in vol.listdir("s3h"):
            dest = dest_dir / Path(entry.path).name
            with open(dest, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
            print("fetched", dest.name)
        return
    fn = {"B200": run_b200, "H100": run_h100, "CPU": run_cpu}[gpu]
    for e in (json.loads(envs) if envs else [json.loads(env)]):
        call = fn.spawn(e)
        print("spawned", e.get("TAG", e["MODULE"]), "on", gpu, call.object_id, time.strftime("%H:%M:%S"), flush=True)
