"""The stage-2 A/B on Modal H100s (INTEGRATION.md, "The A/B"): variant B, the simplified climb (climb_ab.py), next to
the c5 climbs (variant A, modal_climb.py, untouched). Every run's limits are here: the container's timeout and memory,
the climb's own deadline and its time per search. Records go to the Volume `integration` (climb/); --fetch copies
them to integration/lib/records/.

    cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run --detach \
        ../cpu_first/integration/lib/modal_ab.py --mode profile --env '{"TASK": "mul"}'
    ... --mode climb --name b1_exact_h100 --config '{"cap": "measured"}'
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
# the limits of every run (the container is stopped at TIMEOUT_S; the climb itself stops at CLIMB_S)
GPU = "H100"
TIMEOUT_S = 6 * 3600
MEMORY_MB = 64 * 1024
CLIMB_S = 5 * 3600
PROFILE_TIMEOUT_S = 2 * 3600

image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "library", "/root/cpu_first/library", ignore=["records", "inventing", "__pycache__"])
         .add_local_dir(CPU / "anchors", "/root/cpu_first/anchors", ignore=["records", "__pycache__", "*.md"])
         .add_local_file(CPU / "integration" / "__init__.py", "/root/cpu_first/integration/__init__.py")
         .add_local_dir(HERE, "/root/cpu_first/integration/lib", ignore=["records", "__pycache__", "logs", "*.out"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("integration-ab", image=image)
vol = modal.Volume.from_name("integration", create_if_missing=True)


def _save(name: str, text: str) -> None:
    Path("/out/climb").mkdir(parents=True, exist_ok=True)
    Path(f"/out/climb/{name}").write_text(text)
    vol.commit()


@app.function(gpu=GPU, timeout=PROFILE_TIMEOUT_S, memory=MEMORY_MB, volumes={"/out": vol}, max_containers=3)
def profile(env: dict) -> str:
    import os
    import subprocess
    e = dict(os.environ, H1_DATA=f"/root/data/{DATA}", PYTHONPATH="/root/cpu_first", **env)
    proc = subprocess.Popen(["python", "-W", "ignore", "-u", "-m", env.get("MODULE", "integration.lib.profile_search")],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e, cwd="/root/cpu_first")
    lines = []
    for line in proc.stdout:                             # streamed to the container's log as it comes (watchable)
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    out = "".join(lines)
    _save(f"profile_{env.get('TAG', env.get('TASK', 'mul'))}.log", out)
    return out


@app.function(gpu=GPU, timeout=TIMEOUT_S, memory=MEMORY_MB, volumes={"/out": vol}, max_containers=4)
def climb(name: str, config: dict) -> dict:
    import os
    import sys
    os.environ["H1_DATA"] = f"/root/data/{DATA}"
    sys.path.insert(0, "/root/cpu_first")
    from integration.lib import climb_ab
    config = dict(config, seconds=min(config.get("seconds", CLIMB_S), CLIMB_S))

    def save(rec):                                       # the record so far, after every unit of work
        _save(f"{name}.json", json.dumps(rec, indent=1, default=str))
    if config.get("variant") == "core":
        return climb_ab.run_core(name, config, save)
    return climb_ab.run(name, config, save)


# research runs on bigger GPUs (the official certification runs stay on an H100); same limits as `climb`
@app.function(gpu="B200", timeout=TIMEOUT_S, memory=MEMORY_MB, volumes={"/out": vol}, max_containers=2)
def climb_b200(name: str, config: dict) -> dict:
    return climb.local(name, config)


@app.function(gpu="H200", timeout=TIMEOUT_S, memory=MEMORY_MB, volumes={"/out": vol}, max_containers=2)
def climb_h200(name: str, config: dict) -> dict:
    return climb.local(name, config)


@app.function(gpu="B200", timeout=600, max_containers=1)
def probe_b200() -> str:
    import subprocess
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                          text=True).stdout + subprocess.run(["bash", "-c", "nproc; grep -m1 'model name' /proc/cpuinfo"],
                                                             capture_output=True, text=True).stdout


@app.function(gpu="H200", timeout=600, max_containers=1)
def probe_h200() -> str:
    return probe_b200.local()


@app.function(gpu="H100", timeout=600, max_containers=1)
def probe_h100() -> str:
    return probe_b200.local()


@app.local_entrypoint()
def main(mode: str = "climb", name: str = "b1", config: str = "{}", env: str = "{}", gpu: str = "H100") -> None:
    rec_dir = HERE / "records"
    rec_dir.mkdir(exist_ok=True)
    if mode == "fetch":
        for entry in vol.listdir("climb"):
            dest = rec_dir / Path(entry.path).name
            with open(dest, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
            print("fetched", dest.name)
        return
    if mode == "profile":
        call = profile.spawn(json.loads(env))
        print("spawned profile", call.object_id)
        return
    if mode == "probe":
        for fn in (probe_b200, probe_h200, probe_h100):
            try:
                print(fn.tag, "->", fn.remote().strip().replace("\n", " | "), flush=True)
            except Exception as exc:
                print(fn.tag, "-> unavailable:", repr(exc)[:200], flush=True)
        return
    fn = {"H100": climb, "H200": climb_h200, "B200": climb_b200}[gpu]
    call = fn.spawn(name, json.loads(config))
    print("spawned", name, "on", gpu, call.object_id, time.strftime("%H:%M:%S"))
