"""climb_b.py on a Modal GPU: stage 3's climb on an instruction set, one run per container, the record to the Volume
`bar3-bridge` (climb/) and back with --fetch.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_climb_b.py --sets s3i,inv4 --tag c1
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/bridge/modal_climb_b.py --fetch

Limits: each run stops by itself at --seconds (the climb's own deadline) and the container times out an hour later.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "library", "/root/cpu_first/library", ignore=["records", "inventing", "__pycache__"])
         .add_local_file(CPU / "integration" / "__init__.py", "/root/cpu_first/integration/__init__.py")
         .add_local_dir(CPU / "integration" / "lib", "/root/cpu_first/integration/lib",
                        ignore=["records", "__pycache__", "logs", "*.json", "*.txt", "*.out"])
         .add_local_file(CPU / "bar3" / "__init__.py", "/root/cpu_first/bar3/__init__.py")
         .add_local_dir(HERE, "/root/cpu_first/bar3/bridge", ignore=["records", "__pycache__", "*.md"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("bar3-bridge-climb", image=image)
vol = modal.Volume.from_name("bar3-bridge", create_if_missing=True)


def _run(name: str, args: list, gpu: str) -> str:
    import os
    import subprocess
    os.makedirs("/out/climb", exist_ok=True)
    env = dict(os.environ, H1_DATA=f"/root/data/{DATA}", PYTHONPATH="/root/cpu_first")
    out = f"/out/climb/{name}.json"
    r = subprocess.run(["python", "-u", "-m", "bar3.bridge.climb_b", *args, "--out", out], capture_output=True,
                       text=True, env=env, cwd="/root/cpu_first")
    Path(f"/out/climb/{name}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-20000:])
    vol.commit()
    return name


@app.function(gpu="H100", timeout=6 * 3600, volumes={"/out": vol}, max_containers=8)
def climb_h100(name: str, args: list) -> str:
    return _run(name, args, "H100")


@app.function(gpu="B200", timeout=6 * 3600, volumes={"/out": vol}, max_containers=8)
def climb_b200(name: str, args: list) -> str:
    return _run(name, args, "B200")


@app.function(gpu="H100", timeout=2 * 3600, volumes={"/out": vol}, max_containers=4)
def script_h100(module: str, args: list, tag: str) -> str:
    """Any module of the repo on an H100 (benchmarks, checks); its output to climb/script_<tag>.log."""
    import os
    import subprocess
    env = dict(os.environ, H1_DATA=f"/root/data/{DATA}", PYTHONPATH="/root/cpu_first")
    r = subprocess.run(["python", "-u", "-m", module, *args], capture_output=True, text=True, env=env,
                       cwd="/root/cpu_first")
    os.makedirs("/out/climb", exist_ok=True)
    Path(f"/out/climb/script_{tag}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-20000:])
    vol.commit()
    return r.stdout[-3000:]


@app.local_entrypoint()
def main(sets: str = "", tag: str = "c1", seconds: float = 3600, gpu: str = "h100", progs: str = "",
         fetch: bool = False, script: str = "", args: str = "", scale: float = 1.0, modx_above: bool = False) -> None:
    rec = HERE / "records" / "climb"
    rec.mkdir(parents=True, exist_ok=True)
    if fetch:
        for entry in vol.listdir("climb"):
            dest = rec / Path(entry.path).name
            with open(dest, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
            print("fetched", dest.name)
        return
    if script:
        print(script_h100.remote(script, args.split() if args else [], tag))
        return
    fn = climb_b200 if gpu == "b200" else climb_h100
    for s in sets.split(","):
        args = ["--set", s if s in ("s3i", "inv3_sel", "inv4", "lifted") else f"/root/cpu_first/bar3/bridge/{s}",
                "--seconds", str(seconds), "--scale", str(scale)] + (["--modx-above"] if modx_above else [])
        if progs:
            args += ["--progs", f"/root/cpu_first/bar3/bridge/{progs}"]
        name = f"{Path(s).stem}_{gpu}_{tag}"
        call = fn.spawn(name, args)
        print("spawned", name, call.object_id)
