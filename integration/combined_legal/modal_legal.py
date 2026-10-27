"""This folder's tiny cores on Modal (LEGAL_FILE.md §4), as combined_fast/modal_fast.py runs them: core_c.py
(combined_fast's, byte for byte) with env vars on one GPU container, its output streamed to the app log, its log and
JSON record saved to the Volume `combined-legal` (runs/<TAG>.log, .json) every minute. Limits: the container's timeout
and hard memory limit, the script's own SECONDS (a wall-clock guard below the timeout), and the script and its
children killed with the call. The official runs go through the shared runner (tools/modal_runner.py).

    cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach \
        integration/combined_legal/modal_legal.py --envs '[{"TAG": "core_legal", "FILE": "submission_stage3_legal.py"}]'
    ~/github/sotaku/venv/bin/python -m modal run integration/combined_legal/modal_legal.py --mode fetch
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
TIMEOUT_S = 3600                    # one container at most (the script's SECONDS is set below this)
MEMORY = (16384, 65536)             # request, hard limit (MiB)

image = (modal.Image.debian_slim(python_version="3.13").apt_install("gcc")      # (gcc: Triton builds its launcher)
         .pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "learned_remainder" / "chain_local" / "stub",
                        "/root/cpu_first/learned_remainder/chain_local/stub", ignore=["__pycache__"])
         .add_local_dir(HERE, "/root/cpu_first/integration/combined_legal",       # the script, the files
                        ignore=["records", "checks", "__pycache__", "*.md"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("combined-legal", image=image)
vol = modal.Volume.from_name("combined-legal", create_if_missing=True)


def _run(env: dict, python: str = "python") -> str:
    import os
    import signal
    import subprocess
    tag = env.get("TAG", "run")
    Path("/out/runs").mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, H1_DATA=f"/root/data/{DATA}", OUT="/out/runs", PYTHONPATH="/root/cpu_first",
             **{k: str(v) for k, v in env.items()})
    e.setdefault("SECONDS", str(TIMEOUT_S - 1200))
    e["FILE"] = f"/root/cpu_first/integration/combined_legal/{env.get('FILE', 'submission_stage3_legal.py')}"
    script = "/root/cpu_first/integration/combined_legal/core_c.py"
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
        fn = {"h100": run_h100}.get(e.get("GPU"), run_b200)
        call = fn.spawn(e)
        print("spawned", e.get("TAG"), "on", e.get("GPU", "b200"), call.object_id, time.strftime("%H:%M:%S"),
              flush=True)
