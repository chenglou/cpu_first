"""The combined file's tiny core (core_c.py) on Modal, as generic_ops/modal_g.py runs core_g.py: the module run with env
vars on one GPU container, its output streamed to the app log, its log and JSON record saved to the Volume
`combined-sp2` (runs/<TAG>.log, runs/<TAG>.json) every minute. The evaluator runs go through the shared runner
(tools/modal_runner.py) instead. Limits: the container's timeout and hard memory limit, the module's own SECONDS (a
wall-clock guard below the timeout), and the module and its children killed with the call.

    cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach \
        integration/combined_sp2/modal_c.py --envs '[{"TAG": "core_combined"}]'
    ~/github/sotaku/venv/bin/python -m modal run integration/combined_sp2/modal_c.py --mode fetch
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
TIMEOUT_S = 3 * 3600                # one container at most (the module's SECONDS is set below this)
MEMORY = (16384, 65536)             # request, hard limit (MiB)

image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "learned_remainder" / "chain_local" / "stub",
                        "/root/cpu_first/learned_remainder/chain_local/stub", ignore=["__pycache__"])
         .add_local_file(HERE / "core_c.py", "/root/cpu_first/integration/combined_sp2/core_c.py")
         .add_local_file(HERE / "submission_stage3_combined.py",
                         "/root/cpu_first/integration/combined_sp2/submission_stage3_combined.py")
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("combined-sp2", image=image)
vol = modal.Volume.from_name("combined-sp2", create_if_missing=True)


def _run(env: dict) -> str:
    import os
    import signal
    import subprocess
    tag = env.get("TAG", "run")
    Path("/out/runs").mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, H1_DATA=f"/root/data/{DATA}", OUT="/out/runs", **{k: str(v) for k, v in env.items()})
    e.setdefault("SECONDS", str(TIMEOUT_S - 1200))
    proc = subprocess.Popen(["python", "-W", "ignore", "-u", "/root/cpu_first/integration/combined_sp2/core_c.py"],
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


@app.function(gpu="B200", timeout=TIMEOUT_S, memory=MEMORY, volumes={"/out": vol}, max_containers=2)
def run_b200(env: dict) -> str:
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
        call = run_b200.spawn(e)
        print("spawned", e.get("TAG"), "on B200", call.object_id, time.strftime("%H:%M:%S"), flush=True)
