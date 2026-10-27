"""The offline test on Modal (LEGAL_TEST.md; PREREG.md): offline.py on one B200, its own app and Volume (`bar3-legal`:
runs/<TAG>.log, runs/<TAG>.json). The image holds the composition harness's code (rounds 1–4, read only), s4_x2's
record, H1's data, and this folder.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/legal/modal_legal.py --tag lt1
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/legal/modal_legal.py --mode fetch

Limits: the module's own deadlines (SECONDS for Q1's searches, SEARCH_S each for Q2's); the container's timeout
(TIMEOUT_S: at $6.37 a B200-hour, at most $4.8) and hard memory limit; one container.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
TIMEOUT_S = 2700
MEMORY = (16384, 65536)                 # request, hard limit (MiB)
ENV = dict(MODULE="bar3.scale.legal.offline", SPELL="16", ARMS="full,pieces,rows780", LAW_COST="3",
           BUDGETS="1000000,10000000", SEARCH_S="900", MAX_BODY="3", N_ANSWERS="400", SECONDS="2400")

SCALE = CPU / "bar3" / "scale"
image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "library", "/root/cpu_first/library", ignore=["records", "inventing", "__pycache__"])
         .add_local_dir(CPU / "anchors", "/root/cpu_first/anchors", ignore=["records", "__pycache__", "*.md"])
         .add_local_file(CPU / "integration" / "__init__.py", "/root/cpu_first/integration/__init__.py")
         .add_local_dir(CPU / "integration" / "lib", "/root/cpu_first/integration/lib",
                        ignore=["records", "__pycache__", "logs", "*.out", "*.json", "*.txt"])
         .add_local_dir(CPU / "integration" / "self_practice", "/root/cpu_first/integration/self_practice",
                        ignore=["records", "__pycache__", "*.md"])
         .add_local_file(CPU / "bar3" / "__init__.py", "/root/cpu_first/bar3/__init__.py")
         .add_local_file(SCALE / "__init__.py", "/root/cpu_first/bar3/scale/__init__.py")
         .add_local_dir(SCALE / "compose", "/root/cpu_first/bar3/scale/compose",
                        ignore=["logs", "records", "__pycache__", "*.md"])
         .add_local_dir(SCALE / "compose2", "/root/cpu_first/bar3/scale/compose2",
                        ignore=["logs", "records", "__pycache__", "*.md"])
         .add_local_dir(SCALE / "compose3", "/root/cpu_first/bar3/scale/compose3",
                        ignore=["logs", "records", "__pycache__", "*.md"])
         .add_local_dir(SCALE / "slice", "/root/cpu_first/bar3/scale/slice",
                        ignore=["records", "logs", "__pycache__", "*.md"])
         .add_local_file(SCALE / "slice" / "records" / "s4_x2.json",
                         "/root/cpu_first/bar3/scale/slice/records/s4_x2.json")
         .add_local_dir(HERE, "/root/cpu_first/bar3/scale/legal", ignore=["records", "logs", "__pycache__", "*.md"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}",
                        ignore=["test.jsonl", "ood_*.jsonl"]))
app = modal.App("bar3-legal", image=image)
vol = modal.Volume.from_name("bar3-legal", create_if_missing=True)


@app.function(gpu="B200", timeout=TIMEOUT_S, memory=MEMORY, volumes={"/out": vol}, max_containers=1)
def run_b200(env: dict) -> str:
    import os
    import signal
    import subprocess
    tag = env.get("TAG", "run")
    Path("/out/runs").mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, H1_DATA=f"/root/data/{DATA}", PYTHONPATH="/root/cpu_first", OUT="/out/runs",
             **{k: str(v) for k, v in env.items()})
    proc = subprocess.Popen(["python", "-W", "ignore", "-u", "-m", env["MODULE"]], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=e, cwd="/root/cpu_first", start_new_session=True)
    lines, last = [], time.time()
    try:
        for line in proc.stdout:                         # streamed to the container's log as it comes
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
    return out[-3000:]


@app.local_entrypoint()
def main(mode: str = "run", tag: str = "lt1") -> None:
    if mode == "fetch":
        dest = HERE / "records"
        dest.mkdir(parents=True, exist_ok=True)
        for entry in vol.listdir("runs"):
            with open(dest / Path(entry.path).name, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
            print("fetched", Path(entry.path).name)
        return
    env = dict(ENV, TAG=tag)
    call = run_b200.spawn(env)
    print("spawned", tag, call.object_id, time.strftime("%H:%M:%S"), env, flush=True)
