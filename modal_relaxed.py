"""Run the relaxed-rules experiments (RELAXED.md) on Modal instead of the laptop.

One detached job per run; each writes <name>.log and <name>.json (and a .pt checkpoint where the script saves one) to
the `relaxed-runs` Volume:

    modal run --detach cpu_first/modal_relaxed.py --script relax_trace.py --name v2_b12 \
        --args "--bits 12 --data h1like --fmt v2 --steps 8000"

Watch:  modal volume get relaxed-runs <name>.log -   (or modal container exec <id> -- tail /runs/<name>.log)
Fetch:  modal volume get relaxed-runs <name>.json cpu_first/runs/
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
H1_LOCAL = os.environ.get(
    "H1_DATA",
    "/Users/chenglou/github/one-layer/one-layer-deeper/data/generated/"
    "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2",
)
SCRIPTS = ["relax_common.py", "relax_step.py", "relax_trace.py", "relax_periodic.py", "relax_square.py", "relax_trace_errors.py",
           "relax_periodic_cert.py", "h1.py"]
CKPTS = ["relax2_trace_h1like_forward_b12.pt", "relax2_trace_h1like_reverse_b12.pt"]

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy==2.2.6")
for f in SCRIPTS:
    image = image.add_local_file(str(HERE / f), f"/root/cpu_first/{f}")
for f in CKPTS:
    if (HERE / "runs" / f).exists():
        image = image.add_local_file(str(HERE / "runs" / f), f"/root/ckpt/{f}")
image = image.add_local_dir(H1_LOCAL, "/root/h1data")

app = modal.App("relaxed-rules", image=image)
runs_vol = modal.Volume.from_name("relaxed-runs", create_if_missing=True)


@app.function(gpu="H100", timeout=6 * 3600, volumes={"/runs": runs_vol},
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run(script: str, args: list[str], name: str) -> dict:
    runs_vol.reload()
    log = f"/runs/{name}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True).stdout.strip()
        fh.write(f"gpu={gpu} script={script} args={args}\n")
        fh.flush()
        proc = subprocess.Popen(["python", "-u", f"/root/cpu_first/{script}", *args, "--out", f"/runs/{name}.json"],
                                cwd="/root/cpu_first", stdout=fh, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONUNBUFFERED="1", H1_DATA="/root/h1data"))
        last = time.time()
        while proc.poll() is None:  # commit the log now and then so it can be read from the Volume mid-run
            time.sleep(10)
            if time.time() - last > 120:
                runs_vol.commit()
                last = time.time()
        code = proc.returncode
    with open(log, "a") as fh:
        fh.write(f"\nexit code {code}, wall {time.time() - t0:.0f}s\n")
    runs_vol.commit()
    return {"name": name, "returncode": code, "wall": time.time() - t0}


@app.local_entrypoint()
def main(script: str, name: str, args: str = "") -> None:
    extra = args.split() + (["--device", "cuda"] if "--device" not in args else [])
    call = run.spawn(script, extra, name)
    print(f"spawned {name}: {call.object_id}  -> Volume relaxed-runs: {name}.log / {name}.json")
