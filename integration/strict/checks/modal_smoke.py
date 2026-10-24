"""A check script of this folder on a Modal CPU container (integration/checks/modal_checks.py's image and limits, with
the logs kept here, in strict/checks/logs/): for the checks that pass the local limits (a minute, 1 GB).

    cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run integration/strict/checks/modal_smoke.py \
        --script smoke_compare --args "KIND=sp2 STEPS=40" --tag smoke_sp2
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INTEG = HERE.parent.parent
CPU = INTEG.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "learned_remainder" / "chain_local" / "stub", "/root/cpu_first/learned_remainder/chain_local/stub")
         .add_local_file(CPU / "wider_menus" / "local" / "local_wide.py", "/root/cpu_first/wider_menus/local/local_wide.py")
         .add_local_dir(INTEG / "strict", "/root/cpu_first/integration/strict",
                        ignore=["records", "logs", "__pycache__"])
         .add_local_file(INTEG / "self_practice" / "submission_stage3_sp2.py",
                         "/root/cpu_first/integration/self_practice/submission_stage3_sp2.py")
         .add_local_file(INTEG / "submission_stage3_s3i.py", "/root/cpu_first/integration/submission_stage3_s3i.py")
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("strict-checks", image=image)


@app.function(cpu=8, memory=(16384, 32768), timeout=3600, max_containers=4)
def check(script: str, args: list) -> str:
    env = {"H1_DATA": f"/root/data/{DATA}", "OMP_NUM_THREADS": "8", "PATH": "/usr/local/bin:/usr/bin:/bin"}
    out = subprocess.run(["python", "-W", "ignore", f"/root/cpu_first/integration/strict/checks/{script}.py", *args],
                         capture_output=True, text=True, env=env, cwd="/root/cpu_first", timeout=3300)
    return out.stdout + (out.stderr[-3000:] if out.returncode else "")


@app.function(gpu="H100!", memory=(16384, 32768), timeout=3600, max_containers=2)
def check_gpu(script: str, args: list) -> str:
    return check.local(script, args)


@app.local_entrypoint()
def main(script: str = "smoke_compare", args: str = "", tag: str = "check", gpu: bool = False) -> None:
    jobs = [(script, a.split()) for a in args.split(";")]
    (HERE / "logs").mkdir(exist_ok=True)
    for i, ((sc, a), out) in enumerate(zip(jobs, (check_gpu if gpu else check).starmap(jobs))):
        (HERE / "logs" / f"{tag}_{i}.log").write_text(out)
        print(f"== {sc} {' '.join(a)}")
        print("\n".join(out.splitlines()[-40:]))
