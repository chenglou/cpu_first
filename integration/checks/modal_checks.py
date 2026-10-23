"""integration/checks/phase_c_check.py for several seeds at once on Modal CPU containers (the H1 data, the evaluator
stub and the local harness in a plain image). Prints each seed's last lines and keeps every log in checks/logs/.

    cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run integration/checks/modal_checks.py \
        --file submission_stage1.py --seeds 0,1,2,3 --args "STEPS=80" --tag pc1
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INTEG = HERE.parent
CPU = INTEG.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "learned_remainder" / "chain_local" / "stub", "/root/cpu_first/learned_remainder/chain_local/stub")
         .add_local_file(CPU / "wider_menus" / "local" / "local_wide.py", "/root/cpu_first/wider_menus/local/local_wide.py")
         .add_local_file(CPU / "general_lessons" / "local_check.py", "/root/cpu_first/general_lessons/local_check.py")
         .add_local_dir(INTEG, "/root/cpu_first/integration", ignore=["records", "logs", "work", "*.jsonl", "__pycache__"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("integration-checks", image=image)


@app.function(cpu=8, memory=32768, timeout=3600, max_containers=8)
def check(script: str, seed: int, args: list) -> str:
    env = {"H1_DATA": f"/root/data/{DATA}", "OMP_NUM_THREADS": "8", "PATH": "/usr/local/bin:/usr/bin:/bin"}
    out = subprocess.run(["python", "-W", "ignore", f"/root/cpu_first/integration/checks/{script}.py", f"SEED={seed}",
                          *args], capture_output=True, text=True, env=env, cwd="/root/cpu_first")
    return out.stdout + (out.stderr[-3000:] if out.returncode else "")


@app.function(gpu="H100", timeout=3600, max_containers=2)
def check_gpu(script: str, seed: int, args: list) -> str:
    return check.local(script, seed, args)


@app.local_entrypoint()
def main(script: str = "phase_c_check", file: str = "submission_stage1.py", seeds: str = "0,1,2,3",
         args: str = "STEPS=60", tag: str = "check", gpu: bool = False) -> None:
    jobs = [(script, int(s), [f"FILE={file}"] + args.split()) for s in seeds.split(",")]
    (HERE / "logs").mkdir(exist_ok=True)
    for (sc, s, _), out in zip(jobs, (check_gpu if gpu else check).starmap(jobs)):
        (HERE / "logs" / f"{tag}_seed{s}.log").write_text(out)
        lines = [l for l in out.splitlines() if l.startswith(("step", "STEP", "history", "Traceback"))]
        print(f"== {sc} seed {s}")
        for l in (lines[-3:] if lines else out.splitlines()[-8:]):
            print("  ", l[:600])
