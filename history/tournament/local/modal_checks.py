"""local/chain_check.py or local/phase_c_check.py for several seeds at once on Modal CPU containers (the H1 data,
the evaluator stub and the local harness in a plain image). Prints each seed's last lines.

    ~/github/sotaku/venv/bin/python -m modal run tournament/local/modal_checks.py --script chain_check --seeds 0,1,2,3
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
TOUR = HERE.parent
CPU = TOUR.parents[1]  # cpu_first (the tournament is in history/)
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "learned_remainder" / "chain_local" / "stub", "/root/cpu_first/learned_remainder/chain_local/stub")
         .add_local_file(CPU / "wider_menus" / "local" / "local_wide.py", "/root/cpu_first/wider_menus/local/local_wide.py")
         .add_local_file(TOUR / "submission_tournament.py", "/root/cpu_first/tournament/submission_tournament.py")
         .add_local_file(HERE / "chain_check.py", "/root/cpu_first/tournament/local/chain_check.py")
         .add_local_file(HERE / "phase_c_check.py", "/root/cpu_first/tournament/local/phase_c_check.py")
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("tournament-checks", image=image)


@app.function(cpu=8, memory=16384, timeout=3600, max_containers=8)
def check(script: str, seed: int, args: list) -> str:
    env = {"H1_DATA": f"/root/data/{DATA}", "OMP_NUM_THREADS": "8", "PATH": "/usr/local/bin:/usr/bin:/bin"}
    out = subprocess.run(["python", "-W", "ignore", f"/root/cpu_first/tournament/local/{script}.py", f"SEED={seed}",
                          *args], capture_output=True, text=True, env=env, cwd="/root/cpu_first")
    return out.stdout + (out.stderr[-3000:] if out.returncode else "")


@app.local_entrypoint()
def main(script: str = "chain_check", seeds: str = "0,1,2,3,4,5,6,7", args: str = "STEPS=30") -> None:
    jobs = [(script, int(s), args.split()) for s in seeds.split(",")]
    for (sc, s, _), out in zip(jobs, check.starmap(jobs)):
        lines = [l for l in out.splitlines() if l.startswith(("step", "STEP", "history", "Traceback"))]
        print(f"== {sc} seed {s}")
        for l in (lines[-3:] if lines else out.splitlines()[-8:]):
            print("  ", l[:420])
