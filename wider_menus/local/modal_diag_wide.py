"""Run local/local_wide.py variants in parallel on Modal CPU containers (a copy of
history/learned_wiring/local/modal_diag.py with its own app `wider-menus-diag` and Volume `wider-menus`: the same cached
competition image, whose build regenerates H1). Run from the one-layer-deeper repo root:

    modal run /path/to/wider_menus/local/modal_diag_wide.py --plan plan.json      # spawns, then collects the logs
    modal run /path/to/wider_menus/local/modal_diag_wide.py --fetch 1 --prefix d1  # logs already on the Volume

A plan is a JSON list of {"name": ..., "argv": ["submission_wide.py", "train", "2000", "512", "1", "K=V", ...]}. Each job
writes its stdout to the Volume `wider-menus` under diag/: <name>.partial while it runs (committed every minute),
<name>.log when it ends; the client copies them to local/logs/ as they arrive (launch it as a background command; if
the client dies, --fetch 1 gets them, partial logs included). At most 6 jobs run at once.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LW = HERE.parent
LR = LW.parent / "learned_remainder"
REMOTE = "/workspace/one-layer-benchmark"
H1_DIR = "data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
H1_ARGS = (
    f"--output_dir {H1_DIR} --modulus_bits '[13,16,20]' --time_steps '[4,8,16]' --ood_time_steps '[32]' "
    "--ood_examples_per_setting 3334 --examples_per_setting 12222 "
    "--extra_train_examples_by_time_step '{\"1\":1000,\"2\":1000}' "
    "--depth_evaluation_time_steps '[1,2,4,8,16,32,64]' --depth_evaluation_examples_per_setting 256 "
    "--ood_n_depth_evaluation_modulus_bits '[14,18,22,24]' --ood_n_depth_evaluation_examples_per_setting 192 "
    "--train_fraction 0.9090909090909091 --test_fraction 0.09090909090909091 "
    "--split_group modulus --factor_modulus 4 --factor_remainder 3 "
    "--separate_ood_splits true --seed 45 --separate_input_output true"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.13")
    .apt_install("git", "build-essential")
    .pip_install("uv==0.11.3")
    .env({"PYTHONPATH": REMOTE})
    .run_commands(f"mkdir -p {REMOTE}")
    .add_local_file("pyproject.toml", f"{REMOTE}/pyproject.toml", copy=True)
    .add_local_file("uv.lock", f"{REMOTE}/uv.lock", copy=True)
    .add_local_file("submission_validation.py", f"{REMOTE}/submission_validation.py", copy=True)
    .run_commands(f"cd {REMOTE} && uv sync --frozen --no-install-project")
    .add_local_dir("benchmark", f"{REMOTE}/benchmark", copy=True, ignore=["__pycache__"])
    .add_local_dir("data", f"{REMOTE}/data", copy=True, ignore=["__pycache__", "generated"])
    .run_commands(f"cd {REMOTE} && .venv/bin/python -m data.reversed_n_squaring_mod {H1_ARGS}")
)
app = modal.App("wider-menus-diag", image=image)
vol = modal.Volume.from_name("wider-menus", create_if_missing=True)

FILES = {"local/local_wide.py": LW / "local" / "local_wide.py",
         "../learned_remainder/chain_local/stub/benchmark/__init__.py": LR / "chain_local/stub/benchmark/__init__.py"}
for _f in sorted(LW.glob("submission_*.py")):
    FILES[_f.name] = _f


@app.function(cpu=8, memory=16384, timeout=4 * 3600, volumes={"/out": vol}, max_containers=6)
def diag(name: str, files: dict, argv: list) -> str:
    work = Path(f"/tmp/{name}/learned_wiring")
    for rel, src in files.items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text(src)
    env = dict(os.environ, H1_DATA=f"{REMOTE}/{H1_DIR}", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8")
    t0 = time.time()
    Path("/out/diag").mkdir(parents=True, exist_ok=True)
    live, err = Path(f"/out/diag/{name}.partial"), work / "stderr.log"   # the partial log reaches the Volume every minute
    with open(live, "w") as fo, open(err, "w") as fe:
        proc = subprocess.Popen([f"{REMOTE}/.venv/bin/python", "-W", "ignore",
                                 str(work / "local/local_wide.py"), *argv],
                                cwd=str(work / "local"), stdout=fo, stderr=fe, text=True, env=env)
        while proc.poll() is None:
            time.sleep(60)
            fo.flush()
            vol.commit()
    out = (f"# argv={argv} wall={time.time() - t0:.0f}s rc={proc.returncode}\n" + live.read_text()
           + ("\n--- stderr ---\n" + err.read_text()[-20000:] if proc.returncode else ""))
    Path(f"/out/diag/{name}.log").write_text(out)
    live.unlink()
    vol.commit()
    return name


def fetch_all(prefix: str = "") -> list:
    (HERE / "logs").mkdir(exist_ok=True)
    got = []
    for entry in vol.listdir("diag"):
        if not Path(entry.path).name.startswith(prefix):
            continue
        dest = HERE / "logs" / Path(entry.path).name
        if not dest.exists() or dest.stat().st_size != entry.size:
            with open(dest, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
            got.append(dest.name)
    return got


@app.local_entrypoint()
def main(plan: str = "", fetch: int = 0, prefix: str = "") -> None:
    if fetch:
        print("fetched", fetch_all(prefix), flush=True)
        return
    jobs = json.loads(Path(plan).read_text())
    files = {rel: path.read_text() for rel, path in FILES.items()}
    calls = [(j["name"], diag.spawn(j["name"], files, j["argv"])) for j in jobs]
    print("spawned", [n for n, _ in calls], flush=True)
    while calls:                                            # collect in completion order
        pending = []
        for job, call in calls:
            try:
                call.get(timeout=0)
            except TimeoutError:
                pending.append((job, call))
                continue
            except Exception as exc:
                print("FAILED call:", job, repr(exc)[:2000], flush=True)
                continue
            fetch_all(job)
            log = (HERE / "logs" / f"{job}.log").read_text().splitlines()
            tail = [l for l in log if l.startswith("step")][-1:] or log[-3:]
            print(job, "|", " ".join(tail)[:400], flush=True)
        calls = pending
        if calls:
            time.sleep(20)
