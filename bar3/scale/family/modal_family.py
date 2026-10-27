"""This study's runs on Modal (FAMILY.md §4): round 6's runner (bar3/scale/reading/modal_reading.py) on this study's own
app and Volume (`bar3-family`: runs/<TAG>.log, runs/<TAG>.json), with the earlier rounds' code the copies import in
the image (compose, compose2, compose3, slice) and the copies here (core_f). At most 8 B200s at once (max_containers),
the coordinator's limit; the spend watcher's total is the cap less one hour of every container that can run at once.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/family/modal_family.py --arms r7_x2
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/family/modal_family.py --mode fetch

Limits: the module's own deadline (SECONDS), per-search limit (SEARCH_S) and level cap (LEVELS); the container's
timeout and hard memory limit. Only the names in ARMS run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
TIMEOUT_S = 5400                    # one container at most: an arm's SECONDS (2,400 or 3,600) and its overrun
MEMORY = (16384, 65536)             # request, hard limit (MiB)

# Round 6's settings (bar3/scale/reading/modal_reading.py), on this study's copy: the small target and the round's
# search each level, the concept rule, NP 8, deep probes over whole concepts; round 2's flags and TIEFOLLOW; the cuts
# and their loops; no book, the step's cuts (round 6's r6_x2_nobook).
BASE = dict(MODULE="bar3.scale.family.core_f", TARGET="small", ROUND_TOO="1", RULE="lpca", NP="8",
            DEEP_BY="bindings", DEEP_FOCUS="closure", LEVELS="10", SECONDS="2400")
BAR3 = "restrict,project,value,rest"
FULL = dict(CLEAN_VALUES="1", SUBGOAL="1", LAWNEAR="1", ROWCHECK="1", TIEBREAK="1", RESTRICT_ROWS="1", SPELL="16",
            REST_DEEP="1")                                                          # round 2's B2rd
L2 = dict(BASE, SOURCES="law," + BAR3, LAWS="lt;red;add", PATTERNS="pair", **FULL, TIEFOLLOW="1", SLICE="1",
          LOOPS="1")                                                                # round 5's l5_x2
NOBOOK = dict(L2, SOURCES=BAR3, DOWN="1")                                         # round 6's no book
FAMILY = dict(NOBOOK, CONST="1", SUBTERMS="1", SECONDS="3600")                    # this study's setup
ARMS = {                                                                         # PREREG.md, in order
    "replay_r6_x2_nobook": NOBOOK,                                               # round 6's r6_x2_nobook
    "r7_x2": FAMILY,                                                             # H1, the control
    "r7_x3": dict(FAMILY, TARGET_FN="x3"),                                       # the cube
    "r7_ten": dict(FAMILY, TARGET_FN="ten"),
    "r7_rho": dict(FAMILY, TARGET_FN="rho"),
    "r7_plus3": dict(FAMILY, TARGET_FN="plus3"),
    "r7_minus1": dict(FAMILY, TARGET_FN="minus1"),
    "r7_tenrho": dict(FAMILY, TARGET_FN="10x2p1"),
}
FIXED = dict(FAMILY, CONST1="1", HOLES="1")                                       # addendum 1: both fixes
ARMS.update({                                                                    # addendum 1 (PREREG_ADDENDUM1.md)
    "r7_minus1_both": dict(FIXED, TARGET_FN="minus1"),
    "r7_minus1_const1": dict(FAMILY, TARGET_FN="minus1", CONST1="1"),
    "r7_minus1_holes": dict(FAMILY, TARGET_FN="minus1", HOLES="1"),
    "r7_x2_both": FIXED,
    "r7_x3_both": dict(FIXED, TARGET_FN="x3"),
    "r7_ten_both": dict(FIXED, TARGET_FN="ten"),
    "r7_rho_both": dict(FIXED, TARGET_FN="rho"),
    "r7_plus3_both": dict(FIXED, TARGET_FN="plus3"),
    "r7_tenrho_both": dict(FIXED, TARGET_FN="10x2p1"),
    "r7_x2_const1": dict(FAMILY, CONST1="1"),
    "r7_x3_const1": dict(FAMILY, TARGET_FN="x3", CONST1="1"),
})
ALL3 = dict(FIXED, ROUNDT="1")                                                    # addendum 2: all three fixes
ARMS.update({                                                                    # addendum 2 (PREREG_ADDENDUM2.md)
    "r7_minus1_all": dict(ALL3, TARGET_FN="minus1"),
    "r7_minus1_const1_roundt": dict(FAMILY, TARGET_FN="minus1", CONST1="1", ROUNDT="1"),
    "r7_x2_roundt": dict(FAMILY, ROUNDT="1"),
    "r7_x3_roundt": dict(FAMILY, TARGET_FN="x3", ROUNDT="1"),
    "r7_x2_all": ALL3,
    "r7_x3_all": dict(ALL3, TARGET_FN="x3"),
    "r7_ten_all": dict(ALL3, TARGET_FN="ten"),
    "r7_rho_all": dict(ALL3, TARGET_FN="rho"),
    "r7_plus3_all": dict(ALL3, TARGET_FN="plus3"),
    "r7_tenrho_all": dict(ALL3, TARGET_FN="10x2p1"),
})

image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "library", "/root/cpu_first/library", ignore=["records", "inventing", "__pycache__"])
         .add_local_dir(CPU / "anchors", "/root/cpu_first/anchors", ignore=["records", "__pycache__", "*.md"])
         .add_local_file(CPU / "integration" / "__init__.py", "/root/cpu_first/integration/__init__.py")
         .add_local_dir(CPU / "integration" / "lib", "/root/cpu_first/integration/lib",
                        ignore=["records", "__pycache__", "logs", "*.out", "*.json", "*.txt"])
         .add_local_dir(CPU / "integration" / "self_practice", "/root/cpu_first/integration/self_practice",
                        ignore=["records", "__pycache__", "*.md"])
         .add_local_file(CPU / "bar3" / "__init__.py", "/root/cpu_first/bar3/__init__.py")
         .add_local_file(CPU / "bar3" / "scale" / "__init__.py", "/root/cpu_first/bar3/scale/__init__.py")
         .add_local_dir(CPU / "bar3" / "scale" / "compose", "/root/cpu_first/bar3/scale/compose",
                        ignore=["logs", "records", "__pycache__", "*.md"])
         .add_local_dir(CPU / "bar3" / "scale" / "compose2", "/root/cpu_first/bar3/scale/compose2",
                        ignore=["logs", "records", "__pycache__", "*.md"])
         .add_local_dir(CPU / "bar3" / "scale" / "compose3", "/root/cpu_first/bar3/scale/compose3",
                        ignore=["logs", "records", "__pycache__", "*.md"])
         .add_local_dir(CPU / "bar3" / "scale" / "slice", "/root/cpu_first/bar3/scale/slice",
                        ignore=["logs", "records", "__pycache__", "*.md"])
         .add_local_dir(HERE, "/root/cpu_first/bar3/scale/family", ignore=["records", "logs", "__pycache__", "*.md"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("bar3-family", image=image)
vol = modal.Volume.from_name("bar3-family", create_if_missing=True)


def _run(env: dict) -> str:
    import os
    import signal
    import subprocess
    tag = env.get("TAG", "run")
    Path("/out/runs").mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, H1_DATA=f"/root/data/{DATA}", PYTHONPATH="/root/cpu_first", OUT="/out/runs",
             **{k: str(v) for k, v in env.items()})
    proc = subprocess.Popen(["python", "-W", "ignore", "-u", "-m", env["MODULE"]],
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


@app.function(gpu="B200", timeout=TIMEOUT_S, memory=MEMORY, volumes={"/out": vol}, max_containers=8)
def run_b200(env: dict) -> str:
    return _run(env)


@app.local_entrypoint()
def main(mode: str = "run", arms: str = "") -> None:
    if mode == "fetch":
        dest = HERE / "records"
        dest.mkdir(parents=True, exist_ok=True)
        for entry in vol.listdir("runs"):
            with open(dest / Path(entry.path).name, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
            print("fetched", Path(entry.path).name)
        return
    names = [a for a in arms.split(",") if a]
    assert names and all(a in ARMS for a in names), (names, sorted(ARMS))
    for a in names:
        env = dict(ARMS[a], TAG=a)
        call = run_b200.spawn(env)
        print("spawned", a, call.object_id, time.strftime("%H:%M:%S"), json.dumps(env), flush=True)
