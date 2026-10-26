"""The composition test's arms on Modal (COMPOSE.md §3): integration/self_practice/modal_sp.py's run function, on this
study's own app and Volume, with the image's code the same plus this folder (the driver copy core_compose.py and its
climb). Each arm is the tiny core's module with env vars on one B200, its output streamed to the app log, its log and
JSON record saved to the Volume `bar3-compose` (runs/<TAG>.log, runs/<TAG>.json) every minute.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/compose/modal_compose.py --arms cp_book_min,cp_bar3
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/compose/modal_compose.py --mode fetch

Limits: the module's own deadline (SECONDS) and per-search limit (SEARCH_S), its level cap (LEVELS); the container's
timeout and hard memory limit. Only the arms named in ARMS run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
TIMEOUT_S = 5400                    # one container at most: the arm's SECONDS (2,400) and its last search's overrun
MEMORY = (16384, 65536)             # request, hard limit (MiB)

# Every arm: the small target and the round's search (ROUND_TOO), the concept rule, the tiny core's settings as its
# documented run sets them (SELF_PRACTICE.md §7: NP 8, deep probes over whole concepts), 10 levels, 2,400 s.
BASE = dict(MODULE="bar3.scale.compose.core_compose", TARGET="small", ROUND_TOO="1", RULE="lpca", NP="8",
            DEEP_BY="bindings", DEEP_FOCUS="closure", LEVELS="10", SECONDS="2400")
BOOK_MIN = dict(LAWS="lt;red;add;mul;acc.mul", PATTERNS="pair;x+on;x,o;o,x,d")
BAR3 = "restrict,project,value,rest"
ARMS = {
    "cp_book_min": dict(BASE, SOURCES="law", **BOOK_MIN),                                   # C1, the control
    "cp_bar3": dict(BASE, SOURCES=BAR3),                                                    # A1
    "cp_bar3_lp": dict(BASE, SOURCES=BAR3, RULE="lp"),                                      # A1lp
    "cp_bar3_minus_red": dict(BASE, SOURCES="law," + BAR3, LAWS=BOOK_MIN["LAWS"],           # A2: no x + o N
                              PATTERNS="pair;x,o;o,x,d"),
    "cp_none": dict(BASE, SOURCES="none", LEVELS="1"),                                      # C2, no practice
    "cp_bar3_clean": dict(BASE, SOURCES=BAR3, CLEAN_VALUES="1"),                            # A1c (PREREG_ADDENDUM.md)
    "cp_bar3_minus_red_clean": dict(BASE, SOURCES="law," + BAR3, LAWS=BOOK_MIN["LAWS"],     # A2c (PREREG_ADDENDUM2.md)
                                    PATTERNS="pair;x,o;o,x,d", CLEAN_VALUES="1"),
}
# Checks on the arms' GPU (COMPOSE.md §5.5): the oracle's draws come from generators on the device.
CHECKS = {"route": dict(MODULE="bar3.scale.compose.route", REC="/out/runs/cp_bar3_minus_red_clean.json"),
          "route_focus": dict(MODULE="bar3.scale.compose.route", REC="/out/runs/cp_bar3_minus_red_clean.json",
                              FOCUS="T.red.pair.AN|T.red.pair.AY", SKIP_FULL="1")}

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
         .add_local_dir(HERE, "/root/cpu_first/bar3/scale/compose",
                        ignore=["records", "logs", "__pycache__", "*.md"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("bar3-compose", image=image)
vol = modal.Volume.from_name("bar3-compose", create_if_missing=True)


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


@app.function(gpu="B200", timeout=TIMEOUT_S, memory=MEMORY, volumes={"/out": vol}, max_containers=5)
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
    assert names and all(a in ARMS or a in CHECKS for a in names), (names, sorted(ARMS), sorted(CHECKS))
    for a in names:
        env = dict(ARMS.get(a) or CHECKS[a], TAG=a)
        call = run_b200.spawn(env)
        print("spawned", a, call.object_id, time.strftime("%H:%M:%S"), json.dumps(env), flush=True)
