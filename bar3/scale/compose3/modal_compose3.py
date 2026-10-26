"""Round 3's runs on Modal (COMPOSE3.md §4): bar3/scale/compose/modal_compose.py's run function, on this study's own
app and Volume (`bar3-compose3`: runs/<TAG>.log, runs/<TAG>.json), with rounds 1 and 2's code in the image too (round
2's subgoal score is imported from its folder).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/compose3/modal_compose3.py --arms replay_c2_b2_rd,c3_r1
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/compose3/modal_compose3.py --mode fetch

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
TIMEOUT_S = 5400                    # one container at most: an arm's SECONDS (2,400) and its last search's overrun
MEMORY = (16384, 65536)             # request, hard limit (MiB)

# Round 2's settings, on round 3's copy: the small target and the round's search each level, the concept rule,
# SELF_PRACTICE.md §7's documented run (NP 8, deep probes over whole concepts), 2,400 s.
BASE = dict(MODULE="bar3.scale.compose3.core_c3", TARGET="small", ROUND_TOO="1", RULE="lpca", NP="8",
            DEEP_BY="bindings", DEEP_FOCUS="closure", LEVELS="10", SECONDS="2400")
BOOK_MIN = dict(LAWS="lt;red;add;mul;acc.mul", PATTERNS="pair;x+on;x,o;o,x,d")
LESS_RED = dict(LAWS="lt;red;add;mul;acc.mul", PATTERNS="pair;x,o;o,x,d")          # book_min less x + o N
LESS_RED_ROW = dict(LAWS="lt;red;add;mul", PATTERNS="pair;x,o")                   # and less o,x,d (the row)
BAR3 = "restrict,project,value,rest"
FULL = dict(CLEAN_VALUES="1", SUBGOAL="1", LAWNEAR="1", ROWCHECK="1", TIEBREAK="1", RESTRICT_ROWS="1", SPELL="16",
            REST_DEEP="1")                                                          # round 2's B2rd
ARMS = {
    "replay_cp_none": dict(BASE, SOURCES="none", LEVELS="1"),                 # round 1's controls, every new flag off
    "replay_cp_book_min": dict(BASE, SOURCES="law", **BOOK_MIN),
    "replay_c2_b2_rd": dict(BASE, SOURCES="law," + BAR3, **LESS_RED, **FULL),  # round 2's winning arm
    "c3_r1": dict(BASE, SOURCES="law," + BAR3, **LESS_RED_ROW, **FULL, TIEFOLLOW="1"),     # part 1: no row
    "c3_g_x3": dict(BASE, SOURCES="law," + BAR3, **LESS_RED, **FULL, TARGET_FN="x3", LEVELS="7"),      # part 2
    "c3_g_10x2p1": dict(BASE, SOURCES="law," + BAR3, **LESS_RED, **FULL, TARGET_FN="10x2p1", LEVELS="7"),
}
for _k in list(FULL) + ["TIEFOLLOW"]:                                           # one flag off, if an arm gets a round
    for _a in ("c3_r1", "c3_g_x3", "c3_g_10x2p1"):
        if _k in ARMS[_a]:
            ARMS[f"{_a}_no_{_k.lower()}"] = {k: v for k, v in ARMS[_a].items() if k != _k}
CHECKS = {}

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
         .add_local_dir(HERE, "/root/cpu_first/bar3/scale/compose3", ignore=["records", "logs", "__pycache__", "*.md"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("bar3-compose3", image=image)
vol = modal.Volume.from_name("bar3-compose3", create_if_missing=True)


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


@app.function(gpu="B200", timeout=TIMEOUT_S, memory=MEMORY, volumes={"/out": vol}, max_containers=12)
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
