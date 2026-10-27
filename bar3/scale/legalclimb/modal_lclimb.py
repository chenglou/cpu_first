"""This study's runs on Modal (LEGALCLIMB.md §4): round 4's runner (bar3/scale/slice/modal_slice.py) on this study's
own app and Volume (`bar3-legalclimb`: runs/<TAG>.log, runs/<TAG>.json), the copies here (core_l), at most MAX B200s at
once so that the spend watcher's margin of one hour of the containers' cost holds; arms spawned together queue.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/legalclimb/modal_lclimb.py --arms g1,g0
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/legalclimb/modal_lclimb.py --mode fetch

Limits: the module's own deadline (SECONDS), per-search limit (SEARCH_S, 900 s) and level cap (LEVELS); the
container's timeout and hard memory limit. Only the names in ARMS or CHECKS run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent.parent
SCALE = HERE.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
TIMEOUT_S = 3900                    # one container at most: an arm's SECONDS (2,700) and its last search's overrun
MEMORY = (16384, 65536)             # request, hard limit (MiB)
MAX = int(__import__("os").environ.get("LEGALCLIMB_MAX", 5))   # ADDENDUM1.md: 1 for the added arms' own app

# Round 4's settings (SELF_PRACTICE.md §7's documented run: NP 8, deep probes over whole concepts), the concept
# rule, 780 answer rows (the most a full-size state can hold: 8,192 words), 2,700 s.
BASE = dict(MODULE="bar3.scale.legalclimb.core_l", ROUND_TOO="1", RULE="lpca", NP="8", DEEP_BY="bindings",
            DEEP_FOCUS="closure", LEVELS="10", SECONDS="2700", ANSWERS="780", SPELL="16")
ROWS = dict(TARGET="rows", SOURCES="law,value,rest", LAWNEAR="1", SUBGOAL="1", REST_DEEP="1", CLEAN_VALUES="1",
            ROWCHECK="1", LOOK_COST="3")                       # bar 3's sources on the answer rows, at full size
BOOK_MIN = dict(LAWS="lt;red;add;mul;acc.mul", PATTERNS="pair;x+on;x,o;o,x,d")
FLOOR = dict(LAWS="lt;red;add;mul;acc.mul", PATTERNS="pair;x+dn;x,o;o,x,d")       # book_min, its reduction -> the chain
ARMS = {
    "c1": dict(BASE, TARGET="round", SOURCES="law", **BOOK_MIN),                     # the book alone, as sp2 runs it
    "g0": dict(BASE, TARGET="round", SOURCES="law", **FLOOR),                        # the floor's book alone
    "g1": dict(BASE, **ROWS, **FLOOR),                                               # the floor and bar 3's sources
    "g1_id": dict(BASE, **ROWS, **FLOOR, IDCHECK="1"),                               # and the identity check
    "g2": dict(BASE, **ROWS, LAWS="lt;red;add;mul;acc.mul", PATTERNS="pair;x,o;o,x,d"),   # less the chain
    "g3": dict(BASE, **ROWS, LAWS="lt;red;add;mul", PATTERNS="pair;x+dn;x,o"),        # less the row
    "g1_b2": dict(BASE, **ROWS, **FLOOR, FULL_BODY="2"),                             # ADDENDUM1.md: bodies up to 2
    "g1_s1": dict(BASE, **ROWS, **FLOOR, SEED="1"),                                  # ADDENDUM1.md: another seed
    "g1_s2": dict(BASE, **ROWS, **FLOOR, SEED="2"),                                  # ADDENDUM2.md: more seeds
    "g1_s3": dict(BASE, **ROWS, **FLOOR, SEED="3"),
    "g2_m3": dict(BASE, **ROWS, LAWS="lt;red;add;mul;acc.mul", PATTERNS="pair;x,o;o,x,d", MIN_ROWS="3",   # ADDENDUM2.md
                  SECONDS="3600", SEARCH_S="2400"),
}
CHECKS = {                                                                           # the tiny tests (resume.py)
    "rs_b3": dict(MODULE="bar3.scale.legalclimb.resume", MODE="level", IDCHECK="1", ANSWERS="780", SECONDS="2400"),
    "rs_b2": dict(MODULE="bar3.scale.legalclimb.resume", MODE="level", IDCHECK="1", ANSWERS="780", SECONDS="2400",
                  FULL_BODY="2"),
}

image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_file(CPU / "integration" / "__init__.py", "/root/cpu_first/integration/__init__.py")
         .add_local_dir(CPU / "integration" / "lib", "/root/cpu_first/integration/lib",
                        ignore=["records", "__pycache__", "logs", "*.out", "*.json", "*.txt"])
         .add_local_file(CPU / "bar3" / "__init__.py", "/root/cpu_first/bar3/__init__.py")
         .add_local_file(SCALE / "__init__.py", "/root/cpu_first/bar3/scale/__init__.py")
         .add_local_dir(SCALE / "compose2", "/root/cpu_first/bar3/scale/compose2",
                        ignore=["logs", "records", "__pycache__", "*.md"])
         .add_local_dir(SCALE / "slice", "/root/cpu_first/bar3/scale/slice",
                        ignore=["records", "logs", "__pycache__", "*.md"])
         .add_local_file(SCALE / "slice" / "records" / "s4_x2.json",
                         "/root/cpu_first/bar3/scale/slice/records/s4_x2.json")
         .add_local_dir(SCALE / "legal", "/root/cpu_first/bar3/scale/legal",
                        ignore=["records", "logs", "__pycache__", "*.md"])
         .add_local_dir(HERE, "/root/cpu_first/bar3/scale/legalclimb", ignore=["records", "logs", "__pycache__", "*.md"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}",
                        ignore=["test.jsonl", "ood_*.jsonl", "depth_*.jsonl"]))
app = modal.App("bar3-legalclimb", image=image)
vol = modal.Volume.from_name("bar3-legalclimb", create_if_missing=True)


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


@app.function(gpu="B200", timeout=TIMEOUT_S, memory=MEMORY, volumes={"/out": vol}, max_containers=MAX)
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
