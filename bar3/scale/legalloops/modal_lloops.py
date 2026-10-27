"""This study's runs on Modal (LEGALLOOPS.md §4): round 4's runner (bar3/scale/slice/modal_slice.py) on this study's
own app and Volume (`bar3-legalloops`: runs/<TAG>.log, runs/<TAG>.json), the copies here (core_k), at most MAX B200s at
once so that the spend watcher's margin of one hour of the containers' cost holds; arms spawned together queue.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/legalloops/modal_lloops.py --arms k1,k0
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/legalloops/modal_lloops.py --mode fetch

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
MAX = int(__import__("os").environ.get("LEGALLOOPS_MAX", 6))   # at most this many B200s at once

# Round 4's settings (SELF_PRACTICE.md §7's documented run: NP 8, deep probes over whole concepts), the concept
# rule, 780 answer rows (the most a full-size state can hold: 8,192 words), 2,700 s.
BASE = dict(MODULE="bar3.scale.legalloops.core_k", ROUND_TOO="1", RULE="lpca", NP="8", DEEP_BY="bindings",
            DEEP_FOCUS="closure", LEVELS="10", SECONDS="2700", ANSWERS="780", SPELL="16")
ROWS = dict(TARGET="rows", SOURCES="law,value,rest", LAWNEAR="1", SUBGOAL="1", REST_DEEP="1", CLEAN_VALUES="1",
            ROWCHECK="1", LOOK_COST="3")                       # bar 3's sources on the answer rows, at full size
BOOK_MIN = dict(LAWS="lt;red;add;mul;acc.mul", PATTERNS="pair;x+on;x,o;o,x,d")
LESS2 = dict(LAWS="lt;red;add;mul", PATTERNS="pair;x,o")                          # book_min less its reduction and row
LESS3 = dict(LAWS="lt;red;add", PATTERNS="pair")                                  # and less the multiply too
ARMS = {
    "k1": dict(BASE, TARGET="round", SOURCES="law", **LESS3, READ="1"),               # the cuts and loops read legally
    "k2": dict(BASE, **ROWS, **LESS3, READ="1", FULL_BODY="2"),                      # and test 3's sources too
    "k3": dict(BASE, TARGET="round", SOURCES="law", **LESS2, READ="1"),              # the multiply in the book
    "k0": dict(BASE, **ROWS, **LESS3, FULL_BODY="2"),                                # control: no reading
    "c1": dict(BASE, TARGET="round", SOURCES="law", **BOOK_MIN),                     # control: book_min alone
    "k1_s1": dict(BASE, TARGET="round", SOURCES="law", **LESS3, READ="1", SEED="1"),  # another seed
    "k1_less_add": dict(BASE, TARGET="round", SOURCES="law", LAWS="lt;red", PATTERNS="pair", READ="1"),  # less add
}
CHECKS = {                                                                           # the tiny test: k1's first levels
    "t_k1": dict(BASE, TARGET="round", SOURCES="law", **LESS3, READ="1", LEVELS="5"),
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
         .add_local_dir(SCALE / "legal", "/root/cpu_first/bar3/scale/legal",
                        ignore=["records", "logs", "__pycache__", "*.md"])
         .add_local_dir(HERE, "/root/cpu_first/bar3/scale/legalloops", ignore=["records", "logs", "__pycache__", "*.md"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}",
                        ignore=["test.jsonl", "ood_*.jsonl", "depth_*.jsonl"]))
app = modal.App("bar3-legalloops", image=image)
vol = modal.Volume.from_name("bar3-legalloops", create_if_missing=True)


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
