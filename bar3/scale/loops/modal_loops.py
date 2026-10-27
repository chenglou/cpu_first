"""This study's runs on Modal (LOOPS.md §4): round 4's runner (bar3/scale/slice/modal_slice.py) on this study's own
app and Volume (`bar3-loops`: runs/<TAG>.log, runs/<TAG>.json), with rounds 1 to 4's code in the image, the copies
here (core_l). Phase 1 ran at most 2 B200s at once; from phase 2 on, at the coordinator's word (budget soft $60,
cap $80, up to about 10 GPUs at once where arms are independent), at most 8 an app (max_containers), the spend
watcher's total the cap less one hour of every container that can run at once.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/loops/modal_loops.py --arms l5_x2,replay_s4_x2
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/loops/modal_loops.py --mode fetch

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
TIMEOUT_S = 9000                    # one container at most: an arm's SECONDS (2,400; 7,200 for addendum 2's long arms) and its overrun
MEMORY = (16384, 65536)             # request, hard limit (MiB)

# Round 3 and 4's settings, on this study's copy: the small target and the round's search each level, the concept rule,
# SELF_PRACTICE.md §7's documented run (NP 8, deep probes over whole concepts), 2,400 s.
BASE = dict(MODULE="bar3.scale.loops.core_l", TARGET="small", ROUND_TOO="1", RULE="lpca", NP="8",
            DEEP_BY="bindings", DEEP_FOCUS="closure", LEVELS="10", SECONDS="2400")
BOOK_MIN = dict(LAWS="lt;red;add;mul;acc.mul", PATTERNS="pair;x+on;x,o;o,x,d")
LESS_RED = dict(LAWS="lt;red;add;mul;acc.mul", PATTERNS="pair;x,o;o,x,d")          # book_min less x + o N
LESS_RED_ROW = dict(LAWS="lt;red;add;mul", PATTERNS="pair;x,o")                   # and less o,x,d (the row)
BAR3 = "restrict,project,value,rest"
FULL = dict(CLEAN_VALUES="1", SUBGOAL="1", LAWNEAR="1", ROWCHECK="1", TIEBREAK="1", RESTRICT_ROWS="1", SPELL="16",
            REST_DEEP="1")                                                          # round 2's B2rd
R1 = dict(BASE, SOURCES="law," + BAR3, **LESS_RED_ROW, **FULL, TIEFOLLOW="1")       # round 3's c3_r1
S2 = dict(R1, SLICE="1")                                                         # round 4's s4_x2
LESS_RED_ROW_MUL = dict(LAWS="lt;red;add", PATTERNS="pair")                       # book_min less x+on, o,x,d and x,o
L2 = dict(BASE, SOURCES="law," + BAR3, **LESS_RED_ROW_MUL, **FULL, TIEFOLLOW="1", SLICE="1", LOOPS="1")
L3 = dict(S2, LOOPS="1")                                                         # R1's book, the cuts and their loops
SEARCH_FIXES = ("SUBGOAL", "LAWNEAR", "REST_DEEP", "SPELL", "TIEBREAK", "TIEFOLLOW")
ARMS = {
    "l5_x2": L2,                                                                 # phase 1 (PREREG.md), in order
    "replay_s4_x2": S2,
    "l5_x3": dict(L3, TARGET_FN="x3"),
    "l5_10x2p1": dict(L3, TARGET_FN="10x2p1"),
    "l5_x2_bare": {k: v for k, v in L2.items() if k not in SEARCH_FIXES},        # phase 2, if l5_x2 gets a round
    "l5_x2_less_add": dict(L2, LAWS="lt;red"),
    "l5_x2_less_link": dict(L2, LAWS="lt"),
    "l5_x2_lessons": dict({k: v for k, v in L2.items() if k not in ("SUBGOAL", "REST_DEEP")}, RULE="ct2a2"),
}
BARE = ARMS["l5_x2_bare"]
ARMS.update({                                                                    # addendum 1 (PREREG_ADDENDUM1.md)
    "l5_x2_no_book": dict(L2, SOURCES=BAR3),
    "l5_x2_all_off": {k: v for k, v in BARE.items() if k not in ("CLEAN_VALUES", "ROWCHECK", "RESTRICT_ROWS")},
    "l5_x2_bare_no_redloop": dict(BARE, LOOPS="mul"),
    "l5_x3_less_mul": dict(L2, TARGET_FN="x3"),
    "l5_10x2p1_less_mul": dict(L2, TARGET_FN="10x2p1"),
})
ARMS.update({                                                                    # addendum 2 (PREREG_ADDENDUM2.md)
    **{f"s4_x2_no_{_k.lower()}": {k: v for k, v in S2.items() if k != _k} for _k in ("SUBGOAL", "LAWNEAR", "SPELL")},
    "l5_x3_long": dict(L3, TARGET_FN="x3", SECONDS="7200"),
    "l5_x3_less_mul_long": dict(L2, TARGET_FN="x3", SECONDS="7200"),
    "l5_10x2p1_less_mul_long": dict(L2, TARGET_FN="10x2p1", SECONDS="7200"),
    "l5_x2_lessons_long": dict(ARMS["l5_x2_lessons"], SECONDS="7200"),
})
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
         .add_local_dir(CPU / "bar3" / "scale" / "compose3", "/root/cpu_first/bar3/scale/compose3",
                        ignore=["logs", "records", "__pycache__", "*.md"])
         .add_local_dir(CPU / "bar3" / "scale" / "slice", "/root/cpu_first/bar3/scale/slice",
                        ignore=["logs", "records", "__pycache__", "*.md"])
         .add_local_dir(HERE, "/root/cpu_first/bar3/scale/loops", ignore=["records", "logs", "__pycache__", "*.md"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("bar3-loops", image=image)
vol = modal.Volume.from_name("bar3-loops", create_if_missing=True)


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
    assert names and all(a in ARMS or a in CHECKS for a in names), (names, sorted(ARMS), sorted(CHECKS))
    for a in names:
        env = dict(ARMS.get(a) or CHECKS[a], TAG=a)
        call = run_b200.spawn(env)
        print("spawned", a, call.object_id, time.strftime("%H:%M:%S"), json.dumps(env), flush=True)
