"""Stage 2 climbs on a Modal GPU (INTEGRATION.md): learned cells first (cells.py, optional), then the in-run library
climb (inclimb.Climb) on the torch search, one process, one GPU. Each run writes a JSON record (every search, every
level's merges and library, the final program and its H1-size check) to the Volume `integration` (climb/) and returns
it; --fetch copies them to integration/lib/records/.

    cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach integration/lib/modal_climb.py \
        --name c1 --config '{"budget": 3000000}'
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "library", "/root/cpu_first/library", ignore=["records", "inventing", "__pycache__"])
         .add_local_dir(CPU / "anchors", "/root/cpu_first/anchors", ignore=["records", "__pycache__", "*.md"])
         .add_local_file(CPU / "integration" / "__init__.py", "/root/cpu_first/integration/__init__.py")
         .add_local_dir(HERE, "/root/cpu_first/integration/lib", ignore=["records", "__pycache__", "logs"])
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("integration-climb", image=image)
vol = modal.Volume.from_name("integration", create_if_missing=True)


def _climb(name: str, config: dict) -> dict:
    """libcore.Climb (the code the stage-3 file inlines) in one process: optionally the cells learned first (cells.py:
    the subtraction by the anchors' recipe, the addition by sub.inverse, the multiply by mul.zero and mul.ladder; the
    climb starts only if every cell passes its own checks), then the climb, with the round searched against the T = 1
    answers whenever new routines appear."""
    import os
    import sys
    os.environ["H1_DATA"] = f"/root/data/{DATA}"
    sys.path.insert(0, "/root/cpu_first")
    import torch
    from integration.lib import libcore as L, inclimb
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    lines = []

    def log(s):
        lines.append(f"{time.time() - t0:7.1f}s {s}")
        print(lines[-1], flush=True)
    rows = inclimb.load_rows(config.get("rows", 4000), seed=config.get("seed", 0))
    Nd, Xd, _ = (t.to(dev) for t in inclimb.digits_of_rows(rows))
    ans = tuple(t.to(dev) for t in inclimb.digits_of_rows(inclimb.load_rows(config.get("labels", 400),
                                                                              seed=config.get("seed", 0) + 1, T=1)))
    cells = L.Cells(10, dev)
    cell_log = None
    if config.get("learned_cells"):
        from integration.lib import cells as cl
        tabs, cell_log = cl.learn(seed=config.get("seed", 0), log=log, **config.get("cell_args", {}))
        cells.set(**{k: v.long() for k, v in tabs.items()})
        log(f"cells learned (report only, against the truth): {cell_log.get('right_report')}")
        if cell_log.get("sbb_ok_at") is None or not cell_log.get("adc_own") or not cell_log.get("mul_own"):
            log("cells did not pass their own checks: the climb does not start (tables first)")
            out = {"name": name, "config": config, "device": dev, "seconds": round(time.time() - t0, 1),
                   "log": lines, "cells": cell_log, "climb": "not started"}
            Path("/out/climb").mkdir(parents=True, exist_ok=True)
            Path(f"/out/climb/{name}.json").write_text(json.dumps(out, indent=1, default=str))
            vol.commit()
            return out
    c = L.Climb(cells, dev, 16, budget=config.get("budget", 3_000_000),
                fallback_budget=config.get("fallback_budget", 16_000_000), max_body=config.get("max_body", 3),
                max_cost=config.get("max_cost", 7), min_count=config.get("min_count", 2.0),
                max_merges=config.get("max_merges", 12), levels=config.get("levels", 12), NP=config.get("NP", 8),
                round_budget=config.get("round_budget", 4_000_000), round_cost=config.get("round_cost", 5),
                chunk=config.get("chunk", 1 << 21))
    deadline = time.time() + config.get("seconds", 6 * 3600)
    while True:
        line = c.step((Nd, Xd), answers=ans if config.get("round", True) else None, deadline=deadline)
        if line is None:
            break
        log(line)
        if time.time() > deadline:
            log("time up")
            break
        if c.rounds and config.get("stop_at_round", True):
            log(f"round found: {[L.show(p) for p in c.rounds]}")
            break
    out = {"name": name, "config": config, "device": dev, "seconds": round(time.time() - t0, 1), "log": lines,
           "records": c.records, "history": c.history, "solved_at": c.solved_at,
           "lib": {k: L.show(v) for k, v in c.lib.items()},
           "corpus": {t: [L.show(p) for p in ps] for t, ps in c.corpus.items()}, "cells": cell_log,
           "rounds": [L.show(p) for p in c.rounds],
           "rounds_primitive": [L.show(L.expand(p, c.lib)) for p in c.rounds]}
    Path("/out/climb").mkdir(parents=True, exist_ok=True)
    Path(f"/out/climb/{name}.json").write_text(json.dumps(out, indent=1, default=str))
    vol.commit()
    return out


@app.function(gpu="H100", timeout=8 * 3600, volumes={"/out": vol}, max_containers=6)
def climb_gpu(name: str, config: dict) -> dict:
    return _climb(name, config)


@app.function(cpu=8, memory=32768, timeout=6 * 3600, volumes={"/out": vol}, max_containers=6)
def climb_cpu(name: str, config: dict) -> dict:
    return _climb(name, config)


def _script(module: str, env: dict) -> str:
    import os
    import subprocess
    e = dict(os.environ, H1_DATA=f"/root/data/{DATA}", PYTHONPATH="/root/cpu_first", **env)
    r = subprocess.run(["python", "-W", "ignore", "-m", module], capture_output=True, text=True, env=e,
                       cwd="/root/cpu_first")
    out = r.stdout + (r.stderr[-4000:] if r.returncode else "")
    Path("/out/climb").mkdir(parents=True, exist_ok=True)
    Path(f"/out/climb/script_{module.split('.')[-1]}_{env.get('TAG', 'x')}.log").write_text(out)
    vol.commit()
    return out


@app.function(cpu=8, memory=32768, timeout=4 * 3600, volumes={"/out": vol}, max_containers=6)
def script_cpu(module: str, env: dict) -> str:
    return _script(module, env)


@app.function(gpu="L40S", timeout=4 * 3600, volumes={"/out": vol}, max_containers=6)
def script_gpu(module: str, env: dict) -> str:
    return _script(module, env)


@app.local_entrypoint()
def main(name: str = "c1", config: str = "{}", cpu: bool = False, fetch: bool = False, script: str = "",
         env: str = "{}") -> None:
    rec_dir = HERE / "records"
    rec_dir.mkdir(exist_ok=True)
    if fetch:
        for entry in vol.listdir("climb"):
            dest = rec_dir / Path(entry.path).name
            with open(dest, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
            print("fetched", dest.name)
        return
    if script:
        call = (script_cpu if cpu else script_gpu).spawn(script, json.loads(env))
        print("spawned", script, call.object_id)
        return
    fn = climb_cpu if cpu else climb_gpu
    call = fn.spawn(name, json.loads(config))       # detached runs: the record lands on the Volume (--fetch)
    print("spawned", name, call.object_id)
