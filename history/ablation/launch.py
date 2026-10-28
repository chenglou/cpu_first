"""Spawn a plan's jobs on the deployed `ablation` app (modal_ablate.py); the calls run on Modal whether or not this
client stays up, under the deployed functions' container caps.

    ~/github/sotaku/venv/bin/python launch.py plans/<plan>.json [--dry]

A plan is a JSON list of jobs {"name", "file" (default submission_ablate.py; relative to this folder or absolute),
"set": {CONSTANT: python source} (rewrites `CONSTANT = ...` lines, as modal_chain.py's --set does), "seed": int
(a CPU job's manifest: device cpu and this seed, everything else official), "gpu": true (the untouched official
manifest file on an H100; no seed, no overrides)}. Appends the call ids to runs/launch_<plan>.log.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def variant(src: str, sets: dict) -> str:
    for key, value in sets.items():
        src, count = re.subn(rf"^{key} = .*$", f"{key} = {value}", src, count=1, flags=re.M)
        if count != 1:
            raise KeyError(key)
    return src


def jobs_of(plan: Path) -> list:
    out = []
    for j in json.loads(plan.read_text()):
        path = Path(j.get("file", "submission_ablate.py"))
        path = path if path.is_absolute() else HERE / path
        src = variant(path.read_text(), j.get("set", {}))
        overrides = None if j.get("gpu") else {"runtime": {"device": "cpu", "seeds": [j["seed"]]}}
        meta = {"plan": plan.name, "file": str(path.relative_to(HERE.parent)) if HERE.parent in path.parents else str(path),
                "set": j.get("set", {}), "seed": j.get("seed"), "gpu": bool(j.get("gpu"))}
        out.append((j["name"], src, overrides, meta))
    return out


def main(argv):
    plan = Path(argv[0]).resolve()
    jobs = jobs_of(plan)
    if "--dry" in argv:
        for name, src, overrides, meta in jobs:
            print(name, meta, len(src))
        return
    import modal
    run_cpu = modal.Function.from_name("ablation", "run_cpu")
    run_h100 = modal.Function.from_name("ablation", "run_h100")
    log = HERE / "runs" / f"launch_{plan.stem}.log"
    log.parent.mkdir(exist_ok=True)
    with open(log, "a") as f:
        for name, src, overrides, meta in jobs:
            call = (run_h100 if meta["gpu"] else run_cpu).spawn(name, src, overrides, meta)
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} spawned {name} {call.object_id} {json.dumps(meta)}"
            print(line, flush=True)
            f.write(line + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
