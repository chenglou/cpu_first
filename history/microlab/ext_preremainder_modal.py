"""Run a list of ext_preremainder commands on Modal CPU containers, one container per command, all in parallel.

    modal run microlab/ext_preremainder_modal.py --commands runs/pre_remainder/remote.txt

Each line of the commands file is `<name> <ext_preremainder args>` (the sweep.sh format). Inside the container
`--threads` becomes 4 and `--device mps` becomes cpu. Rows are appended to microlab/results_preremainder.jsonl with
host "modal-cpu4", and each command's one-line summary goes to runs/pre_remainder/<name>.log.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "models.py", "tasks.py", "run.py", "ext_preremainder.py"):
    image = image.add_local_file(HERE / _f, f"/root/microlab/{_f}")
app = modal.App("microlab-preremainder", image=image)


@app.function(cpu=4, memory=4096, timeout=3 * 3600, retries=modal.Retries(max_retries=1))
def cell(name: str, argv: list[str]) -> dict:
    import sys
    sys.path.insert(0, "/root")
    from microlab import ext_preremainder as E
    argv = list(argv)
    for flag, value in (("--threads", "4"), ("--device", "cpu")):
        if flag in argv:
            argv[argv.index(flag) + 1] = value
        else:
            argv += [flag, value]
    rows = E.main(argv, log=None)
    for r in rows:
        r["host"] = "modal-cpu4"
    return {"name": name, "rows": rows, "lines": [E.line(r) for r in rows]}


@app.local_entrypoint()
def main(commands: str) -> None:
    jobs = [(l.split()[0], l.split()[1:]) for l in open(commands) if l.strip() and not l.startswith("#")]
    out_dir = HERE.parent / "runs" / "pre_remainder"
    for res in cell.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("failed:", repr(res)[:300], flush=True)
            continue
        with open(HERE / "results_preremainder.jsonl", "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in res["rows"])
        (out_dir / f"{res['name']}.log").write_text("\n".join(res["lines"]) + "\n")
        print(f"done {res['name']}: {res['lines'][-1][:200]}", flush=True)
