"""Run exact_wide.py analyses on Modal (one CPU container each, at most 3 at once):

    cd ~/github/one-layer/cpu_first
    ~/github/sotaku/venv/bin/python -m modal run wider_menus/modal_exact.py --jobs "W=3 div=3 mul=2 part=chain,io,mul;..."

Each job's JSON goes to the Volume `wider-menus` under exact/<name>.json and to wider_menus/runs/exact/<name>.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "wide.py", "tiny_wide.py", "exact_wide.py"):
    image = image.add_local_file(HERE / _f, f"/root/wider_menus/{_f}")
vol = modal.Volume.from_name("wider-menus", create_if_missing=True)
app = modal.App("wider-menus-exact", image=image)


@app.function(cpu=8, memory=32768, timeout=4 * 3600, volumes={"/out": vol}, max_containers=3)
def run_job(argv: str) -> dict:
    import contextlib
    import io
    import os
    import sys
    sys.path.insert(0, "/root")
    import torch
    torch.set_num_threads(8)
    from wider_menus.exact_wide import CFG, run
    cfg = dict(CFG)
    for a in argv.split():
        k, v = a.split("=", 1)
        cfg[k] = tuple(int(t) for t in v.split(",")) if isinstance(CFG[k], tuple) else type(CFG[k])(v)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = run(cfg)
    name = "_".join(a.replace("=", "").replace(",", "-") for a in argv.split())
    out["log"] = buf.getvalue()
    os.makedirs("/out/exact", exist_ok=True)
    Path(f"/out/exact/{name}.json").write_text(json.dumps(out, default=str))
    vol.commit()
    return {"name": name, "out": out}


@app.local_entrypoint()
def main(jobs: str) -> None:
    dest = HERE / "runs" / "exact"
    dest.mkdir(parents=True, exist_ok=True)
    for r in run_job.map(jobs.split(";"), order_outputs=False, return_exceptions=True, wrap_returned_exceptions=False):
        if isinstance(r, Exception):
            print("job failed:", repr(r)[:500], flush=True)
            continue
        (dest / f"{r['name']}.json").write_text(json.dumps(r["out"], indent=1, default=str))
        print(r["name"], r["out"]["log"][-3000:], flush=True)
