"""census.py over a wide menu on Modal: the wiring indices split into ranges, one CPU container per range (at most 16).

    ~/github/sotaku/venv/bin/python -m modal run bridge/modal_census.py --extra 6      # 13 options: 13^7 wirings

Writes bridge/records/census_<options>.json (the merged counts and the wirings that pass both).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "grid.py", "net.py", "census.py"):
    image = image.add_local_file(HERE / _f, f"/root/bridge/{_f}")
app = modal.App("bridge-census", image=image)


@app.function(cpu=4, memory=4096, timeout=3 * 3600, max_containers=16, retries=modal.Retries(max_retries=1))
def census_range(extra: int, lo: int, hi: int) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import torch
    torch.set_num_threads(4)
    from bridge.census import run_range
    t0 = time.time()
    r = run_range(extra, lo, hi)
    r["seconds"] = time.time() - t0
    return r


@app.local_entrypoint()
def main(extra: int = 6, parts: int = 12) -> None:
    import sys
    sys.path.insert(0, str(HERE.parent))
    from bridge.census import PORTS, merge
    n = 7 + extra
    total = n ** len(PORTS)
    edges = [total * i // parts for i in range(parts + 1)]
    t0 = time.time()
    results = list(census_range.starmap([(extra, edges[i], edges[i + 1]) for i in range(parts)]))
    rec = merge(results)
    rec["seconds"] = time.time() - t0
    rec["container_seconds"] = sum(r["seconds"] for r in results)
    out = HERE / "records" / f"census_{n}.json"
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["counts"]), f"{rec['seconds']:.0f}s wall -> {out}", flush=True)
