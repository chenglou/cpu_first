"""The one big run: exhaustive straight-line speculation at L=6 (x^8 mod N, 5-op ISA, 2^47 raw programs) on a Modal
CPU box, to measure where enumeration runs out instead of extrapolating from L <= 5.

    ~/github/tips-for-running-modal/venv/bin/python -m modal run microlab/ext_exact_modal.py --length 6
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.0.2")
    .add_local_dir(str(HERE), "/root/cpu_first/microlab",
                   ignore=["__pycache__", "runs_exact", "*.jsonl", "*.md"])
)
app = modal.App("exact-blocks-spec", image=image)


@app.function(cpu=32, memory=65536, timeout=3 * 3600)
def spec(isa: str, target: str, L: int, bits: int, workers: int, chunk: int) -> dict:
    import sys
    sys.path.insert(0, "/root/cpu_first")
    import numpy as np
    from microlab import ext_exact as E
    T = {"sq": 1, "p4": 2, "p8": 3}[target]
    rows = E.free_rows(1000, bits, np.random.default_rng(0), T=T)
    t0 = time.time()
    res = E.speculate(rows, E.ISAS[isa], L, K=8, workers=workers, chunk=chunk)
    surv = [(pr, m) for pr, m in res["survivors"]
            if (E.run_prog(pr, [rows["N"], rows["x"]]) == rows["y"]).all()]
    held = E.held_out_sets(np.random.default_rng(999), T=T)
    groups = E.check_survivors(surv, target, held) if surv else []
    names = ["N", "x", "0", "1", "2"]
    return {"isa": isa, "target": target, "L": L, "bits": bits, "workers": workers, "space": res["space"],
            "states": res["states"], "curve": res["curve"], "seconds": res["seconds"],
            "wall": time.time() - t0, "survivors_probe": len(res["survivors"]), "survivors_all_rows": len(surv),
            "groups": [{"correct": g["correct"], "raw": g["raw"], "n": g["n"], "acc": g["acc"],
                        "example": E.show(g["example"], names)} for g in groups]}


@app.local_entrypoint()
def main(isa: str = "basic", target: str = "p8", length: int = 6, bits: int = 16, workers: int = 30,
         chunk: int = 2048) -> None:
    L = length
    r = spec.remote(isa, target, L, bits, workers, chunk)
    out = HERE / "runs_exact" / f"modal_spec_{isa}_{target}_L{L}.json"
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "groups"}, indent=1))
    for g in r["groups"][:8]:
        print(("CORRECT" if g["correct"] else "alias  "), g["raw"], g["acc"], g["example"])
    with open(HERE / "results_exact.jsonl", "a") as fh:
        fh.write(json.dumps({"exp": "spec_modal", **r}) + "\n")
