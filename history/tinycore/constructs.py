"""Symmetry breaking by construction at bases 3-5: the shared subtractor (cell="mult": slot s subtracts the exact
digits of s*N) and repeated subtraction (cell="repeat"), joint from answers, with and without the staged curriculum.
Rows: runs/constructs.jsonl.

    python -m tinycore.constructs
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

from .fixes import HERE, run_fix

SIZES = [(3, 2, 1), (3, 2, 2), (4, 2, 1), (4, 2, 2), (5, 2, 1), (5, 2, 2), (3, 3, 2)]


def jobs(seeds=6):
    out = []
    for B, w, k in SIZES:
        for cell in ("mult", "repeat"):
            for ch in ("est", "borrow"):
                for fix in (("joint", "staged") if k > 1 else ("joint",)):
                    for s in range(seeds):
                        out.append({"fix": fix, "label": f"{cell}+{fix}", "cfg": {"B": B, "w": w, "k": k, "choice": ch,
                                    "cell": cell, "init": 0.1}, "seed": s, "kw": {}})
    return out


def run(job):
    r = run_fix(dict(job, kw=dict(job["kw"])))
    r["label"] = job["label"]
    return r


if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex, open(os.path.join(HERE, "runs", "constructs.jsonl"), "a") as f:
        for r in ex.map(run, jobs()):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{r['label']:14s} B{r['B']} w{r['w']} k{r['k']} {r['cell']}/{r['choice']} seed {r['seed']}: hit "
                  f"{r['hit']} final {r['final']:.3f} slots {r['slots']} ({r['seconds']}s)", flush=True)
