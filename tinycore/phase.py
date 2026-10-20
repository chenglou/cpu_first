"""The size sweep: joint training from answers (minibatches of 256 from every example, Adam), for each base B,
digits of N (w) and quotient digits (k), with microlab's two choices. Rows: runs/phase.jsonl.

    python -m tinycore.phase
"""
import itertools
import os

from .sweep import HERE, run_jobs

SIZES = [(2, w, k) for w in (3, 4, 5) for k in (1, 2, 3, 4)] + \
        [(3, w, k) for w in (2, 3) for k in (1, 2, 3)] + [(4, 2, k) for k in (1, 2, 3)] + [(5, 2, k) for k in (1, 2)]


def jobs(seeds=6, steps=4000, choices=("est", "borrow"), sizes=SIZES, extra=None, lr=0.05, init=0.1):
    out = []
    for (B, w, k), ch in itertools.product(sizes, choices):
        for s in range(seeds):
            cfg = {"B": B, "w": w, "k": k, "choice": ch, "cell": "slot", "init": init, **(extra or {})}
            tr = {"steps": steps, "lr": lr, "every": 100, "opt": "adam", "batch": 256}
            out.append({"cfg": cfg, "train": dict(tr), "train_orig": dict(tr), "seed": s})
    return out


if __name__ == "__main__":
    run_jobs(jobs(), os.path.join(HERE, "runs", "phase.jsonl"), workers=6)
