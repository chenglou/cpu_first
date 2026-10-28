"""The wiring given (the true one), the table learned: which table recipes certify at (3, 2, 1) and (3, 2, 2).

    ~/github/sotaku/venv/bin/python -m tinyrouting.baseline
"""

from __future__ import annotations

import time

import torch

from tinyrouting.train import Cfg, Trainer

RECIPES = {
    "answers": dict(),
    "answers, own hard path": dict(own_path=True),
    "answers + cell": dict(checks={"cell": 1.0}),
    "answers + cell on own link inputs": dict(checks={"cellown": 1.0}),
    "answers + laws": dict(checks={"laws": 1.0}),
    "answers + range": dict(checks={"range": 1.0}),
    "cell alone (no answers)": dict(answers=0.0, checks={"cell": 1.0}),
}


def main(seeds=6):
    torch.set_num_threads(2)
    for B, w, k in [(3, 2, 1), (3, 2, 2)]:
        for name, kw in RECIPES.items():
            res, t = [], time.time()
            for seed in range(seeds):
                r = Trainer(Cfg(B=B, w=w, k=k, learn=(), param="given", steps=1500, seed=seed,
                                held=((B, w, k + 1), (B, w + 1, k + 1)), **kw)).train()
                res.append(r)
            fit = sum(r["hit"] is not None for r in res)
            cert = sum(r["certified"] for r in res)
            hits = sorted(r["hit"] for r in res if r["hit"] is not None)
            print(f"{(B, w, k)} {name:36s} train-exact {fit}/{seeds}, certified {cert}/{seeds}, steps {hits}, "
                  f"table rows right {[round(r['final']['rows'], 2) for r in res]} ({time.time() - t:.0f} s)",
                  flush=True)


if __name__ == "__main__":
    main()
