"""Check the certified rules further out, from their saved weights (records with `save_weights`).

    ~/github/sotaku/venv/bin/python -m modal run --detach nca/modal_nca.py --suite verify
    ~/github/sotaku/venv/bin/python -m modal run nca/modal_nca.py --suite verify --fetch
    ~/github/sotaku/venv/bin/python -m nca.verify            # collect runs/verify.jsonl into records/verify.json

For every record that kept its weights (suite cover2): the hard path (argmax states) on every question of each size
up to `FULL` questions, else a random sample; from blank scratch states at the compiled depth, and for three sizes
also from random scratch states and at 4x the ticks. Writes nca/records/verify.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nca import grid as G
from nca import train as T

HERE = Path(__file__).resolve().parent
SIZES = [(3, 3, 2), (3, 3, 3), (3, 4, 3), (3, 3, 4), (3, 4, 4), (3, 5, 4), (3, 5, 5)]
FULL = 30000                  # sizes with at most this many questions are checked on every one
SAMPLE = 3000


@torch.no_grad()
def check(r: T.Run, B, w, k, scratch="blank", mult=1, chunk=1024):
    lay = r.layout(B, w, k)
    n_all = sum(N * B ** k for N in range(max(2, B ** (w - 1)), B ** w))
    data = G.Data(B, w, k, sample=0 if n_all <= FULL else SAMPLE, seed=11)
    right = 0
    for i in range(0, data.X, chunk):
        d = data.subset(np.arange(i, min(i + chunk, data.X)))
        S0 = r.net.prompt_state(lay, d.Nd, d.vd, scratch, torch.Generator().manual_seed(i))
        S, _ = r.net.run(lay, S0, mult * lay.depth, mode="hard")
        got = r.answer(lay, S).argmax(-1).numpy()
        right += int((got == d.yd).all(-1).sum())
    return {"questions": int(data.X), "every": bool(data.X == n_all), "right": right, "exact": right == data.X}


def verify_record(rec):
    r = T.load_state(rec)
    res = {}
    t0 = time.time()
    for B, w, k in SIZES:
        for scratch, mult in (("blank", 1), ("random", 1), ("blank", 4)):
            if (scratch, mult) != ("blank", 1) and (w, k) not in ((3, 3), (4, 3), (4, 4)):
                continue
            key = f"{B},{w},{k} {scratch} x{mult}"
            res[key] = check(r, B, w, k, scratch, mult)
            print(rec["name"], key, res[key], f"{time.time() - t0:.0f}s", flush=True)
    return res


def jobs(suites=("cover2", "cover3")):
    """One Modal job per record that kept its weights."""
    out = []
    for s in suites:
        p = HERE / "runs" / f"{s}.jsonl"
        if p.exists():
            for l in open(p):
                rec = json.loads(l)
                if "weights" in rec:
                    out.append({"kind": "verify", "name": f"verify_{rec['name']}", "record": rec})
    return out


def main():
    rows = [json.loads(l) for l in open(HERE / "runs" / "verify.jsonl")]
    out = {r["name"]: r["checks"] for r in rows}
    (HERE / "records" / "verify.json").write_text(json.dumps(out, indent=1))
    for name, res in out.items():
        print(name, "all exact" if all(v["exact"] for v in res.values()) else
              "NOT exact: " + ", ".join(k for k, v in res.items() if not v["exact"]))


if __name__ == "__main__":
    main()
