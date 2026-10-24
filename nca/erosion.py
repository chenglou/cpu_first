"""Where does a rule that kept its training answers go wrong one size up? (§10)

    ~/github/sotaku/venv/bin/python -m nca.erosion [suite/name ...]      (default: rest_stab/b4_sgd0.01_s0)

From a record's saved weights (a run started at the compiled rule), along the compiled machine's own trajectory at
each size (blank scratch, the hard path, 1.5x the depth's ticks): one tick of the trained rule from each of the
machine's states, and the written cells whose next state differs from the machine's. A cell's situation is what the
causal rule reads: the kinds of its 3 x 3 neighbourhood and the states of the cells up-left, up, up-right, left and
right. At the training size and one size up: how many cell-ticks meet a situation that never occurs on the machine's
trajectory at the training size, how the one-tick errors split between situations met and never met, and the trained
rule's share of answers right when run freely. (3, 2, 1) records only, (3, 3, 2) on a sample of 200 questions.
Writes nca/records/erosion.json (about 30 seconds a record; under 1 GB).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from nca import grid as G
from nca import train as T

HERE = Path(__file__).resolve().parent
READ = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1)]    # the causal rule's state reads (it sees all 9 kinds)
SIZES = {(3, 2, 1): (((3, 2, 2), 0), ((3, 3, 2), 200))}


@torch.no_grad()
def walk(trained: T.Run, machine: T.Run, B, w, k, sample=0):
    """Every written cell-tick of the machine's trajectory at (B, w, k): its situation (bytes) and whether one tick
    of the trained rule from the machine's state gets the cell's next state wrong; and the trained rule's share of
    answers right, run freely."""
    fmt = machine.net.fmt
    snap = lambda S: np.stack([S[:, fmt.sl(n)].argmax(1).numpy() for n, _ in fmt.chans], 1)     # [X, nch, R, C]
    lay = machine.layout(B, w, k)
    data = G.Data(B, w, k, sample=sample, seed=1)
    S0 = machine.net.prompt_state(lay, data.Nd, data.vd, "blank", torch.Generator().manual_seed(0))
    ticks = lay.depth + lay.depth // 2
    _, _, hist = machine.net.run(lay, S0, ticks, mode="hard", record=True)
    states = [S0] + hist
    kind = np.full((lay.R + 2, lay.C + 2), G.OUT)
    kind[1:-1, 1:-1] = lay.kind
    scratch = ~np.asarray(lay.prompt)
    keys, errs = [], []
    for t in range(ticks):
        pad = np.full((data.X, len(fmt.chans), lay.R + 2, lay.C + 2), -1, dtype=np.int8)
        pad[:, :, 1:-1, 1:-1] = snap(states[t])
        bad = (snap(trained.net.run(lay, states[t], 1, mode="hard")[0]) != snap(states[t + 1])).any(1)
        for r, c in zip(*np.nonzero(scratch)):
            kp = bytes(np.array([kind[1 + r + dr, 1 + c + dc] for dr, dc in G.OFFS], dtype=np.int8))
            nb = np.ascontiguousarray(np.concatenate([pad[:, :, 1 + r + dr, 1 + c + dc] for dr, dc in READ], 1))
            keys += [kp + row.tobytes() for row in nb]
            errs.append(bad[:, r, c])
    S, _ = trained.net.run(lay, S0, lay.depth, mode="hard")
    right = float((trained.answer(lay, S).argmax(-1).numpy() == data.yd).all(-1).mean())
    return keys, np.concatenate(errs), right, int(data.X)


def erosion(rec):
    trained = T.load_state(rec)
    machine = T.Run(trained.cfg)                  # the same cfg, nothing trained: the run's start, the compiled rule
    c = trained.cfg
    keys, err, right, X = walk(trained, machine, c.B, c.w, c.k)
    seen = set(keys)
    out = {"steps": rec["steps"], "train": {"size": [c.B, c.w, c.k], "questions": X, "right": right,
                                            "cell_ticks": len(keys), "errors": int(err.sum()),
                                            "situations": len(seen)}}
    for size, sample in SIZES[(c.B, c.w, c.k)]:
        keys, err, right, X = walk(trained, machine, *size, sample=sample)
        new = np.array([key not in seen for key in keys])
        out[",".join(map(str, size))] = {
            "questions": X, "every": not sample, "right": right, "cell_ticks": len(keys),
            "never_met": round(float(new.mean()), 4), "errors": int(err.sum()),
            "errors_never_met": int(err[new].sum()), "error_rate_met": round(float(err[~new].mean()), 6),
            "error_rate_never_met": round(float(err[new].mean()), 6)}
    return out


def main(*names):
    out_path = HERE / "records" / "erosion.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names or ("rest_stab/b4_sgd0.01_s0",):
        suite, run = name.split("/")
        rec = next(r for r in map(json.loads, open(HERE / "runs" / f"{suite}.jsonl")) if r["name"] == run)
        out[name] = erosion(rec)
        print(name, json.dumps(out[name]), flush=True)
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
