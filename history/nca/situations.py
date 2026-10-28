"""How new are a bigger grid's local situations? Counted on the compiled machine's settled states.

    ~/github/sotaku/venv/bin/python -m nca.situations

A cell's situation is what its rule reads, taken three ways:
- kinds: the 3 x 3 neighbourhood's kinds (the edge included);
- full: the kinds and every channel value of the 3 x 3 neighbourhood but the cell's own state (what a noself rule
  sees once the grid has settled);
- relevant: the cell's kind and only the values the compiled rule reads (grid.compiled_clauses), so a rule that
  ignores everything else meets nothing new.
For each bigger size, the share of its scratch cells (over every question, or a sample) whose situation never occurs
at the training size; and the same for the square's grid, trained at (3, 3). Writes nca/records/situations.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nca import grid as G
from nca.grid import MUL, OFFS, OUT, POS, SEL, SHIFT, SPLIT, SUB

HERE = Path(__file__).resolve().parent
CH = ("d", "n", "b", "m", "g")
# what the compiled rule reads, per kind (beyond the cell's own kind)
RELEVANT = {SHIFT: [("UL", "d"), ("U", "n")], SUB: [("U", "d"), ("U", "n"), ("L", "b")],
            SEL: [("U", "b"), ("U", "m"), ("U", "d"), ("R", "g"), ("R", "kind"), ("U", "n")],
            SPLIT: [("U", "d"), ("U", "n")],
            MUL: [("U", "d"), ("U", "a"), ("U", "q"), ("L", "s"), ("L", "c"), ("L", "kind"), ("UL", "a"), ("UR", "q"),
                  ("U", "n")]}
CH_SQ = CH + ("a", "q", "s", "c")


def situations(B, w, k, sample=0, task="div"):
    lay = G.Layout(B, w, k, task=task)
    data = G.SqData(B, w, sample=sample, seed=1) if task == "sq" else G.Data(B, w, k, sample=sample, seed=1)
    ref = G.machine_reference(lay, data)
    chs = CH_SQ if task == "sq" else CH
    R, C = lay.R, lay.C
    kind = np.full((R + 2, C + 2), OUT)
    kind[1:-1, 1:-1] = lay.kind
    vals = {ch: np.zeros((data.X, R + 2, C + 2), dtype=np.int64) for ch in chs}
    for ch in chs:
        vals[ch][:, 1:-1, 1:-1] = ref[ch]
    out = {"kinds": set(), "full": set(), "relevant": set()}
    cells = {"kinds": [], "full": [], "relevant": []}
    for r in range(1, R):
        for c in range(C):
            kd = lay.kind[r, c]
            kpat = tuple(int(kind[1 + r + dr, 1 + c + dc]) for dr, dc in OFFS)
            states = np.stack([vals[ch][:, 1 + r + dr, 1 + c + dc] for dr, dc in OFFS if (dr, dc) != (0, 0)
                               for ch in chs], 1)
            rel = []
            for pos, ch in RELEVANT[kd]:
                dr, dc = OFFS[POS[pos]]
                rel.append(np.full(data.X, kind[1 + r + dr, 1 + c + dc] == OUT) if ch == "kind" else
                           vals[ch][:, 1 + r + dr, 1 + c + dc])
            rel = np.stack(rel, 1)
            for i in range(data.X):
                f = (kpat, tuple(states[i]))
                g = (kd, tuple(rel[i]))
                out["full"].add(f)
                out["relevant"].add(g)
                cells["full"].append(f)
                cells["relevant"].append(g)
            out["kinds"].add(kpat)
            cells["kinds"] += [kpat] * data.X
    return out, cells


def main():
    train = (3, 2, 1)
    seen, _ = situations(*train)
    rec = {"train": train, "distinct_at_train": {k: len(v) for k, v in seen.items()}, "bigger": {}}
    for size in ((3, 2, 2), (3, 3, 2), (3, 3, 3), (3, 4, 3)):
        sample = 3000 if size[1] >= 3 else 0
        pats, cells = situations(*size, sample=sample)
        rec["bigger"][",".join(map(str, size))] = {
            k: {"distinct": len(pats[k]), "new_distinct": len(pats[k] - seen[k]),
                "cells_new": round(float(np.mean([c not in seen[k] for c in cells[k]])), 4)} for k in pats}
        print(size, rec["bigger"][",".join(map(str, size))], flush=True)
    seen2, _ = situations(3, 2, 2)
    seen3, _ = situations(3, 3, 2)
    for name, base in (("3,2,2", seen2), ("3,3,2", seen3)):
        rec[f"from_{name}"] = {}
        for size in ((3, 3, 3), (3, 4, 3)):
            pats, cells = situations(*size, sample=3000)
            rec[f"from_{name}"][",".join(map(str, size))] = {
                k: round(float(np.mean([c not in base[k] for c in cells[k]])), 4) for k in pats}
        print("trained at", name, rec[f"from_{name}"], flush=True)
    sq_seen, _ = situations(3, 3, 3, task="sq")
    rec["square_from_3,3"] = {}
    for w in (4, 5):
        pats, cells = situations(3, w, w, sample=3000, task="sq")
        rec["square_from_3,3"][f"3,{w}"] = {k: round(float(np.mean([c not in sq_seen[k] for c in cells[k]])), 4)
                                           for k in pats}
    print("square trained at (3, 3)", rec["square_from_3,3"], flush=True)
    (HERE / "records" / "situations.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
