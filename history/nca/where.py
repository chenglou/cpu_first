"""Where does a saved rule miss? Every question of a size, then a look at the wrong ones.

    ~/github/sotaku/venv/bin/python -m nca.where cover2 w3_causal_trace_s7 3 4 3

From a record that kept its weights: the hard path on every question of (B, w, k); for the questions it gets wrong,
their moduli; whether the rule is right on them one tick from the machine's own state (every cell, every channel);
how many it gets right from random scratch states and with 4x the ticks; and the first wrong cells of the first one.
Writes nca/records/where_<name>_<B><w><k>.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from nca import grid as G
from nca import train as T

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def where(rec, B, w, k):
    r = T.load_state(rec)
    lay = r.layout(B, w, k)
    data = G.Data(B, w, k)
    bad = []
    for i in range(0, data.X, 2048):
        d = data.subset(np.arange(i, min(i + 2048, data.X)))
        S, _ = r.net.run(lay, r.net.prompt_state(lay, d.Nd, d.vd), lay.depth, mode="hard")
        wrong = np.where(~(r.answer(lay, S).argmax(-1).numpy() == d.yd).all(-1))[0]
        bad += [(int(d.N[j]), int(d.v[j])) for j in wrong]
    out = {"record": rec["name"], "size": [B, w, k], "questions": int(data.X), "wrong": len(bad),
           "wrong_moduli": dict(Counter(n for n, _ in bad)), "wrong_questions": bad[:50]}
    if not bad:
        return out
    d = data.like(np.array([b[0] for b in bad]), np.array([b[1] for b in bad]))
    Sr, ref = r.ref_state(lay, d)
    Z = r.net.logits(Sr, lay, hard=True)
    tf_wrong = 0
    for ch in [n for n, _ in r.fmt.chans if n in ref]:
        got = Z[:, r.fmt.sl(ch)].argmax(1).numpy()
        tf_wrong += int((got != ref[ch])[:, 1:].sum())
    out["cells_wrong_one_tick_from_the_machine"] = tf_wrong
    starts = {}
    for scratch, mult in (("blank", 4), ("random", 1), ("random", 4)):
        S, _ = r.net.run(lay, r.net.prompt_state(lay, d.Nd, d.vd, scratch, torch.Generator().manual_seed(3)),
                         mult * lay.depth, mode="hard")
        starts[f"{scratch} x{mult}"] = int((r.answer(lay, S).argmax(-1).numpy() == d.yd).all(-1).sum())
    out["right_among_them"] = starts
    S, _ = r.net.run(lay, r.net.prompt_state(lay, d.Nd, d.vd), lay.depth, mode="hard")
    rows = [n if t < 0 else f"{n}({t},{s})" for n, t, s in lay.rows]
    wrong = r.net.channel(S, "d").argmax(1).numpy() != ref["d"]
    out["first_wrong_cells_of_the_first"] = [[rows[a], int(b)] for a, b in np.argwhere(wrong[0])][:6]
    return out


def main(suite, name, B, w, k):
    rec = next(json.loads(l) for l in open(HERE / "runs" / f"{suite}.jsonl") if json.loads(l)["name"] == name)
    out = where(rec, int(B), int(w), int(k))
    (HERE / "records" / f"where_{name}_{B}{w}{k}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k_: v for k_, v in out.items() if k_ != "wrong_questions"}))


if __name__ == "__main__":
    main(*sys.argv[1:6])
