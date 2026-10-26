"""The composition test's outcomes (COMPOSE.md §5), from the arms' records (records/<TAG>.json, fetched from the Volume):

  summary: each arm's levels: what it practised (its picks, each a problem of the law book or of bar 3's sources, with
           its cost and program), and each level's two searches (the small target, then the round at full size): the
           cost found, the programs, the small target's programs checked at full size;
  ladder:  a found program checked as the stage-3 studies check the round (SP_ABLATION.md §5.1: "T = 64 on both
           ladders, every rung 768/768"): H1's depth ladders, the seen moduli (depth_t_<T>) and the unseen ones
           (depth_ood_n_t_<T>), T = 1 to 64, each rung's 768 rows, the program applied T times at full size by the
           machine's exact cells. Neither ladder is among the rows the climb saw (train.jsonl).

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose.verify summary cp_book_min cp_bar3 ...
    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose.verify ladder cp_book_min [PROGRAM]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

from integration.lib import libcore as L

HERE = Path(__file__).resolve().parent
H1_DIR = os.environ.get("H1_DATA", str(HERE.parents[3] / "one-layer-deeper" / "data" / "generated" /
                                       "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
P = 16
RUNGS = (1, 2, 4, 8, 16, 32, 64)
BOOK = {"lt.pair": "the comparison", "red.pair": "the link", "add.pair": "the addition",
        "acc.mul.o,x,d": "the multiply-add row", "mul.x,o": "the multiply", "red.x+on": "the reduction"}


def load(tag):
    return json.load(open(tag if tag.endswith(".json") else HERE / "records" / f"{tag}.json"))


def kind(name):
    """What a practised problem is: one of the six pieces (the law book's), or which of bar 3's sources posed it."""
    law = name.split("/")[0]
    pat = name.split("/")[1].split("@")[0] if "/" in name else ""
    if f"{law}.{pat}" in BOOK:
        return BOOK[f"{law}.{pat}"]
    if law.startswith("val."):
        return "a near miss's value"
    if pat.startswith("rest."):
        return "the rest of the way from a near miss"
    if pat.startswith("rst"):
        return "a restriction of the target"
    if pat == "tgt":
        return f"a projection of the target ({law})" if law != "target" else "the target"
    return f"{law} on {pat}"


def summary(tags):
    for tag in tags:
        rec = load(tag)
        cfg = rec["config"]
        print(f"== {tag}: sources {cfg['sources']}, rule {cfg['rule']}, laws {cfg.get('laws')}, patterns "
              f"{cfg.get('patterns')}; {rec['seconds']} s, final {rec['final']}, rounds {rec['rounds'][:2]}")
        for h in rec["history"]:
            picks = [(p, h["learnable"].get(p, [None, None])) for p in h["picks"]]
            print(f"  L{h['level']}: pool {h['pool']} {h['counts']}")
            for p, (c, prog) in picks:
                print(f"     {p:34s} cost {c}  {prog:44s} {kind(p)}")
        for u in rec["units"]:
            if " target: " in u["line"] or " round: " in u["line"] or "done" in u["line"]:
                print(f"  {u['t']:7.1f}s {u['line'][:330]}")


def ladder_rows(fname):
    rows = [json.loads(l) for l in open(os.path.join(H1_DIR, fname))]
    num = lambda key: torch.tensor([[int(ch) for ch in reversed(str(r[key]))] + [0] * (P - len(str(r[key])))
                                    for r in rows])
    return num("modulus"), num("x"), num("result"), rows[0]["time_steps"]


def ladder(tag, prog_text=None, dev="cuda" if torch.cuda.is_available() else "cpu"):
    rec = load(tag)
    lib = {k: L.parse(v) for k, v in rec["lib"].items()}
    progs = [prog_text] if prog_text else rec["rounds"][:1]
    cells = L.Cells(10, dev)
    st = L.round_settings(cells, P)
    for text in progs:
        prog = L.parse(text)
        print(f"{tag}: {text}", flush=True)
        for name in ("depth_t", "depth_ood_n_t"):
            for T in RUNGS:
                t0 = time.time()
                Nd, Xd, Yd, T_ = ladder_rows(f"{name}_{T}.jsonl")
                assert T_ == T
                X = Xd.to(dev)
                for _ in range(T):
                    X, _c = L.run_prog(st, prog, lib, {"X": X, "N": Nd.to(dev)}, L.gen_on(dev, 1), dev)
                ok = (X.cpu() == Yd).all(1)
                print(f"  {name}_{T}: {int(ok.sum())}/{len(ok)} right ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(sys.argv[2:])
    elif sys.argv[1] == "ladder":
        ladder(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
