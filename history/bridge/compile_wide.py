"""The widened compiles of the Net2Net arm (BRIDGE.md §7), each checked exact on every question.

    ~/github/sotaku/venv/bin/python -m bridge.compile_wide

Each case is a function-preserving widening of the compiled machine: head copies (mean or side by side), Net2Net
duplication of the MLP's hidden units, the prompt only at the first tick (prompt cells holding themselves), and the
free-symbol format with one or four symbols per meaning. Checked like compile.py: the hard path, the soft path, and
from random scratch states, at (3, 2, 1), (3, 2, 2) and (4, 2, 2). Writes bridge/records/compile_wide.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from bridge import compile as Cm
from bridge import net as Nn

HERE = Path(__file__).resolve().parent

# name: (net.Cfg fields at B = 3, Net2Net widening of the MLP's units)
CASES = {
    "free attention + MLP, 4 head copies (mean), units x4": (dict(attn="free", rule="mlp", copies=4, hidden=216), 4),
    "free attention + MLP, 4 head copies (side by side)": (dict(attn="free", rule="mlp", copies=4, copy_mode="concat"), 1),
    "free attention + tables, 4 head copies": (dict(attn="free", rule="table", learn=(), copies=4), 1),
    "menu + tables, 4 head copies": (dict(attn="menu", rule="table", learn=(), copies=4), 1),
    "free attention + tables, prompt at the first tick only": (dict(attn="free", rule="table", learn=(), inject="first"), 1),
    "free attention + MLP, prompt at the first tick only": (dict(attn="free", rule="mlp", inject="first", hidden=63), 1),
    "free symbols, K = 6": (dict(attn="free", rule="mlp", fmt="symbols", K=6), 1),
    "free symbols, K = 24 (4 per meaning), 4 head copies side by side, units x4":
        (dict(attn="free", rule="mlp", fmt="symbols", K=24, copies=4, copy_mode="concat", hidden=216), 4),
}


def scaled(kw, B):
    """The same case at base B: hidden units and symbols grow with the tables."""
    kw = dict(kw)
    units = 6 * B * B + (3 * B if kw.get("inject") == "first" else 0)
    widen = 4 if kw.get("hidden") in (216,) else 1
    if kw.get("hidden"):
        kw["hidden"] = units * widen
    if kw.get("fmt") == "symbols":
        kw["K"] = 2 * B * (kw["K"] // 6)
    return kw


def build(kw, widen, B):
    net = Nn.Net(Nn.Cfg(**scaled(kw, B)), B)
    if net.cfg.attn == "free":
        Cm.set_free(net)
    if net.cfg.attn == "menu":
        Cm.set_menu(net)
    if net.cfg.fmt == "symbols":
        Cm.set_symbols(net, widen=widen)
    elif net.cfg.rule == "mlp":
        Cm.set_mlp(net, widen=widen)
    return net


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    rec = {}
    for name, (kw, widen) in CASES.items():
        rows = {}
        for (B, w, k) in [(3, 2, 1), (3, 2, 2), (4, 2, 2)]:
            net = build(kw, widen, B)
            r = Cm.check(net, B, w, k)
            r2 = Cm.check(net, B, w, k, rand_init=True, seed=3)
            rows[f"{B},{w},{k}"] = dict(hard=r["hard"]["exact"], soft=r["soft"]["exact"],
                                       min_p=r["soft"]["min_p_answer"], hard_from_random_scratch=r2["hard"]["exact"],
                                       params=sum(p.numel() for p in net.parameters()))
        rec[name] = rows
        print(name, "|", " | ".join(f"({s}) hard {v['hard']:.3f} soft {v['soft']:.3f} min p {v['min_p']:.4f} "
                                    f"random start {v['hard_from_random_scratch']:.3f}" for s, v in rows.items()),
              flush=True)
    rec["seconds"] = time.time() - t0
    out = HERE / "records" / "compile_wide.json"
    out.write_text(json.dumps(rec, indent=1))
    print(f"{rec['seconds']:.0f} s; wrote {out}")


if __name__ == "__main__":
    main()
