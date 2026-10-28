"""Step 3: one representative failing run per trap, probed against the compiled machine (probe.py).

    ~/github/sotaku/venv/bin/python -m bridge.diagnose [case ...]

Each case trains one notch with one lesson tier at (3, 2, 1) from random, for two seeds, then records the probes:
heads (argmax sources against the compiled ones), cells (every row's agreement with the machine's own values), path
(the loss on the straight line to the compiled weights), grad0 (where the first gradient pushes each head), and for
the free-symbol formats relabel (do the cells hold the machine's digits under a consistent renaming). Writes
bridge/records/diagnose.json (one entry per case and seed).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from bridge import notches as Nt
from bridge import probe as Pr
from bridge import train as T

HERE = Path(__file__).resolve().parent

CASES = {
    # wiring: a wide menu (7 relative rules per port) from the answers, the tables given
    "menu_answers": ("N2_menu_given", "A", {}),
    # wiring: 13 options per port, the tables given, with the one-link law
    "menu13_link": ("N2_rel_given", "AL", {"net": {"menu_extra": 6}}),
    # wiring: free attention, the tables given, with the one-link law
    "free_link": ("N3_free_given", "AL", {}),
    # wiring: free attention, grown one link at a time
    "free_link_staged": ("N3_free_given", "AL", {"link_stages": (1, 2)}),
    # rules: one ReLU MLP for all three kinds, fixed attention
    "mlp_all": ("N1_mlp", "GCL", {"lr": 0.01}),
    # rules: the subtraction cell an MLP, without bijectivity (the order laws only)
    "mlp_sub_nobij": ("N1_mlp_sub", "AC", {"lr": 0.01, "lessons": {"ans": 1.0, "pref": 1.0, "colall": 1.0,
                                                                   "rowall": 1.0}}),
    # rules: every rule a learned table (the copy and the select too), fixed attention
    "tables_all": ("N1_tables", "ACL", {}),
    # wiring: free attention with a 2-row locality window (plus N's row), the tables given, the one-link law
    "free_rows": ("N3_rows_given", "AL", {}),
    # wiring: free attention at lower learning rates
    "free_lr01": ("N3_free_given", "AL", {"lr": 0.01, "steps": 3000}),
    "free_lr003": ("N3_free_given", "AL", {"lr": 0.003, "steps": 3000}),
    # wiring: the 7-option menu by whole options, port by port (each option hard, the other ports soft)
    "menu_pmix": ("N2_menu_given", "A", {"pmix": True, "steps": 400}),
    # format: free symbols, free attention
    "symbols": ("N4_symbols", "GL", {"lr": 0.01}),
    # the bypass fixes (BRIDGE.md §8): the prompt only at the first tick; snapped states
    "free_first": ("N3_free_given_first", "AL", {}),
    "free_st": ("N3_free_given_st", "AL", {}),
}


def run_case(name, seed, steps=2000):
    notch, tier, extra = CASES[name]
    net = dict(Nt.NOTCHES[notch])
    net.update(extra.get("net", {}))
    kw = {k: v for k, v in extra.items() if k != "net"}
    steps = kw.pop("steps", steps)
    lessons = kw.pop("lessons", Nt.TIERS[tier])
    cfg = T.RunCfg(B=3, w=2, k=1, net=net, lessons=lessons, seed=seed, steps=steps, cert=True, **kw)
    r = T.Run(cfg)
    rec = {"case": name, "notch": notch, "tier": tier, "seed": seed, "cfg": cfg.__dict__}
    kinds = net.get("roles", "kinds") == "kinds"
    attn = net.get("attn") in ("menu", "free")
    if kinds and attn:
        rec["grad0_start"] = Pr.grad0(r)
    out = r.train()
    rec.update(hit=out["hit"], certified=out.get("certified"), train_hard=out["train_hard"],
               train_soft=out["train_soft"], bigger=out.get("bigger"), steps=out["steps"])
    for key in ("sub_digits_right", "sub_borrows_right", "shift_right", "sel_right"):
        if key in out:
            rec[key] = out[key]
    if kinds and attn:
        rec["heads"] = Pr.heads(r)
        rec["grad0_end"] = Pr.grad0(r)
    if net.get("fmt", "digits") == "digits" and kinds:
        rec["cells_hard"] = Pr.cells(r)
        rec["path"] = Pr.path(r)
    if net.get("fmt") == "symbols":
        rec["relabel"] = Pr.relabel(r)
    if net.get("inject") == "first":
        rec.update(Pr.hold(r))
    return rec


def main():
    torch.set_num_threads(2)
    names = sys.argv[1:] or list(CASES)
    path = HERE / "records" / "diagnose.json"
    path.parent.mkdir(exist_ok=True)
    recs = json.loads(path.read_text()) if path.exists() else []
    for name in names:
        for seed in (0, 1):
            t0 = time.time()
            rec = run_case(name, seed)
            rec["seconds"] = time.time() - t0
            recs = [x for x in recs if not (x["case"] == name and x["seed"] == seed)] + [rec]
            path.write_text(json.dumps(recs, indent=1, default=str))
            print(name, seed, f"cert {rec['certified']} train {rec['train_hard']:.3f}/{rec['train_soft']:.3f}",
                  f"{rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
