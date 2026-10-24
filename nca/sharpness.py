"""How large must the rule's weights be, and how sharp are the states a trained grid passes? The two measures behind
§9's account of weight decay.

    ~/github/sotaku/venv/bin/python -m nca.sharpness

First, the compiled rule written into the trim runs' network (hidden 128, the causal hood) at a range of output
scales: exact at (3, 2, 1) and one size up on the snapped and the soft path, and its weight norm (the norm the trim
runs log). Second, for every trim record at (3, 2, 1) that kept its weights, and for the compiled rule at scales 4 and
20: on the 99 training questions, every tick, every written cell, the largest probability of each channel's
distribution, as a median and the share above 0.9. Writes nca/records/sharpness.json (seconds; under 0.5 GB).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nca import compile as CP
from nca import suites
from nca import train as T

HERE = Path(__file__).resolve().parent
SCALES = (0.5, 1, 2, 3, 4, 5, 8, 20)


def trim_run():
    """A Run with the trim runs' settings at (3, 2, 1) (its rule is overwritten by the caller)."""
    job = dict(next(j for j in suites.SUITES["trim1"]() if j["name"] == "w2_ctl_s0"))
    for key in ("name", "kind", "gpu", "pack"):
        job.pop(key, None)
    return T.Run(T.RunCfg(**job))


def norm(net):
    return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in net.parameters() if p.requires_grad)))


@torch.no_grad()
def sharp(r):
    """{channel: (median of the largest probability, share above 0.9)} over every tick and written cell."""
    c = r.cfg
    lay, d = r.layout(c.B, c.w, c.k), r.dataset(c.B, c.w, c.k, 0)
    S0 = r.net.prompt_state(lay, d.Nd, d.vd, "blank", r.gen)
    _, _, hist = r.net.run(lay, S0, lay.depth, "soft", record=True)
    written = ~torch.as_tensor(lay.prompt)
    out = {}
    for name, _ in r.net.fmt.chans:
        top = torch.stack([r.net.channel(S, name).max(1).values for S in hist])[:, :, written].flatten().cpu().numpy()
        out[name] = [round(float(np.median(top)), 3), round(float((top > 0.9).mean()), 3)]
    return out


def main():
    r = trim_run()
    rec = {"compiled": {}, "runs": {}}
    for beta in SCALES:
        CP.set_compiled(r.net, 3, beta)
        row = {"norm": round(norm(r.net), 1), "W2": round(float(r.net.W2.norm()), 1),
               "exact 3,2,1": [r.exact(3, 2, 1), r.exact(3, 2, 1, mode="soft")],
               "exact 3,3,2": [r.exact(3, 3, 2, sample=300), r.exact(3, 3, 2, sample=300, mode="soft")]}
        if beta in (4, 20):
            row["sharpness"] = sharp(r)
        rec["compiled"][str(beta)] = row
        print(f"compiled, output scale {beta}:", row, flush=True)
    for suite in ("trim1", "trim_cols", "trim_stab"):
        p = HERE / "runs" / f"{suite}.jsonl"
        for l in (open(p) if p.exists() else []):
            x = json.loads(l)
            if x["cfg"]["w"] != 2 or "weights" not in x:
                continue
            rr = T.load_state(x)
            row = {"norm": round(norm(rr.net), 1), "exact": [x["train_hard"], x["train_soft"]], "sharpness": sharp(rr)}
            rec["runs"][f"{suite}/{x['name']}"] = row
            print(f"{suite}/{x['name']}:", row, flush=True)
    (HERE / "records" / "sharpness.json").write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
