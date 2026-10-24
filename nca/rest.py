"""Does Adam rest at the machine's rule? §9's check of the optimizer behind trim_stab.

    ~/github/sotaku/venv/bin/python -m nca.rest

The compiled rule at output scale 20 in the trim runs' network at (3, 2, 1), where every lesson is 0 and its gradient
about 1e-8; then plain Adam at the trim runs' learning rate (0.003) on the same lessons for 125 steps, with Adam's
default eps (1e-8) and with eps 1e-4. Adam divides each weight's step by the size of its recent gradients plus eps:
with eps 1e-8, gradients of 1e-8 make steps of about half the learning rate on every weight that has one. Prints
the snapped training share every 25 steps (under a minute; under 0.5 GB).
"""

from __future__ import annotations

import sys

from nca import suites
from nca import train as T


def rest(beta=20.0, eps=1e-8, steps=125):
    job = dict(next(j for j in suites.SUITES["trim_stab"]() if j["name"] == "w2_ctl_s0"))
    for key in ("name", "kind", "gpu", "pack"):
        job.pop(key, None)
    job.update(init_beta=beta, steps=steps, every=25, curve_every=0, cert=False, save_weights=False, extra_sizes=())
    r = T.Run(T.RunCfg(**job))
    for g in r.opt.param_groups:
        g["eps"] = eps
    return [(x["step"], x["exact"]) for x in r.train()["log"]]


def main(steps=125):
    for eps in (1e-8, 1e-4):
        log = rest(eps=eps, steps=steps)
        print(f"Adam eps {eps:g}, from the compiled rule at scale 20:", "  ".join(f"{s}: {e:.2f}" for s, e in log),
              flush=True)


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
