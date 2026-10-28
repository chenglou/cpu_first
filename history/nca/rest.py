"""Does the optimizer rest at the machine's rule? §9's check of Adam behind trim_stab, and §10's of the optimizers that
can.

    ~/github/sotaku/venv/bin/python -m nca.rest [steps] [check ...]  (adam, eps, sgd, grads or shrink; default adam eps)

The compiled rule at output scale 20 in the trim runs' network at (3, 2, 1), where every lesson is 0 and its gradient
about 1e-8; then an optimizer on the same lessons for 125 steps: plain Adam at the trim runs' learning rate (0.003)
with its default eps (1e-8) and with eps 1e-4, or plain SGD at learning rate 0.01. Adam divides each weight's step by
the size of its recent gradients plus eps: with eps 1e-8, gradients of 1e-8 make steps of about half the learning
rate on every weight that has one; SGD's step is the gradient times the learning rate. Prints the snapped training
share every 25 steps (each check under 30 seconds; under 0.5 GB). The check `grads` prints instead the lessons'
gradient on every weight at the compiled rule, output scales 4 and 20: the share of weights at 0 and the quantiles of
its size, against which Adam's eps of 1e-4 can be read (seconds). The check `shrink` multiplies every weight of the
compiled rule at scale 20 by one factor, as weight decay alone would (it decays every weight, biases included): the
training share snapped and soft, the lessons' loss and their largest gradient (seconds).
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from nca import suites
from nca import train as T

CHECKS = {"adam": ("Adam eps 1e-8", {}), "eps": ("Adam eps 1e-4", dict(eps=1e-4)),
          "sgd": ("plain SGD at learning rate 0.01", dict(optim="sgd", lr=0.01))}


def start(beta=20.0, steps=125, **opt):
    job = dict(next(j for j in suites.SUITES["trim_stab"]() if j["name"] == "w2_ctl_s0"))
    for key in ("name", "kind", "gpu", "pack"):
        job.pop(key, None)
    job.update(init_beta=beta, steps=steps, every=25, curve_every=0, cert=False, save_weights=False, extra_sizes=(),
               **opt)
    return T.Run(T.RunCfg(**job))


def rest(beta=20.0, steps=125, **opt):
    return [(x["step"], x["exact"]) for x in start(beta, steps, **opt).train()["log"]]


def grads():
    for beta in (4.0, 20.0):
        r = start(beta)
        loss, _ = r.lessons(r.cfg.k)
        loss.backward()
        g = np.concatenate([p.grad.abs().flatten().double().numpy() for p in r.net.parameters() if p.grad is not None])
        q = np.quantile(g, [0.5, 0.9, 0.99])
        print(f"the compiled rule at scale {beta:g}: lesson loss {loss.item():.4f}; of {g.size:,} weights, "
              f"{(g == 0).mean():.0%} with no gradient, {(g >= 1e-3).mean():.1%} with one of 1e-3 or more; |gradient| "
              f"median {q[0]:.1e}, 90% {q[1]:.1e}, 99% {q[2]:.1e}, largest {g.max():.1e}", flush=True)


def shrink(factors=(1, 0.8, 0.64, 0.5, 0.4, 0.3)):
    r = start()
    ps = [p for p in r.net.parameters() if p.requires_grad]
    base = [p.detach().clone() for p in ps]
    for c in factors:
        with torch.no_grad():
            for p, b in zip(ps, base):
                p.copy_(b * c)
                p.grad = None
        loss, _ = r.lessons(r.cfg.k)
        loss.backward()
        norm = float(torch.sqrt(sum((p.detach() ** 2).sum() for p in ps)))
        g = max(float(p.grad.abs().max()) for p in ps if p.grad is not None)
        print(f"every weight x{c:g}: norm {norm:.0f}; training snapped {r.exact(3, 2, 1):.2f}, soft "
              f"{r.exact(3, 2, 1, mode='soft'):.2f}; lesson loss {loss.item():.2g}, largest gradient {g:.1e}",
              flush=True)


def main(*args):
    steps = next((a for a in args if isinstance(a, int)), 125)
    checks = [a for a in args if not isinstance(a, int)]
    if "grads" in checks:
        return grads()
    if "shrink" in checks:
        return shrink()
    for name in checks or ("adam", "eps"):
        lab, opt = CHECKS[name]
        log = rest(steps=steps, **opt)
        print(f"{lab}, from the compiled rule at scale 20:", "  ".join(f"{s}: {e:.2f}" for s, e in log), flush=True)


if __name__ == "__main__":
    main(*(int(a) if a.isdigit() else a for a in sys.argv[1:]))
