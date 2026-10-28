"""Does the choice get a lesson that points at the true digit? From the exact uniform start (all logits 0), train
jointly (full batch, Adam) and every `every` steps take one small normalized descent step on the choice's own
parameters only (the estimator; for the borrow choice, the cell's carry entries). Score the share of (example, step)
choices whose probability of the true quotient digit rises, and the share where it falls.

With learned slots ("slot") the slots stay interchangeable, so the choice has no reason to prefer the true digit's
slot; with constructed slots ("mult": slot s subtracts s*N) it should.

    python -m tinycore.lesson            (runs on Modal: modal_tinycore.py --suite lesson)
"""

from __future__ import annotations

import torch

from .core import Config, Core, Data


def choice_alignment(m: Core, d: Data, lr=1e-3):
    which = ["g"] if m.cfg.choice in ("est", "estc") else ["b"]
    q = torch.stack([d.qd[:, t] for t in d.steps], 1)
    with torch.no_grad():
        _, ch0 = m.forward(d)
    p0 = ch0.gather(-1, q[..., None]).squeeze(-1)
    params = [m.params[n] for n in which]
    for p in m.params.values():
        p.grad = None
    m.loss(d).backward()
    norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params))
    with torch.no_grad():
        saved = [p.detach().clone() for p in params]
        for p in params:
            p -= lr * p.grad / norm.clamp_min(1e-300)
        _, ch1 = m.forward(d)
        for p, s in zip(params, saved):
            p.copy_(s)
    for p in m.params.values():
        p.grad = None
    p1 = ch1.gather(-1, q[..., None]).squeeze(-1)
    up = (p1 > p0 + 1e-15).double().mean().item()
    down = (p1 < p0 - 1e-15).double().mean().item()
    return up, down, norm.item()


def run(B, w, k, cell, choice, steps=200, every=25, lr=0.05):
    d = Data(B, w, k)
    m = Core(Config(B=B, w=w, k=k, cell=cell, choice=choice, init=0.0))
    params = [m.a, m.b] + ([m.g] if choice in ("est", "estc") else [])
    o = torch.optim.Adam(params, lr=lr)
    out = []
    for it in range(steps + 1):
        if it % every == 0:
            up, down, gn = choice_alignment(m, d)
            out.append((it, round(up, 4), round(down, 4), gn, round(m.exact(d), 4)))
        if it == steps:
            break
        L = m.loss(d)
        o.zero_grad()
        L.backward()
        o.step()
    return out


if __name__ == "__main__":
    torch.set_num_threads(1)
    for it, up, down, gn, ex in run(2, 4, 1, "mult", "est", steps=50, every=25):
        print(it, up, down, f"{gn:.2e}", ex)
