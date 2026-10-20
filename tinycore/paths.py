"""The loss along straight lines in parameter space, on every example: from the uniform start (all logits 0) to the
true tables (beta * one-hot), and from a stuck lookup arrangement to the truth (best slot permutation).

    python -m tinycore.paths
"""
import itertools
import json
import os
import sys

import numpy as np
import torch

from .core import Config, Core, Data, train
from .exact import flat, loss_at, names

HERE = os.path.dirname(os.path.abspath(__file__))


def est_truth(m, beta):
    B, w = m.cfg.B, m.cfg.w
    g = torch.zeros_like(m.g)
    for ridx in range(B ** m.kr):
        Rp = sum(((ridx // B ** i) % B) * B ** (i + (w + 1 - m.kr)) for i in range(m.kr))
        for nidx in range(B ** m.kn):
            N = sum(((nidx // B ** i) % B) * B ** (i + (w - m.kn)) for i in range(m.kn))
            q = min(Rp // N, B - 1) if N > 0 else 0
            g[ridx * B ** m.kn + nidx, q] = beta
    return g


def truth_vec(m, beta, perm=None):
    parts = {"a": beta * m.TD, "b": beta * m.TC, "g": est_truth(m, beta)}
    if perm is not None and m.cfg.cell == "slot":
        p = torch.tensor(perm)
        parts = {"a": parts["a"][:, :, p], "b": parts["b"][:, :, p], "g": parts["g"][:, p]}
    return torch.cat([parts[n].flatten() for n in names(m)])


def line(m, data, x0, x1, n=21):
    out = []
    with torch.no_grad():
        for t in np.linspace(0, 1, n):
            out.append(loss_at(m, data, (1 - t) * x0 + t * x1).item())
    return out


def main():
    torch.set_num_threads(2)
    res = {}
    for B, w, k, choice in [(2, 3, 1, "est"), (2, 3, 2, "est"), (2, 4, 3, "est"), (2, 4, 3, "borrow"),
                            (3, 2, 1, "est"), (3, 2, 2, "est"), (4, 2, 1, "est")]:
        data = Data(B, w, k)
        m = Core(Config(B=B, w=w, k=k, choice=choice, init=0.0))
        x0 = flat(m)
        for beta in (2.0, 8.0):
            L = line(m, data, x0, truth_vec(m, beta))
            mono = all(b <= a + 1e-9 for a, b in zip(L, L[1:]))
            print(f"B{B} w{w} k{k} {choice}: start -> truth (beta {beta}): {' '.join(f'{v:.3f}' for v in L[::2])}  "
                  f"monotone {mono}", flush=True)
            res[f"{B},{w},{k},{choice},start,{beta}"] = L
        # a stuck run (joint, Adam, full batch) -> truth under the best permutation
        m2 = Core(Config(B=B, w=w, k=k, choice=choice, init=0.1), seed=0)
        curve, hit = train(m2, data, steps=3000, lr=0.05, every=500)
        xs = flat(m2)
        best = None
        for perm in itertools.permutations(range(B)):
            xt = truth_vec(m2, 8.0, perm)
            L = line(m2, data, xs, xt)
            if best is None or max(L) - L[0] < max(best[1]) - best[1][0]:
                best = (perm, L)
        perm, L = best
        print(f"   stuck run (final exact {curve[-1][2]:.2f}, loss {L[0]:.3f}) -> truth {perm}: "
              f"{' '.join(f'{v:.3f}' for v in L[::2])}  barrier {max(L) - L[0]:+.3f}", flush=True)
        res[f"{B},{w},{k},{choice},stuck"] = {"final": curve[-1][2], "perm": perm, "line": L}
    json.dump(res, open(os.path.join(HERE, "runs", "paths.json"), "w"))


if __name__ == "__main__":
    main()
