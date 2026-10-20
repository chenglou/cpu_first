"""Where the undecided plateau leads: along the exactly slot-symmetric trajectory (gradient descent, lr 1), the
Hessian's most negative eigenvalues on the slot-antisymmetric subspace (Lanczos on Hessian-vector products) and,
for each of those escape directions, how well the slot it favors in every estimator window agrees with the true
quotient digit (best slot permutation, windows weighted by how often the first division step uses them).
1.00 means the direction is the right arrangement; 1/S-ish or 0.5 at S = 2 means chance.

    python -m tinycore.escape "2,3,1;2,4,1;3,2,1" 0,1000          (small sizes only; rows to runs/escape_*.json)
"""
import itertools
import json
import os
import sys
import time

import torch

from .core import Config, Core, Data
from .exact import min_anti_eigs, symmetric_init, unflat
from .saddle import gd

HERE = os.path.dirname(os.path.abspath(__file__))


def true_q_per_window(m):
    B, w = m.cfg.B, m.cfg.w
    out = torch.full((B ** m.kr * B ** m.kn,), -1, dtype=torch.long)
    for ridx in range(B ** m.kr):
        Rp = sum(((ridx // B ** i) % B) * B ** (i + (w + 1 - m.kr)) for i in range(m.kr))
        for nidx in range(B ** m.kn):
            N = sum(((nidx // B ** i) % B) * B ** (i + (w - m.kn)) for i in range(m.kn))
            if N >= max(2, B ** (w - 1)) and Rp < B * N:
                out[ridx * B ** m.kn + nidx] = Rp // N
    return out


def window_weights(m, data):
    with torch.no_grad():
        m.forward(data, keep_Rp=True)
        Rp = m.Rps[0]
        B = m.cfg.B
        idx = (Rp.argmax(-1)[:, -m.kr:] * (B ** torch.arange(m.kr))).sum(1) * B ** m.kn + \
              (data.Nd[:, -m.kn:] * (B ** torch.arange(m.kn))).sum(1)
        return torch.bincount(idx, minlength=m.g.shape[0]).double()


def agreement(m, vec, qwin, wts):
    g = unflat(m, torch.from_numpy(vec))["g"]
    pred = g.argmax(1)
    ok = (qwin >= 0) & (wts > 0)
    best = 0.0
    for perm in itertools.permutations(range(m.S)):
        p = torch.tensor(perm)
        best = max(best, (((p[pred] == qwin) & ok).double() @ wts / (wts * ok.double()).sum()).item())
    return best


def main():
    torch.set_num_threads(2)
    sizes = [tuple(int(x) for x in s.split(",")) for s in sys.argv[1].split(";")]
    steps = [int(x) for x in sys.argv[2].split(",")]
    res = {}
    for B, w, k in sizes:
        data = Data(B, w, k)
        m = Core(Config(B=B, w=w, k=k, cell="slot", choice="est", init=0.1), seed=0)
        symmetric_init(m, init=0.1, eps=0.0, seed=0)
        qwin, wts = true_q_per_window(m), window_weights(m, data)
        rows, done = [], 0
        for T in steps:
            gd(m, data, T - done, 1.0, "gd")
            done = T
            t = time.time()
            vals, vecs = min_anti_eigs(m, data, k=6)
            agree = [agreement(m, vecs[:, i], qwin, wts) for i in range(vecs.shape[1])]
            with torch.no_grad():
                L = m.loss(data).item()
            rows.append({"it": T, "loss": L, "exact": m.exact(data), "eigs": vals.tolist(), "agree": agree})
            print(f"B{B} w{w} k{k} it {T:5d} loss {L:.4f} anti eigs {' '.join(f'{v:+.2e}' for v in vals)} | "
                  f"agreement {' '.join(f'{a:.2f}' for a in agree)} ({time.time() - t:.0f}s)", flush=True)
        res[f"{B},{w},{k}"] = rows
    tag = sys.argv[1].replace(";", "_").replace(",", "")
    json.dump(res, open(os.path.join(HERE, "runs", f"escape_{tag}.json"), "w"))


if __name__ == "__main__":
    main()
