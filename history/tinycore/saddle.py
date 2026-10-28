"""The undecided plateau as a symmetric saddle (Saad & Solla 1995), computed exactly on the tiny core.

- `traj`: start exactly slot-symmetric (every slot the same table, the estimator equal across slots). Gradient
  descent keeps it symmetric forever. Along that trajectory, the Hessian's antisymmetric block gives the escape rate:
  its most negative eigenvalue -lam means a slot-asymmetry grows like exp(lr * lam * t).
- `eps`: the same start plus an antisymmetric perturbation of size eps. If the plateau is this saddle, its length
  grows like log(1/eps) / (lr * lam).

    python -m tinycore.saddle traj --B 2 --w 3 --k 2
    python -m tinycore.saddle eps --B 2 --w 3 --k 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .core import Config, Core, Data
from .exact import anti_basis, flat, hessian, names, split_sym, symmetric_init

HERE = os.path.dirname(os.path.abspath(__file__))


def gd(m: Core, data: Data, steps, lr, opt="gd", callback=None, every=50):
    params = [m.params[n] for n in names(m)]
    o = torch.optim.SGD(params, lr=lr) if opt == "gd" else torch.optim.Adam(params, lr=lr)
    for it in range(steps + 1):
        if it % every == 0 and callback:
            if callback(it, m) is False:
                return it
        if it == steps:
            break
        L = m.loss(data)
        o.zero_grad()
        L.backward()
        o.step()
    return steps


def anti_eigs(m: Core, data: Data, k=3):
    H = hessian(m, data)
    Q = anti_basis(m)
    Ha = Q.T @ H @ Q
    ev, evec = torch.linalg.eigh(Ha)
    Hs_full = H                                                              # symmetric block, for comparison
    evs = torch.linalg.eigvalsh(Hs_full)
    return ev[:k].tolist(), evs[:k].tolist(), (Q @ evec[:, 0])


def traj(cfg: Config, steps, lr, opt, every, seed=0, init=0.1):
    data = Data(cfg.B, cfg.w, cfg.k, odd=cfg.odd, start=cfg.start)
    m = Core(cfg, seed=seed)
    symmetric_init(m, init=init, eps=0.0, seed=seed)
    rows = []

    def cb(it, m):
        with torch.no_grad():
            L = m.loss(data).item()
        ex = m.exact(data)
        ea, eh, v = anti_eigs(m, data)
        TD, TC = m.tables()
        ent = (-(TD.clamp_min(1e-12).log() * TD).sum(-1)).mean().item()
        r = {"it": it, "loss": L, "exact": ex, "anti_min": ea, "full_min": eh, "anti_norm": split_sym(m),
             "cell_entropy": ent}
        rows.append(r)
        print(f"it {it:6d} loss {L:.4f} exact {ex:.3f} anti-block eig {', '.join(f'{x:+.2e}' for x in ea)} "
              f"| full {', '.join(f'{x:+.2e}' for x in eh)} | anti norms {json.dumps({k: round(v, 6) for k, v in r['anti_norm'].items()})} "
              f"| cell entropy {ent:.3f}", flush=True)

    gd(m, data, steps, lr, opt, cb, every)
    return rows


def escape(cfg: Config, eps, steps, lr, opt, seed=0, init=0.1, every=10, level=None):
    """Steps until the exact share leaves the symmetric plateau (rises `level` above it), from a symmetric start
    plus an antisymmetric perturbation eps."""
    data = Data(cfg.B, cfg.w, cfg.k, odd=cfg.odd, start=cfg.start)
    m = Core(cfg, seed=seed)
    symmetric_init(m, init=init, eps=eps, seed=seed)
    hist = []

    def cb(it, m):
        an = split_sym(m)
        ex = m.exact(data)
        hist.append((it, ex, sum(an.values())))
        return ex < 0.99

    end = gd(m, data, steps, lr, opt, cb, every)
    return hist, end


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--w", type=int, default=3)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--choice", default="est")
    p.add_argument("--cell", default="slot")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--opt", default="gd")
    p.add_argument("--every", type=int, default=250)
    p.add_argument("--init", type=float, default=0.1)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--eps", default="1e-1,1e-2,1e-3,1e-4,1e-6,1e-8")
    a = p.parse_args()
    torch.set_num_threads(2)
    cfg = Config(B=a.B, w=a.w, k=a.k, cell=a.cell, choice=a.choice, init=a.init)
    if a.what == "traj":
        traj(cfg, a.steps, a.lr, a.opt, a.every, init=a.init)
    elif a.what == "eps":
        out = []
        for seed in range(a.seeds):
            for eps in [float(x) for x in a.eps.split(",")]:
                t = time.time()
                hist, end = escape(cfg, eps, a.steps, a.lr, a.opt, seed=seed, init=a.init)
                ex = [h[1] for h in hist]
                # the first step at which exact is >= 0.99, and the plateau level
                hit = next((h[0] for h in hist if h[1] >= 0.99), None)
                print(f"seed {seed} eps {eps:.0e}: hit {hit} final {ex[-1]:.3f} max {max(ex):.3f} "
                      f"({time.time() - t:.1f}s)", flush=True)
                out.append({"seed": seed, "eps": eps, "hit": hit, "final": ex[-1], "hist": hist})
        with open(os.path.join(HERE, "runs", f"eps_B{a.B}w{a.w}k{a.k}_{a.cell}_{a.choice}_{a.opt}.json"), "w") as f:
            json.dump({"cfg": cfg.__dict__, "lr": a.lr, "opt": a.opt, "rows": out}, f)


if __name__ == "__main__":
    main()
