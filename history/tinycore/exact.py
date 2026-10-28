"""Exact quantities of the tiny core, computed on every example (no sampling): the gradient at the uniform start,
the Hessian, and the symmetric (undecided) trajectory.

    python -m tinycore.exact grad0            # which table entries the first gradient pushes toward the truth
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from .core import Config, Core, Data

torch.set_num_threads(2)


def names(m: Core):
    out = ["a", "b"]
    if m.cfg.choice in ("est", "estc"):
        out.append("g")
    return out


def flat(m: Core, which=None):
    return torch.cat([m.params[n].detach().flatten() for n in (which or names(m))])


def unflat(m: Core, vec, which=None):
    out, i = {}, 0
    for n in which or names(m):
        p = m.params[n]
        out[n] = vec[i:i + p.numel()].view_as(p)
        i += p.numel()
    return out


def loss_at(m: Core, data: Data, vec, which=None):
    """The loss with the parameters in `which` replaced by vec (differentiable in vec)."""
    keep = (m.a, m.b, m.g)
    try:
        for n, t in unflat(m, vec, which).items():
            setattr(m, n, t)
        return m.loss(data)
    finally:
        m.a, m.b, m.g = keep


def grad_at(m: Core, data: Data, vec=None, which=None):
    vec = (flat(m, which) if vec is None else vec).clone().requires_grad_(True)
    L = loss_at(m, data, vec, which)
    (g,) = torch.autograd.grad(L, vec)
    return L.item(), g


def hessian(m: Core, data: Data, vec=None, which=None):
    vec = flat(m, which) if vec is None else vec
    return torch.autograd.functional.hessian(lambda v: loss_at(m, data, v, which), vec)


def toward_truth(grad, truth, axis=-1):
    """Per table row: the projection of the descent direction (-grad) on the direction from uniform to the truth's
    one-hot (both centered over the output axis). >0: toward the truth; <0: away; 0: no lesson."""
    t = truth - truth.mean(axis, keepdim=True)
    d = -grad - (-grad).mean(axis, keepdim=True)
    return (d * t).sum(axis)


def grad0(cfg: Config, eps=1e-12):
    """The gradient at the exact uniform start (every logit 0), split by table and by row."""
    data = Data(cfg.B, cfg.w, cfg.k, odd=cfg.odd, start=cfg.start)
    m = Core(Config(**{**cfg.__dict__, "init": 0.0}))
    L, g = grad_at(m, data)
    parts = unflat(m, g)
    out = {"loss": L}
    for n, p in parts.items():
        out[n + "_norm"] = p.norm().item()
    proj_a = toward_truth(parts["a"], m.TD)
    proj_b = toward_truth(parts["b"], m.TC)
    # rows the datapath can visit: the truth's carry range is reachable; count every row
    for n, proj in (("a", proj_a), ("b", proj_b)):
        flatp = proj.flatten()
        out[n + "_toward"] = int((flatp > eps).sum())
        out[n + "_away"] = int((flatp < -eps).sum())
        out[n + "_zero"] = int((flatp.abs() <= eps).sum())
        out[n + "_rows"] = flatp.numel()
    if cfg.cell == "slot":                                                    # every slot the same lesson?
        a = parts["a"]
        out["a_slot_spread"] = (a - a.mean(2, keepdim=True)).abs().max().item()
    return out, parts, (proj_a, proj_b), m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what")
    a = p.parse_args()
    if a.what == "grad0":
        for B, w, k in [(2, 3, 1), (2, 3, 2), (2, 4, 3), (3, 2, 2), (3, 3, 2), (4, 2, 2)]:
            for cell, choice in [("slot", "est"), ("slot", "borrow"), ("slot", "given"), ("repeat", "borrow"),
                                 ("repeat", "est")]:
                out, parts, proj, m = grad0(Config(B=B, w=w, k=k, cell=cell, choice=choice))
                spread = f" slot spread {out['a_slot_spread']:.1e}" if "a_slot_spread" in out else ""
                gn = f" |g_est| {out['g_norm']:.1e}" if "g_norm" in out else ""
                print(f"B{B} w{w} k{k} {cell:6s} {choice:6s} digit rows: toward {out['a_toward']:3d} away "
                      f"{out['a_away']:3d} zero {out['a_zero']:3d} | carry rows: toward {out['b_toward']:3d} away "
                      f"{out['b_away']:3d} zero {out['b_zero']:3d} (|g| {out['b_norm']:.1e}){gn}{spread}")


if __name__ == "__main__":
    main()


# ----- the symmetric (undecided) trajectory and its escape -----
SLOT_AXIS = {"a": 2, "b": 2, "g": 1}


def split_sym(m: Core, which=None):
    """Norms of the slot-symmetric and slot-antisymmetric parts of each table (slot cell and estimator)."""
    out = {}
    for n in which or names(m):
        p = m.params[n].detach()
        ax = SLOT_AXIS[n]
        anti = p - p.mean(ax, keepdim=True)
        out[n] = anti.norm().item()
    return out


def symmetric_init(m: Core, init=0.1, eps=0.0, seed=0):
    """Slot-symmetric logits (every slot the same random table, the estimator equal across slots) plus an
    antisymmetric perturbation of norm `eps` per entry (zero mean across slots)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n in ("a", "b", "g"):
            p = m.params[n]
            ax = SLOT_AXIS[n]
            shape = list(p.shape)
            shape[ax] = 1
            base = torch.randn(*shape, generator=gen) * init if n != "g" else torch.zeros(*shape)
            pert = torch.randn(*p.shape, generator=gen)
            pert = pert - pert.mean(ax, keepdim=True)
            p.copy_(base.expand_as(p) + eps * pert)


def anti_basis(m: Core, which=None):
    """Orthonormal basis (columns) of the slot-antisymmetric subspace of the flattened parameters."""
    cols, offset = [], 0
    for n in which or names(m):
        p = m.params[n]
        ax = SLOT_AXIS[n]
        S = p.shape[ax]
        idx = torch.arange(p.numel()).view_as(p).movedim(ax, -1).reshape(-1, S)   # [elements, S]
        # Helmert contrasts across the S slots
        H = torch.zeros(S, S - 1)
        for j in range(1, S):
            H[:j, j - 1] = 1.0
            H[j, j - 1] = -j
            H[:, j - 1] /= H[:, j - 1].norm()
        for row in idx:
            for j in range(S - 1):
                v = torch.zeros(sum(m.params[x].numel() for x in which or names(m)))
                v[offset + row] = H[:, j]
                cols.append(v)
        offset += p.numel()
    return torch.stack(cols, 1)


def anti_project(m: Core, vec, which=None):
    """Remove the slot-mean of every table entry: the projection onto the slot-antisymmetric subspace."""
    parts = unflat(m, vec, which)
    out = []
    for n in which or names(m):
        p = parts[n]
        ax = SLOT_AXIS[n]
        out.append((p - p.mean(ax, keepdim=True)).flatten())
    return torch.cat(out)


def visited_mask(m: Core, data: Data, which=None):
    """Entries the loss depends on at all (nonzero Hessian-vector response to a random probe)."""
    v = torch.randn(sum(m.params[n].numel() for n in which or names(m)), dtype=torch.float64)
    hv = hvp(m, data, flat(m, which), v, which)
    return hv.abs() > 1e-14


def hvp(m: Core, data: Data, x, v, which=None):
    x = x.clone().requires_grad_(True)
    L = loss_at(m, data, x, which)
    (g,) = torch.autograd.grad(L, x, create_graph=True)
    (hv,) = torch.autograd.grad(g @ v, x)
    return hv


def min_anti_eigs(m: Core, data: Data, k=4, which=None, tol=1e-6):
    """The k most negative eigenvalues (and vectors) of the Hessian on the slot-antisymmetric subspace, by Lanczos on
    Hessian-vector products (scipy eigsh, projected operator)."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    x = flat(m, which)
    n = x.numel()
    shift = 0.0

    def mv(v):
        v = torch.from_numpy(np.asarray(v).reshape(-1).copy())
        pv = anti_project(m, v, which)
        out = anti_project(m, hvp(m, data, x, pv, which), which)
        return out.numpy()

    op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    vals, vecs = eigsh(op, k=k, which="SA", tol=tol, maxiter=5000)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]
