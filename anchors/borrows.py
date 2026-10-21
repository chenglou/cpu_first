"""Which order facts teach the borrow table from a random start (ANCHORS.md §4.1)?

Only the cell's borrow outputs (200 x 2 logits) are trained; the facts read nothing else: each is "a - b borrows
exactly when a < b" for a pair of the row's own numbers (x < N, 0 < x, 0 < N, a = a). Base 10, the H1 rows, the
machine's 9-place register.

    python borrows.py STEPS SEEDS VARIANT [hard_cin]
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

import core

VARIANTS = {
    "less_more": [("x", "N", 1), ("N", "x", 0)],
    "+neg": [("x", "N", 1), ("N", "x", 0), ("0", "N", 1)],
    "order": [("x", "N", 1), ("N", "x", 0), ("0", "N", 1), ("0", "x", 1), ("N", "0", 0), ("x", "0", 0),
              ("x", "x", 0), ("N", "N", 0), ("0", "0", 0)],
    "order_strict": [("x", "N", 1), ("N", "x", 0), ("0", "N", 1), ("0", "x", 1), ("N", "0", 0), ("x", "0", 0)],
    "borrows_only": [("x", "N", 1), ("0", "N", 1), ("0", "x", 1)],
}


def final_borrow(Ad, Md, TB, B, hard_c=None):
    """P(final borrow) of A - M on exact digits A, M [X, n]: the borrow ripple alone. TB [B, B, 2, 2] probabilities.
    hard_c: the borrow into each place taken from the hard table's own ripple (the gradient reaches only the entries
    the hard ripple uses)."""
    X, n = Ad.shape
    c = torch.zeros(X, 2, dtype=TB.dtype)
    c[:, 0] = 1
    for j in range(n):
        if hard_c is not None:
            c = F.one_hot(hard_c[:, j], 2).to(TB.dtype)
        c = torch.einsum("xc,xco->xo", c, TB[Ad[:, j], Md[:, j]])
    return c[:, 1]


def train(variant, steps, seed, batch=64, hard_cin=False, data=None, lr=0.05, init=0.1):
    B = 10
    mc = core.Machine(B, 8)
    data = core.h1_rows(4096, seed=0) if data is None else data
    g = torch.Generator().manual_seed(seed)
    b = (init * torch.randn(B, B, 2, 2, generator=g)).requires_grad_()
    opt = torch.optim.Adam([b], lr=lr)
    td, tb = mc.true_tables()
    pairs = VARIANTS.get(variant, [])
    for s in range(1, steps + 1):
        idx = torch.randint(0, len(data["N"]), (batch,), generator=g)
        nums = {"x": mc.digits(data["x"][idx]), "N": mc.digits(data["N"][idx])}
        nums["0"] = torch.zeros_like(nums["x"])
        TB = b.softmax(-1)
        Tb = b.argmax(-1).flatten()
        loss = 0.0
        if variant.startswith("prefix"):                                  # x < N on every top part: x[j:] <= N[j:],
            x_, N_ = nums["x"], nums["N"]                                 # strictly less iff the strings differ
            P = x_.shape[1]
            for j in range(P):
                differ = (x_[:, j:] != N_[:, j:]).any(1)
                p1 = final_borrow(x_[:, j:], N_[:, j:], TB, B)
                p2 = final_borrow(N_[:, j:], x_[:, j:], TB, B)
                loss = loss - torch.where(differ, torch.log(p1.clamp_min(1e-12)), torch.log((1 - p1).clamp_min(1e-12))).mean() / P
                loss = loss - torch.log((1 - p2).clamp_min(1e-12)).mean() / P
            pairs = VARIANTS.get(variant[len("prefix"):], [])
        for a, m, t in pairs:
            hc = None
            if hard_cin:
                hc = (mc.sub(nums[a], nums[m], td, Tb)[2] % 2)
            p = final_borrow(nums[a], nums[m], TB, B, hc)
            loss = loss - (torch.log(p.clamp_min(1e-12)) if t else torch.log((1 - p).clamp_min(1e-12))).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([b], 1.0)
        opt.step()
    Tb = b.argmax(-1).flatten()
    return int((Tb == tb).sum()), b.detach()


if __name__ == "__main__":
    steps, nseeds, variant = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    hard = len(sys.argv) > 4 and sys.argv[4] == "hard_cin"
    data = core.h1_rows(4096, seed=0)
    t0 = time.time()
    res = [train(variant, steps, s, hard_cin=hard, data=data)[0] for s in range(1, nseeds + 1)]
    print(f"{variant:14s} hard_cin={hard}: {sum(r == 200 for r in res)}/{nseeds} exact, borrows right {res} ({time.time() - t0:.0f}s)")
