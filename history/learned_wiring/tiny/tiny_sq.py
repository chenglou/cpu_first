"""Stage 2 in miniature: the combined H1 machine's schoolbook squaring, base 10, with its layout learned (WIRING.md
section 3).

The squaring of s (W digits) is the combined machine's (SQUARE.md): row j holds the products of every digit s_i with
the multiplier digit s_j, from the learned product cell (a, b) -> (lo, hi); each row is two numbers, its low digits
and its high digits, and the 2W rows' numbers are summed by the learned adder A - (0 - B) (the subtraction table is
given exactly here: stage 1 and the combined machine learn it before the ladder starts).

Its layout as switches (the grammar): column c (0..2W + 1) of row j's low-digit number holds the low digit of one
cell (s_i, s_j), i in 0..W - 1, or nothing; the same for the high-digit number. That is 2 W (2W + 2) switches of W + 1
options. The true layout puts cell i's low digit at column i + j and its high digit at column i + j + 1.

Lessons (all label-free):
  * the routed ladder (TINYROUTING.md section 5, SQUARE.md's verified frontier): for X in the rows' numbers, every row j
    and rung b, the soft row_j(X, b + 1) (soft cell, soft switches blended on the wire, the learned adder's two
    ripples carried as one chain) against the hard row_j(X, b) + X 10^j (snapped cell, argmax layout, the learned
    adder), trained only where the hard rungs 0..b check out; rung 0: row_j(X, 0) = 0;
  * ROUTE = "soft" blends the options on the wire; "tied" uses one offset per output kind for every cell (column
    i + j + d, d in -1..2), the cheap baseline; "rule" is H1's stage-2 grammar: column f(i) + g(j) + d with f in
    {W - 1 - i, i}, g in {0, j, W - 1 - j}, d in -1..2, one rule per output kind.
Stop once the argmax machine passes every rung of every row for DONE steps; certify: the hard square (argmax layout,
snapped cell, the learned adder) of sampled W-digit numbers must equal s * s.

    ~/github/sotaku/venv/bin/python -m learned_wiring.tiny.tiny_sq seed=0 [KEY=VALUE ...]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

CFG = dict(
    W=3,               # digits of s; rows j = 0..W - 1, columns 0..2W + 1
    batch=256,         # numbers X per step
    seed=0,
    init=0.1,
    lr=0.05,
    route="soft",      # soft (a source switch per column of every row) | tied (one offset per output kind) | rule
                       # (H1's stage 2: one placement rule f(i) + g(j) + d per output kind, three factors) |
                       # rule_mix (the same rules, one weight per whole rule)
    train_digits=0,    # > 0: training numbers have at most this many digits (H1: x, N have at most W - 1 of W)
    rows_per_step=0,   # 0: every row j every step; k: each number trains k rows (cycling)
    cells="learn",     # learn | given (the product cell exact: the layout alone)
    cell_ladder=1.0,   # weight of the cell's own ladder a (b + 1) = a b + a on single digits (no layout involved)
    layout="learn",    # learn | given (the true layout: the cell alone)
    done=50,
    max_steps=3000,
    log=50,
    cert=20000,
)


def digits_np(v, n):
    v = np.asarray(v, dtype=np.int64).copy()
    out = np.empty((len(v), n), dtype=np.int64)
    for i in range(n):
        out[:, i] = v % 10
        v //= 10
    return out


def exact_sub():
    r, m, c = np.meshgrid(np.arange(10), np.arange(10), np.arange(2), indexing="ij")
    x = r - m - c
    return torch.from_numpy((x % 10).flatten()), torch.from_numpy((x < 0).astype(np.int64).flatten())


def exact_prod():
    v = np.arange(10)[:, None] * np.arange(10)[None, :]
    return torch.from_numpy((v % 10).flatten()), torch.from_numpy((v // 10).flatten())


def sub_hard(D, M, Td, Tb):
    c = torch.zeros_like(D[..., 0])
    outs = []
    for j in range(D.shape[-1]):
        idx = (D[..., j] * 10 + M[..., j]) * 2 + c
        outs.append(Td[idx])
        c = Tb[idx]
    return torch.stack(outs, -1), c


def add_learned(A, B, Td, Tb):
    C, bc = sub_hard(torch.zeros_like(B), B, Td, Tb)
    Y, by = sub_hard(A, C, Td, Tb)
    return Y


def add_soft_joint(A, B, TH):
    """The learned adder on digit distributions (COMBINED.md 4.2): A - (0 - B), both ripples as one chain."""
    X = A.shape[0]
    T = TH.view(10, 2, 10, 12)
    Pd, Pb = T[..., :10], T[..., 10:]
    T0 = (Pd[0][..., :, None] * Pb[0][..., None, :]).permute(1, 0, 2, 3).reshape(10, 40)
    T1 = (Pd[..., :, None] * Pb[..., None, :]).reshape(10, 400)
    s = torch.zeros(X, 2, 2, dtype=A.dtype)
    s[:, 0, 0] = 1
    out = []
    for j in range(A.shape[1]):
        P1 = (B[:, j] @ T0).view(X, 2, 10, 2)
        P2 = (A[:, j] @ T1).view(X, 20, 20)
        Q1 = torch.einsum("xab,xacp->xpbc", s, P1).reshape(X, 2, 20)
        J = torch.bmm(Q1, P2).view(X, 2, 10, 2)
        out.append(J.sum((1, 3)))
        s = J.sum(2)
    return torch.stack(out, 1)


def true_sources(W):
    """The true source of every column of every row: [W, C] for the low and the high digits (W = none)."""
    C = 2 * W + 2
    j = torch.arange(W)[:, None]
    c = torch.arange(C)[None, :]
    lo = torch.where((c - j >= 0) & (c - j < W), c - j, W)
    hi = torch.where((c - j - 1 >= 0) & (c - j - 1 < W), c - j - 1, W)
    return lo, hi


class Sq(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        W = cfg["W"]
        C = 2 * W + 2
        g = torch.Generator().manual_seed(cfg["seed"])
        r = lambda *s: torch.nn.Parameter(cfg["init"] * torch.randn(*s, generator=g))
        self.p_lo, self.p_hi = r(10, 10, 10), r(10, 10, 10)
        self.src_lo, self.src_hi = r(W, C, W + 1), r(W, C, W + 1)       # soft: per row, column -> source cell
        self.off = r(2, 4)                                              # tied: offset of lo / hi, d in -1..2
        self.rf, self.rg, self.rd = r(2, 2), r(2, 3), r(2, 4)          # rule: f(i), g(j), d per output kind
        self.rw = r(2, 24)                                              # rule_mix: one weight per whole rule
        self.W, self.C, self.route = W, C, cfg["route"]

    def rule_maps(self):
        """Source map of every rule (f, g, d) [2, 3, 4, W, C] (W = nothing)."""
        W, C = self.W, self.C
        j = torch.arange(W)[:, None]
        c = torch.arange(C)[None, :]
        out = torch.full((2, 3, 4, W, C), W, dtype=torch.long)
        for gi, g in enumerate((torch.zeros_like(j), j, W - 1 - j)):
            for di in range(4):
                f = c - g - (di - 1)
                out[0, gi, di] = torch.where((f >= 0) & (f < W), W - 1 - f, W)
                out[1, gi, di] = torch.where((f >= 0) & (f < W), f, W)
        return out

    def sources(self):
        """The argmax source of every column [W, C] (lo, hi)."""
        if self.route == "rule":
            m = self.rule_maps()
            return tuple(m[int(self.rf[k].argmax()), int(self.rg[k].argmax()), int(self.rd[k].argmax())]
                         for k in range(2))
        if self.route == "rule_mix":
            m = self.rule_maps().reshape(24, self.W, self.C)
            return tuple(m[int(self.rw[k].argmax())] for k in range(2))
        if self.route == "tied":
            W, C = self.W, self.C
            j = torch.arange(W)[:, None]
            c = torch.arange(C)[None, :]
            out = []
            for k in range(2):
                d = int(self.off[k].argmax()) - 1
                i = c - j - d
                out.append(torch.where((i >= 0) & (i < W), i, W))
            return out
        return self.src_lo.argmax(-1), self.src_hi.argmax(-1)

    def source_probs(self):
        """Per row and column, the probability of every source [W, C, W + 1] (lo, hi)."""
        if self.route == "rule":
            m = F.one_hot(self.rule_maps(), self.W + 1).float()
            return tuple(torch.einsum("a,b,c,abcjxs->jxs", self.rf[k].softmax(-1), self.rg[k].softmax(-1),
                                      self.rd[k].softmax(-1), m) for k in range(2))
        if self.route == "rule_mix":
            m = F.one_hot(self.rule_maps(), self.W + 1).float().reshape(24, self.W, self.C, self.W + 1)
            return tuple(torch.einsum("r,rjxs->jxs", self.rw[k].softmax(-1), m) for k in range(2))
        if self.route == "tied":
            W, C = self.W, self.C
            out = []
            for k in range(2):
                p = self.off[k].softmax(-1)
                rows = []
                for j in range(W):
                    cols = []
                    for c in range(C):
                        terms = []
                        for di in range(4):
                            i = c - j - (di - 1)
                            terms.append(F.one_hot(torch.tensor(i if 0 <= i < W else W), W + 1).float() * p[di])
                        cols.append(sum(terms))
                    rows.append(torch.stack(cols))
                out.append(torch.stack(rows))
            return out
        return self.src_lo.softmax(-1), self.src_hi.softmax(-1)


def row_hard(X, b, j, Plo, Phi, slo, shi, Td, Tb, W, C):
    """Row j of X * b on digits: X [n, W], b [n]; the low and high digits gathered by the sources slo, shi [W, C] of
    row j (W = nothing) and summed by the learned adder. Returns [n, C]."""
    idx = X * 10 + b[:, None]
    z = torch.zeros_like(X[:, :1])
    lo = torch.cat([Plo[idx], z], 1).gather(1, slo[j][None].expand(X.shape[0], -1))
    hi = torch.cat([Phi[idx], z], 1).gather(1, shi[j][None].expand(X.shape[0], -1))
    return add_learned(lo, hi, Td, Tb)


def row_soft(X, b, j, PLO, PHI, plo, phi, TH):
    """Row j of X * b on digit distributions: soft cells PLO, PHI [10, 10, 10], soft sources plo, phi [W, C, W + 1],
    the learned adder's soft form on the exact table TH. Returns [n, C, 10]."""
    n = X.shape[0]
    zero = F.one_hot(torch.zeros(n, 1, dtype=torch.long), 10).float()
    lo = torch.cat([PLO[X, b], zero], 1)                                  # [n, W + 1, 10]
    hi = torch.cat([PHI[X, b], zero], 1)
    lo_c = torch.einsum("ci,nik->nck", plo[j], lo)
    hi_c = torch.einsum("ci,nik->nck", phi[j], hi)
    return add_soft_joint(lo_c, hi_c, TH)


def run(cfg, verbose=True):
    torch.manual_seed(cfg["seed"])
    W = cfg["W"]
    C = 2 * W + 2
    model = Sq(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(cfg["seed"] + 1)
    Td, Tb = exact_sub()
    TH = torch.cat([F.one_hot(Td, 10), F.one_hot(Tb, 2)], -1).float().view(10, 10, 2, 12).permute(0, 2, 1, 3)
    TH = TH.reshape(20, 120)
    tlo, thi = true_sources(W)
    pow10 = 10 ** torch.arange(C)
    streak, stop, t0 = 0, None, time.time()
    for step in range(1, cfg["max_steps"] + 1):
        Xv = rng.integers(0, 10 ** (cfg["train_digits"] or W), cfg["batch"])
        X = torch.from_numpy(digits_np(Xv, W))
        n = X.shape[0]
        Plo, Phi = model.p_lo.argmax(-1).flatten(), model.p_hi.argmax(-1).flatten()
        slo, shi = model.sources()
        PLO, PHI = model.p_lo.softmax(-1), model.p_hi.softmax(-1)
        plo, phi = model.source_probs()
        if cfg["cells"] == "given":
            Plo, Phi = exact_prod()
            PLO, PHI = F.one_hot(Plo, 10).float().view(10, 10, 10), F.one_hot(Phi, 10).float().view(10, 10, 10)
        if cfg["layout"] == "given":
            slo, shi = tlo, thi
            plo, phi = F.one_hot(tlo, W + 1).float(), F.one_hot(thi, W + 1).float()
        js = list(range(W)) if not cfg["rows_per_step"] else \
            [(step * cfg["rows_per_step"] + k) % W for k in range(cfg["rows_per_step"])]
        loss, ok_all = 0.0, []
        for j in js:
            with torch.no_grad():
                hard = torch.stack([row_hard(X, torch.full((n,), b), j, Plo, Phi, slo, shi, Td, Tb, W, C)
                                    for b in range(10)], 1)                # [n, 10, C]
                Xj = F.pad(X, (j, C - W - j))                              # X 10^j
                tgt = add_learned(hard[:, :9], Xj[:, None].expand(n, 9, C), Td, Tb)
                ok = torch.cat([(hard[:, :1] == 0).all(-1), (hard[:, 1:] == tgt).all(-1)], 1)
                ver = ok.long().cumprod(1).bool()
                ok_all.append(ok)
            l = -row_soft(X, 0, j, PLO, PHI, plo, phi, TH)[..., 0].clamp_min(1e-12).log().mean()
            for b in range(9):
                s = row_soft(X, b + 1, j, PLO, PHI, plo, phi, TH)
                nll = -s.clamp_min(1e-12).log().gather(-1, tgt[:, b, :, None]).squeeze(-1).mean(-1)
                l = l + (nll * ver[:, b].float()).sum() / n
            loss = loss + l / 10
        if cfg["cell_ladder"] and cfg["cells"] == "learn":    # the cell's own ladder: X = a single digit a,
            a = torch.arange(10)[:, None]                        # its rows [lo, hi] in two places
            with torch.no_grad():
                h = torch.stack([torch.cat([Plo.view(10, 10)[:, b:b + 1], Phi.view(10, 10)[:, b:b + 1]], 1)
                                 for b in range(10)], 1)           # [10 (a), 10 (b), 2]
                tg = add_learned(h[:, :9], F.pad(a, (0, 1))[:, None].expand(10, 9, 2), Td, Tb)
                okc = torch.cat([(h[:, :1] == 0).all(-1), (h[:, 1:] == tg).all(-1)], 1)
                verc = okc.long().cumprod(1).bool()
            lc = -(PLO[:, 0, 0].clamp_min(1e-12).log().mean() + PHI[:, 0, 0].clamp_min(1e-12).log().mean())
            for b in range(9):
                nll = -(PLO[:, b + 1].clamp_min(1e-12).log().gather(-1, tg[:, b, 0:1]).squeeze(-1)
                        + PHI[:, b + 1].clamp_min(1e-12).log().gather(-1, tg[:, b, 1:2]).squeeze(-1))
                lc = lc + (nll * verc[:, b].float()).mean()
            loss = loss + cfg["cell_ladder"] * lc / 10
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lad_ok = float(torch.stack(ok_all).float().mean())
        with torch.no_grad():
            slo2, shi2 = model.sources()
            layout = int((slo2 == tlo).sum() + (shi2 == thi).sum())
            plo_t, phi_t = exact_prod()
            cells = int((model.p_lo.argmax(-1).flatten() == plo_t).sum() + (model.p_hi.argmax(-1).flatten() == phi_t).sum())
        streak = streak + 1 if lad_ok == 1.0 else 0
        if verbose and (step % cfg["log"] == 0 or step == 1):
            print(f"step={step} loss={float(loss):.4f} lad_ok={lad_ok:.4f} layout={layout}/{2 * W * C} "
                  f"cells={cells}/200 {time.time() - t0:.1f}s", flush=True)
        if streak >= cfg["done"]:
            stop = step
            break
    res = certify(model, cfg)
    res.update(cfg=dict(cfg), stop=stop, secs=round(time.time() - t0, 1), layout=layout, cells=cells,
               true_layout=layout == 2 * W * C)
    return res


@torch.no_grad()
def certify(model, cfg):
    """The hard square (argmax layout, snapped cell, the learned adder) of sampled W-digit numbers against s * s."""
    W = cfg["W"]
    C = 2 * W + 2
    Td, Tb = exact_sub()
    Plo, Phi = model.p_lo.argmax(-1).flatten(), model.p_hi.argmax(-1).flatten()
    slo, shi = model.sources()
    rng = np.random.default_rng(cfg["seed"] + 7)
    sv = rng.integers(0, 10 ** W, cfg["cert"])
    S = torch.from_numpy(digits_np(sv, W))
    acc = torch.zeros(len(sv), C, dtype=torch.long)
    for j in range(W):
        row = row_hard(S, S[:, j], j, Plo, Phi, slo, shi, Td, Tb, W, C)
        acc = add_learned(acc, row, Td, Tb)
    want = torch.from_numpy(digits_np(sv * sv, C))
    exact = float((acc == want).all(1).float().mean())
    return {"square_exact": exact, "certified": exact == 1.0}


def main(argv):
    cfg = dict(CFG)
    for kv in argv:
        k, v = kv.split("=", 1)
        cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
    torch.set_num_threads(2)
    res = run(cfg)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
