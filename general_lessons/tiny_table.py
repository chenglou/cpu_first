"""Tiny core for the law library (LESSONS.md section 1): the subtraction table alone, learned from a random start by
the anchors' lessons of submission_wide_anchored.py (ANCHORS.md), with the reader given (true digits), on one fixed
modulus or on many. The question litmus left open: with one modulus every lesson pairs a digit with one of N's own, so
some entries are in no lesson (LITMUS.md section 4). Does applying the same laws to every pair of numbers fix that?

  pairs = "N"    the file's lessons: x < N on every top part (the borrows); then the order laws and bijectivity on the
                 entries the batch's links read (each register minus N again and again); LATE_AFTER steps later the
                 pieces' laws: (D - N) + N = D complete, R - 0 = R, R - R = 0, 0 + N = N, R + N = N + R, x - N borrows,
                 N - x does not, (x - N) - N does not, 0 - N borrows
  pairs = "all"  the same, and the laws also on a register and another row's register: (R - R') + R' = R complete,
                 R + R' = R' + R, and the order laws on the entries those subtractions read

early = 1 switches the pieces' laws (and with pairs = "all" the laws on every pair) on from step 1, before the order
laws, instead of LATE_AFTER steps after them.

Registers: x and the (W + 1)-place windows of x^2 (exact here: the table is the only thing learned). Base 10, W
digits, registers of P = W + 1 places.

    ~/github/sotaku/venv/bin/python general_lessons/tiny_table.py fixed=323 pairs=all seed=0
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CFG = dict(W=6, fixed=0, widths=(4, 5), pool=4000, batch=256, seed=0, init=0.1, lr=0.05, pairs="N", gate_win=10,
           late_after=100, place0=5.0, early=0, max_steps=800, log=100, threads=0, out="")
EPS = 1e-6


def coprime(w: int) -> np.ndarray:
    v = np.arange(max(10 ** (w - 1), 3), 10 ** w)
    return v[(v % 2 != 0) & (v % 5 != 0)]


def digits(v: torch.Tensor, L: int) -> torch.Tensor:
    return torch.stack([(v // 10 ** i) % 10 for i in range(L)], -1)


def ste(hard, soft):
    return hard + (soft - soft.detach())


def sub_hard(D, M, Td, Tb):
    c = torch.zeros_like(D[..., 0])
    outs, rows = [], []
    for j in range(D.shape[-1]):
        idx = (D[..., j] * 10 + M[..., j]) * 2 + c
        outs.append(Td[idx])
        c = Tb[idx]
        rows.append(idx)
    return torch.stack(outs, -1), c, torch.stack(rows, -1)


def carry_on(c, ste_cin):
    return ste(F.one_hot(c.argmax(-1), 2).to(c.dtype), c) if ste_cin else c


def link_soft(D, M, TDC, ste_cin=False):
    X = D.shape[0]
    c = torch.zeros(X, 2, dtype=D.dtype)
    c[:, 0] = 1
    out, bor = [], []
    for j in range(D.shape[1]):
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = (joint @ TDC).view(X, 10, 12).gather(1, M[:, j, None, None].expand(X, 1, 12)).squeeze(1)
        out.append(Y[:, :10])
        bor.append(Y[:, 10:])
        c = carry_on(Y[:, 10:], ste_cin)
    return torch.stack(out, 1), torch.stack(bor, 1)


def sub_soft(D, M, TDC, ste_cin=False):
    X = D.shape[0]
    c = torch.zeros(X, 2, dtype=D.dtype)
    c[:, 0] = 1
    out, bor = [], []
    for j in range(D.shape[1]):
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = torch.einsum("xmk,xm->xk", (joint @ TDC).view(X, 10, 12), M[:, j])
        out.append(Y[:, :10])
        bor.append(Y[:, 10:])
        c = carry_on(Y[:, 10:], ste_cin)
    return torch.stack(out, 1), torch.stack(bor, 1)


def order_pairs(Tb, used):
    ent = torch.arange(200)
    r, m, c = ent // 20, (ent // 2) % 10, ent % 2
    e0 = ent.repeat_interleave(10)
    k = torch.arange(10).repeat(200)
    out = []
    e1 = k * 20 + (e0 % 20)                                                  # columns
    ok = (r[e0] < k) & used[e0] & used[e1]
    a, b = e0[ok], e1[ok]
    wa, wb = Tb[a], Tb[b]
    lt = Tb[(r[a] * 10 + r[b]) * 2]
    first = (wa < wb) | ((wa == wb) & (lt == 1))
    out.append((torch.where(first, a, b), torch.where(first, b, a)))
    e1 = (r[e0] * 10 + k) * 2 + c[e0]                                        # rows
    ok = (m[e0] < k) & used[e0] & used[e1]
    a, b = e0[ok], e1[ok]
    wa, wb = Tb[a], Tb[b]
    lt = Tb[(m[a] * 10 + m[b]) * 2]
    first = (wa < wb) | ((wa == wb) & (lt == 0))
    out.append((torch.where(first, a, b), torch.where(first, b, a)))
    return out


def run(cfg: dict, verbose: bool = True) -> dict:
    torch.manual_seed(cfg["seed"])
    if cfg["threads"]:
        torch.set_num_threads(cfg["threads"])
    W, P = cfg["W"], cfg["W"] + 1
    rng = np.random.default_rng(cfg["seed"])
    if cfg["fixed"]:
        Nv = np.full(cfg["pool"], cfg["fixed"])
    else:
        Nv = rng.choice(np.concatenate([coprime(w) for w in cfg["widths"]]), cfg["pool"])
    xv = 1 + (rng.random(cfg["pool"]) * (Nv - 1)).astype(np.int64)
    Nd, xd_all = digits(torch.from_numpy(Nv), W), digits(torch.from_numpy(xv), W)
    sq = digits(torch.from_numpy(xv * xv), 2 * W)
    g = torch.Generator().manual_seed(cfg["seed"])
    a = torch.nn.Parameter(cfg["init"] * torch.randn(10, 10, 2, 10, generator=g))
    bpar = torch.nn.Parameter(cfg["init"] * torch.randn(10, 10, 2, 2, generator=g))
    opt = torch.optim.Adam([a, bpar], lr=cfg["lr"])
    rr, mm, cc = torch.meshgrid(torch.arange(10), torch.arange(10), torch.arange(2), indexing="ij")
    TRUE_D, TRUE_B = ((rr - mm - cc) % 10).flatten(), ((rr - mm - cc) < 0).long().flatten()
    hot = lambda v: F.one_hot(v, 10).float()
    log = lambda p: p.clamp_min(1e-12).log()
    dnll = lambda Y, t: -log(Y).gather(-1, t[..., None]).squeeze(-1).sum(-1)
    facts_streak, gated_on, hist, t0 = 0, None, [], time.time()
    first_exact = None
    allp = cfg["pairs"] == "all"
    for step in range(1, cfg["max_steps"] + 1):
        b = torch.from_numpy(rng.choice(cfg["pool"], cfg["batch"], replace=False))
        Xd, M0 = xd_all[b], Nd[b]
        X = Xd.shape[0]
        Td, Tb = a.argmax(-1).flatten(), bpar.argmax(-1).flatten()
        TDC = torch.cat([a.softmax(-1), bpar.softmax(-1)], -1).permute(0, 2, 1, 3).reshape(20, 120)
        M = F.pad(M0, (0, 1))
        xd = F.pad(Xd, (0, 1))
        with torch.no_grad():
            regs = [xd] + [sq[b][:, t:t + P] for t in range(W - 1)]
            Rg = torch.stack(regs, 1).reshape(X * W, P)
            MW = M.repeat_interleave(W, 0)
            links = [Rg]
            for _ in range(8):
                links.append(sub_hard(links[-1], MW, Td, Tb)[0])
            NL = W * len(links)
            D = torch.stack(links, 1).reshape(X * NL, P)
            Mr = M.repeat_interleave(NL, 0)
            Z = torch.zeros_like(M)
            R2 = Rg.view(X, W, P).roll(1, 0).reshape(X * W, P)              # another row's registers
        digits_on = gated_on is not None
        pieces = (digits_on and step - gated_on >= cfg["late_after"]) or bool(cfg["early"])   # early: from step 1
        loss = torch.zeros(())
        for j in range(P):                                                   # x < N on every top part
            differ = (xd[:, j:] != M[:, j:]).any(1)
            p1 = link_soft(hot(xd[:, j:]), M[:, j:], TDC)[1][:, -1, 1]
            p2 = link_soft(hot(M[:, j:]), xd[:, j:], TDC)[1][:, -1, 1]
            loss = loss - (torch.where(differ, log(p1), log(1 - p1)).mean() + log(1 - p2).mean()) / P
        with torch.no_grad():
            okf = (sub_hard(xd, M, Td, Tb)[1] == 1) & (sub_hard(M, xd, Td, Tb)[1] == 0)
            facts_streak = facts_streak + 1 if bool(okf.all()) else 0
        if digits_on:
            with torch.no_grad():
                used = torch.bincount(sub_hard(D, Mr, Td, Tb)[2].flatten(), minlength=200) > 0
                if allp:
                    E2, _, i2 = sub_hard(Rg, R2, Td, Tb)
                    C2, _, i3 = sub_hard(torch.zeros_like(R2), R2, Td, Tb)
                    _, _, i4 = sub_hard(E2, C2, Td, Tb)
                    for ii in (i2, i3, i4):
                        used |= torch.bincount(ii.flatten(), minlength=200) > 0
            pa = a.softmax(-1)
            pd = pa.reshape(200, 10)
            for lo, hi in order_pairs(Tb, used):
                loss = loss - log((pd[lo] * (1 - pd[hi].cumsum(-1))).sum(-1)).mean()
            loss = loss - log(10 * pa.mean(0)).mean() - log(10 * pa.mean(1)).mean()
        if not digits_on and facts_streak >= cfg["gate_win"]:
            gated_on = step
        if pieces:
            weight = torch.ones(P)
            weight[0] = cfg["place0"]

            def inverse(Dv, Bv):                       # (D - B) + B = D, complete: (D - B) - (0 - B) = D, be = bc - bf
                Es, be = link_soft(hot(Dv), Bv, TDC)
                Cs, bc = link_soft(hot(torch.zeros_like(Bv)), Bv, TDC)
                back, bf = sub_soft(Es, Cs, TDC)
                pe, pf, pc = be[:, -1, 1], bf[:, -1, 1], bc[:, -1, 1]
                agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc
                nll = (-log(back).gather(-1, Dv[..., None]).squeeze(-1) * weight).sum(-1) - log(agree)
                return nll.mean() / (weight.sum() + 1)

            def commute(A, B):                          # A + B = B + A: A - (0 - B) = B - (0 - A)
                CB, bcb = link_soft(hot(torch.zeros_like(B)), B, TDC)
                A1, b1 = sub_soft(hot(A), CB, TDC)
                CA, bca = link_soft(hot(torch.zeros_like(A)), A, TDC)
                A2, b2 = sub_soft(hot(B), CA, TDC)
                c1, c2 = bcb[:, -1, 1] * b1[:, -1, 0], bca[:, -1, 1] * b2[:, -1, 0]
                return -(log((A1 * A2).sum(-1)).sum(-1) + log(c1 * c2 + (1 - c1) * (1 - c2))).mean() / (P + 1)

            loss = loss + inverse(D, Mr) + commute(Rg, MW)
            if allp:
                loss = loss + inverse(Rg, R2) + commute(Rg, R2)
            for A, B, tgt in ((Rg, torch.zeros_like(Rg), Rg), (Rg, Rg, torch.zeros_like(Rg))):
                Y, bb = link_soft(hot(A), B, TDC, True)
                loss = loss + (dnll(Y, tgt) - log(bb[:, -1, 0])).mean() / (P + 1)
            Cs, bc = link_soft(hot(Z), M, TDC)
            Y2, b2 = sub_soft(hot(Z), Cs, TDC)
            loss = loss + (dnll(Y2, M) - log(bc[:, -1, 1] * b2[:, -1, 1] + bc[:, -1, 0] * b2[:, -1, 0])).mean() / (P + 1)
            loss = loss - log(link_soft(hot(xd), M, TDC)[1][:, -1, 1]).mean()
            loss = loss - log(link_soft(hot(M), xd, TDC)[1][:, -1, 0]).mean()
            Ew = link_soft(hot(xd), M, TDC)[0]
            loss = loss - log(link_soft(Ew, M, TDC)[1][:, -1, 0]).mean()
            loss = loss - log(bc[:, -1, 1]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            Td, Tb = a.argmax(-1).flatten(), bpar.argmax(-1).flatten()
            dr, br = int((Td == TRUE_D).sum()), int((Tb == TRUE_B).sum())
            ok = int(((Td == TRUE_D) & (Tb == TRUE_B)).sum())
        if ok == 200 and first_exact is None:
            first_exact = step
        if step % cfg["log"] == 0 or step == 1:
            hist.append((step, dr, br, ok))
            if verbose:
                print(f"step={step} loss={float(loss):.3f} digits={dr}/200 borrows={br}/200 both={ok}/200 "
                      f"gated_on={gated_on}", flush=True)
    with torch.no_grad():
        Td, Tb = a.argmax(-1).flatten(), bpar.argmax(-1).flatten()
        wrong = [(i // 20, (i // 2) % 10, i % 2) for i in range(200) if Td[i] != TRUE_D[i] or Tb[i] != TRUE_B[i]]
    res = {"cfg": cfg, "seconds": round(time.time() - t0, 1), "gated_on": gated_on, "first_exact": first_exact,
           "final": hist[-1], "history": hist, "wrong_entries": wrong[:60], "n_wrong": len(wrong)}
    if verbose:
        print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "history"}), flush=True)
    if cfg["out"]:
        Path(cfg["out"]).write_text(json.dumps(res, indent=1))
    return res


def parse(argv) -> dict:
    cfg = dict(CFG)
    for x in argv:
        k, v = x.split("=", 1)
        cfg[k] = type(CFG[k])(v) if not isinstance(CFG[k], tuple) else tuple(int(t) for t in v.split(","))
    return cfg


if __name__ == "__main__":
    run(parse(sys.argv[1:]))
