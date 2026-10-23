"""Tiny core for the fixed-modulus study (FIXED_MODULUS.md section 2): can the subtraction table learn on one modulus,
and does it learn the same thing whichever digit code the reader uses? general_lessons/tiny_table.py (the anchors'
table lessons of submission_wide_anchored.py, the reader given, W = 6, from a random start) with two additions:

  tie = 1      the reader's ranking ties the table's borrows: sub.order on a read number and another row's (x against
               the next row's x), the premise "a < b" being the reader's own comparison of the two label strings (the
               highest place where they differ decides, as the miner's compare); on every top part, both ways round,
               from the same step as the borrow facts. Its hard verdicts join the facts the digit lessons wait for
               (GATE_WIN batches in a row), and the entries its subtractions read join the order laws' entries.
  code = true | wrong   the digit code the reader reads with: the identity, or a random wrong code that still passes
               x < N on every row of the pool (the reader's semantics: at the highest place where the digit strings
               differ x's digit is coded lower; where x has ended there, N's digit is not coded 0). Every number is
               then a label number (its own digits through the code, missing places 0), x^2 is computed on the labels
               (exact here: the table is the only thing learned), and the table is scored against the standard table
               on labels, r - m - c.

The question: with one modulus the anchors' lessons pair digits only with N's own (LITMUS.md section 4), so the table
never became exact even with the true reader (general_lessons/tiny_table.py: 0 of 12 runs). With the tie, does it?
And under a wrong code, does it learn the same standard label table, so that the machine is exact before the code is
right and the answers can then score whole codes through it?

    ~/github/sotaku/venv/bin/python general_lessons/fixed/tiny_tie.py fixed=323 tie=1 code=wrong code_seed=3 seed=0
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
           late_after=100, place0=5.0, early=0, max_steps=800, log=100, threads=0, out="", tie=0, code="true",
           code_seed=0)
EPS = 1e-6


def coprime(w: int) -> np.ndarray:
    v = np.arange(max(10 ** (w - 1), 3), 10 ** w)
    return v[(v % 2 != 0) & (v % 5 != 0)]


def digits(v: torch.Tensor, L: int) -> torch.Tensor:
    return torch.stack([(v // 10 ** i) % 10 for i in range(L)], -1)


def compare(A, B):
    """The reader's comparison of label strings [X, L] (LSB first): -1, 0, 1, decided at the highest differing place."""
    L = A.shape[1]
    top = torch.where(A != B, torch.arange(L), -1).max(-1).values
    a = A.gather(1, top.clamp(min=0)[:, None])[:, 0]
    b = B.gather(1, top.clamp(min=0)[:, None])[:, 0]
    return torch.where(top < 0, 0, torch.where(a < b, -1, 1))


def own_digits(v: np.ndarray) -> list:
    return [[int(c) for c in str(int(t))][::-1] for t in v]


def consistent(code: np.ndarray, Nv: np.ndarray, xv: np.ndarray) -> bool:
    """Does the code pass the reader's x < N on every row (the reader's semantics, see the docstring)?"""
    for n, x in zip(own_digits(Nv), own_digits(xv)):
        L = max(len(n), len(x))
        n2, x2 = n + [-1] * (L - len(n)), x + [-1] * (L - len(x))
        top = max((j for j in range(L) if n2[j] != x2[j]), default=-1)
        if top < 0 or n2[top] < 0:
            return False
        if x2[top] < 0:
            if code[n2[top]] == 0:
                return False
        elif code[x2[top]] >= code[n2[top]]:
            return False
    return True


def label_digits(v: np.ndarray, code: np.ndarray, L: int) -> torch.Tensor:
    """Each number's own digits through the code, missing places 0 [n, L] (LSB first)."""
    out = np.zeros((len(v), L), dtype=np.int64)
    for i, ds in enumerate(own_digits(v)):
        for j, d in enumerate(ds[:L]):
            out[i, j] = code[d]
    return torch.from_numpy(out)


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
    code = np.arange(10)
    if cfg["code"] == "wrong":
        crng = np.random.default_rng(1000 + cfg["code_seed"])
        uN, ix = np.unique(np.stack([Nv, xv], 1), axis=0, return_index=True)
        tries = 0
        while True:
            tries += 1
            code = crng.permutation(10)
            if (code != np.arange(10)).any() and consistent(code, Nv[ix], xv[ix]):
                break
        print(f"code={code.tolist()} found after {tries} draws", flush=True)
    Nd, xd_all = label_digits(Nv, code, W), label_digits(xv, code, W)
    xl = (xd_all * 10 ** torch.arange(W)).sum(-1)                         # x as a label number
    sq = digits(xl * xl, 2 * W)
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
        x2 = xd.roll(1, 0)                                                   # another row's x
        if cfg["tie"]:                                                       # the reader's ranking ties the borrows
            for j in range(P):
                for A_, B_ in ((xd, x2), (x2, xd)):
                    lt = compare(A_[:, j:], B_[:, j:]) < 0
                    pb = link_soft(hot(A_[:, j:]), B_[:, j:], TDC)[1][:, -1, 1]
                    loss = loss - torch.where(lt, log(pb), log(1 - pb)).mean() / P
        with torch.no_grad():
            okf = (sub_hard(xd, M, Td, Tb)[1] == 1) & (sub_hard(M, xd, Td, Tb)[1] == 0)
            if cfg["tie"]:
                for A_, B_ in ((xd, x2), (x2, xd)):
                    okf = okf & (sub_hard(A_, B_, Td, Tb)[1] == (compare(A_, B_) < 0).long())
            facts_streak = facts_streak + 1 if bool(okf.all()) else 0
        if digits_on:
            with torch.no_grad():
                used = torch.bincount(sub_hard(D, Mr, Td, Tb)[2].flatten(), minlength=200) > 0
                if cfg["tie"]:
                    for A_, B_ in ((xd, x2), (x2, xd)):
                        used |= torch.bincount(sub_hard(A_, B_, Td, Tb)[2].flatten(), minlength=200) > 0
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
    res = {"cfg": cfg, "code_used": code.tolist(), "seconds": round(time.time() - t0, 1), "gated_on": gated_on,
           "first_exact": first_exact,
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
