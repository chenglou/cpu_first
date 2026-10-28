"""CADET-M end to end: a decimal Montgomery machine whose arithmetic is learned without task labels, and whose
labels only *select*. Datapath (microcode) after the panel's decimal design; everything here is vectorised.

What is learned, and from what:
  FA[a,b,c]->(s,c'), MUL[a,b]->(lo,hi), COMP[b]  : Peano self-tests (anchored only on digit-token order), no H1 rows
  N' per (N mod 10) class                        : H1 labels, by speculation over all 10 complete machines
  digit -> token output map                      : H1 labels, plain CE (it sits after the last modular op)
What is designed: the microcode (entry x*10^9 mod N by speculative shift-subtract, MonPro squaring with carry-save
rows, exit MonPro(s,1), decimal loop counter via FA), and the I/O plumbing.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

import h1

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--selftest_steps", type=int, default=400)
args = p.parse_args()
torch.manual_seed(args.seed)
K, P = 9, 10

# ---------------------------------------------------------------- phase 0: Peano self-tests (no task data)
Lfa = (0.01 * torch.randn(10, 10, 2, 12)).requires_grad_()   # FA logits: s (10) | carry (2)
Lmu = (0.01 * torch.randn(10, 10, 20)).requires_grad_()      # MUL logits: lo (10) | hi (10)
Lco = (0.01 * torch.randn(10, 10)).requires_grad_()          # COMP logits
opt = torch.optim.Adam([Lfa, Lmu, Lco], lr=0.1)
E10, E2, ar = torch.eye(10), torch.eye(2), torch.arange(10)


def soft_fa(pa, pb, pc, S, Cc):
    w = torch.einsum("na,nb,nc->nabc", pa, pb, pc)
    return torch.einsum("nabc,abcs->ns", w, S), torch.einsum("nabc,abck->nk", w, Cc)


def ce(p_src, p_tgt):
    return -(p_tgt * torch.log(p_src.clamp_min(1e-9))).sum(-1).mean()


t0 = time.time()
for it in range(args.selftest_steps):
    S, Cc = F.softmax(Lfa[..., :10], -1), F.softmax(Lfa[..., 10:], -1)
    lo, hi, co = F.softmax(Lmu[..., :10], -1), F.softmax(Lmu[..., 10:], -1), F.softmax(Lco, -1)
    z = torch.zeros(10, dtype=torch.long)
    loss = ce(S[:, 0, 1], E10[(ar + 1) % 10]) + ce(Cc[:, 0, 1], E2[(ar == 9).long()])  # successor (token order)
    loss = loss + ce(S[:, 0, 0], E10[ar]) + ce(Cc[:, 0, 0], E2[z])                     # a + 0 = a
    A = ar.repeat_interleave(18)
    Bv = torch.arange(9).repeat(10).repeat_interleave(2)
    Cv = torch.arange(2).repeat(90)
    s2, c2 = soft_fa(S[A, Bv, Cv], E10[torch.zeros_like(A)], E2[torch.ones_like(A)], S, Cc)
    c1 = Cc[A, Bv, Cv]
    cout = torch.stack([c1[:, 0] * c2[:, 0], 1 - c1[:, 0] * c2[:, 0]], -1)
    loss = loss + ce(S[A, Bv + 1, Cv], s2.detach()) + ce(Cc[A, Bv + 1, Cv], cout.detach())  # a+(b+1) = inc(a+b)
    loss = loss + 0.5 * (ce(S, S.transpose(0, 1).detach()) + ce(Cc, Cc.transpose(0, 1).detach()))  # commutativity
    loss = loss + ce(lo[:, 0], E10[z]) + ce(hi[:, 0], E10[z])                          # a*0 = 0
    Am, Bm = ar.repeat_interleave(9), torch.arange(9).repeat(10)
    l2, cy = soft_fa(lo[Am, Bm], E10[Am], E2[torch.zeros_like(Am)], S, Cc)
    h2, _ = soft_fa(hi[Am, Bm], E10[torch.zeros_like(Am)], cy, S, Cc)
    loss = loss + ce(lo[Am, Bm + 1], l2.detach()) + ce(hi[Am, Bm + 1], h2.detach())    # a*(b+1) = a*b + a
    s9, c9 = soft_fa(E10[ar], co, E2[z], S.detach(), Cc.detach())
    loss = loss + ce(s9, E10[torch.full((10,), 9)]) + ce(c9, E2[z])                    # b + comp(b) = 9
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    FAs, FAc = Lfa[..., :10].argmax(-1), Lfa[..., 10:].argmax(-1)
    ML, MH, CO = Lmu[..., :10].argmax(-1), Lmu[..., 10:].argmax(-1), Lco.argmax(-1)
a3, b3, c3 = torch.meshgrid(ar, ar, torch.arange(2), indexing="ij")
fa_ok = ((FAs == (a3 + b3 + c3) % 10) & (FAc == (a3 + b3 + c3) // 10)).float().mean().item()
mu_ok = ((ML == (ar[:, None] * ar) % 10) & (MH == (ar[:, None] * ar) // 10)).float().mean().item()
print(f"phase 0 (self-tests, {time.time()-t0:.1f}s): FA exact {fa_ok:.3f}  MUL exact {mu_ok:.3f}  "
      f"COMP exact {(CO == 9 - ar).float().mean().item():.2f}  (checked against true arithmetic after the fact)")


# ---------------------------------------------------------------- the machine (hard tables, integer gathers)
def fa(a, b, c):
    return FAs[a, b, c], FAc[a, b, c]


def multiples(v):  # v [B,P] -> d*v for d = 0..9, [B,10,P]
    B = v.shape[0]
    d = ar.view(1, 10, 1).expand(B, 10, P)
    vj = v[:, None, :].expand(B, 10, P)
    lo, hi = ML[d, vj], MH[d, vj]
    hi = torch.cat([torch.zeros_like(hi[..., :1]), hi[..., :-1]], -1)
    out, cy = [], torch.zeros_like(lo[..., 0])
    for j in range(P):
        s, cy = fa(lo[..., j], hi[..., j], cy)
        out.append(s)
    return torch.stack(out, -1)


def sub_ripple(a, b):  # a - b via a + comp(b) + 1; carry-out 1 <=> a >= b
    out, cy = [], torch.ones_like(a[..., 0])
    for j in range(a.shape[-1]):
        s, cy = fa(a[..., j], CO[b[..., j]], cy)
        out.append(s)
    return torch.stack(out, -1), cy


def entry(x, multN):  # x*10^K mod N by K speculative shift-subtract steps
    B = x.shape[0]
    s = x
    for _ in range(K):
        A = torch.cat([torch.zeros_like(s[:, :1]), s], 1)[:, None, :].expand(B, 10, P)
        res, ok = sub_ripple(A, multN)
        q = (ok * ar).max(-1).values  # largest multiple that fits (ok is monotone)
        s = res[torch.arange(B), q, :K]
    return s


def monpro(a, b, multN, ninv):  # a*b*10^-K mod N, carry-save rows, no comparisons
    B = a.shape[0]
    bi = torch.arange(B)
    multB = multiples(torch.cat([b, torch.zeros_like(b[:, :1])], 1))
    D = C1 = C2 = torch.zeros(B, P, dtype=torch.long)
    zero = torch.zeros(B, dtype=torch.long)
    for i in range(K):
        S1, C1n = fa(D, multB[bi, a[:, i]], C1)
        t0, _ = fa(S1[:, 0], zero, C2[:, 0])
        m = ML[t0, ninv]
        S2, C2n = fa(S1, multN[bi, m], C2)
        D = torch.cat([S2[:, 1:], torch.zeros_like(S2[:, :1])], 1)
        C1, C2 = C1n, C2n
    E, _ = fa(C1, torch.zeros_like(C1), C2)
    out, cy = [], zero
    for j in range(P):
        s, cy = fa(D[:, j], E[:, j], cy)
        out.append(s)
    return torch.stack(out, -1)[:, :K]


def run(N, X, T, ninv_of_class):
    """Returns LSB-first digits of the machine's answer, shape [B, K]."""
    Nd = torch.from_numpy(h1.digits(N, P))
    xd = torch.from_numpy(h1.digits(X, K))
    td = torch.from_numpy(h1.digits(T, 2))
    cls = torch.from_numpy(np.searchsorted([1, 3, 7, 9], Nd[:, 0].numpy()))
    ninv = ninv_of_class[cls]
    multN = multiples(Nd)
    s = entry(xd, multN)
    nine = torch.full_like(td[:, 0], 0)
    while True:  # decimal loop counter: T + 99 via FA ripple; carry-out = "still running"
        s0, cy = fa(td[:, 0], CO[nine], torch.zeros_like(nine))
        s1, cy = fa(td[:, 1], CO[nine], cy)
        active = cy.bool()
        if not active.any():
            break
        s = torch.where(active[:, None], monpro(s, s, multN, ninv), s)
        td = torch.where(active[:, None], torch.stack([s0, s1], 1), td)
    one = torch.zeros_like(s)
    one[:, 0] = 1
    return monpro(s, one, multN, ninv)


# ---------------------------------------------------------------- phase 1: labels select (speculation) + output map
tr = h1.load("train")
sel = tr["T"] <= 2
N, X, T, Y = tr["N"][sel], tr["X"][sel], tr["T"][sel], tr["Y"][sel]
t1 = time.time()
with torch.no_grad():
    match = []  # [10 candidates, rows]: does the machine with N' = c reproduce the label?
    for c in range(10):
        out = run(N, X, T, torch.full((4,), c, dtype=torch.long)).numpy()
        match.append(h1.exact(out, Y))
    match = torch.from_numpy(np.stack(match, 0)).float()
cls = torch.from_numpy(np.searchsorted([1, 3, 7, 9], N % 10))
gate = torch.zeros(4, 10, requires_grad=True)  # "temporarily weak connections" between whole machines
gopt = torch.optim.Adam([gate], lr=0.1)
for _ in range(200):
    pmix = (F.softmax(gate, -1)[cls] * match.T).sum(-1)
    loss = -torch.log(pmix + 1e-6).mean()
    gopt.zero_grad()
    loss.backward()
    gopt.step()
ninv_learned = gate.argmax(-1)
print(f"phase 1 (labels select N' among 10 complete machines per class, {time.time()-t1:.1f}s): "
      f"N' per N mod 10 in (1,3,7,9) = {ninv_learned.tolist()}  (true: [9, 3, 7, 1]); "
      f"per-candidate T<=2 match rate for class N≡1: {[round(v, 2) for v in match[:, cls == 0].mean(1).tolist()]}")

# output map digit -> token, learned from labels on the machine's own digits (transparent: after the last mod)
Wout = torch.zeros(10, 17, requires_grad=True)
oopt = torch.optim.Adam([Wout], lr=0.1)
with torch.no_grad():
    yd = torch.from_numpy(run(N, X, T, ninv_learned).numpy())
lab = torch.from_numpy(h1.digits(Y, K)) + 7
scored = torch.from_numpy(np.arange(K)[None, :] < h1.ndigits(Y)[:, None])
for _ in range(200):
    loss = F.cross_entropy(Wout[yd][scored], lab[scored])
    oopt.zero_grad()
    loss.backward()
    oopt.step()
tokmap = Wout.argmax(-1)
print(f"output map learned: digit->token {tokmap.tolist()}")


# ---------------------------------------------------------------- certification
def predict(N, X, T):
    tok = tokmap[run(N, X, T, ninv_learned)]
    return (tok - 7).numpy()


t2 = time.time()
for prof in ("depth_t", "depth_ood_n_t"):
    r = h1.certify(predict, prof, verbose=False)
    print(f"{prof:14s} certified Max T = {r['max_t']}   rung accuracies {r['accs']}")
print(f"certification wall time {time.time()-t2:.1f}s (CPU)")
