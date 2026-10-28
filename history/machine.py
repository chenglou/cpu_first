"""A decimal Montgomery machine (after the panel's CADET-M) as a reusable module.

ALU cells are learned truth tables (FA, MUL, COMP) calibrated by Peano self-tests — no task labels. The
microcode (entry, MonPro, modular add, exit, loop counter) is designed and vectorised over the batch.
Values between ops are in Montgomery form v~ = v*R mod N with R = 10^K, kept below 2N (Walter's bound).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import h1

K, P = 9, 10
AR = torch.arange(10)


class ALU:
    def __init__(self, FAs, FAc, ML, MH, CO):
        self.FAs, self.FAc, self.ML, self.MH, self.CO = FAs, FAc, ML, MH, CO

    @staticmethod
    def learn(seed: int = 0, steps: int = 400) -> "ALU":
        """Peano self-tests anchored only on digit-token order (successor), commutativity, a*0, a*(b+1)."""
        g = torch.Generator().manual_seed(seed)
        Lfa = (0.01 * torch.randn(10, 10, 2, 12, generator=g)).requires_grad_()
        Lmu = (0.01 * torch.randn(10, 10, 20, generator=g)).requires_grad_()
        Lco = (0.01 * torch.randn(10, 10, generator=g)).requires_grad_()
        opt = torch.optim.Adam([Lfa, Lmu, Lco], lr=0.1)
        E10, E2, ar = torch.eye(10), torch.eye(2), AR

        def soft_fa(pa, pb, pc, S, Cc):
            w = torch.einsum("na,nb,nc->nabc", pa, pb, pc)
            return torch.einsum("nabc,abcs->ns", w, S), torch.einsum("nabc,abck->nk", w, Cc)

        def ce(src, tgt):
            return -(tgt * torch.log(src.clamp_min(1e-9))).sum(-1).mean()

        for _ in range(steps):
            S, Cc = F.softmax(Lfa[..., :10], -1), F.softmax(Lfa[..., 10:], -1)
            lo, hi, co = F.softmax(Lmu[..., :10], -1), F.softmax(Lmu[..., 10:], -1), F.softmax(Lco, -1)
            z = torch.zeros(10, dtype=torch.long)
            loss = ce(S[:, 0, 1], E10[(ar + 1) % 10]) + ce(Cc[:, 0, 1], E2[(ar == 9).long()])
            loss = loss + ce(S[:, 0, 0], E10[ar]) + ce(Cc[:, 0, 0], E2[z])
            A = ar.repeat_interleave(18)
            Bv = torch.arange(9).repeat(10).repeat_interleave(2)
            Cv = torch.arange(2).repeat(90)
            s2, c2 = soft_fa(S[A, Bv, Cv], E10[torch.zeros_like(A)], E2[torch.ones_like(A)], S, Cc)
            c1 = Cc[A, Bv, Cv]
            cout = torch.stack([c1[:, 0] * c2[:, 0], 1 - c1[:, 0] * c2[:, 0]], -1)
            loss = loss + ce(S[A, Bv + 1, Cv], s2.detach()) + ce(Cc[A, Bv + 1, Cv], cout.detach())
            loss = loss + 0.5 * (ce(S, S.transpose(0, 1).detach()) + ce(Cc, Cc.transpose(0, 1).detach()))
            loss = loss + ce(lo[:, 0], E10[z]) + ce(hi[:, 0], E10[z])
            Am, Bm = ar.repeat_interleave(9), torch.arange(9).repeat(10)
            l2, cy = soft_fa(lo[Am, Bm], E10[Am], E2[torch.zeros_like(Am)], S, Cc)
            h2, _ = soft_fa(hi[Am, Bm], E10[torch.zeros_like(Am)], cy, S, Cc)
            loss = loss + ce(lo[Am, Bm + 1], l2.detach()) + ce(hi[Am, Bm + 1], h2.detach())
            s9, c9 = soft_fa(E10[ar], co, E2[z], S.detach(), Cc.detach())
            loss = loss + ce(s9, E10[torch.full((10,), 9)]) + ce(c9, E2[z])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return ALU(Lfa[..., :10].argmax(-1), Lfa[..., 10:].argmax(-1),
                       Lmu[..., :10].argmax(-1), Lmu[..., 10:].argmax(-1), Lco.argmax(-1))

    def exactness(self) -> dict:
        ar = AR
        a3, b3, c3 = torch.meshgrid(ar, ar, torch.arange(2), indexing="ij")
        return {
            "FA": ((self.FAs == (a3 + b3 + c3) % 10) & (self.FAc == (a3 + b3 + c3) // 10)).float().mean().item(),
            "MUL": ((self.ML == (ar[:, None] * ar) % 10) & (self.MH == (ar[:, None] * ar) // 10)).float().mean().item(),
            "COMP": (self.CO == 9 - ar).float().mean().item(),
        }

    # ------------------------------------------------------------ microcode
    def fa(self, a, b, c):
        return self.FAs[a, b, c], self.FAc[a, b, c]

    def multiples(self, v):  # v [B,P] -> d*v for d = 0..9, [B,10,P]
        B = v.shape[0]
        d = AR.view(1, 10, 1).expand(B, 10, P)
        vj = v[:, None, :].expand(B, 10, P)
        lo, hi = self.ML[d, vj], self.MH[d, vj]
        hi = torch.cat([torch.zeros_like(hi[..., :1]), hi[..., :-1]], -1)
        out, cy = [], torch.zeros_like(lo[..., 0])
        for j in range(P):
            s, cy = self.fa(lo[..., j], hi[..., j], cy)
            out.append(s)
        return torch.stack(out, -1)

    def sub_ripple(self, a, b):  # a - b = a + comp(b) + 1; carry-out 1 <=> a >= b
        out, cy = [], torch.ones_like(a[..., 0])
        for j in range(a.shape[-1]):
            s, cy = self.fa(a[..., j], self.CO[b[..., j]], cy)
            out.append(s)
        return torch.stack(out, -1), cy

    def add_ripple(self, a, b):
        out, cy = [], torch.zeros_like(a[..., 0])
        for j in range(a.shape[-1]):
            s, cy = self.fa(a[..., j], b[..., j], cy)
            out.append(s)
        return torch.stack(out, -1), cy

    def entry(self, x, multN):  # x*10^K mod N: K speculative shift-subtract steps (10 candidates each)
        B = x.shape[0]
        s = x
        for _ in range(K):
            A = torch.cat([torch.zeros_like(s[:, :1]), s], 1)[:, None, :].expand(B, 10, P)
            res, ok = self.sub_ripple(A, multN)
            q = (ok * AR).max(-1).values
            s = res[torch.arange(B), q, :K]
        return s

    def monpro(self, a, b, multN, ninv):  # a*b*10^-K mod N, carry-save rows, no comparisons
        B = a.shape[0]
        bi = torch.arange(B)
        multB = self.multiples(torch.cat([b, torch.zeros_like(b[:, :1])], 1))
        D = C1 = C2 = torch.zeros(B, P, dtype=torch.long)
        zero = torch.zeros(B, dtype=torch.long)
        for i in range(K):
            S1, C1n = self.fa(D, multB[bi, a[:, i]], C1)
            t0, _ = self.fa(S1[:, 0], zero, C2[:, 0])
            m = self.ML[t0, ninv]
            S2, C2n = self.fa(S1, multN[bi, m], C2)
            D = torch.cat([S2[:, 1:], torch.zeros_like(S2[:, :1])], 1)
            C1, C2 = C1n, C2n
        E, _ = self.fa(C1, torch.zeros_like(C1), C2)
        out, cy = [], zero
        for j in range(P):
            s, cy = self.fa(D[:, j], E[:, j], cy)
            out.append(s)
        return torch.stack(out, -1)[:, :K]

    def modadd(self, u, v, Nd):  # (u + v) with u < 2N, v < N  ->  < 2N, one speculative subtract
        w, _ = self.add_ripple(torch.cat([u, torch.zeros_like(u[:, :1])], 1), torch.cat([v, torch.zeros_like(v[:, :1])], 1))
        d, ge = self.sub_ripple(w, Nd)
        return torch.where(ge.bool()[:, None], d, w)[:, :K]

    def counter_step(self, td):  # T + 99 via FA ripple; carry-out = still running
        zero = torch.zeros_like(td[:, 0])
        s0, cy = self.fa(td[:, 0], self.CO[zero], zero)
        s1, cy = self.fa(td[:, 1], self.CO[zero], cy)
        return cy.bool(), torch.stack([s0, s1], 1)


def ninv_class(N):
    return torch.from_numpy(np.searchsorted([1, 3, 7, 9], np.asarray(N) % 10))


def run(alu: ALU, program, N, X, T, ninv_of_class):
    """program(alu, s~, ctx) -> next s~ (Montgomery form). Returns LSB-first digits of the answer, [B, K]."""
    Nd = torch.from_numpy(h1.digits(N, P))
    xd = torch.from_numpy(h1.digits(X, K))
    td = torch.from_numpy(h1.digits(T, 2))
    ninv = ninv_of_class[ninv_class(N)]
    multN = alu.multiples(Nd)
    ctx = {"Nd": Nd, "multN": multN, "ninv": ninv}
    ctx["x0"] = s = alu.entry(xd, multN)  # Montgomery form of x
    while True:
        active, nt = alu.counter_step(td)
        if not active.any():
            break
        s = torch.where(active[:, None], program(alu, s, ctx), s)
        td = torch.where(active[:, None], nt, td)
    one = torch.zeros_like(s)
    one[:, 0] = 1
    return alu.monpro(s, one, multN, ninv)
