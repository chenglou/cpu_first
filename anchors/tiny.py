"""Full-batch training on the tiny core: every input, exact gradients, many seeds (ANCHORS.md §3).

The machine's numbers are replaced by ALL numbers the tiny machine can meet (core.py trains on the machine's own link
inputs; exact.py searches tables against all inputs): every modulus N coprime to B with up to W digits, every x < N,
and every link input D < B N. So a failure here is the landscape's, not the data's. The lessons are core.py's.

    python tiny.py B W STEPS SEEDS RECIPE [key=value ...]
"""

from __future__ import annotations

import json
import math
import sys
import time

import torch
import torch.nn.functional as F

import core
from core import ripple, less_soft, add_designed, soft_tables


def all_inputs(B, W):
    mc = core.Machine(B, W)
    Ns = [N for N in range(2, B ** W) if math.gcd(N, B) == 1]
    rows = torch.tensor([(N, x) for N in Ns for x in range(1, N)])
    links = torch.tensor([(D, N) for N in Ns for D in range(B * N)])
    return {"mc": mc, "N": rows[:, 0], "x": rows[:, 1], "D": links[:, 0], "DN": links[:, 1]}


RECIPES = {
    "core": dict(lc=1, zero=1, self=1, less=1, more=1, wrap=1, neg=1, neg2=1, comm=1),
    "simple": dict(lc=1, less=1, more=1, wrap=1),
    "warm": dict(dc=1, zero=1, self=1, less=1, more=1, wrap=1),
}


class Tiny:
    def __init__(self, B, W, weights, seed, init=0.1, lr=0.05, place0=5.0, complete=1.0, tb_fixed=False,
                 dtype=torch.float64, data=None, hard_cin=False, ans_frac=1.0, mon_k=4,
                 coupled=True, gate=None):
        self.B, self.W, self.P = B, W, W + 1
        self.d = all_inputs(B, W) if data is None else data
        self.mc = self.d["mc"]
        self.w, self.place0, self.complete, self.dtype, self.hard_cin = weights, place0, complete, dtype, hard_cin
        self.mon_k, self.coupled = mon_k, coupled
        self.gate, self.gated_on = gate, None                               # order lessons wait for the borrows
        g = torch.Generator().manual_seed(seed)
        self.g = g
        self.a = (init * torch.randn(B, B, 2, B, generator=g, dtype=torch.float64)).to(dtype).requires_grad_()
        self.b = (init * torch.randn(B, B, 2, 2, generator=g, dtype=torch.float64)).to(dtype).requires_grad_()
        td, tb = self.mc.true_tables()
        self.td, self.tb = td, tb
        if tb_fixed:
            with torch.no_grad():
                self.b.copy_(8.0 * (2 * F.one_hot(tb.view(B, B, 2), 2) - 1).to(dtype))
        self.opt = torch.optim.Adam([self.a] if tb_fixed else [self.a, self.b], lr=lr)
        self.fam = core.families(B)
        mc, P = self.mc, self.P
        self.M, self.xd = mc.digits(self.d["N"]), mc.digits(self.d["x"])
        self.Dd, self.DM = mc.digits(self.d["D"]), mc.digits(self.d["DN"])
        self.bl = (self.d["D"] < self.d["DN"]).long()                     # each link's true borrow
        self.Eidx = (self.Dd * B + self.DM) * 2 + self._true_carries(self.Dd, self.DM)   # entry per link place
        self.Rg = mc.digits(torch.unique(self.d["D"]))                       # registers: every number met
        self.Nu = torch.unique(self.d["N"])
        self.MU = mc.digits(self.Nu)
        self.step_no = 0
        n = len(self.d["N"])                                                  # rows that carry an answer
        k = max(1, int(round(ans_frac * n)))
        self.ans_sel = torch.randperm(n, generator=g)[:k]

    def tables(self):
        return self.a.argmax(-1).flatten(), self.b.argmax(-1).flatten()

    def loss(self):
        B, P, w = self.B, self.P, self.w
        hot = lambda v: F.one_hot(v, B).to(self.dtype)
        log = lambda p: p.clamp_min(1e-300).log()
        TS = soft_tables(self.a, self.b)
        TR = TS.detach()
        D, DM, M, xd, Rg = self.Dd, self.DM, self.M, self.xd, self.Rg
        pw = torch.ones(P, dtype=self.dtype)
        pw[0] = self.place0
        nll_w = lambda Y, t: (-log(Y).gather(-1, t[..., None]).squeeze(-1) * pw).sum(-1)
        nll = lambda Y, t: -log(Y).gather(-1, t[..., None]).squeeze(-1).sum(-1)
        L = torch.zeros((), dtype=self.dtype)
        Es, ce = ripple(hot(D), hot(DM), TS, B)
        be = ce[:, -1]
        Z = torch.zeros_like(DM)
        Cs, cc = ripple(hot(Z), hot(DM), TS, B)
        bc = cc[:, -1]
        if w.get("dc", 0):
            back, carry = add_designed(Es, DM, B)
            L = L + w["dc"] * (nll_w(back, D) - self.complete * log(carry * be[:, 1] + (1 - carry) * be[:, 0])).mean() / (pw.sum() + self.complete)
        if w.get("lc", 0):
            back, cf = ripple(Es, Cs, TS, B)
            pc, pe, pf = bc[:, 1], be[:, 1], cf[:, -1, 1]
            agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc
            L = L + w["lc"] * (nll_w(back, D) - self.complete * log(agree)).mean() / (pw.sum() + self.complete)
        hc = (lambda X: torch.zeros_like(X)) if self.hard_cin else (lambda X: None)
        if w.get("zero", 0):
            Y, cy = ripple(hot(Rg), hot(torch.zeros_like(Rg)), TS, B, hard_c=hc(Rg))
            L = L + w["zero"] * (nll(Y, Rg) - log(cy[:, -1, 0])).mean() / (P + 1)
        if w.get("self", 0):
            Y, cy = ripple(hot(Rg), hot(Rg), TS, B, hard_c=hc(Rg))
            L = L + w["self"] * (nll(Y, torch.zeros_like(Rg)) - log(cy[:, -1, 0])).mean() / (P + 1)
        if w.get("less", 0):
            L = L - w["less"] * log(less_soft(hot(xd), hot(M), TS, B)).mean()
        if w.get("more", 0):
            L = L - w["more"] * log(1 - less_soft(hot(M), hot(xd), TS, B)).mean()
        if w.get("wrap", 0):
            Ew = ripple(hot(xd), hot(M), TS, B)[0]
            L = L - w["wrap"] * log(1 - less_soft(Ew, hot(M), TS, B)).mean()
        MU = self.MU
        ZU = torch.zeros_like(MU)
        CU, ccu = ripple(hot(ZU), hot(MU), TS, B)
        if w.get("neg", 0):
            L = L - w["neg"] * log(ccu[:, -1, 1]).mean()
        if w.get("neg2", 0):
            Y2, c2 = ripple(hot(ZU), CU, TS, B)
            pc, p2 = ccu[:, -1, 1], c2[:, -1, 1]
            L = L + w["neg2"] * (nll(Y2, MU) - log(pc * p2 + (1 - pc) * (1 - p2))).mean() / (P + 1)
        if w.get("comm", 0):
            A1, c1 = ripple(hot(D), Cs, TS, B)
            CR, ccr = ripple(hot(Z), hot(D), TS, B)
            A2, c2 = ripple(hot(DM), CR, TS, B)
            k1, k2 = bc[:, 1] * c1[:, -1, 0], ccr[:, -1, 1] * c2[:, -1, 0]
            agree = log((A1 * A2).sum(-1)).sum(-1) + log(k1 * k2 + (1 - k1) * (1 - k2))
            L = L - w["comm"] * agree.mean() / (P + 1)
        if w.get("dec", 0) or w.get("chain", 0) or w.get("mon", 0) or w.get("monk", 0):
            with torch.no_grad():                                            # the referee: the learned hard table
                Td, Tb = self.tables()
                _, blh, rows_e = self.mc.sub(D, DM, Td, Tb)
                self.bl_h, self.Eidx_h = blh, rows_e
            Eo = ripple(hot(D), hot(DM), TS, B, hard_c=rows_e % 2)[0]
        if w.get("dec", 0):
            pl = less_soft(Eo, hot(D), TR, B)
            t = (self.bl_h == 0).to(self.dtype)
            L = L - w["dec"] * (t * log(pl) + (1 - t) * log(1 - pl)).mean()
        if w.get("chain", 0):
            s = self.bl_h == 1
            L = L - w["chain"] * log(1 - less_soft(Eo[s], hot(DM[s]), TR, B)).mean()
        if w.get("mon", 0):                                                  # pairs with the same N
            i, j = self.pairs(self.mon_k)
            with torch.no_grad():
                b0, b1 = self.bl_h[i], self.bl_h[j]
                Td, Tb = self.tables()
                dl = self.mc.sub(D[i], D[j], Td, Tb)[1] == 1                # D_i < D_j, the learned comparator
            t = (b0 < b1) | ((b0 == b1) & dl)                               # f(D_i) < f(D_j)
            lo, hi = torch.where(t, i, j), torch.where(t, j, i)             # which one must come out smaller
            if self.coupled:
                same = self.Eidx_h[lo] == self.Eidx_h[hi]
                pl = core.less_coupled(Eo[lo], Eo[hi], same, self.b.detach().softmax(-1))
            else:
                pl = less_soft(Eo[lo], Eo[hi], TR, B)
            L = L - w["mon"] * log(pl).mean()
        if w.get("nn0", 0):                                                  # N^2 mod N = 0, own hard path
            NU = self.Nu
            L = L + w["nn0"] * self.path_loss(NU * NU, NU, torch.zeros_like(NU), TS)
        if w.get("ans", 0):                                                  # x^2 mod N, own hard path
            sel = self.ans_sel
            N, x = self.d["N"][sel], self.d["x"][sel]
            L = L + w["ans"] * self.path_loss(x * x, N, x * x % N, TS)
        if w.get("monk", 0):                                                 # the same at every width k
            i, j = self.pairs(self.mon_k)
            Tbp = self.b.detach().softmax(-1)
            with torch.no_grad():
                Td, Tb = self.tables()
                rows_d = self.mc.sub(D[i], D[j], Td, Tb)[2]
                bk0, bk1, dlk = Tb[self.Eidx_h[i]], Tb[self.Eidx_h[j]], Tb[rows_d]
                same_all = self.Eidx_h[i] == self.Eidx_h[j]
            tot = 0.0
            for k in range(1, P + 1):
                b0, b1, dl = bk0[:, k - 1], bk1[:, k - 1], dlk[:, k - 1]
                t = (b0 < b1) | ((b0 == b1) & (dl == 1))
                lo, hi = torch.where(t, i, j), torch.where(t, j, i)
                tot = tot - log(core.less_coupled(Eo[lo, :k], Eo[hi, :k], same_all[:, :k], Tbp)).mean()
            L = L + w["monk"] * tot / P
        return L

    def path_loss(self, v, N, y, TS):
        """CE of v mod N's digits below N's width along the machine's own hard path (core.Lab.answer_loss)."""
        mc, B, P = self.mc, self.B, self.P
        Td, Tb = self.tables()
        M = mc.digits(N)
        Rh = mc.digits(v // B ** mc.W)
        Rs = F.one_hot(Rh, B).to(self.dtype)
        Mh = F.one_hot(M, B).to(self.dtype)
        for t in reversed(range(mc.W)):
            vt = (v // B ** t) % B
            with torch.no_grad():
                Rh, q, _, _ = mc.step(torch.cat([vt[:, None], Rh[:, :-1]], 1), M, Td, Tb)
            Dd = torch.cat([F.one_hot(vt, B).to(self.dtype)[:, None], Rs[:, :-1]], 1)
            Rs = Dd * (q == 0).to(self.dtype)[:, None, None]
            for s in range(1, int(q.max()) + 1):
                Dd = ripple(Dd, Mh, TS, B)[0]
                Rs = Rs + Dd * (q == s).to(self.dtype)[:, None, None]
        width = torch.zeros_like(N)
        k = N.clone()
        while bool((k > 0).any()):
            width += (k > 0).long()
            k = k // B
        scored = (torch.arange(P)[None] < width[:, None]).to(self.dtype)
        nll = -Rs.clamp_min(1e-300).log().gather(-1, mc.digits(y)[..., None]).squeeze(-1)
        return (nll * scored).sum() / scored.sum()

    def pairs(self, k):
        """k random partners per link among the links with the same N."""
        DN = self.d["DN"]
        n = len(DN)
        order = torch.argsort(DN, stable=True)
        Ns, counts = torch.unique_consecutive(DN[order], return_counts=True)
        start = torch.cumsum(counts, 0) - counts
        grp = torch.searchsorted(Ns, DN)
        i = torch.arange(n).repeat(k)
        u = torch.rand(len(i), generator=self.g, dtype=torch.float64)
        j = order[start[grp[i]] + (u * counts[grp[i]]).long()]
        keep = i != j
        return i[keep], j[keep]

    def _true_carries(self, A, M):
        B = self.B
        c = torch.zeros_like(A[:, 0])
        out = []
        for j in range(A.shape[1]):
            out.append(c)
            c = (A[:, j] - M[:, j] - c < 0).long()
        return torch.stack(out, 1)

    def facts_exact(self):
        """Share of rows on which the learned borrows pass x - N borrows and N - x does not (hard)."""
        with torch.no_grad():
            Td, Tb = self.tables()
            ok = (self.mc.sub(self.xd, self.M, Td, Tb)[1] == 1) & (self.mc.sub(self.M, self.xd, Td, Tb)[1] == 0)
        return float(ok.double().mean())

    def step(self):
        self.step_no += 1
        if self.gate is not None and self.gated_on is None and self.facts_exact() >= self.gate:
            self.gated_on = self.step_no                                     # the referee is trusted from now on
        w0 = self.w
        if self.gate is not None and self.gated_on is None:
            self.w = {k: v for k, v in w0.items() if k in ("less", "more", "neg")}
        L = self.loss()
        self.w = w0
        self.opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_([self.a, self.b], 1.0)
        self.opt.step()
        return float(L.detach())

    def state(self):
        Td, Tb = self.tables()
        return core.describe(Td, Tb, self.fam)


def census(B, W, weights, steps, seeds, **kw):
    data = all_inputs(B, W)
    out = []
    for s in seeds:
        t = Tiny(B, W, weights, s, data=data, **kw)
        for _ in range(steps):
            L = t.step()
        st = t.state()
        st["loss"] = L
        st["gated_on"] = t.gated_on
        st["wrong"] = core.table_str(*t.tables(), B)
        out.append(st)
    return out


def main(argv):
    B, W, steps, nseeds = int(argv[0]), int(argv[1]), int(argv[2]), int(argv[3])
    w = dict(RECIPES[argv[4]]) if argv[4] in RECIPES else json.loads(argv[4])
    kw = {}
    for kv in argv[5:]:
        k, v = kv.split("=")
        if k.startswith("+"):
            w[k[1:]] = float(v)
        else:
            kw[k] = json.loads(v)
    t0 = time.time()
    res = census(B, W, w, steps, range(nseeds), **kw)
    n = 2 * B * B
    ok = sum(r["rows"] == n for r in res)
    print(f"B={B} W={W} {w} {kw}: {ok}/{len(res)} true  ({time.time() - t0:.0f}s)")
    for r in res:
        if r["rows"] != n:
            print(f"   rows {r['rows']}/{n} d{r['digits']} b{r['borrows']} [{r['closest']}] loss {r['loss']:.4f} :: {r['wrong'][:160]}")


if __name__ == "__main__":
    main(sys.argv[1:])
