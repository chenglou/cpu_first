"""The second tiny core: schoolbook multiplication whose column routing is learned with its times table.

x and y have w base-B digits (LSB first); every product cell (i, j) looks up the times table (a, b) -> (lo, hi) on
(x_i, y_j) and sends lo and hi each to a column; the columns are summed with carries (a designed adder) into the 2w
digits of the product. The true wiring sends lo to column i + j and hi to column i + j + 1 (SQUARE.md's rows).

The routing of every output digit is a mux over columns:
- "tied"      one offset per output kind, shared by every cell: column = i + j + d_lo (resp. d_hi), d in
              {-1, 0, 1, 2} (16 wirings);
- "rel"       untied: every cell's lo and hi choose their own offset in {-1, 0, 1, 2} (4^(2 w^2) wirings);
- "abs"       untied: every cell's lo and hi choose any column 0..2w-1 ((2w)^(2 w^2) wirings).
A digit routed outside 0..2w-1 is lost. Soft routing mixes on the wire: each digit is added to column c with its
routing weight (independent inclusions, the multilinear semantics of the division core).

Answers: the product's digits ("product"), or only x*y mod N after the division core with its true wiring and true
table ("mod": N has w digits, x, y < N, so the dividend has 2w digits and the division w steps): the H1 situation,
where the squaring's wiring is seen only through the reduction.
The times table is given ("exact") or learned with the label-free per-cell ladder a*0 = 0, a*(b+1) = a*b + a (a
designed adder; SQUARE.md's lesson without its verified frontier) plus the answers.
"""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tinyrouting import machine as M

torch.set_default_dtype(torch.float64)
OFFSETS = (-1, 0, 1, 2)


class SData:
    def __init__(self, B, w, labels="product", square=False):
        self.B, self.w = B, w
        if labels == "product":
            xs = np.arange(B ** w)
            x, y = np.meshgrid(xs, xs, indexing="ij")
            x, y = x.ravel(), y.ravel()
            if square:
                x, y = xs, xs
            N = np.ones_like(x)
            target = x * y
        else:                                                                  # every N with w digits, x, y < N
            rows = [(N, a, b) for N in range(max(2, B ** (w - 1)), B ** w) for a in range(N) for b in range(N)
                    if not square or a == b]
            N, x, y = (np.array(c) for c in zip(*rows))
            target = (x * y) % N
        self.N, self.x, self.y, self.t = N, x, y, target
        self.X = len(x)
        self.xd = torch.from_numpy(M.digits(x, B, w))
        self.yd = torch.from_numpy(M.digits(y, B, w))
        self.pd = torch.from_numpy(M.digits(x * y, B, 2 * w))
        self.td = torch.from_numpy(M.digits(target, B, 2 * w if labels == "product" else w))
        self.Nd = torch.from_numpy(M.digits(N, B, w))
        self.labels = labels


def column_sums(D, R, Smax):
    """D [X, K, B] digit distributions of the K routed digits; R [K, C] routing weights onto C columns. Returns the
    column sums' distributions [X, C, Smax+1] (independent inclusions)."""
    X, K, B = D.shape
    C = R.shape[1]
    out = torch.zeros(X, C, Smax + 1)
    out[..., 0] = 1
    for kk in range(K):
        r = R[kk][None, :, None]                                                # [1, C, 1]
        conv = torch.zeros(X, C, Smax + 1)
        for v in range(B):
            if v == 0:
                conv = conv + out * D[:, kk, None, 0, None]
            else:
                conv[..., v:] = conv[..., v:] + out[..., :-v] * D[:, kk, None, v, None]
        out = (1 - r) * out + r * conv
    return out


def add_columns(S, B, digits):
    """Column sums S [X, C, Smax+1] -> the digit distributions [X, digits, B] of sum_c S_c B^c (mod B^digits)."""
    X, C, L = S.shape
    carry = torch.zeros(X, L)
    carry[:, 0] = 1
    out = []
    for c in range(digits):
        col = S[:, c] if c < C else F.one_hot(torch.zeros(X, dtype=torch.long), L).double()
        tot = torch.zeros(X, 2 * L)
        for v in range(L):                                                      # convolution with the carry
            tot[:, v:v + L] += col[:, v, None] * carry
        idx = torch.arange(2 * L)
        dig = torch.zeros(X, B).index_add_(1, idx % B, tot)
        nxt = torch.zeros(X, 2 * L).index_add_(1, idx // B, tot)[:, :L]
        out.append(dig)
        carry = nxt
    return torch.stack(out, 1)


@dataclass
class SCfg:
    B: int = 3
    w: int = 2
    route: str = "tied"            # tied | rel | abs
    param: str = "soft"            # soft | st | mix (tied only: the 16 wirings)
    labels: str = "product"        # product | mod
    table: str = "exact"           # exact | learn
    ladder: float = 0.0            # weight of the label-free ladder (table "learn")
    rladder: float = 0.0           # weight of the routed ladder: row_j(x, b+1) = row_j(x, b) + x B^j through the
                                   # learned placement (a label-free lesson for the wiring; verified frontier)
    answers: float = 1.0
    stages: int = 0                # untied: route cells in waves by i + j (0 = all at once)
    steps: int = 2000
    lr: float = 0.05
    init: float = 0.1
    seed: int = 0
    every: int = 25
    square: bool = False
    post: int = 100                # steps after the first exact step, before the final measure
    div_learn: tuple = ()          # the division core's switches learned too (soft, or straight-through with "st")
    div_table: str = "exact"       # the subtraction table given ("exact") or learned ("learn")
    cell: float = 0.0              # weight of the cell check (a - b) + b = a on the subtraction table
    div_coarse: tuple = ()         # division switches enumerated as a mixture of whole sub-wirings (the rest soft)


class School:
    def __init__(self, c: SCfg):
        self.c = c
        B, w = c.B, c.w
        g = torch.Generator().manual_seed(c.seed)
        self.data = SData(B, w, c.labels, c.square)
        self.cells = [(i, j, o) for i in range(w) for j in range(w) for o in range(2)]   # o: 0 lo, 1 hi
        self.C = 2 * w
        a, b = np.meshgrid(np.arange(B), np.arange(B), indexing="ij")
        self.true_lo = torch.from_numpy((a * b) % B)
        self.true_hi = torch.from_numpy((a * b) // B)
        self.tlo = (torch.randn(B, B, B, generator=g) * c.init).requires_grad_(c.table == "learn")
        self.thi = (torch.randn(B, B, B, generator=g) * c.init).requires_grad_(c.table == "learn")
        if c.route == "tied":
            n = (2, len(OFFSETS))
        elif c.route == "rel":
            n = (len(self.cells), len(OFFSETS))
        else:
            n = (len(self.cells), self.C)
        self.alpha = (torch.randn(*n, generator=g) * c.init).requires_grad_(True)
        self.progs = list(itertools.product(range(len(OFFSETS)), repeat=2)) if c.param == "mix" else None
        if c.param == "mix":
            assert c.route == "tied"
            self.beta = (torch.randn(len(self.progs), generator=g) * c.init).requires_grad_(True)
        params = [self.beta] if c.param == "mix" else [self.alpha]
        if c.table == "learn":
            params += [self.tlo, self.thi]
        self.dalpha = {m: (torch.randn(len(M.MUXES[m]), generator=g) * c.init).requires_grad_(True)
                       for m in M.ORDER if m in c.div_learn and m not in c.div_coarse}
        params += list(self.dalpha.values())
        self.dprogs = M.programs([m for m in M.ORDER if m in c.div_coarse]) if c.div_coarse else None
        if self.dprogs:
            self.dbeta = (torch.randn(len(self.dprogs), generator=g) * c.init).requires_grad_(True)
            params.append(self.dbeta)
        self.dtd = (torch.randn(B, B, 2, B, generator=g) * c.init).requires_grad_(c.div_table == "learn")
        self.dtc = (torch.randn(B, B, 2, 2, generator=g) * c.init).requires_grad_(c.div_table == "learn")
        if c.div_table == "learn":
            params += [self.dtd, self.dtc]
        if c.cell:
            from tinyrouting.train import Checks
            self.ck = Checks(M.Data(B, w, 1))
        self.opt = torch.optim.Adam(params, lr=c.lr)
        if c.labels == "mod":
            self.ddata = self.div_data()
        self.wave = None

    # ----- the times table -----
    def table(self):
        if self.c.table == "exact":
            return (F.one_hot(self.true_lo, self.c.B).double(), F.one_hot(self.true_hi, self.c.B).double())
        return self.tlo.softmax(-1), self.thi.softmax(-1)

    def routed_digits(self, d=None):
        """The 2w^2 routed digits' distributions [X, K, B]: (lo, hi) of every cell."""
        d = d or self.data
        TLO, THI = self.table()
        out = []
        for i, j, o in self.cells:
            T = TLO if o == 0 else THI
            out.append(T[d.xd[:, i], d.yd[:, j]])
        return torch.stack(out, 1)

    # ----- routing -----
    def offsets_to_columns(self, p_off):
        """Offset weights [K, 4] -> column weights [K, C]."""
        R = torch.zeros(len(self.cells), self.C)
        for kk, (i, j, o) in enumerate(self.cells):
            for di, dlt in enumerate(OFFSETS):
                c = i + j + dlt
                if 0 <= c < self.C:
                    R[kk, c] = R[kk, c] + p_off[kk, di]
        return R

    def R_soft(self, hard=False):
        a = self.alpha
        p = a.softmax(-1)
        if hard:
            p = F.one_hot(p.argmax(-1), p.shape[-1]).double() + p - p.detach()
        if self.wave is not None and self.c.route != "tied":                   # staged: later waves held hard true
            p = self.wave_mask(p)
        if self.c.route == "tied":
            p = torch.stack([p[o] for (_, _, o) in self.cells])
        return self.offsets_to_columns(p) if self.c.route != "abs" else p

    def wave_mask(self, p):
        """Cells in waves above the current one are not used yet: their digits are dropped (routed nowhere)."""
        keep = torch.tensor([float(i + j <= self.wave) for (i, j, _) in self.cells])[:, None]
        return p * keep

    def R_prog(self, prog):
        p = torch.stack([F.one_hot(torch.tensor(prog[o]), len(OFFSETS)).double() for (_, _, o) in self.cells])
        return self.offsets_to_columns(p)

    def true_R(self):
        p = torch.zeros(len(self.cells), len(OFFSETS))
        for kk, (i, j, o) in enumerate(self.cells):
            p[kk, OFFSETS.index(o)] = 1
        return self.offsets_to_columns(p)

    def argmax_R(self):
        if self.c.param == "mix":
            return self.R_prog(self.progs[int(self.beta.argmax())])
        p = self.alpha.detach()
        p = F.one_hot(p.argmax(-1), p.shape[-1]).double()
        if self.c.route == "tied":
            p = torch.stack([p[o] for (_, _, o) in self.cells])
        return self.offsets_to_columns(p) if self.c.route != "abs" else p

    # ----- the datapath -----
    def product(self, R, d=None):
        d = d or self.data
        D = self.routed_digits(d)
        Smax = len(self.cells) * (self.c.B - 1)
        S = column_sums(D, R, Smax)
        return add_columns(S, self.c.B, 2 * self.c.w)                            # [X, 2w, B]

    def div_data(self):
        """The division core's data for dividends x*y (2w digits), N (w digits): k = w steps."""
        d = self.data
        dd = object.__new__(M.Data)
        dd.B, dd.w, dd.k = self.c.B, self.c.w, self.c.w
        dd.N, dd.v, dd.y = d.N, d.x * d.y, d.t
        dd.X = d.X
        dd.Nd, dd.vd, dd.yd = d.Nd, d.pd, d.td
        return dd

    def div_tables(self):
        if self.c.div_table == "exact":
            return M.truth_table(self.c.B)
        return self.dtd.softmax(-1), self.dtc.softmax(-1)

    def div_P(self, hard=False):
        """The division's switches: learned ones soft (or straight-through), the rest true."""
        P = {}
        for m in M.ORDER:
            n = len(M.MUXES[m])
            if m in self.dalpha:
                p = self.dalpha[m].softmax(-1)
                if hard:
                    p = F.one_hot(p.argmax(), n).double() + p - p.detach()
                P[m] = p[None]
            else:
                P[m] = F.one_hot(torch.tensor([0]), n).double()
        return P

    def div_prog(self):
        prog = {m: (int(self.dalpha[m].argmax()) if m in self.dalpha else 0) for m in M.ORDER}
        if self.dprogs:
            prog.update(self.dprogs[int(self.dbeta.argmax())])
        return prog

    def div_P_coarse(self, hard=False):
        """The coarse division sub-wirings enumerated (H rows), the other learned switches soft."""
        P1 = self.div_P(hard)
        H = len(self.dprogs)
        P = M.program_P(self.dprogs, tuple(self.c.div_coarse), H=H)
        for m in M.ORDER:
            if m not in self.c.div_coarse:
                P[m] = P1[m].expand(H, -1)
        return P

    def answer(self, R):
        prod = self.product(R)
        if self.c.labels == "product":
            return prod
        TDC = M.table_TDC(*self.div_tables())
        if self.dprogs:
            return self.divide_soft(prod, self.ddata, TDC, self.div_P_coarse(hard=self.c.param == "st"), all_h=True)
        return self.divide_soft(prod, self.ddata, TDC, self.div_P(hard=self.c.param == "st"))

    def divide_soft(self, prod, dd, TDC, P, all_h=False):
        """The division core with a soft dividend [X, 2w, B]: its reading of the dividend register, N's reading, the
        steps and the readout all go through the switches P (H rows: every row a sub-wiring; all_h returns them all)."""
        B, w, k = dd.B, dd.w, dd.k
        vin = M.mix(P["vread"], [prod[None], prod[None].flip(2)])
        _, Nreg = M.inputs(dd, P)
        H = vin.shape[0]
        zero = torch.zeros(H, dd.X, 1, B)
        zero[..., 0] = 1
        Rg = torch.cat([vin[:, :, k:k + w], zero], 2)
        for i, t in enumerate(reversed(range(k))):
            Rp = M.mix(P["shift"], M.shift_options(Rg, vin[:, :, t]))
            Rg = M.reduce_step(Rp, Nreg, P, TDC, B, None, i)
        out = M.mix(P["read"], [Rg[:, :, :w], Rg[:, :, 1:w + 1]])
        return out if all_h else out[0]

    def cell_loss(self):
        """(a - b) + b = a on every pair, the subtraction table soft, a designed adder (train.cell_loss)."""
        from tinyrouting.train import add_soft
        ck, B = self.ck, self.c.B
        TDC = M.table_TDC(*self.div_tables())
        T, cb = M.ripple(F.one_hot(ck.cA, B).double()[None], F.one_hot(ck.cB, B).double()[None], TDC)
        S, _ = add_soft(T, ck.cB)
        nll = -S.clamp_min(1e-300).log().gather(-1, ck.cA[None, ..., None]).mean()
        pb = torch.stack([1 - cb, cb], -1).clamp_min(1e-300).log()
        return nll - pb.gather(-1, ck.cBorrow[None, :, None]).mean()

    def loss(self):
        c, d = self.c, self.data
        L = torch.zeros(())
        if c.answers:
            if c.param == "mix":
                lls = []
                for prog in self.progs:
                    ans = self.answer(self.R_prog(prog))
                    lls.append(ans.clamp_min(1e-300).log().gather(-1, d.td[..., None]).squeeze(-1).sum(-1))
                ll = torch.stack(lls)
                La = -torch.logsumexp(self.beta.log_softmax(0)[:, None] + ll, 0).mean() / d.td.shape[1]
            elif self.dprogs:
                ans = self.answer(self.R_soft(hard=c.param == "st"))                # [H, X, w, B]
                ll = ans.clamp_min(1e-300).log().gather(-1, d.td[None, ..., None].expand(ans.shape[0], -1, -1, 1))
                ll = ll.squeeze(-1).sum(-1)
                La = -torch.logsumexp(self.dbeta.log_softmax(0)[:, None] + ll, 0).mean() / d.td.shape[1]
            else:
                ans = self.answer(self.R_soft(hard=c.param == "st"))
                La = -ans.clamp_min(1e-300).log().gather(-1, d.td[..., None]).mean()
            L = L + c.answers * La
        if c.ladder:
            L = L + c.ladder * self.ladder_loss()
        if c.rladder:
            L = L + c.rladder * self.rladder_loss()
        if c.cell:
            L = L + c.cell * self.cell_loss()
        return L

    def row_digits(self, X, j, b, R, TLO, THI):
        """The row x * b placed at row j through the routing R [K, C]: the digit distributions [n, 2w, B] of the
        sum of the cells (i, j, lo/hi) only. X [n, w] digits of x, b the multiplier digit."""
        B, w = self.c.B, self.c.w
        n = X.shape[0]
        D = torch.zeros(n, len(self.cells), B)
        keep = torch.zeros(len(self.cells), 1)
        for kk, (i, jj, o) in enumerate(self.cells):
            if jj == j:
                D[:, kk] = (TLO if o == 0 else THI)[X[:, i], b]
                keep[kk] = 1
        S = column_sums(D, R * keep, len(self.cells) * (B - 1))
        return add_columns(S, B, 2 * w)

    def rladder_loss(self):
        """Every x of w digits, every row j and rung b: the soft row_j(x, b + 1) against the hard row_j(x, b) + x B^j
        (snapped table, argmax wiring, designed addition), and row_j(x, 0) against 0, trained only where the hard
        rungs 0..b check out (the verified frontier of SQUARE.md)."""
        B, w = self.c.B, self.c.w
        xs = np.arange(B ** w)
        X = torch.from_numpy(M.digits(xs, B, w))
        TLO, THI = self.table()
        R = self.R_soft(hard=self.c.param == "st")
        with torch.no_grad():
            Rh = self.argmax_R()
            lo_h = F.one_hot(TLO.argmax(-1), B).double()
            hi_h = F.one_hot(THI.argmax(-1), B).double()
            place = B ** torch.arange(2 * w)
        L, n = 0.0, 0
        for j in range(w):
            ok = torch.ones(len(xs), dtype=torch.bool)
            prev = torch.zeros(len(xs), dtype=torch.long)                         # the hard rung b's value
            for b in range(B):
                if b == 0:
                    target = torch.zeros(len(xs), dtype=torch.long)
                else:
                    target = prev + torch.from_numpy(xs) * B ** j
                tdig = torch.from_numpy(M.digits(target.numpy() % B ** (2 * w), B, 2 * w))
                soft = self.row_digits(X, j, b, R, TLO, THI)
                nll = -soft.clamp_min(1e-300).log().gather(-1, tdig[..., None]).squeeze(-1).sum(-1)
                L = L + (nll * ok).sum() / (2 * w)
                n += int(ok.sum())
                with torch.no_grad():
                    hard = self.row_digits(X, j, b, Rh, lo_h, hi_h).argmax(-1)
                    val = (hard * place).sum(-1)
                    ok = ok & (val == target)                                     # rung b checks out
                    prev = val
        return L / max(n, 1)

    def ladder_loss(self):
        """a*0 = 0 and a*(b+1) = a*b + a for every (a, b), on the table's own rows: the soft product of (a, b+1)
        against the designed sum of the snapped (a, b) product and a (two digits)."""
        B = self.c.B
        TLO, THI = self.table()
        a = torch.arange(B)
        L = -(TLO[a, 0, 0].clamp_min(1e-300).log().mean() + THI[a, 0, 0].clamp_min(1e-300).log().mean())
        with torch.no_grad():
            lo, hi = TLO.argmax(-1), THI.argmax(-1)
        A, Bb = torch.meshgrid(a, torch.arange(B - 1), indexing="ij")
        val = hi[A, Bb] * B + lo[A, Bb] + A                                     # the hard rung b plus a
        L = L - (TLO[A, Bb + 1, val % B].clamp_min(1e-300).log().mean()
                 + THI[A, Bb + 1, (val // B) % B].clamp_min(1e-300).log().mean())
        return L

    @torch.no_grad()
    def exact(self, d=None):
        d = d or self.data
        TLO, THI = self.table()
        saved = (self.tlo.data.clone(), self.thi.data.clone())
        R = self.argmax_R()
        lo_h = F.one_hot(TLO.argmax(-1), self.c.B).double()
        hi_h = F.one_hot(THI.argmax(-1), self.c.B).double()
        D = []
        for i, j, o in self.cells:
            T = lo_h if o == 0 else hi_h
            D.append(T[d.xd[:, i], d.yd[:, j]])
        D = torch.stack(D, 1)
        S = column_sums(D, R, len(self.cells) * (self.c.B - 1))
        prod = add_columns(S, self.c.B, 2 * self.c.w)
        ok = (prod.argmax(-1) == d.pd).all(-1)
        right = bool(torch.equal(R, self.true_R()))
        if self.c.labels == "mod":
            hard = F.one_hot(prod.argmax(-1), self.c.B).double()
            TD, TC = self.div_tables()
            TDC = M.table_TDC(F.one_hot(TD.argmax(-1), self.c.B).double(), F.one_hot(TC.argmax(-1), 2).double())
            prog = self.div_prog()
            rem = self.divide_soft(hard, self.ddata, TDC, M.program_P([prog], M.ORDER))
            self.last_prod_exact = ok.double().mean().item()
            self.last_div_right = prog == M.true_program()
            ok = (rem.argmax(-1) == d.td).all(-1)
            right = right and self.last_div_right
        return ok.double().mean().item(), right, saved

    def train(self):
        c = self.c
        t0, hit, curve = time.time(), None, []
        waves = list(range(2 * c.w - 1)) if c.stages and c.route != "tied" else [None]
        step = 0
        for wv in waves:
            self.wave = wv
            end = c.steps if wv is None or wv == waves[-1] else step + c.stages
            while step < end:
                if step % c.every == 0:
                    ex, right, _ = self.exact()
                    curve.append((step, wv, round(float(self.loss().detach()), 4), ex, right))
                    if ex == 1.0 and (wv is None or wv == waves[-1]):
                        hit = step
                        break
                L = self.loss()
                self.opt.zero_grad()
                L.backward()
                self.opt.step()
                step += 1
            if hit is not None:
                for _ in range(c.post):
                    L = self.loss()
                    self.opt.zero_grad()
                    L.backward()
                    self.opt.step()
                break
        self.wave = None
        ex, right, _ = self.exact()
        rows = None
        if c.table == "learn":
            rows = ((self.tlo.argmax(-1) == self.true_lo) & (self.thi.argmax(-1) == self.true_hi)).double().mean().item()
        return {"cfg": asdict(c), "hit": hit, "exact": ex, "true_wiring": right, "rows": rows,
                "product_exact": getattr(self, "last_prod_exact", ex),
                "div_prog": M.describe(self.div_prog()) if self.dalpha else None,
                "secs": round(time.time() - t0, 1), "curve": curve}


def main():
    torch.set_num_threads(2)
    for B, w in [(3, 2)]:
        for labels in ("product", "mod"):
            for route, param in [("tied", "mix"), ("tied", "soft"), ("tied", "st"), ("rel", "soft"), ("abs", "soft")]:
                res = []
                for seed in range(3):
                    r = School(SCfg(B=B, w=w, route=route, param=param, labels=labels, seed=seed, steps=1500)).train()
                    res.append((r["hit"], round(r["exact"], 2), round(r["product_exact"], 2), r["true_wiring"]))
                print((B, w), labels, route, param, res, flush=True)


if __name__ == "__main__":
    main()
