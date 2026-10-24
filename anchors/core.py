"""Tiny core of the digit-meaning problem (ANCHORS.md): the H1 chain machine in any base, its lessons, and training.

The machine is CHAIN.md's / ADDER.md's in base B: a register of P = W + 1 places, W long-division steps, each
subtracting N again and again (B - 1 links) with ONE learned cell (r, m, borrow in) -> (digit, borrow out); the stop
rule keeps the last trial before the chain first borrows. Base 10 with W = 8 and the H1 rows is the real machine; a
smaller base or width is a tiny core of it.

Every lesson is a weight in a dict (`LESSONS` names them) and has a hard version: the share of its instances the
snapped (argmax) table passes, computed the way training reads its numbers (the machine's own link inputs).
Tables are flat over (r, m, c) -> (r * B + m) * 2 + c, as in the submissions.
"""

from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.nn.functional as F

H1_DATA = os.environ.get("H1_DATA", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "one-layer-deeper", "data", "generated",
    "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))

LESSONS = {
    "dc": "the check (D - N) + N = D with the hand-written adder (the warm start's teacher)",
    "lc": "the check (D - N) - (0 - N) = D with the learned adder",
    "less": "x < N: x - N borrows",
    "more": "x < N: N - x does not borrow",
    "zero": "R - 0 = R on the registers",
    "self": "R - R = 0 on the registers",
    "wrap": "(x - N) - N does not borrow (x - N wraps to B^P + x - N >= N)",
    "neg": "0 - N borrows",
    "neg2": "0 + N = N with the learned adder: 0 - (0 - N) = N",
    "comm": "R + N = N + R with the learned adder",
    "chain": "on the machine's own chains, the link after a borrowing one does not borrow",
    "xmod": "x mod N = x: x - N borrows, so one division step keeps x (hard path: the kept trial is x)",
    "ans": "the rows' answers (T <= 2) along the machine's own hard path",
    "dec": "order: a link that does not borrow lowers its number, one that borrows raises it (E < D iff no borrow)",
    "mon": "order: subtracting N keeps two link inputs' order, unless exactly one of them wraps",
    "mon0": "order: 0 - N falls as N grows (pairs of rows)",
    "monx": "order: x - N rises with x for the same N (pairs of rows with one modulus)",
}


# ------------------------------------------------------------------------------------------------------------- data
def h1_rows(n: int | None = None, seed: int = 0, split: str = "train") -> dict:
    """H1 rows as int64 tensors N, x, T, y (n random rows, or all)."""
    with open(os.path.join(H1_DATA, split + ".jsonl")) as f:
        rows = [json.loads(line) for line in f]
    if n is not None:
        g = torch.Generator().manual_seed(seed)
        rows = [rows[i] for i in torch.randperm(len(rows), generator=g)[:n].tolist()]
    get = lambda k: torch.tensor([r[k] for r in rows], dtype=torch.long)
    return {"N": get("modulus"), "x": get("x"), "T": get("time_steps"), "y": get("result")}


def tiny_rows(B: int, wmin: int, wmax: int, n: int, seed: int, t_frac: float = 0.02) -> dict:
    """Made-up H1-like rows in base B: N of wmin..wmax digits, coprime to B (H1's N end in 1, 3, 7, 9), x uniform
    in [1, N), a share t_frac of the rows with T = 1 or 2 (H1: 1% + 1%), the rest T = 4."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(wmin, wmax + 1, (4 * n,), generator=g)
    lo, hi = B ** (w - 1), B ** w
    N = lo + (torch.rand(4 * n, generator=g, dtype=torch.float64) * (hi - lo)).long()
    N = N[(torch.tensor([math.gcd(int(v), B) for v in N.tolist()]) == 1) & (N >= 3)][:n]
    x = 1 + (torch.rand(len(N), generator=g, dtype=torch.float64) * (N - 1)).long()
    u = torch.rand(len(N), generator=g)
    T = torch.where(u < t_frac / 2, 1, torch.where(u < t_frac, 2, 4))
    y = torch.tensor([pow(int(a), 2 ** int(t), int(m)) for a, t, m in zip(x.tolist(), T.tolist(), N.tolist())])
    return {"N": N, "x": x, "T": T, "y": y}


# ---------------------------------------------------------------------------------------------------------- machine
class Machine:
    """The hard (integer) machine in base B with W division steps on P = W + 1 places."""

    def __init__(self, B: int, W: int):
        self.B, self.W, self.P, self.S = B, W, W + 1, B
        self.R = 2 * B * B                                                     # table rows

    def digits(self, v: torch.Tensor, n: int | None = None) -> torch.Tensor:
        n = self.P if n is None else n
        p = torch.tensor([self.B ** i for i in range(n)], dtype=torch.long)
        return (v[..., None] // p) % self.B

    def value(self, d: torch.Tensor) -> torch.Tensor:
        p = torch.tensor([self.B ** i for i in range(d.shape[-1])], dtype=torch.long)
        return (d * p).sum(-1)

    def true_tables(self):
        B = self.B
        r = torch.arange(B)[:, None, None]
        m = torch.arange(B)[None, :, None]
        c = torch.arange(2)[None, None, :]
        x = r - m - c
        return (x % B).flatten(), (x < 0).long().flatten()

    def sub(self, A, M, Td, Tb, cin=None):
        """A - M on integer digits [X, n] (any width): digits, final borrow, table rows used [X, n]."""
        c = torch.zeros_like(A[:, 0]) if cin is None else cin
        out, rows = [], []
        for j in range(A.shape[1]):
            idx = (A[:, j] * self.B + M[:, j]) * 2 + c
            out.append(Td[idx])
            c = Tb[idx]
            rows.append(idx)
        return torch.stack(out, 1), c, torch.stack(rows, 1)

    def step(self, Rp, M, Td, Tb):
        """One division step: trials Rp - s N (s = 1 .. S - 1), keep the last before the first borrow. Returns the
        kept trial, the quotient digit, every link's input [X, S - 1, P] and every link's borrow [X, S - 1]."""
        D, R = Rp, Rp
        alive = torch.ones_like(Rp[:, 0], dtype=torch.bool)
        q = torch.zeros_like(Rp[:, 0])
        ins, bs = [], []
        for _ in range(1, self.S):
            ins.append(D)
            D, c, _ = self.sub(D, M, Td, Tb)
            bs.append(c)
            alive = alive & (c == 0)
            R = torch.where(alive[:, None], D, R)
            q = q + alive.long()
        return R, q, torch.stack(ins, 1), torch.stack(bs, 1)

    def divide(self, v, M, Td, Tb, steps=None):
        """v mod N from R = v // B^steps (below N) in `steps` steps. Returns the remainder's digits, plus every
        step's link inputs [X, steps, S - 1, P] and link borrows [X, steps, S - 1]."""
        steps = self.W if steps is None else steps
        R = self.digits(v // self.B ** steps)
        ins, bs = [], []
        for t in reversed(range(steps)):
            vt = (v // self.B ** t) % self.B
            R, _, i, b = self.step(torch.cat([vt[:, None], R[:, :-1]], 1), M, Td, Tb)
            ins.append(i)
            bs.append(b)
        return R, torch.stack(ins, 1), torch.stack(bs, 1)

    def run(self, x, T, M, Td, Tb):
        s = x.clone()
        for t in range(int(T.max()) if T.numel() else 0):
            R = self.divide(s * s, M, Td, Tb)[0]
            s = torch.where(T > t, self.value(R), s)
        return s


# ------------------------------------------------------------------------------------------------------------- soft
def soft_tables(a, b):
    """Digit and borrow distributions of the cell, laid out for the ripples: TS [2B (r, c), B * (B + 2) (m, out)]."""
    B = a.shape[0]
    TS = torch.cat([a.softmax(-1), b.softmax(-1)], -1)                          # [r, m, c, B + 2]
    return TS.permute(0, 2, 1, 3).reshape(2 * B, B * (B + 2))


def ripple(Ad, Md, TS, B, hard_c=None):
    """A - M on digit distributions Ad, Md [X, n, B] (one-hot for exact digits), the cell applied multilinearly.
    hard_c [X, n] (optional): the borrow into each place is taken from it (the hard machine's own), instead of the
    ripple's. Returns digits [X, n, B], every place's borrow-out distribution [X, n, 2]."""
    X, n = Ad.shape[:2]
    c = torch.zeros(X, 2, dtype=Ad.dtype)
    c[:, 0] = 1
    outs, cs = [], []
    for j in range(n):
        if hard_c is not None:
            c = F.one_hot(hard_c[:, j], 2).to(Ad.dtype)
        joint = (Ad[:, j, :, None] * c[:, None, :]).reshape(X, 2 * B)
        Y = torch.einsum("xmk,xm->xk", (joint @ TS).view(X, B, B + 2), Md[:, j])
        outs.append(Y[:, :B])
        c = Y[:, B:]
        cs.append(c)
    return torch.stack(outs, 1), torch.stack(cs, 1)


def less_soft(Ad, Md, TS, B):
    """P(A < M): the final borrow of A - M on digit distributions (the cell's borrow outputs only)."""
    return ripple(Ad, Md, TS, B)[1][:, -1, 1]


def less_coupled(P1, P2, same, Tb):
    """P(A < A'): A, A' on digit distributions P1, P2 [X, n, B] whose places marked `same` [X, n] read the SAME table
    entry (so their digits are equal, not two independent draws), compared by the borrow table Tb [B, B, 2, 2]
    (probabilities; the referee, detached by the caller)."""
    X, n, B = P1.shape
    c = torch.zeros(X, 2, dtype=P1.dtype)
    c[:, 0] = 1
    eye = torch.eye(B, dtype=P1.dtype)
    for j in range(n):
        J = torch.where(same[:, j, None, None], P1[:, j, :, None] * eye, P1[:, j, :, None] * P2[:, j, None, :])
        c = torch.einsum("xuv,xc,uvco->xo", J, c, Tb)
    return c[:, 1]


def add_designed(Ed, M, B):
    """The hand-written adder on digit distributions Ed [X, n, B] plus exact digits M [X, n]: digits and P(carry)."""
    o = torch.arange(B)
    c1 = torch.zeros(Ed.shape[0], dtype=Ed.dtype)
    out = []
    for j in range(Ed.shape[1]):
        d, m = Ed[:, j], M[:, j, None]
        out.append((1 - c1)[:, None] * d.gather(-1, (o - m) % B) + c1[:, None] * d.gather(-1, (o - m - 1) % B))
        below = torch.cat([torch.zeros_like(d[:, :1]), d.cumsum(-1)], -1)        # below[k] = P(d < k)
        g0 = 1 - below.gather(-1, B - m)[:, 0]
        g1 = 1 - below.gather(-1, B - 1 - m)[:, 0]
        c1 = (1 - c1) * g0 + c1 * g1
    return torch.stack(out, 1), c1


def add_hard(A, M, B):
    c = torch.zeros_like(A[:, 0])
    out = []
    for j in range(A.shape[1]):
        s = A[:, j] + M[:, j] + c
        out.append(s % B)
        c = s // B
    return torch.stack(out, 1), c


# ------------------------------------------------------------------------------------------------------- families
def families(B: int) -> dict:
    """Complete tables (digit, borrow) for the consistent wrong systems of ADDER.md §3 (those that exist in base B),
    plus the ones that keep the true borrows."""
    r = torch.arange(B)[:, None, None].expand(B, B, 2)
    m = torch.arange(B)[None, :, None].expand(B, B, 2)
    c = torch.arange(2)[None, None, :].expand(B, B, 2)
    x = r - m - c
    tb = (x < 0).long()
    out = {"true": (x % B, tb)}
    out["copy the minuend"] = (r.clone(), torch.zeros_like(r))
    out["lazy"] = (torch.where((r == m) & (c == 0), 0, r), torch.where(c == 1, 1, ((m != 0) & (r != m)).long()))
    out["no borrows at all"] = (x % B, torch.zeros_like(r))
    xs = r - m - (1 - c)
    out["swapped borrow code"] = (xs % B, 1 - (xs < 0).long())
    out["digits relabeled (d -> -d)"] = ((r - m + c) % B, (((-r) % B) - ((-m) % B) - c < 0).long())
    for k in range(2, B):
        if B % k == 0 and math.gcd(k, B // k) == 1:                       # Z/B = Z/k x Z/(B/k): borrows in one part
            j = B // k
            crt = lambda ak, aj: next(v for v in range(B) if v % k == ak and v % j == aj)
            tab = torch.tensor([[crt(a, b_) for b_ in range(j)] for a in range(k)])
            xk = r % k - m % k - c
            out[f"borrows in the mod-{k} part"] = (tab[xk % k, (r - m) % j], (xk < 0).long())
    # true borrows, wrong digits (pass the learned check: ANCHORS.md §2)
    out["true borrows, digits ignore the borrow"] = ((r - m) % B, tb)
    if B % 2 == 0:
        out["true borrows, digit r - m - [m even]"] = ((r - m - (m % 2 == 0).long()) % B, tb)
    out["true borrows, digit r - m - [m = 0]"] = ((r - m - (m == 0).long()) % B, tb)
    out["true except (r, 0, 1) -> r"] = (torch.where((m == 0) & (c == 1), r, x % B), tb)
    out["true except (0, 0, 1) -> 0"] = (torch.where((r == 0) & (m == 0) & (c == 1), 0, x % B), tb)
    return {k: (v[0].flatten(), v[1].flatten()) for k, v in out.items()}


def describe(Td, Tb, fam: dict) -> dict:
    """How a snapped table compares with the truth and the families: digits right, borrows right, rows right, and the
    family (other than the truth) matching most rows."""
    td, tb = fam["true"]
    best = max(((int(((Td == d) & (Tb == b)).sum()), k) for k, (d, b) in fam.items() if k != "true"))
    return {"digits": int((Td == td).sum()), "borrows": int((Tb == tb).sum()),
            "rows": int(((Td == td) & (Tb == tb)).sum()), "closest": f"{best[1]} {best[0]}"}


# ---------------------------------------------------------------------------------------------------------- lessons
class Lab:
    """The tables, the data and the lessons of one run."""

    def __init__(self, B, W, data, weights, seed=0, init=0.1, lr=0.05, batch=64, place0=5.0, complete=1.0,
                 warm=0, pairs=1, dtype=torch.float32, warm_lessons=None, ans_from=0, hard_cin=False,
                 tb_fixed=False, td_fixed_rows=None, gate=None, gate_win=10, late=None, late_at=0):
        self.mc = Machine(B, W)
        self.B, self.P, self.W = B, W + 1, W
        self.data, self.weights, self.batch = data, dict(weights), batch
        self.place0, self.complete, self.warm, self.pairs = place0, complete, warm, pairs
        self.warm_lessons = warm_lessons                                     # lessons during the warm start
        self.ans_from, self.hard_cin, self.dtype = ans_from, hard_cin, dtype
        self.gate, self.gate_win, self.gated_on, self.fact_win = gate, gate_win, None, []
        self.late, self.late_at = late, late_at                              # lessons that join late_at steps after
        g = torch.Generator().manual_seed(seed)
        self.g = g
        self.a = (init * torch.randn(B, B, 2, B, generator=g)).to(dtype).requires_grad_()
        self.b = (init * torch.randn(B, B, 2, 2, generator=g)).to(dtype).requires_grad_()
        self.tb_fixed = tb_fixed                                             # diagnostic: borrows held true
        td, tb = self.mc.true_tables()
        if tb_fixed:
            with torch.no_grad():
                self.b.copy_(6.0 * (2 * F.one_hot(tb.view(B, B, 2), 2) - 1).to(dtype))
        if td_fixed_rows is not None:                                        # diagnostic: some digit rows start true
            with torch.no_grad():
                av = self.a.view(-1, B)
                av[td_fixed_rows] = 4.0 * F.one_hot(td[td_fixed_rows], B).to(dtype)
        params = [self.a] if tb_fixed else [self.a, self.b]
        self.opt = torch.optim.Adam(params, lr=lr)
        self.fam = families(B)
        self.step_no = 0
        self.Nmax = int(data["N"].max())

    def tables(self):
        return self.a.argmax(-1).flatten(), self.b.argmax(-1).flatten()

    def sample(self):
        n = len(self.data["N"])
        idx = torch.randint(0, n, (self.batch,), generator=self.g)
        return {k: v[idx] for k, v in self.data.items()}

    def lessons(self, rows, weights, check_only=False):
        """Loss (sum of the weighted lessons) and every lesson's hard pass rate on this batch."""
        mc, B, P = self.mc, self.B, self.P
        hot = lambda v: F.one_hot(v, B).to(self.dtype)
        log = lambda p: p.clamp_min(1e-12).log()
        Td, Tb = self.tables()
        TS = soft_tables(self.a, self.b)
        N, x, T, y = rows["N"], rows["x"], rows["T"], rows["y"]
        X = len(N)
        M, xd = mc.digits(N), mc.digits(x)
        Z = torch.zeros_like(M)
        K = mc.W * (mc.S - 1)
        with torch.no_grad():
            _, ins, lb = mc.divide(x * x, M, Td, Tb)
            D = ins.reshape(X * K, P)
            bl = lb.reshape(X * K)                                            # each link's hard borrow
            Mr = M.repeat_interleave(K, 0)
            Rg = ins[:, :, 0].reshape(X * mc.W, P)                            # the registers (first links' inputs)
            MW = M.repeat_interleave(mc.W, 0)
        loss = torch.zeros((), dtype=self.dtype)
        hard = {}
        w = weights
        need_E = any(w.get(k, 0) > 0 for k in ("dc", "lc"))
        if need_E:
            Es, ce = ripple(hot(D), hot(Mr), TS, B)                            # the link, D - N
            be = ce[:, -1]
        Eh, beh, rows_e = mc.sub(D, Mr, Td, Tb)
        TR = TS.detach()                                                     # the order lessons' referee
        if any(w.get(k, 0) > 0 for k in ("dec", "mon", "monk", "chain", "col", "colall")):
            Eo = ripple(hot(D), hot(Mr), TS, B, hard_c=rows_e % 2)[0]        # soft digits, the hard link's carries
        pw = torch.ones(P, dtype=self.dtype)
        pw[0] = self.place0
        nll_digits = lambda Y, target: (-log(Y).gather(-1, target[..., None]).squeeze(-1) * pw).sum(-1)
        nll_plain = lambda Y, target: -log(Y).gather(-1, target[..., None]).squeeze(-1).sum(-1)
        if w.get("dc", 0) > 0:
            back, carry = add_designed(Es, Mr, B)
            nll = nll_digits(back, D) - self.complete * log(carry * be[:, 1] + (1 - carry) * be[:, 0])
            loss = loss + w["dc"] * nll.mean() / (pw.sum() + self.complete)
        Ch, bch, _ = mc.sub(Z, M, Td, Tb)
        if w.get("lc", 0) > 0 or w.get("neg2", 0) > 0 or w.get("comm", 0) > 0 or w.get("neg", 0) > 0:
            Cs, cc = ripple(hot(Z), hot(M), TS, B)
            bc = cc[:, -1]
        if w.get("lc", 0) > 0:
            back, cf = ripple(Es, Cs.repeat_interleave(K, 0), TS, B)
            bf = cf[:, -1]
            pc, pe, pf = bc[:, 1].repeat_interleave(K, 0), be[:, 1], bf[:, 1]
            agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc   # be = bc - bf
            nll = nll_digits(back, D) - self.complete * log(agree)
            loss = loss + w["lc"] * nll.mean() / (pw.sum() + self.complete)
        if w.get("zero", 0) > 0:
            Y, cy = ripple(hot(Rg), hot(torch.zeros_like(Rg)), TS, B,
                           hard_c=torch.zeros_like(Rg) if self.hard_cin else None)
            loss = loss + w["zero"] * (nll_plain(Y, Rg) - log(cy[:, -1, 0])).mean() / (P + 1)
        if w.get("self", 0) > 0:
            Y, cy = ripple(hot(Rg), hot(Rg), TS, B, hard_c=torch.zeros_like(Rg) if self.hard_cin else None)
            loss = loss + w["self"] * (nll_plain(Y, torch.zeros_like(Rg)) - log(cy[:, -1, 0])).mean() / (P + 1)
        if w.get("less", 0) > 0:
            loss = loss - w["less"] * log(less_soft(hot(xd), hot(M), TS, B)).mean()
        if w.get("more", 0) > 0:
            loss = loss - w["more"] * log(1 - less_soft(hot(M), hot(xd), TS, B)).mean()
        if w.get("pref", 0) > 0:                                             # x < N on every top part:
            tot = 0.0                                                         # x[j:] <= N[j:], strictly iff the
            for j in range(P):                                                # digit strings differ there
                differ = (xd[:, j:] != M[:, j:]).any(1)
                p1 = less_soft(hot(xd[:, j:]), hot(M[:, j:]), TS, B)
                p2 = less_soft(hot(M[:, j:]), hot(xd[:, j:]), TS, B)
                tot = tot - torch.where(differ, log(p1), log(1 - p1)).mean() - log(1 - p2).mean()
            loss = loss + w["pref"] * tot / P
        if w.get("wrap", 0) > 0:
            Ew = ripple(hot(xd), hot(M), TS, B)[0]
            loss = loss - w["wrap"] * log(1 - less_soft(Ew, hot(M), TS, B)).mean()
        if w.get("neg", 0) > 0:
            loss = loss - w["neg"] * log(bc[:, 1]).mean()
        if w.get("neg2", 0) > 0:                                             # 0 - (0 - N) = N, complete
            Y2, c2 = ripple(hot(Z), Cs, TS, B)
            pc, p2 = bc[:, 1], c2[:, -1, 1]
            agree2 = pc * p2 + (1 - pc) * (1 - p2)                           # 0 + N does not carry: b2 = bc
            loss = loss + w["neg2"] * (nll_plain(Y2, M) - log(agree2)).mean() / (P + 1)
        if w.get("comm", 0) > 0:                                             # R + N = N + R: R - (0 - N) = N - (0 - R)
            A1, c1 = ripple(hot(Rg), Cs.repeat_interleave(mc.W, 0), TS, B)
            CR, ccr = ripple(hot(torch.zeros_like(Rg)), hot(Rg), TS, B)
            A2, c2 = ripple(hot(MW), CR, TS, B)
            k1 = bc[:, 1].repeat_interleave(mc.W, 0) * c1[:, -1, 0]          # P(carry of R + N)
            k2 = ccr[:, -1, 1] * c2[:, -1, 0]                                # P(carry of N + R)
            agree = log((A1 * A2).sum(-1)).sum(-1) + log(k1 * k2 + (1 - k1) * (1 - k2))
            loss = loss - w["comm"] * agree.mean() / (P + 1)
        if w.get("chain", 0) > 0:                                            # after a borrowing link: no borrow
            sel = bl == 1
            if sel.any():
                pn = less_soft(Eo[sel], hot(Mr[sel]), TR, B)
                loss = loss - w["chain"] * log(1 - pn).mean()
        if w.get("dec", 0) > 0:                                              # E < D iff the link does not borrow
            pl = less_soft(Eo, hot(D), TR, B)
            tgt = (bl == 0).to(self.dtype)
            loss = loss - w["dec"] * (tgt * log(pl) + (1 - tgt) * log(1 - pl)).mean()
        if w.get("mon", 0) > 0:                                              # pairs of links of one row
            j = torch.randint(0, K, (X * K * self.pairs,), generator=self.g)
            i0 = torch.arange(X * K).repeat(self.pairs)
            i1 = (i0 // K) * K + j
            keep = i1 != i0
            i0, i1 = i0[keep], i1[keep]
            with torch.no_grad():
                dl = mc.sub(D[i0], D[i1], Td, Tb)[1]                         # D_i < D_j (the learned comparator)
                b0, b1 = bl[i0], bl[i1]
                tgt = (b0 < b1) | ((b0 == b1) & (dl == 1))                   # [f(D_i) < f(D_j)]: the one that
                lo, hi = torch.where(tgt, i0, i1), torch.where(tgt, i1, i0)  # does not wrap, or the smaller one
                same = rows_e[lo] == rows_e[hi]                              # places reading the same entry
            pl = less_coupled(Eo[lo], Eo[hi], same, self.b.detach().softmax(-1))
            loss = loss - w["mon"] * log(pl).mean()
            hard["mon"] = float((mc.sub(Eh[lo], Eh[hi], Td, Tb)[1] == 1).float().mean())
        if w.get("monk", 0) > 0:                                             # the same at every width k:
            j = torch.randint(0, K, (X * K * self.pairs,), generator=self.g)  # the lowest k places of a
            i0 = torch.arange(X * K).repeat(self.pairs)                       # subtraction are a subtraction
            i1 = (i0 // K) * K + j                                            # mod B^k, so it keeps order too
            keep = i1 != i0
            i0, i1 = i0[keep], i1[keep]
            Tbp = self.b.detach().softmax(-1)
            tot = 0.0
            with torch.no_grad():
                _, _, rows_d = mc.sub(D[i0], D[i1], Td, Tb)                  # the learned comparator's borrows
                bk0 = Tb[rows_e[i0]]                                         # borrow out of each place of D - N
                bk1 = Tb[rows_e[i1]]
                dlk = Tb[rows_d]                                             # [D_i mod B^k < D_j mod B^k]
                same_all = rows_e[i0] == rows_e[i1]
            okh = []
            for k in range(1, P + 1):
                b0, b1, dl = bk0[:, k - 1], bk1[:, k - 1], dlk[:, k - 1]
                tgt = (b0 < b1) | ((b0 == b1) & (dl == 1))
                lo, hi = torch.where(tgt, i0, i1), torch.where(tgt, i1, i0)
                same = torch.where(tgt[:, None], same_all, same_all)[:, :k]
                pl = less_coupled(Eo[lo, :k], Eo[hi, :k], same, Tbp)
                tot = tot - log(pl).mean()
                with torch.no_grad():
                    okh.append(mc.sub(Eh[lo, :k], Eh[hi, :k], Td, Tb)[1] == 1)
            loss = loss + w["monk"] * tot / P
            hard["monk"] = float(torch.stack(okh).float().mean())
        if w.get("col", 0) > 0:                                              # one digit at a time: two places
            j = torch.randint(0, K, (X * K * self.pairs,), generator=self.g)  # of one row's links that read the
            i0 = torch.arange(X * K).repeat(self.pairs)                       # same column (N's digit, borrow in)
            i1 = (i0 // K) * K + j                                            # keep their digits' order, unless
            with torch.no_grad():                                             # exactly one of them wraps
                e0, e1 = rows_e[i0], rows_e[i1]                               # entries [n, P]: (r B + m) 2 + c
                r0, r1 = e0 // (2 * B), e1 // (2 * B)
                ok = ((e0 % (2 * B)) == (e1 % (2 * B))) & (r0 != r1)          # same m and c, different r
                w0, w1 = Tb[e0], Tb[e1]                                       # do they wrap (the learned borrows)
                less_r = Tb[(r0 * B + r1) * 2]                                # [r0 < r1], the learned comparator
                tgt = (w0 < w1) | ((w0 == w1) & (less_r == 1))                # [o0 < o1]
                lo_e, hi_e = torch.where(tgt, e0, e1)[ok], torch.where(tgt, e1, e0)[ok]
            if lo_e.numel():
                pa = self.a.softmax(-1).reshape(-1, B)                        # every entry's digit distribution
                plo, phi = pa[lo_e], pa[hi_e]
                p_less = (plo[:, :, None] * phi[:, None, :] * torch.triu(torch.ones(B, B), 1)).sum((1, 2))
                loss = loss - w["col"] * log(p_less).mean()
                hard["col"] = float((Td[lo_e] < Td[hi_e]).float().mean())
        if w.get("colall", 0) > 0:                                           # the one-digit order law on every
            with torch.no_grad():                                             # pair of digits r < r' of each column
                used = torch.bincount(rows_e.flatten(), minlength=2 * B * B) > 0  # (m, c) the batch's links use
                ent = torch.arange(2 * B * B)
                rr, mc_ = ent // (2 * B), ent % (2 * B)
                e0 = ent[:, None].expand(-1, B).reshape(-1)                   # (r, m, c) paired with (r', m, c)
                r1 = torch.arange(B).repeat(2 * B * B)
                e1 = r1 * 2 * B + mc_[e0]
                ok = (rr[e0] < r1) & used[e0] & used[e1]
                e0, e1 = e0[ok], e1[ok]
                w0, w1 = Tb[e0], Tb[e1]
                lt = Tb[(rr[e0] * B + r1[ok]) * 2]                            # [r < r'], the learned comparator
                first = (w0 < w1) | ((w0 == w1) & (lt == 1))
                lo, hi = torch.where(first, e0, e1), torch.where(first, e1, e0)
            pa = self.a.softmax(-1).reshape(-1, B)
            above = 1 - pa[hi].cumsum(-1)
            loss = loss - w["colall"] * log((pa[lo] * above).sum(-1)).mean()
            hard["colall"] = float((Td[lo] < Td[hi]).float().mean())
        if w.get("rowall", 0) > 0:                                           # subtracting more gives less: the
            with torch.no_grad():                                             # entries (r, m, c), (r, m', c) of
                used = torch.bincount(rows_e.flatten(), minlength=2 * B * B) > 0  # one row r and borrow c keep the
                ent = torch.arange(2 * B * B)                                 # reversed order of m < m', unless
                rr, mm, cc = ent // (2 * B), (ent // 2) % B, ent % 2          # exactly one of them wraps
                e0 = ent[:, None].expand(-1, B).reshape(-1)
                m1 = torch.arange(B).repeat(2 * B * B)
                e1 = (rr[e0] * B + m1) * 2 + cc[e0]
                lt = Tb[(mm[e0] * B + m1) * 2]                                # [m < m'], the learned comparator
                ok = (lt == 1) & used[e0] & used[e1]
                e0, e1 = e0[ok], e1[ok]
                w0, w1 = Tb[e0], Tb[e1]
                first = (w0 < w1)                                             # e0 (less taken) comes out smaller
                lo = torch.where(first | (w0 != w1), torch.where(first, e0, e1), e1)   # same wrap: the larger m
                hi = torch.where(first | (w0 != w1), torch.where(first, e1, e0), e0)
            pa = self.a.softmax(-1).reshape(-1, B)
            above = 1 - pa[hi].cumsum(-1)
            loss = loss - w["rowall"] * log((pa[lo] * above).sum(-1)).mean()
            hard["rowall"] = float((Td[lo] < Td[hi]).float().mean())
        if w.get("bijr", 0) > 0:                                             # one digit minus different digits:
            q = self.a.softmax(-1).mean(1)                                    # each row uses every value once
            loss = loss - w["bijr"] * log(B * q).mean()
        if w.get("bij", 0) > 0:                                              # subtracting a fixed digit with a
            q = self.a.softmax(-1).mean(0)                                    # fixed borrow permutes the digits:
            loss = loss - w["bij"] * log(B * q).mean()                        # each column uses every value once
        if w.get("mon0", 0) > 0:                                             # 0 - N falls as N grows
            perm = torch.randperm(X, generator=self.g)
            with torch.no_grad():
                tgt = (mc.sub(M[perm], M, Td, Tb)[1] == 1).to(self.dtype)    # N' < N  =>  0 - N < 0 - N'
            Co = ripple(hot(Z), hot(M), TS, B, hard_c=mc.sub(Z, M, Td, Tb)[2] % 2)[0]
            pl = less_soft(Co, Co[perm], TR, B)
            loss = loss - w["mon0"] * (tgt * log(pl) + (1 - tgt) * log(1 - pl)).mean()
        if w.get("ans", 0) > 0 and self.step_no > self.ans_from:
            sel = torch.nonzero(T <= 2).flatten()
            if sel.numel():
                loss = loss + w["ans"] * self.answer_loss(rows, sel, TS, Td, Tb)
        if w.get("xmod", 0) > 0:                                             # one step of x / N keeps x
            pass                                                             # (given hard borrows it is "less")
        if not check_only:
            pass
        # hard versions of every lesson (the snapped table on its own numbers)
        with torch.no_grad():
            Fh = mc.sub(Eh, Ch.repeat_interleave(K, 0), Td, Tb)
            hard["lc"] = float((Fh[0] == D).all(1).float().mean())
            hard["lc_complete"] = float(((Fh[0] == D).all(1) & (beh == bch.repeat_interleave(K, 0) - Fh[1])).float().mean())
            bk, cy = add_hard(Eh, Mr, B)
            hard["dc"] = float(((bk == D).all(1) & (cy == beh)).float().mean())
            hard["less"] = float((mc.sub(xd, M, Td, Tb)[1] == 1).float().mean())
            hard["more"] = float((mc.sub(M, xd, Td, Tb)[1] == 0).float().mean())
            Exh = mc.sub(xd, M, Td, Tb)[0]
            hard["wrap"] = float((mc.sub(Exh, M, Td, Tb)[1] == 0).float().mean())
            Yz, cz, _ = mc.sub(Rg, torch.zeros_like(Rg), Td, Tb)
            hard["zero"] = float(((Yz == Rg).all(1) & (cz == 0)).float().mean())
            Ys, cs_, _ = mc.sub(Rg, Rg, Td, Tb)
            hard["self"] = float(((Ys == 0).all(1) & (cs_ == 0)).float().mean())
            hard["neg"] = float((bch == 1).float().mean())
            hard["neg2"] = float((mc.sub(Z, Ch, Td, Tb)[0] == M).all(1).float().mean())
            sel = bl == 1
            hard["chain"] = float((mc.sub(Eh[sel], Mr[sel], Td, Tb)[1] == 0).float().mean()) if sel.any() else 1.0
            hard["dec"] = float(((mc.sub(Eh, D, Td, Tb)[1] == 1) == (bl == 0)).float().mean())
            hard["xmod"] = float((mc.divide(x, M, Td, Tb, steps=1)[0] == xd).all(1).float().mean())
            sel = torch.nonzero(T <= 2).flatten()
            if sel.numel():
                sp = mc.run(x[sel], T[sel], M[sel], Td, Tb)
                hard["ans"] = float((sp == y[sel]).float().mean())
            else:
                hard["ans"] = float("nan")
        return loss, hard

    def answer_loss(self, rows, sel, TS, Td, Tb):
        """The rows' answers along the machine's own hard path: each step's quotient digit from the hard machine, the
        soft cell running that many links from the soft register; CE of the remainder's digits below N's width."""
        mc, B, P = self.mc, self.B, self.P
        N, x, T, y = (rows[k][sel] for k in ("N", "x", "T", "y"))
        M = mc.digits(N)
        with torch.no_grad():
            sp = mc.run(x, T - 1, M, Td, Tb)
        v = sp * sp
        Rh = mc.digits(v // B ** mc.W)
        Rs = F.one_hot(Rh, B).to(self.dtype)
        for t in reversed(range(mc.W)):
            vt = (v // B ** t) % B
            with torch.no_grad():
                Rh, q, _, _ = mc.step(torch.cat([vt[:, None], Rh[:, :-1]], 1), M, Td, Tb)
            Dd = torch.cat([F.one_hot(vt, B).to(self.dtype)[:, None], Rs[:, :-1]], 1)
            Rs = Dd * (q == 0).to(self.dtype)[:, None, None]
            for s in range(1, int(q.max()) + 1):
                Dd = ripple(Dd, F.one_hot(M, B).to(self.dtype), TS, B)[0]
                Rs = Rs + Dd * (q == s).to(self.dtype)[:, None, None]
        width = torch.tensor([len(_digits_of(int(n), B)) for n in N.tolist()])
        scored = (torch.arange(P)[None] < width[:, None]).to(self.dtype)
        nll = -Rs.clamp_min(1e-12).log().gather(-1, mc.digits(y)[..., None]).squeeze(-1)
        return (nll * scored).sum() / scored.sum()

    def train_step(self):
        self.step_no += 1
        rows = self.sample()
        w = self.weights
        if self.step_no <= self.warm and self.warm_lessons is not None:
            w = self.warm_lessons
        if self.gate is not None and self.gated_on is None:
            w = {k: v for k, v in w.items() if k in ("less", "more", "neg", "pref")}
        elif self.late and self.gated_on is not None and self.step_no - self.gated_on >= self.late_at:
            w = {**w, **self.late}
        loss, hard = self.lessons(rows, w)
        if self.gate is not None and self.gated_on is None:
            self.fact_win.append(min(hard["less"], hard["more"]))
            self.fact_win = self.fact_win[-self.gate_win:]
            if len(self.fact_win) == self.gate_win and min(self.fact_win) >= self.gate:
                self.gated_on = self.step_no
        self.opt.zero_grad()
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.a, self.b], 1.0)
            self.opt.step()
        return float(loss), hard

    def state(self):
        Td, Tb = self.tables()
        return describe(Td, Tb, self.fam)


def _digits_of(v: int, B: int) -> list:
    out = []
    while True:
        out.append(v % B)
        v //= B
        if v == 0:
            return out


def table_str(Td, Tb, B):
    """Wrong entries of a snapped table as '(r,m,c)->(d,b)' (true (d', b'))."""
    td, tb = Machine(B, 1).true_tables()
    out = []
    for i in torch.nonzero((Td != td) | (Tb != tb)).flatten().tolist():
        r, m, c = i // (2 * B), (i // 2) % B, i % 2
        out.append(f"({r},{m},{c})->{int(Td[i])}{int(Tb[i])}")
    return " ".join(out)


def run(B=10, W=8, data=None, weights=None, steps=300, seed=0, log_every=25, verbose=True, **kw):
    """One training run; returns the trajectory of (step, state, hard checks)."""
    if data is None:
        data = h1_rows(4096, seed=seed)
    lab = Lab(B, W, data, weights, seed=seed, **kw)
    traj = []
    t0 = time.time()
    for s in range(1, steps + 1):
        loss, hard = lab.train_step()
        if s % log_every == 0 or s == steps:
            st = lab.state()
            traj.append({"step": s, "loss": loss, **st, "gated_on": lab.gated_on,
                         "hard": {k: round(v, 4) for k, v in hard.items()}})
            if verbose:
                hs = " ".join(f"{k}={v:.3f}" for k, v in hard.items() if k in ("lc", "dc", "dec", "wrap", "zero", "self", "ans", "mon", "monk", "col", "colall", "rowall"))
                print(f"step {s:4d} gate {lab.gated_on} loss {loss:.4f} digits {st['digits']:3d} borrows {st['borrows']:3d} rows {st['rows']:3d} "
                      f"| {st['closest']} | {hs} | {time.time() - t0:.0f}s", flush=True)
    return lab, traj
