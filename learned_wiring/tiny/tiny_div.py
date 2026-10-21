"""Stage 1 in miniature: the combined H1 machine's division, base 10, with its wiring learned (WIRING.md section 2).

The machine is CHAIN.md + CONTROL.md at small widths, with H1's own 200-row subtraction table and 2x2 commit gate:
- N has at most W digits (training uses narrower N, as H1's 13-20-bit moduli sit in 8 slots), x < N, and the
  dividend is x^2 in 2W digits;
- the register R (P = W + 1 places) starts at the dividend's top W digits; each of W steps shifts in the next digit
  and runs a chain of 9 links, each a ripple of the learned table, whose result the gate (on the link's final borrow)
  commits or holds; the answer is R's low W places.

Its wiring as tied switches, one per kind of connection (TINYROUTING.md's grammar, on the gated chain). The options
are listed in a fixed order that does not start with the true one; TRUE marks it (only the plumbing check and the
report use it):
  vread  the dividend's digits as given | reversed (which end the steps start from)
  shift  the step's register Rp from R and the incoming digit d: up (10 R + d) | down | stay (d replaces place 0) |
         skip (R)
  read   the answer: R's low W places | its high W places
  nread  the subtrahend register: N's digits as given | reversed
  a      each link's minuend: the chain register (prev) | the step's start Rp | N
  b      each link's subtrahend: N | prev | Rp
  keep   what a link writes into the chain register: its result when the gate commits | always | never
  gate   the learned 2x2 table: borrow -> hold | commit (CONTROL.md); in the mixtures its two rows are two switches

Lessons (WIRING.md section 2 says why each):
  table   the check (A - B) + B = A on every link of a 9-link chain for every (a, b, nread) option from the step
          registers of the machine's own hard division (one step per row; the union does not depend on the wiring),
          complete (final borrow = carry), place 0 weighted 5; the designed adder for the first WARM steps, then the
          learned one (A - B) - (0 - B); and the facts about the row's numbers: x - N borrows, N - x does not,
          (x - N) - N does not. From step WARM also 0 + N = N and R + N = N + R with the learned adder;
  chain   (once every table check has been exact for WIN steps) the division ladder: one step from the register
          x + q N (q = 0..9) must give back x. CHAIN = "fmix": every combination of a, b, keep, nread and the gate's
          two rows runs as a hard step with the snapped table, and the verdicts are blended with weights that are the
          product of the switches' probabilities; CHAIN = "soft": the options are blended on the wire (digit
          distributions, the snapped table), cross-entropy against x;
  io      (once the argmax chain passes the division ladder for WIN steps) x mod N = x and N mod N = 0 through the
          whole division, and the labeled rows' answers x^2 mod N: every combination of vread, shift and read runs
          as a hard division (the argmax chain), verdicts blended by the product weights (IO = "fmix") or by one
          weight per combination (IO = "mix").
Training stops once the argmax machine has been exact on every lesson for DONE steps; then it is certified on the
whole training pool and on N of the full width W (never trained).

    ~/github/sotaku/venv/bin/python -m learned_wiring.tiny.tiny_div seed=0 [KEY=VALUE ...]
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

CFG = dict(
    W=4,               # slots of N and of the answer; the dividend has 2W digits, the register P = W + 1 places
    widths=(2, 3),     # N's digit counts in training (certification adds the full width W)
    pool=40000,        # training rows (N, x) drawn once
    batch=256,
    label_frac=0.02,   # share of rows whose answer x^2 mod N is used (H1: T = 1 rows are 1% of training)
    seed=0,
    init=0.1,
    lr=0.05,
    warm=70,           # designed adder in the table's check for the first `warm` steps
    win=20,            # steps in a row before the next lesson joins
    done=50,           # steps in a row exact on every lesson to stop
    max_steps=3000,
    chain="fmix",      # fmix | soft
    io="fmix",         # fmix | mix | soft (the I/O switches blended on the wire) | soft_all (every switch and the
                       # gate blended on the wire through the whole division, the snapped table)
    io_laws=1,         # 1: x mod N = x and N mod N = 0 join the answers in the I/O lesson; 0: the answers alone
    lad_rows=64,       # rows of the batch used by the division ladder (x 10 registers)
    chain_from=-1,     # >= 0: the chain lesson joins at this step regardless of the table (diagnostics)
    union="all",       # the check's chains: every (a, b, nread) option | "true": only the true one (diagnostics)
    ste_cin=1,         # 1: the table's lessons ripple with the snapped borrows (straight-through)
    facts=1.0,         # weight of the facts about x and N
    operands="own",    # the check's operands: own (the machine's own step registers minus N again and again) |
                       # union (every (a, b, nread) option's chain) | random | rownums | rowchains (diagnostics)
    log=50,
    cert_rows=20000,
)

B = 10
MENU = {
    "vread": ("given", "reversed"),
    "shift": ("down", "stay", "up", "skip"),
    "read": ("high", "low"),
    "nread": ("reversed", "given"),
    "a": ("start", "N", "prev"),
    "b": ("prev", "start", "N"),
    "keep": ("all", "gated", "none"),
}
TRUE = {"vread": 0, "shift": 2, "read": 1, "nread": 1, "a": 2, "b": 2, "keep": 1}
IO_SW = ("vread", "shift", "read")
CHAIN_SW = ("nread", "a", "b", "keep")


def digits_np(v, n):
    v = np.asarray(v, dtype=np.int64).copy()
    out = np.empty((len(v), n), dtype=np.int64)
    for i in range(n):
        out[:, i] = v % 10
        v //= 10
    return out


def value(d):
    return (d * (10 ** torch.arange(d.shape[-1]))).sum(-1)


class Data:
    def __init__(self, W, widths, n, seed, label_frac):
        rng = np.random.default_rng(seed)
        wd = rng.choice(np.array(widths), n)
        lo, hi = 10 ** (wd - 1), 10 ** wd
        N = rng.integers(np.maximum(lo, 2), hi)
        x = (rng.random(n) * N).astype(np.int64)
        V = x * x
        self.N, self.x, self.V, self.y = N, x, V, V % N
        self.Nd = torch.from_numpy(digits_np(N, W))
        self.xd = torch.from_numpy(digits_np(x, W))
        self.Vd = torch.from_numpy(digits_np(V, 2 * W))
        self.yd = torch.from_numpy(digits_np(V % N, W))
        self.lab = torch.from_numpy(rng.random(n) < label_frac)
        self.n = n

    def rows(self, idx):
        return self.Nd[idx], self.xd[idx], self.Vd[idx], self.yd[idx], self.lab[idx]


# ---------------------------------------------------------------------------------------------- hard machine (ints)
def sel(idx, opts):
    """Per-row selection among options [X, ...] by idx [X]."""
    out = opts[0]
    for i, o in enumerate(opts[1:], 1):
        out = torch.where((idx == i).view(-1, *([1] * (o.dim() - 1))), o, out)
    return out


def sub_hard(A, Bo, Td, Tb):
    c = torch.zeros_like(A[:, 0])
    out, rows = [], []
    for j in range(A.shape[1]):
        r = (A[:, j] * 10 + Bo[:, j]) * 2 + c
        out.append(Td[r])
        c = Tb[r]
        rows.append(r)
    return torch.stack(out, 1), c, torch.stack(rows, 1)


def add_hard(A, Bo):
    c = torch.zeros_like(A[:, 0])
    out = []
    for j in range(A.shape[1]):
        s = A[:, j] + Bo[:, j] + c
        out.append(s % 10)
        c = s // 10
    return torch.stack(out, 1), c


def shift_opts(R, d):
    return [torch.cat([R[:, 1:], d[:, None]], 1),          # down
            torch.cat([d[:, None], R[:, 1:]], 1),          # stay
            torch.cat([d[:, None], R[:, :-1]], 1),         # up
            R]                                             # skip


def nreg(Nd, nread, P):
    return F.pad(sel(nread, [Nd.flip(-1), Nd]), (0, P - Nd.shape[1]))


def chain_hard(Rp, M, w, Td, Tb, links=None):
    """One step's chain on ints with per-row options w (a, b, keep: [X]; g0, g1: commit on borrow 0 / 1 [X] bool).
    Returns the register and the number of writes. `links` collects every link's (A, B)."""
    D = Rp
    writes = torch.zeros_like(Rp[:, 0])
    for _ in range(B - 1):
        A = sel(w["a"], [Rp, M, D])
        Bo = sel(w["b"], [D, Rp, M])
        E, c, _ = sub_hard(A, Bo, Td, Tb)
        if links is not None:
            links.append((A, Bo))
        commit = torch.where(c == 0, w["g0"], w["g1"])
        take = torch.where(w["keep"] == 1, commit, w["keep"] == 0)
        D = torch.where(take[:, None], E, D)
        writes = writes + take.long()
    return D, writes


def divide_hard(Vd, Nd, w, Td, Tb, regs=None):
    """The whole division on ints with per-row options w. Returns the answer [X, W]; `regs` collects each step's Rp."""
    W = Nd.shape[1]
    P = W + 1
    Vp = sel(w["vread"], [Vd, Vd.flip(-1)])
    M = nreg(Nd, w["nread"], P)
    R = F.pad(Vp[:, W:2 * W], (0, P - W))
    for t in reversed(range(W)):
        Rp = sel(w["shift"], shift_opts(R, Vp[:, t]))
        if regs is not None:
            regs.append(Rp)
        R, _ = chain_hard(Rp, M, w, Td, Tb)
    return sel(w["read"], [R[:, 1:W + 1], R[:, :W]])


# ---------------------------------------------------------------------------------------------- soft pieces
STE_CIN = [False]      # the check's ripples carry the snapped borrow forward (its gradient the soft one's)


def soft_sub(D, M, TDC):
    """D - M on digit distributions [X, L, 10] (multilinear). TDC [20, 120]: rows (r, c), columns (m, digit + borrow).
    Returns the digits [X, L, 10] and the borrow out of every place [X, L, 2]. With STE_CIN each place's borrow in is
    the snapped borrow of the entry used below it, straight-through: a lesson then trains only the entries it uses
    (ABLATION.md rule 3)."""
    X = D.shape[0]
    c = torch.zeros(X, 2, dtype=D.dtype)
    c[:, 0] = 1
    out, bor = [], []
    for j in range(D.shape[1]):
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = torch.einsum("xmk,xm->xk", (joint @ TDC).view(X, 10, 12), M[:, j])
        out.append(Y[:, :10])
        c = Y[:, 10:]
        bor.append(c)                                         # the soft borrow of the entry used (for the lessons)
        if STE_CIN[0]:
            c = F.one_hot(c.argmax(-1), 2).to(c.dtype) + (c - c.detach())
    return torch.stack(out, 1), torch.stack(bor, 1)


def add_soft(D, M):
    """Designed adder on digit distributions D [X, L, 10] + exact digits M [X, L]; returns digits and P(carry)."""
    o = torch.arange(10)
    c1 = torch.zeros(D.shape[0], dtype=D.dtype)
    out = []
    for j in range(D.shape[1]):
        d, m = D[:, j], M[:, j, None]
        out.append((1 - c1)[:, None] * d.gather(-1, (o - m) % 10) + c1[:, None] * d.gather(-1, (o - m - 1) % 10))
        below = torch.cat([torch.zeros_like(d[:, :1]), d.cumsum(-1)], -1)
        g0 = 1 - below.gather(-1, 10 - m)
        g1 = 1 - below.gather(-1, 9 - m)
        c1 = (1 - c1) * g0[:, 0] + c1 * g1[:, 0]
    return torch.stack(out, 1), c1


def hot(v):
    return F.one_hot(v, 10).float()


def blend(p, opts):
    return sum(p[i] * o for i, o in enumerate(opts))


def divide_soft(V, Nd, p, TH, pg):
    """The whole division on digit distributions V [X, 2W, 10], every switch blended on the wire by its probabilities
    p[name] [n], the gate P(commit | borrow 0, 1) pg [2], the table TH (snapped, one-hot, or soft). Returns [X, W, 10]."""
    W = Nd.shape[1]
    P = W + 1
    X = V.shape[0]
    z = hot(torch.zeros(X, P - W, dtype=torch.long))
    Vp = blend(p["vread"], [V, V.flip(1)])
    M = blend(p["nread"], [torch.cat([hot(Nd.flip(-1)), z], 1), torch.cat([hot(Nd), z], 1)])
    R = torch.cat([Vp[:, W:2 * W], z], 1)
    for t in reversed(range(W)):
        d = Vp[:, t]
        Rp = blend(p["shift"], [torch.cat([R[:, 1:], d[:, None]], 1), torch.cat([d[:, None], R[:, 1:]], 1),
                                torch.cat([d[:, None], R[:, :-1]], 1), R])
        D = Rp
        for _ in range(B - 1):
            A_ = blend(p["a"], [Rp, M, D])
            B_ = blend(p["b"], [D, Rp, M])
            E_, bo_ = soft_sub(A_, B_, TH)
            take = p["keep"][1] * (bo_[:, -1] @ pg) + p["keep"][0]
            D = take[:, None, None] * E_ + (1 - take[:, None, None]) * D
            D = D / D.sum(-1, keepdim=True).clamp_min(1e-12)
        R = D
    return blend(p["read"], [R[:, 1:W + 1], R[:, :W]])


# ---------------------------------------------------------------------------------------------- the model
class Tiny(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        g = torch.Generator().manual_seed(cfg["seed"])
        r = lambda *s: torch.nn.Parameter(cfg["init"] * torch.randn(*s, generator=g))
        self.ta, self.tb = r(10, 10, 2, 10), r(10, 10, 2, 2)
        self.g = r(2, 2)
        self.sw = torch.nn.ParameterDict({k: r(len(v)) for k, v in MENU.items()})
        self.mix_io = r(2 * 4 * 2)                            # IO = "mix": one weight per (vread, shift, read)
        self.io_mode = cfg["io"]

    def tables(self):
        return self.ta.argmax(-1).flatten(), self.tb.argmax(-1).flatten()

    def soft_table(self):
        T = torch.cat([self.ta.softmax(-1), self.tb.softmax(-1)], -1)
        return T.permute(0, 2, 1, 3).reshape(20, 120)

    def hard_table(self):
        T = torch.cat([F.one_hot(self.ta.argmax(-1), 10), F.one_hot(self.tb.argmax(-1), 2)], -1).float()
        return T.permute(0, 2, 1, 3).reshape(20, 120)

    def choice(self):
        """The argmax option of every switch (IO = "mix": the I/O switches from the mixture's best combination)."""
        c = {k: int(v.argmax()) for k, v in self.sw.items()}
        if self.io_mode == "mix":
            best = list(itertools.product(*[range(len(MENU[n])) for n in IO_SW]))[int(self.mix_io.argmax())]
            c.update(zip(IO_SW, best))
        return c

    def argmax_w(self, X):
        w = {k: torch.full((X,), v) for k, v in self.choice().items()}
        gc = self.g.argmax(-1) == 1
        w["g0"], w["g1"] = gc[0].expand(X), gc[1].expand(X)
        return w

    def wiring(self):
        return {k: MENU[k][v] for k, v in self.choice().items()} | {
            "gate": "".join("C" if int(c) == 1 else "H" for c in self.g.argmax(-1))}

    def true_wiring(self):
        c = self.choice()
        return all(c[k] == TRUE[k] for k in MENU) and self.g.argmax(-1).tolist() == [1, 0]


def combos(names, extra_gate=False):
    sizes = [len(MENU[n]) for n in names] + ([2, 2] if extra_gate else [])
    return torch.tensor(list(itertools.product(*[range(s) for s in sizes])))


def combo_logw(model, names, C, gate):
    """log of the product weight of every combination C [K, len(names) (+2)]."""
    lw = 0
    for i, n in enumerate(names):
        lw = lw + model.sw[n].log_softmax(-1)[C[:, i]]
    if gate:
        lg = model.g.log_softmax(-1)                          # [borrow, (hold, commit)]
        lw = lw + lg[0][C[:, -2]] + lg[1][C[:, -1]]
    return lw


def run(cfg, verbose=True):
    torch.manual_seed(cfg["seed"])
    STE_CIN[0] = bool(cfg["ste_cin"])
    W = cfg["W"]
    P = W + 1
    data = Data(W, cfg["widths"], cfg["pool"], cfg["seed"], cfg["label_frac"])
    model = Tiny(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(cfg["seed"] + 1)
    log = lambda p: p.clamp_min(1e-12).log()
    CC = combos(CHAIN_SW, extra_gate=True)                    # 3*3*3*2*2*2 = 216 chain combinations
    CI = combos(IO_SW)                                        # 16 I/O combinations
    phase, streak, hist, t0 = "table", 0, [], time.time()
    stop_at = None
    for step in range(1, cfg["max_steps"] + 1):
        idx = rng.integers(0, data.n, cfg["batch"])
        Nd, xd, Vd, yd, lab = data.rows(torch.from_numpy(idx))
        X = Nd.shape[0]
        Td, Tb = model.tables()
        TDC = model.soft_table()
        warm = step <= cfg["warm"]
        M0 = F.pad(Nd, (0, P - W))
        xp = F.pad(xd, (0, P - W))
        # ---- the table: the check on the union of the (a, b, nread) options' chains from the machine's own registers
        with torch.no_grad():
            regs = []
            divide_hard(Vd, Nd, model.argmax_w(X), Td, Tb, regs)
            Rall = torch.stack(regs, 1)                                                  # [X, W, P]
            Rp = Rall[torch.arange(X), (torch.arange(X) + step) % W]                     # one step per row
            K9 = torch.tensor(list(itertools.product(range(3), range(3), range(2))))     # (a, b, nread)
            if cfg["union"] == "true":
                K9 = torch.tensor([[TRUE["a"], TRUE["b"], TRUE["nread"]]])
            elif cfg["union"] != "all":                       # e.g. "pN,sN": (a, b) codes s/N/p, both N readings
                code = {"s": 0, "N": 1, "p": 2}
                bcode = {"p": 0, "s": 1, "N": 2}
                K9 = torch.tensor([[code[ab[0]], bcode[ab[1]], int(ab[2]) if len(ab) > 2 else r]
                                   for ab in cfg["union"].split(",") for r in range(2 if len(ab) == 2 else 1)])
            n = K9.shape[0]
            wU = {"a": K9[:, 0].repeat_interleave(X), "b": K9[:, 1].repeat_interleave(X),
                  "keep": torch.zeros(n * X, dtype=torch.long),                          # "all": every link writes
                  "g0": torch.ones(n * X, dtype=torch.bool), "g1": torch.ones(n * X, dtype=torch.bool)}
            links = []
            chain_hard(Rp.repeat(n, 1), nreg(Nd.repeat(n, 1), K9[:, 2].repeat_interleave(X), P), wU, Td, Tb, links)
            A = torch.cat([l[0] for l in links])
            Bo = torch.cat([l[1] for l in links])
            if cfg["operands"] == "own":                       # the machine's own step registers minus N, again
                D_, Mr_ = Rall.reshape(-1, P), M0.repeat_interleave(W, 0)               # and again (every step)
                As, Bs = [], []
                for _ in range(B - 1):
                    As.append(D_)
                    Bs.append(Mr_)
                    D_ = sub_hard(D_, Mr_, Td, Tb)[0]
                A, Bo = torch.cat(As), torch.cat(Bs)
            elif cfg["operands"] == "random":                  # uniform random pairs (TINYROUTING's "cell")
                g_ = torch.Generator().manual_seed(step)
                A = torch.randint(0, 10, (9 * X, P), generator=g_)
                Bo = torch.randint(0, 10, (9 * X, P), generator=g_)
            elif cfg["operands"] == "rownums":                 # ordered pairs of the row's own numbers
                Vw = [F.pad(Vd[:, t:t + P], (0, max(0, P - Vd[:, t:t + P].shape[1]))) for t in range(W + 1)]
                S_ = torch.stack([xp, M0, torch.zeros_like(M0)] + Vw, 1)            # [X, k, P]
                k_ = S_.shape[1]
                ii, jj = torch.meshgrid(torch.arange(k_), torch.arange(k_), indexing="ij")
                A = S_[:, ii.flatten()].reshape(-1, P)
                Bo = S_[:, jj.flatten()].reshape(-1, P)
            elif cfg["operands"] == "rowchains":               # the row's numbers minus N again and again
                Vw = [F.pad(Vd[:, t:t + P], (0, max(0, P - Vd[:, t:t + P].shape[1]))) for t in range(W + 1)]
                S_ = torch.stack([xp, M0] + Vw, 1).reshape(-1, P)
                Mr_ = M0.repeat_interleave(len(Vw) + 2, 0)
                As, Bs, D_ = [], [], S_
                for _ in range(B - 1):
                    As.append(D_)
                    Bs.append(Mr_)
                    D_ = sub_hard(D_, Mr_, Td, Tb)[0]
                A, Bo = torch.cat(As), torch.cat(Bs)
        weight = torch.ones(P)
        weight[0] = 5.0
        Es, be = soft_sub(hot(A), hot(Bo), TDC)
        pe = be[:, -1, 1]
        if warm:
            back, carry = add_soft(Es, Bo)
            agree = carry * pe + (1 - carry) * (1 - pe)
        else:
            Cs, bc = soft_sub(hot(torch.zeros_like(Bo)), hot(Bo), TDC)
            back, bf = soft_sub(Es, Cs, TDC)
            pc, pf = bc[:, -1, 1], bf[:, -1, 1]
            agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc
        nll = (-log(back).gather(-1, A[..., None]).squeeze(-1) * weight).sum(-1) - log(agree)
        loss = nll.mean() / (weight.sum() + 1)
        # the facts about the row's numbers
        _, bl = soft_sub(hot(xp), hot(M0), TDC)
        _, bm = soft_sub(hot(M0), hot(xp), TDC)
        Ew, _ = soft_sub(hot(xp), hot(M0), TDC)
        _, bw = soft_sub(Ew, hot(M0), TDC)
        loss = loss - cfg["facts"] * (log(bl[:, -1, 1]).mean() + log(bm[:, -1, 0]).mean() + log(bw[:, -1, 0]).mean())
        if not warm:                                          # 0 + N = N, R + N = N + R with the learned adder
            Cn, bcn = soft_sub(hot(torch.zeros_like(M0)), hot(M0), TDC)
            Y2, b2 = soft_sub(hot(torch.zeros_like(M0)), Cn, TDC)
            agree2 = bcn[:, -1, 1] * b2[:, -1, 1] + bcn[:, -1, 0] * b2[:, -1, 0]
            loss = loss + (-log(Y2).gather(-1, M0[..., None]).squeeze(-1).sum(-1) - log(agree2)).mean() / (P + 1)
            A1, b1 = soft_sub(hot(Rp), Cn, TDC)
            CR, bcr = soft_sub(hot(torch.zeros_like(Rp)), hot(Rp), TDC)
            A2, b2b = soft_sub(hot(M0), CR, TDC)
            c1 = bcn[:, -1, 1] * b1[:, -1, 0]
            c2 = bcr[:, -1, 1] * b2b[:, -1, 0]
            loss = loss - (log((A1 * A2).sum(-1)).sum(-1) + log(c1 * c2 + (1 - c1) * (1 - c2))).mean() / (P + 1)
        with torch.no_grad():                                 # the same checks on the hard machine
            Eh, beh, _ = sub_hard(A, Bo, Td, Tb)
            if warm:
                bk, ch = add_hard(Eh, Bo)
                ok = (bk == A).all(1) & (ch == beh)
            else:
                Ch, bch, _ = sub_hard(torch.zeros_like(Bo), Bo, Td, Tb)
                bk, bfh, _ = sub_hard(Eh, Ch, Td, Tb)
                ok = (bk == A).all(1) & (beh == bch - bfh)
            table_ok = float(ok.float().mean())
            Exh, blh, _ = sub_hard(xp, M0, Td, Tb)
            facts = float(((blh == 1) & (sub_hard(M0, xp, Td, Tb)[1] == 0) & (sub_hard(Exh, M0, Td, Tb)[1] == 0))
                          .float().mean())
            table_ok = min(table_ok, facts)
        # ---- the chain's switches: the division ladder, one step from x + q N must give back x
        chain_on = (phase in ("chain", "io", "done") or (cfg["chain_from"] >= 0 and step >= cfg["chain_from"])) \
            and cfg["io"] != "soft_all"
        lad_ok = 0.0
        if chain_on:
            nl = min(cfg["lad_rows"], X)
            q = torch.arange(10)
            xv, Nv = value(xd[:nl]), value(Nd[:nl])
            Rq = torch.from_numpy(digits_np((xv[:, None] + q[None] * Nv[:, None]).flatten().numpy(), P))  # [nl*10, P]
            xq = xp[:nl].repeat_interleave(10, 0)
            Ndq = Nd[:nl].repeat_interleave(10, 0)
            if cfg["chain"] == "fmix":
                K = CC.shape[0]
                m = Rq.shape[0]
                with torch.no_grad():
                    wK = {"nread": CC[:, 0].repeat_interleave(m), "a": CC[:, 1].repeat_interleave(m),
                          "b": CC[:, 2].repeat_interleave(m), "keep": CC[:, 3].repeat_interleave(m),
                          "g0": (CC[:, 4] == 1).repeat_interleave(m), "g1": (CC[:, 5] == 1).repeat_interleave(m)}
                    R, _ = chain_hard(Rq.repeat(K, 1), nreg(Ndq.repeat(K, 1), wK["nread"], P), wK, Td, Tb)
                    verdict = (R == xq.repeat(K, 1)).all(1).view(K, m).float()
                lw = combo_logw(model, CHAIN_SW, CC, gate=True)                   # [K]
                lik = torch.logsumexp(lw[:, None] + log(verdict), 0)              # [m]
                loss = loss - lik.mean()
            else:                                                                # soft on the wire, snapped table
                TH = model.hard_table()
                p = {k: model.sw[k].softmax(-1) for k in CHAIN_SW}
                pg = model.g.softmax(-1)[:, 1]
                Rs = hot(Rq)
                Mr = p["nread"][0] * hot(F.pad(Ndq.flip(-1), (0, P - W))) + p["nread"][1] * hot(F.pad(Ndq, (0, P - W)))
                D = Rs
                for _ in range(B - 1):
                    A_ = p["a"][0] * Rs + p["a"][1] * Mr + p["a"][2] * D
                    B_ = p["b"][0] * D + p["b"][1] * Rs + p["b"][2] * Mr
                    E_, bo_ = soft_sub(A_, B_, TH)
                    k_ = bo_[:, -1] @ pg
                    take = p["keep"][1] * k_ + p["keep"][0]
                    D = take[:, None, None] * E_ + (1 - take[:, None, None]) * D
                loss = loss - log(D).gather(-1, xq[..., None]).mean()
            with torch.no_grad():
                wa = model.argmax_w(Rq.shape[0])
                Rh, _ = chain_hard(Rq, nreg(Ndq, wa["nread"], P), wa, Td, Tb)
                lad_ok = float((Rh == xq).all(1).float().mean())
        # ---- the division's I/O switches: x mod N = x, N mod N = 0 through the whole division, and the answers
        io_ok = 0.0
        if phase in ("io", "done"):
            K = CI.shape[0]
            with torch.no_grad():
                wa = model.argmax_w(X * K)
                for i, n_ in enumerate(IO_SW):
                    wa[n_] = CI[:, i].repeat_interleave(X)
                Nr = Nd.repeat(K, 1)
                okx = (divide_hard(F.pad(xd, (0, W)).repeat(K, 1), Nr, wa, Td, Tb) == xd.repeat(K, 1)).all(1)
                okn = (divide_hard(F.pad(Nd, (0, W)).repeat(K, 1), Nr, wa, Td, Tb) == 0).all(1)
                if cfg["io_laws"] == 0:                       # the answers alone
                    okx, okn = torch.ones_like(okx), torch.ones_like(okn)
                oky = (divide_hard(Vd.repeat(K, 1), Nr, wa, Td, Tb) == yd.repeat(K, 1)).all(1) | ~lab.repeat(K)
                verdict = (okx & okn & oky).view(K, X).float()
            if cfg["io"] in ("soft", "soft_all"):            # blended on the wire, cross-entropy on the targets
                TH = model.hard_table()
                ch = model.choice()
                p = {k: (model.sw[k].softmax(-1) if (k in IO_SW or cfg["io"] == "soft_all")
                         else F.one_hot(torch.tensor(ch[k]), len(MENU[k])).float()) for k in MENU}
                pg = model.g.softmax(-1)[:, 1] if cfg["io"] == "soft_all" else (model.g.argmax(-1) == 1).float()
                tgt = [(hot(F.pad(xd, (0, W))), xd, torch.ones(X, dtype=torch.bool)),
                       (hot(F.pad(Nd, (0, W))), torch.zeros_like(xd), torch.ones(X, dtype=torch.bool)),
                       (hot(Vd), yd, lab)]
                if cfg["io_laws"] == 0:
                    tgt = tgt[2:]
                for Vin, y_, use in tgt:
                    if use.any():
                        out = divide_soft(Vin[use], Nd[use], p, TH, pg)
                        loss = loss - log(out).gather(-1, y_[use][..., None]).mean()
            else:
                if cfg["io"] == "fmix":
                    lw = combo_logw(model, IO_SW, CI, gate=False)
                else:
                    lw = model.mix_io.log_softmax(-1)
                loss = loss - torch.logsumexp(lw[:, None] + log(verdict), 0).mean()
        with torch.no_grad():                                 # the argmax machine on every lesson
            wa = model.argmax_w(X)
            okx = (divide_hard(F.pad(xd, (0, W)), Nd, wa, Td, Tb) == xd).all(1)
            okn = (divide_hard(F.pad(Nd, (0, W)), Nd, wa, Td, Tb) == 0).all(1)
            oky = (divide_hard(Vd, Nd, wa, Td, Tb) == yd).all(1)
            io_ok = float((okx & okn & (oky | ~lab)).float().mean())
            ans_all = float(oky.float().mean())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # ---- the schedule
        before = phase
        if phase == "table":
            streak = streak + 1 if table_ok == 1.0 else 0
            if streak >= cfg["win"]:
                phase, streak = "chain", 0
        elif phase == "chain" and cfg["io"] == "soft_all":    # every switch is taught through the whole division
            phase, streak = "io", 0
        elif phase == "chain":
            streak = streak + 1 if (table_ok == 1.0 and lad_ok == 1.0) else 0
            if streak >= cfg["win"]:
                phase, streak = "io", 0
        elif phase == "io":
            chain_ok = lad_ok == 1.0 or cfg["io"] == "soft_all"
            streak = streak + 1 if (table_ok == 1.0 and chain_ok and io_ok == 1.0) else 0
            if streak >= cfg["done"]:
                phase = "done"
        if phase != before:
            hist.append((step, phase))
        if verbose and (step % cfg["log"] == 0 or step == 1 or phase != before):
            rows_right = int((Td == torch.from_numpy(exact_td())).sum() + (Tb == torch.from_numpy(exact_tb())).sum())
            print(f"step={step} phase={phase} loss={float(loss):.4f} table_ok={table_ok:.4f} lad_ok={lad_ok:.4f} "
                  f"io_ok={io_ok:.4f} ans_all={ans_all:.4f} table={rows_right}/400 wiring={model.wiring()}",
                  flush=True)
        if phase == "done":
            stop_at = step
            break
    res = certify(model, cfg)
    res.update(cfg={k: v for k, v in cfg.items()}, hist=hist, stop=stop_at, secs=round(time.time() - t0, 1),
               wiring=model.wiring(), true_wiring=model.true_wiring())
    return res


def exact_td():
    r, m, c = np.meshgrid(np.arange(10), np.arange(10), np.arange(2), indexing="ij")
    return ((r - m - c) % 10).flatten()


def exact_tb():
    r, m, c = np.meshgrid(np.arange(10), np.arange(10), np.arange(2), indexing="ij")
    return ((r - m - c) < 0).astype(np.int64).flatten()


@torch.no_grad()
def certify(model, cfg):
    """The argmax machine on the whole training pool and on N of the full width W (unseen)."""
    W = cfg["W"]
    Td, Tb = model.tables()
    table = int((Td == torch.from_numpy(exact_td())).sum() + (Tb == torch.from_numpy(exact_tb())).sum())
    out = {"table": table}
    td, tb = torch.from_numpy(exact_td()), torch.from_numpy(exact_tb())
    out["wrong"] = [((i // 20, (i // 2) % 10, i % 2), int(Td[i]), int(Tb[i])) for i in range(200)
                    if Td[i] != td[i] or Tb[i] != tb[i]][:24]
    for name, widths in (("train", cfg["widths"]), ("full", (W,))):
        d = Data(W, widths, cfg["cert_rows"], cfg["seed"] + 7, 1.0)
        X = d.n
        y = divide_hard(d.Vd, d.Nd, model.argmax_w(X), Td, Tb)
        out[name] = float((y == d.yd).all(1).float().mean())
    out["certified"] = out["train"] == 1.0 and out["full"] == 1.0
    return out


def main(argv):
    cfg = dict(CFG)
    for kv in argv:
        k, v = kv.split("=", 1)
        old = cfg[k]
        cfg[k] = (tuple(int(t) for t in v.split(",")) if isinstance(old, tuple) else
                  type(old)(v) if not isinstance(old, str) else v)
    torch.set_num_threads(2)
    res = run(cfg)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
