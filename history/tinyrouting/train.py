"""Training the tiny machine's wiring and table together, in every parametrization of the wiring.

Parametrizations (`param`), for the muxes in `learn` (the others are fixed to the true option):
- "given"     the wiring is fixed to the truth; only the table learns (the baseline).
- "soft"      per-connection soft routing: one softmax per mux, the options' digit distributions mixed on the wire.
- "st"        hard-path routing: the forward pass runs the argmax option of every mux; the backward pass uses the
              soft routing's gradient at that point (straight-through).
- "gumbel"    the same with a Gumbel-perturbed argmax forward and a tempered softmax backward (`tau`).
- "mix"       a weighted mixture over whole programs: every combination of the learned muxes' options is run as a
              hard machine, and the answer's likelihood is mixed with one softmax weight per program (SpecBit).
- "fmix"      the same mixture with factorized weights: program weight = product of per-mux softmaxes.
- "hybrid"    a mixture over the programs of the `coarse` muxes, with soft routing for the other learned muxes.
- "reinforce" sample `K` programs from per-mux softmaxes per step; the table learns along the samples, the
              routing from the score function with a mean baseline.
The table is learned from the answers ("learn") or given ("exact"). Label-free checks (`checks`, weights):
- "cell"   the table's contract (a - b) + b = a with a designed adder, on every pair of (w+1)-digit numbers;
           it teaches the table its meaning without the wiring.
- "laws"   the reduce unit's laws on spec-made inputs: x mod N = x for every x < N, and N mod N = 0 (the chain and
           write-back run on them; reads a, b, keep).
- "range"  after every step of the machine's own run, the register is below N (a designed comparator).
- "steplabel" (not allowed by the rules; a control) every step's register equals Rp mod N.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from tinyrouting import machine as M
from tinyrouting.machine import MUXES, ORDER, Data, onehot, program_P, programs, run, table_TDC


@dataclass
class Cfg:
    B: int = 3
    w: int = 2
    k: int = 1
    learn: tuple = ("shift", "a", "b", "keep")
    param: str = "mix"
    coarse: tuple = ()
    table: str = "learn"
    init_table: float = 0.1
    init_route: float = 0.1
    lr: float = 0.05
    route_lr: float = 0.0          # 0 = same as lr
    steps: int = 3000
    every: int = 25
    seed: int = 0
    answers: float = 1.0
    checks: dict = field(default_factory=dict)
    tau: float = 1.0
    K: int = 16
    stages: tuple = ()             # one-quotient-digit-at-a-time: k values, each trained until hard-exact
    stage_steps: int = 0           # max steps per stage (0 = steps)
    own_path: bool = False         # tables along the model's own hard path (hard routing and hard write-back)
    held: tuple = ()               # held-out sizes (B, w, k) for certification
    route_freeze: int = 0          # steps with the routing frozen at its init (tables first)
    table_freeze: int = 0          # steps with the table frozen (routing first)
    search: dict = field(default_factory=dict)   # try-and-keep: {"every": n, "tries": m}
    sym_init: bool = False         # exactly uniform routing logits and table logits (no symmetry breaking)
    untie: tuple = ()              # muxes with one routing per step / link instead of one shared (soft, st, hybrid)
    post: int = 100                # steps trained on after the training answers first become exact, before the
                                   # final measure and certification (the hit step is the first exact step)
    patience: int = 600            # stop when neither the argmax wiring nor the exact share has changed for this
                                   # many steps (0 = never)
    batch: int = 0                 # examples per step (0 = every example)


class Model:
    def __init__(self, cfg: Cfg):
        self.cfg = c = cfg
        B = c.B
        g = torch.Generator().manual_seed(c.seed)
        self.gen = g
        self.TDt, self.TCt = M.truth_table(B)
        if c.sym_init:
            self.td = torch.zeros(B, B, 2, B)
            self.tc = torch.zeros(B, B, 2, 2)
        else:
            self.td = torch.randn(B, B, 2, B, generator=g) * c.init_table
            self.tc = torch.randn(B, B, 2, 2, generator=g) * c.init_table
        self.td.requires_grad_(c.table == "learn")
        self.tc.requires_grad_(c.table == "learn")
        self.learn = [m for m in ORDER if m in c.learn] if c.param != "given" else []
        if c.untie:
            assert c.param in ("soft", "st", "gumbel", "hybrid"), "untied muxes need per-mux routing"
            self.learn = [p for m in self.learn for p in (M.untied_keys(m, B, c.k) if m in c.untie else [m])]
        # every learned mux's options in a seed-dependent internal order, so no tie favors the true option 0
        rng = np.random.default_rng(1000 + c.seed)
        self.perm = {m: torch.from_numpy(rng.permutation(len(MUXES[m.split("@")[0]]))) for m in self.learn}
        self.alpha = {}
        coarse = [m for m in self.learn if m in c.coarse] if c.param == "hybrid" else []
        enum = self.learn if c.param == "mix" else coarse
        self.fine = [m for m in self.learn if m not in enum]
        self.enum = enum
        if enum:
            progs = programs(enum)
            order = rng.permutation(len(progs))
            self.progs = [progs[i] for i in order]
            H = len(self.progs)
            self.beta = (torch.zeros(H) if c.sym_init else torch.randn(H, generator=g) * c.init_route)
            self.beta.requires_grad_(True)
        else:
            self.progs, self.beta = None, None
        for m in (self.fine if c.param in ("mix", "hybrid") else self.learn):
            n = len(MUXES[m.split("@")[0]])
            self.alpha[m] = (torch.zeros(n) if c.sym_init else torch.randn(n, generator=g) * c.init_route)
            self.alpha[m].requires_grad_(True)

    # ----- parameters -----
    def table_params(self):
        return [self.td, self.tc] if self.cfg.table == "learn" else []

    def route_params(self):
        return list(self.alpha.values()) + ([self.beta] if self.beta is not None else [])

    def tables(self):
        if self.cfg.table == "exact":
            return self.TDt, self.TCt
        return self.td.softmax(-1), self.tc.softmax(-1)

    def canon(self, m, p_int):
        """Internal-order weights [..., n] -> canonical option order."""
        out = torch.zeros_like(p_int)
        out[..., self.perm[m]] = p_int
        return out

    def soft_route(self, m):
        return self.canon(m, self.alpha[m].softmax(-1))

    def marginals(self):
        """Every learned mux's routing distribution in canonical order (for a mixture: the marginal of the program
        weights)."""
        out = {}
        for m in self.learn:
            if m in self.alpha:
                out[m] = self.soft_route(m).detach()
            else:
                pi = self.beta.softmax(0).detach()
                v = torch.zeros(len(MUXES[m.split("@")[0]]))
                for p, w_ in zip(self.progs, pi):
                    v[p[m]] += w_
                out[m] = v
        return out

    def argmax_program(self):
        prog = M.true_program()
        if self.beta is not None:
            prog.update(self.progs[int(self.beta.argmax())])
        for m in self.alpha:
            prog[m] = int(self.soft_route(m).argmax())
        for m in self.cfg.untie:                                               # a position never trained (a longer
            keys = [k_ for k_ in prog if k_.startswith(m + "@")]               # run) repeats the last position
            if keys:
                prog[m] = prog[keys[-1]]
        return prog

    # ----- routing weights for a forward pass -----
    def P_soft(self, hard=False, gumbel=False):
        """One wiring (H = 1): soft routing, or straight-through hard routing."""
        P = {}
        for m in list(ORDER) + [k_ for k_ in self.alpha if "@" in k_]:
            n = len(MUXES[m.split("@")[0]])
            if m in self.alpha:
                a = self.alpha[m]
                if gumbel:
                    u = torch.rand(n, generator=self.gen).clamp(1e-12, 1 - 1e-12)
                    a = a - torch.log(-torch.log(u))
                    soft = (a / self.cfg.tau).softmax(-1)
                else:
                    soft = a.softmax(-1)
                if hard or gumbel:
                    h = onehot(soft.argmax(), n)
                    soft = h + soft - soft.detach()
                P[m] = self.canon(m, soft)[None]
            else:
                P[m] = onehot(torch.tensor([0]), n)
        return P

    def P_enum(self):
        """Every enumerated program (H rows), with the fine muxes soft (hybrid) and the rest true."""
        H = len(self.progs)
        P = program_P(self.progs, self.enum, H=H)
        for m in self.fine:
            P[m] = self.soft_route(m)[None].expand(H, -1)
        return P

    def log_pi(self):
        return self.beta.log_softmax(0)

    def P_sample(self, K):
        """K programs sampled from the per-mux softmaxes: one-hot rows and their log-probabilities [K]."""
        P, logp = {}, torch.zeros(K)
        for m in ORDER:
            n = len(MUXES[m])
            if m in self.alpha:
                p = self.soft_route(m)
                idx = torch.multinomial(p.detach(), K, replacement=True, generator=self.gen)
                P[m] = onehot(idx, n)
                logp = logp + p.log()[idx]
            else:
                P[m] = onehot(torch.zeros(K, dtype=torch.long), n)
        return P, logp


# ----- label-free checks -----
def add_soft(T, Bd):
    """Designed adder: digit distributions T [H, X, P, B] plus exact digits Bd [X, P] -> sum digit distributions
    [H, X, P, B] mod B^P and P(final carry) [H, X]."""
    H, X, P, B = T.shape
    o = torch.arange(B)
    c1 = torch.zeros(H, X)
    out = []
    for j in range(P):
        d, m = T[:, :, j], Bd[None, :, j, None].expand(H, X, 1)
        s0 = d.gather(-1, ((o[None, None] - m) % B))
        s1 = d.gather(-1, ((o[None, None] - m - 1) % B))
        out.append((1 - c1)[..., None] * s0 + c1[..., None] * s1)
        below = torch.cat([torch.zeros(H, X, 1), d.cumsum(-1)], -1)
        g0 = 1 - below.gather(-1, B - m)
        g1 = 1 - below.gather(-1, (B - 1 - m).clamp_min(0))
        c1 = (1 - c1) * g0[..., 0] + c1 * g1[..., 0]
    return torch.stack(out, 2), c1


class Checks:
    """Spec-made inputs for the label-free checks of one size."""

    def __init__(self, data: Data):
        B, w = data.B, data.w
        P = w + 1
        n = B ** P
        a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        a, b = a.ravel(), b.ravel()
        self.cA = torch.from_numpy(M.digits(a, B, P))
        self.cB = torch.from_numpy(M.digits(b, B, P))
        self.cBorrow = torch.from_numpy((a < b).astype(np.int64))
        Ns, xs = [], []
        for N in range(max(2, B ** (w - 1)), B ** w):
            xs += list(range(N)) + [N]
            Ns += [N] * (N + 1)
        Ns, xs = np.array(Ns), np.array(xs)
        self.lN = torch.from_numpy(M.digits(Ns, B, w))
        self.lx = torch.from_numpy(M.digits(xs, B, P))
        self.ly = torch.from_numpy(M.digits(xs % Ns, B, P))


def cell_loss(model: Model, ck: Checks):
    """(a - b) + b = a on every pair, with the soft table and the designed adder; plus the final borrow = (a < b)."""
    TD, TC = model.tables()
    TDC = table_TDC(TD, TC)
    B = model.cfg.B
    A1, B1 = onehot(ck.cA, B)[None], onehot(ck.cB, B)[None]
    T, c = M.ripple(A1, B1, TDC)
    S, _ = add_soft(T, ck.cB)
    nll = -S.clamp_min(1e-300).log().gather(-1, ck.cA[None, ..., None]).squeeze(-1).mean()
    pb = torch.stack([1 - c, c], -1).clamp_min(1e-300).log()
    nllb = -pb.gather(-1, ck.cBorrow[None, :, None]).mean()
    return nll + nllb


def anchor_loss(model: Model):
    """A fixed meaning for the table's two operands: the law x - 0 = x (rows (x, 0, borrow 0) -> digit x, borrow 0).
    Under the port twin the transposed table would need 0 - x = x, so the anchor breaks that symmetry."""
    TD, TC = model.tables()
    B = model.cfg.B
    x = torch.arange(B)
    return -(TD[x, 0, 0, x].clamp_min(1e-300).log().mean() + TC[x, 0, 0, 0].clamp_min(1e-300).log().mean())


def laws_rows(model: Model, ck: Checks, P, TDC):
    """Per-row log-likelihood [H, rows] of the reduce unit's laws: reduce(x) = x mod N for x <= N, x and N exact."""
    B = model.cfg.B
    H = P["a"].shape[0]
    Rp = onehot(ck.lx, B)[None].expand(H, -1, -1, -1)
    n1 = onehot(ck.lN, B)[None]
    zero = torch.zeros(1, n1.shape[1], 1, B)
    zero[..., 0] = 1
    Nreg = torch.cat([n1, zero], 2).expand(H, -1, -1, -1)
    R = M.reduce_step(Rp, Nreg, P, TDC, B)
    return R.clamp_min(1e-300).log().gather(-1, ck.ly[None, ..., None].expand(H, -1, -1, 1)).squeeze(-1).sum(-1)


def below_N(R, Nd):
    """P(R < N) for digit distributions R [H, X, P, B] and N's exact digits Nd [X, w] (top places 0)."""
    H, X, P, B = R.shape
    n = torch.cat([Nd, torch.zeros(X, P - Nd.shape[1], dtype=torch.long)], 1)
    lt = torch.zeros(H, X)
    eq = torch.ones(H, X)
    for j in reversed(range(P)):
        d = R[:, :, j]
        below = torch.cat([torch.zeros(H, X, 1), d.cumsum(-1)], -1)
        p_lt = below.gather(-1, n[None, :, j, None].expand(H, X, 1))[..., 0]
        p_eq = d.gather(-1, n[None, :, j, None].expand(H, X, 1))[..., 0]
        lt = lt + eq * p_lt
        eq = eq * p_eq
    return lt


class Trainer:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.model = Model(cfg)
        self.rng = np.random.default_rng(cfg.seed)
        self.set_data(cfg.k)
        self.ck = Checks(self.data) if any(n in cfg.checks for n in ("cell", "laws")) else None
        self.held = [Data(*s) for s in cfg.held]
        lr_r = cfg.route_lr or cfg.lr
        groups = []
        if self.model.table_params():
            groups.append({"params": self.model.table_params(), "lr": cfg.lr})
        if self.model.route_params():
            groups.append({"params": self.model.route_params(), "lr": lr_r})
        self.opt = torch.optim.Adam(groups) if groups else None

    # ----- the answer loss in each parametrization -----
    def answer_loss(self, data: Data, record_regs=False):
        m, c = self.model, self.cfg
        TDC = table_TDC(*m.tables())
        regs = [] if record_regs else None
        if c.param in ("given", "soft", "st", "gumbel"):
            P = m.P_soft(hard=c.param == "st", gumbel=c.param == "gumbel")
            if c.own_path:
                ans = self.own_path_answer(data, P)
            else:
                ans = run(data, P, TDC, regs=regs)
            ll = M.row_loglik(ans, data)[0]
            return -ll.mean() / data.w, regs, P, None
        if c.param in ("mix", "hybrid"):
            P = m.P_enum()
            ans = run(data, P, TDC, regs=regs)
            ll = M.row_loglik(ans, data)                                        # [H, X]
            lp = m.log_pi()
            return -torch.logsumexp(lp[:, None] + ll, 0).mean() / data.w, regs, P, lp
        if c.param == "fmix":
            progs = programs(m.learn)
            P = program_P(progs, m.learn)
            ans = run(data, P, TDC, regs=regs)
            ll = M.row_loglik(ans, data)
            lp = sum(m.alpha[mm].log_softmax(-1)[m.perm[mm].argsort()][torch.tensor([p[mm] for p in progs])]
                     for mm in m.learn)
            return -torch.logsumexp(lp[:, None] + ll, 0).mean() / data.w, regs, P, lp
        if c.param == "reinforce":
            P, logp = m.P_sample(c.K)
            ans = run(data, P, TDC, regs=regs)
            ll = M.row_loglik(ans, data)                                        # [K, X]
            r = ll.mean(1).detach()
            adv = r - r.mean()
            surrogate = -(adv * logp).mean() - ll.mean() / data.w
            return surrogate, regs, P, None
        raise ValueError(c.param)

    def own_path_answer(self, data: Data, P):
        """The answer along the model's own hard path: every step keeps the trial that the hard machine (argmax table,
        the routing P's forward values) keeps; the soft table runs the chain on the soft register up to that trial."""
        m = self.model
        B, w, k = data.B, data.w, data.k
        TD, TC = m.tables()
        TDC = table_TDC(TD, TC)
        TDCh = table_TDC(onehot(TD.argmax(-1), B), onehot(TC.argmax(-1), 2))
        Ph = {mm: onehot(P[mm].detach().argmax(-1), P[mm].shape[1]) for mm in P}
        assert not self.cfg.untie, "own path with untied muxes is not implemented"
        vin, Nreg = M.inputs(data, P)
        vinh, Nregh = M.inputs(data, Ph)
        zero = torch.zeros(1, data.X, 1, B)
        zero[..., 0] = 1
        R = torch.cat([vin[:, :, k:k + w], zero], 2)
        Rh = torch.cat([vinh[:, :, k:k + w], zero], 2)
        for t in reversed(range(k)):
            Rp = M.mix(P["shift"], M.shift_options(R, vin[:, :, t]))
            Rph = M.mix(Ph["shift"], M.shift_options(Rh, vinh[:, :, t]))
            # the hard chain and its keep decision
            trials_h, borrows_h, prev = [Rph], [], Rph
            trials, prev_s = [Rp], Rp
            for _ in range(1, B):
                T, cb = M.ripple(M.mix(Ph["a"], [prev, Rph, Nregh]), M.mix(Ph["b"], [Nregh, prev, Rph]), TDCh)
                T = onehot(T.argmax(-1), B)
                trials_h.append(T)
                borrows_h.append((cb > 0.5).double())
                prev = T
                Ts, _ = M.ripple(M.mix(P["a"], [prev_s, Rp, Nreg]), M.mix(P["b"], [Nreg, prev_s, Rp]), TDC)
                trials.append(Ts)
                prev_s = Ts
            kw = M.keep_weights(borrows_h)
            kh = sum(Ph["keep"][:, i, None, None] * kw[i] for i in range(4))     # [1, X, S+1] one-hot
            Rh = torch.einsum("hxs,hxspb->hxpb", kh, torch.stack(trials_h, 2))
            R = torch.einsum("hxs,hxspb->hxpb", kh, torch.stack(trials, 2))
        return M.mix(P["read"], [R[:, :, :w], R[:, :, 1:w + 1]])

    # ----- checks -----
    def check_loss(self, regs, P, lp, data):
        m, c = self.model, self.cfg
        out, total = {}, 0.0
        TDC = table_TDC(*m.tables())
        for name, lam in c.checks.items():
            if not lam:
                continue
            if name == "cell":
                L = cell_loss(m, self.ck)
            elif name == "cellown":
                L = self.cellown_loss(data)
            elif name == "anchor":
                L = anchor_loss(m)
            elif name == "laws":
                ll = laws_rows(m, self.ck, P, TDC)                               # [H, rows]
                if lp is None or ll.shape[0] == 1:
                    L = -ll.mean() / (c.w + 1)
                else:
                    L = -torch.logsumexp(lp[:, None] + ll, 0).mean() / (c.w + 1)
            elif name in ("range", "steplabel"):
                lls = []
                for i, (Rp, R, _) in enumerate(regs):
                    if name == "range":
                        lls.append(below_N(R, data.Nd).clamp_min(1e-300).log())
                    else:
                        y = self.step_targets(data)[i]
                        lls.append(R.clamp_min(1e-300).log().gather(-1, y[None, ..., None].expand(
                            R.shape[0], -1, -1, 1)).squeeze(-1).sum(-1) / (c.w + 1))
                ll = torch.stack(lls, 0).sum(0)                                   # [H, X]
                if lp is None or ll.shape[0] == 1:
                    L = -ll.mean()
                else:
                    L = -torch.logsumexp(lp[:, None] + ll, 0).mean()
            else:
                raise ValueError(name)
            out[name] = float(L.detach())
            total = total + lam * L
        return total, out

    def cellown_loss(self, data):
        """The table's contract (a - b) + b = a on the machine's own link inputs: the argmax wiring run with the
        snapped table (its own hard trials, as H1's chain does), the soft table applied to those exact inputs."""
        m = self.model
        B = data.B
        TD, TC = m.tables()
        TDCh = table_TDC(onehot(TD.argmax(-1), B), onehot(TC.argmax(-1), 2))
        record = []
        with torch.no_grad():
            run(data, program_P([m.argmax_program()], ORDER), TDCh, record=record)
        A = torch.cat([r[0][0] for r in record], 0).argmax(-1)                # [links * X, P]
        Bd = torch.cat([r[1][0] for r in record], 0).argmax(-1)
        T, c = M.ripple(onehot(A, B)[None], onehot(Bd, B)[None], table_TDC(TD, TC))
        S, _ = add_soft(T, Bd)
        nll = -S.clamp_min(1e-300).log().gather(-1, A[None, ..., None]).mean()
        place = B ** torch.arange(A.shape[1])
        lt = ((A * place).sum(1) < (Bd * place).sum(1)).long()
        pb = torch.stack([1 - c, c], -1).clamp_min(1e-300).log()
        return nll - pb.gather(-1, lt[None, :, None]).mean()

    def step_targets(self, d):
        TDC = table_TDC(*M.truth_table(d.B))
        regs = []
        with torch.no_grad():
            run(d, program_P([M.true_program()], ORDER), TDC, regs=regs)
        return [R[0].argmax(-1) for (_, R, _) in regs]

    # ----- measures -----
    @torch.no_grad()
    def measure(self, data=None):
        m = self.model
        data = data or self.data
        prog = m.argmax_program()
        TD, TC = m.tables()
        ex = M.hard_exact(data, prog, TD, TC)
        TDh, TCh = TD.argmax(-1), TC.argmax(-1)
        tdt, tct = self.model.TDt.argmax(-1), self.model.TCt.argmax(-1)
        rows = ((TDh == tdt) & (TCh == tct)).double().mean().item()
        rows_t = ((TDh == tdt.transpose(0, 1)) & (TCh == tct.transpose(0, 1))).double().mean().item()
        marg = m.marginals()
        return {"exact": ex, "prog": {k_: MUXES[k_.split("@")[0]][v] for k_, v in prog.items() if k_ in m.learn},
                "rows": rows, "rows_t": rows_t,
                "route_max": {k_: round(float(v.max()), 3) for k_, v in marg.items()},
                "route_true": {k_: round(float(v[0]), 3) for k_, v in marg.items()}}

    def certify(self):
        m = self.model
        prog = m.argmax_program()
        TD, TC = m.tables()
        return [M.hard_exact(d, prog, TD, TC) for d in self.held]

    def set_data(self, k):
        self.data = Data(self.cfg.B, self.cfg.w, k)
        self.eval_data = self.subset(np.random.default_rng(7).permutation(self.data.X)[:128])

    def subset(self, idx):
        return self.data if len(idx) >= self.data.X else self.data.subset(np.sort(idx))

    def sample_batch(self):
        b = self.cfg.batch
        if not b or b >= self.data.X:
            return self.data
        return self.data.subset(self.rng.choice(self.data.X, b, replace=False))

    # ----- training -----
    def loss(self, data=None):
        c = self.cfg
        data = data or self.data
        need_regs = any(n in c.checks and c.checks[n] for n in ("range", "steplabel"))
        L = torch.zeros(())
        parts = {}
        La, regs, P, lp = self.answer_loss(data, record_regs=need_regs)
        parts["ans"] = float(La.detach())
        if c.answers:
            L = L + c.answers * La
        if c.checks:
            Lc, out = self.check_loss(regs, P, lp, data)
            parts.update(out)
            L = L + Lc
        return L, parts

    def try_and_keep(self, step):
        """Discrete local search over the hard wiring: from the current argmax wiring, try every single-mux change,
        and keep the one whose loss (current table) is lowest, by setting the routing logits to prefer it."""
        m = self.model
        cur = m.argmax_program()
        best = None
        with torch.no_grad():
            TDC = table_TDC(*m.tables())
            for mm in m.learn:
                for o in range(len(MUXES[mm.split("@")[0]])):
                    cand = dict(cur, **{mm: o})
                    ans = run(self.data, program_P([cand], tuple(ORDER) + tuple(m.learn)), TDC)
                    L = -M.row_loglik(ans, self.data).mean().item()
                    if best is None or L < best[0]:
                        best = (L, cand)
            if best[1] != cur:
                for mm in m.alpha:
                    a = m.alpha[mm]
                    idx = (m.perm[mm] == best[1][mm]).nonzero()[0, 0]
                    a.data[idx] = a.data.max() + 1.0
        return best[1] != cur

    def log_row(self, step, k):
        ms = self.measure()
        with torch.no_grad():
            L, parts = self.loss(self.eval_data)
        return {"step": step, "k": k, "loss": round(float(L), 5), **{n: round(v, 5) for n, v in parts.items()}, **ms}

    def train(self, verbose=False):
        """Train until the hard machine (argmax wiring, snapped table) is exact on every example of the last stage,
        or `steps`. With `stages`, each earlier stage trains until exact or `stage_steps`."""
        c, m = self.cfg, self.model
        curve, hit, t0 = [], None, time.time()
        stages = list(c.stages) or [c.k]
        step = 0
        for si, k in enumerate(stages):
            last = si == len(stages) - 1
            if k != self.data.k:
                self.set_data(k)
            end = c.steps if last else min(c.steps, step + (c.stage_steps or c.steps))
            done, stale, seen = False, step, None
            while True:
                if step % c.every == 0 or step >= end:
                    row = self.log_row(step, k)
                    curve.append(row)
                    if verbose:
                        print(row, flush=True)
                    done = row["exact"] == 1.0
                    key = (tuple(sorted(row["prog"].items())), row["exact"])
                    if key != seen:
                        seen, stale = key, step
                if done or step >= end:
                    break
                if last and c.patience and step - stale >= c.patience:
                    break
                if c.search and step and step % c.search.get("every", 100) == 0:
                    self.try_and_keep(step)
                L, _ = self.loss(self.sample_batch())
                self.opt.zero_grad()
                L.backward()
                if step < c.route_freeze:
                    for p in m.route_params():
                        p.grad = None
                if step < c.table_freeze:
                    for p in m.table_params():
                        p.grad = None
                self.opt.step()
                step += 1
            if last and done:
                hit = step
                for _ in range(c.post):
                    L, _ = self.loss(self.sample_batch())
                    self.opt.zero_grad()
                    L.backward()
                    self.opt.step()
                    step += 1
                curve.append(self.log_row(step, k))
            if step >= c.steps:
                break
        final = self.measure()
        cert = self.certify() if self.held else []
        return {"cfg": asdict(self.cfg), "hit": hit, "final": final, "cert": cert,
                "certified": bool(hit is not None and all(x == 1.0 for x in cert)),
                "secs": round(time.time() - t0, 1), "curve": curve}
