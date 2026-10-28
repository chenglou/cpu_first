"""Training the looped transformer of `net.py` from random, at any notch, with the lessons that notch can express.

Lessons (`Run.lessons`, weights):
  network-level, expressible at every notch (they only read and write prompts and answers):
    ans    the answers: cross-entropy of every training question's answer digits
    xmodn  x mod N = x: each question's own register start R0 (v's top w digits, below N) asked as a dividend, and
           N mod N = 0. The answer is known without being told (an order fact of the data: R0 < N)
    step   one step at a time, as a consistency of the network with itself: its answer on (N, v) must agree with its
           answer on (N, B * f(N, v // B) + v_0), where f(N, v // B) is its own (soft) answer on the shorter dividend.
           A remainder is unchanged by reducing a prefix first (Horner). Needs k >= 2
    idem   its answer fed back as a dividend must come back unchanged (a remainder is its own remainder)
  sub-grid laws, expressible while the grid's kinds exist (roles "kinds"): the same network run on a smaller grid
    link   one link alone (the grid with k = 1 and a single link), on the data's own numbers: a link applied to
           x = R0 (below N) must leave x, and applied to N must give 0 (x mod N = x and N mod N = 0 for one
           subtraction). Read from the link's SEL row (through the answer readout); in fmt "digits" also the link's
           SUB row's final borrow, which must be [x < N]
  cell-level, expressible only while the subtraction cell is identifiable (fmt "digits", roles "kinds"): the SUB
  rule is queried directly on constructed ports, bypassing the attention
    pref   ANCHORS.md's top-part order facts: for every top part of (R0, N), R0[j:] - N[j:] borrows exactly when the
           digit strings differ, and N[j:] - R0[j:] never borrows; through the rule's own ripple
    colall ANCHORS.md's one-digit order law on every column of the rule: taking the same digit away keeps the order
           (the learned borrows referee), except that the one that wraps comes out on top
    rowall ANCHORS.md's second law on every row: taking more away from the same digit leaves less, with the same
           exception
    bij    ANCHORS.md's bijectivity: every column and row of the rule's digit table uses each digit once
    cellD  TINYROUTING's cell check, a reference only (it uses a hand-written adder): (a - b) + b = a on every pair of
           P-digit numbers, and the final borrow is [a < b]

A run trains on every question of its size (full batch, or `batch` questions per step), and is measured on the hard
path (argmax attention, snapped states and tables). It stops `post` steps after the training answers first all come
out right, or at `steps`. It is then *certified* if the hard network is also exact on every question of two bigger
sizes, (B, w, k+1) and (B, w+1, k+1) (sampled to `cert_sample` questions where larger).

Training options (RunCfg): `route_freeze` keeps the attention at its start for that many steps (tables first);
`link_stages` grows the grid one link at a time; `stages` one quotient digit at a time; `pmix` replaces the answers'
and the link law's soft blend by whole-option scoring, port by port (each menu option tried as the hard choice, the
other ports soft, the port's weights trained on the mixture).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from bridge import grid as G
from bridge import net as Nn
from bridge.grid import SCRATCH, SEL, SHIFT, SUB, Z


@dataclass
class RunCfg:
    B: int = 3
    w: int = 2
    k: int = 2
    net: dict = field(default_factory=dict)       # net.Cfg fields
    lessons: dict = field(default_factory=lambda: {"ans": 1.0})
    seed: int = 0
    lr: float = 0.05
    steps: int = 3000
    every: int = 25
    post: int = 100
    patience: int = 1500                           # stop if nothing improved for this many steps
    stages: tuple = ()                             # train k = stages[0], then stages[1], ... (each until exact)
    link_stages: tuple = ()                        # grow the grid one link at a time: links = link_stages[0], ...,
                                                   # each on the questions whose quotient digits all fit (<= links)
    stage_steps: int = 0
    cert: bool = True
    cert_soft: bool = False                        # also certify the soft path (softmax attention, unsnapped states)
    cert_sample: int = 20000
    dtype: str = "float64"
    clip: float = 0.0
    batch: int = 0                                 # questions per step for the network-level lessons (0 = all)
    pmix: bool = False                             # menu heads: whole-option scoring, port by port (the answers and
                                                   # the link law as a mixture over each port's options, each tried
                                                   # hard with the other ports soft), instead of the soft blend
    route_freeze: int = 0                          # steps with the attention frozen at its start (tables first)
    lesson_after: dict = field(default_factory=dict)   # lesson -> the step it joins (weight 0 before)
    init: str = ""                                 # "compiled": start at the compiled machine (compile.py) with every
    init_beta: float = 4.0                         # scale (attention scores, output logits, table logits) set to
    init_noise: float = 0.0                        # init_beta, then Gaussian noise of this std on every learned tensor
    widen: int = 1                                 # Net2Net: the compiled MLP's units duplicated this many times
    tag: str = ""


def nll_digits(p, y):
    """Mean negative log-likelihood per digit of digit distributions p [X, n, B] at targets y [X, n]."""
    return -p.clamp_min(1e-12).log().gather(-1, y[..., None]).squeeze(-1).mean()


def agree(p, q):
    """Mean over digits of -log P(two digit distributions [X, n, B] agree)."""
    return -(p * q).sum(-1).clamp_min(1e-12).log().mean()


class Run:
    def __init__(self, cfg: RunCfg):
        self.cfg = c = cfg
        torch.manual_seed(c.seed)
        self.dtype = getattr(torch, c.dtype)
        netc = dict(c.net)
        if not netc.get("tied", True) and not netc.get("nblocks"):
            netc["nblocks"] = G.Layout(c.B, c.w, c.k).depth + netc.get("extra_ticks", 0)
        self.ncfg = Nn.Cfg(**netc)
        self.net = Nn.Net(self.ncfg, c.B, seed=c.seed, dtype=self.dtype)
        if c.init == "compiled":
            self.compile_init()
        params = [p for p in self.net.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=c.lr)
        self.nparams = sum(p.numel() for p in params)
        self.lay = {}
        self.data = {}
        self.t0 = time.time()
        self.log = []
        self.rng = np.random.default_rng(1000 + c.seed)
        self.step = 0

    def compile_init(self):
        """Set the network to the compiled machine, softened to `init_beta` and perturbed by `init_noise`."""
        from bridge import compile as Cm
        c, n = self.cfg, self.net
        cfg = n.cfg
        g = torch.Generator().manual_seed(7000 + c.seed)
        beta, sig = c.init_beta, c.init_noise
        if cfg.attn == "free":
            Cm.set_free(n, beta=beta, noise=sig, gen=g)
        elif cfg.attn == "menu":
            Cm.set_menu(n, beta=beta, noise=sig, gen=g)
        elif cfg.attn != "fixed":
            raise ValueError(f"no compiled point for attention {cfg.attn}")
        if cfg.fmt == "symbols":
            Cm.set_symbols(n, beta=beta, beta_o=beta, noise=sig, gen=g, widen=c.widen)
        elif cfg.fmt != "digits":
            raise ValueError(f"no compiled point for format {cfg.fmt}")
        elif cfg.rule in ("mlp", "mlptab") and (cfg.rule == "mlp" or cfg.learn):
            Cm.set_mlp(n, beta_o=beta, widen=c.widen, noise=sig, gen=g)
        elif cfg.rule == "table":
            blk = n.blocks[0]
            T = G.truth_tables(c.B, self.dtype)
            with torch.no_grad():
                for i, kd in enumerate(SCRATCH):
                    name = G.KINDS[kd]
                    if name in blk.Ld:
                        blk.Ld[name].copy_(beta * T[i, ..., :c.B])
                        blk.Lb[name].copy_(beta * T[i, ..., c.B:])
                        Cm._noise(blk.Ld[name], sig, g)
                        Cm._noise(blk.Lb[name], sig, g)

    # ----- sizes -----
    def layout(self, B, w, k, links=None):
        key = (B, w, k, links)
        if key not in self.lay:
            self.lay[key] = G.Layout(B, w, k, links)
        return self.lay[key]

    def dataset(self, B, w, k, sample=0, links=None):
        key = (B, w, k, sample, links)
        if key not in self.data:
            d = G.Data(B, w, k, sample=sample, seed=1)
            if links is not None and links < B - 1:                              # quotient digits that fit
                q = G.digits(d.v // d.N, B, k)
                d = d.subset(np.where((q <= links).all(1))[0])
            self.data[key] = d
        return self.data[key]

    def hot(self, x):
        return G.onehot(x, self.cfg.B, self.dtype)

    def f(self, lay, Np, vp, hard=False):
        """The network's answer digit distributions [X, w, B] on digit distributions of N and v."""
        return self.net.answer(lay, Np, vp, hard)

    def vdist(self, lay, v):
        return self.hot(torch.from_numpy(G.digits(v, lay.B, lay.w + lay.k)))

    # ----- lessons -----
    def lessons(self, k, links=None):
        c = self.cfg
        B, w = c.B, c.w
        lay = self.layout(B, w, k, links)
        data = self.dataset(B, w, k, links=links)
        if c.batch and data.X > c.batch:
            data = data.subset(self.rng.choice(data.X, c.batch, replace=False))
        W = c.lessons
        terms = {}
        Nh = self.hot(data.Nd)
        if c.pmix:
            terms["pmix"] = self.pmix(lay, data)
        elif W.get("ans", 0):
            terms["ans"] = nll_digits(self.f(lay, Nh, self.hot(data.vd)), data.yd)
        if W.get("xmodn", 0):
            pairs = np.unique(np.stack([data.N, data.R0], 1), axis=0)
            Nn_ = np.concatenate([pairs[:, 0], np.unique(data.N)])
            vv = np.concatenate([pairs[:, 1], np.unique(data.N)])
            yy = vv % Nn_
            Nd = torch.from_numpy(G.digits(Nn_, B, w))
            p = self.f(lay, self.hot(Nd), self.vdist(lay, vv))
            terms["xmodn"] = nll_digits(p, torch.from_numpy(G.digits(yy, B, w)))
        if W.get("step", 0) and k >= 2:
            p_full = self.f(lay, Nh, self.hot(data.vd))
            r1 = self.f(lay, Nh, self.vdist(lay, data.v // B))                      # f(N, v // B), soft
            X = data.X
            v2 = torch.zeros(X, w + k, B, dtype=self.dtype)
            v2[:, 0] = self.hot(data.vd[:, 0])
            v2[:, 1:w + 1] = r1
            v2[:, w + 1:, 0] = 1
            p2 = self.f(lay, Nh, v2)
            terms["step"] = agree(p_full, p2)
        if W.get("idem", 0):
            p = self.f(lay, Nh, self.hot(data.vd))
            v2 = torch.zeros(data.X, w + k, B, dtype=self.dtype)
            v2[:, :w] = p
            v2[:, w:, 0] = 1
            terms["idem"] = agree(p, self.f(lay, Nh, v2))
        if W.get("link", 0):
            terms["link"] = self.link(data)
        if W.get("pref", 0):
            terms["pref"] = self.pref(data)
        if W.get("colall", 0):
            terms["colall"] = self.colall()
        if W.get("rowall", 0):
            terms["rowall"] = self.rowall()
        if W.get("bij", 0):
            terms["bij"] = self.bij()
        if W.get("cellD", 0):
            terms["cellD"] = self.cellD()
        loss = sum(W.get(n, 1.0) * t for n, t in terms.items() if self.step >= c.lesson_after.get(n, 0))
        return loss, {n: float(t.detach()) for n, t in terms.items()}

    def pmix(self, lay, data):
        """For every port a kind uses: -log of the mixture over its options (each tried as the hard choice, the other
        ports soft) of each question's answer likelihood, plus the same for the link law's rows."""
        net = self.net
        blk = net.blocks[0]
        mask = blk.menu_mask
        Nh, vh = self.hot(data.Nd), self.hot(data.vd)
        c = self.cfg
        lay1 = self.layout(c.B, c.w, 1, links=1)
        pairs = np.unique(np.stack([data.N, data.R0], 1), axis=0)
        Ns = np.concatenate([pairs[:, 0], np.unique(data.N)])
        xs = np.concatenate([pairs[:, 1], np.unique(data.N)])
        N1 = self.hot(torch.from_numpy(G.digits(Ns, c.B, c.w)))
        v1 = self.vdist(lay1, xs)
        P = c.w + 1
        last = lay1.Rn - 1
        cells = np.array([lay1.cell(last, j) for j in range(P)])
        tgt1 = torch.from_numpy(G.digits(xs % Ns, c.B, P))
        lg = lambda p, y: p.clamp_min(1e-12).log().gather(-1, y[..., None]).squeeze(-1).sum(-1)
        tot = 0.0
        for (h, kd), opt in Nn.TRUE_MENU.items():
            if opt == "zero":
                continue                                                           # a port its kind does not use
            opts = torch.where(torch.isfinite(mask[h, kd]))[0].tolist()
            logpi = (blk.menu[h, kd] + mask[h, kd]).log_softmax(-1)[opts]
            lls, ll1s = [], []
            for o in opts:
                net.force = ((h, kd), o)
                lls.append(lg(self.f(lay, Nh, vh), data.yd))
                S1 = net(lay1, net.inputs(lay1, N1, v1))
                ll1s.append(lg(S1[:, cells, :c.B], tgt1))
            net.force = None
            for ll in (torch.stack(lls), torch.stack(ll1s)):
                tot = tot - torch.logsumexp(logpi[:, None] + ll, 0).mean() / c.w
        return tot

    def link(self, data):
        """One link alone on x = R0 (below N) and x = N: its register must become x mod N (x, or 0)."""
        c = self.cfg
        B, w = c.B, c.w
        P = w + 1
        lay1 = self.layout(B, w, 1, links=1)
        pairs = np.unique(np.stack([data.N, data.R0], 1), axis=0)
        Ns = np.concatenate([pairs[:, 0], np.unique(data.N)])
        xs = np.concatenate([pairs[:, 1], np.unique(data.N)])
        Nh = self.hot(torch.from_numpy(G.digits(Ns, B, w)))
        S = self.net(lay1, self.net.inputs(lay1, Nh, self.vdist(lay1, xs)))
        last = lay1.Rn - 1
        cells = np.array([lay1.cell(last, j) for j in range(P)])
        target = torch.from_numpy(G.digits(xs % Ns, B, P))
        if self.ncfg.fmt == "digits":
            reg = S[:, cells, :B]
        else:
            lay_cells = lay1.answer
            lay1.answer = cells                                                    # the readout, on that row
            reg = self.net.readout(lay1, S)
            lay1.answer = lay_cells
        loss = nll_digits(reg, target)
        if self.ncfg.fmt == "digits":
            top = S[:, lay1.cell(last - 1, P - 1), B:]                             # the SUB row's final borrow
            lt = torch.from_numpy((xs < Ns).astype(np.int64))
            loss = loss + -top.clamp_min(1e-12).log().gather(1, lt[:, None]).mean()
        return loss

    def ripple(self, Ad, Md):
        """A - M through the network's own SUB rule, from place 0 up: digit distributions [X, n, B] in; digits
        [X, n, B] and every place's borrow-out [X, n, 2] out."""
        X, n, B = Ad.shape
        cin = torch.zeros(X, 2, dtype=self.dtype)
        cin[:, 0] = 1
        outs, cs = [], []
        for j in range(n):
            d, cin = self.net.sub_rule(Ad[:, j], Md[:, j], cin)
            outs.append(d)
            cs.append(cin)
        return torch.stack(outs, 1), torch.stack(cs, 1)

    def pref(self, data):
        B, P = self.cfg.B, self.cfg.w + 1
        pairs = np.unique(np.stack([data.N, data.R0], 1), axis=0)
        Nd = torch.from_numpy(G.digits(pairs[:, 0], B, P))
        Rd = torch.from_numpy(G.digits(pairs[:, 1], B, P))
        tot = 0.0
        for j in range(P):
            differ = (Rd[:, j:] != Nd[:, j:]).any(1).to(self.dtype)
            p1 = self.ripple(self.hot(Rd[:, j:]), self.hot(Nd[:, j:]))[1][:, -1, 1]
            p2 = self.ripple(self.hot(Nd[:, j:]), self.hot(Rd[:, j:]))[1][:, -1, 1]
            lg = lambda x: x.clamp_min(1e-12).log()
            tot = tot - (differ * lg(p1) + (1 - differ) * lg(1 - p1)).mean() - lg(1 - p2).mean()
        return tot / P

    def rule_table(self, hard=False):
        """The SUB rule on every one-hot (a, m, c): digit [B, B, 2, B] and borrow [B, B, 2, 2] distributions."""
        B = self.cfg.B
        a, m, cc = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
        d, b = self.net.sub_rule(self.hot(a.flatten()), self.hot(m.flatten()), G.onehot(cc.flatten(), 2, self.dtype),
                                 hard)
        return d.view(B, B, 2, B), b.view(B, B, 2, 2)

    def colall(self):
        B = self.cfg.B
        Td, Tb = self.rule_table()
        with torch.no_grad():
            bor = Tb.argmax(-1)                                                     # [a, m, c] the learned borrows
        tot, n = 0.0, 0
        tri = torch.triu(torch.ones(B, B, dtype=self.dtype), 1)
        for m in range(B):
            for c in range(2):
                for r in range(B):
                    for r2 in range(r + 1, B):
                        w0, w1 = int(bor[r, m, c]), int(bor[r2, m, c])
                        lt = int(bor[r, r2, 0])                                     # the comparator: [r < r2]
                        first = (w0 < w1) or (w0 == w1 and lt == 1)                 # [o(r) < o(r2)]
                        lo, hi = (Td[r, m, c], Td[r2, m, c]) if first else (Td[r2, m, c], Td[r, m, c])
                        tot = tot - (lo[:, None] * hi[None, :] * tri).sum().clamp_min(1e-12).log()
                        n += 1
        return tot / n

    def rowall(self):
        """ANCHORS.md's second law, one digit at a time: taking more away from the same digit leaves less, except
        that the one that wraps comes out on top. For every row (a, c) and digits m < m2 (the learned comparator
        referees): o(a, m) > o(a, m2) unless exactly one of them borrows."""
        B = self.cfg.B
        Td, Tb = self.rule_table()
        with torch.no_grad():
            bor = Tb.argmax(-1)
        tot, n = 0.0, 0
        tri = torch.triu(torch.ones(B, B, dtype=self.dtype), 1)
        for a in range(B):
            for c in range(2):
                for m in range(B):
                    for m2 in range(m + 1, B):
                        w0, w1 = int(bor[a, m, c]), int(bor[a, m2, c])
                        lt = int(bor[m, m2, 0])                                     # [m < m2]
                        if w0 != w1:
                            first_lower = w0 == 0                                   # the one that wraps is on top
                        else:
                            first_lower = lt == 0                                   # more taken away leaves less
                        lo, hi = (Td[a, m, c], Td[a, m2, c]) if first_lower else (Td[a, m2, c], Td[a, m, c])
                        tot = tot - (lo[:, None] * hi[None, :] * tri).sum().clamp_min(1e-12).log()
                        n += 1
        return tot / n

    def bij(self):
        """ANCHORS.md's bijectivity: every column (m, c) and every row (a, c) of the rule's digit table uses each digit
        once, as a uniform marginal over the column's (row's) entries (the H1 file's form)."""
        B = self.cfg.B
        Td, _ = self.rule_table()
        lg = lambda x: x.clamp_min(1e-12).log()
        return -(lg(B * Td.mean(0)).mean() + lg(B * Td.mean(1)).mean())

    def cellD(self):
        """(a - b) + b = a with a hand-written adder, on every pair of P-digit numbers, and borrow = [a < b]."""
        B, P = self.cfg.B, self.cfg.w + 1
        vals = np.arange(B ** P)
        a, b = np.meshgrid(vals, vals, indexing="ij")
        a, b = a.flatten(), b.flatten()
        Ad = torch.from_numpy(G.digits(a, B, P))
        Bd = torch.from_numpy(G.digits(b, B, P))
        E, cs = self.ripple(self.hot(Ad), self.hot(Bd))
        # designed adder on the distributions E plus the exact digits of b: back = E + b, digit by digit
        X = len(a)
        carry = torch.zeros(X, 2, dtype=self.dtype)
        carry[:, 0] = 1
        tot = 0.0
        for j in range(P):
            bj = Bd[:, j]
            # P(back digit = d) = sum_e E[e] * sum_c carry[c] [ (e + b + c) % B == d ]
            pd = torch.zeros(X, B, dtype=self.dtype)
            pc = torch.zeros(X, 2, dtype=self.dtype)
            for e in range(B):
                for cc in range(2):
                    s = e + bj + cc
                    w_ = E[:, j, e] * carry[:, cc]
                    pd[torch.arange(X), s % B] += w_
                    pc[torch.arange(X), (s >= B).long()] += w_
            carry = pc
            tot = tot - pd.gather(1, Ad[:, j:j + 1]).clamp_min(1e-12).log().mean()
        lt = torch.from_numpy((a < b).astype(np.int64))
        tot = tot - cs[:, -1].gather(1, lt[:, None]).clamp_min(1e-12).log().mean()
        return tot / (P + 1)

    # ----- measures -----
    @torch.no_grad()
    def exact(self, B, w, k, sample=0, hard=True, chunk=4096, links=None):
        lay = self.layout(B, w, k, links)
        data = self.dataset(B, w, k, sample, links)
        n = 0
        for i in range(0, data.X, chunk):
            d = data if data.X <= chunk else data.subset(np.arange(i, min(i + chunk, data.X)))
            p = self.f(lay, self.hot(d.Nd), self.hot(d.vd), hard)
            n += int((p.argmax(-1) == d.yd).all(-1).sum())
        return n / data.X

    @torch.no_grad()
    def inspect(self):
        """Where the parts are identifiable: the SUB rule against the truth, and each head's argmax against the
        compiled pattern (at the training size)."""
        c, nc = self.cfg, self.ncfg
        out = {}
        if nc.fmt == "digits" and nc.roles == "kinds":
            Td, Tb = self.rule_table(hard=True)
            T = G.truth_tables(c.B, self.dtype)[1]
            out["sub_digits_right"] = int((Td.argmax(-1) == T[..., :c.B].argmax(-1)).sum())
            out["sub_borrows_right"] = int((Tb.argmax(-1) == T[..., c.B:].argmax(-1)).sum())
            out["sub_entries"] = 2 * c.B * c.B
            B = c.B
            a, m, cc = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
            a, m, cc = a.flatten(), m.flatten(), cc.flatten()
            Tt = G.truth_tables(B, self.dtype)
            for i, kd in enumerate(SCRATCH):
                if kd == SUB:
                    continue
                d, b = self.net.kind_rule(kd, self.hot(a), self.hot(m), G.onehot(cc, 2, self.dtype), hard=True)
                ok = d.argmax(-1) == Tt[i, a, m, cc, :B].argmax(-1)
                if kd == SHIFT:                                                    # only (a, 0, 0) is ever used
                    ok = ok[(m == 0) & (cc == 0)]
                out[f"{G.KINDS[kd].lower()}_right"] = f"{int(ok.sum())}/{len(ok)}"
        if nc.attn in ("menu", "free") and nc.roles == "kinds":
            lay = self.layout(c.B, c.w, c.k)
            ix = self.net.index(lay)
            A = self.net.port_attention(lay, ix, self.net.blocks[0], None, True)
            right = {}
            for (h, kd), opt in Nn.TRUE_MENU.items():
                if opt == "zero":
                    continue
                cells = np.where(lay.kind == kd)[0]
                src = A[h, cells].argmax(-1).numpy()
                comp = lay.comp[h, cells]
                ok = (src == comp) | ((lay.kind[src] == Z) & (lay.kind[comp] == Z))
                right[f"{G.HEADS[h]}.{G.KINDS[kd]}"] = float(ok.mean())
            out["heads_right"] = right
        return out

    # ----- training -----
    def train(self):
        c = self.cfg
        if c.link_stages:
            stages = [(c.k, L if L < c.B - 1 else None) for L in c.link_stages]
        else:
            stages = [(kk, None) for kk in (list(c.stages) or [c.k])]
        step = 0
        hit = None
        best, best_step = -1.0, 0
        rec = {"stages": []}
        if c.init:                                                             # where training starts
            rec["start_hard"] = self.exact(c.B, c.w, c.k)
            rec["start_soft"] = self.exact(c.B, c.w, c.k, hard=False)
            rec["start_inspect"] = self.inspect()
        for si, (kk, links) in enumerate(stages):
            limit = c.stage_steps or c.steps
            s0 = step
            stage_hit = None
            best, best_step = -1.0, step
            while step < c.steps and step - s0 < limit:
                self.opt.zero_grad()
                loss, terms = self.lessons(kk, links)
                loss.backward()
                if step < c.route_freeze:
                    for name, prm in self.net.named_parameters():
                        if any(t in name for t in ("menu", ".U", ".Rr", ".Rc", ".Tp", "Wq", "Wk", "Prow", "Pcol")):
                            prm.grad = None
                if c.clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), c.clip)
                self.opt.step()
                step += 1
                self.step = step
                if step % c.every == 0:
                    ex = self.exact(c.B, c.w, kk, links=links)
                    row = {"step": step, "k": kk, "links": links, "loss": float(loss.detach()), "exact": ex, **terms}
                    if c.net.get("fmt", "digits") == "digits" and c.net.get("roles", "kinds") == "kinds":
                        row.update({kx: v for kx, v in self.inspect().items() if kx != "heads_right"})
                    self.log.append(row)
                    if ex > best + 1e-9:
                        best, best_step = ex, step
                    if ex == 1.0 and stage_hit is None:
                        stage_hit = step
                    if stage_hit is not None and step - stage_hit >= (c.post if si == len(stages) - 1 else 0):
                        break
                    if step - best_step >= c.patience:
                        break
            rec["stages"].append({"k": kk, "links": links, "hit": stage_hit, "end": step})
            if stage_hit is None:
                break
        hit = rec["stages"][-1]["hit"] if len(rec["stages"]) == len(stages) else None
        rec.update(steps=step, hit=hit, seconds=time.time() - self.t0, params=self.nparams)
        rec["train_hard"] = self.exact(c.B, c.w, c.k)
        rec["train_soft"] = self.exact(c.B, c.w, c.k, hard=False)
        rec.update(self.inspect())
        if c.cert:
            bigger = [(c.B, c.w, c.k + 1), (c.B, c.w + 1, c.k + 1)]
            rec["bigger"] = {f"{b},{w_},{k_}": self.exact(b, w_, k_, sample=c.cert_sample) for b, w_, k_ in bigger}
            rec["certified"] = bool(rec["train_hard"] == 1.0 and all(v == 1.0 for v in rec["bigger"].values()))
            if c.cert_soft:                                                   # the network as trained, unsnapped
                rec["bigger_soft"] = {f"{b},{w_},{k_}": self.exact(b, w_, k_, sample=c.cert_sample, hard=False)
                                      for b, w_, k_ in bigger}
                rec["certified_soft"] = bool(rec["train_soft"] == 1.0 and
                                             all(v == 1.0 for v in rec["bigger_soft"].values()))
        rec["log"] = self.log
        return rec


def run(cfg: RunCfg) -> dict:
    r = Run(cfg)
    out = r.train()
    out["cfg"] = asdict(cfg)
    return out


def summary(rec):
    s = rec["cfg"]
    bits = [f"{s['tag'] or ''} seed {s['seed']}", f"steps {rec['steps']}", f"hit {rec['hit']}",
            f"train {rec['train_hard']:.3f}/{rec['train_soft']:.3f}"]
    if "bigger" in rec:
        bits.append("bigger " + " ".join(f"{v:.3f}" for v in rec["bigger"].values()))
        bits.append("CERT" if rec["certified"] else "no")
    if "sub_digits_right" in rec:
        bits.append(f"sub {rec['sub_digits_right']}/{rec['sub_borrows_right']} of {rec['sub_entries']}")
    if "heads_right" in rec:
        bits.append("heads " + " ".join(f"{k_}:{v:.2f}" for k_, v in rec["heads_right"].items()))
    bits.append(f"{rec['seconds']:.0f}s")
    return " | ".join(bits)
