"""Training the NCA of `model.py` from random, with the lessons its format can express, and certifying it.

Lessons (`RunCfg.lessons`, weights):
  grid-level, expressible for any format (they only write prompts and read answers):
    ans    the answers: cross-entropy of every training question's answer digits
    xmodn  x mod N = x: each question's own register start R0 (below N) asked as a dividend, and N mod N = 0
    step   one step at a time, as a consistency of the NCA with itself: its answer on (N, v) must agree with its
           answer on (N, B * f(N, v // B) + v_0), f(N, v // B) its own soft answer (k >= 2)
    link   one link alone (the grid with k = 1 and one link, a sub-grid the same rule runs on): on x = R0 the
           register must come out x, on x = N it must come out 0; read from every register place of the last row.
           With the compiled format also the SUB row's final borrow, which must be [x < N]
    indep  the answer must not depend on the scratch cells' start: two runs from random scratch states must agree
  cell-level, only for the compiled format (the borrow channel b is identifiable). The rule is queried directly on a
  constructed neighbourhood: a SUB cell (an interior place of a step's first link: SHIFT kinds above, SUB beside,
  SEL below) whose up neighbour holds d = a, n = m and whose left neighbour holds b = c, every other neighbour
  channel blank (`ctx_mode` "blank") or random, fresh each step ("random"). That reads off the rule's subtraction
  table (a, m, c) -> (d, b):
    pref   ANCHORS.md's top-part order facts through that table's own ripple: R0[j:] - N[j:] borrows exactly when
           the digit strings differ, N[j:] - R0[j:] never borrows
    colall, rowall  ANCHORS.md's two one-digit order laws, the learned borrows refereeing
    bij    every row and column of the table uses each digit once
  labels, not lessons of the project (references):
    sheet  the worksheet's visible digits: d at every scratch cell after the last tick is the machine's
    trace  the compiled rule's run: every tick of the learned rule gives the compiled rule's next state in every
           channel of every scratch cell (`trace_chans` for some channels only, `trace_ticks` to sample ticks)
  the square (`task` "sq"): ans and trace on the square's grid, prod (the product x^2 on the last MUL row, a label),
  and the division lessons above on the division grid of size (B, w, div_k), run by the same rule (dans, dtrace
  are the division's ans and trace)
A run trains on every question of its size (full batch, or `batch` questions per step) with a tick count drawn from
depth .. depth * (1 + slack) each step. It is measured on the hard path (argmax states, `depth` ticks). It stops
`post` steps after the training answers first all come out right, or at `steps`, or after `patience` steps without
progress. It is *certified* if the hard NCA is also exact on every question of two bigger sizes, (B, w, k + 1) and
(B, w + 1, k + 1) (up to `cert_sample` sampled where larger): the bridge's test. Extrapolation sizes and extra ticks
are measured too.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from nca import grid as G
from nca import model as M
from nca.grid import MUL, OUT, SEL, SHIFT, SPLIT, SUB


@dataclass
class RunCfg:
    task: str = "div"                                  # div: v mod N; sq: x^2 mod N (the division lessons then run on
    div_k: int = 1                                     # the division grid of size (B, w, div_k), the same rule)
    B: int = 3
    w: int = 2
    k: int = 1
    net: dict = field(default_factory=dict)            # model.Cfg fields
    fmt: str = "compiled"                              # compiled: d, n, b, m, g; "free:2,3,2": d, n + hidden channels
    lessons: dict = field(default_factory=lambda: {"ans": 1.0})
    seed: int = 0
    lr: float = 0.01
    steps: int = 3000
    every: int = 25
    post: int = 100
    patience: int = 1500
    slack: float = 0.5
    stages: tuple = ()                                 # train k = stages[0], then stages[1], ... (each until exact)
    stage_steps: int = 0
    lesson_after: dict = field(default_factory=dict)   # lesson -> the step it joins
    batch: int = 0
    ctx: int = 4                                       # contexts per cell-lesson query
    ctx_mode: str = "random"                           # blank: other neighbour channels blank; random: random
    scratch: str = "blank"                             # the scratch cells' tick-0 state in training: blank | random
    dtype: str = "float32"
    cert: bool = True
    cert_sample: int = 20000
    extra_sizes: tuple = ((0, 2), (1, 2), (2, 2))      # (dw, dk) beyond training, measured (not certification)
    tick_mults: tuple = (2, 4)                         # extra-tick robustness on the training size and (w+1, k+1)
    init: str = ""                                     # "compiled": start at the compiled rule
    init_beta: float = 4.0
    init_noise: float = 0.0
    trace_ticks: int = 0                               # trace: ticks of the teacher's run sampled per step (0: all)
    trace_chans: str = ""                              # trace: only these channels supervised ("" every channel), e.g.
                                                       # "d" or "d,b"; the run is the teacher's, the others unlabelled
    save_weights: bool = False                         # keep the trained rule's weights in the record (base64 float32)
    freeze_g: int = 0                                  # hyper: g (the wiring) frozen for this many steps
    l1: float = 0.0                                    # L1 on the rule's input weights (mean |W1|, and |Wx|): a rule
                                                       # that reads few inputs meets fewer new situations
    tag: str = ""


def make_fmt(spec: str, B: int, task: str = "div") -> G.Fmt:
    if spec == "compiled":
        return G.compiled_fmt(B, task)
    if spec == "dnb":
        return G.compiled_fmt(B, task, minimal=True)
    if spec.startswith("free:"):
        sizes = [int(s) for s in spec[5:].split(",") if s]
        return G.Fmt([("d", B), ("n", B)] + [(f"h{i}", s) for i, s in enumerate(sizes)])
    raise ValueError(spec)


def nll(p, y):
    """Mean negative log-likelihood of distributions p [..., V] at integer targets y [...]."""
    return -p.clamp_min(1e-12).log().gather(-1, y[..., None]).squeeze(-1).mean()


def agree(p, q):
    return -(p * q).sum(-1).clamp_min(1e-12).log().mean()


class Run:
    def __init__(self, cfg: RunCfg):
        self.cfg = c = cfg
        torch.manual_seed(c.seed)
        self.dtype = getattr(torch, c.dtype)
        self.fmt = make_fmt(c.fmt, c.B, c.task)
        self.ncfg = M.Cfg(**dict(c.net, kindset="sq" if c.task == "sq" else "div"))
        self.hyper = self.ncfg.rule == "hyper"
        if self.hyper:
            from nca import hyper as Hy
            self.net = Hy.HyperNCA(self.fmt, self.ncfg, seed=c.seed, dtype=self.dtype)
        else:
            self.net = M.NCA(self.fmt, self.ncfg, seed=c.seed, dtype=self.dtype)
        if c.init == "compiled":
            g = torch.Generator().manual_seed(7000 + c.seed)
            if self.hyper:
                Hy.compile_ports(self.net, c.B, beta=c.init_beta, beta_o=c.init_beta)
                with torch.no_grad():
                    for p in self.net.parameters():
                        p.add_(c.init_noise * torch.randn(p.shape, generator=g, dtype=p.dtype))
            else:
                from nca import compile as Cm
                Cm.set_compiled(self.net, c.B, beta=c.init_beta, noise=c.init_noise, gen=g, task=c.task)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=c.lr)
        self.nparams = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.lay, self.data = {}, {}
        self.rng = np.random.default_rng(1000 + c.seed)
        self.gen = torch.Generator().manual_seed(2000 + c.seed)
        self.t0 = time.time()
        self.log = []
        self.step = 0
        self.compiled = c.fmt in ("compiled", "dnb")
        self.reach = "ports" if c.fmt == "dnb" else (self.ncfg.extra or "3x3")

    # ----- sizes -----
    def layout(self, B, w, k, links=None, task="div"):
        key = (B, w, k, links, task)
        if key not in self.lay:
            self.lay[key] = G.Layout(B, w, k, links, task=task)
        return self.lay[key]

    def dataset(self, B, w, k, sample=0, task="div"):
        key = (B, w, k, sample, task)
        if key not in self.data:
            self.data[key] = (G.SqData(B, w, sample=sample, seed=1) if task == "sq" else
                              G.Data(B, w, k, sample=sample, seed=1))
        return self.data[key]

    def ticks(self, lay, train=True):
        if not train or not self.cfg.slack:
            return lay.depth
        return lay.depth + int(self.rng.integers(0, int(self.cfg.slack * lay.depth) + 1))

    def forward(self, lay, Nd, vd, mode=None, ticks=None, train=True, scratch=None):
        mode = mode or self.ncfg.state
        scratch = scratch or (self.cfg.scratch if train else "blank")
        S0 = self.net.prompt_state(lay, Nd, vd, scratch, self.gen)
        return self.net.run(lay, S0, ticks or self.ticks(lay, train), mode)

    def answer(self, lay, P, places=None):
        """Digit distributions [X, n, B] of the last row at `places` (default: the answer's places)."""
        r = lay.R - 1
        places = lay.answer[1] if places is None else places
        return self.net.channel(P, "d")[:, :, r][:, :, places].permute(0, 2, 1)

    def hot(self, x, n=None):
        return F.one_hot(torch.as_tensor(x), n or self.cfg.B).to(self.dtype)

    # ----- lessons -----
    def lessons(self, k):
        c = self.cfg
        W = {n: v for n, v in c.lessons.items() if v and self.step >= c.lesson_after.get(n, 0)}
        if c.task == "div":
            terms = self.div_terms(k, W)
        else:
            Wd = {("ans" if n == "dans" else "trace" if n == "dtrace" else n): v for n, v in W.items()
                  if n not in ("ans", "trace", "prod")}
            terms = self.div_terms(c.div_k, Wd) if Wd else {}
            terms = {("dans" if n == "ans" else "dtrace" if n == "trace" else n): t for n, t in terms.items()}
            terms.update(self.sq_terms(W))
        Wall = dict(W)
        if c.l1 and self.ncfg.rule == "mlp":
            terms["l1"] = (self.net.W1 * self.net.mask).abs().mean() + (self.net.Wx.abs().mean() if self.net.nextra else 0)
            Wall["l1"] = c.l1
        loss = sum(Wall[n] * t for n, t in terms.items())
        return loss, {n: float(t.detach()) for n, t in terms.items()}

    def sq_terms(self, W):
        """The square's own lessons on its grid: its answers, the product x^2 read from the last MUL row (a label of
        the multiplication, not a law), and the compiled square's trace (the upper bound)."""
        c = self.cfg
        lay = self.layout(c.B, c.w, c.w, task="sq")
        data = self.dataset(c.B, c.w, c.w, task="sq")
        terms = {}
        if "ans" in W or "prod" in W:
            _, P = self.forward(lay, data.Nd, data.vd)
            if "ans" in W:
                terms["ans"] = nll(self.answer(lay, P), torch.from_numpy(data.yd))
            if "prod" in W:
                r = 2 + c.w - 1                                                       # the last MUL row
                pd = self.net.channel(P, "d")[:, :, r].permute(0, 2, 1)
                terms["prod"] = nll(pd, torch.from_numpy(G.digits(data.v, c.B, lay.C)))
        if "trace" in W and self.compiled:
            terms["trace"] = self.trace(lay, data)
        return terms

    def div_terms(self, k, W):
        c = self.cfg
        B, w = c.B, c.w
        lay = self.layout(B, w, k)
        data = self.dataset(B, w, k)
        if c.batch and data.X > c.batch:
            data = data.subset(self.rng.choice(data.X, c.batch, replace=False))
        terms = {}
        if "ans" in W or "step" in W:
            _, P = self.forward(lay, data.Nd, data.vd)
            pa = self.answer(lay, P)
            if "ans" in W:
                terms["ans"] = nll(pa, torch.from_numpy(data.yd))
        if "xmodn" in W:
            pairs = np.unique(np.stack([data.N, data.R0], 1), axis=0)
            Ns = np.concatenate([pairs[:, 0], np.unique(data.N)])
            xs = np.concatenate([pairs[:, 1], np.unique(data.N)])
            _, P = self.forward(lay, G.digits(Ns, B, w), G.digits(xs, B, w + k))
            terms["xmodn"] = nll(self.answer(lay, P), torch.from_numpy(G.digits(xs % Ns, B, w)))
        if "step" in W and k >= 2:
            _, P1 = self.forward(lay, data.Nd, G.digits(data.v // B, B, w + k))
            r1 = self.answer(lay, P1)
            v2 = torch.zeros(data.X, w + k, B, dtype=self.dtype)
            v2[:, 0] = self.hot(data.vd[:, 0])
            v2[:, 1:w + 1] = r1
            v2[:, w + 1:, 0] = 1
            _, P2 = self.forward(lay, self.hot(data.Nd), v2)
            terms["step"] = agree(pa, self.answer(lay, P2))
        if "link" in W:
            terms["link"] = self.link(data)
        if "trace" in W and self.compiled:
            terms["trace"] = self.trace(lay, data)
        if "fixpoint" in W and self.compiled:
            terms["fixpoint"] = self.fixpoint(lay, data)
        if "sheet" in W:                                        # the worksheet's visible digits as labels
            _, P = self.forward(lay, data.Nd, data.vd)
            ref = G.machine_reference(lay, data)
            sc = torch.from_numpy(~lay.prompt)
            lp = self.net.channel(P, "d").clamp_min(1e-12).log()
            terms["sheet"] = -lp.gather(1, torch.from_numpy(ref["d"])[:, None]).squeeze(1)[:, sc].mean()
        if "indep" in W:
            _, Pa = self.forward(lay, data.Nd, data.vd, scratch="random")
            _, Pb = self.forward(lay, data.Nd, data.vd, scratch="random")
            terms["indep"] = agree(self.answer(lay, Pa), self.answer(lay, Pb))
        if self.compiled and any(n in W for n in ("pref", "colall", "rowall", "bij")):
            Td, Tb = self.sub_table()
            if "pref" in W:
                terms["pref"] = self.pref(data, Td, Tb)
            if "colall" in W:
                terms["colall"] = self.colall(Td, Tb)
            if "rowall" in W:
                terms["rowall"] = self.rowall(Td, Tb)
            if "bij" in W:
                terms["bij"] = self.bij(Td)
        return terms

    def link(self, data):
        """One link alone on x = R0 (below N) and x = N: its register must become x mod N (x, or 0)."""
        c = self.cfg
        B, w = c.B, c.w
        lay1 = self.layout(B, w, 1, links=1)
        pairs = np.unique(np.stack([data.N, data.R0], 1), axis=0)
        Ns = np.concatenate([pairs[:, 0], np.unique(data.N)])
        xs = np.concatenate([pairs[:, 1], np.unique(data.N)])
        S, P = self.forward(lay1, G.digits(Ns, B, w), G.digits(xs, B, w + 1))
        reg = self.answer(lay1, P, np.arange(1, w + 2))                         # every register place
        loss = nll(reg, torch.from_numpy(G.digits(xs % Ns, B, w + 1)))
        if self.compiled:
            top = self.net.channel(P, "b")[:, :, 2, lay1.C - 1]                  # the SUB row's final borrow
            loss = loss + nll(top, torch.from_numpy((xs < Ns).astype(np.int64)))
        return loss

    def ref_state(self, lay, data):
        """The compiled machine's state of every cell (grid.machine_reference) as one-hot channels [X, D, R, C]."""
        ref = G.machine_reference(lay, data, self.reach)
        f = self.fmt
        S = self.net.blank(data.X, lay.R, lay.C)
        for n, sz in f.chans:
            if n in ref:
                S[:, f.sl(n)] = F.one_hot(torch.from_numpy(ref[n]), sz).permute(0, 3, 1, 2).to(self.dtype)
        return S, ref

    def fixpoint(self, lay, data):
        """The machine's settled grid must be a resting state: one tick from it gives it back, in every channel of
        every scratch cell. (A rule that reads its own state passes this by copying itself.)"""
        S, ref = self.ref_state(lay, data)
        Z = self.net.logits(S, lay)
        scratch = torch.from_numpy(~lay.prompt)
        tot = 0.0
        for n, sz in self.fmt.chans:
            lp = Z[:, self.fmt.sl(n)].log_softmax(1)
            tot = tot - lp.gather(1, torch.from_numpy(ref[n])[:, None]).squeeze(1)[:, scratch].mean()
        return tot / len(self.fmt.chans)

    def trace(self, lay, data):
        """Supervision by the compiled rule's trace (a reference upper bound, not a lesson of the project): along the
        compiled NCA's own run from random scratch states, every tick of the learned rule must give the compiled
        rule's next state, in every channel of every scratch cell. (Asking only that the machine's final state be
        reproduced is passed by a rule that copies its own state, and that rule never leaves the blank start.)"""
        from nca import compile as Cm
        if not hasattr(self, "_teacher"):
            self._teacher = {}
        if lay.task not in self._teacher:
            self._teacher[lay.task] = Cm.compiled_net(self.cfg.B, clamp=True, dtype=self.dtype, task=lay.task,
                                                      reach=self.reach)
        T_ = self._teacher[lay.task]
        with torch.no_grad():                                   # alternately from random and from blank scratch
            S0 = T_.prompt_state(lay, data.Nd, data.vd, "random" if self.step % 2 == 0 else "blank", self.gen)
            _, _, hist = T_.run(lay, S0, lay.depth, mode="hard", record=True)
            states = [S0] + hist[:-1]
            nxt = hist
        scratch = torch.from_numpy(~lay.prompt)
        tot = 0.0
        tf = T_.fmt
        keep = self.cfg.trace_chans.split(",") if self.cfg.trace_chans else None
        chans = [(n, sz) for n, sz in self.fmt.chans if keep is None or n in keep]
        if self.cfg.trace_ticks and self.cfg.trace_ticks < len(states):
            pick = sorted(self.rng.choice(len(states), self.cfg.trace_ticks, replace=False))
            states, nxt = [states[i] for i in pick], [nxt[i] for i in pick]
        for S, Sn in zip(states, nxt):
            if tf.D != self.fmt.D:                     # a division teacher for a rule that also has the square's
                S = self.widen_state(S, tf)            # channels: those sit at value 0
            Z = self.net.logits(S, lay)
            for n, sz in chans:
                lp = Z[:, self.fmt.sl(n)].log_softmax(1)
                t = Sn[:, tf.sl(n)].argmax(1) if n in tf.size else torch.zeros_like(Sn[:, 0], dtype=torch.long)
                tot = tot - lp.gather(1, t[:, None]).squeeze(1)[:, scratch].mean()
        return tot / (len(states) * len(chans))

    def widen_state(self, S, f_small):
        X, _, R, C = S.shape
        out = self.net.blank(X, R, C)
        for n, sz in f_small.chans:
            out[:, self.fmt.sl(n)] = S[:, f_small.sl(n)]
        return out

    def sub_query(self, a, m, cin, hard=False):
        """The rule at a SUB cell whose up neighbour holds digit distributions a [X, B] and n = m [X, B], whose left
        neighbour holds borrow distribution cin [X, 2], everything else blank or random (`ctx_mode`). Returns the SUB
        cell's digit [X, B] and borrow [X, 2] distributions, averaged over `ctx` contexts. (The generated-wiring rule
        is queried on its heads instead: hyper.HyperNCA.query_sub.)"""
        c, f, net = self.cfg, self.fmt, self.net
        if self.hyper:
            return net.query_sub(a, m, cin)
        X = a.shape[0]
        nctx = 1 if (hard or c.ctx_mode == "blank") else c.ctx
        R, C = 3, 3                                                              # rows SHIFT, SUB, SEL; 3 places
        outs_d, outs_b = [], []
        for _ in range(nctx):
            if hard or c.ctx_mode == "blank":
                S = net.blank(X, R, C)
            else:
                S = torch.zeros(X, f.D, R, C, dtype=self.dtype)
                for n, sz in f.chans:
                    S[:, f.sl(n)] = torch.rand(X, sz, R, C, generator=self.gen, dtype=self.dtype).softmax(1)
            S = S.clone()
            S[:, f.sl("d"), 0, 1] = a
            S[:, f.sl("n"), 0, 1] = m
            S[:, f.sl("b"), 1, 0] = cin
            Z = self.query_logits(S, hard)
            outs_d.append(Z[:, f.sl("d")].softmax(1))
            outs_b.append(Z[:, f.sl("b")].softmax(1))
        return torch.stack(outs_d).mean(0), torch.stack(outs_b).mean(0)

    def query_logits(self, S, hard=False):
        """The rule's logits at the centre of a 3 x 3 patch S [X, D, 3, 3] of kinds SHIFT / SUB / SEL by row (an
        interior place of a first link: no edge in sight)."""
        net = self.net
        kd = torch.tensor([[SHIFT] * 3, [SUB] * 3, [SEL] * 3])
        if net.cfg.kinds == "none":
            kd = torch.full((3, 3), SUB)
        K = F.one_hot(kd, net.nk).to(self.dtype).permute(2, 0, 1)[None].expand(S.shape[0], -1, -1, -1)
        inp = torch.cat([K, S], 1)
        extra = None
        if net.nextra:                                         # the long-range reads see the up neighbour's N digit
            X = S.shape[0]
            ex = inp[:, :, 0:1, 1:2].clone()                    # (as the prompt of its column) and the rest blank
            extra = torch.cat([ex] + [torch.cat([F.one_hot(torch.tensor(OUT), net.nk).to(self.dtype).expand(X, -1),
                                                 net.blank(X, 1, 1)[:, :, 0, 0]], 1)[:, :, None, None]] *
                              (net.nextra - 1), 1)
        return net.apply_rule(inp, hard, extra)[:, :, 0, 0]

    def sub_table(self, hard=False):
        B = self.cfg.B
        a, m, cc = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
        d, b = self.sub_query(self.hot(a.flatten()), self.hot(m.flatten()), self.hot(cc.flatten(), 2), hard)
        return d.view(B, B, 2, B), b.view(B, B, 2, 2)

    def ripple(self, Ad, Md):
        """A - M through the queried subtraction, place by place from 0 up: digit distributions [X, n, B] in;
        every place's borrow-out [X, n, 2] out."""
        X, n, B = Ad.shape
        cin = torch.zeros(X, 2, dtype=self.dtype)
        cin[:, 0] = 1
        cs = []
        for j in range(n):
            _, cin = self.sub_query(Ad[:, j], Md[:, j], cin)
            cs.append(cin)
        return torch.stack(cs, 1)

    def pref(self, data, Td, Tb):
        B, P = self.cfg.B, self.cfg.w + 1
        pairs = np.unique(np.stack([data.N, data.R0], 1), axis=0)
        Nd = G.digits(pairs[:, 0], B, P)
        Rd = G.digits(pairs[:, 1], B, P)
        lg = lambda x: x.clamp_min(1e-12).log()
        tot = 0.0
        for j in range(P):
            differ = torch.from_numpy((Rd[:, j:] != Nd[:, j:]).any(1)).to(self.dtype)
            p1 = self.ripple(self.hot(Rd[:, j:]), self.hot(Nd[:, j:]))[:, -1, 1]
            p2 = self.ripple(self.hot(Nd[:, j:]), self.hot(Rd[:, j:]))[:, -1, 1]
            tot = tot - (differ * lg(p1) + (1 - differ) * lg(1 - p1)).mean() - lg(1 - p2).mean()
        return tot / P

    def colall(self, Td, Tb):
        """ANCHORS.md's first law per column (m, c): taking the same digit away keeps the order, except that the
        one that wraps comes out on top (the learned borrows referee)."""
        B = self.cfg.B
        with torch.no_grad():
            bor = Tb.argmax(-1)
        tri = torch.triu(torch.ones(B, B, dtype=self.dtype), 1)
        tot, n = 0.0, 0
        for m in range(B):
            for c in range(2):
                for r in range(B):
                    for r2 in range(r + 1, B):
                        w0, w1 = int(bor[r, m, c]), int(bor[r2, m, c])
                        lt = int(bor[r, r2, 0])
                        first = (w0 < w1) or (w0 == w1 and lt == 1)
                        lo, hi = (Td[r, m, c], Td[r2, m, c]) if first else (Td[r2, m, c], Td[r, m, c])
                        tot = tot - (lo[:, None] * hi[None, :] * tri).sum().clamp_min(1e-12).log()
                        n += 1
        return tot / n

    def rowall(self, Td, Tb):
        B = self.cfg.B
        with torch.no_grad():
            bor = Tb.argmax(-1)
        tri = torch.triu(torch.ones(B, B, dtype=self.dtype), 1)
        tot, n = 0.0, 0
        for a in range(B):
            for c in range(2):
                for m in range(B):
                    for m2 in range(m + 1, B):
                        w0, w1 = int(bor[a, m, c]), int(bor[a, m2, c])
                        lt = int(bor[m, m2, 0])
                        first_lower = (w0 == 0) if w0 != w1 else (lt == 0)
                        lo, hi = (Td[a, m, c], Td[a, m2, c]) if first_lower else (Td[a, m2, c], Td[a, m, c])
                        tot = tot - (lo[:, None] * hi[None, :] * tri).sum().clamp_min(1e-12).log()
                        n += 1
        return tot / n

    def bij(self, Td):
        B = self.cfg.B
        lg = lambda x: x.clamp_min(1e-12).log()
        return -(lg(B * Td.mean(0)).mean() + lg(B * Td.mean(1)).mean())

    # ----- measures -----
    @torch.no_grad()
    def exact(self, B, w, k, sample=0, ticks=None, mode="hard", chunk=2048, detail=False, task="div"):
        lay = self.layout(B, w, k, task=task)
        data = self.dataset(B, w, k, sample, task=task)
        n, ans = 0, []
        for i in range(0, data.X, chunk):
            d = data if data.X <= chunk else data.subset(np.arange(i, min(i + chunk, data.X)))
            S, P = self.forward(lay, d.Nd, d.vd, mode=mode, ticks=ticks or lay.depth, train=False)
            got = self.answer(lay, S if mode == "hard" else P).argmax(-1).numpy()
            n += int((got == d.yd).all(-1).sum())
            if detail:
                ans.append(G.value(got, B))
        if detail:
            return n / data.X, np.concatenate(ans), data
        return n / data.X

    @torch.no_grad()
    def probe(self, task=None):
        """Where the NCA stands against the compiled machine, at the training size on the hard path: per row kind,
        the share of cells whose channel holds the machine's value (d for every format; every channel for the
        compiled one); the same one tick from the machine's own state (tf_: the rule right where its neighbours
        are); and the impostor answers: the share of questions answered R0 (the register's start), v's low digits,
        or 0."""
        c = self.cfg
        task = task or c.task
        k = c.w if task == "sq" else c.k
        lay = self.layout(c.B, c.w, k, task=task)
        data = self.dataset(c.B, c.w, k, task=task)
        S, _ = self.forward(lay, data.Nd, data.vd, mode="hard", ticks=lay.depth, train=False)
        ref = G.machine_reference(lay, data, self.reach)
        out = {}
        chans = [n for n, _ in self.fmt.chans if n in ref] if self.compiled else ["d"]
        kinds = [kd for kd in (SPLIT, MUL, SHIFT, SUB, SEL) if (lay.kind[:, 0] == kd).any()]
        for ch in chans:
            got = self.net.channel(S, ch).argmax(1).numpy()
            out[f"cells_{ch}"] = {G.KINDS[kd]: round(float((got[:, lay.kind[:, 0] == kd] ==
                                                         ref[ch][:, lay.kind[:, 0] == kd]).mean()), 3)
                                  for kd in kinds}
        if self.compiled:                                                            # one tick from the machine's state
            Sr, _ = self.ref_state(lay, data)
            Z = self.net.logits(Sr, lay, hard=True)
            for ch in chans:
                got1 = Z[:, self.fmt.sl(ch)].argmax(1).numpy()
                out[f"tf_{ch}"] = {G.KINDS[kd]: round(float((got1[:, lay.kind[:, 0] == kd] ==
                                                               ref[ch][:, lay.kind[:, 0] == kd]).mean()), 3)
                                   for kd in kinds}
        got = G.value(self.answer(lay, S).argmax(-1).numpy(), c.B)
        out["answers_R0"] = round(float((got == data.R0).mean()), 3)
        out["answers_vlow"] = round(float((got == data.v % c.B ** c.w).mean()), 3)
        out["answers_0"] = round(float((got == 0).mean()), 3)
        out["answers_right"] = round(float((got == data.y).mean()), 3)
        if task == "sq":
            out["answers_x"] = round(float((got == data.x).mean()), 3)
        if self.compiled:
            out.update(self.sub_right())
        dig = self.net.channel(S, "d").argmax(1).numpy()                           # scratch cells constant over
        const = (dig == dig[:1]).all(0)                                            # every question
        out["const_scratch_cells"] = round(float(const[~lay.prompt].mean()), 3)
        if not self.compiled:
            out["relabel"] = self.relabel(lay, data, S)
        if self.hyper:
            from nca import hyper as Hy
            out["wiring"] = Hy.wiring_probe(self.net, lay)
        if self.ncfg.rule == "mlp":                       # input features (position, kind or channel value) whose
            W = (self.net.W1 * self.net.mask).detach().abs()   # largest weight is at least 10% of the largest
            mx = W.amax(dim=(0,)).flatten()
            out["inputs_read"] = int((mx >= 0.1 * mx.max()).sum())
            out["inputs_total"] = int((self.net.mask.flatten() > 0).sum())
        return out

    def relabel(self, lay, data, S):
        """The free format's hidden channels against the compiled machine's (n, b, m, g): for each hidden channel,
        the machine channel it predicts best through a lookup of its values (per row kind), and how well; and the
        most redundant pair of hidden channels (a symmetric pair carries the same code)."""
        ref = G.machine_reference(lay, data)
        sc = ~lay.prompt
        kinds = lay.kind[None].repeat(data.X, 0)[:, sc]
        hidden = [n for n, _ in self.fmt.chans if n not in ("d", "n")] + ["n"]
        vals = {n: self.net.channel(S, n).argmax(1).numpy()[:, sc] for n in hidden}

        def lookup_acc(x, y):
            acc = 0
            for kd in np.unique(kinds):
                m = kinds == kd
                xs, ys = x[m], y[m]
                for v in np.unique(xs):
                    acc += np.bincount(ys[xs == v]).max()
            return acc / x.size

        const = np.zeros_like(vals["n"])
        lift = lambda x, y: ((lookup_acc(x, y) - lookup_acc(const, y)) / max(1e-9, 1 - lookup_acc(const, y)))
        out = {}
        for n in hidden:                        # lift: 0 = no better than the kind's majority value, 1 = exact
            best = max(((lift(vals[n], ref[r][:, sc]), r) for r in ("n", "b", "m", "g")))
            out[n] = (best[1], round(float(best[0]), 3))
        pairs = [(lift(vals[a], vals[b]), a, b) for a in hidden for b in hidden
                 if a != b and lookup_acc(const, vals[b]) < 1]
        if pairs:
            top = max(pairs)
            out["redundant"] = (top[1], top[2], round(float(top[0]), 3))
        return out

    def sub_right(self):
        Td, Tb = self.sub_table(hard=True)
        B = self.cfg.B
        a, m, cc = np.meshgrid(np.arange(B), np.arange(B), np.arange(2), indexing="ij")
        x = a - m - cc
        return {"sub_digits_right": int((Td.argmax(-1).numpy() == x % B).sum()),
                "sub_borrows_right": int((Tb.argmax(-1).numpy() == (x < 0)).sum())}

    def train_exact(self, kk):
        c = self.cfg
        if c.task == "sq":
            return self.exact(c.B, c.w, c.w, task="sq")
        return self.exact(c.B, c.w, kk)

    # ----- training -----
    def train(self):
        c = self.cfg
        stages = list(c.stages) or [c.k]
        step = 0
        rec = {"stages": []}
        if c.init:
            rec["start_hard"] = self.train_exact(c.k)
            rec["start_soft"] = (self.exact(c.B, c.w, c.w, mode="soft", task="sq") if c.task == "sq" else
                                 self.exact(c.B, c.w, c.k, mode="soft"))
        for si, kk in enumerate(stages):
            limit = c.stage_steps or c.steps
            s0 = step
            stage_hit = None
            best, best_step = -1.0, step
            while step < c.steps and step - s0 < limit:
                self.step = step
                self.opt.zero_grad()
                loss, terms = self.lessons(kk)
                if not torch.is_tensor(loss):
                    break
                loss.backward()
                if self.hyper and step < c.freeze_g:                   # the wiring waits while the rule learns
                    for p in (self.net.G1, self.net.g1, self.net.G2, self.net.g2):
                        p.grad = None
                self.opt.step()
                step += 1
                if step % c.every == 0:
                    ex = self.train_exact(kk)
                    row = {"step": step, "k": kk, "loss": round(float(loss.detach()), 5), "exact": ex,
                           **{n: round(v, 5) for n, v in terms.items()}}
                    if c.task == "sq":
                        row["div_exact"] = self.exact(c.B, c.w, c.div_k)
                    if self.compiled:
                        sr = self.sub_right()
                        row["sub"] = sr["sub_digits_right"] + sr["sub_borrows_right"]
                    self.log.append(row)
                    if ex > best + 1e-9:
                        best, best_step = ex, step
                    if ex == 1.0 and stage_hit is None:
                        stage_hit = step
                    if stage_hit is not None and step - stage_hit >= (c.post if si == len(stages) - 1 else 0):
                        break
                    if step - best_step >= c.patience:
                        break
            rec["stages"].append({"k": kk, "hit": stage_hit, "end": step})
            if stage_hit is None:
                break
        self.step = step
        rec["hit"] = rec["stages"][-1]["hit"] if len(rec["stages"]) == len(stages) else None
        rec.update(steps=step, seconds=round(time.time() - self.t0, 1), params=self.nparams)
        if c.task == "sq":
            self.finish_sq(rec)
        else:
            self.finish_div(rec)
        if c.save_weights:
            rec["weights"] = save_state(self.net)
        rec["log"] = self.log
        return rec

    def finish_div(self, rec):
        c = self.cfg
        rec["train_hard"] = self.exact(c.B, c.w, c.k)
        rec["train_soft"] = self.exact(c.B, c.w, c.k, mode="soft")
        rec["probe"] = self.probe()
        if not c.cert:
            return
        bigger = [(c.B, c.w, c.k + 1), (c.B, c.w + 1, c.k + 1)]
        rec["bigger"] = {f"{b},{w_},{k_}": self.exact(b, w_, k_, sample=c.cert_sample) for b, w_, k_ in bigger}
        rec["certified"] = bool(rec["train_hard"] == 1.0 and all(v == 1.0 for v in rec["bigger"].values()))
        side = c.cert_sample or 5000                   # the soft path and extra ticks: a sample even when certifying
        rec["bigger_soft"] = {f"{b},{w_},{k_}": self.exact(b, w_, k_, sample=side, mode="soft")   # on every question
                              for b, w_, k_ in bigger}
        if rec["train_hard"] == 1.0:
            rec["extra"] = {f"{c.B},{c.w + dw},{c.k + dk}": self.exact(c.B, c.w + dw, c.k + dk, sample=side)
                            for dw, dk in c.extra_sizes}
            rec["ticks"] = {}
            for mult in c.tick_mults:
                for b, w_, k_ in [(c.B, c.w, c.k), (c.B, c.w + 1, c.k + 1)]:
                    lay = self.layout(b, w_, k_)
                    rec["ticks"][f"{b},{w_},{k_} x{mult}"] = self.exact(b, w_, k_, sample=side, ticks=mult * lay.depth)

    def finish_sq(self, rec):
        """The square: exact at its training size (B, w) and certified if also exact at (B, w + 1) and (B, w + 2);
        the division grid measured at its own sizes as well."""
        c = self.cfg
        rec["train_hard"] = self.exact(c.B, c.w, c.w, task="sq")
        rec["train_soft"] = self.exact(c.B, c.w, c.w, mode="soft", task="sq")
        rec["probe"] = self.probe("sq")
        rec["div_probe"] = self.probe("div")
        rec["div"] = {f"{c.B},{w_},{k_}": self.exact(c.B, w_, k_, sample=c.cert_sample)
                      for w_, k_ in [(c.w, c.div_k), (c.w, c.div_k + 1), (c.w + 1, c.div_k + 1)]}
        if not c.cert:
            return
        bigger = [(c.B, c.w + 1), (c.B, c.w + 2)]
        rec["bigger"] = {f"sq {b},{w_}": self.exact(b, w_, w_, sample=c.cert_sample, task="sq") for b, w_ in bigger}
        rec["certified"] = bool(rec["train_hard"] == 1.0 and all(v == 1.0 for v in rec["bigger"].values()))
        rec["bigger_soft"] = {f"sq {b},{w_}": self.exact(b, w_, w_, sample=c.cert_sample, mode="soft", task="sq")
                              for b, w_ in bigger}
        if rec["train_hard"] == 1.0:
            rec["ticks"] = {}
            for mult in c.tick_mults:
                for w_ in (c.w, c.w + 1):
                    lay = self.layout(c.B, w_, w_, task="sq")
                    rec["ticks"][f"sq {c.B},{w_} x{mult}"] = self.exact(c.B, w_, w_, sample=c.cert_sample,
                                                                      ticks=mult * lay.depth, task="sq")


def save_state(net) -> dict:
    """The rule's parameters and buffers as {name: [shape, base64 of float32 bytes]}."""
    import base64
    out = {}
    for name, t in list(net.state_dict().items()):
        a = t.detach().to(torch.float32).contiguous().numpy()
        out[name] = [list(a.shape), base64.b64encode(a.tobytes()).decode()]
    return out


def load_state(rec) -> Run:
    """A Run whose rule holds a record's saved weights (its cfg rebuilt, nothing trained)."""
    import base64
    cfg = RunCfg(**{k: (tuple(map(tuple, v)) if k == "extra_sizes" else tuple(v) if k in ("stages", "tick_mults") else v)
                    for k, v in rec["cfg"].items()})
    r = Run(cfg)
    sd = {}
    for name, (shape, b) in rec["weights"].items():
        sd[name] = torch.from_numpy(np.frombuffer(base64.b64decode(b), dtype=np.float32).reshape(shape).copy()).to(r.dtype)
    r.net.load_state_dict(sd)
    return r


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
    p = rec.get("probe", {})
    if "sub_digits_right" in p:
        bits.append(f"sub {p['sub_digits_right']}/{p['sub_borrows_right']}")
    bits.append(f"R0 {p.get('answers_R0')} vlow {p.get('answers_vlow')} zero {p.get('answers_0')}")
    bits.append(f"{rec['seconds']:.0f}s")
    return " | ".join(bits)


@torch.no_grad()
def situations(r: Run, B, w, k, task="div", sample=3000):
    """Where a trained rule breaks on a size: one tick from the compiled machine's own state (every neighbour
    right), the share of questions on which each cell comes out right, per row and place, in every channel the
    format shares with the machine; and the same for the free run (cells_d). Rows are labelled by kind."""
    lay = r.layout(B, w, k, task=task)
    data = r.dataset(B, w, k, sample, task=task)
    Sr, ref = r.ref_state(lay, data)
    Z = r.net.logits(Sr, lay, hard=True)
    out = {"rows": [n if t < 0 else f"{n}({t},{s})" for n, t, s in lay.rows], "tf": {}, "free_d": None}
    chans = [n for n, _ in r.fmt.chans if n in ref] if r.compiled else ["d"]
    if r.compiled:
        for ch in chans:
            got = Z[:, r.fmt.sl(ch)].argmax(1).numpy()
            out["tf"][ch] = np.round((got == ref[ch]).mean(0), 3).tolist()
    S, _ = r.forward(lay, data.Nd, data.vd, mode="hard", ticks=lay.depth, train=False)
    got = r.net.channel(S, "d").argmax(1).numpy()
    out["free_d"] = np.round((got == ref["d"]).mean(0), 3).tolist()
    return out


def diag(cfg: RunCfg, sizes) -> dict:
    """Train, then map where the rule breaks at each size (`situations`)."""
    r = Run(cfg)
    out = r.train()
    out["cfg"] = asdict(cfg)
    out["situations"] = {",".join(map(str, s)): situations(r, *s) for s in sizes}
    return out
