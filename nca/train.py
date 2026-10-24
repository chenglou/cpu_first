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
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from nca import grid as G
from nca import model as M
from nca.grid import MUL, OUT, SEL, SHIFT, SPLIT, SUB

CKPT_HOOK = None                                       # called after each checkpoint is written (Modal: vol.commit)


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
    device: str = "cpu"                                # cpu | cuda | mps: where the run's tensors live
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
    optim: str = "adam"                                # adam | adamw (decoupled weight decay `wd`: trimming)
    wd: float = 0.0
    agree: str = ""                                    # "": plain; "and": AND-mask over question groups;
                                                       # "and_cols": over blocks of grid columns (cell positions);
                                                       # "dbd": delta-bar-delta, a per-weight speed by agreement over
                                                       # time
    agree_groups: int = 8                              # and: the training questions split into this many groups
    agree_tau: float = 0.75                            # and: keep a component if |mean sign over groups| >= tau
    agree_cols: int = 4                                # and_cols: the grid's columns in this many blocks
    dbd_up: float = 0.02                               # dbd: speed += up where the gradient agrees with its trace,
    dbd_down: float = 0.1                              # speed *= 1 - down where it does not; speeds in [lo, hi]
    dbd_lo: float = 0.1
    dbd_hi: float = 10.0
    nostop: bool = False                               # never stop early: run every one of `steps`
    curve_every: int = 0                               # every so many steps: soft training share, one size up, the
    curve_sample: int = 500                            # weight norm, the sub table, the plumbing probe
    l1: float = 0.0                                    # L1 on the rule's input weights (mean |W1|, and |Wx|): a rule
                                                       # that reads few inputs meets fewer new situations
    ckpt: str = ""                                     # a path: the run saved there at every curve point, and resumed
                                                       # from it after a restart (one-stage runs; same settings)
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


def nll_rows(p, y):
    """nll per row: [X]."""
    t = -p.clamp_min(1e-12).log().gather(-1, y[..., None]).squeeze(-1)
    return t.reshape(len(t), -1).mean(1)


def agree_rows(p, q):
    t = -(p * q).sum(-1).clamp_min(1e-12).log()
    return t.reshape(len(t), -1).mean(1)


class Run:
    def __init__(self, cfg: RunCfg):
        self.cfg = c = cfg
        if c.device != "cpu":
            torch.set_default_device(c.device)          # every tensor the run makes lands there
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
            g = torch.Generator(device=torch.get_default_device()).manual_seed(7000 + c.seed)
            if self.hyper:
                Hy.compile_ports(self.net, c.B, beta=c.init_beta, beta_o=c.init_beta)
                with torch.no_grad():
                    for p in self.net.parameters():
                        p.add_(c.init_noise * torch.randn(p.shape, generator=g, dtype=p.dtype))
            else:
                from nca import compile as Cm
                Cm.set_compiled(self.net, c.B, beta=c.init_beta, noise=c.init_noise, gen=g, task=c.task)
        if c.optim == "adamw":
            self.opt = torch.optim.AdamW(self.net.parameters(), lr=c.lr, weight_decay=c.wd)
        else:
            self.opt = torch.optim.Adam(self.net.parameters(), lr=c.lr)
        self.speed = None                                    # dbd: one speed per weight, and a trace of gradients
        self.trace_g = None
        self.nparams = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.lay, self.data = {}, {}
        self.rng = np.random.default_rng(1000 + c.seed)
        self.gen = torch.Generator(device=torch.get_default_device()).manual_seed(2000 + c.seed)
        self.t0 = time.time()
        self.log = []
        self.curve = []
        self.kept = 1.0
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
    def lessons(self, k, idx=None):
        """Every lesson's term on the training questions (all, a random `batch`, or the questions `idx`)."""
        c = self.cfg
        W = {n: v for n, v in c.lessons.items() if v and self.step >= c.lesson_after.get(n, 0)}
        if c.task == "div":
            terms = self.div_terms(k, W, idx)
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
                terms["ans"] = nll(self.answer(lay, P), torch.as_tensor(data.yd))
            if "prod" in W:
                r = 2 + c.w - 1                                                       # the last MUL row
                pd = self.net.channel(P, "d")[:, :, r].permute(0, 2, 1)
                terms["prod"] = nll(pd, torch.as_tensor(G.digits(data.v, c.B, lay.C)))
        if "trace" in W and self.compiled:
            terms["trace"] = self.trace(lay, data)
        return terms

    def div_terms(self, k, W, idx=None):
        c = self.cfg
        B, w = c.B, c.w
        lay = self.layout(B, w, k)
        data = self.dataset(B, w, k)
        if idx is not None:
            data = data.subset(idx)
        elif c.batch and data.X > c.batch:
            data = data.subset(self.rng.choice(data.X, c.batch, replace=False))
        terms = {}
        if "ans" in W or "step" in W:
            _, P = self.forward(lay, data.Nd, data.vd)
            pa = self.answer(lay, P)
            if "ans" in W:
                terms["ans"] = nll(pa, torch.as_tensor(data.yd))
        if "xmodn" in W:
            pairs = np.unique(np.stack([data.N, data.R0], 1), axis=0)
            Ns = np.concatenate([pairs[:, 0], np.unique(data.N)])
            xs = np.concatenate([pairs[:, 1], np.unique(data.N)])
            _, P = self.forward(lay, G.digits(Ns, B, w), G.digits(xs, B, w + k))
            terms["xmodn"] = nll(self.answer(lay, P), torch.as_tensor(G.digits(xs % Ns, B, w)))
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
            sc = torch.as_tensor(~lay.prompt)
            lp = self.net.channel(P, "d").clamp_min(1e-12).log()
            terms["sheet"] = -lp.gather(1, torch.as_tensor(ref["d"])[:, None]).squeeze(1)[:, sc].mean()
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
        loss = nll(reg, torch.as_tensor(G.digits(xs % Ns, B, w + 1)))
        if self.compiled:
            top = self.net.channel(P, "b")[:, :, 2, lay1.C - 1]                  # the SUB row's final borrow
            loss = loss + nll(top, torch.as_tensor((xs < Ns).astype(np.int64)))
        return loss

    def ref_state(self, lay, data):
        """The compiled machine's state of every cell (grid.machine_reference) as one-hot channels [X, D, R, C]."""
        ref = G.machine_reference(lay, data, self.reach)
        f = self.fmt
        S = self.net.blank(data.X, lay.R, lay.C)
        for n, sz in f.chans:
            if n in ref:
                S[:, f.sl(n)] = F.one_hot(torch.as_tensor(ref[n]), sz).permute(0, 3, 1, 2).to(self.dtype)
        return S, ref

    def fixpoint(self, lay, data):
        """The machine's settled grid must be a resting state: one tick from it gives it back, in every channel of
        every scratch cell. (A rule that reads its own state passes this by copying itself.)"""
        S, ref = self.ref_state(lay, data)
        Z = self.net.logits(S, lay)
        scratch = torch.as_tensor(~lay.prompt)
        tot = 0.0
        for n, sz in self.fmt.chans:
            lp = Z[:, self.fmt.sl(n)].log_softmax(1)
            tot = tot - lp.gather(1, torch.as_tensor(ref[n])[:, None]).squeeze(1)[:, scratch].mean()
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
        scratch = torch.as_tensor(~lay.prompt)
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
            differ = torch.as_tensor((Rd[:, j:] != Nd[:, j:]).any(1)).to(self.dtype)
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
            got = self.answer(lay, S if mode == "hard" else P).argmax(-1).cpu().numpy()
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
            got = self.net.channel(S, ch).argmax(1).cpu().numpy()
            out[f"cells_{ch}"] = {G.KINDS[kd]: round(float((got[:, lay.kind[:, 0] == kd] ==
                                                         ref[ch][:, lay.kind[:, 0] == kd]).mean()), 3)
                                  for kd in kinds}
        if self.compiled:                                                            # one tick from the machine's state
            Sr, _ = self.ref_state(lay, data)
            Z = self.net.logits(Sr, lay, hard=True)
            for ch in chans:
                got1 = Z[:, self.fmt.sl(ch)].argmax(1).cpu().numpy()
                out[f"tf_{ch}"] = {G.KINDS[kd]: round(float((got1[:, lay.kind[:, 0] == kd] ==
                                                               ref[ch][:, lay.kind[:, 0] == kd]).mean()), 3)
                                   for kd in kinds}
        got = G.value(self.answer(lay, S).argmax(-1).cpu().numpy(), c.B)
        out["answers_R0"] = round(float((got == data.R0).mean()), 3)
        out["answers_vlow"] = round(float((got == data.v % c.B ** c.w).mean()), 3)
        out["answers_0"] = round(float((got == 0).mean()), 3)
        out["answers_right"] = round(float((got == data.y).mean()), 3)
        if task == "sq":
            out["answers_x"] = round(float((got == data.x).mean()), 3)
        if self.compiled:
            out.update(self.sub_right())
        dig = self.net.channel(S, "d").argmax(1).cpu().numpy()                           # scratch cells constant over
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
        vals = {n: self.net.channel(S, n).argmax(1).cpu().numpy()[:, sc] for n in hidden}

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
        return {"sub_digits_right": int((Td.argmax(-1).cpu().numpy() == x % B).sum()),
                "sub_borrows_right": int((Tb.argmax(-1).cpu().numpy() == (x < 0)).sum())}

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
        assert not c.ckpt or len(stages) == 1, "resuming is for one-stage runs"
        for si, kk in enumerate(stages):
            limit = c.stage_steps or c.steps
            s0 = step
            stage_hit = None
            best, best_step = -1.0, step
            if c.ckpt:
                ck = self.load_ckpt()
                if ck:
                    step, stage_hit, best, best_step = ck["step"], ck["stage_hit"], ck["best"], ck["best_step"]
                    rec["resumed_at"] = ck["resumed_at"] + [step]
            while step < c.steps and step - s0 < limit:
                self.step = step
                self.opt.zero_grad()
                if c.agree == "and":
                    loss, terms = self.and_mask_grads(kk)
                elif c.agree == "and_cols":
                    loss, terms = self.and_cols_grads(kk)
                else:
                    loss, terms = self.lessons(kk)
                    if not torch.is_tensor(loss):
                        break
                    loss.backward()
                if self.hyper and step < c.freeze_g:                   # the wiring waits while the rule learns
                    for p in (self.net.G1, self.net.g1, self.net.G2, self.net.g2):
                        p.grad = None
                if c.agree == "dbd":
                    self.dbd_step()
                else:
                    self.opt.step()
                step += 1
                if step % c.every == 0:
                    ex = self.train_exact(kk)
                    row = {"step": step, "k": kk, "loss": round(float(loss.detach()), 5), "exact": ex,
                           "t": round(time.time() - self.t0, 1),
                           **{("step_law" if n == "step" else n): round(v, 5) for n, v in terms.items()}}
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
                    if c.curve_every and step % c.curve_every == 0:
                        self.curve.append(self.curve_point(kk, step))
                        if c.ckpt:
                            self.save_ckpt(step, stage_hit, best, best_step, rec.get("resumed_at", []))
                    if c.nostop:
                        continue
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
        if c.curve_every:
            rec["curve"] = self.curve
        return rec

    # ----- checkpoints -----
    def resume_key(self):
        return {k: v for k, v in asdict(self.cfg).items() if k not in ("ckpt", "device", "steps")}

    def same_settings(self, saved):
        """Whether a checkpoint's settings are this run's: every saved setting the same, and every setting RunCfg has
        gained since at its default (the code that wrote the checkpoint ran without it, which is what the default
        does)."""
        now, default = self.resume_key(), asdict(RunCfg())
        return all(now.get(k) == v for k, v in saved.items()) and all(now[k] == default[k] for k in now if k not in saved)

    def save_ckpt(self, step, stage_hit, best, best_step, resumed_at):
        """The run so far, to resume from after a restart: the rule, the optimizer, the speeds, every random state, the
        log and the curve (written to a temporary file first); and the curve alone beside it, to peek at."""
        c = self.cfg
        state = {"key": self.resume_key(), "step": step, "stage_hit": stage_hit, "best": best, "best_step": best_step,
                 "resumed_at": resumed_at, "t": time.time() - self.t0, "net": self.net.state_dict(),
                 "opt": self.opt.state_dict(), "speed": self.speed, "trace_g": self.trace_g, "kept": self.kept,
                 "rng": self.rng.bit_generator.state, "gen": self.gen.get_state(), "torch": torch.get_rng_state(),
                 "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 "log": self.log, "curve": self.curve}
        os.makedirs(os.path.dirname(c.ckpt) or ".", exist_ok=True)
        torch.save(state, c.ckpt + ".tmp")
        os.replace(c.ckpt + ".tmp", c.ckpt)
        with open(c.ckpt + ".json.tmp", "w") as f:
            json.dump({"step": step, "steps": c.steps, "t": round(state["t"]), "curve": self.curve}, f)
        os.replace(c.ckpt + ".json.tmp", c.ckpt + ".json")
        if CKPT_HOOK:
            CKPT_HOOK()

    def load_ckpt(self):
        """The saved run if there is one with the same settings (the step budget and the device may differ)."""
        c = self.cfg
        if not os.path.exists(c.ckpt):
            return None
        ck = torch.load(c.ckpt, map_location="cpu", weights_only=False)
        if not self.same_settings(ck["key"]):
            return None
        dev = torch.get_default_device()
        self.net.load_state_dict(ck["net"])
        self.opt.load_state_dict(ck["opt"])
        self.speed = None if ck["speed"] is None else [x.to(dev) for x in ck["speed"]]
        self.trace_g = None if ck["trace_g"] is None else [x.to(dev) for x in ck["trace_g"]]
        self.kept = ck["kept"]
        self.rng.bit_generator.state = ck["rng"]
        self.gen.set_state(ck["gen"])
        torch.set_rng_state(ck["torch"])
        if ck["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda"])
        self.log, self.curve = ck["log"], ck["curve"]
        self.t0 = time.time() - ck["t"]
        return ck

    # ----- agreement -----
    def and_groups(self, k):
        c = self.cfg
        n = self.dataset(c.B, c.w, k).X
        pool = self.rng.choice(n, c.batch, replace=False) if c.batch and n > c.batch else self.rng.permutation(n)
        return np.array_split(pool, c.agree_groups)

    def and_mask(self, grads, sums=None):
        """grads: per parameter, the groups' gradients [G, ...]. Leaves in .grad the mean gradient (or `sums`) where at
        least a fraction `agree_tau` of the groups agree on its sign (|mean of the signs| >= tau), zero elsewhere."""
        kept = total = 0
        for i, (p, g) in enumerate(zip(self.net.parameters(), grads)):
            mask = g.sign().mean(0).abs() >= self.cfg.agree_tau
            p.grad = mask.to(p.dtype) * (g.mean(0) if sums is None else sums[i])
            kept += int(mask.sum())
            total += mask.numel()
        self.kept = kept / max(1, total)

    def and_mask_grads(self, k):
        """The AND-mask (Parascandolo et al. 2021): the training questions in `agree_groups` random groups, every
        lesson's gradient on each group, and of the mean gradient only the components on whose sign at least a
        fraction `agree_tau` of the groups agree; the rest are zeroed. The groups run in one pass, each on its own copy
        of the rule's weights (model.NCA.groups); the lessons that do not depend on the questions (the cell's table
        laws) run once on the rule itself, and their gradient joins every group's. Leaves the masked gradient in .grad
        and returns the mean loss over groups and the mean terms. (and_mask_grads_seq is the same, a group at a
        time.)"""
        c, net = self.cfg, self.net
        assert c.task == "div" and not self.hyper and self.ncfg.rule == "mlp" and not net.nextra and not c.l1
        groups = self.and_groups(k)
        base = [net.W1, net.b1, net.W2, net.b2]
        assert len(base) == len(list(net.parameters()))
        self.copies = [p.detach()[None].repeat(len(groups), *[1] * p.dim()).requires_grad_() for p in base]
        terms, shared, W = self.group_terms(k, groups)
        loss_q = sum(W[n] * t for n, t in terms.items())
        loss_s = sum(W[n] * t for n, t in shared.items())
        for p in base:
            p.grad = None
        (loss_q + loss_s).backward()
        grads = [cp.grad + (p.grad if p.grad is not None else 0) for cp, p in zip(self.copies, base)]
        self.and_mask(grads)
        self.copies = None
        G_ = len(groups)
        out = {n: float(t.detach()) / G_ for n, t in terms.items()}
        out.update({n: float(t.detach()) for n, t in shared.items()})
        return torch.tensor(float(loss_q.detach()) / G_ + float(loss_s.detach() if torch.is_tensor(loss_s) else loss_s)), out

    def on_groups(self, sizes):
        """Rows in one block per group, each padded to the largest by repeating its last row, on the groups' weight
        copies (model.NCA.groups): the index of each padded row into the groups' rows laid end to end, and the mask
        of the real ones [G, n]."""
        assert min(sizes) > 0
        n = max(sizes)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
        idx = np.concatenate([s0 + np.minimum(np.arange(n), sz - 1) for s0, sz in zip(starts, sizes)])
        real = torch.as_tensor(np.arange(n)[None, :] < np.array(sizes)[:, None]).to(self.dtype)
        self.net.groups = (self.copies, "rows", len(sizes))
        return idx, real

    def group_terms(self, k, groups):
        """div_terms for the AND-mask's groups at once: each question term summed over the groups (each group's own
        mean, as div_terms on the group alone would give), and the terms that do not depend on the questions (colall,
        rowall, bij) once, on the rule itself."""
        c = self.cfg
        B, w = c.B, c.w
        W = {n: v for n, v in c.lessons.items() if v and self.step >= c.lesson_after.get(n, 0)}
        assert set(W) <= {"ans", "xmodn", "step", "link", "pref", "colall", "rowall", "bij"}, W
        lay = self.layout(B, w, k)
        full = self.dataset(B, w, k)
        datas = [full.subset(g) for g in groups]
        gsum = lambda rows, real: ((rows.view(real.shape) * real).sum(1) / real.sum(1)).sum()
        lg = lambda x: x.clamp_min(1e-12).log()

        def pairs_of(d):
            pr = np.unique(np.stack([d.N, d.R0], 1), axis=0)
            return np.concatenate([pr[:, 0], np.unique(d.N)]), np.concatenate([pr[:, 1], np.unique(d.N)])

        terms, shared = {}, {}
        try:
            if "ans" in W or "step" in W:
                idx, real = self.on_groups([d.X for d in datas])
                data = full.subset(np.concatenate(groups)[idx])
                _, P = self.forward(lay, data.Nd, data.vd)
                pa = self.answer(lay, P)
                if "ans" in W:
                    terms["ans"] = gsum(nll_rows(pa, torch.as_tensor(data.yd)), real)
            if "step" in W and k >= 2:
                _, P1 = self.forward(lay, data.Nd, G.digits(data.v // B, B, w + k))
                r1 = self.answer(lay, P1)
                v2 = torch.zeros(data.X, w + k, B, dtype=self.dtype)
                v2[:, 0] = self.hot(data.vd[:, 0])
                v2[:, 1:w + 1] = r1
                v2[:, w + 1:, 0] = 1
                _, P2 = self.forward(lay, self.hot(data.Nd), v2)
                terms["step"] = gsum(agree_rows(pa, self.answer(lay, P2)), real)
            if "xmodn" in W or "link" in W:
                both = [pairs_of(d) for d in datas]
                idx, real = self.on_groups([len(x[0]) for x in both])
                Ns = np.concatenate([x[0] for x in both])[idx]
                xs = np.concatenate([x[1] for x in both])[idx]
            if "xmodn" in W:
                _, P = self.forward(lay, G.digits(Ns, B, w), G.digits(xs, B, w + k))
                terms["xmodn"] = gsum(nll_rows(self.answer(lay, P), torch.as_tensor(G.digits(xs % Ns, B, w))), real)
            if "link" in W:                                                   # link() on each group
                lay1 = self.layout(B, w, 1, links=1)
                _, P = self.forward(lay1, G.digits(Ns, B, w), G.digits(xs, B, w + 1))
                rows = nll_rows(self.answer(lay1, P, np.arange(1, w + 2)), torch.as_tensor(G.digits(xs % Ns, B, w + 1)))
                if self.compiled:
                    top = self.net.channel(P, "b")[:, :, 2, lay1.C - 1]
                    rows = rows + nll_rows(top, torch.as_tensor((xs < Ns).astype(np.int64)))
                terms["link"] = gsum(rows, real)
            if "pref" in W:                                                   # pref() on each group
                Pn = w + 1
                prs = [np.unique(np.stack([d.N, d.R0], 1), axis=0) for d in datas]
                idx, real = self.on_groups([len(x) for x in prs])
                pr = np.concatenate(prs)[idx]
                Nd, Rd = G.digits(pr[:, 0], B, Pn), G.digits(pr[:, 1], B, Pn)
                tot = 0.0
                for j in range(Pn):
                    differ = torch.as_tensor((Rd[:, j:] != Nd[:, j:]).any(1)).to(self.dtype)
                    p1 = self.ripple(self.hot(Rd[:, j:]), self.hot(Nd[:, j:]))[:, -1, 1]
                    p2 = self.ripple(self.hot(Nd[:, j:]), self.hot(Rd[:, j:]))[:, -1, 1]
                    tot = tot - gsum(differ * lg(p1) + (1 - differ) * lg(1 - p1), real) - gsum(lg(1 - p2), real)
                terms["pref"] = tot / Pn
        finally:
            self.net.groups = None
        if any(n in W for n in ("colall", "rowall", "bij")):
            Td, Tb = self.sub_table()
            if "colall" in W:
                shared["colall"] = self.colall(Td, Tb)
            if "rowall" in W:
                shared["rowall"] = self.rowall(Td, Tb)
            if "bij" in W:
                shared["bij"] = self.bij(Td)
        return terms, shared, W

    def and_cols_grads(self, k):
        """The AND-mask across cell positions: the grid's columns in `agree_cols` blocks of neighbouring places, the
        cells of each block on their own copy of the rule's weights (model.NCA.groups), so that a copy's gradient is
        what the cells of its block ask of the rule; a component is kept where at least a fraction `agree_tau` of the
        blocks agree on its sign. The lessons that do not run on the grid (the cell's laws, pref among them) run on
        the rule itself and join every block's gradient. Leaves in .grad the kept part of the sum over blocks (the
        plain gradient), and returns the loss and the terms."""
        c, net = self.cfg, self.net
        assert c.task == "div" and not self.hyper and self.ncfg.rule == "mlp" and not net.nextra and not c.l1
        base = [net.W1, net.b1, net.W2, net.b2]
        assert len(base) == len(list(net.parameters()))
        self.copies = [p.detach()[None].repeat(c.agree_cols, *[1] * p.dim()).requires_grad_() for p in base]
        W = {n: v for n, v in c.lessons.items() if v and self.step >= c.lesson_after.get(n, 0)}
        on_grid = {n: v for n, v in W.items() if n in ("ans", "xmodn", "step", "link")}
        off_grid = {n: v for n, v in W.items() if n not in on_grid}
        net.groups = (self.copies, "cols", c.agree_cols)
        try:
            terms = self.div_terms(k, on_grid)
        finally:
            net.groups = None
        terms.update(self.div_terms(k, off_grid) if off_grid else {})
        loss = sum(W[n] * t for n, t in terms.items())
        for p in base:
            p.grad = None
        loss.backward()
        shared = [p.grad if p.grad is not None else torch.zeros_like(p) for p in base]
        self.and_mask([cp.grad + s for cp, s in zip(self.copies, shared)],
                      sums=[cp.grad.sum(0) + s for cp, s in zip(self.copies, shared)])
        self.copies = None
        return loss.detach(), {n: float(t.detach()) for n, t in terms.items()}

    def and_mask_grads_seq(self, k):
        """The AND-mask a group at a time: the reference and_mask_grads is checked against (and the way for lessons
        group_terms does not cover)."""
        groups = self.and_groups(k)
        params = list(self.net.parameters())
        per = [[] for _ in params]
        loss_tot, terms_tot = 0.0, {}
        for g in groups:
            self.opt.zero_grad()
            loss, terms = self.lessons(k, idx=g)
            loss.backward()
            for i, p in enumerate(params):
                per[i].append(p.grad.clone() if p.grad is not None else torch.zeros_like(p))
            loss_tot += float(loss.detach()) / len(groups)
            for nme, v in terms.items():
                terms_tot[nme] = terms_tot.get(nme, 0.0) + v / len(groups)
        self.and_mask([torch.stack(x) for x in per])
        return torch.tensor(loss_tot), terms_tot

    def dbd_step(self):
        """Delta-bar-delta (Jacobs 1988; the step-size idea Sutton's IDBD makes incremental): one speed per weight,
        raised by `dbd_up` where this step's gradient has the sign of the trace of past gradients and cut by a
        factor 1 - `dbd_down` where it does not; the optimizer's own step is scaled by it."""
        c = self.cfg
        params = [p for p in self.net.parameters() if p.requires_grad and p.grad is not None]
        if self.speed is None:
            self.speed = [torch.ones_like(p) for p in params]
            self.trace_g = [torch.zeros_like(p) for p in params]
        before = [p.detach().clone() for p in params]
        for i, p in enumerate(params):
            agree = (p.grad * self.trace_g[i]) > 0
            self.speed[i] = torch.where(agree, self.speed[i] + c.dbd_up, self.speed[i] * (1 - c.dbd_down))
            self.speed[i].clamp_(c.dbd_lo, c.dbd_hi)
            self.trace_g[i].mul_(0.9).add_(0.1 * p.grad)
        self.opt.step()
        with torch.no_grad():
            for i, p in enumerate(params):
                p.copy_(before[i] + self.speed[i] * (p - before[i]))

    # ----- curves -----
    @torch.no_grad()
    def curve_point(self, k, step):
        """One point of the training curve: snapped and soft training shares, the shares one size up ((B, w, k + 1)
        and (B, w + 1, k + 1), on a fixed sample), the rule's weight norm, the sub table, the plumbing probe."""
        c = self.cfg
        out = {"step": step, "train_hard": self.exact(c.B, c.w, k), "train_soft": self.exact(c.B, c.w, k, mode="soft")}
        for b, w_, k_ in ((c.B, c.w, k + 1), (c.B, c.w + 1, k + 1)):
            out[f"{b},{w_},{k_}"] = self.exact(b, w_, k_, sample=c.curve_sample)
        out["wnorm"] = round(float(torch.sqrt(sum((p.detach() ** 2).sum() for p in self.net.parameters()
                                                  if p.requires_grad))), 3)
        out["t"] = round(time.time() - self.t0, 1)
        if self.compiled:
            sr = self.sub_right()
            out["sub"] = sr["sub_digits_right"] + sr["sub_borrows_right"]
            out["plumbing"] = self.plumbing(k)
        if c.agree in ("and", "and_cols"):
            out["kept"] = round(self.kept, 4)
        if c.agree == "dbd" and self.speed is not None:
            sp = torch.cat([x.flatten() for x in self.speed])
            out["speed"] = [round(float(sp.min()), 3), round(float(sp.median()), 3), round(float(sp.max()), 3)]
        return out

    @torch.no_grad()
    def plumbing(self, k):
        """A read-only probe of the copies, on the hard path from blank cells at the training size: does the
        information arrive where the compiled machine reads it, in whatever code? For each, how well a lookup of
        the source cell's whole state (every channel's value) predicts the machine's value, fitted on half the
        questions and scored on the other half, as a lift over the most common value (0: nothing, 1: all of it):
            N's digit    at a SUB cell, from the cell above it (the compiled rule's U.n)
            old register at a SEL cell, from the cell above it (U.m, the minuend its link started from)
            final borrow at a SEL cell, from the cell to its right (R.g; at the top place, U.b)"""
        c = self.cfg
        lay = self.layout(c.B, c.w, k)
        data = self.dataset(c.B, c.w, k)
        S, _ = self.forward(lay, data.Nd, data.vd, mode="hard", ticks=lay.depth, train=False)
        st = np.stack([self.net.channel(S, n).argmax(1).cpu().numpy() for n, _ in self.fmt.chans], -1)  # [X,R,C,ch]
        ref = G.machine_reference(lay, data)
        X, R, C = data.X, lay.R, lay.C
        key = lambda a: np.ravel_multi_index(np.moveaxis(a, -1, 0), [sz for _, sz in self.fmt.chans])
        half = np.arange(X) % 2 == 0
        cases = {"n_digit": [], "old_register": [], "final_borrow": []}
        for r in range(1, R):
            kd = lay.kind[r, 0]
            for cc in range(C):
                if kd == SUB:
                    cases["n_digit"].append((key(st[:, r - 1, cc]), ref["n"][:, r - 1, cc]))
                elif kd == SEL:
                    cases["old_register"].append((key(st[:, r - 1, cc]), ref["d"][:, r - 2, cc]))
                    src = st[:, r - 1, cc] if cc == C - 1 else st[:, r, cc + 1]
                    cases["final_borrow"].append((key(src), ref["g"][:, r, cc]))
        out = {}
        for name, pairs in cases.items():
            xs = np.concatenate([p[0] for p in pairs])
            ys = np.concatenate([p[1] for p in pairs])
            tr = np.concatenate([half] * len(pairs))
            keys, inv = np.unique(xs, return_inverse=True)
            counts = np.zeros((len(keys), int(ys.max()) + 1), dtype=np.int64)    # the lookup: per state, its values
            np.add.at(counts, (inv[tr], ys[tr]), 1)
            major = np.bincount(ys[tr]).argmax()
            pred = np.where(counts.sum(1) > 0, counts.argmax(1), major)[inv[~tr]]
            acc = float((pred == ys[~tr]).mean())
            base = float((ys[~tr] == major).mean())
            out[name] = round((acc - base) / max(1e-9, 1 - base), 3)
        return out

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
        a = t.detach().to(torch.float32).contiguous().cpu().numpy()
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
        sd[name] = torch.as_tensor(np.frombuffer(base64.b64decode(b), dtype=np.float32).reshape(shape).copy()).to(r.dtype)
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
            got = Z[:, r.fmt.sl(ch)].argmax(1).cpu().numpy()
            out["tf"][ch] = np.round((got == ref[ch]).mean(0), 3).tolist()
    S, _ = r.forward(lay, data.Nd, data.vd, mode="hard", ticks=lay.depth, train=False)
    got = r.net.channel(S, "d").argmax(1).cpu().numpy()
    out["free_d"] = np.round((got == ref["d"]).mean(0), 3).tolist()
    return out


def diag(cfg: RunCfg, sizes) -> dict:
    """Train, then map where the rule breaks at each size (`situations`)."""
    r = Run(cfg)
    out = r.train()
    out["cfg"] = asdict(cfg)
    out["situations"] = {",".join(map(str, s)): situations(r, *s) for s in sizes}
    return out
