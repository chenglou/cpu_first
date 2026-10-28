"""Training the handwired network (`hnet.py`) from random at each notch, with the attention bridge's lessons.

This reuses `bridge.train.Run` whole: the data, the lessons (answers, x mod N = x, the step law, the one-link law on a
sub-grid, the subtraction cell's order facts and bijectivity), the hard-path measure, the stopping rule and the
certification on two bigger sizes, (B, w, k + 1) and (B, w + 1, k + 1). Only the network changes, plus three things a
raw affine-plus-ReGLU network needs:
- **reading a state as a distribution.** The attention bridge's states are distributions by construction. Here they
  are raw values, so the lessons read a digit group as a distribution (`readout` "relu": clamp at 0 and normalize),
  or compare raw values with the exact one-hot the compiled program computes (`readout` "mse");
- **sparsity pressure** (`l1`, joining at step `l1_after`) on the freed routing and on dense cells;
- **projection** (`project_every`): every that many steps, the learned parts are snapped onto the nearest exact
  program (tables to one-hot rows, each menu port to its strongest option, dense routing to its strongest source per
  feature) and scaled to `project_beta`, and training continues from there.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from bridge import grid as G
from bridge import net as BN
from bridge import train as BT
from bridge.grid import SCRATCH, SEL, SHIFT, SUB, Z

from handwired_bridge import hnet as H


@dataclass
class HRunCfg(BT.RunCfg):
    readout: str = "relu"          # "relu" (clamp and normalize, then the bridge's cross-entropy) | "softmax"
                                   # (softmax(wb_beta * values), as the "softmax" write-back) | "mse"
    l1: float = 0.0
    l1_after: int = 0
    project_every: int = 0
    project_beta: float = 1.0
    eps: float = 1e-9


def dist(x, eps=1e-9):
    """A digit group's raw values as a distribution: clamped at 0 and normalized (uniform when all are 0)."""
    x = x.clamp_min(0) + eps
    return x / x.sum(-1, keepdim=True)


class HNetAdapter(H.HNet):
    """The handwired network with the interface `bridge.train.Run` expects: answers and the subtraction cell's
    outputs read as distributions (as the soft write-back reads them)."""

    def as_dist(self, x):
        if self.cfg.writeback == "softmax":
            return (self.cfg.wb_beta * x).softmax(-1)
        return dist(x)

    def answer(self, lay, Np, vp, hard=False):
        return self.as_dist(self.answer_raw(lay, Np, vp, hard))

    def sub_rule(self, a, b, c, hard=False):
        d, bo = self.kind_rule(SUB, a, b, c, hard)
        return self.as_dist(d), self.as_dist(bo)


class Run(BT.Run):
    def __init__(self, cfg: HRunCfg):
        self.cfg = c = cfg
        torch.manual_seed(c.seed)
        self.dtype = getattr(torch, c.dtype)
        netc = dict(c.net)
        if not netc.get("tied", True) and not netc.get("nblocks"):
            netc["nblocks"] = G.Layout(c.B, c.w, c.k).depth
        self.ncfg = H.HCfg(**netc)
        self.ncfg.fmt, self.ncfg.roles, self.ncfg.attn = "digits", "kinds", self.ncfg.route
        self.net = HNetAdapter(self.ncfg, c.B, seed=c.seed, dtype=self.dtype)
        self.lay, self.data = {}, {}
        if c.init == "compiled":
            self.compile_init()
        params = [p for p in self.net.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=c.lr) if params else None
        self.nparams = sum(p.numel() for p in params)
        self.t0 = time.time()
        self.log = []
        self.rng = np.random.default_rng(1000 + c.seed)
        self.step = 0

    # ----- the digit losses on raw values -----
    def raw(self, lay, Np, vp, hard=False):
        return self.net.answer_raw(lay, Np, vp, hard)

    def as_dist(self, x):
        """A raw digit group read as a distribution, the way the network's soft write-back reads it."""
        if self.cfg.readout == "softmax":
            return (self.ncfg.wb_beta * x).softmax(-1)
        return dist(x, self.cfg.eps)

    def dloss(self, x, y):
        """Raw digit groups x [X, n, B] against target digits y [X, n]."""
        if self.cfg.readout == "mse":
            return ((x - F.one_hot(y, x.shape[-1]).to(x.dtype)) ** 2).sum(-1).mean()
        return BT.nll_digits(self.as_dist(x), y)

    def dagree(self, x, y):
        if self.cfg.readout == "mse":
            return ((x - y) ** 2).sum(-1).mean()
        return BT.agree(self.as_dist(x), self.as_dist(y))

    def lessons(self, k, links=None):
        """The bridge's lessons (`bridge.train.Run.lessons`) on raw values."""
        c = self.cfg
        B, w = c.B, c.w
        lay = self.layout(B, w, k, links)
        data = self.dataset(B, w, k, links=links)
        if c.batch and data.X > c.batch:
            data = data.subset(self.rng.choice(data.X, c.batch, replace=False))
        W = c.lessons
        terms = {}
        Nh = self.hot(data.Nd)
        if W.get("ans", 0):
            terms["ans"] = self.dloss(self.raw(lay, Nh, self.hot(data.vd)), data.yd)
        if W.get("xmodn", 0):
            pairs = np.unique(np.stack([data.N, data.R0], 1), axis=0)
            Nn_ = np.concatenate([pairs[:, 0], np.unique(data.N)])
            vv = np.concatenate([pairs[:, 1], np.unique(data.N)])
            Nd = torch.from_numpy(G.digits(Nn_, B, w))
            terms["xmodn"] = self.dloss(self.raw(lay, self.hot(Nd), self.vdist(lay, vv)),
                                        torch.from_numpy(G.digits(vv % Nn_, B, w)))
        if W.get("step", 0) and k >= 2:
            p_full = self.raw(lay, Nh, self.hot(data.vd))
            r1 = self.as_dist(self.raw(lay, Nh, self.vdist(lay, data.v // B)))        # f(N, v // B), soft
            v2 = torch.zeros(data.X, w + k, B, dtype=self.dtype)
            v2[:, 0] = self.hot(data.vd[:, 0])
            v2[:, 1:w + 1] = r1
            v2[:, w + 1:, 0] = 1
            terms["step"] = self.dagree(p_full, self.raw(lay, Nh, v2))
        if W.get("link", 0):
            terms["link"] = self.link(data)
        for name in ("pref", "colall", "rowall", "bij"):
            if W.get(name, 0):                                  # untied blocks: the lesson on every block's cell
                tot = 0.0
                for b in range(len(self.net.blocks)):
                    self.net.rule_block = b
                    tot = tot + (getattr(self, name)() if name != "pref" else self.pref(data))
                self.net.rule_block = 0
                terms[name] = tot / len(self.net.blocks)
        if c.l1 and self.step >= c.l1_after:
            terms["l1"] = c.l1 * self.net.l1()
        loss = sum(W.get(n, 1.0) * t for n, t in terms.items() if self.step >= c.lesson_after.get(n, 0))
        return loss, {n: float(t.detach()) for n, t in terms.items()}

    def link(self, data):
        """One link alone on x = R0 (below N) and x = N: the register must become x mod N, and the subtraction's
        final borrow must be [x < N]."""
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
        loss = self.dloss(S[:, cells, :B], torch.from_numpy(G.digits(xs % Ns, B, P)))
        top = S[:, lay1.cell(last - 1, P - 1), B:]
        lt = torch.from_numpy((xs < Ns).astype(np.int64))
        return loss + self.dloss(top[:, None], lt[:, None])

    # ----- a start at the compiled machine (the Net2Net arm's stability test) -----
    @torch.no_grad()
    def compile_init(self):
        """Every learned part set to the compiled machine, logits and gates at scale `init_beta`, then Gaussian noise
        of std `init_noise` on every learned tensor. With `copies` > 1 the compiled circuit is the Net2Net-widened one
        (every unit duplicated, weights into the copies split), and the noise lets the copies diverge."""
        from handwired_bridge import probe as HP
        c = self.cfg
        g = torch.Generator().manual_seed(7000 + c.seed)
        beta, sig = c.init_beta, c.init_noise
        comp = HP.compiled_copy(self, beta=beta)
        for (n, p), (n2, q) in zip(self.net.named_parameters(), comp.named_parameters()):
            assert n == n2
            p.copy_(q)
        for blk in self.net.blocks:
            for cell in blk.cells.values():
                if getattr(cell, "learn", "") == "gates" and cell.lift:
                    for i in range(cell.nlayers):
                        for name in ("WL", "WG"):
                            cell.p[f"{name}p{i}"].copy_(getattr(cell, f"c{name}p{i}"))
                            cell.p[f"{name}r{i}"].copy_(getattr(cell, f"c{name}r{i}"))
                        cell.p[f"bL{i}"].copy_(getattr(cell, f"cbLl{i}"))
                        cell.p[f"bG{i}"].copy_(getattr(cell, f"cbG{i}"))
                elif getattr(cell, "learn", "") == "gates":
                    for i in range(cell.nlayers):
                        for name in ("WL", "WG", "bL", "bG"):
                            cell.p[f"{name}{i}"].copy_(getattr(cell, f"c{name}{i}"))
        for p in self.net.parameters():
            p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype) * sig)

    # ----- projection: snap the learned parts onto the nearest exact program -----
    @torch.no_grad()
    def project(self):
        beta = self.cfg.project_beta
        for blk in self.net.blocks:
            for cell in blk.cells.values():
                if getattr(cell, "learn", "") == "table" and cell.table_simplex:
                    for name, n in (("Ld", self.cfg.B), ("Lb", 2)):
                        L = cell.p[name]
                        L.copy_(beta * F.one_hot(L.argmax(-1), n).to(L.dtype))
                if getattr(cell, "learn", "") == "gates":                     # weights to the nearest integer
                    for prm in cell.p.values():
                        prm.copy_(prm.round())
            if self.ncfg.route == "menu":
                g = blk.menu_g
                hot = F.one_hot(g.argmax(-1), g.shape[-1]).to(g.dtype)
                g.copy_(beta * hot if self.ncfg.mux not in ("softmax", "sigmoid_free", "softplus") else beta * (2 * hot - 1))

    # ----- measures -----
    @torch.no_grad()
    def inspect(self):
        c, nc = self.cfg, self.ncfg
        out = {}
        B = c.B
        Td, Tb = self.rule_table(hard=True)
        T = G.truth_tables(B, self.dtype)
        out["sub_digits_right"] = int((Td.argmax(-1) == T[1, ..., :B].argmax(-1)).sum())
        out["sub_borrows_right"] = int((Tb.argmax(-1) == T[1, ..., B:].argmax(-1)).sum())
        out["sub_entries"] = 2 * B * B
        a, m, cc = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
        a, m, cc = a.flatten(), m.flatten(), cc.flatten()
        for i, kd in enumerate(SCRATCH):
            if kd == SUB:
                continue
            d, _ = self.net.kind_rule(kd, self.hot(a), self.hot(m), G.onehot(cc, 2, self.dtype), hard=True)
            ok = d.argmax(-1) == T[i, a, m, cc, :B].argmax(-1)
            if kd == SHIFT:
                ok = ok[(m == 0) & (cc == 0)]
            out[f"{G.KINDS[kd].lower()}_right"] = f"{int(ok.sum())}/{len(ok)}"
        if nc.route != "fixed":
            lay = self.layout(c.B, c.w, c.k)
            ix = self.net.index(lay)
            R = self.net.routing(lay, ix, self.net.blocks[0], True)
            right = {}
            for (h, kd), opt in BN.TRUE_MENU.items():
                if opt == "zero":
                    continue
                cells = np.where(lay.kind == kd)[0]
                src = R[h, cells].argmax(-1).numpy()
                comp = lay.comp[h, cells]
                ok = (src == comp) | ((lay.kind[src] == Z) & (lay.kind[comp] == Z))
                right[f"{G.HEADS[h]}.{G.KINDS[kd]}"] = float(ok.mean())
            out["heads_right"] = right
        return out

    def train(self):
        if self.opt is None:                                                    # nothing to learn: just measure
            c = self.cfg
            rec = {"steps": 0, "hit": 0, "seconds": 0.0, "params": 0, "stages": []}
            rec["train_hard"] = self.exact(c.B, c.w, c.k)
            rec["train_soft"] = self.exact(c.B, c.w, c.k, hard=False)
            return rec
        if self.cfg.project_every:
            step_fn = self.opt.step
            every = self.cfg.project_every

            def stepped(*a, **kw):
                r = step_fn(*a, **kw)
                if (self.step + 1) % every == 0:
                    self.project()
                return r
            self.opt.step = stepped
        return super().train()


def run(cfg: HRunCfg) -> dict:
    r = Run(cfg)
    out = r.train()
    out["cfg"] = asdict(cfg)
    return out
