"""The compiled machine as a probe of a trained handwired network (the attention bridge's probes, `bridge/probe.py`,
for routing that is a matrix of weights instead of attention).

    ports(run)   every port's strongest source per cell against the compiled source, classified: right; "input" (a
                 prompt cell the machine does not read: a bypass to clean data); "computed" (another computed cell);
                 "zero" (a constant zero)
    grad0(run)   at the run's current parameters, which source each port's answer gradient pushes up most, per cell
    path(run)    the answer loss on the straight line from the trained routing to the compiled one: falling all the way
                 (a slow road) or rising first (a separate valley)
    cells(run)   every scratch row's digits (and the SUB rows' borrows) against the machine's own values
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from bridge import grid as G
from bridge import net as BN
from bridge.probe import classify, cells  # noqa: F401  (cells works unchanged on a digits-format network)


def _lay(run):
    c = run.cfg
    return run.layout(c.B, c.w, c.k)


@torch.no_grad()
def ports(run, R=None):
    """{port.kind: {class: share of that kind's cells}} for the strongest source of every port."""
    lay = _lay(run)
    net = run.net
    if R is None:
        R = net.routing(lay, net.index(lay), net.blocks[0], True)
    out = {}
    for (h, kd), opt in BN.TRUE_MENU.items():
        if opt == "zero":
            continue
        cells_ = np.where(lay.kind == kd)[0]
        src = R[h, cells_].argmax(-1).numpy()
        cls = [classify(lay, s, lay.comp[h, p]) for s, p in zip(src, cells_)]
        out[f"{G.HEADS[h]}.{G.KINDS[kd]}"] = {k: cls.count(k) / len(cls) for k in ("right", "input", "computed", "zero")
                                               if cls.count(k)}
    return out


def share(d, cls):
    """The mean share of class `cls` over the ports of a ports()/grad0() result."""
    return float(np.mean([v.get(cls, 0.0) for v in d.values()]))


def grad0(run):
    """For every port and cell, the source whose weight the answer loss most wants to raise, classified."""
    c = run.cfg
    lay = _lay(run)
    net = run.net
    data = run.dataset(c.B, c.w, c.k)
    ix = net.index(lay)
    R = net.routing(lay, ix, net.blocks[0], False).detach().requires_grad_(True)
    orig = net.routing
    net.routing = lambda *a, **k: R                                                 # the same weights every tick
    try:
        p = run.f(lay, run.hot(data.Nd), run.hot(data.vd))
        loss = -p.clamp_min(1e-12).log().gather(-1, data.yd[..., None]).mean()
        g, = torch.autograd.grad(loss, R)
    finally:
        net.routing = orig
    push = -g
    if net.cfg.route == "dense":
        push = push.masked_fill(~ix.ok[None], -float("inf"))
    elif net.cfg.route == "menu":
        allowed = torch.zeros_like(push, dtype=torch.bool)                          # only the menu's sources
        allowed.scatter_(2, ix.menu_src.T[None].expand(3, lay.L, -1), True)
        push = push.masked_fill(~allowed, -float("inf"))
    return ports(run, R=F.one_hot(push.argmax(-1), push.shape[-1]).to(push.dtype))


def compiled_copy(run, beta=30.0):
    """The run's network with its routing set to the compiled wiring (menus: the true option; dense: the compiled
    source) and its learned tables to the truth, logits and gates at scale `beta` (relu-type gates: 1 on the true
    source, 0 elsewhere)."""
    net = copy.deepcopy(run.net)
    cfg = net.cfg
    with torch.no_grad():
        for blk in net.blocks:
            if cfg.route == "menu":
                opts = BN.menu_options(cfg)
                logit = cfg.mux in ("softmax", "sigmoid_free", "softplus")
                g = torch.full_like(blk.menu_g, -beta if logit else 0.0)            # [3, copies, NK, M]
                for (h, kd), opt in BN.TRUE_MENU.items():
                    g[h, :, kd, opts.index(opt)] = beta if logit else 1.0
                blk.menu_g.copy_(g)
            elif cfg.route == "dense":
                logit = cfg.mux in ("softmax", "sigmoid_free", "softplus")
                g = torch.full_like(blk.menu_dense, -beta if logit else 0.0)
                c = run.cfg
                for B_, w_, k_ in ((c.B, c.w, c.k), (c.B, c.w, c.k + 1), (c.B, c.w + 1, c.k + 1)):
                    lay = run.layout(B_, w_, k_)                        # the bins of the bigger sizes too
                    ix = net.index(lay)
                    for h in range(3):
                        for p in np.where(lay.comp[h] >= 0)[0]:
                            j = lay.comp[h, p]
                            g[h, :, ix.qk[p, j], ix.kk[p, j], ix.br[p, j], ix.bc[p, j], ix.top[p, j]] = (
                                beta if logit else 1.0)
                blk.menu_dense.copy_(g)
            sub = blk.cells["SUB"]
            if getattr(sub, "learn", "") == "table" and sub.table_simplex:
                T = G.truth_tables(run.cfg.B, run.dtype)[1]
                # the joint units' order: read the compiled circuit's output rows
                WL = getattr(sub, f"cWL{sub.nlayers - 1}")
                lo, hi = sub.joint
                rows = WL[lo:hi]                                                     # [joint, B + 2] 0/1
                sub.p["Ld"].copy_(beta * rows[:, :run.cfg.B])
                sub.p["Lb"].copy_(beta * rows[:, run.cfg.B:])
    return net


@torch.no_grad()
def path(run, points=11):
    """(lambda, answer loss per digit, training share exact on the hard path) along the line from the trained
    parameters (lambda 0) to the compiled ones (lambda 1)."""
    c = run.cfg
    comp = compiled_copy(run)
    trained = {n: p.detach().clone() for n, p in run.net.named_parameters()}
    target = dict(comp.named_parameters())
    lay = _lay(run)
    data = run.dataset(c.B, c.w, c.k)
    out = []
    for lam in np.linspace(0, 1, points):
        for n, p in run.net.named_parameters():
            p.copy_((1 - lam) * trained[n] + lam * target[n].detach())
        ans = run.f(lay, run.hot(data.Nd), run.hot(data.vd))
        loss = float(-ans.clamp_min(1e-12).log().gather(-1, data.yd[..., None]).mean())
        ex = float((run.f(lay, run.hot(data.Nd), run.hot(data.vd), hard=True).argmax(-1) == data.yd).all(-1)
                   .double().mean())
        out.append((float(lam), loss, ex))
    for n, p in run.net.named_parameters():
        p.copy_(trained[n])
    return out


@torch.no_grad()
def irrelevant(run, trials=8, seed=0):
    """For a freed subtraction cell: on every one-hot (a, b, c), the share of random fillings of the entries the
    compiled cell ignores (A's and B's borrows, C's digit) on which the snapped output (digit and borrow) changes."""
    B = run.cfg.B
    S = B + 2
    cell = run.net.blocks[0].cells["SUB"]
    a, m, cc = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
    a, m, cc = a.flatten(), m.flatten(), cc.flatten()
    n = len(a)
    hot = lambda x, k: F.one_hot(x, k).to(run.dtype)
    base = lambda ab, bb, cd: torch.cat([hot(a, B), ab, hot(m, B), bb, cd, hot(cc, 2)], -1)
    ref = cell(base(hot(torch.zeros(n, dtype=torch.long), 2), hot(torch.zeros(n, dtype=torch.long), 2),
                    hot(torch.zeros(n, dtype=torch.long), B)))
    g = torch.Generator().manual_seed(seed)
    changed = 0
    for _ in range(trials):
        out = cell(base(hot(torch.randint(2, (n,), generator=g), 2), hot(torch.randint(2, (n,), generator=g), 2),
                        hot(torch.randint(B, (n,), generator=g), B)))
        changed += int(((out[:, :B].argmax(-1) != ref[:, :B].argmax(-1)) | (out[:, B:].argmax(-1) != ref[:, B:]
                                                                            .argmax(-1))).sum())
    return changed / (trials * n)


@torch.no_grad()
def relabel(run, hard=True):
    """Do the scratch cells hold the machine's digits under a renaming? Per row group: the share of (question, cell)
    pairs whose digit, mapped by the best digit -> digit map for that group (fitted on these questions), equals the
    machine's digit there; and the share without renaming. 1.0 / < 1.0: the cells carry the machine's digits in
    another code (a digit-meaning drift); both 1.0: the machine's own code."""
    c = run.cfg
    lay = _lay(run)
    data = run.dataset(c.B, c.w, c.k)
    net = run.net
    S = net(lay, net.inputs(lay, run.hot(data.Nd), run.hot(data.vd)), hard=hard)
    dig, _ = G.machine_reference(lay, data)
    out = {}
    for r, (name, t, s_) in enumerate(lay.rows):
        if name not in ("SHIFT", "SUB", "SEL"):
            continue
        ps = [lay.cell(r, j) for j in range(lay.P)]
        sym = S[:, ps, :c.B].argmax(-1).numpy().ravel()
        d = dig[:, ps].ravel()
        right = sum(np.bincount(d[sym == k_], minlength=c.B).max() for k_ in range(c.B) if (sym == k_).any())
        out[f"{name}({t},{s_})" if name != "SHIFT" else f"SHIFT({t})"] = (float(right / len(d)), float((sym == d).mean()))
    return out
