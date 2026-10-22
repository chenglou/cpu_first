"""Step 3: the compiled machine as a probe of a trained network.

    heads(run)    every head's argmax source per cell against the compiled source, classified:
                  right; "input" (the cell reads an input cell the machine does not: a bypass to clean data);
                  "computed" (another computed cell); "zero" (a constant zero)
    cells(run)    every scratch cell's state on the training questions against the machine's own value there
                  (grid.machine_reference), by row kind and link: which parts compute what the machine computes
    path(run)     the loss on the straight line from the trained parameters to the compiled ones: does it fall
                  monotonically (the optimizer just did not take the road) or rise first (a separate basin)?
    grad0(run)    at the run's starting point, which key each head's answer gradient pushes up most, per cell:
                  the compiled source, an input (bypass), or something else
    symmetry(run) for the free formats: do heads, kinds or cells compute the same thing (identical copies)?
These apply where the network still has the compiled grid's cells (roles "kinds"); the attention probes where the
attention is menu or free.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from bridge import compile as Cm
from bridge import grid as G
from bridge import net as Nn
from bridge.grid import NIN, SCRATCH, SEL, SHIFT, SUB, VIN, Z


def _lay(run):
    c = run.cfg
    return run.layout(c.B, c.w, c.k)


def classify(lay, src, comp):
    if src == comp or (lay.kind[src] == Z and lay.kind[comp] == Z):
        return "right"
    if lay.kind[src] == Z:
        return "zero"
    if lay.kind[src] in (NIN, VIN):
        return "input"
    return "computed"


@torch.no_grad()
def heads(run, A=None):
    """{head.kind: {class: share of that kind's cells}} for the argmax attention."""
    lay = _lay(run)
    net = run.net
    if A is None:
        A = net.attention(lay, net.index(lay), net.blocks[0], None, True)
    if A.shape[0] > 3:                                                             # head copies: their mean
        A = A.view(3, -1, *A.shape[1:]).mean(1)
    out = {}
    for (h, kd), opt in Nn.TRUE_MENU.items():
        if opt == "zero":
            continue
        cells = np.where(lay.kind == kd)[0]
        src = A[h, cells].argmax(-1).numpy()
        cls = [classify(lay, s, lay.comp[h, p]) for s, p in zip(src, cells)]
        out[f"{G.HEADS[h]}.{G.KINDS[kd]}"] = {k: cls.count(k) / len(cls) for k in ("right", "input", "computed",
                                                                                    "zero") if cls.count(k)}
    return out


@torch.no_grad()
def cells(run, hard=True):
    """Share of training questions on which each group of scratch cells holds the machine's digit (and, for SUB
    cells, its borrow): {row label: (digit share, borrow share)}."""
    c = run.cfg
    lay = _lay(run)
    data = run.dataset(c.B, c.w, c.k)
    net = run.net
    S = net(lay, net.inputs(lay, run.hot(data.Nd), run.hot(data.vd)), hard=hard)
    dig, bor = G.machine_reference(lay, data)
    B = c.B
    out = {}
    for r, (name, t, s) in enumerate(lay.rows):
        if name not in ("SHIFT", "SUB", "SEL"):
            continue
        ps = [lay.cell(r, j) for j in range(lay.P)]
        if net.cfg.fmt == "digits":
            d = (S[:, ps, :B].argmax(-1).numpy() == dig[:, ps]).mean()
            b = (S[:, ps, B:].argmax(-1).numpy() == bor[:, ps]).mean() if name == "SUB" else None
        else:
            d, b = None, None
        out[f"{name}({t},{s})" if name != "SHIFT" else f"SHIFT({t})"] = (float(d) if d is not None else None,
                                                                        None if b is None else float(b))
    return out


def compiled_copy(run):
    """A copy of the run's network with every attention parameter set to the compiled machine (and, where the rule
    is learned, the rule too)."""
    net = copy.deepcopy(run.net)
    cfg = net.cfg
    if cfg.attn == "menu":
        Cm.set_menu(net)
    elif cfg.attn == "free":
        Cm.set_free(net)
    if cfg.rule == "mlp" and cfg.fmt == "digits":
        Cm.set_mlp(net)
    elif cfg.rule == "table":
        with torch.no_grad():
            T = G.truth_tables(run.cfg.B, run.dtype)
            for i, kd in enumerate(SCRATCH):
                name = G.KINDS[kd]
                if name in net.blocks[0].Ld:
                    net.blocks[0].Ld[name].copy_(30 * T[i, ..., :run.cfg.B])
                    net.blocks[0].Lb[name].copy_(30 * T[i, ..., run.cfg.B:])
    return net


@torch.no_grad()
def path(run, points=11, lessons=("ans",)):
    """Loss and exact share at theta(l) = (1 - l) trained + l compiled, l = 0 .. 1 (attention and rule parameters;
    for the menu the logits, for free attention the scores)."""
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
        ex = float((run.f(lay, run.hot(data.Nd), run.hot(data.vd), hard=True).argmax(-1) == data.yd).all(-1).double().mean())
        out.append((float(lam), loss, ex))
    for n, p in run.net.named_parameters():
        p.copy_(trained[n])
    return out


def grad0(run):
    """At the run's current parameters: for every head and cell, the key whose attention weight the answer loss
    most wants to raise (largest negative gradient of the loss with respect to the attention weight), classified.
    Returns {head.kind: {class: share}}."""
    c = run.cfg
    lay = _lay(run)
    net = run.net
    data = run.dataset(c.B, c.w, c.k)
    ix = net.index(lay)
    blk = net.blocks[0]
    A = net.attention(lay, ix, blk, None, False).detach().requires_grad_(True)
    orig = net.attention
    net.attention = lambda *a, **k: A                                              # the same weights every tick
    try:
        p = run.f(lay, run.hot(data.Nd), run.hot(data.vd))
        loss = -p.clamp_min(1e-12).log().gather(-1, data.yd[..., None]).mean()
        g, = torch.autograd.grad(loss, A)
    finally:
        net.attention = orig
    push = -g                                                                      # what descent would raise
    if net.cfg.causal and net.cfg.attn in ("free", "abs"):
        push = push.masked_fill(torch.isinf(ix.causal)[None], -float("inf"))      # keys the mask rules out
    return heads(run, A=torch.nn.functional.one_hot(push.argmax(-1), push.shape[-1]).to(push.dtype))


@torch.no_grad()
def symmetry(run):
    """How alike the three heads' patterns are, and how alike the scratch kinds' rules are (identical copies):
    mean total-variation distance between heads' attention rows (0 = identical)."""
    lay = _lay(run)
    net = run.net
    A = net.port_attention(lay, net.index(lay), net.blocks[0], None, False)
    if A.dim() == 4:
        A = A.mean(1)
    scr = ~lay.is_input
    tv = lambda a, b: 0.5 * (a - b).abs().sum(-1).mean().item()
    return {"A~B": tv(A[0, scr], A[1, scr]), "A~C": tv(A[0, scr], A[2, scr]), "B~C": tv(A[1, scr], A[2, scr])}


@torch.no_grad()
def relabel(run, hard=True):
    """For formats whose symbols have learned meanings: per row group, the share of (question, cell) pairs whose
    argmax symbol, mapped by the best symbol -> digit map for that group (fitted on these questions), equals the
    machine's digit there. 1.0: the cells hold the machine's digits under a consistent renaming."""
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
        sym = S[:, ps].argmax(-1).numpy().ravel()
        d = dig[:, ps].ravel()
        K = S.shape[-1]
        right = 0
        for k_ in range(K):
            m = sym == k_
            if m.any():
                right += np.bincount(d[m], minlength=c.B).max()
        out[f"{name}({t},{s_})" if name != "SHIFT" else f"SHIFT({t})"] = float(right / len(d))
    return out


@torch.no_grad()
def hold(run):
    """With the prompt only at the first tick: the share of prompt cells whose port A reads the cell itself (how they
    hold), and the share of prompt digits still right after the last tick on the hard path."""
    c = run.cfg
    lay = _lay(run)
    net = run.net
    ix = net.index(lay)
    A = net.port_attention(lay, ix, net.blocks[0], None, True)
    inp = np.where(lay.is_input)[0]
    selfread = float((A[0, inp].argmax(-1).numpy() == inp).mean())
    data = run.dataset(c.B, c.w, c.k)
    S = net(lay, net.inputs(lay, run.hot(data.Nd), run.hot(data.vd)), hard=True)
    dig, _ = G.machine_reference(lay, data)
    cells = np.where(np.isin(lay.kind, (NIN, VIN)))[0]
    held = float((S[:, cells, :c.B].argmax(-1).numpy() == dig[:, cells]).mean())
    return {"prompt_reads_itself": selfread, "prompt_digits_held": held}
