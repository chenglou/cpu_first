"""A looped transformer over the grid of `grid.py`, loosened notch by notch.

One block, applied `ticks` times to every cell at once:
    heads    for each head h (A, B, C): an attention pattern over all cells; the head copies (a value projection
             of) the attended cells' states into the cell's port h
    rule     an MLP from (the cell's kind, its three ports) to the cell's new state
    inputs   the input cells are re-injected after every tick (input injection)
The state is overwritten each tick (no residual), except in the "free" format, which is a standard residual block.

The knobs (`Cfg`), from the compiled machine to a plain looped transformer:
    attn   "fixed"  the compiled patterns (grid.Layout.comp)
           "menu"   per head and query kind, a softmax mixture over a menu of relative rules (TINYROUTING's tied soft
                    switches): MENU's 7 rules ("wide"), NARROW's 1-3 per port ("narrow"), or MENU plus up to 13 more
                    relative offsets ("rel", `menu_extra`)
           "free"   per head, softmax over every cell of a learned score: a 2-D relative bias per (query kind, key kind,
                    row offset, column offset) plus a top-place term ("joint", a Swin-style relative bias table per
                    kind pair), or separate additive terms ("factor"); a 2-D causal mask (`causal`, raster order) and
                    an optional locality window (`window`)
           "abs"    content QK attention with learned absolute row and column embeddings (a plain transformer's)
    rule   "table"  the rule is a table over (A's digit, B's digit, C's borrow), applied multilinearly: the machine's
                    learned table. `learn` says which kinds' tables are learned; the others stay compiled
           "mlptab" the same, but each learned table is a ReLU MLP's outputs on every one-hot (kind, a, b, c)
           "mlp"    a ReLU MLP on [kind one-hot, the ports] -> logits, applied to the ports directly; one MLP for every
                    kind, or for `mlp_kinds` only (the others compiled)
    fmt    "digits"  a cell holds a digit distribution and a borrow distribution (the compiled meaning)
           "symbols" a cell holds a distribution over K symbols whose meaning is learned: the inputs are embedded,
                    the answer is read out, both learned; heads have value projections; the rule sees whole ports
           "free"   a cell holds a vector in R^d (LayerNorm residual block), learned embeddings and readout
    roles  "kinds"  scratch cells know their kind (SHIFT, SUB, SEL)
           "none"   every scratch cell is the same kind
    tied   one block for every tick, or one per tick (`nblocks`)
    st     straight-through states: the forward pass snaps every state, the backward pass is the soft one's
`Net.force` sets one menu port hard to one option (train.py's whole-option scoring).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bridge import grid as G
from bridge.grid import NIN, SCRATCH, SEL, SHIFT, SUB, VIN, Z

NK = len(G.KINDS)

# menu options: relative rules a head can follow (the source cell for a query cell at (r, c))
MENU = ("up", "up2", "left", "upleft", "nrow", "uptop", "zero")
# the "rel" menu: the 7 rules plus every other relative offset (row -3..0, column -2..2, causal), kind-agnostic
REL = tuple(f"rel{dr},{dc}" for dr in (-3, -2, -1, 0) for dc in (-2, -1, 0, 1, 2)
            if (dr, dc) not in ((-1, 0), (-2, 0), (0, -1), (-1, -1)) and not (dr == 0 and dc >= 0))
MENU_REL = MENU + REL


def menu_options(cfg) -> tuple:
    if cfg.menu != "rel":
        return MENU
    extra = REL if cfg.menu_extra < 0 else tuple(np.random.default_rng(7).permutation(REL)[:cfg.menu_extra])
    return MENU + extra


def menu_source(lay: G.Layout, p: int, opt: str) -> int:
    r, c = lay.row[p], lay.col[p]
    if opt == "up":
        return lay.cell(r - 1, c)
    if opt == "up2":
        return lay.cell(r - 2, c)
    if opt == "left":
        return lay.cell(r, c - 1)
    if opt == "upleft":
        return lay.cell(r - 1, c - 1)
    if opt == "nrow":
        return lay.cell(0, c)
    if opt == "uptop":
        return lay.cell(r - 1, lay.P - 1)
    if opt == "zero":
        return 0                                                          # the N row's column -1: a constant zero
    if opt.startswith("rel"):
        dr, dc = (int(x) for x in opt[3:].split(","))
        rr, cc = r + dr, c + dc
        if 0 <= rr < lay.Rn and -1 <= cc < lay.P:
            return lay.cell(rr, cc)
        return 0
    raise ValueError(opt)


# the compiled option of each (head, scratch kind); a head a kind does not use reads a constant zero
TRUE_MENU = {
    (0, SHIFT): "upleft", (1, SHIFT): "zero", (2, SHIFT): "zero",
    (0, SUB): "up", (1, SUB): "nrow", (2, SUB): "left",
    (0, SEL): "up2", (1, SEL): "up", (2, SEL): "uptop",
}

# a narrow menu, close to TINYROUTING's switches: a few semantic options per port (the carry route and SHIFT's
# unused ports fixed)
NARROW = {
    (0, SHIFT): ("upleft", "up", "left"), (1, SHIFT): ("zero",), (2, SHIFT): ("zero",),
    (0, SUB): ("up", "up2", "nrow"), (1, SUB): ("nrow", "up", "up2"), (2, SUB): ("left",),
    (0, SEL): ("up2", "up"), (1, SEL): ("up", "up2"), (2, SEL): ("uptop", "up"),
}

# the compiled free-attention scores: per (head, query kind), the key kinds, row offset, column offset, top flag
TRUE_FREE = {
    (0, SHIFT): ((VIN, SEL), -1, -1, None),
    (0, SUB): ((SHIFT, SEL), -1, 0, None),
    (1, SUB): ((NIN, Z), None, 0, None),
    (2, SUB): ((SUB, Z), 0, -1, None),
    (0, SEL): ((SHIFT, SEL), -2, 0, None),
    (1, SEL): ((SUB,), -1, 0, None),
    (2, SEL): ((SUB,), -1, None, True),
    (1, SHIFT): ((Z,), None, None, None),                                   # unused: any constant zero
    (2, SHIFT): ((Z,), None, None, None),
}


@dataclass
class Cfg:
    attn: str = "fixed"
    rule: str = "table"
    learn: tuple = ("SUB",)
    fmt: str = "digits"
    K: int = 0
    d: int = 16
    roles: str = "kinds"
    tied: bool = True
    nblocks: int = 0               # untied: blocks for the first nblocks ticks; later ticks reuse the last one
    hidden: int = 0
    ticks: int = 0
    init: float = 0.1
    Rb: int = 3
    Cb: int = 2
    menu: str = "wide"             # wide: all 7 options for every port; narrow: NARROW; rel: + 13 relative offsets
    menu_extra: int = -1           # menu "rel": keep only the first menu_extra extra offsets (-1 = all 13)
    st: bool = False               # straight-through states in training: the forward pass snaps every state (the
                                   # hard path), the backward pass uses the soft state's gradient
    free: str = "joint"            # free attention's score: "joint" one bias per (query kind, key kind, row offset,
                                   # column offset) plus a top-place term (a Swin-style 2-D relative bias table per
                                   # kind pair), or "factor" (separate additive kind, row, column and top terms)
    mlp_kinds: tuple = ()          # rule "mlp": only these kinds use the MLP; the others keep their compiled tables
                                   # (() = every kind uses the MLP)
    window: tuple = ()             # free attention: (Rw, Cw) keeps keys within Rw rows above and Cw columns either
                                   # side, plus the N row (row 0) from anywhere; () = every cell
    copies: int = 1                # parallel copies of every head (menu / free / abs attention); a port reads ...
    copy_mode: str = "mean"        # ... the mean of its copies' values ("mean"), or, for MLP rules, all of them
                                   # side by side ("concat")
    inject: str = "always"         # the prompt re-injected after every tick ("always"), or only in the initial
                                   # state ("first", TRM-style): input cells are then updated like any cell, by the
                                   # copy rule on their port A, and must learn to attend to themselves to hold
    causal: bool = True            # free / abs attention: a cell attends only to cells before it in raster order
                                   # (earlier rows, or its own row to its left): the 2-D decoder mask
    extra_ticks: int = 0
    maxrows: int = 64
    maxcols: int = 16


class Index:
    """Per-layout index tensors the attention needs (cached)."""

    def __init__(self, lay: G.Layout, cfg: Cfg):
        L = lay.L
        kind = lay.kind.copy()
        if cfg.roles == "none":
            kind[np.isin(kind, SCRATCH)] = SUB
        self.kind = torch.from_numpy(kind)
        self.qkind = self.kind[:, None].expand(L, L)
        self.kkind = self.kind[None, :].expand(L, L)
        dr = lay.row[None, :] - lay.row[:, None]
        dc = lay.col[None, :] - lay.col[:, None]
        self.br = torch.from_numpy(np.clip(dr, -cfg.Rb, cfg.Rb) + cfg.Rb)
        self.bc = torch.from_numpy(np.clip(dc, -cfg.Cb, cfg.Cb) + cfg.Cb)
        self.top = torch.from_numpy(lay.top.astype(np.int64))[None, :].expand(L, L)
        self.inp = torch.from_numpy(lay.is_input)
        self.scratch = ~self.inp
        self.row = torch.from_numpy(lay.row)
        self.col = torch.from_numpy(lay.col + 1)
        order = lay.row * lay.C + lay.col                                     # raster order
        mask = order[None, :] < order[:, None]
        if cfg.inject == "first":                                             # an input cell may read itself
            mask[lay.is_input] = (order[None, :] <= order[:, None])[lay.is_input]
        else:
            mask[lay.is_input] = True                                         # input cells are re-injected anyway
        if cfg.window:
            Rw, Cw = cfg.window
            local = (dr <= 0) & (dr >= -Rw) & (np.abs(dc) <= Cw)
            mask = mask & (local | (lay.row[None, :] == 0))
            if cfg.inject != "first":
                mask[lay.is_input] = True
        self.causal = torch.from_numpy(np.where(mask, 0.0, -np.inf))
        self.menu = None
        if cfg.attn == "menu":
            opts = menu_options(cfg)
            M = torch.zeros(len(opts), L, L)
            for m, opt in enumerate(opts):
                for p in np.where(~lay.is_input)[0]:
                    M[m, p, menu_source(lay, p, opt)] = 1
            self.menu = M
        self.comp = G.compiled_attention(lay, torch.float64)


class Block(nn.Module):
    """The parameters of one block application."""

    def __init__(self, cfg: Cfg, B: int, gen: torch.Generator, dtype):
        super().__init__()
        self.cfg, self.B = cfg, B
        r = lambda *s: nn.Parameter(torch.randn(*s, generator=gen, dtype=dtype) * cfg.init)
        nb, nc = 2 * cfg.Rb + 1, 2 * cfg.Cb + 1
        nh = 3 * cfg.copies                                               # heads: 3 ports x their copies
        assert cfg.copies == 1 or cfg.attn != "fixed", "copies of fixed heads would all be the same"
        if cfg.attn == "menu":
            nopt = len(menu_options(cfg))
            self.menu = r(nh, NK, nopt)
            mask = torch.zeros(3, NK, nopt, dtype=dtype)
            if cfg.menu not in ("wide", "rel"):
                # narrow: NARROW; narrow_carry: NARROW with the carry route (C.SUB) wide; wide_nocarry: every port
                # wide but the carry route fixed
                allowed = {hk: (MENU if cfg.menu == "wide_nocarry" else o) for hk, o in NARROW.items()}
                if cfg.menu == "narrow_carry":
                    allowed[(2, SUB)] = MENU
                if cfg.menu == "wide_nocarry":
                    allowed[(2, SUB)] = ("left",)
                mask.fill_(-math.inf)
                for (h, kd), opts in allowed.items():
                    for o in opts:
                        mask[h, kd, MENU.index(o)] = 0
                mask[:, [Z, NIN, VIN]] = 0
            self.register_buffer("menu_mask", mask.repeat_interleave(cfg.copies, 0))
        elif cfg.attn == "free" and cfg.free == "factor":
            self.U = r(nh, NK, NK)
            self.Rr = r(nh, NK, nb)
            self.Rc = r(nh, NK, nc)
            self.Tp = r(nh, NK, 2)
        elif cfg.attn == "free":
            self.Rel = r(nh, NK, NK, nb, nc)
            self.Tp = r(nh, NK, NK, 2)
        elif cfg.attn == "abs":
            dq = cfg.d
            self.Wq = nn.Parameter(torch.randn(nh, self.state_dim(), dq, generator=gen, dtype=dtype) / math.sqrt(dq))
            self.Wk = nn.Parameter(torch.randn(nh, self.state_dim(), dq, generator=gen, dtype=dtype) / math.sqrt(dq))
        S = self.state_dim()
        if cfg.fmt != "digits":                                         # value projections of the heads
            self.Wv = nn.Parameter(torch.randn(nh, S, S, generator=gen, dtype=dtype) / math.sqrt(S))
        if cfg.rule in ("table", "mlptab"):
            assert cfg.fmt == "digits"
            self.Ld = nn.ParameterDict()
            self.Lb = nn.ParameterDict()
            if cfg.rule == "table":
                for name in cfg.learn:
                    self.Ld[name] = r(B, B, 2, B)
                    self.Lb[name] = r(B, B, 2, 2)
            else:                                                             # the tables are an MLP's outputs
                nin = NK + 2 * B + 2
                H = cfg.hidden or 3 * 2 * B * B
                self.W1 = nn.Parameter(torch.randn(nin, H, generator=gen, dtype=dtype) * math.sqrt(2 / nin))
                self.b1 = nn.Parameter(torch.zeros(H, dtype=dtype))
                self.W2 = nn.Parameter(torch.randn(H, B + 2, generator=gen, dtype=dtype) * math.sqrt(1 / H))
                self.b2 = nn.Parameter(torch.zeros(B + 2, dtype=dtype))
        else:
            ncopy = cfg.copies if cfg.copy_mode == "concat" else 1
            nin = NK + ncopy * (2 * B + 2 if cfg.fmt == "digits" else 3 * S) + (S if cfg.fmt == "free" else 0)
            H = cfg.hidden or 3 * 2 * B * B
            self.W1 = nn.Parameter(torch.randn(nin, H, generator=gen, dtype=dtype) * math.sqrt(2 / nin))
            self.b1 = nn.Parameter(torch.zeros(H, dtype=dtype))
            self.W2 = nn.Parameter(torch.randn(H, S, generator=gen, dtype=dtype) * math.sqrt(1 / H))
            self.b2 = nn.Parameter(torch.zeros(S, dtype=dtype))
            if cfg.fmt == "free":
                self.ln = nn.LayerNorm(S, dtype=dtype)

    def state_dim(self):
        c = self.cfg
        return self.B + 2 if c.fmt == "digits" else (c.K or self.B + 2) if c.fmt == "symbols" else c.d


class Net(nn.Module):
    def __init__(self, cfg: Cfg, B: int, seed: int = 0, dtype=torch.float64):
        super().__init__()
        self.cfg, self.B, self.dtype = cfg, B, dtype
        g = torch.Generator().manual_seed(seed)
        nblocks = 1 if cfg.tied else cfg.nblocks
        assert cfg.tied or cfg.nblocks, "untied blocks need nblocks"
        self.blocks = nn.ModuleList([Block(cfg, B, g, dtype) for _ in range(nblocks)])
        S = self.blocks[0].state_dim()
        self.S = S
        if cfg.fmt != "digits":
            self.Ein = nn.Parameter(torch.randn(3, B, S, generator=g, dtype=dtype) * (1.0 if cfg.fmt == "symbols" else 1.0))
            self.Wout = nn.Parameter(torch.randn(S, B, generator=g, dtype=dtype) * (1.0 if cfg.fmt == "symbols" else 1 / math.sqrt(S)))
            self.s0 = nn.Parameter(torch.zeros(S, dtype=dtype))
        if cfg.fmt == "free" or cfg.attn == "abs":
            self.Kemb = nn.Parameter(torch.randn(NK, S, generator=g, dtype=dtype) * 0.5)
        if cfg.attn == "abs":
            self.Prow = nn.Parameter(torch.randn(cfg.maxrows, S, generator=g, dtype=dtype) * 0.5)
            self.Pcol = nn.Parameter(torch.randn(cfg.maxcols, S, generator=g, dtype=dtype) * 0.5)
        self.truth = G.truth_tables(B, dtype)
        self._idx = {}
        self.force = None                                                     # ((head, kind), option) or None

    # ----- helpers -----
    def index(self, lay: G.Layout) -> Index:
        key = (lay.B, lay.w, lay.k, lay.links)
        if key not in self._idx:
            self._idx[key] = Index(lay, self.cfg)
        return self._idx[key]

    def ticks(self, lay):
        c = self.cfg
        return (c.ticks or lay.depth) + c.extra_ticks

    def block(self, t):
        return self.blocks[0] if self.cfg.tied else self.blocks[min(t, len(self.blocks) - 1)]

    # ----- inputs and outputs -----
    def inputs(self, lay: G.Layout, Np, vp):
        """Input states [X, L, S] from digit distributions of N [X, w, B] and v [X, w + k, B]; scratch cells get the
        format's empty state."""
        c = self.cfg
        if c.fmt == "digits":
            return G.input_states(lay, Np, vp, self.dtype)
        X, B = Np.shape[0], self.B
        dig = G.input_states(lay, Np, vp, self.dtype)[..., :B]               # [X, L, B] digit distributions
        ik = torch.from_numpy(np.where(lay.kind == NIN, 1, np.where(lay.kind == VIN, 2, 0)))
        E = self.Ein[ik]                                                      # [L, B, S]
        if c.fmt == "symbols":
            emb = torch.einsum("xlb,lbs->xls", dig, E.softmax(-1))
            empty = self.s0.softmax(-1)
        else:
            emb = torch.einsum("xlb,lbs->xls", dig, E) + self.Kemb[torch.from_numpy(lay.kind)]
            empty = self.s0
        inp = torch.from_numpy(lay.is_input)
        return torch.where(inp[None, :, None], emb, empty)

    def readout(self, lay: G.Layout, S):
        """The answer's digit distributions [X, w, B]."""
        c = self.cfg
        A = S[:, lay.answer]
        if c.fmt == "digits":
            return A[..., :self.B]
        if c.fmt == "symbols":
            return torch.einsum("xws,sb->xwb", A, self.Wout.softmax(-1))
        return (A @ self.Wout).softmax(-1)

    # ----- the block -----
    def port_attention(self, lay, ix: Index, blk: Block, S, hard):
        """Each port's attention [3, L, L]: the mean over its head copies (what a "mean" port reads)."""
        A = self.attention(lay, ix, blk, S, hard)
        if self.cfg.copies > 1:
            A = A.view(3, self.cfg.copies, *A.shape[1:]).mean(1)
        return A

    def attention(self, lay, ix: Index, blk: Block, S, hard):
        """[heads, L, L] (or [heads, X, L, L]) attention weights; heads = 3 ports x copies, port-major."""
        c = self.cfg
        if c.attn == "fixed":
            return ix.comp.to(self.dtype)
        if c.attn == "menu":
            w = (blk.menu + blk.menu_mask).softmax(-1)                        # [3, NK, M]
            if hard:
                w = F.one_hot(w.argmax(-1), w.shape[-1]).to(self.dtype)
            if self.force is not None:                                        # one port set hard to one option
                (h, kd), o = self.force
                w = w.clone()
                w[h, kd] = F.one_hot(torch.tensor(o), w.shape[-1]).to(self.dtype)
            wq = w[:, ix.kind]                                                # [heads, L, M]
            return torch.einsum("hlm,mlk->hlk", wq, ix.menu.to(self.dtype))
        if c.attn == "free":
            if c.free == "factor":
                s = (blk.U[:, ix.qkind, ix.kkind] + blk.Rr[:, ix.qkind, ix.br] + blk.Rc[:, ix.qkind, ix.bc]
                     + blk.Tp[:, ix.qkind, ix.top])                           # [3, L, L]
            else:
                s = blk.Rel[:, ix.qkind, ix.kkind, ix.br, ix.bc] + blk.Tp[:, ix.qkind, ix.kkind, ix.top]
            if c.causal:
                s = s + ix.causal.to(s.dtype)
            if hard:
                return F.one_hot(s.argmax(-1), s.shape[-1]).to(self.dtype)
            return s.softmax(-1)
        if c.attn == "abs":
            pos = self.Prow[ix.row] + self.Pcol[ix.col] + self.Kemb[ix.kind]   # [L, S]
            h = S + pos
            q = torch.einsum("xls,hsd->hxld", h, blk.Wq)
            k = torch.einsum("xls,hsd->hxld", h, blk.Wk)
            s = torch.einsum("hxqd,hxkd->hxqk", q, k) / math.sqrt(q.shape[-1])
            if c.causal:
                s = s + ix.causal.to(s.dtype)
            if hard:
                return F.one_hot(s.argmax(-1), s.shape[-1]).to(self.dtype)
            return s.softmax(-1)
        raise ValueError(c.attn)

    def rule_tables(self, blk: Block, hard):
        """The table rule's per-kind tables [3, B, B, 2, B + 2] (learned kinds from their logits; for "mlptab",
        the MLP's logits on every one-hot (kind, a, b, c))."""
        T = self.truth.clone()
        B = self.B
        for i, kd in enumerate(SCRATCH):
            name = G.KINDS[kd]
            if self.cfg.rule == "mlptab" and name in self.cfg.learn:
                a, m, cc = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
                x = torch.cat([F.one_hot(torch.full_like(a, kd), NK), F.one_hot(a, B), F.one_hot(m, B),
                               F.one_hot(cc, 2)], -1).to(self.dtype)
                o = torch.relu(x @ blk.W1 + blk.b1) @ blk.W2 + blk.b2               # [B, B, 2, B + 2] logits
                d, b = o[..., :B], o[..., B:]
            elif name in blk.Ld:
                d, b = blk.Ld[name], blk.Lb[name]
            else:
                continue
            if hard:
                d = F.one_hot(d.argmax(-1), self.B).to(self.dtype)
                b = F.one_hot(b.argmax(-1), 2).to(self.dtype)
            else:
                d, b = d.softmax(-1), b.softmax(-1)
            T[i] = torch.cat([d, b], -1)
        return T

    def rule_mlp(self, blk: Block, kind_onehot, ports, own=None):
        """The MLP rule on [X, L, NK] kind features and ports [X, L, 3 * S] -> new states [X, L, S]."""
        c = self.cfg
        feats = [kind_onehot, ports] + ([own] if own is not None else [])
        x = torch.cat(feats, -1)
        h = torch.relu(x @ blk.W1 + blk.b1)
        o = h @ blk.W2 + blk.b2
        B = self.B
        if c.fmt == "digits":
            return torch.cat([o[..., :B].softmax(-1), o[..., B:].softmax(-1)], -1)
        if c.fmt == "symbols":
            return o.softmax(-1)
        return blk.ln(own + o)

    def tick(self, lay, ix: Index, blk: Block, S, hard):
        c = self.cfg
        A = self.attention(lay, ix, blk, S, hard)
        if A.dim() == 3:
            H = torch.einsum("hqk,xkd->hxqd", A, S)
        else:
            H = torch.einsum("hxqk,xkd->hxqd", A, S)
        if c.fmt != "digits":
            H = torch.einsum("hxqd,hde->hxqe", H, blk.Wv)
            if c.fmt == "symbols":
                H = H.softmax(-1)
        X, L = S.shape[:2]
        B = self.B
        Hc = H.view(3, c.copies, *H.shape[1:])                                # [3, copies, X, L, d]
        Hm = Hc.mean(1)                                                       # the ports, copies averaged
        hold = c.inject == "first"                                            # input cells: the copy rule
        if c.rule in ("table", "mlptab"):
            T = self.rule_tables(blk, hard)
            Tcell = torch.zeros(L, 2 * B * B, B + 2, dtype=self.dtype)
            for i, kd in enumerate(SCRATCH):
                m = ix.kind == kd
                Tcell[m] = T[i].reshape(2 * B * B, B + 2)
            if hold:
                Tcell[ix.inp] = self.truth[0].reshape(2 * B * B, B + 2)
            a, b, cc = Hm[0, ..., :B], Hm[1, ..., :B], Hm[2, ..., B:]
            joint = (a[..., :, None, None] * b[..., None, :, None] * cc[..., None, None, :]).reshape(X, L, -1)
            out = torch.einsum("xpj,pjo->xpo", joint, Tcell)
        else:
            kind1 = F.one_hot(ix.kind, NK).to(self.dtype)[None].expand(X, L, NK)
            concat = c.copy_mode == "concat" and c.copies > 1
            if c.fmt == "digits":                                             # the compiled port format
                if concat:                                                    # copy by copy: A_j, B_j, C_j
                    ports = torch.cat([t for j in range(c.copies) for t in
                                       (Hc[0, j, ..., :B], Hc[1, j, ..., :B], Hc[2, j, ..., B:])], -1)
                else:
                    ports = torch.cat([Hm[0, ..., :B], Hm[1, ..., :B], Hm[2, ..., B:]], -1)
            elif concat:
                ports = Hc.permute(2, 3, 1, 0, 4).reshape(X, L, -1)           # copy-major, then port
            else:
                ports = Hm.permute(1, 2, 0, 3).reshape(X, L, -1)
            out = self.rule_mlp(blk, kind1, ports, S if c.fmt == "free" else None)
            if c.mlp_kinds:                                                   # the other kinds: compiled tables
                Tcell = torch.zeros(L, 2 * B * B, B + 2, dtype=self.dtype)
                for i, kd in enumerate(SCRATCH):
                    Tcell[ix.kind == kd] = self.truth[i].reshape(2 * B * B, B + 2)
                a, b, cc = Hm[0, ..., :B], Hm[1, ..., :B], Hm[2, ..., B:]
                joint = (a[..., :, None, None] * b[..., None, :, None] * cc[..., None, None, :]).reshape(X, L, -1)
                tab = torch.einsum("xpj,pjo->xpo", joint, Tcell)
                use = torch.zeros(L, dtype=torch.bool)
                for name in c.mlp_kinds:
                    use |= ix.kind == G.KINDS.index(name)
                out = torch.where(use[None, :, None], out, tab)
        return out

    def snap(self, S):
        c = self.cfg
        if c.fmt == "digits":
            return G.snap_state(S, self.B)
        if c.fmt == "symbols":
            return F.one_hot(S.argmax(-1), S.shape[-1]).to(S.dtype)
        return S

    def forward(self, lay: G.Layout, S_in, hard=False, record=False):
        ix = self.index(lay)
        S = S_in
        hist = []
        for t in range(self.ticks(lay)):
            S = self.tick(lay, ix, self.block(t), S, hard)
            if self.cfg.inject != "first":
                S = torch.where(ix.inp[None, :, None], S_in, S)
            if hard:
                S = self.snap(S)
            elif self.cfg.st and self.cfg.fmt != "free":
                S = self.snap(S.detach()) + S - S.detach()
            if record:
                hist.append(S)
        return (S, hist) if record else S

    def answer(self, lay, Np, vp, hard=False):
        S = self.forward(lay, self.inputs(lay, Np, vp), hard)
        return self.readout(lay, S)

    # ----- the rule on constructed ports (cell-level lessons; only where cells are identifiable) -----
    def sub_rule(self, a, b, c, hard=False):
        """The SUB cell's rule on port distributions a [n, B], b [n, B], borrow c [n, 2] -> digit [n, B], borrow
        [n, 2]. Needs fmt "digits" and roles "kinds"."""
        return self.kind_rule(SUB, a, b, c, hard)

    def kind_rule(self, kd, a, b, c, hard=False):
        cfg = self.cfg
        assert cfg.fmt == "digits" and cfg.roles == "kinds"
        blk = self.blocks[0]
        B = self.B
        if cfg.rule in ("table", "mlptab"):
            T = self.rule_tables(blk, hard)[SCRATCH.index(kd)].reshape(2 * B * B, B + 2)
            joint = (a[:, :, None, None] * b[:, None, :, None] * c[:, None, None, :]).reshape(len(a), -1)
            o = joint @ T
        elif cfg.mlp_kinds and G.KINDS[kd] not in cfg.mlp_kinds:
            T = self.truth[SCRATCH.index(kd)].reshape(2 * B * B, B + 2)
            joint = (a[:, :, None, None] * b[:, None, :, None] * c[:, None, None, :]).reshape(len(a), -1)
            o = joint @ T
        else:
            n = len(a)
            kind1 = F.one_hot(torch.full((n,), kd), NK).to(self.dtype)
            ncopy = cfg.copies if cfg.copy_mode == "concat" else 1               # every copy gets the same ports
            o = self.rule_mlp(blk, kind1, torch.cat([a, b, c] * ncopy, 1))
            if hard:
                o = G.snap_state(o, B)
        return o[:, :B], o[:, B:]
