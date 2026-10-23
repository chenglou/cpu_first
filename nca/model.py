"""The NCA: one small rule network run by every cell of the grid on its 3 x 3 neighbourhood, tick after tick.

A tick, for every cell at once (synchronous updates):
    input    the 3 x 3 neighbourhood's kinds (one-hot) and states, the grid padded with a ring of OUT cells holding
             the blank state (every channel at value 0)
    rule     "mlp": a ReLU MLP, the 3 x 3 input -> `hidden` units -> one logit per value of every channel. Its first
             layer is a 3 x 3 convolution and its second a 1 x 1 one, the shape of Mordvintsev et al.'s NCA
             (perception, then a per-cell MLP), here with the raw neighbourhood as the perception.
             "logic": layers of learned 2-input logic gates over the neighbourhood's bit planes (logic.py), snapped to
             Boolean gates on the hard path
    state    each channel's logits become a distribution: "soft" (softmax), "ste" (straight-through: the forward
             pass snaps to the argmax, the backward pass uses the softmax's gradient) or "hard" (argmax, evaluation)
    prompt   re-injected after every tick (`clamp`), or written only at tick 0 (the rule must then hold it)
`hood` "noself" hides the cell's own state from its rule (not its kind).
Kinds: OUT (the edge), PROMPT, SHIFT, SUB, SEL. With `kinds` "none" every scratch row reads as SUB, so the rule
cannot tell rows apart by kind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nca import grid as G
from nca.grid import OUT, PROMPT, SCRATCH, SUB


@dataclass
class Cfg:
    hidden: int = 64
    state: str = "soft"            # soft | ste
    clamp: bool = True             # the prompt re-injected after every tick
    kinds: str = "rows"            # rows: SHIFT / SUB / SEL given; none: every scratch row the same kind
    init_scale: float = 1.0        # multiplies PyTorch-style fan-in init of both layers
    hood: str = "3x3"              # 3x3: the cell and its 8 neighbours; noself: the cell's own state hidden from
                                   # its rule (its kind still seen), so a cell cannot simply hold its value;
                                   # causal: noself, and the states of the row below hidden too (kinds still seen),
                                   # so information only flows down the grid and along rows
    rule: str = "mlp"              # mlp | logic | hyper (hyper.py: generated all-to-all wiring)
    layers: tuple = (256, 256, 256, 256)   # logic: gates per layer (then one group per output value)
    group: int = 8                 # logic: gates per output value
    tau: float = 0.0               # logic: group-sum temperature (0: group / 4)
    gate_init: str = "residual"    # logic: residual (gates start near pass-through) | random
    kindset: str = "div"           # div: 5 kinds; sq: 7 (adds SPLIT and MUL)
    extra: str = ""                # fixed reads beyond the 3 x 3: "ncol" the prompt cell of the cell's column
                                   # (N's digit: "row-local plus N's column"); "ports" also two rows up and the top
                                   # place of the row above (the bridge's long-range ports)
    split: bool = False            # mlp: the hidden units split into one block per output channel (each channel
                                   # its own small MLP over the same neighbourhood)
    heads: int = 3                 # hyper (hyper.py): generated all-to-all heads
    ghidden: int = 64              # hyper: g's hidden units
    gbias: float = 0.0             # hyper: g's initial output bias (negative: sigmoid weights start small)
    mask: str = "all"              # hyper: all | noself (not the cell itself) | causal (nor the rows below)
    gnorm: str = "softmax"         # hyper: softmax (sources compete) | sigmoid (each connection on its own)


class NCA(nn.Module):
    def __init__(self, fmt: G.Fmt, cfg: Cfg, seed: int = 0, dtype=torch.float32):
        super().__init__()
        self.fmt, self.cfg, self.dtype = fmt, cfg, dtype
        g = torch.Generator().manual_seed(seed)
        self.nk = NK = G.NK_SQ if cfg.kindset == "sq" else G.NK
        nin = NK + fmt.D
        H = cfg.hidden
        s = cfg.init_scale
        self.W1 = nn.Parameter(torch.randn(H, nin, 3, 3, generator=g, dtype=dtype) * s * math.sqrt(2 / (9 * nin)))
        self.b1 = nn.Parameter(torch.zeros(H, dtype=dtype))
        self.W2 = nn.Parameter(torch.randn(fmt.D, H, generator=g, dtype=dtype) * s * math.sqrt(1 / H))
        self.b2 = nn.Parameter(torch.zeros(fmt.D, dtype=dtype))
        self._kinds = {}
        self.nextra = {"": 0, "ncol": 1, "ports": 3}[cfg.extra]
        if self.nextra:
            ne = self.nextra * nin
            self.Wx = nn.Parameter(torch.randn(H, ne, generator=g, dtype=dtype) * s * math.sqrt(2 / (9 * nin)))
        W2mask = torch.ones(fmt.D, H, dtype=dtype)
        if cfg.split:
            nch = len(fmt.chans)
            W2mask.zero_()
            for i, (n, _) in enumerate(fmt.chans):
                W2mask[fmt.sl(n), i * H // nch:(i + 1) * H // nch] = 1
        self.register_buffer("W2mask", W2mask)
        mask = torch.ones(1, nin, 3, 3, dtype=dtype)
        if cfg.hood in ("noself", "causal"):
            mask[:, NK:, 1, 1] = 0
        if cfg.hood == "causal":
            mask[:, NK:, 2, :] = 0
        self.register_buffer("mask", mask)
        if cfg.rule == "logic":
            from nca.logic import LogicRule
            for p in (self.W1, self.b1, self.W2, self.b2):
                p.requires_grad_(False)
            hidden_q = {"3x3": (), "noself": (4,), "causal": (4, 6, 7, 8)}[cfg.hood]
            allowed = [c * 9 + q for c in range(nin) for q in range(9) if not (c >= NK and q in hidden_q)]
            allowed += list(range(9 * nin, 9 * nin + self.nextra * nin))
            self.logic = LogicRule(9 * nin + self.nextra * nin, fmt.D, cfg.layers, cfg.group, cfg.tau, cfg.gate_init,
                                   allowed, seed, dtype)
            if self.nextra:
                self.Wx.requires_grad_(False)

    # ----- the grid -----
    def kinds(self, lay: G.Layout):
        """[1, NK, R + 2, C + 2] one-hot kinds with the OUT ring (cached per size)."""
        key = lay.key()
        if key not in self._kinds:
            kd = np.full((lay.R + 2, lay.C + 2), OUT)
            inner = lay.kind.copy()
            if self.cfg.kinds == "none":
                inner[np.isin(inner, SCRATCH)] = SUB
            assert inner.max() < self.nk, "this grid has kinds the rule was not built for (kindset)"
            kd[1:-1, 1:-1] = inner
            oh = F.one_hot(torch.from_numpy(kd), self.nk).to(self.dtype).permute(2, 0, 1)[None]
            self._kinds[key] = oh.contiguous()
        return self._kinds[key]

    def blank(self, X, R, C):
        """[X, D, R, C]: every channel at value 0."""
        S = torch.zeros(X, self.fmt.D, R, C, dtype=self.dtype)
        for n, _ in self.fmt.chans:
            S[:, self.fmt.off[n]] = 1
        return S

    def prompt_state(self, lay: G.Layout, Nd, vd, scratch="blank", gen=None):
        """The tick-0 state [X, D, R, C]: the prompt row from N's and v's digits (integer arrays [X, w] and
        [X, w + k], or digit distributions [X, w, B] / [X, w + k, B] for soft prompts), scratch cells blank or
        random ("random": random one-hot values, "noise": random distributions)."""
        B = lay.B
        X = len(Nd)
        f = self.fmt
        if scratch == "blank":
            S = self.blank(X, lay.R, lay.C)
        else:
            S = torch.zeros(X, f.D, lay.R, lay.C, dtype=self.dtype)
            for n, sz in f.chans:
                if scratch == "random":
                    v = torch.randint(sz, (X, lay.R, lay.C), generator=gen)
                    S[:, f.sl(n)] = F.one_hot(v, sz).permute(0, 3, 1, 2).to(self.dtype)
                else:
                    S[:, f.sl(n)] = torch.rand(X, sz, lay.R, lay.C, generator=gen, dtype=self.dtype).softmax(1)
        S[:, :, 0] = self.blank(X, 1, lay.C)[:, :, 0]
        S[:, f.sl("d"), 0] = self.digit_row(lay, vd, "d", B)
        S[:, f.sl("n"), 0] = self.digit_row(lay, Nd, "n", B)
        return S

    def digit_row(self, lay, digs, which, B):
        """The prompt row's channel `which` [X, B, C] from digits (or digit distributions)."""
        k, w = lay.k, lay.w
        t = torch.as_tensor(digs)
        if t.dtype in (torch.int64, torch.int32):
            t = F.one_hot(t.long(), B).to(self.dtype)
        X = t.shape[0]
        row = torch.zeros(X, lay.C, B, dtype=self.dtype)
        row[..., 0] = 1
        if which == "d":
            row[:, :t.shape[1]] = t
        else:
            row[:, k:k + w] = t
        return row.permute(0, 2, 1)

    # ----- one tick -----
    def logits(self, S, lay, hard=False):
        """The rule on every cell: S [X, D, R, C] -> logits [X, D, R, C]. `hard` snaps a logic rule's gates."""
        X = S.shape[0]
        P = self.blank(X, lay.R + 2, lay.C + 2)
        P = torch.cat([P[:, :, :1], torch.cat([P[:, :, 1:-1, :1], S, P[:, :, 1:-1, -1:]], 3), P[:, :, -1:]], 2)
        inp = torch.cat([self.kinds(lay).expand(X, -1, -1, -1), P], 1)
        return self.apply_rule(inp, hard, self.extra_inputs(inp) if self.nextra else None)

    def extra_inputs(self, inp):
        """The fixed long-range reads [X, E (NK + D), R, C] of every cell, from the padded input (kinds and states):
        the prompt cell of its column; for "ports" also the cell two rows up and the top place of the row above
        (OUT beyond the grid)."""
        X, nin, R2, C2 = inp.shape
        R, C = R2 - 2, C2 - 2
        inner = inp[:, :, 1:-1, 1:-1]
        outs = [inner[:, :, :1].expand(-1, -1, R, -1)]                        # the prompt row, same column
        if self.cfg.extra == "ports":
            edge = inp[:, :, :1, 1:-1]                                         # an OUT cell's input, per column
            up2 = torch.cat([edge.expand(-1, -1, 2, -1), inner[:, :, :-2]], 2)
            top = inner[:, :, :, -1:]                                          # each row's top place
            uptop = torch.cat([edge[:, :, :, :1], top[:, :, :-1]], 2).expand(-1, -1, -1, C)
            outs += [up2, uptop]
        return torch.cat(outs, 1)

    def apply_rule(self, inp, hard=False, extra=None):
        """The rule on a padded input [X, NK + D, R + 2, C + 2] (kinds, then states), plus the long-range reads
        [X, E (NK + D), R, C] if any -> logits [X, D, R, C]."""
        if self.cfg.rule == "logic":
            X, _, R2, C2 = inp.shape
            R, C = R2 - 2, C2 - 2
            u = F.unfold(inp, 3)                                              # [X, 9 (NK + D), R C], channel-major
            if extra is not None:
                u = torch.cat([u, extra.reshape(X, extra.shape[1], R * C)], 1)
            z = self.logic(u.permute(0, 2, 1).reshape(X * R * C, -1), hard)
            return z.view(X, R, C, -1).permute(0, 3, 1, 2)
        pre = F.conv2d(inp, self.W1 * self.mask, self.b1)
        if extra is not None:
            pre = pre + torch.einsum("xerc,he->xhrc", extra, self.Wx)
        h = F.relu(pre)
        return torch.einsum("xhrc,dh->xdrc", h, self.W2 * self.W2mask) + self.b2[None, :, None, None]

    def normalize(self, Z, mode):
        out = []
        for n, sz in self.fmt.chans:
            z = Z[:, self.fmt.sl(n)]
            if mode == "hard":
                out.append(F.one_hot(z.argmax(1), sz).permute(0, 3, 1, 2).to(Z.dtype))
                continue
            p = z.softmax(1)
            if mode == "ste":
                h = F.one_hot(z.argmax(1), sz).permute(0, 3, 1, 2).to(Z.dtype)
                p = h + p - p.detach()
            out.append(p)
        return torch.cat(out, 1)

    def run(self, lay, S0, ticks, mode="soft", record=False):
        """`ticks` ticks from S0. Returns the final state and the last tick's soft distributions (the answer loss
        reads these, so that a snapped forward pass still has a likelihood), and every tick's state if `record`."""
        clamp = self.cfg.clamp
        pm = torch.from_numpy(lay.prompt)[None, None]
        S = S0
        hist = []
        P = S0
        for _ in range(ticks):
            Z = self.logits(S, lay, hard=mode == "hard")
            P = self.normalize(Z, "soft")
            S = self.normalize(Z, mode) if mode != "soft" else P
            if clamp:
                S = torch.where(pm, S0, S)
                P = torch.where(pm, S0, P)
            if record:
                hist.append(S)
        return (S, P, hist) if record else (S, P)

    def channel(self, S, name):
        return S[:, self.fmt.sl(name)]
