"""Interchangeable small models for microlab. Each maps a Batch's digit fields to logits for the target and for any
requested auxiliary fields. Register new ones with @model("name").

- mlp  : one-hot digits, concatenated, two hidden layers.
- xf   : transformer encoder over digit tokens (digit + field + position-within-field embeddings) plus one query token
         per output digit; bidirectional attention; logits read from the query tokens.
- loop : like xf, but one shared block applied `depth` times (a looped transformer).
Positions are learned per index within a field, so width extrapolation is not expected from these baselines.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

MODELS: dict[str, type] = {}


def model(name):
    def deco(cls):
        MODELS[name] = cls
        return cls
    return deco


class _Base(nn.Module):
    def __init__(self, in_widths: dict[str, int], out_width: int, aux_widths: dict[str, int], **kw):
        super().__init__()
        self.in_widths, self.out_width, self.aux_widths = in_widths, out_width, aux_widths
        self.heads = {"target": out_width, **aux_widths}


@model("mlp")
class MLP(_Base):
    def __init__(self, in_widths, out_width, aux_widths, hidden=512, **kw):
        super().__init__(in_widths, out_width, aux_widths)
        n_in = 10 * sum(in_widths.values())
        n_out = 10 * sum(self.heads.values())
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = torch.cat([F.one_hot(inputs[k], 10).float().flatten(1) for k in self.in_widths], 1)
        z = self.net(x)
        out, i = {}, 0
        for name, w in self.heads.items():
            out[name] = z[:, i:i + 10 * w].view(-1, w, 10)
            i += 10 * w
        return out


@model("xf")
class Transformer(_Base):
    def __init__(self, in_widths, out_width, aux_widths, d=128, heads=4, depth=3, loop=False, max_pos=40, **kw):
        super().__init__(in_widths, out_width, aux_widths)
        self.fields = list(in_widths) + list(self.heads)
        self.digit = nn.Embedding(11, d)                 # 10 = query slot
        self.field = nn.Embedding(len(self.fields), d)
        self.pos = nn.Embedding(max_pos, d)
        layer = lambda: nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.loop, self.depth = loop, depth
        self.blocks = nn.ModuleList([layer()] if loop else [layer() for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 10)

    def forward(self, inputs):
        B = next(iter(inputs.values())).shape[0]
        dev = self.digit.weight.device
        toks = []
        for fi, name in enumerate(self.fields):
            if name in self.in_widths:
                dig = inputs[name]
            else:
                dig = torch.full((B, self.heads[name]), 10, dtype=torch.long, device=dev)
            w = dig.shape[1]
            toks.append(self.digit(dig) + self.field.weight[fi] + self.pos(torch.arange(w, device=dev)))
        h = torch.cat(toks, 1)
        for i in range(self.depth):
            h = (self.blocks[0] if self.loop else self.blocks[i])(h)
        h = self.norm(h)
        out, i = {}, sum(self.in_widths.values())
        for name, w in self.heads.items():
            out[name] = self.out(h[:, i:i + w])
            i += w
        return out


@model("loop")
class Looped(Transformer):
    def __init__(self, in_widths, out_width, aux_widths, depth=6, **kw):
        super().__init__(in_widths, out_width, aux_widths, depth=depth, loop=True, **kw)
