"""Learned lookup-table cells over categorical digits — the 'gate' of a learned digital machine.

A Table maps a tuple of categorical inputs (sizes in_sizes) to a tuple of categorical outputs (sizes
out_sizes). Its only parameters are logits[prod(in_sizes), sum(out_sizes)], randomly initialised: the cell
is a learnable truth table, shared at every digit position and every step (bit-slice weight sharing).

Modes (signal restoration):
  soft  : outputs are softmax distributions (inputs may be soft; joint = outer product)
  ste   : forward = hard one-hot argmax (exactly digital), backward = softmax gradient (straight-through)
  hard  : forward = hard one-hot, no gradient trick (evaluation)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def joint(inputs: list[Tensor]) -> Tensor:
    """Outer product of categorical distributions along the last dim -> [..., prod(sizes)]."""
    out = inputs[0]
    for x in inputs[1:]:
        out = (out.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2)
    return out


def restore(logits: Tensor, mode: str, tau: float = 1.0) -> Tensor:
    soft = F.softmax(logits / tau, dim=-1)
    if mode == "soft":
        return soft
    hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(soft.dtype)
    if mode == "hard":
        return hard
    return hard + soft - soft.detach()  # straight-through


class Table(nn.Module):
    def __init__(self, in_sizes: list[int], out_sizes: list[int], init_scale: float = 0.1) -> None:
        super().__init__()
        self.in_sizes, self.out_sizes = list(in_sizes), list(out_sizes)
        self.logits = nn.Parameter(init_scale * torch.randn(math.prod(in_sizes), sum(out_sizes)))

    def forward(self, *inputs: Tensor, mode: str = "ste", tau: float = 1.0, with_logits: bool = False):
        z = joint(list(inputs)) @ self.logits
        parts = z.split(self.out_sizes, dim=-1)
        outs = [restore(part, mode, tau) for part in parts]
        return (outs, list(parts)) if with_logits else outs

    @torch.no_grad()
    def truth_table(self) -> Tensor:
        """Hard truth table: [prod(in_sizes), n_outputs] of argmax indices."""
        return torch.stack([p.argmax(-1) for p in self.logits.split(self.out_sizes, dim=-1)], dim=-1)


def onehot(idx: Tensor, n: int, dtype=torch.float32) -> Tensor:
    return F.one_hot(idx.long(), n).to(dtype)
