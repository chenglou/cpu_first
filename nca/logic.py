"""A rule made of learned logic gates (the bit-level arm), in the style of differentiable logic gate networks
(Petersen et al., NeurIPS 2022) and their cellular-automaton use (Miotti, Niklasson, Randazzo and Mordvintsev, 2025).

The rule's input is the 3 x 3 neighbourhood as bit planes: every kind and every channel value of every neighbour is
one bit (one-hot), so a cell of the grid sees 9 * (kinds + values) bits. Layers of 2-input gates follow, each gate
wired to two fixed random outputs of the layer before; a gate holds 16 logits, one per Boolean function of two bits,
and computes the softmax mixture of the 16 functions' real-valued relaxations (AND = ab, OR = a + b - ab, XOR =
a + b - 2ab, ...). Every function is c0 + c1 a + c2 b + c3 ab, so a gate is the mixture of four coefficients. The last
layer is split into one group per output value; a value's logit is its group's sum divided by `tau` (difflogic's
group sum). Evaluated hard, every gate is its argmax function on exact bits: the rule is then a Boolean circuit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# (c0, c1, c2, c3) of f(a, b) = c0 + c1 a + c2 b + c3 ab for the 16 Boolean functions of two bits
COEF = torch.tensor([
    [0, 0, 0, 0],      # false
    [0, 0, 0, 1],      # a and b
    [0, 1, 0, -1],     # a and not b
    [0, 1, 0, 0],      # a
    [0, 0, 1, -1],     # not a and b
    [0, 0, 1, 0],      # b
    [0, 1, 1, -2],     # a xor b
    [0, 1, 1, -1],     # a or b
    [1, -1, -1, 1],    # nor
    [1, -1, -1, 2],    # xnor
    [1, 0, -1, 0],     # not b
    [1, 0, -1, 1],     # a or not b
    [1, -1, 0, 0],     # not a
    [1, -1, 0, 1],     # not a or b
    [1, 0, 0, -1],     # nand
    [1, 0, 0, 0],      # true
], dtype=torch.float64)
NAMES = ("0", "and", "a&!b", "a", "!a&b", "b", "xor", "or", "nor", "xnor", "!b", "a|!b", "!a", "!a|b", "nand", "1")


class LogicRule(nn.Module):
    """Bits [N, nin] in (only the columns in `allowed`), value logits [N, nout] out."""

    def __init__(self, nin: int, nout: int, layers=(256, 256, 256, 256), group: int = 8, tau: float = 0.0,
                 init: str = "residual", allowed=None, seed: int = 0, dtype=torch.float32):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.nout, self.group = nout, group
        self.tau = tau or group / 4.0
        allowed = torch.arange(nin) if allowed is None else torch.as_tensor(allowed)
        self.register_buffer("allowed", allowed)
        sizes = list(layers) + [nout * group]
        prev = len(allowed)
        self.wa, self.wb = [], []
        self.logits = nn.ParameterList()
        for i, G in enumerate(sizes):
            ia = torch.randint(prev, (G,), generator=g)
            ib = (ia + 1 + torch.randint(prev - 1, (G,), generator=g)) % prev        # two different inputs
            self.register_buffer(f"ia{i}", ia)
            self.register_buffer(f"ib{i}", ib)
            w = torch.randn(G, 16, generator=g, dtype=dtype)
            if init == "residual":                       # start each gate near the pass-through of its first input
                w[:, 3] += 5.0
            self.logits.append(nn.Parameter(w))
            prev = G
        self.register_buffer("coef", COEF.to(dtype))
        self.nlayers = len(sizes)

    def forward(self, x, hard=False):
        x = x[:, self.allowed]
        for i in range(self.nlayers):
            a = x[:, getattr(self, f"ia{i}")]
            b = x[:, getattr(self, f"ib{i}")]
            w = self.logits[i]
            if hard:
                k = self.coef[w.argmax(-1)]
            else:
                k = w.softmax(-1) @ self.coef
            x = k[:, 0] + k[:, 1] * a + k[:, 2] * b + k[:, 3] * a * b
        return x.view(x.shape[0], self.nout, self.group).sum(-1) / self.tau

    def ngates(self):
        return sum(p.shape[0] for p in self.logits)
