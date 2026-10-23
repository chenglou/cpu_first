"""A found program on one-hot digits (INTEGRATION.md, stage 3's prediction path): the library machine's instructions
with every digit a distribution over 10 values and every cell a straight-through table (its value: the snapped table
exactly, a one-hot row per input; its gradient: the cell's softmax). The pointer is shared by every row and never
depends on data (library/dsl.py), so it is a Python int here and the program runs as straight-line code. With one-hot
inputs every value is exactly the hard machine's (tdsl), and a gradient from the outputs reaches the cells.

State: regs [X, 5, P, 10] (A, X, Y, N, S), c [X, 2] (borrow / carry), h [X, 10] (multiplier digit), k [X, 10] (multiply
carry digit). Tables: SBB, ADC [B * B * 2, 12] (digit 10 + flag 2, rows (a, b, c) flat), MUL [B^4, 20] (low 10 + high
10, rows (a, x, h, k) flat).
"""

from __future__ import annotations

import torch

from library.dsl import fill

REGI = {"A": 0, "X": 1, "Y": 2, "N": 3, "S": 4}


GRAD_DAMP = 0.1


def damp_grad(x):
    """A cell's output: the same value, its gradient times GRAD_DAMP (the certified files' damp, per cell here); without
    autograd, nothing to scale."""
    if not torch.is_grad_enabled():
        return x
    return x.detach() + GRAD_DAMP * (x - x.detach())


class SteMachine:
    def __init__(self, SBB, ADC, MUL, P: int, B: int = 10, W: int | None = None, damp=None):
        self.SBB, self.ADC, self.MUL, self.P, self.B = SBB, ADC, MUL, P, B
        self.W = P // 2 if W is None else W
        self.damp = damp or damp_grad                                     # every cell's output: GRAD_DAMP's role

    def count(self, kind):
        return {"P": self.P, "B": self.B, "W": self.W}[kind]

    def run(self, regs, c, h, k, items, lib, p: int = 0):
        """Run items from regs [X, 5, P, 10] (or a list of 5 registers); returns (regs [X, 5, P, 10], c, h, k, p)."""
        st = [list(regs.unbind(1)) if torch.is_tensor(regs) else list(regs), c, h, k, p]
        self._items(st, items, lib)
        return (torch.stack(st[0], 1), st[1], st[2], st[3], st[4])

    def _items(self, st, items, lib):
        for it in items:
            if isinstance(it, tuple):
                _, kind, body = it
                if kind in lib:
                    self._items(st, fill(lib[kind], body[0]), lib)
                else:
                    for _ in range(self.count(kind)):
                        self._items(st, body, lib)
            elif it in lib:
                if it.startswith("T."):
                    p = st[4]
                    self._items(st, lib[it], lib)
                    st[4] = p
                else:
                    self._items(st, lib[it], lib)
            else:
                self._step(st, it)

    def _cell(self, st, table, a, x, flags):
        """One cell lookup: the joint one-hot of the inputs times the table."""
        joint = a
        for f in (x,) + flags:
            joint = (joint[:, :, None] * f[:, None, :]).reshape(a.shape[0], -1)
        return joint @ table

    @staticmethod
    def _put(R, p, d):
        """Register R [X, P, 10] with place p replaced by digit d [X, 10] (a new tensor: no in-place writes, so autograd
        keeps every earlier value)."""
        return torch.cat([R[:, :p], d[:, None], R[:, p + 1:]], 1)

    def _step(self, st, tok):
        regs, c, h, k, p = st                                             # regs: a list of 5 registers [X, P, 10]
        B = self.B
        op, _, arg = tok.partition(".")
        regs = list(regs)
        zero = torch.zeros_like(regs[0][:, :1])
        zero[..., 0] = 1
        if op in ("SBB", "ADC"):
            out = self.damp(self._cell(st, self.SBB if op == "SBB" else self.ADC, regs[0][:, p], regs[REGI[arg]][:, p],
                                       (c,)))
            regs[0] = self._put(regs[0], p, out[:, :B])
            st[0], st[1] = regs, out[:, B:]
        elif op == "MUL":
            out = self.damp(self._cell(st, self.MUL, regs[0][:, p], regs[REGI[arg]][:, p], (h, k)))
            regs[0] = self._put(regs[0], p, out[:, :B])
            st[0], st[3] = regs, out[:, B:]
        elif op == "SEL":
            sel = c[:, 1][:, None, None]
            regs[0] = self.damp(sel * regs[4] + (1 - sel) * regs[0])
            st[0] = regs
        elif op == "LDA":
            regs[0] = regs[REGI[arg]]
            st[0] = regs
        elif op == "STA":
            regs[REGI[arg]] = regs[0]
            st[0] = regs
        elif op == "CLA":
            regs[0] = zero.expand_as(regs[0])
            st[0] = regs
        elif op == "CLC":
            c2 = torch.zeros_like(c)
            c2[:, 0] = 1
            st[1] = c2
        elif op == "CLK":
            k2 = torch.zeros_like(k)
            k2[:, 0] = 1
            st[3] = k2
        elif op == "LDH":
            st[2] = regs[REGI[arg]][:, p]
        elif op == "INC":
            st[4] = (p + 1) % self.P
        elif op == "SHL":
            r = REGI[arg]
            regs[r] = torch.cat([zero, regs[r][:, :-1]], 1)
            st[0] = regs
        elif op == "SHR":
            r = REGI[arg]
            regs[r] = torch.cat([regs[r][:, 1:], zero], 1)
            st[0] = regs
        elif op == "BRG":
            top = regs[1][:, -1:]
            regs[0] = torch.cat([top, regs[0][:, :-1]], 1)
            regs[1] = torch.cat([zero, regs[1][:, :-1]], 1)
            st[0] = regs
        else:
            raise ValueError(tok)


def tables_onehot(cells, dtype=torch.float32):
    """The snapped tables of tdsl.Cells as one-hot rows (no gradient; the submission uses straight-through ones)."""
    B = cells.B
    oh = lambda t, n: torch.nn.functional.one_hot(t.long().flatten(), n).to(dtype)
    SBB = torch.cat([oh(cells.sbb_d, B), oh(cells.sbb_c, 2)], 1)
    ADC = torch.cat([oh(cells.adc_d, B), oh(cells.adc_c, 2)], 1)
    MUL = torch.cat([oh(cells.mul_lo, B), oh(cells.mul_hi, B)], 1)
    return SBB, ADC, MUL


def start(X: int, P: int, inputs: dict, device="cpu", dtype=torch.float32):
    """A clean start (zeros everywhere, the flags 0) with inputs {reg: digits [X, <= P]} as one-hots."""
    regs = torch.zeros(X, 5, P, 10, device=device, dtype=dtype)
    regs[..., 0] = 1
    for r, d in inputs.items():
        oh = torch.nn.functional.one_hot(d.long(), 10).to(dtype) if d.dim() == 2 else d
        regs[:, REGI[r], :oh.shape[1]] = oh
    z2 = torch.zeros(X, 2, device=device, dtype=dtype)
    z2[:, 0] = 1
    z10 = torch.zeros(X, 10, device=device, dtype=dtype)
    z10[:, 0] = 1
    return regs, z2, z10.clone(), z10.clone()
