"""The bare counting machine of the inventing study (library/inventing/brm.py), with more input and output registers,
so that it can pose one place of a digit machine: a digit operation with its carry.

Registers (whole numbers, all six always present; a problem's vocabulary only names the ones it uses):
    A  the accumulator digit: an input (the digit at the pointer) and the output digit
    R  the operand digit (another register's digit at the pointer)
    C  the carry or flag: an input (the carry from the place below, or a flag) and the output carry
    H  a digit operand that is the same at every place (the multiplier digit)
    N  the base B (the counting machine's modulus, as N is x^2 mod N's)
    T  scratch
A fresh machine holds the problem's inputs and 0 elsewhere (the counter-machine convention). Values saturate at `cap`.

The instructions are brm.py's, one token each, over a problem's registers:
    INC.r  r <- r + 1      DEC.r  r <- r - 1, stopping at 0      CLR.r  r <- 0      MOV.r.s  r <- s
    L.r[...]  run the body r times (r read once, on entry)        Z.r[...]  run the body if r = 0
Nothing here is digit-specific: no places, no carry logic, no wrap-around. A digit operation's wrap at B has to be
invented from the zero test and the counters, as the inventing study's x^2 mod N had to invent its wrap at N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGS = ("A", "R", "C", "H", "N", "T")
RI = {r: i for i, r in enumerate(REGS)}
A, R, C, H, N, T = range(6)


def prims(regs):
    """The tokens over a set of registers (in REGS order)."""
    regs = [r for r in REGS if r in regs]
    return ([f"INC.{r}" for r in regs] + [f"DEC.{r}" for r in regs] + [f"CLR.{r}" for r in regs]
            + [f"MOV.{r}.{s}" for r in regs for s in regs if r != s])


def loop_kinds(regs):
    regs = [r for r in REGS if r in regs]
    return [f"L.{r}" for r in regs] + [f"Z.{r}" for r in regs]


def regs_of(items, lib, seen=None):
    """The registers a program names, through its routines and merges."""
    seen = set() if seen is None else seen
    out = set()
    for it in items:
        if isinstance(it, tuple):
            kind = it[1]
            if kind in lib:
                if kind not in seen:
                    seen.add(kind)
                    out |= regs_of(lib[kind], lib, seen)
            else:
                out.add(kind[2:])
            out |= regs_of(it[2], lib, seen)
        elif it in lib:
            if it not in seen:
                seen.add(it)
                out |= regs_of(lib[it], lib, seen)
        elif it != "_":
            out |= set(it.split(".")[1:])
    return out


@dataclass
class Machine:
    cap: int = 256          # values saturate here
    limit: int = 20000      # steps a single machine may run before it is dropped


@dataclass
class Batch:
    regs: np.ndarray        # [S, NP, 6] int32
    dead: np.ndarray        # [S] bool: some probe ran out of steps

    @property
    def S(self):
        return self.regs.shape[0]

    def take(self, idx):
        return Batch(self.regs[idx], self.dead[idx])

    def copy(self):
        return Batch(self.regs.copy(), self.dead.copy())

    @staticmethod
    def concat(bs):
        return Batch(np.concatenate([b.regs for b in bs]), np.concatenate([b.dead for b in bs]))

    def features(self):
        S = self.S
        return np.ascontiguousarray(self.regs.astype(np.int16)).view(np.uint8).reshape(S, -1)


def fresh(inputs, NP=None):
    """One fresh machine (S = 1) over the probes: inputs {register: values [NP]}; every other register 0."""
    if NP is None:
        NP = len(next(iter(inputs.values())))
    regs = np.zeros((1, NP, len(REGS)), np.int32)
    for r, v in inputs.items():
        regs[0, :, RI[r]] = np.asarray(v)
    return Batch(regs, np.zeros(1, bool))


def random_states(S, NP, rng, top=12):
    return Batch(rng.integers(0, top, size=(S, NP, len(REGS))).astype(np.int32), np.zeros(S, bool))


def fill(pattern, arg):
    out = []
    for it in pattern:
        if isinstance(it, tuple):
            out.append(("REP", it[1], fill(it[2], arg)))
        else:
            out.append(arg if it == "_" else it)
    return tuple(out)


class _Run:
    """One execution over flat machines M [m, 6], with per-machine step counts (brm._Run over six registers)."""

    def __init__(self, m: Machine, M, lib):
        self.m, self.M, self.lib = m, M, lib
        self.steps = np.zeros(len(M), np.int64)
        self.dead = np.zeros(len(M), bool)

    def items(self, items, idx):
        M, cap = self.M, self.m.cap
        for it in items:
            if len(idx) == 0:
                return
            if isinstance(it, tuple):
                kind, body = it[1], it[2]
                if kind in self.lib:                         # an abstraction applied to its argument
                    self.items(fill(self.lib[kind], body[0]), idx)
                    continue
                r = RI[kind[2:]]
                if kind[0] == "Z":
                    self.items(body, idx[M[idx, r] == 0])
                    continue
                cnt = M[idx, r].copy()                       # L.r: r read once, on entry
                cur, t = idx, 0
                while True:
                    keep = cnt > t
                    if not keep.all():
                        cur, cnt = cur[keep], cnt[keep]
                    if len(cur) == 0:
                        break
                    self.steps[cur] += 1
                    over = self.steps[cur] > self.m.limit
                    if over.any():
                        self.dead[cur[over]] = True
                        cur, cnt = cur[~over], cnt[~over]
                        if len(cur) == 0:
                            break
                    self.items(body, cur)
                    t += 1
                continue
            if it in self.lib:
                self.items(self.lib[it], idx)
                continue
            op, _, arg = it.partition(".")
            self.steps[idx] += 1
            if op == "INC":
                r = RI[arg]
                M[idx, r] = np.minimum(M[idx, r] + 1, cap)
            elif op == "DEC":
                r = RI[arg]
                M[idx, r] = np.maximum(M[idx, r] - 1, 0)
            elif op == "CLR":
                M[idx, RI[arg]] = 0
            elif op == "MOV":
                d, s = arg.split(".")
                M[idx, RI[d]] = M[idx, RI[s]]
            else:
                raise ValueError(it)


def run_items(m: Machine, b: Batch, items, lib):
    """Run items on every probe of every state of b, in place."""
    S, NP, nr = b.regs.shape
    M = b.regs.reshape(S * NP, nr)
    run = _Run(m, M, lib)
    run.items(tuple(items), np.arange(S * NP))
    b.regs = M.reshape(S, NP, nr)
    b.dead = b.dead | run.dead.reshape(S, NP).any(axis=1)


def apply(m, b, item, lib):
    out = b.copy()
    run_items(m, out, (item,), lib)
    return out


def evaluate(prog, lib, inputs, m):
    """Every register's final value per probe [NP, 6], and whether any probe ran out of steps."""
    b = fresh(inputs)
    run_items(m, b, prog, lib)
    return b.regs[0].copy(), bool(b.dead[0])
