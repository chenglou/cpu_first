"""A bare register machine for the inventing-learner test (INVENTING.md): nothing digit-specific.

Registers hold whole numbers: X and N (the inputs), A (where the answer is read), T and U (scratch). A fresh machine
has X = x, N = N and the rest 0, the counter-machine convention. Values saturate at `cap`.

The instructions, one token each:
    INC.r      r <- r + 1                 DEC.r   r <- r - 1, stopping at 0          CLR.r   r <- 0
    MOV.r.s    r <- s
and two kinds of block, written like the digit DSL's loops (dsl.py), so merge.py's BPE and MDL apply unchanged:
    L.r[...]   run the body r times, r read once on entry (so every program halts)
    Z.r[...]   run the body if r = 0
That is the LOOP language of Meyer and Ritchie plus a zero test: it computes the primitive recursive functions.
x^2 mod N is 8 tokens in it, for instance a modular counter: MOV.T.N L.X[L.X[INC.A DEC.T Z.T[CLR.A MOV.T.N]]].

A batch holds S machine states over NP probes: regs [S, NP, 5] int32. Every probe of every state runs as its own
machine (loop counts differ between probes), and a machine that runs more than `limit` steps is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGS = ("X", "N", "A", "T", "U")
RI = {r: i for i, r in enumerate(REGS)}
X, N, A, T, U = range(5)

PRIMS = ([f"INC.{r}" for r in REGS] + [f"DEC.{r}" for r in REGS] + [f"CLR.{r}" for r in REGS]
         + [f"MOV.{r}.{s}" for r in REGS for s in REGS if r != s])
LOOP_KINDS = tuple(f"L.{r}" for r in REGS) + tuple(f"Z.{r}" for r in REGS)


@dataclass
class Machine:
    cap: int = 128          # values saturate here
    limit: int = 20000      # steps a single machine may run before it is dropped


@dataclass
class Batch:
    regs: np.ndarray        # [S, NP, 5] int32
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


def fresh(xs, Ns):
    """One fresh machine (S = 1) over the probes (x, N)."""
    NP = len(xs)
    regs = np.zeros((1, NP, 5), np.int32)
    regs[0, :, X] = xs
    regs[0, :, N] = Ns
    return Batch(regs, np.zeros(1, bool))


def random_states(S, NP, rng, top=12):
    return Batch(rng.integers(0, top, size=(S, NP, 5)).astype(np.int32), np.zeros(S, bool))


def fill(pattern, arg):
    out = []
    for it in pattern:
        if isinstance(it, tuple):
            out.append(("REP", it[1], fill(it[2], arg)))
        else:
            out.append(arg if it == "_" else it)
    return tuple(out)


class _Run:
    """One execution over flat machines R [M, 5], with per-machine step counts."""

    def __init__(self, m: Machine, R, lib):
        self.m, self.R, self.lib = m, R, lib
        self.steps = np.zeros(len(R), np.int64)
        self.dead = np.zeros(len(R), bool)

    def items(self, items, idx):
        R, cap = self.R, self.m.cap
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
                    self.items(body, idx[R[idx, r] == 0])
                    continue
                cnt = R[idx, r].copy()                       # L.r: r read once, on entry
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
                R[idx, r] = np.minimum(R[idx, r] + 1, cap)
            elif op == "DEC":
                r = RI[arg]
                R[idx, r] = np.maximum(R[idx, r] - 1, 0)
            elif op == "CLR":
                R[idx, RI[arg]] = 0
            elif op == "MOV":
                d, s = arg.split(".")
                R[idx, RI[d]] = R[idx, RI[s]]
            else:
                raise ValueError(it)


def run_items(m: Machine, b: Batch, items, lib):
    """Run items on every probe of every state of b, in place."""
    S, NP, _ = b.regs.shape
    R = b.regs.reshape(S * NP, 5)
    run = _Run(m, R, lib)
    run.items(tuple(items), np.arange(S * NP))
    b.regs = R.reshape(S, NP, 5)
    b.dead = b.dead | run.dead.reshape(S, NP).any(axis=1)


def apply(m, b, item, lib):
    out = b.copy()
    run_items(m, out, (item,), lib)
    return out


def evaluate(prog, lib, xs, Ns, m=None):
    """A's value for each probe (x, N), and whether any probe ran out of steps."""
    m = m or Machine(cap=max(64, 4 * int(max(Ns)) ** 2), limit=10 ** 6)
    b = fresh(np.asarray(xs), np.asarray(Ns))
    run_items(m, b, prog, lib)
    return b.regs[0, :, A].copy(), bool(b.dead[0])
