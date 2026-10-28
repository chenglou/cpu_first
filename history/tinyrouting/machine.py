"""A tiny long-division machine whose wiring is learnable together with its digit table.

A size is (B, w, k): base B, moduli N with w base-B digits (top digit nonzero), dividends v < N * B^k (k quotient
digits). Digits are LSB first. Every (N, v) of a size is used, so every gradient is exact (float64).

The true machine is H1's chain reduction in miniature (learned_remainder/CHAIN.md):
- read v's digit string and N's digit string;
- the register R (w+1 places) starts at v's top w digits, which are below N;
- each step shifts R up one place and brings the next dividend digit in at place 0: Rp = B R + v_t;
- a chain of B-1 links: trial 0 = Rp, trial s = trial (s-1) - N, each link a ripple of the learned subtraction
  table (a, b, borrow) -> (digit, borrow) from place 0 up, ending in a final borrow;
- the write-back keeps the last trial before the first borrowing link (the subtraction's own borrow decides);
- after the last step the answer is read from R's low w places.

Every connection in that description is a mux with a handful of options; option 0 is the true one:
    vread  v's digit string read as given | reversed
    nread  N's digit string read as given | reversed
    shift  where Rp comes from: up (B R + v_t) | down (R / B + v_t B^w) | stay (v_t replaces place 0) | skip (R)
    a      each link's minuend: prev (the previous trial) | start (Rp) | N
    b      each link's subtrahend: N | prev | start
    keep   which trial is written back: stop (last before the first borrow) | over (the first borrowing trial)
           | all (the last trial) | none (trial 0)
    read   which places are the answer: low (0..w-1) | high (1..w)
One routing weight vector per mux, shared by every step and link (one mux replicated, like the table).

Learned parts: the table's digit logits [B, B, 2, B] and borrow logits [B, B, 2, 2], and the routing. The table is
applied multilinearly to digit distributions (the expected output over independent inputs), as in tinycore.

`run` evaluates H wirings at once: P maps every mux to routing weights [H, n] (rows sum to 1). One-hot rows give
hard programs (the whole-program mixture enumerates them); soft rows mix the options' digit distributions on the
wire (per-connection soft routing); a hybrid enumerates some muxes and mixes the others.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

MUXES = {
    "vread": ("given", "reversed"),
    "nread": ("given", "reversed"),
    "shift": ("up", "down", "stay", "skip"),
    "a": ("prev", "start", "N"),
    "b": ("N", "prev", "start"),
    "keep": ("stop", "over", "all", "none"),
    "read": ("low", "high"),
}
ORDER = tuple(MUXES)


def digits(v, B, width):
    v = np.asarray(v, dtype=np.int64).copy()
    out = np.empty((len(v), width), dtype=np.int64)
    for i in range(width):
        out[:, i] = v % B
        v //= B
    return out


class Data:
    """Every (N, v) of a size: N with w digits (top digit nonzero), v < N * B^k."""

    def __init__(self, B, w, k, sample=0, seed=0):
        self.B, self.w, self.k = B, w, k
        Ns, vs = [], []
        for N in range(max(2, B ** (w - 1)), B ** w):
            vs.append(np.arange(N * B ** k, dtype=np.int64))
            Ns.append(np.full(N * B ** k, N, dtype=np.int64))
        N, v = np.concatenate(Ns), np.concatenate(vs)
        if sample and sample < len(N):
            idx = np.random.default_rng(seed).choice(len(N), sample, replace=False)
            N, v = N[idx], v[idx]
        self.N, self.v = N, v
        self.y = v % N
        self.X = len(N)
        self.Nd = torch.from_numpy(digits(N, B, w))
        self.vd = torch.from_numpy(digits(v, B, w + k))
        self.yd = torch.from_numpy(digits(self.y, B, w))
        self.qd = torch.from_numpy(digits(v // N, B, k))

    def subset(self, idx):
        out = object.__new__(Data)
        out.__dict__.update(self.__dict__)
        idx = np.asarray(idx)
        for a in ("N", "v", "y"):
            setattr(out, a, getattr(self, a)[idx])
        ti = torch.from_numpy(idx)
        for a in ("Nd", "vd", "yd", "qd"):
            setattr(out, a, getattr(self, a)[ti])
        out.X = len(idx)
        return out


def truth_table(B):
    """One-hot (digit, borrow) of a - b - c for every (a, b, c): [B, B, 2, B] and [B, B, 2, 2]."""
    a, b, c = np.meshgrid(np.arange(B), np.arange(B), np.arange(2), indexing="ij")
    x = a - b - c
    return torch.from_numpy(np.eye(B)[x % B]), torch.from_numpy(np.eye(2)[(x < 0).astype(int)])


def transpose_table(TD, TC):
    """The table with its two operands exchanged: (a, b, c) -> T(b, a, c)."""
    return TD.transpose(0, 1), TC.transpose(0, 1)


def onehot(x, n):
    return F.one_hot(x, n).double()


def mix(p, opts):
    """Routing weights p [H, n] over options [H or 1, X, ...] -> the mixed wire [H, X, ...]. A one-hot row that does
    not need a gradient selects without arithmetic when every row picks the same option."""
    if not p.requires_grad:
        col = p.argmax(1)
        if bool((p.max(1).values == 1).all()) and bool((col == col[0]).all()):
            o = opts[int(col[0])]
            return o.expand(p.shape[0], *o.shape[1:])
    out = 0
    for i, o in enumerate(opts):
        out = out + p[:, i].view(-1, *([1] * (o.dim() - 1))) * o
    return out


def ripple(A, Bo, TDC):
    """A - Bo on digit distributions [H, X, P, B] with the table TDC [B*B*2, B+2] (rows (a, b, c); columns digit then
    borrow probabilities), applied multilinearly from place 0 up. Returns the digits [H, X, P, B] and the final
    borrow probability [H, X]."""
    H, X, P, B = A.shape
    c = torch.zeros(H, X, 2)
    c[..., 0] = 1
    out = []
    for j in range(P):
        joint = (A[:, :, j, :, None, None] * Bo[:, :, j, None, :, None] * c[:, :, None, None, :]).reshape(H, X, -1)
        y = joint @ TDC
        out.append(y[..., :B])
        c = y[..., B:]
    return torch.stack(out, 2), c[..., 1]


def keep_weights(borrows):
    """Weights over trials 0..S for the four write-back options, from the links' borrow probabilities [S][H, X].
    stop: last trial before the first borrow; over: the first borrowing trial (the last if none borrows);
    all: the last trial; none: trial 0. Returns a list of 4 tensors [H, X, S+1]."""
    S = len(borrows)
    H, X = borrows[0].shape
    alive = [torch.ones(H, X)]
    for b in borrows:
        alive.append(alive[-1] * (1 - b))
    stop = [alive[s] * borrows[s] for s in range(S)] + [alive[S]]
    over = [torch.zeros(H, X)] + [alive[s - 1] * borrows[s - 1] for s in range(1, S + 1)]
    over[S] = over[S] + alive[S]
    allw = [torch.zeros(H, X)] * S + [torch.ones(H, X)]
    none = [torch.ones(H, X)] + [torch.zeros(H, X)] * S
    return [torch.stack(x, 2) for x in (stop, over, allw, none)]


def route(P, m, t=None, s=None):
    """The routing weights of mux m at step t, link s: an untied mux has its own entry "m@t" or "m@t.s"."""
    if s is not None and f"{m}@{t}.{s}" in P:
        return P[f"{m}@{t}.{s}"]
    if t is not None and f"{m}@{t}" in P:
        return P[f"{m}@{t}"]
    return P[m]


def untied_keys(m, B, k):
    """The positions of an untied mux: shift and keep per step, a and b per step and link."""
    if m in ("shift", "keep"):
        return [f"{m}@{t}" for t in range(k)]
    if m in ("a", "b"):
        return [f"{m}@{t}.{s}" for t in range(k) for s in range(1, B)]
    raise ValueError(f"{m} cannot be untied")


def reduce_step(Rp, Nreg, P, TDC, B, record=None, t=None):
    """The chain and the write-back on the register Rp [H, X, P, B]: returns the new register."""
    trials, borrows, prev = [Rp], [], Rp
    for s in range(1, B):
        A = mix(route(P, "a", t, s), [prev, Rp, Nreg])
        Bo = mix(route(P, "b", t, s), [Nreg, prev, Rp])
        T, c = ripple(A, Bo, TDC)
        if record is not None:
            record.append((A, Bo, T, c))
        trials.append(T)
        borrows.append(c)
        prev = T
    Tr = torch.stack(trials, 2)                                                # [H, X, S+1, P, B]
    kw = keep_weights(borrows)
    opts = [torch.einsum("hxs,hxspb->hxpb", w_, Tr) for w_ in kw]
    return mix(route(P, "keep", t), opts)


def shift_options(R, vt):
    """The four shift options for the register R [H, X, P, B] and the incoming digit vt [H, X, B]."""
    v = vt[:, :, None]
    return [torch.cat([v, R[:, :, :-1]], 2),                                   # up: B R + v_t
            torch.cat([R[:, :, 1:], v], 2),                                    # down: R / B + v_t B^w
            torch.cat([v, R[:, :, 1:]], 2),                                    # stay: v_t replaces place 0
            R]                                                                 # skip


def inputs(data: Data, P):
    """The read digit strings: v [H, X, w+k, B] and N's register [H, X, w+1, B] (top place 0)."""
    B = data.B
    v1 = onehot(data.vd, B)[None]
    vin = mix(P["vread"], [v1, v1.flip(2)])
    n1 = onehot(data.Nd, B)[None]
    nin = mix(P["nread"], [n1, n1.flip(2)])
    zero = torch.zeros(*nin.shape[:2], 1, B)
    zero[..., 0] = 1
    return vin, torch.cat([nin, zero], 2)


def run(data: Data, P, TDC, record=None, regs=None):
    """The machine on every example for H wirings at once. P: mux -> routing weights [H, n]. TDC [B*B*2, B+2].
    Returns the answer's digit distributions [H, X, w, B]. `record` (a list) collects every link's (A, B, T, borrow);
    `regs` (a list) collects every step's (Rp, R) and N's register."""
    B, w, k = data.B, data.w, data.k
    vin, Nreg = inputs(data, P)
    zero = torch.zeros(*vin.shape[:2], 1, B)
    zero[..., 0] = 1
    R = torch.cat([vin[:, :, k:k + w], zero], 2)
    for i, t in enumerate(reversed(range(k))):
        Rp = mix(route(P, "shift", i), shift_options(R, vin[:, :, t]))
        R = reduce_step(Rp, Nreg, P, TDC, B, record, i)
        if regs is not None:
            regs.append((Rp, R, Nreg))
    return mix(P["read"], [R[:, :, :w], R[:, :, 1:w + 1]])


def row_loglik(ans, data: Data):
    """Per-row log-likelihood of the answer digits [H, X] (sum over the w digits)."""
    y = data.yd[None].expand(ans.shape[0], -1, -1)
    return ans.clamp_min(1e-300).log().gather(-1, y[..., None]).squeeze(-1).sum(-1)


def exact_rows(ans, data: Data):
    """Which rows' argmax answer digits are all right [H, X] (bool)."""
    return (ans.argmax(-1) == data.yd[None]).all(-1)


def programs(muxes):
    """Every combination of options of the given muxes: a list of dicts mux -> option index."""
    names = list(muxes)
    return [dict(zip(names, c)) for c in itertools.product(*[range(len(MUXES[m])) for m in names])]


def program_P(progs, learn, fixed=None, H=None):
    """Routing weights for the given programs: one-hot rows [len(progs), n] for every mux in `learn`; every other mux
    is fixed to `fixed[m]` (an option index, default 0 = the true option) or to soft weights [n]."""
    fixed = fixed or {}
    H = H or len(progs)
    P = {}
    for m in ORDER:
        n = len(MUXES[m])
        if m in learn:
            idx = torch.tensor([p[m] for p in progs])
            P[m] = onehot(idx, n)
        else:
            f = fixed.get(m, 0)
            P[m] = (onehot(torch.tensor([f]), n) if isinstance(f, int) else f[None]).expand(H, n)
    for key in progs[0]:                                                       # untied positions ("m@t", "m@t.s")
        if "@" in key and key in learn:
            P[key] = onehot(torch.tensor([p[key] for p in progs]), len(MUXES[key.split("@")[0]]))
    return P


def table_TDC(TD, TC):
    B = TD.shape[0]
    return torch.cat([TD, TC], -1).reshape(B * B * 2, B + 2)


def true_program():
    return {m: 0 for m in ORDER}


def describe(prog):
    return " ".join(f"{m}={MUXES[m.split('@')[0]][v]}" for m, v in prog.items())


@torch.no_grad()
def hard_exact(data: Data, prog, TD, TC, chunk=4096):
    """Exact-match share of the hard machine: program `prog` (mux -> option; missing muxes true), the snapped tables
    (argmax digit and borrow)."""
    B = data.B
    TDh = onehot(TD.argmax(-1), B)
    TCh = onehot(TC.argmax(-1), 2)
    TDC = table_TDC(TDh, TCh)
    full = true_program()
    full.update(prog)
    n = 0
    for i in range(0, data.X, chunk):
        d = data if data.X <= chunk else data.subset(np.arange(i, min(i + chunk, data.X)))
        P = program_P([full], tuple(ORDER) + tuple(k_ for k_ in full if "@" in k_))
        n += exact_rows(run(d, P, TDC), d).sum().item()
    return n / data.X
