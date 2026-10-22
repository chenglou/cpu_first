"""The tiny long-division machine laid out as a grid of tokens, and its compiled looped-transformer weights.

The task is TINYROUTING's division core: base B, moduli N with w base-B digits (top digit nonzero), dividends
v < N * B^k (k quotient digits), answer v mod N. Digits are lowest first.

The machine (CHAIN.md / TINYROUTING.md's truth, with H1's commit-gate chain) on a register of P = w + 1 places:
- the register starts at v's top w digits (below N, since v < N B^k);
- each of the k steps shifts the register up one place and brings the next dividend digit in at place 0;
- then B - 1 links: each subtracts N from the chain register with a ripple of the subtraction table, and the result
  replaces the chain register only if the subtraction did not borrow (the commit gate);
- the answer is the register's low w places.

**As a grid.** Every table application is one token (a "cell"); a row of P + 1 cells holds one register, at columns
c = -1 .. P-1 (column -1 carries the digit brought in next, or a constant zero). Rows, top to bottom:
    N      N's digits (inputs), a zero at place w
    R0     v's top w digits (inputs), a zero at place w; at column -1 the first digit to bring in
    for each step t:  SHIFT(t), then for each link s = 1 .. B-1:  SUB(t, s), SEL(t, s)
The last SEL row of step t holds, at column -1, the digit that step t + 1 brings in. Kinds of cell:
    Z      a constant zero (digit 0, no borrow)
    NIN    an input digit of N
    VIN    an input digit of v
    SHIFT  copy port A                           (the register moved up one place)
    SUB    port A - port B - borrow of port C    (one place of a link's ripple)
    SEL    port C's borrow ? port A : port B     (the commit gate: keep the old chain register if the link borrowed)
Each cell's state is a digit distribution (B) and a borrow distribution (2).

**As a looped transformer.** One block, applied D times to every cell at once (synchronous updates; input cells are
re-injected every tick):
- three attention heads, A, B and C, each copying another cell's whole state into the cell's port of that name. Their
  compiled patterns (relative to the cell at row r, column c):
      SHIFT  A <- (r-1, c-1)
      SUB    A <- (r-1, c)          B <- the N row at column c      C <- (r, c-1)
      SEL    A <- (r-2, c)          B <- (r-1, c)                   C <- the top place of row r-1
- one MLP, the rule: (kind, port A's digit, port B's digit, port C's borrow) -> (digit, borrow). Compiled, its hidden
  layer has one unit per (kind, a, b, c) and its output layer holds the truth tables.
After D = k (1 + (B-1)(P+1)) ticks (the circuit's depth) every cell holds the machine's value, whatever the cells
held before. The answer is read from the last row's places 0 .. w-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

KINDS = ("Z", "NIN", "VIN", "SHIFT", "SUB", "SEL")
Z, NIN, VIN, SHIFT, SUB, SEL = range(6)
SCRATCH = (SHIFT, SUB, SEL)
HEADS = ("A", "B", "C")


def digits(v, B, width):
    v = np.asarray(v, dtype=np.int64).copy()
    out = np.empty((len(v), width), dtype=np.int64)
    for i in range(width):
        out[:, i] = v % B
        v //= B
    return out


def value(d, B):
    """Digit tensors [..., n] (lowest first) -> integers."""
    d = np.asarray(d)
    return (d * (B ** np.arange(d.shape[-1]))).sum(-1)


class Layout:
    """The grid for one size (B, w, k): kinds, coordinates, input wiring, compiled attention, answer cells."""

    def __init__(self, B: int, w: int, k: int, links: int | None = None):
        self.B, self.w, self.k = B, w, k
        links = self.links = B - 1 if links is None else links            # links per step (fewer: a sub-grid)
        P = self.P = w + 1
        C = self.C = P + 1
        rows = [("N", -1, 0), ("R0", -1, 0)]
        for t in range(k):
            rows.append(("SHIFT", t, 0))
            for s in range(1, links + 1):
                rows += [("SUB", t, s), ("SEL", t, s)]
        self.rows = rows
        Rn = self.Rn = len(rows)
        L = self.L = Rn * C
        kind = np.full(L, Z)
        row = np.repeat(np.arange(Rn), C)
        col = np.tile(np.arange(-1, P), Rn)
        step = np.array([rows[r][1] for r in row])
        link = np.array([rows[r][2] for r in row])
        # input cells: where each takes its digit from (-1: a constant zero; i >= 0 an index into N's w digits for
        # NIN cells, into v's w + k digits for VIN cells)
        src = np.full(L, -1)
        cell = lambda r, c: r * C + c + 1
        for c in range(w):
            kind[cell(0, c)] = NIN
            src[cell(0, c)] = c
            kind[cell(1, c)] = VIN
            src[cell(1, c)] = k + c
        kind[cell(1, -1)] = VIN
        src[cell(1, -1)] = k - 1
        for r, (name, t, s) in enumerate(rows):
            if name in ("SHIFT", "SUB", "SEL"):
                for c in range(P):
                    kind[cell(r, c)] = KINDS.index(name)
            if name == "SEL" and s == links and t < k - 1:
                kind[cell(r, -1)] = VIN
                src[cell(r, -1)] = k - 2 - t
        self.kind, self.row, self.col, self.step, self.link, self.src = kind, row, col, step, link, src
        self.is_input = np.isin(kind, (Z, NIN, VIN))
        self.top = col == P - 1
        # compiled attention: for every scratch cell and head, the cell it reads (-1: unused)
        comp = np.full((3, L), -1)
        nrow = 0
        for p in range(L):
            r, c, kd = row[p], col[p], kind[p]
            if kd == SHIFT:
                comp[0, p] = cell(r - 1, c - 1)
            elif kd == SUB:
                comp[0, p] = cell(r - 1, c)
                comp[1, p] = cell(nrow, c)
                comp[2, p] = cell(r, c - 1)
            elif kd == SEL:
                comp[0, p] = cell(r - 2, c)
                comp[1, p] = cell(r - 1, c)
                comp[2, p] = cell(r - 1, P - 1)
        self.comp = comp
        last = Rn - 1
        self.answer = np.array([cell(last, c) for c in range(w)])
        self.depth = k * (1 + links * (P + 1))

    def cell(self, r, c):
        return r * self.C + c + 1

    def describe(self):
        lines = []
        for r, (name, t, s) in enumerate(self.rows):
            tag = name if t < 0 else (f"{name}({t})" if name == "SHIFT" else f"{name}({t},{s})")
            cells = []
            for c in range(-1, self.P):
                p = self.cell(r, c)
                kd = KINDS[self.kind[p]]
                if kd == "NIN":
                    cells.append(f"N{self.src[p]}")
                elif kd == "VIN":
                    cells.append(f"v{self.src[p]}")
                elif kd == "Z":
                    cells.append(".")
                else:
                    cells.append(kd[:2].lower())
            lines.append(f"{tag:>10}  " + " ".join(f"{x:>3}" for x in cells))
        return "\n".join(lines)


class Data:
    """Every (N, v) of a size: N with w digits (top digit nonzero), v < N * B^k. `moduli` restricts N."""

    def __init__(self, B, w, k, moduli=None, sample=0, seed=0):
        self.B, self.w, self.k = B, w, k
        Ns, vs = [], []
        for N in range(max(2, B ** (w - 1)), B ** w):
            if moduli is not None and N not in moduli:
                continue
            vs.append(np.arange(N * B ** k, dtype=np.int64))
            Ns.append(np.full(N * B ** k, N, dtype=np.int64))
        N, v = np.concatenate(Ns), np.concatenate(vs)
        if sample and sample < len(N):
            idx = np.random.default_rng(seed).choice(len(N), sample, replace=False)
            N, v = N[idx], v[idx]
        self.set(N, v)

    def set(self, N, v):
        B, w, k = self.B, self.w, self.k
        self.N, self.v = np.asarray(N), np.asarray(v)
        self.y = self.v % self.N
        self.X = len(self.N)
        self.Nd = torch.from_numpy(digits(self.N, B, w))
        self.vd = torch.from_numpy(digits(self.v, B, w + k))
        self.yd = torch.from_numpy(digits(self.y, B, w))
        self.R0 = self.v // B ** k                                      # the register's start: below N
        return self

    def like(self, N, v):
        out = object.__new__(Data)
        out.B, out.w, out.k = self.B, self.w, self.k
        return out.set(N, v)

    def subset(self, idx):
        return self.like(self.N[idx], self.v[idx])


def truth_tables(B, dtype=torch.float64):
    """The rule's truth for each scratch kind: [3 (SHIFT, SUB, SEL), B (a), B (b), 2 (c), B + 2] probabilities."""
    T = torch.zeros(3, B, B, 2, B + 2, dtype=dtype)
    for a in range(B):
        for b in range(B):
            for c in range(2):
                T[0, a, b, c, a] = 1
                T[0, a, b, c, B] = 1
                x = a - b - c
                T[1, a, b, c, x % B] = 1
                T[1, a, b, c, B + int(x < 0)] = 1
                T[2, a, b, c, a if c else b] = 1
                T[2, a, b, c, B] = 1
    return T


def input_states(lay: Layout, Np, vp, dtype=torch.float64):
    """The input cells' states [X, L, B + 2] from digit distributions of N [X, w, B] and v [X, w + k, B] (one-hot for
    exact digits). Scratch cells start blank: digit 0, no borrow (the constant zero's state)."""
    X = Np.shape[0]
    B = lay.B
    S = torch.zeros(X, lay.L, B + 2, dtype=dtype)
    S[:, :, B] = 1                                                     # no borrow
    S[:, :, 0] = 1                                                     # digit 0 (scratch cells start blank)
    nin = np.where(lay.kind == NIN)[0]
    vin = np.where((lay.kind == VIN) & (lay.src >= 0))[0]
    S[:, nin, :B] = Np[:, lay.src[nin]].to(dtype)
    S[:, vin, :B] = vp[:, lay.src[vin]].to(dtype)
    return S


def onehot(x, n, dtype=torch.float64):
    return F.one_hot(torch.as_tensor(x), n).to(dtype)


def data_inputs(lay: Layout, data: Data, dtype=torch.float64):
    return input_states(lay, onehot(data.Nd, lay.B, dtype), onehot(data.vd, lay.B, dtype), dtype)


def compiled_attention(lay: Layout, dtype=torch.float64):
    """The compiled heads as [3, L, L] one-hot rows. A head a kind does not use, and every head of an input cell,
    reads cell 0 (a constant zero), so every port holds a distribution."""
    A = torch.zeros(3, lay.L, lay.L, dtype=dtype)
    for h in range(3):
        src = np.where(lay.comp[h] >= 0, lay.comp[h], 0)
        A[h, np.arange(lay.L), src] = 1
    return A


def run_compiled(lay: Layout, S_in, tables=None, ticks=None, snap=True, record=False):
    """The compiled looped transformer: D ticks of (three heads, the rule MLP) on every cell. Returns the final states
    (and every tick's states when `record`)."""
    B = lay.B
    dtype = S_in.dtype
    A = compiled_attention(lay, dtype)
    T = truth_tables(B, dtype) if tables is None else tables
    Tcell = torch.zeros(lay.L, 2 * B * B, B + 2, dtype=dtype)
    for i, kd in enumerate(SCRATCH):
        Tcell[lay.kind == kd] = T[i].reshape(2 * B * B, B + 2)
    inp = torch.tensor(lay.is_input)
    S = S_in.clone()
    hist = []
    for _ in range(ticks or lay.depth):
        S = tick_table(S, A, Tcell, B)
        S = torch.where(inp[None, :, None], S_in, S)
        if snap:
            S = snap_state(S, B)
        if record:
            hist.append(S)
    return (S, hist) if record else S


def tick_table(S, A, Tcell, B):
    """One block application with the rule as a table: S [X, L, B + 2], A [3, L, L] (or [3, X, L, L]), Tcell
    [L, 2 B B, B + 2]. The table is applied multilinearly (the expected output over independent port inputs)."""
    if A.dim() == 3:
        H = torch.einsum("hqk,xkd->hxqd", A, S)
    else:
        H = torch.einsum("hxqk,xkd->hxqd", A, S)
    a, b, c = H[0, ..., :B], H[1, ..., :B], H[2, ..., B:]
    X, L = S.shape[:2]
    joint = (a[..., :, None, None] * b[..., None, :, None] * c[..., None, None, :]).reshape(X, L, 2 * B * B)
    return torch.einsum("xpj,pjo->xpo", joint, Tcell)


def snap_state(S, B):
    d = F.one_hot(S[..., :B].argmax(-1), B).to(S.dtype)
    c = F.one_hot(S[..., B:].argmax(-1), 2).to(S.dtype)
    return torch.cat([d, c], -1)


def answer_digits(lay: Layout, S):
    return S[:, lay.answer, :lay.B]


def machine_reference(lay: Layout, data: Data):
    """The machine's own value at every cell, computed directly in integers (not through the network): digit [X, L]
    and borrow [X, L]. Used to check the compile and as the probe's ground truth."""
    B, w, k, P = lay.B, lay.w, lay.k, lay.P
    X = data.X
    dig = np.zeros((X, lay.L), dtype=np.int64)
    bor = np.zeros((X, lay.L), dtype=np.int64)
    Nd = data.Nd.numpy()
    vd = data.vd.numpy()
    for p in range(lay.L):
        if lay.kind[p] == NIN:
            dig[:, p] = Nd[:, lay.src[p]]
        elif lay.kind[p] == VIN and lay.src[p] >= 0:
            dig[:, p] = vd[:, lay.src[p]]
    order = np.argsort(lay.row * lay.C + lay.col, kind="stable")             # rows top-down, columns left-right
    for p in order:
        kd = lay.kind[p]
        if kd in (Z, NIN, VIN):
            continue
        a = lay.comp[0, p]
        if kd == SHIFT:
            dig[:, p] = dig[:, a]
        elif kd == SUB:
            x = dig[:, a] - dig[:, lay.comp[1, p]] - bor[:, lay.comp[2, p]]
            dig[:, p] = x % B
            bor[:, p] = x < 0
        else:
            g = bor[:, lay.comp[2, p]]
            dig[:, p] = np.where(g == 1, dig[:, a], dig[:, lay.comp[1, p]])
    return dig, bor
