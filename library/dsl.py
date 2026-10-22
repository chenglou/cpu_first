"""A primitive digit machine, and programs over it, for library learning (LIBRARY.md).

The machine. A setting is (B, W): base B, operands of W digits, registers of P = 2W places, digits LSB first.
- Five registers A (the accumulator, where answers are read), X, Y, N (inputs) and S (scratch), P digits each.
- A pointer p (0..P-1, shared by every probe: control flow never depends on data), a borrow/carry bit c, a
  multiplier digit h and a multiply carry digit k.
- A fresh machine has p = 0; the inputs in X, Y, N (a task's unused inputs are random); A, S, c, h, k random.
  Starting dirty makes every found program set up what it uses, so its pieces work wherever they are reused.

The primitive instructions (each one token):
    SBB.r   (A[p], c) <- A[p] - r[p] - c            the subtraction cell, r in X Y N S
    ADC.r   (A[p], c) <- A[p] + r[p] + c            the addition cell, r in X Y N S
    MUL.r   (A[p], k) <- A[p] + r[p] * h + k        the multiply cell (low digit, high digit), r in X Y N
    SEL.S   A <- S where c = 1                      the keep/discard gate, on the borrow
    LDA.r   A <- r (r in X Y N S)     STA.r  r <- A (r in X Y S)     CLA  A <- 0
    CLC     c <- 0      CLK  k <- 0      LDH.r  h <- r[p] (r in X Y)
    INC     p <- p + 1 mod P
    SHL.r   r <- r * B mod B^P     SHR.r  r <- r div B     (r in A X Y)
    BRG.X   A <- A * B + X[P-1], X <- X * B mod B^P      (bring the next digit in: a two-register shift)
and loops, REP.P / REP.B / REP.W [body]: run the body P, B or W times.

A program is a tuple of items; an item is a token (a primitive or a library macro, str) or a loop
("REP", kind, body). A macro named T.<task> is a routine (a solved task's whole program, wake_sleep.promote): calling
it restores the pointer afterwards, so routines compose whatever their loops do to the pointer. Its cost is its token count in prefix form with the loop's arity in its header:
a token is 1, a loop is 1 + its body's cost. So REP.P[SBB.N INC] costs 3.

Batches. `Batch` holds S machine states of one setting at once, each over NP probes:
regs [S, 5, P, NP] uint8, c / h / k [S, NP] uint8, p [S] int64. Every instruction runs on a whole batch.
Cells are pluggable (`Cells`): exact tables by default; learned tables (LIBRARY.md §7) swap in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REGS = ("A", "X", "Y", "N", "S")
RI = {r: i for i, r in enumerate(REGS)}
A, X, Y, N, S = range(5)

PRIMS = (
    [f"SBB.{r}" for r in "XYNS"] + [f"ADC.{r}" for r in "XYNS"] + [f"MUL.{r}" for r in "XYN"]
    + ["SEL.S"] + [f"LDA.{r}" for r in "XYNS"] + [f"STA.{r}" for r in "XYS"] + ["CLA", "CLC", "CLK"]
    + [f"LDH.{r}" for r in "XY"] + ["INC"] + [f"SHL.{r}" for r in "AXY"] + [f"SHR.{r}" for r in "AXY"] + ["BRG.X"]
)
LOOP_KINDS = ("P", "B", "W")


# ---------------------------------------------------------------------------------------------------------------
# numbers <-> digits

def to_digits(v, B, P):
    v = np.asarray(v, dtype=object)
    out = np.zeros(v.shape + (P,), np.uint8)
    for i in range(P):
        out[..., i] = (v % B).astype(np.uint8)
        v = v // B
    return out


def from_digits(d, B):
    d = np.asarray(d)
    out = np.zeros(d.shape[:-1], dtype=object)
    for i in reversed(range(d.shape[-1])):
        out = out * B + d[..., i].astype(object)
    return out


# ---------------------------------------------------------------------------------------------------------------
# cells

class Cells:
    """The digit cells as tables. Exact by default; `sbb`, `adc`, `mul` can be replaced by learned tables.

    sbb[a, b, c] -> (digit, borrow); adc[a, b, c] -> (digit, carry); mul[a, x, h, k] -> (low, high)."""

    def __init__(self, B):
        self.B = B
        a, b, c = np.meshgrid(np.arange(B), np.arange(B), np.arange(2), indexing="ij")
        d = a - b - c
        self.sbb_d, self.sbb_c = (d % B).astype(np.uint8), (d < 0).astype(np.uint8)
        d = a + b + c
        self.adc_d, self.adc_c = (d % B).astype(np.uint8), (d >= B).astype(np.uint8)
        a, x, h, k = np.meshgrid(*(np.arange(B),) * 4, indexing="ij")
        t = a + x * h + k
        self.mul_lo, self.mul_hi = (t % B).astype(np.uint8), (t // B).astype(np.uint8)

    def exact(self):
        ref = Cells(self.B)
        return all(np.array_equal(getattr(self, n), getattr(ref, n))
                   for n in ("sbb_d", "sbb_c", "adc_d", "adc_c", "mul_lo", "mul_hi"))


# ---------------------------------------------------------------------------------------------------------------
# batches

@dataclass
class Setting:
    B: int
    W: int
    cells: Cells = None

    def __post_init__(self):
        self.P = 2 * self.W
        if self.cells is None:
            self.cells = Cells(self.B)

    def count(self, kind):
        return {"P": self.P, "B": self.B, "W": self.W}[kind]

    def __repr__(self):
        return f"({self.B},{self.W})"


@dataclass
class Batch:
    regs: np.ndarray   # [S, 5, P, NP] uint8
    c: np.ndarray      # [S, NP]
    h: np.ndarray
    k: np.ndarray
    p: np.ndarray      # [S] int64

    @property
    def S(self):
        return self.regs.shape[0]

    def copy(self):
        return Batch(self.regs.copy(), self.c.copy(), self.h.copy(), self.k.copy(), self.p.copy())

    def take(self, idx):
        return Batch(self.regs[idx], self.c[idx], self.h[idx], self.k[idx], self.p[idx])

    def repeat(self, n):
        return Batch(np.repeat(self.regs, n, 0), np.repeat(self.c, n, 0), np.repeat(self.h, n, 0),
                     np.repeat(self.k, n, 0), np.repeat(self.p, n, 0))

    @staticmethod
    def concat(bs):
        return Batch(*(np.concatenate([getattr(b, f) for b in bs], 0) for f in ("regs", "c", "h", "k", "p")))

    def features(self):
        S = self.S
        return np.concatenate([self.regs.reshape(S, -1), self.c, self.h, self.k,
                               self.p.astype(np.uint8)[:, None]], axis=1)


def fresh(setting, inputs, rng, NP=None):
    """One fresh machine (S = 1) over NP probes: inputs {reg: [NP, P] digits}; everything else random."""
    B, P = setting.B, setting.P
    if NP is None:
        NP = next(iter(inputs.values())).shape[0]
    regs = rng.integers(0, B, size=(1, 5, P, NP)).astype(np.uint8)
    for r, v in inputs.items():
        regs[0, RI[r]] = np.asarray(v, np.uint8).T
    c = rng.integers(0, 2, size=(1, NP)).astype(np.uint8)
    h = rng.integers(0, B, size=(1, NP)).astype(np.uint8)
    k = rng.integers(0, B, size=(1, NP)).astype(np.uint8)
    return Batch(regs, c, h, k, np.zeros(1, np.int64))


def random_states(setting, S, NP, rng):
    """S fully random machines (random pointer too): the test bed for deciding when two items are the same."""
    B, P = setting.B, setting.P
    return Batch(rng.integers(0, B, size=(S, 5, P, NP)).astype(np.uint8),
                 rng.integers(0, 2, size=(S, NP)).astype(np.uint8),
                 rng.integers(0, B, size=(S, NP)).astype(np.uint8),
                 rng.integers(0, B, size=(S, NP)).astype(np.uint8),
                 rng.integers(0, P, size=S).astype(np.int64))


# ---------------------------------------------------------------------------------------------------------------
# the instructions (in place on a batch)

def _cell(st, b, reg):
    ar = np.arange(b.S)
    return ar, b.regs[ar, A, b.p], b.regs[ar, reg, b.p]


def _shl(b, r):
    b.regs[:, r, 1:] = b.regs[:, r, :-1].copy()
    b.regs[:, r, 0] = 0


def _shr(b, r):
    b.regs[:, r, :-1] = b.regs[:, r, 1:].copy()
    b.regs[:, r, -1] = 0


def step(st: Setting, b: Batch, tok: str):
    """Run one primitive on batch b in place."""
    cl = st.cells
    op, _, arg = tok.partition(".")
    if op == "SBB" or op == "ADC":
        ar, a, x = _cell(st, b, RI[arg])
        if op == "SBB":
            d, c = cl.sbb_d[a, x, b.c], cl.sbb_c[a, x, b.c]
        else:
            d, c = cl.adc_d[a, x, b.c], cl.adc_c[a, x, b.c]
        b.regs[ar, A, b.p] = d
        b.c = c
    elif op == "MUL":
        ar, a, x = _cell(st, b, RI[arg])
        lo, hi = cl.mul_lo[a, x, b.h, b.k], cl.mul_hi[a, x, b.h, b.k]
        b.regs[ar, A, b.p] = lo
        b.k = hi
    elif op == "SEL":
        b.regs[:, A] = np.where(b.c[:, None, :].astype(bool), b.regs[:, S], b.regs[:, A])
    elif op == "LDA":
        b.regs[:, A] = b.regs[:, RI[arg]]
    elif op == "STA":
        b.regs[:, RI[arg]] = b.regs[:, A]
    elif op == "CLA":
        b.regs[:, A] = 0
    elif op == "CLC":
        b.c = np.zeros_like(b.c)
    elif op == "CLK":
        b.k = np.zeros_like(b.k)
    elif op == "LDH":
        ar = np.arange(b.S)
        b.h = b.regs[ar, RI[arg], b.p].copy()
    elif op == "INC":
        b.p = (b.p + 1) % st.P
    elif op == "SHL":
        _shl(b, RI[arg])
    elif op == "SHR":
        _shr(b, RI[arg])
    elif op == "BRG":
        top = b.regs[:, X, -1].copy()
        _shl(b, A)
        b.regs[:, A, 0] = top
        _shl(b, X)
    else:
        raise ValueError(tok)


def fill(pattern, arg):
    """An abstraction's pattern with its hole "_" replaced by arg (an item)."""
    out = []
    for it in pattern:
        if isinstance(it, tuple):
            out.append(("REP", it[1], fill(it[2], arg)))
        else:
            out.append(arg if it == "_" else it)
    return tuple(out)


def run_items(st, b, items, lib):
    """Run a sequence of items on batch b in place. lib: macro name -> items (an abstraction's items hold "_")."""
    for it in items:
        if isinstance(it, tuple):
            _, kind, body = it
            if kind in lib:         # an abstraction applied to its argument
                run_items(st, b, fill(lib[kind], body[0]), lib)
            else:
                for _ in range(st.count(kind)):
                    run_items(st, b, body, lib)
        elif it in lib:
            if it.startswith("T."):     # a routine (a solved task's program) restores the pointer it was called with
                p = b.p.copy()
                run_items(st, b, lib[it], lib)
                b.p = p
            else:
                run_items(st, b, lib[it], lib)
        else:
            step(st, b, it)


def apply(st, b, item, lib):
    """A new batch: item applied to every state of b."""
    out = b.copy()
    run_items(st, out, (item,), lib)
    return out


# ---------------------------------------------------------------------------------------------------------------
# programs

def is_loop(it):
    return isinstance(it, tuple)


def cost(items):
    return sum(1 + cost(it[2]) if is_loop(it) else 1 for it in items)


def item_cost(it):
    return cost((it,))


def expand(items, lib):
    """The program with every macro replaced by its definition, recursively (primitives and loops only)."""
    out = []
    for it in items:
        if is_loop(it) and it[1] in lib:
            out.extend(expand(fill(lib[it[1]], it[2][0]), lib))
        elif is_loop(it):
            out.append(("REP", it[1], tuple(expand(it[2], lib))))
        elif it in lib:
            out.extend(expand(lib[it], lib))
        else:
            out.append(it)
    return tuple(out)


def show(items):
    parts = []
    for it in items:
        if is_loop(it):
            parts.append(f"{it[1]}[{show(it[2])}]")
        else:
            parts.append(it)
    return " ".join(parts)


def parse(s):
    """Inverse of show: 'LDA.X CLC P[SBB.N INC]' -> items."""
    toks = s.replace("[", " [ ").replace("]", " ] ").split()
    pos = 0

    def seq():
        nonlocal pos
        out = []
        while pos < len(toks) and toks[pos] != "]":
            t = toks[pos]
            if pos + 1 < len(toks) and toks[pos + 1] == "[":
                pos += 2
                body = seq()
                assert toks[pos] == "]"
                pos += 1
                out.append(("REP", t, tuple(body)))
            else:
                out.append(t)
                pos += 1
        return out

    items = seq()
    assert pos == len(toks), s
    return tuple(items)


def raw_count(n_tokens, n_kinds, L, max_arity=None):
    """How many syntactically valid programs of cost exactly L: sequences of items, a token costing 1 and a loop
    1 + its body (a nonempty sequence of at most max_arity items). Exact integers."""
    from functools import lru_cache

    @lru_cache(None)
    def seq(c, n):          # sequences of exactly n items, total cost c
        if n == 0:
            return 1 if c == 0 else 0
        return sum(item(ci) * seq(c - ci, n - 1) for ci in range(1, c + 1))

    @lru_cache(None)
    def item(c):
        if c == 1:
            return n_tokens
        top = (c - 1) if max_arity is None else min(c - 1, max_arity)
        return n_kinds * sum(seq(c - 1, n) for n in range(1, top + 1))

    return sum(seq(L, n) for n in range(0, L + 1))
