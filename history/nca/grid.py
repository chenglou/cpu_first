"""The worksheet grid of a neural cellular automaton (NCA): the bridge's long division, laid out so that every cell
runs the same rule on its 3 x 3 neighbourhood.

The task is the bridge's tiny division core (bridge/grid.py): base B, moduli N with w base-B digits (top digit
nonzero), dividends v < N * B^k, answer v mod N. Digits are lowest first.

**The grid.** A column is a place of one long register of k + w + 1 places, lowest at the left: places 0 .. k - 1
hold the dividend's low digits that are still to be brought in (the queue), places k .. k + w are the machine's
register (the window). Rows, top to bottom:
    PROMPT   the prompt, written at tick 0: digit channel d = v's digits (place k + w is 0), channel n = N's digits
             at places k .. k + w - 1 (0 elsewhere)
    for each step t:  SHIFT, then for each link s = 1 .. B - 1:  SUB, SEL
The answer is the last row's d at places k .. k + w - 1. Every row is uniform across its places: subtracting N's
conveyed digit 0 from a queue place with no borrow in is the identity, so the queue rides along unchanged and
enters the window one place per SHIFT, the way a shift register brings digits in. (The bridge instead wrote each
digit to bring in as a fresh input cell at column -1 of the row that needs it.) Around the grid is a ring of OUT
cells holding the blank state (every channel at value 0); a cell sees its neighbours' kinds, so it can tell the
edge.

**The compiled rule** (`compiled_clauses`), relative to the cell: UL, U, L, R are its up-left, up, left and right
neighbours. Channels: d (digit), n (N's digit, conveyed down), b (borrow), m (memo: the minuend), g (gate):
    SHIFT  d = UL.d                              (the register moved up one place; the left edge brings in 0)
    SUB    d, b = U.d - U.n - L.b  (digit, borrow)   m = U.d   (one place of the link's ripple, left to right)
    SEL    g = (R is the edge) ? U.b : R.g       (the link's final borrow, passed right to left)
           d = g ? U.m : U.d                     (keep the old register if the link borrowed)
    every scratch kind:  n = U.n                 (N's digits conveyed straight down)
Channels a kind does not set are 0. Every read is inside the 3 x 3 neighbourhood; the bridge's long-range ports
(N's row, the top place of the row above, two rows up) became the conveyor n, the gate chain g and the memo m.
"""

from __future__ import annotations

import numpy as np
import torch

KINDS = ("OUT", "PROMPT", "SHIFT", "SUB", "SEL", "SPLIT", "MUL")
OUT, PROMPT, SHIFT, SUB, SEL, SPLIT, MUL = range(len(KINDS))
NK = 5                                          # the division's kinds; the square adds SPLIT and MUL (NK_SQ)
NK_SQ = 7
SCRATCH = (SHIFT, SUB, SEL, SPLIT, MUL)
# the 3 x 3 neighbourhood, in the order the rule's input lists them
OFFS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))
POS = {"UL": 0, "U": 1, "UR": 2, "L": 3, "S": 4, "R": 5, "DL": 6, "D": 7, "DR": 8}


def digits(v, B, width):
    v = np.asarray(v, dtype=np.int64).copy()
    out = np.empty((len(v), width), dtype=np.int64)
    for i in range(width):
        out[:, i] = v % B
        v //= B
    return out


def value(d, B):
    d = np.asarray(d)
    return (d * (B ** np.arange(d.shape[-1]))).sum(-1)


class Fmt:
    """A cell's state: named discrete channels, each a distribution over its values, concatenated."""

    def __init__(self, chans):
        self.chans = tuple(chans)
        self.size = {n: s for n, s in self.chans}
        self.off = {}
        o = 0
        for n, s in self.chans:
            self.off[n] = o
            o += s
        self.D = o

    def sl(self, name):
        return slice(self.off[name], self.off[name] + self.size[name])

    def __repr__(self):
        return "Fmt(" + ", ".join(f"{n}:{s}" for n, s in self.chans) + ")"


def compiled_fmt(B, task="div", minimal=False):
    """The compiled machine's channels; `minimal` keeps only d, n, b (enough when the rule reads the bridge's
    long-range ports, `reach` "ports")."""
    ch = [("d", B), ("n", B), ("b", 2)] if minimal else [("d", B), ("n", B), ("b", 2), ("m", B), ("g", 2)]
    if task == "sq":
        ch += [("a", B), ("q", B), ("s", B), ("c", B)]
    return Fmt(ch)


class Layout:
    """The grid for one size (B, w, k): row kinds, places, the prompt's cells, the answer's cells. `links` < B - 1
    gives a sub-grid (fewer links per step), as the bridge's one-link law uses. `task` "sq" is x^2 mod N with x < N
    (square.py's worksheet): the rows SPLIT and w MUL rows square x into the long register, then the division's rows
    reduce it, with k = w."""

    def __init__(self, B: int, w: int, k: int, links: int | None = None, task: str = "div"):
        self.task = task
        if task == "sq":
            k = w
        self.B, self.w, self.k = B, w, k
        self.links = links = B - 1 if links is None else links
        self.C = C = k + w + 1                                       # places 0 .. k + w
        rows = [("PROMPT", -1, 0)]
        if task == "sq":
            rows.append(("SPLIT", -1, 0))
            rows += [("MUL", i, 0) for i in range(w)]
        for t in range(k):
            rows.append(("SHIFT", t, 0))
            for s in range(1, links + 1):
                rows += [("SUB", t, s), ("SEL", t, s)]
        self.rows = rows
        self.R = R = len(rows)
        self.kind = np.array([[KINDS.index(name)] * C for name, _, _ in rows])        # [R, C]
        self.prompt = self.kind == PROMPT
        self.win = (k, k + w)                                        # the register's places, top included
        self.answer = (R - 1, np.arange(k, k + w))                   # row, places
        self.depth = settle_depth(self)

    def key(self):
        return (self.B, self.w, self.k, self.links, self.task)

    def describe(self):
        lines = []
        for r, (name, t, s) in enumerate(self.rows):
            tag = name if t < 0 else (f"{name}({t})" if name in ("SHIFT", "MUL") else f"{name}({t},{s})")
            lines.append(f"{tag:>10}  " + " ".join(f"{name[:2].lower():>3}" for _ in range(self.C)))
        return "\n".join(lines)


# the cells each kind's compiled rule reads, to find the settling depth
READS = {SHIFT: ("UL", "U"), SUB: ("U", "L"), SEL: ("U", "R"), SPLIT: ("U",), MUL: ("U", "UL", "UR", "L")}
EXTRA_POS = {"NC": 0, "U2": 1, "UT": 2}                   # the long-range reads' order in model.extra_inputs


def settle_depth(lay: Layout) -> int:
    """Ticks until every cell holds its final value whatever the scratch cells started as: a cell is settled one
    tick after every cell its compiled rule reads (the edge and the prompt are settled from the start)."""
    R, C = lay.R, lay.C
    done = np.full((R, C), -1)
    done[lay.prompt] = 0
    t = 0
    while (done < 0).any():
        t += 1
        new = done.copy()
        for r in range(R):
            for c in range(C):
                if done[r, c] >= 0:
                    continue
                ok = True
                for p in READS[lay.kind[r, c]]:
                    dr, dc = OFFS[POS[p]]
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < R and 0 <= cc < C and done[rr, cc] < 0:
                        ok = False
                if ok:
                    new[r, c] = t
        done = new
        if t > 10 * R * C:
            raise RuntimeError("does not settle")
    return int(done.max())


class Data:
    """Every (N, v) of a size: N with w digits (top digit nonzero), v < N * B^k (the bridge's Data). `moduli`
    restricts N; `sample` draws that many."""

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
        self.N, self.v = np.asarray(N, dtype=np.int64), np.asarray(v, dtype=np.int64)
        self.y = self.v % self.N
        self.X = len(self.N)
        self.Nd = digits(self.N, B, w)
        self.vd = digits(self.v, B, w + k)
        self.yd = digits(self.y, B, w)
        self.R0 = self.v // B ** k                                   # the register's start: below N
        return self

    def like(self, N, v):
        out = object.__new__(Data)
        out.B, out.w, out.k = self.B, self.w, self.k
        return out.set(N, v)

    def subset(self, idx):
        return self.like(self.N[idx], self.v[idx])


def prompt_digits(lay: Layout, Nd, vd):
    """The prompt row's digits per place: d [X, C] (v, then 0 at the top place; for the square, x at places
    0 .. w - 1) and n [X, C] (N at the window's low w places, 0 elsewhere)."""
    X = len(Nd)
    k, w = lay.k, lay.w
    d = np.zeros((X, lay.C), dtype=np.int64)
    d[:, :vd.shape[1]] = vd
    n = np.zeros((X, lay.C), dtype=np.int64)
    n[:, k:k + w] = Nd
    return d, n


def machine_reference(lay: Layout, data, reach="3x3"):
    """The compiled machine's value of every channel at every cell, computed directly on integer digit arrays (not
    through any network): dict channel -> [X, R, C]. The probe's ground truth. With a long-range `reach` the
    channels it makes unnecessary are 0 in the scratch rows (n for "ncol"; n, m, g for "ports")."""
    B, X, R, C = lay.B, data.X, lay.R, lay.C
    chans = ("d", "n", "b", "m", "g") + (("a", "q", "s", "c") if lay.task == "sq" else ())
    ref = {ch: np.zeros((X, R, C), dtype=np.int64) for ch in chans}
    d0, n0 = prompt_digits(lay, data.Nd, data.vd)
    ref["d"][:, 0], ref["n"][:, 0] = d0, n0
    for r in range(1, R):
        kd = lay.kind[r, 0]
        up = ref["d"][:, r - 1]
        ref["n"][:, r] = n0
        if kd == SHIFT:
            ref["d"][:, r, 1:] = up[:, :-1]
        elif kd == SUB:
            bor = np.zeros(X, dtype=np.int64)
            for c in range(C):
                x = up[:, c] - n0[:, c] - bor
                ref["d"][:, r, c] = x % B
                bor = (x < 0).astype(np.int64)
                ref["b"][:, r, c] = bor
            ref["m"][:, r] = up
        elif kd == SEL:
            g = ref["b"][:, r - 1, C - 1]
            ref["g"][:, r] = g[:, None]
            ref["d"][:, r] = np.where(g[:, None] == 1, ref["m"][:, r - 1], up)
        elif kd == SPLIT:
            ref["a"][:, r] = up
            ref["q"][:, r] = up
        elif kd == MUL:
            ua, uq = ref["a"][:, r - 1], ref["q"][:, r - 1]
            xi = uq[:, 0]
            carry = np.zeros(X, dtype=np.int64)
            for c in range(C):
                t = up[:, c] + ua[:, c] * xi + carry
                ref["d"][:, r, c] = t % B
                carry = t // B
                ref["c"][:, r, c] = carry
            ref["s"][:, r] = xi[:, None]
            ref["a"][:, r, 1:] = ua[:, :-1]
            ref["q"][:, r, :-1] = uq[:, 1:]
    unused = {"3x3": (), "ncol": ("n",), "ports": ("n", "m", "g")}[reach]
    for ch in unused:
        ref[ch][:, 1:] = 0
    return ref


def compiled_clauses(B, task="div", reach="3x3", chans=None):
    """The compiled rule as AND clauses: (kind, [(position, channel, value, positive)], [(position, kind,
    positive)], {channel: value}). One clause fires for every (cell, channel) on exact states; `compile.py` turns
    each into one ReLU unit. Every channel a kind does not set gets a clause setting it to 0.
    `reach` says what a cell reads beyond its 3 x 3 neighbourhood (model.Cfg.extra): "ncol" N's digit straight from
    the prompt cell of its column (position NC), so nothing conveys N; "ports" also the cell two rows up (U2) and the
    top place of the row above (UT), the bridge's long-range ports, so the select reads the old register and the
    final borrow directly and needs neither the memo nor the gate chain."""
    cl = []
    nsrc = "U" if reach == "3x3" else "NC"
    for j in range(B):
        cl.append((SHIFT, [("UL", "d", j, True)], [], {"d": j}))
        if reach == "3x3":
            for kd in SCRATCH:
                if task == "sq" or kd in (SHIFT, SUB, SEL):
                    cl.append((kd, [("U", "n", j, True)], [], {"n": j}))
    for a in range(B):
        for nn_ in range(B):
            for c in range(2):
                x = a - nn_ - c
                o = {"d": x % B, "b": int(x < 0)}
                if reach != "ports":
                    o["m"] = a
                cl.append((SUB, [("U", "d", a, True), (nsrc, "n", nn_, True), ("L", "b", c, True)], [], o))
    if reach == "ports":
        for j in range(B):
            cl.append((SEL, [("UT", "b", 1, True), ("U2", "d", j, True)], [], {"d": j}))
            cl.append((SEL, [("UT", "b", 0, True), ("U", "d", j, True)], [], {"d": j}))
    else:
        for j in range(B):
            cl.append((SEL, [("U", "b", 1, True), ("U", "m", j, True)], [("R", OUT, True)], {"d": j}))
            cl.append((SEL, [("U", "b", 0, True), ("U", "d", j, True)], [("R", OUT, True)], {"d": j}))
            cl.append((SEL, [("R", "g", 1, True), ("U", "m", j, True)], [("R", OUT, False)], {"d": j}))
            cl.append((SEL, [("R", "g", 0, True), ("U", "d", j, True)], [("R", OUT, False)], {"d": j}))
        for c in range(2):
            cl.append((SEL, [("U", "b", c, True)], [("R", OUT, True)], {"g": c}))
            cl.append((SEL, [("R", "g", c, True)], [("R", OUT, False)], {"g": c}))
    # the prompt holds itself (used only when the prompt is not re-injected after tick 0)
    for j in range(B):
        cl.append((PROMPT, [("S", "d", j, True)], [], {"d": j}))
        cl.append((PROMPT, [("S", "n", j, True)], [], {"n": j}))
    if task == "sq":
        for j in range(B):
            cl.append((SPLIT, [("U", "d", j, True)], [], {"a": j, "q": j}))
            cl.append((MUL, [("UL", "a", j, True)], [], {"a": j}))
            cl.append((MUL, [("UR", "q", j, True)], [], {"q": j}))
            cl.append((MUL, [("U", "q", j, True)], [("L", OUT, True)], {"s": j}))
            cl.append((MUL, [("L", "s", j, True)], [("L", OUT, False)], {"s": j}))
        cl.append((SPLIT, [], [], {"d": 0}))
        for u in range(B):
            for a in range(B):
                for xi in range(B):
                    for c in range(B):
                        t = u + a * xi + c
                        if t >= B * B:
                            continue                     # never reached: the carry stays below B
                        o = {"d": t % B, "c": t // B}
                        base = [("U", "d", u, True), ("U", "a", a, True), ("L", "c", c, True)]
                        cl.append((MUL, base + [("U", "q", xi, True)], [("L", OUT, True)], o))
                        cl.append((MUL, base + [("L", "s", xi, True)], [("L", OUT, False)], o))
    if chans is None:
        chans = ("d", "n", "b", "m", "g") + (("a", "q", "s", "c") if task == "sq" else ())
    cl = [(k_, c_, kc, {ch: v for ch, v in o.items() if ch in chans}) for k_, c_, kc, o in cl]
    cl = [x for x in cl if x[3]]
    kinds = (PROMPT, SHIFT, SUB, SEL) + ((SPLIT, MUL) if task == "sq" else ())
    for kd in kinds:
        set_ = {ch for k_, _, _, o in cl if k_ == kd for ch in o}
        rest = {ch: 0 for ch in chans if ch not in set_}
        if rest:
            cl.append((kd, [], [], rest))
    return cl


class SqData:
    """Every (N, x) of a size for the square: N with w digits (top digit nonzero, N >= 2), x < N; the answer is
    x^2 mod N. `vd` holds x's w digits (the prompt's d), so the grid's prompt code is the division's."""

    def __init__(self, B, w, sample=0, seed=0):
        self.B, self.w, self.k = B, w, w
        Ns, xs = [], []
        for N in range(max(2, B ** (w - 1)), B ** w):
            xs.append(np.arange(N, dtype=np.int64))
            Ns.append(np.full(N, N, dtype=np.int64))
        N, x = np.concatenate(Ns), np.concatenate(xs)
        if sample and sample < len(N):
            idx = np.random.default_rng(seed).choice(len(N), sample, replace=False)
            N, x = N[idx], x[idx]
        self.set(N, x)

    def set(self, N, x):
        B, w = self.B, self.w
        self.N, self.x = np.asarray(N, dtype=np.int64), np.asarray(x, dtype=np.int64)
        self.v = self.x * self.x
        self.y = self.v % self.N
        self.X = len(self.N)
        self.Nd = digits(self.N, B, w)
        self.vd = digits(self.x, B, w)
        self.yd = digits(self.y, B, w)
        self.R0 = self.v // B ** w
        return self

    def subset(self, idx):
        out = object.__new__(SqData)
        out.B, out.w, out.k = self.B, self.w, self.k
        return out.set(self.N[idx], self.x[idx])
