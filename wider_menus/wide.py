"""The wide wiring grammar (MENUS.md): menus at growing levels, and the machine they span, computed on integer values
with every table at the truth (the tiny core's fast path; the H1 file runs the same grammar on learned tables).

The division of a 2W-digit dividend V by a W-digit N (base 10) is W steps on a P = W + 1 place register R. A step
brings in one dividend digit, runs a chain of S - 1 = 9 links, and moves the register. Its wiring is a set of tied
switches (one per kind of connection, shared by every step and link), grouped in two blocks:

  the chain (one step's links)
    a, b    each link's two operands, from any register: the chain register D, the step's start Rp, N's register M,
            the link's multiple of N (link s: s M, made by the machine's own multiplier), zero
    op      subtract (A - B) | add (A + B, the learned adder A - (0 - B))
    flag    what the gate reads: the operation's final borrow (or carry) | "D's lowest digit is 0" | "E's lowest digit
            is 0" (a zero test is the table's borrow of d - 1)
    keep    what a link writes into D: always | its result when the gate commits | never
    nread   N's register: N's digits as read | reversed
    g0, g1  the gate (learned table): commit or hold on flag 0, on flag 1
  the I/O (how the steps meet the dividend and the answer)
    vread   the dividend's digits as given | reversed
    start   the register's first content: the top W digits | zero | the bottom W digits
    order   the digit brought in at step t: digit W - 1 - t (downwards) | digit t (upwards)
    bring   the register moved by k places (any k in -(P - 1)..P - 1), then the digit written at place p (or not at
            all) by replacing that place's digit or by adding it (the learned adder)
    post    after the chain the register moves by -1, 0 or +1 places
    final   after the last step: nothing | one conditional subtraction of N (a gated link) | add the top W digits of
            the dividend, then the conditional subtraction
    read    the answer: the W places from offset 0 or 1, in order or reversed

Long division (CHAIN.md) is a = D, b = M, sub, borrow, gated, given, commit on 0 / hold on 1; given, top, down,
(k, p) = (1, 0) replace, post 0, final none, read (0, given). The menu also holds, among others:
  * long division by trial multiples: a = Rp, b = the link's multiple (link s computes Rp - s N);
  * a Montgomery-like right-to-left reduction (REDC with R = 10^W): start zero, order up, bring (0, 0) add, op add,
    b = M, flag "D's lowest digit is 0" (hold once it is 0), post -1, final add-high: it computes V 10^-W mod N, the
    Montgomery form, not V mod N;
  * repeated subtraction with other stop rules (keep always, the zero-digit flags, the other gates).

The multiply X * Y (squaring: X = Y = s) places the product cell's low and high digit of every pair (X_i, Y_j) in a
column and a number: column = a i + b j + c (one affine rule per output kind), number = alpha i + beta j + gamma (one
rule for both kinds); the numbers are summed by the learned adder. Schoolbook rows are number = j; number = i and the
diagonals number = i - j are the other collision-free groupings (both correct); number = i + j collides.
"""

from __future__ import annotations

import itertools

import torch

B = 10
S = 10                                         # a step's chain has S - 1 links
SRC = {"D": 0, "Rp": 1, "M": 2, "Mq": 3, "Z": 4, "R": 5, "d": 6}
CODES = {
    "op": {"sub": 0, "add": 1},
    "flag": {"borrow": 0, "zD": 1, "zE": 2, "lt": 3},
    "keep": {"always": 0, "gated": 1, "never": 2},
    "nread": {"given": 0, "reversed": 1},
    "g0": {"hold": 0, "commit": 1},
    "g1": {"hold": 0, "commit": 1},
    "vread": {"given": 0, "reversed": 1},
    "start": {"top": 0, "zero": 1, "bottom": 2, "N": 3},
    "order": {"down": 0, "up": 1},
    "final": {"none": 0, "csub": 1, "addhigh": 2, "csub2": 3},
}
CHAIN = ("a", "b", "op", "flag", "keep", "nread", "g0", "g1")
IO = ("vread", "start", "order", "bring", "post", "final", "read")
DIV_TRUE = {"a": "D", "b": "M", "op": "sub", "flag": "borrow", "keep": "gated", "nread": "given", "g0": "commit",
            "g1": "hold", "vread": "given", "start": "top", "order": "down", "bring": (1, 0, "rep"), "post": 0,
            "final": "none", "read": (0, "given")}
MUL_TRUE = {"lo": (1, 1, 0), "hi": (1, 1, 1), "num": (0, 1)}
IO3_SHIFTS = lambda P: [(-1, P - 1), (0, 0), (1, 0), (0, -1)]   # io3's down, stay, up, skip as (k, p)


def div_menu(level: int, W: int) -> dict:
    """The division's switches at a menu level (option lists; no list starts with the true option):
    0  io3's menu (WIRING.md: 864 wirings times the gate's 4 tables);
    1  + any register on either port (the multiples of N and zero too), add or subtract, three flags;
    2  + the start, the digit order, the bring-in's replace / add, the post-chain move, the final correction, the
       readout's direction;
    3  + any move k and any insert place p (or none);
    4  + the register before the bring-in and the incoming digit on either port, the flag "A < N", moves of the
       register after the chain by up to 2 places, N as the start, two conditional subtractions at the end."""
    P = W + 1
    m = {}
    if level == 0:
        m["a"], m["b"], m["op"], m["flag"] = ["Rp", "M", "D"], ["D", "Rp", "M"], ["sub"], ["borrow"]
    elif level >= 4:
        m["a"], m["b"] = ["d", "R", "Z", "Mq", "M", "Rp", "D"], ["d", "R", "Z", "D", "Rp", "Mq", "M"]
        m["op"], m["flag"] = ["add", "sub"], ["lt", "zD", "zE", "borrow"]
    else:
        m["a"], m["b"] = ["Z", "Mq", "M", "Rp", "D"], ["Z", "D", "Rp", "Mq", "M"]
        m["op"], m["flag"] = ["add", "sub"], ["zD", "zE", "borrow"]
    m["keep"] = ["always", "gated", "never"]
    m["nread"] = ["reversed", "given"]
    m["g0"], m["g1"] = ["hold", "commit"], ["commit", "hold"]
    m["vread"] = ["reversed", "given"]
    if level <= 1:
        m["start"], m["order"] = ["top"], ["down"]
        m["bring"] = [(k, p, "rep") for k, p in IO3_SHIFTS(P)]
        m["post"], m["final"], m["read"] = [0], ["none"], [(1, "given"), (0, "given")]
    else:
        m["start"], m["order"] = ["zero", "bottom", "top"], ["up", "down"]
        if level == 2:
            m["bring"] = [(k, p, md) for k, p in IO3_SHIFTS(P) for md in ("add", "rep")]
        else:
            m["bring"] = [(k, p, md) for k in range(-(P - 1), P) for p in range(-1, P) for md in ("add", "rep")]
        m["post"], m["final"] = [-1, 1, 0], ["addhigh", "csub", "none"]
        m["read"] = [(o, d) for o in (1, 0) for d in ("reversed", "given")]
        if level >= 4:
            m["start"], m["post"] = ["N", "zero", "bottom", "top"], [-2, -1, 2, 1, 0]
            m["final"] = ["csub2", "addhigh", "csub", "none"]
    return m


def mul_menu(level: int, W: int) -> dict:
    """The multiply's switches: one column rule per output kind (lo, hi) and one number rule.
    0  io3's 24 rules per kind, f(i) + g(j) + d (f in {W-1-i, i}, g in {0, j, W-1-j}, d in -1..2), numbers by j;
    1  every affine column rule a i + b j + c, a, b in -2..2, c in -2(W-1)..C-1, numbers by j;
    2  + the number rule alpha i + beta j + gamma, alpha, beta in -1..1;
    3  column rules with a, b in -3..3, c in -3(W-1)..C-1."""
    C = 2 * W + 2
    if level >= 3:
        rules = [(a, b, c) for a in range(-3, 4) for b in range(-3, 4) for c in range(-3 * (W - 1), C)]
    elif level == 0:
        rules = [(fa, gb, fc + gc + d) for fa, fc in ((-1, W - 1), (1, 0)) for gb, gc in ((0, 0), (1, 0), (-1, W - 1))
                 for d in (-1, 0, 1, 2)]
    if level in (1, 2):
        rules = [(a, b, c) for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2) for c in range(-2 * (W - 1), C)]
    num = [(0, 1)] if level <= 1 else [(al, be) for al in (-1, 0, 1) for be in (-1, 0, 1)]
    return {"lo": rules, "hi": list(rules), "num": num}


def size(menu: dict, names=None) -> int:
    n = 1
    for k in (names or menu):
        n *= len(menu[k])
    return n


def true_index(menu: dict, truth: dict) -> dict:
    return {k: menu[k].index(truth[k]) for k in menu}


# ------------------------------------------------------------------------------------------------ the division (values)
def decode(menu: dict, idx: dict, W: int) -> dict:
    """Per-row option indices idx[name] [X] -> the numeric codes the machine reads."""
    dev = next(iter(idx.values())).device
    out = {}
    for k in ("a", "b"):
        out[k] = torch.tensor([SRC[v] for v in menu[k]], device=dev)[idx[k]]
    for k, codes in CODES.items():
        out[k] = torch.tensor([codes[v] for v in menu[k]], device=dev)[idx[k]]
    br = menu["bring"]
    out["k"] = torch.tensor([v[0] for v in br], device=dev)[idx["bring"]]
    out["p"] = torch.tensor([v[1] for v in br], device=dev)[idx["bring"]]
    out["mode"] = torch.tensor([int(v[2] == "add") for v in br], device=dev)[idx["bring"]]
    out["post"] = torch.tensor(menu["post"], device=dev)[idx["post"]]
    out["off"] = torch.tensor([v[0] for v in menu["read"]], device=dev)[idx["read"]]
    out["rdir"] = torch.tensor([int(v[1] == "reversed") for v in menu["read"]], device=dev)[idx["read"]]
    return out


def value(D: torch.Tensor) -> torch.Tensor:
    """Digits [..., L] (LSB first) -> int64 values."""
    return (D * (10 ** torch.arange(D.shape[-1], device=D.device))).sum(-1)


def digits(v: torch.Tensor, L: int) -> torch.Tensor:
    return torch.stack([(v // 10 ** i) % 10 for i in range(L)], -1)


def move(R: torch.Tensor, k: torch.Tensor, MOD: int) -> torch.Tensor:
    """The register moved up by k places (k < 0: down), zero-filled, P places."""
    up = 10 ** k.clamp(min=0)
    down = 10 ** (-k).clamp(min=0)
    return (R * up) % MOD // down


def sel(code: torch.Tensor, opts: list) -> torch.Tensor:
    out = opts[0]
    for i, o in enumerate(opts[1:], 1):
        out = torch.where(code == i, o, out)
    return out


def divide_val(Vd: torch.Tensor, Nd: torch.Tensor, w: dict, mult=None, steps: int | None = None,
               R0: torch.Tensor | None = None) -> torch.Tensor:
    """V mod N (or whatever the wiring computes) on values, per-row codes w (decode's output). Vd [X, 2W], Nd [X, W]
    digits; mult(M) -> [X, 9] the multiples 1..9 of the N register (default: exact). Returns the answer's W digits."""
    W = Nd.shape[1]
    P = W + 1
    MOD = 10 ** P
    Vp = torch.where(w["vread"][:, None] == 1, Vd.flip(-1), Vd)
    Nr = torch.where(w["nread"][:, None] == 1, Nd.flip(-1), Nd)
    M = value(Nr)
    Mq = (torch.arange(1, S, device=M.device)[None] * M[:, None]) % MOD if mult is None else mult(Nr)
    top = value(Vp[:, W:])
    R = sel(w["start"], [top, torch.zeros_like(top), value(Vp[:, :W]), M])
    for t in range(W):
        d = torch.where(w["order"] == 0, Vp[:, W - 1 - t], Vp[:, t])
        R0 = R
        R = move(R, w["k"], MOD)
        pw = 10 ** w["p"].clamp(min=0)
        rep = R - ((R // pw) % 10) * pw + d * pw
        R = torch.where(w["p"] < 0, R, torch.where(w["mode"] == 0, rep, (R + d * pw) % MOD))
        R = chain_val(R, M, Mq, w, MOD, R0, d)
        R = move(R, w["post"], MOD)
    fin = w["final"]
    if bool((fin > 0).any()):
        R2 = torch.where(fin == 2, (R + top) % MOD, R)
        for rep in range(2):
            commit = torch.where(R2 < M, w["g1"], w["g0"]) == 1
            R2 = torch.where(commit & ((rep == 0) | (fin == 3)), (R2 - M) % MOD, R2)
        R = torch.where(fin > 0, R2, R)
    out = digits((R // 10 ** w["off"]) % 10 ** W, W)
    return torch.where(w["rdir"][:, None] == 1, out.flip(-1), out)


def chain_val(Rp: torch.Tensor, M: torch.Tensor, Mq: torch.Tensor, w: dict, MOD: int, R0: torch.Tensor | None = None,
              d: torch.Tensor | None = None) -> torch.Tensor:
    """One step's chain on values: S - 1 links. R0, d: the register before the bring-in and the incoming digit (the
    menu's sources R and d; default: Rp // 10 and Rp % 10, as the true bring-in made them). Returns the chain
    register."""
    D = Rp
    Z = torch.zeros_like(Rp)
    R0 = Rp // 10 if R0 is None else R0
    d = Rp % 10 if d is None else d
    for s in range(1, S):
        srcs = [D, Rp, M, Mq[:, s - 1], Z, R0, d]
        A, Bv = sel(w["a"], srcs), sel(w["b"], srcs)
        E = torch.where(w["op"] == 0, (A - Bv) % MOD, (A + Bv) % MOD)
        fb = torch.where(w["op"] == 0, A < Bv, A + Bv >= MOD)
        fl = sel(w["flag"], [fb, D % 10 == 0, E % 10 == 0, A < M])
        commit = torch.where(fl, w["g1"], w["g0"]) == 1
        take = (w["keep"] == 0) | ((w["keep"] == 1) & commit)
        D = torch.where(take, E, D)
    return D


# ------------------------------------------------------------------------------------------------ the multiply (values)
def num_gamma(al: int, be: int, W: int) -> int:
    return (W - 1) * (al < 0) + (W - 1) * (be < 0)


def claims(rule_lo, rule_hi, num, W: int, rows_only: bool = False) -> torch.Tensor:
    """The column of every digit (kind lo / hi, pair i, j) under the rules, or -1 where it is dropped (outside the C
    columns, or its number and column already taken by an earlier pair: pairs in order j, then i). rows_only: each
    multiplier digit j alone (the routed ladder's view: the numbers of other rows do not exist). [2, W, W] (i, j)."""
    C = 2 * W + 2
    out = torch.full((2, W, W), -1, dtype=torch.long)
    al, be = num
    ga = num_gamma(al, be, W)
    for kind, (a, b, c) in enumerate((rule_lo, rule_hi)):
        taken = set()
        for j in range(W):
            if rows_only:
                taken = set()
            for i in range(W):
                col = a * i + b * j + c
                n = (0, j) if rows_only else al * i + be * j + ga
                if 0 <= col < C and (n, col) not in taken:
                    taken.add((n, col))
                    out[kind, i, j] = col
    return out


def mul_val(Xd: torch.Tensor, Yd: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """X * Y on values (exact product cell) under the claims cols [2, W, W] (or per row [X, 2, W, W]): the sum of the
    placed digits mod 10^C."""
    W = Xd.shape[1]
    C = 2 * W + 2
    prod = Xd[:, :, None] * Yd[:, None, :]                                  # [X, i, j]
    dig = torch.stack([prod % 10, prod // 10], 1)                           # [X, kind, i, j]
    if cols.dim() == 3:
        cols = cols[None]
    wgt = torch.where(cols >= 0, 10 ** cols.clamp(min=0), torch.zeros_like(cols))
    return (dig * wgt).sum((1, 2, 3)) % 10 ** C


def row_val(Xd: torch.Tensor, b: torch.Tensor, j: int, cols_rows: torch.Tensor) -> torch.Tensor:
    """Row j of X * b (single-digit b [X]) under the routed ladder's view cols_rows [2, W, W]: values."""
    W = Xd.shape[1]
    C = 2 * W + 2
    prod = Xd * b[:, None]
    dig = torch.stack([prod % 10, prod // 10], 1)                           # [X, kind, i]
    c = cols_rows[:, :, j]
    wgt = torch.where(c >= 0, 10 ** c.clamp(min=0), torch.zeros_like(c))
    return (dig * wgt[None]).sum((1, 2)) % 10 ** C


def rule_maps(rules: list, W: int) -> torch.Tensor:
    """The routed ladder's soft-map pieces: for every column rule, row j and column c the source pair i (the first i
    whose column is c), W = nothing [R, W (j), C]."""
    C = 2 * W + 2
    out = torch.full((len(rules), W, C), W, dtype=torch.long)
    for r, (a, b, c0) in enumerate(rules):
        for j in range(W):
            for i in reversed(range(W)):
                col = a * i + b * j + c0
                if 0 <= col < C:
                    out[r, j, col] = i
    return out


# ------------------------------------------------------------------------------------------------ the soft adder
def add_soft_exact(A: torch.Tensor, Bd: torch.Tensor) -> torch.Tensor:
    """A + B mod 10^L on digit distributions [X, L, 10] with the exact adder (the routed ladder's soft rows):
    carries tracked as a distribution."""
    X, L, _ = A.shape
    o = torch.arange(10)
    c = torch.zeros(X, 2, dtype=A.dtype)
    c[:, 0] = 1
    out = []
    # joint over (a, b, cin): the sum digit and carry
    s = o[:, None, None] + o[None, :, None] + torch.arange(2)[None, None, :]    # [a, b, cin]
    dig = torch.nn.functional.one_hot(s % 10, 10).to(A.dtype)                  # [a, b, cin, 10]
    car = torch.nn.functional.one_hot(s // 10, 2).to(A.dtype)
    for k in range(L):
        J = A[:, k, :, None, None] * Bd[:, k, None, :, None] * c[:, None, None, :]   # [X, a, b, cin]
        out.append(torch.einsum("xabc,abcd->xd", J, dig))
        c = torch.einsum("xabc,abcd->xd", J, car)
    return torch.stack(out, 1)


def combos(menu: dict, names: tuple) -> torch.Tensor:
    """Every combination of the switches `names` [K, len(names)] (option indices)."""
    return torch.tensor(list(itertools.product(*[range(len(menu[n])) for n in names])), dtype=torch.long)
