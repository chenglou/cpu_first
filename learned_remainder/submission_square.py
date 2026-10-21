"""LearnedSquare: x^(2^T) mod N in which the squaring is a schoolbook multiplication with a learned digit-product cell
and the reduction is submission_chain.py's restoring long division with its learned subtraction table.

Datapath (designed wiring, the schoolbook algorithms in base 10):
  * read N (its digits come LSB first), x and T from the prompt; the state is x's W = 8 digits, LSB first;
  * loop T times:
      - square the state by schoolbook: the row s * s_j puts the product cell's lo digit of (s_i, s_j) at place i and
        its hi digit at place i + 1 and adds the two (the adder); the W rows, shifted by j, are summed by the adder into
        the 2W digits of s^2;
      - reduce s^2 mod N by submission_chain.py's restoring long division: W steps, each a chain of up to S - 1
        subtractions of N's prompt digits by ONE shared subtractor cell, stopped at the first borrow;
  * write the state's digits right-aligned on the prompt.
No integer arithmetic touches the numbers; only the loop counter T is read as an integer.

Learned (MODE = "learned"):
  * the product cell (a, b) -> (lo digit, hi digit): 100 rows x (10 + 10) logits;
  * the subtractor cell (r, m, borrow in) -> (digit, borrow out): 200 rows x (10 + 2) logits (submission_chain.py).
The squaring's adder (ADDER):
  * "learned": the learned subtraction table in ten's complement, A + B = A - (0 - B), both subtractions the table
    (ADDER.md's add_learned). So every arithmetic step of the forward pass is one of the two learned tables;
  * "designed": a HAND-WRITTEN ripple adder (the stand-in before the learned adder was available).
Either way the training checks still use a hand-written adder (and the ladder a hand-written soft adder): the learned
adder's own training without one did not learn (ADDER.md). Evaluation snaps both cells to their argmax tables and runs
the datapath on integer digits. MODE = "designed" uses the exact tables instead (the plumbing check).

What teaches the product cell (label-free, on the rows' own numbers X = x and N; SQUARE.md has the tiny core):
  * X * 0 = 0;
  * the multiplication ladder along the verified frontier: the soft row X * (b + 1) learns the digits of the machine's
    own hard X * b plus X (the adder), b = 0, ..., 8, but only for the numbers whose rungs 0..b all check out on the
    hard machine (X * 0 = 0 and X * c = X * (c - 1) + X for c <= b). So every target the cell learns from is exact.
    Without the gate, early wrong targets leave cells in a carry-compensated blur (a wrong lo digit whose carry makes
    up for a wrong hi digit); with both rungs soft (X * (b + 1) - X * b = X) the ladder freezes into domain walls.
What teaches the subtractor: submission_chain.py's lessons unchanged (every link of the machine's own division of its
own square of x checks itself, "x mod N = x", the rows' answers along the own hard path). The rows' answers are used
only in steps in which every ladder check passes, and only on rows whose squarings (of x, and of the state before the
last step) equal the ones the check's adder makes (ADDER = "learned"): a wrong square turns the answer into noise for
the subtractor. Their dividend is the machine's own hard square, so labels never reach the product cell. The model freezes after FREEZE_WIN
consecutive steps in which every check (links, "x mod N = x", ladder, squares) is exact on the hard machine.
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
)

MODE = "learned"       # "learned" | "designed" (the exact product and subtraction tables: the plumbing check)
ADDER = "learned"      # the squaring's adder: "learned" (A - (0 - B) with the learned subtraction table) | "designed"
                       # (a hand-written ripple adder: the stand-in)
DIGIT, MARK_N, MARK_X, MARK_T = 7, 2, 3, 4  # token ids: digit d is DIGIT + d
W = 8                  # digits of the widest modulus (a 24-bit N is below 10^8); one reduction takes W steps
P = W + 1              # register places: 10 R + v_t < 10 N < 10^9
Q = W + 2              # places of a schoolbook row X * b (X * 9 < 10^9; one more keeps any table's row whole)
S = 10                 # trials per step: trial s is s links down the chain (Rp - s*N until a link borrows)
INIT = 0.1             # std of the initial logits
LR = 0.05              # Adam on the logits
ID_SWITCH = 0.8        # phase B (the other lessons join) once this share of the link checks passes, averaged ...
STAGE_WIN = 20         # ... over this many steps,
ID_STEPS = 300         # ... or after this many steps
ID_PLACE0 = 5.0        # weight of the lowest place in the check (the swapped borrow code fails it only there)
ID_LINKS = "all"       # which links check themselves: "all" | "live" (as submission_chain.py)
ID_BORROW = 1.0        # weight of the final borrow in the link check (the complete identity, as submission_chain.py)
LADDER_GATE = True     # the product ladder trains rung b + 1 only where rungs 0..b check out (False: every number)
LADDER_W = 1.0         # weight of the ladder's loss
SELF_LESSON = True     # phase B: also "x mod N = x" (each row's x divided by N, soft own-borrow choice)
SELF_STEPS = 1         # ... in this many steps, from x // 10^SELF_STEPS (one step: the chain must stop at x itself)
REAL_MAX_T = 2         # phase B also trains on the batch's real rows with T <= REAL_MAX_T (0: never sees a label)
FREEZE_WIN = 200       # phase B: freeze after this many consecutive steps with every check exact
BATCH = 512
EVAL_BATCH = 16384
MAX_STEPS = 20000
LOG_EVERY = 250

POW10 = [10 ** i for i in range(19)]
AR10 = torch.arange(10)
SUMM = F.one_hot((AR10[:, None] + AR10[None, :]).flatten(), 19).float()  # [100, 19]: a + b


def to_digits(v: Tensor, n: int) -> Tensor:
    """Non-negative int64 values -> their n lowest decimal digits, LSB first (designed I/O)."""
    return (v[..., None] // torch.tensor(POW10[:n], device=v.device)) % 10


def from_digits(d: Tensor) -> Tensor:
    return (d * torch.tensor(POW10[:d.shape[-1]], device=d.device)).sum(-1)


def parse(input_ids: Tensor, attention_mask: Tensor):
    """Prompt `N <N digits, LSB first> X <x, MSB first> T <T, MSB first>` -> LSB-first digits [rows, W], N's width."""
    pos = lambda tok: (input_ids == tok).int().argmax(1)
    i_n, i_x, i_t = pos(MARK_N), pos(MARK_X), pos(MARK_T)
    end = attention_mask.long().sum(1)
    k = torch.arange(W, device=input_ids.device)[None]

    def field(start, step, count):
        idx = (start[:, None] + step * k).clamp(0, input_ids.shape[1] - 1)
        d = (input_ids.gather(1, idx) - DIGIT).clamp(0, 9)
        return torch.where(k < count[:, None], d, 0)

    width = i_x - i_n - 1
    return field(i_n + 1, 1, width), field(i_t - 1, -1, i_t - i_x - 1), field(end - 1, -1, end - i_t - 1), end, width


def exact_tables(device) -> tuple[Tensor, Tensor]:
    """The designed subtractor (the plumbing check only): digit and borrow of r - m - c, flat over (r, m, c)."""
    r = torch.arange(10, device=device)[:, None, None]
    m = torch.arange(10, device=device)[None, :, None]
    c = torch.arange(2, device=device)[None, None, :]
    x = r - m - c
    return (x % 10).flatten(), (x < 0).long().flatten()


def exact_products(device) -> tuple[Tensor, Tensor]:
    """The designed product cell (the plumbing check only): lo and hi digit of a * b, flat over (a, b)."""
    v = torch.arange(10, device=device)[:, None] * torch.arange(10, device=device)[None, :]
    return (v % 10).flatten(), (v // 10).flatten()


# ------------------------------------------------------------------------------------------------ the adder (designed)
def add_hard(D: Tensor, M: Tensor) -> tuple[Tensor, Tensor]:
    """The designed adder on integers: digits D [..., L] + M [..., L], mod 10^L, and the final carry."""
    c = torch.zeros_like(D[..., 0])
    out = []
    for j in range(D.shape[-1]):
        x = D[..., j] + M[..., j] + c
        out.append(x % 10)
        c = x // 10
    return torch.stack(out, -1), c


def add_soft(D: Tensor, M: Tensor) -> tuple[Tensor, Tensor]:
    """The designed adder on digit distributions D [..., P, 10] plus exact digits M [..., P], mod 10^P (multilinear),
    and the probability of a final carry."""
    o = torch.arange(10, device=D.device)
    c1 = torch.zeros(D.shape[:-2], device=D.device)                         # P(carry in = 1)
    out = []
    for j in range(D.shape[-2]):
        d, m = D[..., j, :], M[..., j, None]
        out.append((1 - c1)[..., None] * d.gather(-1, (o - m) % 10) + c1[..., None] * d.gather(-1, (o - m - 1) % 10))
        below = torch.cat([torch.zeros_like(d[..., :1]), d.cumsum(-1)], -1)  # below[k] = P(d < k)
        g0 = 1 - below.gather(-1, 10 - m)                                    # P(d + m >= 10)
        g1 = 1 - below.gather(-1, 9 - m)                                     # P(d + m + 1 >= 10)
        c1 = (1 - c1) * g0[..., 0] + c1 * g1[..., 0]
    return torch.stack(out, -2), c1


def add_dd(A: Tensor, B: Tensor) -> Tensor:
    """The designed adder on two digit distributions [..., L, 10] (multilinear, places independent): the sum's digit
    distributions mod 10^L."""
    summ = SUMM.to(A.device)
    c = torch.zeros(A.shape[:-2], device=A.device)                          # P(carry in = 1)
    out = []
    for j in range(A.shape[-2]):
        s = (A[..., j, :, None] * B[..., j, None, :]).flatten(-2) @ summ     # over a + b
        s = (1 - c)[..., None] * F.pad(s, (0, 1)) + c[..., None] * F.pad(s, (1, 0))   # over a + b + c
        out.append(s[..., :10] + s[..., 10:])
        c = s[..., 10:].sum(-1)
    return torch.stack(out, -2)


def sub_hard(D: Tensor, M: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """D - M on integers with the learned subtraction table, any width and batch shape: digits [..., L], LSB first.
    Returns the digits mod 10^L and the final borrow. `visits` accumulates every table row used."""
    c = torch.zeros_like(D[..., 0])
    out, rows = [], []
    for j in range(D.shape[-1]):
        idx = (D[..., j] * 10 + M[..., j]) * 2 + c
        out.append(Td[idx])
        c = Tb[idx]
        rows.append(idx)
    if visits is not None:
        visits += torch.bincount(torch.stack(rows).flatten(), minlength=200)
    return torch.stack(out, -1), c


def add_fwd(A: Tensor, B: Tensor, tabs, adder: str, visits: Tensor | None = None) -> Tensor:
    """The squaring's adder, A + B mod 10^L on integer digits [..., L]: "designed" is the hand-written ripple adder;
    "learned" is A - (0 - B) with the learned subtraction table (in ten's complement 0 - B is 10^L - B)."""
    if adder == "designed":
        return add_hard(A, B)[0]
    C = sub_hard(torch.zeros_like(B), B, tabs[0], tabs[1], visits)[0]
    return sub_hard(A, C, tabs[0], tabs[1], visits)[0]


# ------------------------------------------------------------------------------------------------ the squaring
def row_hard(X: Tensor, b: Tensor, tabs, adder: str | None = None, pvisits: Tensor | None = None,
             visits: Tensor | None = None) -> Tensor:
    """Schoolbook row X * b on integers: X [..., W] digits times the digits b [...] with the snapped product cell
    (tabs[2], tabs[3]: flat lo and hi tables [100] indexed a * 10 + b): lo at place i, hi at place i + 1, summed by the
    adder. Returns the row's digits [..., Q]. `pvisits` accumulates every product cell used, `visits` every subtractor
    row the learned adder uses. `adder` defaults to ADDER."""
    idx = X * 10 + b[..., None]
    z = torch.zeros_like(X[..., :1])
    lo = torch.cat([tabs[2][idx], z, z], -1)
    hi = torch.cat([z, tabs[3][idx], z], -1)
    if pvisits is not None:
        pvisits += torch.bincount(idx.flatten(), minlength=100)
    return add_fwd(lo, hi, tabs, adder or ADDER, visits)


def square_hard(Sd: Tensor, tabs, adder: str | None = None, pvisits: Tensor | None = None,
                visits: Tensor | None = None) -> Tensor:
    """s^2 by schoolbook on integers: Sd [X, W] digits -> the 2W digits of s^2 [X, 2W]. The W rows s * s_j are formed
    side by side, then summed one by one, shifted by j, by the adder (in 2W + 2 places, so no table can wrap)."""
    X, adder = Sd.shape[0], adder or ADDER
    rows = row_hard(Sd[:, None, :].expand(X, W, W), Sd, tabs, adder, pvisits, visits)      # [X, W (j), Q]
    acc = torch.zeros(X, 2 * W + 2, dtype=Sd.dtype, device=Sd.device)
    for j in range(W):
        acc = add_fwd(acc, F.pad(rows[:, j], (j, 2 * W + 2 - Q - j)), tabs, adder, visits)
    return acc[:, :2 * W]


def row_soft(X: Tensor, b: int, PLO: Tensor, PHI: Tensor) -> Tensor:
    """The row X * b on digit distributions: the soft product cell PLO, PHI [10, 10, 10] (a, b) -> digit
    probabilities, applied to the exact digits X [n, W], summed by the designed adder. Returns [n, Q, 10]."""
    n = X.shape[0]
    z = torch.zeros(n, 1, 10, device=X.device)
    z[..., 0] = 1
    lo = torch.cat([PLO[X, b], z, z], 1)
    hi = torch.cat([z, PHI[X, b], z], 1)
    return add_dd(lo, hi)


# ------------------------------------------------------------------------------------------------ the reduction
def link_hard(D: Tensor, M: Tensor, Td: Tensor, Tb: Tensor):
    """One link of the chain on integers, D - N: D [X, P] digits, M [X, P] N's digits, flat tables Td, Tb [200]
    indexed (r, m, c) -> r * 20 + m * 2 + c. Returns the digits [X, P], the final borrow [X] and the table rows
    [X, P]."""
    c = torch.zeros_like(D[:, 0])
    outs, rows = [], []
    for j in range(P):
        idx = (D[:, j] * 10 + M[:, j]) * 2 + c
        outs.append(Td[idx])
        c = Tb[idx]
        rows.append(idx)
    return torch.stack(outs, 1), c, torch.stack(rows, 1)


def step_hard(Rp: Tensor, M: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None):
    """One division step on integers: trial s = trial (s - 1) - N for s = 1, ..., S - 1, then the stop rule keeps the
    last trial before the chain first borrows. Returns the kept trial [X, P], its index s (the quotient digit) [X] and
    every link's input [X, S - 1, P] with whether the link was live (no earlier link borrowed) [X, S - 1]. `visits`
    accumulates every table row the links use."""
    D, R = Rp, Rp
    alive = torch.ones_like(Rp[:, 0], dtype=torch.bool)                 # no link has borrowed yet
    q = torch.zeros_like(Rp[:, 0])
    ins, rows, live = [], [], []
    for s in range(1, S):
        ins.append(D)
        live.append(alive)
        D, c, idx = link_hard(D, M, Td, Tb)
        rows.append(idx)
        alive = alive & (c == 0)
        R = torch.where(alive[:, None], D, R)
        q = q + alive.long()
    if visits is not None:
        visits += torch.bincount(torch.stack(rows).flatten(), minlength=200)
    return R, q, torch.stack(ins, 1), torch.stack(live, 1)


def top(V: Tensor, steps: int) -> Tensor:
    """The digits of V // 10^steps in P places (V [X, L] digits): the division's starting register."""
    R = V[:, steps:steps + P]
    return F.pad(R, (0, P - R.shape[1]))


def divide_hard(V: Tensor, steps: int, M: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None) -> Tensor:
    """V mod N on integer digits V [X, L], from R = V // 10^steps (below N) in `steps` steps of the chain. Returns the
    remainder's digits [X, P]. `visits` accumulates every table row used."""
    R = top(V, steps)
    for t in reversed(range(steps)):
        R = step_hard(torch.cat([V[:, t, None], R[:, :P - 1]], 1), M, Td, Tb, visits)[0]
    return R


def link_inputs(V: Tensor, steps: int, M: Tensor, Td: Tensor, Tb: Tensor) -> tuple[Tensor, Tensor]:
    """The hard division's link inputs at every step [X, steps, S - 1, P]: the register Rp = 10 R + v_t feeds link 1,
    trial s - 1 feeds link s (the machine's own numbers, every link it computes); and whether each link was live
    [X, steps, S - 1]."""
    R = top(V, steps)
    out, live = [], []
    for t in reversed(range(steps)):
        R, _, ins, lv = step_hard(torch.cat([V[:, t, None], R[:, :P - 1]], 1), M, Td, Tb)
        out.append(ins)
        live.append(lv)
    return torch.stack(out, 1), torch.stack(live, 1)


def link_soft(D: Tensor, M: Tensor, TDC: Tensor):
    """One link on digit distributions, the cell applied multilinearly: D [X, P, 10] minus N's exact digits M [X, P].
    TDC [20, 120]: rows (r, c), columns (m, 10 digit + 2 borrow probabilities). Returns the digits [X, P, 10] and the
    final borrow [X, 2]."""
    X = D.shape[0]
    c = torch.zeros(X, 2, device=D.device)
    c[:, 0] = 1
    out = []
    for j in range(P):
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = (joint @ TDC).view(X, 10, 12)                                       # every m at once
        Y = Y.gather(1, M[:, j, None, None].expand(X, 1, 12)).squeeze(1)        # N's own digit
        out.append(Y[:, :10])
        c = Y[:, 10:]
    return torch.stack(out, 1), c


def divide_soft(V: Tensor, steps: int, M: Tensor, TDC: Tensor) -> Tensor:
    """The same division on digit distributions with the soft own-borrow choice: trial s is kept with the probability
    that links 1..s do not borrow and link s + 1 does (trial S - 1 when no link borrows). V [X, L] exact digits.
    Returns the final remainder's digit distributions [X, P, 10]."""
    R = F.one_hot(top(V, steps), 10).float()
    for t in reversed(range(steps)):
        D = torch.cat([F.one_hot(V[:, t], 10).float()[:, None], R[:, :P - 1]], 1)
        R = torch.zeros_like(D)
        alive = torch.ones(V.shape[0], device=V.device)                       # P(no link has borrowed yet)
        for s in range(1, S):
            Dn, c = link_soft(D, M, TDC)
            R = R + (alive * c[:, 1])[:, None, None] * D                        # link s borrows: keep trial s - 1
            alive = alive * c[:, 0]
            D = Dn
        R = R + alive[:, None, None] * D
    return R


def divide_path(V: Tensor, steps: int, M: Tensor, TDC: Tensor, Td: Tensor, Tb: Tensor,
                visits: Tensor | None = None):
    """The model's own hard path, learned along: each step's chain length (the quotient digit) is chosen by the hard
    datapath (argmax table, stop at the first borrow), and the soft cell runs that many links on digit distributions
    from the soft register. V [X, L] exact digits (the machine's own hard square). Returns the soft remainder
    [X, P, 10] (the gradient reaches the subtractor through the kept trials' links only) and the hard one [X, P]."""
    Rh = top(V, steps)
    Rs = F.one_hot(Rh, 10).float()
    for t in reversed(range(steps)):
        vt = V[:, t]
        Rh, q, _, _ = step_hard(torch.cat([vt[:, None], Rh[:, :P - 1]], 1), M, Td, Tb, visits)
        D = torch.cat([F.one_hot(vt, 10).float()[:, None], Rs[:, :P - 1]], 1)
        Rs = D * (q == 0).float()[:, None, None]
        for s in range(1, int(q.max()) + 1):
            D = link_soft(D, M, TDC)[0]
            Rs = Rs + D * (q == s).float()[:, None, None]
    return Rs, Rh


def ce(R: Tensor, target: Tensor, width: Tensor) -> Tensor:
    """Cross-entropy of the remainder's digits below N's width (a remainder below N has `width` digits)."""
    scored = (torch.arange(P, device=R.device)[None] < width[:, None]).float()
    nll = -R.clamp_min(1e-12).log().gather(-1, target[..., None]).squeeze(-1)
    return (nll * scored).sum() / scored.sum()


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Control:
    """The training schedule (Python-side, not model state): phase A, phase B, frozen."""

    def __init__(self) -> None:
        self.step = 0
        self.phase = "frozen" if MODE == "designed" else "A"
        self.in_phase = 0
        self.streak = 0
        self.reset = False                    # the scheduler clears Adam's state at the switch
        self.win: deque = deque(maxlen=STAGE_WIN)
        self.visits: Tensor | None = None     # subtractor rows used by training's hard checks (the audit)
        self.used: Tensor | None = None       # subtractor rows used at evaluation
        self.pvisits: Tensor | None = None    # product cells used by training's hard checks (the ladder)
        self.pused: Tensor | None = None      # product cells used at evaluation
        self.reported = (-1, -1)
        self.history: list = []


class LearnedSquare(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.a = nn.Parameter(INIT * torch.randn(10, 10, 2, 10))           # subtractor: (r, m, borrow in) -> digit
        self.b = nn.Parameter(INIT * torch.randn(10, 10, 2, 2))            # ... -> borrow out
        self.p_lo = nn.Parameter(INIT * torch.randn(10, 10, 10))           # product cell: (a, b) -> lo digit
        self.p_hi = nn.Parameter(INIT * torch.randn(10, 10, 10))           # ... -> hi digit
        self.ctl = Control()

    def tables(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if MODE == "designed":
            return exact_tables(self.a.device) + exact_products(self.a.device)
        return (self.a.argmax(-1).flatten(), self.b.argmax(-1).flatten(),
                self.p_lo.argmax(-1).flatten(), self.p_hi.argmax(-1).flatten())

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        if attention_mask is None:
            attention_mask = input_ids != 0
        with torch.autocast(input_ids.device.type, enabled=False):
            return self._forward(input_ids, attention_mask)

    def _forward(self, input_ids: Tensor, attention_mask: Tensor):
        dev = input_ids.device
        Nd, Xd, Td_, end, width = parse(input_ids, attention_mask)
        T = from_digits(Td_[:, :2])
        M = F.pad(Nd, (0, P - W))                                            # N's digits from the prompt, P places
        tabs = self.tables()
        rows, L, V = input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
        if self.training:
            aux = self.lessons(Nd, Xd, T, width, M, tabs)
            return torch.zeros(rows, L, V, device=dev), aux
        ctl = self.ctl
        if ctl.used is None:
            ctl.used = torch.zeros(200, dtype=torch.long, device=dev)
            ctl.pused = torch.zeros(100, dtype=torch.long, device=dev)
            self.report_tables(tabs)
        s = self.run(Xd, T, M, tabs, ctl.used, ctl.pused)
        n_used = (int((ctl.used > 0).sum()), int((ctl.pused > 0).sum()))
        if n_used != ctl.reported:
            ctl.reported = n_used
            unvisited = int(((ctl.used > 0) & (ctl.visits == 0)).sum()) if ctl.visits is not None else -1
            punvisited = int(((ctl.pused > 0) & (ctl.pvisits == 0)).sum()) if ctl.pvisits is not None else -1
            print(f"AUDIT eval_rows_used={n_used[0]}/200 used_but_never_trained={unvisited} "
                  f"eval_product_cells_used={n_used[1]}/100 product_used_but_never_trained={punvisited}", flush=True)
        place = end[:, None] - 1 - torch.arange(L, device=dev)[None]          # position p holds digit end - 1 - p
        d = s.gather(1, place.clamp(0, W - 1))
        tok = torch.where((place >= 0) & (place < W), d + DIGIT, 0)
        logits = torch.full((rows, L, V), -20.0, device=dev)
        return logits.scatter_(2, tok[..., None], 0.0), None

    def run(self, Sd: Tensor, T: Tensor, M: Tensor, tabs, visits: Tensor | None = None,
            pvisits: Tensor | None = None) -> Tensor:
        """The hard machine: T squarings (schoolbook, the product cell), each reduced by the chained long division."""
        Td, Tb, Plo, Phi = tabs
        s = Sd.clone()
        for t in range(int(T.max()) if T.numel() else 0):
            R = divide_hard(square_hard(s, tabs, ADDER, pvisits, visits), W, M, Td, Tb, visits)
            s = torch.where((T > t)[:, None], R[:, :W], s)
        return s

    def ladder(self, X: Tensor, tabs, PLO: Tensor, PHI: Tensor):
        """The product cell's lessons on the rows' own numbers X [n, W] (digits): X * 0 = 0, and the soft row
        X * (b + 1) against the machine's own hard X * b + X, where X's rungs 0..b all check out (LADDER_GATE). The
        check's adder is the designed one (as in the link checks). Returns the loss and the share of the hard rung
        checks that pass."""
        n = X.shape[0]
        Xq = F.pad(X, (0, Q - W))
        bs = torch.arange(10, device=X.device)
        with torch.no_grad():
            Xr = X[:, None, :].expand(n, 10, W)
            hard = row_hard(Xr, bs[None].expand(n, 10), tabs, "designed")
            tgt = add_hard(hard[:, :9], Xq[:, None].expand(n, 9, Q))[0]       # X * b + X, b = 0..8 (the adder)
            ok = torch.cat([(hard[:, :1] == 0).all(-1), (hard[:, 1:] == tgt).all(-1)], 1)   # rung c checks out [n, 10]
            verified = ok.long().cumprod(1).bool()                            # rungs 0..c all check out
            w = verified[:, :9].float() if LADDER_GATE else torch.ones(n, 9, device=X.device)
            trained = torch.cat([torch.ones_like(w[:, :1]), w], 1).bool()     # the rungs that get a lesson [n, 10]
            cells = (Xr * 10 + bs[None, :, None])[trained]
            self.ctl.pvisits += torch.bincount(cells.flatten(), minlength=100)
        nll0 = -row_soft(X, 0, PLO, PHI)[..., 0].clamp_min(1e-12).log().mean()
        loss = nll0
        for b in range(9):
            nll = -row_soft(X, b + 1, PLO, PHI).clamp_min(1e-12).log().gather(-1, tgt[:, b, :, None]).squeeze(-1)
            loss = loss + (nll.mean(-1) * w[:, b]).sum() / n
        return loss / 10, float(ok.float().mean())

    def lessons(self, Nd, Xd, T, width, M, tabs) -> dict:
        """Label-free lessons on the rows' own numbers (nothing made up):
          * the product cell's ladder on x and N (see `ladder`);
          * every link of the machine's own division of its own square of x checks itself, (D - N) + N = D, with the
            learned subtractor and a designed adder: the digits, and (ID_BORROW) the final borrow against the adder's
            final carry;
          * from phase B on (once the link checks pass on ID_SWITCH of the links): x is its own remainder (x < N).
            Dividing x by N with the soft own-borrow choice must stop the chain at trial 0 (SELF_STEPS steps from
            x // 10^SELF_STEPS), so the first link must end in a borrow;
          * phase B also adds the real rows' answers along the own path (REAL_MAX_T > 0), in steps in which every
            ladder check passes, on the rows whose squarings check out against the check's adder."""
        ctl = self.ctl
        ctl.step += 1
        dev = Xd.device
        Td, Tb, Plo, Phi = tabs
        if ctl.visits is None:
            ctl.visits = torch.zeros(200, dtype=torch.long, device=dev)
            ctl.pvisits = torch.zeros(100, dtype=torch.long, device=dev)
        zero = self.a.sum() * 0.0 + self.p_lo.sum() * 0.0
        if ctl.phase == "frozen":
            return {"loss": zero, "real": None, "ctl": ctl}
        TDC = torch.cat([self.a.float().softmax(-1), self.b.float().softmax(-1)], -1)
        TDC = TDC.permute(0, 2, 1, 3).reshape(20, 120)
        PLO, PHI = self.p_lo.float().softmax(-1), self.p_hi.float().softmax(-1)
        lad_loss, lad_ok = self.ladder(torch.cat([Xd, Nd]), tabs, PLO, PHI)
        X = Xd.shape[0]
        with torch.no_grad():
            V = square_hard(Xd, tabs)                                        # the machine's own square of x
            sq_row = (V == square_hard(Xd, tabs, "designed")).all(-1)        # its adder agrees with the check's
            sq_ok = float(sq_row.float().mean())
            D, live = link_inputs(V, W, M, Td, Tb)
            D, Mr = D.reshape(X * W * (S - 1), P), M.repeat_interleave(W * (S - 1), 0)
            if ID_LINKS == "live":
                keep = torch.nonzero(live.flatten()).flatten()
                D, Mr = D[keep], Mr[keep]
        Ds, bs = link_soft(F.one_hot(D, 10).float(), Mr, TDC)
        back, carry = add_soft(Ds, Mr)                                       # (D - N) + N on distributions
        weight = torch.ones(P, device=dev)
        weight[0] = ID_PLACE0                                               # the lowest place tells the swapped code
        nll = (-back.clamp_min(1e-12).log().gather(-1, D[..., None]).squeeze(-1) * weight).sum(-1)
        if ID_BORROW > 0:                                                   # final borrow = the adder's final carry
            agree = carry * bs[:, 1] + (1 - carry) * bs[:, 0]
            nll = nll - ID_BORROW * agree.clamp_min(1e-12).log()
        loss = nll.mean() / (weight.sum() + ID_BORROW) + LADDER_W * lad_loss
        xd = F.pad(Xd, (0, P - W))
        selfl = SELF_LESSON and ctl.phase == "B"
        if selfl:
            loss = loss + ce(divide_soft(Xd, SELF_STEPS, M, TDC), xd, width)
        with torch.no_grad():
            Dh, bh, idx = link_hard(D, Mr, Td, Tb)
            ctl.visits += torch.bincount(idx.flatten(), minlength=200)
            backh, carryh = add_hard(Dh, Mr)
            ok = (backh == D).all(-1)
            if ID_BORROW > 0:
                ok = ok & (carryh == bh)
            link_exact = float(ok.float().mean())
            exact = min(link_exact, lad_ok, sq_ok)
            if selfl:
                exact = min(exact, float((divide_hard(Xd, SELF_STEPS, M, Td, Tb) == xd).all(1).float().mean()))
        real = None
        if ctl.phase == "B" and REAL_MAX_T > 0 and lad_ok == 1.0:
            sel = torch.nonzero(T <= REAL_MAX_T).flatten()
            with torch.no_grad():                                            # rows whose squarings check out
                sp = self.run(Xd[sel], T[sel] - 1, M[sel], tabs)
                Vr = square_hard(sp, tabs)
                good = sq_row[sel] & (Vr == square_hard(sp, tabs, "designed")).all(-1)
                sel, Vr = sel[good], Vr[good]
            if sel.numel():
                Rr, Rrh = divide_path(Vr, W, M[sel], TDC, Td, Tb, ctl.visits)
                real = {"sel": sel, "R": Rr, "Rh": Rrh, "width": width[sel]}
        self.control(link_exact, exact, lad_ok, sq_ok, float(loss.detach()))
        return {"loss": loss, "real": real, "ctl": ctl}

    def control(self, link_exact: float, exact: float, lad_ok: float, sq_ok: float, loss: float) -> None:
        ctl = self.ctl
        ctl.win.append(link_exact)
        ctl.in_phase += 1
        mean = sum(ctl.win) / len(ctl.win)
        if ctl.step % LOG_EVERY == 0 or ctl.step == 1:
            print(f"LR step={ctl.step} phase={ctl.phase} in_phase={ctl.in_phase} loss={loss:.4f} "
                  f"hard_exact={exact:.4f} link_exact={link_exact:.4f} ladder_ok={lad_ok:.4f} squares_ok={sq_ok:.4f} "
                  f"win={mean:.4f} "
                  f"rows_visited={int((ctl.visits > 0).sum())}/200 cells_visited={int((ctl.pvisits > 0).sum())}/100",
                  flush=True)
        if ctl.phase == "A" and ((len(ctl.win) == STAGE_WIN and mean >= ID_SWITCH) or ctl.in_phase >= ID_STEPS):
            # the link checks pass on most links: "x mod N = x" and the rows' answers join, with a fresh optimizer
            ctl.history.append((ctl.step, "switch", ctl.in_phase, round(mean, 4)))
            print(f"LR switch_to_own_borrow step={ctl.step} win={mean:.4f} ladder_ok={lad_ok:.4f}", flush=True)
            ctl.phase, ctl.reset, ctl.in_phase = "B", True, 0
            ctl.win.clear()
        elif ctl.phase == "B":
            ctl.streak = ctl.streak + 1 if exact == 1.0 else 0
            if ctl.streak >= FREEZE_WIN:
                ctl.history.append((ctl.step, "frozen", ctl.in_phase, 1.0))
                print(f"LR frozen step={ctl.step} history={ctl.history}", flush=True)
                ctl.phase = "frozen"

    def report_tables(self, tabs) -> None:
        """At the first evaluation batch: the snapped tables and how often training's hard checks used each row."""
        ctl = self.ctl
        Td, Tb, Plo, Phi = tabs
        print("TABLE digit=" + "".join(str(int(d)) for d in Td.tolist()), flush=True)
        print("TABLE borrow=" + "".join(str(int(d)) for d in Tb.tolist()), flush=True)
        print("TABLE prod_lo=" + "".join(str(int(d)) for d in Plo.tolist()), flush=True)
        print("TABLE prod_hi=" + "".join(str(int(d)) for d in Phi.tolist()), flush=True)
        if ctl.visits is not None:
            v = ctl.visits
            print(f"TABLE train_visits min={int(v.min())} rows_never={int((v == 0).sum())} "
                  f"counts={','.join(str(int(c)) for c in v.tolist())}", flush=True)
            v = ctl.pvisits
            print(f"TABLE prod_train_visits min={int(v.min())} cells_never={int((v == 0).sum())} "
                  f"counts={','.join(str(int(c)) for c in v.tolist())}", flush=True)
        print(f"TABLE schedule phase={ctl.phase} step={ctl.step} history={ctl.history}", flush=True)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    aux = batch.auxiliary
    with torch.autocast(batch.labels.device.type, enabled=False):
        loss = aux["loss"]
        real = aux["real"]
        if real is None:
            return loss
        sel = real["sel"]
        labels, valid = batch.labels[sel], batch.valid_mask[sel]
        cnt = valid.long().sum(1)
        k = torch.arange(P, device=labels.device)[None]
        yd = (labels.gather(1, (cnt[:, None] - 1 - k).clamp(0, labels.shape[1] - 1)) - DIGIT).clamp(0, 9)
        yd = torch.where(k < cnt[:, None], yd, 0)                           # the label's digits, LSB first
        ctl = aux["ctl"]
        lesson = ce(real["R"], yd, real["width"])
        if ctl.step % LOG_EVERY == 0:
            ex = float((real["Rh"] == yd).all(1).float().mean())
            print(f"LR step={ctl.step} real_rows={sel.numel()} real_hard_exact={ex:.4f}", flush=True)
        return loss + lesson


class Schedule:
    """Scheduler: a fresh Adam state at the switch to phase B; learning rate 0 once the model is frozen."""

    def __init__(self, optimizer: torch.optim.Optimizer, ctl: Control) -> None:
        self.optimizer, self.ctl = optimizer, ctl

    def step(self) -> None:
        if self.ctl.reset:
            self.optimizer.state.clear()
            self.ctl.reset = False
        if self.ctl.phase == "frozen":
            for group in self.optimizer.param_groups:
                group["lr"] = 0.0


def build_model(spec: ModelSpec) -> LearnedSquare:
    model = LearnedSquare(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    # fused keeps Adam's step counter on the accelerator (the evaluator checks every optimizer state tensor)
    opt = torch.optim.Adam(model.parameters(), lr=LR, fused=spec.device_type in ("cuda", "mps"))
    return OptimizerBundle(opt, scheduler=Schedule(opt, model.ctl))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=BATCH,
    max_steps=MAX_STEPS,
    eval_batch_size=EVAL_BATCH,
)
