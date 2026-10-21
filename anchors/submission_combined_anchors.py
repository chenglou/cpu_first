"""LearnedCombined: x^(2^T) mod N in which every part of the calculator that computes something is a learned table:
the prompt reader and the answer writer (IO.md), the schoolbook squaring's times table (SQUARE.md), the subtraction
table that does every addition and subtraction (CHAIN.md, ADDER.md), and the commit gate that runs the T countdown and
each division step (CONTROL.md). One model, trained from a random start in one run.

Datapath (designed wiring, the schoolbook algorithms in base 10); the state is digits from the prompt to the answer:
  * read: three fields, N (8 slots), x (8 slots) and T (K = 2 slots), each by a learned pointer (anchor + one shared
    step), with a learned digit-ness per token and a learned code (token -> digit) per reading hypothesis (IO.md);
  * rounds: each round subtracts 1 from T's digits with the subtraction table, and the commit gate reads that
    subtraction's final borrow. A committing round squares the state and reduces it mod N and takes T - 1; a holding
    round keeps both (CONTROL.md). The rounds end when no row commits (at most 10^K);
      - squaring (SQUARE.md): row j of s * s is the product cell's lo digits of (s_i, s_j) plus its hi digits shifted
        by one place; the W rows, shifted by j, are summed one by one. Every sum is A - (0 - B) with the subtraction
        table (ADDER.md's learned adder);
      - reduction (CHAIN.md, CONTROL.md): restoring long division over the 2W digits of s^2. W steps, each a chain of
        S - 1 subtractions of N's read digits from one register with the table, each committed or held by the gate on
        its own final borrow;
  * write: the state's slot j at the position the writer's pointer gives slot j, its token by the writer's code.

Learned (MODE = "learned"): the subtraction table (r, m, borrow in) -> (digit, borrow out), 200 rows; the product
cell (a, b) -> (lo, hi), 100 rows; the commit gate borrow -> (hold, commit), 2 rows; the reader's digit-ness, codes,
anchors and steps and the writer's code (IO.md). MODE = "designed" starts every one of them at the truth and never
trains (the plumbing check).

The prediction path (PATH = "ste", rule 8): the digits are one-hot vectors and every table is applied to them as a
matrix, multilinearly, with a straight-through table: one-hot(argmax) + (softmax - softmax.detach()). The forward pass is
exactly the snapped machine (every value is 0 or 1), and the gradient of the final logits reaches every table's
logits through every lookup. The same holds for the reader (pointer positions, steps, digit-ness, the code of the
selected reading) and the writer. No integer value is formed anywhere: no int64 number, no integer T. The exact
machine's multilinear gradient grows about 5x per link, so the state's gradient is discounted by GRAD_DAMP after every
link, addition and round (the value is unchanged). PATH = "index" runs the same machine with the digits as integers
indexing the argmax tables (the pieces' way; outside autograd).

This file is submission_combined.py with the subtraction table taught as in anchors/submission_anchors.py
(TABLE_TEACH = "order", ANCHORS.md): no hand-written adder at any step. Until the reading is chosen the table has no
lesson. From then on: x < N on every top part (x[j:] - N[j:] borrows exactly when the digit strings differ there,
N[j:] - x[j:] never borrows) teaches the borrows; once x - N borrows and N - x does not on every row of GATE_WIN
batches, the order laws (r - m - c keeps the order of r, reverses the order of m, unless exactly one of the two wraps,
which then comes out larger; refereed by the learned borrows) and bijectivity (each column (m, c) and row (r, c) of the
cell uses every digit once) teach the digits; LATE_AFTER steps later the pieces' own table lessons below join, with the
learned adder from their first step (TABLE_TEACH = "adder" is the certified recipe, line for line).

What teaches the tables (the pieces' lessons; COMBINED.md says what changed where they meet):
  * the reader and writer: IO.md's stages (the answer round trip, the code round trips, Bradley-Terry on x < N, the
    reading's normalized mixture, the tiling, T's direction by speculation once the machine is exact);
  * the subtraction table, phase A: every link of the machine's own division of its own square of x checks itself,
    (D - N) + N = D, complete (the final borrow against the adder's final carry), the lowest place weighted ID_PLACE0;
    the adder is the designed one until the reading is chosen (and for at least ADDER_WARM steps), then the learned
    one, (D - N) - (0 - N) = D. With it the anchors that hold for any numbers: R - 0 = R, R - R = 0, and after the
    warm start 0 + N = N, R + N = N + R;
  * once the reading is chosen (IO's stage 4): the facts about the rows' numbers, x - N borrows, N - x does not,
    (x - N) - N does not (x < N), 0 - N borrows (N > 0); the learned adder takes over the check at the same step;
  * phase B, once the reading is chosen and the checks pass on ID_SWITCH of the links: Adam reset;
  * the product cell, once every check of the table has been exact for LADDER_WIN steps in a row in phase B: X * 0 = 0
    and the ladder X * (b + 1) = X * b + X along the verified frontier (SQUARE.md), on the rows' x and N, with the
    learned adder (the snapped table) for its targets, its checks and its soft rows (the soft rows' two ripples carried
    as one chain over their joint borrows, add_soft_joint);
  * phase C, once x - N borrows on CMP_SWITCH of the rows: the gate's facts "x mod N = x" and "N mod N = 0" on a
    detached table, and the rows' answers (T <= REAL_MAX_T) along the own hard path with the soft gate at the hard
    borrows, on the rows whose countdown, squarings and ladder check out (only learned arithmetic in the checks);
  * freeze (learning rate 0) after FREEZE_WIN consecutive phase-C steps in which every check is exact on the hard
    machine and T's direction is settled.
The schedule is Python-side state of the model (not model state). Everything runs in fp32 with autocast off.
"""

from __future__ import annotations

import itertools
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

MODE = "learned"       # "learned" | "designed" (every table, the gate and the I/O start at the truth, frozen: plumbing)
PATH = "ste"           # the prediction path: "ste" (one-hot digits through straight-through tables, inside autograd) |
                       # "index" (integer digits indexing the argmax tables, as the pieces)
GRAD_DAMP = 0.1        # "ste": the state's gradient is multiplied by this after every link, addition and round (the
                       # value is unchanged)
W = 8                  # digits of the widest modulus (a 24-bit N is below 10^8); one reduction takes W steps
P = W + 1              # register places: 10 R + v_t < 10 N < 10^9
Q = W + 2              # places of a schoolbook row X * b (X * 9 < 10^9; one more keeps any table's row whole)
S = 10                 # a step's chain has S - 1 links, enough for Rp - 9N
K = 2                  # the countdown's places: T's field
SLOTS = (W, W, K, W)   # the fields' slots: N, x, T; the writer
INIT = 0.1             # std of the initial logits
LR = 0.05              # Adam on the logits
# the subtraction table (CHAIN.md, ADDER.md)
ID_SWITCH = 0.8        # phase B once this share of the table's checks passes, averaged ...
STAGE_WIN = 20         # ... over this many steps,
ID_STEPS = 300         # ... or after this many steps (and in both cases only once the reading is chosen)
ID_PLACE0 = 5.0        # weight of the lowest place in the check (the swapped borrow code fails it only there)
ID_BORROW = 1.0        # weight of the final borrow in the check: the complete identity (D - N) + N = D
ADDER_WARM = 70        # the designed adder in the check for at least the first ADDER_WARM steps, then the learned one ...
WARM_TO_FACTS = True   # ... and (True) until the facts about x and N have joined: the learned adder needs them
                       # (ADDER.md: without "x - N borrows", "N - x does not", "(x - N) - N does not" it drifts to a
                       # consistent wrong table), and they need the reading (COMBINED.md section 2)
FACTS_FROM = "reading"  # the facts about x and N join: "reading" (once the reading is chosen) | "B" (with phase B)
TABLE_TEACH = "order"  # "order": the table learns from the order of the numbers (ANCHORS.md), no designed adder |
                       # "adder": the certified recipe (the designed adder until the reading is chosen)
GATE_WIN = 10          # "order": the digit lessons start once the borrow facts hold on every row of this many batches
LATE_AFTER = 100       # "order": the pieces' own table lessons join this many steps after that
ANCHOR_ZERO = 1.0      # R - 0 = R on the registers
ANCHOR_SELF = 1.0      # R - R = 0 on the registers
ANCHOR_NEG2 = 1.0      # 0 + N = N with the learned adder: 0 - (0 - N) = N (after the warm start)
ANCHOR_COMM = 1.0      # R + N = N + R with the learned adder (after the warm start)
ANCHOR_LESS = 1.0      # x < N: x - N borrows (once the reading is chosen; CONTROL.md's "x - N borrows")
ANCHOR_MORE = 1.0      # x < N: N - x does not borrow (once the reading is chosen)
ANCHOR_WRAP = 1.0      # x - N wraps to 10^P + x - N >= N: (x - N) - N does not borrow (once the reading is chosen)
ANCHOR_NEG = 1.0       # N > 0: 0 - N borrows (once the reading is chosen, after the warm start)
# the commit gate (CONTROL.md)
CMP_SWITCH = 0.99      # phase C once x - N borrows on this share of the rows, averaged over STAGE_WIN steps, ...
CMP_STEPS = 300        # ... or after this many steps of phase B
SELF_LESSON = True     # phase C: "x mod N = x" (one step from the register x keeps trial 0)
NMOD_LESSON = True     # phase C: "N mod N = 0" (one step from the register N keeps trial 1)
FACT_TABLE = False     # the two facts also teach the table (False: a detached copy, so they teach only the gate)
OWN_GATE = "mix"       # the own path's soft register keeps each trial with the soft gate at the hard borrows | "off"
# the product cell (SQUARE.md)
LADDER_W = 1.0         # weight of the ladder's loss
LADDER_WIN = 20        # the ladder joins once every check of the table has been exact this many steps in a row ...
LADDER_STEPS = 300     # ... or after this many steps of phase B
# the reader and writer (IO.md)
BT_TAU = 1.0           # Bradley-Terry temperature of the rank codes
PICK_PASS = 0.999      # the reading is chosen once a hypothesis passes x < N on this share of the batch ...
PICK_WIN = 20          # ... for this many steps in a row
CHOSEN = 0.99          # then the tiling joins once the mixture puts this posterior on one reading
T_AFTER = 20           # T's direction: speculate once the machine has been exact this many steps in a row ...
T_ROWS = 64            # ... on at most this many rows whose two readings of T differ
T_DONE = 0.999         # ... until the snapped reading reproduces their answers and carries this weight
REAL_MAX_T = 2         # phase C also trains on the batch's real rows with T <= REAL_MAX_T (one digit)
FREEZE_WIN = 200       # phase C: freeze after this many consecutive steps with every check exact
BATCH = 512
EVAL_BATCH = 16384
MAX_STEPS = 20000
LOG_EVERY = 250
EPS = 1e-6

PERMS = list(itertools.permutations(range(3)))
HYP = [(rn, dn, rx, dx) for rn in range(3) for rx in range(3) if rn != rx for dn in range(2) for dx in range(2)]
HIDX = {h: i for i, h in enumerate(HYP)}  # (N's run, N's direction, x's run, x's direction); direction 0: step +1


def tally(acc: Tensor | None, idx: Tensor, n: int) -> None:
    """Training / evaluation audits: how often each table row is used."""
    if acc is not None:
        acc += torch.bincount(idx.flatten(), minlength=n)


def exact_tables(device) -> tuple[Tensor, Tensor]:
    """The true subtractor (the designed start only): digit and borrow of r - m - c, flat over (r, m, c)."""
    r = torch.arange(10, device=device)[:, None, None]
    m = torch.arange(10, device=device)[None, :, None]
    c = torch.arange(2, device=device)[None, None, :]
    x = r - m - c
    return (x % 10).flatten(), (x < 0).long().flatten()


def exact_products(device) -> tuple[Tensor, Tensor]:
    """The true product cell (the designed start only): lo and hi digit of a * b, flat over (a, b)."""
    v = torch.arange(10, device=device)[:, None] * torch.arange(10, device=device)[None, :]
    return (v % 10).flatten(), (v // 10).flatten()


def ste(hard: Tensor, soft: Tensor) -> Tensor:
    """Straight-through: the value is exactly `hard` (soft - soft.detach() is exactly 0), the gradient is soft's."""
    return hard + (soft - soft.detach())


def damp(x: Tensor) -> Tensor:
    """The prediction path's state after a link, an addition or a round: the same value, its gradient times GRAD_DAMP.
    The multilinear gradient of the exact machine grows several times per link (each gate decision gathers every
    place's gradient into the borrow chain), so undamped it overflows within one division. (Without autograd, as at
    evaluation, there is no gradient to scale.)"""
    if GRAD_DAMP == 1.0 or not torch.is_grad_enabled():
        return x
    return x.detach() + GRAD_DAMP * (x - x.detach())


# ------------------------------------------------------------------------------------------------ hard digits (integers)
def sub_hard(D: Tensor, M: Tensor, Td: Tensor, Tb: Tensor):
    """D - M place by place from the lowest with the snapped table (flat Td, Tb [200] indexed r * 20 + m * 2 + c), any
    width and batch shape: digits [..., L], LSB first. Returns the digits, the final borrow and the table rows."""
    c = torch.zeros_like(D[..., 0])
    outs, rows = [], []
    for j in range(D.shape[-1]):
        idx = (D[..., j] * 10 + M[..., j]) * 2 + c
        outs.append(Td[idx])
        c = Tb[idx]
        rows.append(idx)
    return torch.stack(outs, -1), c, torch.stack(rows, -1)


def add_learned(A: Tensor, B: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None):
    """The learned adder (ADDER.md): A + B mod 10^L as A - (0 - B), both subtractions the table (in ten's complement
    0 - B is 10^L - B), and the final carry borrow(0 - B) - borrow(A - (0 - B))."""
    C, bc, r1 = sub_hard(torch.zeros_like(B), B, Td, Tb)
    Y, by, r2 = sub_hard(A, C, Td, Tb)
    tally(visits, r1, 200)
    tally(visits, r2, 200)
    return Y, bc - by


def add_hard(D: Tensor, M: Tensor) -> tuple[Tensor, Tensor]:
    """The designed adder (the warm start's check only): digits D + M, mod 10^L, and the final carry."""
    c = torch.zeros_like(D[..., 0])
    out = []
    for j in range(D.shape[-1]):
        x = D[..., j] + M[..., j] + c
        out.append(x % 10)
        c = x // 10
    return torch.stack(out, -1), c


def row_hard(X: Tensor, b: Tensor, tabs, visits: Tensor | None = None, pvisits: Tensor | None = None) -> Tensor:
    """Schoolbook row X * b on digits: X [..., W] times the digits b [...] with the snapped product cell (flat lo and hi
    tables [100] indexed a * 10 + b): lo at place i, hi at place i + 1, summed by the learned adder. Returns [..., Q]."""
    Td, Tb, Plo, Phi = tabs[:4]
    idx = X * 10 + b[..., None]
    z = torch.zeros_like(X[..., :1])
    tally(pvisits, idx, 100)
    return add_learned(torch.cat([Plo[idx], z, z], -1), torch.cat([z, Phi[idx], z], -1), Td, Tb, visits)[0]


def square_hard(Sd: Tensor, tabs, visits: Tensor | None = None, pvisits: Tensor | None = None, check: bool = False):
    """s^2 by schoolbook on digits Sd [X, W] -> the 2W digits of s^2 [X, 2W]. The W rows s * s_j side by side, then
    summed one by one, shifted by j, by the learned adder in 2W + 2 places (no table can wrap). With `check` also, per
    row, whether every one of its additions passes its own subtraction check (A + B) - B = A with borrow = carry."""
    X, (Td, Tb) = Sd.shape[0], tabs[:2]
    rows = row_hard(Sd[:, None, :].expand(X, W, W), Sd, tabs, visits, pvisits)       # [X, W (j), Q]
    acc = torch.zeros(X, 2 * W + 2, dtype=Sd.dtype, device=Sd.device)
    ok = torch.ones(X, dtype=torch.bool, device=Sd.device)
    for j in range(W):
        B = F.pad(rows[:, j], (j, 2 * W + 2 - Q - j))
        Y, cy = add_learned(acc, B, Td, Tb, visits)
        if check:
            A2, b2, _ = sub_hard(Y, B, Td, Tb)
            ok = ok & (A2 == acc).all(-1) & (b2 == cy)
        acc = Y
    return (acc[:, :2 * W], ok) if check else acc[:, :2 * W]


def ladder_hard(Xn: Tensor, tabs, visits: Tensor | None = None):
    """The ladder's hard rungs on the numbers Xn [n, W]: the machine's own rows X * b [n, 10, Q], the targets
    X * b + X (b = 0..8, the learned adder) and whether rung c checks out [n, 10] (rung 0: X * 0 = 0)."""
    n = Xn.shape[0]
    bs = torch.arange(10, device=Xn.device)
    hard = row_hard(Xn[:, None, :].expand(n, 10, W), bs[None].expand(n, 10), tabs, visits)
    tgt = add_learned(hard[:, :9], F.pad(Xn, (0, Q - W))[:, None].expand(n, 9, Q), tabs[0], tabs[1], visits)[0]
    ok = torch.cat([(hard[:, :1] == 0).all(-1), (hard[:, 1:] == tgt).all(-1)], 1)
    return hard, tgt, ok


def square_ok(Sd: Tensor, tabs) -> Tensor:
    """Per row, whether the machine's square of Sd checks out with learned parts only: the ladder verifies every rung up
    to its largest digit (the rows s * s_j), and every addition passes its subtraction check."""
    verified = ladder_hard(Sd, tabs)[2].long().cumprod(1).bool()
    rows_ok = verified.gather(1, Sd.max(1).values[:, None])[:, 0]
    return rows_ok & square_hard(Sd, tabs, check=True)[1]


def top(V: Tensor, steps: int) -> Tensor:
    """The digits of V // 10^steps in P places (V [X, L] digits): the division's starting register."""
    R = V[:, steps:steps + P]
    return F.pad(R, (0, P - R.shape[1]))


def step_hard(Rp: Tensor, M: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor, visits: Tensor | None = None,
              gvisits: Tensor | None = None):
    """One division step on digits: S - 1 links on one register, each D - N with the table, and the gate Gc [2]
    (1 = commit) on the link's final borrow: the register takes the difference or keeps D. Returns the register [X, P],
    every link's input [X, S - 1, P] and whether each link committed [X, S - 1]."""
    D = Rp
    ins, commits = [], []
    for _ in range(1, S):
        ins.append(D)
        E, c, idx = sub_hard(D, M, Td, Tb)
        k = Gc[c] == 1
        D = torch.where(k[:, None], E, D)
        tally(visits, idx, 200)
        tally(gvisits, c, 2)
        commits.append(k)
    return D, torch.stack(ins, 1), torch.stack(commits, 1)


def divide_hard(V: Tensor, M: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor, visits: Tensor | None = None,
                gvisits: Tensor | None = None) -> Tensor:
    """V mod N on digits V [X, 2W], from R = V // 10^W (below N) in W steps of the chain. Returns [X, P]."""
    R = top(V, W)
    for t in reversed(range(W)):
        R = step_hard(torch.cat([V[:, t, None], R[:, :P - 1]], 1), M, Td, Tb, Gc, visits, gvisits)[0]
    return R


def link_inputs(V: Tensor, M: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor) -> Tensor:
    """The hard division's link inputs at every step [X, W, S - 1, P]: the machine's own numbers, every link it
    computes (a held register makes the next link recompute the same subtraction)."""
    R = top(V, W)
    out = []
    for t in reversed(range(W)):
        R, ins, _ = step_hard(torch.cat([V[:, t, None], R[:, :P - 1]], 1), M, Td, Tb, Gc)
        out.append(ins)
    return torch.stack(out, 1)


def count_round(C: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor):
    """One round of the countdown C [X, K]: C - 1 with the table and the gate on its final borrow. Returns the
    difference, whether the round commits [X], the final borrow [X] and the table rows."""
    Cn, b, idx = sub_hard(C, F.one_hot(torch.zeros_like(C[:, 0]), C.shape[1]), Td, Tb)   # the digits of 1
    return Cn, Gc[b] == 1, b, idx


def run_hard(Sd: Tensor, C: Tensor, M: Tensor, tabs, cap: int | None = None, visits: Tensor | None = None,
             gvisits: Tensor | None = None, cvisits: Tensor | None = None, pvisits: Tensor | None = None):
    """The hard machine on digits: rounds until no row commits (at most `cap`, by default 10^K). A committing row
    squares and reduces its state and takes the countdown's difference; a holding row keeps both. Returns the state
    [X, W], the input of each row's last committed squaring [X, W] (x if none), the number of committed rounds [X]
    and whether the row has held [X]. `visits` / `pvisits` / `gvisits` accumulate the table / product / gate rows of
    the squarings and reductions, `cvisits` the table rows of the countdown."""
    Td, Tb, Plo, Phi, Gc = tabs
    s, prev = Sd.clone(), Sd.clone()
    rounds = torch.zeros_like(Sd[:, 0])
    held = torch.zeros_like(Sd[:, 0], dtype=torch.bool)
    for _ in range(10 ** C.shape[1] if cap is None else cap):
        Cn, k, b, idx = count_round(C, Td, Tb, Gc)
        tally(cvisits, idx, 200)
        tally(gvisits, b, 2)
        held = held | ~k
        if not bool(k.any()):
            break
        prev = torch.where(k[:, None], s, prev)
        R = divide_hard(square_hard(s, tabs, visits, pvisits), M, Td, Tb, Gc, visits, gvisits)
        s = torch.where(k[:, None], R[:, :W], s)
        C = torch.where(k[:, None], Cn, C)
        rounds = rounds + k.long()
    return s, prev, rounds, held


# ------------------------------------------------------------------------------------------------ digit distributions
def onehot0(n: int, L: int, device, dtype=torch.float32) -> Tensor:
    """n numbers of L exact 0 digits [n, L, 10]."""
    z = torch.zeros(n, L, 10, device=device, dtype=dtype)
    z[..., 0] = 1
    return z


def zpad(D: Tensor, low: int, high: int) -> Tensor:
    """Digit distributions [X, L, 10] padded with exact 0 digits below (low) and above (high)."""
    return torch.cat([onehot0(D.shape[0], low, D.device, D.dtype), D, onehot0(D.shape[0], high, D.device, D.dtype)], 1)


def link_soft(D: Tensor, M: Tensor, TDC: Tensor):
    """D - M on digit distributions D [X, L, 10] minus exact digits M [X, L], the cell applied multilinearly. TDC
    [20, 120]: rows (r, c), columns (m, 10 digit + 2 borrow probabilities). Returns the digits [X, L, 10] and the
    borrow out of every place [X, L, 2] (the last is the final borrow)."""
    X = D.shape[0]
    c = torch.zeros(X, 2, device=D.device)
    c[:, 0] = 1
    out, borrows = [], []
    for j in range(D.shape[1]):
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = (joint @ TDC).view(X, 10, 12)                                       # every m at once
        Y = Y.gather(1, M[:, j, None, None].expand(X, 1, 12)).squeeze(1)        # M's own digit
        out.append(Y[:, :10])
        c = Y[:, 10:]
        borrows.append(c)
    return torch.stack(out, 1), torch.stack(borrows, 1)


def sub_soft(D: Tensor, M: Tensor, TDC: Tensor, used: Tensor | None = None):
    """D - M with both operands digit distributions [X, L, 10] (multilinear). Returns the digits [X, L, 10] and the
    borrow out of every place [X, L, 2]. With one-hot operands and a straight-through table this is the prediction
    path's subtraction: every value is exactly the snapped table's. `used` accumulates the rows (r, m, c) met."""
    X = D.shape[0]
    c = torch.zeros(X, 2, device=D.device, dtype=D.dtype)
    c[:, 0] = 1
    out, borrows = [], []
    for j in range(D.shape[1]):
        if used is not None:
            used += torch.einsum("xr,xm,xc->rmc", D[:, j], M[:, j], c).detach().flatten()
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = torch.einsum("xmk,xm->xk", (joint @ TDC).view(X, 10, 12), M[:, j])   # expected over the subtrahend's digit
        out.append(Y[:, :10])
        c = Y[:, 10:]
        borrows.append(c)
    return torch.stack(out, 1), torch.stack(borrows, 1)


def add_soft_learned(A: Tensor, B: Tensor, TDC: Tensor, used: Tensor | None = None):
    """The learned adder on digit distributions [X, L, 10]: A - (0 - B). Returns the sum's digits and P(carry = 1)."""
    C, bc = sub_soft(onehot0(B.shape[0], B.shape[1], B.device, B.dtype), B, TDC, used)
    Y, by = sub_soft(A, C, TDC, used)
    return Y, bc[:, -1, 1] * by[:, -1, 0]


def add_soft_joint(A: Tensor, B: Tensor, TH: Tensor) -> Tensor:
    """The learned adder on digit distributions A, B [X, L, 10] for the ladder's soft rows: A - (0 - B) with the table
    TH (soft_table's layout), the two ripples carried as one chain over their joint borrow state (the borrow of 0 - B,
    the borrow of A - (0 - B)). The sum's digit marginals are then exact for independent digits; with the true table this
    is the designed soft adder exactly. add_soft_learned's two separate ripples treat the digits of 0 - B as independent
    although its borrow chain ties them: a soft hi digit then puts spurious errors on every place above it, which hold
    the ladder near P(9) = 0.93 for a hi digit that should be 0 (COMBINED.md section 2). Returns [X, L, 10]."""
    X = A.shape[0]
    T = TH.view(10, 2, 10, 12)                                              # [r, borrow in, m, digit + borrow out]
    Pd, Pb = T[..., :10], T[..., 10:]
    T0 = (Pd[0][..., :, None] * Pb[0][..., None, :]).permute(1, 0, 2, 3).reshape(10, 40)   # m -> (bc, C, bc')
    T1 = (Pd[..., :, None] * Pb[..., None, :]).reshape(10, 400)             # r -> (by, C, Y, by')
    s = torch.zeros(X, 2, 2, device=A.device, dtype=A.dtype)                # P(bc, by): the two borrows in
    s[:, 0, 0] = 1
    out = []
    for j in range(A.shape[1]):
        P1 = (B[:, j] @ T0).view(X, 2, 10, 2)                               # 0 - B: [x, bc, C, bc']
        P2 = (A[:, j] @ T1).view(X, 20, 20)                                 # A - C: [x, (by, C), (Y, by')]
        Q1 = torch.einsum("xab,xacp->xpbc", s, P1).reshape(X, 2, 20)        # [x, bc', (by, C)]
        J = torch.bmm(Q1, P2).view(X, 2, 10, 2)                             # [x, bc', Y, by']
        out.append(J.sum((1, 3)))
        s = J.sum(2)
    return torch.stack(out, 1)


def add_soft(D: Tensor, M: Tensor) -> tuple[Tensor, Tensor]:
    """The designed adder (the warm start's check only) on digit distributions D [..., L, 10] plus exact digits
    M [..., L], mod 10^L (multilinear), and the probability of a final carry."""
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


def row_soft(X: Tensor, b: int, PLO: Tensor, PHI: Tensor, TH: Tensor) -> Tensor:
    """The row X * b on digit distributions: the soft product cell PLO, PHI [10, 10, 10] (a, b) -> digit
    probabilities on the exact digits X [n, W], summed by the learned adder on the snapped table TH (one joint chain
    over both ripples' borrows, add_soft_joint). Returns [n, Q, 10]."""
    return add_soft_joint(zpad(PLO[X, b], 0, 2), zpad(PHI[X, b], 1, 1), TH)


def trials_soft(D: Tensor, M: Tensor, TDC: Tensor) -> tuple[Tensor, Tensor]:
    """A step's trials on digit distributions: trial 0 is the register D [X, P, 10], trial j is trial j - 1 minus N.
    Returns the trials [X, S, P, 10] and each link's final borrow [X, S - 1, 2]."""
    trials, borrows = [D], []
    for _ in range(1, S):
        E, c = link_soft(trials[-1], M, TDC)
        trials.append(E)
        borrows.append(c[:, -1])
    return torch.stack(trials, 1), torch.stack(borrows, 1)


def keep_soft(trials: Tensor, pc: Tensor) -> Tensor:
    """The register after a step whose link j + 1 commits with probability pc[:, j] [X, S - 1]: trial j is kept with the
    probability that links 1..j commit and link j + 1 holds (trial S - 1 when every link commits)."""
    alive = torch.ones_like(pc[:, 0])
    R = torch.zeros_like(trials[:, 0])
    for j in range(S - 1):
        R = R + (alive * (1 - pc[:, j]))[:, None, None] * trials[:, j]
        alive = alive * pc[:, j]
    return R + alive[:, None, None] * trials[:, S - 1]


def chain_soft(D: Tensor, M: Tensor, TDC: Tensor, GS: Tensor) -> Tensor:
    """One division step on digit distributions from the register D [X, P, 10], the gate GS [2] = P(commit | borrow 0,
    1) on each link's soft borrow (CONTROL.md's trial mixture)."""
    trials, b = trials_soft(D, M, TDC)
    return keep_soft(trials, (b * GS).sum(-1))


def divide_path(V: Tensor, M: Tensor, TDC: Tensor, GS: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor,
                visits: Tensor | None = None, gvisits: Tensor | None = None):
    """The model's own hard path, learned along (CONTROL.md): at each step the hard datapath computes the trials and
    their borrows and keeps the trial before the gate's first hold; the soft cell computes the same trials from the soft
    register and keeps them with the soft gate's probabilities at the hard borrows. V [X, 2W] digits (the machine's own
    hard square). Returns the soft remainder [X, P, 10] and the hard one [X, P]."""
    Rh = top(V, W)
    Rs = F.one_hot(Rh, 10).float()
    for t in reversed(range(W)):
        vt = V[:, t]
        Dh = torch.cat([vt[:, None], Rh[:, :P - 1]], 1)
        hard, bh = [Dh], []
        for _ in range(1, S):
            Eh, c, idx = sub_hard(hard[-1], M, Td, Tb)
            hard.append(Eh)
            bh.append(c)
            tally(visits, idx, 200)
            tally(gvisits, c, 2)
        bh = torch.stack(bh, 1)                                             # [X, S - 1]
        q = (Gc[bh] == 1).long().cumprod(1).sum(1)                          # commits before the first hold
        Rh = torch.stack(hard, 1).gather(1, q[:, None, None].expand(-1, 1, P)).squeeze(1)
        trials = trials_soft(torch.cat([F.one_hot(vt, 10).float()[:, None], Rs[:, :P - 1]], 1), M, TDC)[0]
        pc = GS[bh] if OWN_GATE == "mix" else (torch.arange(S - 1, device=V.device)[None] < q[:, None]).float()
        Rs = keep_soft(trials, pc)
    return Rs, Rh


def ce(R: Tensor, target: Tensor, width: Tensor) -> Tensor:
    """Cross-entropy of the remainder's digits below N's width (a remainder below N has `width` digits)."""
    scored = (torch.arange(P, device=R.device)[None] < width[:, None]).float()
    nll = -R.clamp_min(1e-12).log().gather(-1, target[..., None]).squeeze(-1)
    return (nll * scored).sum() / scored.sum().clamp_min(1)


# ------------------------------------------------------------------------------------------------ the prediction path
def square_ste(Sd: Tensor, TDC: Tensor, PROD: Tensor, used: Tensor | None = None, pused: Tensor | None = None):
    """s^2 by schoolbook on one-hot digits Sd [X, W, 10] with the straight-through product cell PROD [100, 20] and
    subtraction table TDC: square_hard's wiring. Returns [X, 2W, 10]."""
    X = Sd.shape[0]
    cell = (Sd[:, None, :, :, None] * Sd[:, :, None, None, :]).reshape(X, W, W, 100)   # [x, j, i, (s_i, s_j)]
    if pused is not None:
        pused += cell.detach().sum((0, 1, 2))
    lohi = cell @ PROD
    lo, hi = lohi[..., :10].reshape(X * W, W, 10), lohi[..., 10:].reshape(X * W, W, 10)
    rows = damp(add_soft_learned(zpad(lo, 0, 2), zpad(hi, 1, 1), TDC, used)[0]).view(X, W, Q, 10)
    acc = onehot0(X, 2 * W + 2, Sd.device, Sd.dtype)
    for j in range(W):
        acc = damp(add_soft_learned(acc, zpad(rows[:, j], j, 2 * W + 2 - Q - j), TDC, used)[0])
    return acc[:, :2 * W]


def divide_ste(V: Tensor, M: Tensor, TDC: Tensor, G: Tensor, used: Tensor | None = None,
               gused: Tensor | None = None) -> Tensor:
    """V mod N on one-hot digits V [X, 2W, 10], M [X, P, 10] (divide_hard's wiring): each link's final borrow, read by
    the straight-through gate G [2, 2], commits (the register takes the difference) or holds. Returns [X, P, 10]."""
    R = zpad(V[:, W:], 0, P - W)
    for t in reversed(range(W)):
        D = torch.cat([V[:, t:t + 1], R[:, :P - 1]], 1)
        for _ in range(1, S):
            E, bor = sub_soft(D, M, TDC, used)
            b = bor[:, -1]
            if gused is not None:
                gused += b.detach().sum(0)
            k = (b * G[:, 1]).sum(-1)[:, None, None]                        # 1: commit, 0: hold
            D = damp(k * E + (1 - k) * D)
        R = D
    return R


def run_ste(Sd: Tensor, C: Tensor, M: Tensor, TDC: Tensor, PROD: Tensor, G: Tensor, used: Tensor | None = None,
            cused: Tensor | None = None, pused: Tensor | None = None, gused: Tensor | None = None) -> Tensor:
    """run_hard on one-hot digits: the countdown C [X, K, 10] takes C - 1 with the table each round and the gate on
    its final borrow decides; a committing row squares and reduces its state. The rounds end when no row commits."""
    one = onehot0(C.shape[0], C.shape[1], C.device, C.dtype)                # the digits of 1
    one[:, 0] = F.one_hot(torch.tensor(1, device=C.device), 10).to(C.dtype)
    s = Sd
    for _ in range(10 ** C.shape[1]):
        Cn, bor = sub_soft(C, one, TDC, cused)
        b = bor[:, -1]
        if gused is not None:
            gused += b.detach().sum(0)
        k = (b * G[:, 1]).sum(-1)
        if not bool((k > 0.5).any()):
            break
        R = divide_ste(square_ste(s, TDC, PROD, used, pused), M, TDC, G, used, gused)
        kk = k[:, None, None]
        s = damp(kk * R[:, :W] + (1 - kk) * s)
        C = damp(kk * Cn + (1 - kk) * C)
    return s


def shift(A: Tensor, d: int) -> Tensor:
    """Position distributions [X, Lp] moved by d places (mass at q goes to q + d; zero-filled)."""
    if d >= 0:
        return F.pad(A, (d, 0))[:, :A.shape[1]]
    return F.pad(A, (0, -d))[:, -d:]


def order_pairs(Tb: Tensor, used: Tensor) -> list:
    """The pairs of table entries the two order laws compare, each oriented (lower, upper) by the learned borrows
    (anchors/submission_anchors.py). Entries are flat (r, m, c) -> r * 20 + m * 2 + c; only entries in `used` take
    part. An entry wraps when its borrow out is 1, and r < r' when r - r' borrows.
      * columns: (r, m, c) and (r', m, c), r < r': the digits keep their order, unless exactly one wraps, which then
        comes out larger;
      * rows: (r, m, c) and (r, m', c), m < m': subtracting more gives less, unless exactly one wraps, which then comes
        out larger."""
    dev = Tb.device
    ent = torch.arange(200, device=dev)
    r, m, c = ent // 20, (ent // 2) % 10, ent % 2
    e0 = ent.repeat_interleave(10)
    k = torch.arange(10, device=dev).repeat(200)
    out = []
    e1 = k * 20 + (e0 % 20)                                                # columns
    ok = (r[e0] < k) & used[e0] & used[e1]
    a, b = e0[ok], e1[ok]
    wa, wb = Tb[a], Tb[b]
    lt = Tb[(r[a] * 10 + r[b]) * 2]                                        # r < r'
    first = (wa < wb) | ((wa == wb) & (lt == 1))
    out.append((torch.where(first, a, b), torch.where(first, b, a)))
    e1 = (r[e0] * 10 + k) * 2 + c[e0]                                      # rows
    ok = (m[e0] < k) & used[e0] & used[e1]
    a, b = e0[ok], e1[ok]
    wa, wb = Tb[a], Tb[b]
    lt = Tb[(m[a] * 10 + m[b]) * 2]                                        # m < m'
    first = (wa < wb) | ((wa == wb) & (lt == 0))                           # a (less taken) comes out smaller
    out.append((torch.where(first, a, b), torch.where(first, b, a)))
    return out


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Control:
    """The training schedule (Python-side, not model state): the I/O stages, the machine's phases A, B, C, frozen,
    whether the ladder has joined; and the audits' counters."""

    def __init__(self) -> None:
        designed = MODE == "designed"
        self.step = 0
        self.phase = "frozen" if designed else "A"
        self.io = 5 if designed else 1        # 1 answers, 2 ranks, 3 reading, 4 tiling, 5 T's direction done
        self.ladder = designed                # the product cell's ladder has joined
        self.table_streak = 0                 # steps in a row with every check of the table exact (phase B on)
        self.ladder_wait = 0
        self.in_phase = 0
        self.streak = 0
        self.pick = 0                         # steps in a row with a hypothesis passing x < N
        self.best = 0
        self.rows3 = 0.0
        self.chosen = 0.0                     # the mixture's mean posterior on its favourite reading
        self.io_exact = False                 # the I/O's own hard checks, this step
        self.reset = False                    # the scheduler clears Adam's state at the switch to phase B
        self.reset_params: list = []          # ... or only these parameters' (the product cell when the ladder joins)
        self.win: deque = deque(maxlen=STAGE_WIN)
        self.visits: Tensor | None = None     # table rows used by training's hard checks and own path (the audit)
        self.pvisits: Tensor | None = None    # product cells the ladder trained
        self.gvisits: Tensor | None = None    # gate rows (borrow 0, 1) used by training
        self.used: Tensor | None = None       # at evaluation: table rows of the squarings and reductions, ...
        self.used_count: Tensor | None = None  # ... of the countdown, ...
        self.pused: Tensor | None = None      # ... product cells, ...
        self.used_gate: Tensor | None = None  # ... gate rows
        self.reported = None
        self.history: list = []
        self.t_log = ""
        self.facts_streak = 0                 # TABLE_TEACH "order": batches in a row passing the borrow facts
        self.gated_on: int | None = None      # ... the step the digit lessons started


class LearnedCombined(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        V = spec.vocab_size
        self.config = Config(V, spec.max_seq_len)
        self.V, self.END, self.OUT = V, V, V + 1                             # virtual tokens: the end, beyond it
        self.a = nn.Parameter(INIT * torch.randn(10, 10, 2, 10))           # subtractor: (r, m, borrow in) -> digit
        self.b = nn.Parameter(INIT * torch.randn(10, 10, 2, 2))            # ... -> borrow out
        self.p_lo = nn.Parameter(INIT * torch.randn(10, 10, 10))           # product cell: (a, b) -> lo digit
        self.p_hi = nn.Parameter(INIT * torch.randn(10, 10, 10))           # ... -> hi digit
        self.g = nn.Parameter(INIT * torch.randn(2, 2))                    # commit gate: borrow -> (hold, commit)
        self.dg = nn.Parameter(INIT * torch.randn(V))                      # token -> is a digit
        self.u = nn.Parameter(INIT * torch.randn(len(HYP), V))             # per reading hypothesis: token scores
        self.wc = nn.Parameter(INIT * torch.randn(10, V))                  # writer: value -> token
        self.anc = nn.Parameter(INIT * torch.randn(4, V + 1))              # field anchor: a token id, or the end
        self.strd = nn.Parameter(INIT * torch.randn(4, 2))                 # field step: +1 | -1
        self.ctl = Control()
        if MODE == "designed":
            self.designed()

    @torch.no_grad()
    def designed(self) -> None:
        """Every table at the truth (the plumbing check only): the subtractor, the product cell, the gate (commit unless
        the subtraction borrows); N read after its marker, x before T's marker, T and the answer at the end, reading
        leftwards; digits 7..16 valued 0..9."""
        hot = lambda t, n: 10.0 * F.one_hot(t, n).float() - 5.0
        Td, Tb = exact_tables("cpu")
        Plo, Phi = exact_products("cpu")
        self.a.copy_(hot(Td, 10).view(10, 10, 2, 10))
        self.b.copy_(hot(Tb, 2).view(10, 10, 2, 2))
        self.p_lo.copy_(hot(Plo, 10).view(10, 10, 10))
        self.p_hi.copy_(hot(Phi, 10).view(10, 10, 10))
        self.g.copy_(hot(torch.tensor([1, 0]), 2))
        DIGIT, MARK_N, MARK_T = 7, 2, 4
        self.dg.fill_(-5.0)
        self.dg[DIGIT:DIGIT + 10] = 5.0
        self.u.copy_(torch.arange(self.V, dtype=torch.float)[None].expand_as(self.u))
        self.wc.fill_(-5.0)
        self.wc[torch.arange(10), DIGIT + torch.arange(10)] = 5.0
        self.anc.fill_(-5.0)
        self.strd.fill_(-5.0)
        for f, (tok, step) in enumerate([(MARK_N, 0), (MARK_T, 1), (self.END, 1), (self.END, 1)]):
            self.anc[f, tok] = 5.0
            self.strd[f, step] = 5.0

    def zero(self) -> Tensor:
        return sum(p.sum() for p in self.parameters()) * 0.0

    # ------------------------------------------------------------------ the tables
    def tables(self):
        """The snapped tables: subtractor digit and borrow [200], product lo and hi [100], gate [2] (1 = commit)."""
        return (self.a.argmax(-1).flatten(), self.b.argmax(-1).flatten(), self.p_lo.argmax(-1).flatten(),
                self.p_hi.argmax(-1).flatten(), self.g.argmax(-1))

    def soft_table(self) -> Tensor:
        """The subtractor's softmax, laid out [20 (r, c), 120 (m, 10 digit + 2 borrow)]."""
        T = torch.cat([self.a.float().softmax(-1), self.b.float().softmax(-1)], -1)
        return T.permute(0, 2, 1, 3).reshape(20, 120)

    def hard_table(self) -> Tensor:
        """The snapped subtractor as one-hot rows in soft_table's layout (no gradient): the learned adder of the
        ladder's soft rows."""
        T = torch.cat([F.one_hot(self.a.argmax(-1), 10), F.one_hot(self.b.argmax(-1), 2)], -1).float()
        return T.permute(0, 2, 1, 3).reshape(20, 120)

    def gate_soft(self) -> Tensor:
        """P(commit | borrow 0, 1) [2]."""
        return self.g.float().softmax(-1)[:, 1]

    def ste_tables(self):
        """The straight-through tables of the prediction path: subtractor [20, 120] (soft_table's layout), product cell
        [100, 20] (lo, hi), gate [2, 2] (borrow -> hold, commit). Values are the snapped tables exactly."""
        hot = lambda p: ste(F.one_hot(p.argmax(-1), p.shape[-1]).float(), p.float().softmax(-1))
        T = torch.cat([hot(self.a), hot(self.b)], -1).permute(0, 2, 1, 3).reshape(20, 120)
        return T, torch.cat([hot(self.p_lo), hot(self.p_hi)], -1).reshape(100, 20), hot(self.g)

    # ------------------------------------------------------------------ the I/O pieces (IO.md)
    def ext(self, ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Token ids with END at the prompt's end and OUT beyond, padded with OUT by W + 1 on each side (so every
        pointer stays inside): [X, L + 1 + 2 (W + 1)], and the prompt's length [X]."""
        end = mask.long().sum(1)
        p = torch.arange(ids.shape[1] + 1, device=ids.device)[None]
        e = torch.where(p < end[:, None], F.pad(ids, (0, 1)),
                        torch.where(p == end[:, None], self.END, self.OUT))
        return F.pad(e, (W + 1, W + 1), value=self.OUT), end

    def isdig(self, e: Tensor) -> Tensor:
        return (e < self.V) & (self.dg[e.clamp(max=self.V - 1)] > 0)

    def ranks(self, k: Tensor) -> Tensor:
        """The code of hypothesis k [X] -> [X, V]: each digit token's rank among the digit tokens (others 0)."""
        isd = self.dg > 0
        u = self.u[k.clamp(min=0)]
        below = ((u[..., None, :] < u[..., :, None]) & isd).sum(-1)
        return torch.where(isd, below, 0).clamp(max=9)

    def wsm(self) -> Tensor:
        """The writer's code P(token | value) [10, V]: only tokens believed to be digits can be written."""
        return (self.wc.float() + F.logsigmoid(self.dg.float())[None]).softmax(-1)

    def valid_anchor(self, e: Tensor, end: Tensor) -> Tensor:
        p = torch.arange(e.shape[1], device=e.device)[None]
        return (p >= W + 1) & (p <= W + 1 + end[:, None])

    def anchor_logits(self, f: int, e: Tensor, end: Tensor) -> Tensor:
        return self.anc[f].float()[e.clamp(max=self.END)].masked_fill(~self.valid_anchor(e, end), -1e9)

    def anchor(self, f: int, e: Tensor, end: Tensor) -> Tensor:
        """Field f's soft anchor over the prompt's positions and its end [X, Lp] (zero elsewhere)."""
        return self.anchor_logits(f, e, end).softmax(-1)

    def hard_pos(self, f: int, e: Tensor, end: Tensor):
        """Field f's snapped pointer: its slots' positions [X, n], its anchor [X] and its step."""
        q = self.anchor_logits(f, e, end).argmax(-1)
        s = 1 if int(self.strd[f].argmax()) == 0 else -1
        return q[:, None] + (torch.arange(SLOTS[f], device=e.device)[None] + 1) * s, q, s

    def runs(self, e: Tensor, end: Tensor):
        """The snapped digit runs: rows with exactly three; each run's first and last position and its length."""
        p = torch.arange(e.shape[1], device=e.device)[None]
        isd = (p >= W + 1) & (p < W + 1 + end[:, None]) & self.isdig(e)
        start = isd & ~F.pad(isd[:, :-1], (1, 0))
        stop = isd & ~F.pad(isd[:, 1:], (0, 1))
        ok = start.sum(1) == 3
        si = torch.where(start, p, e.shape[1] - 2).sort(1).values[:, :3]
        ti = torch.where(stop, p, e.shape[1] - 2).sort(1).values[:, :3]
        return ok, si, ti, ti - si + 1

    def which_run(self, f: int, e: Tensor, end: Tensor, si: Tensor, ti: Tensor, ok: Tensor, length: Tensor):
        """The whole run field f's snapped pointer reads: its index, its direction (0: step +1) and whether it is one."""
        _, q, s = self.hard_pos(f, e, end)
        first = q + s
        inside = (first[:, None] >= si) & (first[:, None] <= ti)
        r = inside.float().argmax(1)
        edge = (si if s > 0 else ti).gather(1, r[:, None])[:, 0] - s
        whole = inside.any(1) & (q == edge) & ok & (length.gather(1, r[:, None])[:, 0] <= SLOTS[f])
        return r, int(s < 0), whole

    def selected(self, e: Tensor, end: Tensor) -> Tensor:
        """Per row, the hypothesis the snapped N and x pointers make (-1: not a reading of two whole runs)."""
        ok, si, ti, length = self.runs(e, end)
        rn, dn, okn = self.which_run(0, e, end, si, ti, ok, length)
        rx, dx, okx = self.which_run(1, e, end, si, ti, ok, length)
        lut = torch.full((3, 2, 3, 2), -1, dtype=torch.long, device=e.device)
        for h, i in HIDX.items():
            lut[h] = i
        return torch.where(okn & okx & (rn != rx), lut[rn, dn, rx, dx], -1)

    def read(self, f: int, e: Tensor, end: Tensor, code: Tensor):
        """Field f read by its snapped pointer with the code [X, V]: digits [X, n] (LSB first, 0 after the first
        non-digit) and the number of digits read [X]."""
        pos, _, _ = self.hard_pos(f, e, end)
        t = e.gather(1, pos)
        live = self.isdig(t).long().cumprod(-1).bool()
        return torch.where(live, code.gather(1, t.clamp(max=self.V - 1)), 0), live.sum(1)

    def label_seq(self, e: Tensor, labels: Tensor, valid: Tensor, tpos: Tensor) -> Tensor:
        """The answer's tokens placed at their prompt positions (the evaluator's target positions), OUT elsewhere."""
        labe = torch.full_like(e, self.OUT)
        labe.scatter_(1, torch.where(valid, tpos + W + 1, 0), torch.where(valid, labels, self.OUT))
        labe[:, 0] = self.OUT
        return labe

    def read_answer(self, labe: Tensor, e: Tensor, end: Tensor, code: Tensor) -> Tensor:
        """The answer read at the writer's positions with the reader's code: its digits [X, W], LSB first."""
        pos, _, _ = self.hard_pos(3, e, end)
        t = labe.gather(1, pos)
        live = self.isdig(t).long().cumprod(-1).bool()
        return torch.where(live, code.gather(1, t.clamp(max=self.V - 1)), 0)

    def reading_weights(self, e: Tensor, end: Tensor) -> Tensor:
        """Per row, the soft weight of each reading hypothesis [X, H]: the N and x pointers' probabilities of reading
        that hypothesis's whole runs (io_lessons' mixture), normalized. The gradient side of the selected reading."""
        X, Lp = e.shape
        _, si, ti, length = self.runs(e, end)
        A = torch.stack([self.anchor(f, e, end) for f in range(2)], 1)                     # [X, field, Lp]
        sl = torch.stack([self.strd[f].float().softmax(-1) for f in range(2)])             # [field, dir]
        fits = length[:, None, :] <= torch.tensor(SLOTS[:2], device=e.device)[None, :, None]
        left = A.gather(2, (si - 1)[:, None].expand(X, 2, 3)) * sl[None, :, None, 0]
        right = A.gather(2, (ti + 1).clamp(max=Lp - 1)[:, None].expand(X, 2, 3)) * sl[None, :, None, 1]
        rho = torch.stack([left, right], -1) * fits[..., None]                             # [X, field, run, dir]
        hyp = torch.tensor(HYP, device=e.device)
        w = rho[:, 0, hyp[:, 0], hyp[:, 1]] * rho[:, 1, hyp[:, 2], hyp[:, 3]] + EPS
        return w / w.sum(-1, keepdim=True)

    def codes_ste(self) -> Tensor:
        """Every hypothesis's code as straight-through one-hots [H, V, 10]: the value is the rank among the digit tokens
        (ranks), the gradient a soft rank's (each token's expected number of digit tokens scored below it)."""
        u, isd = self.u.float(), self.dg > 0
        below = ((u[:, None, :] < u[:, :, None]) & isd[None, None, :]).sum(-1)
        hard = F.one_hot(torch.where(isd[None], below, 0).clamp(max=9), 10).float()
        pd = torch.sigmoid(self.dg.float())
        sr = (torch.sigmoid((u[:, :, None] - u[:, None, :]) / BT_TAU) * pd[None, None, :]).sum(-1) - 0.5 * pd[None]
        soft = (-(torch.arange(10, device=u.device)[None, None] - sr[..., None]) ** 2).softmax(-1)
        return ste(hard, soft)

    # ------------------------------------------------------------------ the model
    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        if attention_mask is None:
            attention_mask = input_ids != 0
        with torch.autocast(input_ids.device.type, enabled=False):
            return self._forward(input_ids, attention_mask)

    def _forward(self, input_ids: Tensor, attention_mask: Tensor):
        dev = input_ids.device
        e, end = self.ext(input_ids, attention_mask)
        k = self.selected(e, end)
        rows, L, V = input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
        if self.training:
            code = self.ranks(k)                                             # [X, V] the selected reader's code
            Nd, width = self.read(0, e, end, code)
            Xd, _ = self.read(1, e, end, code)
            Cd, tcount = self.read(2, e, end, code)                          # T's digits, LSB first: the countdown
            M = F.pad(Nd, (0, P - W))                                        # N's digits as read, P places
            small = (tcount == 1) & (Cd[:, 0] >= 1) & (Cd[:, 0] <= REAL_MAX_T)  # the rows whose answers are used
            aux = self.lessons(Nd, Xd, Cd, small, width, M, self.tables(), e, end, k, code)
            return torch.zeros(rows, L, V, device=dev), aux
        ctl = self.ctl
        if ctl.used is None:
            ctl.used = torch.zeros(200, device=dev)
            ctl.used_count = torch.zeros(200, device=dev)
            ctl.pused = torch.zeros(100, device=dev)
            ctl.used_gate = torch.zeros(2, device=dev)
            self.report_tables(e, end)
        if PATH == "ste":
            logits = self.predict_ste(e, end, k, L)
        else:
            logits = self.predict_index(e, end, k, L)
        self.audit()
        return logits, None

    def predict_index(self, e: Tensor, end: Tensor, k: Tensor, L: int) -> Tensor:
        """The machine on integer digits indexing the argmax tables (the pieces' prediction path)."""
        ctl, dev, rows = self.ctl, e.device, e.shape[0]
        code = self.ranks(k)
        Nd, _ = self.read(0, e, end, code)
        Xd, _ = self.read(1, e, end, code)
        Cd, _ = self.read(2, e, end, code)
        used, cused, pused, gused = [torch.zeros_like(t, dtype=torch.long) for t in
                                     (ctl.used, ctl.used_count, ctl.pused, ctl.used_gate)]
        s = run_hard(Xd, Cd, F.pad(Nd, (0, P - W)), self.tables(), None, used, gused, cused, pused)[0]
        for acc, n in ((ctl.used, used), (ctl.used_count, cused), (ctl.pused, pused), (ctl.used_gate, gused)):
            acc += n
        pos, _, _ = self.hard_pos(3, e, end)
        p = pos - (W + 1)
        tok = self.wsm().argmax(-1)[s]
        inside = (p >= 0) & (p < L)
        logits = torch.full((rows, L, self.V), -20.0, device=dev)
        r = torch.arange(rows, device=dev)[:, None].expand_as(p)
        logits[r[inside], p[inside], tok[inside]] = 0.0
        return logits

    def predict_ste(self, e: Tensor, end: Tensor, k: Tensor, L: int) -> Tensor:
        """The prediction path inside autograd: the reader, the machine and the writer on one-hot digits with
        straight-through tables. Every value equals predict_index's; the final logits' gradient reaches every table."""
        ctl, V = self.ctl, self.V
        X, Lp = e.shape
        E1 = F.one_hot(e, V + 2).float()                                     # the tokens, with END and OUT [X, Lp, V + 2]
        DG = ste((self.dg > 0).float(), torch.sigmoid(self.dg.float()))     # digit-ness per token [V]
        isdv = F.pad(DG, (0, 2))                                             # END and OUT are not digits
        hot = lambda lg: ste(F.one_hot(lg.argmax(-1), lg.shape[-1]).float(), lg.softmax(-1))
        anchors = [hot(self.anchor_logits(f, e, end)) for f in range(4)]     # [X, Lp] each
        steps = [hot(self.strd[f].float()) for f in range(4)]               # [2] each: +1, -1
        H = len(HYP)
        Wh = ste(F.one_hot(k.clamp(min=0), H).float(), self.reading_weights(e, end))
        code = torch.einsum("xh,hvk->xvk", Wh, self.codes_ste())            # the selected reading's code [X, V, 10]

        def slot(f: int, j: int) -> Tensor:                                   # field f's slot j: a position [X, Lp]
            return steps[f][0] * shift(anchors[f], j + 1) + steps[f][1] * shift(anchors[f], -(j + 1))

        def field(f: int) -> Tensor:                                          # field f's digits [X, n, 10], LSB first
            live, out = torch.ones(X, device=e.device), []
            zero = F.one_hot(torch.zeros(X, dtype=torch.long, device=e.device), 10).float()
            for j in range(SLOTS[f]):
                tok = torch.einsum("xp,xpv->xv", slot(f, j), E1)
                live = live * (tok @ isdv)
                v = torch.einsum("xv,xvk->xk", tok[:, :V], code)
                out.append(live[:, None] * v + (1 - live)[:, None] * zero)
            return torch.stack(out, 1)

        M = zpad(field(0), 0, P - W)
        TDC, PROD, G = self.ste_tables()
        s = run_ste(field(1), field(2), M, TDC, PROD, G, ctl.used, ctl.used_count, ctl.pused, ctl.used_gate)
        wcode = hot(self.wc.float() + F.logsigmoid(self.dg.float())[None])  # the writer's code [10, V]
        out = torch.zeros(X, Lp, V, device=e.device)
        for j in range(W):
            out = out + slot(3, j)[:, :, None] * (s[:, j] @ wcode)[:, None, :]
        return 20.0 * out[:, W + 1:W + 1 + L] - 20.0

    # ------------------------------------------------------------------ lessons
    def ladder(self, Xn: Tensor, tabs, train: bool):
        """The product cell's lessons on the numbers Xn [n, W] (the rows' x and N): X * 0 = 0, and the soft row
        X * (b + 1) against the machine's own hard X * b + X where X's rungs 0..b all check out (the verified frontier).
        The adder of the targets, the checks and the soft rows is the learned one (the snapped table). Returns the loss
        (when `train`), the share of the hard rung checks that pass and which rungs are verified [n, 10]."""
        n = Xn.shape[0]
        with torch.no_grad():
            _, tgt, ok = ladder_hard(Xn, tabs, self.ctl.visits)
            verified = ok.long().cumprod(1).bool()
            w = verified[:, :9].float()
            if train:
                trained = torch.cat([torch.ones_like(w[:, :1]), w], 1).bool()   # the rungs that get a lesson [n, 10]
                cells = (Xn[:, None, :] * 10 + torch.arange(10, device=Xn.device)[None, :, None]).expand(n, 10, W)
                tally(self.ctl.pvisits, cells[trained], 100)
        lad_ok = float(ok.float().mean())
        if not train:
            return None, lad_ok, verified
        PLO, PHI, TH = self.p_lo.float().softmax(-1), self.p_hi.float().softmax(-1), self.hard_table()
        loss = -row_soft(Xn, 0, PLO, PHI, TH)[..., 0].clamp_min(1e-12).log().mean()
        for b in range(9):
            nll = -row_soft(Xn, b + 1, PLO, PHI, TH).clamp_min(1e-12).log().gather(-1, tgt[:, b, :, None]).squeeze(-1)
            loss = loss + (nll.mean(-1) * w[:, b]).sum() / n
        return loss / 10, lad_ok, verified

    def lessons(self, Nd, Xd, Cd, small, width, M, tabs, e, end, k, code) -> dict:
        """The label-free lessons (the label-dependent ones run in token_training_loss on what this returns): see the
        module docstring and COMBINED.md."""
        ctl = self.ctl
        ctl.step += 1
        dev = Xd.device
        Td, Tb, Plo, Phi, Gc = tabs
        if ctl.visits is None:
            ctl.visits = torch.zeros(200, dtype=torch.long, device=dev)
            ctl.pvisits = torch.zeros(100, dtype=torch.long, device=dev)
            ctl.gvisits = torch.zeros(2, dtype=torch.long, device=dev)
        common = {"ctl": ctl, "model": self, "e": e, "end": end, "k": k, "code": code, "Xd": Xd, "M": M,
                  "width": width, "tabs": tabs, "real": None, "exact": 0.0, "table_exact": 0.0, "cmp": 0.0}
        if ctl.phase == "frozen":
            return {"loss": self.zero(), **common}
        TDC, GS = self.soft_table(), self.gate_soft()
        X, NL = Xd.shape[0], W * (S - 1)
        hot = lambda v: F.one_hot(v, 10).float()
        log = lambda p: p.clamp_min(1e-12).log()
        digits_nll = lambda Y, target: -log(Y).gather(-1, target[..., None]).squeeze(-1).sum(-1)
        with torch.no_grad():
            V, sq_add = square_hard(Xd, tabs, check=True)                    # the machine's own square of x
            D = link_inputs(V, M, Td, Tb, Gc)
            D, Mr = D.reshape(X * NL, P), M.repeat_interleave(NL, 0)
            Z = torch.zeros_like(M)
        facts = ctl.phase in ("B", "C") or (FACTS_FROM == "reading" and ctl.io >= 4)   # x < N, N > 0 may be used
        warm = ctl.step <= ADDER_WARM or (WARM_TO_FACTS and not facts)
        gate_on = ctl.phase == "C"
        order = TABLE_TEACH == "order"
        if order:                                                            # no designed adder at any step; the
            warm = False                                                     # pieces' table lessons wait (below)
        digits_on = order and ctl.gated_on is not None
        pieces = not order or (digits_on and ctl.step - ctl.gated_on >= LATE_AFTER)
        xd0 = F.pad(Xd, (0, P - W))
        loss = self.zero()
        if order and facts:
            # x < N on every top part: x[j:] - N[j:] borrows iff the digit strings differ, N[j:] - x[j:] never does
            for j in range(P):
                differ = (xd0[:, j:] != M[:, j:]).any(1)
                p1 = link_soft(hot(xd0[:, j:]), M[:, j:], TDC)[1][:, -1, 1]
                p2 = link_soft(hot(M[:, j:]), xd0[:, j:], TDC)[1][:, -1, 1]
                loss = loss - (torch.where(differ, log(p1), log(1 - p1)).mean() + log(1 - p2).mean()) / P
            with torch.no_grad():
                okf = (sub_hard(xd0, M, Td, Tb)[1] == 1) & (sub_hard(M, xd0, Td, Tb)[1] == 0)
                ctl.facts_streak = ctl.facts_streak + 1 if bool(okf.all()) else 0
        if digits_on:
            # the order laws and bijectivity on the entries the batch's links read
            with torch.no_grad():
                used = torch.bincount(sub_hard(D, Mr, Td, Tb)[2].flatten(), minlength=200) > 0
            pa = self.a.float().softmax(-1)
            pf = pa.reshape(200, 10)
            for lo, hi in order_pairs(Tb, used):
                loss = loss - log((pf[lo] * (1 - pf[hi].cumsum(-1))).sum(-1)).mean()
            loss = loss - log(10 * pa.mean(0)).mean() - log(10 * pa.mean(1)).mean()
        if order and facts and ctl.gated_on is None and ctl.facts_streak >= GATE_WIN:
            ctl.gated_on = ctl.step                                          # the borrows referee from the next step
            ctl.history.append((ctl.step, "order_digits"))
            print(f"LR order_digits step={ctl.step}", flush=True)
        # every link checks itself, (D - N) + N = D, complete
        Es, be = link_soft(hot(D), Mr, TDC)
        be = be[:, -1]
        if not warm:
            Cs, bc = link_soft(hot(Z), M, TDC)                               # 0 - N, once per row
            bc = bc[:, -1]
        if warm:                                                             # the designed adder
            back, carry = add_soft(Es, Mr)
            agree = carry * be[:, 1] + (1 - carry) * be[:, 0]
        else:                                                                # (D - N) - (0 - N), the learned adder
            back, bf = sub_soft(Es, Cs.repeat_interleave(NL, 0), TDC)
            pe, pf, pc = be[:, 1], bf[:, -1, 1], bc[:, 1].repeat_interleave(NL, 0)
            agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc   # be = bc - bf
        weight = torch.ones(P, device=dev)
        weight[0] = ID_PLACE0                                               # the lowest place tells the swapped code
        nll = (-log(back).gather(-1, D[..., None]).squeeze(-1) * weight).sum(-1) - ID_BORROW * log(agree)
        pieces_w = 1.0 if pieces else 0.0                                    # "order": 0 until LATE_AFTER
        loss = loss + pieces_w * nll.mean() / (weight.sum() + ID_BORROW)
        # the anchors (ADDER.md): on any numbers, then (phase B on) the facts about x and N
        Rg = D.view(X, W, S - 1, P)[:, :, 0].reshape(X * W, P)               # the registers Rp (first links' inputs)
        MW, xd, zero = M.repeat_interleave(W, 0), F.pad(Xd, (0, P - W)), torch.zeros_like(Rg)
        checks = []                                                         # hard versions: (A, B, target, borrow)
        for wgt, A, B, target in ((ANCHOR_ZERO, Rg, zero, Rg), (ANCHOR_SELF, Rg, Rg, zero)):
            if wgt > 0:                                                     # R - 0 = R, R - R = 0, no final borrow
                Y, b = link_soft(hot(A), B, TDC)
                loss = loss + pieces_w * wgt * (digits_nll(Y, target) - log(b[:, -1, 0])).mean() / (P + 1)
                checks.append((A, B, target, 0))
        if not warm and ANCHOR_NEG2 > 0:                                    # 0 + N = N: 0 - (0 - N) = N, complete
            Y2, b2 = sub_soft(hot(Z), Cs, TDC)
            agree2 = bc[:, 1] * b2[:, -1, 1] + bc[:, 0] * b2[:, -1, 0]
            loss = loss + pieces_w * ANCHOR_NEG2 * (digits_nll(Y2, M) - log(agree2)).mean() / (P + 1)
        if not warm and ANCHOR_COMM > 0:                                    # R + N = N + R: R - (0 - N) = N - (0 - R)
            A1, b1 = sub_soft(hot(Rg), Cs.repeat_interleave(W, 0), TDC)
            CR, bcr = link_soft(hot(torch.zeros_like(Rg)), Rg, TDC)
            A2, b2 = sub_soft(hot(MW), CR, TDC)
            c1 = bc[:, 1].repeat_interleave(W, 0) * b1[:, -1, 0]             # P(carry = 1)
            c2 = bcr[:, -1, 1] * b2[:, -1, 0]
            agree = log((A1 * A2).sum(-1)).sum(-1) + log(c1 * c2 + (1 - c1) * (1 - c2))
            loss = loss - pieces_w * ANCHOR_COMM * agree.mean() / (P + 1)
        if facts and ANCHOR_LESS > 0:                                       # x < N: x - N borrows
            loss = loss - pieces_w * ANCHOR_LESS * log(link_soft(hot(xd), M, TDC)[1][:, -1, 1]).mean()
            checks.append((xd, M, None, 1))
        if facts and ANCHOR_MORE > 0:                                       # x < N: N - x does not borrow
            loss = loss - pieces_w * ANCHOR_MORE * log(link_soft(hot(M), xd, TDC)[1][:, -1, 0]).mean()
            checks.append((M, xd, None, 0))
        if facts and ANCHOR_WRAP > 0:                                       # (x - N) - N does not borrow
            Ew = link_soft(hot(xd), M, TDC)[0]
            loss = loss - pieces_w * ANCHOR_WRAP * log(link_soft(Ew, M, TDC)[1][:, -1, 0]).mean()
        if facts and not warm and ANCHOR_NEG > 0:                           # N > 0: 0 - N borrows
            loss = loss - pieces_w * ANCHOR_NEG * log(bc[:, 1]).mean()
        # the product cell (SQUARE.md), once the table can be trusted
        lad_loss, lad_ok, verified = self.ladder(torch.cat([Xd, Nd]), tabs, ctl.ladder)
        if ctl.ladder:
            loss = loss + LADDER_W * lad_loss
        # the gate's facts (CONTROL.md), on a detached table
        TF = TDC if FACT_TABLE else TDC.detach()
        if gate_on and SELF_LESSON:
            loss = loss + ce(chain_soft(hot(xd), M, TF, GS), xd, width)
        if gate_on and NMOD_LESSON:
            loss = loss + ce(chain_soft(hot(M), M, TF, GS), Z, width)
        with torch.no_grad():                                               # the same checks on the hard machine
            Eh, beh, idx = sub_hard(D, Mr, Td, Tb)
            tally(ctl.visits, idx, 200)
            if warm:
                backh, carryh = add_hard(Eh, Mr)
                ok = (backh == D).all(-1) & (carryh == beh)
            else:
                Ch, bch, idc = sub_hard(Z, M, Td, Tb)
                backh, bfh, idf = sub_hard(Eh, Ch.repeat_interleave(NL, 0), Td, Tb)
                tally(ctl.visits, idc, 200)
                tally(ctl.visits, idf, 200)
                ok = (backh == D).all(-1) & (beh == bch.repeat_interleave(NL, 0) - bfh)
            table = float(ok.float().mean())
            if not warm and ANCHOR_NEG2 > 0:
                Y2h, b2h, id2 = sub_hard(Z, Ch, Td, Tb)
                tally(ctl.visits, id2, 200)
                table = min(table, float(((Y2h == M).all(-1) & (b2h == bch)).float().mean()))
            if not warm and ANCHOR_COMM > 0:
                A1h, c1h = add_learned(Rg, MW, Td, Tb, ctl.visits)
                A2h, c2h = add_learned(MW, Rg, Td, Tb, ctl.visits)
                table = min(table, float(((A1h == A2h).all(-1) & (c1h == c2h)).float().mean()))
            for A, B, target, borrow in checks:
                Yh, bh, ida = sub_hard(A, B, Td, Tb)
                tally(ctl.visits, ida, 200)
                okc = bh == borrow
                if target is not None:
                    okc = okc & (Yh == target).all(-1)
                table = min(table, float(okc.float().mean()))
            if facts and ANCHOR_WRAP > 0:
                Ewh, _, idw = sub_hard(xd, M, Td, Tb)
                table = min(table, float((sub_hard(Ewh, M, Td, Tb)[1] == 0).float().mean()))
            if facts and not warm and ANCHOR_NEG > 0:
                table = min(table, float((bch == 1).float().mean()))
            cmp = float((sub_hard(xd, M, Td, Tb)[1] == 1).float().mean())
            sq = verified[:X].gather(1, Xd.max(1).values[:, None])[:, 0] & sq_add   # the squares of x check out
            exact = min(table, lad_ok, float(sq.float().mean()))
            if gate_on:                                                     # the hard machine on the gate's facts
                selfx = step_hard(xd, M, Td, Tb, Gc, ctl.visits, ctl.gvisits)[0]
                selfn = step_hard(M, M, Td, Tb, Gc, ctl.visits, ctl.gvisits)[0]
                exact = min(exact, cmp, float((selfx == xd).all(1).float().mean()),
                            float((selfn == 0).all(1).float().mean()))
        if gate_on and REAL_MAX_T > 0 and lad_ok == 1.0:                    # the rows' answers along the own path
            sel = torch.nonzero(small).flatten()
            if sel.numel():
                with torch.no_grad():                                       # countdown, squarings and ladder check out
                    _, sp, n, held = run_hard(Xd[sel], Cd[sel], M[sel], tabs, cap=REAL_MAX_T + 1)
                    good = torch.nonzero((n > 0) & held & sq[sel] & square_ok(sp, tabs)).flatten()
                    sel, sp = sel[good], sp[good]
                    Vr = square_hard(sp, tabs)
                if sel.numel():
                    Rr, Rrh = divide_path(Vr, M[sel], TDC, GS, Td, Tb, Gc, ctl.visits, ctl.gvisits)
                    common["real"] = {"sel": sel, "R": Rr, "Rh": Rrh, "width": width[sel]}
        if MODE == "learned":
            loss = loss + self.io_lessons(e, end, k, code, V[:, :W])
        common.update(exact=exact, table_exact=table, cmp=cmp, lad_ok=lad_ok)
        return {"loss": loss, **common}

    def io_lessons(self, e: Tensor, end: Tensor, k: Tensor, code: Tensor, vals: Tensor) -> Tensor:
        """The I/O's label-free lessons (IO.md): the rank codes (Bradley-Terry on x < N, every hypothesis, from stage
        2), the reading (the normalized mixture over complete readings, from stage 3), the tiling (from stage 4) and the
        writer's code round trip on the machine's own numbers `vals` [X, W] (from stage 3)."""
        ctl = self.ctl
        X, Lp = e.shape
        dev = e.device
        loss = self.u.sum() * 0.0
        ok, si, ti, length = self.runs(e, end)
        ctl.rows3 = float(ok.float().mean())
        if ctl.io < 2 or not ok.any():
            return loss
        okf = ok.float()
        j = torch.arange(W, device=dev)
        # each run read in each direction: tokens [X, run, dir, P] (-1 after the run's end)
        tokp = e.gather(1, (si[..., None] + j).clamp(max=Lp - 1).reshape(X, -1)).view(X, 3, W)
        tokm = e.gather(1, (ti[..., None] - j).clamp(min=0).reshape(X, -1)).view(X, 3, W)
        liv = j[None, None] < length[..., None]
        R = F.pad(torch.stack([torch.where(liv, tokp, -1), torch.where(liv, tokm, -1)], 2), (0, P - W), value=-1)
        hyp = torch.tensor(HYP, device=dev)
        Tn, Tx = R[:, hyp[:, 0], hyp[:, 1]], R[:, hyp[:, 2], hyp[:, 3]]     # [X, H, P]: N's and x's readings
        place = torch.where(Tx != Tn, torch.arange(P, device=dev), -1).max(-1).values   # the decisive place
        ta = Tx.gather(2, place.clamp(min=0)[..., None])[..., 0]
        tb = Tn.gather(2, place.clamp(min=0)[..., None])[..., 0]
        u = self.u.float()
        isd = (self.dg > 0).detach()
        umin = u.masked_fill(~isd[None], 1e9).min(-1).values.detach()     # the score of each code's 0
        ua = u[None].expand(X, -1, -1).gather(2, ta.clamp(min=0)[..., None])[..., 0]
        ub = u[None].expand(X, -1, -1).gather(2, tb.clamp(min=0)[..., None])[..., 0]
        both = (ta >= 0) & (tb >= 0) & (place >= 0)                       # two digits differ there
        padx = (ta < 0) & (tb >= 0) & (place >= 0)                        # x' ended first: N's digit must not be 0
        bt = -(F.logsigmoid((ub - ua) / BT_TAU) * both * okf[:, None]).sum() / (both * okf[:, None]).sum().clamp_min(1)
        loss = loss + bt
        hp = torch.where(both, ub > ua, torch.where(padx, ub > umin[None], torch.zeros_like(both)))  # exact verdicts
        rest = 3 - hyp[:, 0] - hyp[:, 2]                                   # the run left for T's two slots
        tfits = length.gather(1, rest[None].expand(X, -1)) <= SLOTS[2]
        passes = ((hp & tfits).float() * okf[:, None]).sum(0) / okf.sum()  # [H]
        ctl.pick = ctl.pick + 1 if float(passes.max()) >= PICK_PASS else 0
        ctl.best = int(passes.argmax())
        if ctl.io >= 3:
            A = torch.stack([self.anchor(f, e, end) for f in range(3)], 1)                 # [X, field, Lp]
            sl = torch.stack([self.strd[f].float().softmax(-1) for f in range(3)])         # [field, dir]
            fits = length[:, None, :] <= torch.tensor(SLOTS[:3], device=dev)[None, :, None]
            left = A.gather(2, (si - 1)[:, None].expand(X, 3, 3)) * sl[None, :, None, 0]
            right = A.gather(2, (ti + 1).clamp(max=Lp - 1)[:, None].expand(X, 3, 3)) * sl[None, :, None, 1]
            rho = torch.stack([left, right], -1) * fits[..., None]                         # [X, field, run, dir]
            Lk = hp.float() + EPS
            lik, norm = torch.zeros(X, device=dev), torch.zeros(X, device=dev)
            post = torch.zeros(X, len(HYP), device=dev)
            for sN, sx, sT in PERMS:
                for dn in range(2):
                    for dx in range(2):
                        w = rho[:, 2, sT].sum(-1) * rho[:, 0, sN, dn] * rho[:, 1, sx, dx]
                        h = HIDX[(sN, dn, sx, dx)]
                        lik = lik + w * Lk[:, h]
                        norm = norm + w
                        post[:, h] = post[:, h] + w.detach()
            loss = loss - ((lik.clamp_min(1e-30).log() - norm.clamp_min(1e-30).log()) * okf).sum() / okf.sum()
            ctl.chosen = float((post / norm.detach()[:, None].clamp_min(1e-30)).max(1).values[ok].mean())
            if ctl.io >= 4:                                                # whole runs, distinct fields
                loss = loss - (norm.clamp_min(EPS).log() * okf).sum() / okf.sum()
            # the writer's code: write the machine's own numbers, read them back
            with torch.no_grad():
                inv = torch.zeros(X, 10, dtype=torch.long, device=dev)
                for t in torch.nonzero(isd).flatten().tolist():
                    inv.scatter_(1, code[:, t:t + 1], t)                    # the token whose code is v
            rv = self.wsm()[vals, inv.gather(1, vals)]
            loss = loss - rv.clamp_min(EPS).log().mean()
        return loss

    def io_label_lessons(self, aux: dict, labels: Tensor, valid: Tensor, tpos: Tensor):
        """The I/O's label lessons (IO.md): the answer round trip (the writer's pointer, digit-ness), the token-side code
        round trip (the writer's code) and, once the machine is exact, T's direction by speculation."""
        ctl = self.ctl
        e, end, code = aux["e"], aux["end"], aux["code"]
        X, Lp = e.shape
        dev = e.device
        labe = self.label_seq(e, labels, valid, tpos)
        wsm = self.wsm()
        dl = torch.where(labe < self.V, torch.sigmoid(self.dg.float()[labe.clamp(max=self.V - 1)]),
                         torch.zeros((), device=dev))
        a = self.anchor(3, e, end)
        q = torch.arange(Lp, device=dev)[None, :, None]
        lps = []
        for kk, s in enumerate((1, -1)):
            pos = (q + (torch.arange(W, device=dev)[None, None] + 1) * s).clamp(0, Lp - 1).expand(X, Lp, W)
            tok = labe.gather(1, pos.reshape(X, -1)).view(X, Lp, W)
            live = dl.gather(1, pos.reshape(X, -1)).view(X, Lp, W).cumprod(-1)
            tgt = tok < self.V
            tc = tok.clamp(max=self.V - 1)
            back = live * wsm.detach()[code.gather(1, tc.reshape(X, -1)).view(X, Lp, W), tc]   # read, write back
            lp = ((tgt * back.clamp_min(EPS).log()).sum(-1)
                  + (valid.sum(1)[:, None] - tgt.sum(-1)) * float(torch.log(torch.tensor(EPS))))
            lps.append(lp + a.clamp_min(1e-30).log() + self.strd[3].float().log_softmax(-1)[kk])
        loss = -torch.logsumexp(torch.stack(lps, -1).reshape(X, -1), -1).mean()
        lv = labe < self.V
        rows_l, tl = torch.nonzero(lv)[:, 0], labe[lv]
        loss = loss - wsm[code[rows_l, tl], tl].clamp_min(EPS).log().mean()   # read an answer token, write it back
        with torch.no_grad():                                               # the hard round trips
            yd = self.read_answer(labe, e, end, code)
            pos, _, _ = self.hard_pos(3, e, end)
            back = self.wsm().argmax(-1)[yd]
            o = torch.full_like(e, -1).scatter_(1, pos, back)
            ans = (torch.where(valid, o.gather(1, torch.where(valid, tpos + W + 1, 0)), labels) == labels).all(1)
            isd = self.dg > 0
            inv_ok = bool((self.wsm().argmax(-1)[code[:, isd]] == torch.nonzero(isd).flatten()[None]).all())
            ctl.io_exact = bool(ans.all()) and inv_ok and int(isd.sum()) == 10 and bool((aux["k"] >= 0).all())
        if ctl.io == 4 and ctl.phase == "C" and ctl.streak >= T_AFTER:
            loss = loss + self.t_lesson(aux, yd)
        return loss, yd

    def t_lesson(self, aux: dict, yd: Tensor) -> Tensor:
        """T's direction by speculation (IO.md): on rows whose two readings of T differ, the exact machine runs its
        countdown from both, and the answers keep the reading that reproduces them (a normalized mixture over T's two
        whole-run programs)."""
        ctl = self.ctl
        e, end, code, Xd, M = aux["e"], aux["end"], aux["code"], aux["Xd"], aux["M"]
        X, Lp = e.shape
        dev = e.device
        ok, si, ti, length = self.runs(e, end)
        rn, _, _ = self.which_run(0, e, end, si, ti, ok, length)
        rx, _, _ = self.which_run(1, e, end, si, ti, ok, length)
        rt = (3 - rn - rx).clamp(0, 2)                                      # T's run
        a0, b0 = si.gather(1, rt[:, None])[:, 0], ti.gather(1, rt[:, None])[:, 0]
        j = torch.arange(K, device=dev)
        tp = e.gather(1, (a0[:, None] + j).clamp(max=Lp - 1))
        tm = e.gather(1, (b0[:, None] - j).clamp(min=0))
        lp = j[None] < (b0 - a0 + 1)[:, None]
        Cp = torch.where(lp, code.gather(1, tp.clamp(max=self.V - 1)), 0)   # T read from its run's left end (step +1)
        Cm = torch.where(lp, code.gather(1, tm.clamp(max=self.V - 1)), 0)   # ... from its right end (step -1)
        rows = torch.nonzero(ok & (Cp != Cm).any(1)).flatten()[:T_ROWS]
        if rows.numel() == 0:
            return self.strd.sum() * 0.0
        with torch.no_grad():
            n = rows.numel()
            s2 = run_hard(Xd[rows].repeat(2, 1), torch.cat([Cp[rows], Cm[rows]]), M[rows].repeat(2, 1), aux["tabs"])[0]
            Lpl, Lmi = (s2[:n] == yd[rows]).all(1).float(), (s2[n:] == yd[rows]).all(1).float()
        A = self.anchor(2, e, end)[rows]
        sl = self.strd[2].float().softmax(-1)
        wp = A.gather(1, (a0[rows] - 1)[:, None])[:, 0] * sl[0]                # read from the run's left end, +1
        wm = A.gather(1, (b0[rows] + 1)[:, None])[:, 0] * sl[1]                # read from its right end, -1
        hit = (Lpl + Lmi) > 0
        lt = -(((wp * Lpl + wm * Lmi).clamp_min(1e-30).log() - (wp + wm).clamp_min(1e-30).log()) * hit).sum()
        lt = lt / hit.sum().clamp_min(1)
        with torch.no_grad():
            _, _, hs = self.hard_pos(2, e, end)
            mine = Lpl if hs > 0 else Lmi                                      # the snapped reading's verdict
            share = float(((wp if hs > 0 else wm) / (wp + wm).clamp_min(1e-30)).mean())
            ctl.t_log = f"rows={n} hits={int(hit.sum())} snapped_right={float(mine.mean()):.3f} share={share:.4f}"
            if bool(mine.all()) and share >= T_DONE:
                ctl.io = 5
                ctl.history.append((ctl.step, "T_resolved", n))
                print(f"LR io_stage=5 T_resolved step={ctl.step} {ctl.t_log}", flush=True)
        return lt

    def control(self, aux: dict, loss: float) -> None:
        """The schedule, once per step (after the label lessons): the I/O stages, the machine's phases, the ladder, the
        freeze."""
        ctl = self.ctl
        exact, table, cmp = aux["exact"], aux["table_exact"], aux["cmp"]
        before = ctl.io
        if ctl.io == 1 and ctl.rows3 >= 0.99:
            ctl.io = 2                                                       # three digit runs: rank the codes
            ctl.history.append((ctl.step, "io2_digits"))
        elif ctl.io == 2 and ctl.pick >= PICK_WIN:
            ctl.io = 3                                                       # a hypothesis passes: choose the reading
            ctl.history.append((ctl.step, "io3_ranked", HYP[ctl.best]))
        elif ctl.io == 3 and ctl.chosen >= CHOSEN:
            ctl.io = 4                                                       # one reading carries the mixture: tiling
            ctl.history.append((ctl.step, "io4_reading", round(ctl.chosen, 4)))
        if ctl.io != before:
            print(f"LR io_stage={ctl.io} step={ctl.step} history={ctl.history[-1]}", flush=True)
        if not ctl.ladder and ctl.phase in ("B", "C"):
            ctl.table_streak = ctl.table_streak + 1 if table == 1.0 else 0
            ctl.ladder_wait += 1
            if ctl.table_streak >= LADDER_WIN or ctl.ladder_wait >= LADDER_STEPS:
                # the table passes every check: the learned adder can be trusted with the product cell's targets
                ctl.ladder = True
                ctl.reset_params = [self.p_lo, self.p_hi]
                ctl.history.append((ctl.step, "ladder", ctl.table_streak))
                print(f"LR ladder_joins step={ctl.step} table_streak={ctl.table_streak}", flush=True)
        ctl.win.append(table if ctl.phase == "A" else cmp if ctl.phase == "B" else exact)
        ctl.in_phase += 1
        mean = sum(ctl.win) / len(ctl.win)
        if ctl.step % LOG_EVERY == 0 or ctl.step == 1:
            gate = "".join(str(int(g)) for g in self.g.argmax(-1).tolist())
            print(f"LR step={ctl.step} phase={ctl.phase} io={ctl.io} ladder={int(ctl.ladder)} in_phase={ctl.in_phase} "
                  f"loss={loss:.4f} hard_exact={exact:.4f} table_exact={table:.4f} cmp={cmp:.4f} "
                  f"ladder_ok={aux.get('lad_ok', 0.0):.4f} win={mean:.4f} gate={gate} io_exact={ctl.io_exact} "
                  f"chosen={ctl.chosen:.4f} rows_visited={int((ctl.visits > 0).sum())}/200 "
                  f"cells_visited={int((ctl.pvisits > 0).sum())}/100 order_digits={ctl.gated_on} {ctl.t_log}",
                  flush=True)
        if ctl.phase == "A":
            if ctl.io >= 4 and ((len(ctl.win) == STAGE_WIN and mean >= ID_SWITCH) or ctl.in_phase >= ID_STEPS):
                # the checks pass on most links and the reading is chosen: the facts about x and N join, with a fresh
                # optimizer (Adam's memory of phase A's gradients would slow the repairs)
                ctl.history.append((ctl.step, "switch", ctl.in_phase, round(mean, 4)))
                print(f"LR switch_to_facts step={ctl.step} win={mean:.4f}", flush=True)
                ctl.phase, ctl.reset, ctl.in_phase = "B", True, 0
                ctl.win.clear()
        elif ctl.phase == "B":
            if (len(ctl.win) == STAGE_WIN and mean >= CMP_SWITCH) or ctl.in_phase >= CMP_STEPS:
                # the borrows say "below N" where they should: the gate can learn what to do with them
                ctl.history.append((ctl.step, "gate", ctl.in_phase, round(mean, 4)))
                print(f"LR switch_to_gate step={ctl.step} win={mean:.4f}", flush=True)
                ctl.phase, ctl.in_phase = "C", 0
                ctl.win.clear()
        elif ctl.phase == "C":
            ctl.streak = ctl.streak + 1 if exact == 1.0 and ctl.io_exact else 0
            if ctl.streak >= FREEZE_WIN and ctl.io == 5:
                ctl.history.append((ctl.step, "frozen", ctl.in_phase, 1.0))
                print(f"LR frozen step={ctl.step} history={ctl.history}", flush=True)
                ctl.phase = "frozen"

    # ------------------------------------------------------------------ evaluation reports
    def report_tables(self, e: Tensor, end: Tensor) -> None:
        """At the first evaluation batch: the snapped tables, the I/O, and how often training used each row."""
        ctl = self.ctl
        Td, Tb, Plo, Phi, Gc = self.tables()
        print("TABLE digit=" + "".join(str(int(d)) for d in Td.tolist()), flush=True)
        print("TABLE borrow=" + "".join(str(int(d)) for d in Tb.tolist()), flush=True)
        print("TABLE prod_lo=" + "".join(str(int(d)) for d in Plo.tolist()), flush=True)
        print("TABLE prod_hi=" + "".join(str(int(d)) for d in Phi.tolist()), flush=True)
        print("TABLE gate=" + "".join(str(int(d)) for d in Gc.tolist())
              + " p_commit=" + ",".join(f"{float(p):.4f}" for p in self.gate_soft().tolist()), flush=True)
        if ctl.visits is not None:
            for name, v in (("train_visits", ctl.visits), ("prod_train_visits", ctl.pvisits)):
                print(f"TABLE {name} min={int(v.min())} rows_never={int((v == 0).sum())} "
                      f"counts={','.join(str(int(c)) for c in v.tolist())}", flush=True)
            print(f"TABLE gate_train_visits={','.join(str(int(c)) for c in ctl.gvisits.tolist())}", flush=True)
        print(f"TABLE schedule phase={ctl.phase} io={ctl.io} step={ctl.step} history={ctl.history}", flush=True)
        names = ["N", "x", "T", "writer"]
        for f in range(4):
            a = int(self.anc[f].argmax())
            st = "+1" if int(self.strd[f].argmax()) == 0 else "-1"
            print(f"IO field={names[f]} anchor={'END' if a == self.END else a} step={st}", flush=True)
        k = self.selected(e, end)
        kk = int(torch.mode(k).values)
        code = self.ranks(torch.tensor([max(kk, 0)], device=e.device))[0]
        isd = self.dg > 0
        print(f"IO digit_tokens={torch.nonzero(isd).flatten().tolist()} reading={HYP[kk] if kk >= 0 else None} "
              f"rows_with_that_reading={float((k == kk).float().mean()):.4f}", flush=True)
        print("IO code=" + ",".join(f"{t}:{int(code[t])}" for t in torch.nonzero(isd).flatten().tolist()), flush=True)
        print("IO writer=" + ",".join(f"{v}:{int(t)}" for v, t in enumerate(self.wsm().argmax(-1).tolist())), flush=True)
        print(f"IO path={PATH}", flush=True)

    def audit(self) -> None:
        """Rows used at evaluation so far (subtractor: all uses and the countdown's; product cells; gate) and how many
        of them training never used."""
        ctl = self.ctl
        used = (ctl.used + ctl.used_count) > 0
        count, cells, gate = ctl.used_count > 0, ctl.pused > 0, ctl.used_gate > 0
        key = (int(used.sum()), int(count.sum()), int(cells.sum()), int(gate.sum()))
        if key == ctl.reported:
            return
        ctl.reported = key
        if ctl.visits is None:
            print(f"AUDIT eval_rows_used={key[0]}/200 eval_product_cells_used={key[2]}/100 (no training ran)", flush=True)
            return
        never = lambda u, v: int((u & (v == 0)).sum())
        rows = [(i // 20, (i // 2) % 10, i % 2) for i in torch.nonzero(count).flatten().tolist()]
        print(f"AUDIT eval_rows_used={key[0]}/200 used_but_never_trained={never(used, ctl.visits)} "
              f"eval_product_cells_used={key[2]}/100 product_used_but_never_trained={never(cells, ctl.pvisits)}",
              flush=True)
        print(f"AUDIT countdown_rows_used={key[1]} used_but_never_trained={never(count, ctl.visits)} "
              f"min_train_visits={int(ctl.visits[count].min()) if key[1] else -1} rows={rows}", flush=True)
        print(f"AUDIT gate_rows_used={key[3]}/2 used_but_never_trained={never(gate, ctl.gvisits)} "
              f"train_visits={ctl.gvisits.tolist()}", flush=True)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    aux = batch.auxiliary
    with torch.autocast(batch.labels.device.type, enabled=False):
        loss = aux["loss"]
        ctl, model = aux["ctl"], aux["model"]
        if ctl.phase == "frozen":
            return loss
        labels, valid = batch.labels, batch.valid_mask
        tpos = batch.target_positions
        if tpos is None:                                                    # right-aligned answers (the evaluator's)
            cnt = valid.long().sum(1)
            kk = torch.arange(labels.shape[1], device=labels.device)[None]
            tpos = torch.where(kk < cnt[:, None], aux["end"][:, None] - cnt[:, None] + kk, -1)
        if MODE == "learned":
            lio, yd = model.io_label_lessons(aux, labels, valid, tpos)
            loss = loss + lio
        else:
            ctl.io_exact = True
            yd = model.read_answer(model.label_seq(aux["e"], labels, valid, tpos), aux["e"], aux["end"], aux["code"])
        real = aux["real"]
        if real is not None:
            sel = real["sel"]
            ydr = F.pad(yd[sel], (0, P - W))                                 # the answers as the learned reader reads them
            lesson = ce(real["R"], ydr, real["width"])
            if ctl.step % LOG_EVERY == 0:
                ex = float((real["Rh"] == ydr).all(1).float().mean())
                print(f"LR step={ctl.step} real_rows={sel.numel()} real_hard_exact={ex:.4f}", flush=True)
            loss = loss + lesson
        model.control(aux, float(loss.detach()))
        return loss


class Schedule:
    """Scheduler: a fresh Adam state at the switch to phase B (and for the product cell when the ladder joins);
    learning rate 0 once the model is frozen."""

    def __init__(self, optimizer: torch.optim.Optimizer, ctl: Control) -> None:
        self.optimizer, self.ctl = optimizer, ctl

    def step(self) -> None:
        if self.ctl.reset:
            self.optimizer.state.clear()
            self.ctl.reset = False
        for p in self.ctl.reset_params:
            self.optimizer.state.pop(p, None)
        self.ctl.reset_params = []
        if self.ctl.phase == "frozen":
            for group in self.optimizer.param_groups:
                group["lr"] = 0.0


def build_model(spec: ModelSpec) -> LearnedCombined:
    model = LearnedCombined(spec)
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
