"""LearnedWiringSq: the combined H1 machine (submission_combined.py, COMBINED.md) with the long division's wiring
learned (WIRING.md, stage 1: submission_div.py) and the schoolbook squaring's layout learned too (stage 2): which
product digits each column of each row holds, so which digit pairs combine in which column. Every table, the reader and
the writer are learned as in the combined machine. One model, trained from a random start in one run.

Datapath (base 10); the state is digits from the prompt to the answer:
  * read: three fields, N (8 slots), x (8 slots) and T (K = 2 slots), each by a learned pointer (anchor + one shared
    step), with a learned digit-ness per token and a learned code (token -> digit) per reading hypothesis (IO.md);
  * rounds: each round subtracts 1 from T's digits with the subtraction table, and the commit gate reads that
    subtraction's final borrow. A committing round squares the state and reduces it mod N and takes T - 1; a holding
    round keeps both (CONTROL.md). The rounds end when no row commits (at most 10^K);
      - squaring (SQUARE.md): row j of s * s is made from the product cell's digits of the pairs (s_i, s_j); it is two
        numbers of C = 2W + 2 places, a low-digit number and a high-digit number, summed by the learned adder
        A - (0 - B) (ADDER.md), and the W rows are summed one by one the same way. Its layout (stage 2): column c of
        row j's low-digit number holds the low digit of one pair (s_i, s_j), or nothing; the high-digit number
        likewise. LAYOUT = "rule": one placement rule per output kind, shared by every pair (LAY_MENU): the column
        of pair (i, j) is f(i) + g(j) + d, f in {W - 1 - i, i}, g in {0, j, W - 1 - j}, d in {-1, 0, 1, 2}: 24
        rules per kind, 576 layouts, one weight per whole rule (the rule's three parts depend on each other: weighed
        apart they co-adapt, WIRING.md section 3); the truth puts the low digit in column i + j and the high digit
        in column i + j + 1. LAYOUT = "columns": every column of every row its own source switch (2 W C = 288 switches of
        W + 1 options; WIRING.md section 3: pair 7 is never exercised in training, so it does not generalize);
      - reduction: a long division of the 2W digits of s^2 by N in W steps on a register of P places, each step a
        chain of S - 1 links of the subtraction table, each link's result written by the commit gate on the link's
        final borrow (CHAIN.md, CONTROL.md). Its wiring is a set of tied switches, one per kind of connection, shared
        by every step, link and round (DIV_MENU; TINYROUTING.md's grammar on the gated chain):
            vread  the dividend's digits as given | reversed (the end the steps start from; the register starts
                   with the other end's W digits)
            shift  the step's register from the register R and the incoming digit d: down | stay (d replaces place 0)
                   | up (10 R + d) | skip (R)
            read   the remainder: the register's high W places | its low W places
            nread  the subtrahend register: N's read digits reversed | as given
            a      each link's minuend: the step's start | N | the chain register (prev)
            b      each link's subtrahend: prev | the step's start | N
            keep   what a link writes into the chain register: always | its result when the gate commits | never
        The true wiring is (given, up, low, given, prev, N, gated): DIV_TRUE, used only by MODE = "designed" and the
        reports. No option list starts with it, and every lesson treats the options alike;
  * write: the state's slot j at the position the writer's pointer gives slot j, its token by the writer's code.

Learned (MODE = "learned"): the subtraction table (r, m, borrow in) -> (digit, borrow out), 200 rows; the product
cell (a, b) -> (lo, hi), 100 rows; the commit gate borrow -> (hold, commit), 2 rows; the division's 7 switches; the
squaring's layout, a 24-way rule switch per output kind (LAYOUT = "rule"); the reader's digit-ness, codes, anchors
and steps and the writer's code (IO.md). MODE = "designed" starts every one of them
at the truth and never trains (the plumbing check).

The prediction path (PATH = "ste", rule 8): one-hot digits through straight-through tables and switches
(one-hot(argmax) + (softmax - softmax.detach())): the forward pass is exactly the snapped machine, and the final
logits' gradient reaches every table and switch. The state's gradient is discounted by GRAD_DAMP after every link,
addition and round (the value is unchanged). PATH = "index" runs the same machine on integer digits.

What teaches what (WIRING.md section 2):
  * the reader and writer: IO.md's stages, as in the combined machine;
  * the subtraction table (phase A, then B): the row's own numbers, x and the P-place windows of the machine's own
    square of x (which do not depend on the division's wiring), minus N again and again (S - 1 links: the trials of a
    chain, also past zero) check every subtraction, (D - N) + N = D,
    complete, the lowest place weighted ID_PLACE0: with the designed adder until ADDER_WARM steps after the reading
    is chosen, then the learned one. With it R - 0 = R, R - R = 0, and after the warm start 0 + N = N and
    R + N = N + R; from the reading on the facts x - N borrows, N - x does not, (x - N) - N does not, 0 - N borrows.
    R - 0 = R and R - R = 0 ripple with the snapped borrow carried forward straight-through (STE_CIN), so they train
    only the entries they use (ABLATION.md's HARD_CIN; with it on every lesson one seed's table collapsed, WIRING.md);
  * the product cell and the squaring's layout, once the table's checks are exact: the cell's own ladder on single
    digits, a * (b + 1) = a * b + a as two-place numbers (it does not involve the layout: TINYROUTING.md's condition 1),
    and the routed ladder (TINYROUTING.md section 5): for X in the rows' x and N, the soft row_j(X, b + 1) (soft cell,
    soft layout blended on the wire, the learned adder) against the hard row_j(X, b) + X 10^j, along the verified
    frontier (SQUARE.md), LAYOUT_ROWS rows j per number per step;
  * the chain's switches (nread, a, b, keep) and the gate (phase C, once the table and the ladder have been exact for
    WIRE_WIN steps): the division ladder, one division step at a time. From the register x + N * q (q = 0..9, the
    ladder's own row 0 of N * q added to x by the learned adder) one step must give back x. Every combination of the four
    switches and the gate's two rows (216) runs that step on the hard machine, and the verdicts are blended by the
    product of the switches' probabilities (a factorized mixture: a law is a crisp test of a whole wiring, while in a
    blend on the wire it rewards the do-nothing half-and-half wiring, TINYROUTING.md section 6);
  * the division's I/O switches (vread, shift, read), once the argmax chain has passed the division ladder for
    WIRE_WIN steps: x mod N = x and N mod N = 0 through the whole division, and the answers of the rows with
    T <= REAL_MAX_T through the whole machine, each of the 16 combinations on the hard machine, verdicts blended as
    above;
  * T's direction by speculation once the machine is exact (IO.md); freeze after FREEZE_WIN exact steps.
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

MODE = "learned"       # "learned" | "designed" (every table, switch, the gate and the I/O start at the truth, frozen)
PATH = "ste"           # the prediction path: "ste" (one-hot digits through straight-through tables and switches,
                       # inside autograd) | "index" (integer digits indexing the argmax tables)
GRAD_DAMP = 0.1        # "ste": the state's gradient is multiplied by this after every link, addition and round
W = 8                  # digits of the widest modulus (a 24-bit N is below 10^8); one reduction takes W steps
P = W + 1              # register places: 10 R + v_t < 10 N < 10^9
C = 2 * W + 2          # columns of the square and of its rows (s^2 < 10^2W; two more keep any layout's sum whole)
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
ADDER_WARM = 70        # the designed adder in the check until ADDER_WARM steps after the facts about x and N have joined
                       # (the reading is chosen): before the reading the untrained reader may read every N as 0, and
                       # the check then subtracts 0 (WIRING.md section 3.4), then the learned one
STE_CIN = "anchors"    # which table lessons ripple with the snapped borrow straight-through: "all" | "anchors" (R - 0 = R
                       # and R - R = 0 only, ABLATION.md's HARD_CIN) | "none" (WIRING.md section 2.2)
CHECK_REGS = "rows"    # the registers the check subtracts N from: "rows" (x and the dividend's P-place windows: the
                       # row's own numbers, whatever the wiring) | "own" (the argmax division's step registers)
ANCHOR_ZERO = 1.0      # R - 0 = R on the registers
ANCHOR_SELF = 1.0      # R - R = 0 on the registers
ANCHOR_NEG2 = 1.0      # 0 + N = N with the learned adder: 0 - (0 - N) = N (after the warm start)
ANCHOR_COMM = 1.0      # R + N = N + R with the learned adder (after the warm start)
ANCHOR_LESS = 1.0      # x < N: x - N borrows (once the reading is chosen)
ANCHOR_MORE = 1.0      # x < N: N - x does not borrow (once the reading is chosen)
ANCHOR_WRAP = 1.0      # x - N wraps to 10^P + x - N >= N: (x - N) - N does not borrow (once the reading is chosen)
ANCHOR_NEG = 1.0       # N > 0: 0 - N borrows (once the reading is chosen, after the warm start)
# the product cell (SQUARE.md)
LADDER_W = 1.0         # weight of the ladder's loss
LADDER_WIN = 20        # the ladder joins once every check of the table has been exact this many steps in a row ...
LADDER_STEPS = 300     # ... or after this many steps of phase B
CELL_LADDER = 1.0      # weight of the product cell's own ladder a (b + 1) = a b + a on single digits (no layout)
LAYOUT_ROWS = 2        # rows j of the routed ladder each number trains per step (cycling through the W rows)
LAYOUT = "rule"        # the squaring's layout: "rule" (one placement rule per output kind) | "columns" (a source switch
                       # per column of every row)
# the division's wiring (WIRING.md)
WIRE_WIN = 20          # the chain lesson joins once the table and the ladder have been exact this many steps in a row;
                       # the I/O lesson once the argmax chain has passed the division ladder this many steps in a row
CHAIN_ROWS = 64        # rows of the batch whose x + N * q (q = 0..9) the division ladder divides
IO_ROWS = 128          # rows of the batch whose x and N the I/O lesson divides
# the reader and writer (IO.md)
BT_TAU = 1.0           # Bradley-Terry temperature of the rank codes
PICK_PASS = 0.999      # the reading is chosen once a hypothesis passes x < N on this share of the batch ...
PICK_WIN = 20          # ... for this many steps in a row
CHOSEN = 0.99          # then the tiling joins once the mixture puts this posterior on one reading
T_AFTER = 20           # T's direction: speculate once the machine has been exact this many steps in a row ...
T_ROWS = 64            # ... on at most this many rows whose two readings of T differ
T_DONE = 0.999         # ... until the snapped reading reproduces their answers and carries this weight
REAL_MAX_T = 2         # the I/O lesson uses the answers of the batch's rows with T <= REAL_MAX_T (one digit)
FREEZE_WIN = 200       # phase C: freeze after this many consecutive steps with every check exact
BATCH = 512
EVAL_BATCH = 16384
MAX_STEPS = 20000
LOG_EVERY = 250
EPS = 1e-6

PERMS = list(itertools.permutations(range(3)))
HYP = [(rn, dn, rx, dx) for rn in range(3) for rx in range(3) if rn != rx for dn in range(2) for dx in range(2)]
HIDX = {h: i for i, h in enumerate(HYP)}  # (N's run, N's direction, x's run, x's direction); direction 0: step +1
DIV_MENU = {
    "vread": ("given", "reversed"),
    "shift": ("down", "stay", "up", "skip"),
    "read": ("high", "low"),
    "nread": ("reversed", "given"),
    "a": ("start", "N", "prev"),
    "b": ("prev", "start", "N"),
    "keep": ("all", "gated", "none"),
}
DIV_TRUE = {"vread": 0, "shift": 2, "read": 1, "nread": 1, "a": 2, "b": 2, "keep": 1}
CHAIN_SW = ("nread", "a", "b", "keep")                 # taught by the division ladder, with the gate's two rows
IO_SW = ("vread", "shift", "read")                     # taught by x mod N = x, N mod N = 0 and the answers
CHAIN_COMBOS = list(itertools.product(*[range(len(DIV_MENU[n])) for n in CHAIN_SW], range(2), range(2)))
LAY_MENU = {"i": ("W-1-i", "i"), "j": ("0", "j", "W-1-j"), "d": (-1, 0, 1, 2)}   # f(i), g(j), d of a placement rule
LAY_TRUE = {"lo": (1, 1, 1), "hi": (1, 1, 2)}          # column i + j (low digit), i + j + 1 (high digit)
LAY_RULES = list(itertools.product(*[range(len(v)) for v in LAY_MENU.values()]))   # the 24 rules (f, g, d)
IO_COMBOS = list(itertools.product(*[range(len(DIV_MENU[n])) for n in IO_SW]))


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
    """The prediction path's state after a link, an addition or a round: the same value, its gradient times GRAD_DAMP
    (COMBINED.md section 3). Without autograd, as at evaluation, there is no gradient to scale."""
    if GRAD_DAMP == 1.0 or not torch.is_grad_enabled():
        return x
    return x.detach() + GRAD_DAMP * (x - x.detach())


# ------------------------------------------------------------------------------------------------ hard digits (integers)
def sel(idx: Tensor, opts: list) -> Tensor:
    """A switch on the hard machine: per row, the option idx [X] among opts [X, ...]."""
    out = opts[0]
    for i, o in enumerate(opts[1:], 1):
        out = torch.where((idx == i).view(-1, *([1] * (o.dim() - 1))), o, out)
    return out


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


def row_hard(X: Tensor, b: Tensor, j: Tensor, tabs, visits: Tensor | None = None,
             pvisits: Tensor | None = None) -> Tensor:
    """Schoolbook row j of X * b on digits: X [..., W] times the digits b [...], row indices j [...], with the snapped
    product cell (flat lo and hi tables [100] indexed a * 10 + b) and the argmax layout (tabs[5], tabs[6]: the source of
    every column [W, C], W = nothing): column c of the low-digit number holds the low digit of cell src_lo[j, c], the
    high-digit number likewise; the two are summed by the learned adder. Returns [..., C]."""
    Td, Tb, Plo, Phi = tabs[:4]
    idx = X * 10 + b[..., None]
    z = torch.zeros_like(X[..., :1])
    tally(pvisits, idx, 100)
    lo = torch.cat([Plo[idx], z], -1).gather(-1, tabs[5][j])
    hi = torch.cat([Phi[idx], z], -1).gather(-1, tabs[6][j])
    return add_learned(lo, hi, Td, Tb, visits)[0]


def square_hard(Sd: Tensor, tabs, visits: Tensor | None = None, pvisits: Tensor | None = None, check: bool = False):
    """s^2 by schoolbook on digits Sd [X, W] -> the 2W digits of s^2 [X, 2W]. The W rows j (s times s_j, laid out by the
    argmax layout) side by side, then summed one by one by the learned adder in C places (no table can wrap). With
    `check` also, per row, whether every one of its additions passes its own subtraction check (A + B) - B = A with
    borrow = carry."""
    X, (Td, Tb) = Sd.shape[0], tabs[:2]
    js = torch.arange(W, device=Sd.device)[None].expand(X, W)
    rows = row_hard(Sd[:, None, :].expand(X, W, W), Sd, js, tabs, visits, pvisits)   # [X, W (j), C]
    acc = torch.zeros(X, C, dtype=Sd.dtype, device=Sd.device)
    ok = torch.ones(X, dtype=torch.bool, device=Sd.device)
    for j in range(W):
        B = rows[:, j]
        Y, cy = add_learned(acc, B, Td, Tb, visits)
        if check:
            A2, b2, _ = sub_hard(Y, B, Td, Tb)
            ok = ok & (A2 == acc).all(-1) & (b2 == cy)
        acc = Y
    return (acc[:, :2 * W], ok) if check else acc[:, :2 * W]


def ladder_hard(Xn: Tensor, tabs, visits: Tensor | None = None):
    """The routed ladder's hard rungs on the numbers Xn [n, W]: the machine's own rows j of X * b [n, W, 10, C], the
    targets row_j(X, b) + X 10^j (b = 0..8, the learned adder) [n, W, 9, C] and whether rung b of row j checks out
    [n, W, 10] (rung 0: row_j(X, 0) = 0)."""
    n, dev = Xn.shape[0], Xn.device
    hard = row_hard(Xn[:, None, None, :].expand(n, W, 10, W), torch.arange(10, device=dev)[None, None].expand(n, W, 10),
                    torch.arange(W, device=dev)[None, :, None].expand(n, W, 10), tabs, visits)
    Xj = torch.stack([F.pad(Xn, (j, C - W - j)) for j in range(W)], 1)            # X 10^j [n, W, C]
    tgt = add_learned(hard[:, :, :9], Xj[:, :, None].expand(n, W, 9, C), tabs[0], tabs[1], visits)[0]
    ok = torch.cat([(hard[:, :, :1] == 0).all(-1), (hard[:, :, 1:] == tgt).all(-1)], 2)
    return hard, tgt, ok


def square_ok(Sd: Tensor, tabs) -> Tensor:
    """Per row, whether the machine's square of Sd checks out with learned parts only: the routed ladder verifies every
    rung of every row up to s's largest digit, and every addition passes its subtraction check."""
    verified = ladder_hard(Sd, tabs)[2].long().cumprod(2).bool()                  # [X, W, 10]
    top_ = Sd.max(1).values[:, None, None].expand(-1, W, 1)
    rows_ok = verified.gather(2, top_)[..., 0].all(1)
    return rows_ok & square_hard(Sd, tabs, check=True)[1]


def top(V: Tensor, steps: int) -> Tensor:
    """The digits of V // 10^steps in P places (V [X, L] digits): the division's starting register."""
    R = V[:, steps:steps + P]
    return F.pad(R, (0, P - R.shape[1]))


def shift_opts(R: Tensor, d: Tensor) -> list:
    """The shift switch's options for the register R [..., P] and the incoming digit d [...]: down, stay, up, skip."""
    d = d[..., None]
    return [torch.cat([R[..., 1:], d], -1), torch.cat([d, R[..., 1:]], -1), torch.cat([d, R[..., :-1]], -1), R]


def shift_opts_soft(R: Tensor, d: Tensor) -> list:
    """shift_opts on digit distributions: R [X, P, 10], d [X, 10]."""
    d = d[:, None]
    return [torch.cat([R[:, 1:], d], 1), torch.cat([d, R[:, 1:]], 1), torch.cat([d, R[:, :-1]], 1), R]


def n_register(Nd: Tensor, nread: Tensor) -> Tensor:
    """The subtrahend register in P places: N's W read digits reversed or as given (nread [X])."""
    return F.pad(sel(nread, [Nd.flip(-1), Nd]), (0, P - Nd.shape[-1]))


def chain_hard(Rp: Tensor, M: Tensor, w: dict, Td: Tensor, Tb: Tensor, visits: Tensor | None = None,
               gvisits: Tensor | None = None) -> Tensor:
    """One division step's chain on digits with per-row wiring w (a, b, keep [X]; g0, g1 [X] bool: the gate commits
    on borrow 0 / 1): S - 1 links, each a subtraction with the table; the chain register takes the link's result as
    `keep` says. Returns the register [X, P]."""
    D = Rp
    for _ in range(1, S):
        A = sel(w["a"], [Rp, M, D])
        B = sel(w["b"], [D, Rp, M])
        E, c, idx = sub_hard(A, B, Td, Tb)
        commit = torch.where(c == 0, w["g0"], w["g1"])
        take = torch.where(w["keep"] == 1, commit, w["keep"] == 0)
        D = torch.where(take[:, None], E, D)
        tally(visits, idx, 200)
        tally(gvisits, c, 2)
    return D


def divide_hard(V: Tensor, Nd: Tensor, w: dict, Td: Tensor, Tb: Tensor, visits: Tensor | None = None,
                gvisits: Tensor | None = None, regs: list | None = None) -> Tensor:
    """V mod N on digits V [X, 2W] by the division with per-row wiring w (vread, shift, read, nread, a, b, keep, g0,
    g1 [X]): the register starts with the dividend's top W digits (after vread) and takes W steps. Returns the read W
    places [X, W]. `regs` collects every step's register Rp."""
    Vp = sel(w["vread"], [V, V.flip(-1)])
    M = n_register(Nd, w["nread"])
    R = top(Vp, W)
    for t in reversed(range(W)):
        Rp = sel(w["shift"], shift_opts(R, Vp[:, t]))
        if regs is not None:
            regs.append(Rp)
        R = chain_hard(Rp, M, w, Td, Tb, visits, gvisits)
    return sel(w["read"], [R[:, 1:W + 1], R[:, :W]])


def count_round(C: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor):
    """One round of the countdown C [X, K]: C - 1 with the table and the gate on its final borrow. Returns the
    difference, whether the round commits [X], the final borrow [X] and the table rows."""
    Cn, b, idx = sub_hard(C, F.one_hot(torch.zeros_like(C[:, 0]), C.shape[1]), Td, Tb)   # the digits of 1
    return Cn, Gc[b] == 1, b, idx


def run_hard(Sd: Tensor, C: Tensor, Nd: Tensor, tabs, w: dict, cap: int | None = None, visits: Tensor | None = None,
             gvisits: Tensor | None = None, cvisits: Tensor | None = None, pvisits: Tensor | None = None):
    """The hard machine on digits with per-row division wiring w: rounds until no row commits (at most `cap`, by
    default 10^K). A committing row squares and reduces its state and takes the countdown's difference; a holding row
    keeps both. Returns the state [X, W], the input of each row's last committed squaring [X, W] (x if none), the
    number of committed rounds [X] and whether the row has held [X]."""
    Td, Tb, Plo, Phi, Gc = tabs[:5]
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
        R = divide_hard(square_hard(s, tabs, visits, pvisits), Nd, w, Td, Tb, visits, gvisits)
        s = torch.where(k[:, None], R, s)
        C = torch.where(k[:, None], Cn, C)
        rounds = rounds + k.long()
    return s, prev, rounds, held


def repeat_wiring(w: dict, K: int) -> dict:
    """The wiring w (per-row [X]) for K copies of the rows, copy-major."""
    return {k: v.repeat(K) for k, v in w.items()}


def combo_wiring(w: dict, names: tuple, combos: Tensor, n: int, gate: bool = False) -> dict:
    """The wiring for K = len(combos) copies of n rows (copy-major): the switches `names` (and with `gate` the gate's
    two rows) take each combination's options, every other switch keeps w's (per row, [n])."""
    K = combos.shape[0]
    out = repeat_wiring(w, K)
    for i, name in enumerate(names):
        out[name] = combos[:, i].repeat_interleave(n)
    if gate:
        out["g0"] = (combos[:, -2] == 1).repeat_interleave(n)
        out["g1"] = (combos[:, -1] == 1).repeat_interleave(n)
    return out


# ------------------------------------------------------------------------------------------------ digit distributions
def onehot0(n: int, L: int, device, dtype=torch.float32) -> Tensor:
    """n numbers of L exact 0 digits [n, L, 10]."""
    z = torch.zeros(n, L, 10, device=device, dtype=dtype)
    z[..., 0] = 1
    return z


def zpad(D: Tensor, low: int, high: int) -> Tensor:
    """Digit distributions [X, L, 10] padded with exact 0 digits below (low) and above (high)."""
    return torch.cat([onehot0(D.shape[0], low, D.device, D.dtype), D, onehot0(D.shape[0], high, D.device, D.dtype)], 1)


def carry_on(c: Tensor, ste_cin: bool) -> Tensor:
    """The borrow passed to the next place: the soft one, or (ste_cin) the snapped borrow of the entry used, with the
    soft one's gradient. With ste_cin a lesson's next place uses exactly one table entry, so a lesson trains only the
    entries it uses (ABLATION.md rule 3); the borrow returned for the lesson itself stays the soft one."""
    if not ste_cin:
        return c
    return ste(F.one_hot(c.argmax(-1), 2).to(c.dtype), c)


def link_soft(D: Tensor, M: Tensor, TDC: Tensor, ste_cin: bool = False):
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
        borrows.append(Y[:, 10:])
        c = carry_on(Y[:, 10:], ste_cin)
    return torch.stack(out, 1), torch.stack(borrows, 1)


def sub_soft(D: Tensor, M: Tensor, TDC: Tensor, used: Tensor | None = None, ste_cin: bool = False):
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
        borrows.append(Y[:, 10:])
        c = carry_on(Y[:, 10:], ste_cin)
    return torch.stack(out, 1), torch.stack(borrows, 1)


def add_soft_learned(A: Tensor, B: Tensor, TDC: Tensor, used: Tensor | None = None):
    """The learned adder on digit distributions [X, L, 10]: A - (0 - B). Returns the sum's digits and P(carry = 1)."""
    C, bc = sub_soft(onehot0(B.shape[0], B.shape[1], B.device, B.dtype), B, TDC, used)
    Y, by = sub_soft(A, C, TDC, used)
    return Y, bc[:, -1, 1] * by[:, -1, 0]


def add_soft_joint(A: Tensor, B: Tensor, TH: Tensor) -> Tensor:
    """The learned adder on digit distributions A, B [X, L, 10] for the ladder's soft rows: A - (0 - B) with the table
    TH (soft_table's layout), the two ripples carried as one chain over their joint borrow state (COMBINED.md 4.2)."""
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


def row_soft(X: Tensor, b: int, j: Tensor, PLO: Tensor, PHI: Tensor, SLO: Tensor, SHI: Tensor, TH: Tensor) -> Tensor:
    """Row j of X * b on digit distributions: the soft product cell PLO, PHI [10, 10, 10] (a, b) -> digit
    probabilities on the exact digits X [n, W], row indices j [n], the soft layout SLO, SHI [W, C, W + 1] blended on
    the wire (column c of the low-digit number: sum_i SLO[j, c, i] times cell i's low digit, the last source a 0),
    summed by the learned adder on the snapped table TH (one joint chain over both ripples' borrows, add_soft_joint).
    Returns [n, C, 10]."""
    z = onehot0(X.shape[0], 1, X.device)
    lo = torch.cat([PLO[X, b], z], 1)                                       # [n, W + 1, 10]
    hi = torch.cat([PHI[X, b], z], 1)
    return add_soft_joint(torch.einsum("nci,nik->nck", SLO[j], lo), torch.einsum("nci,nik->nck", SHI[j], hi), TH)


def mixture_nll(lw: Tensor, verdict: Tensor) -> Tensor:
    """-log of the mixture's likelihood per observation: lw [K] log weights of K whole sub-wirings (summing to 1),
    verdict [K, n] whether sub-wiring k passes observation n. Zero gradient when no sub-wiring passes."""
    return -torch.logsumexp(lw[:, None] + verdict.float().clamp_min(1e-12).log(), 0)


# ------------------------------------------------------------------------------------------------ the prediction path
def pick(w: Tensor, opts: list) -> Tensor:
    """A switch on the prediction path: w [n] its straight-through one-hot (values exactly 0 or 1), opts the options'
    digit distributions. With autograd the blend sum_o w_o opts_o (its value is the chosen option exactly, its gradient
    reaches every option's logit); without autograd, as at evaluation, the chosen option itself (the same value)."""
    if not torch.is_grad_enabled():
        return opts[int(w.argmax())]
    return sum(w[i] * o for i, o in enumerate(opts))


def square_ste(Sd: Tensor, TDC: Tensor, PROD: Tensor, SRC: tuple, used: Tensor | None = None,
               pused: Tensor | None = None):
    """s^2 by schoolbook on one-hot digits Sd [X, W, 10] with the straight-through product cell PROD [100, 20], layout
    SRC (straight-through one-hot sources [W, C, W + 1], low and high) and subtraction table TDC: square_hard's
    datapath. Returns [X, 2W, 10]."""
    X = Sd.shape[0]
    cell = (Sd[:, None, :, :, None] * Sd[:, :, None, None, :]).reshape(X, W, W, 100)   # [x, j, i, (s_i, s_j)]
    if pused is not None:
        pused += cell.detach().sum((0, 1, 2))
    lohi = cell @ PROD
    z = onehot0(X * W, 1, Sd.device, Sd.dtype).view(X, W, 1, 10)
    lo = torch.cat([lohi[..., :10], z], 2)                                  # [X, W (j), W + 1 (source), 10]
    hi = torch.cat([lohi[..., 10:], z], 2)
    Lc = torch.einsum("jci,xjik->xjck", SRC[0], lo).reshape(X * W, C, 10)
    Hc = torch.einsum("jci,xjik->xjck", SRC[1], hi).reshape(X * W, C, 10)
    rows = damp(add_soft_learned(Lc, Hc, TDC, used)[0]).view(X, W, C, 10)
    acc = onehot0(X, C, Sd.device, Sd.dtype)
    for j in range(W):
        acc = damp(add_soft_learned(acc, rows[:, j], TDC, used)[0])
    return acc[:, :2 * W]


def divide_ste(V: Tensor, N1: Tensor, TDC: Tensor, G: Tensor, SW: dict, used: Tensor | None = None,
               gused: Tensor | None = None) -> Tensor:
    """V mod N on one-hot digits V [X, 2W, 10], N1 [X, W, 10] with the straight-through switches SW (divide_hard's
    datapath): each link's final borrow, read by the straight-through gate G [2, 2], decides the gated write.
    Returns the read W places [X, W, 10]."""
    Vp = pick(SW["vread"], [V, V.flip(1)])
    M = pick(SW["nread"], [zpad(N1.flip(1), 0, P - W), zpad(N1, 0, P - W)])
    R = zpad(Vp[:, W:], 0, P - W)
    for t in reversed(range(W)):
        Rp = pick(SW["shift"], shift_opts_soft(R, Vp[:, t]))
        D = Rp
        for _ in range(1, S):
            A = pick(SW["a"], [Rp, M, D])
            B = pick(SW["b"], [D, Rp, M])
            E, bor = sub_soft(A, B, TDC, used)
            b = bor[:, -1]
            if gused is not None:
                gused += b.detach().sum(0)
            k = (b * G[:, 1]).sum(-1)                                       # 1: the gate commits
            kk = pick(SW["keep"], [torch.ones_like(k), k, torch.zeros_like(k)])[:, None, None]
            D = damp(kk * E + (1 - kk) * D)
        R = D
    return pick(SW["read"], [R[:, 1:W + 1], R[:, :W]])


def run_ste(Sd: Tensor, Cnt: Tensor, N1: Tensor, TDC: Tensor, PROD: Tensor, G: Tensor, SW: dict, SRC: tuple,
            used: Tensor | None = None, cused: Tensor | None = None, pused: Tensor | None = None,
            gused: Tensor | None = None) -> Tensor:
    """run_hard on one-hot digits: the countdown C [X, K, 10] takes C - 1 with the table each round and the gate on
    its final borrow decides; a committing row squares and reduces its state. The rounds end when no row commits."""
    one = onehot0(Cnt.shape[0], Cnt.shape[1], Cnt.device, Cnt.dtype)       # the digits of 1
    one[:, 0] = F.one_hot(torch.tensor(1, device=Cnt.device), 10).to(Cnt.dtype)
    s = Sd
    for _ in range(10 ** Cnt.shape[1]):
        Cn, bor = sub_soft(Cnt, one, TDC, cused)
        b = bor[:, -1]
        if gused is not None:
            gused += b.detach().sum(0)
        k = (b * G[:, 1]).sum(-1)
        if not bool((k > 0.5).any()):
            break
        R = divide_ste(square_ste(s, TDC, PROD, SRC, used, pused), N1, TDC, G, SW, used, gused)
        kk = k[:, None, None]
        s = damp(kk * R + (1 - kk) * s)
        Cnt = damp(kk * Cn + (1 - kk) * Cnt)
    return s


def shift(A: Tensor, d: int) -> Tensor:
    """Position distributions [X, Lp] moved by d places (mass at q goes to q + d; zero-filled)."""
    if d >= 0:
        return F.pad(A, (d, 0))[:, :A.shape[1]]
    return F.pad(A, (0, -d))[:, -d:]


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Control:
    """The training schedule (Python-side, not model state): the I/O stages, the machine's phases A, B, C, frozen,
    whether the ladder and the wiring lessons have joined; and the audits' counters."""

    def __init__(self) -> None:
        designed = MODE == "designed"
        self.step = 0
        self.phase = "frozen" if designed else "A"
        self.io = 5 if designed else 1        # 1 answers, 2 ranks, 3 reading, 4 tiling, 5 T's direction done
        self.ladder = designed                # the product cell's ladder has joined
        self.wire_io = designed               # the division's I/O lesson has joined (the chain's: phase C)
        self.table_streak = 0                 # steps in a row with every check of the table exact (phase B on)
        self.ladder_wait = 0
        self.ready = 0                        # steps in a row with the table and the ladder exact (for phase C)
        self.chain_streak = 0                 # steps in a row with the argmax chain exact on the division ladder
        self.in_phase = 0
        self.streak = 0
        self.pick = 0                         # steps in a row with a hypothesis passing x < N
        self.best = 0
        self.rows3 = 0.0
        self.chosen = 0.0                     # the mixture's mean posterior on its favourite reading
        self.io_exact = False                 # the I/O's own hard checks, this step
        self.ans_ok = 0.0                     # the argmax machine on the rows' answers (T <= REAL_MAX_T), this step
        self.facts_from = 0                   # the step the facts about x and N joined (0: not yet)
        self.reset = False                    # the scheduler clears Adam's state at the switch to phase B
        self.reset_params: list = []          # ... or only these parameters' (the product cell when the ladder joins)
        self.win: deque = deque(maxlen=STAGE_WIN)
        self.visits: Tensor | None = None     # table rows used by training's lessons (the audit)
        self.pvisits: Tensor | None = None    # product cells the ladder trained
        self.gvisits: Tensor | None = None    # gate rows (borrow 0, 1) met by the division ladder's argmax chain
        self.used: Tensor | None = None       # at evaluation: table rows of the squarings and reductions, ...
        self.used_count: Tensor | None = None  # ... of the countdown, ...
        self.pused: Tensor | None = None      # ... product cells, ...
        self.used_gate: Tensor | None = None  # ... gate rows
        self.reported = None
        self.history: list = []
        self.t_log = ""


def true_sources(device) -> tuple[Tensor, Tensor]:
    """The true layout (the designed start and the reports only): column c of row j holds pair c - j's low digit and
    pair c - j - 1's high digit (W: nothing) [W, C]."""
    j = torch.arange(W, device=device)[:, None]
    c = torch.arange(C, device=device)[None, :]
    lo = torch.where((c - j >= 0) & (c - j < W), c - j, W)
    hi = torch.where((c - j - 1 >= 0) & (c - j - 1 < W), c - j - 1, W)
    return lo, hi


def rule_maps(device) -> Tensor:
    """The source map of every placement rule (fi, gj, d) [2, 3, 4, W, C] (W: nothing): column c of row j holds the
    pair i whose column f(i) + g(j) + d is c."""
    j = torch.arange(W, device=device)[:, None]
    c = torch.arange(C, device=device)[None, :]
    out = torch.full((2, 3, 4, W, C), W, dtype=torch.long, device=device)
    for gi, g in enumerate((torch.zeros_like(j), j, W - 1 - j)):
        for di, d in enumerate(LAY_MENU["d"]):
            f = c - g - d                                                    # f(i) of the pair in column c
            out[0, gi, di] = torch.where((f >= 0) & (f < W), W - 1 - f, W)   # f(i) = W - 1 - i
            out[1, gi, di] = torch.where((f >= 0) & (f < W), f, W)           # f(i) = i
    return out


class LearnedWiringSq(nn.Module):
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
        self.sw = nn.ParameterDict({k: nn.Parameter(INIT * torch.randn(len(v))) for k, v in DIV_MENU.items()})
        if LAYOUT == "rule":                                                # layout: a placement rule per output kind
            self.lay = nn.Parameter(INIT * torch.randn(2, len(LAY_RULES)))   # one weight per whole rule (lo, hi)
        else:                                                               # layout: row j, column c -> source pair
            self.src_lo = nn.Parameter(INIT * torch.randn(W, C, W + 1))    # ... (the last option: nothing)
            self.src_hi = nn.Parameter(INIT * torch.randn(W, C, W + 1))
        self.maps = None                                                    # rule_maps, made on the first use
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
        """Every table and switch at the truth (the plumbing check only): the subtractor, the product cell, the gate
        (commit unless the subtraction borrows), the division's wiring DIV_TRUE; N read after its marker, x before T's
        marker, T and the answer at the end, reading leftwards; digits 7..16 valued 0..9."""
        hot = lambda t, n: 10.0 * F.one_hot(t, n).float() - 5.0
        Td, Tb = exact_tables("cpu")
        Plo, Phi = exact_products("cpu")
        self.a.copy_(hot(Td, 10).view(10, 10, 2, 10))
        self.b.copy_(hot(Tb, 2).view(10, 10, 2, 2))
        self.p_lo.copy_(hot(Plo, 10).view(10, 10, 10))
        self.p_hi.copy_(hot(Phi, 10).view(10, 10, 10))
        self.g.copy_(hot(torch.tensor([1, 0]), 2))
        for k, v in DIV_MENU.items():
            self.sw[k].copy_(hot(torch.tensor(DIV_TRUE[k]), len(v)))
        if LAYOUT == "rule":
            for k, t in enumerate(LAY_TRUE.values()):
                self.lay[k].copy_(hot(torch.tensor(LAY_RULES.index(t)), len(LAY_RULES)))
        else:
            lo, hi = true_sources("cpu")
            self.src_lo.copy_(hot(lo, W + 1))
            self.src_hi.copy_(hot(hi, W + 1))
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

    # ------------------------------------------------------------------ the tables and the wiring
    def tables(self):
        """The snapped tables: subtractor digit and borrow [200], product lo and hi [100], gate [2] (1 = commit), and
        the argmax layout (the source of every column of every row, low and high [W, C])."""
        lo, hi = self.sources()
        return (self.a.argmax(-1).flatten(), self.b.argmax(-1).flatten(), self.p_lo.argmax(-1).flatten(),
                self.p_hi.argmax(-1).flatten(), self.g.argmax(-1), lo, hi)

    def rule_maps(self) -> Tensor:
        if self.maps is None or self.maps.device != self.a.device:
            self.maps = rule_maps(self.a.device)
        return self.maps

    def sources(self) -> tuple:
        """The argmax layout: the source pair of every column of every row [W, C] (W: nothing), low and high."""
        if LAYOUT != "rule":
            return self.src_lo.argmax(-1), self.src_hi.argmax(-1)
        m = self.rule_maps().reshape(len(LAY_RULES), W, C)
        return m[int(self.lay[0].argmax())], m[int(self.lay[1].argmax())]

    def source_probs(self) -> tuple:
        """The soft layout: every column's distribution over its sources [W, C, W + 1], low and high (LAYOUT "rule":
        the rules' one-hot maps weighted by the product of the switches' probabilities)."""
        if LAYOUT != "rule":
            return self.src_lo.float().softmax(-1), self.src_hi.float().softmax(-1)
        m = F.one_hot(self.rule_maps(), W + 1).float().reshape(len(LAY_RULES), W, C, W + 1)
        return tuple(torch.einsum("r,rjxs->jxs", self.lay[k].float().softmax(-1), m) for k in range(2))

    def ste_sources(self) -> tuple:
        """The layout on the prediction path: straight-through one-hot sources [W, C, W + 1], low and high (the value
        is the argmax layout exactly, the gradient the soft layout's)."""
        soft = self.source_probs()
        return tuple(ste(F.one_hot(h, W + 1).float(), s_) for h, s_ in zip(self.sources(), soft))

    def layout_right(self) -> int:
        """How many of the 2 W C columns hold their true source (the reports only)."""
        lo, hi = true_sources(self.a.device)
        slo, shi = self.sources()
        return int((slo == lo).sum() + (shi == hi).sum())

    def layout_names(self) -> str:
        if LAYOUT != "rule":
            return f"columns right={self.layout_right()}/{2 * W * C}"
        names = lambda r: "+".join(str(LAY_MENU[q][v]) for q, v in zip(LAY_MENU, r))
        return " ".join(f"{k}={names(LAY_RULES[int(self.lay[i].argmax())])}" for i, k in enumerate(("lo", "hi"))) \
            + f" right={self.layout_right()}/{2 * W * C}"

    def soft_table(self) -> Tensor:
        """The subtractor's softmax, laid out [20 (r, c), 120 (m, 10 digit + 2 borrow)]."""
        T = torch.cat([self.a.float().softmax(-1), self.b.float().softmax(-1)], -1)
        return T.permute(0, 2, 1, 3).reshape(20, 120)

    def hard_table(self) -> Tensor:
        """The snapped subtractor as one-hot rows in soft_table's layout (no gradient): the learned adder of the
        ladder's soft rows."""
        T = torch.cat([F.one_hot(self.a.argmax(-1), 10), F.one_hot(self.b.argmax(-1), 2)], -1).float()
        return T.permute(0, 2, 1, 3).reshape(20, 120)

    def ste_tables(self):
        """The straight-through tables of the prediction path: subtractor [20, 120] (soft_table's layout), product cell
        [100, 20] (lo, hi), gate [2, 2] (borrow -> hold, commit). Values are the snapped tables exactly."""
        hot = lambda p: ste(F.one_hot(p.argmax(-1), p.shape[-1]).float(), p.float().softmax(-1))
        T = torch.cat([hot(self.a), hot(self.b)], -1).permute(0, 2, 1, 3).reshape(20, 120)
        return T, torch.cat([hot(self.p_lo), hot(self.p_hi)], -1).reshape(100, 20), hot(self.g)

    def ste_switches(self) -> dict:
        """The division's switches on the prediction path: straight-through one-hots [n] (values exactly 0 or 1)."""
        return {k: ste(F.one_hot(v.argmax(), v.shape[0]).float(), v.float().softmax(-1)) for k, v in self.sw.items()}

    def wiring(self, X: int, device) -> dict:
        """The argmax wiring, per row [X]: every switch's option and the gate's two rows (commit on borrow 0 / 1)."""
        w = {k: torch.full((X,), int(v.argmax()), dtype=torch.long, device=device) for k, v in self.sw.items()}
        gc = self.g.argmax(-1) == 1
        w["g0"], w["g1"] = gc[0].expand(X).clone(), gc[1].expand(X).clone()
        return w

    def wiring_names(self) -> str:
        gate = "".join("C" if int(c) == 1 else "H" for c in self.g.argmax(-1))
        return " ".join(f"{k}={DIV_MENU[k][int(v.argmax())]}" for k, v in self.sw.items()) + f" gate={gate}"

    def combo_logw(self, names: tuple, combos: Tensor, gate: bool) -> Tensor:
        """log of the factorized mixture's weight of every combination combos [K, len(names) (+ 2 gate rows)]: the
        product of the switches' probabilities (and of the gate's rows)."""
        lw = 0
        for i, n in enumerate(names):
            lw = lw + self.sw[n].float().log_softmax(-1)[combos[:, i]]
        if gate:
            lg = self.g.float().log_softmax(-1)                             # [borrow, (hold, commit)]
            lw = lw + lg[0][combos[:, -2]] + lg[1][combos[:, -1]]
        return lw

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
            small = (tcount == 1) & (Cd[:, 0] >= 1) & (Cd[:, 0] <= REAL_MAX_T)  # the rows whose answers are used
            aux = self.lessons(Nd, Xd, Cd, small, width, self.tables(), e, end, k, code)
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
        """The machine on integer digits indexing the argmax tables and switches."""
        ctl, dev, rows = self.ctl, e.device, e.shape[0]
        code = self.ranks(k)
        Nd, _ = self.read(0, e, end, code)
        Xd, _ = self.read(1, e, end, code)
        Cd, _ = self.read(2, e, end, code)
        used, cused, pused, gused = [torch.zeros_like(t, dtype=torch.long) for t in
                                     (ctl.used, ctl.used_count, ctl.pused, ctl.used_gate)]
        s = run_hard(Xd, Cd, Nd, self.tables(), self.wiring(rows, dev), None, used, gused, cused, pused)[0]
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
        straight-through tables and switches. Every value equals predict_index's; the final logits' gradient reaches
        every table and switch."""
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

        TDC, PROD, G = self.ste_tables()
        s = run_ste(field(1), field(2), field(0), TDC, PROD, G, self.ste_switches(), self.ste_sources(), ctl.used,
                    ctl.used_count, ctl.pused, ctl.used_gate)
        wcode = hot(self.wc.float() + F.logsigmoid(self.dg.float())[None])  # the writer's code [10, V]
        out = torch.zeros(X, Lp, V, device=e.device)
        for j in range(W):
            out = out + slot(3, j)[:, :, None] * (s[:, j] @ wcode)[:, None, :]
        return 20.0 * out[:, W + 1:W + 1 + L] - 20.0

    # ------------------------------------------------------------------ lessons
    def cell_ladder(self, tabs):
        """The product cell's own ladder on single digits (it does not involve the layout): a * 0 = 0 and the soft
        cell (a, b + 1) as the two-place number [lo, hi] against the hard [lo, hi] of (a, b) plus a (the learned adder),
        along the verified frontier. Returns the loss and whether every rung checks out."""
        Td, Tb, Plo, Phi = tabs[:4]
        dev = Td.device
        PLO, PHI = self.p_lo.float().softmax(-1), self.p_hi.float().softmax(-1)
        log = lambda p: p.clamp_min(1e-12).log()
        with torch.no_grad():
            a = torch.arange(10, device=dev)
            h = torch.stack([Plo.view(10, 10), Phi.view(10, 10)], -1)       # [a, b, (lo, hi)]
            tg = add_learned(h[:, :9], F.pad(a[:, None], (0, 1))[:, None].expand(10, 9, 2), Td, Tb)[0]
            ok = torch.cat([(h[:, :1] == 0).all(-1), (h[:, 1:] == tg).all(-1)], 1)
            ver = ok.long().cumprod(1).bool()
            tally(self.ctl.pvisits, (a[:, None] * 10 + a[None]).flatten(), 100)
        loss = -(log(PLO[:, 0, 0]).mean() + log(PHI[:, 0, 0]).mean())
        for b in range(9):
            nll = -(log(PLO[:, b + 1]).gather(-1, tg[:, b, 0:1]) + log(PHI[:, b + 1]).gather(-1, tg[:, b, 1:2]))[:, 0]
            loss = loss + (nll * ver[:, b].float()).mean()
        return loss / 10, bool(ok.all())

    def ladder(self, Xn: Tensor, tabs, train: bool):
        """The product cell's and the layout's lessons on the numbers Xn [n, W] (the rows' x and N): the cell's own
        ladder, and the routed ladder: row_j(X, 0) = 0, and the soft row_j(X, b + 1) against the machine's own hard
        row_j(X, b) + X 10^j where row j's rungs 0..b all check out (the verified frontier), on LAYOUT_ROWS rows j per
        number (cycling). The adder of the targets, the checks and the soft rows is the learned one (the snapped table).
        Returns the loss (when `train`), the share of the hard rung checks that pass (0 unless the cell's own ladder
        passes too), which rungs are verified [n, W, 10] and the hard rows [n, W, 10, C]."""
        n, dev = Xn.shape[0], Xn.device
        with torch.no_grad():
            hard, tgt, ok = ladder_hard(Xn, tabs, self.ctl.visits)
            verified = ok.long().cumprod(2).bool()
            k = min(LAYOUT_ROWS, W)
            r = torch.arange(n, device=dev).repeat_interleave(k)
            js = (r * k + torch.arange(k, device=dev).repeat(n) + self.ctl.step * k) % W   # the rows j trained
            w = verified[r, js, :9].float()
            if train:
                trained = torch.cat([torch.ones_like(w[:, :1]), w], 1).bool()   # the rungs that get a lesson
                cells = (Xn[r][:, None, :] * 10 + torch.arange(10, device=dev)[None, :, None])
                tally(self.ctl.pvisits, cells[trained], 100)
        cell_loss, cell_ok = self.cell_ladder(tabs)
        lad_ok = float(ok.float().mean()) if cell_ok else 0.0
        if not train:
            return None, lad_ok, verified, hard
        PLO, PHI, TH = self.p_lo.float().softmax(-1), self.p_hi.float().softmax(-1), self.hard_table()
        SLO, SHI = self.source_probs()
        Xr = Xn[r]
        loss = -row_soft(Xr, 0, js, PLO, PHI, SLO, SHI, TH)[..., 0].clamp_min(1e-12).log().mean()
        for b in range(9):
            soft = row_soft(Xr, b + 1, js, PLO, PHI, SLO, SHI, TH)
            nll = -soft.clamp_min(1e-12).log().gather(-1, tgt[r, js, b, :, None]).squeeze(-1)
            loss = loss + (nll.mean(-1) * w[:, b]).sum() / r.shape[0]
        return loss / 10 + CELL_LADDER * cell_loss, lad_ok, verified, hard

    def chain_lesson(self, Xd: Tensor, Nd: Tensor, rowsN: Tensor, verified: Tensor, tabs, train: bool):
        """The division ladder (one division step at a time): from the register x + N * q, q = 0..9 (the ladder's own
        row N * q added to x by the learned adder), one step must give back x. Rows whose N rungs are not all verified
        are left out. Every combination of the chain's switches and the gate's rows runs the step on the hard machine;
        the loss is the factorized mixture's (when `train`). Returns the loss and whether the argmax chain passes."""
        Td, Tb, Gc = tabs[0], tabs[1], tabs[4]
        dev = Xd.device
        n = min(CHAIN_ROWS, Xd.shape[0])
        with torch.no_grad():
            keep = verified[:n].all(1)
            Xs, Ns, rows = Xd[:n][keep], Nd[:n][keep], rowsN[:n][keep]        # rows [m, 10, C]: N * q (row 0)
            m = Xs.shape[0]
            if m == 0:
                return None, 0.0
            reg = add_learned(F.pad(Xs, (0, C - W))[:, None].expand(m, 10, C), rows, Td, Tb)[0][..., :P]
            reg = reg.reshape(m * 10, P)
            Nq = Ns.repeat_interleave(10, 0)
            xq = F.pad(Xs, (0, P - W)).repeat_interleave(10, 0)
            wa = self.wiring(m * 10, dev)
            Ra = chain_hard(reg, n_register(Nq, wa["nread"]), wa, Td, Tb, None, self.ctl.gvisits)
            div_ok = float((Ra == xq).all(1).float().mean())
            if not train:
                return None, div_ok
            combos = torch.tensor(CHAIN_COMBOS, device=dev)
            Kc = combos.shape[0]
            wk = combo_wiring(wa, CHAIN_SW, combos, m * 10, gate=True)
            Rk = chain_hard(reg.repeat(Kc, 1), n_register(Nq.repeat(Kc, 1), wk["nread"]), wk, Td, Tb)
            verdict = (Rk == xq.repeat(Kc, 1)).all(1).view(Kc, m * 10)
        lw = self.combo_logw(CHAIN_SW, combos, gate=True)
        return mixture_nll(lw, verdict).mean(), div_ok

    def io_lesson(self, Xd: Tensor, Nd: Tensor, Cd: Tensor, small: Tensor, tabs, train: bool):
        """The division's I/O switches: x mod N = x and N mod N = 0 through the whole division (IO_ROWS rows), and the
        rows with T <= REAL_MAX_T through the whole machine (their outputs, compared with the answers in
        token_training_loss). Every combination of vread, shift and read runs on the hard machine (the argmax chain
        and gate). Returns the laws' verdicts [K, n], the answer rows' outputs [K, m, W] and their indices, and whether
        the argmax wiring passes the laws."""
        Td, Tb = tabs[0], tabs[1]
        dev = Xd.device
        n = min(IO_ROWS, Xd.shape[0])
        with torch.no_grad():
            Xs, Ns = Xd[:n], Nd[:n]
            wa = self.wiring(n, dev)
            okx = (divide_hard(F.pad(Xs, (0, W)), Ns, wa, Td, Tb) == Xs).all(1)
            okn = (divide_hard(F.pad(Ns, (0, W)), Ns, wa, Td, Tb) == 0).all(1)
            law_ok = float((okx & okn).float().mean())
            if not train:
                return None, law_ok
            combos = torch.tensor(IO_COMBOS, device=dev)
            Kc = combos.shape[0]
            wk = combo_wiring(wa, IO_SW, combos, n)
            okx = (divide_hard(F.pad(Xs, (0, W)).repeat(Kc, 1), Ns.repeat(Kc, 1), wk, Td, Tb) == Xs.repeat(Kc, 1))
            okn = divide_hard(F.pad(Ns, (0, W)).repeat(Kc, 1), Ns.repeat(Kc, 1), wk, Td, Tb) == 0
            verdict = (okx.all(1) & okn.all(1)).view(Kc, n)
            sel_rows = torch.nonzero(small).flatten()
            outs = None
            if sel_rows.numel():
                mr = sel_rows.numel()
                wm = combo_wiring(self.wiring(mr, dev), IO_SW, combos, mr)
                outs = run_hard(Xd[sel_rows].repeat(Kc, 1), Cd[sel_rows].repeat(Kc, 1), Nd[sel_rows].repeat(Kc, 1),
                                tabs, wm, cap=REAL_MAX_T + 1)[0].view(Kc, mr, W)
        return {"verdict": verdict, "outs": outs, "sel": sel_rows, "combos": combos}, law_ok

    def lessons(self, Nd, Xd, Cd, small, width, tabs, e, end, k, code) -> dict:
        """The label-free lessons (the label-dependent ones run in token_training_loss on what this returns): see the
        module docstring and WIRING.md."""
        ctl = self.ctl
        ctl.step += 1
        dev = Xd.device
        Td, Tb, Plo, Phi, Gc = tabs[:5]
        M = F.pad(Nd, (0, P - W))                                            # N's digits as read, P places
        if ctl.visits is None:
            ctl.visits = torch.zeros(200, dtype=torch.long, device=dev)
            ctl.pvisits = torch.zeros(100, dtype=torch.long, device=dev)
            ctl.gvisits = torch.zeros(2, dtype=torch.long, device=dev)
        common = {"ctl": ctl, "model": self, "e": e, "end": end, "k": k, "code": code, "Xd": Xd, "Nd": Nd, "M": M,
                  "Cd": Cd, "small": small, "width": width, "tabs": tabs, "io": None, "exact": 0.0,
                  "table_exact": 0.0}
        if ctl.phase == "frozen":
            return {"loss": self.zero(), **common}
        TDC = self.soft_table()
        X, NL = Xd.shape[0], W * (S - 1)
        hot = lambda v: F.one_hot(v, 10).float()
        log = lambda p: p.clamp_min(1e-12).log()
        digits_nll = lambda Y, target: -log(Y).gather(-1, target[..., None]).squeeze(-1).sum(-1)
        with torch.no_grad():
            V, sq_add = square_hard(Xd, tabs, check=True)                    # the machine's own square of x
            if CHECK_REGS == "own":                                          # the argmax division's step registers
                regs = []
                divide_hard(V, Nd, self.wiring(X, dev), Td, Tb, regs=regs)
            else:                                                            # x and the square's P-place windows
                regs = [F.pad(Xd, (0, P - W))] + [V[:, t:t + P] for t in range(W - 1)]
            Rg = torch.stack(regs, 1).reshape(X * W, P)
            MW = M.repeat_interleave(W, 0)
            links = [Rg]
            for _ in range(S - 2):                                           # each register minus N again and again
                links.append(sub_hard(links[-1], MW, Td, Tb)[0])
            D = torch.stack(links, 1).reshape(X * NL, P)
            Mr = M.repeat_interleave(NL, 0)
            Z = torch.zeros_like(M)
        facts = ctl.phase in ("B", "C") or ctl.io >= 4                       # x < N, N > 0 may be used
        if facts and not ctl.facts_from:
            ctl.facts_from = ctl.step
        warm = not facts or ctl.step < ctl.facts_from + ADDER_WARM
        sc = STE_CIN == "all"
        sa = STE_CIN in ("all", "anchors")
        # every link checks itself, (D - N) + N = D, complete
        Es, be = link_soft(hot(D), Mr, TDC, sc)
        be = be[:, -1]
        if not warm:
            Cs, bc = link_soft(hot(Z), M, TDC, sc)                           # 0 - N, once per row
            bc = bc[:, -1]
        if warm:                                                             # the designed adder
            back, carry = add_soft(Es, Mr)
            agree = carry * be[:, 1] + (1 - carry) * be[:, 0]
        else:                                                                # (D - N) - (0 - N), the learned adder
            back, bf = sub_soft(Es, Cs.repeat_interleave(NL, 0), TDC, None, sc)
            pe, pf, pc = be[:, 1], bf[:, -1, 1], bc[:, 1].repeat_interleave(NL, 0)
            agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc   # be = bc - bf
        weight = torch.ones(P, device=dev)
        weight[0] = ID_PLACE0                                               # the lowest place tells the swapped code
        nll = (-log(back).gather(-1, D[..., None]).squeeze(-1) * weight).sum(-1) - ID_BORROW * log(agree)
        loss = nll.mean() / (weight.sum() + ID_BORROW)
        # the anchors (ADDER.md): on any numbers, then the facts about x and N
        xd, zero = F.pad(Xd, (0, P - W)), torch.zeros_like(Rg)
        checks = []                                                         # hard versions: (A, B, target, borrow)
        for wgt, A, B, target in ((ANCHOR_ZERO, Rg, zero, Rg), (ANCHOR_SELF, Rg, Rg, zero)):
            if wgt > 0:                                                     # R - 0 = R, R - R = 0, no final borrow
                Y, b = link_soft(hot(A), B, TDC, sa)
                loss = loss + wgt * (digits_nll(Y, target) - log(b[:, -1, 0])).mean() / (P + 1)
                checks.append((A, B, target, 0))
        if not warm and ANCHOR_NEG2 > 0:                                    # 0 + N = N: 0 - (0 - N) = N, complete
            Y2, b2 = sub_soft(hot(Z), Cs, TDC, None, sc)
            agree2 = bc[:, 1] * b2[:, -1, 1] + bc[:, 0] * b2[:, -1, 0]
            loss = loss + ANCHOR_NEG2 * (digits_nll(Y2, M) - log(agree2)).mean() / (P + 1)
        if not warm and ANCHOR_COMM > 0:                                    # R + N = N + R: R - (0 - N) = N - (0 - R)
            A1, b1 = sub_soft(hot(Rg), Cs.repeat_interleave(W, 0), TDC, None, sc)
            CR, bcr = link_soft(hot(torch.zeros_like(Rg)), Rg, TDC, sc)
            A2, b2 = sub_soft(hot(MW), CR, TDC, None, sc)
            c1 = bc[:, 1].repeat_interleave(W, 0) * b1[:, -1, 0]             # P(carry = 1)
            c2 = bcr[:, -1, 1] * b2[:, -1, 0]
            agree = log((A1 * A2).sum(-1)).sum(-1) + log(c1 * c2 + (1 - c1) * (1 - c2))
            loss = loss - ANCHOR_COMM * agree.mean() / (P + 1)
        if facts and ANCHOR_LESS > 0:                                       # x < N: x - N borrows
            loss = loss - ANCHOR_LESS * log(link_soft(hot(xd), M, TDC, sc)[1][:, -1, 1]).mean()
            checks.append((xd, M, None, 1))
        if facts and ANCHOR_MORE > 0:                                       # x < N: N - x does not borrow
            loss = loss - ANCHOR_MORE * log(link_soft(hot(M), xd, TDC, sc)[1][:, -1, 0]).mean()
            checks.append((M, xd, None, 0))
        if facts and ANCHOR_WRAP > 0:                                       # (x - N) - N does not borrow
            Ew = link_soft(hot(xd), M, TDC, sc)[0]
            loss = loss - ANCHOR_WRAP * log(link_soft(Ew, M, TDC, sc)[1][:, -1, 0]).mean()
        if facts and not warm and ANCHOR_NEG > 0:                           # N > 0: 0 - N borrows
            loss = loss - ANCHOR_NEG * log(bc[:, 1]).mean()
        # the product cell (SQUARE.md), once the table can be trusted
        lad_loss, lad_ok, verified, lad_rows = self.ladder(torch.cat([Xd, Nd]), tabs, ctl.ladder)
        if ctl.ladder:
            loss = loss + LADDER_W * lad_loss
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
                tally(ctl.visits, idw, 200)
                table = min(table, float((sub_hard(Ewh, M, Td, Tb)[1] == 0).float().mean()))
            if facts and not warm and ANCHOR_NEG > 0:
                table = min(table, float((bch == 1).float().mean()))
            top_ = Xd.max(1).values[:, None, None].expand(-1, W, 1)
            sq = verified[:X].gather(2, top_)[..., 0].all(1) & sq_add       # the squares of x check out
            exact = min(table, lad_ok, float(sq.float().mean()))
        # the division's wiring (WIRING.md): the chain's switches and the gate, then the division's I/O switches
        chain_on = ctl.phase == "C"
        div_loss, div_ok = self.chain_lesson(Xd, Nd, lad_rows[X:, 0], verified[X:, 0], tabs, chain_on)
        if chain_on and div_loss is not None:
            loss = loss + div_loss
        io, law_ok = self.io_lesson(Xd, Nd, Cd, small, tabs, ctl.wire_io)
        if ctl.wire_io:
            common["io"] = io
        exact = min(exact, div_ok, law_ok)
        if MODE == "learned":
            loss = loss + self.io_lessons(e, end, k, code, V[:, :W])
        common.update(exact=exact, table_exact=table, lad_ok=lad_ok, div_ok=div_ok, law_ok=law_ok)
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
        e, end, code, Xd, Nd = aux["e"], aux["end"], aux["code"], aux["Xd"], aux["Nd"]
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
            s2 = run_hard(Xd[rows].repeat(2, 1), torch.cat([Cp[rows], Cm[rows]]), Nd[rows].repeat(2, 1), aux["tabs"],
                          self.wiring(2 * n, dev))[0]
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

    def wire_answers(self, aux: dict, yd: Tensor) -> Tensor:
        """The division's I/O lesson (with the labels): the factorized mixture over the 16 combinations of vread, shift
        and read, on the laws' verdicts and on the answers of the rows with T <= REAL_MAX_T."""
        io = aux["io"]
        lw = self.combo_logw(IO_SW, io["combos"], gate=False)
        loss = mixture_nll(lw, io["verdict"]).mean()
        if io["outs"] is not None:
            ans = (io["outs"] == yd[io["sel"]][None]).all(-1)                  # [K, m]
            loss = loss + mixture_nll(lw, ans).mean()
        return loss

    def control(self, aux: dict, loss: float) -> None:
        """The schedule, once per step (after the label lessons): the I/O stages, the machine's phases, the ladder, the
        wiring lessons, the freeze."""
        ctl = self.ctl
        exact, table = aux["exact"], aux["table_exact"]
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
        ctl.win.append(table)
        ctl.in_phase += 1
        mean = sum(ctl.win) / len(ctl.win)
        if ctl.step % LOG_EVERY == 0 or ctl.step == 1:
            print(f"LR step={ctl.step} phase={ctl.phase} io={ctl.io} ladder={int(ctl.ladder)} wire_io={int(ctl.wire_io)} "
                  f"in_phase={ctl.in_phase} loss={loss:.4f} hard_exact={exact:.4f} table_exact={table:.4f} "
                  f"ladder_ok={aux.get('lad_ok', 0.0):.4f} div_ok={aux.get('div_ok', 0.0):.4f} "
                  f"law_ok={aux.get('law_ok', 0.0):.4f} ans_ok={ctl.ans_ok:.4f} win={mean:.4f} "
                  f"io_exact={ctl.io_exact} chosen={ctl.chosen:.4f} rows_visited={int((ctl.visits > 0).sum())}/200 "
                  f"cells_visited={int((ctl.pvisits > 0).sum())}/100 layout: {self.layout_names()} "
                  f"wiring: {self.wiring_names()} {ctl.t_log}",
                  flush=True)
        if ctl.phase == "A":
            if ctl.io >= 4 and ((len(ctl.win) == STAGE_WIN and mean >= ID_SWITCH) or ctl.in_phase >= ID_STEPS):
                # the checks pass on most links and the reading is chosen: a fresh optimizer for the repairs
                ctl.history.append((ctl.step, "switch", ctl.in_phase, round(mean, 4)))
                print(f"LR switch_to_facts step={ctl.step} win={mean:.4f}", flush=True)
                ctl.phase, ctl.reset, ctl.in_phase = "B", True, 0
                ctl.win.clear()
        elif ctl.phase == "B":
            ctl.ready = ctl.ready + 1 if (ctl.ladder and table == 1.0 and aux.get("lad_ok", 0.0) == 1.0) else 0
            if ctl.ready >= WIRE_WIN:
                # the table and the product cell are exact: the chain's switches and the gate can learn their wiring
                ctl.history.append((ctl.step, "chain", ctl.in_phase))
                print(f"LR chain_lesson_joins step={ctl.step}", flush=True)
                ctl.phase, ctl.in_phase = "C", 0
        elif ctl.phase == "C":
            if not ctl.wire_io:
                ctl.chain_streak = ctl.chain_streak + 1 if aux.get("div_ok", 0.0) == 1.0 else 0
                if ctl.chain_streak >= WIRE_WIN:
                    ctl.wire_io = True
                    ctl.history.append((ctl.step, "wire_io", ctl.in_phase))
                    print(f"LR io_wiring_lesson_joins step={ctl.step} wiring: {self.wiring_names()}", flush=True)
            ok = exact == 1.0 and ctl.io_exact and ctl.wire_io and ctl.ans_ok == 1.0
            ctl.streak = ctl.streak + 1 if ok else 0
            if ctl.streak >= FREEZE_WIN and ctl.io == 5:
                ctl.history.append((ctl.step, "frozen", ctl.in_phase, 1.0))
                print(f"LR frozen step={ctl.step} history={ctl.history} wiring: {self.wiring_names()}", flush=True)
                ctl.phase = "frozen"

    # ------------------------------------------------------------------ evaluation reports
    def report_tables(self, e: Tensor, end: Tensor) -> None:
        """At the first evaluation batch: the snapped tables, the wiring, the I/O, and how often training used each
        row."""
        ctl = self.ctl
        Td, Tb, Plo, Phi, Gc = self.tables()[:5]
        print("TABLE digit=" + "".join(str(int(d)) for d in Td.tolist()), flush=True)
        print("TABLE borrow=" + "".join(str(int(d)) for d in Tb.tolist()), flush=True)
        print("TABLE prod_lo=" + "".join(str(int(d)) for d in Plo.tolist()), flush=True)
        print("TABLE prod_hi=" + "".join(str(int(d)) for d in Phi.tolist()), flush=True)
        print("TABLE gate=" + "".join(str(int(d)) for d in Gc.tolist())
              + " p_commit=" + ",".join(f"{float(p):.4f}" for p in self.g.float().softmax(-1)[:, 1].tolist()),
              flush=True)
        print("WIRING " + self.wiring_names() + " true=" + str(all(int(v.argmax()) == DIV_TRUE[k]
                                                                  for k, v in self.sw.items())), flush=True)
        slo, shi = self.sources()
        print(f"LAYOUT right={self.layout_right()}/{2 * W * C} {self.layout_names()} "
              + " ".join(f"row{j}=lo:{','.join(str(int(v)) for v in slo[j].tolist())}"
                         f"/hi:{','.join(str(int(v)) for v in shi[j].tolist())}" for j in range(W)), flush=True)
        print("WIRING probabilities " + " ".join(f"{k}=" + ",".join(f"{float(p):.4f}" for p in v.softmax(-1).tolist())
                                                 for k, v in self.sw.items()), flush=True)
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
        if aux["io"] is not None:                                           # the division's I/O lesson
            loss = loss + model.wire_answers(aux, yd)
        with torch.no_grad():                                               # the argmax machine on the small rows
            sel = torch.nonzero(aux["small"]).flatten()
            if ctl.phase != "C":
                ctl.ans_ok = 0.0
            elif sel.numel() == 0:
                ctl.ans_ok = 1.0
            else:
                s = run_hard(aux["Xd"][sel], aux["Cd"][sel], aux["Nd"][sel], aux["tabs"],
                             model.wiring(sel.numel(), sel.device), cap=REAL_MAX_T + 1)[0]
                ctl.ans_ok = float((s == yd[sel]).all(1).float().mean())
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


def build_model(spec: ModelSpec) -> LearnedWiringSq:
    model = LearnedWiringSq(spec)
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
