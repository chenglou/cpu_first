"""LITMUS VARIANT (cpu_first/litmus/LITMUS.md): submission_wide_anchored.py with T's pointer stepping the way x's
does (one tie, STRD_OF): T's field uses x's step parameter instead of its own. On data whose T always has one digit
nothing else can tell T = 16 from T = 61. The rest of the file is unchanged.

WideMenus: the learned-wiring H1 machine (learned_wiring/submission_io3.py, WIRING.md) with its wiring chosen from
wide menus (MENUS.md): a register-transfer grammar for the reduction that holds long division in two forms, a
Montgomery-like right-to-left reduction and many other machines, and affine placement rules plus a number rule for the
multiply, which hold the schoolbook rows, the diagonals and other groupings. Every table is learned as in the combined
machine. One model, trained from a random start in one run.

This file is submission_wide.py (MENUS.md) with the subtraction table taught as in
learned_wiring/submission_anchored.py (ANCHORED.md, from anchors/submission_combined_anchors.py, ANCHORS.md): no
hand-written adder at any step. The designed adder of the 70-step warm start is gone from the file; until the reading
is chosen the table has no lesson. From then on x < N on every top part teaches the borrows; once x - N borrows and
N - x does not on every row of GATE_WIN batches, the order laws and bijectivity teach the digits; LATE_AFTER steps
later the pieces' own table lessons join, with the learned adder (they did so from the warm start's end before).
Every other lesson and the whole datapath are submission_wide.py's.

Datapath (base 10); the state is digits from the prompt to the answer:
  * read: three fields, N (8 slots), x (8 slots) and T (K = 2 slots), each by a learned pointer, with a learned
    digit-ness per token and a learned code (token -> digit) per reading hypothesis (IO.md). The pointer's shape is a
    switch (WIRING.md stage 3: step, offsets or absolute), one for all four fields; a slot reads 0 once a slot before
    it met a non-digit;
  * rounds: each round subtracts 1 from T's digits with the subtraction table, and the commit gate reads that
    subtraction's final borrow. A committing round squares the state and reduces it mod N and takes T - 1; a holding
    round keeps both (CONTROL.md). The rounds end when no row commits (at most 10^K);
      - the multiply (SQUARE.md's parts): every pair (s_i, s_j) gives the learned product cell's low and high digit;
        a column rule per output kind puts the digit in column a i + b j + c (MUL_MENU: io3's 24 rules, or every
        affine rule with small a, b), a number rule puts the pair in number alpha i + beta j + gamma (the rows by
        multiplier digit, by multiplicand digit, the diagonals, ...), a pair whose place is taken is dropped; each
        number's low and high parts are summed by the learned adder A - (0 - B) (ADDER.md), then the numbers one by
        one, in C = 2W + 2 places. The truth (SQUARE.md): columns i + j and i + j + 1, numbers by j;
      - the reduction: a register of P = W + 1 places runs W steps; a step brings in one digit of the dividend, runs a
        chain of S - 1 links and moves the register; then a final correction and the readout (CHAIN.md, CONTROL.md
        for the truth). Its wiring is a set of tied switches (DIV_MENU, one per kind of connection, shared by every
        step, link and round):
            a, b    each link's operands, from any register: the chain register D, the step's start Rp, N's register
                    M, the link's multiple of N (link s: s M, by the machine's own row 0), zero (level 4: also the
                    register before the bring-in R and the incoming digit d)
            op      subtract A - B | add A + B (the learned adder)
            flag    what the gate reads: the operation's final borrow (or carry) | the table's zero test of D's or E's
                    lowest digit (the borrow of d - 1) (level 4: also A < N, the borrow of A - M)
            keep    what a link writes into D: always | its result when the gate commits | never
            nread   N's register: N's digits as read | reversed
            vread   the dividend's digits as given | reversed
            start   the register's first content: the top W digits | zero | the bottom W digits (level 4: N)
            order   the digit brought in at step t: W - 1 - t (downwards) | t (upwards)
            bring   the register moved by k places, then the digit written at place p (or not at all), replacing that
                    place's digit or added there by the learned adder
            post    the register moved after the chain by -1..1 places (level 4: -2..2)
            final   nothing | a conditional subtraction of N | add the top W digits, then the conditional subtraction
                    (level 4: also two conditional subtractions)
            read    the answer: the W places from offset 0 or 1, in order or reversed
        The truth (CHAIN.md): D - M, borrow, gated, given; given, top, downwards, moved up 1 with the digit replacing
        place 0, no post move, no final correction, the low W places in order. The menu also holds long division by
        trial multiples (a = Rp, b = the link's multiple), and a Montgomery reduction (start zero, upwards, digit added
        at place 0, D + M while D's lowest digit is not 0, moved down 1, add the top digits, subtract N if needed: it
        computes V 10^-W mod N, the Montgomery form). No option list starts with the true option; every lesson treats
        the options alike; DIV_TRUE and MUL_TRUE are used only by MODE = "designed" and the reports;
  * write: the state's slot j at the position the writer's pointer (the same shape switch) gives slot j, its token by
    the writer's code.

Learned (MODE = "learned"): the subtraction table (r, m, borrow in) -> (digit, borrow out), 200 rows; the product
cell (a, b) -> (lo, hi), 100 rows; the commit gate flag -> (hold, commit), 2 rows; the division's switches; the
multiply's column rule per output kind and its number rule; the pointer shape switch and every shape's parameters per
field; the reader's digit-ness and codes and the writer's code (IO.md). MODE = "designed" starts every one of them at
the truth and never trains (the plumbing check).

The prediction path (PATH = "ste", rule 8): one-hot digits through straight-through tables and switches
(one-hot(argmax) + (softmax - softmax.detach())): the forward pass is exactly the snapped machine, and the final
logits' gradient reaches every table and switch (each switch's options are made lazily: all of them inside autograd,
only the chosen one without). The state's gradient is discounted by GRAD_DAMP after every link, addition and round
(the value is unchanged). PATH = "index" runs the same machine on integer digits.

What teaches what (WIRING.md section 2, MENUS.md):
  * the reader and writer: IO.md's stages, as in WIRING.md stage 3;
  * the subtraction table (ANCHORS.md, ANCHORED.md), no lesson until the reading is chosen (io stage 4: before it the
    reader can read every N as 0, WIRING.md section 3.4), then x < N on every top part (x[j:] - N[j:] borrows exactly
    when the digit strings differ there, N[j:] - x[j:] never borrows: soft ripples of the borrow outputs); once x - N
    borrows and N - x does not on every row of GATE_WIN consecutive batches, the column law and the row law on every
    pair of the entries the batch's links read (r - m - c keeps the order of r and reverses the order of m, unless
    exactly one of the two wraps, which then comes out larger; the learned borrows referee, order_pairs) and
    bijectivity of every column (m, c) and row (r, c) of the cell; LATE_AFTER steps after that the pieces' own lessons:
    the row's own numbers, x and the P-place windows of the machine's own square of x, minus N again and again (S - 1
    links), check every subtraction with the learned adder, (D - N) - (0 - N) = D, complete, the lowest place weighted
    ID_PLACE0; R - 0 = R, R - R = 0, 0 + N = N, R + N = N + R, and the facts x - N borrows, N - x does not,
    (x - N) - N does not, 0 - N borrows (STE_CIN as in WIRING.md). Their hard versions are the table's checks from
    step 1, which the phases and the ladder wait on;
  * the product cell and the column rules, once the table's checks are exact: the cell's own ladder on single digits,
    a * (b + 1) = a * b + a, and the routed ladder: for X in the rows' x and N, the soft row_j(X, b + 1) (soft cell,
    column rules blended on the wire, the learned adder) against the hard row_j(X, b) + X 10^j along the verified
    frontier (SQUARE.md), LAYOUT_ROWS rows j per number per step; the number rule by distributivity: X * X under every
    number rule against the sum of its single-digit rows, verdicts on the hard machine blended as a mixture;
  * the chain's switches (a, b, op, flag, keep, nread) and the gate (phase C, once the table and the ladder have been
    exact for WIRE_WIN steps): the division ladder, one division step at a time: from the register x + N * q (q =
    0..9) one step must give back x; every combination of the chain's switches and the gate's two rows runs on the
    hard machine, verdicts blended by the product of the switches' probabilities (a factorized mixture);
  * the I/O switches, once the argmax chain has passed the division ladder for WIRE_WIN steps: IO_MODE "enum": every
    combination of the I/O block by x mod N = x and N mod N = 0 through the whole division and the answers of the rows
    with T <= REAL_MAX_T through the whole machine; IO_MODE "local": the step's switches (bring, post) by the step law
    (the register x + N * q split into its upper places and its lowest digit: bringing the digit back in and running
    the step gives back x), the reading's (vread, start, order, final, read) by the whole-division laws and answers;
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
GATE_WIN = 10          # the digit lessons start once the borrow facts hold on every row of this many batches in a row
LATE_AFTER = 100       # the pieces' own table lessons join this many steps after the digit lessons start
STE_CIN = "anchors"    # which table lessons ripple with the snapped borrow straight-through: "all" | "anchors" (R - 0 = R
                       # and R - R = 0 only, ABLATION.md's HARD_CIN) | "none" (WIRING.md section 2.2)
CHECK_REGS = "rows"    # the registers the check subtracts N from: "rows" (x and the dividend's P-place windows: the
                       # row's own numbers, whatever the wiring) | "own" (the argmax division's step registers)
ANCHOR_ZERO = 1.0      # R - 0 = R on the registers
ANCHOR_SELF = 1.0      # R - R = 0 on the registers
ANCHOR_NEG2 = 1.0      # 0 + N = N with the learned adder: 0 - (0 - N) = N
ANCHOR_COMM = 1.0      # R + N = N + R with the learned adder
ANCHOR_LESS = 1.0      # x < N: x - N borrows (once the reading is chosen)
ANCHOR_MORE = 1.0      # x < N: N - x does not borrow (once the reading is chosen)
ANCHOR_WRAP = 1.0      # x - N wraps to 10^P + x - N >= N: (x - N) - N does not borrow (once the reading is chosen)
ANCHOR_NEG = 1.0       # N > 0: 0 - N borrows (once the reading is chosen)
# the product cell (SQUARE.md)
LADDER_W = 1.0         # weight of the ladder's loss
LADDER_WIN = 20        # the ladder joins once every check of the table has been exact this many steps in a row ...
LADDER_STEPS = 300     # ... or after this many steps of phase B
CELL_LADDER = 1.0      # weight of the product cell's own ladder a (b + 1) = a b + a on single digits (no layout)
LAYOUT_ROWS = 2        # rows j of the routed ladder each number trains per step (cycling through the W rows)
MUL_LEVEL = 3          # the multiply's menu (MUL_MENU, MENUS.md): 0 io3's 24 placement rules per output kind, numbers
                       # by multiplier digit | 1 every affine column rule a i + b j + c, a, b in -2..2 | 2 + the number
                       # rule alpha i + beta j + gamma, alpha, beta in -1..1 | 3 column rules with a, b in -3..3
NUM_ROWS = 128         # rows (the batch's x and N) the number rule's lesson squares under every number rule
# the division's wiring (MENUS.md, WIRING.md)
DIV_LEVEL = 4          # the division's menu (DIV_MENU, MENUS.md): 0 io3's | 1 + any register on either port, add, three
                       # flags | 2 + the start, the digit order, replace or add, the post-chain move, the final
                       # correction, the readout's direction | 3 + any move and any insert place | 4 + the register before
                       # the bring-in and the incoming digit as sources, the flag "A < N", moves by 2, N as the start,
                       # two final subtractions
IO_MODE = "local"      # how the I/O block is taught: "enum" (every combination of it, by the whole-division laws and the
                       # answers) | "local" (the step's switches bring and post by the step law on the division ladder's
                       # registers; the reading's switches by the whole-division laws and the answers) | "joint" (no
                       # step law at all: the chain's and the I/O's switches and the gate's rows enumerated together, by
                       # the whole-division laws and the answers alone, from phase C on)
WIRE_WIN = 20          # the chain lesson joins once the table and the ladder have been exact this many steps in a row;
                       # the I/O lessons once the argmax chain has passed the division ladder this many steps in a row
CHAIN_ROWS = 64 if DIV_LEVEL == 0 else 16   # rows of the batch whose x + N * q (q = 0..9) the division ladder (and the
                       # step law) divides
IO_ROWS = 128 if DIV_LEVEL == 0 else 32     # rows of the batch whose x and N the I/O lesson divides ...
IO_ROWS_ENUM = 4       # ... when every combination of the whole I/O block is enumerated (IO_MODE "enum", levels 2 on)
IO_LAWS = "xnq"        # the whole-division laws of the I/O lesson: x (x mod N = x), n (N mod N = 0), q (x + Q N mod N =
                       # x, Q another row's x: the dividend fills both halves; Q N by the machine's own rows)
# the reader and writer (IO.md)
SHAPES = ("step", "offsets", "absolute")   # the pointer's shapes (stage 3); one switch for the four fields
OFF = W + 1            # the offsets shape's range: -OFF..OFF from the anchor
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

STRD_OF = (0, 1, 1, 3)  # litmus: the step parameter each field uses (N, x, T, the writer): T steps the way x does
PERMS = list(itertools.permutations(range(3)))
HYP = [(rn, dn, rx, dx) for rn in range(3) for rx in range(3) if rn != rx for dn in range(2) for dx in range(2)]
HIDX = {h: i for i, h in enumerate(HYP)}  # (N's run, N's direction, x's run, x's direction); direction 0: step +1
SRC = {"D": 0, "Rp": 1, "M": 2, "Mq": 3, "Z": 4, "R": 5, "d": 6}
CODES = {"op": {"sub": 0, "add": 1}, "flag": {"borrow": 0, "zD": 1, "zE": 2, "lt": 3},
         "keep": {"always": 0, "gated": 1, "never": 2}, "nread": {"given": 0, "reversed": 1},
         "vread": {"given": 0, "reversed": 1}, "start": {"top": 0, "zero": 1, "bottom": 2, "N": 3},
         "order": {"down": 0, "up": 1}, "final": {"none": 0, "csub": 1, "addhigh": 2, "csub2": 3}}
IO3_SHIFTS = [(-1, P - 1), (0, 0), (1, 0), (0, -1)]   # io3's down, stay, up, skip as (move k, insert place p)


def div_menu(level: int) -> dict:
    """The division's switches at a menu level (option lists; no list starts with the true option; MENUS.md)."""
    m = {}
    if level == 0:
        m["a"], m["b"], m["op"], m["flag"] = ["Rp", "M", "D"], ["D", "Rp", "M"], ["sub"], ["borrow"]
    elif level >= 4:
        m["a"], m["b"] = ["d", "R", "Z", "Mq", "M", "Rp", "D"], ["d", "R", "Z", "D", "Rp", "Mq", "M"]
        m["op"], m["flag"] = ["add", "sub"], ["lt", "zD", "zE", "borrow"]
    else:
        m["a"], m["b"] = ["Z", "Mq", "M", "Rp", "D"], ["Z", "D", "Rp", "Mq", "M"]
        m["op"], m["flag"] = ["add", "sub"], ["zD", "zE", "borrow"]
    m["keep"], m["nread"], m["vread"] = ["always", "gated", "never"], ["reversed", "given"], ["reversed", "given"]
    if level <= 1:
        m["start"], m["order"] = ["top"], ["down"]
        m["bring"] = [(k, p, "rep") for k, p in IO3_SHIFTS]
        m["post"], m["final"], m["read"] = [0], ["none"], [(1, "given"), (0, "given")]
    else:
        m["start"], m["order"] = ["zero", "bottom", "top"], ["up", "down"]
        if level == 2:
            m["bring"] = [(k, p, md) for k, p in IO3_SHIFTS for md in ("add", "rep")]
        else:
            m["bring"] = [(k, p, md) for k in range(-(P - 1), P) for p in range(-1, P) for md in ("add", "rep")]
        m["post"], m["final"] = [-1, 1, 0], ["addhigh", "csub", "none"]
        m["read"] = [(o, d) for o in (1, 0) for d in ("reversed", "given")]
        if level >= 4:
            m["start"], m["post"] = ["N", "zero", "bottom", "top"], [-2, -1, 2, 1, 0]
            m["final"] = ["csub2", "addhigh", "csub", "none"]
    return m


def mul_menu(level: int) -> dict:
    """The multiply's switches: one column rule (a, b, c) per output kind (column a i + b j + c) and one number rule
    (alpha, beta) (number alpha i + beta j + gamma) (MENUS.md)."""
    if level == 0:
        rules = [(fa, gb, fc + gc + d) for fa, fc in ((-1, W - 1), (1, 0)) for gb, gc in ((0, 0), (1, 0), (-1, W - 1))
                 for d in (-1, 0, 1, 2)]
    elif level <= 2:
        rules = [(a, b, c) for a in range(-2, 3) for b in range(-2, 3) for c in range(-2 * (W - 1), C)]
    else:
        rules = [(a, b, c) for a in range(-3, 4) for b in range(-3, 4) for c in range(-3 * (W - 1), C)]
    num = [(0, 1)] if level <= 1 else [(al, be) for al in (-1, 0, 1) for be in (-1, 0, 1)]
    return {"lo": rules, "num": num}


DIV_MENU = div_menu(DIV_LEVEL)
MUL_MENU = mul_menu(MUL_LEVEL)
DIV_TRUE = {"a": "D", "b": "M", "op": "sub", "flag": "borrow", "keep": "gated", "nread": "given", "vread": "given",
            "start": "top", "order": "down", "bring": (1, 0, "rep"), "post": 0, "final": "none", "read": (0, "given")}
MUL_TRUE = {"lo": (1, 1, 0), "hi": (1, 1, 1), "num": (0, 1)}   # columns i + j and i + j + 1, numbers by j
SWITCHED = tuple(k for k, v in DIV_MENU.items() if len(v) > 1)  # the division's switches (single options are fixed)
CHAIN_SW = tuple(k for k in ("a", "b", "op", "flag", "keep", "nread") if k in SWITCHED)  # + the gate's two rows
IO_SW = tuple(k for k in ("vread", "start", "order", "bring", "post", "final", "read") if k in SWITCHED)
if IO_MODE == "joint":
    IO_BLOCKS = [("joint", CHAIN_SW + IO_SW)]
elif IO_MODE == "enum":
    IO_BLOCKS = [("io", IO_SW)]
else:
    IO_BLOCKS = [(n, b) for n, b in (("step", tuple(k for k in ("bring", "post") if k in IO_SW)),
                                     ("read", tuple(k for k in IO_SW if k not in ("bring", "post")))) if b]
CHAIN_COMBOS = list(itertools.product(*[range(len(DIV_MENU[n])) for n in CHAIN_SW], range(2), range(2)))
R_NUM = 2 * W - 1      # the multiply's numbers (a number rule places a pair in 0..2W - 2)
N_RULES = len(MUL_MENU["lo"])


def menu_size(menu: dict, names=None) -> int:
    n = 1
    for k in (names or menu):
        n *= len(menu[k])
    return n


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


def row_hard(X: Tensor, b: Tensor, j: Tensor, tabs, visits: Tensor | None = None,
             pvisits: Tensor | None = None) -> Tensor:
    """Row j of X * b on digits: X [..., W] times the digits b [...], row indices j [...], with the snapped
    product cell (flat lo and hi tables [100] indexed a * 10 + b) and the argmax column rules' row maps (tabs[5],
    tabs[6]: the source of every column [W, C], W = nothing): column c of the low-digit number holds the low digit of
    cell src_lo[j, c], the high-digit number likewise; the two are summed by the learned adder. Returns [..., C]. (The
    routed ladder's view: the multiplier has one nonzero digit, so the number rule does not matter.)"""
    Td, Tb, Plo, Phi = tabs[:4]
    idx = X * 10 + b[..., None]
    z = torch.zeros_like(X[..., :1])
    tally(pvisits, idx, 100)
    lo = torch.cat([Plo[idx], z], -1).gather(-1, tabs[5][j])
    hi = torch.cat([Phi[idx], z], -1).gather(-1, tabs[6][j])
    return add_learned(lo, hi, Td, Tb, visits)[0]


def square_hard(Sd: Tensor, tabs, visits: Tensor | None = None, pvisits: Tensor | None = None, check: bool = False,
                full: Tensor | None = None):
    """s^2 on digits Sd [X, W] -> the 2W digits of s^2 [X, 2W]: every pair's (s_i, s_j) low and high digit from the
    snapped product cell, placed by the argmax layout's full map (tabs[7], or `full`: [2, R_NUM, C], number n's column
    c holds pair i W + j, W^2 = nothing); each number's low and high part summed by the learned adder, then the numbers
    one by one in C places (empty numbers skipped). With `check` also, per row, whether every addition of the sum
    passes its own subtraction check (A + B) - B = A with borrow = carry."""
    X, (Td, Tb, Plo, Phi) = Sd.shape[0], tabs[:4]
    fm = tabs[7] if full is None else full
    idx = (Sd[:, :, None] * 10 + Sd[:, None, :]).reshape(X, W * W)          # pair (i, j): cell (s_i, s_j)
    tally(pvisits, idx, 100)
    z = torch.zeros_like(idx[:, :1])
    lo, hi = torch.cat([Plo[idx], z], 1), torch.cat([Phi[idx], z], 1)        # [X, W^2 + 1]
    used = [n for n in range(fm.shape[1]) if bool((fm[:, n] < W * W).any())]
    Ln = lo[:, fm[0, used].flatten()].view(X, len(used), C)
    Hn = hi[:, fm[1, used].flatten()].view(X, len(used), C)
    rows = add_learned(Ln, Hn, Td, Tb, visits)[0]                            # [X, n, C]
    acc = torch.zeros(X, C, dtype=Sd.dtype, device=Sd.device)
    ok = torch.ones(X, dtype=torch.bool, device=Sd.device)
    for n in range(len(used)):
        B = rows[:, n]
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


def move_digits(R: Tensor, k: Tensor) -> Tensor:
    """The digits R [X, L] moved up by k [X] places (down for k < 0), zero-filled."""
    L = R.shape[-1]
    src = torch.arange(L, device=R.device)[None] - k[:, None]
    return torch.where((src >= 0) & (src < L), R.gather(-1, src.clamp(0, L - 1)), torch.zeros_like(R))


def multiples(Nr: Tensor, tabs, visits: Tensor | None = None) -> Tensor:
    """The multiples 1..S-1 of the N register (Nr [X, W] digits) by the machine's own row 0 (the snapped product cell,
    the argmax layout, the learned adder): [X, S - 1, P] (link s's source Mq is the s-th)."""
    X = Nr.shape[0]
    q = torch.arange(1, S, device=Nr.device).repeat(X)
    rows = row_hard(Nr.repeat_interleave(S - 1, 0), q, torch.zeros_like(q), tabs, visits)
    return rows.view(X, S - 1, C)[..., :P]


def n_register(Nd: Tensor, nread: Tensor) -> Tensor:
    """The subtrahend register in P places: N's W read digits as given or reversed (nread [X] code 1)."""
    return F.pad(torch.where(nread[:, None] == 1, Nd.flip(-1), Nd), (0, P - Nd.shape[-1]))


def masked_tally(acc: Tensor | None, idx: Tensor, mask: Tensor, n: int) -> None:
    if acc is not None and bool(mask.any()):
        tally(acc, idx[mask], n)


def zero_flag(Dg: Tensor, Tb: Tensor) -> tuple[Tensor, Tensor]:
    """The table's zero test of the digits Dg [X]: the borrow of d - 1 (1 iff d = 0 with the true table), and the
    table rows."""
    idx = (Dg * 10 + 1) * 2
    return Tb[idx], idx


def chain_hard(Rp: Tensor, M: Tensor, w: dict, Td: Tensor, Tb: Tensor, visits: Tensor | None = None,
               gvisits: Tensor | None = None, Mq: Tensor | None = None, R0: Tensor | None = None,
               dd: Tensor | None = None) -> Tensor:
    """One division step's chain on digits with per-row codes w (a, b: sources; op, flag, keep; g0, g1 [X] bool: the
    gate commits on flag 0 / 1): S - 1 links, each E = A - B (or A + B, the learned adder A - (0 - B)) with the
    table; the gate reads the flag (the operation's final borrow or carry, a zero test of D's or E's lowest digit, or
    A < N); the chain register takes E as `keep` says. Sources: D, Rp, M, the link's multiple Mq [X, S - 1, P], zero,
    the register before the bring-in R0, the incoming digit dd (Mq may also be a function of the link). Returns the
    register [X, P]."""
    D = Rp
    Z = torch.zeros_like(Rp)
    add = w["op"] == 1
    any_add, any_lt = bool(add.any()), bool((w["flag"] == 3).any())
    mq = (lambda s: Z) if Mq is None else Mq if callable(Mq) else (lambda s: Mq[:, s - 1])
    for s in range(1, S):
        srcs = [D, Rp, M, mq(s), Z, Z if R0 is None else R0, Z if dd is None else dd]
        A, Bv = sel(w["a"], srcs), sel(w["b"], srcs)
        if any_add:
            Cn, bc, r1 = sub_hard(Z, Bv, Td, Tb)
            masked_tally(visits, r1, add, 200)
            Bv = torch.where(add[:, None], Cn, Bv)
        E, c, idx = sub_hard(A, Bv, Td, Tb)
        tally(visits, idx, 200)
        fb = torch.where(add, bc - c, c) if any_add else c
        zD, iD = zero_flag(D[:, 0], Tb)
        zE, iE = zero_flag(E[:, 0], Tb)
        flags = [fb, zD, zE]
        masked_tally(visits, iD, w["flag"] == 1, 200)
        masked_tally(visits, iE, w["flag"] == 2, 200)
        if any_lt:
            _, bl, rl = sub_hard(A, M, Td, Tb)
            masked_tally(visits, rl, w["flag"] == 3, 200)
            flags.append(bl)
        fl = sel(w["flag"], flags)
        commit = torch.where(fl == 0, w["g0"], w["g1"])
        take = torch.where(w["keep"] == 1, commit, w["keep"] == 0)
        D = torch.where(take[:, None], E, D)
        tally(gvisits, fl, 2)
    return D


def bring_in(R: Tensor, d: Tensor, w: dict, Td: Tensor, Tb: Tensor, visits: Tensor | None = None) -> Tensor:
    """A step's bring-in on digits: R [X, P] moved by k places, then the digit d [X] written at place p (p = -1: not
    at all), replacing that place's digit or added there by the learned adder (mode 1)."""
    R = move_digits(R, w["k"])
    at = torch.arange(P, device=R.device)[None] == w["p"][:, None]
    out = torch.where(at, d[:, None], R)
    if bool((w["mode"] == 1).any()):
        out = torch.where((w["mode"] == 1)[:, None], add_learned(R, torch.where(at, d[:, None], 0), Td, Tb, visits)[0],
                          out)
    return torch.where((w["p"] >= 0)[:, None], out, R)


def divide_hard(V: Tensor, Nd: Tensor, w: dict, Td: Tensor, Tb: Tensor, visits: Tensor | None = None,
                gvisits: Tensor | None = None, regs: list | None = None, tabs=None, mults: Tensor | None = None):
    """The division of the digits V [X, 2W] by N [X, W] with per-row codes w (MENUS.md): the register starts as
    `start` says, W steps each bring a dividend digit in (order, bring), run the chain and move the register (post);
    then the final correction and the readout. The multiples of N (source Mq) come from `mults` or the machine's own
    rows (tabs). Returns the read W places [X, W]. `regs` collects every step's register after the bring-in."""
    Vp = torch.where(w["vread"][:, None] == 1, V.flip(-1), V)
    Nr = torch.where(w["nread"][:, None] == 1, Nd.flip(-1), Nd)
    M = F.pad(Nr, (0, P - W))
    Mq = None
    if bool(((w["a"] == 3) | (w["b"] == 3)).any()):
        Mq = mults if mults is not None else multiples(Nr, tabs, visits)
    top_, bot = F.pad(Vp[:, W:2 * W], (0, P - W)), F.pad(Vp[:, :W], (0, P - W))
    R = sel(w["start"], [top_, torch.zeros_like(top_), bot, M])
    for t in range(W):
        d = torch.where(w["order"] == 0, Vp[:, W - 1 - t], Vp[:, t])
        R0 = R
        R = bring_in(R, d, w, Td, Tb, visits)
        if regs is not None:
            regs.append(R)
        R = chain_hard(R, M, w, Td, Tb, visits, gvisits, Mq, R0, F.pad(d[:, None], (0, P - 1)))
        R = move_digits(R, w["post"])
    fin = w["final"]
    if bool((fin > 0).any()):
        R2 = R
        if bool((fin == 2).any()):
            R2 = torch.where((fin == 2)[:, None], add_learned(R, top_, Td, Tb, visits)[0], R)
        for rep in range(2 if bool((fin == 3).any()) else 1):
            E, c, idx = sub_hard(R2, M, Td, Tb)
            do = (fin > 0) & ((fin == 3) | (rep == 0))
            masked_tally(visits, idx, do, 200)
            commit = torch.where(c == 0, w["g0"], w["g1"])
            R2 = torch.where((do & commit)[:, None], E, R2)
        R = torch.where((fin > 0)[:, None], R2, R)
    out = torch.where((w["off"] == 1)[:, None], R[:, 1:W + 1], R[:, :W])
    return torch.where((w["rdir"] == 1)[:, None], out.flip(-1), out)


def count_round(C: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor):
    """One round of the countdown C [X, K]: C - 1 with the table and the gate on its final borrow. Returns the
    difference, whether the round commits [X], the final borrow [X] and the table rows."""
    Cn, b, idx = sub_hard(C, F.one_hot(torch.zeros_like(C[:, 0]), C.shape[1]), Td, Tb)   # the digits of 1
    return Cn, Gc[b] == 1, b, idx


def run_hard(Sd: Tensor, C: Tensor, Nd: Tensor, tabs, w: dict, cap: int | None = None, visits: Tensor | None = None,
             gvisits: Tensor | None = None, cvisits: Tensor | None = None, pvisits: Tensor | None = None):
    """The hard machine on digits with per-row division codes w: rounds until no row commits (at most `cap`, by
    default 10^K). A committing row squares and reduces its state and takes the countdown's difference; a holding row
    keeps both. Returns the state [X, W], the input of each row's last committed squaring [X, W] (x if none), the
    number of committed rounds [X] and whether the row has held [X]."""
    Td, Tb, Plo, Phi, Gc = tabs[:5]
    s, prev = Sd.clone(), Sd.clone()
    rounds = torch.zeros_like(Sd[:, 0])
    held = torch.zeros_like(Sd[:, 0], dtype=torch.bool)
    mults = None
    if bool(((w["a"] == 3) | (w["b"] == 3)).any()):                        # N's multiples, once (N stays)
        mults = multiples(torch.where(w["nread"][:, None] == 1, Nd.flip(-1), Nd), tabs, visits)
    for _ in range(10 ** C.shape[1] if cap is None else cap):
        Cn, k, b, idx = count_round(C, Td, Tb, Gc)
        tally(cvisits, idx, 200)
        tally(gvisits, b, 2)
        held = held | ~k
        if not bool(k.any()):
            break
        prev = torch.where(k[:, None], s, prev)
        R = divide_hard(square_hard(s, tabs, visits, pvisits), Nd, w, Td, Tb, visits, gvisits, tabs=tabs, mults=mults)
        s = torch.where(k[:, None], R, s)
        C = torch.where(k[:, None], Cn, C)
        rounds = rounds + k.long()
    return s, prev, rounds, held


def codes(idx: dict) -> dict:
    """Per-row option indices of the division's switches (absent: a single fixed option) and the gate's rows g0, g1
    -> the codes the machine reads (sources, op, flag, keep, nread, vread, start, order, k, p, mode, post, final, off,
    rdir)."""
    dev = idx["g0"].device
    X = idx["g0"].shape[0]
    ix = {k: idx[k] if k in idx else torch.zeros(X, dtype=torch.long, device=dev) for k in DIV_MENU}
    lut = lambda vals: torch.tensor(vals, dtype=torch.long, device=dev)
    out = {"g0": idx["g0"], "g1": idx["g1"]}
    for k in ("a", "b"):
        out[k] = lut([SRC[v] for v in DIV_MENU[k]])[ix[k]]
    for k, cd in CODES.items():
        out[k] = lut([cd[v] for v in DIV_MENU[k]])[ix[k]]
    br = DIV_MENU["bring"]
    out["k"] = lut([v[0] for v in br])[ix["bring"]]
    out["p"] = lut([v[1] for v in br])[ix["bring"]]
    out["mode"] = lut([int(v[2] == "add") for v in br])[ix["bring"]]
    out["post"] = lut(DIV_MENU["post"])[ix["post"]]
    out["off"] = lut([v[0] for v in DIV_MENU["read"]])[ix["read"]]
    out["rdir"] = lut([int(v[1] == "reversed") for v in DIV_MENU["read"]])[ix["read"]]
    return out


def repeat_wiring(w: dict, K: int) -> dict:
    """The wiring w (per-row [X]) for K copies of the rows, copy-major."""
    return {k: v.repeat(K) for k, v in w.items()}


def combo_wiring(w: dict, names: tuple, combos: Tensor, n: int, gate: bool = False) -> dict:
    """The per-row option indices for K = len(combos) copies of n rows (copy-major): the switches `names` (and with
    `gate` the gate's two rows) take each combination's options, every other switch keeps w's (per row, [n])."""
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


# ------------------------------------------------------------------------------------------------ the prediction path
def pick(w: Tensor, opts: list) -> Tensor:
    """A switch on the prediction path: w [n] its straight-through one-hot (values exactly 0 or 1), opts the options'
    digit distributions. With autograd the blend sum_o w_o opts_o (its value is the chosen option exactly, its gradient
    reaches every option's logit); without autograd, as at evaluation, the chosen option itself (the same value)."""
    if not torch.is_grad_enabled():
        return opts[int(w.argmax())]
    return sum(w[i] * o for i, o in enumerate(opts))


def pickl(w: Tensor, fns: list):
    """pick with lazily made options (functions): with autograd every option is made and blended (tuples blended
    element by element); without autograd only the chosen one is made."""
    if not torch.is_grad_enabled() or len(fns) == 1:
        return fns[int(w.argmax())]()
    outs = [f() for f in fns]
    if isinstance(outs[0], tuple):
        return tuple(sum(w[i] * o[t] for i, o in enumerate(outs)) for t in range(len(outs[0])))
    return sum(w[i] * o for i, o in enumerate(outs))


def move_soft(R: Tensor, k: int) -> Tensor:
    """Digit distributions R [X, L, 10] moved up by k places (down for k < 0), filled with exact 0 digits."""
    X, L, _ = R.shape
    if k == 0:
        return R
    if abs(k) >= L:
        return onehot0(X, L, R.device, R.dtype)
    if k > 0:
        return torch.cat([onehot0(X, k, R.device, R.dtype), R[:, :L - k]], 1)
    return torch.cat([R[:, -k:], onehot0(X, -k, R.device, R.dtype)], 1)


def zero_soft(Dg: Tensor, TDC: Tensor, used: Tensor | None = None) -> Tensor:
    """The table's zero test on a digit distribution Dg [X, 10]: the borrow distribution of d - 1 [X, 2]."""
    one = torch.ones(Dg.shape[0], 1, dtype=torch.long, device=Dg.device)
    return sub_soft(Dg[:, None], F.one_hot(one, 10).to(Dg.dtype), TDC, used)[1][:, -1]


def square_ste(Sd: Tensor, TDC: Tensor, PROD: Tensor, FULL: Tensor, used: Tensor | None = None,
               pused: Tensor | None = None):
    """s^2 on one-hot digits Sd [X, W, 10] with the straight-through product cell PROD [100, 20], the
    straight-through full map FULL [2, R_NUM, C, W^2 + 1] (number n's column c: which pair's digit, the last
    nothing) and the subtraction table TDC: square_hard's datapath. Returns [X, 2W, 10]."""
    X = Sd.shape[0]
    cell = (Sd[:, :, None, :, None] * Sd[:, None, :, None, :]).reshape(X, W * W, 100)   # pair (i, j): (s_i, s_j)
    if pused is not None:
        pused += cell.detach().sum((0, 1))
    lohi = cell @ PROD
    z = onehot0(X, 1, Sd.device, Sd.dtype)
    lo = torch.cat([lohi[..., :10], z], 1)                                  # [X, W^2 + 1, 10]
    hi = torch.cat([lohi[..., 10:], z], 1)
    nz = [n for n in range(R_NUM) if bool((FULL[:, n, :, :W * W] != 0).any())]
    Ln = torch.einsum("ncp,xpk->xnck", FULL[0, nz], lo).reshape(X * len(nz), C, 10)
    Hn = torch.einsum("ncp,xpk->xnck", FULL[1, nz], hi).reshape(X * len(nz), C, 10)
    rows = damp(add_soft_learned(Ln, Hn, TDC, used)[0]).view(X, len(nz), C, 10)
    acc = onehot0(X, C, Sd.device, Sd.dtype)
    for n in range(len(nz)):
        acc = damp(add_soft_learned(acc, rows[:, n], TDC, used)[0])
    return acc[:, :2 * W]


def mult_ste(Nr: Tensor, TDC: Tensor, PROD: Tensor, ROW0: Tensor, used: Tensor | None = None) -> Tensor:
    """The multiples 1..S-1 of the N register Nr [X, W, 10] by the prediction path's own row 0 (straight-through cell,
    the straight-through row-0 maps ROW0 [2, C, W + 1], the learned adder): [X, S - 1, P, 10]."""
    X = Nr.shape[0]
    out = []
    for q in range(1, S):
        cell = (Nr[..., :, None] * F.one_hot(torch.tensor(q, device=Nr.device), 10).to(Nr.dtype)).reshape(X, W, 100)
        lohi = cell @ PROD
        z = onehot0(X, 1, Nr.device, Nr.dtype)
        Lc = torch.einsum("ci,xik->xck", ROW0[0], torch.cat([lohi[..., :10], z], 1))
        Hc = torch.einsum("ci,xik->xck", ROW0[1], torch.cat([lohi[..., 10:], z], 1))
        out.append(damp(add_soft_learned(Lc, Hc, TDC, used)[0])[:, :P])
    return torch.stack(out, 1)


def chain_ste(Rp: Tensor, M: Tensor, Mq, R0: Tensor, dd: Tensor, TDC: Tensor, G: Tensor, SW: dict,
              used: Tensor | None = None, gused: Tensor | None = None) -> Tensor:
    """chain_hard on one-hot digits [X, P, 10] with the straight-through switches SW (and the straight-through gate G
    [2, 2]: flag -> hold, commit). Mq: a function of the link (its multiple [X, P, 10])."""
    D = Rp
    Z = onehot0(Rp.shape[0], P, Rp.device, Rp.dtype)
    for s in range(1, S):
        Dc = D
        srcs = [lambda: Dc, lambda: Rp, lambda: M, lambda: Mq(s), lambda: Z, lambda: R0, lambda: dd]
        srcs = [srcs[SRC[v]] for v in DIV_MENU["a"]], [srcs[SRC[v]] for v in DIV_MENU["b"]]
        A, B = pickl(SW["a"], srcs[0]), pickl(SW["b"], srcs[1])

        def sub_():
            E_, bor = sub_soft(A, B, TDC, used)
            return E_, bor[:, -1]

        def add_():
            Cn, bc = sub_soft(Z, B, TDC, used)
            E_, by = sub_soft(A, Cn, TDC, used)
            pc = bc[:, -1, 1] * by[:, -1, 0]                                # P(carry = 1)
            return E_, torch.stack([1 - pc, pc], -1)
        ops = {"sub": sub_, "add": add_}
        E, fb = pickl(SW["op"], [ops[v] for v in DIV_MENU["op"]])
        flags = {"borrow": lambda: fb, "zD": lambda: zero_soft(Dc[:, 0], TDC, used),
                 "zE": lambda: zero_soft(E[:, 0], TDC, used), "lt": lambda: sub_soft(A, M, TDC, used)[1][:, -1]}
        fl = pickl(SW["flag"], [flags[v] for v in DIV_MENU["flag"]])
        if gused is not None:
            gused += fl.detach().sum(0)
        k = (fl * G[:, 1]).sum(-1)                                         # 1: the gate commits
        keeps = {"always": lambda: torch.ones_like(k), "gated": lambda: k, "never": lambda: torch.zeros_like(k)}
        kk = pickl(SW["keep"], [keeps[v] for v in DIV_MENU["keep"]])[:, None, None]
        D = damp(kk * E + (1 - kk) * Dc)
    return D


def divide_ste(V: Tensor, M: Tensor, Mq, TDC: Tensor, G: Tensor, SW: dict, used: Tensor | None = None,
               gused: Tensor | None = None) -> Tensor:
    """The division on one-hot digits V [X, 2W, 10] by the N register M [X, P, 10] with the straight-through switches
    SW (divide_hard's datapath). Returns the read W places [X, W, 10]."""
    X = V.shape[0]
    Vp = pickl(SW["vread"], [{"given": lambda: V, "reversed": lambda: V.flip(1)}[v] for v in DIV_MENU["vread"]])
    top_, bot = zpad(Vp[:, W:], 0, P - W), zpad(Vp[:, :W], 0, P - W)
    starts = {"top": lambda: top_, "zero": lambda: onehot0(X, P, V.device, V.dtype), "bottom": lambda: bot,
              "N": lambda: M}
    R = pickl(SW["start"], [starts[v] for v in DIV_MENU["start"]])
    for t in range(W):
        d = pickl(SW["order"], [{"down": lambda: Vp[:, W - 1 - t], "up": lambda: Vp[:, t]}[v] for v in DIV_MENU["order"]])
        dd = zpad(d[:, None], 0, P - 1)
        R0 = R

        def bring(k, p, mode, R=R, d=d):
            Rm = move_soft(R, k)
            if p < 0:
                return Rm
            if mode == "rep":
                return torch.cat([Rm[:, :p], d[:, None], Rm[:, p + 1:]], 1)
            return add_soft_learned(Rm, zpad(d[:, None], p, P - 1 - p), TDC, used)[0]
        Rp = pickl(SW["bring"], [lambda o=o: bring(*o) for o in DIV_MENU["bring"]])
        D = chain_ste(Rp, M, Mq, R0, dd, TDC, G, SW, used, gused)
        R = pickl(SW["post"], [lambda m=m, D=D: move_soft(D, m) for m in DIV_MENU["post"]])

    def csub(R2):
        E, bor = sub_soft(R2, M, TDC, used)
        k = (bor[:, -1] * G[:, 1]).sum(-1)[:, None, None]
        return damp(k * E + (1 - k) * R2)
    finals = {"none": lambda: R, "csub": lambda: csub(R),
              "addhigh": lambda: csub(add_soft_learned(R, top_, TDC, used)[0]), "csub2": lambda: csub(csub(R))}
    R = pickl(SW["final"], [finals[v] for v in DIV_MENU["final"]])
    reads = [lambda o=o, dr=dr: (R[:, o:o + W].flip(1) if dr == "reversed" else R[:, o:o + W])
             for o, dr in DIV_MENU["read"]]
    return pickl(SW["read"], reads)


def run_ste(Sd: Tensor, Cnt: Tensor, N1: Tensor, TDC: Tensor, PROD: Tensor, G: Tensor, SW: dict, MAPS: tuple,
            used: Tensor | None = None, cused: Tensor | None = None, pused: Tensor | None = None,
            gused: Tensor | None = None) -> Tensor:
    """run_hard on one-hot digits: the countdown C [X, K, 10] takes C - 1 with the table each round and the gate on
    its final borrow decides; a committing row squares and reduces its state. The rounds end when no row commits.
    MAPS: the straight-through row-0 maps [2, C, W + 1] (N's multiples) and full map [2, R_NUM, C, W^2 + 1]."""
    one = onehot0(Cnt.shape[0], Cnt.shape[1], Cnt.device, Cnt.dtype)       # the digits of 1
    one[:, 0] = F.one_hot(torch.tensor(1, device=Cnt.device), 10).to(Cnt.dtype)
    Nr = pickl(SW["nread"], [{"given": lambda: N1, "reversed": lambda: N1.flip(1)}[v] for v in DIV_MENU["nread"]])
    M = zpad(Nr, 0, P - W)
    cache = {}

    def Mq(s):                                                              # N's multiples, made once when needed
        if "m" not in cache:
            cache["m"] = mult_ste(Nr, TDC, PROD, MAPS[0], used)
        return cache["m"][:, s - 1]
    s = Sd
    for _ in range(10 ** Cnt.shape[1]):
        Cn, bor = sub_soft(Cnt, one, TDC, cused)
        b = bor[:, -1]
        if gused is not None:
            gused += b.detach().sum(0)
        k = (b * G[:, 1]).sum(-1)
        if not bool((k > 0.5).any()):
            break
        R = divide_ste(square_ste(s, TDC, PROD, MAPS[1], used, pused), M, Mq, TDC, G, SW, used, gused)
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
        self.facts_streak = 0                 # batches in a row passing the borrow facts (x - N borrows, N - x not)
        self.gated_on: int | None = None      # the step the digit lessons started
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
        self.io_parts = ""


def rule_maps(device) -> Tensor:
    """The row maps of every column rule (a, b, c0) of MUL_MENU (the routed ladder's view): row j, column c -> the
    source pair i whose column a i + b j + c0 is c (the first such i), W = nothing [N_RULES, W, C]."""
    out = torch.full((N_RULES, W, C), W, dtype=torch.long)
    for r, (a, b, c0) in enumerate(MUL_MENU["lo"]):
        for j in range(W):
            for i in reversed(range(W)):
                col = a * i + b * j + c0
                if 0 <= col < C:
                    out[r, j, col] = i
    return out.to(device)


def num_index(al: int, be: int) -> Tensor:
    """The number of every pair under the number rule (alpha, beta): alpha i + beta j + gamma [W (i), W (j)]."""
    i = torch.arange(W)[:, None]
    j = torch.arange(W)[None, :]
    return al * i + be * j + (W - 1) * (al < 0) + (W - 1) * (be < 0)


def full_map(lo: tuple, hi: tuple, num: tuple) -> Tensor:
    """The squaring's full map under the column rules lo, hi and the number rule num: number n, column c -> the pair
    i W + j whose digit is placed there, W^2 = nothing [2, R_NUM, C] (a pair is dropped where its column is outside
    the C columns or its place is taken by an earlier pair, pairs in order j, then i)."""
    out = torch.full((2, R_NUM, C), W * W, dtype=torch.long)
    nidx = num_index(*num)
    for kind, (a, b, c0) in enumerate((lo, hi)):
        for j in range(W):
            for i in range(W):
                col, n = a * i + b * j + c0, int(nidx[i, j])
                if 0 <= col < C and out[kind, n, col] == W * W:
                    out[kind, n, col] = i * W + j
    return out


def true_sources(device) -> tuple[Tensor, Tensor]:
    """The true layout's row maps (the designed start and the reports only): column c of row j holds pair c - j's low
    digit and pair c - j - 1's high digit (W: nothing) [W, C]."""
    j = torch.arange(W, device=device)[:, None]
    c = torch.arange(C, device=device)[None, :]
    lo = torch.where((c - j >= 0) & (c - j < W), c - j, W)
    hi = torch.where((c - j - 1 >= 0) & (c - j - 1 < W), c - j - 1, W)
    return lo, hi


def classify(names: dict) -> dict:
    """Which known algorithm a wiring is, part by part (the reports' labels)."""
    key = tuple(names[k] for k in ("a", "b", "op", "flag", "keep"))
    chain = {("D", "M", "sub", "borrow", "gated"): "repeated subtraction",
             ("D", "M", "sub", "lt", "gated"): "repeated subtraction",          # D < N is D - N's borrow
             ("Rp", "Mq", "sub", "borrow", "gated"): "trial multiples"}.get(key, "other")
    num = {(0, 1): "rows by multiplier digit", (1, 0): "rows by multiplicand digit", (1, -1): "diagonals i - j",
           (-1, 1): "diagonals j - i", (0, -1): "rows by multiplier digit, reversed",
           (-1, 0): "rows by multiplicand digit, reversed"}.get(tuple(names["num"]), "other")
    return {"chain": chain, "numbers": num}


class LearnedWiringIO(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        V = spec.vocab_size
        self.config = Config(V, spec.max_seq_len)
        self.V, self.END, self.OUT = V, V, V + 1                             # virtual tokens: the end, beyond it
        self.a = nn.Parameter(INIT * torch.randn(10, 10, 2, 10))           # subtractor: (r, m, borrow in) -> digit
        self.b = nn.Parameter(INIT * torch.randn(10, 10, 2, 2))            # ... -> borrow out
        self.p_lo = nn.Parameter(INIT * torch.randn(10, 10, 10))           # product cell: (a, b) -> lo digit
        self.p_hi = nn.Parameter(INIT * torch.randn(10, 10, 10))           # ... -> hi digit
        self.g = nn.Parameter(INIT * torch.randn(2, 2))                    # commit gate: flag -> (hold, commit)
        self.sw = nn.ParameterDict({k: nn.Parameter(INIT * torch.randn(len(DIV_MENU[k]))) for k in SWITCHED})
        self.lay = nn.Parameter(INIT * torch.randn(2, N_RULES))            # one weight per whole column rule (lo, hi)
        self.num = (nn.Parameter(INIT * torch.randn(len(MUL_MENU["num"]))) if len(MUL_MENU["num"]) > 1
                    else None)                                              # the number rule
        self.maps = None                                                    # rule_maps, made on the first use
        self.fmaps: dict = {}                                               # full maps, by (lo, hi, num) rule
        self.dg = nn.Parameter(INIT * torch.randn(V))                      # token -> is a digit
        self.u = nn.Parameter(INIT * torch.randn(len(HYP), V))             # per reading hypothesis: token scores
        self.wc = nn.Parameter(INIT * torch.randn(10, V))                  # writer: value -> token
        self.anc = nn.Parameter(INIT * torch.randn(4, V + 1))              # field anchor: a token id, or the end
        self.strd = nn.Parameter(INIT * torch.randn(4, 2))                 # field step: +1 | -1 (shape "step")
        self.shape = nn.Parameter(INIT * torch.randn(len(SHAPES)))         # the pointer's shape, all four fields
        self.off = nn.Parameter(INIT * torch.randn(4, W, 2 * OFF + 1))     # shape "offsets": slot j's offset
        self.LPM = spec.max_seq_len + 1 + 2 * (W + 1)                      # positions of the padded prompt
        self.absp = nn.Parameter(INIT * torch.randn(4, W, self.LPM))       # shape "absolute": slot j's position
        self.ctl = Control()
        if MODE == "designed":
            self.designed()

    @torch.no_grad()
    def designed(self) -> None:
        """Every table and switch at the truth (the plumbing check only): the subtractor, the product cell, the gate
        (commit unless the flag is 1), the division's wiring DIV_TRUE, the layout MUL_TRUE; N read after its marker, x
        before T's marker, T and the answer at the end, reading leftwards; digits 7..16 valued 0..9."""
        hot = lambda t, n: 10.0 * F.one_hot(t, n).float() - 5.0
        Td, Tb = exact_tables("cpu")
        Plo, Phi = exact_products("cpu")
        self.a.copy_(hot(Td, 10).view(10, 10, 2, 10))
        self.b.copy_(hot(Tb, 2).view(10, 10, 2, 2))
        self.p_lo.copy_(hot(Plo, 10).view(10, 10, 10))
        self.p_hi.copy_(hot(Phi, 10).view(10, 10, 10))
        self.g.copy_(hot(torch.tensor([1, 0]), 2))
        for k in SWITCHED:
            self.sw[k].copy_(hot(torch.tensor(DIV_MENU[k].index(DIV_TRUE[k])), len(DIV_MENU[k])))
        for kind, key in enumerate(("lo", "hi")):
            self.lay[kind].copy_(hot(torch.tensor(MUL_MENU["lo"].index(MUL_TRUE[key])), N_RULES))
        if self.num is not None:
            self.num.copy_(hot(torch.tensor(MUL_MENU["num"].index(MUL_TRUE["num"])), len(MUL_MENU["num"])))
        DIGIT, MARK_N, MARK_T = 7, 2, 4
        self.dg.fill_(-5.0)
        self.dg[DIGIT:DIGIT + 10] = 5.0
        self.u.copy_(torch.arange(self.V, dtype=torch.float)[None].expand_as(self.u))
        self.wc.fill_(-5.0)
        self.wc[torch.arange(10), DIGIT + torch.arange(10)] = 5.0
        self.anc.fill_(-5.0)
        self.strd.fill_(-5.0)
        self.shape.copy_(hot(torch.tensor(0), len(SHAPES)))                # the step shape
        for f, (tok, step) in enumerate([(MARK_N, 0), (MARK_T, 1), (self.END, 1), (self.END, 1)]):
            self.anc[f, tok] = 5.0
            self.strd[f, step] = 5.0

    def zero(self) -> Tensor:
        return sum(p.sum() for p in self.parameters()) * 0.0

    # ------------------------------------------------------------------ the tables and the wiring
    def tables(self):
        """The snapped tables: subtractor digit and borrow [200], product lo and hi [100], gate [2] (1 = commit), the
        argmax column rules' row maps (low and high [W, C]) and the argmax layout's full map [2, R_NUM, C]."""
        lo, hi = self.sources()
        return (self.a.argmax(-1).flatten(), self.b.argmax(-1).flatten(), self.p_lo.argmax(-1).flatten(),
                self.p_hi.argmax(-1).flatten(), self.g.argmax(-1), lo, hi, self.full())

    def rule_maps(self) -> Tensor:
        if self.maps is None or self.maps.device != self.a.device:
            self.maps = rule_maps(self.a.device)
        return self.maps

    def choice(self, k: str) -> int:
        """The argmax option of the division's switch k (0: a single fixed option)."""
        return int(self.sw[k].argmax()) if k in SWITCHED else 0

    def num_choice(self) -> int:
        return int(self.num.argmax()) if self.num is not None else 0

    def sources(self) -> tuple:
        """The argmax column rules' row maps: the source pair of every column of every row [W, C] (W: nothing), low
        and high."""
        m = self.rule_maps()
        return m[int(self.lay[0].argmax())], m[int(self.lay[1].argmax())]

    def full(self, num: int | None = None) -> Tensor:
        """The full map of the argmax column rules under the argmax number rule (or `num`) [2, R_NUM, C]."""
        key = (int(self.lay[0].argmax()), int(self.lay[1].argmax()), self.num_choice() if num is None else num)
        if key not in self.fmaps:
            rules = MUL_MENU["lo"]
            self.fmaps[key] = full_map(rules[key[0]], rules[key[1]], MUL_MENU["num"][key[2]])
        return self.fmaps[key].to(self.a.device)

    def source_probs(self) -> tuple:
        """The soft row maps: every column's distribution over its sources [W, C, W + 1], low and high (the rules'
        one-hot row maps weighted by the rule switch's probabilities)."""
        m = F.one_hot(self.rule_maps(), W + 1).float()
        return tuple(torch.einsum("r,rjxs->jxs", self.lay[k].float().softmax(-1), m) for k in range(2))

    def ste_sources(self) -> tuple:
        """The layout on the prediction path: the row-0 maps [2, C, W + 1] (N's multiples) and the full map
        [2, R_NUM, C, W^2 + 1], straight-through (the value is the argmax layout exactly; the gradient: the soft row
        maps, placed in numbers by the soft number rule)."""
        dev = self.a.device
        soft = self.source_probs()
        hard = [F.one_hot(h, W + 1).float() for h in self.sources()]
        row0 = torch.stack([ste(hard[k][0], soft[k][0]) for k in range(2)])
        full = F.one_hot(self.full(), W * W + 1).float()
        if not torch.is_grad_enabled():
            return row0, full
        nums = torch.stack([F.one_hot(num_index(*r), R_NUM).float() for r in MUL_MENU["num"]]).to(dev)
        pn = self.num.float().softmax(-1) if self.num is not None else torch.ones(1, device=dev)
        NUMS = torch.einsum("r,rijn->ijn", pn, nums)
        sf = []
        for k in range(2):
            s_ = torch.einsum("jci,ijn->ncij", soft[k][:, :, :W], NUMS).reshape(R_NUM, C, W * W)
            sf.append(torch.cat([s_, 1 - s_.sum(-1, keepdim=True)], -1))
        return row0, ste(full, torch.stack(sf))

    def layout_right(self) -> int:
        """How many of the 2 W C row-map columns hold their true source (the reports only)."""
        lo, hi = true_sources(self.a.device)
        slo, shi = self.sources()
        return int((slo == lo).sum() + (shi == hi).sum())

    def layout_names(self) -> str:
        rules = MUL_MENU["lo"]
        return (f"lo={rules[int(self.lay[0].argmax())]} hi={rules[int(self.lay[1].argmax())]} "
                f"num={MUL_MENU['num'][self.num_choice()]} right={self.layout_right()}/{2 * W * C}").replace(" ", "")

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
        [100, 20] (lo, hi), gate [2, 2] (flag -> hold, commit). Values are the snapped tables exactly."""
        hot = lambda p: ste(F.one_hot(p.argmax(-1), p.shape[-1]).float(), p.float().softmax(-1))
        T = torch.cat([hot(self.a), hot(self.b)], -1).permute(0, 2, 1, 3).reshape(20, 120)
        return T, torch.cat([hot(self.p_lo), hot(self.p_hi)], -1).reshape(100, 20), hot(self.g)

    def ste_switches(self) -> dict:
        """The division's switches on the prediction path: straight-through one-hots [n] (values exactly 0 or 1); a
        single fixed option is the constant [1]."""
        out = {k: torch.ones(1, device=self.a.device) for k in DIV_MENU}
        out.update({k: ste(F.one_hot(v.argmax(), v.shape[0]).float(), v.float().softmax(-1)) for k, v in self.sw.items()})
        return out

    def wiring(self, X: int, device) -> dict:
        """The argmax wiring, per row [X]: every switch's option index and the gate's two rows (commit on flag 0 / 1)."""
        w = {k: torch.full((X,), int(v.argmax()), dtype=torch.long, device=device) for k, v in self.sw.items()}
        gc = self.g.argmax(-1) == 1
        w["g0"], w["g1"] = gc[0].expand(X).clone(), gc[1].expand(X).clone()
        return w

    def wcodes(self, X: int, device) -> dict:
        return codes(self.wiring(X, device))

    def names(self) -> dict:
        out = {k: DIV_MENU[k][self.choice(k)] for k in DIV_MENU}
        rules = MUL_MENU["lo"]
        out.update(lo=rules[int(self.lay[0].argmax())], hi=rules[int(self.lay[1].argmax())],
                   num=MUL_MENU["num"][self.num_choice()])
        return out

    def wiring_true(self) -> bool:
        return all(self.choice(k) == DIV_MENU[k].index(DIV_TRUE[k]) for k in DIV_MENU)

    def wiring_names(self) -> str:
        gate = "".join("C" if int(c) == 1 else "H" for c in self.g.argmax(-1))
        return (" ".join(f"{k}={DIV_MENU[k][self.choice(k)]}".replace(" ", "") for k in DIV_MENU if k in SWITCHED)
                + f" gate={gate}")

    def combo_logw(self, names: tuple, combos: Tensor, gate: bool) -> Tensor:
        """log of the factorized mixture's weight of every combination combos [K, len(names) (+ 2 gate rows)]: the
        product of the switches' probabilities (and of the gate's rows)."""
        lw = 0
        for i, n in enumerate(names):
            lw = lw + self.sw[n].float().log_softmax(-1)[combos[:, i]]
        if gate:
            lg = self.g.float().log_softmax(-1)                             # [flag, (hold, commit)]
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

    def shape_logp(self) -> Tensor:
        return self.shape.float().log_softmax(-1)

    def hard_pos(self, f: int, e: Tensor, end: Tensor) -> Tensor:
        """Field f's snapped pointer: its slots' positions [X, n] under the argmax shape and its argmax parameters."""
        n, Lp = SLOTS[f], e.shape[1]
        m = int(self.shape.argmax())
        j = torch.arange(n, device=e.device)
        if m == 0:
            q = self.anchor_logits(f, e, end).argmax(-1)
            s = 1 if int(self.strd[STRD_OF[f]].argmax()) == 0 else -1
            pos = q[:, None] + (j[None] + 1) * s
        elif m == 1:
            q = self.anchor_logits(f, e, end).argmax(-1)
            pos = q[:, None] + (self.off[f, :n].argmax(-1) - OFF)[None]
        else:
            pos = self.absp[f, :n, :Lp].argmax(-1)[None].expand(e.shape[0], n)
        return pos.clamp(0, Lp - 1)

    def shape_probs(self, f: int, e: Tensor, end: Tensor, pos: Tensor, length: Tensor) -> Tensor:
        """The probability that field f reads the positions pos [X, K, W + 1] (reading order) as a whole number of
        length [X, K] digits under each shape: slots 0..len-1 on pos[..., :len] and, when the field has more slots,
        slot len on pos[..., len] (right past the number, where the read stops). Returns [X, K, 3]."""
        X, K, _ = pos.shape
        n, Lp = SLOTS[f], e.shape[1]
        dev = e.device
        A = self.anchor(f, e, end)                                           # [X, Lp]
        j = torch.arange(n, device=dev)
        use = (j[None, None] <= length[..., None]).float()                   # [X, K, n]
        pos = pos[..., :n].clamp(0, Lp - 1)
        d = (pos[..., 1] - pos[..., 0]) if n > 1 else torch.ones_like(pos[..., 0])
        sl = self.strd[STRD_OF[f]].float().softmax(-1)
        step = A.gather(1, (pos[..., 0] - d).clamp(0, Lp - 1)) * torch.where(d > 0, sl[0], sl[1])
        lo = self.off[f, :n].float().log_softmax(-1)                         # [n, 2 OFF + 1]
        rel = pos[..., None] - torch.arange(Lp, device=dev)                  # [X, K, n, Lp]: slot j relative to q
        inr = (rel >= -OFF) & (rel <= OFF)
        lp = lo[j[None, None, :, None].expand_as(rel), (rel + OFF).clamp(0, 2 * OFF)].masked_fill(~inr, -1e4)
        offs = (A.clamp_min(1e-30).log()[:, None] + (lp * use[..., None]).sum(2)).logsumexp(-1).exp()
        la = self.absp[f, :n, :Lp].float().log_softmax(-1)                   # [n, Lp]
        absl = (la[j[None, None].expand(X, K, n), pos] * use).sum(-1).exp()
        return torch.stack([step, offs, absl], -1)

    def shape_rho(self, fields: tuple, e: Tensor, end: Tensor, si: Tensor, ti: Tensor, length: Tensor,
                  hold_shape: bool = False) -> Tensor:
        """P(field f reads run r whole in direction d) [X, len(fields), 3 (runs), 2 (directions: from the run's left
        end, from its right end)], the shapes mixed by the shape switch (held fixed with hold_shape: the lesson then
        teaches each shape's own pointer and leaves the choice of shape to the reading lessons)."""
        X = e.shape[0]
        j = torch.arange(W + 1, device=e.device)
        pos = torch.stack([si[..., None] + j, ti[..., None] - j], 2).view(X, 6, W + 1)
        ln = length[..., None].expand(X, 3, 2).reshape(X, 6)
        w = self.shape_logp().exp()
        w = w.detach() if hold_shape else w
        out = []
        for f in fields:
            fits = (length <= SLOTS[f])[..., None]
            out.append((self.shape_probs(f, e, end, pos, ln) * w).sum(-1).view(X, 3, 2) * fits)
        return torch.stack(out, 1)

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
        """The whole run field f's snapped pointer reads: its index [X], its direction [X] (0: from the run's left end)
        and whether it is one [X]: slots 0..len-1 on the run's digits in reading order and, if the field has more
        slots, slot len right past the run in that order (where the read stops; it also tells the direction of a
        one-digit run)."""
        X = e.shape[0]
        pos = self.hard_pos(f, e, end)                                       # [X, n]
        n = SLOTS[f]
        j = torch.arange(n, device=e.device)
        exp = torch.stack([si[..., None] + j, ti[..., None] - j], 2)        # [X, 3, 2, n]
        use = j <= length[..., None, None]                                   # slots 0..len (len < n: the stop slot)
        on = ((pos[:, None, None] == exp) | ~use).all(-1)
        match = (on & (length <= n)[..., None] & ok[:, None, None]).view(X, 6)
        idx = match.float().argmax(1)
        return idx // 2, idx % 2, match.any(1)

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
        pos = self.hard_pos(f, e, end)
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
        pos = self.hard_pos(3, e, end)
        t = labe.gather(1, pos)
        live = self.isdig(t).long().cumprod(-1).bool()
        return torch.where(live, code.gather(1, t.clamp(max=self.V - 1)), 0)

    def reading_weights(self, e: Tensor, end: Tensor) -> Tensor:
        """Per row, the soft weight of each reading hypothesis [X, H]: the N and x pointers' probabilities of reading
        that hypothesis's whole runs (io_lessons' mixture), normalized. The gradient side of the selected reading."""
        _, si, ti, length = self.runs(e, end)
        rho = self.shape_rho((0, 1), e, end, si, ti, length)                               # [X, field, run, dir]
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
        s = run_hard(Xd, Cd, Nd, self.tables(), self.wcodes(rows, dev), None, used, gused, cused, pused)[0]
        for acc, n in ((ctl.used, used), (ctl.used_count, cused), (ctl.pused, pused), (ctl.used_gate, gused)):
            acc += n
        pos = self.hard_pos(3, e, end)
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
        steps = [hot(self.strd[STRD_OF[f]].float()) for f in range(4)]      # [2] each: +1, -1
        H = len(HYP)
        Wh = ste(F.one_hot(k.clamp(min=0), H).float(), self.reading_weights(e, end))
        code = torch.einsum("xh,hvk->xvk", Wh, self.codes_ste())            # the selected reading's code [X, V, 10]

        shp = hot(self.shape.float())                                        # [3] the pointer's shape

        def slot(f: int, j: int) -> Tensor:                                   # field f's slot j: a position [X, Lp]
            off = hot(self.off[f, j].float())                                 # [2 OFF + 1]
            ab = hot(self.absp[f, j, :Lp].float())[None].expand(X, Lp)       # [X, Lp]
            opts = [lambda: steps[f][0] * shift(anchors[f], j + 1) + steps[f][1] * shift(anchors[f], -(j + 1)),
                    lambda: (pick(off, [shift(anchors[f], o - OFF) for o in range(2 * OFF + 1)])
                             if torch.is_grad_enabled() else shift(anchors[f], int(off.argmax()) - OFF)),
                    lambda: ab]
            if not torch.is_grad_enabled():
                return opts[int(shp.argmax())]()
            return sum(shp[m] * opts[m]() for m in range(len(SHAPES)))

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

    def num_lesson(self, Xn: Tensor, tabs, train: bool):
        """The number rule by distributivity: X * X under every number rule (the argmax column rules) must equal the
        sum of its single-digit rows (row j = X times X_j by the routed ladder's view, summed one by one by the learned
        adder). Every number rule runs on the hard machine; the loss is the mixture's (when `train`). Returns the loss
        and whether the argmax number rule passes on every row."""
        n = min(NUM_ROWS, Xn.shape[0])
        X = Xn[:n]
        Td, Tb = tabs[0], tabs[1]
        nopts = len(MUL_MENU["num"])
        with torch.no_grad():
            js = torch.arange(W, device=X.device)[None].expand(n, W)
            rows = row_hard(X[:, None, :].expand(n, W, W), X, js, tabs)    # [n, W (j), C]
            ref = torch.zeros(n, C, dtype=X.dtype, device=X.device)
            for j in range(W):
                ref = add_learned(ref, rows[:, j], Td, Tb)[0]
            verdict = torch.stack([(square_hard(X, tabs, full=self.full(r)) == ref[:, :2 * W]).all(1)
                                   for r in range(nopts)]) if (train and nopts > 1) else None
            ok = bool((square_hard(X, tabs) == ref[:, :2 * W]).all())
        if verdict is None:
            return None, ok
        return mixture_nll(self.num.float().log_softmax(-1), verdict).mean(), ok

    def chain_lesson(self, Xd: Tensor, Nd: Tensor, rowsN: Tensor, verified: Tensor, tabs, train: bool):
        """The division ladder (one division step at a time): from the register x + N * q, q = 0..9 (the ladder's own
        row N * q added to x by the learned adder), one step's chain must give back x. Rows whose N rungs are not all
        verified are left out. Every combination of the chain's switches and the gate's rows runs the step on the hard
        machine; the loss is the factorized mixture's (when `train`). The sources R and d are the register's upper
        places and its lowest digit (what the true bring-in made it from); the multiples of N come from the machine's
        own rows. Returns the loss, whether the argmax chain passes, and the registers (for the step law)."""
        Td, Tb = tabs[0], tabs[1]
        dev = Xd.device
        n = min(CHAIN_ROWS, Xd.shape[0])
        with torch.no_grad():
            keep = verified[:n].all(1)
            Xs, Ns, rows = Xd[:n][keep], Nd[:n][keep], rowsN[:n][keep]        # rows [m, 10, C]: N * q (row 0)
            m = Xs.shape[0]
            if m == 0:
                return None, 0.0, None
            reg = add_learned(F.pad(Xs, (0, C - W))[:, None].expand(m, 10, C), rows, Td, Tb)[0][..., :P]
            reg = reg.reshape(m * 10, P)
            Nq = Ns.repeat_interleave(10, 0)
            xq = F.pad(Xs, (0, P - W)).repeat_interleave(10, 0)
            mq = None
            if "Mq" in DIV_MENU["a"] + DIV_MENU["b"]:
                mq = (multiples(Nq, tabs), multiples(Nq.flip(-1), tabs))      # N's multiples, as read and reversed
            regs = {"reg": reg, "Nq": Nq, "xq": xq, "mq": mq}
            ca = self.wcodes(m * 10, dev)
            Ra = self.chain_run(reg, ca, regs, 1, tabs, self.ctl.gvisits)
            div_ok = float((Ra == xq).all(1).float().mean())
            if not train:
                return None, div_ok, regs
            combos = torch.tensor(CHAIN_COMBOS, device=dev)
            Kc = combos.shape[0]
            wk = codes(combo_wiring(self.wiring(m * 10, dev), CHAIN_SW, combos, m * 10, gate=True))
            verdict = (self.chain_run(reg.repeat(Kc, 1), wk, regs, Kc, tabs) == xq.repeat(Kc, 1)).all(1)
            verdict = verdict.view(Kc, m * 10)
        lw = self.combo_logw(CHAIN_SW, combos, gate=True)
        return mixture_nll(lw, verdict).mean(), div_ok, regs

    def chain_run(self, reg: Tensor, w: dict, regs: dict, Kc: int, tabs, gvisits: Tensor | None = None,
                  bring: bool = False) -> Tensor:
        """One step on the division ladder's registers (Kc copies, copy-major) with per-row codes w: the chain alone
        from the register reg, or (bring) the whole step from its upper places with its lowest digit coming in (the
        bring-in, the chain, the post move)."""
        Td, Tb = tabs[0], tabs[1]
        Nq = regs["Nq"].repeat(Kc, 1)
        R0 = F.pad(reg[:, 1:], (0, 1))                                      # the register's upper places
        d = reg[:, 0]
        mq = None
        if regs["mq"] is not None:
            rev = (w["nread"] == 1)[:, None, None]
            mq = lambda s: torch.where(rev[:, 0], regs["mq"][1][:, s - 1].repeat(Kc, 1),
                                       regs["mq"][0][:, s - 1].repeat(Kc, 1))
        Rp = bring_in(R0, d, w, Td, Tb) if bring else reg
        D = chain_hard(Rp, n_register(Nq, w["nread"]), w, Td, Tb, None, gvisits, mq, R0, F.pad(d[:, None], (0, P - 1)))
        return move_digits(D, w["post"]) if bring else D

    def step_lesson(self, regs: dict | None, tabs, train: bool, names: tuple):
        """The step law (IO_MODE "local"): the register x + N * q split into its upper places and its lowest digit; one
        whole step from the upper places with the lowest digit coming in (the bring-in, the argmax chain, the post
        move) must give back x. Every combination of the step block's switches runs on the hard machine; the loss is
        the factorized mixture's (when `train`). Returns the loss and whether the argmax step passes."""
        if regs is None:
            return None, 0.0
        reg, xq = regs["reg"], regs["xq"]
        dev = reg.device
        n = reg.shape[0]
        with torch.no_grad():
            ok = float((self.chain_run(reg, self.wcodes(n, dev), regs, 1, tabs, bring=True) == xq).all(1).float().mean())
            if not train:
                return None, ok
            combos = torch.tensor(list(itertools.product(*[range(len(DIV_MENU[k])) for k in names])), device=dev)
            Kc = combos.shape[0]
            wk = codes(combo_wiring(self.wiring(n, dev), names, combos, n))
            verdict = (self.chain_run(reg.repeat(Kc, 1), wk, regs, Kc, tabs, bring=True) == xq.repeat(Kc, 1)).all(1)
            verdict = verdict.view(Kc, n)
        return mixture_nll(self.combo_logw(names, combos, gate=False), verdict).mean(), ok

    def io_lesson(self, Xd: Tensor, Nd: Tensor, Cd: Tensor, small: Tensor, tabs, train: bool, names: tuple):
        """The division's I/O switches `names` (the whole I/O block, or the reading's): the laws IO_LAWS (x mod N = x,
        N mod N = 0, x + Q N mod N = x) through the whole division, and the rows with T <= REAL_MAX_T through the
        whole machine (their outputs,
        compared with the answers in token_training_loss). Every combination runs on the hard machine (the other
        switches at their argmax). Returns the laws' verdicts [K, n], the answer rows' outputs [K, m, W] and their
        indices, and whether the argmax wiring passes the laws."""
        Td, Tb = tabs[0], tabs[1]
        dev = Xd.device
        gate = IO_MODE == "joint"                                           # the joint block holds the gate's rows
        combos = torch.tensor(list(itertools.product(*[range(len(DIV_MENU[k])) for k in names],
                                                     *([range(2), range(2)] if gate else []))), device=dev)
        Kc = combos.shape[0]
        n = min(IO_ROWS if Kc <= 4096 else IO_ROWS_ENUM, Xd.shape[0])
        with torch.no_grad():
            Xs, Ns = Xd[:n], Nd[:n]
            wa = self.wiring(n, dev)
            ca = codes(wa)
            mults = None
            if bool(((ca["a"] == 3) | (ca["b"] == 3)).any()):
                mults = multiples(torch.where(ca["nread"][:, None] == 1, Ns.flip(-1), Ns), tabs)
            laws = []                                                       # (dividend [n, 2W], the remainder [n, W])
            if "x" in IO_LAWS:
                laws.append((F.pad(Xs, (0, W)), Xs))
            if "n" in IO_LAWS:
                laws.append((F.pad(Ns, (0, W)), torch.zeros_like(Ns)))
            if "q" in IO_LAWS:                                              # x + Q N, Q another row's x (Q N by rows)
                Q = Xd.roll(1, 0)[:n]
                rows = row_hard(Ns[:, None, :].expand(n, W, W), Q, torch.arange(W, device=dev)[None].expand(n, W), tabs)
                Dq = F.pad(Xs, (0, C - W))
                for j in range(W):
                    Dq = add_learned(Dq, rows[:, j], Td, Tb)[0]
                laws.append((Dq[:, :2 * W], Xs))
            ok = torch.ones(n, dtype=torch.bool, device=dev)
            for Dv, r in laws:
                ok = ok & (divide_hard(Dv, Ns, ca, Td, Tb, tabs=tabs, mults=mults) == r).all(1)
            law_ok = float(ok.float().mean())
            if not train:
                return None, law_ok
            wk = codes(combo_wiring(wa, names, combos, n, gate))
            mk = None
            if mults is not None or "Mq" in DIV_MENU["a"] + DIV_MENU["b"]:          # every row's own nread
                both = [multiples(Ns, tabs), multiples(Ns.flip(-1), tabs)]
                mk = torch.where((wk["nread"] == 1)[:, None, None], both[1].repeat(Kc, 1, 1), both[0].repeat(Kc, 1, 1))
            verdict = torch.ones(Kc * n, dtype=torch.bool, device=dev)
            for Dv, r in laws:
                verdict = verdict & (divide_hard(Dv.repeat(Kc, 1), Ns.repeat(Kc, 1), wk, Td, Tb, tabs=tabs, mults=mk)
                                     == r.repeat(Kc, 1)).all(1)
            verdict = verdict.view(Kc, n)
            sel_rows = torch.nonzero(small).flatten()
            outs = None
            if sel_rows.numel() and Kc * sel_rows.numel() <= 65536:
                mr = sel_rows.numel()
                wm = codes(combo_wiring(self.wiring(mr, dev), names, combos, mr, gate))
                outs = run_hard(Xd[sel_rows].repeat(Kc, 1), Cd[sel_rows].repeat(Kc, 1), Nd[sel_rows].repeat(Kc, 1),
                                tabs, wm, cap=REAL_MAX_T + 1)[0].view(Kc, mr, W)
        return {"verdict": verdict, "outs": outs, "sel": sel_rows, "combos": combos, "names": names, "gate": gate}, law_ok

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
                divide_hard(V, Nd, self.wcodes(X, dev), Td, Tb, regs=regs, tabs=tabs)
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
        digits_on = ctl.gated_on is not None
        pieces = digits_on and ctl.step - ctl.gated_on >= LATE_AFTER         # the pieces' own table lessons
        if pieces and ctl.step - ctl.gated_on == LATE_AFTER:
            ctl.history.append((ctl.step, "pieces"))
            print(f"LR pieces_join step={ctl.step}", flush=True)
        sc = STE_CIN == "all"
        sa = STE_CIN in ("all", "anchors")
        xd, zero = F.pad(Xd, (0, P - W)), torch.zeros_like(Rg)
        loss = self.zero()
        if facts:
            # x < N on every top part: x[j:] - N[j:] borrows iff the digit strings differ, N[j:] - x[j:] never does
            for j in range(P):
                differ = (xd[:, j:] != M[:, j:]).any(1)
                p1 = link_soft(hot(xd[:, j:]), M[:, j:], TDC)[1][:, -1, 1]
                p2 = link_soft(hot(M[:, j:]), xd[:, j:], TDC)[1][:, -1, 1]
                loss = loss - (torch.where(differ, log(p1), log(1 - p1)).mean() + log(1 - p2).mean()) / P
            with torch.no_grad():
                okf = (sub_hard(xd, M, Td, Tb)[1] == 1) & (sub_hard(M, xd, Td, Tb)[1] == 0)
                ctl.facts_streak = ctl.facts_streak + 1 if bool(okf.all()) else 0
        if digits_on:
            # the order laws and bijectivity on the entries the batch's links read
            with torch.no_grad():
                used = torch.bincount(sub_hard(D, Mr, Td, Tb)[2].flatten(), minlength=200) > 0
            pa = self.a.float().softmax(-1)
            pd = pa.reshape(200, 10)
            for lo, hi in order_pairs(Tb, used):
                loss = loss - log((pd[lo] * (1 - pd[hi].cumsum(-1))).sum(-1)).mean()
            loss = loss - log(10 * pa.mean(0)).mean() - log(10 * pa.mean(1)).mean()
        if facts and not digits_on and ctl.facts_streak >= GATE_WIN:
            ctl.gated_on = ctl.step                                          # the borrows referee from the next step
            ctl.history.append((ctl.step, "order_digits"))
            print(f"LR order_digits step={ctl.step}", flush=True)
        # the pieces' own lessons, LATE_AFTER steps after the digit lessons start (their hard versions below are the
        # table's checks from step 1): every link checks itself with the learned adder, (D - N) - (0 - N) = D, complete
        if pieces:
            Es, be = link_soft(hot(D), Mr, TDC, sc)
            be = be[:, -1]
            Cs, bc = link_soft(hot(Z), M, TDC, sc)                           # 0 - N, once per row
            bc = bc[:, -1]
            back, bf = sub_soft(Es, Cs.repeat_interleave(NL, 0), TDC, None, sc)
            pe, pf, pc = be[:, 1], bf[:, -1, 1], bc[:, 1].repeat_interleave(NL, 0)
            agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc   # be = bc - bf
            weight = torch.ones(P, device=dev)
            weight[0] = ID_PLACE0                                           # the lowest place tells the swapped code
            nll = (-log(back).gather(-1, D[..., None]).squeeze(-1) * weight).sum(-1) - ID_BORROW * log(agree)
            loss = loss + nll.mean() / (weight.sum() + ID_BORROW)
        # the anchors (ADDER.md): on any numbers, then the facts about x and N
        checks = []                                                         # hard versions: (A, B, target, borrow)
        for wgt, A, B, target in ((ANCHOR_ZERO, Rg, zero, Rg), (ANCHOR_SELF, Rg, Rg, zero)):
            if wgt > 0:                                                     # R - 0 = R, R - R = 0, no final borrow
                if pieces:
                    Y, b = link_soft(hot(A), B, TDC, sa)
                    loss = loss + wgt * (digits_nll(Y, target) - log(b[:, -1, 0])).mean() / (P + 1)
                checks.append((A, B, target, 0))
        if pieces and ANCHOR_NEG2 > 0:                                      # 0 + N = N: 0 - (0 - N) = N, complete
            Y2, b2 = sub_soft(hot(Z), Cs, TDC, None, sc)
            agree2 = bc[:, 1] * b2[:, -1, 1] + bc[:, 0] * b2[:, -1, 0]
            loss = loss + ANCHOR_NEG2 * (digits_nll(Y2, M) - log(agree2)).mean() / (P + 1)
        if pieces and ANCHOR_COMM > 0:                                      # R + N = N + R: R - (0 - N) = N - (0 - R)
            A1, b1 = sub_soft(hot(Rg), Cs.repeat_interleave(W, 0), TDC, None, sc)
            CR, bcr = link_soft(hot(torch.zeros_like(Rg)), Rg, TDC, sc)
            A2, b2 = sub_soft(hot(MW), CR, TDC, None, sc)
            c1 = bc[:, 1].repeat_interleave(W, 0) * b1[:, -1, 0]             # P(carry = 1)
            c2 = bcr[:, -1, 1] * b2[:, -1, 0]
            agree = log((A1 * A2).sum(-1)).sum(-1) + log(c1 * c2 + (1 - c1) * (1 - c2))
            loss = loss - ANCHOR_COMM * agree.mean() / (P + 1)
        if facts and ANCHOR_LESS > 0:                                       # x < N: x - N borrows
            if pieces:
                loss = loss - ANCHOR_LESS * log(link_soft(hot(xd), M, TDC, sc)[1][:, -1, 1]).mean()
            checks.append((xd, M, None, 1))
        if facts and ANCHOR_MORE > 0:                                       # x < N: N - x does not borrow
            if pieces:
                loss = loss - ANCHOR_MORE * log(link_soft(hot(M), xd, TDC, sc)[1][:, -1, 0]).mean()
            checks.append((M, xd, None, 0))
        if pieces and ANCHOR_WRAP > 0:                                      # (x - N) - N does not borrow
            Ew = link_soft(hot(xd), M, TDC, sc)[0]
            loss = loss - ANCHOR_WRAP * log(link_soft(Ew, M, TDC, sc)[1][:, -1, 0]).mean()
        if pieces and ANCHOR_NEG > 0:                                       # N > 0: 0 - N borrows
            loss = loss - ANCHOR_NEG * log(bc[:, 1]).mean()
        # the product cell (SQUARE.md), once the table can be trusted
        lad_loss, lad_ok, verified, lad_rows = self.ladder(torch.cat([Xd, Nd]), tabs, ctl.ladder)
        if ctl.ladder:
            loss = loss + LADDER_W * lad_loss
        num_loss, num_ok = self.num_lesson(torch.cat([Xd, Nd]), tabs, ctl.ladder)   # the number rule
        if ctl.ladder and num_loss is not None:
            loss = loss + num_loss
        with torch.no_grad():                                               # the same checks on the hard machine
            Eh, beh, idx = sub_hard(D, Mr, Td, Tb)
            tally(ctl.visits, idx, 200)
            Ch, bch, idc = sub_hard(Z, M, Td, Tb)
            backh, bfh, idf = sub_hard(Eh, Ch.repeat_interleave(NL, 0), Td, Tb)
            tally(ctl.visits, idc, 200)
            tally(ctl.visits, idf, 200)
            ok = (backh == D).all(-1) & (beh == bch.repeat_interleave(NL, 0) - bfh)
            table = float(ok.float().mean())
            if ANCHOR_NEG2 > 0:
                Y2h, b2h, id2 = sub_hard(Z, Ch, Td, Tb)
                tally(ctl.visits, id2, 200)
                table = min(table, float(((Y2h == M).all(-1) & (b2h == bch)).float().mean()))
            if ANCHOR_COMM > 0:
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
            if facts and ANCHOR_NEG > 0:
                table = min(table, float((bch == 1).float().mean()))
            top_ = Xd.max(1).values[:, None, None].expand(-1, W, 1)
            sq = verified[:X].gather(2, top_)[..., 0].all(1) & sq_add       # the squares of x check out
            exact = min(table, lad_ok, float(sq.float().mean()), float(num_ok))
        # the division's wiring (MENUS.md): the chain's switches and the gate, then the I/O switches (the whole block,
        # or the step's by the step law and the reading's)
        chain_on = ctl.phase == "C" and IO_MODE != "joint"
        div_loss, div_ok, regs = self.chain_lesson(Xd, Nd, lad_rows[X:, 0], verified[X:, 0], tabs, chain_on)
        if chain_on and div_loss is not None:
            loss = loss + div_loss
        io_parts, law_ok, step_ok = [], 1.0, 1.0
        for name, names in IO_BLOCKS:
            if name == "step":
                step_loss, step_ok = self.step_lesson(regs, tabs, ctl.wire_io, names)
                if ctl.wire_io and step_loss is not None:
                    loss = loss + step_loss
            else:
                io, law_ok = self.io_lesson(Xd, Nd, Cd, small, tabs, ctl.wire_io, names)
                if ctl.wire_io:
                    io_parts.append(io)
        if ctl.wire_io:
            common["io"] = io_parts
        exact = min(exact, law_ok, step_ok, 1.0 if IO_MODE == "joint" else div_ok)   # joint: no step law, not even
                                                                                       # as a check
        if MODE == "learned":
            loss = loss + self.io_lessons(e, end, k, code, V[:, :W])
        common.update(exact=exact, table_exact=table, lad_ok=lad_ok, div_ok=div_ok, law_ok=law_ok, step_ok=step_ok,
                      num_ok=float(num_ok))
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
            rho = self.shape_rho((0, 1, 2), e, end, si, ti, length)                      # [X, field, run, dir]
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
        # the writer's pointer: its slots 0..m-1 on the answer's own positions, LSB first, and slot m right past them
        m = valid.long().sum(1)
        last = tpos.gather(1, (m - 1).clamp(min=0)[:, None])[:, 0] + W + 1   # the answer's last position: its LSB
        apos = last[:, None] - torch.arange(W + 1, device=dev)[None]
        pw = (self.shape_probs(3, e, end, apos[:, None], m[:, None])[:, 0] * self.shape_logp().exp()).sum(-1)
        loss = -pw.clamp_min(1e-30).log().mean()
        lv = labe < self.V
        rows_l, tl = torch.nonzero(lv)[:, 0], labe[lv]
        loss = loss - wsm[code[rows_l, tl], tl].clamp_min(EPS).log().mean()   # read an answer token, write it back
        # the answer's tokens are digits, read slot by slot: IO.md's round trip's liveness term on the answer's own
        # positions (slot j is live only if slots 0..j read digits)
        atok = labe.gather(1, apos[:, :W].clamp(0, Lp - 1))
        dl = torch.where(atok < self.V, torch.sigmoid(self.dg.float()[atok.clamp(max=self.V - 1)]),
                         torch.zeros((), device=dev))
        inans = torch.arange(W, device=dev)[None] < m[:, None]
        loss = loss - (dl.clamp_min(EPS).log().cumsum(-1) * inans).sum(-1).mean()
        with torch.no_grad():                                               # the hard round trips
            yd = self.read_answer(labe, e, end, code)
            pos = self.hard_pos(3, e, end)
            back = self.wsm().argmax(-1)[yd]
            o = torch.full_like(e, -1).scatter_(1, pos, back)
            ans = (torch.where(valid, o.gather(1, torch.where(valid, tpos + W + 1, 0)), labels) == labels).all(1)
            isd = self.dg > 0
            inv_ok = bool((self.wsm().argmax(-1)[code[:, isd]] == torch.nonzero(isd).flatten()[None]).all())
            ctl.io_exact = bool(ans.all()) and inv_ok and int(isd.sum()) == 10 and bool((aux["k"] >= 0).all())
            ctl.io_parts = (f"answers_ok={float(ans.float().mean()):.4f} code_inverse={int(inv_ok)} "
                            f"digit_tokens={int(isd.sum())} rows_read={float((aux['k'] >= 0).float().mean()):.4f}")
            if not ctl.io_exact and ctl.phase == "C":
                c0 = code[0, isd].tolist()
                wa = self.wsm().argmax(-1).tolist()
                print(f"LR io_not_exact step={ctl.step} {ctl.io_parts} digit_tokens={torch.nonzero(isd).flatten().tolist()} "
                      f"code_row0={c0} writer={wa} codes_differ={int((code[:, isd] != code[:1, isd]).any())}", flush=True)
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
                          self.wcodes(2 * n, dev))[0]
            Lpl, Lmi = (s2[:n] == yd[rows]).all(1).float(), (s2[n:] == yd[rows]).all(1).float()
        # the shape switch is held fixed: the lesson is normalized over T's two directions, so a shape that has not
        # chosen one (an untrained shape: about half each way) would beat the trained shape whose direction is still
        # wrong, and the shared switch would leave the shape that reads N, x and the writer
        rhoT = self.shape_rho((2,), e[rows], end[rows], si[rows], ti[rows], length[rows], hold_shape=True)[:, 0]
        wp = rhoT.gather(1, rt[rows][:, None, None].expand(-1, 1, 2))[:, 0, 0]  # read from the run's left end
        wm = rhoT.gather(1, rt[rows][:, None, None].expand(-1, 1, 2))[:, 0, 1]  # read from its right end
        hit = (Lpl + Lmi) > 0
        lt = -(((wp * Lpl + wm * Lmi).clamp_min(1e-30).log() - (wp + wm).clamp_min(1e-30).log()) * hit).sum()
        lt = lt / hit.sum().clamp_min(1)
        with torch.no_grad():
            pT = self.hard_pos(2, e, end)[rows]
            hs = int(((pT[:, 1] - pT[:, 0]) > 0).float().mean() > 0.5) * 2 - 1     # the snapped reading's direction
            mine = Lpl if hs > 0 else Lmi                                      # the snapped reading's verdict
            share = float(((wp if hs > 0 else wm) / (wp + wm).clamp_min(1e-30)).mean())
            ctl.t_log = f"rows={n} hits={int(hit.sum())} snapped_right={float(mine.mean()):.3f} share={share:.4f}"
            if bool(mine.all()) and share >= T_DONE:
                ctl.io = 5
                ctl.history.append((ctl.step, "T_resolved", n))
                print(f"LR io_stage=5 T_resolved step={ctl.step} {ctl.t_log}", flush=True)
        return lt

    def wire_answers(self, aux: dict, yd: Tensor) -> Tensor:
        """The division's I/O lessons (with the labels): for each I/O block, the factorized mixture over its
        combinations on the laws' verdicts and on the answers of the rows with T <= REAL_MAX_T."""
        loss = 0.0
        for io in aux["io"]:
            lw = self.combo_logw(io["names"], io["combos"], gate=io["gate"])
            loss = loss + mixture_nll(lw, io["verdict"]).mean()
            if io["outs"] is not None:
                ans = (io["outs"] == yd[io["sel"]][None]).all(-1)              # [K, m]
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
                  f"law_ok={aux.get('law_ok', 0.0):.4f} step_ok={aux.get('step_ok', 0.0):.4f} num_ok={aux.get('num_ok', 0.0):.0f} "
                  f"ans_ok={ctl.ans_ok:.4f} win={mean:.4f} "
                  f"io_exact={ctl.io_exact} chosen={ctl.chosen:.4f} rows_visited={int((ctl.visits > 0).sum())}/200 "
                  f"cells_visited={int((ctl.pvisits > 0).sum())}/100 facts_streak={ctl.facts_streak} "
                  f"order_digits={ctl.gated_on} layout: {self.layout_names()} shape: "
                  f"{SHAPES[int(self.shape.argmax())]} "
                  f"{','.join(f'{float(p):.3f}' for p in self.shape_logp().exp().tolist())} "
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
                if ctl.chain_streak >= WIRE_WIN or IO_MODE == "joint":
                    ctl.wire_io = True
                    ctl.history.append((ctl.step, "wire_io", ctl.in_phase))
                    print(f"LR io_wiring_lesson_joins step={ctl.step} wiring: {self.wiring_names()}", flush=True)
            ok = exact == 1.0 and ctl.io_exact and ctl.wire_io and ctl.ans_ok == 1.0
            ctl.streak = ctl.streak + 1 if ok else 0
            if ctl.streak >= FREEZE_WIN and ctl.io == 5:
                ctl.history.append((ctl.step, "frozen", ctl.in_phase, 1.0))
                print(f"LR frozen step={ctl.step} history={ctl.history} wiring: {self.wiring_names()} layout: "
                      f"{self.layout_names()} {classify(self.names())}", flush=True)
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
        names = self.names()
        print("WIRING " + self.wiring_names() + " true=" + str(self.wiring_true())
              + " chain=" + classify(names)["chain"].replace(" ", "_"), flush=True)
        slo, shi = self.sources()
        print(f"LAYOUT right={self.layout_right()}/{2 * W * C} {self.layout_names()} "
              f"numbers={classify(names)['numbers'].replace(' ', '_')} "
              + " ".join(f"row{j}=lo:{','.join(str(int(v)) for v in slo[j].tolist())}"
                         f"/hi:{','.join(str(int(v)) for v in shi[j].tolist())}" for j in range(W)), flush=True)
        print("WIRING probabilities " + " ".join(f"{k}={float(v.float().softmax(-1).max()):.4f}"
                                                 for k, v in self.sw.items())
              + f" lo={float(self.lay[0].float().softmax(-1).max()):.4f}"
              + f" hi={float(self.lay[1].float().softmax(-1).max()):.4f}"
              + (f" num={float(self.num.float().softmax(-1).max()):.4f}" if self.num is not None else ""), flush=True)
        print(f"MENUS div_level={DIV_LEVEL} mul_level={MUL_LEVEL} io_mode={IO_MODE} division={menu_size(DIV_MENU)} "
              f"chain_combos={len(CHAIN_COMBOS)} io={menu_size(DIV_MENU, IO_SW)} multiply={N_RULES ** 2 * len(MUL_MENU['num'])} "
              f"total={menu_size(DIV_MENU) * N_RULES ** 2 * len(MUL_MENU['num'])}", flush=True)
        if ctl.visits is not None:
            for name, v in (("train_visits", ctl.visits), ("prod_train_visits", ctl.pvisits)):
                print(f"TABLE {name} min={int(v.min())} rows_never={int((v == 0).sum())} "
                      f"counts={','.join(str(int(c)) for c in v.tolist())}", flush=True)
            print(f"TABLE gate_train_visits={','.join(str(int(c)) for c in ctl.gvisits.tolist())}", flush=True)
        print(f"TABLE schedule phase={ctl.phase} io={ctl.io} step={ctl.step} history={ctl.history}", flush=True)
        names = ["N", "x", "T", "writer"]
        print("IO shape=" + SHAPES[int(self.shape.argmax())] + " probabilities="
              + ",".join(f"{float(p):.4f}" for p in self.shape_logp().exp().tolist()), flush=True)
        for f in range(4):
            a = int(self.anc[f].argmax())
            st = "+1" if int(self.strd[STRD_OF[f]].argmax()) == 0 else "-1"
            print(f"IO field={names[f]} anchor={'END' if a == self.END else a} step={st} offsets="
                  f"{(self.off[f].argmax(-1) - OFF).tolist()} absolute={self.absp[f].argmax(-1).tolist()}", flush=True)
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
                             model.wcodes(sel.numel(), sel.device), cap=REAL_MAX_T + 1)[0]
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


def build_model(spec: ModelSpec) -> LearnedWiringIO:
    model = LearnedWiringIO(spec)
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
