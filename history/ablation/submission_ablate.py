"""Ablation copy of cpu_first/learned_remainder/submission_adder.py (md5 44e658ab2dfd270baa1d7625d734c3da), the core
recipe of ABLATION.md. With every constant at its default it trains exactly as that file does; the constants after
MAX_STEPS switch off the ingredients that the original has no constant for, and TABLE_EVERY logs the snapped table.

LearnedAdder: submission_chain.py with the hand-written adder taken out of the training self-check. The reduction is
the same restoring long division built only from a learned subtraction table; addition is now learned too.

Datapath (designed wiring, unchanged from submission_chain.py, the CPU's schoolbook algorithm in base 10):
  * read N (its digits come LSB first), x and T from the prompt;
  * loop T times: square the state (designed integer product), then reduce it mod N by long division:
      - the running remainder R has P = W + 1 digit places (W = 8 digits of the widest 24-bit modulus); it starts at the
        dividend's top digits v // 10^W (below N) and takes W steps, each shifting in the next dividend digit,
        Rp = 10 R + v_t;
      - every step subtracts N again and again (restoring division): trial 0 is Rp and trial s is trial s - 1 minus N,
        for s = 1, ..., S - 1. Each link of this chain is a ripple of ONE shared subtractor cell (r, m, borrow in) ->
        (digit, borrow out) over N's digits as the prompt gives them, place by place from the lowest;
      - the stop rule keeps the last trial before the chain first borrows (the subtraction's own borrow);
  * write the state's digits right-aligned on the prompt.
Learned (MODE = "learned"): the subtractor cell, a 200-row table of digit and borrow logits. Evaluation snaps the cell
to its argmax table and runs the datapath on integers. MODE = "designed" uses the exact subtraction table (plumbing
check).

The learned adder (ADDER = "learned", route A): in ten's complement 0 - B is 10^P - B, so A + B = A - (0 - B) mod
10^P, both subtractions the learned table, with final carry borrow(0 - B) - borrow(A - (0 - B)) (add_learned /
add_learned_soft, any width: the functions later squaring work can call). ADDER = "table" (route B) learns a second cell
for addition instead; ADDER = "designed" is submission_chain.py's hand-written adder (for comparison).

What teaches the cell (submission_chain.py's lessons, the designed adder replaced after the warm start; see ADDER.md):
  * phase A: every link of the machine's own division of each row's x^2 checks itself, (D - N) + N = D: with the
    designed adder for the first ADDER_WARM steps, then with the learned adder, (D - N) - (0 - N) = D. Complete: the
    P digits come back and the link's final borrow equals the adder's final carry (ID_BORROW). The lowest place
    weighs ID_PLACE0. Anchors that do not ask the table only to agree with itself (weights ANCHOR_*): R - 0 = R and
    R - R = 0 on the registers; x - N borrows and N - x does not (x < N); x - N wraps to 10^P + x - N >= N, so
    (x - N) - N does not borrow. After the warm start also 0 - N borrows (N > 0) and the learned adder's own rules,
    0 + N = N and R + N = N + R;
  * phase B, once ID_SWITCH of the checks pass or after ID_STEPS steps: also "x mod N = x" (one division step from
    x // 10 with the soft own-borrow choice must stop the chain at trial 0) and the answers of the batch's rows with
    T <= REAL_MAX_T along the model's own hard path: each step's chain length is chosen by the hard datapath (argmax
    table, stop at the first borrow) and the soft cell runs that many links on digit distributions.
Adam's state is reset at the switch; the model freezes (learning rate 0) after FREEZE_WIN consecutive steps in which
every check is exact on the hard datapath. The schedule is Python-side state of the model (not model state). Everything
runs in fp32 with autocast off; evaluation is integer arithmetic.

Status (ADDER.md): from random init the learned adder alone does NOT teach the table (ADDER_WARM = 0: no seed, no
variant tried). So the designed adder stays as a warm start: the check uses it for the first ADDER_WARM = 70 steps only
(the table is then about 185/200 right); from step 71 on every lesson and every check is learned arithmetic, and the
learned adder finishes the table. The other options are the variants of that search. The adder functions are exact
whenever the table is.
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

MODE = "learned"       # "learned" | "designed" (the exact subtraction table: the plumbing check)
DIGIT, MARK_N, MARK_X, MARK_T = 7, 2, 3, 4  # token ids: digit d is DIGIT + d
W = 8                  # digits of the widest modulus (a 24-bit N is below 10^8); one reduction takes W steps
P = W + 1              # register places: 10 R + v_t < 10 N < 10^9
S = 10                 # trials per step: trial s is s links down the chain (Rp - s*N until a link borrows)
INIT = 0.1             # std of the initial logits
LR = 0.05              # Adam on the logits
ID_SWITCH = 0.8        # phase B (the other lessons join) once this share of the link checks passes, averaged ...
STAGE_WIN = 20         # ... over this many steps,
ID_STEPS = 300         # ... or after this many steps
ID_PLACE0 = 5.0        # weight of the lowest place in the check (the swapped borrow code fails it only there)
ID_BORROW = 1.0        # weight of the final borrow in the check: the complete identity (D - N) + N = D also needs the
                       # subtraction's final borrow to equal the adder's final carry
ADDER = "learned"      # the check's adder: "learned" (E + N = E - (0 - N), both subtractions the learned table) |
                       # "table" (a second learned table, an adder cell (r, m, carry in) -> (digit, carry out)) |
                       # "designed" (submission_chain.py's hand-written adder, for comparison)
ANCHOR_INV2 = 0.0      # ADDER = "table": weight of (R + N) - N = R on the registers
ANCHOR_INV = 0.0       # weight of R - (R - N) = N on the registers (subtraction only)
ANCHOR_PAIRS = 0.0     # weight of the check (R - R') + R' = R on pairs of registers (R' = another row's register)
ANCHOR_NEG = 1.0       # weight of "0 - N borrows" (N > 0: the learned adder's carry is 1 - borrow(E - (0 - N)))
ANCHOR_NEG2 = 1.0      # weight of 0 + N = N with the learned adder: 0 - (0 - N) = N
ANCHOR_COMM = 1.0      # weight of R + N = N + R with the learned adder, on the registers
ANCHOR_ZERO = 1.0      # weight of R - 0 = R on the registers
ANCHOR_SELF = 1.0      # weight of R - R = 0 on the registers
ANCHOR_LESS = 1.0      # weight of "x < N, so x - N borrows"
ANCHOR_MORE = 1.0      # weight of "x < N, so N - x does not borrow"
ANCHOR_WRAP = 1.0      # weight of "x - N wraps to 10^P + x - N >= N, so (x - N) - N does not borrow"
SELF_FROM = "B"        # phase in which "x mod N = x" joins: "A" (from step 1) | "B"
SELF_LESSON = True     # phase B: also "x mod N = x" (each row's x divided by N, soft own-borrow choice)
SELF_STEPS = 1         # ... in this many steps, from x // 10^SELF_STEPS (one step: the chain must stop at x itself)
REAL_MAX_T = 2         # phase B also trains on the batch's real rows with T <= REAL_MAX_T (0: never sees a label)
ANSWER_PATH = "hard"   # the rows' answers along the own hard path | "soft": the soft own-borrow choice at every step
ID_WEIGHT = 1.0        # weight of the check (D - N) + N = D
ID_CURRIC = 0          # > 0: the check holds mod 10^k on the lowest k places, k = 1 + step // ID_CURRIC up to P
ADDER_WARM = 70        # the designed adder for the first ADDER_WARM steps only, then ADDER alone (0: from scratch,
                       # which does not learn; ADDER.md)
FREEZE_WIN = 200       # phase B: freeze after this many consecutive steps with every check exact
BATCH = 512
EVAL_BATCH = 16384
MAX_STEPS = 3000       # the model freezes near step 330 (with ADDER_WARM = 0 it never does: this keeps that run short)
LOG_EVERY = 250
# Ablation switches (ABLATION.md); the defaults are submission_adder.py's recipe.
ID_AFTER_WARM = 1.0    # weight of the check once the learned adder runs it, after ADDER_WARM (0: no check after that)
ID_GATE = False        # ID_AFTER_WARM = 0: the check's hard version still joins the checks the schedule reads
ID_LINKS = "all"       # links the check reads: "all" (every link the hard chain computes) | "live" (up to the first borrow)
CE_SCOPE = "width"     # places the remainder lessons score: "width" (below N's width) | "all" (all P places)
ADAM_RESET = True      # clear Adam's state at the switch to phase B
HARD_ANCHORS = True    # every lesson's hard version joins the checks the schedule reads (False: the check, x mod N = x)
TABLE_EVERY = 0        # > 0: log the snapped table every this many steps (logging only)
# General replacements (ABLATION.md section 4), off by default:
ANCHOR_ORDER = 0.0     # 0 < x < N: for every ordered pair (a, b) of different numbers among 0, x, N, and for a - a, the
                       # final borrow of a - b is [a < b] (one rule for x - N, N - x, 0 - N and the borrows of R - 0, R - R)
ANCHOR_CHAIN = 0.0     # a link that borrows wraps to 10^P + D - N >= N, so on every link of the machine's own chains the
                       # link after a borrowing one does not borrow (the wrap-around fact on the machine's own numbers)
HARD_CIN = False       # R - 0 = R and R - R = 0 ripple with the hard machine's borrow into each place, so they train only
                       # the rows they use (with soft borrows they also teach the borrow-in-1 rows next to them)

POW10 = [10 ** i for i in range(19)]


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


def link_hard(D: Tensor, M: Tensor, Td: Tensor, Tb: Tensor):
    """One link of the chain on integers, D - N: D [X, P] digits, M [X, P] N's digits (any width), flat tables Td, Tb
    [200] indexed (r, m, c) -> r * 20 + m * 2 + c. Returns the digits [X, P], the final borrow [X] and the table rows
    [X, P]."""
    c = torch.zeros_like(D[:, 0])
    outs, rows = [], []
    for j in range(D.shape[1]):
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


def divide_hard(v: Tensor, steps: int, M: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None) -> Tensor:
    """v mod N on integers, from R = v // 10^steps (below N) in `steps` steps of the chain. Returns the remainder's
    digits [X, P]. `visits` accumulates every table row used."""
    R = to_digits(v // POW10[steps], P)
    for t in reversed(range(steps)):
        R = step_hard(torch.cat([((v // POW10[t]) % 10)[:, None], R[:, :P - 1]], 1), M, Td, Tb, visits)[0]
    return R


def link_inputs(v: Tensor, steps: int, M: Tensor, Td: Tensor, Tb: Tensor) -> tuple[Tensor, Tensor]:
    """The hard division's link inputs at every step [X, steps, S - 1, P]: the register Rp = 10 R + v_t feeds link 1,
    trial s - 1 feeds link s (the machine's own numbers, every link it computes); and whether each link was live
    [X, steps, S - 1]."""
    R = to_digits(v // POW10[steps], P)
    out, live = [], []
    for t in reversed(range(steps)):
        R, _, ins, lv = step_hard(torch.cat([((v // POW10[t]) % 10)[:, None], R[:, :P - 1]], 1), M, Td, Tb)
        out.append(ins)
        live.append(lv)
    return torch.stack(out, 1), torch.stack(live, 1)


def add_hard(D: Tensor, M: Tensor) -> tuple[Tensor, Tensor]:
    """The designed adder on integers: digits D [..., P] + M [..., P], mod 10^P, and the final carry."""
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


def link_soft(D: Tensor, M: Tensor, TDC: Tensor):
    """One link on digit distributions, the cell applied multilinearly: D [X, P, 10] minus N's exact digits M [X, P].
    TDC [20, 120]: rows (r, c), columns (m, 10 digit + 2 borrow probabilities). Returns the digits [X, P, 10] and the
    final borrow [X, 2]."""
    X = D.shape[0]
    c = torch.zeros(X, 2, device=D.device)
    c[:, 0] = 1
    out = []
    for j in range(D.shape[1]):
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = (joint @ TDC).view(X, 10, 12)                                       # every m at once
        Y = Y.gather(1, M[:, j, None, None].expand(X, 1, 12)).squeeze(1)        # N's own digit
        out.append(Y[:, :10])
        c = Y[:, 10:]
    return torch.stack(out, 1), c


def link_soft_hc(D: Tensor, M: Tensor, TDC: Tensor, Td: Tensor, Tb: Tensor):
    """link_soft with the hard machine's borrow into each place (HARD_CIN): the cell is applied to the exact digits D, M
    [X, P] with the borrow that the snapped table carries in, so the gradient reaches only the rows the hard ripple uses.
    Returns the digits [X, P, 10] and the final borrow [X, 2]."""
    X = D.shape[0]
    cin = link_hard(D, M, Td, Tb)[2] % 2                                    # the hard borrow into each place
    out = []
    for j in range(D.shape[1]):
        Y = TDC[D[:, j] * 2 + cin[:, j]].view(X, 10, 12)
        Y = Y.gather(1, M[:, j, None, None].expand(X, 1, 12)).squeeze(1)
        out.append(Y[:, :10])
    return torch.stack(out, 1), Y[:, 10:]


def sub_soft(D: Tensor, M: Tensor, TDC: Tensor):
    """D - M with both operands on digit distributions, D, M [X, width, 10] (multilinear, as link_soft). Returns the
    digits [X, width, 10] and the final borrow [X, 2]."""
    X = D.shape[0]
    c = torch.zeros(X, 2, device=D.device)
    c[:, 0] = 1
    out = []
    for j in range(D.shape[1]):
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = torch.einsum("xmk,xm->xk", (joint @ TDC).view(X, 10, 12), M[:, j])  # expected over the subtrahend's digit
        out.append(Y[:, :10])
        c = Y[:, 10:]
    return torch.stack(out, 1), c


def add_learned(A: Tensor, B: Tensor, Td: Tensor, Tb: Tensor) -> tuple[Tensor, Tensor]:
    """The learned adder on integers, any width: A + B mod 10^width computed as A - (0 - B), both subtractions the
    learned table (in ten's complement 0 - B is 10^width - B), and the final carry borrow(0 - B) - borrow(A - (0 - B))
    (0 or 1 when the table is true subtraction). A, B [X, width] digits, LSB first."""
    C, bc, _ = link_hard(torch.zeros_like(B), B, Td, Tb)
    Y, by, _ = link_hard(A, C, Td, Tb)
    return Y, bc - by


def add_learned_soft(A: Tensor, B: Tensor, TDC: Tensor) -> tuple[Tensor, Tensor]:
    """The same on digit distributions A, B [X, width, 10]: the sum's digits [X, width, 10] and P(final carry = 1)."""
    C, bc = sub_soft(torch.zeros_like(B).index_fill_(-1, torch.tensor([0], device=B.device), 1.0), B, TDC)
    Y, by = sub_soft(A, C, TDC)
    return Y, bc[:, 1] * by[:, 0]


def divide_soft(v: Tensor, steps: int, M: Tensor, TDC: Tensor) -> Tensor:
    """The same division on digit distributions with the soft own-borrow choice: trial s is kept with the probability
    that links 1..s do not borrow and link s + 1 does (trial S - 1 when no link borrows). Returns the final
    remainder's digit distributions [X, P, 10]."""
    R = F.one_hot(to_digits(v // POW10[steps], P), 10).float()
    for t in reversed(range(steps)):
        D = torch.cat([F.one_hot((v // POW10[t]) % 10, 10).float()[:, None], R[:, :P - 1]], 1)
        R = torch.zeros_like(D)
        alive = torch.ones(v.shape[0], device=v.device)                       # P(no link has borrowed yet)
        for s in range(1, S):
            Dn, c = link_soft(D, M, TDC)
            # renormalize: the float32 rounding of the table's rows compounds over places and links (8 steps x 9 links
            # overflow without it)
            Dn, c = Dn / Dn.sum(-1, keepdim=True), c / c.sum(-1, keepdim=True)
            R = R + (alive * c[:, 1])[:, None, None] * D                        # link s borrows: keep trial s - 1
            alive = alive * c[:, 0]
            D = Dn
        R = R + alive[:, None, None] * D
    return R


def divide_path(v: Tensor, steps: int, M: Tensor, TDC: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None):
    """The model's own hard path, learned along: each step's chain length (the quotient digit) is chosen by the hard
    datapath (argmax table, stop at the first borrow), and the soft cell runs that many links on digit distributions
    from the soft register. Returns the soft remainder [X, P, 10] (the gradient reaches the cell through the kept
    trials' links only) and the hard one [X, P]."""
    Rh = to_digits(v // POW10[steps], P)
    Rs = F.one_hot(Rh, 10).float()
    for t in reversed(range(steps)):
        vt = (v // POW10[t]) % 10
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
    if CE_SCOPE == "all":
        scored = torch.ones_like(scored)
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
        self.visits: Tensor | None = None     # table rows used by training's hard checks (the audit)
        self.used: Tensor | None = None       # table rows used at evaluation
        self.reported = -1
        self.history: list = []


class LearnedAdder(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.a = nn.Parameter(INIT * torch.randn(10, 10, 2, 10))           # subtractor: (r, m, borrow in) -> digit
        self.b = nn.Parameter(INIT * torch.randn(10, 10, 2, 2))            # ... -> borrow out
        if ADDER == "table":
            self.g = nn.Parameter(INIT * torch.randn(10, 10, 2, 10))       # adder: (r, m, carry in) -> digit
            self.h = nn.Parameter(INIT * torch.randn(10, 10, 2, 2))        # ... -> carry out
        self.ctl = Control()

    def tables(self) -> tuple[Tensor, Tensor]:
        if MODE == "designed":
            return exact_tables(self.a.device)
        return self.a.argmax(-1).flatten(), self.b.argmax(-1).flatten()

    def adder_tables(self) -> tuple[Tensor, Tensor]:
        """ADDER = "table": the snapped adder cell, flat over (r, m, carry in) as the subtractor (link_hard runs it)."""
        return self.g.argmax(-1).flatten(), self.h.argmax(-1).flatten()

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        if attention_mask is None:
            attention_mask = input_ids != 0
        with torch.autocast(input_ids.device.type, enabled=False):
            return self._forward(input_ids, attention_mask)

    def _forward(self, input_ids: Tensor, attention_mask: Tensor):
        dev = input_ids.device
        Nd, Xd, Td_, end, width = parse(input_ids, attention_mask)
        x, T = from_digits(Xd), from_digits(Td_[:, :2])
        M = F.pad(Nd, (0, P - W))                                            # N's digits from the prompt, P places
        Td, Tb = self.tables()
        rows, L, V = input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
        if self.training:
            aux = self.lessons(x, T, width, M, Td, Tb)
            return torch.zeros(rows, L, V, device=dev), aux
        ctl = self.ctl
        if ctl.used is None:
            ctl.used = torch.zeros(200, dtype=torch.long, device=dev)
            self.report_tables(Td, Tb)
        s = self.run(x, T, M, Td, Tb, ctl.used)
        n_used = int((ctl.used > 0).sum())
        if n_used != ctl.reported:
            ctl.reported = n_used
            unvisited = int(((ctl.used > 0) & (ctl.visits == 0)).sum()) if ctl.visits is not None else -1
            print(f"AUDIT eval_rows_used={n_used}/200 used_but_never_trained={unvisited}", flush=True)
        place = end[:, None] - 1 - torch.arange(L, device=dev)[None]          # position p holds digit end - 1 - p
        d = to_digits(s, W).gather(1, place.clamp(0, W - 1))
        tok = torch.where((place >= 0) & (place < W), d + DIGIT, 0)
        logits = torch.full((rows, L, V), -20.0, device=dev)
        return logits.scatter_(2, tok[..., None], 0.0), None

    def run(self, x: Tensor, T: Tensor, M: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None) -> Tensor:
        """The hard machine: T squarings, each reduced by the chained long division."""
        s = x.clone()
        for t in range(int(T.max()) if T.numel() else 0):
            R = divide_hard(s * s, W, M, Td, Tb, visits)
            s = torch.where(T > t, from_digits(R), s)
        return s

    def lessons(self, x, T, width, M, Td, Tb) -> dict:
        """Label-free lessons on the rows' own numbers (nothing made up):
          * every link of the machine's own division of each row's x^2 checks itself, (D - N) + N = D, with the adder
            ADDER (route A: (D - N) - (0 - N), all three subtractions the learned table): the digits, and (ID_BORROW)
            the final borrow against the adder's final carry; optionally only mod 10^k (ID_CURRIC);
          * the anchors and identities weighted by ANCHOR_* (module constants), each with a hard version that joins
            the checks the schedule reads;
          * "x mod N = x" from phase B (or from step 1, SELF_FROM = "A"): dividing x by N with the soft own-borrow
            choice must stop the chain at trial 0 (SELF_STEPS steps from x // 10^SELF_STEPS);
          * phase B also adds the real rows' answers along the own path (REAL_MAX_T > 0, ANSWER_PATH)."""
        ctl = self.ctl
        ctl.step += 1
        dev = x.device
        if ctl.visits is None:
            ctl.visits = torch.zeros(200, dtype=torch.long, device=dev)
        if ctl.phase == "frozen":
            return {"loss": self.a.sum() * 0.0, "real": None, "ctl": ctl}
        TDC = torch.cat([self.a.float().softmax(-1), self.b.float().softmax(-1)], -1)
        TDC = TDC.permute(0, 2, 1, 3).reshape(20, 120)
        X, K = x.shape[0], W * (S - 1)
        with torch.no_grad():
            D, live = link_inputs(x * x, W, M, Td, Tb)
            D, Mr = D.reshape(X * K, P), M.repeat_interleave(K, 0)
            Z = torch.zeros_like(M)
        keep = live.reshape(X * K) if ID_LINKS == "live" else None           # the links the check reads
        rep = (lambda t: t.repeat_interleave(K, 0)) if keep is None else (lambda t: t.repeat_interleave(K, 0)[keep])
        k = P if ID_CURRIC <= 0 else min(P, 1 + (ctl.step - 1) // ID_CURRIC)   # the check holds mod 10^k
        Dk, Mk = (D[:, :k], Mr[:, :k]) if keep is None else (D[keep][:, :k], Mr[keep][:, :k])
        Es, be = link_soft(F.one_hot(Dk, 10).float(), Mk, TDC)              # the link, D - N
        adder = "designed" if ctl.step <= ADDER_WARM else ADDER
        w_id = ID_WEIGHT if ctl.step <= ADDER_WARM else ID_WEIGHT * ID_AFTER_WARM  # the check's weight
        if ADDER == "learned":
            Cs, bc = link_soft(F.one_hot(Z, 10).float(), M, TDC)            # 0 - N, once per row
        if adder == "learned":
            bck = link_soft(F.one_hot(Z[:, :k], 10).float(), M[:, :k], TDC)[1] if k < P else bc
            back, bf = sub_soft(Es, rep(Cs[:, :k]), TDC)                     # (D - N) - (0 - N) = (D - N) + N
            pc = rep(bck[:, 1])
            pe, pf = be[:, 1], bf[:, 1]
            agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc   # be = bc - bf
        elif adder == "table":
            TAC = torch.cat([self.g.float().softmax(-1), self.h.float().softmax(-1)], -1)
            TAC = TAC.permute(0, 2, 1, 3).reshape(20, 120)
            back, cy = link_soft(Es, Mk, TAC)                                # (D - N) + N, the learned adder cell
            agree = cy[:, 1] * be[:, 1] + cy[:, 0] * be[:, 0]
        else:
            back, carry = add_soft(Es, Mk)                                   # (D - N) + N, the designed adder
            agree = carry * be[:, 1] + (1 - carry) * be[:, 0]
        weight = torch.ones(k, device=dev)
        weight[0] = ID_PLACE0                                               # the lowest place tells the swapped code
        nll = (-back.clamp_min(1e-12).log().gather(-1, Dk[..., None]).squeeze(-1) * weight).sum(-1)
        if ID_BORROW > 0:                                                   # final borrow = the adder's final carry
            nll = nll - ID_BORROW * agree.clamp_min(1e-12).log()
        loss = w_id * nll.mean() / (weight.sum() + ID_BORROW)
        Rg = D.view(X, W, S - 1, P)[:, :, 0].reshape(X * W, P)               # the registers Rp (first links' inputs)
        MW, xd = M.repeat_interleave(W, 0), to_digits(x, P)
        hot = lambda v: F.one_hot(v, 10).float()
        log = lambda p: p.clamp_min(1e-12).log()
        digits_nll = lambda Y, target: -log(Y).gather(-1, target[..., None]).squeeze(-1).sum(-1)
        checks = []                                                         # hard versions: (A, B, target, borrow)
        zero = torch.zeros_like(Rg)
        for wgt, A, B, target in ((ANCHOR_ZERO, Rg, zero, Rg), (ANCHOR_SELF, Rg, Rg, zero)):
            if wgt > 0:                                                     # R - 0 = R, R - R = 0, no final borrow
                Y, b = link_soft_hc(A, B, TDC, Td, Tb) if HARD_CIN else link_soft(hot(A), B, TDC)
                loss = loss + wgt * (digits_nll(Y, target) - log(b[:, 0])).mean() / (P + 1)
                checks.append((A, B, target, 0))
        if ANCHOR_LESS > 0:                                                 # x < N: x - N borrows
            loss = loss - ANCHOR_LESS * log(link_soft(hot(xd), M, TDC)[1][:, 1]).mean()
            checks.append((xd, M, None, 1))
        if ANCHOR_MORE > 0:                                                 # x < N: N - x does not borrow
            loss = loss - ANCHOR_MORE * log(link_soft(hot(M), xd, TDC)[1][:, 0]).mean()
            checks.append((M, xd, None, 0))
        if ANCHOR_WRAP > 0:                                                 # (x - N) - N does not borrow
            Ew = link_soft(hot(xd), M, TDC)[0]
            loss = loss - ANCHOR_WRAP * log(link_soft(Ew, M, TDC)[1][:, 0]).mean()
        if ANCHOR_ORDER > 0:                                                # 0 < x < N: borrow(a - b) = [a < b]
            zx = torch.zeros_like(xd)
            for A, B, t in ((xd, M, 1), (M, xd, 0), (zx, M, 1), (M, zx, 0), (zx, xd, 1), (xd, zx, 0), (xd, xd, 0),
                            (M, M, 0)):
                loss = loss - ANCHOR_ORDER * log(link_soft(hot(A), B, TDC)[1][:, t]).mean()
                checks.append((A, B, None, t))
        if ANCHOR_CHAIN > 0:                                                # after a borrowing link, no borrow
            with torch.no_grad():
                wrapped = link_hard(Dk, Mk, Td, Tb)[1] == 1
            if bool(wrapped.any()):
                loss = loss - ANCHOR_CHAIN * log(link_soft(Es[wrapped], Mk[wrapped], TDC)[1][:, 0]).mean()
        if ANCHOR_PAIRS > 0 and ADDER == "learned":                         # (R - R') + R' = R, the learned adder
            R2 = Rg.roll(1, 0)
            Ep, bep = link_soft(hot(Rg), R2, TDC)
            Cp, bcp = link_soft(hot(torch.zeros_like(R2)), R2, TDC)
            Fp, bfp = sub_soft(Ep, Cp, TDC)
            pe, pf, pc = bep[:, 1], bfp[:, 1], bcp[:, 1]
            agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc
            loss = loss + ANCHOR_PAIRS * (digits_nll(Fp, Rg) - log(agree)).mean() / (P + 1)
        if ANCHOR_INV > 0:                                                  # R - (R - N) = N, complete
            E1, b1 = link_soft(hot(Rg), MW, TDC)
            G1, g1 = sub_soft(hot(Rg), E1, TDC)
            agree = b1[:, 1] * g1[:, 1] + b1[:, 0] * g1[:, 0]
            loss = loss + ANCHOR_INV * (digits_nll(G1, MW) - log(agree)).mean() / (P + 1)
        if adder == "learned":
            if ANCHOR_NEG > 0:                                              # N > 0: 0 - N borrows
                loss = loss - ANCHOR_NEG * log(bc[:, 1]).mean()
            if ANCHOR_NEG2 > 0:                                             # 0 + N = N: 0 - (0 - N) = N, complete
                Y2, b2 = sub_soft(hot(Z), Cs, TDC)
                agree2 = bc[:, 1] * b2[:, 1] + bc[:, 0] * b2[:, 0]
                loss = loss + ANCHOR_NEG2 * (digits_nll(Y2, M) - log(agree2)).mean() / (P + 1)
            if ANCHOR_COMM > 0:                                             # R + N = N + R: R - (0 - N) = N - (0 - R)
                A1, b1 = sub_soft(hot(Rg), Cs.repeat_interleave(W, 0), TDC)
                CR, bcr = link_soft(hot(torch.zeros_like(Rg)), Rg, TDC)
                A2, b2 = sub_soft(hot(MW), CR, TDC)
                c1, c2 = bc[:, 1].repeat_interleave(W, 0) * b1[:, 0], bcr[:, 1] * b2[:, 0]   # P(carry = 1)
                agree = log((A1 * A2).sum(-1)).sum(-1) + log(c1 * c2 + (1 - c1) * (1 - c2))
                loss = loss - ANCHOR_COMM * agree.mean() / (P + 1)
        if ADDER == "table":
            if ANCHOR_NEG2 > 0:                                             # 0 + N = N, no carry
                Y2, c2 = link_soft(hot(Z), M, TAC)
                loss = loss + ANCHOR_NEG2 * (digits_nll(Y2, M) - log(c2[:, 0])).mean() / (P + 1)
            if ANCHOR_COMM > 0:                                             # R + N = N + R
                A1, c1 = link_soft(hot(Rg), MW, TAC)
                A2, c2 = link_soft(hot(MW), Rg, TAC)
                agree = log((A1 * A2).sum(-1)).sum(-1) + log(c1[:, 1] * c2[:, 1] + c1[:, 0] * c2[:, 0])
                loss = loss - ANCHOR_COMM * agree.mean() / (P + 1)
            if ANCHOR_INV2 > 0:                                             # (R + N) - N = R, complete
                G, cg = link_soft(hot(Rg), MW, TAC)
                Bk, bk = link_soft(G, MW, TDC)
                agree = cg[:, 1] * bk[:, 1] + cg[:, 0] * bk[:, 0]
                loss = loss + ANCHOR_INV2 * (digits_nll(Bk, Rg) - log(agree)).mean() / (P + 1)
        selfl = SELF_LESSON and (ctl.phase == "B" or SELF_FROM == "A")
        if selfl:
            loss = loss + ce(divide_soft(x, SELF_STEPS, M, TDC), xd, width)
        with torch.no_grad():
            Eh, beh, idx = link_hard(Dk, Mk, Td, Tb)
            ctl.visits += torch.bincount(idx.flatten(), minlength=200)
            if adder == "learned":
                Ch, bch, idc = link_hard(Z, M, Td, Tb)
                bchk = Tb[idc[:, k - 1]]                                    # the borrow out of place k - 1
                backh, bfh, idf = link_hard(Eh, rep(Ch[:, :k]), Td, Tb)
                ctl.visits += torch.bincount(torch.cat([idc.flatten(), idf.flatten()]), minlength=200)
                ok = (backh == Dk).all(-1)
                if ID_BORROW > 0:
                    ok = ok & (beh == rep(bchk) - bfh)
                exact = float(ok.float().mean()) if w_id > 0 or ID_GATE else 1.0
                exact_main = exact
                if ANCHOR_NEG > 0:
                    exact = min(exact, float((bch == 1).float().mean()))
                if ANCHOR_NEG2 > 0:
                    Y2h, b2h, id2 = link_hard(Z, Ch, Td, Tb)
                    ctl.visits += torch.bincount(id2.flatten(), minlength=200)
                    exact = min(exact, float(((Y2h == M).all(-1) & (b2h == bch)).float().mean()))
                if ANCHOR_COMM > 0:
                    A1h, c1h = add_learned(Rg, MW, Td, Tb)
                    A2h, c2h = add_learned(MW, Rg, Td, Tb)
                    exact = min(exact, float(((A1h == A2h).all(-1) & (c1h == c2h)).float().mean()))
            elif adder == "table":
                Ad, Ac = self.adder_tables()
                backh, cyh, _ = link_hard(Eh, Mk, Ad, Ac)
                ok = (backh == Dk).all(-1)
                if ID_BORROW > 0:
                    ok = ok & (cyh == beh)
                exact = float(ok.float().mean())
                exact_main = exact
                if ANCHOR_NEG2 > 0:
                    Y2h, c2h, _ = link_hard(Z, M, Ad, Ac)
                    exact = min(exact, float(((Y2h == M).all(-1) & (c2h == 0)).float().mean()))
                if ANCHOR_COMM > 0:
                    A1h, c1h, _ = link_hard(Rg, MW, Ad, Ac)
                    A2h, c2h, _ = link_hard(MW, Rg, Ad, Ac)
                    exact = min(exact, float(((A1h == A2h).all(-1) & (c1h == c2h)).float().mean()))
                if ANCHOR_INV2 > 0:
                    Gh, cgh, _ = link_hard(Rg, MW, Ad, Ac)
                    Bh, bh, idb = link_hard(Gh, MW, Td, Tb)
                    ctl.visits += torch.bincount(idb.flatten(), minlength=200)
                    exact = min(exact, float(((Bh == Rg).all(-1) & (bh == cgh)).float().mean()))
            else:
                backh, carryh = add_hard(Eh, Mk)
                ok = (backh == Dk).all(-1)
                if ID_BORROW > 0:
                    ok = ok & (carryh == beh)
                exact = float(ok.float().mean())
                exact_main = exact
            if ANCHOR_PAIRS > 0 and ADDER == "learned":
                R2 = Rg.roll(1, 0)
                Eph, beph, _ = link_hard(Rg, R2, Td, Tb)
                Fph, cph = add_learned(Eph, R2, Td, Tb)
                exact = min(exact, float(((Fph == Rg).all(-1) & (cph == beph)).float().mean()))
            if ANCHOR_INV > 0:
                E1h, b1h, _ = link_hard(Rg, MW, Td, Tb)
                G1h, g1h, idg = link_hard(Rg, E1h, Td, Tb)
                ctl.visits += torch.bincount(idg.flatten(), minlength=200)
                exact = min(exact, float(((G1h == MW).all(-1) & (g1h == b1h)).float().mean()))
            for A, B, target, borrow in checks:
                Yh, bh, ida = link_hard(A, B, Td, Tb)
                ctl.visits += torch.bincount(ida.flatten(), minlength=200)
                ok = bh == borrow
                if target is not None:
                    ok = ok & (Yh == target).all(-1)
                exact = min(exact, float(ok.float().mean()))
            if ANCHOR_WRAP > 0:
                Ewh, _, idw = link_hard(xd, M, Td, Tb)
                exact = min(exact, float((link_hard(Ewh, M, Td, Tb)[1] == 0).float().mean()))
            if ANCHOR_CHAIN > 0 and bool((beh == 1).any()):
                exact = min(exact, float((link_hard(Eh[beh == 1], Mk[beh == 1], Td, Tb)[1] == 0).float().mean()))
            if not HARD_ANCHORS:                                            # the schedule reads the check only
                exact = exact_main
            if selfl:
                exact = min(exact, float((divide_hard(x, SELF_STEPS, M, Td, Tb) == xd).all(1).float().mean()))
        real = None
        if ctl.phase == "B" and REAL_MAX_T > 0:
            sel = torch.nonzero(T <= REAL_MAX_T).flatten()
            if sel.numel():
                with torch.no_grad():
                    sp = self.run(x[sel], T[sel] - 1, M[sel], Td, Tb)
                if ANSWER_PATH == "soft":                                   # the soft own-borrow choice at every step
                    Rr = divide_soft(sp * sp, W, M[sel], TDC)
                    Rrh = divide_hard(sp * sp, W, M[sel], Td, Tb, ctl.visits)
                else:
                    Rr, Rrh = divide_path(sp * sp, W, M[sel], TDC, Td, Tb, ctl.visits)
                real = {"sel": sel, "R": Rr, "Rh": Rrh, "width": width[sel]}
        if TABLE_EVERY > 0 and ctl.step % TABLE_EVERY == 0:
            print(f"LR table step={ctl.step} phase={ctl.phase} exact={exact:.4f} digit=" + "".join(map(str, Td.tolist()))
                  + " borrow=" + "".join(map(str, Tb.tolist())), flush=True)
        self.control(exact, float(loss.detach()))
        return {"loss": loss, "real": real, "ctl": ctl}

    def control(self, exact: float, loss: float) -> None:
        ctl = self.ctl
        ctl.win.append(exact)
        ctl.in_phase += 1
        mean = sum(ctl.win) / len(ctl.win)
        if ctl.step % LOG_EVERY == 0 or ctl.step == 1:
            print(f"LR step={ctl.step} phase={ctl.phase} in_phase={ctl.in_phase} loss={loss:.4f} "
                  f"hard_exact={exact:.4f} win={mean:.4f} rows_visited={int((ctl.visits > 0).sum())}/200", flush=True)
        if ctl.phase == "A" and ((len(ctl.win) == STAGE_WIN and mean >= ID_SWITCH) or ctl.in_phase >= ID_STEPS):
            # the checks pass on most links: "x mod N = x" and the rows' answers join, with a fresh optimizer (Adam's
            # memory of phase A's gradients would slow the repairs)
            ctl.history.append((ctl.step, "switch", ctl.in_phase, round(mean, 4)))
            print(f"LR switch_to_own_borrow step={ctl.step} win={mean:.4f}", flush=True)
            ctl.phase, ctl.reset, ctl.in_phase = "B", ADAM_RESET, 0
            ctl.win.clear()
        elif ctl.phase == "B":
            ctl.streak = ctl.streak + 1 if exact == 1.0 else 0
            if ctl.streak >= FREEZE_WIN:
                ctl.history.append((ctl.step, "frozen", ctl.in_phase, 1.0))
                print(f"LR frozen step={ctl.step} history={ctl.history}", flush=True)
                ctl.phase = "frozen"

    def report_tables(self, Td: Tensor, Tb: Tensor) -> None:
        """At the first evaluation batch: the snapped table and how often training's hard checks used each row."""
        ctl = self.ctl
        print("TABLE digit=" + "".join(str(int(d)) for d in Td.tolist()), flush=True)
        print("TABLE borrow=" + "".join(str(int(d)) for d in Tb.tolist()), flush=True)
        if ADDER == "table":
            Ad, Ac = self.adder_tables()
            print("TABLE adder_digit=" + "".join(str(int(d)) for d in Ad.tolist()), flush=True)
            print("TABLE adder_carry=" + "".join(str(int(d)) for d in Ac.tolist()), flush=True)
        if ctl.visits is not None:
            v = ctl.visits
            print(f"TABLE train_visits min={int(v.min())} rows_never={int((v == 0).sum())} "
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


def build_model(spec: ModelSpec) -> LearnedAdder:
    model = LearnedAdder(spec)
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
