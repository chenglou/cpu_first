"""LearnedControl: x^(2^T) mod N on submission_chain.py's reduction (a restoring long division built only from one learned
subtraction table), with its two control pieces made of learned parts too:
  * the loop over T is a countdown on T's digits by the same learned table, T - 1 each round;
  * the chain's stop rule ("keep the last trial before the first borrow") is gone. Every subtraction, of the chain and of
    the countdown, is committed or held, and a small learned gate reading that subtraction's own final borrow decides.

Datapath (designed wiring):
  * read N (its digits come LSB first), x and T from the prompt. T stays digits: a countdown register of K places, LSB
    first, K = T's width in the prompt (the widest in the batch);
  * rounds: each round subtracts 1 from the countdown with the learned table, and the commit gate reads that
    subtraction's final borrow. A committing round squares the state (designed integer product), reduces it mod N and
    takes the countdown's difference; a holding round leaves both as they are. Once the countdown is 0, 0 - 1 borrows, the
    round holds and the countdown stays 0, so every later round holds the same way: the same inputs give the same borrow,
    so no latch is needed. The rounds end when no row commits, or after 10^K rounds (a safety cap: a K-digit countdown
    that works ends sooner);
  * the reduction is long division. The running remainder R has P = W + 1 digit places (W = 8 digits of the widest
    24-bit modulus); it starts at the dividend's top digits v // 10^W (below N) and takes W steps, each shifting in the
    next dividend digit, Rp = 10 R + v_t. Each step is a chain of S - 1 links on one register: link s subtracts N's digits
    (as the prompt gives them) from the register with the learned table, place by place from the lowest, and the commit
    gate reads the link's final borrow: the register takes the difference (commit) or keeps its value (hold). With the
    exact table and gate this is restoring division: the register takes Rp - N, Rp - 2N, ... while no link borrows, then
    holds, and every later link of the step recomputes the same subtraction and holds again;
  * write the state's digits right-aligned on the prompt.
Learned (MODE = "learned"): the subtractor cell (r, m, borrow in) -> (digit, borrow out), a 200-row table of digit and
borrow logits shared by every place, link, step, width and by the countdown; and the commit gate, a 2-row table
borrow -> (hold, commit) logits shared by every link and every round. GATE = "borrow" wires the borrow as the hold
signal instead (no gate). Evaluation snaps both tables to their argmax and runs the datapath on integers. MODE =
"designed" uses the exact table and gate (the plumbing check).

What teaches them (label-free lessons on the rows' own numbers, then the rows' answers; see CONTROL.md for what broke
and why). First the table learns what a borrow means, then the gate learns what to do with it:
  * phase A (the table): every link of the machine's own division of each row's x^2 checks itself, (D - N) + N = D,
    with the learned subtractor and a designed adder (D is the link's input). Complete (ID_BORROW): the P digits must
    come back and the final borrow must equal the adder's final carry. The lowest place weighs ID_PLACE0 times the others;
  * phase B (the table), once ID_SWITCH of the checks pass or after ID_STEPS steps: also "x - N borrows" (CMP_LESSON:
    x < N, so the subtraction's final borrow must be 1). submission_chain.py got this from "x mod N = x" through its
    designed stop rule; without that rule nothing else teaches the borrow of the zeros above N's top digit;
  * phase C (the gate), once "x - N borrows" holds on CMP_SWITCH of the rows or after CMP_STEPS steps: also two facts
    about remainders, each one division step on digit distributions whose links run on a detached copy of the soft table
    (FACT_TABLE), so they teach only the gate: "x mod N = x" (x < N: from the register x the step must keep trial 0, so
    hold on a borrow) and "N mod N = 0" (NMOD_LESSON: from the register N it must keep trial 1, so commit when N - N
    does not borrow, then hold on 0 - N's borrow). Trial j is kept with the probability that links 1..j commit and link
    j + 1 holds, each link's commit probability being the soft gate on that trial's soft borrow (submission_chain.py's
    soft own-borrow choice, with the gate in place of the stop rule); and the answers of the batch's rows with
    T <= REAL_MAX_T along the model's own hard path (below).
The own path: the hard machine runs its own countdown and records the input of each row's last committed squaring. Each
step of its division of that number's square computes the trials and their borrows with the argmax table and keeps the
trial before the gate's first hold; the soft cell computes the same trials on digit distributions from the soft register
and keeps them with the soft gate's probabilities at the hard borrows (OWN_GATE), so the answers teach the table and the
gate. The rows' dividend is the machine's own s_{T-1}^2 and the answer is the row's label. Adam's state is reset at the
switch to phase B; the model freezes (learning rate 0) after FREEZE_WIN consecutive phase-C steps in which every check,
"x - N borrows" and both facts are exact on the hard datapath. The schedule is Python-side state of the model (not model
state). Loss: cross-entropy of the soft remainder's digits below N's width. Everything runs in fp32 with autocast off;
evaluation is integer arithmetic.
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

MODE = "learned"       # "learned" | "designed" (the exact subtraction table and gate: the plumbing check)
GATE = "learned"       # "learned": the commit gate is a learned 2-row table borrow -> (hold, commit) | "borrow": a
                       # subtraction's own final borrow is wired as its hold signal (no gate)
DIGIT, MARK_N, MARK_X, MARK_T = 7, 2, 3, 4  # token ids: digit d is DIGIT + d
W = 8                  # digits of the widest modulus (a 24-bit N is below 10^8); one reduction takes W steps
P = W + 1              # register places: 10 R + v_t < 10 N < 10^9
S = 10                 # a step's chain has S - 1 links, enough for Rp - 9N
INIT = 0.1             # std of the initial logits
LR = 0.05              # Adam on the logits
ID_SWITCH = 0.8        # phase B (the other lessons join) once this share of the link checks passes, averaged ...
STAGE_WIN = 20         # ... over this many steps,
ID_STEPS = 300         # ... or after this many steps
ID_PLACE0 = 5.0        # weight of the lowest place in the check (the swapped borrow code fails it only there)
ID_LINKS = "all"       # which links check themselves: "all" (every link computed, also the held links after a step's
                       # first hold, which recompute the same subtraction) | "live" (up to and including the first hold)
ID_BORROW = 1.0        # weight of the borrows in the check (0: off): the complete identity (D - N) + N = D also needs
                       # the subtraction's final borrow to equal the designed adder's final carry ...
ID_BORROWS = "final"   # ... "final": that one | "all": at every place (the identity holds for every block of low places,
                       # so the borrow out of each place equals the adder's carry out of it; deepens the (r, r, 1) saddle)
CMP_LESSON = True      # phase B: "x - N borrows" (x < N: the subtraction's final borrow must be 1)
GATE_PHASE = "C"       # when the gate's lessons and the rows' answers join: "C", once "x - N borrows" holds on CMP_SWITCH
                       # of the rows (averaged over STAGE_WIN steps) or after CMP_STEPS steps of phase B | "B": at once
CMP_SWITCH = 0.99
CMP_STEPS = 300
SELF_LESSON = True     # phase C: "x mod N = x" (one step from the register x keeps trial 0)
NMOD_LESSON = True     # phase C: "N mod N = 0" (one step from the register N keeps trial 1)
FACT_TABLE = False     # the two facts also teach the table (False: their links use a detached copy of the soft table, so
                       # they teach only the gate: the gate and the borrows' meaning cannot co-adapt)
OWN_GATE = "mix"       # the own path's soft register keeps each trial: "mix" with the soft gate's probabilities at the
                       # hard borrows | "off" where the hard gate keeps it (the rows' answers do not reach the gate)
REAL_MAX_T = 2         # phase C also trains on the batch's real rows with T <= REAL_MAX_T (0: never sees a label; below 10)
FREEZE_WIN = 200       # phase C: freeze after this many consecutive steps with every check exact
BATCH = 512
EVAL_BATCH = 16384
MAX_STEPS = 20000
LOG_EVERY = 250

POW10 = [10 ** i for i in range(19)]


def to_digits(v: Tensor, n: int) -> Tensor:
    """Non-negative int64 values -> their n lowest decimal digits, LSB first (designed I/O)."""
    return (v[..., None] // torch.tensor(POW10[:n], device=v.device)) % 10


def from_digits(d: Tensor) -> Tensor:
    return (d * torch.tensor(POW10[:d.shape[-1]], device=d.device)).sum(-1)


def parse(input_ids: Tensor, attention_mask: Tensor):
    """Prompt `N <N digits, LSB first> X <x, MSB first> T <T, MSB first>` -> LSB-first digits [rows, W] of N, x and T,
    the prompt's end, N's width and T's width."""
    pos = lambda tok: (input_ids == tok).int().argmax(1)
    i_n, i_x, i_t = pos(MARK_N), pos(MARK_X), pos(MARK_T)
    end = attention_mask.long().sum(1)
    k = torch.arange(W, device=input_ids.device)[None]

    def field(start, step, count):
        idx = (start[:, None] + step * k).clamp(0, input_ids.shape[1] - 1)
        d = (input_ids.gather(1, idx) - DIGIT).clamp(0, 9)
        return torch.where(k < count[:, None], d, 0)

    width, twidth = i_x - i_n - 1, end - i_t - 1
    return (field(i_n + 1, 1, width), field(i_t - 1, -1, i_t - i_x - 1), field(end - 1, -1, twidth), end, width,
            twidth)


def exact_tables(device) -> tuple[Tensor, Tensor]:
    """The designed subtractor (the plumbing check only): digit and borrow of r - m - c, flat over (r, m, c)."""
    r = torch.arange(10, device=device)[:, None, None]
    m = torch.arange(10, device=device)[None, :, None]
    c = torch.arange(2, device=device)[None, None, :]
    x = r - m - c
    return (x % 10).flatten(), (x < 0).long().flatten()


def exact_gate(device) -> Tensor:
    """The designed commit gate (the plumbing check, and GATE = "borrow"): commit unless the subtraction borrows."""
    return torch.tensor([1, 0], device=device)


def link_hard(D: Tensor, M: Tensor, Td: Tensor, Tb: Tensor):
    """One subtraction on integers, D - M place by place from the lowest: D, M [X, K] digits, flat tables Td, Tb [200]
    indexed (r, m, c) -> r * 20 + m * 2 + c. Returns the digits [X, K], the final borrow [X] and the table rows
    [X, K]."""
    c = torch.zeros_like(D[:, 0])
    outs, rows = [], []
    for j in range(D.shape[1]):
        idx = (D[:, j] * 10 + M[:, j]) * 2 + c
        outs.append(Td[idx])
        c = Tb[idx]
        rows.append(idx)
    return torch.stack(outs, 1), c, torch.stack(rows, 1)


def step_hard(Rp: Tensor, M: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor, visits: Tensor | None = None,
              gvisits: Tensor | None = None):
    """One division step on integers: S - 1 links on one register, each D - N with the table, and the gate Gc [2]
    (1 = commit) on the link's final borrow: the register takes the difference or keeps D. Returns the register [X, P],
    every link's input [X, S - 1, P] and whether each link committed [X, S - 1]. `visits` / `gvisits` accumulate every
    table / gate row the links use."""
    D = Rp
    ins, rows, borrows, commits = [], [], [], []
    for s in range(1, S):
        ins.append(D)
        E, c, idx = link_hard(D, M, Td, Tb)
        k = Gc[c] == 1
        D = torch.where(k[:, None], E, D)
        rows.append(idx)
        borrows.append(c)
        commits.append(k)
    if visits is not None:
        visits += torch.bincount(torch.stack(rows).flatten(), minlength=200)
    if gvisits is not None:
        gvisits += torch.bincount(torch.stack(borrows).flatten(), minlength=2)
    return D, torch.stack(ins, 1), torch.stack(commits, 1)


def divide_hard(v: Tensor, steps: int, M: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor, visits: Tensor | None = None,
                gvisits: Tensor | None = None) -> Tensor:
    """v mod N on integers, from R = v // 10^steps (below N) in `steps` steps of the chain. Returns the remainder's
    digits [X, P]. `visits` / `gvisits` accumulate every table / gate row used."""
    R = to_digits(v // POW10[steps], P)
    for t in reversed(range(steps)):
        R = step_hard(torch.cat([((v // POW10[t]) % 10)[:, None], R[:, :P - 1]], 1), M, Td, Tb, Gc, visits, gvisits)[0]
    return R


def link_inputs(v: Tensor, steps: int, M: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor) -> tuple[Tensor, Tensor]:
    """The hard division's link inputs at every step [X, steps, S - 1, P] (the machine's own numbers, every link it
    computes) and whether each link was live [X, steps, S - 1]: every earlier link of its step committed."""
    R = to_digits(v // POW10[steps], P)
    out, live = [], []
    for t in reversed(range(steps)):
        R, ins, k = step_hard(torch.cat([((v // POW10[t]) % 10)[:, None], R[:, :P - 1]], 1), M, Td, Tb, Gc)
        out.append(ins)
        live.append(torch.cat([torch.ones_like(k[:, :1]), k[:, :-1]], 1).long().cumprod(1) == 1)
    return torch.stack(out, 1), torch.stack(live, 1)


def count_round(C: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor):
    """One round of the countdown C [X, K]: C - 1 with the table and the gate on its final borrow. Returns the
    difference [X, K], whether the round commits [X], the final borrow [X] and the table rows [X, K]."""
    Cn, b, idx = link_hard(C, F.one_hot(torch.zeros_like(C[:, 0]), C.shape[1]), Td, Tb)   # the digits of 1
    return Cn, Gc[b] == 1, b, idx


def countdown_hard(C: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor, cvisits: Tensor) -> None:
    """The countdown alone (no squaring), for the audit: accumulates the table rows it uses until every row holds."""
    for _ in range(10 ** C.shape[1]):
        Cn, k, _, idx = count_round(C, Td, Tb, Gc)
        cvisits += torch.bincount(idx.flatten(), minlength=200)
        if not bool(k.any()):
            return
        C = torch.where(k[:, None], Cn, C)


def run_hard(x: Tensor, C: Tensor, M: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor, visits: Tensor | None = None,
             gvisits: Tensor | None = None, cvisits: Tensor | None = None, cap: int | None = None):
    """The hard machine: rounds until no row commits (at most `cap`, by default 10^K). A committing row squares and
    reduces its state and takes the countdown's difference, a holding row keeps both. Returns the state [X], the input
    of each row's last committed squaring [X] (x if none), the number of committed rounds [X] and whether the row has
    held [X] (False: it was still committing at the cap). `visits` / `gvisits` accumulate the table / gate rows of the
    reductions and the countdown, `cvisits` the table rows of the countdown."""
    s, prev = x.clone(), x.clone()
    rounds = torch.zeros_like(x)
    held = torch.zeros_like(x, dtype=torch.bool)
    for _ in range(10 ** C.shape[1] if cap is None else cap):
        Cn, k, b, idx = count_round(C, Td, Tb, Gc)
        if cvisits is not None:
            cvisits += torch.bincount(idx.flatten(), minlength=200)
        if gvisits is not None:
            gvisits += torch.bincount(b, minlength=2)
        held = held | ~k
        if not bool(k.any()):
            break
        prev = torch.where(k, s, prev)
        s = torch.where(k, from_digits(divide_hard(s * s, W, M, Td, Tb, Gc, visits, gvisits)), s)
        C = torch.where(k[:, None], Cn, C)
        rounds = rounds + k.long()
    return s, prev, rounds, held


def add_hard(D: Tensor, M: Tensor) -> tuple[Tensor, Tensor]:
    """The designed adder on integers: digits D [..., P] + M [..., P], mod 10^P, and the carry out of every place
    [..., P] (the last is the final carry)."""
    c = torch.zeros_like(D[..., 0])
    out, carries = [], []
    for j in range(P):
        x = D[..., j] + M[..., j] + c
        out.append(x % 10)
        c = x // 10
        carries.append(c)
    return torch.stack(out, -1), torch.stack(carries, -1)


def add_soft(D: Tensor, M: Tensor) -> tuple[Tensor, Tensor]:
    """The designed adder on digit distributions D [..., P, 10] plus exact digits M [..., P], mod 10^P (multilinear),
    and the probability of a carry out of every place [..., P]."""
    o = torch.arange(10, device=D.device)
    c1 = torch.zeros(D.shape[:-2], device=D.device)                         # P(carry in = 1)
    out, carries = [], []
    for j in range(P):
        d, m = D[..., j, :], M[..., j, None]
        out.append((1 - c1)[..., None] * d.gather(-1, (o - m) % 10) + c1[..., None] * d.gather(-1, (o - m - 1) % 10))
        below = torch.cat([torch.zeros_like(d[..., :1]), d.cumsum(-1)], -1)  # below[k] = P(d < k)
        g0 = 1 - below.gather(-1, 10 - m)                                    # P(d + m >= 10)
        g1 = 1 - below.gather(-1, 9 - m)                                     # P(d + m + 1 >= 10)
        c1 = (1 - c1) * g0[..., 0] + c1 * g1[..., 0]
        carries.append(c1)
    return torch.stack(out, -2), torch.stack(carries, -1)


def link_soft(D: Tensor, M: Tensor, TDC: Tensor):
    """One link on digit distributions, the cell applied multilinearly: D [X, P, 10] minus N's exact digits M [X, P].
    TDC [20, 120]: rows (r, c), columns (m, 10 digit + 2 borrow probabilities). Returns the digits [X, P, 10] and the
    borrow out of every place [X, P, 2] (the last is the final borrow)."""
    X = D.shape[0]
    c = torch.zeros(X, 2, device=D.device)
    c[:, 0] = 1
    out, borrows = [], []
    for j in range(P):
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = (joint @ TDC).view(X, 10, 12)                                       # every m at once
        Y = Y.gather(1, M[:, j, None, None].expand(X, 1, 12)).squeeze(1)        # N's own digit
        out.append(Y[:, :10])
        c = Y[:, 10:]
        borrows.append(c)
    return torch.stack(out, 1), torch.stack(borrows, 1)


def trials_soft(D: Tensor, M: Tensor, TDC: Tensor) -> tuple[Tensor, Tensor]:
    """A step's trials on digit distributions: trial 0 is the register D [X, P, 10], trial j is trial j - 1 minus N
    (the numbers a step's links can reach). Returns the trials [X, S, P, 10] and each link's final borrow [X, S - 1, 2]
    (link j + 1 subtracts N from trial j)."""
    trials, borrows = [D], []
    for _ in range(1, S):
        E, c = link_soft(trials[-1], M, TDC)
        trials.append(E)
        borrows.append(c[:, -1])
    return torch.stack(trials, 1), torch.stack(borrows, 1)


def keep_soft(trials: Tensor, pc: Tensor) -> Tensor:
    """The register after a step whose link j + 1 commits with probability pc[:, j] [X, S - 1]: trial j is kept with the
    probability that links 1..j commit and link j + 1 holds (trial S - 1 when every link commits). A held register makes
    every later link recompute the same subtraction, so the first hold is final."""
    alive = torch.ones_like(pc[:, 0])
    R = torch.zeros_like(trials[:, 0])
    for j in range(S - 1):
        R = R + (alive * (1 - pc[:, j]))[:, None, None] * trials[:, j]
        alive = alive * pc[:, j]
    return R + alive[:, None, None] * trials[:, S - 1]


def chain_soft(D: Tensor, M: Tensor, TDC: Tensor, GS: Tensor) -> Tensor:
    """One division step on digit distributions from the register D [X, P, 10], the gate GS [2] = P(commit | borrow 0,
    1) on each link's soft borrow."""
    trials, b = trials_soft(D, M, TDC)
    return keep_soft(trials, (b * GS).sum(-1))


def divide_path(v: Tensor, steps: int, M: Tensor, TDC: Tensor, GS: Tensor, Td: Tensor, Tb: Tensor, Gc: Tensor,
                visits: Tensor | None = None, gvisits: Tensor | None = None):
    """The model's own hard path, learned along. At each step the hard datapath (argmax table) computes the trials and
    their borrows, and its gate keeps the trial before the first hold (the machine's own register); the soft cell
    computes the same trials on digit distributions from the soft register. The soft register keeps trial j with the
    soft gate's probabilities at the hard borrows (OWN_GATE = "mix", so the answers also teach the gate) or where the
    hard gate keeps it ("off"). Returns the soft remainder [X, P, 10] and the hard one [X, P]."""
    Rh = to_digits(v // POW10[steps], P)
    Rs = F.one_hot(Rh, 10).float()
    for t in reversed(range(steps)):
        vt = (v // POW10[t]) % 10
        Dh = torch.cat([vt[:, None], Rh[:, :P - 1]], 1)
        hard, bh = [Dh], []
        for _ in range(1, S):
            Eh, c, idx = link_hard(hard[-1], M, Td, Tb)
            hard.append(Eh)
            bh.append(c)
            if visits is not None:
                visits += torch.bincount(idx.flatten(), minlength=200)
                gvisits += torch.bincount(c, minlength=2)
        bh = torch.stack(bh, 1)                                             # [X, S - 1]
        q = (Gc[bh] == 1).long().cumprod(1).sum(1)                          # commits before the first hold
        Rh = torch.stack(hard, 1).gather(1, q[:, None, None].expand(-1, 1, P)).squeeze(1)
        trials = trials_soft(torch.cat([F.one_hot(vt, 10).float()[:, None], Rs[:, :P - 1]], 1), M, TDC)[0]
        pc = GS[bh] if OWN_GATE == "mix" else (torch.arange(S - 1, device=v.device)[None] < q[:, None]).float()
        Rs = keep_soft(trials, pc)
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
    """The training schedule (Python-side, not model state): phase A, phase B, frozen; and the audit's counters."""

    def __init__(self) -> None:
        self.step = 0
        self.phase = "frozen" if MODE == "designed" else "A"
        self.in_phase = 0
        self.streak = 0
        self.reset = False                    # the scheduler clears Adam's state at the switch
        self.win: deque = deque(maxlen=STAGE_WIN)
        self.visits: Tensor | None = None     # table rows used by training's hard checks and own path (the audit)
        self.gvisits: Tensor | None = None    # gate rows (borrow 0, 1) used by the same
        self.cvisits: Tensor | None = None    # table rows the countdown uses on the training rows' T
        self.used: Tensor | None = None       # table rows used at evaluation by the reductions
        self.used_count: Tensor | None = None  # ... by the countdown
        self.used_gate: Tensor | None = None  # gate rows used at evaluation
        self.reported = None
        self.history: list = []


class LearnedControl(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.a = nn.Parameter(INIT * torch.randn(10, 10, 2, 10))           # subtractor: (r, m, borrow in) -> digit
        self.b = nn.Parameter(INIT * torch.randn(10, 10, 2, 2))            # ... -> borrow out
        if GATE == "learned":
            self.g = nn.Parameter(INIT * torch.randn(2, 2))                # commit gate: borrow -> (hold, commit)
        self.ctl = Control()

    def tables(self) -> tuple[Tensor, Tensor, Tensor]:
        """The snapped subtractor (digit, borrow) and gate (1 = commit), each indexed by its input."""
        dev = self.a.device
        if MODE == "designed":
            return (*exact_tables(dev), exact_gate(dev))
        gate = self.g.argmax(-1) if GATE == "learned" else exact_gate(dev)
        return self.a.argmax(-1).flatten(), self.b.argmax(-1).flatten(), gate

    def gate_soft(self) -> Tensor:
        """P(commit | borrow 0, 1) [2]."""
        if GATE == "learned" and MODE == "learned":
            return self.g.float().softmax(-1)[:, 1]
        return exact_gate(self.a.device).float()

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        if attention_mask is None:
            attention_mask = input_ids != 0
        with torch.autocast(input_ids.device.type, enabled=False):
            return self._forward(input_ids, attention_mask)

    def _forward(self, input_ids: Tensor, attention_mask: Tensor):
        dev = input_ids.device
        Nd, Xd, Cd, end, width, twidth = parse(input_ids, attention_mask)
        x = from_digits(Xd)
        C = Cd[:, :max(1, int(twidth.max()))]                                # the countdown: T's digits, K places
        M = F.pad(Nd, (0, P - W))                                            # N's digits from the prompt, P places
        Td, Tb, Gc = self.tables()
        rows, L, V = input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
        if self.training:
            small = (twidth == 1) & (Cd[:, 0] <= REAL_MAX_T)                 # the curriculum's rows: T <= REAL_MAX_T
            aux = self.lessons(x, C, small, width, M, Td, Tb, Gc)
            return torch.zeros(rows, L, V, device=dev), aux
        ctl = self.ctl
        if ctl.used is None:
            ctl.used = torch.zeros(200, dtype=torch.long, device=dev)
            ctl.used_count = torch.zeros(200, dtype=torch.long, device=dev)
            ctl.used_gate = torch.zeros(2, dtype=torch.long, device=dev)
            self.report_tables(Td, Tb, Gc)
        s = run_hard(x, C, M, Td, Tb, Gc, ctl.used, ctl.used_gate, ctl.used_count)[0]
        self.audit()
        place = end[:, None] - 1 - torch.arange(L, device=dev)[None]          # position p holds digit end - 1 - p
        d = to_digits(s, W).gather(1, place.clamp(0, W - 1))
        tok = torch.where((place >= 0) & (place < W), d + DIGIT, 0)
        logits = torch.full((rows, L, V), -20.0, device=dev)
        return logits.scatter_(2, tok[..., None], 0.0), None

    def lessons(self, x, C, small, width, M, Td, Tb, Gc) -> dict:
        """Label-free lessons on the rows' own numbers (nothing made up):
          * every link of the machine's own division of each row's x^2 checks itself, (D - N) + N = D, with the learned
            subtractor and a designed adder: the digits, and (ID_BORROW) the final borrow against the adder's final
            carry;
          * from phase B on (once the checks pass on ID_SWITCH of the links): x - N ends in a borrow (x < N). This
            teaches the borrows the check is slow on (the zeros above N's top digit);
          * from phase C on (once that holds): x is its own remainder and N's is 0. One division step from the register
            x must keep trial 0 (x - N borrows: hold) and one from the register N must keep trial 1 (N - N does not
            borrow: commit; 0 - N borrows: hold). On a detached table, so they teach the gate both of its rows and
            nothing else;
          * phase C also adds the real rows' answers along the own path (REAL_MAX_T > 0)."""
        ctl = self.ctl
        ctl.step += 1
        dev = x.device
        if ctl.visits is None:
            ctl.visits = torch.zeros(200, dtype=torch.long, device=dev)
            ctl.gvisits = torch.zeros(2, dtype=torch.long, device=dev)
            ctl.cvisits = torch.zeros(200, dtype=torch.long, device=dev)
        if ctl.phase == "frozen":
            return {"loss": self.a.sum() * 0.0, "real": None, "ctl": ctl}
        with torch.no_grad():
            countdown_hard(C, Td, Tb, Gc, ctl.cvisits)                       # the audit: what the countdown meets
        TDC = torch.cat([self.a.float().softmax(-1), self.b.float().softmax(-1)], -1)
        TDC = TDC.permute(0, 2, 1, 3).reshape(20, 120)
        GS = self.gate_soft()
        X = x.shape[0]
        with torch.no_grad():
            D, live = link_inputs(x * x, W, M, Td, Tb, Gc)
            D, Mr = D.reshape(X * W * (S - 1), P), M.repeat_interleave(W * (S - 1), 0)
            if ID_LINKS == "live":
                keep = torch.nonzero(live.flatten()).flatten()
                D, Mr = D[keep], Mr[keep]
        Ds, bs = link_soft(F.one_hot(D, 10).float(), Mr, TDC)
        back, carry = add_soft(Ds, Mr)                                       # (D - N) + N on distributions
        weight = torch.ones(P, device=dev)
        weight[0] = ID_PLACE0                                               # the lowest place tells the swapped code
        nll = (-back.clamp_min(1e-12).log().gather(-1, D[..., None]).squeeze(-1) * weight).sum(-1)
        nb = P if ID_BORROWS == "all" else 1
        if ID_BORROW > 0:                                                   # borrow out = the adder's carry out
            agree = carry[:, -nb:] * bs[:, -nb:, 1] + (1 - carry[:, -nb:]) * bs[:, -nb:, 0]
            nll = nll - ID_BORROW * agree.clamp_min(1e-12).log().sum(-1)
        loss = nll.mean() / (weight.sum() + ID_BORROW * nb)
        xd, zero = to_digits(x, P), torch.zeros_like(M)
        phase_b = ctl.phase in ("B", "C")
        gate_on = ctl.phase == "C" or (phase_b and GATE_PHASE == "B")
        if phase_b and CMP_LESSON:                                          # x < N: x - N ends in a borrow
            loss = loss - link_soft(F.one_hot(xd, 10).float(), M, TDC)[1][:, -1, 1].clamp_min(1e-12).log().mean()
        TF = TDC if FACT_TABLE else TDC.detach()                            # the facts' table
        if gate_on and SELF_LESSON:
            loss = loss + ce(chain_soft(F.one_hot(xd, 10).float(), M, TF, GS), xd, width)
        if gate_on and NMOD_LESSON:
            loss = loss + ce(chain_soft(F.one_hot(M, 10).float(), M, TF, GS), zero, width)
        with torch.no_grad():
            Dh, bh, idx = link_hard(D, Mr, Td, Tb)
            ctl.visits += torch.bincount(idx.flatten(), minlength=200)
            backh, carryh = add_hard(Dh, Mr)
            ok = (backh == D).all(-1)
            if ID_BORROW > 0:
                ok = ok & (carryh[:, -nb:] == Tb[idx][:, -nb:]).all(-1)
            exact = float(ok.float().mean())
            cmp = float((link_hard(xd, M, Td, Tb)[1] == 1).float().mean())
            if gate_on:                                                     # the hard machine on the facts
                selfx = step_hard(xd, M, Td, Tb, Gc, ctl.visits, ctl.gvisits)[0]
                selfn = step_hard(M, M, Td, Tb, Gc, ctl.visits, ctl.gvisits)[0]
                exact = min(exact, cmp, float((selfx == xd).all(1).float().mean()),
                            float((selfn == zero).all(1).float().mean()))
        real = None
        if gate_on and REAL_MAX_T > 0:
            sel = torch.nonzero(small).flatten()
            if sel.numel():
                with torch.no_grad():                                       # its own countdown committed, then held
                    _, sp, n, held = run_hard(x[sel], C[sel], M[sel], Td, Tb, Gc, cap=REAL_MAX_T + 1)
                    ok = torch.nonzero((n > 0) & held).flatten()
                sel, sp = sel[ok], sp[ok]
            if sel.numel():
                Rr, Rrh = divide_path(sp * sp, W, M[sel], TDC, GS, Td, Tb, Gc, ctl.visits, ctl.gvisits)
                real = {"sel": sel, "R": Rr, "Rh": Rrh, "width": width[sel]}
        self.control(exact, cmp, gate_on, float(loss.detach()))
        return {"loss": loss, "real": real, "ctl": ctl}

    def control(self, exact: float, cmp: float, gate_on: bool, loss: float) -> None:
        ctl = self.ctl
        ctl.win.append(cmp if ctl.phase == "B" and not gate_on else exact)
        ctl.in_phase += 1
        mean = sum(ctl.win) / len(ctl.win)
        if ctl.step % LOG_EVERY == 0 or ctl.step == 1:
            gate = "".join(str(int(g)) for g in self.tables()[2].tolist())
            print(f"LR step={ctl.step} phase={ctl.phase} in_phase={ctl.in_phase} loss={loss:.4f} "
                  f"hard_exact={exact:.4f} cmp={cmp:.4f} win={mean:.4f} gate={gate} "
                  f"rows_visited={int((ctl.visits > 0).sum())}/200", flush=True)
        if ctl.phase == "A" and ((len(ctl.win) == STAGE_WIN and mean >= ID_SWITCH) or ctl.in_phase >= ID_STEPS):
            # the checks pass on most links: "x - N borrows" joins (and with GATE_PHASE = "B" the gate's lessons and the
            # rows' answers), with a fresh optimizer (Adam's memory of phase A's gradients would slow the repairs)
            ctl.history.append((ctl.step, "switch", ctl.in_phase, round(mean, 4)))
            print(f"LR switch_to_own_borrow step={ctl.step} win={mean:.4f}", flush=True)
            ctl.phase, ctl.reset, ctl.in_phase = "B", True, 0
            ctl.win.clear()
        elif ctl.phase == "B" and not gate_on:
            if (len(ctl.win) == STAGE_WIN and mean >= CMP_SWITCH) or ctl.in_phase >= CMP_STEPS:
                # the borrows say "below N" where they should: the gate can learn what to do with them
                ctl.history.append((ctl.step, "gate", ctl.in_phase, round(mean, 4)))
                print(f"LR switch_to_gate step={ctl.step} win={mean:.4f}", flush=True)
                ctl.phase, ctl.in_phase = "C", 0
                ctl.win.clear()
        elif gate_on:
            ctl.streak = ctl.streak + 1 if exact == 1.0 else 0
            if ctl.streak >= FREEZE_WIN:
                ctl.history.append((ctl.step, "frozen", ctl.in_phase, 1.0))
                print(f"LR frozen step={ctl.step} history={ctl.history}", flush=True)
                ctl.phase = "frozen"

    def report_tables(self, Td: Tensor, Tb: Tensor, Gc: Tensor) -> None:
        """At the first evaluation batch: the snapped tables and how often training used each row."""
        ctl = self.ctl
        print("TABLE digit=" + "".join(str(int(d)) for d in Td.tolist()), flush=True)
        print("TABLE borrow=" + "".join(str(int(d)) for d in Tb.tolist()), flush=True)
        print("TABLE gate=" + "".join(str(int(d)) for d in Gc.tolist())
              + " p_commit=" + ",".join(f"{float(p):.4f}" for p in self.gate_soft().tolist()), flush=True)
        if ctl.visits is not None:
            v = ctl.visits
            print(f"TABLE train_visits min={int(v.min())} rows_never={int((v == 0).sum())} "
                  f"counts={','.join(str(int(c)) for c in v.tolist())}", flush=True)
            print(f"TABLE gate_train_visits={','.join(str(int(c)) for c in ctl.gvisits.tolist())}", flush=True)
            print(f"TABLE countdown_train_rows={','.join(str(int(c)) for c in ctl.cvisits.tolist())}", flush=True)
        print(f"TABLE schedule phase={ctl.phase} step={ctl.step} history={ctl.history}", flush=True)

    def audit(self) -> None:
        """Rows used at evaluation so far: all of them, the countdown's, the gate's; and which were never trained."""
        ctl = self.ctl
        used = (ctl.used + ctl.used_count) > 0
        count = ctl.used_count > 0
        gate = ctl.used_gate > 0
        key = (int(used.sum()), int(count.sum()), int(gate.sum()))
        if key == ctl.reported:
            return
        ctl.reported = key
        if ctl.visits is None:
            print(f"AUDIT eval_rows_used={key[0]}/200 (never trained: no training ran)", flush=True)
            return
        never = lambda u, v: int((u & (v == 0)).sum())
        rows = [(i // 20, (i // 2) % 10, i % 2) for i in torch.nonzero(count).flatten().tolist()]
        unmet = [(i // 20, (i // 2) % 10, i % 2)
                 for i in torch.nonzero(count & (ctl.cvisits == 0)).flatten().tolist()]
        print(f"AUDIT eval_rows_used={key[0]}/200 used_but_never_trained={never(used, ctl.visits)}", flush=True)
        print(f"AUDIT countdown_rows_used={key[1]} used_but_never_trained={never(count, ctl.visits)} "
              f"min_train_visits={int(ctl.visits[count].min())} not_met_by_the_countdown_in_training={len(unmet)} "
              f"rows={rows} unmet={unmet}", flush=True)
        print(f"AUDIT gate_rows_used={key[2]}/2 used_but_never_trained={never(gate, ctl.gvisits)} "
              f"train_visits={ctl.gvisits.tolist()}", flush=True)


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


def build_model(spec: ModelSpec) -> LearnedControl:
    model = LearnedControl(spec)
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
