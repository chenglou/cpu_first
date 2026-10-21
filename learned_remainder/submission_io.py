"""LearnedIO: submission_chain.py (x^(2^T) mod N, the reduction a chain of one learned subtraction table) with the prompt
reading and the answer writing learned as well: no `parse`, no `token - 7`, no designed right-alignment.

The I/O (designed grammar, learned contents):
  * a field is read by a pointer: an anchor (the position of a token the field chose, or the prompt's end) and a step
    (+1 or -1); slot j reads the token at anchor + (j + 1) * step, while every token from the anchor's neighbour to it is
    a digit (a learned digit-ness per token); the slots after the first non-digit read 0. One shared step per field is
    what lets slot 7 (an 8-digit N, x or answer, never seen in training) work like the others;
  * the machine has three input fields, N (8 slots), x (8 slots) and T (2 slots), and one output field (8 slots): the
    writer puts the state's slot j at the position its own pointer gives slot j;
  * a token's digit value (the code): per candidate reading (which digit run the N and x fields read, in which
    direction: 24 hypotheses), one learned score per token, and a digit token's value is the rank of its score among
    the digit tokens. The reader uses the code of the hypothesis its snapped pointers make;
  * the writer's code: value -> token logits, plus the log digit-ness of the token (only digits are written).
Everything is snapped at evaluation (argmax pointers, signs of the digit-ness, ranks, argmax writer) and exact.

What teaches the I/O (see IO.md):
  1. the answer round trip: read the answer (the label) at the writer's positions and write it back; per row a mixture
     over the writer's programs, each of which must reproduce every answer token. It pins the writer's pointer (the
     prompt's end, step -1: right-aligned, lowest digit last) and which tokens are digits;
  2. the code round trips, against the reader's code held fixed: read an answer token and write it back; write the
     machine's own numbers (the digits of its square of x) and read them back. They teach the writer's code;
  3. per hypothesis, Bradley-Terry on "x < N" (x is a residue mod N): at the highest place where the two readings
     differ, x's digit is the smaller one. Every one of the 45 digit pairs is such a place in training, so for the true
     reading the only consistent ranking is the identity; wrong readings cannot be ranked consistently;
  4. speculation: per row a mixture over the complete readings (fields -> distinct whole digit runs, directions),
     normalized over those readings, each scored by its hypothesis's exact verdict on x < N. This picks the reading;
     then the tiling (each field reads one whole run, T's two slots only a short one) pins the pointers;
  5. T's direction: no label-free lesson tells "16" from "61"; once the machine is exact, the T = 16 rows run the
     machine under both readings of T and the answers keep the one that reproduces them (a crisp mixture).
The machine's own lessons are submission_chain.py's, fed by the learned reader; its phase B (which relies on x < N and
on the answers) waits until the reading is chosen, and the rows' answers are read by the learned answer reader.

Datapath (designed, as in submission_chain.py): T squarings (designed integer product), each reduced mod N by
restoring division whose every subtraction is one link of the learned subtractor table over N's read digits.
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

MODE = "learned"       # "learned" | "designed" (the exact subtraction table: the plumbing check)
IO_MODE = "learned"    # "learned" | "designed" (the I/O parameters set at the true reading: the plumbing check)
W = 8                  # digits of the widest modulus (a 24-bit N is below 10^8); one reduction takes W steps
P = W + 1              # register places: 10 R + v_t < 10 N < 10^9
S = 10                 # trials per step: trial s is s links down the chain (Rp - s*N until a link borrows)
SLOTS = (W, W, 2, W)   # the fields' slots: N, x, T; the writer
INIT = 0.1             # std of the initial logits
LR = 0.05              # Adam on the logits
ID_SWITCH = 0.8        # phase B (the other lessons join) once this share of the link checks passes, averaged ...
STAGE_WIN = 20         # ... over this many steps,
ID_STEPS = 300         # ... or after this many steps (and in both cases only once the reading is chosen)
ID_PLACE0 = 5.0        # weight of the lowest place in the check (the swapped borrow code fails it only there)
ID_BORROW = 1.0        # weight of the final borrow in the check: the complete identity (D - N) + N = D
SELF_LESSON = True     # phase B: also "x mod N = x" (each row's x divided by N, soft own-borrow choice)
SELF_STEPS = 1         # ... in this many steps, from x // 10^SELF_STEPS (one step: the chain must stop at x itself)
REAL_MAX_T = 2         # phase B also trains on the batch's real rows with T <= REAL_MAX_T (0: never sees a label)
FREEZE_WIN = 200       # phase B: freeze after this many consecutive steps with every check exact
BT_TAU = 1.0           # Bradley-Terry temperature of the rank codes
PICK_PASS = 0.999      # the reading is chosen once a hypothesis passes x < N on this share of the batch ...
PICK_WIN = 20          # ... for this many steps in a row
CHOSEN = 0.99          # then the tiling joins once the mixture puts this posterior on one reading
T_AFTER = 20           # T's direction: speculate once the machine has been exact this many steps in a row ...
T_ROWS = 64            # ... on at most this many rows whose two readings of T differ
T_DONE = 0.999         # ... until the snapped reading reproduces their answers and carries this weight
BATCH = 512
EVAL_BATCH = 16384
MAX_STEPS = 20000
LOG_EVERY = 250
EPS = 1e-6

POW10 = [10 ** i for i in range(19)]
PERMS = list(itertools.permutations(range(3)))
HYP = [(rn, dn, rx, dx) for rn in range(3) for rx in range(3) if rn != rx for dn in range(2) for dx in range(2)]
HIDX = {h: i for i, h in enumerate(HYP)}  # (N's run, N's direction, x's run, x's direction); direction 0: step +1


def to_digits(v: Tensor, n: int) -> Tensor:
    """Non-negative int64 values -> their n lowest decimal digits, LSB first (the integer square's designed I/O)."""
    return (v[..., None] // torch.tensor(POW10[:n], device=v.device)) % 10


def from_digits(d: Tensor) -> Tensor:
    return (d * torch.tensor(POW10[:d.shape[-1]], device=d.device)).sum(-1)


def exact_tables(device) -> tuple[Tensor, Tensor]:
    """The designed subtractor (the plumbing check only): digit and borrow of r - m - c, flat over (r, m, c)."""
    r = torch.arange(10, device=device)[:, None, None]
    m = torch.arange(10, device=device)[None, :, None]
    c = torch.arange(2, device=device)[None, None, :]
    x = r - m - c
    return (x % 10).flatten(), (x < 0).long().flatten()


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
    for j in range(P):
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
    for j in range(P):
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
    for j in range(P):
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = (joint @ TDC).view(X, 10, 12)                                       # every m at once
        Y = Y.gather(1, M[:, j, None, None].expand(X, 1, 12)).squeeze(1)        # N's own digit
        out.append(Y[:, :10])
        c = Y[:, 10:]
    return torch.stack(out, 1), c


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
    nll = -R.clamp_min(1e-12).log().gather(-1, target[..., None]).squeeze(-1)
    return (nll * scored).sum() / scored.sum().clamp_min(1)


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Control:
    """The training schedule (Python-side, not model state): the I/O stages, the machine's phases A, B, frozen."""

    def __init__(self) -> None:
        self.step = 0
        self.phase = "frozen" if MODE == "designed" and IO_MODE == "designed" else "A"
        self.io = 5 if IO_MODE == "designed" else 1   # 1 answers, 2 ranks, 3 reading, 4 tiling, 5 T's direction done
        self.in_phase = 0
        self.streak = 0
        self.pick = 0                         # steps in a row with a hypothesis passing x < N
        self.chosen = 0.0                     # the mixture's mean posterior on its favourite reading
        self.io_exact = False                 # the I/O's own hard checks, this step
        self.reset = False                    # the scheduler clears Adam's state at the switch
        self.win: deque = deque(maxlen=STAGE_WIN)
        self.visits: Tensor | None = None     # table rows used by training's hard checks (the audit)
        self.used: Tensor | None = None       # table rows used at evaluation
        self.reported = -1
        self.history: list = []
        self.t_log = ""


class LearnedIO(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        V = spec.vocab_size
        self.config = Config(V, spec.max_seq_len)
        self.V, self.END, self.OUT = V, V, V + 1                             # virtual tokens: the end, beyond it
        self.a = nn.Parameter(INIT * torch.randn(10, 10, 2, 10))           # subtractor: (r, m, borrow in) -> digit
        self.b = nn.Parameter(INIT * torch.randn(10, 10, 2, 2))            # ... -> borrow out
        self.dg = nn.Parameter(INIT * torch.randn(V))                      # token -> is a digit
        self.u = nn.Parameter(INIT * torch.randn(len(HYP), V))             # per reading hypothesis: token scores
        self.wc = nn.Parameter(INIT * torch.randn(10, V))                  # writer: value -> token
        self.anc = nn.Parameter(INIT * torch.randn(4, V + 1))              # field anchor: a token id, or the end
        self.strd = nn.Parameter(INIT * torch.randn(4, 2))                 # field step: +1 | -1
        self.ctl = Control()
        if IO_MODE == "designed":
            self.designed_io()

    @torch.no_grad()
    def designed_io(self) -> None:
        """The true reading (the plumbing check only): N after its marker, x before T's marker, T and the answer at the
        end, reading leftwards; digits 7..16 valued 0..9."""
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

    def tables(self) -> tuple[Tensor, Tensor]:
        if MODE == "designed":
            return exact_tables(self.a.device)
        return self.a.argmax(-1).flatten(), self.b.argmax(-1).flatten()

    # ------------------------------------------------------------------ the I/O pieces
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

    def anchor(self, f: int, e: Tensor, end: Tensor) -> Tensor:
        """Field f's soft anchor over the prompt's positions and its end [X, Lp] (zero elsewhere)."""
        lg = self.anc[f].float()[e.clamp(max=self.END)]
        return lg.masked_fill(~self.valid_anchor(e, end), -1e9).softmax(-1)

    def hard_pos(self, f: int, e: Tensor, end: Tensor):
        """Field f's snapped pointer: its slots' positions [X, n], its anchor [X] and its step."""
        lg = self.anc[f][e.clamp(max=self.END)].masked_fill(~self.valid_anchor(e, end), -1e9)
        q = lg.argmax(-1)
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
        code = self.ranks(k)                                                 # [X, V] the selected reader's code
        Nd, width = self.read(0, e, end, code)
        Xd, _ = self.read(1, e, end, code)
        Td_, _ = self.read(2, e, end, code)
        x, T = from_digits(Xd), from_digits(Td_)
        M = F.pad(Nd, (0, P - W))                                            # N's digits as read, P places
        Td, Tb = self.tables()
        rows, L, V = input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
        if self.training:
            aux = self.lessons(x, T, width, M, Td, Tb, e, end, k, code)
            return torch.zeros(rows, L, V, device=dev), aux
        ctl = self.ctl
        if ctl.used is None:
            ctl.used = torch.zeros(200, dtype=torch.long, device=dev)
            self.report_tables(Td, Tb, e, end)
        s = self.run(x, T, M, Td, Tb, ctl.used)
        n_used = int((ctl.used > 0).sum())
        if n_used != ctl.reported:
            ctl.reported = n_used
            unvisited = int(((ctl.used > 0) & (ctl.visits == 0)).sum()) if ctl.visits is not None else -1
            print(f"AUDIT eval_rows_used={n_used}/200 used_but_never_trained={unvisited}", flush=True)
        # the learned writer: slot j of the state at the writer's slot-j position, its token by the writer's code
        pos, _, _ = self.hard_pos(3, e, end)
        p = pos - (W + 1)
        tok = self.wsm().argmax(-1)[to_digits(s, W)]
        inside = (p >= 0) & (p < L)
        logits = torch.full((rows, L, V), -20.0, device=dev)
        r = torch.arange(rows, device=dev)[:, None].expand_as(p)
        logits[r[inside], p[inside], tok[inside]] = 0.0
        return logits, None

    def run(self, x: Tensor, T: Tensor, M: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None) -> Tensor:
        """The hard machine: T squarings, each reduced by the chained long division."""
        s = x.clone()
        for t in range(int(T.max()) if T.numel() else 0):
            R = divide_hard(s * s, W, M, Td, Tb, visits)
            s = torch.where(T > t, from_digits(R), s)
        return s

    def run_many(self, x: Tensor, Ts: list, M: Tensor, Td: Tensor, Tb: Tensor) -> list:
        """The hard machine once, to the largest count: the state after each row's Ts[h] squarings, for every h."""
        s, outs = x.clone(), [torch.where(Th == 0, x, 0) for Th in Ts]
        for t in range(max(int(Th.max()) for Th in Ts)):
            s = from_digits(divide_hard(s * s, W, M, Td, Tb))
            outs = [torch.where(Th == t + 1, s, o) for Th, o in zip(Ts, outs)]
        return outs

    # ------------------------------------------------------------------ lessons
    def lessons(self, x, T, width, M, Td, Tb, e, end, k, code) -> dict:
        """The machine's label-free lessons (submission_chain.py's, on the learned reader's numbers) and the I/O's
        label-free ones; the label-dependent ones run in token_training_loss on what this returns:
          * every link of the machine's own division of each row's x^2 checks itself, (D - N) + N = D, with the learned
            subtractor and a designed adder: the digits and the final borrow against the adder's final carry;
          * from phase B on (once the checks pass and the reading is chosen): "x mod N = x" and the real rows' answers;
          * the I/O: the rank codes' Bradley-Terry on x < N, the speculative choice of the reading, the tiling, and the
            writer's code round trip on the machine's own numbers (see the module docstring)."""
        ctl = self.ctl
        ctl.step += 1
        dev = x.device
        if ctl.visits is None:
            ctl.visits = torch.zeros(200, dtype=torch.long, device=dev)
        common = {"ctl": ctl, "model": self, "e": e, "end": end, "k": k, "code": code, "x": x, "T": T, "M": M,
                  "width": width, "Td": Td, "Tb": Tb, "real": None}
        if ctl.phase == "frozen":
            return {"loss": self.a.sum() * 0.0 + self.dg.sum() * 0.0, **common}
        X = x.shape[0]
        loss = self.a.sum() * 0.0
        exact = 1.0
        if MODE == "learned":
            TDC = torch.cat([self.a.float().softmax(-1), self.b.float().softmax(-1)], -1)
            TDC = TDC.permute(0, 2, 1, 3).reshape(20, 120)
            with torch.no_grad():
                D, live = link_inputs(x * x, W, M, Td, Tb)
                D, Mr = D.reshape(X * W * (S - 1), P), M.repeat_interleave(W * (S - 1), 0)
            Ds, bs = link_soft(F.one_hot(D, 10).float(), Mr, TDC)
            back, carry = add_soft(Ds, Mr)                                   # (D - N) + N on distributions
            weight = torch.ones(P, device=dev)
            weight[0] = ID_PLACE0                                           # the lowest place tells the swapped code
            nll = (-back.clamp_min(1e-12).log().gather(-1, D[..., None]).squeeze(-1) * weight).sum(-1)
            if ID_BORROW > 0:                                               # final borrow = the adder's final carry
                agree = carry * bs[:, 1] + (1 - carry) * bs[:, 0]
                nll = nll - ID_BORROW * agree.clamp_min(1e-12).log()
            loss = nll.mean() / (weight.sum() + ID_BORROW)
            xd = to_digits(x, P)
            selfl = SELF_LESSON and ctl.phase == "B"
            if selfl:
                loss = loss + ce(divide_soft(x, SELF_STEPS, M, TDC), xd, width)
            with torch.no_grad():
                Dh, bh, idx = link_hard(D, Mr, Td, Tb)
                ctl.visits += torch.bincount(idx.flatten(), minlength=200)
                backh, carryh = add_hard(Dh, Mr)
                ok = (backh == D).all(-1)
                if ID_BORROW > 0:
                    ok = ok & (carryh == bh)
                exact = float(ok.float().mean())
                if selfl:
                    exact = min(exact, float((divide_hard(x, SELF_STEPS, M, Td, Tb) == xd).all(1).float().mean()))
            if ctl.phase == "B" and REAL_MAX_T > 0:
                sel = torch.nonzero(T <= REAL_MAX_T).flatten()
                if sel.numel():
                    with torch.no_grad():
                        sp = self.run(x[sel], T[sel] - 1, M[sel], Td, Tb)
                    Rr, Rrh = divide_path(sp * sp, W, M[sel], TDC, Td, Tb, ctl.visits)
                    common["real"] = {"sel": sel, "R": Rr, "Rh": Rrh, "width": width[sel]}
        if IO_MODE == "learned":
            loss = loss + self.io_lessons(e, end, k, code, x)
        common["exact"] = exact
        return {"loss": loss, **common}

    def io_lessons(self, e: Tensor, end: Tensor, k: Tensor, code: Tensor, x: Tensor) -> Tensor:
        """The I/O's label-free lessons: the rank codes (Bradley-Terry on x < N, every hypothesis, from stage 2), the
        reading (the normalized mixture over complete readings, from stage 3), the tiling (from stage 4) and the
        writer's code round trip on the machine's own numbers (from stage 3)."""
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
            # the writer's code: write the machine's own numbers (its square of x), read them back
            with torch.no_grad():
                vals = to_digits((x * x) % POW10[W], W)
                inv = torch.zeros(X, 10, dtype=torch.long, device=dev)
                for t in torch.nonzero(isd).flatten().tolist():
                    inv.scatter_(1, code[:, t:t + 1], t)                    # the token whose code is v
            rv = self.wsm()[vals, inv.gather(1, vals)]
            loss = loss - rv.clamp_min(EPS).log().mean()
        return loss

    def io_label_lessons(self, aux: dict, labels: Tensor, valid: Tensor, tpos: Tensor) -> Tensor:
        """The I/O's label lessons: the answer round trip (the writer's pointer, digit-ness), the token-side code round
        trip (the writer's code) and, once the machine is exact, T's direction by speculation."""
        ctl = self.ctl
        e, end, code = aux["e"], aux["end"], aux["code"]
        X, Lp = e.shape
        dev = e.device
        labe = self.label_seq(e, labels, valid, tpos)
        wsm = self.wsm()
        dl = torch.where(labe < self.V, torch.sigmoid(self.dg.float()[labe.clamp(max=self.V - 1)]), torch.zeros((), device=dev))
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
            lp = (tgt * back.clamp_min(EPS).log()).sum(-1) + (valid.sum(1)[:, None] - tgt.sum(-1)) * float(torch.log(torch.tensor(EPS)))
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
        if ctl.io == 4 and ctl.phase == "B" and ctl.streak >= T_AFTER:
            loss = loss + self.t_lesson(aux, yd)
        return loss, yd

    def t_lesson(self, aux: dict, yd: Tensor) -> Tensor:
        """T's direction by speculation: on rows whose two readings of T differ, the exact machine runs both, and the
        answers keep the reading that reproduces them (a mixture over T's two whole-run programs, normalized)."""
        ctl = self.ctl
        e, end, code, x, M = aux["e"], aux["end"], aux["code"], aux["x"], aux["M"]
        X, Lp = e.shape
        dev = e.device
        ok, si, ti, length = self.runs(e, end)
        rn, _, _ = self.which_run(0, e, end, si, ti, ok, length)
        rx, _, _ = self.which_run(1, e, end, si, ti, ok, length)
        rt = (3 - rn - rx).clamp(0, 2)                                      # T's run
        a0, b0 = si.gather(1, rt[:, None])[:, 0], ti.gather(1, rt[:, None])[:, 0]
        j = torch.arange(2, device=dev)
        tp = e.gather(1, (a0[:, None] + j).clamp(max=Lp - 1))
        tm = e.gather(1, (b0[:, None] - j).clamp(min=0))
        lp = j[None] < (b0 - a0 + 1)[:, None]
        Tp = from_digits(torch.where(lp, code.gather(1, tp.clamp(max=self.V - 1)), 0))
        Tm = from_digits(torch.where(lp, code.gather(1, tm.clamp(max=self.V - 1)), 0))
        rows = torch.nonzero(ok & (Tp != Tm)).flatten()[:T_ROWS]
        if rows.numel() == 0:
            return self.strd.sum() * 0.0
        Td, Tb = aux["Td"], aux["Tb"]
        with torch.no_grad():
            sp, sm = self.run_many(x[rows], [Tp[rows], Tm[rows]], M[rows], Td, Tb)
            y = from_digits(yd[rows])
            Lpl, Lmi = (sp == y).float(), (sm == y).float()
        A = self.anchor(2, e, end)[rows]
        sl = self.strd[2].float().softmax(-1)
        wp = A.gather(1, (a0[rows] - 1)[:, None])[:, 0] * sl[0]                # read from the run's left end, +1
        wm = A.gather(1, (b0[rows] + 1)[:, None])[:, 0] * sl[1]                # read from its right end, -1
        hit = (Lpl + Lmi) > 0
        lt = -(((wp * Lpl + wm * Lmi).clamp_min(1e-30).log() - (wp + wm).clamp_min(1e-30).log()) * hit).sum() / hit.sum().clamp_min(1)
        with torch.no_grad():
            _, _, hs = self.hard_pos(2, e, end)
            mine = Lpl if hs > 0 else Lmi                                      # the snapped reading's verdict
            share = float(((wp if hs > 0 else wm) / (wp + wm).clamp_min(1e-30)).mean())
            ctl.t_log = f"rows={rows.numel()} hits={int(hit.sum())} snapped_right={float(mine.mean()):.3f} share={share:.4f}"
            if bool(mine.all()) and share >= T_DONE:
                ctl.io = 5
                ctl.history.append((ctl.step, "T_resolved", rows.numel()))
                print(f"LR io_stage=5 T_resolved step={ctl.step} {ctl.t_log}", flush=True)
        return lt

    def control(self, exact: float, loss: float) -> None:
        ctl = self.ctl
        ctl.win.append(exact)
        ctl.in_phase += 1
        mean = sum(ctl.win) / len(ctl.win)
        before = ctl.io
        if ctl.io == 1 and getattr(ctl, "rows3", 0.0) >= 0.99:
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
        if ctl.step % LOG_EVERY == 0 or ctl.step == 1:
            print(f"LR step={ctl.step} phase={ctl.phase} io={ctl.io} in_phase={ctl.in_phase} loss={loss:.4f} "
                  f"hard_exact={exact:.4f} win={mean:.4f} io_exact={ctl.io_exact} chosen={ctl.chosen:.4f} "
                  f"rows_visited={int((ctl.visits > 0).sum())}/200 {ctl.t_log}", flush=True)
        if (ctl.phase == "A" and ctl.io >= 4
                and ((len(ctl.win) == STAGE_WIN and mean >= ID_SWITCH) or ctl.in_phase >= ID_STEPS)):
            # the checks pass on most links and the reading is chosen: "x mod N = x" and the rows' answers join, with
            # a fresh optimizer (Adam's memory of phase A's gradients would slow the repairs)
            ctl.history.append((ctl.step, "switch", ctl.in_phase, round(mean, 4)))
            print(f"LR switch_to_own_borrow step={ctl.step} win={mean:.4f}", flush=True)
            ctl.phase, ctl.reset, ctl.in_phase = "B", True, 0
            ctl.win.clear()
        elif ctl.phase == "B":
            ctl.streak = ctl.streak + 1 if exact == 1.0 and ctl.io_exact else 0
            if ctl.streak >= FREEZE_WIN and ctl.io == 5:
                ctl.history.append((ctl.step, "frozen", ctl.in_phase, 1.0))
                print(f"LR frozen step={ctl.step} history={ctl.history}", flush=True)
                ctl.phase = "frozen"

    def report_tables(self, Td: Tensor, Tb: Tensor, e: Tensor, end: Tensor) -> None:
        """At the first evaluation batch: the snapped table and I/O, and how often training's hard checks used each
        table row."""
        ctl = self.ctl
        print("TABLE digit=" + "".join(str(int(d)) for d in Td.tolist()), flush=True)
        print("TABLE borrow=" + "".join(str(int(d)) for d in Tb.tolist()), flush=True)
        if ctl.visits is not None:
            v = ctl.visits
            print(f"TABLE train_visits min={int(v.min())} rows_never={int((v == 0).sum())} "
                  f"counts={','.join(str(int(c)) for c in v.tolist())}", flush=True)
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
        if IO_MODE == "learned":
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
        model.control(aux["exact"], float(loss.detach()))
        return loss


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


def build_model(spec: ModelSpec) -> LearnedIO:
    model = LearnedIO(spec)
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
