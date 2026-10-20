"""LearnedChain: x^(2^T) mod N whose mod-N reduction is a restoring long division built only from a learned subtraction
table: no designed multiples of N.

Datapath (designed wiring, the CPU's schoolbook algorithm in base 10):
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
So every arithmetic step of the reduction is the table (submission.py's trials subtract the exact digits of s*N, made by
designed integer multiplication). Learned (MODE = "learned"): the subtractor cell, a 200-row table of digit and borrow
logits. Evaluation snaps the cell to its argmax table and runs the datapath on integers. MODE = "designed" uses the
exact subtraction table instead (the plumbing check).

What teaches the cell (submission.py's Stage 2 lessons, moved onto single links; see CHAIN.md for what broke and why):
  * phase A: every link of the machine's own division of each row's x^2 checks itself, (D - N) + N = D, with the
    learned subtractor and a designed adder (label-free; D is the link's input: the register Rp or the previous trial).
    The check is complete (ID_BORROW): the P digits must come back AND the subtraction's final borrow must equal the
    adder's final carry, the one output of a link that the digits alone do not check (it is the stop rule's signal;
    without it a third of the seeds stalled with rows (0, 0, 1) and (9, 0, 1) wrong). The lowest place weighs ID_PLACE0
    times the others: it is the one place where the true borrow code and the swapped one (r - m - (1 - c), the two
    borrow symbols exchanged: a consistent subtractor of D - N - 1) differ;
  * phase B, once ID_SWITCH of the checks pass or after ID_STEPS steps: also "x mod N = x" (x < N, so one division step
    from x // 10 with the soft own-borrow choice must stop the chain at trial 0: the first link must borrow), and the
    answers of the batch's rows with T <= REAL_MAX_T along the model's own hard path (below).
The own path: each step's chain length (the quotient digit) is chosen by the hard datapath (argmax table, stop at the
first borrow), and the soft cell runs that many links on digit distributions, from the soft register. The rows'
dividend is the machine's own s_{T-1}^2 and the answer is the row's label. Adam's state is reset at the switch to
phase B; the model freezes (learning rate 0) after FREEZE_WIN consecutive steps in which every check is exact on the
hard datapath. The schedule is Python-side state of the model (not model state). Loss: cross-entropy of the soft
remainder's digits below N's width. Everything runs in fp32 with autocast off; evaluation is integer arithmetic.
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
ID_LINKS = "all"       # which links check themselves: "all" (every link computed, also those after the chain first
                       # borrows, whose inputs are wrapped numbers the table itself made) | "live" (the links the machine
                       # relies on: up to and including the first that borrows)
ID_BORROW = 1.0        # weight of the final borrow in the check (0: off, the first recipe): the complete identity
                       # (D - N) + N = D also needs the subtraction's final borrow to equal the designed adder's final
                       # carry (D - N = digits - borrow * 10^P and digits + N = D + carry * 10^P)
SELF_LESSON = True     # phase B: also "x mod N = x" (each row's x divided by N, soft own-borrow choice)
SELF_STEPS = 1         # ... in this many steps, from x // 10^SELF_STEPS (one step: the chain must stop at x itself)
REAL_MAX_T = 2         # phase B also trains on the batch's real rows with T <= REAL_MAX_T (0: never sees a label)
FREEZE_WIN = 200       # phase B: freeze after this many consecutive steps with every check exact
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


class LearnedChain(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.a = nn.Parameter(INIT * torch.randn(10, 10, 2, 10))           # subtractor: (r, m, borrow in) -> digit
        self.b = nn.Parameter(INIT * torch.randn(10, 10, 2, 2))            # ... -> borrow out
        self.ctl = Control()

    def tables(self) -> tuple[Tensor, Tensor]:
        if MODE == "designed":
            return exact_tables(self.a.device)
        return self.a.argmax(-1).flatten(), self.b.argmax(-1).flatten()

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
          * every link of the machine's own division of each row's x^2 checks itself, (D - N) + N = D, with the learned
            subtractor and a designed adder: the digits, and (ID_BORROW) the final borrow against the adder's final
            carry. This pins the digits, but it has a saddle at the borrow of (r, r, borrow in 1) (the adder's carry
            makes up for a wrong digit), so alone it is slow;
          * from phase B on (once the checks pass on ID_SWITCH of the links): x is its own remainder (x < N). Dividing x
            by N with the soft own-borrow choice must stop the chain at trial 0 (SELF_STEPS steps from
            x // 10^SELF_STEPS), so the first link must end in a borrow. This teaches the borrows the check is slow on;
          * phase B also adds the real rows' answers along the own path (REAL_MAX_T > 0)."""
        ctl = self.ctl
        ctl.step += 1
        dev = x.device
        if ctl.visits is None:
            ctl.visits = torch.zeros(200, dtype=torch.long, device=dev)
        if ctl.phase == "frozen":
            return {"loss": self.a.sum() * 0.0, "real": None, "ctl": ctl}
        TDC = torch.cat([self.a.float().softmax(-1), self.b.float().softmax(-1)], -1)
        TDC = TDC.permute(0, 2, 1, 3).reshape(20, 120)
        X = x.shape[0]
        with torch.no_grad():
            D, live = link_inputs(x * x, W, M, Td, Tb)
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
        real = None
        if ctl.phase == "B" and REAL_MAX_T > 0:
            sel = torch.nonzero(T <= REAL_MAX_T).flatten()
            if sel.numel():
                with torch.no_grad():
                    sp = self.run(x[sel], T[sel] - 1, M[sel], Td, Tb)
                Rr, Rrh = divide_path(sp * sp, W, M[sel], TDC, Td, Tb, ctl.visits)
                real = {"sel": sel, "R": Rr, "Rh": Rrh, "width": width[sel]}
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
            ctl.phase, ctl.reset, ctl.in_phase = "B", True, 0
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


def build_model(spec: ModelSpec) -> LearnedChain:
    model = LearnedChain(spec)
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
