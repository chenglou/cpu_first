"""LearnedRemainder: x^(2^T) mod N whose mod-N reduction is a long division with a learned subtraction table.

Datapath (designed wiring, the CPU's schoolbook algorithm in base 10):
  * read N (its digits come LSB first), x and T from the prompt;
  * loop T times: square the state (designed integer product), then reduce it mod N by long division:
      - the running remainder R has P = W + 1 digit places (W = 8 digits of the widest 24-bit modulus); it starts at the
        dividend's top digits v // 10^W (below N) and takes W steps, each shifting in the next dividend digit,
        Rp = 10 R + v_t;
      - every step tries all ten quotient digits at once: trial s subtracts the exact digits of s*N from Rp, place by
        place from the lowest, with ONE shared subtractor cell (r, m, borrow in) -> (digit, borrow out);
      - the choice keeps one trial: the largest s whose trial ends without a borrow (the subtraction's own borrow);
  * write the state's digits right-aligned on the prompt.
Learned (MODE = "learned"): the subtractor cell, a 200-row table of digit and borrow logits (and, for LESSON "madeup"
or "rows", the estimator that bootstraps it). Evaluation snaps the cell to its argmax table and runs the datapath on
integers. MODE = "designed" uses the exact subtraction table instead (Stage 0: the
pipeline with the remainder designed).

What teaches the cell (LESSON; see LEARNED_REMAINDER.md for what broke and why each piece is there):
  * "identity" (Stage 2, the default; nothing made up, the inputs are the rows' and the machine's own):
      - phase A: every trial of the machine's own division of each row's x^2 checks itself, (Rp - s*N) + s*N = Rp,
        with the learned subtractor and a designed adder (label-free). The lowest place weighs ID_PLACE0 times the
        others: it is the one place where the true borrow code and the swapped one (r - m - (1 - c), the two borrow
        symbols exchanged: a consistent subtractor of r - m - 1) differ;
      - phase B, once ID_SWITCH of the checks pass or after ID_STEPS steps: also "x mod N = x" (x < N, so one division
        step from x // 10 with the soft own-borrow choice must keep trial 0: it teaches the borrows that the identity
        cannot, at its saddle rows (r, r, borrow 1)), and the answers of the batch's rows with T <= REAL_MAX_T along
        the model's own hard path (below);
  * "madeup" (Stage 1; GRAY AREA: dividends made up inside the forward pass from each row's own N and x,
    v = q*N + x with q a random k-digit number, designed product and sum; x < N, so the remainder is x, read from the
    prompt). Stage k = 1, ..., W divides in k steps from v // 10^k:
      - phase A: the estimator's soft choice (a table over the top KR places of Rp and the top KN digits of N ->
        trial), from the first step; the remainder's lowest place weighs PLACE0 times the others (the swapped code
        again); ends once the hard datapath is STAGE_EXACT exact on the stage's lessons over STAGE_WIN steps;
      - phase B: the own path, stages continuing to k = W; at k = W the rows with T <= REAL_MAX_T join;
  * "rows" (a Stage 2 experiment that fails: the T = 1 rows' own divisions only, staged by their quotient length).
The own path: each step's trial is chosen by the hard datapath (argmax table, own-borrow choice) and the soft cell is
trained along the chosen trials only; a soft choice teaches wrong trials' lessons that are correlated with the answer.
The rows' dividend is the machine's own s_{T-1}^2 and the answer is the row's label. Adam's state is reset at the
switch to phase B; the model freezes (learning rate 0) after FREEZE_WIN consecutive steps in which every check (every
full-size lesson for "madeup") is exact on the hard datapath. The schedule is Python-side state of the model (not model
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

MODE = "learned"       # "learned" | "designed" (the exact subtraction table: Stage 0)
LESSON = "identity"    # what teaches the cell: "identity" (Stage 2: the label-free check (Rp - s*N) + s*N = Rp on
                       # every trial of the machine's own division of each row's x^2, with a designed adder; no made-up
                       # numbers) | "madeup" (Stage 1: dividends q*N + x made up from the rows' N and x) | "rows" (only
                       # the T = 1 rows' own divisions, x^2 against the label, staged by each row's quotient length; the
                       # schedule then reads the labels' exactness through the loss; it fails)
DIGIT, MARK_N, MARK_X, MARK_T = 7, 2, 3, 4  # token ids: digit d is DIGIT + d
W = 8                  # digits of the widest modulus (a 24-bit N is below 10^8); one reduction takes W steps
P = W + 1              # register places: 10 R + v_t < 10 N < 10^9
S = 10                 # trials per step: trial s subtracts s*N
KR, KN = 3, 2          # estimator window: top KR places of Rp, top KN digits of N, aligned on N's own width
INIT = 0.1             # std of the initial logits
LR = 0.05              # Adam on the logits
HOLD = 0               # phase A: the first HOLD steps choose uniformly (the estimator is not consulted). 0: a
                       # uniform hold cannot tell the true borrow code from the swapped one (their losses differ by
                       # 0.04 nats), and seeds land in either
MIX_BORROW = 0.0       # phase A: this share of the soft choice is the own borrow (tried; did not stop the swapped code)
PLACE_RAMP = 0         # phase A: score the remainder's lowest 1 + step // PLACE_RAMP places only (tried; too slow)
PLACE0 = 5.0           # phase A: weight of the remainder's lowest place in the loss, the one place that tells the
                       # true borrow code from the swapped one (r - m - (1 - c), borrow symbols exchanged)
BORROW_INIT = 0.0      # initial logit bias of every row's borrow out toward symbol 0, the symbol fed into place 0
                       # (tried; did not stop the swapped code, which a random start picks in about half of the seeds)
STAGE_EXACT = 0.99     # a stage ends once the hard datapath is this exact on its lessons, averaged ...
STAGE_WIN = 20         # ... over this many steps,
STAGE_MAX = 1500       # ... or after this many steps
FREEZE_WIN = 200       # phase B: freeze after this many consecutive steps with every full-size lesson exact
REAL_MAX_T = 2         # phase B also trains on the batch's real rows with T <= REAL_MAX_T
ID_SWITCH = 0.8        # identity lessons: phase B (the rows' answers join) once this share of trials checks out,
ID_STEPS = 300         # ... or after this many identity steps (its plateau, 0.78-0.86, is reached by step ~150)
SELF_LESSON = True     # identity lessons: also "x mod N = x" (each row's x divided by N, soft own-borrow choice)
ID_PLACE0 = 5.0        # identity lessons: weight of the lowest place in the check (the swapped borrow code is a local
                       # minimum of the identity too: it fails it only at place 0)
SELF_STEPS = 1         # ... in this many steps, from x // 10^SELF_STEPS (one step: x itself against every s*N)
LESSONS = 1            # lessons per row and step (each its own random q)
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
    """The designed subtractor (Stage 0 only): digit and borrow of r - m - c, flat over (r, m, c)."""
    r = torch.arange(10, device=device)[:, None, None]
    m = torch.arange(10, device=device)[None, :, None]
    c = torch.arange(2, device=device)[None, None, :]
    x = r - m - c
    return (x % 10).flatten(), (x < 0).long().flatten()


def ripple_hard(Rp: Tensor, Md: Tensor, Td: Tensor, Tb: Tensor):
    """Every trial on integers: Rp [X, P] digits, Md [X, S, P] digits of s*N, flat tables Td, Tb [200] indexed
    (r, m, c) -> r * 20 + m * 2 + c. Returns the trials' digits [X, S, P], final borrows [X, S] and table rows
    [X, S, P]."""
    c = torch.zeros(Md.shape[:2], dtype=torch.long, device=Md.device)
    outs, rows = [], []
    for j in range(P):
        idx = (Rp[:, j, None] * 10 + Md[:, :, j]) * 2 + c
        outs.append(Td[idx])
        c = Tb[idx]
        rows.append(idx)
    return torch.stack(outs, 2), c, torch.stack(rows, 2)


def registers(v: Tensor, steps: int, Md: Tensor, Td: Tensor, Tb: Tensor) -> Tensor:
    """The hard division's register Rp = 10 R + v_t at every step [X, steps, P] (the machine's own inputs)."""
    R = to_digits(v // POW10[steps], P)
    slot = torch.arange(S, device=v.device)
    row = torch.arange(v.shape[0], device=v.device)
    out = []
    for t in reversed(range(steps)):
        Rp = torch.cat([((v // POW10[t]) % 10)[:, None], R[:, :P - 1]], 1)
        out.append(Rp)
        D, c, _ = ripple_hard(Rp, Md, Td, Tb)
        R = D[row, torch.where(c == 0, slot, -1).argmax(1)]
    return torch.stack(out, 1)


def add_hard(D: Tensor, M: Tensor) -> Tensor:
    """The designed adder on integers: digits D [..., P] + M [..., P], mod 10^P."""
    c = torch.zeros_like(D[..., 0])
    out = []
    for j in range(P):
        x = D[..., j] + M[..., j] + c
        out.append(x % 10)
        c = x // 10
    return torch.stack(out, -1)


def add_soft(D: Tensor, M: Tensor) -> Tensor:
    """The designed adder on digit distributions D [..., P, 10] plus exact digits M [..., P], mod 10^P (multilinear)."""
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
    return torch.stack(out, -2)


def divide_hard(v: Tensor, steps: int, Md: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None) -> Tensor:
    """v mod N on integers, from R = v // 10^steps (below N) in `steps` steps; the choice keeps the largest trial
    that does not borrow. Returns the remainder's digits [X, P]. `visits` accumulates every table row used."""
    R = to_digits(v // POW10[steps], P)
    slot = torch.arange(S, device=v.device)
    row = torch.arange(v.shape[0], device=v.device)
    for t in reversed(range(steps)):
        Rp = torch.cat([((v // POW10[t]) % 10)[:, None], R[:, :P - 1]], 1)
        D, c, idx = ripple_hard(Rp, Md, Td, Tb)
        R = D[row, torch.where(c == 0, slot, -1).argmax(1)]
        if visits is not None:
            visits += torch.bincount(idx.flatten(), minlength=200)
    return R


def ripple_soft(Rp: Tensor, Md: Tensor, TDC: Tensor):
    """Trials on digit distributions, the cell applied multilinearly. Rp [X, P, 10], Md [X, n, P] digits of the n
    trials, TDC [20, 120]: rows (r, c), columns (m, 10 digit + 2 borrow probabilities). Returns D [X, n, P, 10] and the
    final borrows [X, n, 2]."""
    X, n = Md.shape[:2]
    c = torch.zeros(X, n, 2, device=Rp.device)
    c[..., 0] = 1
    D = []
    for j in range(P):
        joint = (Rp[:, None, j, :, None] * c[:, :, None, :]).reshape(X * n, 20)
        Y = (joint @ TDC).view(X, n, 10, 12)                                       # every m at once
        Y = Y.gather(2, Md[:, :, j, None, None].expand(X, n, 1, 12)).squeeze(2)    # the trial's own m
        D.append(Y[..., :10])
        c = Y[..., 10:]
    return torch.stack(D, 2), c


def divide_soft(v: Tensor, steps: int, Md: Tensor, TDC: Tensor, choice: str, est=None) -> Tensor:
    """The same division on digit distributions with a soft choice: "uniform", "est" (the estimator) or "borrow"
    (the subtraction's own borrow). Returns the final remainder's digit distributions [X, P, 10]."""
    X = v.shape[0]
    R = F.one_hot(to_digits(v // POW10[steps], P), 10).float()
    for t in reversed(range(steps)):
        Rp = torch.cat([F.one_hot((v // POW10[t]) % 10, 10).float()[:, None], R[:, :P - 1]], 1)
        D, c = ripple_soft(Rp, Md, TDC)
        valid = c[..., 0]
        above = torch.flip(torch.cumprod(torch.flip(1 - valid, [1]), 1), [1])         # prod_{s' >= s} (1 - valid)
        own = valid * torch.cat([above[:, 1:], torch.ones(X, 1, device=v.device)], 1)
        own = own / own.sum(1, keepdim=True).clamp_min(1e-12)
        if choice == "uniform":
            wq = torch.full((X, S), 1.0 / S, device=v.device)
        elif choice == "est":
            wq = est(Rp)
        else:
            wq = own
        if choice != "borrow" and MIX_BORROW > 0:                          # phase A: part of the weight on the own
            wq = (1 - MIX_BORROW) * wq + MIX_BORROW * own                    # borrow (the final borrow is read)
        R = torch.einsum("xs,xspo->xpo", wq, D)
    return R


def divide_path(v: Tensor, steps: int, Md: Tensor, TDC: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None):
    """The model's own hard path, learned along: each step's trial is chosen by the hard datapath (argmax table, the
    largest trial that does not borrow), and the soft cell computes that trial on digit distributions. Returns the
    soft remainder [X, P, 10] (the gradient reaches the cell through the chosen trials only) and the hard one [X, P]."""
    Rh = to_digits(v // POW10[steps], P)
    Rs = F.one_hot(Rh, 10).float()
    slot = torch.arange(S, device=v.device)
    row = torch.arange(v.shape[0], device=v.device)
    for t in reversed(range(steps)):
        vt = (v // POW10[t]) % 10
        D, c, idx = ripple_hard(torch.cat([vt[:, None], Rh[:, :P - 1]], 1), Md, Td, Tb)
        best = torch.where(c == 0, slot, -1).argmax(1)
        Rh = D[row, best]
        if visits is not None:
            visits += torch.bincount(idx.flatten(), minlength=200)
        Rp = torch.cat([F.one_hot(vt, 10).float()[:, None], Rs[:, :P - 1]], 1)
        Rs = ripple_soft(Rp, Md[row, best][:, None], TDC)[0][:, 0]
    return Rs, Rh


def ce(R: Tensor, target: Tensor, width: Tensor, places: int = P, place0: float = 1.0) -> Tensor:
    """Cross-entropy of the remainder's digits below N's width (a remainder below N has `width` digits), and below
    `places` (the place ramp of phase A)."""
    scored = (torch.arange(P, device=R.device)[None] < width.clamp(max=places)[:, None]).float()
    if place0 != 1.0:
        scored[:, 0] *= place0
    nll = -R.clamp_min(1e-12).log().gather(-1, target[..., None]).squeeze(-1)
    return (nll * scored).sum() / scored.sum()


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Control:
    """The training schedule (Python-side, not model state): phase A stages, phase B, frozen."""

    def __init__(self) -> None:
        self.step = 0
        self.phase = "frozen" if MODE == "designed" else "A"
        self.k = 1
        self.in_stage = 0
        self.streak = 0
        self.reset = False                    # the scheduler clears Adam's state at the switch
        self.win: deque = deque(maxlen=STAGE_WIN)
        self.visits: Tensor | None = None     # table rows used by training's hard checks (the audit)
        self.used: Tensor | None = None       # table rows used at evaluation
        self.reported = -1
        self.history: list = []


class LearnedRemainder(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.a = nn.Parameter(INIT * torch.randn(10, 10, 2, 10))           # subtractor: (r, m, borrow in) -> digit
        self.b = nn.Parameter(INIT * torch.randn(10, 10, 2, 2))            # ... -> borrow out
        with torch.no_grad():                                               # every row starts out emitting the symbol
            self.b[..., 0] += BORROW_INIT                                    # fed into place 0 (see BORROW_INIT)
        if LESSON != "identity":                                            # (rule 6: every weight is trained)
            self.g = nn.Parameter(INIT * torch.randn(10 ** KN, 10 ** KR, S))  # estimator: (N window, Rp window) -> slot
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
        N, x, T = from_digits(Nd), from_digits(Xd), from_digits(Td_[:, :2])
        Md = to_digits(N[:, None] * torch.arange(S, device=dev)[None], P)     # the exact multiples s*N (designed)
        Td, Tb = self.tables()
        rows, L, V = input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
        if self.training:
            aux = self.lessons(N, x, T, Nd, width, Md, Td, Tb)
            return torch.zeros(rows, L, V, device=dev), aux
        ctl = self.ctl
        if ctl.used is None:
            ctl.used = torch.zeros(200, dtype=torch.long, device=dev)
            self.report_tables(Td, Tb)
        s = self.run(x, T, Md, Td, Tb, ctl.used)
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

    def run(self, x: Tensor, T: Tensor, Md: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None) -> Tensor:
        """The hard machine: T squarings, each reduced by the long division."""
        s = x.clone()
        for t in range(int(T.max()) if T.numel() else 0):
            R = divide_hard(s * s, W, Md, Td, Tb, visits)
            s = torch.where(T > t, from_digits(R), s)
        return s

    def estimate(self, Rp: Tensor, nidx: Tensor, rpos: Tensor) -> Tensor:
        top = Rp.gather(1, rpos[..., None].expand(-1, -1, 10))              # [X, KR, 10], the top place first
        joint = top[:, 0]
        for i in range(1, KR):
            joint = (joint[:, :, None] * top[:, i, None, :]).flatten(1)
        G = self.g.float().softmax(-1)[nidx]                                 # [X, 10^KR, S]
        return torch.bmm(joint[:, None, :], G).squeeze(1)

    def lessons(self, N, x, T, Nd, width, Md, Td, Tb) -> dict:
        ctl = self.ctl
        ctl.step += 1
        dev = N.device
        if ctl.visits is None:
            ctl.visits = torch.zeros(200, dtype=torch.long, device=dev)
        if ctl.phase == "frozen":
            return {"loss": self.a.sum() * 0.0, "real": None, "ctl": ctl}
        TDC = torch.cat([self.a.float().softmax(-1), self.b.float().softmax(-1)], -1)
        TDC = TDC.permute(0, 2, 1, 3).reshape(20, 120)
        if LESSON == "identity":
            return self.identity_lessons(N, x, T, width, Md, Td, Tb, TDC)
        if LESSON == "rows":
            return self.row_lessons(N, x, T, Nd, width, Md, Td, Tb, TDC)
        k = ctl.k
        rep = lambda t: t.repeat_interleave(LESSONS, 0)
        Nl, xl, Ml, wl, Ndl = rep(N), rep(x), rep(Md), rep(width), rep(Nd)
        v = torch.randint(0, POW10[k], Nl.shape, device=dev) * Nl + xl         # made-up dividend; remainder x
        if ctl.phase == "A":
            choice = "uniform" if ctl.step <= HOLD else "est"
        else:
            choice = "borrow"
        rpos = wl[:, None] - torch.arange(KR, device=dev)[None]
        nidx = sum(Ndl.gather(1, (wl - 1 - i)[:, None])[:, 0] * 10 ** (KN - 1 - i) for i in range(KN))
        target = to_digits(xl, P)
        if ctl.phase == "A":
            R = divide_soft(v, k, Ml, TDC, choice, lambda Rp: self.estimate(Rp, nidx, rpos))
            with torch.no_grad():
                Rh = divide_hard(v, k, Ml, Td, Tb, ctl.visits)
        else:
            R, Rh = divide_path(v, k, Ml, TDC, Td, Tb, ctl.visits)
        places = 1 + ctl.step // PLACE_RAMP if (PLACE_RAMP and ctl.phase == "A") else P
        loss = ce(R, target, wl, places, PLACE0 if ctl.phase == "A" else 1.0)
        exact = float((Rh == target).all(1).float().mean())
        real = None
        if ctl.phase == "B" and k == W:
            sel = torch.nonzero(T <= REAL_MAX_T).flatten()
            if sel.numel():
                with torch.no_grad():
                    sp = self.run(x[sel], T[sel] - 1, Md[sel], Td, Tb)
                Rr, Rrh = divide_path(sp * sp, W, Md[sel], TDC, Td, Tb, ctl.visits)
                real = {"sel": sel, "R": Rr, "Rh": Rrh, "width": width[sel]}
        self.control(exact, float(loss.detach()))
        return {"loss": loss, "real": real, "ctl": ctl}

    def identity_lessons(self, N, x, T, width, Md, Td, Tb, TDC) -> dict:
        """Stage 2, label-free lessons on the rows' own numbers (nothing made up):
          * every trial of the machine's own division of each row's x^2 checks itself, (Rp - s*N) + s*N = Rp, with the
            learned subtractor and a designed adder. This pins the digits, but it has a saddle at the borrow of
            (0, 0, borrow in 1) (the adder's carry makes up for a wrong digit), so alone it stalls;
          * from phase B on (once the identity checks out on ID_SWITCH of the trials): x is its own remainder (x < N).
            Dividing x by N with the soft own-borrow choice must keep slot 0 at every step (SELF_STEPS steps from
            x // 10^SELF_STEPS), so every larger trial must end in a borrow. This teaches the borrows the identity
            cannot (from the start it biases every borrow);
          * phase B can add the real rows' answers along the own path, as in Stage 1 (REAL_MAX_T > 0)."""
        ctl = self.ctl
        X = N.shape[0]
        with torch.no_grad():
            Rp = registers(x * x, W, Md, Td, Tb).reshape(X * W, P)
        M = Md.repeat_interleave(W, 0)                                       # [X W, S, P]
        D, _ = ripple_soft(F.one_hot(Rp, 10).float(), M, TDC)
        back = add_soft(D, M)                                                # [X W, S, P, 10]
        target = Rp[:, None, :].expand(-1, S, -1)
        weight = torch.ones(P, device=N.device)
        weight[0] = ID_PLACE0                                               # the lowest place tells the swapped code
        nll = -back.clamp_min(1e-12).log().gather(-1, target[..., None]).squeeze(-1)
        loss = (nll * weight).sum(-1).mean() / weight.sum()
        xd = to_digits(x, P)
        selfl = SELF_LESSON and ctl.phase == "B"
        if selfl:
            loss = loss + ce(divide_soft(x, SELF_STEPS, Md, TDC, "borrow"), xd, width)
        with torch.no_grad():
            Dh, _, idx = ripple_hard(Rp, M, Td, Tb)
            ctl.visits += torch.bincount(idx.flatten(), minlength=200)
            exact = float((add_hard(Dh, M) == target).all(-1).float().mean())
            if selfl:
                exact = min(exact, float((divide_hard(x, SELF_STEPS, Md, Td, Tb) == xd).all(1).float().mean()))
        real = None
        if ctl.phase == "B" and REAL_MAX_T > 0:
            sel = torch.nonzero(T <= REAL_MAX_T).flatten()
            if sel.numel():
                with torch.no_grad():
                    sp = self.run(x[sel], T[sel] - 1, Md[sel], Td, Tb)
                Rr, Rrh = divide_path(sp * sp, W, Md[sel], TDC, Td, Tb, ctl.visits)
                real = {"sel": sel, "R": Rr, "Rh": Rrh, "width": width[sel]}
        self.control(exact, float(loss.detach()))
        return {"loss": loss, "real": real, "ctl": ctl}

    def row_lessons(self, N, x, T, Nd, width, Md, Td, Tb, TDC) -> dict:
        """Stage 2, the rows only: stage k trains on the batch's T = 1 rows whose dividend x^2 has at most k quotient
        digits (x^2 < N 10^k), from x^2 // 10^k in k steps, against the label (in the loss). Same phases as Stage 1;
        at k = W phase B also takes the T <= REAL_MAX_T rows. The loss reports the hard exactness back (control)."""
        ctl = self.ctl
        k, v = ctl.k, x * x
        sel = (T == 1) & (v < N * POW10[k])
        if ctl.phase == "B" and k == W:
            sel = sel | (T <= REAL_MAX_T)
        sel = torch.nonzero(sel).flatten()
        if sel.numel() == 0:
            ctl.in_stage += 1
            return {"loss": self.a.sum() * 0.0, "real": None, "ctl": ctl}
        with torch.no_grad():
            sp = self.run(x[sel], T[sel] - 1, Md[sel], Td, Tb)                # T = 1: x itself
        vs, Ms, ws = sp * sp, Md[sel], width[sel]
        if ctl.phase == "A":
            choice = "uniform" if ctl.step <= HOLD else "est"
            rpos = ws[:, None] - torch.arange(KR, device=N.device)[None]
            nidx = sum(Nd[sel].gather(1, (ws - 1 - i)[:, None])[:, 0] * 10 ** (KN - 1 - i) for i in range(KN))
            R = divide_soft(vs, k, Ms, TDC, choice, lambda Rp: self.estimate(Rp, nidx, rpos))
            with torch.no_grad():
                Rh = divide_hard(vs, k, Ms, Td, Tb, ctl.visits)
        else:
            R, Rh = divide_path(vs, k, Ms, TDC, Td, Tb, ctl.visits)
        return {"loss": self.a.sum() * 0.0, "real": {"sel": sel, "R": R, "Rh": Rh, "width": ws},
                "ctl": ctl, "control": self.control}

    def control(self, exact: float, loss: float) -> None:
        ctl = self.ctl
        ctl.win.append(exact)
        ctl.in_stage += 1
        mean = sum(ctl.win) / len(ctl.win)
        if ctl.step % LOG_EVERY == 0 or ctl.step == 1:
            print(f"LR step={ctl.step} phase={ctl.phase} k={ctl.k} in_stage={ctl.in_stage} loss={loss:.4f} "
                  f"hard_exact={exact:.4f} win={mean:.4f} rows_visited={int((ctl.visits > 0).sum())}/200", flush=True)
        ready = len(ctl.win) == STAGE_WIN and mean >= STAGE_EXACT
        timeout = ctl.in_stage >= STAGE_MAX
        if LESSON == "identity":                                            # one size: k plays no part
            ctl.k = W
            ready = len(ctl.win) == STAGE_WIN and mean >= ID_SWITCH
            timeout = ctl.in_stage >= ID_STEPS                              # the identity's plateau varies by seed
        if ctl.phase == "A" and (ready or (ctl.k == W and timeout)):
            # the hard datapath with the own-borrow choice is exact on this stage: from now on the soft choice is the
            # own borrow, with a fresh optimizer (Adam's memory of phase A's lessons would slow the repairs)
            ctl.history.append((ctl.step, "switch", ctl.k, ctl.in_stage, round(mean, 4)))
            print(f"LR switch_to_own_borrow step={ctl.step} k={ctl.k} win={mean:.4f}", flush=True)
            ctl.phase, ctl.reset = "B", True
            if ctl.k < W:
                self.advance(mean)
            else:
                ctl.in_stage = 0
                ctl.win.clear()
        elif ctl.k < W and (ready or timeout):
            self.advance(mean)
        elif ctl.phase == "B" and ctl.k == W:
            ctl.streak = ctl.streak + 1 if exact == 1.0 else 0
            if ctl.streak >= FREEZE_WIN:
                ctl.history.append((ctl.step, "frozen", W, ctl.in_stage, 1.0))
                print(f"LR frozen step={ctl.step} history={ctl.history}", flush=True)
                ctl.phase = "frozen"

    def advance(self, mean: float) -> None:
        ctl = self.ctl
        ctl.history.append((ctl.step, ctl.phase, ctl.k, ctl.in_stage, round(mean, 4)))
        print(f"LR stage_done step={ctl.step} phase={ctl.phase} k={ctl.k} steps={ctl.in_stage} win={mean:.4f}",
              flush=True)
        ctl.k, ctl.in_stage = ctl.k + 1, 0
        ctl.win.clear()

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
        if "control" in aux:                                                # LESSON = "rows": the schedule reads labels
            aux["control"](float((real["Rh"] == yd).all(1).float().mean()), float(lesson.detach()))
        elif ctl.step % LOG_EVERY == 0:
            ex = float((real["Rh"] == yd).all(1).float().mean())
            print(f"LR step={ctl.step} real_rows={sel.numel()} real_hard_exact={ex:.4f}", flush=True)
        return loss + lesson


class Schedule:
    """Scheduler: a fresh Adam state at the switch to the own borrow; learning rate 0 once the model is frozen."""

    def __init__(self, optimizer: torch.optim.Optimizer, ctl: Control) -> None:
        self.optimizer, self.ctl = optimizer, ctl

    def step(self) -> None:
        if self.ctl.reset:
            self.optimizer.state.clear()
            self.ctl.reset = False
        if self.ctl.phase == "frozen":
            for group in self.optimizer.param_groups:
                group["lr"] = 0.0


def build_model(spec: ModelSpec) -> LearnedRemainder:
    model = LearnedRemainder(spec)
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
