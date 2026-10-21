"""LearnedAnchors: submission_simple.py's machine, its subtraction table learned from a random start with no
hand-written adder at any step and no answer. Meaning comes from the order of the numbers (ANCHORS.md).

Datapath (designed wiring, unchanged from submission_chain.py / submission_adder.py / submission_simple.py, base 10):
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
Learned: the subtractor cell, a 200-row table of digit and borrow logits. Evaluation snaps the cell to its argmax table
and runs the datapath on integers. MODE = "designed" uses the exact subtraction table (plumbing check).

What teaches the cell (on the training rows' own numbers; no label is ever used, no adder is hand-written):
  1. the borrows, from x < N on every top part: the number formed by x's digits from place j up is at most N's,
     strictly less exactly when the two digit strings differ there; so x[j:] - N[j:] borrows exactly when they differ,
     and N[j:] - x[j:] never borrows (every j). Only the borrow outputs read these facts.
  2. once x - N borrows and N - x does not on every row of GATE_WIN consecutive batches (the borrows can referee),
     the digits, from two laws of subtraction applied one digit at a time to the entries the batch's links read:
       - order: r - m - c keeps the order of r (two entries (r, m, c), (r', m, c), r < r') and reverses the order of
         m (two entries (r, m, c), (r, m', c), m < m': subtracting more gives less), unless exactly one of the two
         wraps (goes below zero, its borrow out), and then the wrapped one comes out larger. Whether an entry wraps
         and which of two digits is smaller are read from the learned borrows;
       - bijectivity: different digits minus the same digit, or the same digit minus different digits, give
         different digits, so each column (m, c) and each row (r, c) of the cell uses every digit once;
  3. optionally (LATE_AFTER not None), LATE_AFTER steps later also the learned-adder check of ADDER.md, (D - N) - (0 - N) = D on every link of the
     machine's own division of each row's x^2 (all three subtractions the learned table; complete: the link's final
     borrow equals the adder's carry; lowest place weighted ID_PLACE0), and the wrap fact (x - N) - N does not borrow.
That is all: no answers, no phases beyond the two starts above, no optimizer reset, no freeze. The model trains for
MAX_STEPS steps. Everything runs in fp32 with autocast off; evaluation is integer arithmetic.
"""

from __future__ import annotations

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
GATE_WIN = 10          # the digit lessons start once the borrow facts hold on every row of this many batches
LATE_AFTER = None      # ... and (if a number) the learned-adder check and the wrap fact this many steps after that
ID_PLACE0 = 5.0        # the check's weight on the remainder's lowest place
ID_BORROW = 1.0        # the check's weight on its final borrow (the complete identity)
BATCH = 512
EVAL_BATCH = 16384
MAX_STEPS = 3000
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
    every link's input [X, S - 1, P]. `visits` accumulates every table row the links use."""
    D, R = Rp, Rp
    alive = torch.ones_like(Rp[:, 0], dtype=torch.bool)                 # no link has borrowed yet
    q = torch.zeros_like(Rp[:, 0])
    ins, rows = [], []
    for s in range(1, S):
        ins.append(D)
        D, c, idx = link_hard(D, M, Td, Tb)
        rows.append(idx)
        alive = alive & (c == 0)
        R = torch.where(alive[:, None], D, R)
        q = q + alive.long()
    if visits is not None:
        visits += torch.bincount(torch.stack(rows).flatten(), minlength=200)
    return R, q, torch.stack(ins, 1)


def divide_hard(v: Tensor, steps: int, M: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None) -> Tensor:
    """v mod N on integers, from R = v // 10^steps (below N) in `steps` steps of the chain. Returns the remainder's
    digits [X, P]. `visits` accumulates every table row used."""
    R = to_digits(v // POW10[steps], P)
    for t in reversed(range(steps)):
        R = step_hard(torch.cat([((v // POW10[t]) % 10)[:, None], R[:, :P - 1]], 1), M, Td, Tb, visits)[0]
    return R


def link_inputs(v: Tensor, steps: int, M: Tensor, Td: Tensor, Tb: Tensor) -> Tensor:
    """The hard division's link inputs at every step [X, steps, S - 1, P]: the register Rp = 10 R + v_t feeds link 1,
    trial s - 1 feeds link s (the machine's own numbers, every link it computes)."""
    R = to_digits(v // POW10[steps], P)
    out = []
    for t in reversed(range(steps)):
        R, _, ins = step_hard(torch.cat([((v // POW10[t]) % 10)[:, None], R[:, :P - 1]], 1), M, Td, Tb)
        out.append(ins)
    return torch.stack(out, 1)


def link_soft(D: Tensor, M: Tensor, TDC: Tensor):
    """One link on digit distributions, the cell applied multilinearly: D [X, n, 10] minus exact digits M [X, n] (any
    width n). TDC [20, 120]: rows (r, c), columns (m, 10 digit + 2 borrow probabilities). Returns the digits
    [X, n, 10] and the final borrow [X, 2]."""
    X = D.shape[0]
    c = torch.zeros(X, 2, device=D.device)
    c[:, 0] = 1
    out = []
    for j in range(D.shape[1]):
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = (joint @ TDC).view(X, 10, 12)                                       # every m at once
        Y = Y.gather(1, M[:, j, None, None].expand(X, 1, 12)).squeeze(1)        # the subtrahend's own digit
        out.append(Y[:, :10])
        c = Y[:, 10:]
    return torch.stack(out, 1), c


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


def order_pairs(Tb: Tensor, used: Tensor) -> list:
    """The pairs of table entries the two order laws compare, each oriented (lower, upper) by the learned borrows.
    Entries are flat (r, m, c) -> r * 20 + m * 2 + c; only entries in `used` take part. An entry wraps when its
    borrow out is 1, and r < r' when r - r' borrows.
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
    # columns
    e1 = k * 20 + (e0 % 20)
    ok = (r[e0] < k) & used[e0] & used[e1]
    a, b = e0[ok], e1[ok]
    wa, wb = Tb[a], Tb[b]
    lt = Tb[(r[a] * 10 + r[b]) * 2]                                        # r < r'
    first = (wa < wb) | ((wa == wb) & (lt == 1))
    out.append((torch.where(first, a, b), torch.where(first, b, a)))
    # rows
    e1 = (r[e0] * 10 + k) * 2 + c[e0]
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
    """Python-side bookkeeping (not model state): the step, the two starts, and the audit's table-row counts."""

    def __init__(self) -> None:
        self.step = 0
        self.facts_streak = 0
        self.gated_on: int | None = None      # the step the digit lessons started (the borrows referee from then on)
        self.visits: Tensor | None = None     # table rows used by training's lessons (their hard ripples)
        self.used: Tensor | None = None       # table rows used at evaluation
        self.reported = -1


class LearnedAnchors(nn.Module):
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
            loss = self.lessons(x, M, Td, Tb)
            return torch.zeros(rows, L, V, device=dev), {"loss": loss}
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

    def lessons(self, x: Tensor, M: Tensor, Td: Tensor, Tb: Tensor) -> Tensor:
        """The order facts about x and N for the borrows, then the order law and bijectivity for the digits, then the
        learned-adder check and the wrap fact (module docstring)."""
        ctl = self.ctl
        ctl.step += 1
        dev = x.device
        if ctl.visits is None:
            ctl.visits = torch.zeros(200, dtype=torch.long, device=dev)
        if MODE == "designed":
            return self.a.sum() * 0.0
        TDC = torch.cat([self.a.float().softmax(-1), self.b.float().softmax(-1)], -1)
        TDC = TDC.permute(0, 2, 1, 3).reshape(20, 120)
        X, K = x.shape[0], W * (S - 1)
        hot = lambda v: F.one_hot(v, 10).float()
        log = lambda p: p.clamp_min(1e-12).log()
        xd = to_digits(x, P)
        digits_on = ctl.gated_on is not None
        late_on = digits_on and LATE_AFTER is not None and ctl.step - ctl.gated_on >= LATE_AFTER
        seen = []
        # 1. x < N on every top part: x[j:] - N[j:] borrows iff the digit strings differ, N[j:] - x[j:] never borrows
        loss = torch.zeros((), device=dev)
        for j in range(P):
            differ = (xd[:, j:] != M[:, j:]).any(1)
            p1 = link_soft(hot(xd[:, j:]), M[:, j:], TDC)[1][:, 1]
            p2 = link_soft(hot(M[:, j:]), xd[:, j:], TDC)[1][:, 1]
            loss = loss - (torch.where(differ, log(p1), log(1 - p1)).mean() + log(1 - p2).mean()) / P
        with torch.no_grad():
            D = link_inputs(x * x, W, M, Td, Tb)
            D, Mr = D.reshape(X * K, P), M.repeat_interleave(K, 0)
            Z = torch.zeros_like(M)
            Eh, bh, idx = link_hard(D, Mr, Td, Tb)                         # the machine's own links, D - N
            _, bl, idl = link_hard(xd, M, Td, Tb)
            _, bm, idm = link_hard(M, xd, Td, Tb)
            seen += [idx, idl, idm]
            facts = bool(((bl == 1) & (bm == 0)).all())                    # x - N borrows, N - x does not
            ctl.facts_streak = ctl.facts_streak + 1 if facts else 0
        if digits_on:
            # 2a. order: in r (one column (m, c)) and, reversed, in m (one row (r, c)); 2b. bijectivity in both
            used = torch.bincount(idx.flatten(), minlength=200) > 0        # the entries the batch's links read
            pa = self.a.float().softmax(-1)
            pf = pa.reshape(200, 10)
            for lo, hi in order_pairs(Tb, used):
                above = 1 - pf[hi].cumsum(-1)                              # P(hi's digit > u)
                loss = loss - log((pf[lo] * above).sum(-1)).mean()
            loss = loss - log(10 * pa.mean(0)).mean() - log(10 * pa.mean(1)).mean()
        if late_on:
            # 3a. the learned-adder check on every link: (D - N) - (0 - N) = D, complete
            Es, be = link_soft(hot(D), Mr, TDC)
            Cs, bc = link_soft(hot(Z), M, TDC)
            back, bf = sub_soft(Es, Cs.repeat_interleave(K, 0), TDC)
            pc, pe, pf = bc[:, 1].repeat_interleave(K, 0), be[:, 1], bf[:, 1]
            agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc   # be = bc - bf
            weight = torch.ones(P, device=dev)
            weight[0] = ID_PLACE0
            nll = (-log(back).gather(-1, D[..., None]).squeeze(-1) * weight).sum(-1) - ID_BORROW * log(agree)
            loss = loss + nll.mean() / (weight.sum() + ID_BORROW)
            # 3b. x - N wraps to 10^P + x - N >= N, so (x - N) - N does not borrow
            Ew = link_soft(hot(xd), M, TDC)[0]
            loss = loss - log(link_soft(Ew, M, TDC)[1][:, 0]).mean()
        with torch.no_grad():
            if late_on or ctl.step % LOG_EVERY == 0 or ctl.step == 1:
                Ch, bch, idc = link_hard(Z, M, Td, Tb)
                Fh, bfh, idf = link_hard(Eh, Ch.repeat_interleave(K, 0), Td, Tb)
                Exh, _, idw0 = link_hard(xd, M, Td, Tb)
                bw, idw = link_hard(Exh, M, Td, Tb)[1:]
                if late_on:
                    seen += [idc, idf, idw0, idw]
            ctl.visits += torch.bincount(torch.cat([t.flatten() for t in seen]), minlength=200)
            if ctl.step % LOG_EVERY == 0 or ctl.step == 1:
                check = float(((Fh == D).all(-1) & (bh == bch.repeat_interleave(K, 0) - bfh)).float().mean())
                print(f"LR step={ctl.step} digits_from={ctl.gated_on} late={late_on} loss={float(loss):.4f} "
                      f"facts_exact={float(((bl == 1) & (bm == 0)).float().mean()):.4f} check_exact={check:.4f} "
                      f"wrap_exact={float((bw == 0).float().mean()):.4f} "
                      f"rows_visited={int((ctl.visits > 0).sum())}/200", flush=True)
        if ctl.gated_on is None and ctl.facts_streak >= GATE_WIN:
            ctl.gated_on = ctl.step                                        # the borrows referee from the next step
            print(f"LR digits_start step={ctl.step}", flush=True)
        return loss

    def report_tables(self, Td: Tensor, Tb: Tensor) -> None:
        """At the first evaluation batch: the snapped table and how often training's lessons used each row."""
        ctl = self.ctl
        print("TABLE digit=" + "".join(str(int(d)) for d in Td.tolist()), flush=True)
        print("TABLE borrow=" + "".join(str(int(d)) for d in Tb.tolist()), flush=True)
        if ctl.visits is not None:
            v = ctl.visits
            print(f"TABLE train_visits min={int(v.min())} rows_never={int((v == 0).sum())} "
                  f"counts={','.join(str(int(c)) for c in v.tolist())}", flush=True)
        print(f"TABLE schedule step={ctl.step} digits_from={ctl.gated_on}", flush=True)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    with torch.autocast(batch.labels.device.type, enabled=False):
        return batch.auxiliary["loss"]


def build_model(spec: ModelSpec) -> LearnedAnchors:
    model = LearnedAnchors(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    # fused keeps Adam's step counter on the accelerator (the evaluator checks every optimizer state tensor)
    opt = torch.optim.Adam(model.parameters(), lr=LR, fused=spec.device_type in ("cuda", "mps"))
    return OptimizerBundle(opt)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=BATCH,
    max_steps=MAX_STEPS,
    eval_batch_size=EVAL_BATCH,
)
