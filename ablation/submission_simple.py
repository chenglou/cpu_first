"""LearnedAdderSimple: submission_adder.py's machine with its training recipe cut down to what ABLATION.md found holds
it up. The datapath is unchanged; only the lessons and the schedule are shorter.

Datapath (designed wiring, unchanged from submission_chain.py / submission_adder.py, base 10):
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

What teaches the cell (every step, on the training rows' own numbers; no label is ever used):
  * the check: every link of the machine's own hard division of each row's x^2, D - N, adds N back and must give D's
    digits. The adder is the hand-written one for the first ADDER_WARM steps, then the learned one, (D - N) - (0 - N)
    (in ten's complement 0 - N = 10^P - N), all three subtractions the learned table;
  * x < N, so x - N borrows and N - x does not;
  * x - N wraps around to 10^P + x - N, which is at least N, so (x - N) - N does not borrow.
That is all: no other identity, no anchor on the registers, no answer, no phase, no optimizer reset, no freeze. The
model trains for MAX_STEPS steps. Everything runs in fp32 with autocast off; evaluation is integer arithmetic.
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
ADDER_WARM = 70        # the check's adder: the hand-written one for the first ADDER_WARM steps, then the learned one
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


def add_hard(D: Tensor, M: Tensor) -> tuple[Tensor, Tensor]:
    """The designed adder on integers: digits D [..., P] + M [..., P], mod 10^P, and the final carry."""
    c = torch.zeros_like(D[..., 0])
    out = []
    for j in range(D.shape[-1]):
        x = D[..., j] + M[..., j] + c
        out.append(x % 10)
        c = x // 10
    return torch.stack(out, -1), c


def add_soft(D: Tensor, M: Tensor) -> Tensor:
    """The designed adder on digit distributions D [..., P, 10] plus exact digits M [..., P], mod 10^P (multilinear)."""
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
    return torch.stack(out, -2)


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


def sub_soft(D: Tensor, M: Tensor, TDC: Tensor) -> Tensor:
    """D - M with both operands on digit distributions, D, M [X, width, 10] (multilinear, as link_soft). Returns the
    digits [X, width, 10]."""
    X = D.shape[0]
    c = torch.zeros(X, 2, device=D.device)
    c[:, 0] = 1
    out = []
    for j in range(D.shape[1]):
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = torch.einsum("xmk,xm->xk", (joint @ TDC).view(X, 10, 12), M[:, j])  # expected over the subtrahend's digit
        out.append(Y[:, :10])
        c = Y[:, 10:]
    return torch.stack(out, 1)


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Control:
    """Python-side bookkeeping (not model state): the step, and the audit's table-row counts."""

    def __init__(self) -> None:
        self.step = 0
        self.visits: Tensor | None = None     # table rows used by training's lessons (their hard ripples)
        self.used: Tensor | None = None       # table rows used at evaluation
        self.reported = -1


class LearnedAdderSimple(nn.Module):
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
        """The check on every link of the machine's own division of each row's x^2, and the two facts about x < N."""
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
        with torch.no_grad():
            D = link_inputs(x * x, W, M, Td, Tb)
            D, Mr = D.reshape(X * K, P), M.repeat_interleave(K, 0)
            Z = torch.zeros_like(M)
        hot = lambda v: F.one_hot(v, 10).float()
        log = lambda p: p.clamp_min(1e-12).log()
        warm = ctl.step <= ADDER_WARM
        Es = link_soft(hot(D), Mr, TDC)[0]                                     # the link, D - N
        if warm:
            back = add_soft(Es, Mr)                                            # (D - N) + N, the hand-written adder
        else:
            Cs = link_soft(hot(Z), M, TDC)[0]                                  # 0 - N, once per row
            back = sub_soft(Es, Cs.repeat_interleave(K, 0), TDC)               # (D - N) - (0 - N) = (D - N) + N
        loss = -log(back).gather(-1, D[..., None]).squeeze(-1).sum(-1).mean() / P   # D's digits come back
        xd = to_digits(x, P)
        loss = loss - log(link_soft(hot(xd), M, TDC)[1][:, 1]).mean()           # x < N: x - N borrows
        loss = loss - log(link_soft(hot(M), xd, TDC)[1][:, 0]).mean()           # x < N: N - x does not borrow
        Ew = link_soft(hot(xd), M, TDC)[0]
        loss = loss - log(link_soft(Ew, M, TDC)[1][:, 0]).mean()               # (x - N) - N does not borrow
        with torch.no_grad():                                                  # the same lessons on the hard machine
            Eh, _, idx = link_hard(D, Mr, Td, Tb)
            seen = [idx]
            if warm:
                ok = (add_hard(Eh, Mr)[0] == D).all(-1)
            else:
                Ch, _, idc = link_hard(Z, M, Td, Tb)
                Fh, _, idf = link_hard(Eh, Ch.repeat_interleave(K, 0), Td, Tb)
                ok = (Fh == D).all(-1)
                seen += [idc, idf]
            Exh, bl, idl = link_hard(xd, M, Td, Tb)
            bm, idm = link_hard(M, xd, Td, Tb)[1:]
            bw, idw = link_hard(Exh, M, Td, Tb)[1:]
            seen += [idl, idm, idw]
            ctl.visits += torch.bincount(torch.cat([t.flatten() for t in seen]), minlength=200)
            if ctl.step % LOG_EVERY == 0 or ctl.step == 1:
                facts = float(((bl == 1) & (bm == 0) & (bw == 0)).float().mean())
                print(f"LR step={ctl.step} adder={'designed' if warm else 'learned'} loss={float(loss):.4f} "
                      f"check_exact={float(ok.float().mean()):.4f} facts_exact={facts:.4f} "
                      f"rows_visited={int((ctl.visits > 0).sum())}/200", flush=True)
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
        print(f"TABLE schedule step={ctl.step}", flush=True)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    with torch.autocast(batch.labels.device.type, enabled=False):
        return batch.auxiliary["loss"]


def build_model(spec: ModelSpec) -> LearnedAdderSimple:
    model = LearnedAdderSimple(spec)
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
