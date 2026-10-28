"""SpecBit: a gate-level binary Montgomery machine whose one gate is learned from labels by speculation.

Port of cpu_first/bitcpu_netlist.py to a One Layer Deeper submission. Every gate in the datapath is the same
learned 3->2 cell g(a, b, c) = (sum, carry): AND = carry(a, b, 0), OR = carry(a, b, 1), XOR = sum(a, b, 0),
NOT = sum(a, 1, 0), MUX = OR(AND(c, a), AND(NOT c, b)). The wiring is designed (decimal->binary Horner, radix-4
entry x*2^26 mod N, predicated carry-save Montgomery squarings with a 7-bit down counter, 26-halving exit,
double-dabble back to BCD); the cell's semantics are learned.

Parameters: theta, one logit per candidate cell (8 sum bits | 8 carry bits << 8: 65,536 truth tables).
Training forward: the machine runs under every hypothesis h with pi_h >= 1e-9 max pi (all 65,536 at first) on the
batch's T <= 2 rows, bitsliced: lane j of an int64 word is hypothesis 64*w + j, so one bitwise op advances 64
machines. Loss: -mean_rows logsumexp_h[log pi_h - 1e4 * 1(digits_h != y)]. Prediction: the argmax cell's machine,
logits = log(pi* onehot(digit) + (1 - pi*) / V), so the loss reaches theta and argmax is the machine's digit.
Everything is int64 bitwise ops or fp32 (no autocast-sensitive math).
"""

from __future__ import annotations

import math

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

W, K, ND, NB, TW = 28, 26, 8, 8, 7  # datapath bits, Montgomery R = 2^K, decimal digits in, BCD digits out, counter
NH = 1 << 16                         # candidate cells: sum8 | carry8 << 8, table index m = a + 2b + 4c
DIGIT, MARK_N, MARK_X, MARK_T = 7, 2, 3, 4
PRUNE = math.log(1e-9)               # hypotheses below 1e-9 * max pi are dropped from the forward pass
MISMATCH = 1e4
SPEC_STEPS = 2                       # clock budget of the speculative pass (it only sees T <= 2 rows)
MAX_STEPS = (1 << TW) - 1            # the counter's range


class Cells:
    """One shared 3->2 cell, bitsliced over hypotheses: lane j of word w evaluates table ids[64 w + j]."""

    def __init__(self, ids: Tensor) -> None:
        lanes = ids.view(-1, 64)
        shifts = torch.arange(64, device=ids.device)
        pack = lambda bits: (bits << shifts).sum(-1)  # one word per 64 lanes
        self.t = [[pack((lanes >> (8 * o + m)) & 1) for m in range(8)] for o in (0, 1)]

    @staticmethod
    def _eval(t: list, ins: list):
        """Shannon expansion of a per-lane truth table; constant (int) inputs are folded away first."""
        for i, x in enumerate(ins):
            if isinstance(x, int):
                return Cells._eval([t[m] for m in range(len(t)) if (m >> i) & 1 == x], ins[:i] + ins[i + 1:])
        if not ins:
            return t[0]
        x = ins[0]
        return Cells._eval([t[2 * j] ^ (x & (t[2 * j] ^ t[2 * j + 1])) for j in range(len(t) // 2)], ins[1:])

    def cell(self, o: int, ins: list):
        rank = max(x.dim() for x in ins if torch.is_tensor(x))  # lanes lead; broadcast the table over the rest
        return self._eval([w.view(-1, *([1] * (rank - 1))) for w in self.t[o]], ins)

    def fa(self, a, b, c):
        return self.cell(0, [a, b, c]), self.cell(1, [a, b, c])

    def AND(self, a, b): return self.cell(1, [a, b, 0])
    def OR(self, a, b): return self.cell(1, [a, b, 1])
    def XOR(self, a, b): return self.cell(0, [a, b, 0])
    def NOT(self, a): return self.cell(0, [a, 1, 0])
    def MUX(self, c, a, b): return self.OR(self.AND(c, a), self.AND(self.NOT(c), b))


# ---- wiring (no gates): bit vectors are int64 words [..., rows, bits], bit 0 = LSB ----
def cat(*vs: Tensor) -> Tensor:
    lead = torch.broadcast_shapes(*[v.shape[:-1] for v in vs])
    return torch.cat([v.expand(*lead, v.shape[-1]) for v in vs], -1)


def shl(v: Tensor, k: int) -> Tensor:
    return F.pad(v[..., : v.shape[-1] - k], (k, 0)) if k else v


def shr(v: Tensor, k: int) -> Tensor:
    return F.pad(v[..., k:], (0, k)) if k else v


def zeros(like: Tensor, w: int) -> Tensor:
    return torch.zeros(1, 1, w, dtype=torch.int64, device=like.device)


def word_bits(bits: Tensor) -> Tensor:
    """0/1 [rows, w] -> words [1, rows, w] (the same value in every lane)."""
    return torch.where(bits.bool(), -1, 0).to(torch.int64)[None]


# ---- blocks built only from the learned cell ----
def ks_add(g: Cells, a: Tensor, b: Tensor, cin):
    """Kogge-Stone adder; returns (sum, carry out)."""
    gen, prop = g.AND(a, b), g.XOR(a, b)
    G = cat(cin, gen) if torch.is_tensor(cin) else cat(torch.full_like(gen[..., :1], -cin), gen)
    P = cat(torch.zeros_like(prop[..., :1]), prop)
    n, dist = G.shape[-1], 1
    while dist < n:
        Gn, Pn = g.OR(G, g.AND(P, shl(G, dist))), g.AND(P, shl(P, dist))
        G, P = cat(G[..., :dist], Gn[..., dist:]), cat(P[..., :dist], Pn[..., dist:])
        dist *= 2
    return g.XOR(prop, G[..., :-1]), G[..., n - 1:]


def ripple_add(g: Cells, a: Tensor, b: Tensor):
    outs, c = [], 0
    for j in range(a.shape[-1]):
        s, c = g.fa(a[..., j:j + 1], b[..., j:j + 1], c)
        outs.append(s)
    return cat(*outs)


def d2b(g: Cells, digs: Tensor, w: int) -> Tensor:
    """LSB-first decimal digits [rows, n] -> w-bit binary: carry-save Horner v <- 8v + 2v + d, then a CPA."""
    vs = vc = zeros(digs, w)
    for i in range(digs.shape[1] - 1, -1, -1):
        d = F.pad(word_bits((digs[:, i:i + 1] >> torch.arange(4, device=digs.device)) & 1), (0, w - 4))
        s1, c1 = g.fa(shl(vs, 3), shl(vc, 3), shl(vs, 1))
        s2, c2 = g.fa(s1, shl(c1, 1), shl(vc, 1))
        s3, c3 = g.fa(s2, shl(c2, 1), d)
        vs, vc = s3, shl(c3, 1)
    return ks_add(g, vs, vc, 0)[0]


def montgomery_square(g: Cells, s: Tensor, N: Tensor) -> Tensor:
    """s*s / 2^K mod N (up to one N), carry-save, quotient bit = parity of the sum row."""
    pp = g.AND(s[..., :K, None], s[..., None, :])  # [..., rows, K, W] partial products
    Ss = Sc = zeros(s, W)
    for i in range(K):
        u, v = g.fa(Ss, Sc, pp[..., i, :])
        s2, c2 = g.fa(u, shl(v, 1), g.AND(u[..., :1], N))
        Ss, Sc = shr(s2, 1), c2
    return ks_add(g, Ss, Sc, 0)[0]


def run_machine(g: Cells, Nd: Tensor, Xd: Tensor, Td: Tensor, max_steps: int) -> Tensor:
    """Digits (LSB first) -> 8 BCD digits of x^(2^T) mod N as words [lanes, rows, 32]."""
    N, X, cnt = d2b(g, Nd, W), d2b(g, Xd, W), d2b(g, Td, TW)
    N2 = shl(N, 1)
    N3 = ks_add(g, N, N2, 0)[0]
    negs = [g.NOT(N), g.NOT(N2), g.NOT(N3)]
    s = X
    for _ in range(K // 2):  # entry: x * 2^K mod N, radix 4: 4s - kN for k = 1..3, select by borrow-outs
        t = shl(s, 2)
        diffs = [ks_add(g, t, negs[k], 1) for k in range(3)]
        s = g.MUX(diffs[0][1], diffs[0][0], t)
        s = g.MUX(diffs[1][1], diffs[1][0], s)
        s = g.MUX(diffs[2][1], diffs[2][0], s)
    for _ in range(max_steps):  # predicated loop; the machine's own counter halts it
        active = g.OR(g.OR(g.OR(cnt[..., 0:1], cnt[..., 1:2]), g.OR(cnt[..., 2:3], cnt[..., 3:4])),
                      g.OR(g.OR(cnt[..., 4:5], cnt[..., 5:6]), cnt[..., 6:7]))
        if not bool(active.any()):
            break
        s = g.MUX(active, montgomery_square(g, s, N), s)
        cnt = ripple_add(g, cnt, active.expand(*active.shape[:-1], TW))  # cnt += active * (2^TW - 1) = -active
    Ss, Sc = s, zeros(s, W)
    for _ in range(K):  # exit: MonPro(s, 1), K halvings mod N
        q = g.XOR(Ss[..., :1], Sc[..., :1])
        u, v = g.fa(Ss, Sc, g.AND(q, N))
        Ss, Sc = shr(u, 1), v
    y = ks_add(g, Ss, Sc, 0)[0]
    reg = zeros(y, 4 * NB)
    for bit in range(23, -1, -1):  # double dabble: add 3 to every nibble >= 5, shift in the next bit
        nib = reg.reshape(*reg.shape[:-1], NB, 4)
        n0, n1, n2, n3 = (nib[..., j:j + 1] for j in range(4))
        cond = g.OR(n3, g.AND(n2, g.OR(n1, n0)))
        a0, c = g.fa(n0, cond, 0)
        a1, c = g.fa(n1, cond, c)
        a2, c = g.fa(n2, 0, c)
        a3, _ = g.fa(n3, 0, c)
        nibs = cat(a0, a1, a2, a3).flatten(-2)
        reg = cat(y[..., bit:bit + 1], nibs[..., :-1])
    return reg


def parse(input_ids: Tensor, attention_mask: Tensor):
    """Prompt `N <N LSB first> X <x MSB first> T <T MSB first>` -> LSB-first digit arrays."""
    pos = lambda tok: (input_ids == tok).int().argmax(1)
    i_n, i_x, i_t = pos(MARK_N), pos(MARK_X), pos(MARK_T)
    end = attention_mask.long().sum(1)
    k = lambda n: torch.arange(n, device=input_ids.device)[None]

    def field(start, step, count, n):
        idx = (start[:, None] + step * k(n)).clamp(0, input_ids.shape[1] - 1)
        d = input_ids.gather(1, idx) - DIGIT
        return torch.where(k(n) < count[:, None], d, 0)

    Nd = field(i_n + 1, 1, i_x - i_n - 1, ND)
    Xd = field(i_t - 1, -1, i_t - i_x - 1, ND)
    Td = field(end - 1, -1, end - i_t - 1, 2)
    return Nd, Xd, Td, end


def nibbles(reg: Tensor) -> Tensor:
    """Lane 0 of BCD words [lanes, rows, 32] -> digit values [rows, NB]."""
    bits = (reg[0] & 1).view(reg.shape[1], NB, 4)
    return (bits << torch.arange(4, device=reg.device)).sum(-1)


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class SpecBit(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.theta = nn.Parameter(0.01 * torch.randn(NH))

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        if attention_mask is None:
            attention_mask = input_ids != 0
        Nd, Xd, Td, end = parse(input_ids, attention_mask)
        logp = torch.log_softmax(self.theta.float(), 0)
        aux = None
        if self.training:
            rows = torch.nonzero((Td[:, 1] == 0) & (Td[:, 0] <= 2)).flatten()  # the batch's T <= 2 rows
            ids = torch.nonzero(logp >= logp.max() + PRUNE).flatten()
            lanes = torch.cat([ids, ids[:1].expand((-ids.numel()) % 64)])
            reg = run_machine(Cells(lanes), Nd[rows], Xd[rows], Td[rows], SPEC_STEPS) if rows.numel() else None
            aux = {"rows": rows, "ids": ids, "logp": logp[ids], "reg": reg, "logp_all": logp}
        best = logp.argmax()
        digits = nibbles(run_machine(Cells(best.expand(64)), Nd, Xd, Td, MAX_STEPS))  # [B, NB]
        # position p holds digit (end - 1 - p): answers are right-aligned on the prompt
        place = end[:, None] - 1 - torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        d = digits.gather(1, place.clamp(0, NB - 1))
        tok = torch.where((place >= 0) & (place < NB) & (d < 10), d + DIGIT, 0)
        pi = logp[best].exp()
        V = self.config.vocab_size
        probs = F.one_hot(tok, V).float() * pi + (1 - pi) / V
        return probs.clamp_min(1e-30).log(), aux


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    aux = batch.auxiliary
    if aux["reg"] is None:
        return aux["logp_all"].sum() * 0.0
    rows, reg = aux["rows"], aux["reg"]
    labels, valid = batch.labels[rows], batch.valid_mask[rows]
    n = valid.long().sum(1)
    k = torch.arange(NB, device=labels.device)[None]
    digit = (labels.gather(1, (n[:, None] - 1 - k).clamp_min(0)) - DIGIT).clamp_min(0)  # LSB first
    j = torch.arange(4, device=labels.device)
    want = ((digit[..., None] >> j) & 1).flatten(1)                                     # [rows, 32]
    scored = (k < n[:, None])[..., None].expand(-1, -1, 4).flatten(1)
    diff = (reg ^ word_bits(want)) & word_bits(scored.long())                           # [words, rows, 32]
    while diff.shape[-1] > 1:                                                            # OR-reduce the bits
        diff = diff[..., 0::2] | diff[..., 1::2]
    lane = torch.arange(64, device=labels.device)
    wrong = ((diff[..., 0][:, None, :] >> lane[None, :, None]) & 1).flatten(0, 1)       # [lanes, rows]
    wrong = wrong[: aux["ids"].numel()].float()
    score = aux["logp"][:, None] - MISMATCH * wrong
    return -torch.logsumexp(score, 0).mean()


def build_model(spec: ModelSpec) -> SpecBit:
    model = SpecBit(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    # fused keeps Adam's step counter on the accelerator (the evaluator checks every optimizer state tensor)
    return OptimizerBundle(torch.optim.Adam(model.parameters(), lr=5.0, fused=spec.device_type in ("cuda", "mps")))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    max_steps=20,
    eval_batch_size=1024,
)
