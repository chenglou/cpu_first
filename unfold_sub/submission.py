"""UnfoldB: x^(2^T) mod N from one learned base and a learned quadratic, trained on answers only.

Port of cpu_first/UNFOLD.md section 4 (the self-unfolding clock loss) to a One Layer Deeper submission.

Model (fp64 inside, autocast off):
  * every digit string (N, x, T, and the label inside the loss) is worth sum_i B^i d_i, one learned base B;
  * one squaring step maps a state value s to V = w0 + w1 s + w2 s^2 (learned) and snaps it to the integer
    r = round(V) mod round(n), n the modulus string's value;
  * the step loops round(t) times (t the T string's value), the state an integer after every step;
  * the final state is written in base round(B) and read out as a phase over the 10 digit tokens (a clock with
    10 hours per digit); the digit is exact and a straight-through term carries the gradient of V.
Training (answer-only labels):
  * single-step rows (the model reads t = 1): clock residual e = (V - y - k n) / n in turns of the live modulus n,
    against the model's own nearest unfolding k = max(0, round((V - y) / n)); y is the label string read through B;
  * easiest first by the model's own difficulty: the P + 1 rows whose phase moves least with w (|d phase / d w| ~
    s^2 / n under its own base B) are always in; every other row comes in once the model can predict its phase to
    within U_MAX turns from the rows already in (linearized covariance of the fit) and its residual agrees with that
    prediction. The curriculum is recomputed from the batch and the current parameters at every step (no schedule);
  * Levenberg-Marquardt inside the OptimizerBundle: the gradient is the evaluator's backward (the loss is divided by
    a detached nu so the evaluator's clipping never bites; the optimizer multiplies nu back), the Gauss-Newton
    curvature comes from the loss's closed-form residual Jacobian, and trust regions keep every included row's phase
    move under TRUST turns and the base's move under BTRUST * B per update.
One batch of 100,000 rows (about the whole training split) is reused for 8 updates, the evaluator's maximum.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
)

DIGIT, MARK_N, MARK_X, MARK_T = 7, 2, 3, 4  # token ids: digit d is DIGIT + d
D = 9                                       # digit slots (24-bit values have at most 8 decimal digits)
P = 4                                       # parameters, in model.parameters() order: B, w0, w1, w2

B_INIT = (4.0, 16.0)   # base B ~ U(B_INIT)
W_INIT = 1e-3          # w ~ W_INIT * N(0, 1): V starts near 0, so every unfolding starts at k = 0 (plain regression)
KAPPA = 10.0           # logit scale of the phase readout
FRONT0 = P + 1         # always in: the P + 1 easiest single-step rows of the batch
U_MAX = 0.05           # admit a row once its predicted phase std (turns) from the admitted rows is below U_MAX
GATE = 3.0             # ... and its residual is within GATE predicted std
S_FLOOR = 1e-9         # floor on the admitted rows' residual std (turns), above fp64 noise at 24 bits
GATE_ITERS = 16        # admit / re-estimate rounds per step
DAMP = 1e-9            # LM damping on the column-normalized Gauss-Newton matrix
TRUST = 0.25           # trust region: no included row's phase moves more than TRUST turns per update
BTRUST = 0.1           # ... and the base moves by at most BTRUST * B (the model's own reading changes slowly)
BATCH = 100_000        # one batch per epoch: about every training row
EVAL_BATCH = 4096
MAX_STEPS = 1000
LOG_EVERY = 10


def parse(input_ids: Tensor, attention_mask: Tensor):
    """Prompt `N <N digits, LSB first> X <x, MSB first> T <T, MSB first>` -> LSB-first digit arrays [rows, D]."""
    pos = lambda tok: (input_ids == tok).int().argmax(1)
    i_n, i_x, i_t = pos(MARK_N), pos(MARK_X), pos(MARK_T)
    end = attention_mask.long().sum(1)
    k = torch.arange(D, device=input_ids.device)[None]

    def field(start, step, count):
        idx = (start[:, None] + step * k).clamp(0, input_ids.shape[1] - 1)
        d = (input_ids.gather(1, idx) - DIGIT).clamp(0, 9)
        return torch.where(k < count[:, None], d, 0)

    Nd = field(i_n + 1, 1, i_x - i_n - 1)
    Xd = field(i_t - 1, -1, i_t - i_x - 1)
    Td = field(end - 1, -1, end - i_t - 1)
    return Nd, Xd, Td, end


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class UnfoldB(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.base = nn.Parameter(torch.empty(1).uniform_(*B_INIT))
        self.w = nn.Parameter(W_INIT * torch.randn(3))
        self.stats: dict = {}  # the last loss's Gauss-Newton pieces, read by the optimizer

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        if attention_mask is None:
            attention_mask = input_ids != 0
        with torch.autocast(input_ids.device.type, enabled=False):
            return self._forward(input_ids, attention_mask)

    def _forward(self, input_ids: Tensor, attention_mask: Tensor):
        f64, dev = torch.float64, input_ids.device
        Nd, Xd, Td, end = parse(input_ids, attention_mask)
        B, w = self.base.to(f64)[0], self.w.to(f64)
        i = torch.arange(D, device=dev, dtype=f64)
        pw = B ** i                                  # place values B^i
        dpw = i * B ** (i - 1).clamp_min(0)          # d(B^i)/dB, closed form
        n, t, s = Nd.to(f64) @ pw, Td.to(f64) @ pw, Xd.to(f64) @ pw
        nr = torch.round(n.detach()).clamp_min(1)    # snapped modulus
        steps = torch.round(t.detach()).clamp_min(0)

        state = torch.remainder(torch.round(s.detach()), nr)  # snapped state (an integer)
        cont = s                                     # the last step's continuous V, for the straight-through readout
        v1 = None
        for it in range(int(steps.max().item()) if steps.numel() else 0):
            live = steps > it
            src = s if it == 0 else state            # step 1 reads x through B; later steps read the snapped state
            v = w[0] + w[1] * src + w[2] * src * src
            if it == 0:
                v1 = v
            cont = torch.where(live, v, cont)
            state = torch.where(live, torch.remainder(torch.round(v.detach()), nr), state)

        # write the state in base round(B); digit k carries the gradient of the continuous remainder / b^k
        b = torch.round(B.detach()).clamp(2, 36)
        place = b.long() ** torch.arange(D, device=dev)
        digs = (state.long()[:, None] // place[None]) % b.long()
        rho = cont - nr * torch.floor(torch.round(cont.detach()) / nr)
        dig = digs.to(f64) + (rho - rho.detach())[:, None] / b ** i
        k = (end[:, None] - 1 - torch.arange(input_ids.shape[1], device=dev)[None]).clamp(0, D - 1)
        dk = dig.gather(1, k)                                         # answers are right-aligned on the prompt
        c = torch.arange(10, device=dev, dtype=f64)
        phase = KAPPA * torch.cos(2 * math.pi * (dk[..., None] - c) / 10)
        rows, L, V = input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
        low = lambda width: torch.full((rows, L, width), -2 * KAPPA, device=dev, dtype=f64)
        logits = torch.cat([low(DIGIT), phase, low(V - DIGIT - 10)], -1).float()

        aux = None
        if self.training:
            aux = {"v1": v1, "s": s, "ds": Xd.to(f64) @ dpw, "n": n, "dn": Nd.to(f64) @ dpw, "pw": pw,
                   "dpw": dpw, "w": w, "single": steps == 1, "stats": self.stats}
        return logits, aux


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    with torch.autocast(batch.labels.device.type, enabled=False):
        return _clock_loss(batch, batch.auxiliary)


def _clock_loss(batch: TokenLossBatch, aux: dict) -> Tensor:
    f64 = torch.float64
    stats = aux["stats"]
    stats.clear()
    sel = torch.nonzero(aux["single"]).flatten()
    if aux["v1"] is None or sel.numel() == 0:
        return aux["w"].sum() * 0.0
    labels, valid = batch.labels[sel], batch.valid_mask[sel]
    cnt = valid.long().sum(1)
    k = torch.arange(D, device=labels.device)[None]
    yd = (labels.gather(1, (cnt[:, None] - 1 - k).clamp_min(0)) - DIGIT).clamp(0, 9)
    yd = torch.where(k < cnt[:, None], yd, 0).to(f64)
    y, dy = yd @ aux["pw"], yd @ aux["dpw"]                      # the label's value through the same base
    V, s, ds, n, dn, w = (aux["v1"][sel], aux["s"][sel], aux["ds"][sel], aux["n"][sel], aux["dn"][sel],
                          aux["w"])
    nb = n.detach().clamp_min(1.0)
    kstar = torch.clamp(torch.round((V - y).detach() / nb), min=0)   # the model's own nearest unfolding, k >= 0
    e = (V - y - kstar * n) / n.clamp_min(1.0)                         # clock residual, in turns of the live modulus
    sd, ed_ = s.detach(), e.detach()
    J = torch.stack([((w[1] + 2 * w[2] * sd) * ds - dy - (kstar + ed_) * dn).detach(), torch.ones_like(sd), sd,
                     sd * sd], 1) / nb[:, None]                        # d e / d (B, w0, w1, w2), closed form

    # easiest first by the model's own difficulty. Always in: the P + 1 rows whose phase moves least with w. Then
    # every row whose phase the model predicts to within U_MAX turns from the rows in (variance s^2 J (J^T J)^-1 J^T,
    # s^2 the residual variance of the rows in) and whose residual agrees with that prediction; re-estimated until
    # the admitted set stops changing.
    count = sel.numel()
    ed = torch.nan_to_num(e.detach(), nan=1e30, posinf=1e30, neginf=-1e30)
    seed = torch.zeros(count, dtype=torch.bool, device=e.device)
    seed[torch.argsort(J[:, 1:].norm(dim=1))[:FRONT0]] = True
    inc, eye = seed, torch.eye(P, dtype=f64, device=e.device)
    for _ in range(GATE_ITERS):
        col = J[inc].norm(dim=0).clamp_min(1e-300)
        Jn = J / col
        lev = (Jn * torch.linalg.solve(Jn[inc].T @ Jn[inc] + 1e-12 * eye, Jn.T).T).sum(1)
        s2 = float((ed[inc] ** 2).sum()) / max(int(inc.sum()) - P, 1) + S_FLOOR ** 2
        var = s2 * lev
        new = seed | ((var < U_MAX ** 2) & (ed * ed < GATE ** 2 * (var + s2)))
        if bool((new == inc).all()):
            break
        inc = new
    inc = torch.nonzero(inc).flatten()

    ei, Ji = e[inc], J[inc]
    loss = 0.5 * (ei * ei).sum()
    g = Ji.T @ ei.detach()
    nu = max(2.0 * float(g.norm()), 1e-30)
    stats.update(J=Ji, e=ei.detach(), nu=nu, m=inc.numel(), count=count, s=math.sqrt(s2),
                 emax=float(ei.detach().abs().max()))
    return loss / nu + (loss - loss / nu).detach()   # value: the clock cost; gradient: scaled by 1 / nu


class LevenbergMarquardt(torch.optim.Optimizer):
    """Gauss-Newton with Marquardt column scaling and trust regions, fp64 master parameters.

    Gradient: the evaluator's backward of the scaled loss, times nu. Curvature: J^T J from the loss's closed-form
    residual Jacobian of the included rows. The step is shrunk so that no included row's predicted phase moves more
    than TRUST turns and the base moves by at most BTRUST * B."""

    def __init__(self, params, stats: dict) -> None:
        super().__init__(params, {})
        self.stats = stats
        self.updates = 0

    @torch.no_grad()
    def step(self, closure=None):
        ps = [p for group in self.param_groups for p in group["params"]]
        for p in ps:
            if "master" not in self.state[p]:
                self.state[p]["master"] = p.detach().to(torch.float64).clone()
        st = self.stats
        self.updates += 1
        if "J" not in st:
            return None
        theta = torch.cat([self.state[p]["master"].flatten() for p in ps])
        grad = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().to(torch.float64)
                          for p in ps]) * st["nu"]
        J = st["J"]
        H = J.T @ J
        col = H.diagonal().clamp_min(0).sqrt()
        col = col.clamp_min(float(col.max()) * 1e-12 + 1e-300)
        A = H / col[:, None] / col[None, :] + DAMP * torch.eye(len(col), dtype=H.dtype, device=H.device)
        delta = torch.linalg.solve(A, grad / col) / col
        move = float((J @ delta).abs().max())
        delta = delta * min(1.0, TRUST / max(move, 1e-300),
                            BTRUST * abs(float(theta[0])) / max(abs(float(delta[0])), 1e-300))
        theta = theta - delta
        at = 0
        for p in ps:
            k = p.numel()
            self.state[p]["master"].copy_(theta[at:at + k].view_as(p))
            p.copy_(theta[at:at + k].view_as(p).to(p.dtype))
            at += k
        if LOG_EVERY and (self.updates <= 20 or self.updates % LOG_EVERY == 0):
            print(f"lm update={self.updates} B={float(theta[0]):.12f} w={[float(v) for v in theta[1:]]} "
                  f"rows={st['m']}/{st['count']} s={st['s']:.2e} max|e|={st['emax']:.3e} move={move:.3e}",
                  flush=True)
        return None


def build_model(spec: ModelSpec) -> UnfoldB:
    model = UnfoldB(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(LevenbergMarquardt(model.parameters(), model.stats),
                           should_reuse_batch=lambda context: True)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=BATCH,
    eval_batch_size=EVAL_BATCH,
    max_steps=MAX_STEPS,
)
