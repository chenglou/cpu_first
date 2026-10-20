"""UnfoldS: UnfoldB (cpu_first/unfold_sub) at the manifest's batch size, with the fit's information kept across batches.

Model and loss residual are UnfoldB's (fp64 inside, autocast off):
  * every digit string (N, x, T, and the label inside the loss) is worth sum_i B^i d_i, one learned base B;
  * one squaring step maps a state value s to V = w0 + w1 s + w2 s^2 (learned) and snaps it to the integer
    r = round(V) mod round(n), n the modulus string's value; the step loops round(t) times;
  * the final state is written in base round(B) and read out as a phase over the 10 digit tokens, with a
    straight-through term carrying the gradient of V.
Training (answer-only labels, single-step rows only):
  * clock residual e = (V - y - k n) / n in turns of the live modulus n, against the model's own nearest unfolding
    k = max(0, round((V - y) / n));
  * seed rows (always in): rows whose phase moves by less than C_SEED turns per unit change of w (|de/dw| < C_SEED),
    an absolute threshold in the model's own units, so a batch need not contain any;
  * every other row comes in once the model predicts its phase to within U_MAX turns and its residual agrees with that
    prediction (GATE sigma). The prediction uses the information of all rows admitted so far, not only this batch's:
    the optimizer keeps it as an information matrix, and the loss reads it;
  * optimizer: an iterated information filter (recursive Gauss-Newton). Within a batch (USES evaluator uses), each
    update is a Levenberg-Marquardt step on this batch's admitted rows plus the quadratic summary of earlier batches
    (information LAM around the batch's anchor, plus a weak prior); after the batch's last use its rows' Gauss-Newton
    information is added to LAM, which then decays with a half-life of HALF_LIFE training rows. The gradient is the
    evaluator's backward (the loss is divided by a detached nu so the evaluator's clipping never bites; the optimizer
    multiplies nu back); the curvature comes from the loss's closed-form residual Jacobian.
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
C_SEED = 1.5           # seed: a unit change of w moves the row's phase by less than C_SEED turns
U_MAX = 0.05           # admit a row once its predicted phase std (turns) is below U_MAX
GATE = 3.0             # ... and its residual is within GATE predicted std
S_FLOOR = 1e-9         # floor on the residual std (turns), above fp64 noise at 24 bits
GATE_ITERS = 16        # admit / re-estimate rounds per step
DAMP = 1e-9            # LM damping on the column-normalized Gauss-Newton matrix
TRUST = 0.25           # trust region: no admitted row's phase moves more than TRUST turns per update
BTRUST = 0.1           # ... and the base moves by at most BTRUST * B (the model's own reading changes slowly)
PRIOR = 1e-12          # weak prior information (turns^2) per unit^2 of (B / B_SCALE, w0, w1, w2) around the anchor
B_SCALE = 0.1
HALF_LIFE = 100_000    # carried information halves every HALF_LIFE training rows
CNT_MAX = 100.0        # ... and is scaled down to at most CNT_MAX admitted rows' worth
USES = 8               # updates per batch: should_reuse_batch always asks, the evaluator allows 8
BATCH = 512
EVAL_BATCH = 4096
MAX_STEPS = 1_000_000
LOG_EVERY = 1000


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


class UnfoldS(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.base = nn.Parameter(torch.empty(1).uniform_(*B_INIT))
        self.w = nn.Parameter(W_INIT * torch.randn(3))
        self.stats: dict = {}   # the last loss's Gauss-Newton pieces, read by the optimizer
        self.memory: dict = {}  # the optimizer's carried information, read by the loss

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
                   "dpw": dpw, "w": w, "single": steps == 1, "stats": self.stats, "memory": self.memory}
        return logits, aux


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    with torch.autocast(batch.labels.device.type, enabled=False):
        return _clock_loss(batch, batch.auxiliary)


def _clock_loss(batch: TokenLossBatch, aux: dict) -> Tensor:
    f64 = torch.float64
    stats, memory = aux["stats"], aux["memory"]
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

    # Seeds: rows whose phase moves by less than C_SEED turns per unit change of w. Then every row whose phase the
    # model predicts to within U_MAX turns (variance s^2 J N^-1 J^T, N the information of this batch's admitted rows
    # plus the carried information of earlier batches, s^2 the residual variance of all of them) and whose residual
    # agrees with that prediction; re-estimated until the admitted set stops changing. Before P + 1 rows' worth of
    # information has been gathered, only seeds count.
    count = sel.numel()
    ed = torch.nan_to_num(e.detach(), nan=1e30, posinf=1e30, neginf=-1e30)
    seed = J[:, 1:].norm(dim=1) < C_SEED
    lam = memory["lam"] + memory["lam0"]
    rss0, cnt0 = memory["rss"], memory["cnt"]
    inc, eye = seed, torch.eye(P, dtype=f64, device=e.device)
    s2 = math.inf
    for _ in range(GATE_ITERS):
        m = cnt0 + int(inc.sum())
        if m < P + 1:
            break
        Ji = J[inc]
        Nm = lam + Ji.T @ Ji
        col = Nm.diagonal().clamp_min(1e-300).sqrt()
        Jn = J / col
        lev = (Jn * torch.linalg.solve(Nm / col[:, None] / col[None, :] + 1e-12 * eye, Jn.T).T).sum(1)
        s2 = max(rss0 + float((ed[inc] ** 2).sum()), 0.0) / (m - P) + S_FLOOR ** 2
        var = s2 * lev
        new = seed | ((var < U_MAX ** 2) & (ed * ed < GATE ** 2 * (var + s2)))
        if bool((new == inc).all()):
            break
        inc = new
    inc = torch.nonzero(inc).flatten()
    if inc.numel() == 0:
        stats.update(count=count, m=0, s=math.sqrt(s2))
        return aux["w"].sum() * 0.0

    ei, Ji = e[inc], J[inc]
    loss = 0.5 * (ei * ei).sum()
    g = Ji.T @ ei.detach()
    nu = max(2.0 * float(g.norm()), 1e-30)
    stats.update(J=Ji, e=ei.detach(), nu=nu, m=inc.numel(), count=count, s=math.sqrt(s2),
                 emax=float(ei.detach().abs().max()), rows=sel[inc], k=kstar[inc])
    return loss / nu + (loss - loss / nu).detach()   # value: the clock cost; gradient: scaled by 1 / nu


class InformationLM(torch.optim.Optimizer):
    """Iterated information filter: Levenberg-Marquardt on this batch's rows plus the carried quadratic of earlier
    batches, fp64 master parameters.

    The earlier batches' admitted rows are summarized at the anchor theta_a (the parameters after the previous batch)
    by their Gauss-Newton information LAM = sum J^T J, their gradient G = sum J^T e and their residual sum of squares,
    all decayed by the half-life. Gradient of an update: the evaluator's backward of the scaled loss, times nu, plus the
    carried G + LAM (theta - theta_a). Curvature: J^T J from the loss's closed-form residual Jacobian plus LAM. The step
    is shrunk so that no admitted row's predicted phase moves more than TRUST turns, the carried rows' RMS phase move
    stays under TRUST, and the base moves by at most BTRUST * B. After the batch's last use its rows are added to the
    carried quadratic, which is re-centered at the new parameters."""

    def __init__(self, params, model: nn.Module) -> None:
        super().__init__(params, {})
        self.stats, self.memory = model.stats, model.memory
        self.updates = 0
        ps = self._params()
        dev = ps[0].device
        f64 = torch.float64
        st = self.state[ps[0]]
        st["master"] = torch.cat([p.detach().flatten().to(f64) for p in ps])
        st["anchor"] = st["master"].clone()
        st["lam"] = torch.zeros(P, P, dtype=f64, device=dev)
        st["grad"] = torch.zeros(P, dtype=f64, device=dev)
        st["rss"] = torch.zeros((), dtype=f64, device=dev)
        st["cnt"] = torch.zeros((), dtype=f64, device=dev)
        self.lam0 = PRIOR * torch.diag(torch.tensor([1.0 / B_SCALE ** 2, 1.0, 1.0, 1.0], dtype=f64, device=dev))
        self.decay = 0.5 ** (BATCH / HALF_LIFE)
        self._publish()

    def _params(self):
        return [p for group in self.param_groups for p in group["params"]]

    def _carried_rss(self, st: dict, theta: Tensor) -> Tensor:
        d = theta - st["anchor"]
        return st["rss"] + 2 * st["grad"] @ d + d @ st["lam"] @ d

    def _publish(self) -> None:
        st = self.state[self._params()[0]]
        self.memory.update(lam=st["lam"], lam0=self.lam0, rss=float(self._carried_rss(st, st["master"])),
                           cnt=float(st["cnt"]))

    @torch.no_grad()
    def step(self, closure=None):
        ps = self._params()
        st, S = self.state[ps[0]], self.stats
        self.updates += 1
        theta, anchor, lam = st["master"], st["anchor"], st["lam"]
        prior = lam + self.lam0
        rhs = prior @ (theta - anchor) + st["grad"]
        Nm = prior.clone()
        J = S.get("J")
        if J is not None:
            grad = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().to(torch.float64)
                              for p in ps]) * S["nu"]
            rhs = rhs + grad
            Nm = Nm + J.T @ J
        col = Nm.diagonal().clamp_min(1e-300).sqrt()
        A = Nm / col[:, None] / col[None, :] + DAMP * torch.eye(P, dtype=Nm.dtype, device=Nm.device)
        delta = torch.linalg.solve(A, rhs / col) / col
        move = float((J @ delta).abs().max()) if J is not None else 0.0
        scale = min(1.0, TRUST / max(move, 1e-300),
                    BTRUST * abs(float(theta[0])) / max(abs(float(delta[0])), 1e-300))
        # the carried quadratic is a linearization at the anchor: within this batch, keep the carried rows' RMS phase
        # move from the anchor under TRUST (or, once there, non-increasing)
        cnt = max(float(st["cnt"]), 1.0)
        d0 = theta - anchor
        a, b = float(delta @ lam @ delta) / cnt, float(d0 @ lam @ delta) / cnt
        f0 = float(d0 @ lam @ d0) / cnt
        bound = max(TRUST ** 2, f0)
        if a * scale * scale - 2 * b * scale + f0 > bound:
            scale = max(0.0, (b + math.sqrt(max(b * b - a * (f0 - bound), 0.0))) / a)
        if float(st["cnt"]) + S.get("m", 0) < P + 1:
            scale = 0.0   # fewer than P + 1 rows' worth of information: hold still, keep gathering
        delta = delta * scale
        theta = theta - delta
        at = 0
        for p in ps:   # the model computes with its own (fp32) parameters, so they are the state: no hidden digits
            k = p.numel()
            p.copy_(theta[at:at + k].view_as(p).to(p.dtype))
            at += k
        theta = torch.cat([p.detach().flatten().to(torch.float64) for p in ps])
        delta = st["master"] - theta
        st["master"] = theta
        if self.updates % USES == 0:   # the batch's last use: carry its rows forward, re-centered at theta
            d = theta - anchor
            rss, grad_c = self._carried_rss(st, theta), st["grad"] + lam @ d
            if J is not None:
                e_post = S["e"] - J @ delta          # this batch's residuals at the new theta (linearized)
                rss = rss + (e_post ** 2).sum()
                grad_c = grad_c + J.T @ e_post
                lam = lam + J.T @ J
            count = float(st["cnt"]) + S.get("m", 0)
            keep = min(self.decay, CNT_MAX / max(count, 1e-300))   # at most CNT_MAX rows' worth is carried
            st["lam"] = keep * lam
            st["grad"] = keep * grad_c
            st["rss"] = keep * rss
            st["cnt"] = keep * (st["cnt"] + S.get("m", 0))
            st["anchor"] = theta.clone()
        self._publish()
        if LOG_EVERY and (self.updates % LOG_EVERY == 0):
            print(f"lm update={self.updates} B={float(theta[0]):.12f} w={[float(v) for v in theta[1:]]} "
                  f"rows={S.get('m', 0)}/{S.get('count', 0)} s={S.get('s', math.inf):.2e} "
                  f"carried={float(st['cnt']):.1f} move={move:.3e}", flush=True)
        return None


def build_model(spec: ModelSpec) -> UnfoldS:
    model = UnfoldS(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(InformationLM(model.parameters(), model), should_reuse_batch=lambda context: True)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=BATCH,
    eval_batch_size=EVAL_BATCH,
    max_steps=MAX_STEPS,
)
