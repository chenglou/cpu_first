"""longdiv with fixed slot meanings: one shared subtractor fed the exact digits of q*N, bootstrapped by the leading-digit
estimator one quotient digit at a time, then choosing by its own borrow. ../tinycore/TINYCORE.md §6's predicted fix,
in microlab's MLP-cell datapath.

FINDINGS_quotient.md: trained from answers, the digit cell and the quotient-digit choice each learn when the other is
given and stall together. TINYCORE.md traces that to the learned slot labels (a symmetric saddle whose exit lands in
permuted or merged lookup arrangements) and predicts this route:
1. Slot meanings fixed by construction. longdiv's learned slot embedding eq goes. One shared subtractor cell
   (r_j, m_j, borrow) -> (digit, borrow), an MLP like longdiv's cell but with a binary borrow, runs every trial, and
   slot q is fed the exact digits m of q*N (multiplying N by a known digit: computed, not learned).
2. The estimator bootstraps the subtractor: longdiv_est's soft choice (window: the top kr digits of 10R + v_t and the
   top kn of N), starting undecided, staged one quotient digit at a time. Stage k trains on v < N*10^k and runs the k
   division steps whose digit can be nonzero, from R = v // 10^k (below N). At the last stage, k = w and v < N^2, that
   is ext_quotient_prefix's start (the dividend's top w digits). `start="top"` instead always starts there and runs w
   steps, with only the data staged.
3. Then longdiv's own choice, the largest q whose trial does not borrow, on v < N^2 for `switch_steps` more steps.
`slots="learned"` runs the same route with longdiv's own cell (a learned label per slot, carries 0..9): the control.

Measures, after each phase: exact with the model's own choices (microlab's 2,000 test examples), exact given the true
quotient digits (is the subtractor right?), the hard datapath (argmax choices, snapped digits), the slot maps
(ext_quotient.choices), held-out moduli (holdout=k runs), the subtractor's 200 one-hot rows (all, and those the trials
visit at the trained width), and the same weights at other widths. Rows go to microlab/results_quotient_shared.jsonl.

Two changes found here, both options: `hold` gives every slot equal odds for the first steps (the MLP estimator
otherwise commits to one slot within ~100 steps), and `hard_borrow` makes the borrow a bit (straight-through).
Diagnostics: `start` (the estimator's first steps), `probe` (how crisp and calibrated a saved subtractor's borrows
are, and what the own borrow makes of them).

    python -m microlab.ext_quotient_shared selftest
    python -m microlab.ext_quotient_shared window --bits 6,7,10
    python -m microlab.ext_quotient_shared run --bits 6 --seed 0 --stage 200 --final 200 --switch_steps 200 --nolog
    python -m microlab.ext_quotient_shared start --seeds 0,1 --hold 0
    python -m microlab.ext_quotient_shared probe shared_7_s0_ho0_hold400_hardb_a.pt --bits 7 --hard_borrow
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import ext_ladder
from .ext_ladder import held_out, keys
from .ext_quotient import _number, choices, top
from .ext_quotient import exact_cell as exact_slot_cell
from .models import MODELS, model
from .run import _tensors
from .tasks import TASKS, Batch, _odd_moduli, digits, ndig

RESULTS = os.path.join(os.path.dirname(__file__), "results_quotient_shared.jsonl")
TRANSFER = (6, 7, 10, 14, 20)


def multiples(N, P):
    """One-hot digits of q*N for q = 0..9 in P places: N [B, w] digits (LSB-first) -> [B, 10 (q), P, 10]."""
    Nv = (N * 10 ** torch.arange(N.shape[1], device=N.device)).sum(1)
    qN = Nv[:, None] * torch.arange(10, device=N.device)
    return F.one_hot(qN[..., None] // 10 ** torch.arange(P, device=N.device) % 10, 10).float()


def exact_cell(r, m, c):
    """The subtractor's truth table on one-hot inputs: the digit and borrow of r - m - c."""
    x = r.argmax(-1) - m.argmax(-1) - c.argmax(-1)
    return F.one_hot(x % 10, 10).float(), F.one_hot((x < 0).long(), 2).float()


@model("longdiv_shared")
class LongDivShared(nn.Module):
    def __init__(self, in_widths, out_width, aux_widths, task="mod", slots="shared", choose="est", d=32, hidden=128,
                 kr=3, kn=2, est_hidden=128, hard_borrow=False, **kw):
        super().__init__()
        self.slots, self.choose, self.kr, self.kn, self.hard_borrow = slots, choose, kr, kn, hard_borrow
        self.alpha = 1.0                                                        # the own borrow's share (handover)
        self.detach_digits = False                                              # lessons for the choice alone
        self.C = 2 if slots == "shared" else 10                                 # binary borrow, or longdiv's carries
        self.er, self.en = nn.Linear(10, d, bias=False), nn.Linear(10, d, bias=False)   # r_j; m_j (or N's digit)
        self.ec = nn.Linear(self.C, d, bias=False)
        if slots == "learned":                                                  # longdiv's ten trial digits
            self.eq = nn.Parameter(torch.randn(10, d) * 0.3)
        self.mlp = nn.Sequential(nn.Linear((3 if slots == "shared" else 4) * d, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 10 + self.C))
        self.est = nn.Sequential(nn.Linear(10 * (kr + kn), est_hidden), nn.GELU(),
                                 nn.Linear(est_hidden, est_hidden), nn.GELU(), nn.Linear(est_hidden, 10))

    def cell(self, r, m, c):
        """One place of every slot's trial: r, m [B, 10 (q), 10] and the incoming borrow c [B, 10, C], as distributions.
        Shared: m is the digit of q*N; learned: m is N's digit and the slot's label eq says what to do with it."""
        z = [self.er(r), self.en(m), self.ec(c)]
        if self.slots == "learned":
            z.insert(2, self.eq.expand(r.shape[0], 10, -1))
        o = self.mlp(torch.cat(z, -1))
        return o[..., :10].softmax(-1), o[..., 10:].softmax(-1)

    def slot_digits(self, N, P):
        """What each slot subtracts, per place: [B, 10 (q), P, 10]."""
        if self.slots == "shared":
            return multiples(N, P)
        return F.one_hot(F.pad(N, (0, P - N.shape[1])), 10).float()[:, None].expand(N.shape[0], 10, P, 10)

    def trials(self, Rp, m):
        """Every slot's trial on Rp [B, P, 10]: the digits [B, 10 (q), P, 10] and P(no final borrow) [B, 10]."""
        B = Rp.shape[0]
        c = F.one_hot(torch.zeros(B, 10, dtype=torch.long, device=Rp.device), self.C).float()
        D = []
        for j in range(Rp.shape[1]):                                            # LSB-first within the trial
            d, c = self.cell(Rp[:, j, None].expand(B, 10, 10), m[:, :, j], c)
            if self.hard_borrow and (j < Rp.shape[1] - 1 or self.hard_borrow == "all"):   # a bit (straight
                c = c + (F.one_hot(c.argmax(-1), self.C).float() - c).detach()  # through); True: between places only
            D.append(d)
        return torch.stack(D, 2), c[..., 0]

    def estimate(self, Rp, N):
        """P(q) from the top kr places of 10R + v_t [B, w+1, 10] and the top kn digits of N [B, w] (longdiv_est's)."""
        n = F.one_hot(N, 10).float()
        return self.est(torch.cat([top(Rp, self.kr).flatten(1), top(n, self.kn).flatten(1)], 1)).softmax(-1)

    def reduce(self, v, N, q=None, k=None, hard=False):
        """k division steps (default w), t = k-1 .. 0, from R = v // 10^k (the dividend's top digits; below N when
        v < N*10^k). v [B, L, 10] (LSB-first); q: the true quotient digits [B, w], which then replace the choice;
        hard: argmax choices and snapped digits. Returns R [B, w, 10] and the choices and estimates [B, k, 10] (by t)."""
        B, w = v.shape[0], N.shape[1]
        k = w if k is None else k
        m = self.slot_digits(N, w + 1)
        zero = F.one_hot(torch.zeros(B, 1, dtype=torch.long, device=v.device), 10).float()
        R = torch.cat([v[:, k:k + w], zero], 1)                                 # w+1 places, top one 0
        choice, est = [None] * k, [None] * k
        for t in reversed(range(k)):                                           # dividend digits, MSB first
            Rp = torch.cat([v[:, t:t + 1], R[:, :w]], 1)                       # 10R + v_t
            D, valid = self.trials(Rp, m)
            pe = self.estimate(Rp, N)
            if q is not None:                                                  # the quotient digit, given
                wq = F.one_hot(q[:, t], 10).float()
            elif self.choose == "borrow":                                     # longdiv: the largest valid q
                higher = torch.cat([torch.flip(torch.cumprod(torch.flip(1 - valid, [1]), 1), [1])[:, 1:],
                                    valid.new_ones(B, 1)], 1)
                wq = valid * higher
                wq = wq / wq.sum(1, keepdim=True).clamp_min(1e-9)
                if self.alpha < 1:                                             # handing over from the estimator
                    wq = (1 - self.alpha) * pe + self.alpha * wq
            elif self.choose == "uniform":                                    # undecided: equal odds on every slot
                wq = torch.full_like(pe, 0.1)
            else:
                wq = pe
            if hard:
                wq = F.one_hot(wq.argmax(1), 10).float()
            choice[t], est[t] = wq, pe
            R = (wq[:, :, None, None] * (D.detach() if self.detach_digits else D)).sum(1)
            if hard:
                R = F.one_hot(R.argmax(-1), 10).float()
        return R[:, :w], torch.stack(choice, 1), torch.stack(est, 1)

    def forward(self, inputs):
        R, choice, est = self.reduce(F.one_hot(inputs["v"], 10).float(), inputs["N"], inputs.get("q"))
        return {"target": R.clamp_min(1e-9).log(), "choice": choice, "est": est}


def sample(n, bits, k, rng, holdout=0):
    """n examples of stage k: odd N of `bits` bits (minus the held-out moduli), v uniform below N*10^k, or below N^2 at
    the last stage (k = w: mod's own distribution). Digits as microlab's mod task (v has 2w places)."""
    w = ndig((1 << bits) - 1)
    N = np.empty(0, np.int64)
    while len(N) < n:
        m = _odd_moduli(2 * n, bits, rng)
        N = np.concatenate([N, m[(m // 2) % holdout != 0] if holdout else m])
    N = N[:n]
    top_v = N * N if k >= w else N * 10 ** k
    v = np.minimum((rng.random(n) * top_v.astype(np.float64)).astype(np.int64), top_v - 1)
    return Batch({"N": digits(N, w), "v": digits(v, 2 * w)}, digits(v % N, w), {"q": digits(v // N, w)})


@torch.no_grad()
def exact_on(net, x, y, **kw):
    """Per-example exact match [B] of net.reduce on x (kw: q, k, hard)."""
    R, _, _ = net.reduce(F.one_hot(x["v"], 10).float(), x["N"], **kw)
    return (R.argmax(-1) == y).all(1)


@torch.no_grad()
def table(net, bits, n=20_000, seed=5):
    """The shared subtractor on its 200 one-hot rows (r, m, borrow in): the share whose digit and borrow out are both
    right; the same on the rows the true slot's trial visits at this width (exact long division on the test
    distribution); the borrow alone on the rows the over-subtracting trials visit (the own borrow reads only their
    final borrows); and the wrong rows."""
    r, m, c = (a.reshape(-1) for a in torch.meshgrid(torch.arange(10), torch.arange(10), torch.arange(2), indexing="ij"))
    x = r - m - c
    d, b = net.cell(F.one_hot(r, 10).float()[:, None], F.one_hot(m, 10).float()[:, None],
                    F.one_hot(c, 2).float()[:, None])
    ok_b = (b[:, 0].argmax(-1) == (x < 0).long()).numpy()
    ok = (d[:, 0].argmax(-1) == x % 10).numpy() & ok_b
    w = ndig((1 << bits) - 1)
    bt = sample(n, bits, w, np.random.default_rng(seed))
    N, vd = _number(bt.inputs["N"]), bt.inputs["v"]
    qN = N[:, None] * np.arange(10)
    path, over, visits = np.zeros(200, bool), np.zeros(200, bool), np.zeros(200, np.int64)
    R = vd[:, w:] @ (10 ** np.arange(w))
    for t in reversed(range(w)):
        Rp = 10 * R + vd[:, t]
        q = Rp // N
        c_ = np.zeros((n, 10), np.int64)
        for j in range(w + 1):                                                 # every trial, every place
            rj, mj = Rp[:, None] // 10 ** j % 10, qN // 10 ** j % 10
            row = rj * 20 + mj * 2 + c_
            path[row[np.arange(n), q]] = True
            np.add.at(visits, row[np.arange(n), q], 1)
            over[row[np.arange(10) > q[:, None]]] = True
            c_ = (rj - mj - c_ < 0).astype(np.int64)
        R = Rp - q * N
    rest = ~path & ~over
    part = lambda a, s: None if not s.any() else round(float(a[s].mean()), 4)
    return {"right": round(float(ok.mean()), 4), "path_rows": int(path.sum()), "right_path": part(ok, path),
            "over_rows": int(over.sum()), "borrow_right_over": part(ok_b, over), "other_rows": int(rest.sum()),
            "right_other": part(ok, rest), "wrong": [[int(r[i]), int(m[i]), int(c[i])] for i in np.flatnonzero(~ok)],
            "pred": torch.stack([d[:, 0].argmax(-1), b[:, 0].argmax(-1)], 1).tolist(), "visits": visits.tolist()}


def scores(net, bits, train_keys, holdout=0, transfer=TRANSFER):
    """Everything measured on the trained width (microlab's 2,000 test examples, seed 10,000) and at other widths."""
    task = TASKS["mod"]
    exact, per_digit, split = ext_ladder.evaluate_split(net, task, bits, train_keys, holdout=holdout)
    b = task.sample(2000, bits, np.random.default_rng(10_000))
    x, y, aux = _tensors(b, "cpu")
    given, hard = exact_on(net, x, y, q=aux["q"]).numpy(), exact_on(net, x, y, hard=True).numpy()
    out = {"exact": round(exact, 4), "per_digit": [round(v, 3) for v in per_digit], **split,
           "exact_given": round(float(given.mean()), 4), "exact_hard": round(float(hard.mean()), 4)}
    if holdout:
        ho = held_out(b.inputs, holdout)
        out |= {"exact_given_heldout_moduli": round(float(given[ho].mean()), 4),
                "exact_hard_heldout_moduli": round(float(hard[ho].mean()), 4)}
    ch = choices(net, "mod", bits, holdout=holdout)
    out["choice"] = {k: ch[k] for k in ("choice_raw", "choice_mapped", "choice_map", "est_raw", "est_mapped", "est_map",
                                        "by_top_digit_of_N")}
    if net.slots == "shared":
        out["table"] = table(net, bits)
    out["transfer"] = {}
    for tb in transfer:
        bt = task.sample(2000, tb, np.random.default_rng(10_000))
        xt, yt, at = _tensors(bt, "cpu")
        out["transfer"][str(tb)] = {"exact": round(exact_on(net, xt, yt).float().mean().item(), 4),
                                    "exact_given": round(exact_on(net, xt, yt, q=at["q"]).float().mean().item(), 4),
                                    "exact_hard": round(exact_on(net, xt, yt, hard=True).float().mean().item(), 4)}
    return out


def run(bits=6, seed=0, slots="shared", lr=3e-3, batch=256, stage=1000, final=1500, switch_steps=2000, every=100,
        holdout=0, kr=3, kn=2, warmup=100, stage_pass=0.99, start="tight", staged=True, first="est", hold=0,
        handover=0, lr_b=None, hard_borrow=False, hard_b=None, freeze_digits=0, from_a=None, transfer=TRANSFER, tag="",
        threads=None, log=RESULTS, save=None, verbose=False):
    """Phase A: the estimator chooses (or, first="borrow", the own borrow already), stage k = 1..w (at most `stage`
    steps each, `final` at the last; a stage ends early once exact on its own validation sample reaches stage_pass;
    staged=False: only the last stage), AdamW at lr after a linear warmup. hold > 0: for the first `hold` steps every
    slot gets equal odds (the estimator is not consulted), so the subtractor's first lessons cannot depend on the
    choice. hard_borrow=True: the borrow passed between places is a bit (argmax, straight-through gradient); "all":
    the final borrow too, so the own borrow reads bits. Phase B: the own borrow chooses, on v < N^2, `switch_steps`
    steps of a fresh AdamW (OneCycle) on the cell alone. handover > 0: the choice is (1 - a) * estimate + a * own
    borrow, a rising linearly from 0 to 1 over the first `handover` steps, so the borrows are taught while the digits
    still get the estimator's (mostly right) lessons. hard_b: phase B's hard_borrow (default: phase A's).
    freeze_digits > 0: for the first `freeze_digits` steps of phase B the trials' digits are detached from the answer,
    so the loss teaches only the choice, i.e. the borrows. from_a: start phase B from saved phase-A weights."""
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    in_w, out_w, _ = TASKS["mod"].widths(bits)
    w = in_w["N"]
    net = MODELS["longdiv_shared"](in_w, out_w, {}, slots=slots, kr=kr, kn=kn, choose=first, hard_borrow=hard_borrow)
    t0, seen = time.time(), []

    def train_step(opt, sched, k):
        b = sample(batch, bits, k, rng, holdout)
        seen.append(keys(b.inputs))
        x, y, _ = _tensors(b, "cpu")
        R, _, _ = net.reduce(F.one_hot(x["v"], 10).float(), x["N"], k=k if start == "tight" else w)
        loss = F.cross_entropy(R.clamp_min(1e-9).log().flatten(0, 1), y.flatten())
        net.zero_grad(set_to_none=True)                                        # phase B: no stale estimator grads
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        return loss.item()

    def validation(k):
        x, y, _ = _tensors(sample(2000, bits, k, np.random.default_rng(20_000 + k), holdout), "cpu")
        return lambda: round(exact_on(net, x, y, k=k if start == "tight" else w).float().mean().item(), 4)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    curve, stages, step = [], [], 0
    if from_a:
        net.load_state_dict(torch.load(from_a))
    for k in ([] if from_a else range(1, w + 1) if staged else [w]):
        val, budget, losses = validation(k), (final if k == w else stage), []
        for i in range(budget + 1):
            net.choose = "uniform" if step < hold else first
            if i % every == 0:
                ex = val()
                curve.append([step, k, ex, round(float(np.mean(losses[-every:])), 4) if losses else None])
                if verbose:
                    print(f"  A step {step} k {k} exact {ex:.3f} loss {curve[-1][3]}", flush=True)
                if ex >= stage_pass and step >= hold:
                    break
            if i == budget:
                break
            losses.append(train_step(opt, sched, k))
            step += 1
        stages.append({"k": k, "steps": i, "exact": ex})
    after_a = scores(net, bits, np.unique(np.concatenate(seen or [np.zeros(0, np.int64)])), holdout, transfer)
    after_a["seconds"] = round(time.time() - t0, 1)
    if save:                                                                   # the weights after each phase
        torch.save(net.state_dict(), f"{save}_a.pt")
    if verbose:
        print(f"  A done: {show_scores(after_a)}", flush=True)

    net.choose = "borrow"
    net.hard_borrow = hard_borrow if hard_b is None else hard_b
    cell_params = [p for n_, p in net.named_parameters() if not n_.startswith("est.")]
    lr_b = lr_b or lr
    opt = torch.optim.AdamW(cell_params, lr=lr_b, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr_b, total_steps=max(switch_steps, 1), pct_start=0.1)
    val, curve_b, losses, hit = validation(w), [], [], None
    for i in range(switch_steps + 1):
        net.alpha = min(1.0, i / handover) if handover else 1.0
        net.detach_digits = i < freeze_digits
        if i % every == 0:
            ex = val()
            curve_b.append([i, ex, round(float(np.mean(losses[-every:])), 4) if losses else None])
            hit = i if hit is None and ex >= stage_pass and net.alpha >= 1 else hit
            if verbose:
                print(f"  B step {i} exact {ex:.3f} loss {curve_b[-1][2]}", flush=True)
        if i == switch_steps:
            break
        losses.append(train_step(opt, sched, w))
    after_b = scores(net, bits, np.unique(np.concatenate(seen)), holdout, transfer)
    if save:
        torch.save(net.state_dict(), f"{save}_b.pt")
    r = {"tag": tag, "bits": bits, "w": w, "seed": seed, "slots": slots, "kr": kr, "kn": kn, "lr": lr, "batch": batch,
         "stage": stage, "final": final, "switch_steps": switch_steps, "warmup": warmup, "holdout": holdout,
         "start": start, "staged": staged, "first": first, "hold": hold, "handover": handover, "lr_b": lr_b, "hard_borrow": hard_borrow,
         "hard_b": hard_b, "freeze_digits": freeze_digits, "from_a": from_a, "params": sum(p.numel() for p in net.parameters()),
         "stages": stages, "curve_a": curve, "after_a": after_a, "curve_b": curve_b, "hit_b": hit, "after_b": after_b,
         "seconds": round(time.time() - t0, 1), "threads": torch.get_num_threads()}
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(r) + "\n")
    return r


def show_scores(s):
    tr = " ".join(f"@{b}:{v['exact']:.2f}/{v['exact_given']:.2f}" for b, v in s["transfer"].items())
    ho = "" if s.get("exact_heldout_moduli") is None else f"  held-out moduli {s['exact_heldout_moduli']:.3f}"
    tb = "" if "table" not in s else (f"  rows right {s['table']['right']:.3f} (path {s['table']['right_path']:.3f}"
                                      f" of {s['table']['path_rows']}, over-trial borrows {s['table']['borrow_right_over']})")
    return (f"exact {s['exact']:.3f} given {s['exact_given']:.3f} hard {s['exact_hard']:.3f}{ho}{tb}"
            f"  map {s['choice']['choice_map']}  widths own/given {tr}")


def show(r):
    return (f"{r['bits']} bits {r['slots']} seed {r['seed']} ho{r['holdout']} {r['tag']}: stages "
            + " ".join(f"k{s['k']}:{s['steps']}/{s['exact']:.2f}" for s in r["stages"])
            + f"\n  after estimator: {show_scores(r['after_a'])}"
            + f"\n  after own borrow (hit {r['hit_b']}): {show_scores(r['after_b'])}  {r['seconds']}s")


def selftest(bits=(6, 7, 10, 14, 20), n=400, seed=0):
    """With the exact subtractor: the own borrow is exact at every width (from the top w digits), the true digits are
    exact at every stage from R = v // 10^k, and an oracle estimate is exact. The learned-slot trials with longdiv's
    exact cell are exact with the own borrow. A learned net runs forward and backward at every stage."""
    for b in bits:
        w = ndig((1 << b) - 1)
        in_w, out_w, _ = TASKS["mod"].widths(b)
        bt = TASKS["mod"].sample(n, b, np.random.default_rng(seed))
        x, y, aux = _tensors(bt, "cpu")
        net = MODELS["longdiv_shared"](in_w, out_w, {}, choose="borrow")
        net.cell = exact_cell
        own, given = exact_on(net, x, y).float().mean(), exact_on(net, x, y, q=aux["q"]).float().mean()
        stage = []
        for k in range(1, w + 1):
            sb = sample(n, b, k, np.random.default_rng(seed + k))
            xs, ys, a_s = _tensors(sb, "cpu")
            stage.append(f"k{k} {exact_on(net, xs, ys, k=k).float().mean():.3f}/"
                         f"{exact_on(net, xs, ys, k=k, q=a_s['q']).float().mean():.3f}")
        net.choose = "est"
        Nv = _number(bt.inputs["N"])
        net.estimate = lambda Rp, N_: F.one_hot((torch.from_numpy(_number(Rp.argmax(-1).numpy()) // Nv)).clamp(0, 9), 10).float()
        oracle = exact_on(net, x, y).float().mean()
        slot = MODELS["longdiv_shared"](in_w, out_w, {}, slots="learned", choose="borrow")
        slot.cell = exact_slot_cell
        print(f"{b} bits exact subtractor: own borrow {own:.3f}, given {given:.3f}, oracle estimate {oracle:.3f};"
              f" stages (own/given) {' '.join(stage)}; learned slots, longdiv's exact cell: "
              f"{exact_on(slot, x, y).float().mean():.3f}")
    torch.manual_seed(0)
    in_w, out_w, _ = TASKS["mod"].widths(7)
    for slots in ("shared", "learned"):
        net = MODELS["longdiv_shared"](in_w, out_w, {}, slots=slots)
        for k in (1, 2, 3):
            xs, ys, _ = _tensors(sample(64, 7, k, np.random.default_rng(k)), "cpu")
            R, ch, est = net.reduce(F.one_hot(xs["v"], 10).float(), xs["N"], k=k)
            F.cross_entropy(R.clamp_min(1e-9).log().flatten(0, 1), ys.flatten()).backward()
        print(f"{slots}: params {sum(p.numel() for p in net.parameters())}, choice {tuple(ch.shape)},"
              f" est at init: max P {est.max().item():.3f} min P {est.min().item():.3f}")


def window(bits=(6, 7, 10), kr=3, kn=2, n=100_000, seed=0):
    """For every stage k: how far the estimator's window fixes the quotient digit on the k steps the stage runs from
    R = v // 10^k (the best window-only estimator, the most common digit per window, as ext_quotient.window)."""
    rows = []
    for b in bits:
        w = ndig((1 << b) - 1)
        for k in range(1, w + 1):
            bt = sample(n, b, k, np.random.default_rng(seed))
            N, vd = _number(bt.inputs["N"]), bt.inputs["v"]
            R, ks, qs = vd[:, k:] @ (10 ** np.arange(vd.shape[1] - k)), [], []
            for t in reversed(range(k)):
                Rp = 10 * R + vd[:, t]
                q = Rp // N
                rw = Rp // 10 ** (w + 1 - kr) if w + 1 >= kr else Rp * 10 ** (kr - w - 1)
                nw = N // 10 ** (w - kn) if w >= kn else N * 10 ** (kn - w)
                ks.append(rw * 10 ** kn + nw)
                qs.append(q)
                R = Rp - q * N
            K, Q = np.stack(ks, 1).ravel(), np.stack(qs, 1).ravel()
            uk, inv = np.unique(K, return_inverse=True)
            counts = np.zeros((len(uk), 10), np.int64)
            np.add.at(counts, (inv, Q), 1)
            right = (counts.argmax(1)[inv] == Q).reshape(n, k)
            rows.append({"bits": b, "k": k, "kr": kr, "kn": kn, "windows": int(len(uk)),
                         "step_right": round(float(right.mean()), 4), "all_steps_right": round(float(right.all(1).mean()), 4)})
    return rows


def start(seeds=(0, 1), hold=0, bits=6, steps=400, every=50):
    """The first `steps` of stage 1 (phase A's training): the estimator's entropy (nats), P(true digit), the slots its
    argmax uses, the subtractor's rows right and exact given the true digits. Seconds at 6 bits."""
    in_w, out_w, _ = TASKS["mod"].widths(bits)
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = MODELS["longdiv_shared"](in_w, out_w, {})
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
        x, y, a = _tensors(sample(2000, bits, 1, np.random.default_rng(21)), "cpu")
        for step in range(steps + 1):
            if step % every == 0:
                net.choose = "est"
                with torch.no_grad():
                    _, _, est = net.reduce(F.one_hot(x["v"], 10).float(), x["N"], k=1)
                p = est[:, 0]
                used = torch.bincount(p.argmax(1), minlength=10)
                print(f"seed {seed} hold {hold} step {step}: entropy {-(p * p.clamp_min(1e-9).log()).sum(1).mean():.2f}"
                      f"  P(true) {p[torch.arange(len(p)), a['q'][:, 0]].mean():.3f}  argmax slots "
                      f"{ {i: int(c) for i, c in enumerate(used) if c} }  rows right {table(net, bits, n=2000)['right']:.2f}"
                      f"  given {exact_on(net, x, y, k=1, q=a['q']).float().mean():.3f}", flush=True)
            net.choose = "uniform" if step < hold else "est"
            xb, yb, _ = _tensors(sample(256, bits, 1, rng), "cpu")
            R, _, _ = net.reduce(F.one_hot(xb["v"], 10).float(), xb["N"], k=1)
            loss = F.cross_entropy(R.clamp_min(1e-9).log().flatten(0, 1), yb.flatten())
            net.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()


@torch.no_grad()
def probe(path, bits, hard_borrow=False, n=2000, widths=(6, 7, 10, 14)):
    """Saved weights (a run's <stem>_a.pt or _b.pt): on the true path, P(borrow out) of the true trial by its true
    borrow, per step and place; the final P(no borrow) of every trial by whether it really borrows (what the own
    borrow reads); and exact with the own borrow, reading the final borrow as is or as a bit (and every borrow a bit)."""
    in_w, out_w, _ = TASKS["mod"].widths(bits)
    net = MODELS["longdiv_shared"](in_w, out_w, {}, choose="borrow", hard_borrow=hard_borrow)
    net.load_state_dict(torch.load(path))
    b = TASKS["mod"].sample(n, bits, np.random.default_rng(10_000))
    x, y, a = _tensors(b, "cpu")
    w = x["N"].shape[1]
    Nv = (x["N"] * 10 ** torch.arange(w)).sum(1)
    m = multiples(x["N"], w + 1)
    Rv = (x["v"][:, w:] * 10 ** torch.arange(w)).sum(1)
    for t in reversed(range(w)):                                               # exact long division, one-hot inputs
        Rpv = 10 * Rv + x["v"][:, t]
        Rp = F.one_hot(torch.stack([Rpv // 10 ** j % 10 for j in range(w + 1)], 1), 10).float()
        q, c, cs, out = a["q"][:, t], torch.zeros(n, dtype=torch.long), F.one_hot(torch.zeros(n, 10, dtype=torch.long), 2).float(), []
        for j in range(w + 1):
            rj, mj = Rpv // 10 ** j % 10, Nv * q // 10 ** j % 10
            tb = (rj - mj - c < 0).long()
            _, soft = net.cell(Rp[:, j, None].expand(-1, 10, -1), m[:, :, j], cs)
            cs = F.one_hot(soft.argmax(-1), 2).float() if hard_borrow and j < w else soft
            pb = soft[torch.arange(n), q, 1]
            out.append(" ".join(f"{pb[tb == k].mean():.2f}" if (tb == k).any() else " · " for k in (0, 1)))
            c = tb
        _, valid = net.trials(Rp, m)
        borrows = Rpv[:, None] < Nv[:, None] * torch.arange(10)
        print(f"step t={t}: P(borrow out) of the true trial with true borrow 0 / 1, by place: {' | '.join(out)};"
              f"  final P(no borrow): trials that do not borrow {valid[~borrows].mean():.3f}, that borrow "
              f"{valid[borrows].mean():.3f} ({(valid[borrows] > 0.1).float().mean():.2f} of them above 0.1)")
        Rv = Rpv % Nv
    res = {}
    for bw in widths:
        bt = TASKS["mod"].sample(n, bw, np.random.default_rng(10_000))
        xb, yb, _ = _tensors(bt, "cpu")
        as_is = exact_on(net, xb, yb).float().mean().item()
        trials = net.trials
        net.trials = lambda Rp_, m_: (lambda D, v: (D, v + ((v > 0.5).float() - v).detach()))(*trials(Rp_, m_))
        final_bit = exact_on(net, xb, yb).float().mean().item()
        net.trials, net.hard_borrow = trials, "all"
        every_bit = exact_on(net, xb, yb).float().mean().item()
        net.hard_borrow = hard_borrow
        res[bw] = f"{as_is:.3f} / {final_bit:.3f} / {every_bit:.3f}"
    print(f"own borrow, exact (final borrow as is / as a bit / every borrow a bit): {res}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run")
    q.add_argument("--bits", type=int, default=6)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--slots", default="shared", choices=("shared", "learned"))
    q.add_argument("--stage", type=int, default=1000)
    q.add_argument("--final", type=int, default=1500)
    q.add_argument("--switch_steps", type=int, default=2000)
    q.add_argument("--holdout", type=int, default=0)
    q.add_argument("--threads", type=int, default=2)
    q.add_argument("--extra", default="{}", help="run options as JSON, e.g. '{\"kr\": 2, \"kn\": 1}'")
    q.add_argument("--tag", default="")
    q.add_argument("--nolog", action="store_true")
    s = sub.add_parser("window")
    s.add_argument("--bits", default="6,7,10")
    s.add_argument("--kr", type=int, default=3)
    s.add_argument("--kn", type=int, default=2)
    sub.add_parser("selftest")
    s = sub.add_parser("start")
    s.add_argument("--seeds", default="0,1")
    s.add_argument("--hold", type=int, default=0)
    s = sub.add_parser("probe")
    s.add_argument("path")
    s.add_argument("--bits", type=int, default=7)
    s.add_argument("--hard_borrow", action="store_true", help="the weights were trained with hard_borrow=True")
    a = p.parse_args()
    if a.cmd == "start":
        torch.set_num_threads(2)
        start(tuple(int(s_) for s_ in a.seeds.split(",")), a.hold)
    elif a.cmd == "probe":
        torch.set_num_threads(2)
        probe(a.path, a.bits, a.hard_borrow)
    elif a.cmd == "run":
        r = run(a.bits, a.seed, slots=a.slots, stage=a.stage, final=a.final, switch_steps=a.switch_steps,
                holdout=a.holdout, threads=a.threads, tag=a.tag, log=None if a.nolog else RESULTS, verbose=True,
                **json.loads(a.extra))
        print(show(r), flush=True)
    elif a.cmd == "window":
        for r in window(tuple(int(b) for b in a.bits.split(",")), a.kr, a.kn):
            print(json.dumps(r))
    else:
        selftest()


if __name__ == "__main__":
    main()
