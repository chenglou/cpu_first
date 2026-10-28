"""Clock scoring for ext_quotient's long-division datapath: does a mod-N-invariant loss break the quotient-digit
chicken-and-egg of FINDINGS_quotient.md?

In the long-division datapath a wrong quotient digit changes the running remainder by an exact multiple of N, as long
as the subtraction is exact and the register does not overflow, so the final value stays congruent to the answer mod
N. Scored on the answer's exact digits (cross-entropy), a bad guess and a bad subtraction look equally wrong. Scored
on the clock, 1 - cos(2π(r - y)/N) with r the final register's value, guesses cost nothing and only subtraction
mistakes show. Hypothesis: the shared digit cell then learns the subtraction first, whatever the guesses are, and the
choice can learn afterwards from the exact subtraction.

Model longdiv_clock (longdiv's digit cell and longdiv_est's estimator, created for every option, so every option has
the same init per seed):
- the register has W = L + w + 1 digits (L dividend digits, w digits of N), one extra digit per division step, so no
  sequence of slots 0..9 overflows it. Arithmetic is mod 10^W (the shift and the final borrow drop the top), and the
  value is read in ten's complement (top digit >= 5 is negative).
- one slot per step, in training and evaluation (a blend of slots is congruent to none of them), from `pick`:
  "given" takes the quotient digits from the input (task modq), with `noise` (a share of steps off by one) while
  training; "borrow" takes longdiv's choice, the largest slot whose trial does not borrow out of the register, taken
  hard; "est" takes the estimator's top slot. The cell trains through that path only.
- `score`: "ce" is the cross-entropy of the answer's w digits against the register's low w digits (as longdiv);
  "clock" is 1 - cos(2π(r - y)/N) in fp64 with r read from the register's argmax digits, with a straight-through
  gradient for each digit's phasor: digit j of value d turns the clock by 10^j d (mod N), and the gradient for its
  distribution is that of the expected phasor with the other digits held at their argmax. Bounded, and it does not
  vanish when many digits are unsure. "clock_mean" is the expected phasor over independent digit distributions
  instead (exact for one-hot digits too; its gradient shrinks with the product of the other digits' phasors).
- the clock gives the choice no gradient. With pick="est" the estimator is taught label-free (`est_teach`): at every
  step, cross-entropy toward the slot whose trial result (read from the cell's argmax digits) is the smallest
  non-negative value, or the largest value if none is non-negative. With an exact cell, that is the slot whose
  result lands in [0, N): the true digit.

Measures: ext_quotient.cell's (exact with the model's own hard choices, held-out moduli, wide, matched, choices),
plus `clock_test`: exact mod N (the register's value congruent to the answer, which is what the clock asks for), and
the cell on its own: the share of its truth table right (digit and borrow of r - q*n - c, slot q read as q), the best
multiplier for each slot with the share right under that map (the clock cannot tell which multiple of N a slot
subtracts), and exact / exact mod N with the true quotient digits given (oracle digits). Rows go to
microlab/results_quotient_clock.jsonl.

    python -m microlab.ext_quotient_clock selftest       # with the exact cell: invariance, the copy cell, the lessons
    python -m microlab.ext_quotient_clock toy            # the loss alone fits free per-example digits
    python -m microlab.ext_quotient_clock bench --size 10
    python -m microlab.ext_quotient_clock cell modq --size 6 --extra '{"pick": "given", "score": "clock", "noise": 0.3}'
    python -m microlab.ext_quotient_clock report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import ext_quotient_report as qr
from .ext_ladder import evaluate_split, keys, sample_kept
from .ext_matched import matched, wide, wider
from .ext_quotient import LongDivEst, choices, show_q, top
from .models import MODELS, model
from .run import _tensors, evaluate
from .tasks import TASKS

RESULTS = os.path.join(os.path.dirname(__file__), "results_quotient_clock.jsonl")
TAU = 2 * math.pi


def oh(a):
    return F.one_hot(a, 10).float()


def value(d):
    """LSB-first digits [..., k] (int) -> int64 values."""
    return (d.long() * 10 ** torch.arange(d.shape[-1], dtype=torch.long)).sum(-1)


def signed(d):
    """A register's digits [..., W] read in ten's complement: top digit >= 5 is negative."""
    W = d.shape[-1]
    return value(d) - (d[..., -1] >= 5).long() * 10 ** W


def turns(W, Nv):
    """The clock angle of digit value d at register place j, 2π (10^j s(d) mod N)/N in fp64: [B, W, 10]. s(d) = d,
    except the top place, read in ten's complement (d - 10 for d >= 5)."""
    d = torch.arange(10, dtype=torch.long)
    s = d.repeat(W, 1)
    s[-1] = d - 10 * (d >= 5).long()
    m = torch.remainder((10 ** torch.arange(W, dtype=torch.long))[:, None] * s, Nv[:, None, None])
    return TAU * m.double() / Nv.double()[:, None, None]


def clock(R, N, y, mode="st"):
    """The clock loss per example, fp64. R [B, W, 10]: the register's final digit distributions; N [B, w], y [B, w]
    digits. Forward: 1 - cos(2π(r - y)/N) with r read from R's argmax digits ("st") or the expected phasor over
    independent digits ("mean"); both are exact on one-hot digits."""
    Nv = value(N)
    ang = turns(R.shape[1], Nv)
    P = R.double()
    re, im = (P * ang.cos()).sum(-1), (P * ang.sin()).sum(-1)                  # each place's expected phasor
    if mode == "st":                                                           # straight-through per place
        hard = ang.gather(-1, R.argmax(-1, keepdim=True)).squeeze(-1)
        re, im = hard.cos() + re - re.detach(), hard.sin() + im - im.detach()
    zr, zi = torch.ones_like(re[:, 0]), torch.zeros_like(re[:, 0])
    for j in range(re.shape[1]):
        zr, zi = zr * re[:, j] - zi * im[:, j], zr * im[:, j] + zi * re[:, j]
    a = TAU * torch.remainder(value(y), Nv).double() / Nv.double()
    return 1 - (zr * a.cos() + zi * a.sin())


@model("longdiv_clock")
class LongDivClock(LongDivEst):
    choose = "est"                                                             # LongDivEst then builds the estimator

    def __init__(self, in_widths, out_width, aux_widths, pick="est", score="clock", noise=0.0, est_teach="range",
                 pre_score=None, pre_noise=0.0, pre_frac=0.0, **kw):
        super().__init__(in_widths, out_width, aux_widths, **kw)
        self.pick, self.score, self.noise, self.est_teach = pick, score, noise, est_teach
        self.pre = (pre_score, pre_noise, pre_frac)                          # a first phase (train() switches)

    def cell_at(self, r, n, c, slot):
        """The digit cell for one slot per row (LongDiv.cell computes all ten at once): r, n, c [..., 10]; slot [...]."""
        o = self.mlp(torch.cat([self.er(r), self.en(n), self.eq[slot], self.ec(c)], -1))
        return o[..., :10].softmax(-1), o[..., 10:].softmax(-1)

    def trial(self, Rp, n, slot):
        """Rp - slot*N through the cell, LSB-first: Rp, n [..., W, 10]; slot [...]. The digits [..., W, 10] and the
        borrow out of the register [..., 10] (dropped from the value: arithmetic mod 10^W)."""
        c, D = oh(torch.zeros_like(slot)), []
        for j in range(Rp.shape[-2]):
            d, c = self.cell_at(Rp[..., j, :], n[..., j, :], c, slot)
            D.append(d)
        return torch.stack(D, -2), c

    def trials(self, Rp, n):
        lead = Rp.shape[:-2]
        ex = lambda a: a.unsqueeze(-3).expand(*lead, 10, *a.shape[-2:])
        return self.trial(ex(Rp), ex(n), torch.arange(10).expand(*lead, 10))

    def est_logits(self, Rp, n, w):
        """The estimator on the leading digits of 10R + v_t's low w+1 places (where a remainder below N lives) and N."""
        return self.est(torch.cat([top(Rp[:, :w + 1], self.kr).flatten(1), top(n[:, :w], self.kn).flatten(1)], 1))

    @staticmethod
    def prep(inputs):
        v, N = inputs["v"], inputs["N"]
        w, L = N.shape[1], v.shape[1]
        W = L + w + 1
        return oh(v), oh(F.pad(N, (0, W - w))), W, w

    def given(self, inputs, noise=0.0):
        """The quotient digits per division step [B, L] (0 above the w digits of q), a share `noise` off by one."""
        q = inputs["q"]
        q = F.pad(q, (0, inputs["v"].shape[1] - q.shape[1]))
        if noise:
            off = torch.rand(q.shape) < noise
            step = torch.where(q == 0, 1, torch.where(q == 9, -1, torch.randint(0, 2, q.shape) * 2 - 1))
            q = torch.where(off, q + step, q)
        return q

    @torch.no_grad()
    def choose_path(self, inputs, q=None):
        """The division with one slot per step, chosen by `pick` (or q [B, L] when given). Returns the final register
        [B, W, 10], the slots [B, L] and choice distributions [B, L, 10] (indexed by the dividend digit t; the division
        runs t = L-1 down to 0), and the label-free estimator targets [B, L] (pick="est")."""
        v, n, W, w = self.prep(inputs)
        B, L = v.shape[:2]
        bi = torch.arange(B)
        R = oh(torch.zeros(B, W, dtype=torch.long))
        slots, lesson = torch.zeros(B, L, dtype=torch.long), torch.zeros(B, L, dtype=torch.long)
        dist = torch.zeros(B, L, 10)
        for t in reversed(range(L)):
            Rp = torch.cat([v[:, t:t + 1], R[:, :W - 1]], 1)                  # 10R + v_t, mod 10^W
            if q is not None:
                slots[:, t] = q[:, t]
                dist[:, t] = oh(q[:, t])
                R = self.trial(Rp, n, q[:, t])[0]
                continue
            D, c = self.trials(Rp, n)                                          # [B, 10, W, 10], [B, 10, 10]
            if self.pick == "borrow":                                          # the largest slot that does not borrow
                valid = c[..., 0]
                higher = torch.cat([torch.flip(torch.cumprod(torch.flip(1 - valid, [1]), 1), [1])[:, 1:],
                                    valid.new_ones(B, 1)], 1)
                wq = valid * higher
                dist[:, t] = wq / wq.sum(1, keepdim=True).clamp_min(1e-9)
            else:
                dist[:, t] = self.est_logits(Rp, n, w).softmax(-1)
                val = signed(D.argmax(-1))                                     # every trial's value, [B, 10]
                big = torch.iinfo(torch.long).max
                low = torch.where(val >= 0, val, big).argmin(1)
                lesson[:, t] = torch.where((val >= 0).any(1), low, val.argmax(1))
            slots[:, t] = dist[:, t].argmax(-1)
            R = D[bi, slots[:, t]]
        return R, slots, dist, lesson

    def follow(self, inputs, slots):
        """The register along fixed slots, with gradient; also each step's 10R + v_t (by t)."""
        v, n, W, w = self.prep(inputs)
        B, L = v.shape[:2]
        R, Rps = oh(torch.zeros(B, W, dtype=torch.long)), [None] * L
        for t in reversed(range(L)):
            Rps[t] = torch.cat([v[:, t:t + 1], R[:, :W - 1]], 1)
            R = self.trial(Rps[t], n, slots[:, t])[0]
        return R, Rps

    def forward(self, inputs):
        q = self.given(inputs) if self.pick == "given" else None
        R, slots, dist, _ = self.choose_path(inputs, q)
        w = inputs["N"].shape[1]
        return {"target": R[:, :w].clamp_min(1e-9).log(), "choice": oh(slots), "est": dist, "register": R}

    def train_loss(self, inputs, y):
        q = self.given(inputs, self.noise) if self.pick == "given" else None
        _, slots, dist, lesson = self.choose_path(inputs, q)
        R, Rps = self.follow(inputs, slots)
        w = y.shape[1]
        r = signed(R.argmax(-1))
        Nv, yv = value(inputs["N"]), value(y)
        stats = {"exact": (R[:, :w].argmax(-1) == y).all(1).float().mean().item(),
                 "exact_mod_n": (torch.remainder(r - yv, Nv) == 0).float().mean().item()}
        if self.score == "ce":
            loss = F.cross_entropy(R[:, :w].clamp_min(1e-9).log().flatten(0, 1), y.flatten())
        else:
            loss = clock(R, inputs["N"], y, "mean" if self.score == "clock_mean" else "st").mean()
        stats["score_loss"] = loss.item()
        if self.pick == "est":
            n = self.prep(inputs)[1]
            logits = torch.stack([self.est_logits(Rps[t].detach(), n, w) for t in range(len(Rps))], 1)
            est = F.cross_entropy(logits.flatten(0, 1), lesson.flatten())
            stats |= {"est_loss": est.item(), "est_agree": (logits.argmax(-1) == lesson).float().mean().item()}
            loss = loss + est
        return loss, stats


def exact_cell_at(r, n, c, slot):
    """The digit cell's truth table for one slot per row, on one-hot inputs: one digit of r - slot*n - c and its borrow."""
    x = r.argmax(-1) - slot * n.argmax(-1) - c.argmax(-1)
    return oh(x % 10), oh((-x).clamp_min(0).add(9).div(10, rounding_mode="floor"))


def copy_cell_at(r, n, c, slot):
    """Subtract nothing, for every slot: the digit passes through (minus the borrow in)."""
    return exact_cell_at(r, n, c, torch.zeros_like(slot))


@torch.no_grad()
def cell_table(net, size, n=1000, seed=50_000):
    """The cell's trial subtraction on its own, where long division uses it: for slot k and digit m, Rp uniform in
    [m*N, (m+1)*N) (so m is the right quotient digit), the share whose trial result has the digits of the remainder
    Rp - m*N, on the w places the answer reads ("low") and on all W places. Digits only: the
    10-way borrow is a code the cell passes to itself. Returns the low share with slot k read as digit k, the best
    digit per slot (by the low share) and both shares under that map. A copy cell (subtract nothing) maps every slot
    to 0."""
    rng = np.random.default_rng(seed)
    w = len(str((1 << size) - 1))
    W = 3 * w + 1
    N = torch.from_numpy(((rng.integers(1 << (size - 2), 1 << (size - 1), n) * 2) + 1).astype(np.int64))
    u = torch.from_numpy(rng.random(n))
    dig = lambda v: torch.remainder(v[:, None] // 10 ** torch.arange(W), 10)
    No = oh(dig(N))
    low, full = torch.zeros(10, 10), torch.zeros(10, 10)
    for m in range(10):
        Rp = m * N + (u * N.double()).long()
        Ro, want = oh(dig(Rp)), dig(Rp - m * N)
        for k in range(10):
            ok = net.trial(Ro, No, torch.full((n,), k))[0].argmax(-1) == want
            low[k, m], full[k, m] = ok[:, :w].all(1).float().mean(), ok.all(1).float().mean()
    best = low.argmax(1)
    pick = lambda A: float(A.gather(1, best[:, None]).mean())
    return {"table_right": round(float(low.diag().mean()), 4), "table_right_full": round(float(full.diag().mean()), 4),
            "slot_multiplier": best.tolist(), "table_right_best_map": round(pick(low), 4),
            "table_right_best_map_full": round(pick(full), 4), "multipliers_used": int(len(set(best.tolist())))}


@torch.no_grad()
def clock_test(net, task, size, n=2000, seed=10_000):
    """Exact mod N with the model's own choices, and the cell with the true quotient digits given (oracle digits):
    exact and exact mod N. Plus the cell's truth table."""
    b = task.sample(n, size, np.random.default_rng(seed))
    x, y, _ = _tensors(b, "cpu")
    q = torch.from_numpy(b.aux["q"] if "q" in b.aux else b.inputs["q"]).long()
    Nv, yv = value(x["N"]), value(y)
    out = {}
    for name, qq in (("own", None), ("oracle", F.pad(q, (0, x["v"].shape[1] - q.shape[1])))):
        if name == "own" and net.pick == "given":
            qq = net.given(x)
        R = net.choose_path(x, qq)[0]
        out[f"exact_{name}"] = round(float((R[:, :y.shape[1]].argmax(-1) == y).all(1).float().mean()), 4)
        out[f"exact_mod_n_{name}"] = round(float((torch.remainder(signed(R.argmax(-1)) - yv, Nv) == 0)
                                                 .float().mean()), 4)
    return out | cell_table(net, size)


def train(task_name, size, model="longdiv_clock", seed=0, steps=3000, batch=256, lr=3e-3, checkpoints=4, marks_n=12,
          threads=None, tag="", holdout=0, save=None, **model_kw):
    """ext_ladder.cell's training (same init, AdamW, OneCycle, clipping, data stream, checkpoints and evaluation) with
    the model's own loss (net.train_loss). With pre_frac > 0 the first pre_frac of the steps use (pre_score,
    pre_noise) instead of the model's score and noise. `save`: a path for the trained state_dict.
    Returns (row, net, the training inputs' keys)."""
    if threads:
        torch.set_num_threads(threads)
    task = TASKS[task_name]
    in_w, out_w, _ = task.widths(size)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = MODELS[model](in_w, out_w, {}, **model_kw)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    marks = {round(steps * (i + 1) / checkpoints) for i in range(checkpoints - 1)}
    at = {round(steps * (i + 1) / marks_n) for i in range(marks_n)}
    curve, train_curve, losses, seen, recent = [], [], [], [], []
    main = (net.score, net.noise)
    pre_steps = round(steps * net.pre[2]) if net.pre[0] else 0
    t0 = time.time()
    for step in range(steps):
        net.score, net.noise = net.pre[:2] if step < pre_steps else main
        b = sample_kept(task, batch, size, rng, holdout)
        seen.append(keys(b.inputs))
        x, y, _ = _tensors(b, "cpu")
        loss, st = net.train_loss(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        recent = (recent + [st])[-100:]
        if step + 1 in marks:
            curve.append([step + 1, round(evaluate(net, task, size, n=500, seed=20_000 + step)[0], 4)])
        if step + 1 in at:
            row = {"step": step + 1, **{k: round(float(np.mean([s[k] for s in recent])), 4) for k in st}}
            train_curve.append(row | {"table_right": cell_table(net, size, n=500)["table_right"]})
        if step + 1 == pre_steps:
            train_curve.append({"step": step + 1, "switch": True, **clock_test(net, TASKS[model_kw.get("task", task_name)], size)})
    train_keys = np.unique(np.concatenate(seen))
    if save:
        torch.save(net.state_dict(), save)
    exact, per_digit, split = evaluate_split(net, task, size, train_keys, holdout=holdout)
    curve.append([steps, round(exact, 4)])
    ood_size = ood_exact = None
    if task.kind == "bits" and size + 1 <= 24 and task.widths(size + 1) == task.widths(size):
        ood_size, ood_exact = size + 1, round(evaluate(net, task, size + 1)[0], 4)
    r = {"task": task_name, "size": size, "model": model, "data": "fresh", "pool": 0, "seed": seed, "steps": steps,
         "batch": batch, "lr": lr, "model_kw": model_kw, "holdout": holdout,
         "params": sum(p.numel() for p in net.parameters()), "exact": round(exact, 4),
         "per_digit": [round(v, 3) for v in per_digit], **split, "train_exact": None, "ood_size": ood_size,
         "ood_exact": ood_exact, "curve": curve, "train_curve": train_curve,
         "final_loss": round(float(np.mean(losses[-100:])), 4), "seconds": round(time.time() - t0, 1),
         "device": "cpu", "threads": torch.get_num_threads(), "tag": tag}
    return r, net, train_keys


def cell(task_name, size, model="longdiv_clock", log=RESULTS, train_on=None, **kw):
    """ext_quotient.cell for these models: `train_on` trains on another task with the same inputs (e.g. modmix) and
    scores `task_name`; then wide, matched, choices and clock_test."""
    kw.setdefault("task", task_name)
    r, net, train_keys = train(train_on or task_name, size, model=model, **kw)
    task = TASKS[task_name]
    if train_on:
        exact, per_digit, split = evaluate_split(net, task, size, train_keys, holdout=kw.get("holdout", 0))
        r.update(task=task_name, train_task=train_on, exact_train_dist=r["exact"], exact=round(exact, 4),
                 per_digit=[round(v, 3) for v in per_digit], **split)
        if r["ood_size"] is not None:
            r["ood_exact"] = round(evaluate(net, task, r["ood_size"])[0], 4)
    r["wide"] = wide(net, task, size, wider(task_name, size))
    r["matched"] = matched(net, task, size, train_keys)
    r["choice"] = choices(net, task_name, size, holdout=kw.get("holdout", 0))
    r["clock_test"] = clock_test(net, task, size)
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(r) + "\n")
    return r


def show_clock(r):
    c = r["clock_test"]
    return (show_q(r) + f"  mod-N {c['exact_mod_n_own']:.2f}  oracle {c['exact_oracle']:.2f}/{c['exact_mod_n_oracle']:.2f}"
            f"  table {c['table_right']:.2f} (best map {c['table_right_best_map']:.2f} {c['slot_multiplier']})"
            f"  ho{r.get('holdout', 0)} {r['steps']}")


def selftest(bits=(6, 7, 10), n=300, seed=0):
    """With the exact cell: the given digits are exact; with noisy or random slots the answer is wrong but still exact
    mod N (the clock is 0); the copy cell (subtract nothing) is also exact mod N with clock 0; the borrow rule is
    exact; the estimator's label-free targets are the true digits. With the learned cell: gradients for every score."""
    for b in bits:
        task = TASKS["modq"]
        bt = task.sample(n, b, np.random.default_rng(seed))
        x, y, _ = _tensors(bt, "cpu")
        in_w, out_w, _ = task.widths(b)
        net = MODELS["longdiv_clock"](in_w, out_w, {}, pick="given")
        Nv, yv = value(x["N"]), value(y)
        L = x["v"].shape[1]
        truth = F.pad(x["q"], (0, L - x["q"].shape[1]))

        def show(name, R):
            ok = (R[:, :y.shape[1]].argmax(-1) == y).all(1).float().mean()
            mod = (torch.remainder(signed(R.argmax(-1)) - yv, Nv) == 0).float().mean()
            print(f"{b} bits {name:34} exact {ok:.3f}  exact mod N {mod:.3f}  clock {clock(R, x['N'], y).mean():.2e}"
                  f" (mean {clock(R, x['N'], y, 'mean').mean():.2e})")

        net.cell_at = exact_cell_at
        show("exact cell, true digits", net.choose_path(x, truth)[0])
        show("exact cell, 30% of digits off by one", net.choose_path(x, net.given(x, 0.3))[0])
        show("exact cell, random slots", net.choose_path(x, torch.randint(0, 10, truth.shape))[0])
        net.pick = "borrow"
        R, slots, _, _ = net.choose_path(x)
        show("exact cell, borrow rule", R)
        print(f"{b} bits borrow rule slots == true digits {(slots == truth).all(1).float().mean():.3f}")
        v, n_, W, w = net.prep(x)
        S, ok = oh(torch.zeros(len(y), W, dtype=torch.long)), []
        for t in reversed(range(L)):                                           # the label-free targets on the true path
            Rp = torch.cat([v[:, t:t + 1], S[:, :W - 1]], 1)
            D, _ = net.trials(Rp, n_)
            val = signed(D.argmax(-1))
            low = torch.where(val >= 0, val, torch.iinfo(torch.long).max).argmin(1)
            ok.append((torch.where((val >= 0).any(1), low, val.argmax(1)) == truth[:, t]).float().mean().item())
            S = D[torch.arange(len(y)), truth[:, t]]
        print(f"{b} bits label-free targets on the true path == true digits, per step (first step first): "
              + " ".join(f"{a:.3f}" for a in ok))
        net.pick = "given"
        net.cell_at = copy_cell_at
        show("copy cell (subtract nothing)", net.choose_path(x, torch.randint(0, 10, truth.shape))[0])
        for pick in ("given", "borrow", "est"):
            for score in ("ce", "clock", "clock_mean"):
                torch.manual_seed(1)
                m = MODELS["longdiv_clock"](in_w, out_w, {}, pick=pick, score=score, noise=0.3)
                loss, st = m.train_loss(x, y)
                loss.backward()
                g = sum(float(p.grad.abs().sum()) for p in m.cell_params() if p.grad is not None)
                ge = sum(float(p.grad.abs().sum()) for p in m.est.parameters() if p.grad is not None)
                print(f"{b} bits learned cell {pick:6} {score:10} loss {loss.item():.4f}  |grad| cell {g:.3g} "
                      f"estimator {ge:.3g}")


def _step_grads(net, x, y, kind, multiplier=None):
    """The gradient of a loss for the cell's parameters, split by division step (summed over the examples): [L, P].
    kind "clock" / "ce": the ST clock or the answer's cross-entropy (w digits), along the given quotient digits from the
    model's own registers; each step's share is its own trial's vector-Jacobian product. kind "teach": teacher-forced,
    every step's trial from the exact previous register toward the exact next one (digits of all W places), with the
    true digits (multiplier None) or every slot subtracting multiplier*N (0: the copy cell)."""
    v, n, W, w = net.prep(x)
    B, L = v.shape[:2]
    q = net.given(x)
    params = net.cell_params()
    flat = lambda gs: torch.cat([g.reshape(-1) for g in gs])
    out = torch.zeros(L, sum(p.numel() for p in params))
    if kind == "teach":
        Nv, Rv = value(x["N"]), torch.zeros(B, dtype=torch.long)
        dig = lambda val: oh(torch.remainder(torch.remainder(val, 10 ** W)[:, None] // 10 ** torch.arange(W), 10))
        for t in reversed(range(L)):
            Rp = 10 * Rv + x["v"][:, t]
            m = q[:, t] if multiplier is None else torch.full_like(q[:, t], multiplier)
            Rv = Rp - m * Nv
            D = net.trial(dig(Rp), n, q[:, t])[0]
            loss = -(D.clamp_min(1e-9).log() * dig(Rv)).sum()
            out[t] = flat(torch.autograd.grad(loss, params))
        return out
    R, Rps, Ds = oh(torch.zeros(B, W, dtype=torch.long)), {}, {}
    for t in reversed(range(L)):
        Rps[t] = torch.cat([v[:, t:t + 1], R[:, :W - 1]], 1)
        R = Ds[t] = net.trial(Rps[t], n, q[:, t])[0]
    loss = (clock(R, x["N"], y).sum() if kind == "clock"
            else F.cross_entropy(R[:, :w].clamp_min(1e-9).log().flatten(0, 1), y.flatten(), reduction="sum"))
    gD = torch.autograd.grad(loss, [Ds[t] for t in range(L)])
    for t in range(L):
        D = net.trial(Rps[t].detach(), n, q[:, t])[0]
        out[t] = flat(torch.autograd.grad(D, params, grad_outputs=gD[t]))
    return out


def gradcheck(size=6, batches=(256, 1024, 4096, 16384, 65536), ref_batch=65536, chunk=4096, seed=0, pre_steps=0,
              threads=4):
    """Run 3: does the clock's gradient for the cell point along the subtraction's own lesson? With the true digits
    given (modq), on the cell at init or after `pre_steps` of cross-entropy training (given digits, batch 256, lr
    3e-3), compare per division step and in total the gradient of the ST clock and of the answer's cross-entropy (B
    examples) with two teacher-forced references on ref_batch other examples: toward the exact subtraction ("true")
    and toward the copy cell ("copy", every slot subtracting nothing, a zero-loss solution of the clock). Returns
    cosines by B and step (first division step first), plus the losses and the cell's measures."""
    torch.set_num_threads(threads)
    task = TASKS["modq"]
    in_w, out_w, _ = task.widths(size)
    torch.manual_seed(seed)
    net = MODELS["longdiv_clock"](in_w, out_w, {}, pick="given", score="ce")
    if pre_steps:
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=pre_steps, pct_start=0.1)
        rng = np.random.default_rng(seed)
        for _ in range(pre_steps):
            x, y, _ = _tensors(task.sample(256, size, rng), "cpu")
            loss = net.train_loss(x, y)[0]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()

    def grads(kind, B, sd, multiplier=None):
        rng, g = np.random.default_rng(sd), 0
        for i in range(0, B, chunk):
            x, y, _ = _tensors(task.sample(min(chunk, B - i), size, rng), "cpu")
            g = g + _step_grads(net, x, y, kind, multiplier)
        return g

    def cos(a, b):                                    # 0 for a step whose gradient is (numerically) absent
        a, b = a.double(), b.double()
        return 0.0 if min(a.norm(), b.norm()) < 1e-20 else float((a * b).sum() / (a.norm() * b.norm()))
    ref = {"true": grads("teach", ref_batch, 70_000), "copy": grads("teach", ref_batch, 70_000, multiplier=0)}
    x, y, _ = _tensors(task.sample(4096, size, np.random.default_rng(80_000)), "cpu")
    with torch.no_grad():
        R = net.choose_path(x, net.given(x))[0]
        losses = {"clock": round(float(clock(R, x["N"], y).mean()), 4),
                  "ce": round(float(F.cross_entropy(R[:, :y.shape[1]].clamp_min(1e-9).log().flatten(0, 1),
                                                    y.flatten())), 4)}
    out = {"size": size, "seed": seed, "pre_steps": pre_steps, "ref_batch": ref_batch, "losses": losses,
           "cell": clock_test(net, task, size), "by_batch": []}
    for i, B in enumerate(batches):
        row = {"batch": B}
        for kind in ("clock", "ce"):
            g = grads(kind, B, 90_000 + i)
            for name, r in ref.items():
                row[f"{kind}_vs_{name}"] = round(cos(g.sum(0), r.sum(0)), 4)
                row[f"{kind}_vs_{name}_by_step"] = [round(cos(g[t], r[t]), 4) for t in reversed(range(len(g)))]
        out["by_batch"].append(row)
    return out


def recheck(paths, out=os.path.join(os.path.dirname(__file__), "results_quotient_clock_cells.jsonl")):
    """Recompute clock_test (with this cell_table) for saved cells: each path is a cell's .pt next to its row .json
    (as ext_quotient_clock_modal.py writes them). Appends {tag, file, size, model_kw, seed, steps, clock_test} rows."""
    for p in paths:
        r = json.loads(open(p.removesuffix(".pt") + ".json").read())
        task = TASKS[r["task"]]
        in_w, out_w, _ = task.widths(r["size"])
        net = MODELS[r["model"]](in_w, out_w, {}, **r["model_kw"])
        net.load_state_dict(torch.load(p))
        row = {"tag": r["tag"], "file": os.path.basename(p), "size": r["size"], "model_kw": r["model_kw"],
               "seed": r["seed"], "steps": r["steps"], "batch": r["batch"], "clock_test": clock_test(net, task, r["size"])}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        c = row["clock_test"]
        print(f"{row['file']:55} oracle {c['exact_oracle']:.2f}/{c['exact_mod_n_oracle']:.2f}  table {c['table_right']:.2f}"
              f" (full {c['table_right_full']:.2f})  best map {c['table_right_best_map']:.2f} {c['slot_multiplier']}", flush=True)


def toy(bits=((6, 7), (7, 10)), n=512, steps=300, seed=0):
    """Is the loss sound? Free digit logits for each example's W-digit register (no cell, no sharing), trained by the
    clock alone with Adam: the share whose register value becomes congruent to the answer mod N."""
    torch.manual_seed(seed)
    for b, W in bits:
        bt = TASKS["mod"].sample(n, b, np.random.default_rng(seed))
        x, y, _ = _tensors(bt, "cpu")
        for mode in ("st", "mean"):
            for scale in (0.1, 3.0):
                Z = (scale * torch.randn(n, W, 10)).requires_grad_()
                opt = torch.optim.Adam([Z], lr=0.1)
                for _ in range(steps):
                    loss = clock(Z.softmax(-1), x["N"], y, mode).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                ok = (torch.remainder(signed(Z.argmax(-1)) - value(y), value(x["N"])) == 0).float().mean()
                print(f"{b} bits, W = {W}, {mode:4}, logits ~ {scale} N(0,1): after {steps} steps, r = y mod N on {ok:.3f}")


def bench(size=7, steps=3, batch=256, threads=4, seed=0):
    """Seconds per training step for each pick (clock score), against longdiv_est's soft step."""
    torch.set_num_threads(threads)
    task = TASKS["modq"]
    in_w, out_w, _ = task.widths(size)
    x, y, _ = _tensors(task.sample(batch, size, np.random.default_rng(seed)), "cpu")
    est = MODELS["longdiv_est"](in_w, out_w, {})
    xs = {k: v for k, v in x.items() if k != "q"}
    runs = [(p, lambda p=p: MODELS["longdiv_clock"](in_w, out_w, {}, pick=p).train_loss(x, y)[0])
            for p in ("given", "borrow", "est")]
    runs.append(("longdiv_est soft", lambda: F.cross_entropy(est(xs)["target"].flatten(0, 1), y.flatten())))
    for name, f in runs:
        f().backward()
        t0 = time.time()
        for _ in range(steps):
            f().backward()
        print(f"{size} bits {name}: {(time.time() - t0) / steps:.3f} s/step ({threads} threads)")


def name(r):
    kw = {k: v for k, v in r["model_kw"].items() if k != "task"}
    pick = {"given": "digits given (modq)", "borrow": "borrow choice, hard", "est": "estimator, hard"}[kw["pick"]]
    out = f"{pick}, {'clock' if kw.get('score', 'clock') == 'clock' else kw['score']}"
    if kw.get("noise"):
        out += f", {kw['noise']:.0%} of digits off by one"
    if kw.get("pre_frac"):
        out += f", after {kw['pre_score']} for the first {kw['pre_frac']:.0%} of steps"
    if r.get("batch", 256) != 256:
        out += f", batch {r['batch']}"
    if r.get("train_task"):
        out += f", trained on {r['train_task']}"
    return out


CELLS = os.path.join(os.path.dirname(__file__), "results_quotient_clock_cells.jsonl")


def _rechecked(rows):
    """Rows with clock_test replaced by its recheck (results_quotient_clock_cells.jsonl) where there is one; other rows
    lose the cell-table entries, which an earlier version of cell_table measured differently."""
    key = lambda r: (r["tag"], r["size"], r["seed"], r["steps"], json.dumps(r["model_kw"], sort_keys=True))
    fresh = {key(c): c["clock_test"] for c in qr.load(CELLS)}
    for r in rows:
        if key(r) in fresh:
            r["clock_test"] = fresh[key(r)]
        else:
            r["clock_test"] = {k: v for k, v in r["clock_test"].items() if not k.startswith(("table", "slot", "mult"))}
    return rows


def report(path=RESULTS):
    """The tables of FINDINGS_quotient_clock.md."""
    groups = {}
    for r in _rechecked(qr.load(path)):
        groups.setdefault((name(r), r["size"], r["steps"]), {}).setdefault(r.get("holdout", 0), []).append(r)
    lines = ["| variant | bits | steps | exact | exact mod N | held-out moduli (trained) | one bit wider | "
             "one digit wider | every choice right | cell table right (best slot map) | oracle digits: exact (mod N) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(groups):
        main, ho = groups[k].get(0, []), groups[k].get(4, [])
        c = [r["clock_test"] for r in main]
        hos = qr.span([r.get("exact_heldout_moduli") for r in ho]) + (f" ({qr.span([qr.kept(r) for r in ho])})"
                                                                      if ho else "")
        new = [x for x in c if "table_right_full" in x]           # rows before the trial-level table: no cell column
        st = f"{k[2] // 1000}k" if k[2] >= 1000 else str(k[2])
        lines.append(
            f"| {k[0]} | {k[1]} | {st} | {qr.span([r['exact'] for r in main])} | "
            f"{qr.span([x['exact_mod_n_own'] for x in c])} | {hos} | {qr.span([qr.wider1(r) for r in main])} | "
            f"{qr.span([qr.digit1(r) for r in main])} | "
            f"{qr.span([r['choice']['choice_mapped']['all'] for r in main])} | "
            f"{qr.span([x['table_right'] for x in new])} ({qr.span([x['table_right_best_map'] for x in new])}) | "
            f"{qr.span([x['exact_oracle'] for x in c])} ({qr.span([x['exact_mod_n_oracle'] for x in c])}) |")
    return "\n".join(lines)


GRAD = os.path.join(os.path.dirname(__file__), "results_quotient_clock_grad.jsonl")


def grad_report(path=GRAD):
    """Run 3's gradient checks: cosine of the clock's and the answer cross-entropy's gradient for the cell with the
    teacher-forced lesson toward the exact subtraction (and toward the copy cell), by batch, over seeds; per step for
    the largest batch (first division step first)."""
    groups = {}
    for r in qr.load(path):
        groups.setdefault((r["size"], r["pre_steps"]), []).append(r)
    lines = ["| bits | cell | batch | clock vs exact lesson | clock vs copy lesson | answer CE vs exact lesson |",
             "|---|---|---|---|---|---|"]
    steps = []
    for (size, pre), rs in sorted(groups.items()):
        state = "at init" if not pre else f"after {pre} CE steps (oracle exact {qr.span([r['cell']['exact_oracle'] for r in rs])})"
        for i, b in enumerate(rs[0]["by_batch"]):
            col = lambda k: qr.span([r["by_batch"][i][k] for r in rs], "{:+.2f}")
            lines.append(f"| {size} | {state} | {b['batch']:,} | {col('clock_vs_true')} | {col('clock_vs_copy')} | "
                         f"{col('ce_vs_true')} |")
        last = lambda k: " / ".join("[" + " ".join(f"{a:+.2f}" if abs(a) <= 1 else " 0  " for a in r["by_batch"][-1][k])
                                    + "]" for r in rs)                # |cos| > 1: an old row's vanishing gradient
        steps.append(f"{size} bits, {state}, batch {rs[0]['by_batch'][-1]['batch']:,}: clock {last('clock_vs_true_by_step')}"
                     f"; answer CE {last('ce_vs_true_by_step')}")
    return "\n".join(lines + [""] + steps)


def runs_report(path=RESULTS):
    """Per run: the cell's slot map, and the training curve (exact, exact mod N, table right)."""
    out = []
    for r in sorted(qr.load(path), key=lambda r: (name(r), r["size"], r["steps"], r.get("holdout", 0), r["seed"])):
        c = r["clock_test"]
        curve = " ".join(f"{p['step']}:{p['exact']:.2f}/{p['exact_mod_n']:.2f}/{p['table_right']:.2f}"
                         for p in [p for p in r["train_curve"] if not p.get("switch")][::3])
        out.append(f"{name(r):55} {r['size']} {r['steps'] // 1000:>2}k ho{r.get('holdout', 0)} s{r['seed']}  "
                   f"exact {r['exact']:.2f} mod-N {c['exact_mod_n_own']:.2f}  table {c['table_right']:.2f} "
                   f"best {c['table_right_best_map']:.2f} map {c['slot_multiplier']}  "
                   f"oracle {c['exact_oracle']:.2f}/{c['exact_mod_n_oracle']:.2f}  train exact/mod-N/table {curve}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("cell")
    q.add_argument("task", choices=sorted(TASKS))
    q.add_argument("--size", type=int, required=True)
    q.add_argument("--steps", type=int, default=3000)
    q.add_argument("--lr", type=float, default=3e-3)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--threads", type=int, default=2)
    q.add_argument("--holdout", type=int, default=0)
    q.add_argument("--train_on", default=None)
    q.add_argument("--extra", default="{}", help="model options as JSON, e.g. '{\"pick\": \"borrow\"}'")
    q.add_argument("--tag", default="")
    q.add_argument("--nolog", action="store_true")
    s = sub.add_parser("bench")
    s.add_argument("--size", type=int, default=7)
    s.add_argument("--steps", type=int, default=3)
    s.add_argument("--threads", type=int, default=4)
    sub.add_parser("selftest")
    sub.add_parser("toy")
    s = sub.add_parser("recheck")
    s.add_argument("paths", nargs="+", help="saved cells (.pt next to their row .json)")
    s = sub.add_parser("report")
    s.add_argument("--runs", action="store_true", help="per run: the slot map and the training curve")
    s.add_argument("--grad", action="store_true", help="run 3's gradient checks")
    a = p.parse_args()
    if a.cmd == "cell":
        r = cell(a.task, a.size, steps=a.steps, lr=a.lr, seed=a.seed, threads=a.threads, holdout=a.holdout,
                 train_on=a.train_on, tag=a.tag, log=None if a.nolog else RESULTS, **json.loads(a.extra))
        print(show_clock(r), flush=True)
    elif a.cmd == "recheck":
        recheck(a.paths)
    elif a.cmd == "toy":
        toy()
    elif a.cmd == "bench":
        bench(a.size, a.steps, threads=a.threads)
    elif a.cmd == "report":
        print(grad_report() if a.grad else runs_report() if a.runs else report())
    else:
        selftest()


if __name__ == "__main__":
    main()
