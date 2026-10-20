"""A learned leading-digit quotient estimator inside ext_matched's long-division datapath: does a better-matched
quotient-digit choice make the remainder learnable from answers?

ext_matched.LongDiv chooses each quotient digit from its own digit cell: it keeps the largest q whose trial
subtraction 10R + v_t - q*N does not borrow, softly, from the cells' final-borrow probabilities. Trained from answers
that is slow (LADDER_MATCHED.md §3): the choice is right only once the cell is, and the cell gets consistent lessons
only once the choice is. Here a separate small estimator picks q from the leading digits instead, the way SRT and
Knuth's Algorithm D estimate a quotient digit from the top digits of the partial remainder and of the divisor.

Models (registered into microlab's MODELS and ext_matched.FOR, tasks mod / modq). Same digit cell, scan and ten
trial subtractions as longdiv; only the choice differs:
- longdiv_tr    longdiv's own choice (the largest non-borrowing q), with the choice traced. Same parameters and
                computation as longdiv, so it reproduces it; the trace gives the choice measures below.
- longdiv_est   q from the estimator alone: an MLP on the top `kr` digits of 10R + v_t and the top `kn` digits of N
                (as digit distributions; defaults 3 and 2) gives P(q). The cell only subtracts; its borrow is unused.
- longdiv_estc  the estimate checked by the cell's borrow: from estimate e, take e - 1 if trial e borrows, e + 1 if
                trial e + 1 does not, else e. Softly, with v(q) = P(trial q does not borrow):
                P(e-1 | e) = 1 - v(e), P(e | e) = v(e) (1 - v(e+1)), P(e+1 | e) = v(e) v(e+1).
                Only the two borrows next to the estimate matter, where longdiv's rule needs all ten right.
At 6 bits (w = 2) the default window is every digit of both numbers. At 7 bits (w = 3) it drops each one's units
digit, so the estimate alone is sometimes off by one (see `window`) and only longdiv_estc can be exact, in principle.

Model options: `kr`/`kn` set the window (kr=4, kn=3 sees everything at 7 bits); `est_bias0` starts the estimator at
q = 0. Diagnostics, not from answers alone: `soft_cell` replaces the digit cell by its truth table applied to digit
distributions (multilinear: exact on one-hot inputs, smooth on the mixtures training produces), so only the choice is
learned, the mirror image of modq; `fit_cell`/`freeze_cell` instead fit the MLP cell to its truth table first, which
is not a faithful stand-in (soft mixtures are inputs it never saw). Cell option `train_on` trains on another task
with the same inputs (e.g. modmix, easy dividends mixed in) and scores the given one.

Measures: ext_matched.cell's (exact, seen/unseen at equal lengths, wider sizes, held-out moduli with holdout=k), plus
the choice scored against the true quotient digits, which are never trained on: per step and all steps right, with
the slots read as digits directly ("raw") and through their best slot -> digit map ("mapped", since an estimator
alone may label its slots in any order); and exact given every choice right (what the subtraction still gets wrong).
Rows go to microlab/results_quotient.jsonl.

    python -m microlab.ext_quotient window --bits 6,7,8,10          # how much a window of leading digits fixes q
    python -m microlab.ext_quotient selftest                         # with exact parts, every variant is exact
    python -m microlab.ext_quotient blurtest                         # a blurred right estimate, with the exact cell
    python -m microlab.ext_quotient cell mod --size 6 --model longdiv_estc --steps 3000 --lr 3e-3
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import ext_ladder, ext_matched
from .ext_matched import LongDiv, matched, show, wide, wider
from .models import MODELS, model
from .run import _tensors, evaluate
from .tasks import TASKS

RESULTS = os.path.join(os.path.dirname(__file__), "results_quotient.jsonl")
VARIANTS = ("longdiv_tr", "longdiv_est", "longdiv_estc")


def top(x, k):
    """The top k places of LSB-first digit distributions x [B, P, 10]; zeros below the units place when P < k."""
    P = x.shape[1]
    if P >= k:
        return x[:, P - k:]
    pad = F.one_hot(torch.zeros(x.shape[0], k - P, dtype=torch.long, device=x.device), 10).float()
    return torch.cat([pad, x], 1)


def truth_table():
    """One-hot digit and borrow of r - q*n - c for every (q, r, n, c): two [10, 10, 10, 10, 10] tensors (q r n c out)."""
    q, r, n, c = torch.meshgrid(*[torch.arange(10)] * 4, indexing="ij")
    x = r - q * n - c
    borrow = (-x).clamp_min(0).add(9).div(10, rounding_mode="floor")
    return torch.stack([F.one_hot(x % 10, 10).float(), F.one_hot(borrow, 10).float()])


def checked(pe, v):
    """The estimate e corrected by the borrows next to it: e-1 if trial e borrows, e+1 if trial e+1 does not, else
    e. pe: [B, 10] P(e); v: [B, 10] P(trial q does not borrow). Returns P(q), which sums to 1."""
    v1 = F.pad(v[:, 1:], (0, 1))                                               # v(e+1), v(10) = 0
    down, stay, up = pe * (1 - v), pe * v * (1 - v1), pe * v * v1
    return stay + F.pad(down[:, 1:], (0, 1)) + F.pad(down[:, :1], (0, 9)) + F.pad(up[:, :-1], (1, 0))


class LongDivEst(LongDiv):
    choose = "pm1"                                                             # "borrow" | "est" | "pm1"

    def __init__(self, in_widths, out_width, aux_widths, task="mod", kr=3, kn=2, est_hidden=128, est_bias0=0.0,
                 fit_cell=0, fit_lr=3e-3, freeze_cell=False, soft_cell=False, **kw):
        super().__init__(in_widths, out_width, aux_widths, task=task, **kw)
        self.kr, self.kn = kr, kn
        if self.choose != "borrow":
            self.est = nn.Sequential(nn.Linear(10 * (kr + kn), est_hidden), nn.GELU(),
                                     nn.Linear(est_hidden, est_hidden), nn.GELU(), nn.Linear(est_hidden, 10))
            with torch.no_grad():                                             # est_bias0 > 0: start at q = 0
                self.est[-1].bias[0] += est_bias0
        if fit_cell:                                                          # diagnostic, not from answers
            self.cell_fit = self.fit(fit_cell, fit_lr)
        if freeze_cell or soft_cell:
            for p in self.cell_params():
                p.requires_grad_(False)
        if soft_cell:                                                         # diagnostic: the exact cell, no learning
            self.register_buffer("table", truth_table(), persistent=False)
            self.cell = self.soft_cell

    def soft_cell(self, r, n, c):
        """The cell's truth table applied to digit distributions: multilinear, so exact on one-hot inputs and smooth
        (with useful gradients) on the soft mixtures training produces. r, n, c: [B, 10 (q), 10]."""
        D, C = self.table
        t = torch.einsum("bqn,qrnco->bqrco", n, D), torch.einsum("bqn,qrnco->bqrco", n, C)
        t = torch.einsum("bqr,bqrco->bqco", r, t[0]), torch.einsum("bqr,bqrco->bqco", r, t[1])
        return torch.einsum("bqc,bqco->bqo", c, t[0]), torch.einsum("bqc,bqco->bqo", c, t[1])

    def cell_params(self):
        return [*self.er.parameters(), *self.en.parameters(), *self.ec.parameters(), self.eq, *self.mlp.parameters()]

    def fit(self, steps, lr=3e-3):
        """Fit the digit cell to its truth table (all 10^4 one-hot entries (r, n, q, c), full batch). Returns the
        share of entries whose digit and borrow are both right."""
        q, r, n, c = (a.reshape(-1) for a in torch.meshgrid(*[torch.arange(10)] * 4, indexing="ij"))
        digit, borrow = (t.argmax(-1).reshape(-1) for t in truth_table())
        oh = lambda a: F.one_hot(a, 10).float()
        opt = torch.optim.Adam(self.cell_params(), lr=lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
        for _ in range(steps):
            o = self.mlp(torch.cat([self.er(oh(r)), self.en(oh(n)), self.eq[q], self.ec(oh(c))], -1))
            loss = F.cross_entropy(o[:, :10], digit) + F.cross_entropy(o[:, 10:], borrow)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            o = self.mlp(torch.cat([self.er(oh(r)), self.en(oh(n)), self.eq[q], self.ec(oh(c))], -1))
            return ((o[:, :10].argmax(-1) == digit) & (o[:, 10:].argmax(-1) == borrow)).float().mean().item()

    def estimate(self, Rp, n):
        """P(q) from the leading digits: the top kr places of 10R + v_t [B, w+1, 10] and the top kn of N [B, w, 10]."""
        return self.est(torch.cat([top(Rp, self.kr).flatten(1), top(n, self.kn).flatten(1)], 1)).softmax(-1)

    def reduce(self, v, N, q=None):
        """LongDiv.reduce with this model's choice. Also returns the choice and the estimate per step, [B, L, 10]
        indexed by the dividend digit t (LSB-first; the division runs t = L-1 down to 0)."""
        B, L, w = v.shape[0], v.shape[1], N.shape[1]
        n = F.one_hot(F.pad(N, (0, 1)), 10).float()                          # w+1 places, top one 0
        zero = F.one_hot(torch.zeros(B, dtype=torch.long, device=N.device), 10).float()
        R = zero[:, None].expand(B, w + 1, 10)
        choice, est = [None] * L, [None] * L
        for t in reversed(range(L)):                                          # dividend digits, MSB first
            Rp = torch.cat([v[:, t:t + 1], R[:, :w]], 1)                      # 10R + v_t
            c, D = zero[:, None].expand(B, 10, 10), []
            for j in range(w + 1):                                            # LSB-first within the trial
                d, c = self.cell(Rp[:, j, None].expand(B, 10, 10), n[:, j, None].expand(B, 10, 10), c)
                D.append(d)
            valid = c[..., 0]                                                 # P(no final borrow), per q
            if self.choose == "borrow":                                       # longdiv: the largest valid q
                higher = torch.cat([torch.flip(torch.cumprod(torch.flip(1 - valid, [1]), 1), [1])[:, 1:],
                                    valid.new_ones(B, 1)], 1)
                wq = valid * higher
                wq = pe = wq / wq.sum(1, keepdim=True).clamp_min(1e-9)
            else:
                pe = self.estimate(Rp, n[:, :w])
                wq = pe if self.choose == "est" else checked(pe, valid)
            if q is not None:                                                 # the quotient digit, given
                wq = F.one_hot(q[:, t] if t < q.shape[1] else torch.zeros_like(q[:, 0]), 10).float()
            choice[t], est[t] = wq, pe
            R = (wq[:, :, None, None] * torch.stack(D, 2)).sum(1)
        return R[:, :w], torch.stack(choice, 1), torch.stack(est, 1)

    def forward(self, inputs):
        R, choice, est = self.reduce(F.one_hot(inputs["v"], 10).float(), inputs["N"], inputs.get("q"))
        return {"target": R.clamp_min(1e-9).log(), "choice": choice, "est": est}


@model("longdiv_tr")
class LongDivTraced(LongDivEst):
    choose = "borrow"


@model("longdiv_est")
class LongDivEstOnly(LongDivEst):
    choose = "est"


@model("longdiv_estc")
class LongDivEstChecked(LongDivEst):
    choose = "pm1"


ext_matched.FOR.update({m: {"mod", "modq"} for m in VARIANTS})


def _number(d):
    return d @ (10 ** np.arange(d.shape[1], dtype=np.int64))


def choices(net, task_name, size, n=2000, seed=10_000, holdout=0):
    """The choice per division step against the true quotient digits, on microlab's usual 2,000 test examples."""
    b = TASKS[task_name].sample(n, size, np.random.default_rng(seed))
    x, y, _ = _tensors(b, next(net.parameters()).device)
    with torch.no_grad():
        out = net(x)
    ok = (out["target"].argmax(-1) == y).all(1).cpu().numpy()
    qd = b.aux["q"] if "q" in b.aux else b.inputs["q"]
    L = out["choice"].shape[1]
    truth = np.pad(qd, ((0, 0), (0, L - qd.shape[1])))[:, ::-1]               # division order: first step first
    ho = ext_ladder.held_out(b.inputs, holdout) if holdout else None
    res, chosen = {}, None
    for name in ("choice", "est"):
        got = out[name].argmax(-1).cpu().numpy()[:, ::-1]
        M = np.zeros((10, 10), np.int64)
        np.add.at(M, (got.ravel(), truth.ravel()), 1)
        for how, g in (("raw", got), ("mapped", M.argmax(1)[got])):
            right = g == truth
            every = right.all(1)
            r = {"per_step": [round(float(a), 4) for a in right.mean(0)], "all": round(float(every.mean()), 4),
                 "exact_if_all": None if not every.any() else round(float(ok[every].mean()), 4)}
            if ho is not None and ho.any():
                r["all_heldout_moduli"] = round(float(every[ho].mean()), 4)
            res[f"{name}_{how}"] = r
            if (name, how) == ("choice", "mapped"):
                chosen = every
        res[f"{name}_map"] = M.argmax(1).tolist()
    digit_ok = (out["target"].argmax(-1) == y).cpu().numpy()
    lead = b.inputs["N"][:, -1]                                               # N's top digit (fixed width)
    res["by_top_digit_of_N"] = {
        str(d): {"n": int((lead == d).sum()), "exact": round(float(ok[lead == d].mean()), 4),
                 "every_choice_right": round(float(chosen[lead == d].mean()), 4),
                 "per_digit": [round(float(a), 3) for a in digit_ok[lead == d].mean(0)]}
        for d in np.unique(lead)}
    res["exact_by_N_digit"] = {str(j): {str(d): round(float(ok[b.inputs["N"][:, j] == d].mean()), 3)
                                        for d in np.unique(b.inputs["N"][:, j])} for j in range(b.inputs["N"].shape[1])}
    return res


def cell(task_name, size, model="longdiv_estc", log=RESULTS, train_on=None, **kw):
    """ext_matched.cell for these models (same training, `wide` and `matched`, and `train_on`: train on another task
    with the same inputs, e.g. "modmix", and score `task_name`), plus the choice measures."""
    kw.setdefault("task", task_name)
    r, net, train_keys = ext_ladder.cell(train_on or task_name, size, model=model, log=None, return_net=True, **kw)
    task = TASKS[task_name]
    if train_on:
        exact, per_digit, split = ext_ladder.evaluate_split(net, task, size, train_keys, holdout=kw.get("holdout", 0))
        r.update(task=task_name, train_task=train_on, exact_train_dist=r["exact"], exact=round(exact, 4),
                 per_digit=[round(v, 3) for v in per_digit], **split)
        if r["ood_size"] is not None:                                         # ext_ladder scored the train task
            r["ood_exact"] = round(evaluate(net, task, r["ood_size"])[0], 4)
    r["wide"] = wide(net, task, size, wider(task_name, size))
    r["matched"] = matched(net, task, size, train_keys)
    if model in VARIANTS:
        r["choice"] = choices(net, task_name, size, holdout=kw.get("holdout", 0))
    if hasattr(net, "cell_fit"):
        r["cell_fit"] = round(net.cell_fit, 5)
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(r) + "\n")
    return r


def show_q(r):
    out = show(r)
    ch = r.get("choice")
    if ch:
        out += (f"  choice all-right raw {ch['choice_raw']['all']:.2f} mapped {ch['choice_mapped']['all']:.2f}"
                f" (exact if all right {ch['choice_mapped']['exact_if_all'] or 0:.2f})"
                f"  est raw {ch['est_raw']['all']:.2f} mapped {ch['est_mapped']['all']:.2f}")
    return out


def window(bits=(6, 7, 8, 10), kr=3, kn=2, n=100_000, seed=0):
    """Run exact long division on mod's distribution and ask how far the window (top kr digits of 10R + v_t, top kn
    of N) fixes q: the best estimator that sees only the window (the most common q per window) and how far off it
    can be. With every error within one, the borrow check (longdiv_estc) can repair all of them."""
    rows = []
    for b in bits:
        bt = TASKS["mod"].sample(n, b, np.random.default_rng(seed))
        N, vd = _number(bt.inputs["N"]), bt.inputs["v"]
        w, L = bt.inputs["N"].shape[1], vd.shape[1]
        R, keys, qs = np.zeros(n, np.int64), [], []
        for t in reversed(range(L)):
            Rp = 10 * R + vd[:, t]
            q = Rp // N
            rw = Rp // 10 ** (w + 1 - kr) if w + 1 >= kr else Rp * 10 ** (kr - w - 1)
            nw = N // 10 ** (w - kn) if w >= kn else N * 10 ** (kn - w)
            keys.append(rw * 10 ** kn + nw)
            qs.append(q)
            R = Rp - q * N
        K, Q = np.stack(keys, 1).ravel(), np.stack(qs, 1).ravel()
        uk, inv = np.unique(K, return_inverse=True)
        counts = np.zeros((len(uk), 10), np.int64)
        np.add.at(counts, (inv, Q), 1)
        best = counts.argmax(1)[inv]
        right = (best == Q).reshape(n, L)
        off = np.abs(best - Q)
        rows.append({"bits": b, "w": w, "kr": kr, "kn": kn, "windows": int(len(uk)),
                     "ambiguous_windows": int(((counts > 0).sum(1) > 1).sum()),
                     "step_right": round(float(right.mean()), 4), "all_steps_right": round(float(right.all(1).mean()), 4),
                     "max_off": int(off.max()), "within_one": round(float((off <= 1).mean()), 4)})
    return rows


def exact_cell(r, n, c):
    """The digit cell's truth table on one-hot inputs: one digit of r - q*n - c and its borrow, for q = 0..9."""
    x = r.argmax(-1) - torch.arange(10, device=r.device) * n.argmax(-1) - c.argmax(-1)
    return F.one_hot(x % 10, 10).float(), F.one_hot((-x).clamp_min(0).add(9).div(10, rounding_mode="floor"), 10).float()


def selftest(bits=(6, 7, 10), n=400, seed=0):
    """With the exact digit cell, longdiv_tr is exact; longdiv_est is exact given the exact quotient digit; and
    longdiv_estc is exact even when the estimate is off by one either way. Also: longdiv_tr == longdiv at init."""
    gen = torch.Generator().manual_seed(seed)

    def oracle(off):
        def est(Rp, n_):
            val = lambda d: (d.argmax(-1) * 10 ** torch.arange(d.shape[1])).sum(1)
            q = (val(Rp) // val(n_)).clamp(0, 9)
            q = (q + torch.randint(-off, off + 1, q.shape, generator=gen)).clamp(0, 9)
            return F.one_hot(q, 10).float()
        return est

    for b in bits:
        bt = TASKS["mod"].sample(n, b, np.random.default_rng(seed))
        x, y, _ = _tensors(bt, "cpu")
        in_w, out_w, _ = TASKS["mod"].widths(b)
        for name, off in (("longdiv_tr", 0), ("longdiv_est", 0), ("longdiv_estc", 0), ("longdiv_estc", 1),
                          ("longdiv_est", 1)):
            net = MODELS[name](in_w, out_w, {})
            net.cell = exact_cell
            if name != "longdiv_tr":
                net.estimate = oracle(off)
            with torch.no_grad():
                out = net(x)
            ex = (out["target"].argmax(-1) == y).all(1).float().mean().item()
            print(f"{b} bits {name:13} estimate off by <= {off}: exact {ex:.3f}")
        torch.manual_seed(1)
        a = MODELS["longdiv"](in_w, out_w, {})
        torch.manual_seed(1)
        t = MODELS["longdiv_tr"](in_w, out_w, {})
        with torch.no_grad():
            same = torch.allclose(a(x)["target"], t(x)["target"])
        print(f"{b} bits longdiv_tr == longdiv: {same}")


def blurtest(bits=(6, 7, 10), blurs=(0.0, 0.1, 0.3), n=500, seed=3):
    """The exact cell (soft_cell) with a right quotient estimate blurred toward uniform: P = (1-b) onehot(q) + b/10.
    The estimate alone stays exact; checked by the borrow, the blur compounds (a blurred remainder's borrow is
    unreliable, which blurs the next choice)."""
    def est(b):
        def f(Rp, n_):
            val = lambda d: (d.argmax(-1) * 10 ** torch.arange(d.shape[1])).sum(1)
            return (1 - b) * F.one_hot((val(Rp) // val(n_)).clamp(0, 9), 10).float() + b / 10
        return f
    for bits in bits:
        bt = TASKS["mod"].sample(n, bits, np.random.default_rng(seed))
        x, y, _ = _tensors(bt, "cpu")
        in_w, out_w, _ = TASKS["mod"].widths(bits)
        for name in ("longdiv_est", "longdiv_estc"):
            row = []
            for b in blurs:
                net = MODELS[name](in_w, out_w, {}, soft_cell=True)
                net.estimate = est(b)
                with torch.no_grad():
                    row.append((net(x)["target"].argmax(-1) == y).all(1).float().mean().item())
            print(f"{bits} bits {name:13} " + "  ".join(f"blur {b}: {e:.3f}" for b, e in zip(blurs, row)))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("cell")
    q.add_argument("task", choices=sorted(TASKS))
    q.add_argument("--size", type=int, required=True)
    q.add_argument("--model", default="longdiv_estc", choices=sorted(MODELS))
    q.add_argument("--steps", type=int, default=3000)
    q.add_argument("--lr", type=float, default=3e-3)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--threads", type=int, default=2)
    q.add_argument("--holdout", type=int, default=0)
    q.add_argument("--extra", default="{}", help="model options as JSON, e.g. '{\"kr\": 4, \"kn\": 3}'")
    q.add_argument("--tag", default="")
    q.add_argument("--nolog", action="store_true")
    s = sub.add_parser("window")
    s.add_argument("--bits", default="6,7,8,10")
    s.add_argument("--kr", type=int, default=3)
    s.add_argument("--kn", type=int, default=2)
    sub.add_parser("selftest")
    sub.add_parser("blurtest")
    a = p.parse_args()
    if a.cmd == "cell":
        r = cell(a.task, a.size, model=a.model, steps=a.steps, lr=a.lr, seed=a.seed, threads=a.threads,
                 holdout=a.holdout, tag=a.tag, log=None if a.nolog else RESULTS, **json.loads(a.extra))
        print(show_q(r), flush=True)
    elif a.cmd == "window":
        for r in window(tuple(int(b) for b in a.bits.split(",")), a.kr, a.kn):
            print(json.dumps(r))
    elif a.cmd == "blurtest":
        blurtest()
    else:
        selftest()


if __name__ == "__main__":
    main()
