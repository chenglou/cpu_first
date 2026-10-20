"""The long-division datapath started the schoolbook way, with the dividend's top w digits as the running remainder:
does the quotient-digit choice learn from answers once every choice is visible in the answer?

In ext_quotient's datapath the first w division steps take in the dividend's top w digits. Their quotient digit is
always 0 (q < N < 10^w), and they never reach the answer's digits: the register keeps w+1 digits and drops its top one
at every shift, so the answer is (v mod 10^w - q*N) mod 10^w, with q made of the last w choices. Whatever slot the model
picks there is never corrected, yet the remainders it produces are what the estimator reads at the later steps.
Here the datapath starts at step w-1 with R = v's top w digits (always below N), which is what those steps compute when
each subtracts nothing, so it runs only the w steps the answer can see. No example-specific information is added: the
skipped digit is 0 for every input of mod and modmix.

Models (registered into microlab's MODELS):
- longdiv_est_prefix  longdiv_est with this start (soft choice from the estimator), trained from answers as in
                      ext_quotient (soft path, cross-entropy of the answer);
- longdiv_em_prefix   longdiv_em with this start, trained by ext_quotient_em's hard EM.
Both report their choices for the w steps they run (so ext_quotient.choices scores those steps). Measures and rows as
in ext_quotient_em (results_quotient_prefix.jsonl).

    python -m microlab.ext_quotient_prefix selftest
    python -m microlab.ext_quotient_prefix reach        # which choices the answer can see (exact cell)
    python -m microlab.ext_quotient_prefix cell mod --size 6 --model longdiv_em_prefix --steps 3000
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import ext_ladder, ext_matched, ext_quotient_em
from .ext_ladder import evaluate_split
from .ext_matched import matched, wide, wider
from .ext_quotient import LongDivEst, choices, show_q
from .ext_quotient_em import LongDivEM, em_test, match
from .models import MODELS, model
from .run import _tensors, evaluate
from .tasks import TASKS

RESULTS = os.path.join(os.path.dirname(__file__), "results_quotient_prefix.jsonl")
MODELS_HERE = ("longdiv_est_prefix", "longdiv_em_prefix")


def prefix(v, w):
    """The register after the first L-w steps when each subtracts nothing: v's top w digits and a 0 on top.
    v [B, L, 10] one-hot (LSB-first, L = 2w) -> [B, w+1, 10]."""
    zero = F.one_hot(torch.zeros(v.shape[0], 1, dtype=torch.long), 10).float()
    return torch.cat([v[:, v.shape[1] - w:], zero], 1)


@model("longdiv_est_prefix")
class LongDivEstPrefix(LongDivEst):
    choose = "est"

    def reduce(self, v, N, q=None):
        """LongDivEst.reduce from the prefix: steps t = w-1 .. 0 only. Choices and estimates are [B, w, 10] (by t)."""
        B, w = v.shape[0], N.shape[1]
        n = F.one_hot(F.pad(N, (0, 1)), 10).float()
        zero = F.one_hot(torch.zeros(B, dtype=torch.long), 10).float()
        R = prefix(v, w)
        choice, est = [None] * w, [None] * w
        for t in reversed(range(w)):
            Rp = torch.cat([v[:, t:t + 1], R[:, :w]], 1)
            c, D = zero[:, None].expand(B, 10, 10), []
            for j in range(w + 1):
                d, c = self.cell(Rp[:, j, None].expand(B, 10, 10), n[:, j, None].expand(B, 10, 10), c)
                D.append(d)
            wq = pe = self.estimate(Rp, n[:, :w])
            if q is not None:
                wq = F.one_hot(q[:, t], 10).float()
            choice[t], est[t] = wq, pe
            R = (wq[:, :, None, None] * torch.stack(D, 2)).sum(1)
        return R[:, :w], torch.stack(choice, 1), torch.stack(est, 1)


@model("longdiv_em_prefix")
class LongDivEMPrefix(LongDivEM):
    def forward(self, inputs):
        v, n = self.digits_in(inputs)
        R, choice, est = LongDivEstPrefix.reduce(self, v, inputs["N"])
        return {"target": R.clamp_min(1e-9).log(), "choice": choice, "est": est}

    @torch.no_grad()
    def estep(self, v, n, y):
        """LongDivEM.estep from the prefix (steps w-1 .. 0). Slots are [B, w], by t."""
        B, W1 = v.shape[0], n.shape[1]
        w = W1 - 1
        bi = torch.arange(B)
        S = prefix(v, w)
        a, D = torch.zeros(B, w, dtype=torch.long), [None] * w
        for t in reversed(range(w)):
            Rp = torch.cat([v[:, t:t + 1], S[:, :w]], 1)
            D[t] = self.trials(Rp, n)
            a[:, t] = self.est_logits(Rp, n[:, :w]).argmax(-1)
            S = D[t][bi, a[:, t]]
        right, ll = match(D[0], y)
        own_exact = right[bi, a[:, 0]] == w
        best, s0 = (right * 1000.0 + ll).max(-1)
        slots, exact = a.clone(), right[bi, s0] == w
        slots[:, 0] = s0
        fixed = torch.zeros(B, dtype=torch.bool)
        nb, k10 = n[:, None].expand(B, 10, W1, 10), torch.arange(10)
        for t in range(1, (w - 1 if self.em_depth is None else min(self.em_depth, w - 1)) + 1):
            X, cs = D[t], a[:, None].repeat(1, 10, 1)
            cs[:, :, t] = k10
            for u in reversed(range(1, t)):
                Rp = torch.cat([v[:, None, u:u + 1].expand(B, 10, 1, 10), X[:, :, :w]], 2)
                cs[:, :, u] = self.est_logits(Rp, nb[:, :, :w]).argmax(-1)
                X = self.trial(Rp, nb, cs[:, :, u])
            Rp = torch.cat([v[:, None, :1].expand(B, 10, 1, 10), X[:, :, :w]], 2)
            r_, l_ = match(self.trials(Rp, nb), y)
            kb, sl = (r_ * 1000.0 + l_).max(-1)
            kb, k = kb.max(-1)
            better = kb > best + 1e-3
            if better.any():
                cs[:, :, 0] = sl
                best = torch.where(better, kb, best)
                slots[better] = cs[bi, k][better]
                exact[better] = (r_[bi, k, sl[bi, k]] == w)[better]
                fixed |= better
        return slots, exact, {"own_exact": own_exact.float().mean().item(), "best_exact": exact.float().mean().item(),
                              "fixed": fixed.float().mean().item()}

    def em_loss(self, inputs, y):
        v, n = self.digits_in(inputs)
        slots, exact, stats = self.estep(v, n, y)
        B, W1 = v.shape[0], n.shape[1]
        w = W1 - 1
        S, ce, agree = prefix(v, w), [None] * w, [None] * w
        for t in reversed(range(w)):
            Rp = torch.cat([v[:, t:t + 1], S[:, :w]], 1)
            logits = self.est_logits(Rp.detach(), n[:, :w])
            ce[t] = F.cross_entropy(logits, slots[:, t], reduction="none")
            agree[t] = logits.argmax(-1) == slots[:, t]
            S = self.trial(Rp, n, slots[:, t])
        cell = F.cross_entropy(S[:, :w].clamp_min(1e-9).log().flatten(0, 1), y.flatten())
        mask = torch.ones(B, w) if self.em_lesson == "all" else exact[:, None].float().expand(B, w).clone()
        mask[:, 0] = 1.0
        est = (torch.stack(ce, 1) * mask).sum() / mask.sum()
        stats |= {"est_agree": ((torch.stack(agree, 1).float() * mask).sum() / mask.sum()).item(),
                  "lessons": mask.mean().item(), "cell_loss": cell.item(), "est_loss": est.item()}
        return cell + self.est_weight * est, stats

    @torch.no_grad()
    def answer_hard(self, inputs):
        v, n = self.digits_in(inputs)
        w = n.shape[1] - 1
        S = prefix(v, w)
        for t in reversed(range(w)):
            Rp = torch.cat([v[:, t:t + 1], S[:, :w]], 1)
            S = self.trial(Rp, n, self.est_logits(Rp, n[:, :w]).argmax(-1))
        return S[:, :w].argmax(-1)


ext_matched.FOR.update({m: {"mod"} for m in MODELS_HERE})


def cell(task_name, size, model="longdiv_em_prefix", log=RESULTS, train_on=None, **kw):
    """Hard EM (ext_quotient_em.cell's training) for longdiv_em_prefix; ext_ladder's soft training from answers (as
    ext_quotient.cell) for longdiv_est_prefix. Then wide, matched, choices, em_test (EM model only)."""
    if model == "longdiv_em_prefix":
        return ext_quotient_em.cell(task_name, size, model=model, log=log, train_on=train_on, **kw)
    kw.setdefault("task", task_name)
    r, net, train_keys = ext_ladder.cell(train_on or task_name, size, model=model, log=None, return_net=True, **kw)
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
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(r) + "\n")
    return r


def selftest(bits=(6, 7, 10), n=400, seed=0):
    """With the exact cell: the prefix start plus the true digits is exact (both models); the EM model's E-step with
    an estimator that always takes slot 0 finds the exact path on every example at 6 bits (two steps: one change and
    the best last slot cover every pair) and repairs one wrong step at 7 and 10 bits."""
    for b in bits:
        bt = TASKS["mod"].sample(n, b, np.random.default_rng(seed))
        x, y, _ = _tensors(bt, "cpu")
        in_w, out_w, _ = TASKS["mod"].widths(b)
        q = torch.from_numpy(bt.aux["q"])
        est = MODELS["longdiv_est_prefix"](in_w, out_w, {})
        est.cell = lambda r, n_, c: ext_quotient_em_cell_all(r, n_, c)
        with torch.no_grad():
            R, _, _ = est.reduce(F.one_hot(x["v"], 10).float(), x["N"], q)
        print(f"{b} bits prefix start, exact cell, true digits: exact {(R.argmax(-1) == y).all(1).float().mean():.3f}")
        em = MODELS["longdiv_em_prefix"](in_w, out_w, {})
        em.cell_at = ext_quotient_em.exact_cell_at
        em.est_logits = lambda Rp, n_: F.one_hot(torch.zeros(Rp.shape[:-2], dtype=torch.long), 10).float()
        slots, exact, st = em.estep(*em.digits_in(x), y)
        fixable = (q[:, 1:] > 0).sum(1) <= 1
        print(f"{b} bits EM, exact cell, always slot 0: {st}  expected best_exact >= {fixable.float().mean():.3f}")
        torch.manual_seed(1)
        em = MODELS["longdiv_em_prefix"](in_w, out_w, {})
        loss, st = em.em_loss(x, y)
        loss.backward()
        print(f"{b} bits EM learned cell: loss {loss.item():.3f}  {st}  forward {tuple(em(x)['choice'].shape)}")


def reach(bits=(6, 7, 8, 10), n=2000, seed=1):
    """What the answer can see, with the exact cell on the full datapath: (1) random slots at the first w steps and the
    true digits after them leave the answer exact (those choices never reach it); (2) the share of examples where
    changing one of the last w digits to another value still gives the exact answer (N divisible by 5)."""
    for b in bits:
        bt = TASKS["mod"].sample(n, b, np.random.default_rng(seed))
        x, y, _ = _tensors(bt, "cpu")
        in_w, out_w, _ = TASKS["mod"].widths(b)
        L, w = in_w["v"], in_w["N"]
        truth = F.pad(torch.from_numpy(bt.aux["q"]), (0, L - w))
        net = MODELS["longdiv_em"](in_w, out_w, {})
        net.cell_at = ext_quotient_em.exact_cell_at
        v, nn_ = net.digits_in(x)

        def run(slots):
            S = F.one_hot(torch.zeros(n, w + 1, dtype=torch.long), 10).float()
            for t in reversed(range(L)):
                S = net.trial(torch.cat([v[:, t:t + 1], S[:, :w]], 1), nn_, slots[:, t])
            return (S[:, :w].argmax(-1) == y).all(1)

        rnd = truth.clone()
        rnd[:, w:] = torch.randint(0, 10, (n, L - w))
        alt = torch.zeros(n, dtype=torch.bool)
        for t in range(w):
            for k in range(10):
                s2 = truth.clone()
                s2[:, t] = k
                alt |= run(s2) & (truth[:, t] != k)
        print(f"{b} bits: random slots at the first {w} steps, true digits after: exact {run(rnd).float().mean():.3f};"
              f"  one of the last {w} digits changed and still exact: {alt.float().mean():.3f}")


def ext_quotient_em_cell_all(r, n, c):
    """The exact cell for all ten slots at once (LongDiv.cell's shapes: r, n, c [B, 10, 10])."""
    x = r.argmax(-1) - torch.arange(10) * n.argmax(-1) - c.argmax(-1)
    return F.one_hot(x % 10, 10).float(), F.one_hot((-x).clamp_min(0).add(9).div(10, rounding_mode="floor"), 10).float()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("cell")
    q.add_argument("task", choices=sorted(TASKS))
    q.add_argument("--size", type=int, required=True)
    q.add_argument("--model", default="longdiv_em_prefix", choices=MODELS_HERE)
    q.add_argument("--steps", type=int, default=3000)
    q.add_argument("--lr", type=float, default=3e-3)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--threads", type=int, default=2)
    q.add_argument("--holdout", type=int, default=0)
    q.add_argument("--train_on", default=None)
    q.add_argument("--extra", default="{}")
    q.add_argument("--tag", default="")
    q.add_argument("--nolog", action="store_true")
    sub.add_parser("selftest")
    sub.add_parser("reach")
    a = p.parse_args()
    if a.cmd == "cell":
        r = cell(a.task, a.size, model=a.model, steps=a.steps, lr=a.lr, seed=a.seed, threads=a.threads,
                 holdout=a.holdout, train_on=a.train_on, tag=a.tag, log=None if a.nolog else RESULTS,
                 **json.loads(a.extra))
        print(show_q(r) + (f"  hard {r['em_test']['exact_hard']:.2f}" if "em_test" in r else ""), flush=True)
    elif a.cmd == "reach":
        reach()
    else:
        selftest()


if __name__ == "__main__":
    main()
