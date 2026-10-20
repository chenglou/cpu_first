"""Hard-EM (best-slot) lessons for the quotient-digit choice in ext_quotient's long-division datapath: does teaching each
choice by the slot that best explains the answer let the datapath learn the remainder from answers alone?

FINDINGS_quotient.md: the leading-digit estimator learns the choice from answers when the subtraction is exact, and the
digit cell learns the subtraction when the quotient digits are given (modq). Trained together from answers, both stall
from a symmetric start: every slot computes the same blend, so neither half has the other to learn from. Here the
choice is a latent variable, filled in during training by a search through the model's own datapath, and both halves
are trained on the path found (hard EM, or Viterbi training).

E-step, per training example (no gradient):
  - the model's path: at every division step the estimator's top slot, through the learned cell;
  - the last step: every slot's trial subtraction is scored against the answer digits (most digits right, then the
    log-likelihood of the answer), and the best slot is taken;
  - one-step changes: for every earlier step t and slot k, step t takes slot k, the steps after it take the
    estimator's top slot (on the changed remainders), and the last step takes its best slot.
  The path that best explains the answer wins; ties keep the model's own path.
M-step:
  - the digit cell: the answer's cross-entropy through the winning path only, one slot per step (as modq trains it
    through the given digits);
  - the estimator: cross-entropy toward the winning path's slot at the last step always, and at every step when the
    winning path gets the whole answer right (the earlier steps' lesson, once the later ones are right; `em_lesson`
    "all" drops that condition). Its inputs are detached, so this lesson does not reach the cell.
The answer only ranks candidate paths during training. Evaluation runs the model exactly as longdiv_est (the soft
choice from the estimator, nothing from the answer), so every ext_quotient measure applies unchanged.

Model longdiv_em: longdiv_est (same parameters, same init per seed) trained this way. Options: em_depth (one-step
changes searched at steps 1..em_depth; None = every step, 0 = the last step's best slot only), em_lesson, est_weight,
and ext_quotient's kr, kn, est_bias0. Measures: ext_quotient.cell's (exact, held-out moduli with holdout=k, wide,
matched, choices), plus exact with the estimator's top slot taken hard (em_test.exact_hard), the E-step's statistics
during training (em_curve) and on the test examples (em_test: how often the model's own path, and the best one-step
change of it, is exactly right). Rows go to microlab/results_quotient_em.jsonl.

    python -m microlab.ext_quotient_em selftest                   # the E-step with the exact cell finds the digits
    python -m microlab.ext_quotient_em bench --size 7              # seconds per training step
    python -m microlab.ext_quotient_em cell mod --size 6 --steps 3000
    python -m microlab.ext_quotient_em report                      # the table of FINDINGS_quotient_em.md
"""

from __future__ import annotations

import argparse
import json
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

RESULTS = os.path.join(os.path.dirname(__file__), "results_quotient_em.jsonl")


def match(D, y):
    """How well trial results D [B, ..., w+1, 10] explain the answer y [B, w]: digits right and the answer's
    log-likelihood, both [B, ...]."""
    w = y.shape[1]
    D = D[..., :w, :]
    yy = y.view(y.shape[0], *[1] * (D.dim() - 3), w).expand(D.shape[:-1])
    right = (D.argmax(-1) == yy).sum(-1)
    return right, D.gather(-1, yy.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9).log().sum(-1)


@model("longdiv_em")
class LongDivEM(LongDivEst):
    choose = "est"

    def __init__(self, in_widths, out_width, aux_widths, em_depth=None, em_lesson="exact", est_weight=1.0, **kw):
        super().__init__(in_widths, out_width, aux_widths, **kw)
        self.em_depth, self.em_lesson, self.est_weight = em_depth, em_lesson, est_weight

    def cell_at(self, r, n, c, slot):
        """The digit cell for one slot per row (LongDiv.cell computes all ten at once): r, n, c [..., 10]; slot [...]."""
        o = self.mlp(torch.cat([self.er(r), self.en(n), self.eq[slot], self.ec(c)], -1))
        return o[..., :10].softmax(-1), o[..., 10:].softmax(-1)

    def trial(self, Rp, n, slot):
        """The trial subtraction for one slot per row, LSB-first through the cell: Rp, n [..., w+1, 10]; slot [...]."""
        c, D = F.one_hot(torch.zeros_like(slot), 10).float(), []
        for j in range(Rp.shape[-2]):
            d, c = self.cell_at(Rp[..., j, :], n[..., j, :], c, slot)
            D.append(d)
        return torch.stack(D, -2)

    def trials(self, Rp, n):
        """All ten trial subtractions: Rp, n [..., w+1, 10] -> [..., 10, w+1, 10]."""
        lead = Rp.shape[:-2]
        ex = lambda a: a.unsqueeze(-3).expand(*lead, 10, *a.shape[-2:])
        return self.trial(ex(Rp), ex(n), torch.arange(10, device=Rp.device).expand(*lead, 10))

    def est_logits(self, Rp, n):
        """The estimator's logits from the leading digits: Rp [..., w+1, 10], n [..., w, 10] -> [..., 10]."""
        lead = Rp.shape[:-2]
        Rp, n = Rp.reshape(-1, *Rp.shape[-2:]), n.reshape(-1, *n.shape[-2:])
        return self.est(torch.cat([top(Rp, self.kr).flatten(1), top(n, self.kn).flatten(1)], 1)).view(*lead, 10)

    @staticmethod
    def digits_in(inputs):
        return F.one_hot(inputs["v"], 10).float(), F.one_hot(F.pad(inputs["N"], (0, 1)), 10).float()

    @torch.no_grad()
    def estep(self, v, n, y):
        """The winning path per example. v [B, L, 10] dividend digits, n [B, w+1, 10] (N with a top 0), y [B, w].
        Returns its slots [B, L] (indexed by the dividend digit t; the division runs t = L-1 down to 0), whether it
        gets the whole answer right [B], and statistics."""
        B, L, W1 = v.shape[0], v.shape[1], n.shape[1]
        w, dev = W1 - 1, v.device
        bi = torch.arange(B, device=dev)
        S = F.one_hot(torch.zeros(B, W1, dtype=torch.long, device=dev), 10).float()
        a, D = torch.zeros(B, L, dtype=torch.long, device=dev), [None] * L
        for t in reversed(range(L)):                                           # the model's own path
            Rp = torch.cat([v[:, t:t + 1], S[:, :w]], 1)
            D[t] = self.trials(Rp, n)                                          # [B, 10, w+1, 10]
            a[:, t] = self.est_logits(Rp, n[:, :w]).argmax(-1)
            S = D[t][bi, a[:, t]]
        right, ll = match(D[0], y)                                             # every last slot: [B, 10]
        own_exact = right[bi, a[:, 0]] == w
        best, s0 = (right * 1000.0 + ll).max(-1)
        slots, exact = a.clone(), right[bi, s0] == w
        slots[:, 0] = s0
        fixed = torch.zeros(B, dtype=torch.bool, device=dev)
        nb, k10 = n[:, None].expand(B, 10, W1, 10), torch.arange(10, device=dev)
        for t in range(1, (L - 1 if self.em_depth is None else min(self.em_depth, L - 1)) + 1):
            X, cs = D[t], a[:, None].repeat(1, 10, 1)                          # step t takes slot k: [B, 10 (k), ...]
            cs[:, :, t] = k10
            for u in reversed(range(1, t)):                                    # later steps: the estimator's top slot
                Rp = torch.cat([v[:, None, u:u + 1].expand(B, 10, 1, 10), X[:, :, :w]], 2)
                cs[:, :, u] = self.est_logits(Rp, nb[:, :, :w]).argmax(-1)
                X = self.trial(Rp, nb, cs[:, :, u])
            Rp = torch.cat([v[:, None, :1].expand(B, 10, 1, 10), X[:, :, :w]], 2)
            r_, l_ = match(self.trials(Rp, nb), y)                             # [B, 10 (k), 10 (last slot)]
            kb, sl = (r_ * 1000.0 + l_).max(-1)                                # the last step's best slot, per k
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
        """The hard-EM loss of a training batch: inputs as for forward, y [B, w] the answer digits."""
        v, n = self.digits_in(inputs)
        slots, exact, stats = self.estep(v, n, y)
        B, L, W1 = v.shape[0], v.shape[1], n.shape[1]
        w = W1 - 1
        S = F.one_hot(torch.zeros(B, W1, dtype=torch.long, device=v.device), 10).float()
        ce, agree = [None] * L, [None] * L
        for t in reversed(range(L)):                                           # the winning path, with gradient
            Rp = torch.cat([v[:, t:t + 1], S[:, :w]], 1)
            logits = self.est_logits(Rp.detach(), n[:, :w])
            ce[t] = F.cross_entropy(logits, slots[:, t], reduction="none")
            agree[t] = logits.argmax(-1) == slots[:, t]
            S = self.trial(Rp, n, slots[:, t])
        cell = F.cross_entropy(S[:, :w].clamp_min(1e-9).log().flatten(0, 1), y.flatten())
        mask = torch.ones(B, L) if self.em_lesson == "all" else exact[:, None].float().expand(B, L).clone()
        mask[:, 0] = 1.0                                                       # the last step: always
        est = (torch.stack(ce, 1) * mask).sum() / mask.sum()
        stats |= {"est_agree": ((torch.stack(agree, 1).float() * mask).sum() / mask.sum()).item(),
                  "lessons": mask.mean().item(), "cell_loss": cell.item(), "est_loss": est.item()}
        return cell + self.est_weight * est, stats

    @torch.no_grad()
    def answer_hard(self, inputs):
        """The answer with the estimator's top slot taken hard at every step: [B, w] digits."""
        v, n = self.digits_in(inputs)
        w = n.shape[1] - 1
        S = F.one_hot(torch.zeros(v.shape[0], w + 1, dtype=torch.long), 10).float()
        for t in reversed(range(v.shape[1])):
            Rp = torch.cat([v[:, t:t + 1], S[:, :w]], 1)
            S = self.trial(Rp, n, self.est_logits(Rp, n[:, :w]).argmax(-1))
        return S[:, :w].argmax(-1)


def em_test(net, task, size, n=2000, seed=10_000, chunk=500):
    """On microlab's usual test examples: exact with hard choices, and the E-step's view (uses the answer, so a
    diagnostic only): the model's own hard path right, and the best one-step change of it right."""
    b = task.sample(n, size, np.random.default_rng(seed))
    x, y, _ = _tensors(b, "cpu")
    out = {"exact_hard": [], "own_exact": [], "best_exact": [], "fixed": []}
    for i in range(0, n, chunk):
        xi, yi = {k: t[i:i + chunk] for k, t in x.items()}, y[i:i + chunk]
        out["exact_hard"].append((net.answer_hard(xi) == yi).all(1).float().mean().item())
        for k, val in net.estep(*net.digits_in(xi), yi)[2].items():
            out[k].append(val)
    return {k: round(float(np.mean(val)), 4) for k, val in out.items()}


def train(task_name, size, model="longdiv_em", seed=0, steps=3000, batch=256, lr=3e-3, checkpoints=4, em_marks=12,
          threads=None, tag="", holdout=0, **model_kw):
    """ext_ladder.cell's training (same init, AdamW, OneCycle, clipping, data stream, checkpoints and evaluation) with
    the hard-EM loss in place of the answer's cross-entropy. Returns (row, net, the training inputs' keys)."""
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
    em_at = {round(steps * (i + 1) / em_marks) for i in range(em_marks)}
    curve, em_curve, losses, seen, recent = [], [], [], [], []
    t0 = time.time()
    for step in range(steps):
        b = sample_kept(task, batch, size, rng, holdout)
        seen.append(keys(b.inputs))
        x, y, _ = _tensors(b, "cpu")
        loss, st = net.em_loss(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        recent = (recent + [st])[-100:]
        if step + 1 in marks:
            curve.append([step + 1, round(evaluate(net, task, size, n=500, seed=20_000 + step)[0], 4)])
        if step + 1 in em_at:
            em_curve.append({"step": step + 1, **{k: round(float(np.mean([s[k] for s in recent])), 4) for k in st}})
    train_keys = np.unique(np.concatenate(seen))
    exact, per_digit, split = evaluate_split(net, task, size, train_keys, holdout=holdout)
    curve.append([steps, round(exact, 4)])
    ood_size = ood_exact = None
    if task.kind == "bits" and size + 1 <= 24 and task.widths(size + 1) == task.widths(size):
        ood_size, ood_exact = size + 1, round(evaluate(net, task, size + 1)[0], 4)
    r = {"task": task_name, "size": size, "model": model, "data": "fresh", "pool": 0, "seed": seed, "steps": steps,
         "batch": batch, "lr": lr, "model_kw": model_kw, "holdout": holdout,
         "params": sum(p.numel() for p in net.parameters()), "exact": round(exact, 4),
         "per_digit": [round(v, 3) for v in per_digit], **split, "train_exact": None, "ood_size": ood_size,
         "ood_exact": ood_exact, "curve": curve, "em_curve": em_curve,
         "final_loss": round(float(np.mean(losses[-100:])), 4), "seconds": round(time.time() - t0, 1),
         "device": "cpu", "threads": torch.get_num_threads(), "tag": tag}
    return r, net, train_keys


def cell(task_name, size, model="longdiv_em", log=RESULTS, train_on=None, **kw):
    """ext_quotient.cell for hard-EM training: `train_on` trains on another task with the same inputs (e.g. modmix) and
    scores `task_name`; then wide, matched, choices and em_test."""
    kw.setdefault("task", task_name)
    r, net, train_keys = train(train_on or task_name, size, model=model, **kw)
    task = TASKS[task_name]
    if train_on:
        exact, per_digit, split = evaluate_split(net, task, size, train_keys, holdout=kw.get("holdout", 0))
        r.update(task=task_name, train_task=train_on, exact_train_dist=r["exact"], exact=round(exact, 4),
                 per_digit=[round(v, 3) for v in per_digit], **split)
        if r["ood_size"] is not None:                                         # train() scored the train task
            r["ood_exact"] = round(evaluate(net, task, r["ood_size"])[0], 4)
    r["wide"] = wide(net, task, size, wider(task_name, size))
    r["matched"] = matched(net, task, size, train_keys)
    r["choice"] = choices(net, task_name, size, holdout=kw.get("holdout", 0))
    r["em_test"] = em_test(net, task, size)
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(r) + "\n")
    return r


def show_em(r):
    e = r["em_test"]
    return (show_q(r) + f"  hard {e['exact_hard']:.2f}  E-step own/best {e['own_exact']:.2f}/{e['best_exact']:.2f}"
            f"  ho{r.get('holdout', 0)} {r['steps']}")


def exact_cell_at(r, n, c, slot):
    """The digit cell's truth table for one slot per row, on one-hot inputs: one digit of r - slot*n - c and its borrow."""
    x = r.argmax(-1) - slot * n.argmax(-1) - c.argmax(-1)
    return F.one_hot(x % 10, 10).float(), F.one_hot((-x).clamp_min(0).add(9).div(10, rounding_mode="floor"), 10).float()


def selftest(bits=(6, 7), n=400, seed=0):
    """With the exact cell: an oracle estimator's path is right and the E-step keeps it; an estimator that always takes
    slot 0 is repaired exactly on the examples with at most one nonzero quotient digit before the last step, and the
    repaired slots are the true digits. With the learned cell: the loss reaches the cell and the estimator, and
    longdiv_em is longdiv_est at init."""
    val = lambda d: (d.argmax(-1) * 10 ** torch.arange(d.shape[-2])).sum(-1)
    for b in bits:
        bt = TASKS["mod"].sample(n, b, np.random.default_rng(seed))
        x, y, _ = _tensors(bt, "cpu")
        in_w, out_w, _ = TASKS["mod"].widths(b)
        L = in_w["v"]
        truth = F.pad(torch.from_numpy(bt.aux["q"]), (0, L - in_w["N"]))     # the digit at division step t
        net = MODELS["longdiv_em"](in_w, out_w, {})
        net.cell_at = exact_cell_at
        net.est_logits = lambda Rp, n_: 10 * F.one_hot((val(Rp) // val(n_)).clamp(0, 9), 10).float()
        slots, exact, st = net.estep(*net.digits_in(x), y)
        print(f"{b} bits oracle estimator: {st}  slots == digits {(slots == truth).all(1).float().mean():.3f}"
              f"  hard exact {(net.answer_hard(x) == y).all(1).float().mean():.3f}")
        net.est_logits = lambda Rp, n_: F.one_hot(torch.zeros(Rp.shape[:-2], dtype=torch.long), 10).float()
        slots, exact, st = net.estep(*net.digits_in(x), y)
        fixable = (truth[:, 1:] > 0).sum(1) <= 1
        print(f"{b} bits always slot 0: {st}  expected best_exact {fixable.float().mean():.3f}"
              f"  exact == fixable {bool((exact == fixable).all())}"
              f"  slots == digits where exact {bool((slots[exact] == truth[exact]).all())}")
        torch.manual_seed(1)
        em = MODELS["longdiv_em"](in_w, out_w, {})
        torch.manual_seed(1)
        est = MODELS["longdiv_est"](in_w, out_w, {})
        with torch.no_grad():
            same = torch.allclose(em(x)["target"], est(x)["target"])
        loss, st = em.em_loss(x, y)
        loss.backward()
        g = lambda ps: sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)
        print(f"{b} bits learned cell: longdiv_em == longdiv_est at init {same}  loss {loss.item():.3f}"
              f"  |grad| cell {g(em.cell_params()):.3g} estimator {g(em.est.parameters()):.3g}  {st}")


def bench(size=7, steps=5, batch=256, threads=4, seed=0):
    """Seconds per hard-EM training step (E-step, M-step, backward), against longdiv_est's soft training step."""
    torch.set_num_threads(threads)
    task = TASKS["mod"]
    in_w, out_w, _ = task.widths(size)
    torch.manual_seed(seed)
    em, est = MODELS["longdiv_em"](in_w, out_w, {}), MODELS["longdiv_est"](in_w, out_w, {})
    x, y, _ = _tensors(task.sample(batch, size, np.random.default_rng(seed)), "cpu")
    for name, f in (("hard-EM", lambda: em.em_loss(x, y)[0]),
                    ("soft", lambda: F.cross_entropy(est(x)["target"].flatten(0, 1), y.flatten()))):
        f().backward()
        t0 = time.time()
        for _ in range(steps):
            f().backward()
        print(f"{size} bits {name}: {(time.time() - t0) / steps:.3f} s/step ({threads} threads)")


VARIANT = "hard-EM best slot"


def name(r):
    kw = {k: v for k, v in r["model_kw"].items() if k != "task"}
    out = VARIANT
    if kw.get("em_depth") == 0:
        out += ", last step only"
    if kw.get("em_lesson") == "all":
        out += ", every step taught"
    if kw.get("est_bias0"):
        out += ", starts at q = 0"
    if r.get("train_task"):
        out += f", trained on {r['train_task']}"
    return out


BASELINES = ("longdiv, borrow choice (rerun)", "estimator", "estimator, starts at q = 0",
             "estimator, trained on modmix", "quotient digits given (modq)")


def report(path=RESULTS):
    """The table of FINDINGS_quotient_em.md: these runs, then ext_quotient's matching rows (via ext_quotient_report)."""
    groups = {}
    for r in qr.load(path):
        groups.setdefault((name(r), r["size"], r["steps"]), {}).setdefault(r.get("holdout", 0), []).append(r)
    for r in qr.rows():
        n = qr.name(r)
        if n in BASELINES and r["size"] in (6, 7) and not r["model_kw"].get("soft_cell"):
            groups.setdefault((n, r["size"], r["steps"]), {}).setdefault(r.get("holdout", 0), []).append(r)
    order = lambda k: (not k[0].startswith(VARIANT), BASELINES.index(k[0]) if k[0] in BASELINES else 0, k[0], k[1], k[2])
    lines = ["| variant | bits | steps | exact | held-out moduli (trained) | one bit wider | one digit wider | "
             "every choice right | exact, hard choice |", "|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(groups, key=order):
        main, ho = groups[k].get(0, []), groups[k].get(4, [])
        hos = qr.span([r.get("exact_heldout_moduli") for r in ho]) + (f" ({qr.span([qr.kept(r) for r in ho])})"
                                                                      if ho else "")
        ch = [r["choice"]["choice_mapped"]["all"] for r in main if r.get("choice")]
        hard = [r["em_test"]["exact_hard"] for r in main if r.get("em_test")]
        lines.append(f"| {k[0]} | {k[1]} | {k[2] // 1000}k | {qr.span([r['exact'] for r in main])} | {hos} | "
                     f"{qr.span([qr.wider1(r) for r in main])} | {qr.span([qr.digit1(r) for r in main])} | "
                     f"{qr.span(ch)} | {qr.span(hard)} |")
    return "\n".join(lines)


def steps_report(path=RESULTS):
    """Per run: the choice per division step (slots mapped to digits), the slot map, and the E-step's curve."""
    out = []
    for r in sorted(qr.load(path), key=lambda r: (name(r), r["size"], r["steps"], r.get("holdout", 0), r["seed"])):
        c, e = r["choice"], r["em_test"]
        curve = " ".join(f"{p['step'] // 1000 if p['step'] >= 1000 else p['step']}"
                         f"{'k' if p['step'] >= 1000 else ''}:{p['best_exact']:.2f}/{p['own_exact']:.2f}"
                         for p in r["em_curve"][::3])
        out.append(f"{name(r):45} {r['size']} {r['steps'] // 1000:>2}k ho{r.get('holdout', 0)} s{r['seed']}  "
                   f"exact {r['exact']:.2f} hard {e['exact_hard']:.2f}  every choice {c['choice_mapped']['all']:.2f} "
                   f"[{' '.join(f'{a:.2f}' for a in c['choice_mapped']['per_step'])}]  map {c['choice_map']}  "
                   f"E-step best/own {curve}")
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
    q.add_argument("--extra", default="{}", help="model options as JSON, e.g. '{\"em_depth\": 0}'")
    q.add_argument("--tag", default="")
    q.add_argument("--nolog", action="store_true")
    s = sub.add_parser("bench")
    s.add_argument("--size", type=int, default=7)
    s.add_argument("--steps", type=int, default=5)
    s.add_argument("--threads", type=int, default=4)
    sub.add_parser("selftest")
    s = sub.add_parser("report")
    s.add_argument("--steps", action="store_true", help="per run: choices per step and the E-step curve")
    a = p.parse_args()
    if a.cmd == "cell":
        r = cell(a.task, a.size, steps=a.steps, lr=a.lr, seed=a.seed, threads=a.threads, holdout=a.holdout,
                 train_on=a.train_on, tag=a.tag, log=None if a.nolog else RESULTS, **json.loads(a.extra))
        print(show_em(r), flush=True)
    elif a.cmd == "bench":
        bench(a.size, a.steps, threads=a.threads)
    elif a.cmd == "report":
        print(steps_report() if a.steps else report())
    else:
        selftest()


if __name__ == "__main__":
    main()
