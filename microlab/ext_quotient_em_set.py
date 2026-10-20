"""Hard EM with tie-aware lessons, and diagnostics of what the hard-EM cell and lessons look like.

ext_quotient_em's E-step teaches one slot per step: at the last step the slot whose trial best explains the answer
(most digits right, then the answer's log-likelihood), and the best one-step change elsewhere. When several slots give
the same answer (two slots that subtract the same multiple of N, or N divisible by 5 at the top visible digit), that
picks among equals by log-likelihood alone, so the lesson for one digit can flip between slots from example to example
and the estimator gets inconsistent targets. FINDINGS_pre_remainder.md's best-fix taught the whole set of tied-best
values instead (log of their total probability), which lets the network keep whichever of them it already prefers.

Model longdiv_em_set (ext_quotient_em.LongDivEM otherwise):
- last step: the estimator's own slot when its trial gets as many answer digits right as the best one, else the best;
- one-step changes apply only when they get strictly more answer digits right than the model's path;
- the estimator's lesson at the last step and at the changed step is the set of slots that do as well as the best
  (cross-entropy of their total probability); at the other steps of an exactly right path it is the slot taken.

Diagnostics for longdiv_em and longdiv_em_set (rows to results_quotient_em_set.jsonl):
- slot_table: which multiple of N each slot's trial subtracts (digits of Rp - m*N on the register's w+1 places, Rp
  uniform in [0, 10N)); duplicates mean two slots stand for one digit;
- lessons: on the test examples' winning paths (the E-step, which uses the answer), per division step, how far the
  taught slot is a function of the true digit (share where it is that digit's most common slot) and the reverse.

    python -m microlab.ext_quotient_em_set selftest
    python -m microlab.ext_quotient_em_set cell mod --size 6 --model longdiv_em_set --steps 3000
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import ext_quotient_em
from .ext_ladder import evaluate_split
from .ext_matched import matched, wide, wider
from .ext_quotient import choices, show_q
from .ext_quotient_em import LongDivEM, em_test, match
from .models import MODELS, model
from .run import _tensors, evaluate
from .tasks import TASKS

RESULTS = os.path.join(os.path.dirname(__file__), "results_quotient_em_set.jsonl")


@model("longdiv_em_set")
class LongDivEMSet(LongDivEM):
    @torch.no_grad()
    def estep(self, v, n, y, sets=False):
        """As LongDivEM.estep, with the model's own choice kept among equals. With sets=True also returns the
        acceptable slots per step [B, L, 10] (one-hot at steps that were not decided by the answer)."""
        B, L, W1 = v.shape[0], v.shape[1], n.shape[1]
        w = W1 - 1
        bi = torch.arange(B)
        S = F.one_hot(torch.zeros(B, W1, dtype=torch.long), 10).float()
        a, D = torch.zeros(B, L, dtype=torch.long), [None] * L
        for t in reversed(range(L)):
            Rp = torch.cat([v[:, t:t + 1], S[:, :w]], 1)
            D[t] = self.trials(Rp, n)
            a[:, t] = self.est_logits(Rp, n[:, :w]).argmax(-1)
            S = D[t][bi, a[:, t]]
        right, ll = match(D[0], y)
        top = right.max(-1).values
        own_exact = right[bi, a[:, 0]] == w
        s0 = torch.where(right[bi, a[:, 0]] == top, a[:, 0], (right * 1000.0 + ll).argmax(-1))
        slots = a.clone()
        slots[:, 0] = s0
        best = top.clone()                                                     # digits right on the winning path
        accept = F.one_hot(slots, 10).bool()
        accept[:, 0] = right == top[:, None]
        fixed = torch.zeros(B, dtype=torch.bool)
        nb, k10 = n[:, None].expand(B, 10, W1, 10), torch.arange(10)
        for t in range(1, (L - 1 if self.em_depth is None else min(self.em_depth, L - 1)) + 1):
            X, cs = D[t], a[:, None].repeat(1, 10, 1)
            cs[:, :, t] = k10
            for u in reversed(range(1, t)):
                Rp = torch.cat([v[:, None, u:u + 1].expand(B, 10, 1, 10), X[:, :, :w]], 2)
                cs[:, :, u] = self.est_logits(Rp, nb[:, :, :w]).argmax(-1)
                X = self.trial(Rp, nb, cs[:, :, u])
            Rp = torch.cat([v[:, None, :1].expand(B, 10, 1, 10), X[:, :, :w]], 2)
            r_, l_ = match(self.trials(Rp, nb), y)                             # [B, 10 (k), 10 (last slot)]
            rk = r_.max(-1).values                                             # digits right per k (best last slot)
            sl = (r_ * 1000.0 + l_).argmax(-1)
            llk = l_.gather(-1, sl[..., None]).squeeze(-1)
            k = torch.where(rk == rk.max(-1, keepdim=True).values, llk, torch.full_like(llk, -1e9)).argmax(-1)
            better = rk[bi, k] > best
            if better.any():
                cs[:, :, 0] = sl
                new = cs[bi, k]
                slots[better] = new[better]
                best = torch.where(better, rk[bi, k], best)
                acc = F.one_hot(new, 10).bool()
                acc[:, t] = rk == rk[bi, k][:, None]
                acc[:, 0] = r_[bi, k] == r_[bi, k].max(-1, keepdim=True).values
                accept[better] = acc[better]
                fixed |= better
        exact = best == w
        stats = {"own_exact": own_exact.float().mean().item(), "best_exact": exact.float().mean().item(),
                 "fixed": fixed.float().mean().item()}
        if not sets:                                                           # the keys ext_quotient_em.em_test reads
            return slots, exact, stats
        return slots, exact, accept, stats | {"ties_last": (accept[:, 0].sum(-1) > 1).float().mean().item()}

    def em_loss(self, inputs, y):
        v, n = self.digits_in(inputs)
        slots, exact, accept, stats = self.estep(v, n, y, sets=True)
        B, L, W1 = v.shape[0], v.shape[1], n.shape[1]
        w = W1 - 1
        S = F.one_hot(torch.zeros(B, W1, dtype=torch.long), 10).float()
        ce, agree = [None] * L, [None] * L
        for t in reversed(range(L)):
            Rp = torch.cat([v[:, t:t + 1], S[:, :w]], 1)
            lp = self.est_logits(Rp.detach(), n[:, :w]).log_softmax(-1)
            ce[t] = -lp.masked_fill(~accept[:, t], float("-inf")).logsumexp(-1)
            agree[t] = accept[:, t].gather(1, lp.argmax(-1, keepdim=True)).squeeze(1)
            S = self.trial(Rp, n, slots[:, t])
        cell = F.cross_entropy(S[:, :w].clamp_min(1e-9).log().flatten(0, 1), y.flatten())
        mask = torch.ones(B, L) if self.em_lesson == "all" else exact[:, None].float().expand(B, L).clone()
        mask[:, 0] = 1.0
        est = (torch.stack(ce, 1) * mask).sum() / mask.sum()
        stats |= {"est_agree": ((torch.stack(agree, 1).float() * mask).sum() / mask.sum()).item(),
                  "lessons": mask.mean().item(), "cell_loss": cell.item(), "est_loss": est.item()}
        return cell + self.est_weight * est, stats


@torch.no_grad()
def slot_table(net, size, n=1000, seed=50_000):
    """Which digit each slot's trial computes the remainder for, where long division uses it: for slot k and digit m,
    Rp uniform in [m*N, (m+1)*N) (m is the right quotient digit); the share whose trial digits are those of the
    remainder Rp - m*N, on the w places the answer reads ("low") and on all w+1. Returns the best digit per slot (by
    the low share), the shares, and how many digits the slots cover (share > 0.5)."""
    rng = np.random.default_rng(seed)
    w = len(str((1 << size) - 1))
    N = torch.from_numpy(((rng.integers(1 << (size - 2), 1 << (size - 1), n) * 2) + 1).astype(np.int64))
    u = torch.from_numpy(rng.random(n))
    dig = lambda val: torch.remainder(torch.remainder(val, 10 ** (w + 1))[:, None] // 10 ** torch.arange(w + 1), 10)
    oh = lambda d: F.one_hot(d, 10).float()
    A, Lo = torch.zeros(10, 10), torch.zeros(10, 10)
    for m in range(10):
        Rp = m * N + (u * N.double()).long()
        Ro, want = oh(dig(Rp)), dig(Rp - m * N)
        for k in range(10):
            ok = net.trial(Ro, oh(dig(N)), torch.full((n,), k)).argmax(-1) == want
            A[k, m], Lo[k, m] = ok.all(1).float().mean(), ok[:, :w].all(1).float().mean()
    best = Lo.argmax(1)
    return {"slot_digit": best.tolist(), "slot_share_low": [round(float(x), 3) for x in Lo.max(1).values],
            "slot_share_all": [round(float(A[k, best[k]]), 3) for k in range(10)],
            "digits_covered": int(len(set(best[Lo.max(1).values > 0.5].tolist()))),
            "identity_share_low": round(float(Lo.diag().mean()), 4)}


@torch.no_grad()
def lessons(net, task, size, n=2000, seed=10_000):
    """On the test examples' winning paths (the E-step), per division step whose true digit varies: the share of
    examples whose taught slot is the most common slot for their true digit (is the lesson a function of the digit?),
    and whose true digit is the most common digit for their taught slot (does a slot stand for one digit?)."""
    b = task.sample(n, size, np.random.default_rng(seed))
    x, y, _ = _tensors(b, "cpu")
    slots, exact = net.estep(*net.digits_in(x), y)[:2]
    q = torch.from_numpy(b.aux["q"])
    L = slots.shape[1]
    truth = F.pad(q, (0, L - q.shape[1])) if L > q.shape[1] else q[:, :L]
    out = {"exact_paths": round(float(exact.float().mean()), 4), "slot_given_digit": [], "digit_given_slot": []}
    for t in reversed(range(L)):
        m = exact
        if m.sum() < 20 or truth[m, t].unique().numel() < 2:
            continue
        C = torch.zeros(10, 10)
        C.index_put_((slots[m, t], truth[m, t]), torch.ones(int(m.sum())), accumulate=True)
        out["slot_given_digit"].append(round(float(C.max(0).values.sum() / C.sum()), 3))
        out["digit_given_slot"].append(round(float(C.max(1).values.sum() / C.sum()), 3))
    return out


@torch.no_grad()
def last_step_slots(net, task, size, n=2000, seed=10_000):
    """On the test examples' winning paths: how many of the ten last-step slots give the whole answer. With a cell
    whose slots subtract distinct fixed multiples of N it is 1 (2 for the parity cases of N divisible by 5); many
    means the slot hardly matters and the search, not the cell, finds the answer. Returns the shares of 0, 1, 2, 3+."""
    b = task.sample(n, size, np.random.default_rng(seed))
    x, y, _ = _tensors(b, "cpu")
    v, nn_ = net.digits_in(x)
    slots = net.estep(v, nn_, y)[0]
    w = nn_.shape[1] - 1
    S = F.one_hot(torch.zeros(n, w + 1, dtype=torch.long), 10).float()
    for t in reversed(range(1, slots.shape[1])):
        S = net.trial(torch.cat([v[:, t:t + 1], S[:, :w]], 1), nn_, slots[:, t])
    D = net.trials(torch.cat([v[:, :1], S[:, :w]], 1), nn_)
    k = (match(D, y)[0] == w).sum(1)
    return {"0": round(float((k == 0).float().mean()), 4), "1": round(float((k == 1).float().mean()), 4),
            "2": round(float((k == 2).float().mean()), 4), "3+": round(float((k >= 3).float().mean()), 4)}


def cell(task_name, size, model="longdiv_em_set", log=RESULTS, train_on=None, save=None, **kw):
    """ext_quotient_em.cell's training and measures, plus slot_table and lessons."""
    kw.setdefault("task", task_name)
    r, net, train_keys = ext_quotient_em.train(train_on or task_name, size, model=model, **kw)
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
    r["em_test"] = em_test(net, task, size)
    r["slot_table"] = slot_table(net, size)
    r["lessons"] = lessons(net, task, size)
    r["last_step_slots"] = last_step_slots(net, task, size)
    if save:
        torch.save(net.state_dict(), save)
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(r) + "\n")
    return r


CELLS = os.path.join(os.path.dirname(__file__), "results_quotient_em_set_cells.jsonl")


def recheck(paths, out=CELLS):
    """em_test, slot_table, lessons and last_step_slots for saved cells: each path is a cell's .pt next to its row
    .json (as ext_quotient_em_set_modal.py writes them)."""
    for p in paths:
        r = json.loads(open(p.removesuffix(".pt") + ".json").read())
        task = TASKS[r["task"]]
        in_w, out_w, _ = task.widths(r["size"])
        net = MODELS[r["model"]](in_w, out_w, {}, **r["model_kw"])
        net.load_state_dict(torch.load(p))
        row = {"tag": r["tag"], "file": os.path.basename(p), "model": r["model"], "size": r["size"],
               "steps": r["steps"], "seed": r["seed"], "em_test": em_test(net, task, r["size"]),
               "slot_table": slot_table(net, r["size"]), "lessons": lessons(net, task, r["size"]),
               "last_step_slots": last_step_slots(net, task, r["size"])}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ("file", "em_test", "slot_table", "lessons", "last_step_slots")}),
              flush=True)


def selftest(bits=(6, 7), n=400, seed=0):
    """With the exact cell: the oracle estimator's path is kept with single lessons; an estimator stuck on slot 0 is
    repaired as by LongDivEM; slot_table reads the exact cell as slot k = multiplier k, and a cell whose slots 1 and 2
    both subtract 1*N as a duplicate; the loss reaches both parts of the learned model."""
    val = lambda d: (d.argmax(-1) * 10 ** torch.arange(d.shape[-2])).sum(-1)
    for b in bits:
        bt = TASKS["mod"].sample(n, b, np.random.default_rng(seed))
        x, y, _ = _tensors(bt, "cpu")
        in_w, out_w, _ = TASKS["mod"].widths(b)
        net = MODELS["longdiv_em_set"](in_w, out_w, {})
        net.cell_at = ext_quotient_em.exact_cell_at
        net.est_logits = lambda Rp, n_: 10 * F.one_hot((val(Rp) // val(n_)).clamp(0, 9), 10).float()
        slots, exact, accept, st = net.estep(*net.digits_in(x), y, sets=True)
        print(f"{b} bits oracle: {st}  accept size mean per step {accept.sum(-1).float().mean(0).tolist()}")
        net.est_logits = lambda Rp, n_: F.one_hot(torch.zeros(Rp.shape[:-2], dtype=torch.long), 10).float()
        print(f"{b} bits always slot 0: {net.estep(*net.digits_in(x), y)[2]}")
        print(f"{b} bits exact cell slot_table {slot_table(net, b, n=500)}")
        net.cell_at = lambda r, n_, c, slot: ext_quotient_em.exact_cell_at(r, n_, c, torch.where(slot == 2, 1, slot))
        print(f"{b} bits slots 1 and 2 both subtract N: {slot_table(net, b, n=500)}  lessons {lessons(net, TASKS['mod'], b, n=500)}")
        torch.manual_seed(1)
        m = MODELS["longdiv_em_set"](in_w, out_w, {})
        loss, st = m.em_loss(x, y)
        loss.backward()
        g = lambda ps: sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)
        print(f"{b} bits learned: loss {loss.item():.3f} |grad| cell {g(m.cell_params()):.3g} est {g(m.est.parameters()):.3g} {st}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("cell")
    q.add_argument("task", choices=sorted(TASKS))
    q.add_argument("--size", type=int, required=True)
    q.add_argument("--model", default="longdiv_em_set", choices=("longdiv_em", "longdiv_em_set"))
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
    s = sub.add_parser("recheck")
    s.add_argument("paths", nargs="+")
    a = p.parse_args()
    if a.cmd == "cell":
        r = cell(a.task, a.size, model=a.model, steps=a.steps, lr=a.lr, seed=a.seed, threads=a.threads,
                 holdout=a.holdout, train_on=a.train_on, tag=a.tag, log=None if a.nolog else RESULTS,
                 **json.loads(a.extra))
        print(show_q(r) + f"  hard {r['em_test']['exact_hard']:.2f}  slots {r['slot_table']}  lessons {r['lessons']}",
              flush=True)
    elif a.cmd == "recheck":
        recheck(a.paths)
    else:
        selftest()


if __name__ == "__main__":
    main()
