"""Architectures shaped like the arithmetic, on the same ladder as ext_ladder: does a piece become a rule once the
network's structure matches it?

Models (registered into microlab's MODELS). Each reads its widths from the inputs, so it also runs at widths it never
trained on:
- colsum   multiplication made local, the way addition already is. A learned digit-pair table P[a_i, b_j] (one vector
           per pair: a bilinear form on one-hot digits) is summed along anti-diagonals into the schoolbook columns
           c_k = sum_{i+j=k} P[a_i, b_j]; one shared learned carry cell (a GRU cell) then scans the columns LSB-first and
           reads out one answer digit per column. Addition uses the aligned pairs (i = j); squaring uses x * x. Nothing
           says what a product or a carry is.
- abacus   transformer encoder whose position embedding is the digit's significance, shared by every operand and the
           answer (position coupling), with a random offset per example while training (Abacus embeddings) so wider
           numbers reuse trained positions. `abacus_loop`: one shared block applied `depth` times.
- longdiv  the remainder as long division. An MSB-first scan over v's digits keeps a running remainder R as w+1 digit
           distributions. Each step forms 10R + v_t, tries all ten quotient digits q at once with one shared learned
           digit cell (r_j, n_j, q, carry) -> (digit, carry) scanned LSB-first, and keeps the largest q whose
           subtraction does not borrow; softly, as P(q is the largest non-borrowing digit) from the cells' final-carry
           probabilities. Only the digit cell is learned; the scan, the ten trials and the keep-largest rule are given.
- longdiv_st  the same with a hard (straight-through) choice of q and digits snapped after every step.
  On task `modq` (the quotient's digits given as an input) longdiv uses them instead of choosing: a diagnostic that
  leaves only the subtraction to learn.
- colsum_longdiv  x^2 mod N end to end: colsum's x^2 columns, as digit distributions, are longdiv's dividend.

Rule-vs-table measures added to ext_ladder.cell's: exact at wider sizes than trained, on all test inputs and on those
with a digit past the trained width (`wide`); exact on seen vs unseen inputs of equal operand lengths (`matched`).
`train_on="modmix"` / `"sqmodmix"` trains on a mix with easy (few-wrap) examples and scores the plain task. Results go
to microlab/ladder_matched_results.jsonl.

    python -m microlab.ext_matched cell mul --size 4 --model colsum --steps 3000
    python -m microlab.ext_matched climb mod --model longdiv --data fresh --seeds 2
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import ext_ladder
from .ext_ladder import PASS, SIZES, line
from .models import MODELS, model
from .run import _tensors, evaluate
from .tasks import TASKS, Batch, Mod, SqMod, _odd_moduli, digits, ndig

RESULTS = os.path.join(os.path.dirname(__file__), "ladder_matched_results.jsonl")
FOR = {"colsum": {"add", "mul", "sq"}, "longdiv": {"mod", "modq"}, "longdiv_st": {"mod"}, "colsum_longdiv": {"sqmod"},
       "abacus": {"add", "mul", "sq", "mod", "sqmod"}, "abacus_loop": {"add", "mul", "sq", "mod", "sqmod"}}


class ModMix(Mod):
    """mod with easy dividends mixed in (a curriculum over wraps, as data): v uniform below N * 10^k for k uniform in
    0..w-1, or below N^2 (k = w). k = 0 needs no reduction at all; each k adds one real quotient digit."""

    def __init__(self):
        super().__init__()
        self.name, self.describe = "modmix", "v mod N, v below N * 10^k, k uniform (easy dividends mixed in)"

    def sample(self, n, b, rng):
        N = _odd_moduli(n, b, rng)
        w = ndig((1 << b) - 1)
        k = rng.integers(0, w + 1, n)
        bound = np.where(k == w, N.astype(np.float64) ** 2, N * 10.0 ** k)
        v = np.minimum((rng.random(n) * bound).astype(np.int64), N * N - 1)
        return Batch({"N": digits(N, w), "v": digits(v, 2 * w)}, digits(v % N, w), {"q": digits(v // N, w)})


class SqModMix(SqMod):
    """sqmod with small x mixed in: x uniform below N^e, e uniform in {1/2, 3/4, 7/8, 1} (x < sqrt N never wraps)."""

    def __init__(self):
        super().__init__()
        self.name, self.describe = "sqmodmix", "x^2 mod N, x below N^e, e in {1/2, 3/4, 7/8, 1}"

    def sample(self, n, b, rng):
        N = _odd_moduli(n, b, rng)
        e = rng.choice([0.5, 0.75, 0.875, 1.0], n)
        x = np.minimum((rng.random(n) * N.astype(np.float64) ** e).astype(np.int64), N - 1)
        w = ndig((1 << b) - 1)
        p = x * x
        return Batch({"N": digits(N, w), "x": digits(x, w)}, digits(p % N, w),
                     {"pre": digits(p, 2 * w), "q": digits(p // N, w)})


class ModQ(Mod):
    """mod with the quotient's digits as an extra input (a diagnostic: longdiv then only has to subtract)."""

    def __init__(self):
        super().__init__()
        self.name, self.describe = "modq", "v mod N, given q = v // N as an input"

    def widths(self, b):
        w = ndig((1 << b) - 1)
        return {"N": w, "v": 2 * w, "q": w}, w, {}

    def sample(self, n, b, rng):
        out = super().sample(n, b, rng)
        return Batch({**out.inputs, "q": out.aux["q"]}, out.target)


TASKS["modmix"], TASKS["sqmodmix"], TASKS["modq"] = ModMix(), SqModMix(), ModQ()


def out_width(task, inputs):
    if task == "add":
        return max(inputs["a"].shape[1], inputs["b"].shape[1]) + 1
    if task == "mul":
        return inputs["a"].shape[1] + inputs["b"].shape[1]
    if task == "sq":
        return 2 * inputs["x"].shape[1]
    return inputs["N"].shape[1]


@model("colsum")
class ColSum(nn.Module):
    def __init__(self, in_widths, out_width, aux_widths, task="mul", d=64, hidden=128, **kw):
        super().__init__()
        self.task = task
        self.pair = nn.Parameter(torch.randn(10, 10, d) * 0.3)
        self.inp = nn.Linear(d, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, 10)

    def columns(self, a, b):
        B, ka, kb = a.shape[0], a.shape[1], b.shape[1]
        if self.task == "add":
            k = max(ka, kb)
            a, b = F.pad(a, (0, k - ka)), F.pad(b, (0, k - kb))
            p = self.pair[a, b]                                                   # aligned pairs
            return torch.cat([p, p.new_zeros(B, 1, p.shape[-1])], 1)              # + the final carry's column
        p = self.pair[a[:, :, None], b[:, None, :]]                               # [B, ka, kb, d]: every pair
        idx = (torch.arange(ka, device=a.device)[:, None] + torch.arange(kb, device=a.device)[None, :]).flatten()
        return p.new_zeros(B, ka + kb, p.shape[-1]).index_add(1, idx, p.flatten(1, 2))   # anti-diagonal sums

    def forward(self, inputs):
        a, b = (inputs["x"], inputs["x"]) if self.task in ("sq", "sqmod") else (inputs["a"], inputs["b"])
        z = self.inp(self.columns(a, b))
        h, hs = z.new_zeros(z.shape[0], self.cell.hidden_size), []
        for t in range(z.shape[1]):                                               # LSB-first carry scan
            h = self.cell(z[:, t], h)
            hs.append(h)
        return {"target": self.out(torch.stack(hs, 1))}


@model("abacus")
class Abacus(nn.Module):
    def __init__(self, in_widths, out_width, aux_widths, task="add", d=128, heads=4, depth=3, loop=False, max_pos=96,
                 max_offset=32, **kw):
        super().__init__()
        self.task, self.names, self.depth, self.loop, self.max_offset = task, list(in_widths), depth, loop, max_offset
        self.digit = nn.Embedding(11, d)                                          # 10 = answer slot
        self.field = nn.Embedding(len(self.names) + 1, d)
        self.pos = nn.Embedding(max_pos, d)                                       # indexed by significance
        layer = lambda: nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.ModuleList([layer()] if loop else [layer() for _ in range(depth)])
        self.norm, self.out = nn.LayerNorm(d), nn.Linear(d, 10)

    def forward(self, inputs):
        first = inputs[self.names[0]]
        B, dev = first.shape[0], first.device
        off = (torch.randint(0, self.max_offset + 1, (B, 1), device=dev) if torch.is_grad_enabled()   # training only
               else torch.zeros(B, 1, dtype=torch.long, device=dev))
        wo = out_width(self.task, inputs)
        digs = [inputs[n] for n in self.names] + [torch.full((B, wo), 10, dtype=torch.long, device=dev)]
        h = torch.cat([self.digit(dg) + self.field.weight[i] + self.pos(torch.arange(dg.shape[1], device=dev) + off)
                       for i, dg in enumerate(digs)], 1)
        for i in range(self.depth):
            h = self.blocks[0 if self.loop else i](h)
        return {"target": self.out(self.norm(h[:, -wo:]))}


@model("abacus_loop")
class AbacusLoop(Abacus):
    def __init__(self, in_widths, out_width, aux_widths, depth=6, **kw):
        super().__init__(in_widths, out_width, aux_widths, depth=depth, loop=True, **kw)


@model("longdiv")
class LongDiv(nn.Module):
    def __init__(self, in_widths, out_width, aux_widths, task="mod", d=32, hidden=128, hard=False, **kw):
        super().__init__()
        self.task, self.hard = task, hard
        self.er, self.en, self.ec = (nn.Linear(10, d, bias=False) for _ in range(3))   # digit distributions in
        self.eq = nn.Parameter(torch.randn(10, d) * 0.3)                                  # the ten trial digits
        self.mlp = nn.Sequential(nn.Linear(4 * d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 20))

    def cell(self, r, n, c):
        """One digit of 10R + v_t - q*N for all ten q: r, n, c are [B, 10, 10] distributions (q, value)."""
        z = torch.cat([self.er(r), self.en(n), self.eq.expand(r.shape[0], 10, -1), self.ec(c)], -1)
        o = self.mlp(z)
        return o[..., :10].softmax(-1), o[..., 10:].softmax(-1)

    def reduce(self, v, N, q=None):
        """v: [B, L, 10] digit distributions of the dividend (LSB-first); N: [B, w] digits; q: optional [B, w] digits of
        the quotient, which then replace the keep-the-largest-valid choice. Returns [B, w, 10]."""
        B, L, w = v.shape[0], v.shape[1], N.shape[1]
        n = F.one_hot(F.pad(N, (0, 1)), 10).float()                              # w+1 places, top one 0
        zero = F.one_hot(torch.zeros(B, dtype=torch.long, device=N.device), 10).float()
        R = zero[:, None].expand(B, w + 1, 10)
        for t in reversed(range(L)):                                              # dividend digits, MSB first
            Rp = torch.cat([v[:, t:t + 1], R[:, :w]], 1)                          # 10R + v_t
            c, D = zero[:, None].expand(B, 10, 10), []
            for j in range(w + 1):                                                # LSB-first within the trial
                d, c = self.cell(Rp[:, j, None].expand(B, 10, 10), n[:, j, None].expand(B, 10, 10), c)
                D.append(d)
            valid = c[..., 0]                                                     # P(no final borrow), per q
            higher = torch.cat([torch.flip(torch.cumprod(torch.flip(1 - valid, [1]), 1), [1])[:, 1:],
                                valid.new_ones(B, 1)], 1)                         # P(every larger q borrows)
            wq = valid * higher
            wq = wq / wq.sum(1, keepdim=True).clamp_min(1e-9)
            if q is not None:                                                     # the quotient digit, given
                wq = F.one_hot(q[:, t] if t < q.shape[1] else torch.zeros_like(q[:, 0]), 10).float()
            elif self.hard:                                                         # straight-through choice
                wq = wq + F.one_hot(wq.argmax(1), 10).float() - wq.detach()
            R = (wq[:, :, None, None] * torch.stack(D, 2)).sum(1)                 # keep the largest valid q
            if self.hard:                                                         # and snapped digits
                R = R + F.one_hot(R.argmax(-1), 10).float() - R.detach()
        return R[:, :w]

    def forward(self, inputs):
        R = self.reduce(F.one_hot(inputs["v"], 10).float(), inputs["N"], inputs.get("q"))
        return {"target": R.clamp_min(1e-9).log()}


@model("longdiv_st")
class LongDivST(LongDiv):
    def __init__(self, in_widths, out_width, aux_widths, **kw):
        super().__init__(in_widths, out_width, aux_widths, hard=True, **kw)


@model("colsum_longdiv")
class ColSumLongDiv(nn.Module):
    def __init__(self, in_widths, out_width, aux_widths, task="sqmod", d=64, hidden=128, ld_d=32, ld_hidden=128, **kw):
        super().__init__()
        self.square = ColSum(in_widths, out_width, aux_widths, task="sq", d=d, hidden=hidden)
        self.ld = LongDiv(in_widths, out_width, aux_widths, d=ld_d, hidden=ld_hidden)

    def forward(self, inputs):
        v = self.square({"x": inputs["x"]})["target"].softmax(-1)               # x^2 as digit distributions
        return {"target": self.ld.reduce(v, inputs["N"]).clamp_min(1e-9).log()}


def wider(task_name, size):
    """Sizes past the trained width: digits k+1, k+2, k+4; bits: the first sizes with one and two more digits."""
    task = TASKS[task_name]
    if task.kind == "digits":                       # products of 10+ digit numbers overflow int64
        return [s for s in (size + 1, size + 2, size + 4) if task_name == "add" or s <= 9]
    w = ndig((1 << size) - 1)
    return [next(b for b in range(size + 1, 25) if ndig((1 << b) - 1) == w + k) for k in (1, 2)]


def _score(net, task, size, n=2000, seed=10_000):
    b = task.sample(n, size, np.random.default_rng(seed))
    x, y, _ = _tensors(b, next(net.parameters()).device)
    with torch.no_grad():
        return b, (net(x)["target"].argmax(-1) == y).all(1).cpu().numpy()


def _lengths(inputs):
    """Significant digits of every input field, as one bucket key per example."""
    lens = [np.where(inputs[k].any(1), inputs[k].shape[1] - np.argmax(inputs[k][:, ::-1] > 0, 1), 0)
            for k in sorted(inputs)]
    return np.stack(lens, 1) @ (100 ** np.arange(len(lens)))


def matched(net, task, size, train_keys, n=10_000, seed=40_000):
    """Exact on seen vs unseen test inputs within the same operand lengths, weighted by the unseen ones: a table
    shows unseen << seen, a rule unseen ~ seen."""
    b, ok = _score(net, task, size, n, seed)
    seen, lens = np.isin(ext_ladder.keys(b.inputs), train_keys), _lengths(b.inputs)
    ws = us = sw = 0.0
    for L in np.unique(lens):
        m = lens == L
        if (m & seen).sum() >= 5 and (m & ~seen).sum() >= 5:
            wt = (m & ~seen).sum()
            ws, us, sw = ws + wt * ok[m & seen].mean(), us + wt * ok[m & ~seen].mean(), sw + wt
    return None if not sw else {"seen": round(ws / sw, 4), "unseen": round(us / sw, 4), "n_unseen": int(sw)}


def wide(net, task, size, sizes):
    """Exact at wider sizes: on all test inputs, and on those with a digit past the trained width of its field."""
    trained, _, _ = task.widths(size)
    out = {}
    for s in sizes:
        b, ok = _score(net, task, s)
        past = np.zeros(len(ok), bool)
        for k, w in trained.items():
            past |= b.inputs[k][:, w:].any(1)
        out[str(s)] = {"all": round(float(ok.mean()), 4), "wider": round(float(ok[past].mean()), 4) if past.any()
                       else None, "n_wider": int(past.sum())}
    return out


def cell(task_name, size, model="colsum", log=RESULTS, train_on=None, **kw):
    """ext_ladder.cell plus, for the width-free models, `wide` (exact at wider sizes) and `matched` (seen vs unseen at
    equal operand lengths). `train_on` trains on another task with the same inputs (e.g. "modmix") and scores
    `task_name`."""
    if model in FOR:
        kw.setdefault("task", task_name)
    r, net, train_keys = ext_ladder.cell(train_on or task_name, size, model=model, log=None, return_net=True, **kw)
    task = TASKS[task_name]
    if train_on:
        exact, per_digit, split = ext_ladder.evaluate_split(net, task, size, train_keys, holdout=kw.get("holdout", 0))
        r.update(task=task_name, train_task=train_on, exact_train_dist=r["exact"], exact=round(exact, 4),
                 per_digit=[round(v, 3) for v in per_digit], **split)
    if model in FOR:
        r["wide"] = wide(net, task, size, wider(task_name, size))
        r["matched"] = matched(net, task, size, train_keys)
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(r) + "\n")
    return r


def show(r):
    out = line(r)
    if r.get("matched"):
        out += f"  matched seen/unseen {r['matched']['seen']:.2f}/{r['matched']['unseen']:.2f}"
    if r.get("wide"):
        out += "  wide " + " ".join(f"@{s}:{v['all']:.2f}" + ("" if v["wider"] is None else f"/{v['wider']:.2f}")
                                    for s, v in r["wide"].items())
    return out


def _job(kw):
    return cell(**kw)


def climb(task_name, model="colsum", data="fresh", seeds=2, sizes=None, patience=2, workers=1, threads=2, log=RESULTS,
          verbose=True, after=None, **cell_kw):
    """ext_ladder.climb with this module's cell."""
    pool = 0 if data == "fresh" else int(data.removeprefix("pool"))
    sizes = sizes or SIZES[TASKS[task_name].kind]
    out, misses = [], 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) if workers > 1 else ext_ladder._Serial() as ex:
        for size in sizes:
            jobs = [dict(task_name=task_name, size=size, model=model, seed=s, pool=pool, threads=threads, log=log,
                         **cell_kw) for s in range(seeds)]
            rs = list(ex.map(_job, jobs))
            out += rs
            if verbose:
                for r in rs:
                    print(show(r), flush=True)
            if after:
                after(rs)
            misses = 0 if max(r["exact"] for r in rs) >= PASS else misses + 1
            if misses >= patience:
                break
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("cell", "climb"):
        q = sub.add_parser(name)
        q.add_argument("task", choices=sorted(TASKS))
        q.add_argument("--model", default="colsum", choices=sorted(MODELS))
        q.add_argument("--steps", type=int, default=3000)
        q.add_argument("--lr", type=float, default=None)
        q.add_argument("--threads", type=int, default=2)
        q.add_argument("--holdout", type=int, default=0)
        q.add_argument("--train_on", default=None, help="train on this task, score the given one (e.g. modmix)")
        q.add_argument("--tag", default="")
        if name == "cell":
            q.add_argument("--size", type=int, required=True)
            q.add_argument("--seed", type=int, default=0)
            q.add_argument("--pool", type=int, default=0)
        else:
            q.add_argument("--data", default="fresh")
            q.add_argument("--seeds", type=int, default=2)
            q.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    common = dict(model=a.model, steps=a.steps, lr=a.lr, threads=a.threads, holdout=a.holdout, tag=a.tag,
                  train_on=a.train_on)
    if a.cmd == "cell":
        print(show(cell(a.task, a.size, seed=a.seed, pool=a.pool, **common)), flush=True)
    else:
        climb(a.task, data=a.data, seeds=a.seeds, workers=a.workers, **common)


if __name__ == "__main__":
    main()
