"""Ladder: where generic networks stop learning each piece of x^(2^T) mod N.

For one task x model x data regime, climb the sizes until the best seed misses 99% exact at two sizes in a row. Each
cell records exact match, per-digit accuracy (lo->hi), a per-digit majority-class baseline, a short learning curve,
the training-pool accuracy for --pool runs, exact match one size up (ood) when the digit widths allow it, and exact
match on the test inputs that never occurred in training (the line between a rule and a table). `--holdout k` keeps a
1/k share of the moduli out of training and scores them separately; `by_quotient` splits unseen inputs by the
quotient value // N.

    python -m microlab.ext_ladder cell sqmod --size 8 --model xf --steps 3000      # one cell, same numbers as microlab.run
    python -m microlab.ext_ladder climb sqmod --model xf --data fresh --seeds 2     # one ladder
    python -m microlab.ext_ladder_report --tags S3k,S12k                            # the LADDER.md tables

Training mirrors microlab.run exactly (init, AdamW, OneCycle, clipping, the same 2,000 evaluation examples), so a cell
without --holdout reproduces the same microlab.run command; the extra checkpoints evaluate on separate examples and do
not touch the training stream. Results go to microlab/ladder_results.jsonl, not microlab/results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from .models import MODELS
from .run import _tensors, evaluate
from .tasks import TASKS, Batch

RESULTS = os.path.join(os.path.dirname(__file__), "ladder_results.jsonl")
SIZES = {"digits": list(range(2, 9)), "bits": list(range(6, 17))}
PASS = 0.99


def baseline(y):
    """Per-digit accuracy of always guessing that digit's most common value."""
    return [float(np.bincount(y[:, i], minlength=10).max() / len(y)) for i in range(y.shape[1])]


def keys(inputs):
    """One int64 per example from all of its input digits (at most 16 digits here), to tell seen from unseen."""
    cols = np.concatenate([inputs[k] for k in sorted(inputs)], 1)
    return cols @ (10 ** np.arange(cols.shape[1], dtype=np.int64))


def moduli(inputs):
    return inputs["N"] @ (10 ** np.arange(inputs["N"].shape[1]))


def held_out(inputs, k):
    """Moduli kept out of training by `holdout=k`: N with (N // 2) % k == 0, a quarter of the moduli for k = 4."""
    return (moduli(inputs) // 2) % k == 0


def sample_kept(task, n, size, rng, holdout=0):
    """task.sample, minus the held-out moduli when holdout > 0."""
    if not holdout:
        return task.sample(n, size, rng)
    parts, got = [], 0
    while got < n:
        b = task.sample(2 * n, size, rng)
        idx = np.flatnonzero(~held_out(b.inputs, holdout))
        parts.append(Batch({k: v[idx] for k, v in b.inputs.items()}, b.target[idx]))
        got += len(idx)
    return Batch({k: np.concatenate([p.inputs[k] for p in parts])[:n] for k in parts[0].inputs},
                 np.concatenate([p.target for p in parts])[:n])


def evaluate_split(net, task, size, train_keys, n=2000, seed=10_000, holdout=0):
    """microlab.run.evaluate on the same examples, plus exact match split by whether the input was trained on (and,
    with holdout, on the examples whose modulus never occurred in training)."""
    b = task.sample(n, size, np.random.default_rng(seed))
    x, y, _ = _tensors(b, next(net.parameters()).device)
    with torch.no_grad():
        ok = (net(x)["target"].argmax(-1) == y).cpu()
    row = ok.all(1).numpy()
    seen = np.isin(keys(b.inputs), train_keys)
    part = lambda m: None if not m.any() else round(float(row[m].mean()), 4)
    unseen_digits = None if seen.all() else [round(v, 3) for v in ok[torch.from_numpy(~seen)].float().mean(0).tolist()]
    split = {"seen_frac": round(float(seen.mean()), 4), "exact_seen": part(seen), "exact_unseen": part(~seen),
             "per_digit_unseen": unseen_digits, "baseline": [round(v, 3) for v in baseline(b.target)]}
    if holdout:
        ho = held_out(b.inputs, holdout)
        split |= {"heldout_frac": round(float(ho.mean()), 4), "exact_heldout_moduli": part(ho),
                  "exact_unseen_kept_moduli": part(~seen & ~ho),
                  "per_digit_heldout": [round(v, 3) for v in ok[torch.from_numpy(ho)].float().mean(0).tolist()]}
    return ok.all(1).float().mean().item(), ok.float().mean(0).tolist(), split


def cell(task_name, size, model="xf", seed=0, steps=3000, batch=256, lr=None, pool=0, checkpoints=4, device="cpu",
         threads=None, tag="", log=RESULTS, return_net=False, holdout=0, **model_kw):
    if threads:
        torch.set_num_threads(threads)
    task = TASKS[task_name]
    in_w, out_w, _ = task.widths(size)
    lr = lr or (3e-3 if model == "mlp" else 5e-4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = MODELS[model](in_w, out_w, {}, **model_kw).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    fixed = sample_kept(task, pool, size, rng, holdout) if pool else None
    marks = {round(steps * (i + 1) / checkpoints) for i in range(checkpoints - 1)}
    curve, losses, seen = [], [], [keys(fixed.inputs)] if fixed is not None else []
    t0 = time.time()
    for step in range(steps):
        if fixed is not None:
            idx = rng.integers(0, pool, batch)
            b = Batch({k: v[idx] for k, v in fixed.inputs.items()}, fixed.target[idx])
        else:
            b = sample_kept(task, batch, size, rng, holdout)
            seen.append(keys(b.inputs))
        x, y, _ = _tensors(b, device)
        loss = F.cross_entropy(net(x)["target"].flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step + 1 in marks:
            curve.append([step + 1, round(evaluate(net, task, size, n=500, seed=20_000 + step)[0], 4)])
    exact, per_digit, split = evaluate_split(net, task, size, np.unique(np.concatenate(seen)), holdout=holdout)
    curve.append([steps, round(exact, 4)])
    train_exact = None
    if fixed is not None:
        x, y, _ = _tensors(fixed, device)
        with torch.no_grad():
            train_exact = (net(x)["target"].argmax(-1) == y).all(1).float().mean().item()
    ood_size = ood_exact = None
    if task.kind == "bits" and size + 1 <= 24 and task.widths(size + 1) == task.widths(size):
        ood_size, ood_exact = size + 1, round(evaluate(net, task, size + 1)[0], 4)
    r = {"task": task_name, "size": size, "model": model, "data": f"pool{pool}" if pool else "fresh", "pool": pool,
         "seed": seed, "steps": steps, "batch": batch, "lr": lr, "model_kw": model_kw, "holdout": holdout,
         "params": sum(p.numel() for p in net.parameters()), "exact": round(exact, 4),
         "per_digit": [round(v, 3) for v in per_digit], **split,
         "train_exact": None if train_exact is None else round(train_exact, 4),
         "ood_size": ood_size, "ood_exact": ood_exact, "curve": curve,
         "final_loss": round(float(np.mean(losses[-100:])), 4), "seconds": round(time.time() - t0, 1),
         "device": str(device), "threads": torch.get_num_threads(), "tag": tag}
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(r) + "\n")
    return (r, net, np.unique(np.concatenate(seen))) if return_net else r


def by_quotient(task_name="sqmod", size=7, model="xf", pool=1000, steps=3000, seed=0, n=20_000, threads=2, **kw):
    """Train one cell, then score fresh examples split by seen/unseen and by the quotient q = value // N (q = 0: the
    remainder changes nothing). Tests whether what a network gets right on unseen inputs is only the q = 0 part."""
    r, net, train_keys = cell(task_name, size, model=model, pool=pool, steps=steps, seed=seed, threads=threads,
                              log=None, return_net=True, **kw)
    task = TASKS[task_name]
    b = task.sample(n, size, np.random.default_rng(30_000))
    x, y, _ = _tensors(b, next(net.parameters()).device)
    with torch.no_grad():
        ok = (net(x)["target"].argmax(-1) == y).all(1).cpu().numpy()
    number = lambda d: d @ (10 ** np.arange(d.shape[1]))
    v = number(b.inputs["x"]) ** 2 if "x" in b.inputs else number(b.inputs["v"])
    q = v // moduli(b.inputs)
    seen = np.isin(keys(b.inputs), train_keys)
    out = {"cell": line(r)}
    for name, m in (("seen", seen), ("unseen", ~seen)):
        for qn, qm in (("q=0", q == 0), ("q=1", q == 1), ("q>=2", q >= 2)):
            sel = m & qm
            out[f"{name} {qn}"] = (int(sel.sum()), round(float(ok[sel].mean()), 3) if sel.any() else None)
    return out


def line(r):
    pd = " ".join(f"{v:.2f}" for v in r["per_digit"])
    extra = "" if r["ood_exact"] is None else f"  ood@{r['ood_size']} {r['ood_exact']:.3f}"
    extra += "" if r["train_exact"] is None else f"  train {r['train_exact']:.3f}"
    extra += f"  seen {r['seen_frac']:.2f}" + ("" if r["exact_unseen"] is None else f" unseen {r['exact_unseen']:.3f}")
    if r.get("exact_heldout_moduli") is not None:
        extra += f"  held-out moduli {r['exact_heldout_moduli']:.3f}"
    curve = " ".join(f"{e:.2f}" for _, e in r["curve"])
    return (f"{r['task']}@{r['size']} {r['model']} {r['data']} seed{r['seed']}: exact {r['exact']:.3f}{extra}  "
            f"digits(lo->hi) [{pd}]  curve [{curve}]  {r['seconds']}s")


def _cell_job(kw):
    return cell(**kw)


def climb(task_name, model="xf", data="fresh", seeds=2, sizes=None, patience=2, workers=1, threads=2, log=RESULTS,
          verbose=True, after=None, **cell_kw):
    """Climb sizes; stop once the best seed misses PASS at `patience` sizes in a row. Returns every cell.
    Seeds run in `workers` spawned processes; `after(cells)` is called once per size (e.g. to commit a Volume)."""
    task = TASKS[task_name]
    pool = 0 if data == "fresh" else int(data.removeprefix("pool"))
    sizes = sizes or SIZES[task.kind]
    out, misses = [], 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) if workers > 1 else _Serial() as ex:
        for size in sizes:
            jobs = [dict(task_name=task_name, size=size, model=model, seed=s, pool=pool, threads=threads, log=log,
                         **cell_kw) for s in range(seeds)]
            rs = list(ex.map(_cell_job, jobs))
            out += rs
            if verbose:
                for r in rs:
                    print(line(r), flush=True)
            if after:
                after(rs)
            misses = 0 if max(r["exact"] for r in rs) >= PASS else misses + 1
            if misses >= patience:
                break
    return out


class _Serial:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, it):
        return map(fn, it)


def load(path=RESULTS, tag=None):
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    return [r for r in rows if tag is None or r["tag"] == tag]


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("cell", "climb"):
        q = sub.add_parser(name)
        q.add_argument("task", choices=sorted(TASKS))
        q.add_argument("--model", default="xf", choices=sorted(MODELS))
        q.add_argument("--steps", type=int, default=3000)
        q.add_argument("--batch", type=int, default=256)
        q.add_argument("--lr", type=float, default=None)
        q.add_argument("--depth", type=int, default=None)
        q.add_argument("--d", type=int, default=None)
        q.add_argument("--hidden", type=int, default=None)
        q.add_argument("--threads", type=int, default=2)
        q.add_argument("--device", default="cpu")
        q.add_argument("--tag", default="")
        q.add_argument("--holdout", type=int, default=0, help="keep 1/k of the moduli out of training")
        if name == "cell":
            q.add_argument("--size", type=int, required=True)
            q.add_argument("--seed", type=int, default=0)
            q.add_argument("--pool", type=int, default=0)
        else:
            q.add_argument("--data", default="fresh", help="fresh or poolN")
            q.add_argument("--seeds", type=int, default=2)
            q.add_argument("--sizes", default="", help="comma list; default digits 2-8 or bits 6-16")
            q.add_argument("--workers", type=int, default=1, help="seeds trained in parallel processes")
    a = p.parse_args()
    kw = {k: v for k, v in (("depth", a.depth), ("d", a.d), ("hidden", a.hidden)) if v is not None}
    common = dict(model=a.model, steps=a.steps, batch=a.batch, lr=a.lr, threads=a.threads, device=a.device, tag=a.tag,
                  holdout=a.holdout, **kw)
    if a.cmd == "cell":
        print(line(cell(a.task, a.size, seed=a.seed, pool=a.pool, **common)), flush=True)
    else:
        sizes = [int(s) for s in a.sizes.split(",") if s] or None
        climb(a.task, data=a.data, seeds=a.seeds, sizes=sizes, workers=a.workers, **common)


if __name__ == "__main__":
    main()
