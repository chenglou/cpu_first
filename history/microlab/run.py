"""Run one small hypothesis and get a standard one-line answer.

    python -m microlab.run sqmod --size 8 --model xf --aux pre --steps 3000 --seeds 2
    python -m microlab.run add --size 5 --model xf --ood 3

Python:  from microlab.run import run;  run("sqmod", 8, model="xf", aux=("pre",))

Report per seed: exact match (every target digit right, leading zeros included) at the training size, per-digit
accuracy from the lowest digit up (shows *which* digits learned), exact match at size + ood when that size exists,
and wall time. Every run is appended to microlab/results.jsonl. Data is fresh each step unless --pool N fixes a
training set of N examples (to mimic the competition's data budget).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .models import MODELS
from .tasks import TASKS, Batch

RESULTS = os.path.join(os.path.dirname(__file__), "results.jsonl")


def _tensors(batch: Batch, dev):
    t = lambda a: torch.from_numpy(a).long().to(dev)
    return {k: t(v) for k, v in batch.inputs.items()}, t(batch.target), {k: t(v) for k, v in batch.aux.items()}


def evaluate(net, task, size, n=2000, seed=10_000):
    rng = np.random.default_rng(seed)
    b = task.sample(n, size, rng)
    dev = next(net.parameters()).device
    x, y, _ = _tensors(b, dev)
    with torch.no_grad():
        pred = net(x)["target"].argmax(-1)
    ok = (pred == y).cpu()
    return ok.all(1).float().mean().item(), ok.float().mean(0).tolist()


def run(task_name, size, model="xf", aux=(), aux_weight=1.0, steps=3000, batch=256, lr=None, seeds=1, pool=0,
        ood=1, device="cpu", tag="", verbose=True, **model_kw):
    task = TASKS[task_name]
    in_w, out_w, aux_w = task.widths(size)
    use_aux = {k: aux_w[k] for k in aux}
    lr = lr or (3e-3 if model == "mlp" else 5e-4)
    results = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = MODELS[model](in_w, out_w, use_aux, **model_kw).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
        fixed = task.sample(pool, size, rng) if pool else None
        t0 = time.time()
        for step in range(steps):
            if fixed is not None:
                idx = rng.integers(0, pool, batch)
                b = Batch({k: v[idx] for k, v in fixed.inputs.items()}, fixed.target[idx],
                          {k: v[idx] for k, v in fixed.aux.items()})
            else:
                b = task.sample(batch, size, rng)
            x, y, a = _tensors(b, device)
            out = net(x)
            loss = F.cross_entropy(out["target"].flatten(0, 1), y.flatten())
            for k in use_aux:
                loss = loss + aux_weight * F.cross_entropy(out[k].flatten(0, 1), a[k].flatten())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        exact, per_digit = evaluate(net, task, size)
        ood_exact = None
        try:
            if ood and task.widths(size + ood) == task.widths(size):
                ood_exact = evaluate(net, task, size + ood)[0]
        except Exception:
            ood_exact = None
        r = {"task": task_name, "size": size, "model": model, "aux": list(aux), "steps": steps, "pool": pool,
             "seed": seed, "exact": round(exact, 4), "per_digit": [round(v, 3) for v in per_digit],
             "ood_size": size + ood if ood_exact is not None else None,
             "ood_exact": None if ood_exact is None else round(ood_exact, 4),
             "seconds": round(time.time() - t0, 1), "tag": tag, "model_kw": model_kw}
        results.append(r)
        with open(RESULTS, "a") as f:
            f.write(json.dumps(r) + "\n")
        if verbose:
            pd = " ".join(f"{v:.2f}" for v in r["per_digit"])
            ood_s = "" if ood_exact is None else f"  ood@{size + ood} {ood_exact:.3f}"
            print(f"{task_name}@{size} {model}{'+' + ','.join(aux) if aux else ''} seed{seed}: exact {exact:.3f}"
                  f"{ood_s}  digits(lo->hi) [{pd}]  {r['seconds']}s", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=sorted(TASKS))
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--model", default="xf", choices=sorted(MODELS))
    p.add_argument("--aux", default="", help="comma list of auxiliary targets, e.g. pre,q")
    p.add_argument("--aux_weight", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--pool", type=int, default=0)
    p.add_argument("--ood", type=int, default=1)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="cpu", help="cpu or mps")
    p.add_argument("--threads", type=int, default=4, help="CPU threads per run (parallel runs share cores)")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    kw = {k: v for k, v in (("depth", a.depth), ("d", a.d)) if v is not None}
    run(a.task, a.size, model=a.model, aux=tuple(x for x in a.aux.split(",") if x), aux_weight=a.aux_weight,
        steps=a.steps, batch=a.batch, lr=a.lr, seeds=a.seeds, pool=a.pool, ood=a.ood, tag=a.tag,
        device=a.device, **kw)


if __name__ == "__main__":
    main()
