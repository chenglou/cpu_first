"""Small-batch dynamics without the evaluator: the submission's model, loss and optimizer on the evaluator's own batches.

    python mini_small.py <submission.py> '<json grid>'     (from the one-layer-deeper repo root, H1 generated)
    python mini_small.py --verify-order                     (replicated batch order == the real DataLoader's)

Grid entries: {"name", "seed": runner seed (init and data order), "batch": batch size, "epochs", "B0": float or null,
"set": {CONSTANT: python source}}. The batches are the evaluator's: torch's RandomSampler + BatchSampler(drop_last)
on a generator seeded by `seed`, with the DataLoader iterator's base-seed draw each epoch. Each batch is used 8 times
(zero_grad, forward, token_training_loss, backward, clip_grad_norm_(1), step), as the evaluator does when
should_reuse_batch returns True. Only a batch's single-step rows (plus one T=4 filler row, so no batch is empty) are
run: the loss uses nothing else, so the parameter trajectory is the runner's.
Prints one JSON line per run: per-epoch trace, exact share on all 1,000 T=1 rows, certified T on both depth profiles.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from dataclasses import replace

import torch
from torch.utils.data import BatchSampler, RandomSampler

from benchmark import OptimizerSpec, TokenLossBatch
from benchmark.batches import prepare_batch
from benchmark.manifest import load_manifest
from benchmark.runner import _make_model_spec
from data.squaring_mod import collate_squaring_mod

ROOT = "data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
MANIFEST = "benchmark/manifests/h100_hard_h1.json"
LADDER = (1, 2, 4, 8, 16, 32, 64)


def sampler_batches(n: int, batch: int, seed: int, epochs: int):
    g = torch.Generator().manual_seed(seed)
    sampler = BatchSampler(RandomSampler(range(n), generator=g), batch, drop_last=True)
    for _ in range(epochs):
        it = iter(sampler)
        torch.empty((), dtype=torch.int64).random_(generator=g)   # the DataLoader iterator's base seed
        for idx in it:
            yield torch.tensor(idx)


if sys.argv[1] == "--verify-order":
    from data import make_dataloaders
    man = load_manifest(MANIFEST)
    recs = [json.loads(l) for l in open(f"{ROOT}/train.jsonl")]
    for bs, workers in ((512, 0), (512, 2), (4096, 0)):
        loaders = make_dataloaders(replace(man.data, seed=74, batch_size=bs, num_workers=workers,
                                           pin_memory=False), device=torch.device("cpu"))
        it, mine = iter(loaders["train"]), sampler_batches(len(recs), bs, 74, 3)
        nb = len(recs) // bs
        ok = True
        for b in range(2 * nb + 3):   # into the third epoch
            try:
                real = next(it)
            except StopIteration:
                it = iter(loaders["train"])
                real = next(it)
            idx = next(mine)
            if b % 37 == 0 or b in (nb - 1, nb, 2 * nb):
                ref = collate_squaring_mod([{"input_ids": recs[i]["input_ids"], "labels": recs[i]["labels"]}
                                            for i in idx.tolist()])
                ok &= bool(torch.equal(ref["input_ids"], real["input_ids"]))
        print(f"batch {bs} workers {workers}: replicated order matches the DataLoader: {ok}", flush=True)
    sys.exit(0)

src0 = open(sys.argv[1]).read()
grid = json.loads(sys.argv[2])
recs = [(r["time_steps"], r["input_ids"], r["labels"], r["x"] ** 2 // r["modulus"])
        for r in map(json.loads, open(f"{ROOT}/train.jsonl"))]
t1 = [i for i, r in enumerate(recs) if r[0] == 1]
filler = next(i for i, r in enumerate(recs) if r[0] == 4)
pos = {i: k for k, i in enumerate(t1)}
rows = [{"input_ids": recs[i][1], "labels": recs[i][2]} for i in t1 + [filler]]
ids, tgt, mask, tpos = prepare_batch(collate_squaring_mod(rows), torch.device("cpu"))
spec = _make_model_spec(load_manifest(MANIFEST))
depth = {}
for prefix in ("depth_t_", "depth_ood_n_t_"):
    for T in LADDER:
        dr = [json.loads(l) for l in open(f"{ROOT}/{prefix}{T}.jsonl")]
        depth[(prefix, T)] = prepare_batch(collate_squaring_mod(
            [{"input_ids": r["input_ids"], "labels": r["labels"]} for r in dr]), torch.device("cpu"))


def load(sets: dict):
    src = src0
    for key, value in {"LOG_EVERY": "0", **sets}.items():
        src, count = re.subn(rf"^{key} = .*$", f"{key} = {value}", src, count=1, flags=re.M)
        assert count == 1, key
    path = f"/tmp/mini_{abs(hash(src))}.py"
    open(path, "w").write(src)
    spec_ = importlib.util.spec_from_file_location(f"sub_mini_{abs(hash(src))}", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def exact(model, data) -> float:
    i, t, m, tp = data
    model.eval()
    with torch.no_grad():
        logits, _ = model(i, attention_mask=m)
    model.train()
    tl = logits[torch.arange(i.shape[0])[:, None], tp.clamp_min(0)]
    valid = t != -100
    return float(((tl.argmax(-1) == t) | ~valid).all(1).float().mean())


def certified(model, prefix: str):
    best = None
    for T in LADDER:
        if exact(model, depth[(prefix, T)]) < 1.0:
            break
        best = T
    return best


t1data = (ids[:-1], tgt[:-1], mask[:-1], tpos[:-1])
for job in grid:
    seed, batch, epochs = job.get("seed", 74), job.get("batch", 512), job.get("epochs", 20)
    sub = load({"BATCH": str(batch), **job.get("set", {})})
    torch.manual_seed(seed)
    model = sub.SUBMISSION.build_model(spec).to(dtype=torch.float32)
    if job.get("B0") is not None:
        with torch.no_grad():
            model.base.fill_(job["B0"])
    opt = sub.SUBMISSION.build_optimizer(model, OptimizerSpec(3600.0, "cpu")).optimizer
    nb = len(recs) // batch
    trace, first, cert, t0, u = [], None, None, time.time(), 0
    admitted, wrong = 0, 0
    for b, idx in enumerate(sampler_batches(len(recs), batch, seed, epochs)):
        take = torch.tensor([pos[i] for i in idx.tolist() if i in pos] + [len(t1)])
        for _ in range(8):
            u += 1
            opt.zero_grad(set_to_none=True)
            logits, aux = model(ids[take], attention_mask=mask[take])
            tp = tpos[take]
            tl = logits[torch.arange(len(take))[:, None], tp.clamp_min(0)].float()
            loss = sub.token_training_loss(TokenLossBatch(logits=tl, labels=tgt[take], valid_mask=tgt[take] != -100,
                                                          target_positions=tp, auxiliary=aux))
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"non-finite loss at update {u}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        admitted += model.stats.get("m", 0)
        if "rows" in model.stats:   # admitted rows whose unfolding k is not the true wrap count floor(x^2 / N)
            ktrue = torch.tensor([recs[t1[int(take[r])]][3] if int(take[r]) < len(t1) else -1
                                  for r in model.stats["rows"].tolist()])
            wrong += int((model.stats["k"].long() != ktrue).sum())
        if job.get("debug") and model.stats.get("m", 0) and b < job["debug"]:
            print(f"batch {b}: rows {model.stats['m']}/{model.stats['count']} s={model.stats.get('s'):.2e} "
                  f"B={float(model.base.detach()):.6f} w={[f'{float(v):.4g}' for v in model.w.detach()]} "
                  f"carried={model.memory['cnt']:.2f}", flush=True)
        if (b + 1) % nb == 0:
            ex = exact(model, t1data)
            if first is None and ex == 1.0:
                first = (b + 1) // nb
            mem = model.memory
            both = (certified(model, "depth_t_"), certified(model, "depth_ood_n_t_")) if ex == 1.0 else (None, None)
            if cert is None and both == (64, 64):
                cert = (b + 1) // nb
            trace.append({"epoch": (b + 1) // nb, "exact_t1": round(ex, 4), "admitted": admitted, "wrong_k": wrong,
                          "cert": both, "B": float(model.base.detach()), "w": [float(v) for v in model.w.detach()],
                          "carried": round(mem["cnt"], 2), "s": model.stats.get("s")})
            admitted, wrong = 0, 0
            if job.get("verbose"):
                print(json.dumps(trace[-1]), flush=True)
            if cert is not None and (b + 1) // nb >= cert + job.get("extra_epochs", 2):
                break
    out = {**job, "exact_t1": exact(model, t1data), "first_exact_epoch": first, "cert_epoch": cert,
           "max_T": certified(model, "depth_t_"), "ood_n_max_T": certified(model, "depth_ood_n_t_"),
           "B": float(model.base.detach()), "w": model.w.tolist(), "updates": u, "sec": round(time.time() - t0, 1),
           "trace": trace}
    print(json.dumps(out), flush=True)
