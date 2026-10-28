"""Stage dynamics without the evaluator: the submission's model, loss and optimizer on H1's single-step rows alone.

    python mini.py <submission.py> '<json grid>'     (from the one-layer-deeper repo root, H1 generated)

The grid is a list of {"B0": float or null, "seed": int, "set": {CONSTANT: source}}. Each run follows the evaluator's
update (zero_grad, forward, token_training_loss, backward, clip_grad_norm_(1), step) on a random `subset` share of the
1,000 T=1 training rows, redrawn every 8 updates as the evaluator's 100,000-row batches with batch reuse would be.
Only T=1 rows enter the loss, so the other rows of a real batch would change nothing but the reported accuracy.
Prints one JSON line per run: exact share on all 1,000 T=1 rows, the update at which they first were all exact, B, w.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time

import torch

from benchmark import OptimizerSpec, TokenLossBatch
from benchmark.batches import prepare_batch
from benchmark.manifest import load_manifest
from benchmark.runner import _make_model_spec
from data.squaring_mod import collate_squaring_mod

DATA = "data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2/train.jsonl"
src0 = open(sys.argv[1]).read()
grid = json.loads(sys.argv[2])
rows = [r for r in map(json.loads, open(DATA)) if r["time_steps"] == 1]
ids, tgt, mask, tpos = prepare_batch(collate_squaring_mod(rows), torch.device("cpu"))
spec = _make_model_spec(load_manifest("benchmark/manifests/h100_hard_h1.json"))


def load(sets: dict):
    src = src0
    for key, value in {"LOG_EVERY": "0", **sets}.items():
        src, n = re.subn(rf"^{key} = .*$", f"{key} = {value}", src, count=1, flags=re.M)
        assert n == 1, key
    path = f"/tmp/mini_{abs(hash(src))}.py"
    open(path, "w").write(src)
    spec_ = importlib.util.spec_from_file_location("sub_mini", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def exact(model) -> float:
    model.eval()
    with torch.no_grad():
        logits, _ = model(ids, attention_mask=mask)
    model.train()
    tl = logits[torch.arange(ids.shape[0])[:, None], tpos.clamp_min(0)]
    valid = tgt != -100
    return float(((tl.argmax(-1) == tgt) | ~valid).all(1).float().mean())


for job in grid:
    sub = load(job.get("set", {}))
    torch.manual_seed(job["seed"])
    model = sub.SUBMISSION.build_model(spec).to(dtype=torch.float32)
    if job.get("B0") is not None:
        with torch.no_grad():
            model.base.fill_(job["B0"])
    opt = sub.SUBMISSION.build_optimizer(model, OptimizerSpec(3600.0, "cpu")).optimizer
    gen = torch.Generator().manual_seed(1000 + job["seed"])
    first, t0, updates, subset = None, time.time(), job.get("updates", 400), job.get("subset", 0.98)
    for u in range(1, updates + 1):
        if (u - 1) % 8 == 0:
            take = torch.nonzero(torch.rand(len(rows), generator=gen) < subset).flatten()
        opt.zero_grad(set_to_none=True)
        logits, aux = model(ids[take], attention_mask=mask[take])
        tp = tpos[take]
        tl = logits[torch.arange(len(take))[:, None], tp.clamp_min(0)].float()
        loss = sub.token_training_loss(TokenLossBatch(logits=tl, labels=tgt[take], valid_mask=tgt[take] != -100,
                                                      target_positions=tp, auxiliary=aux))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first is None and u % 4 == 0 and exact(model) == 1.0:
            first = u
    print(json.dumps({**job, "exact_t1": exact(model), "first_exact_update": first,
                      "B": float(model.base), "w": model.w.tolist(), "sec": round(time.time() - t0, 1)}), flush=True)
