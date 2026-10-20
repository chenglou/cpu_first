"""Check UnfoldS's closed-form residual Jacobian and its scaled-gradient channel against autograd.

    python check_grad_small.py <submission.py>      (from the one-layer-deeper repo root, H1 generated)

On one real training batch (8,192 rows, seed 74), at three generic parameter points, with every single-step row
admitted (C_SEED raised, so every row is a seed, unfolded with k = 0): the evaluator-style backward of the
submission's loss, times nu, must equal J^T e from the loss's own stats, and each row of J must match autograd. Then
with the shipped C_SEED (seeds k = 0, other rows their own unfolding) the per-row check covers both kinds of rows; with
the optimizer's initial (empty) carried information only seed rows may enter the loss.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import replace

import torch

from benchmark import OptimizerSpec
from benchmark.batches import prepare_batch
from benchmark.manifest import load_manifest
from benchmark.runner import _make_model_spec
from data import make_dataloaders

src0 = open(sys.argv[1]).read()


def load(src: str, tag: str):
    path = f"/tmp/check_{tag}.py"
    open(path, "w").write(src)
    spec_ = importlib.util.spec_from_file_location(f"sub_{tag}", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


man = load_manifest("benchmark/manifests/h100_hard_h1.json")
torch.manual_seed(74)
loaders = make_dataloaders(replace(man.data, seed=74, batch_size=8192, num_workers=0), device=torch.device("cpu"))
batch = next(iter(loaders["train"]))
ids, tgt, mask, tpos = prepare_batch(batch, torch.device("cpu"))
f64 = torch.float64

for tag, src in (("all", re.sub(r"^C_SEED = .*$", "C_SEED = 1e30", src0, flags=re.M)), ("shipped", src0)):
    sub = load(src, tag)
    model = sub.SUBMISSION.build_model(_make_model_spec(man)).to(dtype=torch.float32)
    sub.SUBMISSION.build_optimizer(model, OptimizerSpec(3600.0, "cpu"))   # publishes the (empty) carried information
    for trial in range(3):
        with torch.no_grad():  # a generic point, so every column is live
            model.base.fill_(9.3 + 0.4 * trial)
            model.w.copy_(torch.tensor([3.0, -0.2, 0.97 + 0.01 * trial]))
        model.train()
        model.zero_grad()
        logits, aux = model(ids, attention_mask=mask)
        tl = logits[torch.arange(ids.shape[0])[:, None], tpos.clamp_min(0)].float()
        valid = tgt != -100
        loss = sub.token_training_loss(sub.TokenLossBatch(logits=tl, labels=tgt, valid_mask=valid,
                                                          target_positions=tpos, auxiliary=aux))
        loss.backward()
        st = dict(model.stats)
        g_auto = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().to(f64)
                            for p in model.parameters()]) * st.get("nu", 1.0)

        # per-row autograd Jacobian of the same residual
        logits, aux = model(ids, attention_mask=mask)
        sel = torch.nonzero(aux["single"]).flatten()
        labels, vm = tgt[sel], valid[sel]
        cnt = vm.long().sum(1)
        k = torch.arange(sub.D)[None]
        yd = (labels.gather(1, (cnt[:, None] - 1 - k).clamp_min(0)) - sub.DIGIT).clamp(0, 9)
        yd = torch.where(k < cnt[:, None], yd, 0).to(f64)
        y = yd @ aux["pw"]
        V, n = aux["v1"][sel], aux["n"][sel]
        nb = n.detach()
        sd0 = aux["s"][sel].detach()
        is_seed = torch.stack([torch.ones_like(sd0), sd0, sd0 * sd0], 1).norm(dim=1) / nb < sub.C_SEED
        ks = torch.where(is_seed, 0.0, torch.clamp(torch.round((V - y).detach() / nb), min=0))   # the loss's rule
        e = (V - y - ks * n) / n
        rows = list(range(0, len(sel), max(1, len(sel) // 40)))
        Ja = torch.stack([torch.cat([g.flatten().to(f64) for g in
                                     torch.autograd.grad(e[r], list(model.parameters()), retain_graph=True)])
                          for r in rows])
        sd, w = aux["s"][sel].detach(), model.w.to(f64).detach()
        Jc = torch.stack([(w[1] + 2 * w[2] * sd) * aux["ds"][sel].detach() - (yd @ aux["dpw"]).detach()
                          - (ks + e.detach()) * aux["dn"][sel].detach(), torch.ones_like(sd), sd, sd * sd],
                         1) / nb[:, None]
        rel = (Ja - Jc[rows]).abs().max(0).values / Ja.abs().max(0).values.clamp_min(1e-300)
        seeds, wrapped = int(is_seed.sum()), int((ks > 0).sum())
        msg = (f"[{tag}] trial {trial}: B={float(model.base):.2f} {len(sel)} single-step rows (seeds {seeds}, rows "
               f"unfolded with k > 0: {wrapped}); per-row J max rel err per column "
               f"{[f'{v:.1e}' for v in rel.tolist()]}")
        if "J" in st:
            g_closed = st["J"].T @ st["e"]
            gerr = float((g_auto - g_closed).norm() / g_closed.norm().clamp_min(1e-300))
            where = {int(r): i for i, r in enumerate(sel.tolist())}
            e_mine = e.detach()[torch.tensor([where[int(r)] for r in st["rows"].tolist()])]
            msg += (f"; admitted {st['m']}: |grad*nu - J^T e| / |J^T e| = {gerr:.2e}, scaled grad norm "
                    f"{float(g_auto.norm() / st['nu']):.3f}, max |e_loss - e_check| = "
                    f"{float((st['e'] - e_mine).abs().max()):.1e}")
        else:
            msg += "; no rows admitted (empty carried information, no seed in this batch at this point)"
        print(msg)
