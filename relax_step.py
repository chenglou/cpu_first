"""Relaxations 1 and 4: learn ONE squaring step s -> s^2 mod N with a generic looped transformer, then loop it.

Model (nothing arithmetic designed in): tokens are the digits of N and s (LSB-first) plus 2W blank output/scratch
slots; one pre-LN transformer block (bidirectional attention + GELU MLP) applied K times with input injection
(a looped / universal transformer); rotary positions on the digit's place value, so attention sees relative place.
The low W slots are read out as y's digits (cross-entropy). Between squarings the prediction is hard-snapped
(argmax digits) and fed back as the next s.

--data fresh   relaxation 4: unlimited fresh (N, s) at b bits, N any odd b-bit number.
--data h1like  relaxation 1: per-step labels along an H1-shaped finite set at b bits (<=430 moduli, 1k/1k/33k x3 rows).
--data h1      relaxation 1 on the real H1 training rows (13/16/20-bit), W=8, certified with h1.certify.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import relax_common as rc


def rope(x, pos, base=64.0):
    # x [B,H,L,D], pos [L] float
    D = x.shape[-1]
    inv = base ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    ang = pos[:, None] * inv[None, :]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h, mlp=4):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x, pos):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos), rope(k, pos)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o(a.transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.ln2(x))


class LoopedStep(nn.Module):
    def __init__(self, W, d=256, h=8, K=12, layers=1):
        super().__init__()
        self.W, self.K = W, K
        self.dig = nn.Embedding(11, d)  # 10 digits + blank
        self.typ = nn.Embedding(3, d)  # N, s, out
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 10)
        self.head_sq = nn.Linear(d, 10)  # optional: digits of s^2 on the 2W scratch slots (--aux_square)
        pos = torch.cat([torch.arange(W), torch.arange(W), torch.arange(2 * W)]).float()
        typ = torch.cat([torch.zeros(W), torch.ones(W), 2 * torch.ones(2 * W)]).long()
        self.register_buffer("pos", pos)
        self.register_buffer("typs", typ)

    def forward(self, nd, sd):
        B = nd.shape[0]
        blank = torch.full((B, 2 * self.W), 10, device=nd.device, dtype=torch.long)
        tok = torch.cat([nd, sd, blank], 1)
        e = self.dig(tok) + self.typ(self.typs)[None]
        x = e
        for _ in range(self.K):
            for blk in self.blocks:
                x = blk(x + e, self.pos)
        h = self.ln(x[:, 2 * self.W:])
        self.sq_logits = self.head_sq(h)  # [B, 2W, 10], used only with --aux_square
        return self.head(h[:, :self.W])  # [B, W, 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--data", default="fresh", choices=["fresh", "h1like", "h1"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--minutes", type=float, default=1e9)
    ap.add_argument("--aux_square", action="store_true",
                    help="also supervise the digits of s^2 (the value before the remainder) on the scratch slots")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    W = 8 if a.data == "h1" else rc.width(a.bits)
    model = LoopedStep(W, a.d, 8, a.K, a.layers).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = lambda s: min(1.0, s / 500) * (0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)) * 0.95 + 0.05)

    pool = unseen = None
    if a.data == "h1like":
        d = rc.h1like(a.bits, a.seed)
        PN, PS, PY = rc.step_pairs(d)
        pool, unseen = d["pool"], d["unseen"]
        print(f"h1like b={a.bits}: {len(PN)} step pairs, {len(pool)} pool moduli, {len(unseen)} unseen", flush=True)
    elif a.data == "h1":
        import h1
        tr = h1.load("train")
        traj_N, traj_S, traj_Y = [], [], []
        for n, x, t in zip(tr["N"], tr["X"], tr["T"]):
            s = int(x) % int(n)
            for _ in range(int(t)):
                y = s * s % int(n)
                traj_N.append(n), traj_S.append(s), traj_Y.append(y)
                s = y
        PN, PS, PY = (np.array(v, dtype=np.int64) for v in (traj_N, traj_S, traj_Y))
        pool = np.unique(tr["N"])
        print(f"h1 real: {len(PN)} step pairs from {len(tr['N'])} rows, {len(pool)} moduli", flush=True)

    def batch(B):
        if a.data == "fresh":
            return rc.fresh_step(rng, a.bits, B)
        i = rng.integers(0, len(PN), B)
        return PN[i], PS[i], PY[i]

    def to_t(v):
        return torch.from_numpy(rc.digits(v, W)).to(dev)

    @torch.no_grad()
    def predict(N, s, bs=4096):
        model.eval()
        out = []
        for i in range(0, len(N), bs):
            lo = model(to_t(N[i:i + bs]), to_t(s[i:i + bs]))
            out.append(lo.argmax(-1).cpu().numpy())
        model.train()
        return rc.undigits(np.concatenate(out))

    def exact_on(N, s, y):
        return float((predict(N, s) == y).mean())

    @torch.no_grad()
    def square_exact(N, s, bs=4096):
        model.eval()
        ok = []
        for i in range(0, len(N), bs):
            model(to_t(N[i:i + bs]), to_t(s[i:i + bs]))
            pred = rc.undigits(model.sq_logits.argmax(-1).cpu().numpy())
            ok.append(pred == s[i:i + bs] * s[i:i + bs])
        model.train()
        return float(np.concatenate(ok).mean())

    erng = np.random.default_rng(12345)
    if a.data == "h1":
        ev = {}
        # held-out: fresh s on pool moduli, all bits mixed (13/16/20)
        Ne = pool[erng.integers(0, len(pool), 8192)]
        se = (erng.random(8192) * Ne).astype(np.int64)
        ev["pool"] = (Ne, se, rc.sqmod(se, Ne))
    else:
        ev = {"fresh": rc.fresh_step(erng, a.bits, 8192)}
        if pool is not None:
            ev["pool"] = rc.fresh_step(erng, a.bits, 8192, pool)
            if len(unseen):
                ev["unseen"] = rc.fresh_step(erng, a.bits, 8192, unseen)
    print(f"model params={nparam} W={W} K={a.K} layers={a.layers} d={a.d} device={dev}", flush=True)

    t0 = time.time()
    log = []
    best = 0.0
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        N, s, y = batch(a.batch)
        lo = model(to_t(N), to_t(s))
        loss = F.cross_entropy(lo.reshape(-1, 10), to_t(y).reshape(-1))
        if a.aux_square:
            sq = torch.from_numpy(rc.digits(s * s, 2 * W)).to(dev)
            loss_sq = F.cross_entropy(model.sq_logits.reshape(-1, 10), sq.reshape(-1))
            loss = loss + loss_sq
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            accs = {k: exact_on(*v) for k, v in ev.items()}
            if a.aux_square:
                accs["square"] = square_exact(*next(iter(ev.values()))[:2])
            el = time.time() - t0
            log.append({"step": step, "loss": float(loss), "time": el, **accs})
            print(f"step {step} loss {float(loss):.4f} " + " ".join(f"{k}={v:.4f}" for k, v in accs.items())
                  + f" t={el:.0f}s", flush=True)
            if el > a.minutes * 60:
                print("time budget reached", flush=True)
                break
    train_time = time.time() - t0

    # final: big single-step test + ladder with hard snapping
    res = {"args": vars(a), "params": nparam, "train_time": train_time, "log": log}
    frng = np.random.default_rng(777)
    if a.data == "h1":
        import h1

        def predict_h1(N, X, T):
            s = X % N
            out = s.copy()
            for t in np.unique(T):
                m = T == t
                cur = s[m].copy()
                for _ in range(int(t)):
                    cur = predict(N[m], cur)
                out[m] = cur
            return rc.digits(out, h1.L)

        Nf = pool[frng.integers(0, len(pool), 100000)]
        sf = (frng.random(100000) * Nf).astype(np.int64)
        res["single_step_pool_100k"] = exact_on(Nf, sf, rc.sqmod(sf, Nf))
        print("single-step exact (pool, 100k):", res["single_step_pool_100k"], flush=True)
        for prof in ("depth_t", "depth_ood_n_t"):
            res[prof] = h1.certify(predict_h1, prof)
    else:
        Nf, sf, yf = rc.fresh_step(frng, a.bits, 100000)
        res["single_step_fresh_100k"] = exact_on(Nf, sf, yf)
        print("single-step exact (fresh moduli, 100k):", res["single_step_fresh_100k"], flush=True)
        res["ladder_fresh"] = rc.ladder(lambda N, s: predict(N, s), a.bits)
        if pool is not None:
            Nf, sf, yf = rc.fresh_step(frng, a.bits, 100000, pool)
            res["single_step_pool_100k"] = exact_on(Nf, sf, yf)
            print("single-step exact (pool moduli, 100k):", res["single_step_pool_100k"], flush=True)
            res["ladder_pool"] = rc.ladder(lambda N, s: predict(N, s), a.bits, moduli=pool)
    res["total_time"] = time.time() - t0
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print("done", json.dumps({k: v for k, v in res.items() if k not in ("log", "args")})[:2000], flush=True)


if __name__ == "__main__":
    main()
