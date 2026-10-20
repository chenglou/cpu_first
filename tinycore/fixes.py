"""Candidate fixes for the quotient-digit chicken-and-egg, each tested on the tiny core in seconds.

Every fix trains from final remainders only (the answer v mod N) unless it says otherwise, and is scored like the
baseline: exact match of the soft datapath on every example (`exact`, the model's own choices), plus the slot map.
- joint       the baseline: cell and choice trained together (core.train).
- alternate   freeze one half, train the other, swap every `period` steps (est: cell vs estimator; borrow: the
              cell's digit entries vs its carry entries, which make the choice).
- staged      easiest first: k = 1 quotient digit until exact (or `stage` steps), then k = 2, ... (same tables).
- mixed       every batch mixes k = 1..K (microlab's modmix, a curriculum as data).
- repeat      symmetry broken by construction: slot s is s runs of one subtract-N cell (core cell="repeat").
- hardem      hard EM: the E-step tries every path of slots (all S^steps) or, with estep="local", the model's own
              path and its one-step changes (as microlab/ext_quotient_em); the M-step trains the cell through the
              path that best explains the answer and the estimator toward its slots.
- clock       a mod-N-invariant loss on a register wide enough never to overflow: 1 - cos(2 pi (r - y) / N) of the
              final value r, with hard choices; the estimator is taught label-free, toward the slot whose trial is
              the smallest non-negative value (microlab/ext_quotient_clock).
- checks      the cell first learns subtraction from label-free identities, a - 0*b = a and (a - s*b) + s*b = a
              with an exact adder, on random numbers (no remainders); then joint training from answers.

    python -m tinycore.fixes --fix joint,staged,hardem --B 2 --w 4 --k 3 --choice est --seeds 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from .core import Config, Core, Data, digits, train

HERE = os.path.dirname(os.path.abspath(__file__))


def batch_of(data, rng, batch):
    return data if not batch or batch >= data.X else data.subset(rng.choice(data.X, batch, replace=False))


def cell_params(m):
    return [] if m.cfg.exact_cell else [m.a, m.b]


def choice_params(m):
    return [m.g] if m.cfg.choice in ("est", "estc") else []


def evaluate(m, data):
    return m.exact(data)


# ----- alternating -----
def alternate(m, data, steps=4000, lr=0.05, period=200, batch=256, seed=0, every=100):
    rng = np.random.default_rng(seed)
    if m.cfg.choice in ("est", "estc"):
        groups = [cell_params(m), choice_params(m)]
    else:                                                                       # borrow: digits vs carries
        groups = [[m.a], [m.b]]
    opts = [torch.optim.Adam(g, lr=lr) for g in groups]
    curve, hit = [], None
    for it in range(steps + 1):
        if it % every == 0:
            ex = evaluate(m, data)
            curve.append((it, None, ex))
            if ex >= 0.99:
                hit = it
                break
        if it == steps:
            break
        gi = (it // period) % 2
        L = m.loss(batch_of(data, rng, batch))
        for o in opts:
            o.zero_grad()
        L.backward()
        opts[gi].step()
    return curve, hit


# ----- curricula over the number of quotient digits -----
def staged(m, data, steps=4000, lr=0.05, stage=1500, batch=256, seed=0, every=100):
    cfg = m.cfg
    rng = np.random.default_rng(seed)
    params = cell_params(m) + choice_params(m)
    o = torch.optim.Adam(params, lr=lr)
    stages = [Data(cfg.B, cfg.w, kk, odd=cfg.odd, start=cfg.start) for kk in range(1, cfg.k + 1)]
    curve, hit, si, since = [], None, 0, 0
    for it in range(steps + 1):
        if it % every == 0:
            ex = evaluate(m, data)
            exs = evaluate(m, stages[si]) if si < len(stages) - 1 else ex
            curve.append((it, si + 1, ex))
            if ex >= 0.99:
                hit = it
                break
            if si < len(stages) - 1 and (exs >= 0.99 or since >= stage):
                si, since = si + 1, 0
        if it == steps:
            break
        L = m.loss(batch_of(stages[si], rng, batch))
        o.zero_grad()
        L.backward()
        o.step()
        since += 1
    return curve, hit


def mixed(m, data, steps=4000, lr=0.05, batch=256, seed=0, every=100):
    cfg = m.cfg
    rng = np.random.default_rng(seed)
    params = cell_params(m) + choice_params(m)
    o = torch.optim.Adam(params, lr=lr)
    parts = [Data(cfg.B, cfg.w, kk, odd=cfg.odd, start=cfg.start) for kk in range(1, cfg.k + 1)]
    curve, hit = [], None
    for it in range(steps + 1):
        if it % every == 0:
            ex = evaluate(m, data)
            curve.append((it, None, ex))
            if ex >= 0.99:
                hit = it
                break
        if it == steps:
            break
        L = sum(m.loss(batch_of(p, rng, batch // len(parts))) for p in parts) / len(parts)
        o.zero_grad()
        L.backward()
        o.step()
    return curve, hit


# ----- hard EM -----
def model_path(m, d):
    with torch.no_grad():
        _, ch = m.forward(d, hard=True)
    return ch.argmax(-1)                                                        # [X, steps]


def estep_all(m, d):
    """The path (slot per step) that best explains each answer, over every path; ties keep the model's own."""
    S, n = m.S, len(d.steps)
    own = model_path(m, d)
    best_ll = torch.full((d.X,), -math.inf)
    best = own.clone()
    with torch.no_grad():
        R, _ = m.forward(d, path=own)
        best_ll = m.loglik(d, R) + 1e-9
        for p in itertools.product(range(S), repeat=n):
            path = torch.tensor(p).expand(d.X, n)
            R, _ = m.forward(d, path=path)
            ll = m.loglik(d, R)
            better = ll > best_ll
            best_ll = torch.where(better, ll, best_ll)
            best = torch.where(better[:, None], path, best)
    return best


def estep_local(m, d):
    """microlab's E-step: the model's own path; for every step t and slot s, step t takes s, the steps after it take
    the estimator's top slot on the changed remainders, and the last step takes its best slot."""
    S, n = m.S, len(d.steps)
    own = model_path(m, d)
    cands = [own]
    with torch.no_grad():
        for t in range(n):
            for s in range(S):
                path = own.clone()
                path[:, t] = s
                # re-choose the later steps with the model's own (hard) choice on the changed remainders
                if t < n - 1:
                    TD, TC = m.tables()
                    R = F.one_hot(d.R0, m.cfg.B).double()
                    for i, tt in enumerate(d.steps):
                        if i <= t:
                            R, wq, D, _ = m.step(R, d.vd[:, tt], d.Nd, path[:, i], TD, TC, choice="given")
                        else:
                            R, wq, D, _ = m.step(R, d.vd[:, tt], d.Nd, d.qd[:, tt], TD, TC)
                            path[:, i] = wq.argmax(1)
                            R = torch.einsum("xs,xspo->xpo", F.one_hot(path[:, i], m.S).double(), D)
                cands.append(path)
        # the last step: its best slot, for every candidate
        out = []
        for path in cands:
            for s in range(S):
                p2 = path.clone()
                p2[:, -1] = s
                out.append(p2)
        best_ll, best = None, own
        R, _ = m.forward(d, path=own)
        best_ll = m.loglik(d, R) + 1e-9
        for path in out:
            R, _ = m.forward(d, path=path)
            ll = m.loglik(d, R)
            better = ll > best_ll
            best_ll = torch.where(better, ll, best_ll)
            best = torch.where(better[:, None], path, best)
    return best


def hardem(m, data, steps=4000, lr=0.05, batch=256, seed=0, every=100, estep="all", est_weight=1.0):
    rng = np.random.default_rng(seed)
    params = cell_params(m) + choice_params(m)
    o = torch.optim.Adam(params, lr=lr)
    curve, hit = [], None
    E = estep_all if estep == "all" else estep_local
    for it in range(steps + 1):
        if it % every == 0:
            ex = evaluate(m, data)
            curve.append((it, None, ex))
            if ex >= 0.99:
                hit = it
                break
        if it == steps:
            break
        d = batch_of(data, rng, batch)
        best = E(m, d)
        # M-step: the cell through the best path; the model's own choice (estimator, or the cell's borrows) toward
        # the path's slot, on the path's remainders (detached, so that lesson does not reach earlier steps)
        TD, TC = m.tables()
        R = F.one_hot(d.R0, m.cfg.B).double()
        Lch = 0
        for i, t in enumerate(d.steps):
            Rn, _, D, _ = m.step(R, d.vd[:, t], d.Nd, best[:, i], TD, TC, choice="given")
            _, wq, _, _ = m.step(R.detach(), d.vd[:, t], d.Nd, best[:, i], TD, TC)
            Lch = Lch - wq.clamp_min(1e-12).log().gather(1, best[:, i:i + 1]).mean() / len(d.steps)
            R = Rn
        L = -m.loglik(d, R).mean() / m.cfg.w + est_weight * Lch
        o.zero_grad()
        L.backward()
        o.step()
    return curve, hit


# ----- clock scoring -----
def signed_place(B, W, N):
    """For every place j and digit d, (signed place value of d at j) mod N: [X, W, B] ints. The register is read in
    B's complement (top digit >= ceil(B/2) is negative)."""
    X = len(N)
    out = np.zeros((X, W, B), dtype=np.int64)
    for j in range(W):
        for dd in range(B):
            v = dd * B ** j
            if j == W - 1 and dd >= (B + 1) // 2:
                v -= B ** W
            out[:, j, dd] = np.mod(v, N)
    return torch.from_numpy(out)


def clock_run(m, d, pick="est", noise=0.0, rng=None, train_est=True):
    """The datapath on a register of W = w + k + 2 places with one hard slot per step. Returns the clock loss, the
    estimator's label-free loss, and the register's final digits (distributions)."""
    cfg = m.cfg
    B, w = cfg.B, cfg.w
    W = w + cfg.k + 2
    TD, TC = m.tables()
    R = F.one_hot(F.pad(d.R0, (0, W - d.R0.shape[1])), B).double()
    est_loss = torch.zeros(())
    for i, t in enumerate(d.steps):
        Rp = torch.cat([F.one_hot(d.vd[:, t], B).double()[:, None], R[:, :W - 1]], 1)
        D, c = m.ripple_slot(Rp, d.Nd, TD, TC)
        with torch.no_grad():                                                 # each trial's value, by argmax digits
            dig = D.argmax(-1)                                                 # [X, S, W]
            val = (dig * (B ** torch.arange(W))).sum(-1)
            val = val - (dig[..., -1] >= (B + 1) // 2).long() * B ** W
            nonneg = torch.where(val >= 0, val, torch.full_like(val, 1 << 60))
            target = torch.where((val >= 0).any(1), nonneg.argmin(1), val.argmax(1))
        if pick == "given":
            s = d.qd[:, t].clone()
            if noise:
                flip = torch.from_numpy(rng.random(d.X) < noise)
                step = torch.from_numpy(rng.choice([-1, 1], d.X))
                s = torch.where(flip, (s + step).clamp(0, m.S - 1), s)
        else:
            pe = m.estimate(Rp[:, :w + 1].detach(), d.Nd)
            s = pe.argmax(1)
            if train_est:
                est_loss = est_loss - pe.clamp_min(1e-12).log().gather(1, target[:, None]).mean()
        R = D[torch.arange(d.X), s]
    pv = signed_place(B, W, d.N)                                               # [X, W, B]
    theta = 2 * math.pi / torch.from_numpy(d.N).double()
    ang = theta[:, None, None] * pv.double()
    phi = (R * torch.exp(1j * ang)).sum(-1)                                    # [X, W] expected phasor per place
    z = torch.prod(phi, 1) * torch.exp(-1j * theta * torch.from_numpy(d.y).double())
    loss = (1 - z.real).mean()
    return loss, est_loss, R, W


def clock_exact(m, d, pick="est"):
    with torch.no_grad():
        _, _, R, W = clock_run(m, d, pick=pick, train_est=False)
        dig = R.argmax(-1)
        B = m.cfg.B
        val = (dig * (B ** torch.arange(W))).sum(-1) - (dig[:, -1] >= (B + 1) // 2).long() * B ** W
        y = torch.from_numpy(d.y)
        N = torch.from_numpy(d.N)
        return (val == y).double().mean().item(), (torch.remainder(val - y, N) == 0).double().mean().item()


def slot_multipliers(m, d):
    """Per slot, the multiple of N its trial subtracts most often (from argmax digits on the datapath's first step)."""
    with torch.no_grad():
        B, w = m.cfg.B, m.cfg.w
        W = w + m.cfg.k + 2
        TD, TC = m.tables()
        R = F.one_hot(F.pad(d.R0, (0, W - d.R0.shape[1])), B).double()
        t = d.steps[0]
        Rp = torch.cat([F.one_hot(d.vd[:, t], B).double()[:, None], R[:, :W - 1]], 1)
        D, _ = m.ripple_slot(Rp, d.Nd, TD, TC)
        dig = D.argmax(-1)
        val = (dig * (B ** torch.arange(W))).sum(-1) - (dig[..., -1] >= (B + 1) // 2).long() * B ** W
        rp = (Rp.argmax(-1) * (B ** torch.arange(W))).sum(-1)
        N = torch.from_numpy(d.N)
        out = []
        for s in range(m.S):
            diff = rp - val[:, s]
            mult = torch.where(diff % N == 0, diff // N, torch.full_like(diff, -99))
            vals, counts = torch.unique(mult, return_counts=True)
            j = counts.argmax()
            out.append((int(vals[j]), round(counts[j].item() / d.X, 2)))
        return out


def clock(m, data, steps=4000, lr=0.05, batch=256, seed=0, every=100, pick="est", noise=0.0, est_weight=1.0):
    rng = np.random.default_rng(seed)
    params = cell_params(m) + (choice_params(m) if pick == "est" else [])
    o = torch.optim.Adam(params, lr=lr)
    curve, hit = [], None
    for it in range(steps + 1):
        if it % every == 0:
            ex, exm = clock_exact(m, data, pick="est" if pick == "est" else "given")
            curve.append((it, exm, ex))
            if ex >= 0.99:
                hit = it
                break
        if it == steps:
            break
        d = batch_of(data, rng, batch)
        L, Le, _, _ = clock_run(m, d, pick=pick, noise=noise, rng=rng)
        o.zero_grad()
        (L + est_weight * Le).backward()
        o.step()
    return curve, hit


# ----- label-free checks for the cell -----
def exact_adder(B):
    """One-hot (digit, carry) of r + n + c, c in {0, 1}: [B, B, 2, B], [B, B, 2, 2]."""
    r, n, c = np.meshgrid(np.arange(B), np.arange(B), np.arange(2), indexing="ij")
    x = r + n + c
    return torch.from_numpy(np.eye(B)[x % B]), torch.from_numpy(np.eye(2)[(x >= B).astype(int)])


def add_exact(x, bd, AD, AC):
    """x [X, P, B] digit distributions + b (ints [X, P]) mod B^P, with the exact adder applied multilinearly."""
    X, P, B = x.shape
    c = torch.zeros(X, 2)
    c[:, 0] = 1
    out = []
    for j in range(P):
        ad, ac = AD[:, bd[:, j]], AC[:, bd[:, j]]                              # [B(r), X, 2, out]
        out.append(torch.einsum("xr,rxco,xc->xo", x[:, j], ad, c))
        c = torch.einsum("xr,rxco,xc->xo", x[:, j], ac, c)
    return torch.stack(out, 1)


def checks(m, data, steps=4000, lr=0.05, batch=256, seed=0, every=100, pre=1000, then="joint"):
    """Label-free pretraining of the slot cell for `pre` steps, then `then` ("joint" or "est": estimator only) from
    answers for the rest."""
    cfg = m.cfg
    B, w = cfg.B, cfg.w
    rng = np.random.default_rng(seed)
    AD, AC = exact_adder(B)
    o = torch.optim.Adam(cell_params(m), lr=lr)
    lo, hi = max(2, B ** (w - 1)), B ** w
    for it in range(pre):
        a = rng.integers(0, B ** (w + 1), batch)
        b = rng.integers(lo, hi, batch)
        ad = torch.from_numpy(digits(a, B, w + 1))
        bd = torch.from_numpy(digits(b, B, w))
        x = F.one_hot(ad, B).double()
        TD, TC = m.tables()
        D, _ = m.ripple_slot(x, bd, TD, TC)                                   # [X, S, w+1, B]
        L = 0
        for s in range(m.S):
            y = D[:, s]
            for _ in range(s):                                                 # add b back s times
                y = add_exact(y, F.pad(bd, (0, 1)), AD, AC)
            L = L - y.clamp_min(1e-12).log().gather(-1, ad[..., None]).mean()
        o.zero_grad()
        L.backward()
        o.step()
    cell_after = m.exact_given(data)
    rest = steps - pre
    if then == "est":
        params = choice_params(m)
    else:
        params = cell_params(m) + choice_params(m)
    o2 = torch.optim.Adam(params, lr=lr)
    curve, hit = [(0, cell_after, evaluate(m, data))], None
    for it in range(rest + 1):
        if it % every == 0:
            ex = evaluate(m, data)
            curve.append((pre + it, None, ex))
            if ex >= 0.99:
                hit = pre + it
                break
        if it == rest:
            break
        L = m.loss(batch_of(data, rng, batch))
        o2.zero_grad()
        L.backward()
        o2.step()
    return curve, hit


# ----- driver -----
FIXES = {"joint": None, "alternate": alternate, "staged": staged, "mixed": mixed, "hardem": hardem,
         "clock": clock, "checks": checks}


def run_fix(job):
    torch.set_num_threads(1)
    fix, cfg_kw, seed, kw = job["fix"], job["cfg"], job["seed"], job.get("kw", {})
    if fix == "repeat":
        cfg_kw = {**cfg_kw, "cell": "repeat"}
    cfg = Config(**cfg_kw)
    data = Data(cfg.B, cfg.w, cfg.k, odd=cfg.odd, start=cfg.start)
    m = Core(cfg, seed=seed)
    t = time.time()
    steps, lr = kw.pop("steps", 4000), kw.pop("lr", 0.05)
    if fix in ("joint", "repeat"):
        curve, hit = train(m, data, steps=steps, lr=lr, every=100, batch=256, seed=seed, stop_at=0.99, **kw)
    else:
        curve, hit = FIXES[fix](m, data, steps=steps, lr=lr, seed=seed, **kw)
    extra = {}
    if fix == "clock":
        extra["multipliers"] = slot_multipliers(m, data)
        extra["exact_mod_N"] = curve[-1][1]
    smap, raw, mapped = m.slot_map(data)
    return {"fix": fix, **cfg_kw, "seed": seed, "kw": job.get("kw", {}), "hit": hit, "final": curve[-1][2],
            "curve": curve, "slots": smap, "raw": raw, "mapped": mapped, "cell_right": m.cell_right(),
            "exact_given": m.exact_given(data),
            "seconds": round(time.time() - t, 1), **extra}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fix", default="joint")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--w", type=int, default=4)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--choice", default="est")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--kw", default="{}")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default="fixes.jsonl")
    a = p.parse_args()
    jobs = []
    for fix in a.fix.split(","):
        for s in range(a.seeds):
            jobs.append({"fix": fix, "cfg": {"B": a.B, "w": a.w, "k": a.k, "choice": a.choice, "cell": "slot",
                                             "init": 0.1}, "seed": s, "kw": json.loads(a.kw)})
    with ProcessPoolExecutor(a.workers) as ex, open(os.path.join(HERE, "runs", a.out), "a") as f:
        for r in ex.map(run_fix, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            extra = f" mult {r['multipliers']} modN {r['exact_mod_N']:.2f}" if "multipliers" in r else ""
            print(f"{r['fix']:9s} B{r['B']} w{r['w']} k{r['k']} {r['cell']}/{r['choice']} seed {r['seed']}: hit "
                  f"{r['hit']} final {r['final']:.3f} slots {r['slots']} cell {r['cell_right']:.2f}{extra} "
                  f"({r['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
