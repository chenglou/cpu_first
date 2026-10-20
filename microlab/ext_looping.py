"""Looped x^(2^T) mod N on small moduli: does sotaku's "train on later iterations" trick keep a learned loop exact
for many iterations and let it extrapolate in T?

One squaring step is learnable at 6-8 bits (`python -m microlab.run sqmod --size 6` reaches 100%), so the remainder
difficulty is out of the way and only the loop is tested. State = residue digits plus a countdown: each iteration
squares mod N while the countdown is above 0 and holds once it reaches 0, so the answer is a resting point and extra
iterations should be harmless.

Part 1, the loop (`train`). The countdown is exact here (an integer outside the network); the network sees
go = [countdown > 0] each iteration and must square or hold. One step network for every variant:
    z = h + MLP(LN(h) + emb(N) + emb(go)),   logits = R(LN(z))        (residue digits)
The state kinds differ only in what is fed to the next iteration:
    latent   z                                   continuous state
    fb       z + P(softmax(logits))              sotaku's recurrence: latent plus fed-back predictions
    soft     E(softmax(logits))                  digit distribution, re-embedded
    hard     E(onehot(argmax logits))            snapped to digits; no gradient across iterations
    ste      hard forward, soft backward         straight-through
h0 = E(onehot(x)) for every kind, so a snapped state is exactly the embedding of its digits.
Training methods:
    plain    16 iterations from the start, backprop through all; target = final answer at every iteration >= T
    late     sotaku's recipe: 80% of batches as plain; 20% run 32/64/128/256/512 iterations without gradients from
             the start, detach, then train a 16-iteration window with the final answer at every iteration
    steps    per-step labels: the true state is the target at every iteration (hard: the inputs are true too)
Eval: T in 1..64 (training has T <= tmax), 512 iterations for every example (at least 8x T), then on to 2048:
exact at T, 2T, 4T, 8T; chain (right after each of the T squarings); strict (right at every iteration 1..8T);
held (right at every iteration T..8T); the answer at 512 and 2048; how far and how fast the state still moves; and
one iteration from the digit embedding of every reachable input (squaring every (N, r), holding every square; for
hard, all of them exact = exact for every T and every extra iteration). Small moduli fall into short squaring
cycles, so each T also gets the scores of a loop that is exact through tmax squarings and then stops ("cap").

Part 2, the counter (`counter`): learned countdown digits instead of the exact counter. Variants: one MLP over all
digits (decimal or binary), and one small cell shared by every digit position with a fixed ripple borrow
(decimal or binary). Training sees counts 0..tmax only; tests every start count up to the width's limit.
`joint` puts residue and decimal countdown digits in one snapped state, the formulation taken literally. `full`
drives the snapped residue step with the learned binary ripple counter: nothing exact but the wiring.

    python -m microlab.ext_looping train --state latent --method late --tmax 4
    python -m microlab.ext_looping counter --tmax 16
    python -m microlab.ext_looping joint --tmax 16
    python -m microlab.ext_looping full --tmax 4
    python -m microlab.ext_looping report
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .tasks import _odd_moduli, ndig

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "results_looping.jsonl")
HORIZONS = (32, 64, 128, 256, 512)
TEST_T = (1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 32, 48, 64)
EVAL_ITERS = 512


def digits2(v, w, base=10):
    """int array of any shape -> [..., w] LSB-first digits."""
    v = np.asarray(v, dtype=np.int64)
    return (v[..., None] // base ** np.arange(w)) % base


def trajectory(N, x, steps):
    """r[:, i] = x^(2^i) mod N for i = 0..steps."""
    r = np.empty((len(x), steps + 1), np.int64)
    r[:, 0] = x
    for i in range(steps):
        r[:, i + 1] = r[:, i] * r[:, i] % N
    return r


def sample(n, bits, rng, tmin, tmax):
    N = _odd_moduli(n, bits, rng)
    x = (rng.random(n) * N).astype(np.int64)
    return N, x, rng.integers(tmin, tmax + 1, n)


def all_pairs(bits):
    """Every (N, r) with N a b-bit odd modulus and 0 <= r < N."""
    Ns = np.arange((1 << (bits - 1)) + 1, 1 << bits, 2)
    N = np.concatenate([np.full(n, n) for n in Ns])
    r = np.concatenate([np.arange(n) for n in Ns])
    return N.astype(np.int64), r.astype(np.int64)


# ---------------------------------------------------------------- part 1: the loop

class Step(nn.Module):
    def __init__(self, w, state="latent", d=64, hidden=256):
        super().__init__()
        self.w, self.state = w, state
        self.E = nn.Linear(10 * w, d, bias=False)     # residue digits (one-hot or probabilities) -> state
        self.EN = nn.Linear(10 * w, d, bias=False)    # modulus digits
        self.go = nn.Embedding(2, d)
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, d))
        self.ln_out = nn.LayerNorm(d)
        self.R = nn.Linear(d, 10 * w)
        if state == "fb":
            self.P = nn.Linear(10 * w, d)

    def onehot(self, v):
        return F.one_hot(torch.from_numpy(digits2(v, self.w)), 10).float().flatten(1)

    def init(self, x):
        return self.E(self.onehot(x))

    def embed_n(self, N):
        return self.EN(self.onehot(N))

    def iterate(self, h, n, go):
        """One loop iteration. h [B, d], n = embed_n(N), go [B] long -> (next state, residue logits [B, w, 10])."""
        z = h + self.mlp(self.ln(h) + n + self.go(go))
        logits = self.R(self.ln_out(z)).view(-1, self.w, 10)
        if self.state == "latent":
            return z, logits
        p = logits.softmax(-1)
        if self.state == "fb":
            return z + self.P(p.flatten(1)), logits
        if self.state == "soft":
            return self.E(p.flatten(1)), logits
        hard = F.one_hot(logits.argmax(-1), 10).float()
        if self.state == "ste":
            hard = hard + p - p.detach()
        return self.E(hard.flatten(1)), logits


def train(state="latent", method="plain", bits=6, tmax=4, steps=3000, batch=256, lr=1e-3, window=16, late_p=0.2,
          horizons=HORIZONS, seed=0, d=64, hidden=256, tmin=1, tag="", verbose=True, eval_n=256, log=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    w = ndig((1 << bits) - 1)
    net = Step(w, state, d, hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    forced = method == "steps" and state == "hard"
    t0, ar = time.time(), np.arange(batch)
    late_batches = 0
    for step in range(steps):
        N, x, T = sample(batch, bits, rng, tmin, tmax)
        traj = trajectory(N, x, tmax)
        Tt = torch.from_numpy(T)
        k = int(rng.choice(horizons)) if method == "late" and rng.random() < late_p else 0
        late_batches += k > 0
        its = np.arange(k + 1, k + window + 1)
        tgt = torch.from_numpy(digits2(traj[ar[:, None], np.minimum(its[None, :], T[:, None])], w))  # [B, W, w]
        if forced:  # inputs are the true states, so the window's iterations are independent: one batched call
            prev = traj[ar[:, None], np.minimum(its[None, :] - 1, T[:, None])].flatten()
            go = torch.from_numpy((its[None, :] <= T[:, None]).astype(np.int64)).flatten()
            _, logits = net.iterate(net.init(prev), net.embed_n(N).repeat_interleave(window, 0), go)
            loss = F.cross_entropy(logits.flatten(0, 1), tgt.flatten())
        else:
            n, h = net.embed_n(N), net.init(x)
            if k:
                with torch.no_grad():
                    for i in range(1, k + 1):
                        h, _ = net.iterate(h, n, (i <= Tt).long())
                h, n = h.detach(), net.embed_n(N)
            total, count = 0.0, 0.0
            for j, i in enumerate(its):
                h, logits = net.iterate(h, n, (int(i) <= Tt).long())
                ce = F.cross_entropy(logits.flatten(0, 1), tgt[:, j].flatten(), reduction="none").view(batch, w).mean(1)
                m = torch.ones(batch) if method == "steps" else (int(i) >= Tt).float()
                total = total + (ce * m).sum()
                count = count + m.sum()
            loss = total / count.clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and (step + 1) % max(1, steps // 5) == 0:
            print(f"  step {step + 1}: loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    train_s = time.time() - t0
    res = evaluate(net, bits, tmax, n_per_t=eval_n)
    res.update(state=state, method=method, bits=bits, tmax=tmax, steps=steps, batch=batch, lr=lr, window=window,
               late_p=late_p if method == "late" else 0, late_batches=int(late_batches), seed=seed, d=d,
               hidden=hidden, tmin=tmin, tag=tag, train_s=round(train_s, 1), eval_s=round(time.time() - t0 - train_s, 1))
    if log:
        with open(LOG, "a") as f:
            f.write(json.dumps(res) + "\n")
    if verbose:
        print(line(res), flush=True)
    return net, res


@torch.no_grad()
def step_check(net, bits):
    """One iteration from the digit embedding of every input the loop can reach: squaring every (N, r), and holding
    every square (answers for T >= 1 are squares). go0_all also asks to hold non-squares, which never happens."""
    N, r = all_pairs(bits)
    sq = np.zeros(len(N), bool)
    for n_ in np.unique(N):
        g = N == n_
        sq[g] = np.isin(r[g], r[g] * r[g] % n_)
    n, h = net.embed_n(N), net.init(r)
    out = {}
    for name, go, want, keep in (("go1", 1, r * r % N, np.ones(len(N), bool)), ("go0", 0, r, sq),
                                 ("go0_all", 0, r, np.ones(len(N), bool))):
        _, logits = net.iterate(h, n, torch.full((len(N),), go, dtype=torch.long))
        ok = (logits.argmax(-1) == torch.from_numpy(digits2(want, net.w))).all(1).numpy()
        out[name] = round(float(ok[keep].mean()), 4)
    return out


@torch.no_grad()
def evaluate(net, bits, tmax, n_per_t=256, test_t=TEST_T, iters=EVAL_ITERS, seed=12345, long_iters=2048):
    rng = np.random.default_rng(seed)
    w, B = net.w, n_per_t * len(test_t)
    N = _odd_moduli(B, bits, rng)
    x = (rng.random(B) * N).astype(np.int64)
    T = np.repeat(np.array(test_t), n_per_t)
    traj = trajectory(N, x, max(test_t))
    tgt_all = torch.from_numpy(digits2(traj, w))                                        # [B, maxT+1, w]
    Tt, ar = torch.from_numpy(T), torch.arange(B)
    n, h = net.embed_n(N), net.init(x)
    correct = torch.zeros(B, iters, dtype=torch.bool)
    dist = torch.zeros(B, iters)
    hT = torch.zeros_like(h)
    norm_T = torch.zeros(B)
    for i in range(1, iters + 1):
        h_prev = h
        h, logits = net.iterate(h, n, (i <= Tt).long())
        correct[:, i - 1] = (logits.argmax(-1) == tgt_all[ar, Tt.clamp(max=i)]).all(1)
        at = Tt == i
        hT[at], norm_T[at] = h[at], h[at].norm(dim=1)
        dist[:, i - 1] = (h - hT).norm(dim=1)
    rel = dist / norm_T.clamp(min=1e-9)[:, None]
    grow = h.norm(dim=1) / norm_T.clamp(min=1e-9)                       # state norm at the last iteration vs at T
    move = (h - h_prev).norm(dim=1) / h.norm(dim=1).clamp(min=1e-9)     # how much the last iteration still moves it
    for i in range(iters + 1, long_iters + 1):                         # keep resting, past late training's 528
        h, logits = net.iterate(h, n, (i <= Tt).long())
    at_long = (logits.argmax(-1) == tgt_all[ar, Tt]).all(1)
    per_t = {}
    for t in test_t:
        g = torch.from_numpy(T == t)
        c, tr = correct[g], traj[T == t]
        # baselines for a loop that is exact through tmax squarings and then holds: small moduli fall into short
        # squaring cycles, so its answer at T is often right anyway ("cap"); "cap_strict" needs every iteration
        # right, which it gets only when x^(2^tmax) is already a fixed point of squaring
        cap = float((tr[:, min(t, tmax)] == tr[:, t]).mean())
        cap_strict = 1.0 if t <= tmax else float((tr[:, tmax:t + 1] == tr[:, tmax:tmax + 1]).all(1).mean())
        per_t[str(t)] = {
            "x1": round(c[:, t - 1].float().mean().item(), 4),
            "x2": round(c[:, 2 * t - 1].float().mean().item(), 4),
            "x4": round(c[:, 4 * t - 1].float().mean().item(), 4),
            "x8": round(c[:, 8 * t - 1].float().mean().item(), 4),
            "chain": round(c[:, :t].all(1).float().mean().item(), 4),       # right after every squaring 1..T
            "held": round(c[:, t - 1:8 * t].all(1).float().mean().item(), 4),  # right at T and every iteration to 8T
            "strict": round(c[:, :8 * t].all(1).float().mean().item(), 4),     # right at every iteration 1..8T
            "at512": round(c[:, -1].float().mean().item(), 4),
            f"at{long_iters}": round(at_long[g].float().mean().item(), 4),
            "drift8": round(rel[g, 8 * t - 1].median().item(), 4),
            "grow512": round(grow[g].median().item(), 4),
            "move512": round(move[g].median().item(), 6),
            "cap": round(cap, 4),
            "cap_strict": round(cap_strict, 4),
        }
    t_last = max(test_t)
    g = torch.from_numpy(T == t_last)
    chain = {str(i): round(correct[g, i - 1].float().mean().item(), 4) for i in (1, 2, 4, 8, 16, 32, 64) if i <= t_last}
    return {"step": step_check(net, bits), "per_t": per_t, "chain": chain}


def line(r):
    pt, s = r["per_t"], r["step"]
    show = [t for t in ("1", "4", "8", "16", "32", "64") if t in pt]
    ex = " ".join(f"{t}:{pt[t]['x1']:.2f}" for t in show)
    strict = " ".join(f"{t}:{pt[t].get('strict', float('nan')):.2f}" for t in show)
    held = " ".join(f"{t}:{pt[t]['held']:.2f}" for t in show)
    return (f"{r['state']}/{r['method']} b{r['bits']} tmax{r['tmax']} seed{r['seed']}: step go1 {s['go1']:.3f} "
            f"go0 {s['go0']:.3f} | exact@T {ex} | strict(1..8T) {strict} | held(T..8T) {held} "
            f"| {r['train_s']:.0f}s")


# ---------------------------------------------------------------- part 2: the counter

class FlatCounter(nn.Module):
    """One MLP over every countdown digit (positions not shared): digits -> next digits."""

    def __init__(self, base, width, hidden=128):
        super().__init__()
        self.base, self.width = base, width
        self.net = nn.Sequential(nn.Linear(base * width, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, base * width))

    def logits(self, d):                                   # d [B, width] long -> [B, width, base]
        return self.net(F.one_hot(d, self.base).float().flatten(1)).view(-1, self.width, self.base)

    def forward(self, d):
        return self.logits(d).argmax(-1)


class RippleCounter(nn.Module):
    """One small cell shared by every digit position: (digit, borrow in) -> (digit, borrow out). The decrement ripples
    up from the lowest digit with borrow 1; a borrow out of the top digit means the count was 0, so the count is kept.
    Only the cell is learned; the ripple and the hold are fixed wiring."""

    def __init__(self, base, width, hidden=32):
        super().__init__()
        self.base, self.width = base, width
        self.cell = nn.Sequential(nn.Linear(base + 1, hidden), nn.GELU(), nn.Linear(hidden, base + 1))

    def cell_logits(self, d, b):                           # d [B] long, b [B] float -> (digit logits, borrow logit)
        o = self.cell(torch.cat([F.one_hot(d, self.base).float(), b[:, None]], 1))
        return o[:, :self.base], o[:, self.base]

    def tick(self, d):
        """-> (next digits, go): go = 0 when the ripple borrows out of the top digit (the count was 0)."""
        b = torch.ones(len(d))
        out = []
        for j in range(self.width):
            dl, bl = self.cell_logits(d[:, j], b)
            out.append(dl.argmax(-1))
            b = (bl > 0).float()
        under = b > 0
        return torch.where(under[:, None], d, torch.stack(out, 1)), (~under).long()

    def forward(self, d):
        return self.tick(d)[0]


def count_down(c):
    return np.maximum(c - 1, 0)


def train_counter(kind="ripple", base=10, width=2, tmax=16, steps=1500, batch=256, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = (RippleCounter if kind == "ripple" else FlatCounter)(base, width)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(steps):
        c = rng.integers(0, tmax + 1, batch)              # every count a T <= tmax trajectory visits
        d, nd = torch.from_numpy(digits2(c, width, base)), torch.from_numpy(digits2(count_down(c), width, base))
        if kind == "flat":
            loss = F.cross_entropy(net.logits(d).flatten(0, 1), nd.flatten())
        else:                                              # teacher-forced ripple: true borrow into every cell
            b, loss = torch.ones(batch), 0.0
            raw = torch.from_numpy(digits2(c - 1, width, base))   # c - 1 with wraparound below 0 (c = 0 underflows)
            raw = torch.where(torch.from_numpy(c)[:, None] == 0, torch.full_like(raw, base - 1), raw)
            for j in range(width):
                dl, bl = net.cell_logits(d[:, j], b)
                b_out = ((d[:, j] == 0) & (b > 0)).float()
                loss = loss + F.cross_entropy(dl, raw[:, j]) + F.binary_cross_entropy_with_logits(bl, b_out)
                b = b_out
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def test_counter(net, base, width, starts):
    """For each start count T: run 2T + 4 iterations; exact iff the count reads T - i (then 0) every iteration."""
    ok = []
    for T in starts:
        d = torch.from_numpy(digits2(np.array([T]), width, base))
        good = True
        for i in range(1, 2 * T + 5):
            d = net(d)
            if not (d == torch.from_numpy(digits2(np.array([max(T - i, 0)]), width, base))).all():
                good = False
                break
        ok.append(good)
    return np.array(ok)


def counter_main(tmax=16, seeds=2, verbose=True):
    rows = []
    for kind, base, width in (("flat", 10, 2), ("ripple", 10, 2), ("flat", 2, 7), ("ripple", 2, 7)):
        top = base ** width - 1
        starts = np.arange(1, top + 1)
        for seed in range(seeds):
            net = train_counter(kind, base, width, tmax, seed=seed)
            ok = test_counter(net, base, width, starts)
            bad = starts[~ok]
            r = {"part": "counter", "kind": kind, "base": base, "width": width, "tmax": tmax, "seed": seed,
                 "exact_upto_64": round(float(ok[:64].mean()), 4), "exact_all": round(float(ok.mean()), 4),
                 "T32": bool(ok[31]), "T64": bool(ok[63]), "first_bad": int(bad[0]) if len(bad) else None,
                 "bad_upto_64": [int(v) for v in bad[bad <= 64]][:40], "max_start": int(top)}
            rows.append(r)
            with open(LOG, "a") as f:
                f.write(json.dumps(r) + "\n")
            if verbose:
                print(f"counter {kind} base{base} x{width} tmax{tmax} seed{seed}: exact T<=64 {r['exact_upto_64']:.3f} "
                      f"(all <= {top}: {r['exact_all']:.3f})  T32 {r['T32']}  T64 {r['T64']}  first bad {r['first_bad']}",
                      flush=True)
    return rows


# ---------------------------------------------------------------- part 2b: residue + countdown in one snapped state

class JointStep(nn.Module):
    """Residue digits and decimal countdown digits in one state, one MLP, snapped every iteration. No exact go:
    the network decides from the countdown digits whether to square."""

    def __init__(self, w, cw=2, d=64, hidden=256):
        super().__init__()
        self.w, self.cw = w, cw
        self.E = nn.Linear(10 * (w + cw), d, bias=False)
        self.EN = nn.Linear(10 * w, d, bias=False)
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, d))
        self.ln_out = nn.LayerNorm(d)
        self.R = nn.Linear(d, 10 * (w + cw))

    def embed(self, r, c):
        dig = np.concatenate([digits2(r, self.w), digits2(c, self.cw)], -1)
        return self.E(F.one_hot(torch.from_numpy(dig), 10).float().flatten(1))

    def logits(self, h, N):
        n = self.EN(F.one_hot(torch.from_numpy(digits2(N, self.w)), 10).float().flatten(1))
        z = h + self.mlp(self.ln(h) + n)
        return self.R(self.ln_out(z)).view(-1, self.w + self.cw, 10)


def joint_main(bits=6, tmax=16, steps=3000, batch=256, lr=1e-3, seed=0, eval_n=256, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    w = ndig((1 << bits) - 1)
    net = JointStep(w)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for _ in range(steps):                                 # per-step labels on states from T <= tmax trajectories
        N, x, T = sample(batch, bits, rng, 1, tmax)
        traj = trajectory(N, x, tmax)
        i = rng.integers(0, tmax + 1, batch)               # iteration within the trajectory (some past T: holds)
        r, c = traj[np.arange(batch), np.minimum(i, T)], np.maximum(T - i, 0)
        r2, c2 = traj[np.arange(batch), np.minimum(i + 1, T)], np.maximum(T - i - 1, 0)
        tgt = torch.from_numpy(np.concatenate([digits2(r2, w), digits2(c2, 2)], -1))
        loss = F.cross_entropy(net.logits(net.embed(r, c), N).flatten(0, 1), tgt.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
    train_s = time.time() - t0
    with torch.no_grad():
        erng = np.random.default_rng(12345)
        B = eval_n * len(TEST_T)
        N = _odd_moduli(B, bits, erng)
        x = (erng.random(B) * N).astype(np.int64)
        T = np.repeat(np.array(TEST_T), eval_n)
        traj = trajectory(N, x, max(TEST_T))
        r, c = x.copy(), T.copy()
        ok_r = np.zeros((B, EVAL_ITERS), bool)
        ok_c = np.zeros((B, EVAL_ITERS), bool)
        for it in range(1, EVAL_ITERS + 1):
            pred = net.logits(net.embed(r, c), N).argmax(-1).numpy()
            r = (pred[:, :w] * 10 ** np.arange(w)).sum(1)
            c = (pred[:, w:] * 10 ** np.arange(2)).sum(1)
            ok_r[:, it - 1] = r == traj[np.arange(B), np.minimum(it, T)]
            ok_c[:, it - 1] = c == np.maximum(T - it, 0)
    per_t = {}
    for t in TEST_T:
        g = T == t
        per_t[str(t)] = {"x1": round(float(ok_r[g, t - 1].mean()), 4),
                         "held": round(float(ok_r[g, t - 1:8 * t].all(1).mean()), 4),
                         "strict": round(float(ok_r[g, :8 * t].all(1).mean()), 4),
                         "counter": round(float(ok_c[g, :8 * t].all(1).mean()), 4),
                         "cap": round(float((traj[g, min(t, tmax)] == traj[g, t]).mean()), 4)}
    res = {"part": "joint", "bits": bits, "tmax": tmax, "steps": steps, "seed": seed, "per_t": per_t,
           "train_s": round(train_s, 1)}
    with open(LOG, "a") as f:
        f.write(json.dumps(res) + "\n")
    if verbose:
        print(f"joint b{bits} tmax{tmax} seed{seed}: exact@T " + " ".join(f"{t}:{v['x1']:.2f}" for t, v in per_t.items())
              + " | strict " + " ".join(f"{t}:{v['strict']:.2f}" for t, v in per_t.items())
              + " | counter exact " + " ".join(f"{t}:{v['counter']:.2f}" for t, v in per_t.items())
              + f" | {train_s:.0f}s", flush=True)
    return res


# ---------------------------------------------------------------- part 2c: everything learned

def full_main(bits=6, tmax=4, steps=3000, lr=3e-3, seed=0, base=2, width=7, eval_n=256, verbose=True):
    """The snapped residue step (per-step labels) driven by a learned countdown (one digit cell shared by every
    position, fixed ripple) instead of the exact counter. Both see T <= tmax only; tested on T up to 64, 8x T
    iterations each."""
    net, _ = train("hard", "steps", bits, tmax, steps, lr=lr, seed=seed, verbose=False, log=False)
    cnet = train_counter("ripple", base, width, tmax, seed=seed)
    with torch.no_grad():
        rng = np.random.default_rng(12345)
        B = eval_n * len(TEST_T)
        N = _odd_moduli(B, bits, rng)
        x = (rng.random(B) * N).astype(np.int64)
        T = np.repeat(np.array(TEST_T), eval_n)
        traj = trajectory(N, x, max(TEST_T))
        tgt_all = torch.from_numpy(digits2(traj, net.w))
        Tt, ar = torch.from_numpy(T), torch.arange(B)
        n, h = net.embed_n(N), net.init(x)
        d = torch.from_numpy(digits2(T, width, base))
        correct = torch.zeros(B, EVAL_ITERS, dtype=torch.bool)
        go_ok = torch.zeros(B, EVAL_ITERS, dtype=torch.bool)
        for i in range(1, EVAL_ITERS + 1):
            d, go = cnet.tick(d)
            go_ok[:, i - 1] = go == (i <= Tt).long()
            h, logits = net.iterate(h, n, go)
            correct[:, i - 1] = (logits.argmax(-1) == tgt_all[ar, Tt.clamp(max=i)]).all(1)
    per_t = {}
    for t in TEST_T:
        g = torch.from_numpy(T == t)
        c = correct[g]
        per_t[str(t)] = {"x1": round(c[:, t - 1].float().mean().item(), 4),
                         "strict": round(c[:, :8 * t].all(1).float().mean().item(), 4),
                         "go": round(go_ok[g, :8 * t].all(1).float().mean().item(), 4)}
    res = {"part": "full", "bits": bits, "tmax": tmax, "steps": steps, "seed": seed,
           "counter": f"ripple base {base} x{width}", "step": step_check(net, bits), "per_t": per_t}
    with open(LOG, "a") as f:
        f.write(json.dumps(res) + "\n")
    if verbose:
        print(f"full b{bits} tmax{tmax} seed{seed} (ripple base {base}): step {res['step']} | strict "
              + " ".join(f"{t}:{v['strict']:.2f}" for t, v in per_t.items())
              + " | counter go right " + " ".join(f"{t}:{v['go']:.2f}" for t, v in per_t.items()), flush=True)
    return res


# ---------------------------------------------------------------- report

ORDER = [("latent", "plain"), ("latent", "late"), ("fb", "plain"), ("fb", "late"), ("soft", "plain"),
         ("soft", "late"), ("ste", "plain"), ("ste", "late"), ("hard", "plain"), ("hard", "steps")]


def _mean(vals):
    m, lo, hi = sum(vals) / len(vals), min(vals), max(vals)
    return f"{m:.2f}" if hi - lo < 0.005 else f"{m:.2f} ({lo:.2f}-{hi:.2f})"


def report(path=LOG):
    """Markdown tables from the log: part 1 per (bits, tmax, steps, tag) with seeds averaged, then part 2."""
    rows = [json.loads(l) for l in open(path)]
    loop = [r for r in rows if "state" in r and r.get("tag") != "probe"]
    groups = {}
    for r in loop:
        groups.setdefault((r["bits"], r["tmax"], r["steps"], r.get("tag", "")), {}).setdefault(
            (r["state"], r["method"]), []).append(r)
    show = ("4", "8", "16", "32", "64")
    for (bits, tmax, steps, tag), by in sorted(groups.items()):
        print(f"\n### {bits} bits, T <= {tmax}, {steps} steps [{tag}]\n")
        cols = ["state / method", "seeds", "step square / hold", f"exact T<={tmax}"]
        cols += [f"strict@{t}" for t in show] + [f"chain@{t}" for t in ("8", "16", "32", "64")]
        cols += [f"exact@64", f"at512 T={tmax}", f"at2048 T={tmax}", f"norm growth T={tmax}"]
        print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
        for key in ORDER:
            rs = by.get(key)
            if not rs:
                continue
            pt = lambda t, k: [r["per_t"][t][k] for r in rs if k in r["per_t"][t]]
            ind = [sum(r["per_t"][t]["x1"] for t in r["per_t"] if int(t) <= tmax) /
                   sum(1 for t in r["per_t"] if int(t) <= tmax) for r in rs]
            opt = lambda v: _mean(v) if v else "-"
            cells = [f"{key[0]} / {key[1]}", str(len(rs)),
                     _mean([r["step"]["go1"] for r in rs]) + " / " + _mean([r["step"]["go0"] for r in rs]), _mean(ind)]
            cells += [_mean(pt(t, "strict")) for t in show] + [_mean(pt(t, "chain")) for t in ("8", "16", "32", "64")]
            cells += [_mean(pt("64", "x1")), _mean(pt(str(tmax), "at512")), opt(pt(str(tmax), "at2048")),
                      opt(pt(str(tmax), "grow512"))]
            print("| " + " | ".join(cells) + " |")
        p = next(iter(by.values()))[0]["per_t"]
        cells = [f"stop after {tmax} (baseline)", "", "", "1.00"] + [f"{p[t]['cap_strict']:.2f}" for t in show]
        cells += [""] * 4 + [f"{p['64']['cap']:.2f}", "", "", ""]
        print("| " + " | ".join(cells) + " |")
    counters = [r for r in rows if r.get("part") == "counter"]
    if counters:
        print("\n### counter alone: first wrong start count, right starts <= 64\n")
        for r in counters:
            print(f"- {r['kind']} base {r['base']} x{r['width']}, counts <= {r['tmax']}, seed {r['seed']}: "
                  f"first wrong {r['first_bad']}, right <= 64: {round(r['exact_upto_64'] * 64)}/64")
    for part in ("joint", "full"):
        rs = [r for r in rows if r.get("part") == part]
        if rs:
            print(f"\n### {part}: strict at T (8T iterations)\n")
            for r in rs:
                print(f"- T <= {r['tmax']}, seed {r['seed']}: "
                      + " ".join(f"{t}:{v['strict']:.2f}" for t, v in r["per_t"].items()))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--state", default="latent", choices=["latent", "fb", "soft", "hard", "ste"])
    t.add_argument("--method", default="plain", choices=["plain", "late", "steps"])
    t.add_argument("--bits", type=int, default=6)
    t.add_argument("--tmax", type=int, default=4)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--window", type=int, default=16)
    t.add_argument("--seeds", default="0", help="comma list")
    t.add_argument("--d", type=int, default=64)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--tag", default="")
    c = sub.add_parser("counter")
    c.add_argument("--tmax", type=int, default=16)
    c.add_argument("--seeds", type=int, default=2)
    j = sub.add_parser("joint")
    j.add_argument("--bits", type=int, default=6)
    j.add_argument("--tmax", type=int, default=16)
    j.add_argument("--steps", type=int, default=3000)
    j.add_argument("--seeds", default="0")
    fu = sub.add_parser("full")
    fu.add_argument("--bits", type=int, default=6)
    fu.add_argument("--tmax", type=int, default=4)
    fu.add_argument("--steps", type=int, default=3000)
    fu.add_argument("--seeds", default="0")
    sub.add_parser("report")
    for s in (t, c, j, fu):
        s.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    if a.cmd == "report":
        return report()
    torch.set_num_threads(a.threads)
    if a.cmd == "train":
        for seed in (int(s) for s in a.seeds.split(",")):
            train(a.state, a.method, a.bits, a.tmax, a.steps, a.batch, a.lr, a.window, seed=seed, d=a.d,
                  hidden=a.hidden, tag=a.tag)
    elif a.cmd == "counter":
        counter_main(a.tmax, a.seeds)
    elif a.cmd == "joint":
        for seed in (int(s) for s in a.seeds.split(",")):
            joint_main(a.bits, a.tmax, a.steps, seed=seed)
    elif a.cmd == "full":
        for seed in (int(s) for s in a.seeds.split(",")):
            full_main(a.bits, a.tmax, a.steps, seed=seed)


if __name__ == "__main__":
    main()
