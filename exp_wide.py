"""Wide-wire learned tables: data efficiency (fixed example pool) and depth (full multiplication).

All tables are learned truth tables (cells.Table, straight-through). Internal wires (partial-product lo/hi)
use a W-way alphabet; I/O digits are canonical 10-way. The accumulator reuses ADD, whose output is
supervised at the end, so weight sharing ties every intermediate accumulator to the output code.

task mul1: a * d (k-digit times one digit)
task mul : a * b (k-digit times k-digit), schoolbook: acc = ADD(acc, (b_i * a) << i) for each digit of b
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from cells import Table, onehot

p = argparse.ArgumentParser()
p.add_argument("--task", default="mul1")
p.add_argument("--W", type=int, default=24)
p.add_argument("--pool", type=int, default=0, help="fixed training pool size (0 = fresh data each step)")
p.add_argument("--k", type=int, default=4)
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--C", type=int, default=2, help="carry wire alphabet")
p.add_argument("--curriculum", type=int, default=0, help="grow multiplier digits 1..k over training")
a_ = p.parse_args()
torch.manual_seed(a_.seed)
W = a_.W

C = a_.C
PPW = 10 if a_.task == "mul1" else W     # partial-product digit wire: supervised in mul1, latent (wide) in mul
MUL = Table([10, 10], [W, W])           # (a_j, d) -> (lo code, hi code)
PP = Table([W, W, C], [PPW, C])         # (lo_j, hi_{j-1}, carry) -> (pp digit code, carry)
ADD = Table([10, PPW, C], [10, C])      # (acc digit, pp digit code, carry) -> (acc digit, carry)
opt = torch.optim.Adam([q for m in (MUL, PP, ADD) for q in m.parameters()], lr=a_.lr)

digs = lambda v, L: torch.stack([(v // 10**i) % 10 for i in range(L)], 1)


def ints(B, k, g=None):
    kk = torch.randint(1, k + 1, (B,), generator=g)
    return (torch.rand(B, dtype=torch.float64, generator=g) * (10.0 ** kk)).long()


def scan(table, xs, ys, mode):
    B = xs[0].shape[0]
    c = onehot(torch.zeros(B, dtype=torch.long), C)
    outs, lg = [], []
    for x, y in zip(xs, ys):
        (d, c), (dl, _) = table(x, y, c, mode=mode, with_logits=True)
        outs.append(d)
        lg.append(dl)
    return outs, torch.stack(lg, 1)


def mul1(a, d, mode):  # a: list of L [B,10]; d: [B,10]
    lo, hi = zip(*[MUL(aj, d, mode=mode) for aj in a])
    hz = onehot(torch.zeros(d.shape[0], dtype=torch.long), W)
    return scan(PP, list(lo), [hz] + list(hi[:-1]), mode)


KB = a_.k


def forward(a_int, b_int, L, mode):
    a = list(onehot(digs(a_int, L), 10).unbind(1))
    if a_.task == "mul1":
        return mul1(a, onehot(b_int % 10, 10), mode)
    bd = digs(b_int, L)
    z = onehot(torch.zeros_like(a_int), 10)
    zp = onehot(torch.zeros_like(a_int), PPW)  # fixed 'zero' code on the partial-product wire (shift padding)
    acc, lg = [z] * L, None
    for i in range(KB):
        pp, _ = mul1(a, onehot(bd[:, i], 10), mode)
        pp = [zp] * i + pp[: L - i]  # shift by i digits (wiring)
        acc, lg = scan(ADD, acc, pp, mode)
    return acc, lg


def target(a, b):
    return a * (b % 10) if a_.task == "mul1" else a * b


L = a_.k + 2 if a_.task == "mul1" else 2 * a_.k + 1
if a_.pool:
    g = torch.Generator().manual_seed(1234 + a_.seed)
    PA, PB = ints(a_.pool, a_.k, g), ints(a_.pool, a_.k, g)


def acc_at(k, n=2000):
    with torch.no_grad():
        a, b = ints(n, k), ints(n, k)
        LL = k + 2 if a_.task == "mul1" else 2 * k + 1
        global KB
        saved = KB
        KB = k
        out, _ = forward(a, b, LL, "hard")
        KB = saved
        return (torch.stack(out, 1).argmax(-1) == digs(target(a, b), LL)).all(1).float().mean().item()


t0 = time.time()
for step in range(1, a_.steps + 1):
    if a_.pool:
        idx = torch.randint(0, a_.pool, (a_.batch,))
        a, b = PA[idx], PB[idx]
    else:
        a = ints(a_.batch, a_.k)
        if a_.curriculum and a_.task == "mul":
            KB = min(a_.k, 1 + (step * a_.k) // a_.steps)
        b = ints(a_.batch, KB) if a_.task == "mul" else ints(a_.batch, a_.k)
    _, lg = forward(a, b, L, "ste")
    loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), digs(target(a, b), L).reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
kx = a_.k + 3 if a_.task == "mul1" else min(a_.k + 2, 9)
print(f"{a_.task} W={W} C={C} cur={a_.curriculum} pool={a_.pool} seed={a_.seed}: acc@{a_.k}={acc_at(a_.k):.3f} acc@{kx}={acc_at(kx):.3f} "
      f"loss={loss.item():.4f} t={time.time()-t0:.0f}s", flush=True)
