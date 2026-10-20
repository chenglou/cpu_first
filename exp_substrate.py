"""Substrate test: can shared learned digit tables, wired in a fixed schoolbook dataflow, be learned
end-to-end (straight-through, answers only) and extrapolate in width?

task add : a + b            tables ADD(a_j, b_j, c) -> (digit, carry)
task mul1: a * d (digit)    tables MUL(a_j, d) -> (lo, hi); ADD3(lo_j, hi_{j-1}, c) -> (digit, carry)
task mul : a * b            schoolbook: acc += (b_i * a) << i, reusing MUL/ADD3 for partial products and
                            ADD for accumulation (sequential depth ~ k scans of length L)
Train widths 1..k_train digits, test at k_train+2 .. k_train+4 digits. Exact-match accuracy.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from cells import Table, onehot

p = argparse.ArgumentParser()
p.add_argument("--task", default="add", choices=["add", "mul1", "mul"])
p.add_argument("--k_train", type=int, default=5)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--mode", default="ste")
args = p.parse_args()
torch.manual_seed(args.seed)

ADD = Table([10, 10, 2], [10, 2])
MUL = Table([10, 10], [10, 9])
ADD3 = Table([10, 9, 2], [10, 2])
params = list(ADD.parameters()) + list(MUL.parameters()) + list(ADD3.parameters())
opt = torch.optim.Adam(params, lr=args.lr)


def rand_ints(B, kmax, kmin=1):
    k = torch.randint(kmin, kmax + 1, (B,))
    hi = (10 ** k).double()
    return (torch.rand(B, dtype=torch.float64) * hi).long()


def to_digits(v, L):
    return torch.stack([(v // 10**i) % 10 for i in range(L)], dim=1)


def add_scan(a, b, mode):  # a, b: [B, L, 10] one-hot-ish; returns digits [B,L,10], logits list
    B, L, _ = a.shape
    c = onehot(torch.zeros(B, dtype=torch.long), 2)
    outs, logits = [], []
    for j in range(L):
        (d, c), (dl, _) = ADD(a[:, j], b[:, j], c, mode=mode, with_logits=True)
        outs.append(d)
        logits.append(dl)
    return torch.stack(outs, 1), torch.stack(logits, 1)


def mul1(a, dgt, mode):  # a [B,L,10], dgt [B,10] -> a*dgt digits [B,L,10]
    B, L, _ = a.shape
    lo, hi = zip(*[MUL(a[:, j], dgt, mode=mode) for j in range(L)])
    c = onehot(torch.zeros(B, dtype=torch.long), 2)
    hz = onehot(torch.zeros(B, dtype=torch.long), 9)
    outs, logits = [], []
    for j in range(L):
        (d, c), (dl, _) = ADD3(lo[j], hi[j - 1] if j > 0 else hz, c, mode=mode, with_logits=True)
        outs.append(d)
        logits.append(dl)
    return torch.stack(outs, 1), torch.stack(logits, 1)


def run(a_int, b_int, L, mode):
    if args.task == "add":
        return add_scan(onehot(to_digits(a_int, L), 10), onehot(to_digits(b_int, L), 10), mode)
    if args.task == "mul1":
        return mul1(onehot(to_digits(a_int, L), 10), onehot(b_int % 10, 10), mode)
    a = onehot(to_digits(a_int, L), 10)
    bd = to_digits(b_int, L)
    acc = onehot(torch.zeros_like(bd), 10)
    logits = None
    kb = int((b_int.max().item() and len(str(int(b_int.max().item())))) or 1)
    for i in range(kb):
        pp, _ = mul1(a, onehot(bd[:, i], 10), mode)
        pp = torch.cat([onehot(torch.zeros_like(bd[:, :i]), 10), pp[:, : L - i]], 1)  # shift left i (designed)
        acc, logits = add_scan(acc, pp, mode)
    return acc, logits


def target(a_int, b_int):
    return {"add": a_int + b_int, "mul1": a_int * (b_int % 10), "mul": a_int * b_int}[args.task]


def evaluate(kmin, kmax, n=2000):
    with torch.no_grad():
        a, b = rand_ints(n, kmax, kmin), rand_ints(n, kmax, kmin)
        L = min(kmax + 2 if args.task == 'add' else 2 * kmax + 1, 19)
        out, _ = run(a, b, L, "hard")
        return (out.argmax(-1) == to_digits(target(a, b), L)).all(1).float().mean().item()


t0 = time.time()
for step in range(1, args.steps + 1):
    a, b = rand_ints(args.batch, args.k_train), rand_ints(args.batch, args.k_train)
    L = args.k_train + 2 if args.task == 'add' else 2 * args.k_train + 1
    _, logits = run(a, b, L, args.mode)
    loss = F.cross_entropy(logits.reshape(-1, 10), to_digits(target(a, b), L).reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 250 == 0 or step == 1:
        print(f"step {step} loss {loss.item():.4f} train-width acc {evaluate(1, args.k_train, 500):.3f} "
              f"t={time.time()-t0:.0f}s", flush=True)
for kmax in (args.k_train, args.k_train + 2, min(args.k_train + 4, 9 if args.task != 'add' else 16)):
    print(f"exact acc at width {kmax}: {evaluate(kmax, kmax):.4f}")
