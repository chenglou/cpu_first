"""Why do coupled tables get stuck? Wire-encoding experiment on a*d (digit times number).

variant latent  : MUL(a_j,d)->(lo10,hi9), ADD3(lo,hi_prev,c)->(digit,carry)     [private latent codes]
variant shared  : MUL(a_j,d)->(lo10,hi10), the SAME ADD(x,y,c) table as addition [codes must be digits
                  wherever ADD is also used on canonical inputs]
variant shared+ : shared, plus multitask: the ADD table is also trained on plain a+b
variant wide    : private latent codes but overcomplete (lo 24, hi 24)
"""
import argparse, time
import torch, torch.nn.functional as F
from cells import Table, onehot
p = argparse.ArgumentParser()
p.add_argument("--variant", default="latent"); p.add_argument("--steps", type=int, default=3000)
p.add_argument("--seed", type=int, default=0); p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--k", type=int, default=5)
a_ = p.parse_args(); torch.manual_seed(a_.seed)
V = a_.variant
LO, HI = (int(V[4:] or 24), int(V[4:] or 24)) if V.startswith("wide") else (10, 9 if V == "latent" else 10)
MUL = Table([10, 10], [LO, HI])
ADDX = Table([LO, HI, 2], [10, 2]) if V == "latent" or V.startswith("wide") else None
ADD = Table([10, 10, 2], [10, 2])
opt = torch.optim.Adam([q for m in (MUL, ADDX, ADD) if m is not None for q in m.parameters()], lr=a_.lr)
digs = lambda v, L: torch.stack([(v // 10**i) % 10 for i in range(L)], 1)
def ints(B, k): kk = torch.randint(1, k + 1, (B,)); return (torch.rand(B, dtype=torch.float64) * (10.0 ** kk)).long()
def scan(table, xs, ys, mode):
    B = xs[0].shape[0]; c = onehot(torch.zeros(B, dtype=torch.long), 2); outs, lg = [], []
    for x, y in zip(xs, ys):
        (d, c), (dl, _) = table(x, y, c, mode=mode, with_logits=True); outs.append(d); lg.append(dl)
    return torch.stack(outs, 1), torch.stack(lg, 1)
def mul1(a_int, d_int, L, mode):
    a = onehot(digs(a_int, L), 10); d = onehot(d_int, 10)
    lo, hi = zip(*[MUL(a[:, j], d, mode=mode) for j in range(L)])
    hz = onehot(torch.zeros(len(a_int), dtype=torch.long), HI)
    his = [hz] + list(hi[:-1])
    return scan(ADDX if ADDX is not None else ADD, list(lo), his, mode)
def add(a_int, b_int, L, mode):
    a = onehot(digs(a_int, L), 10); b = onehot(digs(b_int, L), 10)
    return scan(ADD, [a[:, j] for j in range(L)], [b[:, j] for j in range(L)], mode)
def acc(k, n=2000):
    with torch.no_grad():
        a, d = ints(n, k), torch.randint(0, 10, (n,)); L = k + 2
        out, _ = mul1(a, d, L, "hard"); return (out.argmax(-1) == digs(a * d, L)).all(1).float().mean().item()
t0 = time.time()
for step in range(1, a_.steps + 1):
    a, d = ints(256, a_.k), torch.randint(0, 10, (256,)); L = a_.k + 2
    _, lg = mul1(a, d, L, "ste")
    loss = F.cross_entropy(lg.reshape(-1, 10), digs(a * d, L).reshape(-1))
    if V == "shared+":
        b = ints(256, a_.k); _, lg2 = add(a, b, L, "ste")
        loss = loss + F.cross_entropy(lg2.reshape(-1, 10), digs(a + b, L).reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
print(f"{V} seed={a_.seed} acc@5={acc(5):.3f} acc@9={acc(9):.3f} t={time.time()-t0:.0f}s")
