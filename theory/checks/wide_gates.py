# Gates at every step on short programs over a wide instruction set: the follow-up to step_gates.py. Backs
# theory/SEARCH_AND_GRADIENTS.md §5.3.
#
# The deep file's searches are short in calls (its H1 round is 2 calls of routines) but wide (30 instructions and a
# library). step_gates.py found gates at every step holding to about 4 steps over 3 instructions; does that hold over
# 30? The setup is step_gates.py's: one register y mod N, every instruction a permutation of the residues, each task's
# rows its program's whole table. Here N is in {5, 11, 17, 23} (56 rows), and the 30 instructions are y + c for
# c = 1..10, c·y for c in {2, 3, 4, 6, 7, 8, 9, 12, 13, 14} and y^k for k in {3, 7, 9, 13, 17, 19, 21, 23, 27, 29}.
# step_gates.py's 3 instructions (INC, DBL, CUBE) run on the same moduli as a control. The same three learners:
#   search       the first program, in a fixed order, that fits every row;
#   whole gates  softmax weights over all I^L programs, a row's label probability their mixture's, by Adam;
#   step gates   one softmax over the I instructions at each step, the register a distribution over residues, by Adam.
# Run: python3 theory/checks/wide_gates.py   (needs numpy and torch; CPU, about 23 minutes)
import time

import numpy as np
import torch

MODS = (5, 11, 17, 23)
M = max(MODS)
PAIRS = [(x, n) for n in MODS for x in range(n)]
R = len(PAIRS)
X = np.array([x for x, _ in PAIRS])
NS = torch.tensor([float(n) for _, n in PAIRS])
EPS = 1e-3          # as in step_gates.py
STEPS = 2000
TASKS = 50
STARTS = 4

NARROW = [("INC", lambda a, n: (a + 1) % n), ("DBL", lambda a, n: 2 * a % n), ("CUBE", lambda a, n: a ** 3 % n)]
WIDE = ([(f"ADD{c}", lambda a, n, c=c: (a + c) % n) for c in range(1, 11)]
        + [(f"MUL{c}", lambda a, n, c=c: c * a % n) for c in (2, 3, 4, 6, 7, 8, 9, 12, 13, 14)]
        + [(f"POW{k}", lambda a, n, k=k: np.array([pow(int(v), k, n) for v in a]))
           for k in (3, 7, 9, 13, 17, 19, 21, 23, 27, 29)])


def loss_of(p):
    """Minus the labels' mean log-probability, for each task; p[t, r] is the probability task t gives row r's label."""
    return -torch.log((1 - EPS) * p + EPS / NS).mean(1)


class Machine:
    def __init__(self, ops):
        self.size = len(ops)
        # fwd[o, r, a]: instruction o applied to residue a, mod row r's N (residues at or above N are padding).
        self.fwd = np.tile(np.arange(M, dtype=np.uint8), (self.size, R, 1))
        for r, (_, n) in enumerate(PAIRS):
            for o, (_, f) in enumerate(ops):
                self.fwd[o, r, :n] = f(np.arange(n), n)
        assert all(sorted(self.fwd[o, r]) == list(range(M)) for o in range(self.size) for r in range(R))
        assert len({self.fwd[o].tobytes() for o in range(self.size)}) == self.size        # no two alike
        self.inv = np.argsort(self.fwd, axis=2)     # residue b came from inv[o, r, b]

    def run(self, prog):
        y = X.astype(np.uint8)
        for o in prog:
            y = self.fwd[o, np.arange(R), y]
        return y

    def table(self, L):
        """Every program's outputs, out[k] for program k, whose base-I digits are its instructions, first one first."""
        out = X[None, :].astype(np.uint8)
        for _ in range(L):
            out = np.stack([self.fwd[o, np.arange(R)[None, :], out] for o in range(self.size)], 1).reshape(-1, R)
        return out

    def search(self, Y, L):
        """For each task, how many programs the search tried to reach the first that fits every row. It takes programs
        by their last instruction, then by the rest in base-I order."""
        prefix = self.table(L - 1)
        tried = np.zeros(len(Y), int)
        for last in range(self.size):
            out = self.fwd[last, np.arange(R)[None, :], prefix]
            for t in np.flatnonzero(tried == 0):
                fits = np.flatnonzero((out == Y[t]).all(1))
                if len(fits):
                    tried[t] = last * len(prefix) + fits[0] + 1
        assert (tried > 0).all()
        return tried

    def whole_gates(self, Y, L, lr=0.1, chunk=10):
        out = self.table(L)
        ok = []
        for i in range(0, len(Y), chunk):
            y = Y[i:i + chunk]
            A = torch.tensor(out[None, :, :] == y[:, None, :], dtype=torch.float32)   # (tasks, programs, rows)
            w = torch.zeros(A.shape[:2], requires_grad=True)
            opt = torch.optim.Adam([w], lr=lr)
            for _ in range(STEPS):
                p = torch.einsum("tk,tkr->tr", torch.softmax(w, 1), A)
                opt.zero_grad()
                loss_of(p).sum().backward()
                opt.step()
            ok += list((out[w.argmax(1).numpy()] == y).all(1))
        return np.array(ok)

    def step_gates(self, Y, L, lr, scale, seed):
        """As in step_gates.py: for each task, whether its decoded program fits, the share of rows it fits, its weakest
        step's top gate weight, and how much its loss moved over the second half of training."""
        T = len(Y)
        g = torch.Generator().manual_seed(seed)
        theta = (scale * torch.randn(T, L, self.size, generator=g)).requires_grad_()
        idx = torch.tensor(self.inv.transpose(1, 0, 2).reshape(R, self.size * M)).expand(T, R, self.size * M)
        labels = torch.tensor(Y.astype(np.int64))[:, :, None]
        start = torch.zeros(T, R, M)
        start[:, np.arange(R), X] = 1
        opt = torch.optim.Adam([theta], lr=lr)
        for step in range(STEPS + 1):
            v = start
            gates = torch.softmax(theta, 2)
            for t in range(L):
                v = torch.einsum("to,trom->trm", gates[:, t], v.gather(2, idx).view(T, R, self.size, M))
            loss = loss_of(v.gather(2, labels)[:, :, 0])
            if step == STEPS // 2:
                half = loss.detach().numpy()
            if step == STEPS:
                break
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
        progs = theta.argmax(2).numpy()
        share = np.array([(self.run(p) == y).mean() for p, y in zip(progs, Y)])
        weakest = gates.max(2).values.min(1).values.detach().numpy()
        return share == 1, share, weakest, np.abs(loss.detach().numpy() - half)


def targets(rng, machine, L):
    return np.array([machine.run(p) for p in rng.integers(0, machine.size, (TASKS, L))])


def main(lengths=((2, 3, 4, 6, 8), (1, 2, 3, 4, 5, 6, 8)), sweep=(2, 3, 5)):
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    print(f"Programs of L instructions on y mod N, N in {MODS}; each task's rows are its whole table ({R} rows)."
          f" {TASKS} random programs a length; step gates from {STARTS} random starts each (logits ~ N(0, 1)); Adam,"
          f" {STEPS} steps, learning rate 0.1. Failed step-gate runs: as in step_gates.py (medians).")
    wide = Machine(WIDE)
    machines = (("3 instructions (INC, DBL, CUBE)", Machine(NARROW), lengths[0], 12, 8),
                ("30 instructions", wide, lengths[1], 5, 3))
    for name, machine, Ls, search_to, whole_to in machines:
        print(f"\n{name}:")
        print(f"{'L':>3} {'programs':>18} | {'search: tried':>17} | {'whole gates':>11} | {'step gates: runs':>16}"
              f" {'programs':>8} | {'failed: rows fit':>16} {'settled':>7} {'loss moved':>10} | {'time':>5}")
        for L in Ls:
            Y = targets(rng, machine, L)
            cells = [f"{L:>3} {machine.size ** L:>18,}"]
            if L <= search_to:
                cells.append(f"median {int(np.median(machine.search(Y, L))):>10,}")
            else:
                cells.append(f"{'not run':>17}")
            cells.append(f"{machine.whole_gates(Y, L).mean():>11.0%}" if L <= whole_to else f"{'not run':>11}")
            clock = time.time()
            ok, share, weakest, moved = machine.step_gates(np.repeat(Y, STARTS, 0), L, 0.1, 1.0, seed=L)
            cells.append(f"{ok.mean():>16.0%} {ok.reshape(TASKS, STARTS).any(1).mean():>8.0%}")
            if ok.all():
                cells.append(f"{'-':>16} {'-':>7} {'-':>10}")
            else:
                cells.append(f"{np.median(share[~ok]):>16.0%} {(weakest[~ok] >= 0.99).mean():>7.0%}"
                             f" {np.median(moved[~ok]):>10.1e}")
            cells.append(f"{time.time() - clock:>4.0f}s")
            print(" | ".join(cells), flush=True)

    print("\n30 instructions, step gates swept: the share of runs whose program fits, by learning rate and starting"
          " spread (logits ~ N(0, s²); s = 0.01 starts every step near an even mix), 2 starts a program.")
    for L in sweep:
        Y = targets(rng, wide, L)
        print(f"  L = {L}:")
        for lr in (0.03, 0.1, 0.3, 1.0):
            row = []
            for scale in (0.01, 1.0):
                ok = wide.step_gates(np.repeat(Y, 2, 0), L, lr, scale, seed=1000 + L)[0]
                row.append(f"s = {scale:<4} {ok.mean():>4.0%}")
            print(f"    lr {lr:<4}  " + "   ".join(row), flush=True)


if __name__ == "__main__":
    main()
