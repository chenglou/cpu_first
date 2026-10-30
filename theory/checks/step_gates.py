# Gates at every step against gates over whole programs, on tiny straight-line programs. Backs
# theory/SEARCH_AND_GRADIENTS.md §4.4.
#
# A program is L instructions on one register y mod N, each one of INC (y + 1), DBL (2y) and CUBE (y³). For N in
# {5, 11, 17} all three permute the residues, so no program collapses to a constant. A task is a random program, and
# its rows are its whole table: every (x, N), 33 rows. Three learners, all over the same 3^L programs:
#   search       the first program, in a fixed order, that fits every row;
#   whole gates  softmax weights over all 3^L programs, a row's label probability their mixture's (§4.3), by Adam;
#   step gates   one softmax over the 3 instructions at each step. The register holds a distribution over residues,
#                and each step moves it by the mix of its instructions, so the model is a soft program with 3L
#                weights, trained by Adam through all L steps: backprop through per-step gates ("forward marginals"
#                gradient descent in Gaunt et al. 2016).
# A learner succeeds if its program fits every row; for step gates, the program of each step's top instruction.
# Run: python3 theory/checks/step_gates.py   (needs numpy and torch; CPU, about 12 minutes)
import time

import numpy as np
import torch

MODS = (5, 11, 17)
M = max(MODS)
OPS = ("INC", "DBL", "CUBE")
PAIRS = [(x, n) for n in MODS for x in range(n)]
R = len(PAIRS)
X = np.array([x for x, _ in PAIRS])
NS = np.array([n for _, n in PAIRS])
EPS = 1e-3          # keeps a row's log-probability finite, as in search_and_gradients.py
STEPS = 2000        # Adam steps, for both kinds of gates
TASKS = 50          # random programs per length
STARTS = 4          # random starts per program, for step gates

# FWD[o, r, a]: instruction o applied to residue a, mod row r's N (residues at or above N are padding, left alone).
FWD = np.tile(np.arange(M), (len(OPS), R, 1))
for r, (_, n) in enumerate(PAIRS):
    a = np.arange(n)
    FWD[0, r, :n], FWD[1, r, :n], FWD[2, r, :n] = (a + 1) % n, 2 * a % n, a ** 3 % n
assert all(sorted(FWD[o, r]) == list(range(M)) for o in range(len(OPS)) for r in range(R))   # all permutations
INV = np.argsort(FWD, axis=2)       # the inverse permutations: residue b came from INV[o, r, b]


def run(prog):
    y = X.copy()
    for o in prog:
        y = FWD[o, np.arange(R), y]
    return y


def table(L):
    """Every program's outputs, out[k] for program k, whose base-3 digits are its instructions, first one first."""
    out = X[None, :]
    for _ in range(L):
        out = np.stack([FWD[o, np.arange(R)[None, :], out] for o in range(len(OPS))], axis=1).reshape(-1, R)
    return out


def loss_of(p):
    """Minus the labels' mean log-probability, for each task; p[t, r] is the probability task t gives row r's label."""
    return -torch.log((1 - EPS) * p + EPS / torch.tensor(NS)).mean(1)


def search(out, Y):
    """For each task, how many programs the search tried to reach the first that fits every row."""
    tried = []
    for y in Y:
        fits = (out == y).all(1)
        assert fits.any()
        tried.append(int(fits.argmax()) + 1)
    return np.array(tried)


def whole_gates(out, Y, lr):
    A = torch.tensor(out[None, :, :] == Y[:, None, :], dtype=torch.float32)        # (tasks, programs, rows)
    w = torch.zeros(A.shape[:2], requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(STEPS):
        p = torch.einsum("tk,tkr->tr", torch.softmax(w, 1), A)
        opt.zero_grad()
        loss_of(p).sum().backward()
        opt.step()
    pick = w.argmax(1).numpy()
    return (out[pick] == Y).all(1)


def step_gates(Y, L, lr, scale, seed):
    """Per-step gates on len(Y) tasks; returns, for each, whether its decoded program fits, the share of rows it fits,
    its weakest step's top gate weight (1 means every step settled on one instruction), and how much its loss moved
    over the second half of training."""
    g = torch.Generator().manual_seed(seed)
    theta = (scale * torch.randn(len(Y), L, len(OPS), generator=g)).requires_grad_()
    inv = torch.tensor(INV)
    labels = torch.tensor(Y)[:, :, None]
    start = torch.zeros(len(Y), R, M)
    start[:, np.arange(R), X] = 1
    opt = torch.optim.Adam([theta], lr=lr)
    for step in range(STEPS + 1):
        v = start
        gates = torch.softmax(theta, 2)
        for t in range(L):
            v = sum(gates[:, t, o, None, None] * v.gather(2, inv[o].expand(len(Y), R, M)) for o in range(len(OPS)))
        loss = loss_of(v.gather(2, labels)[:, :, 0])
        if step == STEPS // 2:
            half = loss.detach().numpy()
        if step == STEPS:
            break
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    progs = theta.argmax(2).numpy()
    share = np.array([(run(p) == y).mean() for p, y in zip(progs, Y)])
    weakest = gates.max(2).values.min(1).values.detach().numpy()
    return share == 1, share, weakest, np.abs(loss.detach().numpy() - half)


def targets(rng, L):
    progs = rng.integers(0, len(OPS), (TASKS, L))
    return progs, np.array([run(p) for p in progs])


def main():
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    print(f"Programs of L instructions from {', '.join(OPS)} on y mod N, N in {MODS}; each task's rows are its whole"
          f" table ({R} rows). {TASKS} random programs a length; step gates from {STARTS} random starts each"
          f" (logits ~ N(0, 1)); Adam, {STEPS} steps, learning rate 0.1.")
    print("Failed step-gate runs: the share of rows their program fits, the share settled on one instruction at every"
          f" step (top weight >= 0.99), and how much their loss moved over the last {STEPS // 2:,} steps (medians).")
    print(f"\n{'L':>3} {'programs':>21} | {'search: tried':>14} | {'whole gates':>11} | {'step gates: runs':>16}"
          f" {'programs':>8} | {'failed: rows fit':>16} {'settled':>7} {'loss moved':>10} | {'time':>5}")
    for L in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        progs, Y = targets(rng, L)
        cells = [f"{L:>3} {3 ** L:>21,}"]
        if L <= 12:
            out = table(L)
            cells.append(f"median {int(np.median(search(out, Y))):>7,}")
        else:
            cells.append(f"{'not run':>14}")
        cells.append(f"{whole_gates(out, Y, 0.1).mean():>11.0%}" if L <= 8 else f"{'not run':>11}")
        clock = time.time()
        ok, share, weakest, moved = step_gates(np.repeat(Y, STARTS, 0), L, 0.1, 1.0, seed=L)
        per_prog = ok.reshape(TASKS, STARTS).any(1)
        cells.append(f"{ok.mean():>16.0%} {per_prog.mean():>8.0%}")
        if ok.all():
            cells.append(f"{'-':>16} {'-':>7} {'-':>10}")
        else:
            cells.append(f"{np.median(share[~ok]):>16.0%} {(weakest[~ok] >= 0.99).mean():>7.0%}"
                         f" {np.median(moved[~ok]):>10.1e}")
        cells.append(f"{time.time() - clock:>4.0f}s")
        print(" | ".join(cells), flush=True)

    # Not a tuning artefact: sweep the learning rate and the starting spread at two lengths, 2 starts a program.
    print("\nStep gates, swept: the share of runs whose program fits, by learning rate and starting spread"
          " (logits ~ N(0, s²); s = 0.01 starts every step near an even mix).")
    for L in (8, 16):
        progs, Y = targets(rng, L)
        print(f"  L = {L}:")
        for lr in (0.03, 0.1, 0.3, 1.0):
            row = []
            for scale in (0.01, 1.0):
                ok = step_gates(np.repeat(Y, 2, 0), L, lr, scale, seed=1000 + L)[0]
                row.append(f"s = {scale:<4} {ok.mean():>4.0%}")
            print(f"    lr {lr:<4}  " + "   ".join(row), flush=True)


if __name__ == "__main__":
    main()
