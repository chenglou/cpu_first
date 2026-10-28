"""Soft selection over exact blocks: can gradient descent choose the ops and operands of x^2 mod N from answers?

Same straight-line program space as ext_exact.py (registers N, x, 0, 1, 2, earlier results; blocks exact in fp64).
Three ways to make the choice soft:

  darts    value mixing: each slot's operands are softmax mixtures of register *values*, its output a softmax mixture
           of every op applied to them. One forward pass, cheap. Temperature annealed, argmax at the end.
  mixture  output-level mixture over complete hard programs: p(y) = sum_prog pi(prog) [prog(x) = y], pi factorised
           per slot. Exact, but the forward pass enumerates every program (it is speculation with soft weights).
  reinforce sample K hard programs per step from pi, reward = share of rows exactly right, score-function update.

    python -m microlab.ext_exact_soft darts --isa basic --L 2 --seeds 10
    python -m microlab.ext_exact_soft interp --isa basic --target sq      # loss along the path to the right program
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import numpy as np
import torch

from . import ext_exact as E

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- exact blocks on reals (fp64)

def _tdiv(a, b):
    safe = torch.where(b.abs() < 1e-9, torch.ones_like(b), b)
    return torch.where(b.abs() < 1e-9, torch.zeros_like(a), torch.floor(a / safe))


def _tmod(a, b):
    safe = torch.where(b.abs() < 1e-9, torch.ones_like(b), b)
    return torch.where(b.abs() < 1e-9, a, a - safe * torch.floor(a / safe))


TOPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": _tdiv,
    "mod": _tmod,
    "min": torch.minimum,
    "max": torch.maximum,
    "lt": lambda a, b: (a < b).to(a.dtype),
}

# the reference program for each target, as (op, a, b) over registers N=0, x=1, 0=2, 1=3, 2=4, t0=5, ...
REFERENCE = {
    ("sq", "tiny"): [("mul", 1, 1), ("mod", 5, 0)],
    ("sq", "basic"): [("mul", 1, 1), ("mod", 5, 0)],
    ("sq", "nomod"): [("mul", 1, 1), ("div", 5, 0), ("mul", 0, 6), ("sub", 5, 7)],
    ("p4", "basic"): [("mul", 1, 1), ("mod", 5, 0), ("mul", 6, 6), ("mod", 7, 0)],
    ("sqN", "basic"): [("mul", 1, 1), ("add", 5, 0)],         # control: x*x + N, no reduction
    ("sqN", "nomod"): [("mul", 1, 1), ("add", 5, 0)],
    ("x2x", "basic"): [("add", 1, 4), ("mul", 1, 5)],         # control: x*(x+2), no reduction
    ("sqN", "ring"): [("mul", 1, 1), ("add", 5, 0)],
    ("nxx", "ring"): [("sub", 0, 1), ("mul", 5, 1)],          # control: (N - x) * x, no reduction
    ("nxx", "basic"): [("sub", 0, 1), ("mul", 5, 1)],
    ("nxxmod", "basic"): [("sub", 0, 1), ("mul", 5, 1), ("mod", 6, 0)],   # the same, then mod N
    ("x2x", "ring"): [("add", 1, 4), ("mul", 1, 5)],
}


def make_rows(target, data, bits, rng, n=1000):
    if data == "h1" and target in ("sq", "p4"):
        r = E.h1_rows(T={"sq": 1, "p4": 2}[target])
        perm = rng.permutation(len(r["N"]))
        return {k: v[perm][:n] for k, v in r.items()}
    r = E.free_rows(n, bits, rng, T=2 if target == "p4" else 1)
    if target == "sqN":
        r["y"] = r["x"] * r["x"] + r["N"]
    if target == "x2x":
        r["y"] = r["x"] * (r["x"] + 2)
    if target in ("nxx", "nxxmod"):
        r["y"] = ref_fn(target)(r["N"], r["x"])
    return r


def ref_fn(target):
    if target == "sqN":
        return lambda N, x: x * x + N
    if target == "x2x":
        return lambda N, x: x * (x + 2)
    if target == "nxx":
        return lambda N, x: (N - x) * x
    if target == "nxxmod":
        return lambda N, x: (N - x) * x % N
    return E.target_fn(target)


def loss_fn(out, y, N, kind):
    if kind == "circ":     # distance on the circle Z/N: right up to multiples of N
        return (1 - torch.cos(2 * math.pi * (out - y) / N)).mean()
    if kind == "log":      # scale-free: compare magnitudes and signs
        f = lambda v: torch.sign(v) * torch.log1p(v.abs())
        return ((f(out) - f(y)) ** 2).mean()
    scale = torch.clamp(y.abs(), min=1.0) if kind == "rel" else N
    return torch.log1p(((out - y) / scale) ** 2).mean()   # robust MSE


class SoftProgram(torch.nn.Module):
    def __init__(self, isa, L, R0=5, init_logits=None, noise=0.01, gen=None):
        super().__init__()
        self.isa, self.L, self.R0 = tuple(isa), L, R0
        mk = lambda n: torch.nn.Parameter(noise * torch.randn(n, generator=gen))
        self.op = torch.nn.ParameterList([mk(len(isa)) for _ in range(L)])
        self.a = torch.nn.ParameterList([mk(R0 + i) for i in range(L)])
        self.b = torch.nn.ParameterList([mk(R0 + i) for i in range(L)])

    def set_probs(self, slot_probs):
        """slot_probs: list over slots of (p_op, p_a, p_b) arrays -> logits = log p."""
        with torch.no_grad():
            for i, (po, pa, pb) in enumerate(slot_probs):
                self.op[i].copy_(torch.log(torch.as_tensor(po).clamp_min(1e-300)))
                self.a[i].copy_(torch.log(torch.as_tensor(pa).clamp_min(1e-300)))
                self.b[i].copy_(torch.log(torch.as_tensor(pb).clamp_min(1e-300)))

    def forward(self, N, x, tau=1.0, hard=False):
        """hard=False: DARTS value mixing. hard=True: straight-through Gumbel-softmax, i.e. the forward pass runs
        one sampled hard program exactly and the backward pass uses the soft relaxation's gradient."""
        regs = [N, x, torch.zeros_like(N), torch.ones_like(N), 2 * torch.ones_like(N)]

        def sel(logits):
            if not hard:
                return torch.softmax(logits / tau, 0)
            g = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-20, 1.0)))
            soft = torch.softmax((logits + g) / tau, 0)
            onehot = torch.zeros_like(soft)
            onehot[soft.argmax()] = 1.0
            return onehot - soft.detach() + soft

        for i in range(self.L):
            R = torch.stack(regs)                                   # [R, B]
            A = sel(self.a[i]) @ R
            Bv = sel(self.b[i]) @ R
            outs = torch.stack([TOPS[o](A, Bv) for o in self.isa])  # [ops, B]
            regs.append(sel(self.op[i]) @ outs)
        return regs[-1]

    def harden(self):
        prog = []
        for i in range(self.L):
            prog.append((self.isa[int(self.op[i].argmax())], int(self.a[i].argmax()), int(self.b[i].argmax())))
        return prog


def correct_on(prog, target, held):
    f = ref_fn(target)
    return all(bool((E.run_prog(prog, [r["N"], r["x"]]) == f(r["N"], r["x"])).all()) for r in held.values())


def equivalent_programs(target, isa, L, held):
    """Every raw program of length L equivalent to the target on the held-out rows (brute force for small spaces)."""
    R0 = 5
    slots = []
    for d in range(L):
        slots.append([(o, a, b) for o in isa for a in range(R0 + d) for b in range(R0 + d)])
    n = 1
    for s in slots:
        n *= len(s)
    if n > 3_000_000:
        return None, n
    f = ref_fn(target)
    good = []
    r = held["16"]
    want = f(r["N"][:64], r["x"][:64])
    for prog in itertools.product(*slots):
        if (E.run_prog(list(prog), [r["N"][:64], r["x"][:64]]) == want).all():
            if correct_on(list(prog), target, held):
                good.append(list(prog))
    return good, n


def train_darts(isa, L, target, rows, steps=1500, lr=0.05, batch=256, tau0=1.0, tau1=0.05, loss="circ",
                seed=0, init=None, verbose=False, hard=False):
    gen = torch.Generator().manual_seed(seed)
    net = SoftProgram(isa, L, gen=gen)
    if init is not None:
        net.set_probs(init)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    N = torch.as_tensor(rows["N"], dtype=torch.float64)
    x = torch.as_tensor(rows["x"], dtype=torch.float64)
    y = torch.as_tensor(rows["y"], dtype=torch.float64)
    rng = np.random.default_rng(seed)
    hist = []
    for step in range(steps):
        tau = tau0 * (tau1 / tau0) ** (step / max(steps - 1, 1))
        idx = torch.as_tensor(rng.integers(0, len(N), batch))
        out = net(N[idx], x[idx], tau, hard=hard)
        l = loss_fn(out, y[idx], N[idx], loss)
        opt.zero_grad()
        l.backward()
        for p in net.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(0.0, 0.0, 0.0)
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            hist.append((step, l.item()))
            if verbose:
                print(f"    step {step} tau {tau:.3f} loss {float(l):.4f} prog {net.harden()}", flush=True)
    return net, hist


def cmd_darts(a):
    isa = E.ISAS[a.isa]
    held = E.held_out_sets(np.random.default_rng(999))
    if a.target in ("sqN", "x2x", "nxx", "nxxmod"):
        f = ref_fn(a.target)
        for r in held.values():
            r["y"] = f(r["N"], r["x"])
    good, space = equivalent_programs(a.target, isa, a.L, held)
    chance = None if good is None else len(good) / space
    print(f"[{a.mode} {a.isa} L={a.L} target {a.target} data {a.data}@{a.bits} loss {a.loss}] raw space {space:,}; "
          f"equivalent-correct {None if good is None else len(good)} -> chance {chance}", flush=True)
    wins = 0
    t0 = time.time()
    for seed in range(a.seeds):
        rows = make_rows(a.target, a.data, a.bits, np.random.default_rng(seed))
        torch.manual_seed(seed)
        net, hist = train_darts(isa, a.L, a.target, rows, steps=a.steps, lr=a.lr, loss=a.loss, seed=seed,
                                verbose=a.verbose, hard=a.mode == "st", tau0=a.tau0, tau1=a.tau1)
        prog = net.harden()
        ok = correct_on(prog, a.target, held)
        wins += ok
        print(f"  seed {seed}: {'OK ' if ok else 'no '} {E.show(prog, ['N', 'x', '0', '1', '2'])}  "
              f"loss {hist[0][1]:.3f}->{hist[-1][1]:.3f}", flush=True)
    rec = {"exp": "darts", "mode": a.mode, "isa": a.isa, "L": a.L, "target": a.target, "data": a.data, "bits": a.bits,
           "loss": a.loss, "steps": a.steps, "lr": a.lr, "seeds": a.seeds, "wins": wins, "space": space,
           "n_correct_programs": None if good is None else len(good), "chance": chance,
           "seconds": round(time.time() - t0, 1)}
    print(f"  => {wins}/{a.seeds} seeds harden to a correct program ({time.time() - t0:.0f}s)")
    with open(E.LOG, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def onehot_mix(n, idx, eps):
    p = np.full(n, eps / n)
    p[idx] += 1 - eps
    return p


def cmd_interp(a):
    """Loss along pi = (1-eps) onehot(reference) + eps uniform, every slot. Where does the loss leave chance?"""
    isa = E.ISAS[a.isa]
    ref = REFERENCE[(a.target, a.isa)]
    L = len(ref)
    out = []
    for bits in a.bits_list:
        rows = make_rows(a.target, "free", bits, np.random.default_rng(0), n=4000)
        N = torch.as_tensor(rows["N"], dtype=torch.float64)
        x = torch.as_tensor(rows["x"], dtype=torch.float64)
        y = torch.as_tensor(rows["y"], dtype=torch.float64)
        net = SoftProgram(isa, L)
        line = []
        for eps in [10.0 ** -e for e in range(0, 17)] + [0.0]:
            probs = [(onehot_mix(len(isa), isa.index(o), eps), onehot_mix(5 + i, ai, eps), onehot_mix(5 + i, bi, eps))
                     for i, (o, ai, bi) in enumerate(ref)]
            net.set_probs(probs)
            with torch.no_grad():
                o_ = net(N, x, 1.0)
                lc = float(loss_fn(o_, y, N, a.loss))
                ex = float((torch.round(o_) == y).double().mean())
            line.append((eps, lc, ex))
        out.append({"bits": bits, "path": line})
        print(f"  bits {bits}: " + "  ".join(f"{e:.0e}:{l:.3g}/{x_:.2f}" for e, l, x_ in line), flush=True)
    with open(E.LOG, "a") as fh:
        fh.write(json.dumps({"exp": "interp", "isa": a.isa, "target": a.target, "loss": a.loss, "out": out}) + "\n")


def cmd_basin(a):
    """Start from mass p on the reference choice (every slot), the rest on one random wrong choice; train; does it
    harden to the reference? p < 0.5 means the initial argmax is wrong."""
    isa = E.ISAS[a.isa]
    ref = REFERENCE[(a.target, a.isa)]
    L = len(ref)
    held = E.held_out_sets(np.random.default_rng(999))
    if a.target in ("sqN", "x2x", "nxx", "nxxmod"):
        f = ref_fn(a.target)
        for r in held.values():
            r["y"] = f(r["N"], r["x"])
    res = []
    for p in a.p_list:
        wins = init_right = 0
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            probs = []
            for i, (o, ai, bi) in enumerate(ref):
                trip = []
                for n, idx in ((len(isa), isa.index(o)), (5 + i, ai), (5 + i, bi)):
                    w = rng.choice([j for j in range(n) if j != idx]) if n > 1 else idx
                    v = np.full(n, 1e-6 / n)
                    v[idx] += p
                    v[w] += 1 - p
                    trip.append(v / v.sum())
                probs.append(tuple(trip))
            rows = make_rows(a.target, a.data, a.bits, rng)
            init_prog = [(isa[int(np.argmax(po))], int(np.argmax(pa)), int(np.argmax(pb))) for po, pa, pb in probs]
            init_right += correct_on(init_prog, a.target, held)
            net, hist = train_darts(isa, L, a.target, rows, steps=a.steps, lr=a.lr, loss=a.loss, seed=seed,
                                    init=probs, tau0=1.0, tau1=a.tau1)
            wins += correct_on(net.harden(), a.target, held)
        print(f"  p={p:.3f}: argmax right at init {init_right}/{a.seeds}, after training {wins}/{a.seeds}",
              flush=True)
        res.append({"p": p, "init_right": init_right, "wins": wins})
    with open(E.LOG, "a") as fh:
        fh.write(json.dumps({"exp": "darts_basin", "isa": a.isa, "target": a.target, "data": a.data,
                             "bits": a.bits, "loss": a.loss, "seeds": a.seeds, "res": res}) + "\n")


# ---------------------------------------------------------------- output-level mixture and REINFORCE

def all_programs(isa, L, R0=5):
    slots = [[(o, a, b) for o in isa for a in range(R0 + d) for b in range(R0 + d)] for d in range(L)]
    return slots, list(itertools.product(*[range(len(s)) for s in slots]))


def cmd_mixture(a):
    """Exact likelihood mixture over every complete program, pi factorised per slot (slot = op x a x b)."""
    isa = E.ISAS[a.isa]
    slots, combos = all_programs(isa, a.L)
    held = E.held_out_sets(np.random.default_rng(999))
    rows = make_rows(a.target, a.data, a.bits, np.random.default_rng(0))
    t0 = time.time()
    M = np.zeros((len(combos), a.rows), bool)
    for i, c in enumerate(combos):
        prog = [slots[d][j] for d, j in enumerate(c)]
        M[i] = E.run_prog(prog, [rows["N"][:a.rows], rows["x"][:a.rows]]) == rows["y"][:a.rows]
    t_enum = time.time() - t0
    Mt = torch.as_tensor(M, dtype=torch.float64)
    C = torch.as_tensor(np.array(combos))
    logits = [torch.nn.Parameter(torch.zeros(len(s))) for s in slots]
    opt = torch.optim.Adam(logits, lr=a.lr)
    for step in range(a.steps):
        logp = sum(torch.log_softmax(logits[d], 0)[C[:, d]] for d in range(a.L))   # [programs]
        # per-row likelihood of the mixture, floor so wrong rows don't explode
        lik = (torch.exp(logp)[:, None] * Mt).sum(0) * (1 - 1e-6) + 1e-6 / 1e6
        loss = -torch.log(lik).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    best = [slots[d][int(logits[d].argmax())] for d in range(a.L)]
    pmax = [float(torch.softmax(logits[d], 0).max()) for d in range(a.L)]
    ok = correct_on(best, a.target, held)
    print(f"[mixture {a.isa} L={a.L}] {len(combos):,} programs enumerated on {a.rows} rows in {t_enum:.1f}s; "
          f"after {a.steps} Adam steps argmax = {E.show(best, ['N', 'x', '0', '1', '2'])} "
          f"({'correct' if ok else 'WRONG'}), slot max prob {[round(p, 4) for p in pmax]}")
    with open(E.LOG, "a") as fh:
        fh.write(json.dumps({"exp": "mixture", "isa": a.isa, "L": a.L, "target": a.target, "programs": len(combos),
                             "rows": a.rows, "steps": a.steps, "correct": ok, "pmax": pmax,
                             "enum_s": round(t_enum, 2)}) + "\n")


def cmd_reinforce(a):
    """Sample K hard programs per step from a factorised pi (op, a, b per slot); reward = share of a batch of rows
    exactly right; REINFORCE with a mean baseline. Count samples until the first correct program."""
    isa = E.ISAS[a.isa]
    held = E.held_out_sets(np.random.default_rng(999))
    good, space = equivalent_programs(a.target, isa, a.L, held) if a.L <= 2 else (None, None)
    res = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        rows = make_rows(a.target, a.data, a.bits, rng)
        lo = [torch.nn.Parameter(torch.zeros(len(isa))) for _ in range(a.L)]
        la = [torch.nn.Parameter(torch.zeros(5 + d)) for d in range(a.L)]
        lb = [torch.nn.Parameter(torch.zeros(5 + d)) for d in range(a.L)]
        opt = torch.optim.Adam(lo + la + lb, lr=a.lr)
        first_hit, converged = None, None
        for step in range(a.steps):
            idx = rng.integers(0, len(rows["N"]), 64)
            logp = torch.zeros(a.K)
            progs = [[] for _ in range(a.K)]
            for d in range(a.L):
                for dist_l, k in ((lo[d], "op"), (la[d], "a"), (lb[d], "b")):
                    dist = torch.distributions.Categorical(logits=dist_l)
                    s = dist.sample((a.K,))
                    logp = logp + dist.log_prob(s)
                    for i in range(a.K):
                        progs[i].append(int(s[i]))
            rewards = []
            for i in range(a.K):
                pr = [(isa[progs[i][3 * d]], progs[i][3 * d + 1], progs[i][3 * d + 2]) for d in range(a.L)]
                out = E.run_prog(pr, [rows["N"][idx], rows["x"][idx]])
                r = float((out == rows["y"][idx]).mean())
                rewards.append(r)
                if r == 1.0 and first_hit is None and correct_on(pr, a.target, held):
                    first_hit = (step + 1) * a.K
            R = torch.as_tensor(rewards)
            adv = R - R.mean()
            loss = -(adv * logp).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            greedy = [(isa[int(lo[d].argmax())], int(la[d].argmax()), int(lb[d].argmax())) for d in range(a.L)]
            if converged is None and correct_on(greedy, a.target, held) and \
                    min(float(torch.softmax(t, 0).max()) for t in lo + la + lb) > 0.9:
                converged = (step + 1) * a.K
                break
        res.append({"seed": seed, "first_hit_samples": first_hit, "converged_samples": converged})
        print(f"  seed {seed}: first correct sample after {first_hit} samples; greedy converged after {converged}",
              flush=True)
    exp_hits = None if not good else space / len(good)
    print(f"  random sampling would need ~{exp_hits} samples on average (space {space}, correct {None if good is None else len(good)})")
    with open(E.LOG, "a") as fh:
        fh.write(json.dumps({"exp": "reinforce", "isa": a.isa, "L": a.L, "target": a.target, "K": a.K,
                             "steps": a.steps, "res": res, "expected_random": exp_hits}) + "\n")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--isa", default="basic", choices=sorted(E.ISAS))
    common.add_argument("--L", type=int, default=2)
    common.add_argument("--target", default="sq", choices=["sq", "p4", "sqN", "x2x", "nxx", "nxxmod"])
    common.add_argument("--data", default="h1", choices=["h1", "free"])
    common.add_argument("--bits", type=int, default=16)
    common.add_argument("--seeds", type=int, default=10)
    common.add_argument("--steps", type=int, default=1500)
    common.add_argument("--lr", type=float, default=0.05)
    common.add_argument("--loss", default="circ", choices=["circ", "mse", "rel", "log"])
    common.add_argument("--threads", type=int, default=2)
    common.add_argument("--verbose", action="store_true")
    s = sub.add_parser("darts", parents=[common])
    s.add_argument("--mode", default="mix", choices=["mix", "st"])
    s.add_argument("--tau0", type=float, default=1.0)
    s.add_argument("--tau1", type=float, default=0.05)
    s = sub.add_parser("interp", parents=[common])
    s.add_argument("--bits_list", type=int, nargs="+", default=[4, 8, 13, 16, 20, 24])
    s = sub.add_parser("basin", parents=[common])
    s.add_argument("--p_list", type=float, nargs="+", default=[0.1, 0.3, 0.45, 0.55, 0.7, 0.9, 0.99])
    s.add_argument("--tau1", type=float, default=0.05)
    s = sub.add_parser("mixture", parents=[common])
    s.add_argument("--rows", type=int, default=64)
    s = sub.add_parser("reinforce", parents=[common])
    s.add_argument("--K", type=int, default=64)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    {"darts": cmd_darts, "interp": cmd_interp, "basin": cmd_basin, "mixture": cmd_mixture,
     "reinforce": cmd_reinforce}[a.cmd](a)


if __name__ == "__main__":
    main()
