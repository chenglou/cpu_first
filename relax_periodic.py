"""Relaxation 3: periodic units tied to N (SIREN-style cos/sin(2*pi*v/N), N an input), fp64, integer snapping.

The step network sees s and N as numbers (atoi/itoa are plumbing here, like N entering the activation). Its
pre-activation v is a learned quadratic of (1, s, N):
  mono      v = w . (1, s, N, s^2, sN, N^2)                         6 weights
  bilinear  v = sum_k (a_k . m)(b_k . m) + c . m,  m = (1, s, N)    K bilinear units (a generic GLU-style product)
The output unit is z = (cos, sin)(2*pi*v/N); snapping decodes the phase to the nearest residue,
y = round(N * angle(z) / 2pi) mod N, which is fed back as the next s. Loss: 1 - cos(2*pi*(v - y)/N).
Exact solutions form a lattice: v = s^2 + integer multiples of N, sN, N^2.

  --mode precision   how exact the weights must be vs bits (fp64 and fp32), and the gradient basin width
  --mode train       gradient descent from small random init, fresh data at b bits, with/without curriculum
  --mode h1          the real rules' data only (H1's 1,000 T=1 rows [+ T=2 rows]), then certify T=64 on H1
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import relax_common as rc

torch.set_default_dtype(torch.float64)
TWO_PI = 2 * math.pi


class PeriodicStep(torch.nn.Module):
    def __init__(self, kind="mono", K=4, init=1e-3, seed=0, dtype=torch.float64):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.kind = kind
        if kind == "mono":
            self.w = torch.nn.Parameter(init * torch.randn(6, generator=g, dtype=dtype))
        else:
            M = 10 if kind == "bidig" else 3
            self.a = torch.nn.Parameter(init * torch.randn(M, K, generator=g, dtype=dtype))
            self.b = torch.nn.Parameter(init * torch.randn(M, K, generator=g, dtype=dtype))
            self.c = torch.nn.Parameter(init * torch.randn(M, generator=g, dtype=dtype))

    def feats(self, s, N):
        one = torch.ones_like(s)
        if self.kind == "bidig":  # the 8 decimal digits of s (LSB first) as separate inputs
            d = torch.stack([torch.floor(s / 10 ** i) % 10 for i in range(8)], -1)
            return torch.cat([one[:, None], d, N[:, None]], -1)
        return torch.stack([one, s, N], -1)

    @torch.no_grad()
    def phase_small_init(self, bits, scale=1e-3, seed=0):
        """Data-dependent init (like LSUV): every term starts contributing ~scale radians at the target size."""
        N, s, _ = rc.fresh_step(np.random.default_rng(seed), bits, 4096)
        N, s = tt(N), tt(s)
        g = torch.Generator().manual_seed(seed)
        one = torch.ones_like(s)
        if self.kind == "mono":
            f = torch.stack([one, s, N, s * s, s * N, N * N], -1)
            self.w.copy_(scale * torch.randn(6, generator=g) / (TWO_PI * (f.abs() / N[:, None]).mean(0)))
        else:
            m = self.feats(s, N)
            mag = m.abs().mean(0).clamp_min(1e-3)
            r = math.sqrt(N.mean())
            self.a.copy_(torch.randn(self.a.shape, generator=g) * math.sqrt(scale) * r / mag[:, None] / 3)
            self.b.copy_(torch.randn(self.b.shape, generator=g) * math.sqrt(scale) * r / mag[:, None] / 3)
            self.c.copy_(scale * torch.randn(self.c.shape, generator=g) / (TWO_PI * (m.abs() / N[:, None]).mean(0).clamp_min(1e-12)))

    def v(self, s, N):
        one = torch.ones_like(s)
        if self.kind == "mono":
            f = torch.stack([one, s, N, s * s, s * N, N * N], -1)
            return f @ self.w
        m = self.feats(s, N)
        return ((m @ self.a) * (m @ self.b)).sum(-1) + m @ self.c

    def v_and_jac(self, s, N):
        one = torch.ones_like(s)
        if self.kind == "mono":
            f = torch.stack([one, s, N, s * s, s * N, N * N], -1)
            return f @ self.w, f
        m = self.feats(s, N)
        pa, pb = m @ self.a, m @ self.b  # [B,K]
        v = (pa * pb).sum(-1) + m @ self.c
        Ja = (m[:, :, None] * pb[:, None, :]).reshape(len(s), -1)  # d v / d a[j,k] = m_j * pb_k
        Jb = (m[:, :, None] * pa[:, None, :]).reshape(len(s), -1)
        return v, torch.cat([Ja, Jb, m], 1)

    @torch.no_grad()
    def add_flat(self, d):
        if self.kind == "mono":
            self.w += d
            return
        na = self.a.numel()
        self.a += d[:na].view_as(self.a)
        self.b += d[na:2 * na].view_as(self.b)
        self.c += d[2 * na:]

    def loss(self, s, N, y):
        # cos(2*pi*(v-y)/N), range-reduced mod N first (what libm does for any large argument)
        r = torch.remainder(self.v(s, N) - y, N)
        return (2 * torch.sin(math.pi * r / N) ** 2).mean()

    @torch.no_grad()
    def snap(self, s, N):
        r = torch.remainder(self.v(s, N), N)
        ang = torch.atan2(torch.sin(TWO_PI * r / N), torch.cos(TWO_PI * r / N))
        return torch.remainder(torch.round(N * ang / TWO_PI), N)


def tt(v, dtype=torch.float64):
    return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=dtype)


def exact_frac(model, bits, n=20000, seed=5, pool=None, dtype=torch.float64):
    N, s, y = rc.fresh_step(np.random.default_rng(seed), bits, n, pool)
    pred = model.snap(tt(s, dtype), tt(N, dtype)).numpy().astype(np.int64)
    return float((pred == y).mean())


def step_fn(model):
    return lambda N, s: model.snap(tt(s), tt(N)).numpy().astype(np.int64)


# ---------------------------------------------------------------- precision
def precision(args):
    out = {}
    for bits in range(6, 29, 2):
        row = {}
        # exact solution, evaluated in fp64 and fp32
        for name, dt in (("fp64", torch.float64), ("fp32", torch.float32)):
            m = PeriodicStep("mono", init=0.0, dtype=dt)
            with torch.no_grad():
                m.w.zero_()
                m.w[3] = 1.0
            row[f"exact_solution_{name}"] = exact_frac(m, bits, dtype=dt)
        # largest relative error on w_ss (and on an atoi-like scale of s) that stays 100% exact
        m = PeriodicStep("mono", init=0.0)
        tol = None
        for e in np.arange(-2.0, -60.0, -0.25):
            with torch.no_grad():
                m.w.zero_()
                m.w[3] = 1.0 + 2.0 ** e
            if exact_frac(m, bits, n=5000) == 1.0:
                tol = float(2.0 ** e)
                break
        row["max_rel_err_w_ss"] = tol
        # basin: from w_ss = 1 + d, does plain gradient descent on fresh data return to an exact solution?
        basin = None
        for e in np.arange(-1.0, -40.0, -0.5):
            d = 2.0 ** e
            m = PeriodicStep("mono", init=0.0)
            with torch.no_grad():
                m.w.zero_()
                m.w[3] = 1.0 + d
            ok = train_loop(m, bits, steps=300, lr=d / 4, verbose=False, seed=int(-e * 10))["exact"] == 1.0
            if ok:
                basin = float(d)
                break
        row["gd_basin_rel_w_ss"] = basin
        out[bits] = row
        print(f"bits {bits:2d}: " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return out


# ---------------------------------------------------------------- training
def train_loop(model, bits, steps=3000, lr=1e-2, batch=1024, curriculum="none", seed=0, verbose=True,
               pool=None, data=None, eval_bits=None, polish=True):
    """curriculum: none | smax (s < S_max grows from sqrt(N) to N) | bits (modulus size grows 4 -> bits)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for step in range(1, steps + 1):
        frac = step / steps
        if data is not None:
            i = rng.integers(0, len(data[0]), batch)
            N, s, y = data[0][i], data[1][i], data[2][i]
        elif curriculum == "bits":
            b = int(round(4 + (bits - 4) * min(1.0, frac / 0.7)))
            N, s, y = rc.fresh_step(rng, b, batch, pool)
        else:
            N, s, y = rc.fresh_step(rng, bits, batch, pool)
            if curriculum == "smax":
                # s uniform in [0, S) with log S growing from 0.5 log N to log N over 70% of training
                expo = 0.5 + 0.5 * min(1.0, frac / 0.7)
                S = np.maximum(2, (N.astype(np.float64) ** expo)).astype(np.int64)
                s = (rng.random(batch) * S).astype(np.int64)
                y = rc.sqmod(s, N)
        lr_t = lr * (0.5 * (1 + math.cos(math.pi * frac)) * 0.999 + 0.001)
        for g in opt.param_groups:
            g["lr"] = lr_t
        loss = model.loss(tt(s), tt(N), tt(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (step % max(1, steps // 10) == 0):
            ex = exact_frac(model, eval_bits or bits, n=4000)
            hist.append((step, float(loss), ex))
            print(f"    step {step} loss {float(loss):.3e} exact@{eval_bits or bits}b {ex:.4f}", flush=True)
    pre = exact_frac(model, eval_bits or bits, n=20000, pool=pool)
    if polish:
        if data is not None:
            Np, sp, yp = data
        else:
            Np, sp, yp = rc.fresh_step(np.random.default_rng(seed + 1), bits, 8192, pool)
        lbfgs(model, Np, sp, yp)
    return {"exact": exact_frac(model, eval_bits or bits, n=20000, pool=pool), "exact_before_polish": pre,
            "hist": hist}


def curriculum_batch(rng, bits, curriculum, frac, B, pool=None):
    if curriculum == "bits":
        return rc.fresh_step(rng, int(round(4 + (bits - 4) * frac)), B, pool)
    N, s, y = rc.fresh_step(rng, bits, B, pool)
    if curriculum == "smax":
        S = np.maximum(2, N.astype(np.float64) ** (0.25 + 0.75 * frac)).astype(np.int64)
        s = (rng.random(B) * S).astype(np.int64)
        y = rc.sqmod(s, N)
    return N, s, y


def train_stagewise(model, bits, curriculum="none", stages=24, B=4096, iters=50, seed=0, pool=None, opt="gn"):
    """Scale-free optimizer per curriculum stage (Gauss-Newton or full-batch L-BFGS), warm-started, then a polish."""
    rng = np.random.default_rng(seed)
    step = (lambda *d: gauss_newton(model, *d, iters=3)) if opt == "gn" else (lambda *d: lbfgs(model, *d, iters=iters))
    for k in range(stages):
        step(*curriculum_batch(rng, bits, curriculum, k / (stages - 1), B, pool))
    pre = exact_frac(model, bits, n=20000, pool=pool)
    step(*rc.fresh_step(rng, bits, 8192, pool))
    return {"exact": exact_frac(model, bits, n=20000, pool=pool), "exact_before_polish": pre}


def gauss_newton(model, N, s, y, iters=3, damp=1e-6):
    """Levenberg-Marquardt on the circular loss: residual r_i = wrap(v_i - y_i) / N_i (the wrap count each example
    implies is read off the current model), closed-form Jacobian of v, normalized columns, adaptive damping."""
    S, Nn, Y = tt(s), tt(N), tt(y)

    def resid():
        v, J = model.v_and_jac(S, Nn)
        return (torch.remainder(v - Y + Nn / 2, Nn) - Nn / 2) / Nn, J / Nn[:, None]

    with torch.no_grad():
        r, J = resid()
        cost = float((r * r).sum())
        for _ in range(iters * 4):
            col = J.norm(dim=0).clamp_min(1e-300)
            Jn = J / col
            A = Jn.T @ Jn
            delta = torch.linalg.solve(A + damp * torch.eye(A.shape[0]), Jn.T @ r) / col
            model.add_flat(-delta)
            r2, J2 = resid()
            c2 = float((r2 * r2).sum())
            if c2 <= cost:
                r, J, cost, damp = r2, J2, c2, max(damp / 10, 1e-12)
            else:
                model.add_flat(delta)
                damp *= 10
            if cost == 0.0:
                break


def lbfgs(model, N, s, y, iters=300):
    """Full-batch L-BFGS in fp64: the generic second-order polish that Adam's ~lr jitter cannot replace."""
    opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=iters, tolerance_grad=0, tolerance_change=0,
                            history_size=20, line_search_fn="strong_wolfe")
    S, Nn, Y = tt(s), tt(N), tt(y)

    def closure():
        opt.zero_grad()
        l = model.loss(S, Nn, Y)
        l.backward()
        return l

    opt.step(closure)


def train(args):
    out = {}
    for bits in args.bits_list:
        for cur in args.curricula:
            succ, t0, details = 0, time.time(), []
            for seed in range(args.seeds):
                m = PeriodicStep(args.kind, K=args.K, init=args.init, seed=seed)
                if args.init_mode == "phase_small":
                    m.phase_small_init(bits, args.init, seed)
                if args.opt in ("lbfgs", "gn"):
                    r = train_stagewise(m, bits, curriculum=cur, seed=seed, stages=args.stages, opt=args.opt)
                else:
                    r = train_loop(m, bits, steps=args.steps, lr=args.lr, curriculum=cur, seed=seed, verbose=False)
                succ += r["exact"] == 1.0
                details.append((r["exact_before_polish"], r["exact"]))
            key = f"{args.kind}_{args.opt}_{args.init_mode}_b{bits}_{cur}"
            out[key] = {"success": succ, "seeds": args.seeds, "exact_per_seed": details,
                        "time_per_seed": (time.time() - t0) / args.seeds}
            print(f"{key}: {succ}/{args.seeds} reach 100% exact; per-seed (adam, +lbfgs) {np.round(details, 4).tolist()} "
                  f"({out[key]['time_per_seed']:.1f}s/seed)", flush=True)
    return out


def h1_mode(args):
    """Only what the real rules give: H1's 1,000 T=1 training rows (13/16/20-bit moduli). Easy-first = stages over
    the rows sorted by x^2/N (how many times the square wraps around N), a curriculum that exists inside the data."""
    import h1
    tr = h1.load("train")
    t1 = tr["T"] == 1
    N, X, Y = tr["N"][t1], tr["X"][t1], tr["Y"][t1]
    wraps = X.astype(np.float64) ** 2 / N
    order = np.argsort(wraps)
    print(f"T=1 rows: {len(N)}; rows with x^2 < N: {(wraps < 1).sum()}, < 16N: {(wraps < 16).sum()}, "
          f"< 256N: {(wraps < 256).sum()}", flush=True)
    out = {}
    for seed in range(args.seeds):
        t0 = time.time()
        m = PeriodicStep(args.kind, K=args.K, init=args.init, seed=seed)
        m.phase_small_init(20, args.init, seed)
        if args.order == "easy_first":
            for k in range(args.stages):
                n = int(round(16 * (len(N) / 16) ** (k / (args.stages - 1))))
                idx = order[:n]
                gauss_newton(m, N[idx], X[idx], Y[idx], iters=3)
        else:
            for k in range(args.stages):
                gauss_newton(m, N, X, Y, iters=3)
        train_t = time.time() - t0
        ex_train = float((m.snap(tt(X), tt(N)).numpy().astype(np.int64) == Y).mean())

        def predict(Nn, Xx, Tt):
            s = Xx % Nn
            for tv in np.unique(Tt):
                msk = Tt == tv
                c = s[msk].copy()
                for _ in range(int(tv)):
                    c = step_fn(m)(Nn[msk], c)
                s[msk] = c
            return h1.digits(s)

        res = {"train_t1_exact": ex_train, "train_time": train_t,
               "params": {k: v.detach().numpy().tolist() for k, v in m.named_parameters()}}
        for prof in ("depth_t", "depth_ood_n_t"):
            res[prof] = h1.certify(predict, prof, verbose=False)
        res["eval_time"] = time.time() - t0 - train_t
        key = f"h1_t1rows_{args.order}_{args.kind}_seed{seed}"
        out[key] = res
        print(key, json.dumps({k: v for k, v in res.items() if k != "params"}), flush=True)
        if args.kind == "mono":
            print("   w =", np.array2string(m.w.detach().numpy(), precision=17), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="precision", choices=["precision", "train", "h1"])
    ap.add_argument("--kind", default="mono", choices=["mono", "bilinear", "bidig"])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--init", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--bits_list", type=int, nargs="+", default=[6, 8, 12, 16, 20, 24])
    ap.add_argument("--curricula", nargs="+", default=["none", "smax", "bits"])
    ap.add_argument("--h1_rows", nargs="+", default=["t1", "t1t2"])
    ap.add_argument("--order", default="easy_first", choices=["easy_first", "uniform"])
    ap.add_argument("--opt", default="gn", choices=["gn", "lbfgs", "adam"])
    ap.add_argument("--init_mode", default="phase_small", choices=["phase_small", "plain"])
    ap.add_argument("--stages", type=int, default=24)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    t0 = time.time()
    res = {"precision": precision, "train": train, "h1": h1_mode}[args.mode](args)
    res = {"args": vars(args), "results": res, "wall": time.time() - t0}
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1, default=str)
    print(f"wall {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
