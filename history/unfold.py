"""Can a model unfold the remainder by itself? Write-up: UNFOLD.md.

y = s^2 mod N is s^2 folded by N. A model that predicts the pre-remainder value V(s, N) can train on the folded
label alone with the clock loss 1 - cos(2 pi (V - y) / N). Near its minima that is plain regression toward
V* = y + N round((V - y) / N): the model's own wrap count unfolds the label. Levenberg-Marquardt on that wrapped
residual is relax_periodic.gauss_newton. This file measures when the self-unfolding works (fp64, value space,
snapping round(V) mod N):

  basin      start at an exact solution, perturb so rows are off by about A*N, run LM on the clock residual:
             success vs the share of rows with |V - s^2| < N/2 at the start                     (test 1)
  topdigits  how often the wrap count is right when s is known only to its top j digits      (test 2)
  fold       fold curriculum: s < N^e, e from 1/4 to 1, from a small random init               (test 3)
  h1         H1's own 1,000 T=1 rows, easiest first by s^2/N, then certify T=64 on both profiles
  rung       test 2 as a model: a log-space rung (wrap count) head trained toward its own nearest unfolded rung,
             with no curriculum, an oracle easiest-first order, or a self-paced order by its own rung estimate
  lattice    analytic unfolding: smallest integer coefficients c with c . phi(s) = y (mod N) on a few random H1 rows,
             all wrap counts solved at once by LLL (no curriculum, no gradient); code in unfold_lattice.py

Models (closed-form Jacobians; s' is s read as a number, or sum_i B^i d_i from the digits with one learned radix B):
  value   V = w . (1, s, N, s^2, sN, N^2)      6 weights, s a number   (relax_periodic "mono")
  value3  V = w . (1, s, s^2)                   3 weights, s a number
  tied    V = s'^2                              1 weight: the radix B (place value i is B^i)
  tied3   V = w . (1, s', s'^2)                 3 weights + radix
  tied6   V = w . (1, s', N, s'^2, s'N, N^2)    6 weights + radix
  untied  V = (sum_i w_i d_i)^2                 8 free place values
  bidig   relax_periodic's 4 bilinear units on (1, 8 digits of s, N)   (70 weights, place values free)
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import relax_common as rc
import relax_periodic as P

torch.set_default_dtype(torch.float64)
D = 8                                            # digit slots of s (24-bit residues have 8 digits)
KINDS = ("value", "value3", "tied", "tied3", "tied6", "untied", "bidig")


def digs(S):
    return torch.stack([torch.floor(S / 10 ** i) % 10 for i in range(D)], -1)


class Tied:
    """V = (sum_i B^i d_i)^2 with one learned radix B."""

    def __init__(self, B=10.0):
        self.B = torch.tensor([float(B)])

    def v_and_jac(self, S, N):
        d, i = digs(S), torch.arange(D, dtype=torch.float64)
        sp = d @ self.B[0] ** i
        dsp = d @ (i * self.B[0] ** (i - 1).clamp_min(0))
        return sp * sp, (2 * sp * dsp)[:, None]

    def add_flat(self, delta):
        self.B += delta


class Untied:
    """V = (sum_i w_i d_i)^2 with 8 free place values."""

    def __init__(self, w):
        self.w = torch.as_tensor(w, dtype=torch.float64).clone()

    def v_and_jac(self, S, N):
        d = digs(S)
        sp = d @ self.w
        return sp * sp, 2 * sp[:, None] * d

    def add_flat(self, delta):
        self.w += delta


class Poly:
    """V = w . phi(s', N), phi = (1, s', s'^2) or (1, s', N, s'^2, s'N, N^2); s' = s, or sum_i B^i d_i (radix)."""

    def __init__(self, nf, radix, w, B=10.0):
        self.nf, self.radix = nf, radix
        self.w = torch.as_tensor(w, dtype=torch.float64).clone()
        self.B = torch.tensor([float(B)])

    def v_and_jac(self, S, N):
        if self.radix:
            d, i = digs(S), torch.arange(D, dtype=torch.float64)
            sp = d @ self.B[0] ** i
            dsp = d @ (i * self.B[0] ** (i - 1).clamp_min(0))
        else:
            sp = S
        one = torch.ones_like(sp)
        if self.nf == 3:
            f = torch.stack([one, sp, sp * sp], -1)
            dv = self.w[1] + 2 * self.w[2] * sp
        else:
            f = torch.stack([one, sp, N, sp * sp, sp * N, N * N], -1)
            dv = self.w[1] + 2 * self.w[3] * sp + self.w[4] * N
        v = f @ self.w
        return (v, torch.cat([f, (dv * dsp)[:, None]], 1)) if self.radix else (v, f)

    def add_flat(self, delta):
        p = len(self.w)
        self.w += delta[:p]
        if self.radix:
            self.B += delta[p:]


POLY = {"value3": (3, False), "tied3": (3, True), "tied6": (6, True)}


def exact_model(kind):
    if kind in POLY:
        nf, radix = POLY[kind]
        w = np.zeros(nf)
        w[2 if nf == 3 else 3] = 1.0
        return Poly(nf, radix, w, 10.0)
    if kind == "value":
        m = P.PeriodicStep("mono", init=0.0)
        with torch.no_grad():
            m.w.zero_()
            m.w[3] = 1.0
        return m
    if kind == "tied":
        return Tied(10.0)
    if kind == "untied":
        return Untied(10.0 ** np.arange(D))
    m = P.PeriodicStep("bidig", init=0.0)                    # m = (1, d_0..d_7, N); unit 0 is s * s
    with torch.no_grad():
        for p in (m.a, m.b, m.c):
            p.zero_()
        m.a[1:1 + D, 0] = 10.0 ** torch.arange(D)
        m.b[1:1 + D, 0] = 10.0 ** torch.arange(D)
    return m


def random_model(kind, bits, rng, seed):
    """Small random start. value/bidig: relax_periodic's data-dependent phase_small_init; tied: B in [1.5, 4];
    untied: place values in [0, 2]."""
    if kind in ("value", "bidig"):
        m = P.PeriodicStep("mono" if kind == "value" else "bidig", init=1e-3, seed=seed)
        m.phase_small_init(bits, 1e-3, seed)
        return m
    if kind == "tied":
        return Tied(rng.uniform(1.5, 4.0))
    if kind in POLY:                                         # phase_small_init's rule: each term ~1e-3 rad at start
        nf, radix = POLY[kind]
        m = Poly(nf, radix, np.zeros(nf), rng.uniform(1.5, 4.0))
        N, s, _ = rc.fresh_step(np.random.default_rng(seed), bits, 4096)
        _, J = m.v_and_jac(tt(s), tt(N))
        mag = (J[:, :nf].abs() / tt(N)[:, None]).mean(0).clamp_min(1e-300)
        m.w = torch.as_tensor(1e-3 * rng.standard_normal(nf)) / (2 * math.pi * mag)
        return m
    return Untied(rng.uniform(0.0, 2.0, D))


def flat(m):
    if isinstance(m, Tied):
        return m.B.clone()
    if isinstance(m, Untied):
        return m.w.clone()
    if isinstance(m, Poly):
        return torch.cat([m.w, m.B]) if m.radix else m.w.clone()
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def tt(v):
    return torch.as_tensor(np.asarray(v, dtype=np.float64))


LM_WEIGHT = "none"


def lm(m, N, s, y, iters=20, damp=1e-6, wrap=True):
    """Levenberg-Marquardt on the clock residual wrap(V - y) / N (wrap to [-N/2, N/2)): every step regresses V toward
    its own nearest unfolding V* = y + N round((V - y) / N). Same algorithm as relax_periodic.gauss_newton.
    LM_WEIGHT "sens" divides each row by its sensitivity |dV/dtheta| / N (column-normalized): the rows whose phase
    moves fastest are the first to alias onto a wrong wrap count, so each row gets one unit-length vote."""
    S, Nn, Y = tt(s), tt(N), tt(y)

    def resid():
        v, J = m.v_and_jac(S, Nn)
        r = torch.remainder(v - Y + Nn / 2, Nn) - Nn / 2 if wrap else v - Y
        r, J = r / Nn, J / Nn[:, None]
        if LM_WEIGHT == "sens":
            rho = 1 / ((J / J.norm(dim=0).clamp_min(1e-300)).norm(dim=1) + 1e-12)
            r, J = r * rho, J * rho[:, None]
        return r, J

    with torch.no_grad():
        r, J = resid()
        cost = float((r * r).sum())
        for _ in range(iters * 4):
            col = J.norm(dim=0).clamp_min(1e-300)
            Jn = J / col
            A = Jn.T @ Jn
            delta = torch.linalg.solve(A + damp * torch.eye(A.shape[0]), Jn.T @ r) / col
            m.add_flat(-delta)
            r2, J2 = resid()
            c2 = float((r2 * r2).sum())
            if c2 <= cost:
                r, J, cost, damp = r2, J2, c2, max(damp / 10, 1e-12)
            else:
                m.add_flat(delta)
                damp *= 10
            if cost == 0.0:
                break
    return cost


def snap(m, N, s):
    with torch.no_grad():
        v, _ = m.v_and_jac(tt(s), tt(N))
        return torch.remainder(torch.round(v), tt(N)).numpy().astype(np.int64)


def exact_frac(m, N, s, y):
    return float((snap(m, N, s) == y).mean())


# ---------------------------------------------------------------------------------------------- test 1: basin
def basin(a):
    out = []
    for kind in a.kinds:
        for bits in a.bits_list:
            rng = np.random.default_rng(bits)
            N, s, y = rc.fresh_step(rng, bits, 4096)
            Ne, se, ye = rc.fresh_step(np.random.default_rng(bits + 100), bits, 20000)
            S, Nn = tt(s), tt(N)
            sq = S * S
            for A in a.amps:
                succ, inb, inbj, med = 0, [], [], []
                for d in range(a.dirs):
                    m = exact_model(kind)
                    with torch.no_grad():
                        _, J = m.v_and_jac(S, Nn)
                        col = J.norm(dim=0)
                        live = col > 1e-12 * col.max()                     # skip directions the exact point ignores
                        u = torch.where(live, torch.as_tensor(rng.standard_normal(J.shape[1])) / col.clamp_min(1e-300),
                                        torch.zeros(J.shape[1]))
                        u /= (J @ u / Nn).abs().median()             # median row moves by N (linearized)
                        sens = (J @ u).abs()                          # each row's pull in LM's normal equations
                        m.add_flat(A * u)
                        err = torch.nan_to_num(((m.v_and_jac(S, Nn)[0] - sq) / Nn).abs(), nan=float("inf"))
                    inside = err < 0.5
                    inb.append(float(inside.double().mean()))
                    inbj.append(float((sens[inside] ** 2).sum() / (sens ** 2).sum()))
                    med.append(float(err.median()))
                    lm(m, N, s, y, iters=a.iters)
                    succ += exact_frac(m, Ne, se, ye) == 1.0
                r = {"kind": kind, "bits": bits, "A": A, "success": succ, "dirs": a.dirs,
                     "in_basin": float(np.mean(inb)), "in_basin_lm_weighted": float(np.mean(inbj)),
                     "median_err_N": float(np.median(med))}
                out.append(r)
                print(f"basin {kind:6} b={bits:2d} A={A:<6g} in-basin rows {r['in_basin']:.3f} (LM-weighted "
                      f"{r['in_basin_lm_weighted']:.3f})  median |V-s^2|/N {r['median_err_N']:.3g}  -> exact {succ}/{a.dirs}",
                      flush=True)
    return out


# ---------------------------------------------------------------------------- test 2: wrap count from top digits
def topdigits(a):
    """k = floor(s^2/N) is what unfolds a row. With s known to its top j significant digits (the rest replaced by
    their midpoint), the unfolded target y + N round(s_hat^2/N - y/N) is right iff |s_hat^2 - s^2| < N/2."""
    out = []
    sets = [(f"fresh{b}", rc.fresh_step(np.random.default_rng(b), b, 100000)) for b in a.bits_list]
    try:
        import h1
        tr = h1.load("train")
        t1 = tr["T"] == 1
        sets.append(("h1_t1", (tr["N"][t1], tr["X"][t1], tr["Y"][t1])))
    except FileNotFoundError:
        pass
    for name, (N, s, y) in sets:
        s_f, N_f = s.astype(np.float64), N.astype(np.float64)
        L = np.floor(np.log10(np.maximum(s_f, 1))) + 1                         # digits of s
        row = {"set": name}
        for j in range(1, D + 1):
            unit = 10.0 ** np.maximum(L - j, 0)
            s_hat = np.floor(s_f / unit) * unit + np.where(L > j, unit / 2, 0)  # top j digits, midpoint for the rest
            row[j] = float((np.abs(s_hat ** 2 - s_f ** 2) < N_f / 2).mean())
        out.append(row)
        print(f"topdigits {name:9}: share of rows unfolded right with the top j digits of s: "
              + " ".join(f"j={j}:{row[j]:.3f}" for j in range(1, D + 1)), flush=True)
    return out


# ------------------------------------------------------------------------------------- test 3: fold curriculum
def fold_batch(rng, bits, e, B, pool=None):
    N, s, y = rc.fresh_step(rng, bits, B, pool)
    S = np.maximum(2, N.astype(np.float64) ** e).astype(np.int64)
    s = (rng.random(B) * S).astype(np.int64)
    return N, s, rc.sqmod(s, N)


def fold(a):
    out = []
    for kind in a.kinds:
        for bits in a.bits_list:
            succ, t0, rows = 0, time.time(), []
            for seed in range(a.seeds):
                rng = np.random.default_rng(1000 * bits + seed)
                m = random_model(kind, bits, rng, seed)
                carry, broke = [], None
                for k in range(a.stages):
                    e = 0.25 + 0.75 * k / (a.stages - 1)
                    N, s, y = fold_batch(rng, bits, e, 4096)
                    before = exact_frac(m, N, s, y)                          # this fold, before training on it
                    lm(m, N, s, y, iters=a.iters)
                    after = exact_frac(m, N, s, y)
                    carry.append((round(e, 3), round(before, 3), round(after, 3)))
                    if broke is None and after < 1.0 and e > 0.5:
                        broke = round(e, 3)
                Ne, se, ye = rc.fresh_step(np.random.default_rng(seed + 77), bits, 20000)
                ex = exact_frac(m, Ne, se, ye)
                succ += ex == 1.0
                rows.append({"seed": seed, "exact": ex, "first_inexact_stage_e": broke, "carry": carry,
                             "theta": flat(m).tolist() if kind in ("tied", "untied") else None})
            r = {"kind": kind, "bits": bits, "success": succ, "seeds": a.seeds, "runs": rows,
                 "sec_per_seed": (time.time() - t0) / a.seeds}
            out.append(r)
            ex = [x["exact"] for x in rows]
            brk = [x["first_inexact_stage_e"] for x in rows]
            print(f"fold {kind:6} b={bits:2d}: {succ}/{a.seeds} exact; full-range exact per seed "
                  f"{np.round(ex, 3).tolist()}; first fold (s < N^e) not fitted: {brk}  ({r['sec_per_seed']:.1f}s/seed)",
                  flush=True)
            if kind in ("tied", "untied"):
                print("   learned:", np.array2string(np.array(rows[0]["theta"]), precision=6), flush=True)
    return out


# -------------------------------------------------------------------------------------------- H1's own rows
def h1_mode(a):
    import h1
    tr = h1.load("train")
    t1 = tr["T"] == 1
    N, X, Y = tr["N"][t1], tr["X"][t1], tr["Y"][t1]
    order = np.argsort(X.astype(np.float64) ** 2 / N)
    out = []
    for kind in a.kinds:
        for seed in range(a.seeds):
            t0 = time.time()
            rng = np.random.default_rng(seed)
            m = random_model(kind, 20, rng, seed)
            if a.stage0 == "plain":                              # the rows the sort puts first have x^2 < N: y is V
                idx = order[:int((X.astype(np.float64) ** 2 < N).sum())]
                lm(m, N[idx], X[idx], Y[idx], iters=a.iters * 10, wrap=False)
            for k in range(a.stages):
                n = int(round(5 * (len(N) / 5) ** (k / (a.stages - 1))))
                idx = order[:n]
                lm(m, N[idx], X[idx], Y[idx], iters=a.iters)
            train_ex = exact_frac(m, N, X, Y)
            res = {"kind": kind, "seed": seed, "stage0": a.stage0, "train_t1_exact": train_ex,
                   "train_sec": time.time() - t0,
                   "theta": flat(m).tolist() if kind in ("tied", "untied") else None}
            if train_ex == 1.0 or a.certify_all:
                def predict(Nn, Xx, Tt):
                    s = Xx % Nn
                    for t in range(int(Tt.max())):
                        live = Tt > t
                        s[live] = snap(m, Nn[live], s[live])
                    return h1.digits(s)
                for prof in ("depth_t", "depth_ood_n_t"):
                    res[prof] = h1.certify(predict, prof, verbose=False)
            out.append(res)
            cert = " ".join(f"{p} max T={res[p]['max_t']}" for p in ("depth_t", "depth_ood_n_t") if p in res)
            print(f"h1 {kind:6} stage0={a.stage0} seed {seed}: train T=1 exact {train_ex:.3f}  {cert}  "
                  f"({res['train_sec']:.1f}s)",
                  flush=True)
            if res["theta"] is not None:
                print("   learned:", np.array2string(np.array(res["theta"]), precision=10), flush=True)
    return out


# ------------------------------------------------------------------------ test 2 as a model: a rung (wrap count) head
class Rung:
    """Rung estimate in log space: log k = alpha log s' + beta log N + gamma, and V = N k. s' is s read as a number
    (value), sum_i B^i d_i (tied) or sum_i w_i d_i (untied). The exact rung is k = s^2/N: alpha 2, beta -1, gamma 0.
    In fp64 exp(log k) is precise enough for exact snapping up to about 20-bit moduli."""

    def __init__(self, kind, rng, init="random"):
        self.kind = kind
        self.abg = torch.as_tensor(rng.standard_normal(3))
        if init == "low":                                    # every row starts below rung 1 (k_hat = e^-10)
            self.abg = torch.tensor([0.0, 0.0, -10.0])
        self.B = torch.tensor([rng.uniform(1.5, 4.0)])
        self.w = torch.as_tensor(rng.uniform(0.0, 2.0, D))

    def u_and_jac(self, S, N):
        a, b, g = self.abg
        if self.kind == "value":
            sp, extra = S, None
        elif self.kind == "tied":
            d, i = digs(S), torch.arange(D, dtype=torch.float64)
            sp = d @ self.B[0] ** i
            extra = (a * (d @ (i * self.B[0] ** (i - 1).clamp_min(0))) / sp.clamp_min(1e-12))[:, None]
        else:
            d = digs(S)
            sp = d @ self.w
            extra = a * d / sp.clamp_min(1e-12)[:, None]
        ls, lN = torch.log(sp.clamp_min(1e-12)), torch.log(N)
        u = a * ls + b * lN + g
        J = torch.stack([ls, lN, torch.ones_like(ls)], 1)
        return u, (J if extra is None else torch.cat([J, extra], 1))

    def add_flat(self, delta):
        self.abg += delta[:3]
        if self.kind == "tied":
            self.B += delta[3:]
        elif self.kind == "untied":
            self.w += delta[3:]

    def params(self):
        extra = self.B if self.kind == "tied" else self.w if self.kind == "untied" else torch.zeros(0)
        return torch.cat([self.abg, extra]).tolist()


def rung_step(m, N, s, y, inner=3, damp=1e-6):
    """One self-unfolding round: assign every row its nearest unfolded rung k* = round(k_hat - y/N) (>= 0), then a few
    LM steps on the log-space residual log k_hat - log(k* + y/N) with those targets held fixed. A row assigned the wrong
    rung is off by about 1/k in log space, so heavily wrapped rows that alias barely pull."""
    S, Nn, Y = tt(s), tt(N), tt(y)
    with torch.no_grad():
        u, J = m.u_and_jac(S, Nn)
        kap = torch.clamp(torch.round(torch.exp(u) - Y / Nn), min=0) + Y / Nn
        ok = (S >= 1) & (kap > 0) & torch.isfinite(u)
        tgt = torch.log(kap.clamp_min(1e-300))

        def resid():
            u2, J2 = m.u_and_jac(S, Nn)
            r = torch.where(ok, u2 - tgt, torch.zeros_like(u2))
            return r, J2 * ok[:, None]

        r, J = resid()
        cost = float((r * r).sum())
        for _ in range(inner * 4):
            if not torch.isfinite(r).all() or not torch.isfinite(J).all():
                break
            col = J.norm(dim=0).clamp_min(1e-300)
            Jn = J / col
            A = Jn.T @ Jn
            delta = torch.linalg.solve(A + damp * torch.eye(A.shape[0]), Jn.T @ r) / col
            m.add_flat(-delta)
            r2, J2 = resid()
            c2 = float((r2 * r2).sum())
            if c2 <= cost and np.isfinite(c2):
                r, J, cost, damp = r2, J2, c2, max(damp / 10, 1e-12)
            else:
                m.add_flat(delta)
                damp *= 10


def rung_snap(m, N, s):
    with torch.no_grad():
        u, _ = m.u_and_jac(tt(s), tt(N))
        v = torch.nan_to_num(tt(N) * torch.exp(u), nan=0.0, posinf=0.0, neginf=0.0)
        return torch.remainder(torch.round(v), tt(N)).numpy().astype(np.int64)


def rung(a):
    """Rung-estimate self-unfolding on fresh rows (`--bits_list`) or H1's 1,000 T=1 rows (`--data h1`).
    Schedules: none (all rows every round), oracle (easiest first by the true s^2/N: s < N^e on fresh data, the sorted
    prefix on H1), self (the rows the model itself rates lowest-rung first, so no oracle at all)."""
    out = []
    if a.data == "h1":
        import h1
        tr = h1.load("train")
        t1 = tr["T"] == 1
        Nh, Xh, Yh = tr["N"][t1], tr["X"][t1], tr["Y"][t1]
        order = np.argsort(Xh.astype(np.float64) ** 2 / Nh)
        sizes = ["h1"]
    else:
        sizes = a.bits_list
    for kind in a.rung_kinds:
        for sched in a.schedules:
            for bits in sizes:
                succ, t0, runs = 0, time.time(), []
                for seed in range(a.seeds):
                    rng = np.random.default_rng(10_000 + seed)
                    m = Rung(kind, rng, a.rung_init)
                    for k in range(a.stages):
                        frac = k / (a.stages - 1)
                        if bits == "h1":
                            if sched == "oracle":
                                idx = order[:int(round(5 * (len(Nh) / 5) ** frac))]
                            elif sched == "self":
                                with torch.no_grad():
                                    u, _ = m.u_and_jac(tt(Xh), tt(Nh))
                                kk = torch.nan_to_num(u, nan=1e9).numpy()
                                idx = np.argsort(kk)[:int(round(5 * (len(Nh) / 5) ** frac))]
                            else:
                                idx = np.arange(len(Nh))
                            N, s, y = Nh[idx], Xh[idx], Yh[idx]
                        else:
                            if sched == "oracle":
                                N, s, y = fold_batch(rng, bits, 0.25 + 0.75 * frac, 4096)
                            else:
                                N, s, y = rc.fresh_step(rng, bits, 4096 if sched == "none" else 16384)
                                if sched == "self":
                                    with torch.no_grad():
                                        u, _ = m.u_and_jac(tt(s), tt(N))
                                    keep = np.argsort(torch.nan_to_num(u, nan=1e9).numpy())
                                    keep = keep[:max(64, int(round(len(keep) * frac ** 2)))]
                                    N, s, y = N[keep], s[keep], y[keep]
                        rung_step(m, N, s, y, inner=a.iters)
                    if bits == "h1":
                        ex = float((rung_snap(m, Nh, Xh) == Yh).mean())
                    else:
                        Ne, se, ye = rc.fresh_step(np.random.default_rng(seed + 77), bits, 20000)
                        ex = float((rung_snap(m, Ne, se) == ye).mean())
                    succ += ex == 1.0
                    run_r = {"seed": seed, "exact": ex, "params": m.params()}
                    if bits == "h1" and ex == 1.0:                   # loop the snapped step: T = 1..64, both profiles
                        def predict(Nn, Xx, Tt):
                            st = Xx % Nn
                            for t in range(int(Tt.max())):
                                live = Tt > t
                                st[live] = rung_snap(m, Nn[live], st[live])
                            return h1.digits(st)
                        for prof in ("depth_t", "depth_ood_n_t"):
                            run_r[prof] = h1.certify(predict, prof, verbose=False)["max_t"]
                    runs.append(run_r)
                r = {"kind": kind, "schedule": sched, "init": a.rung_init, "bits": bits, "success": succ,
                     "seeds": a.seeds, "runs": runs,
                     "sec_per_seed": (time.time() - t0) / a.seeds}
                out.append(r)
                print(f"rung {kind:6} {sched:6} init={a.rung_init} {str(bits):>3}: {succ}/{a.seeds} exact; per seed "
                      f"{np.round([x['exact'] for x in runs], 3).tolist()}; certified max T (depth_t, ood) "
                      f"{[(x.get('depth_t'), x.get('depth_ood_n_t')) for x in runs if 'depth_t' in x]}; alpha, beta, gamma (seed 0) "
                      f"{np.round(runs[0]['params'][:3], 6).tolist()}  ({r['sec_per_seed']:.1f}s/seed)", flush=True)
    return out


# ------------------------------------------------------------------- analytic unfolding: integer ambiguity resolution
def lattice(a):
    """Integer-coefficient hypotheses on random H1 rows, all wrap counts solved at once by LLL (unfold_lattice.py)."""
    import unfold_lattice
    return unfold_lattice.run(a.lattice_feats, a.lattice_rows, a.seeds, a.h1_bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["basin", "topdigits", "fold", "h1", "lattice", "rung"])
    ap.add_argument("--rung_kinds", nargs="+", default=["value", "tied", "untied"])
    ap.add_argument("--schedules", nargs="+", default=["none", "oracle", "self"])
    ap.add_argument("--data", default="fresh", choices=["fresh", "h1"])
    ap.add_argument("--rung_init", default="random", choices=["random", "low"])
    ap.add_argument("--lattice_feats", nargs="+", default=["value", "digits"])
    ap.add_argument("--lattice_rows", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--h1_bits", type=int, default=0, help="lattice: only H1 rows with this modulus size")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--bits_list", type=int, nargs="+", default=[8, 12, 16, 20, 24])
    ap.add_argument("--amps", type=float, nargs="+", default=[0.1, 0.3, 1, 3, 10, 30, 100, 1000])
    ap.add_argument("--dirs", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--stages", type=int, default=48)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--certify_all", action="store_true")
    ap.add_argument("--stage0", default="clock", choices=["clock", "plain"],
                    help="h1: fit the x^2 < N rows by plain regression before the clock-loss stages")
    ap.add_argument("--lm_weight", default="none", choices=["none", "sens"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    global LM_WEIGHT
    LM_WEIGHT = a.lm_weight
    t0 = time.time()
    res = {"basin": basin, "topdigits": topdigits, "fold": fold, "h1": h1_mode, "lattice": lattice,
           "rung": rung}[a.mode](a)
    if a.out:
        json.dump({"args": vars(a), "results": res, "wall": time.time() - t0}, open(a.out, "w"), indent=1)
    print(f"wall {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
