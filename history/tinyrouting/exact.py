"""Exact computations on the tiny routing machine (every example, float64).

    python -m tinyrouting.exact grad0       first gradients on the routing, by mux and objective
    python -m tinyrouting.exact eps         loss along the straight path from the uniform wiring to the true one
    python -m tinyrouting.exact impostors   every wiring with the true table, and with its best-fit table
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from tinyrouting import machine as M
from tinyrouting.machine import MUXES, ORDER, Data, onehot, program_P, programs, run, table_TDC
from tinyrouting.train import Cfg, Checks, Trainer, below_N, cell_loss, laws_rows

L4 = ("shift", "a", "b", "keep")
ALL = ORDER


def route_push(tr: Trainer):
    """After loss.backward(): for every learned mux, the descent direction on its canonical options' probabilities
    (the change of each option's probability under a small gradient step on the routing parameters)."""
    m = tr.model
    out = {}
    with torch.no_grad():
        for mm in m.learn:
            n = len(MUXES[mm.split("@")[0]])
            if mm in m.alpha:
                a = m.alpha[mm]
                p = a.softmax(-1)
                g = a.grad if a.grad is not None else torch.zeros_like(a)
                dp = p * (-g - (p * -g).sum())                                  # softmax Jacobian times -g
                out[mm] = m.canon(mm, dp)
            else:
                pi = m.beta.softmax(0)
                g = m.beta.grad if m.beta.grad is not None else torch.zeros_like(pi)
                dpi = pi * (-g - (pi * -g).sum())
                v = torch.zeros(n)
                for p_, d in zip(m.progs, dpi):
                    v[p_[mm]] += d
                out[mm] = v
    return out


def table_toward(tr: Trainer):
    """Share of table rows whose descent direction raises the true (digit, borrow) logit relative to the rest."""
    m = tr.model
    if m.td.grad is None:
        return float("nan")
    with torch.no_grad():
        gd = -m.td.grad
        rel = (gd * m.TDt).sum(-1) - gd.mean(-1)
        return (rel > 0).double().mean().item()


def grad0(sizes=((3, 2, 1), (3, 2, 2), (4, 2, 1)), seeds=10):
    """First gradients. For every objective alone, at the exact uniform start and at random starts (logits
    N(0, 0.1^2)): which options each mux's lesson pushes up, and how big the routing lesson is next to the table's."""
    objectives = {"answers": dict(answers=1.0), "laws": dict(answers=0.0, checks={"laws": 1.0}),
                  "range": dict(answers=0.0, checks={"range": 1.0})}
    for B, w, k in sizes:
        for param in ("soft", "mix"):
            for oname, okw in objectives.items():
                for start in ("uniform", "random"):
                    pushes, norms, toward = [], [], []
                    for seed in (range(seeds) if start == "random" else [0]):
                        cfg = Cfg(B=B, w=w, k=k, learn=ALL, param=param, seed=seed, sym_init=start == "uniform",
                                  **okw)
                        tr = Trainer(cfg)
                        L, _ = tr.loss()
                        L.backward()
                        pushes.append(route_push(tr))
                        gr = torch.cat([p.grad.flatten() for p in tr.model.route_params() if p.grad is not None]
                                       or [torch.zeros(1)])
                        gt = torch.cat([p.grad.flatten() for p in tr.model.table_params() if p.grad is not None]
                                       or [torch.zeros(1)])
                        norms.append((gr.norm().item(), gt.norm().item()))
                        toward.append(table_toward(tr))
                    desc = []
                    for mm in ALL:
                        v = torch.stack([p[mm] for p in pushes]).mean(0)
                        top = int(v.argmax())
                        scale = v.abs().max().item()
                        desc.append(f"{mm}:{MUXES[mm][top] if scale > 1e-12 else '-'}"
                                    f"({scale:.1e})")
                    nr = np.mean([a for a, _ in norms]), np.mean([b for _, b in norms])
                    print(f"{(B, w, k)} {param:4s} {oname:8s} {start:7s} |g_route| {nr[0]:.2e} |g_table| {nr[1]:.2e}"
                          f" table rows toward truth {np.nanmean(toward):.2f} | " + " ".join(desc), flush=True)


def grad_true_table(sizes=((3, 2, 1), (3, 2, 2), (4, 2, 1)), params=("soft", "fmix")):
    """With the true table and the exactly uniform wiring: the first lesson of every mux from each objective alone.
    Prints, per mux, the option pushed up most and whether it is the true one (T) or not (x)."""
    objectives = {"answers": dict(answers=1.0), "laws": dict(answers=0.0, checks={"laws": 1.0}),
                  "range": dict(answers=0.0, checks={"range": 1.0})}
    for B, w, k in sizes:
        for param in params:
            for oname, okw in objectives.items():
                cfg = Cfg(B=B, w=w, k=k, learn=ALL, param=param, sym_init=True, table="exact", **okw)
                tr = Trainer(cfg)
                L, _ = tr.loss()
                L.backward()
                push = route_push(tr)
                desc = []
                for mm in ALL:
                    v = push[mm]
                    scale = v.abs().max().item()
                    if scale < 1e-12:
                        desc.append(f"{mm}:-")
                        continue
                    top = int(v.argmax())
                    desc.append(f"{mm}:{MUXES[mm][top]}{'T' if top == 0 else 'x'}({scale:.0e})")
                print(f"{(B, w, k)} {param:4s} {oname:8s} " + " ".join(desc), flush=True)


def eps_path(sizes=((3, 2, 1), (3, 2, 2), (4, 2, 1), (3, 3, 1)), learn=ALL):
    """With the true table: the answer loss (nats per digit) along routing = (1 - eps) * true + eps * uniform, for
    soft routing (mixed on the wire) and for the program mixture (mixed at the answer). Also for each mux alone."""
    epss = [1.0, 0.5, 0.3, 0.1, 0.03, 0.01, 1e-3, 0.0]
    for B, w, k in sizes:
        d = Data(B, w, k)
        TDC = table_TDC(*M.truth_table(B))
        for which in [learn] + [(mm,) for mm in learn]:
            rows_s, rows_m = [], []
            progs = programs(which)
            Pm = program_P(progs, which)
            with torch.no_grad():
                llm = M.row_loglik(run(d, Pm, TDC), d)                         # [H, X]
            itrue = [i for i, p in enumerate(progs) if all(v == 0 for v in p.values())][0]
            for e in epss:
                P = {}
                for mm in ORDER:
                    n = len(MUXES[mm])
                    t = onehot(torch.tensor([0]), n)
                    P[mm] = ((1 - e) * t + e / n) if mm in which else t
                with torch.no_grad():
                    ll = M.row_loglik(run(d, P, TDC), d)[0]
                rows_s.append(-ll.mean().item() / w)
                pi = torch.full((len(progs),), e / len(progs))
                pi[itrue] += 1 - e
                rows_m.append(-torch.logsumexp(pi.log()[:, None] + llm, 0).mean().item() / w)
            name = "all" if len(which) > 1 else which[0]
            print(f"{(B, w, k)} {name:6s} soft  " + " ".join(f"{e:g}:{x:.3f}" for e, x in zip(epss, rows_s)))
            print(f"{(B, w, k)} {name:6s} mix   " + " ".join(f"{e:g}:{x:.3f}" for e, x in zip(epss, rows_m)),
                  flush=True)


def fit_tables(data: Data, progs, learn, steps=600, lr=0.05, seed=0, cell=0.0, init=0.1):
    """For every program at once (one table each), train the table with the wiring fixed, from the answers (plus the
    cell check with weight `cell`). Returns the per-program tables (TD, TC) [H, ...] and train exact [H]."""
    B = data.B
    H = len(progs)
    g = torch.Generator().manual_seed(seed)
    td = (torch.randn(H, B, B, 2, B, generator=g) * init).requires_grad_(True)
    tc = (torch.randn(H, B, B, 2, 2, generator=g) * init).requires_grad_(True)
    P = program_P(progs, learn)
    opt = torch.optim.Adam([td, tc], lr=lr)
    ck = Checks(data) if cell else None
    for _ in range(steps):
        TD, TC = td.softmax(-1), tc.softmax(-1)
        TDC = torch.cat([TD, TC], -1).reshape(H, B * B * 2, B + 2)
        L = -M.row_loglik(run(data, P, TDC), data).mean(1).sum()
        if cell:
            A1, B1 = onehot(ck.cA, B)[None].expand(H, -1, -1, -1), onehot(ck.cB, B)[None].expand(H, -1, -1, -1)
            T, c = M.ripple(A1, B1, TDC)
            from tinyrouting.train import add_soft
            S, _ = add_soft(T, ck.cB)
            L = L + cell * -S.clamp_min(1e-300).log().gather(-1, ck.cA[None, ..., None].expand(H, -1, -1, 1)).mean(
                (1, 2, 3)).sum()
            pb = torch.stack([1 - c, c], -1).clamp_min(1e-300).log()
            L = L + cell * -pb.gather(-1, ck.cBorrow[None, :, None].expand(H, -1, 1)).mean((1, 2)).sum()
        opt.zero_grad()
        L.backward()
        opt.step()
    with torch.no_grad():
        TDh, TCh = onehot(td.argmax(-1), B), onehot(tc.argmax(-1), 2)
        TDC = torch.cat([TDh, TCh], -1).reshape(H, B * B * 2, B + 2)
        ex = M.exact_rows(run(data, P, TDC), data).double().mean(1)
    return TDh, TCh, ex


def exact_with(data: Data, progs, learn, TDh, TCh):
    """Exact share [H] of every program with its own snapped table [H, ...] (or one table for all)."""
    B = data.B
    H = len(progs)
    with torch.no_grad():
        if TDh.dim() == 4:
            TDC = table_TDC(TDh, TCh)
        else:
            TDC = torch.cat([TDh, TCh], -1).reshape(H, B * B * 2, B + 2)
        P = program_P(progs, learn)
        return M.exact_rows(run(data, P, TDC), data).double().mean(1)


def impostors(size=(3, 2, 1), learn=L4, held=((3, 2, 2), (3, 3, 2)), steps=600, seeds=2):
    """Every wiring of the learned muxes: (1) with the true table: exact on train and held out; (2) with its own
    best-fit table from the answers alone, and from the answers plus the cell check (best of `seeds`): exact on train
    and held out. Lists the wirings that fit every training answer."""
    d = Data(*size)
    hd = [Data(*s) for s in held]
    progs = programs(learn)
    TD, TC = M.truth_table(d.B)
    ex_true = exact_with(d, progs, learn, onehot(TD.argmax(-1), d.B), onehot(TC.argmax(-1), 2))
    ex_true_h = [exact_with(h, progs, learn, onehot(TD.argmax(-1), d.B), onehot(TC.argmax(-1), 2)) for h in hd]
    print(f"{size} {len(progs)} wirings of {learn}")
    print(f"true table: train-exact wirings {int((ex_true == 1).sum())}; held-out exact on all: "
          f"{int(((ex_true == 1) & torch.stack([e == 1 for e in ex_true_h]).all(0)).sum())}")
    for i in torch.nonzero(ex_true == 1).flatten().tolist():
        print("   ", M.describe(progs[i]), "held", [round(e[i].item(), 3) for e in ex_true_h])
    for cell in (0.0, 1.0):
        best = None
        for seed in range(seeds):
            TDh, TCh, ex = fit_tables(d, progs, learn, steps=steps, seed=seed, cell=cell)
            exh = [exact_with(h, progs, learn, TDh, TCh) for h in hd]
            if best is None:
                best = (ex, exh)
            else:
                better = ex > best[0]
                best = (torch.where(better, ex, best[0]), [torch.where(better, a, b) for a, b in zip(exh, best[1])])
        ex, exh = best
        fit = torch.nonzero(ex == 1).flatten().tolist()
        cert = [i for i in fit if all(e[i] == 1 for e in exh)]
        print(f"best-fit table ({'answers + cell' if cell else 'answers only'}, best of {seeds}): "
              f"{len(fit)} wirings fit every training answer, {len(cert)} also every held-out one; "
              f"train-exact quantiles {np.quantile(ex.numpy(), [0.5, 0.9, 0.99]).round(2).tolist()}")
        for i in fit[:40]:
            print("   ", M.describe(progs[i]), "train", round(ex[i].item(), 3), "held",
                  [round(e[i].item(), 3) for e in exh])


def program_scores(data: Data, progs, learn, TDC, chunk=128):
    """Exact share and mean answer log-likelihood per digit [H] of every program (one table for all)."""
    ex, ll = [], []
    with torch.no_grad():
        for i in range(0, len(progs), chunk):
            P = program_P(progs[i:i + chunk], learn)
            ans = run(data, P, TDC)
            ex.append(M.exact_rows(ans, data).double().mean(1))
            ll.append(M.row_loglik(ans.clamp_min(1e-6), data).mean(1) / data.w)
    return torch.cat(ex), torch.cat(ll)


def flips(sizes=((3, 2, 1), (3, 2, 2), (4, 2, 1)), learn=ALL):
    """The discrete landscape of whole wirings with the true table: how many wirings are local optima of the exact
    share under every single-mux change (where hard-path routing and one-change try-and-keep stop), and how far the
    greedy one-change climb gets from random wirings."""
    for B, w, k in sizes:
        d = Data(B, w, k)
        TDC = table_TDC(*M.truth_table(B))
        progs = programs(learn)
        ex, ll = program_scores(d, progs, learn, TDC)
        index = {tuple(p[m] for m in learn): i for i, p in enumerate(progs)}

        def neighbors(i):
            p = progs[i]
            for m in learn:
                for o in range(len(MUXES[m])):
                    if o != p[m]:
                        yield index[tuple((o if mm == m else p[mm]) for mm in learn)]

        for name, score in (("exact share", ex), ("log-likelihood", ll)):
            local = [i for i in range(len(progs)) if all(score[j] <= score[i] for j in neighbors(i))]
            strict = [i for i in local if score[i] < score.max()]
            climbs, rng = [], np.random.default_rng(0)
            for _ in range(200):
                i = int(rng.integers(len(progs)))
                while True:
                    j = max(neighbors(i), key=lambda j: float(score[j]))
                    if score[j] <= score[i]:
                        break
                    i = j
                climbs.append(float(ex[i]) == 1.0)
            print(f"{(B, w, k)} {len(progs)} wirings, score = {name}: {len(local)} local optima under one change, "
                  f"{len(strict)} of them wrong; greedy climb from 200 random wirings reaches the truth "
                  f"{np.mean(climbs):.2f}", flush=True)
            top = sorted(strict, key=lambda i: -float(score[i]))[:5]
            for i in top:
                print(f"     {M.describe(progs[i])}: exact {ex[i]:.3f} loglik {ll[i]:.3f}")


def census_chunk(size=(3, 2, 1), learn=ALL, chunk=0, nchunks=8, cell=0.0, seeds=2, steps=600, held=None):
    """One chunk of the impostor census over every wiring: each wiring gets its own table fitted from the answers
    (plus the cell check with weight `cell`), best of `seeds`; returns train and held-out exact per wiring."""
    d = Data(*size)
    B, w, k = size
    held = held or ((B, w, k + 1), (B, w + 1, k + 1))
    hd = [Data(*h) for h in held]
    progs = programs(learn)
    idx = list(range(chunk, len(progs), nchunks))
    sub = [progs[i] for i in idx]
    best = None
    for seed in range(seeds):
        TDh, TCh, ex = fit_tables(d, sub, learn, steps=steps, seed=seed, cell=cell)
        exh = [exact_with(h, sub, learn, TDh, TCh) for h in hd]
        if best is None:
            best = (ex, exh)
        else:
            better = ex > best[0]
            best = (torch.where(better, ex, best[0]), [torch.where(better, a, b) for a, b in zip(exh, best[1])])
    ex, exh = best
    TD, TC = M.truth_table(B)
    ext = exact_with(d, sub, learn, onehot(TD.argmax(-1), B), onehot(TC.argmax(-1), 2))
    return [{"prog": M.describe(p), "train": round(float(ex[i]), 4), "held": [round(float(e[i]), 4) for e in exh],
             "true_table": round(float(ext[i]), 4)} for i, p in enumerate(sub)]


# ----- the port twin: an exact symmetry, its saddle and the escape -----
# Registers in a fixed order (prev, start, N); mux a lists them as (prev, start, N), mux b as (N, prev, start).
A_OF_REG = torch.tensor([0, 1, 2])
B_OF_REG = torch.tensor([1, 2, 0])


class PortCore:
    """The machine with only the two port muxes learned (soft routing or the 9-program mixture) and the table, as one
    flat parameter vector, in register coordinates so the twin symmetry is a fixed linear map."""

    def __init__(self, B, w, k, param="soft"):
        self.d = Data(B, w, k)
        self.B, self.param = B, param
        self.nt = B * B * 2 * (B + 2)
        self.nr = 6 if param == "soft" else 9
        self.n = self.nt + self.nr
        self.progs = [{"a": int(A_OF_REG[ra]), "b": int(B_OF_REG[rb])} for ra in range(3) for rb in range(3)]
        self.Pm = program_P(self.progs, ("a", "b"))

    def split(self, x):
        B = self.B
        t = x[:self.nt].view(B, B, 2, B + 2)
        return t[..., :B], t[..., B:], x[self.nt:]

    def loss(self, x):
        td, tc, r = self.split(x)
        TDC = table_TDC(td.softmax(-1), tc.softmax(-1))
        if self.param == "soft":
            pa = torch.zeros(3, dtype=x.dtype).index_put((A_OF_REG,), r[:3].softmax(0))
            pb = torch.zeros(3, dtype=x.dtype).index_put((B_OF_REG,), r[3:].softmax(0))
            P = program_P([M.true_program()], ())
            P["a"], P["b"] = pa[None], pb[None]
            ll = M.row_loglik(run(self.d, P, TDC), self.d)[0]
            return -ll.mean() / self.d.w
        ll = M.row_loglik(run(self.d, self.Pm, TDC), self.d)
        return -torch.logsumexp(r.log_softmax(0)[:, None] + ll, 0).mean() / self.d.w

    def sigma(self, x):
        """The twin map: transpose the table, exchange the two muxes (register-wise)."""
        td, tc, r = self.split(x)
        t = torch.cat([td.transpose(0, 1), tc.transpose(0, 1)], -1).flatten()
        if self.param == "soft":
            r2 = torch.cat([r[3:], r[:3]])
        else:
            R = r.view(3, 3)                                                    # [reg of a, reg of b]
            r2 = R.t().flatten()
        return torch.cat([t, r2])

    def exact(self, x):
        td, tc, r = self.split(x)
        if self.param == "soft":
            pa, pb = r[:3], r[3:]
            prog = {"a": int(A_OF_REG[int(pa.argmax())]), "b": int(B_OF_REG[int(pb.argmax())])}
        else:
            prog = self.progs[int(r.argmax())]
        full = dict(M.true_program(), **prog)
        return M.hard_exact(self.d, full, td.softmax(-1), tc.softmax(-1)), prog


def saddle(B=3, w=2, k=1, param="soft", lr=1.0, steps=6000, seeds=3, epss=(1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8),
           sym_steps=None):
    """Plain gradient descent from an exactly twin-symmetric start (symmetric table, equal muxes), plus an
    antisymmetric push of size eps: the antisymmetric part stays 0 without the push; with it, steps to exact grow
    like ln(1/eps)/lambda, lambda = the most negative eigenvalue of the Hessian's antisymmetric block on the plateau."""
    core = PortCore(B, w, k, param)
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        x0 = torch.randn(core.n, generator=g) * 0.1
        x0 = (x0 + core.sigma(x0)) / 2                                          # symmetric start
        u = torch.randn(core.n, generator=g)
        u = (u - core.sigma(u)) / 2
        u = u / u.norm()                                                        # antisymmetric direction
        # the symmetric trajectory: does it stay symmetric, where does it settle, and the Hessian there
        x = x0.clone().requires_grad_(True)
        sym_steps = sym_steps or steps // 2
        for t in range(sym_steps):
            L = core.loss(x)
            gx, = torch.autograd.grad(L, x)
            with torch.no_grad():
                x -= lr * gx
        anti = ((x - core.sigma(x)) / 2).norm().item()
        Hm = torch.autograd.functional.hessian(core.loss, x.detach())
        # basis of the antisymmetric subspace
        S = torch.stack([core.sigma(e) for e in torch.eye(core.n)], 1)         # sigma as a matrix
        Pa = (torch.eye(core.n) - S) / 2
        ev_full = torch.linalg.eigvalsh((Hm + Hm.t()) / 2)
        Q, _ = torch.linalg.qr(Pa)
        rank = int(torch.linalg.matrix_rank(Pa))
        U, Sv, _ = torch.linalg.svd(Pa)
        Qa = U[:, :rank]
        ev_anti = torch.linalg.eigvalsh(Qa.t() @ ((Hm + Hm.t()) / 2) @ Qa)
        ex_sym, _ = core.exact(x.detach())
        print(f"{(B, w, k)} {param} seed {seed}: symmetric run after {sym_steps} steps: loss {float(core.loss(x)):.4f}"
              f" exact {ex_sym:.3f} antisym norm {anti:.1e}; Hessian min eig full {ev_full[0]:.3e}, antisym block "
              f"{ev_anti[0]:.3e} (pred. steps per decade of eps at lr {lr}: "
              f"{math_ln10() / (-lr * ev_anti[0].item()) if ev_anti[0] < 0 else float('inf'):.0f})", flush=True)
        res = []
        for eps in epss:
            x = (x0 + eps * u).clone().requires_grad_(True)
            hit = None
            for t in range(steps):
                if t % 10 == 0:
                    ex, prog = core.exact(x.detach())
                    if ex == 1.0:
                        hit = t
                        break
                L = core.loss(x)
                gx, = torch.autograd.grad(L, x)
                with torch.no_grad():
                    x -= lr * gx
            ex, prog = core.exact(x.detach())
            res.append(f"eps {eps:g}: {hit} ({'true' if prog == {'a': 0, 'b': 0} else 'twin' if prog == {'a': 2, 'b': 1} else M.describe(prog)}, exact {ex:.2f})")
        print("    ", "; ".join(res), flush=True)


def math_ln10():
    import math
    return math.log(10)



def check_impostors(sizes=((3, 2, 1), (3, 2, 2)), learn=ALL):
    """With the true table: which wirings pass each label-free check exactly (laws: x mod N = x and N mod N = 0 through
    the chain and write-back; range: every step's register below N), and how many of those get every answer right."""
    for B, w, k in sizes:
        d = Data(B, w, k)
        ck = Checks(d)
        TD, TC = M.truth_table(B)
        TDC = table_TDC(TD, TC)
        progs = programs(learn)
        ans_ok, laws_ok, range_ok = [], [], []
        with torch.no_grad():
            for i in range(0, len(progs), 128):
                P = program_P(progs[i:i + 128], learn)
                regs = []
                ans = run(d, P, TDC, regs=regs)
                ans_ok.append(M.exact_rows(ans, d).all(1))
                H = P["a"].shape[0]
                Rp = onehot(ck.lx, B)[None].expand(H, -1, -1, -1)
                n1 = onehot(ck.lN, B)[None]
                zero = torch.zeros(1, n1.shape[1], 1, B)
                zero[..., 0] = 1
                Nreg = torch.cat([n1, zero], 2).expand(H, -1, -1, -1)
                R = M.reduce_step(Rp, Nreg, P, TDC, B)
                laws_ok.append((R.argmax(-1) == ck.ly[None]).all(-1).all(-1))
                below = torch.stack([below_N(Rg, d.Nd) for (_, Rg, _) in regs], 0)   # [steps, H, X]
                range_ok.append((below > 0.5).all(0).all(-1))
        a, l, r = torch.cat(ans_ok), torch.cat(laws_ok), torch.cat(range_ok)
        print(f"{(B, w, k)} {len(progs)} wirings, true table: all answers right {int(a.sum())}; pass laws {int(l.sum())} "
              f"(right {int((l & a).sum())}); pass range {int(r.sum())} (right {int((r & a).sum())}); pass both "
              f"{int((l & r).sum())} (right {int((l & r & a).sum())})")
        for i in torch.nonzero(l & r & ~a).flatten()[:6].tolist():
            print("     passes both, wrong:", M.describe(progs[i]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    what = sys.argv[1] if len(sys.argv) > 1 else "grad0"
    {"grad0": grad0, "eps": eps_path, "impostors": impostors, "saddle": saddle, "flips": flips,
     "gradtrue": grad_true_table, "checks": check_impostors}[what]()
