"""Tiny core of learned schoolbook squaring: which signals teach the digit-product cell (a, b) -> (lo, hi)?

Base 10. The rows' own numbers are K-digit moduli N (N mod 10 in {1, 3, 7, 9}, as in H1) and x < N, both written in
K-digit registers (leading zeros). The machine multiplies by schoolbook: the row X * b puts cell (x_i, b)'s lo digit at
place i and its hi digit at place i + 1, a designed adder sums them, and X * Y sums the rows X * y_j shifted by j. The
cell is the only learned part: 100 rows x (10 lo + 10 hi) logits. Soft versions apply the cell to digit distributions
multilinearly (places independent), as the chain's soft subtractor does. Everything is small: well under 1 GB, seconds.

  python tiny.py null                          # exact: the table directions each identity set leaves free
  python tiny.py train LESSONS [SEED] [STEPS]  # LESSONS joined by '+', from: unit zero peano peanoboot comm sqstep
                                               #   mulstep answers answersfree (see LESSONS below)
  python tiny.py align [K_WRONG ...]           # near the truth: does each lesson repair k wrong entries?
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
K = 3                     # digits of the moduli
ROWS = 2048               # rows (N, x) in the universe
BATCH = 256
LR = 0.05
INIT = 0.1
POW = 10 ** torch.arange(19)
AR = torch.arange(10)
SUMM = F.one_hot((AR[:, None] + AR[None, :]).flatten(), 19).float()      # [100, 19]: a + b
DIFM = F.one_hot((AR[:, None] - AR[None, :] + 9).flatten(), 19).float()  # [100, 19]: a - b + 9
Z1 = F.one_hot(torch.tensor(0), 10).float()
TRUE_F = AR[:, None] * AR[None, :]


def digits(v, n):
    return (v[..., None] // POW[:n]) % 10


def universe(n_rows: int, seed: int, k: int = K):
    g = torch.Generator().manual_seed(seed)
    N = torch.randint(10 ** (k - 1), 10 ** k, (n_rows * 4,), generator=g)
    N = N[(N % 2 == 1) & (N % 5 != 0)][:n_rows]
    x = (torch.rand(n_rows, generator=g) * (N - 1).double()).long() + 1
    return N, x


# ------------------------------------------------------------------ designed adder / subtractor on distributions
def add_dd(A, B):
    """Designed adder on two digit distributions [..., P, 10]: the sum's digits mod 10^P and P(final carry)."""
    c = torch.zeros(A.shape[:-2])
    out = []
    for j in range(A.shape[-2]):
        s = (A[..., j, :, None] * B[..., j, None, :]).flatten(-2) @ SUMM
        s = (1 - c)[..., None] * F.pad(s, (0, 1)) + c[..., None] * F.pad(s, (1, 0))   # over a + b + c
        out.append(s[..., :10] + s[..., 10:])
        c = s[..., 10:].sum(-1)
    return torch.stack(out, -2), c


def sub_dd(A, B):
    """Designed subtractor on two digit distributions: A - B mod 10^P and P(final borrow)."""
    c = torch.zeros(A.shape[:-2])
    out = []
    for j in range(A.shape[-2]):
        d = (A[..., j, :, None] * B[..., j, None, :]).flatten(-2) @ DIFM
        d = (1 - c)[..., None] * F.pad(d, (1, 0)) + c[..., None] * F.pad(d, (0, 1))   # over a - b - c + 10
        out.append(d[..., :10] + d[..., 10:])
        c = d[..., :10].sum(-1)
    return torch.stack(out, -2), c


def zeros(n, p):
    return Z1.expand(n, p, 10)


def onehot(v, p):
    return F.one_hot(digits(v, p), 10).float()


# ------------------------------------------------------------------ the machine: soft and hard
def row_soft(Xd, b, Lo, Hi):
    """X (exact digits [n, k]) times the digit b [n]: [n, k + 2, 10] (the final carry of this add is always 0)."""
    n, k = Xd.shape
    bb = b[:, None].expand(n, k)
    lo, hi = Lo[Xd, bb], Hi[Xd, bb]
    return add_dd(torch.cat([lo, zeros(n, 2)], 1), torch.cat([zeros(n, 1), hi, zeros(n, 1)], 1))[0]


def mul_soft(Xd, Yd, Lo, Hi):
    """X * Y by schoolbook on distributions: [n, kx + ky + 1, 10]."""
    n, kx = Xd.shape
    ky = Yd.shape[1]
    P = kx + ky + 1
    acc = None
    for j in range(ky):
        r = row_soft(Xd, Yd[:, j], Lo, Hi)
        r = torch.cat([zeros(n, j), r, zeros(n, P - j - kx - 2)], 1)
        acc = r if acc is None else add_dd(acc, r)[0]
    return acc


def mul_hard(Xd, Yd, f):
    """X * Y on integers with the snapped cell (f = lo + 10 hi; the designed adder is exact)."""
    v = torch.zeros(Xd.shape[0], dtype=torch.long)
    for i in range(Xd.shape[1]):
        for j in range(Yd.shape[1]):
            v = v + f[Xd[:, i], Yd[:, j]] * POW[i + j]
    return v


def row_hard(Xd, b, f):
    return (f[Xd, b[:, None].expand_as(Xd)] * POW[:Xd.shape[1]]).sum(-1)


def divide_path(vs, vh, N, k):
    """vs: soft product [n, 2k + 1, 10]; vh: the machine's hard product; N [n]. Restoring long division with the EXACT
    subtractor (the remainder is assumed solved): register of k + 1 places from v // 10^k, k steps; each step's
    quotient digit comes from the hard division of vh (the model's own hard path), and the soft register runs that many
    exact subtractions of N. Returns the soft remainder [n, k + 1, 10] and the hard one [n]."""
    n = vh.shape[0]
    Nd1 = onehot(N, k + 1)
    Rh = vh // POW[k]
    Rs = vs[:, k:2 * k + 1]
    for t in reversed(range(k)):
        vt = (vh // POW[t]) % 10
        Rp = 10 * (Rh % POW[k]) + vt                                   # the register drops its top place
        q = torch.zeros_like(Rp)
        for s in range(1, 10):
            q = q + (Rp - s * N >= 0).long()
        Rh = Rp - q * N
        D = torch.cat([vs[:, t:t + 1], Rs[:, :k]], 1)
        Rs = D * (q == 0).float()[:, None, None]
        for s in range(1, int(q.max()) + 1):
            D = sub_dd(D, Nd1)[0]
            Rs = Rs + D * (q == s).float()[:, None, None]
    return Rs, Rh


# ------------------------------------------------------------------ lessons: (soft loss, hard pass rate)
def ce(D, t):
    return -D.clamp_min(1e-12).log().gather(-1, t[..., None]).squeeze(-1).mean()


def nb(p):  # -log P(no final borrow)
    return -(1 - p).clamp_min(1e-12).log().mean()


def lesson(name, N, x, Lo, Hi, f):
    k = K
    Xs = torch.cat([x, N])                       # the rows' own numbers
    Xd = digits(Xs, k)
    n = Xs.shape[0]
    if name == "unit":                           # X * 1 = X
        one = torch.ones(n, dtype=torch.long)
        return ce(row_soft(Xd, one, Lo, Hi), digits(Xs, k + 2)), float((row_hard(Xd, one, f) == Xs).float().mean())
    if name == "zero":                           # X * 0 = 0
        z = torch.zeros(n, dtype=torch.long)
        return ce(row_soft(Xd, z, Lo, Hi), digits(z, k + 2)), float((row_hard(Xd, z, f) == 0).float().mean())
    if name == "peano":                          # X * (b + 1) - X * b = X, b = 0..8 (both rows soft)
        loss, ok = 0.0, []
        for b in range(9):
            bb = torch.full((n,), b)
            Dd, bor = sub_dd(row_soft(Xd, bb + 1, Lo, Hi), row_soft(Xd, bb, Lo, Hi))
            loss = loss + ce(Dd, digits(Xs, k + 2)) + nb(bor) / (k + 2)
            ok.append(row_hard(Xd, bb + 1, f) - row_hard(Xd, bb, f) == Xs)
        return loss / 9, float(torch.stack(ok).float().mean())
    if name == "peanosg":                        # X * (b + 1) - X * b = X with the lower rung a soft target (no grad)
        loss, ok = 0.0, []
        for b in range(9):
            bb = torch.full((n,), b)
            Dd, bor = sub_dd(row_soft(Xd, bb + 1, Lo, Hi), row_soft(Xd, bb, Lo, Hi).detach())
            loss = loss + ce(Dd, digits(Xs, k + 2)) + nb(bor) / (k + 2)
            ok.append(row_hard(Xd, bb + 1, f) - row_hard(Xd, bb, f) == Xs)
        return loss / 9, float(torch.stack(ok).float().mean())
    if name in ("peanoboot", "peanoboot1"):      # X * (b + 1) learns from the machine's own hard X * b + X
        loss, ok = 0.0, []
        for b in range(1 if name == "peanoboot1" else 0, 9):
            bb = torch.full((n,), b)
            tgt = row_hard(Xd, bb, f) + Xs
            loss = loss + ce(row_soft(Xd, bb + 1, Lo, Hi), digits(tgt.clamp(0, 10 ** (k + 2) - 1), k + 2))
            ok.append(row_hard(Xd, bb + 1, f) == tgt)
        return loss / len(ok), float(torch.stack(ok).float().mean())
    if name == "peanogate":                      # the ladder along the verified frontier: X * (b + 1) learns from the
        loss, ok = 0.0, []                       # machine's own hard X * b + X only where X's rungs 0..b all pass
        rows = [row_hard(Xd, torch.full((n,), b), f) for b in range(10)]
        verified = rows[0] == 0
        for b in range(9):
            bb = torch.full((n,), b)
            tgt = rows[b] + Xs
            w = verified.float()
            nll = -row_soft(Xd, bb + 1, Lo, Hi).clamp_min(1e-12).log().gather(
                -1, digits(tgt.clamp(0, 10 ** (k + 2) - 1), k + 2)[..., None]).squeeze(-1).mean(-1)
            loss = loss + (nll * w).sum() / n
            ok.append(rows[b + 1] == tgt)
            verified = verified & ok[-1]
        return loss / 9, float(torch.stack(ok).float().mean())
    if name == "rowsup":                         # DIAGNOSTIC ONLY (a designed multiplier in the loss): exact rows X * b
        loss, ok = 0.0, []
        for b in range(10):
            bb = torch.full((n,), b)
            loss = loss + ce(row_soft(Xd, bb, Lo, Hi), digits(Xs * b, k + 2))
            ok.append(row_hard(Xd, bb, f) == Xs * b)
        return loss / 10, float(torch.stack(ok).float().mean())
    if name == "comm":                           # x * N - N * x = 0
        xd, Nd = digits(x, k), digits(N, k)
        Dd, bor = sub_dd(mul_soft(xd, Nd, Lo, Hi), mul_soft(Nd, xd, Lo, Hi))
        return ce(Dd, digits(torch.zeros_like(x), 2 * k + 1)) + nb(bor) / (2 * k + 1), \
            float((mul_hard(xd, Nd, f) == mul_hard(Nd, xd, f)).float().mean())
    if name == "sqstep":                         # (s + 1)^2 - s^2 = 2 s + 1, s = x
        sd, s1 = digits(x, k), digits(x + 1, k)
        Dd, bor = sub_dd(mul_soft(s1, s1, Lo, Hi), mul_soft(sd, sd, Lo, Hi))
        return ce(Dd, digits(2 * x + 1, 2 * k + 1)) + nb(bor) / (2 * k + 1), \
            float((mul_hard(s1, s1, f) - mul_hard(sd, sd, f) == 2 * x + 1).float().mean())
    if name == "mulstep":                        # s * (s + 1) - s * s = s
        sd, s1 = digits(x, k), digits(x + 1, k)
        Dd, bor = sub_dd(mul_soft(sd, s1, Lo, Hi), mul_soft(sd, sd, Lo, Hi))
        return ce(Dd, digits(x, 2 * k + 1)) + nb(bor) / (2 * k + 1), \
            float((mul_hard(sd, s1, f) - mul_hard(sd, sd, f) == x).float().mean())
    if name == "twoway":                         # the square computed two ways: s * s - (s - 1) * (s + 1) = 1
        sd, s0, s1 = digits(x, k), digits(x - 1, k), digits(x + 1, k)
        Dd, bor = sub_dd(mul_soft(sd, sd, Lo, Hi), mul_soft(s0, s1, Lo, Hi))
        return ce(Dd, digits(torch.ones_like(x), 2 * k + 1)) + nb(bor) / (2 * k + 1), \
            float((mul_hard(sd, sd, f) - mul_hard(s0, s1, f) == 1).float().mean())
    if name in ("answers", "answersfree"):       # x^2 mod N through the exact reduction, along the own hard path
        sd = digits(x, k)
        vs = mul_soft(sd, sd, Lo, Hi)
        vh = mul_hard(sd, sd, f)
        if name == "answersfree":                # the hard product is free to exceed N * 10^k (garbage dividend)
            vh = vh % POW[2 * k]
        else:                                    # clamp the hard dividend below N * 10^k (the division's range)
            vh = torch.minimum(vh, N * POW[k] - 1)
        Rs, Rh = divide_path(vs, vh, N, k)
        y = x * x % N
        return ce(Rs, digits(y, k + 1)), float((Rh == y).float().mean())
    raise KeyError(name)


def report(Llo, Lhi, test):
    lo, hi = Llo.argmax(-1), Lhi.argmax(-1)
    f = lo + 10 * hi
    right = int(((lo == TRUE_F % 10) & (hi == TRUE_F // 10)).sum())
    sd = digits(test, K)
    sq = float((mul_hard(sd, sd, f) == test * test).float().mean())
    anti = int(((f + f.T) == 2 * TRUE_F).all() and (f.diagonal() == TRUE_F.diagonal()).all())
    return right, sq, anti, f


def train(lessons, seed=1, steps=1500, start=None, quiet=False, log_every=100):
    N, x = universe(ROWS, 1000 + seed)
    test = torch.arange(10 ** K)                 # every K-digit number: does the snapped machine square it exactly?
    torch.manual_seed(seed)
    if start is None:
        Llo = (INIT * torch.randn(10, 10, 10)).requires_grad_()
        Lhi = (INIT * torch.randn(10, 10, 10)).requires_grad_()
    else:
        Llo, Lhi = (t.clone().requires_grad_() for t in start)
    opt = torch.optim.Adam([Llo, Lhi], lr=LR)
    g = torch.Generator().manual_seed(seed)
    t0, hist, done_at = time.time(), [], None
    for step in range(1, steps + 1):
        idx = torch.randint(0, N.shape[0], (BATCH,), generator=g)
        Lo, Hi = Llo.softmax(-1), Lhi.softmax(-1)
        f = Llo.argmax(-1) + 10 * Lhi.argmax(-1)
        loss, oks = 0.0, []
        for name in lessons:
            l, ok = lesson(name, N[idx], x[idx], Lo, Hi, f)
            loss = loss + l
            oks.append(ok)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([Llo, Lhi], 1.0)
        opt.step()
        if step % log_every == 0 or step == 1 or step == steps:
            right, sq, anti, _ = report(Llo.detach(), Lhi.detach(), test)
            hist.append((step, round(float(loss.detach()), 4), right, round(sq, 3)))
            if not quiet:
                print(f"step {step} loss {float(loss.detach()):.4f} checks {[round(o, 3) for o in oks]} rows_right {right}/100 "
                      f"squares_exact {sq:.3f} square-equivalent {anti} {time.time() - t0:.1f}s", flush=True)
            if right == 100 and done_at is None:
                done_at = step
    right, sq, anti, f = report(Llo.detach(), Lhi.detach(), test)
    return {"right": right, "squares": sq, "anti": anti, "f": f, "done_at": done_at, "hist": hist,
            "logits": (Llo.detach(), Lhi.detach())}


# ------------------------------------------------------------------ exact: free directions of each identity set
P_MOD = 2147483629


def rank_mod(A):
    A = np.array(A, dtype=np.int64) % P_MOD
    r, rows, cols = 0, A.shape[0], A.shape[1]
    pivots = []
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c] != 0), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), P_MOD - 2, P_MOD)
        A[r] = (A[r] * inv) % P_MOD
        nz = np.nonzero(A[:, c])[0]
        for i in nz:
            if i != r:
                A[i] = (A[i] - A[i, c] * A[r]) % P_MOD
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return r, pivots


def equations(name, N, x, k=K):
    """Rows of the linear system on f(a, b) (100 unknowns, cell (a, b) -> column 10 a + b): coefficient vector, target."""
    eqs = []
    Xs = torch.cat([x, N]).tolist()
    dg = lambda v: [(v // 10 ** i) % 10 for i in range(k)]
    if name == "unit":
        for X in Xs:
            c = np.zeros(100, dtype=object)
            for i, a in enumerate(dg(X)):
                c[10 * a + 1] += 10 ** i
            eqs.append((c, X))
    if name == "zero":
        for X in Xs:
            c = np.zeros(100, dtype=object)
            for i, a in enumerate(dg(X)):
                c[10 * a] += 10 ** i
            eqs.append((c, 0))
    if name == "peano":
        for X in Xs:
            for b in range(9):
                c = np.zeros(100, dtype=object)
                for i, a in enumerate(dg(X)):
                    c[10 * a + b + 1] += 10 ** i
                    c[10 * a + b] -= 10 ** i
                eqs.append((c, X))

    def prod(c, X, Y, sign):
        for i, a in enumerate(dg(X)):
            for j, b in enumerate(dg(Y)):
                c[10 * a + b] += sign * 10 ** (i + j)

    for X, Nn in zip(x.tolist(), N.tolist()):
        c = np.zeros(100, dtype=object)
        if name == "comm":
            prod(c, X, Nn, 1), prod(c, Nn, X, -1)
            eqs.append((c, 0))
        if name == "sqstep":
            prod(c, X + 1, X + 1, 1), prod(c, X, X, -1)
            eqs.append((c, 2 * X + 1))
        if name == "mulstep":
            prod(c, X, X + 1, 1), prod(c, X, X, -1)
            eqs.append((c, X))
        if name == "twoway":
            prod(c, X, X, 1), prod(c, X - 1, X + 1, -1)
            eqs.append((c, 1))
        if name == "squares":                    # reference: exact square LABELS (not available in H1)
            prod(c, X, X, 1)
            eqs.append((c, X * X))
    return eqs


def null(sets=None):
    N, x = universe(400, 7)
    truth = np.array([a * b for a in range(10) for b in range(10)], dtype=np.int64)
    sets = sets or ["unit", "zero", "peano", "zero+peano", "unit+peano", "comm", "sqstep", "mulstep", "twoway",
                    "squares", "sqstep+mulstep", "comm+sqstep", "twoway+sqstep", "unit+zero+peano+comm", "squares+comm"]
    for s in sets:
        eqs = sum((equations(n, N, x) for n in s.split("+")), [])
        A = np.array([[int(v) for v in c] for c, _ in eqs], dtype=np.int64)
        t = np.array([int(v) for _, v in eqs], dtype=np.int64)
        assert (A @ truth == t).all(), s                                   # the truth satisfies every equation
        used = [j for j in range(100) if (A[:, j] != 0).any()]
        r, piv = rank_mod(A)
        in_null = lambda v: not (A @ v).any()
        anti = sum(in_null(np.eye(100, dtype=np.int64)[10 * a + b] - np.eye(100, dtype=np.int64)[10 * b + a])
                   for a in range(10) for b in range(a + 1, 10))
        single = [divmod(j, 10) for j in range(100) if in_null(np.eye(100, dtype=np.int64)[j])]
        print(f"{s:24s} equations {len(eqs):5d} cells_used {len(used):3d} rank {r:3d} free_dims_among_used "
              f"{len(used) - r:3d}  antisymmetric_pairs_free {anti:2d}  single_cells_free {single[:12]}"
              f"{'...' if len(single) > 12 else ''}", flush=True)


# ------------------------------------------------------------------ near the truth: does a lesson repair k wrong cells?
def align(ks=(1, 2, 4, 8, 16), seeds=4, steps=300):
    for kw in ks:
        for names in (["answers"], ["zero", "peanogate"], ["zero", "peanogate", "answers"]):
            fixed = []
            for seed in range(seeds):
                g = torch.Generator().manual_seed(100 + seed)
                Llo = torch.full((10, 10, 10), -2.0)
                Lhi = torch.full((10, 10, 10), -2.0)
                lo, hi = (TRUE_F % 10).clone(), (TRUE_F // 10).clone()
                for c in torch.randperm(100, generator=g)[:kw].tolist():
                    a, b = divmod(c, 10)
                    if torch.rand(1, generator=g) < 0.5:
                        lo[a, b] = (lo[a, b] + torch.randint(1, 10, (1,), generator=g)) % 10
                    else:
                        hi[a, b] = (hi[a, b] + torch.randint(1, 10, (1,), generator=g)) % 10
                Llo.scatter_(-1, lo[..., None], 2.0)
                Lhi.scatter_(-1, hi[..., None], 2.0)
                out = train(names, seed=seed, steps=steps, start=(Llo, Lhi), quiet=True, log_every=steps)
                fixed.append(out["right"])
            print(f"wrong {kw:2d}  lessons {'+'.join(names):24s} rows_right after {steps} steps: {fixed}", flush=True)


def scramble(n_corrupt=200, seed=0):
    """One wrong cell: which rows' T = 1 answers (x^2 mod N, exact reduction) it spoils, and how close a spoiled answer
    still is to the true one (digits below N's width that agree; chance is 0.1 per digit)."""
    N, x = universe(ROWS, 1000 + seed)
    g = torch.Generator().manual_seed(seed)
    sd = digits(x, K)
    y = x * x % N
    rows_hit, exact_hit, digit_agree = [], [], []
    for _ in range(n_corrupt):
        f = TRUE_F.clone()
        a, b = torch.randint(0, 10, (2,), generator=g).tolist()
        d = int(torch.randint(1, 10, (1,), generator=g))
        if torch.rand(1, generator=g) < 0.5:
            f[a, b] = f[a, b] // 10 * 10 + (f[a, b] % 10 + d) % 10
        else:
            f[a, b] = ((f[a, b] // 10 + d) % 10) * 10 + f[a, b] % 10
        v = mul_hard(sd, sd, f)
        hit = v != x * x
        yw = v % N
        rows_hit.append(float(hit.float().mean()))
        if hit.any():
            exact_hit.append(float((yw[hit] == y[hit]).float().mean()))
            w = torch.tensor([len(str(int(n))) for n in N[hit]])
            same = (digits(yw[hit], K) == digits(y[hit], K)).float()
            digit_agree.append(float((same * (torch.arange(K)[None] < w[:, None])).sum() / w.sum()))
    m = lambda v: sum(v) / len(v)
    print(f"one wrong cell ({n_corrupt} draws): rows whose square it changes {m(rows_hit):.3f}; of those, answers still "
          f"exact {m(exact_hit):.4f}, answer digits that agree {m(digit_agree):.3f} (chance 0.100)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "null":
        null(sys.argv[2:] or None)
    elif cmd == "train":
        out = train(sys.argv[2].split("+"), int(sys.argv[3]) if len(sys.argv) > 3 else 1,
                    int(sys.argv[4]) if len(sys.argv) > 4 else 1500)
        f = out["f"]
        wrong = [(a, b, int(f[a, b])) for a in range(10) for b in range(10) if f[a, b] != a * b]
        print("wrong cells (a, b, f):", wrong[:40], "done_at", out["done_at"])
    elif cmd == "scramble":
        scramble()
    elif cmd == "align":
        align(tuple(int(v) for v in sys.argv[2:]) or (1, 2, 4, 8, 16))
