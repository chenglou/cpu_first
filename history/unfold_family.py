"""Learn which formula: self-unfolding for an unknown polynomial recurrence. Write-up: UNFOLD_FAMILY.md.

UNFOLD.md learned the squaring step s -> s^2 mod N from answer-only rows. Here the step is unknown: s -> f(s, x0, N)
mod N for a hidden integer polynomial f of total degree <= 3 in (s, x0, N) (20 monomials), and the labels are
y = f^T(x0 mod N) on H1's (N, x, T) rows (menu.py's hidden recurrences, relabeled). Same two methods as UNFOLD.md:

  lattice  integer ambiguity resolution: c . phi(s, x0, N) = y (mod N) on every row; the 20 integer coefficients and
           every wrap count at once by LLL (unfold_lattice.lattice_solve). T=1 rows have s = x0, so they only see the
           diagonal f(x, x). Stage 2 runs the model's own stage-1 step on the T=2 rows' x0 to get their middle state,
           then solves again on T=1 + T=2 rows, where s and x0 differ.
           `--family sn` drops x0 (the step reads s and N only), an Occam prior for T=1 rows.
  random   the same on random hidden polynomials with coefficients in [-C, C] (rows needed vs C)
  digits   s and x0 read from their decimal digits through one shared base B (B = 2..16 scanned, a lattice per B)
  ambiguity  the one change T=1 + T=2 rows cannot see, k (s - x0)(s - f(x, x)) when f(x, x) is quadratic, and
           adversarial hidden steps that a few T=4 rows (stage 3) settle
  grad     clock loss + easiest-first on H1's own rows (UNFOLD.md section 4) with the same model: T=1 rows first,
           then T=2 rows unrolled through the model's own snapped state; the coefficients are rounded at the end
  fold     the same on fresh rows with a fold curriculum (x0 < N^e, e growing; UNFOLD.md test 3)
  qgrad    grad / fold with the learned diagonal frozen: f = g(x0) + (s - x0) q(s, x0), only q trained on T=2 rows
  report   markdown tables from runs/unfold_family/*.json

Every learned formula is checked the same way: exact integer arithmetic, all T=1 and T=2 training rows, and the
certification ladder T = 1..64 on both H1 profiles relabeled under the hidden recurrence. numpy + python-flint only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from decimal import Decimal, getcontext

import numpy as np

import h1

getcontext().prec = 80

# exponents (a, b, c) of s^a x0^b N^c, total degree <= 3: 20 monomials. Monomials with c >= 1 are multiples of N,
# so an integer coefficient on them never changes anything mod N.
MONOS = [(a, b, d - a - b) for d in range(4) for a in range(d, -1, -1) for b in range(d - a, -1, -1)]
SX = [m for m in MONOS if m[2] == 0]                                   # the 10 monomials in (s, x0) alone


def mono_name(m):
    parts = [v if e == 1 else f"{v}^{e}" for v, e in zip(("s", "x0", "N"), m) if e]
    return "*".join(parts) or "1"


def as_vec(spec, monos=MONOS):
    return [int(spec.get(m, 0)) for m in monos]


TASKS = {                                                               # menu.py's hidden recurrences
    "square": {(2, 0, 0): 1},
    "cube": {(3, 0, 0): 1},
    "square+1": {(2, 0, 0): 1, (0, 0, 0): 1},
    "square*x0": {(2, 1, 0): 1},
    "square+x0": {(2, 0, 0): 1, (0, 1, 0): 1},
    "cube+2": {(3, 0, 0): 1, (0, 0, 0): 2},
}
MENU_LAMBDAS = {                       # menu.py's TASKS verbatim (menu.py runs its experiment on import)
    "square": lambda s, x0, n: s * s % n,
    "cube": lambda s, x0, n: s * s * s % n,
    "square+1": lambda s, x0, n: (s * s + 1) % n,
    "square*x0": lambda s, x0, n: s * s * x0 % n,
    "square+x0": lambda s, x0, n: (s * s + x0) % n,
    "cube+2": lambda s, x0, n: (s * s * s + 2) % n,
}


def canon(c, monos=MONOS):
    """Drop the N-monomials: they are multiples of N, invisible to every label and to the recurrence."""
    return [int(v) if m[2] == 0 else 0 for v, m in zip(c, monos)]


def formula(c, monos=MONOS, base=None):
    terms = []
    for v, m in sorted(zip(canon(c, monos), monos), key=lambda t: (-sum(t[1]), -t[1][0], -t[1][1])):
        if v:
            name = mono_name(m)
            terms.append(f"{v}" if name == "1" else name if v == 1 else f"-{name}" if v == -1 else f"{v}*{name}")
    out = " + ".join(terms).replace("+ -", "- ") or "0"
    return out if base in (None, 10) else f"[{out}] reading digits in base {base}"


# ----------------------------------------------------------------------------------- exact evaluation mod N
def read_base(v, base, N=None):
    """sum_i base^i d_i over the decimal digits d_i of v (base 10 gives v back); exact ints, reduced mod N if given."""
    v = np.asarray(v, dtype=np.int64)
    if base in (None, 10):
        return v.copy() if N is None else v % np.asarray(N, dtype=np.int64)
    r = (h1.digits(v, 9).astype(object) * np.array([int(base) ** i for i in range(9)], dtype=object)).sum(1)
    return r if N is None else np.array(r % np.asarray(N).astype(object), dtype=np.int64)


def step_mod(c, s, x0, N, monos=MONOS, base=None):
    """f(s, x0) mod N for integer coefficients c over `monos`, s and x0 read in `base`. int64 throughout (N < 2^31)."""
    N = np.asarray(N, dtype=np.int64)
    s = read_base(s, base, N)
    x0 = read_base(x0, base, N)
    sp, xp = [np.ones_like(N)], [np.ones_like(N)]
    for _ in range(3):
        sp.append(sp[-1] * s % N)
        xp.append(xp[-1] * x0 % N)
    out = np.zeros_like(N)
    No = N.astype(object)
    for coef, (a, b, cN) in zip(c, monos):
        if coef == 0 or cN > 0:
            continue
        cm = np.array(int(coef) % No, dtype=np.int64)
        out = (out + cm * (sp[a] * xp[b] % N)) % N
    return out


def relabel(c, N, X, T, monos=MONOS, base=None):
    """y = f^T(x mod N) row by row (T may differ per row)."""
    N, T = np.asarray(N, dtype=np.int64), np.asarray(T, dtype=np.int64)
    s = np.asarray(X, dtype=np.int64) % N
    for t in range(int(T.max())):
        live = T > t
        s[live] = step_mod(c, s[live], np.asarray(X)[live], N[live], monos, base)
    return s


def self_check():
    """TASKS as coefficient vectors reproduce menu.py's lambdas on H1 rows (T up to 16)."""
    tr = h1.load("train")
    idx = np.random.default_rng(0).choice(len(tr["N"]), 300, replace=False)
    for name, f in MENU_LAMBDAS.items():
        want = []
        for n, x, t in zip(tr["N"][idx].tolist(), tr["X"][idx].tolist(), tr["T"][idx].tolist()):
            s = x % n
            for _ in range(t):
                s = f(s, x, n)
            want.append(s)
        got = relabel(as_vec(TASKS[name]), tr["N"][idx], tr["X"][idx], tr["T"][idx])
        assert np.array_equal(got, np.array(want)), name


# -------------------------------------------------------------------------------------------- data, certify
_TRAIN = None
_PROF = None
_CERT = {}
_PROF_Y = {}


def train_rows(c_true, monos=MONOS):
    """H1's T=1 and T=2 training rows relabeled under the hidden recurrence."""
    global _TRAIN
    if _TRAIN is None:
        _TRAIN = h1.load("train")
    out = {}
    for t in (1, 2, 4):
        m = _TRAIN["T"] == t
        N, X = _TRAIN["N"][m], _TRAIN["X"][m]
        out[t] = {"N": N, "X": X, "Y": relabel(c_true, N, X, np.full(len(N), t), monos)}
    return out


def certify(c_hat, c_true, monos=MONOS, base=None, monos_true=MONOS):
    """Largest T on H1's ladder with every rung 1..T exact, per profile, labels relabeled under c_true."""
    global _PROF
    if _PROF is None:
        _PROF = {p: {t: h1.load(f"{p}_{t}") for t in h1.LADDER} for p in ("depth_t", "depth_ood_n_t")}
    key = (tuple(canon(c_hat, monos)), tuple(monos), base, tuple(canon(c_true, monos_true)), tuple(monos_true))
    if key in _CERT:
        return _CERT[key]
    res = {}
    for prof, rungs in _PROF.items():
        max_t = 0
        for t in h1.LADDER:
            d = rungs[t]
            yk = (tuple(canon(c_true, monos_true)), tuple(monos_true), prof, t)
            if yk not in _PROF_Y:
                _PROF_Y[yk] = relabel(c_true, d["N"], d["X"], d["T"], monos_true)
            y = _PROF_Y[yk]
            p = relabel(c_hat, d["N"], d["X"], d["T"], monos, base)
            if not h1.exact(h1.digits(p), y).all():
                break
            max_t = t
        res[prof] = max_t
    _CERT[key] = res
    return res


def check(c_hat, c_true, rows, monos=MONOS, base=None, monos_true=MONOS):
    """Exactness of a learned formula on all T=1 / T=2 training rows, and its certification."""
    r1, r2 = rows[1], rows[2]
    ok1 = float((relabel(c_hat, r1["N"], r1["X"], np.ones(len(r1["N"]), np.int64), monos, base) == r1["Y"]).mean())
    ok2 = float((relabel(c_hat, r2["N"], r2["X"], np.full(len(r2["N"]), 2), monos, base) == r2["Y"]).mean())
    cert = certify(c_hat, c_true, monos, base, monos_true)
    true = base in (None, 10) and canon(c_hat, monos) == canon(proj(c_true, monos_true, monos), monos)
    return {"t1_exact": ok1, "t2_exact": ok2, "depth_t": cert["depth_t"], "depth_ood_n_t": cert["depth_ood_n_t"],
            "certified": cert["depth_t"] == 64 and cert["depth_ood_n_t"] == 64, "is_hidden": bool(true)}


def proj(c, monos_from, monos_to):
    d = dict(zip(monos_from, c))
    return [int(d.get(m, 0)) for m in monos_to]


# ------------------------------------------------------------------------------------------------- lattice
def feature_rows(monos, N, S, X0, base=None):
    """phi(s, x0, N) reduced mod N, exact Python ints (object array)."""
    S, X0 = read_base(S, base), read_base(X0, base)
    out = np.empty((len(N), len(monos)), dtype=object)
    for i, (n, s, x) in enumerate(zip(N.tolist(), S.tolist(), X0.tolist())):
        for j, (a, b, c) in enumerate(monos):
            out[i, j] = pow(s, a, n) * pow(x, b, n) * pow(n, c, n) % n
    return out


def solve(monos, perm, N, S, X0, Y, base=None, M=2 ** 64):
    """Smallest integer c (columns in the order `perm`) with c . phi = y (mod N) on every row, or None."""
    import unfold_lattice
    F = feature_rows(monos, N, S, X0, base)[:, perm]
    cp = unfold_lattice.lattice_solve(F, N.astype(object), (np.asarray(Y) % N).astype(object), M=M)
    if cp is None:
        return None
    c = [0] * len(monos)
    for k, j in enumerate(perm):
        c[j] = int(cp[k])
    return c


def seeded(*key):
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def lattice_trial(c_true, rows, i1, i2, perm, monos=MONOS, base=None, monos_true=MONOS):
    """Stage 1 on the T=1 rows i1 (s = x0). If T=2 rows i2 are given, the stage-1 model's own step on their x0 is their
    middle state, and stage 2 solves again on T=1 + T=2 rows. In the digit model the state is a number whose decimal
    digits are read through the base, like any input."""
    r1, r2 = rows[1], rows[2]
    N1, X1, Y1 = r1["N"][i1], r1["X"][i1], r1["Y"][i1]
    t0 = time.perf_counter()
    c1 = solve(monos, perm, N1, X1 % N1, X1, Y1, base)
    out = {"m1": len(i1), "m2": len(i2), "sec1": time.perf_counter() - t0, "sec2": 0.0, "found1": c1 is not None}
    if c1 is not None:
        out["formula1"] = formula(c1, monos, base)
        out["coef1"] = c1
        out.update({k + "_1": v for k, v in check(c1, c_true, rows, monos, base, monos_true).items()})
    c = c1
    if len(i2):
        c = None
        if c1 is not None:
            N2, X2, Y2 = r2["N"][i2], r2["X"][i2], r2["Y"][i2]
            s1 = step_mod(c1, X2, X2, N2, monos, base)
            t0 = time.perf_counter()
            c = solve(monos, perm, np.concatenate([N1, N2]), np.concatenate([X1 % N1, s1]),
                      np.concatenate([X1, X2]), np.concatenate([Y1, Y2]), base)
            out["sec2"] = time.perf_counter() - t0
    out["found"] = c is not None
    if c is not None:
        out["formula"] = formula(c, monos, base)
        out["coef"] = c
        out.update(check(c, c_true, rows, monos, base, monos_true))
    else:
        out.update({"certified": False, "is_hidden": False})
    out["sec"] = out["sec1"] + out["sec2"]
    out["least_wraps_x0cubed"] = min(int(x) ** 3 // int(n) for x, n in zip(X1, N1))
    return out


def pick(rows, m1, m2, rng, p):
    return (rng.choice(len(rows[1]["N"]), m1, replace=False), rng.choice(len(rows[2]["N"]), m2, replace=False),
            rng.permutation(p))


FAMILIES = {"all": MONOS, "sn": [m for m in MONOS if m[1] == 0]}   # sn: the step never reads x0 (an Occam prior)


def lattice_mode(a):
    out = []
    monos = FAMILIES[a.family]
    for task in a.tasks:
        c_true = as_vec(TASKS[task])
        rows = train_rows(c_true)
        for m1 in a.m1:
            for m2 in a.m2:
                rng = seeded("lattice", task, m1, m2, a.seed) if a.family == "all" else \
                    seeded("lattice", a.family, task, m1, m2, a.seed)
                trials = [lattice_trial(c_true, rows, *pick(rows, m1, m2, rng, len(monos)), monos=monos)
                          for _ in range(a.trials)]
                r = summarize(trials, {"task": task, "m1": m1, "m2": m2, "family": a.family})
                out.append(r)
                print_summary("lattice" if a.family == "all" else f"lattice[{a.family}]", r)
    return out


# ------------------------------------------------------------------- what T=1 + T=2 rows cannot tell apart
def diagonal(c, monos=MONOS):
    """Coefficients d_k of g(x) = f(x, x) (N-monomials dropped)."""
    d = [0, 0, 0, 0]
    for v, (a, b, cN) in zip(c, monos):
        if cN == 0:
            d[a + b] += int(v)
    return d


def blind_spot(c, monos=MONOS):
    """h = (s - x0)(s - g(x0)) over MONOS when g = f(x, x) has degree <= 2, else None. h vanishes at s = x0 (T=1
    rows) and at s = g(x0) mod N (the middle state of every T=2 row), so f + k h fits every T=1 and T=2 row."""
    d = diagonal(c, monos)
    if d[3] != 0:
        return None
    h = {(2, 0, 0): 1, (1, 1, 0): -1}
    for k, dk in enumerate(d[:3]):
        if dk:
            h[(1, k, 0)] = h.get((1, k, 0), 0) - dk
            h[(0, k + 1, 0)] = h.get((0, k + 1, 0), 0) + dk
    return as_vec(h)


def stage3(c, rows, i4, monos=MONOS, K=20):
    """Settle the blind spot with T=4 rows: among c + k h (|k| <= K, shortest first), the first that matches the T=4
    rows i4 (exact evaluation; no lattice). Returns (c, k)."""
    h = blind_spot(c, monos)
    if h is None or not len(i4):
        return c, 0
    h = proj(h, MONOS, monos)
    r4 = rows[4]
    N4, X4, Y4 = r4["N"][i4], r4["X"][i4], r4["Y"][i4]
    cands = sorted(range(-K, K + 1), key=lambda k: (sum((u + k * v) ** 2 for u, v in zip(canon(c, monos), h)), abs(k)))
    for k in cands:
        ck = [u + k * v for u, v in zip(c, h)]
        if np.array_equal(relabel(ck, N4, X4, np.full(len(N4), 4), monos), Y4):
            return ck, k
    return c, None


ADVERSARIAL = {   # hidden steps that are not the shortest polynomial agreeing with them on every T=1 and T=2 row
    "square+blind": {(2, 0, 0): 2, (1, 2, 0): -1, (1, 1, 0): -1, (0, 3, 0): 1},       # s^2 + (s - x0)(s - x0^2)
    "square+x0-blind": {(1, 2, 0): 1, (1, 1, 0): 2, (0, 3, 0): -1, (0, 2, 0): -1, (0, 1, 0): 1},  # s^2 + x0 - h
}


def ambiguity_mode(a):
    """(1) For each hidden step whose diagonal is quadratic, the blind-spot neighbours f +- h: exact on all T=1 and
    T=2 rows, yet certify only T=2. (2) Adversarial hidden steps (f + h is hidden, f is shorter): the T<=2 lattice
    finds f; one or two T=4 rows (stage 3) move it to the hidden one."""
    out = {"neighbours": [], "adversarial": []}
    for task, spec in TASKS.items():
        c_true = as_vec(spec)
        h = blind_spot(c_true)
        rows = train_rows(c_true)
        if h is None:
            print(f"ambiguity {task:10s}: diagonal {diagonal(c_true)} is cubic, (s - x0)(s - g) has degree 4: no blind spot")
            out["neighbours"].append({"task": task, "diagonal": diagonal(c_true), "blind_spot": None})
            continue
        for k in (-1, 1):
            ck = [u + k * v for u, v in zip(c_true, h)]
            r4 = rows[4]
            agree4 = float((relabel(ck, r4["N"], r4["X"], np.full(len(r4["N"]), 4)) == r4["Y"]).mean())
            r = {"task": task, "k": k, "formula": formula(ck), "norm2": sum(v * v for v in canon(ck)),
                 "norm2_hidden": sum(v * v for v in canon(c_true)), "t4_agree": agree4, "blind_spot": formula(h)}
            r.update(check(ck, c_true, rows))
            out["neighbours"].append(r)
            print(f"ambiguity {task:10s} k={k:+d}: {r['formula']:32s} norm^2 {r['norm2']} (hidden {r['norm2_hidden']}); "
                  f"T=1 {r['t1_exact']:.3f} T=2 {r['t2_exact']:.3f} T=4 {agree4:.4f}; certified max T "
                  f"{r['depth_t']}/{r['depth_ood_n_t']}", flush=True)
    for task, spec in ADVERSARIAL.items():
        c_true = as_vec(spec)
        rows = train_rows(c_true)
        for m4 in a.m4:
            rng = seeded("adversarial", task, m4, a.seed)
            trials = []
            for _ in range(a.trials):
                i1, i2, perm = pick(rows, 2, 2, rng, len(MONOS))
                i4 = rng.choice(len(rows[4]["N"]), m4, replace=False)
                t = lattice_trial(c_true, rows, i1, i2, perm)
                t["formula_t12"] = t.get("formula")
                if t["found"] and m4:
                    c3, k = stage3(t["coef"], rows, i4)
                    t["stage3_k"] = k
                    t["formula"], t["coef"] = formula(c3), c3
                    t.update(check(c3, c_true, rows))
                trials.append(t)
            r = summarize(trials, {"task": task, "m1": 2, "m2": 2, "m4": m4, "hidden": formula(c_true)})
            out["adversarial"].append(r)
            print(f"adversarial {task} (hidden {formula(c_true)}), 2 T=1 + 2 T=2 rows + {m4} T=4 rows: certified "
                  f"{r['certified']}/{r['trials']}, found {r['formulas']}", flush=True)
    return out


def summarize(trials, head):
    n = len(trials)
    r = dict(head)
    r.update({
        "trials": n,
        "certified": sum(t.get("certified", False) for t in trials),
        "is_hidden": sum(t.get("is_hidden", False) for t in trials),
        "found": sum(t["found"] for t in trials),
        "t1_exact_all": sum(t.get("t1_exact", 0) == 1.0 for t in trials),
        "t2_exact_all": sum(t.get("t2_exact", 0) == 1.0 for t in trials),
        "diag_ok_stage1": sum(t.get("t1_exact_1", 0) == 1.0 for t in trials),
        "certified_stage1": sum(t.get("certified_1", False) for t in trials),
        "sec_median": float(np.median([t["sec"] for t in trials])),
        "sec_max": float(np.max([t["sec"] for t in trials])),
        "formulas": count([t.get("formula", "none") for t in trials]),
        "formulas_stage1": count([t.get("formula1", "none") for t in trials]),
        "runs": trials,
    })
    return r


def count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def print_summary(mode, r):
    top = list(r["formulas"].items())[:3]
    top1 = list(r["formulas_stage1"].items())[:4]
    print(f"{mode} {r['task']:12s} m1={r['m1']:<2d} m2={r['m2']:<2d}: certified {r['certified']}/{r['trials']}, "
          f"hidden formula {r['is_hidden']}/{r['trials']}, stage-1 diagonal exact {r['diag_ok_stage1']}/{r['trials']}"
          f"  ({r['sec_median'] * 1e3:.0f} ms median)  found {top}  stage 1 {top1}", flush=True)


def random_mode(a):
    """Random hidden polynomials over the 10 (s, x0) monomials, coefficients uniform in [-C, C]; m1 = m2 = m rows."""
    out = []
    for C in a.coef_max:
        prng = seeded("random-polys", C)
        polys = []
        while len(polys) < a.polys:
            c = prng.integers(-C, C + 1, len(SX)).tolist()
            if any(v and m[0] for v, m in zip(c, SX)):                # keep only steps that read s
                polys.append(c)
        for pi, c_sx in enumerate(polys):
            c_true = proj(c_sx, SX, MONOS)
            rows = train_rows(c_true)
            for m in a.m1:
                rng = seeded("random", C, pi, m, a.seed)
                trials = [lattice_trial(c_true, rows, *pick(rows, m, m, rng, len(MONOS))) for _ in range(a.trials)]
                r = summarize(trials, {"task": f"C={C} #{pi}", "C": C, "poly": formula(c_true), "m1": m, "m2": m})
                out.append(r)
                print_summary("random", r)
            print(f"      hidden C={C} #{pi}: {formula(c_true)}", flush=True)
    return out


def digits_mode(a):
    """One shared base: s and x0 enter as sum_i B^i d_i over their decimal digits. Each integer B in base_lo..base_hi
    gets its own stage-1 lattice on the same T=1 rows; the shortest solution picks B; stage 2 runs with that B."""
    out = []
    bases = list(range(a.base_lo, a.base_hi + 1))
    for task in a.tasks:
        c_true = as_vec(TASKS[task])
        rows = train_rows(c_true)
        for m1 in a.m1:
            for m2 in a.m2:
                rng = seeded("digits", task, m1, m2, a.seed)
                trials = []
                for _ in range(a.trials):
                    i1, i2, perm = pick(rows, m1, m2, rng, len(MONOS))
                    r1 = rows[1]
                    N1, X1, Y1 = r1["N"][i1], r1["X"][i1], r1["Y"][i1]
                    t0 = time.perf_counter()
                    norms = {}
                    for B in bases:
                        c = solve(MONOS, perm, N1, X1 % N1, X1, Y1, B)
                        norms[B] = math.sqrt(sum(v * v for v in canon(c))) if c is not None else float("inf")
                    sec_scan = time.perf_counter() - t0
                    best = min(norms.values())
                    winners = [B for B, v in norms.items() if v == best]
                    tr = lattice_trial(c_true, rows, i1, i2, perm, MONOS, winners[0])
                    tr.update({"base": winners[0], "base_unique": len(winners) == 1, "sec_scan": sec_scan,
                               "norm_base10": norms.get(10), "norms": {str(k): v for k, v in norms.items()}})
                    tr["sec"] += sec_scan
                    trials.append(tr)
                r = summarize(trials, {"task": task, "m1": m1, "m2": m2})
                r["base10_unique"] = sum(t["base"] == 10 and t["base_unique"] for t in trials)
                out.append(r)
                print_summary("digits", r)
                print(f"      base 10 chosen alone {r['base10_unique']}/{r['trials']}; bases picked "
                      f"{count([t['base'] for t in trials])}", flush=True)
    return out


# ------------------------------------------------------------------------------------- gradient: clock loss
GRAD_MODELS = {                       # name: (monomials, shared radix?)
    "poly20": (MONOS, False),         # the full model: 20 monomials in (s, x0, N), s and x0 read as numbers
    "poly10": (SX, False),            # without the N-monomials (multiples of N: zero mod N for integer weights)
    "tied10": (SX, True),             # s and x0 read from their digits through one learned radix B
    "tied20": (MONOS, True),
    "value3": ([(0, 0, 0), (1, 0, 0), (2, 0, 0)], False),   # UNFOLD.md's value3, (1, s, s^2): calibration
}
DIG = 9                                                                 # digit slots read by the radix models


class GradModel:
    """V = w . phi(s', x0', N) with real weights; s' = s (value) or sum_i B^i d_i(s) with one learned radix B."""

    def __init__(self, monos, tied, w, B=10.0):
        self.monos, self.tied = list(monos), tied
        self.w = np.asarray(w, dtype=np.float64).copy()
        self.B = float(B)

    def reads(self, v):
        v = np.asarray(v, dtype=np.int64)
        if not self.tied:
            return v.astype(np.float64), None
        d, i = h1.digits(v, DIG).astype(np.float64), np.arange(DIG, dtype=np.float64)
        return d @ self.B ** i, d @ (i * self.B ** np.maximum(i - 1, 0))

    def v_and_jac(self, s, x0, N):
        sp, dsp = self.reads(s)
        xp, dxp = self.reads(x0)
        Nf = np.asarray(N, dtype=np.float64)
        F = np.stack([sp ** a * xp ** b * Nf ** c for a, b, c in self.monos], 1)
        V = F @ self.w
        if not self.tied:
            return V, F
        dB = np.zeros_like(V)
        for wm, (a, b, c) in zip(self.w, self.monos):
            if a:
                dB += wm * a * sp ** (a - 1) * xp ** b * Nf ** c * dsp
            if b:
                dB += wm * b * sp ** a * xp ** (b - 1) * Nf ** c * dxp
        return V, np.concatenate([F, dB[:, None]], 1)

    def add_flat(self, delta):
        self.w += delta[:len(self.w)]
        if self.tied:
            self.B += float(delta[len(self.w)])

    def exact_state(self, s, x0, N):
        """round(V) mod N, with V computed exactly for the current fp64 parameters (fp64 itself cannot hold s^3 at
        20 bits to +-0.5). Used for the model's own snapped state and for its exactness."""
        out = np.empty(len(N), dtype=np.int64)
        if not (np.isfinite(self.w).all() and np.isfinite(self.B)):
            return np.full(len(N), -1, dtype=np.int64)                 # diverged: matches no label
        if not self.tied:
            ratios = [float(w).as_integer_ratio() for w in self.w]
            D = max(d for _, d in ratios)
            W = [n * (D // d) for n, d in ratios]
            for k, (si, xi, ni) in enumerate(zip(np.asarray(s).tolist(), np.asarray(x0).tolist(),
                                                  np.asarray(N).tolist())):
                v = sum(Wm * si ** a * xi ** b * ni ** c for Wm, (a, b, c) in zip(W, self.monos) if Wm)
                out[k] = ((2 * v + D) // (2 * D)) % ni
            return out
        Bd = Decimal(self.B)
        pw = [Bd ** i for i in range(DIG)]
        Wd = [Decimal(float(w)) for w in self.w]
        ds, dx = h1.digits(np.asarray(s), DIG), h1.digits(np.asarray(x0), DIG)
        for k, ni in enumerate(np.asarray(N).tolist()):
            sp = sum(pw[i] * int(ds[k, i]) for i in range(DIG))
            xp = sum(pw[i] * int(dx[k, i]) for i in range(DIG))
            pw_s, pw_x, pw_n = ([Decimal(1), v, v * v, v * v * v] for v in (sp, xp, Decimal(ni)))  # no 0 ** 0
            v = sum(Wm * pw_s[a] * pw_x[b] * pw_n[c] for Wm, (a, b, c) in zip(Wd, self.monos) if Wm)
            out[k] = int(v.to_integral_value()) % ni
        return out

    def rounded(self):
        """The integer hypothesis nearest the fp64 model: coefficients rounded, radix rounded."""
        if not (np.isfinite(self.w).all() and np.isfinite(self.B)):
            return [0] * len(self.w), (10 if self.tied else None)
        return [int(round(w)) for w in self.w], (int(round(self.B)) if self.tied else None)


def lm(m, s, x0, N, y, iters=3, damp=1e-6, wrap=True, weight="none"):
    """UNFOLD.md's Levenberg-Marquardt on the clock residual wrap(V - y) / N: each step regresses V toward its own
    nearest unfolding y + N round((V - y) / N). weight "sens" divides each row by its sensitivity (one unit vote)."""
    Nf, Y = np.asarray(N, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def resid():
        V, J = m.v_and_jac(s, x0, N)
        r = np.mod(V - Y + Nf / 2, Nf) - Nf / 2 if wrap else V - Y
        r, J = r / Nf, J / Nf[:, None]
        if weight == "sens":
            col = np.maximum(np.linalg.norm(J, axis=0), 1e-300)
            rho = 1 / (np.linalg.norm(J / col, axis=1) + 1e-12)
            r, J = r * rho, J * rho[:, None]
        return r, J

    r, J = resid()
    cost = float(r @ r)
    if not np.isfinite(cost):
        return cost
    for _ in range(iters * 4):
        col = np.maximum(np.linalg.norm(J, axis=0), 1e-300)
        Jn = J / col
        A = Jn.T @ Jn
        try:
            delta = np.linalg.solve(A + damp * np.eye(A.shape[0]), Jn.T @ r) / col
        except np.linalg.LinAlgError:
            break
        m.add_flat(-delta)
        r2, J2 = resid()
        c2 = float(r2 @ r2)
        if np.isfinite(c2) and c2 <= cost:
            r, J, cost, damp = r2, J2, c2, max(damp / 10, 1e-12)
        else:
            m.add_flat(delta)
            damp *= 10
        if cost == 0.0:
            break
    return cost


def small_init(monos, tied, s, x0, N, rng):
    """UNFOLD.md's start: each term moves the phase by ~1e-3 rad; radix B uniform in [1.5, 4]."""
    m = GradModel(monos, tied, np.zeros(len(monos)), rng.uniform(1.5, 4.0) if tied else 10.0)
    _, J = m.v_and_jac(s, x0, N)
    mag = np.maximum((np.abs(J[:, :len(monos)]) / np.asarray(N, dtype=np.float64)[:, None]).mean(0), 1e-300)
    m.w = 1e-3 * rng.standard_normal(len(monos)) / (2 * math.pi * mag)
    return m


def true_wraps(c_true, s, x0, N):
    """floor(f(s, x0) / N) under the hidden recurrence, exact (the oracle's easiest-first key)."""
    out = []
    for si, xi, ni in zip(np.asarray(s).tolist(), np.asarray(x0).tolist(), np.asarray(N).tolist()):
        v = sum(cv * si ** a * xi ** b * ni ** c for cv, (a, b, c) in zip(c_true, MONOS) if cv)
        out.append(v // ni)
    return np.array(out, dtype=np.float64)


def grad_run(c_true, rows, model, order, weight, seed, stages=48, iters=3):
    """UNFOLD.md section 4 on the relabeled rows. Phase A: T=1 rows, easiest first; stage 0 fits the rows the order
    calls non-wrapping by plain regression, then 48 clock-loss stages grow the prefix from 5 to 1,000 rows.
    Phase B: all T=1 rows plus a growing prefix of T=2 rows; a T=2 row's middle state is the model's own snapped
    T=1 answer, recomputed every stage. Orders: 'oracle' = the true wrap count under the hidden recurrence;
    'free' = label-free, x0^3 / N (T=1) and max(s1_hat, x0)^3 / N (T=2)."""
    monos, tied = GRAD_MODELS[model]
    rng = np.random.default_rng(seed)
    r1, r2 = rows[1], rows[2]
    N1, X1, Y1 = r1["N"], r1["X"], r1["Y"]
    N2, X2, Y2 = r2["N"], r2["X"], r2["Y"]
    t0 = time.time()
    m = small_init(monos, tied, X1, X1, N1, rng)
    x3 = X1.astype(np.float64) ** 3 / N1
    if order == "oracle":
        k1 = true_wraps(c_true, X1, X1, N1)
        o1, easy1 = np.argsort(k1, kind="stable"), int((k1 == 0).sum())
    else:
        o1, easy1 = np.argsort(x3, kind="stable"), int((X1.astype(object) ** 3 < N1.astype(object)).sum())
    h1_phase_a(m, r1, o1, easy1, weight, stages, iters)
    diag_float = float((m.exact_state(X1, X1, N1) == Y1).mean())
    phase_a = {"w": m.w.tolist(), "B": m.B, "rounded": formula(m.rounded()[0], monos, m.rounded()[1])}
    # phase B
    s1_true = relabel(c_true, N2, X2, np.ones(len(N2), np.int64))
    k2 = true_wraps(c_true, s1_true, X2, N2) if order == "oracle" else None
    for k in range(-1, stages):
        s1 = m.exact_state(X2, X2, N2)
        if order == "oracle":
            o2, easy2 = np.argsort(k2, kind="stable"), int((k2 == 0).sum())
        else:
            mx = np.maximum(s1, X2).astype(object)
            o2 = np.argsort((mx.astype(np.float64) ** 3) / N2, kind="stable")
            easy2 = int((mx ** 3 < N2.astype(object)).sum())
        if k == -1:                                         # stage 0: plain regression on the non-wrapping rows
            if easy1 + easy2 == 0:
                continue
            i1, i2 = o1[:easy1], o2[:easy2]
            wrap, it = False, iters * 10
        else:
            i1, i2 = np.arange(len(N1)), o2[:int(round(5 * (len(N2) / 5) ** (k / (stages - 1))))]
            wrap, it = True, iters
        lm(m, np.concatenate([X1[i1], s1[i2]]), np.concatenate([X1[i1], X2[i2]]),
           np.concatenate([N1[i1], N2[i2]]), np.concatenate([Y1[i1], Y2[i2]]), iters=it, wrap=wrap, weight=weight)
    s1 = m.exact_state(X2, X2, N2)
    t1_float = float((m.exact_state(X1, X1, N1) == Y1).mean())
    t2_float = float((m.exact_state(s1, X2, N2) == Y2).mean())
    c_round, B_round = m.rounded()
    res = {"seed": seed, "sec": time.time() - t0, "easy_t1": easy1, "diag_float_after_A": diag_float,
           "t1_float": t1_float, "t2_float": t2_float, "w": m.w.tolist(), "B": m.B,
           "formula_rounded": formula(c_round, monos, B_round), "coef_rounded": c_round, "phase_a": phase_a}
    res.update(check(c_round, c_true, rows, monos, B_round))
    return res


def h1_phase_a(m, r1, o1, easy1, weight, stages, iters):
    """T=1 rows in the order o1: plain regression on the first easy1 rows, then clock-loss stages from 5 to all."""
    N1, X1, Y1 = r1["N"], r1["X"], r1["Y"]
    if easy1:
        idx = o1[:easy1]
        lm(m, X1[idx], X1[idx], N1[idx], Y1[idx], iters=iters * 10, wrap=False, weight=weight)
    for k in range(stages):
        idx = o1[:int(round(5 * (len(N1) / 5) ** (k / (stages - 1))))]
        lm(m, X1[idx], X1[idx], N1[idx], Y1[idx], iters=iters, weight=weight)


def grad_mode(a):
    out = []
    for task in a.tasks:
        c_true = as_vec(TASKS[task])
        rows = train_rows(c_true)
        for model in a.models:
            for order in a.orders:
                for weight in a.weights:
                    runs = [grad_run(c_true, rows, model, order, weight, seed, a.stages, a.iters)
                            for seed in range(a.seeds)]
                    r = {"task": task, "model": model, "order": order, "weight": weight, "seeds": a.seeds,
                         "certified": sum(x["certified"] for x in runs),
                         "is_hidden": sum(x["is_hidden"] for x in runs),
                         "diag_float_after_A": sum(x["diag_float_after_A"] == 1.0 for x in runs),
                         "t1_float": sum(x["t1_float"] == 1.0 for x in runs),
                         "t2_float": sum(x["t2_float"] == 1.0 for x in runs),
                         "sec_per_seed": float(np.mean([x["sec"] for x in runs])),
                         "formulas": count([x["formula_rounded"] for x in runs]), "runs": runs}
                    out.append(r)
                    print(f"grad {task:10s} {model:6s} {order:6s} {weight:4s}: certified {r['certified']}/{a.seeds}; "
                          f"fp64 model exact on T=1 after phase A {r['diag_float_after_A']}/{a.seeds}, at the end "
                          f"T=1 {r['t1_float']}/{a.seeds} T=2 {r['t2_float']}/{a.seeds}; rounded "
                          f"{list(r['formulas'].items())[:3]}  ({r['sec_per_seed']:.1f}s/seed)", flush=True)
    return out


# ------------------------------------------------------------------------------- gradient on a fold curriculum
def fresh_rows(rng, bits, n, e, c_true, T):
    """n fresh rows: N odd with `bits` bits, x0 uniform in [1, max(2, N^e)), y = f^T(x0) under the hidden step."""
    N = rng.integers(1 << (bits - 2), 1 << (bits - 1), n, dtype=np.int64) * 2 + 1
    hi = np.maximum(2.0, np.minimum(N.astype(np.float64) ** e, N - 1.0))
    x0 = 1 + (rng.random(n) * (hi - 1)).astype(np.int64)
    return N, x0, relabel(c_true, N, x0, np.full(n, T))


def fold_phase_a(m, rng, bits, n, c_true, stages, iters, weight):
    """Fresh T=1 rows with x0 < N^e, e from 0.1 to 1, one clock-loss LM call per stage."""
    for k in range(stages):
        e = 0.1 + 0.9 * k / (stages - 1)
        N, x0, y = fresh_rows(rng, bits, n, e, c_true, 1)
        lm(m, x0, x0, N, y, iters=iters, weight=weight)


def fold_run(c_true, rows, model, bits, seed, stages=32, n=2048, iters=3, weight="none"):
    """Phase A: T=1 rows with x0 < N^e, e from 0.1 to 1. Phase B: full-range T=1 rows plus T=2 rows with x0 < N^e,
    e from 0.05 to 1, the middle state from the model's own snapped T=1 answer. Fresh rows every stage."""
    monos, tied = GRAD_MODELS[model]
    rng = np.random.default_rng(1000 * bits + seed)
    t0 = time.time()
    N, x0, y = fresh_rows(rng, bits, n, 0.1, c_true, 1)
    m = small_init(monos, tied, x0, x0, N, rng)
    carry = []
    fold_phase_a(m, rng, bits, n, c_true, stages, iters, weight)
    Ne, xe, ye = fresh_rows(np.random.default_rng(seed + 77), bits, 4096, 1.0, c_true, 1)
    diag_fresh = float((m.exact_state(xe, xe, Ne) == ye).mean())
    for k in range(stages):
        e = 0.05 + 0.95 * k / (stages - 1)
        Na, xa, ya = fresh_rows(rng, bits, n, 1.0, c_true, 1)
        Nb, xb, yb = fresh_rows(rng, bits, n, e, c_true, 2)
        sb = m.exact_state(xb, xb, Nb)
        before = float((m.exact_state(sb, xb, Nb) == yb).mean())
        lm(m, np.concatenate([xa, sb]), np.concatenate([xa, xb]), np.concatenate([Na, Nb]),
           np.concatenate([ya, yb]), iters=iters, weight=weight)
        carry.append((round(e, 3), round(before, 3)))
    Ne2, xe2, ye2 = fresh_rows(np.random.default_rng(seed + 78), bits, 4096, 1.0, c_true, 2)
    t2_fresh = float((m.exact_state(m.exact_state(xe2, xe2, Ne2), xe2, Ne2) == ye2).mean())
    c_round, B_round = m.rounded()
    res = {"seed": seed, "bits": bits, "sec": time.time() - t0, "diag_fresh_after_A": diag_fresh,
           "t2_fresh_float": t2_fresh, "carry_t2_before": carry, "w": m.w.tolist(), "B": m.B,
           "formula_rounded": formula(c_round, monos, B_round), "coef_rounded": c_round}
    res.update(check(c_round, c_true, rows, monos, B_round))
    return res


def fold_mode(a):
    out = []
    for task in a.tasks:
        c_true = as_vec(TASKS[task])
        rows = train_rows(c_true)
        for model in a.models:
            for bits in a.bits_list:
                runs = [fold_run(c_true, rows, model, bits, seed, a.stages, a.fold_rows, a.iters, a.weights[0])
                        for seed in range(a.seeds)]
                r = {"task": task, "model": model, "bits": bits, "seeds": a.seeds,
                     "certified": sum(x["certified"] for x in runs), "is_hidden": sum(x["is_hidden"] for x in runs),
                     "diag_fresh_after_A": sum(x["diag_fresh_after_A"] == 1.0 for x in runs),
                     "t2_fresh_float": sum(x["t2_fresh_float"] == 1.0 for x in runs),
                     "sec_per_seed": float(np.mean([x["sec"] for x in runs])),
                     "formulas": count([x["formula_rounded"] for x in runs]), "runs": runs}
                out.append(r)
                print(f"fold {task:10s} {model:6s} b={bits}: certified {r['certified']}/{a.seeds}; fp64 model exact "
                      f"on fresh T=1 after phase A {r['diag_fresh_after_A']}/{a.seeds}, fresh T=2 at the end "
                      f"{r['t2_fresh_float']}/{a.seeds}; rounded {list(r['formulas'].items())[:3]}  "
                      f"({r['sec_per_seed']:.1f}s/seed)", flush=True)
    return out


# ------------------------------------------------------- gradient with the diagonal frozen: learn only the rest
Q_MONOS = [(a, d - a) for d in range(3) for a in range(d, -1, -1)]      # q(s, x0): the 6 monomials of degree <= 2


def expand_q(diag, q):
    """f(s, x0) = g(x0) + (s - x0) q(s, x0) as integer coefficients over MONOS. Every f with f(x, x) = g(x) has
    this form, so with g frozen the T=1 rows constrain nothing else and only T=2 rows can move q."""
    f = {}
    for k, dk in enumerate(diag):
        f[(0, k, 0)] = f.get((0, k, 0), 0) + dk
    for qv, (a, b) in zip(q, Q_MONOS):
        f[(a + 1, b, 0)] = f.get((a + 1, b, 0), 0) + qv
        f[(a, b + 1, 0)] = f.get((a, b + 1, 0), 0) - qv
    return as_vec(f)


def q_system(diag, N, X, Y):
    """T=2 rows: the middle state s1 = g(x0) mod N from the frozen diagonal, the residual V2 - y = off + J q with
    off = g(x0) - y exact and J = (s1 - x0) * monomials of q."""
    s1 = step_mod(as_vec({(0, k, 0): dk for k, dk in enumerate(diag)}), X, X, N)
    off = [sum(dk * x ** k for k, dk in enumerate(diag)) - yy for x, yy in zip(X.tolist(), Y.tolist())]
    sf, xf = s1.astype(np.float64), X.astype(np.float64)
    J = (sf - xf)[:, None] * np.stack([sf ** a * xf ** b for a, b in Q_MONOS], 1)
    return s1, off, J


def lm_q(q, J, off, N, iters=3, wrap=True, damp=1e-6):
    """LM on (off + J q) / N, wrapped to [-1/2, 1/2) for the clock loss; off is reduced mod N exactly first."""
    Nf = np.asarray(N, dtype=np.float64)
    offf = np.array([o % n if wrap else o for o, n in zip(off, np.asarray(N).tolist())], dtype=np.float64)

    def resid(qq):
        r = offf + J @ qq
        return (np.mod(r + Nf / 2, Nf) - Nf / 2 if wrap else r) / Nf

    Jn0 = J / Nf[:, None]
    col = np.maximum(np.linalg.norm(Jn0, axis=0), 1e-300)
    Jn = Jn0 / col
    A = Jn.T @ Jn
    r = resid(q)
    cost = float(r @ r)
    for _ in range(iters * 4):
        q2 = q - np.linalg.solve(A + damp * np.eye(len(q)), Jn.T @ r) / col
        r2 = resid(q2)
        c2 = float(r2 @ r2)
        if c2 <= cost:
            q, r, cost, damp = q2, r2, c2, max(damp / 10, 1e-12)
        else:
            damp *= 10
        if cost == 0.0:
            break
    return q


def qgrad_run(c_true, rows, seed, order="oracle", bits=None, stages=48, iters=3, n=2048):
    """T=1 phase as in grad (H1 rows) or fold (fresh rows, `bits`), poly10. Then the diagonal is rounded and frozen,
    and only q is trained on T=2 rows by the clock loss. Order on H1: 'oracle' (true wrap count, stage 0 = plain
    regression on the rows that don't wrap) or 'free' (|s1 - x0| max(s1, x0)^2 / N, the size of the q part)."""
    monos, tied = GRAD_MODELS["poly10"]
    t0 = time.time()
    if bits is None:
        rng = np.random.default_rng(seed)
        r1 = rows[1]
        m = small_init(monos, tied, r1["X"], r1["X"], r1["N"], rng)
        if order == "oracle":
            k1 = true_wraps(c_true, r1["X"], r1["X"], r1["N"])
            o1, easy1 = np.argsort(k1, kind="stable"), int((k1 == 0).sum())
        else:
            o1 = np.argsort(r1["X"].astype(np.float64) ** 3 / r1["N"], kind="stable")
            easy1 = int((r1["X"].astype(object) ** 3 < r1["N"].astype(object)).sum())
        h1_phase_a(m, r1, o1, easy1, "none", stages, iters)
    else:
        rng = np.random.default_rng(1000 * bits + seed)
        N, x0, y = fresh_rows(rng, bits, n, 0.1, c_true, 1)
        m = small_init(monos, tied, x0, x0, N, rng)
        fold_phase_a(m, rng, bits, n, c_true, stages, iters, "none")
    diag = [int(round(sum(w for w, (a, b, c) in zip(m.w, monos) if a + b == k))) for k in range(4)]
    q = np.zeros(len(Q_MONOS))
    if bits is None:
        r2 = rows[2]
        N2, X2, Y2 = r2["N"], r2["X"], r2["Y"]
        s1, off, J = q_system(diag, N2, X2, Y2)
        q = 1e-3 * rng.standard_normal(len(Q_MONOS)) / (2 * math.pi * np.maximum(np.abs(J / N2[:, None]).mean(0), 1e-300))
        if order == "oracle":
            k2 = true_wraps(c_true, relabel(c_true, N2, X2, np.ones(len(N2), np.int64)), X2, N2)
            o2, easy2 = np.argsort(k2, kind="stable"), int((k2 == 0).sum())
        else:
            mx = np.maximum(s1, X2).astype(np.float64)
            o2, easy2 = np.argsort(np.abs(s1 - X2) * mx ** 2 / N2, kind="stable"), 0
        if easy2:
            idx = o2[:easy2]
            q = lm_q(q, J[idx], [off[i] for i in idx], N2[idx], iters * 10, wrap=False)
        for k in range(stages):
            idx = o2[:int(round(5 * (len(N2) / 5) ** (k / (stages - 1))))]
            q = lm_q(q, J[idx], [off[i] for i in idx], N2[idx], iters)
    else:
        for k in range(stages):
            e = 0.05 + 0.95 * k / (stages - 1)
            Nb, xb, yb = fresh_rows(rng, bits, n, e, c_true, 2)
            _, off, J = q_system(diag, Nb, xb, yb)
            q = lm_q(q, J, off, Nb, iters)
    c_hat = expand_q(diag, [int(round(v)) for v in q])
    res = {"seed": seed, "bits": bits, "order": order, "sec": time.time() - t0, "diag": diag,
           "diag_ok": diag == diagonal(c_true), "q": q.tolist(), "formula_rounded": formula(c_hat), "coef_rounded": c_hat}
    res.update(check(c_hat, c_true, rows))
    return res


def qgrad_mode(a):
    out = []
    for task in a.tasks:
        c_true = as_vec(TASKS[task])
        rows = train_rows(c_true)
        settings = [("h1", o, None) for o in a.orders] + [("fold", "fold", b) for b in a.bits_list]
        for data, order, bits in settings:
            runs = [qgrad_run(c_true, rows, seed, order, bits, a.stages if data == "h1" else 32, a.iters, a.fold_rows)
                    for seed in range(a.seeds)]
            r = {"task": task, "data": data, "order": order, "bits": bits, "seeds": a.seeds,
                 "certified": sum(x["certified"] for x in runs), "diag_ok": sum(x["diag_ok"] for x in runs),
                 "sec_per_seed": float(np.mean([x["sec"] for x in runs])),
                 "formulas": count([x["formula_rounded"] for x in runs]), "runs": runs}
            out.append(r)
            print(f"qgrad {task:10s} {data:4s} {order:6s} {bits or '':>2}: certified {r['certified']}/{a.seeds}; "
                  f"rounded diagonal right {r['diag_ok']}/{a.seeds}; found {list(r['formulas'].items())[:3]}  "
                  f"({r['sec_per_seed']:.1f}s/seed)", flush=True)
    return out


def report(run_dir="runs/unfold_family"):
    """Markdown tables for UNFOLD_FAMILY.md from the Modal runs (runs/unfold_family/*.json)."""
    import glob
    import os

    def load(pattern):
        out = []
        for f in sorted(glob.glob(os.path.join(run_dir, pattern))):
            out.append(json.load(open(f)))
        return out

    def cell(r):
        return f"{r['certified']}/{r['trials']}"

    order = list(TASKS)
    lat = {d["results"][0]["task"]: d["results"] for d in load("lattice_*.json")
           if d["results"] and d["results"][0].get("family", "all") == "all"}
    if lat:
        print("### Lattice, menu recurrences: certified / 20 trials (T=1 rows m1, T=2 rows m2)\n")
        m1s = sorted({r["m1"] for rs in lat.values() for r in rs})
        m2s = sorted({r["m2"] for rs in lat.values() for r in rs})
        for m2 in m2s:
            print(f"m2 = {m2}:\n")
            print("| hidden step | " + " | ".join(f"m1 = {m}" for m in m1s) + " |")
            print("|---|" + "---|" * len(m1s))
            for t in order:
                if t in lat:
                    cs = [next(r for r in lat[t] if r["m1"] == m and r["m2"] == m2) for m in m1s]
                    print(f"| {t} | " + " | ".join(cell(r) for r in cs) + " |")
            print()
        print("Stage-1 diagonal exact on all 1,000 T=1 rows, m2 = 0 cells, per m1:\n")
        for t in order:
            if t in lat:
                print(f"- {t}: " + ", ".join(f"m1={r['m1']}: {r['diag_ok_stage1']}/{r['trials']}"
                                             for r in lat[t] if r["m2"] == 0))
        print("\nT=1 rows only (m2 = 0), formulas found over all m1 cells:\n")
        for t in order:
            if t in lat:
                agg = {}
                for r in lat[t]:
                    if r["m2"] == 0:
                        for k, v in r["formulas"].items():
                            agg[k] = agg.get(k, 0) + v
                print(f"- {t}: " + ", ".join(f"{k} {v}" for k, v in sorted(agg.items(), key=lambda kv: -kv[1])))
        tot = [(r["certified"], r["trials"]) for rs in lat.values() for r in rs if r["m2"] >= 1 and r["m1"] >= 2]
        tot1 = [(r["certified"], r["trials"]) for rs in lat.values() for r in rs if r["m2"] >= 1 and r["m1"] == 1]
        print(f"\nWith T=2 rows: m1 >= 2: {sum(a for a, _ in tot)}/{sum(b for _, b in tot)} certified; "
              f"m1 = 1: {sum(a for a, _ in tot1)}/{sum(b for _, b in tot1)}.")
        secs = [t["sec"] for rs in lat.values() for r in rs for t in r["runs"]]
        print(f"Time per trial (both stages, python-flint LLL): median {np.median(secs) * 1e3:.1f} ms, "
              f"max {np.max(secs) * 1e3:.0f} ms over {len(secs)} trials.\n")
    son = [d for d in load("lattice_sonly.json")]
    if son:
        print("### Lattice, T=1 rows only, step reads s and N but never x0 (certified / 20)\n")
        rs = son[0]["results"]
        m1s = sorted({r["m1"] for r in rs})
        print("| hidden step | " + " | ".join(f"m1 = {m}" for m in m1s) + " | formula found |")
        print("|---|" + "---|" * (len(m1s) + 1))
        for t in order:
            cs = [next(r for r in rs if r["task"] == t and r["m1"] == m) for m in m1s]
            print(f"| {t} | " + " | ".join(cell(r) for r in cs) + f" | {', '.join(cs[-1]['formulas'])} |")
        print()
    amb = load("ambiguity.json")
    if amb:
        a = amb[0]["results"]
        print("### The T<=2 blind spot (s - x0)(s - g(x0)), g = f(x, x) quadratic\n")
        print("| hidden step | neighbour f + k h | k | norm^2 (hidden) | T=1 rows | T=2 rows | T=4 rows | certified max T |")
        print("|---|---|---|---|---|---|---|---|")
        for r in a["neighbours"]:
            if r.get("blind_spot") is None:
                continue
            print(f"| {r['task']} | {r['formula']} | {r['k']:+d} | {r['norm2']} ({r['norm2_hidden']}) | "
                  f"{r['t1_exact']:.3f} | {r['t2_exact']:.3f} | {r['t4_agree']:.4f} | {r['depth_t']} / {r['depth_ood_n_t']} |")
        print("\nAdversarial hidden steps (2 T=1 + 2 T=2 rows, then m4 T=4 rows for stage 3):\n")
        print("| hidden step | m4 = 0 | m4 = 1 | m4 = 2 | found at m4 = 0 |")
        print("|---|---|---|---|---|")
        for t in ADVERSARIAL:
            rs = [r for r in a["adversarial"] if r["task"] == t]
            by = {r["m4"]: r for r in rs}
            print(f"| {by[0]['hidden']} | " + " | ".join(cell(by[m]) if m in by else "" for m in (0, 1, 2))
                  + f" | {', '.join(by[0]['formulas'])} |")
        print()
    rnd = load("random_C*.json")
    if rnd:
        print("### Lattice, random hidden steps: 10 polynomials x 5 trials per cell, m T=1 + m T=2 rows\n")
        rnd.sort(key=lambda d: d["results"][0]["C"])
        ms = sorted({r["m1"] for r in rnd[0]["results"]})
        print("| coefficients in | " + " | ".join(f"m = {m}" for m in ms) + " | median time at m = 24 |")
        print("|---|" + "---|" * (len(ms) + 1))
        for d in rnd:
            rs = d["results"]
            C = rs[0]["C"]
            cs = [(sum(r["certified"] for r in rs if r["m1"] == m), sum(r["trials"] for r in rs if r["m1"] == m))
                  for m in ms]
            t24 = np.median([t["sec"] for r in rs if r["m1"] == max(ms) for t in r["runs"]])
            print(f"| [-{C}, {C}] | " + " | ".join(f"{a}/{b}" for a, b in cs) + f" | {t24:.2f} s |")
        print()
    dig = {d["results"][0]["task"]: d["results"] for d in load("digits_*.json")}
    if dig:
        print("### Lattice, digits through one shared base (B = 2..16 scanned): base 10 picked alone, certified (of 10)\n")
        m1s = sorted({r["m1"] for rs in dig.values() for r in rs})
        m2s = sorted({r["m2"] for rs in dig.values() for r in rs})
        print("| hidden step | " + " | ".join(f"m1={a}, m2={b}" for a in m1s for b in m2s) + " |")
        print("|---|" + "---|" * (len(m1s) * len(m2s)))
        for t in order:
            if t in dig:
                cs = [next(r for r in dig[t] if r["m1"] == a and r["m2"] == b) for a in m1s for b in m2s]
                print(f"| {t} | " + " | ".join(f"{r['base10_unique']}, {r['certified']}" for r in cs) + " |")
        secs = [t["sec"] for rs in dig.values() for r in rs for t in r["runs"]]
        print(f"\nTime per trial (15 stage-1 lattices + stage 2): median {np.median(secs) * 1e3:.0f} ms.\n")
    grad = [d for d in load("grad_*.json")]
    if grad:
        print("### Gradient on H1's own rows: certified / 5 seeds (oracle order / label-free order; weighting none, sens)\n")
        rows = {}
        for d in grad:
            for r in d["results"]:
                rows[(r["task"], r["model"], r["order"], r["weight"])] = r
        models = [m for m in GRAD_MODELS if any(k[1] == m for k in rows)]
        print("| hidden step | " + " | ".join(models) + " |")
        print("|---|" + "---|" * len(models))
        for t in order:
            cells = []
            for m in models:
                rr = [rows.get((t, m, o, w)) for o in ("oracle", "free") for w in ("none", "sens")]
                cells.append("" if rr[0] is None else
                             f"{rr[0]['certified']}, {rr[1]['certified']} / {rr[2]['certified']}, {rr[3]['certified']}")
            print(f"| {t} | " + " | ".join(cells) + " |")
        print("\nfp64 model exact on all T=1 rows after the T=1 phase (oracle order, no weighting):\n")
        for t in order:
            print(f"- {t}: " + ", ".join(f"{m} {rows[(t, m, 'oracle', 'none')]['diag_float_after_A']}/5"
                                         for m in models if (t, m, "oracle", "none") in rows))
        secs = [x["sec"] for r in rows.values() for x in r["runs"]]
        print(f"\nTime per seed: median {np.median(secs):.1f} s, max {np.max(secs):.1f} s.\n")
    fold = [d for d in load("fold_*.json")]
    if fold:
        print("### Gradient on fresh rows, fold curriculum: certified / 5 (T=1 phase exact on fresh rows / 5)\n")
        rows = {}
        for d in fold:
            for r in d["results"]:
                rows[(r["task"], r["model"], r["bits"])] = r
        models = [m for m in GRAD_MODELS if any(k[1] == m for k in rows)]
        bits = sorted({k[2] for k in rows})
        print("| hidden step | model | " + " | ".join(f"{b} bits" for b in bits) + " |")
        print("|---|---|" + "---|" * len(bits))
        for t in order:
            for m in models:
                if (t, m, bits[0]) in rows:
                    print(f"| {t} | {m} | " + " | ".join(
                        f"{rows[(t, m, b)]['certified']} ({rows[(t, m, b)]['diag_fresh_after_A']})" for b in bits) + " |")
        secs = [x["sec"] for r in rows.values() for x in r["runs"]]
        print(f"\nTime per seed: median {np.median(secs):.1f} s.\n")
    qg = [r for d in load("qgrad_*.json") for r in d["results"]]
    if qg:
        print("### Gradient with the diagonal frozen (only q trained on T=2 rows): certified / 5 (rounded diagonal right / 5)\n")
        cols = [("h1", "oracle", None), ("h1", "free", None)] + [("fold", "fold", b) for b in (12, 16, 20, 24)]
        print("| hidden step | H1 oracle | H1 label-free | " + " | ".join(f"fold {b} bits" for _, _, b in cols[2:]) + " |")
        print("|---|" + "---|" * len(cols))
        for t in order:
            cs = [next((r for r in qg if r["task"] == t and r["data"] == d and r["order"] == o and r["bits"] == b), None)
                  for d, o, b in cols]
            if all(c is not None for c in cs):
                print(f"| {t} | " + " | ".join(f"{c['certified']} ({c['diag_ok']})" for c in cs) + " |")
        print()


# ------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "lattice", "random", "digits", "ambiguity", "grad", "fold", "qgrad",
                                     "report"])
    ap.add_argument("--family", default="all", choices=list(FAMILIES), help="lattice: monomials (sn = no x0)")
    ap.add_argument("--m4", type=int, nargs="+", default=[0, 1, 2], help="ambiguity: T=4 rows for stage 3")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--m1", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--m2", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coef_max", type=int, nargs="+", default=[1, 3, 10, 100, 1000])
    ap.add_argument("--polys", type=int, default=10)
    ap.add_argument("--base_lo", type=int, default=2)
    ap.add_argument("--base_hi", type=int, default=16)
    ap.add_argument("--models", nargs="+", default=["poly20", "poly10", "tied10", "tied20"])
    ap.add_argument("--orders", nargs="+", default=["oracle", "free"])
    ap.add_argument("--weights", nargs="+", default=["none", "sens"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--stages", type=int, default=48)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--bits_list", type=int, nargs="+", default=[12, 16, 20, 24])
    ap.add_argument("--fold_rows", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=0, help="accepted for the Modal runner; numpy uses its default")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    if a.mode == "check":
        self_check()
        print("TASKS match menu.py's lambdas on 300 H1 rows (T up to 16)")
        for task, spec in TASKS.items():
            c = as_vec(spec)
            print(task, formula(c), certify(c, c))
        return
    if a.mode == "report":
        report()
        return
    res = {"lattice": lattice_mode, "random": random_mode, "digits": digits_mode, "ambiguity": ambiguity_mode,
           "grad": grad_mode, "fold": fold_mode, "qgrad": qgrad_mode}[a.mode](a)
    if a.out:
        json.dump({"args": vars(a), "results": res, "wall": time.time() - t0}, open(a.out, "w"), indent=1)
    print(f"wall {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
