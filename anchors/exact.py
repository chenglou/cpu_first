"""Exact map of which subtraction tables pass which lessons, by complete search (ANCHORS.md §2).

A table is 2 B^2 digit entries (values 0..B-1) and 2 B^2 borrow entries (0/1), flat over (r, m, c) -> (r B + m) 2 + c.
Every lesson is a list of instances on fixed inputs (all inputs of the tiny machine, or a data sample). An instance is
evaluated on a PARTIAL table: it passes, fails, or gets stuck on the first entry it needs that is still unassigned.
A depth-first search assigns entries one at a time, always one that some instance is stuck on, and backtracks when
an instance fails; when no instance is stuck the remaining entries are free (counted, not enumerated). So the search
returns every table passing all instances, exactly, as a list of (assignment of the entries that matter, number of
free entries).

    python exact.py B W LESSON[,LESSON...] [--borrows-true] [--inputs all|data] [--limit K]
"""

from __future__ import annotations

import itertools
import math
import sys
import time

STUCK = object()


class Table:
    def __init__(self, B, digits=None, borrows=None):
        self.B, self.n = B, 2 * B * B
        self.d = list(digits) if digits is not None else [-1] * self.n
        self.b = list(borrows) if borrows is not None else [-1] * self.n


class Stuck(Exception):
    def __init__(self, var):
        self.var = var                                                    # ("d", i) or ("b", i)


def idx(B, r, m, c):
    return (r * B + m) * 2 + c


def sub(t: Table, A, M, need_digits=True):
    """A - M (digit lists, LSB first) through the partial table: digits, final borrow. Raises Stuck on the first
    entry it needs that is unassigned (with need_digits=False only the borrows are read)."""
    B, c, out = t.B, 0, []
    for a, m in zip(A, M):
        i = idx(B, a, m, c)
        if t.b[i] < 0:
            raise Stuck(("b", i))
        if need_digits:
            if t.d[i] < 0:
                raise Stuck(("d", i))
            out.append(t.d[i])
        c = t.b[i]
    return out, c


def borrow(t, A, M):
    return sub(t, A, M, need_digits=False)[1]


def digits(v, B, n):
    return [(v // B ** j) % B for j in range(n)]


def value(d, B):
    return sum(x * B ** j for j, x in enumerate(d))


def step(t, Rp, M, S):
    """One division step through the partial table: the kept trial (the last before the first borrow)."""
    D, R = Rp, Rp
    for _ in range(1, S):
        if borrow(t, D, M) == 1:
            return R
        D, _ = sub(t, D, M)
        R = D
    return R


def divide(t, v, M, B, W, P):
    R = digits(v // B ** W, B, P)
    for s in reversed(range(W)):
        R = step(t, [(v // B ** s) % B] + R[:-1], M, B)
    return R


# --------------------------------------------------------------------------------------------------------- lessons
def lesson_instances(name, B, W, rows, links):
    """Instances of one lesson as closures t -> bool (raise Stuck). rows: (N, x) pairs; links: (D, N) pairs (the
    link inputs the lesson reads)."""
    P = W + 1
    dg = lambda v: digits(v, B, P)
    out = []
    if name == "lc":                                                     # (D - N) - (0 - N) = D, digits
        for D, N in links:
            Dd, M = dg(D), dg(N)
            def f(t, Dd=Dd, M=M):
                E, _ = sub(t, Dd, M)
                C, _ = sub(t, [0] * P, M)
                F, _ = sub(t, E, C)
                return F == Dd
            out.append(f)
    elif name == "lcc":                                                  # the same, complete (final borrows too)
        for D, N in links:
            Dd, M = dg(D), dg(N)
            def f(t, Dd=Dd, M=M):
                E, be = sub(t, Dd, M)
                C, bc = sub(t, [0] * P, M)
                F, bf = sub(t, E, C)
                return F == Dd and be == bc - bf
            out.append(f)
    elif name == "dc":                                                   # (D - N) + N = D, hand-written adder
        for D, N in links:
            Dd, M = dg(D), dg(N)
            def f(t, Dd=Dd, M=M, D=D, N=N):
                E, be = sub(t, Dd, M)
                return value(E, B) == (D - N) % B ** P and be == int(D < N)
            out.append(f)
    elif name == "less":
        for N, x in rows:
            out.append(lambda t, a=dg(x), m=dg(N): borrow(t, a, m) == 1)
    elif name == "more":
        for N, x in rows:
            out.append(lambda t, a=dg(N), m=dg(x): borrow(t, a, m) == 0)
    elif name == "wrap":
        for N, x in rows:
            def f(t, a=dg(x), m=dg(N)):
                E, _ = sub(t, a, m)
                return borrow(t, E, m) == 0
            out.append(f)
    elif name == "zero":                                                 # R - 0 = R on the registers
        for R in sorted({D for D, N in links}):
            out.append(lambda t, a=dg(R): sub(t, a, [0] * P) == (a, 0))
    elif name == "self":                                                 # R - R = 0
        for R in sorted({D for D, N in links}):
            out.append(lambda t, a=dg(R): sub(t, a, a) == ([0] * P, 0))
    elif name == "neg":                                                  # 0 - N borrows
        for N in sorted({N for N, x in rows}):
            out.append(lambda t, m=dg(N): borrow(t, [0] * P, m) == 1)
    elif name == "neg2":                                                 # 0 - (0 - N) = N, complete
        for N in sorted({N for N, x in rows}):
            def f(t, m=dg(N)):
                C, bc = sub(t, [0] * P, m)
                Y, by = sub(t, [0] * P, C)
                return Y == m and by == bc
            out.append(f)
    elif name == "comm":                                                 # R + N = N + R (learned adder, complete)
        for D, N in links:
            def f(t, a=dg(D), m=dg(N)):
                C, bc = sub(t, [0] * P, m)
                A1, b1 = sub(t, a, C)
                CR, bcr = sub(t, [0] * P, a)
                A2, b2 = sub(t, m, CR)
                return A1 == A2 and bc - b1 == bcr - b2
            out.append(f)
    elif name == "xmod":                                                 # x mod N = x (one division step)
        for N, x in rows:
            out.append(lambda t, a=dg(x), m=dg(N): step(t, a, m, B) == a)
    elif name == "chain":                                                # after a borrowing link: no borrow
        for D, N in links:
            if D < N:
                def f(t, a=dg(D), m=dg(N)):
                    E, _ = sub(t, a, m)
                    return borrow(t, E, m) == 0
                out.append(f)
    elif name == "dec":                                                  # E < D iff the link does not borrow
        for D, N in links:
            if N > 0:
                def f(t, a=dg(D), m=dg(N), nb=int(D >= N)):
                    E, _ = sub(t, a, m)
                    return borrow(t, E, a) == nb
                out.append(f)
    elif name == "mon":                                                  # subtracting N keeps the order of two
        byN = {}                                                         # numbers, unless exactly one wraps
        for D, N in links:
            byN.setdefault(N, []).append(D)
        for N, Ds in byN.items():
            for D1, D2 in itertools.combinations(sorted(Ds), 2):          # D1 < D2
                t_ = int(D1 >= N or D2 < N)                              # [f(D1) < f(D2)]
                def f(t, a=dg(D1), b_=dg(D2), m=dg(N), t_=t_):
                    E1, _ = sub(t, a, m)
                    E2, _ = sub(t, b_, m)
                    return borrow(t, E1, E2) == t_
                out.append(f)
    elif name == "pref":                                                 # x < N on every top part
        for N, x in rows:
            a, m = dg(x), dg(N)
            for j in range(P):
                out.append(lambda t, a=a[j:], m=m[j:], dif=int(a[j:] != m[j:]): borrow(t, a, m) == dif)
                out.append(lambda t, a=a[j:], m=m[j:]: borrow(t, m, a) == 0)
    elif name in ("colord", "roword"):                                   # the order laws on the cell's entries,
        n = 2 * B * B                                                    # refereed by the table's own borrows
        for e0 in range(n):
            r, m, c = e0 // (2 * B), (e0 // 2) % B, e0 % 2
            for k in range(B):
                if name == "colord" and k > r:
                    e1, cmp = idx(B, k, m, c), idx(B, r, k, 0)           # r - k borrows: r < k
                    def f(t, e0=e0, e1=e1, cmp=cmp, kind=0):
                        return _order(t, e0, e1, cmp, kind)
                    out.append(f)
                if name == "roword" and k > m:
                    e1, cmp = idx(B, r, k, c), idx(B, m, k, 0)           # m - k borrows: m < k
                    def f(t, e0=e0, e1=e1, cmp=cmp, kind=1):
                        return _order(t, e0, e1, cmp, kind)
                    out.append(f)
    elif name in ("bijc", "bijr"):                                       # each column (m, c) / row (r, c): a
        for u in range(B):                                               # permutation of the digits
            for c in range(2):
                ents = [idx(B, r, u, c) for r in range(B)] if name == "bijc" else [idx(B, u, m, c) for m in range(B)]
                for e0, e1 in itertools.combinations(ents, 2):
                    out.append(lambda t, e0=e0, e1=e1: _digit(t, e0) != _digit(t, e1))
    elif name == "ans":                                                  # x^2 mod N through the machine
        for N, x in rows:
            out.append(lambda t, v=x * x, m=dg(N), y=dg(x * x % N): divide(t, v, m, B, W, P) == y)
    elif name == "ans2":                                                 # x^4 mod N (two squarings)
        for N, x in rows:
            def f(t, v=x * x, m=dg(N), y=dg(pow(x, 4, N))):
                s = value(divide(t, v, m, B, W, P), B)
                return divide(t, s * s, m, B, W, P) == y
            out.append(f)
    elif name == "nn0":                                                  # N^2 mod N = 0
        for N in sorted({N for N, x in rows}):
            out.append(lambda t, v=N * N, m=dg(N): divide(t, v, m, B, W, P) == [0] * P)
    else:
        raise KeyError(name)
    return out


def _digit(t, e):
    if t.d[e] < 0:
        raise Stuck(("d", e))
    return t.d[e]


def _bor(t, e):
    if t.b[e] < 0:
        raise Stuck(("b", e))
    return t.b[e]


def _order(t, e0, e1, cmp, kind):
    """The order law on two entries: kind 0, a column ((r, m, c), (k, m, c)); kind 1, a row ((r, m, c), (r, k, c)).
    cmp is the entry whose borrow says whether the first index is the smaller one."""
    w0, w1, lt = _bor(t, e0), _bor(t, e1), _bor(t, cmp)
    if w0 != w1:
        first = w0 < w1                                                  # the one that does not wrap is smaller
    else:
        first = (lt == 1) if kind == 0 else (lt == 0)                    # columns keep, rows reverse the order
    lo, hi = (e0, e1) if first else (e1, e0)
    return _digit(t, lo) < _digit(t, hi)


def tiny_inputs(B, W, nmax=None):
    """Every modulus N coprime to B with at most W digits (N >= 2), every x in [1, N), and every link input
    D in [0, B N) (all numbers a division step's chain can meet)."""
    Ns = [N for N in range(2, B ** W) if math.gcd(N, B) == 1]
    if nmax:
        Ns = Ns[:nmax]
    rows = [(N, x) for N in Ns for x in range(1, N)]
    links = [(D, N) for N in Ns for D in range(B * N)]
    return rows, links


def data_inputs(n_rows, seed=0, B=10, W=8):
    """H1 rows (N, x) and the TRUE machine's link inputs D of each row's x^2 division (the numbers training reads
    once the table is right)."""
    import core
    import torch
    d = core.h1_rows(n_rows, seed=seed)
    mc = core.Machine(B, W)
    td, tb = mc.true_tables()
    M = mc.digits(d["N"])
    _, ins, _ = mc.divide(d["x"] * d["x"], M, td, tb)
    X, K = ins.shape[0], ins.shape[1] * ins.shape[2]
    D = mc.value(ins.reshape(X * K, mc.P))
    Ns = d["N"].repeat_interleave(K)
    rows = list(zip(d["N"].tolist(), d["x"].tolist()))
    links = sorted(set(zip(D.tolist(), Ns.tolist())))
    return rows, links


# ---------------------------------------------------------------------------------------------------------- search
def search(B, instances, fixed_digits=None, fixed_borrows=None, limit=10 ** 6, order=None, seconds=None):
    """Every partial table that passes all instances (DFS with watched instances). Returns a list of
    (digits, borrows) with -1 for free entries, and the number of complete tables they stand for."""
    t = Table(B, fixed_digits, fixed_borrows)
    n = t.n
    sols, count = [], 0
    watch: dict = {}
    pending = list(range(len(instances)))

    def check(k):
        try:
            return instances[k](t), None
        except Stuck as s:
            return None, s.var

    def assign_and_check(var, val, todo):
        """Assign var, re-check the instances watching it; returns (ok, undo-info)."""
        kind, i = var
        (t.d if kind == "d" else t.b)[i] = val
        moved = []
        stack = list(todo)
        while stack:
            k = stack.pop()
            ok, v = check(k)
            if ok is False:
                return False, moved
            if ok is None:
                watch.setdefault(v, []).append(k)
                moved.append((v, k))
        return True, moved

    # initial pass
    for k in pending:
        ok, v = check(k)
        if ok is False:
            return [], 0
        if ok is None:
            watch.setdefault(v, []).append(k)

    def free_count():
        return sum(1 for x in t.d if x < 0), sum(1 for x in t.b if x < 0)

    t_end = None if seconds is None else time.time() + seconds
    timed_out = [False]

    def rec():
        nonlocal count
        if len(sols) >= limit or timed_out[0]:
            return
        if t_end is not None and time.time() > t_end:
            timed_out[0] = True
            return
        live = [v for v, ks in watch.items() if ks and ((t.d if v[0] == "d" else t.b)[v[1]] < 0)]
        if not live:
            fd, fb = free_count()
            sols.append((list(t.d), list(t.b)))
            count += B ** fd * 2 ** fb
            return
        var = max(live, key=lambda v: len(watch[v]))                     # the entry most instances wait on
        todo = watch.pop(var)
        dom = range(B) if var[0] == "d" else range(2)
        for val in dom:
            ok, moved = assign_and_check(var, val, todo)
            if ok:
                rec()
            for v, k in moved:
                watch[v].remove(k)
            (t.d if var[0] == "d" else t.b)[var[1]] = -1
            if len(sols) >= limit or timed_out[0]:
                break
        watch[var] = todo

    rec()
    search.timed_out = timed_out[0]
    return sols, count


def true_table(B):
    d, b = [], []
    for r in range(B):
        for m in range(B):
            for c in range(2):
                x = r - m - c
                d.append(x % B)
                b.append(int(x < 0))
    return d, b


def show(B, d, b):
    """Wrong (or free) entries against the truth."""
    td, tb = true_table(B)
    out = []
    for i in range(2 * B * B):
        r, m, c = i // (2 * B), (i // 2) % B, i % 2
        if d[i] != td[i] or b[i] != tb[i]:
            dd = "?" if d[i] < 0 else d[i]
            bb = "?" if b[i] < 0 else b[i]
            out.append(f"({r}{m}{c})->{dd}{bb}")
    return " ".join(out) if out else "TRUE"


def main(argv):
    B, W = int(argv[0]), int(argv[1])
    names = argv[2].split(",")
    borrows_true = "--borrows-true" in argv
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 20
    if "--data" in argv:                                                 # H1 rows and the true machine's links
        rows, links = data_inputs(int(argv[argv.index("--data") + 1]))
    else:
        rows, links = tiny_inputs(B, W)
    inst = [f for nm in names for f in lesson_instances(nm, B, W, rows, links)]
    td, tb = true_table(B)
    t0 = time.time()
    sols, count = search(B, inst, fixed_borrows=tb if borrows_true else None, limit=limit)
    print(f"B={B} W={W} lessons={names} borrows_true={borrows_true} instances={len(inst)} "
          f"solution_groups={len(sols)}{'+' if len(sols) >= limit else ''} tables={count} ({time.time() - t0:.1f}s)")
    for d, b in sols[:limit]:
        print("  ", show(B, d, b))


if __name__ == "__main__":
    main(sys.argv[1:])
