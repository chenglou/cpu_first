"""Candidate practice problems without curation (PRACTICE.md §2): every problem is a function the bare machine
(library/inventing/brm.py) should compute, A from X and N, posed on instances at the search size k and checked on
instances at a bigger size k2.

Three generic sources:
- laws: a general arithmetic library (succ, pred, add, monus, absdiff, mul, div, mod, min, max, lt, le, eq, gcd),
  each operation applied to the machine's two inputs in every binding, on inputs built from the data's own numbers
  by the division law's form a + q b: X = x, X = x + N, X = x + qN with q drawn from {0, 1}, X = x + QN with Q
  another row's x. Nothing here knows the target; most of these are irrelevant to it.
- transforms of the target: its pairs restricted by a predicate from a small grammar (t1 < t2 over the terms 1, 2,
  3, X, N, X+X, N+N, X+N, X·X, X·N, N·N), its answer projected (lowest bit, the rest, is it 0, below x, equal to
  x), and the partial computations of the learner's own near-miss programs for it: the values its registers hold
  (as problems of their own), and the rest of the way from each such value to the answer (y from X = value, N,
  when that is a function), with that remainder's restrictions too.
- dreams: random programs of the current library run on the target's inputs (arm D's, library/inventing/loop.py).

Plus the two references: the target itself and the curated list of arm B (library/inventing/target.py).
A problem is kept when it is defined on at least 4 instances at each size, every input and answer fits the machine
(values below its cap), the answers vary, and it is a function of the inputs (the same (X, N) never wants two answers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from library import dsl
from library.inventing import brm, target
from library.inventing.jobs import machine_for

K2 = 10                     # the check size: every pair with N <= 10


@dataclass
class Cand:
    name: str               # unique, e.g. "law/mod(X,N)/X=x+qN"
    source: str             # law | restrict | project | value | remainder | dream | curated | target
    formula: str            # the function in plain notation
    xs: list
    Ns: list
    want: list
    cx: list                # the check instances (size k2)
    cN: list
    cw: list
    parent: str = ""
    also: list = field(default_factory=list)     # other names that posed the same problem

    def key(self):
        return (tuple(self.xs), tuple(self.Ns), tuple(self.want), tuple(self.cx), tuple(self.cN), tuple(self.cw))

    def to_dict(self):
        return dict(name=self.name, source=self.source, formula=self.formula, parent=self.parent, also=self.also,
                    n=len(self.xs), n_check=len(self.cx))


def _ok(xs, Ns, want, k):
    cap = machine_for(k).cap
    if len(xs) < 4:
        return False
    if max(max(xs), max(Ns), max(want)) >= cap or min(min(xs), min(want)) < 0:
        return False
    if len(set(want)) < 2:
        return False
    seen = {}
    for a, b, w in zip(xs, Ns, want):
        if seen.setdefault((a, b), w) != w:
            return False
    return True


def make(name, source, formula, gen, k, k2=K2, parent=""):
    """gen(size) -> (xs, Ns, want) as int lists, or None. Returns a Cand or None if not well posed."""
    a, b = gen(k), gen(k2)
    if a is None or b is None or not _ok(*a, k) or not _ok(*b, k2):
        return None
    return Cand(name, source, formula, *[list(map(int, v)) for v in a], *[list(map(int, v)) for v in b],
                parent=parent)


# ---------------------------------------------------------------------------------------------------------------
# the data's own numbers

def data(size, seed=0):
    """The target's pairs at a size (x < N <= size), with Q (another row's x) and q (a coin) for the laws."""
    xs, Ns = target.pairs(size)
    rng = np.random.default_rng(seed + size)
    Q = xs[rng.permutation(len(xs))]
    q = rng.integers(0, 2, len(xs))
    return xs.astype(np.int64), Ns.astype(np.int64), Q.astype(np.int64), q.astype(np.int64)


# ---------------------------------------------------------------------------------------------------------------
# (a) the law library

OPS2 = {
    "add": (lambda a, b: a + b, "{a} + {b}"),
    "monus": (lambda a, b: max(a - b, 0), "{a} ∸ {b}"),
    "absdiff": (lambda a, b: abs(a - b), "|{a} − {b}|"),
    "mul": (lambda a, b: a * b, "{a}·{b}"),
    "div": (lambda a, b: a // b if b else None, "{a} div {b}"),
    "mod": (lambda a, b: a % b if b else None, "{a} mod {b}"),
    "min": (lambda a, b: min(a, b), "min({a}, {b})"),
    "max": (lambda a, b: max(a, b), "max({a}, {b})"),
    "lt": (lambda a, b: int(a < b), "[{a} < {b}]"),
    "le": (lambda a, b: int(a <= b), "[{a} ≤ {b}]"),
    "eq": (lambda a, b: int(a == b), "[{a} = {b}]"),
    "gcd": (lambda a, b: math.gcd(a, b), "gcd({a}, {b})"),
}
OPS1 = {"succ": (lambda a: a + 1, "{a} + 1"), "pred": (lambda a: max(a - 1, 0), "{a} ∸ 1")}
PATTERNS = {                     # what the X register holds, from the data's x, N and the law's q / Q
    "x": lambda x, N, Q, q: x,
    "x+N": lambda x, N, Q, q: x + N,
    "x+qN": lambda x, N, Q, q: x + q * N,
    "x+QN": lambda x, N, Q, q: x + Q * N,
}
PAT_WORDS = {"x": "X = x", "x+N": "X = x + N", "x+qN": "X = x + qN, q in {0, 1}", "x+QN": "X = x + QN, Q another x"}


def laws(k, k2=K2):
    out = []

    def gen_for(op, b1, b2, pat):
        def gen(size):
            x, N, Q, q = data(size)
            X = PATTERNS[pat](x, N, Q, q)
            reg = {"X": X, "N": N}
            xs, Ns, want = [], [], []
            for i in range(len(x)):
                v = op(int(reg[b1][i]), int(reg[b2][i])) if b2 else op(int(reg[b1][i]))
                if v is None:
                    continue
                xs.append(int(X[i]))
                Ns.append(int(N[i]))
                want.append(v)
            return xs, Ns, want
        return gen

    for name, (fn, words) in OPS2.items():
        for pat in PATTERNS:
            for b1, b2 in (("X", "N"), ("N", "X"), ("X", "X")) + ((("N", "N"),) if pat == "x" else ()):
                f = words.format(a=b1, b=b2)
                c = make(f"law/{name}({b1},{b2})/{pat}", "law", f"{f}, {PAT_WORDS[pat]}", gen_for(fn, b1, b2, pat),
                         k, k2)
                if c:
                    out.append(c)
    for name, (fn, words) in OPS1.items():
        for pat in PATTERNS:
            for b1 in ("X",) + (("N",) if pat == "x" else ()):
                f = words.format(a=b1)
                c = make(f"law/{name}({b1})/{pat}", "law", f"{f}, {PAT_WORDS[pat]}", gen_for(fn, b1, None, pat), k, k2)
                if c:
                    out.append(c)
    return out


INNER = {                        # depth-2 laws: one operation applied to the result of another (on X = x)
    "succ(X)": lambda X, N: X + 1, "pred(X)": lambda X, N: np.maximum(X - 1, 0), "add(X,N)": lambda X, N: X + N,
    "add(X,X)": lambda X, N: 2 * X, "monus(N,X)": lambda X, N: np.maximum(N - X, 0), "mul(X,X)": lambda X, N: X * X,
    "mul(X,N)": lambda X, N: X * N,
}


def laws2(k, k2=K2):
    """Every law operation applied to the result of a first one and one of the inputs, on the data's own pairs: e.g.
    mod(succ(X), N) = (x + 1) mod N, the step that adds one and wraps around at N. (The target itself, mod(mul(X,X),
    N), is among them; the learner drops it as the target.)"""
    out = []
    for iname, inner in INNER.items():
        for name, (fn, words) in OPS2.items():
            for side in ("iN", "Ni", "iX", "Xi"):
                def gen(size, inner=inner, fn=fn, side=side):
                    xs, Ns = (v.astype(np.int64) for v in target.pairs(size))
                    I = inner(xs, Ns)
                    a, b = {"iN": (I, Ns), "Ni": (Ns, I), "iX": (I, xs), "Xi": (xs, I)}[side]
                    keep = [i for i in range(len(xs)) if fn(int(a[i]), int(b[i])) is not None]
                    return xs[keep], Ns[keep], [fn(int(a[i]), int(b[i])) for i in keep]
                arg = {"iN": (iname, "N"), "Ni": ("N", iname), "iX": (iname, "X"), "Xi": ("X", iname)}[side]
                c = make(f"law2/{name}({arg[0]},{arg[1]})", "law2", words.format(a=arg[0], b=arg[1]) + ", X = x",
                         gen, k, k2)
                if c:
                    out.append(c)
        for name, (fn, words) in OPS1.items():
            def gen1(size, inner=inner, fn=fn):
                xs, Ns = (v.astype(np.int64) for v in target.pairs(size))
                return xs, Ns, [fn(int(v)) for v in inner(xs, Ns)]
            c = make(f"law2/{name}({iname})", "law2", words.format(a=iname) + ", X = x", gen1, k, k2)
            if c:
                out.append(c)
    return out


# ---------------------------------------------------------------------------------------------------------------
# (b) transforms of the target (or of any problem given as a generator)

def target_gen(size):
    xs, Ns = target.pairs(size)
    return xs, Ns, target.answers(xs, Ns)


TERMS = {
    "1": lambda X, N: np.ones_like(X), "2": lambda X, N: np.full_like(X, 2), "3": lambda X, N: np.full_like(X, 3),
    "X": lambda X, N: X, "N": lambda X, N: N, "X+X": lambda X, N: 2 * X, "N+N": lambda X, N: 2 * N,
    "X+N": lambda X, N: X + N, "X·X": lambda X, N: X * X, "X·N": lambda X, N: X * N, "N·N": lambda X, N: N * N,
}


def restrictions(gen, k, k2=K2, prefix="restrict", parent="target", min_n=6, max_share=0.8):
    """The problem restricted to the instances where t1 < t2, for every ordered pair of terms; one per distinct subset
    (at size k), proper and not too small."""
    out, seen = [], set()
    xs, Ns, want = (np.asarray(v, np.int64) for v in gen(k))
    for a in TERMS:
        for b in TERMS:
            if a == b or (a in "123" and b in "123"):
                continue
            mask = TERMS[a](xs, Ns) < TERMS[b](xs, Ns)
            n = int(mask.sum())
            if n < min_n or n > max_share * len(xs):
                continue
            sig = tuple(np.nonzero(mask)[0])
            if sig in seen:
                continue
            seen.add(sig)

            def g(size, a=a, b=b):
                X, N, w = (np.asarray(v, np.int64) for v in gen(size))
                m = TERMS[a](X, N) < TERMS[b](X, N)
                return X[m], N[m], w[m]

            c = make(f"{prefix}/{parent}/{a}<{b}", "restrict", f"{parent} where {a} < {b}", g, k, k2, parent=parent)
            if c:
                out.append(c)
    return out


PROJECTIONS = {
    "lowbit": (lambda y, X, N: y % 2, "y mod 2 (its lowest binary digit)"),
    "high": (lambda y, X, N: y // 2, "y div 2 (the rest)"),
    "zero": (lambda y, X, N: (y == 0).astype(np.int64), "[y = 0]"),
    "ltx": (lambda y, X, N: (y < X).astype(np.int64), "[y < x]"),
    "eqx": (lambda y, X, N: (y == X).astype(np.int64), "[y = x]"),
}


def projections(gen, k, k2=K2, parent="target"):
    out = []
    for name, (fn, words) in PROJECTIONS.items():
        def g(size, fn=fn):
            X, N, w = (np.asarray(v, np.int64) for v in gen(size))
            return X, N, fn(w, X, N)
        c = make(f"project/{parent}/{name}", "project", f"{parent}: {words}", g, k, k2, parent=parent)
        if c:
            out.append(c)
    return out


def _values(prog, lib, X, N, size):
    """Every register's value after each top-level item of prog, on fresh machines (X, N): {(item index, reg): v}."""
    m = machine_for(size)
    out = {}
    b = brm.fresh(np.asarray(X), np.asarray(N))
    for i, it in enumerate(prog):
        brm.run_items(m, b, (it,), lib)
        if b.dead[0]:
            return None
        for r in ("A", "T", "U", "X", "N"):
            out[(i, r)] = b.regs[0, :, brm.RI[r]].astype(np.int64).copy()
    return out


def from_near_misses(near, lib, k, k2=K2, gen=target_gen, parent="target"):
    """near: [(fit, cost, program string)] of the target. For each near miss: every value one of its registers holds
    after one of its top-level items that is neither an input nor constant becomes a problem ("value"), and the rest of
    the way to the answer, y from (X = that value, N), becomes one ("remainder") when it is a function; each remainder
    also comes with its restrictions."""
    out = []
    pl = {t: dsl.parse(v) for t, v in lib.items()}
    for fit, cost, ptxt in near:
        prog = dsl.parse(ptxt) if ptxt else ()
        if not prog:
            continue
        X, N, _ = gen(k)
        vals = _values(prog, pl, X, N, k)
        if vals is None:
            continue
        for (i, r), v in vals.items():
            if (v == X).all() or (v == N).all() or len(set(v.tolist())) < 2:
                continue
            pre = dsl.show(prog[: i + 1])
            tag = f"{pre} -> {r}"

            def vg(size, i=i, r=r):
                X, N, _ = gen(size)
                vv = _values(prog, pl, X, N, size)
                return None if vv is None else (X, N, vv[(i, r)])

            c = make(f"value/{tag}", "value", f"the value {r} holds after `{pre}`", vg, k, k2, parent=parent)
            if c:
                out.append(c)

            def rg(size, i=i, r=r):
                X, N, w = gen(size)
                vv = _values(prog, pl, X, N, size)
                return None if vv is None else (vv[(i, r)], N, w)

            rem = make(f"remainder/{tag}", "remainder", f"{parent}'s answer from X = the value {r} holds after `{pre}`",
                       rg, k, k2, parent=parent)
            if rem:
                out.append(rem)
                out += restrictions(rg, k, k2, prefix="restrict", parent=f"remainder[{tag}]")
    return out


# ---------------------------------------------------------------------------------------------------------------
# (c) dreams, and the references

def dreams(lib, kinds, k, rng, known, n_want=16, k2=K2):
    from library.inventing.loop import make_dreams
    pl = {t: dsl.parse(v) for t, v in lib.items()}
    out = []
    for prog, got in make_dreams(pl, kinds, k, rng, known, n_want=n_want):
        def g(size, prog=prog):
            X, N = target.pairs(size)
            v, dead = brm.evaluate(prog, pl, X, N, machine_for(size))
            return None if dead else (X, N, v)
        c = make(f"dream/{dsl.show(prog)}", "dream", f"what `{dsl.show(prog)}` computes", g, k, k2)
        if c:
            out.append(c)
    return out


def the_target(k, k2=K2):
    return make("target", "target", "x² mod N", target_gen, k, k2)


def curated(k, k2=K2):
    out = []
    for t in target.CURATED:
        if t == "sqmod":
            continue
        c = make(f"curated/{t}", "curated", t, lambda size, t=t: target.curated(t, size), k, k2)
        if c:
            out.append(c)
    return out


def dedupe(cands, known=()):
    """One candidate per distinct problem (inputs and answers at both sizes); later duplicates are noted in `also`."""
    by, out = {c.key(): c for c in known}, []
    for c in cands:
        k = c.key()
        if k in by:
            by[k].also.append(c.name)
            continue
        by[k] = c
        out.append(c)
    return out
