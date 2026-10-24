"""New practice sources for the bar-3 learner (SCALE.md §2), and the problems they need: problems whose inputs sit in
any registers and whose answer may be several registers at once.

The learner's problems (bar3/problems.py) all have one shape: x in X, N in N, every other register 0, and one answer
in A. Two of the formulas it missed need other shapes:
- Fibonacci's repeated step moves two numbers at once, (a, b) -> (b, a + b mod N): a problem with inputs in two
  registers and an answer in two.
- 10·x² needs the number 10, which nothing poses: a counting machine has to build it, and no source asks for it.

Three sources, each generic (it reads the data's table of answers, or what a routine computes, not the formula):
- `recurrences`: the answer at x + 1 (or x + 2) as a function of the answers before it, when the data says it is
  one. Order 1: y(x + 1) = f(y(x), N). Order 2: y(x + 2) = f(y(x), y(x + 1), N), posed both as the one number it
  computes and as the step that moves the pair (y(x), y(x + 1)) to (y(x + 1), y(x + 2)), in every placement of the
  numbers in the registers. Fibonacci's order-2 step is "add mod N"; x² mod N's is "2b − a + 2 mod N".
- `constants`: the numbers the answers hold at a fixed x. With x fixed, the answer depends on N alone, and when it is
  c mod N for one small c on every N the data shows (the smallest c that fits all the remainders), c is posed as a
  problem with no input: "make c" in A, T or U. 10·x² at x = 1 holds 10; x² + 3 at x = 0 holds 3. Such a problem is
  searched on one instance, x = 0 and N = 1 (it has no inputs to tell apart, and one instance keeps the search small:
  "make 10" takes 7 instructions, found among 7,418 states instead of 2.5M on two instances), and checked on every
  instance of both sizes, so a program that leans on that instance's x or N is rejected.

- `routine_remainders` ("targets made short in terms of learned routines"): for every routine the learner has, the
  rest of the way from what it leaves in A, y from X = that value: bar3/problems.from_near_misses's "remainder",
  taken from every routine instead of the target probe's near misses. With x² mod N carried, 10·x² + 1 is "10·x²
  mod N, then the rest": the remainder from that routine is "X + 1 mod N".

A problem here is an `RCand`: bar3/problems.Cand plus the registers set before the program runs (`init`, beyond X
and N), the registers that hold the answer (`outs`), and optionally a separate set of instances to check (`every`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from library.inventing import brm
from library.inventing.jobs import machine_for
from bar3 import problems as P

FREE = ("A", "T", "U")             # the registers a fresh machine holds at 0


@dataclass
class RCand(P.Cand):
    init: dict = field(default_factory=dict)      # register -> values per instance, size k (beyond X and N)
    cinit: dict = field(default_factory=dict)     # the same at the check size k2
    outs: tuple = ("A",)                          # registers that hold the answer; want[i] is a tuple if several
    every: tuple = ()                             # (xs, Ns, init, want) to accept on, if not the search instances

    def key(self):
        tup = lambda w: tuple(tuple(v) if isinstance(v, (list, tuple)) else v for v in w)   # noqa: E731
        reg = lambda d: tuple((r, tuple(v)) for r, v in sorted(d.items()))                  # noqa: E731
        ev = () if not self.every else (tuple(self.every[0]), tuple(self.every[1]), reg(self.every[2]),
                                        tup(self.every[3]))
        return (tuple(self.xs), tuple(self.Ns), tup(self.want), tuple(self.cx), tuple(self.cN), tup(self.cw),
                reg(self.init), reg(self.cinit), self.outs, ev)

    def to_dict(self):
        return dict(super().to_dict(), outs=list(self.outs), init=sorted(self.init))


def _ok(xs, Ns, init, want, k, outs):
    """bar3/problems._ok for several registers: at least 4 instances, every value below the machine's cap, answers
    that vary, and a function of the inputs."""
    cap = machine_for(k).cap
    if len(xs) < 4:
        return False
    flat_w = [v for w in want for v in (w if len(outs) > 1 else (w,))]
    vals = list(xs) + list(Ns) + flat_w + [v for vs in init.values() for v in vs]
    if max(vals) >= cap or min(vals) < 0:
        return False
    if len(set(map(tuple, want)) if len(outs) > 1 else set(want)) < 2:
        return False
    seen = {}
    for i in range(len(xs)):
        key = (xs[i], Ns[i]) + tuple(init[r][i] for r in sorted(init))
        w = tuple(want[i]) if len(outs) > 1 else want[i]
        if seen.setdefault(key, w) != w:
            return False
    return True


def make(name, source, formula, gen, k, k2=P.K2, outs=("A",), parent="target", every=None):
    """gen(size) -> (xs, Ns, init {reg: values}, want) or None; want[i] a tuple when outs has several registers.
    every(size) -> the same, for problems searched on fewer instances than they are checked on."""
    a, b = gen(k), gen(k2)
    if a is None or b is None or not _ok(*a, k, outs) or not _ok(*b, k2, outs):
        return None
    clean = lambda v: [list(map(int, w)) if len(outs) > 1 else int(w) for w in v]   # noqa: E731
    ints = lambda v: list(map(int, v))                                                # noqa: E731
    ev = ()
    if every is not None:
        e = every(k)
        ev = (ints(e[0]), ints(e[1]), {r: ints(v) for r, v in e[2].items()}, clean(e[3]))
    return RCand(name, source, formula, ints(a[0]), ints(a[1]), clean(a[3]), ints(b[0]), ints(b[1]), clean(b[3]),
                 parent=parent, init={r: ints(v) for r, v in a[2].items()},
                 cinit={r: ints(v) for r, v in b[2].items()}, outs=tuple(outs), every=ev)


# ---------------------------------------------------------------------------------------------------------------
# the data's table of answers

def table(gen, size):
    """{N: [y(0, N), y(1, N), ...]} for the N whose every x < N is in the data, from gen(size) -> (xs, Ns, y)."""
    xs, Ns, y = (np.asarray(v, np.int64) for v in gen(size))
    out = {}
    for n in sorted(set(Ns.tolist())):
        m = Ns == n
        row = dict(zip(xs[m].tolist(), y[m].tolist()))
        if sorted(row) == list(range(n)):
            out[n] = [row[x] for x in range(n)]
    return out


def _is_function(rows):
    seen = {}
    return all(seen.setdefault(key, v) == v for key, v in rows)


def recurrences(gen, k, k2=P.K2, parent="target"):
    """The answer as a function of the answers before it (order 1, else order 2), posed in every placement."""
    def rows1(size):
        return [((a, n), b) for n, ys in table(gen, size).items() for a, b in zip(ys, ys[1:])]

    def rows2(size):
        return [((a, b, n), c) for n, ys in table(gen, size).items() for a, b, c in zip(ys, ys[1:], ys[2:])]

    out = []
    if _is_function(rows1(k)) and _is_function(rows1(k2)) and len(rows1(k)) >= 4:
        for reg in ("A", "X"):
            def g1(size, reg=reg):
                rs = rows1(size)
                a = [r[0][0] for r in rs]
                xs = a if reg == "X" else [0] * len(rs)
                init = {} if reg == "X" else {"A": a}
                return xs, [r[0][1] for r in rs], init, [r[1] for r in rs]
            c = make(f"recur1/{reg}", "recur", f"y(x + 1) from y(x) in {reg}", g1, k, k2, parent=parent)
            if c:
                out.append(c)
        return out
    if not (_is_function(rows2(k)) and _is_function(rows2(k2)) and len(rows2(k)) >= 4):
        return out

    def placed(size, ra, rb):
        """(xs, Ns, init) with y(x) in register ra and y(x + 1) in rb, and the rows."""
        rs = rows2(size)
        a, b = [r[0][0] for r in rs], [r[0][1] for r in rs]
        vals = {ra: a, rb: b}
        xs = vals.get("X", [0] * len(rs))
        return xs, [r[0][2] for r in rs], {r: v for r, v in vals.items() if r != "X"}, rs

    for ra in ("X", "A", "U"):                    # the number it computes, y(x + 2), in A
        for rb in ("X", "A", "U"):
            if ra == rb:
                continue
            def g2(size, ra=ra, rb=rb):
                xs, Ns, init, rs = placed(size, ra, rb)
                return xs, Ns, init, [r[1] for r in rs]
            c = make(f"recur2/{ra}{rb}->A", "recur", f"y(x + 2) from y(x) in {ra}, y(x + 1) in {rb}", g2, k, k2,
                     parent=parent)
            if c:
                out.append(c)
    for ra in ("A", "U", "T", "X"):               # the step that moves the pair: (a, b) -> (b, c) in place
        for rb in ("A", "U", "T", "X"):
            if ra == rb:
                continue
            def g3(size, ra=ra, rb=rb):
                xs, Ns, init, rs = placed(size, ra, rb)
                return xs, Ns, init, [(r[0][1], r[1]) for r in rs]
            c = make(f"recur2/step{ra}{rb}", "recur",
                     f"the pair (y(x), y(x + 1)) in ({ra}, {rb}) moved to (y(x + 1), y(x + 2))", g3, k, k2,
                     outs=(ra, rb), parent=parent)
            if c:
                out.append(c)
    return out


def constants(gen, k, k2=P.K2, at=(0, 1), regs=FREE, parent="target"):
    """For x fixed at each of `at`: when the answer is c mod N on every N the data shows (sizes k and k2) for one c
    with 2 <= c < the machine's cap (the smallest c that fits every remainder), pose "make c" in each of regs."""
    cap = machine_for(k).cap
    found = []
    for x0 in at:
        slices = []
        for size in (k, k2):
            xs, Ns, y = (np.asarray(v, np.int64) for v in gen(size))
            m = xs == x0
            slices.append((Ns[m], y[m]))
        if any(len(n) < 3 for n, _ in slices):
            continue
        Ns = np.concatenate([s[0] for s in slices])
        ys = np.concatenate([s[1] for s in slices])
        period = math.lcm(*map(int, set(Ns.tolist())))
        c = next((c for c in range(2, min(cap, period)) if ((c % Ns) == ys).all()), None)
        if c is not None and c not in found:
            found.append(c)
    out = []
    xk, Nk = (list(map(int, v)) for v in P.target.pairs(k))
    x2, N2 = (list(map(int, v)) for v in P.target.pairs(k2))
    for c in found:
        for reg in regs:          # searched on one instance (there are no inputs to tell apart), checked on all
            out.append(RCand(f"const/{c}->{reg}", "const", f"make {c} in {reg}", [0], [1], [c], x2, N2,
                             [c] * len(x2), parent=parent, outs=(reg,), every=(xk, Nk, {}, [c] * len(xk))))
    return out


def routine_remainders(lib, k, k2=P.K2, gen=None):
    """For each routine T.* of lib: the target's answer from X = the value the routine leaves in A (run alone on the
    target's inputs), when that is a function of (X, N) and the value is not an input or constant."""
    from library import dsl
    gen = gen or P.target_gen
    pl = {t: dsl.parse(v) for t, v in lib.items()}
    out = []
    for t in sorted(lib):
        if not t.startswith("T."):
            continue

        def rg(size, t=t):
            X, N, w = gen(size)
            v, dead = brm.evaluate((t,), pl, X, N, machine_for(size))
            if dead or (v == X).all() or len(set(v.tolist())) < 2:
                return None
            return v, N, w

        c = P.make(f"rrem/{t}", "rrem", f"the answer from X = what {t} leaves in A", rg, k, k2, parent="target")
        if c:
            out.append(c)
    return out
