"""The "one more digit" source (DIGITS.md §2): is the answer for x·B + d a function of the answer for x, the digit d
and N? If so, that step is posed as a practice problem, in every placement, as the two-number step is posed.

It reads the data's rows only (gen(size) -> xs, Ns, answers): for each N, every x whose B children x·B + d,
d < B, are all among the rows (in unary, B = 1, the one child x + 1 with d = 1), gives a row
(answer(x), d, N) -> answer(x·B + d). The answer is its own state when those rows are a function at both sizes.

- The state is the target's answer, or a routine's: what each routine of the library leaves in A on the target's
  inputs (as sources.routine_remainders reads it).
- A state may be a pair: the target's answer with a routine's, when the target's alone is not a state. x² mod N is not
  its own state; with x mod N beside it, it is.
- Placements: the state in A, U or T, updated in place; the digit in X, U or T. A pair is placed in two of A, U and
  T, with the digit in X, where the digit block (machine.py) puts it.

`truth_table` is the pose rule's test on its own (LARGE_NUMBERS.md §5.5, test 1).
"""

from __future__ import annotations

import itertools

import numpy as np

from library import dsl
from library.inventing import brm
from library.inventing.jobs import machine_for
from bar3 import problems as P
from bar3.scale import sources as S

STATE = ("A", "U", "T")
DIGIT = ("X", "U", "T")


def children(x, B):
    return [(x + 1, 1)] if B == 1 else [(x * B + d, d) for d in range(B)]


def step_rows(xs, Ns, vals, B, unique=True):
    """[(state(x) as a tuple, d, N), state(x·B + d) as a tuple] for every x whose children are all in the rows, each
    distinct row once (unique=False: one per x, to test). vals: one value array per component of the state."""
    by = {}
    for i, (x, n) in enumerate(zip(xs.tolist(), Ns.tolist())):
        by[(n, x)] = tuple(int(v[i]) for v in vals)
    out, seen = [], set()
    for (n, x), s in sorted(by.items()):
        kids = children(x, B)
        if all((n, y) in by for y, _ in kids):
            for y, d in kids:
                row = ((s, d, n), by[(n, y)])
                if not unique or row not in seen:
                    seen.add(row)
                    out.append(row)
    return out


def is_state(rows):
    seen = {}
    return len(rows) >= 4 and all(seen.setdefault(key, v) == v for key, v in rows)


def _pose(name, rows_at, k, k2, regs, digit, parent):
    """The step on rows_at(size) as a problem: state components in regs (in place), the digit in `digit`."""
    def gen(size):
        rs = rows_at(size)
        if not rs:
            return None
        ds = [r[0][1] for r in rs]
        init = {reg: [r[0][0][j] for r in rs] for j, reg in enumerate(regs)}
        if digit != "X":
            init[digit] = ds
        xs = ds if digit == "X" else [0] * len(rs)
        want = [r[1][0] for r in rs] if len(regs) == 1 else [list(r[1]) for r in rs]
        return xs, [r[0][2] for r in rs], init, want
    return S.make(name, "digit", f"{parent}: the state after one more digit", gen, k, k2, outs=regs, parent=parent)


def routine_values(lib, gen, size):
    """{routine token: the value it leaves in A, run alone on the target's inputs at this size}."""
    pl = {t: dsl.parse(v) for t, v in lib.items()}
    xs, Ns, _ = gen(size)
    out = {}
    for t in sorted(lib):
        if t.startswith("T."):
            v, dead = brm.evaluate((t,), pl, xs, Ns, machine_for(size))
            if not dead and len(set(v.tolist())) > 1 and not (v == xs).all():
                out[t] = v.astype(np.int64)
    return out


def digit_steps(gen, B, k, k2=P.K2, lib=None, parent="target"):
    """The source's problems for the target (gen) and the routines of lib, in base B."""
    lib = lib or {}
    vk, v2 = routine_values(lib, gen, k), routine_values(lib, gen, k2)
    states = {"target": lambda size: [gen(size)[2]]}
    for t in sorted(set(vk) & set(v2)):
        states[t] = lambda size, t=t: [(vk if size == k else v2)[t]]
    out = []

    def rows_for(comp):
        def rows_at(size):
            xs, Ns, _ = gen(size)
            return step_rows(xs, Ns, [v for c in comp for v in states[c](size)], B)
        return rows_at

    alone = {c: is_state(rows_for((c,))(k)) and is_state(rows_for((c,))(k2)) for c in states}
    for c, ok in alone.items():
        if not ok:
            continue
        for s, d in itertools.product(STATE, DIGIT):
            if s != d:
                p = _pose(f"digit/{c}/{s}.{d}", rows_for((c,)), k, k2, (s,), d, parent if c == "target" else c)
                if p:
                    out.append(p)
    if not alone["target"]:
        for c in states:
            if c == "target":
                continue
            comp = ("target", c)
            if not (is_state(rows_for(comp)(k)) and is_state(rows_for(comp)(k2))):
                continue
            for s1, s2 in itertools.permutations(STATE, 2):
                p = _pose(f"digit/target+{c}/{s1}{s2}.X", rows_for(comp), k, k2, (s1, s2), "X", parent)
                if p:
                    out.append(p)
    return out


# ---------------------------------------------------------------------------------------------------------------
# the law book on the target's rows (every arm: part of the data, not of the new source)

def row_laws(gen, k, k2=P.K2):
    """bar3/problems.laws' depth-1 book with X = x on the target's own rows (it stays on the pairs x < N as well,
    where it poses "subtract N if you can"): on rows with x of several digits, mod(X, N) is x mod N, the state the
    polynomial targets need beside their answer. Named law/<op>/rows."""
    out = []

    def gen_for(fn, b1, b2):
        def g(size):
            xs, Ns, _ = gen(size)
            reg = {"X": xs, "N": Ns}
            keep, want = [], []
            for i in range(len(xs)):
                v = fn(int(reg[b1][i]), int(reg[b2][i])) if b2 else fn(int(reg[b1][i]))
                if v is not None:
                    keep.append(i)
                    want.append(v)
            return xs[keep], Ns[keep], want
        return g

    for name, (fn, words) in P.OPS2.items():
        for b1, b2 in (("X", "N"), ("N", "X"), ("X", "X"), ("N", "N")):
            c = P.make(f"law/{name}({b1},{b2})/rows", "law", words.format(a=b1, b=b2) + ", X = x on the target's rows",
                       gen_for(fn, b1, b2), k, k2)
            if c:
                out.append(c)
    for name, (fn, words) in P.OPS1.items():
        for b1 in ("X", "N"):
            c = P.make(f"law/{name}({b1})/rows", "law", words.format(a=b1) + ", X = x on the target's rows",
                       gen_for(fn, b1, None), k, k2)
            if c:
                out.append(c)
    return out


# ---------------------------------------------------------------------------------------------------------------
# the pose rule's truth table (LARGE_NUMBERS.md §5.5, test 1)

def _fib(x, n):
    a, b = 0, 1 % n
    for _ in range(x):
        a, b = b, (a + b) % n
    return a % n


THEORY = {                                   # LARGE_NUMBERS.md §5.2's twelve targets, and its verdicts
    "x mod N": (lambda x, n: x % n, True),
    "2x + 1": (lambda x, n: (2 * x + 1) % n, True),
    "2^x": (lambda x, n: pow(2, x, n), True),
    "x²": (lambda x, n: x * x % n, False),
    "x² + 1": (lambda x, n: (x * x + 1) % n, False),
    "x² − 1": (lambda x, n: (x * x - 1) % n, False),
    "x² + 3": (lambda x, n: (x * x + 3) % n, False),
    "10·x²": (lambda x, n: 10 * x * x % n, False),
    "10·x² + 1": (lambda x, n: (10 * x * x + 1) % n, False),
    "x³": (lambda x, n: pow(x, 3, n), False),
    "Montgomery's x²·2⁻¹ (N odd)": (lambda x, n: x * x * ((n + 1) // 2) % n, False),
    "Fibonacci": (_fib, False),
}


def truth_table(span_k=10, k=7, k2=10, theory_x=2000):
    """For each target and base: the verdict on the learner's own rows (x < 10·k at both sizes, as the source sees
    them) and on the theory note's (every x < 2,000, N <= 7)."""
    out = []
    for name, (f, want) in THEORY.items():
        odd = name.startswith("Montgomery")
        row = dict(target=name, theory=want)
        for B in (1, 2, 10):
            verdicts = []
            for size, xmax in ((k, span_k * k), (k2, span_k * k2), (7, theory_x)):
                xs, Ns, ys = [], [], []
                for n in range(1, size + 1):
                    if odd and n % 2 == 0:
                        continue
                    for x in range(xmax):
                        xs.append(x)
                        Ns.append(n)
                        ys.append(f(x, n))
                verdicts.append(is_state(step_rows(np.array(xs), np.array(Ns), [np.array(ys)], B)))
            row[B] = dict(learner=verdicts[0] and verdicts[1], theory_rows=verdicts[2])
        out.append(row)
    return out
