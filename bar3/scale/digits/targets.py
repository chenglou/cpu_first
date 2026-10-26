"""The digits study's targets and data (DIGITS.md §2): three formulas, their rows with x of several digits, and the
check at big x.

- The targets, added to bar3/breadth/targets.py's table on import (data, not a rule): x mod N, x² mod N and 2^x mod N.
  Their `pairs` stay the original x < N <= k, so the law book, the depth-2 book and the dreams are posed on the data
  they always were, and keep their "subtract N if you can".
- The target's rows (`rows`): every (x, N) with N <= k and x < 10·k: x of two decimal digits at k = 7 and k = 10,
  up to 7 binary ones, and every residue among the numbers that get a digit appended (x < k). The learner searches,
  checks and scores the target on these.
- The check at big x (`bigx`): N <= 10 and 20 values of x with 6–8 decimal digits for each N (seeded), on a machine
  that holds such x (cap 10⁹) and stops a program after 100,000 steps. A loop that runs once per unit of x needs at
  least two steps per unit, so at least 2·10⁵ at x >= 10⁵: it cannot pass. A loop over x's digits takes 1,400 to
  42,000 steps on tiny.py's hand-written programs. A program passes when it is right on all 200.
- The target's own machines keep the caps of library/inventing/jobs.machine_for, with more steps: 20,000 at the
  search size (3,450 there before) and 100,000 at the check size (6,000 before), since a digit loop multiplies a
  step's cost by the number of digits. Every other problem's machine is as before.
"""

from __future__ import annotations

import numpy as np

from library.inventing import brm
from bar3.breadth import targets as TG

FORMULAS = {
    "dxmod": ("x mod N", lambda x, n: x % n),
    "dxsq": ("x² mod N", lambda x, n: (x * x) % n),
    "dpow2": ("2^x mod N", lambda x, n: pow(2, x, n)),
}
NAMES = tuple(FORMULAS)
SPAN = 10                       # the target's rows: x < SPAN·k
BIG_N, BIG_PER_N, BIG_LO, BIG_HI = 10, 20, 10 ** 5, 10 ** 8
BIG_STEPS = 100_000
BIG = brm.Machine(cap=10 ** 9, limit=BIG_STEPS)
LIMITS = {7: 20_000, 10: BIG_STEPS}            # the target's machines: steps at the search and the check size


def machine(size):
    """The target's machine at a size: machine_for's cap, LIMITS' steps."""
    from library.inventing.jobs import machine_for
    return brm.Machine(cap=machine_for(size).cap, limit=LIMITS[size])


def answers(name, xs, Ns):
    f = FORMULAS[name][1]
    return np.array([f(int(x), int(n)) for x, n in zip(xs, Ns)], np.int64)


for _n, (_f, _fn) in FORMULAS.items():
    TG.TARGETS[_n] = TG.Target(_n, _f, "digits", lambda xs, Ns, n=_n: answers(n, xs, Ns))


def rows(name, size):
    """(xs, Ns, answers): every x < SPAN·size for every N <= size, ordered by N then x."""
    xs = np.array([x for n in range(1, size + 1) for x in range(SPAN * size)], np.int64)
    Ns = np.array([n for n in range(1, size + 1) for _ in range(SPAN * size)], np.int64)
    return xs, Ns, answers(name, xs, Ns)


def gen_for(name):
    """The target's rows as a problem generator: size -> (xs, Ns, answers)."""
    return lambda size: rows(name, size)


def small_gen(name):
    """The target on the original pairs x < N <= size (the recurrence source reads these, as before)."""
    def gen(size):
        xs, Ns = TG.ORIGINAL_PAIRS(size)
        return xs.astype(np.int64), Ns.astype(np.int64), answers(name, xs, Ns)
    return gen


def bigx(name, seed=0):
    """The check at big x: (xs, Ns, answers), 20 values of x in [10⁵, 10⁸) for each N <= 10."""
    rng = np.random.default_rng(seed)
    xs = np.concatenate([rng.integers(BIG_LO, BIG_HI, BIG_PER_N) for _ in range(BIG_N)]).astype(np.int64)
    Ns = np.repeat(np.arange(1, BIG_N + 1), BIG_PER_N).astype(np.int64)
    return xs, Ns, answers(name, xs, Ns)


_BIG = {}


def bigx_share(prog, lib, name, m=BIG):
    """The share of the big-x instances prog gets right; 0 when it runs out of steps on any of them (on the compiled
    kernel, machine.evaluate: a program that counts up to x would take the interpreter minutes to stop)."""
    from bar3.scale.digits import machine as M
    if name not in _BIG:
        _BIG[name] = bigx(name)
    xs, Ns, want = _BIG[name]
    got, dead = M.evaluate(prog, lib, xs, Ns, m)
    return 0.0 if dead else float(np.mean(got == want))


def steps_on(prog, lib, x, n, cap=10 ** 9, limit=10 ** 9):
    """(steps, A) for one machine (x, N) on the compiled kernel (bar3/scale/digits/machine.py), or (None, None)
    when it passes the limit: for tiny.py's budget table, where counting runs to 10⁸ steps."""
    from bar3.scale.digits import machine as M
    code, depth = M.compile_items(tuple(prog), lib)
    regs = np.zeros(5, np.int32)
    regs[brm.X], regs[brm.N] = x, n
    st = [np.empty(depth + 1, np.int64) for _ in range(4)]
    s = M._run_block(code, 0, code.shape[0], regs, 0, cap, limit, *st)
    return (None, None) if s < 0 else (int(s), int(regs[brm.A]))
