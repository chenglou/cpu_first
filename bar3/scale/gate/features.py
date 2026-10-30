"""The gate's inputs (GATE.md §2): property signatures in the sense of Odena and Sutton (ICLR 2020), of a machine state
against the problem's rows, and of the rows themselves. Every test is a comparison of numbers the rows or the state
already hold: equality, order, the lowest bit (parity). No number is made by arithmetic, and nothing names a target.

A state is five registers (X, N, A, T, U) on every instance. For each register v and each test t below, the signature
keeps t's AllTrue and its AnyTrue over the instances (AllFalse = not AnyTrue, Mixed = AnyTrue and not AllTrue):
    v = a (the answer), v = x, v = N, v = 0, v < N, v <= a, v >= a, v and a have the same parity,
    v is as it was at the start, v is constant in x at each N, v depends on x only;
plus the share of instances right (A, or the problem's first answer register, = a) and the best share over the
registers; and three tests of what the state still determines (equality only: probes grouped by equal values):
    the answer is a function of (v, N) (no two probes with the same v and N want different answers), of v alone,
    and of the whole state (if not, the state is dead: gengine.dead_states).
The problem's own signature: a < N, a = x, a = 0, a <= x, a >= x, a and x have the same parity, a constant in x at each
N, a depends on x only (AllTrue, AnyTrue each), and three shape flags (registers set before the program runs, several
answer registers, the answer outside A).
"""

from __future__ import annotations

import numpy as np

from library.inventing import brm

CHUNK = 1 << 15
STATE_TESTS = ("eq_a", "eq_x", "eq_N", "zero", "lt_N", "le_a", "ge_a", "par_a", "kept", "const_x", "x_only")
PROB_TESTS = ("lt_N", "eq_x", "zero", "le_x", "ge_x", "par_x", "const_x", "x_only")
N_STATE = len(STATE_TESTS) * 2 * 5 + 2 + 5 + 5 + 1
N_PROB = len(PROB_TESTS) * 2 + 3
NAMES = ([f"{t}_{q}_{r}" for t in STATE_TESTS for q in ("all", "any") for r in brm.REGS] + ["fit", "fit_any"]
         + [f"det_N_{r}" for r in brm.REGS] + [f"det_{r}" for r in brm.REGS] + ["live"]
         + [f"p_{t}_{q}" for t in PROB_TESTS for q in ("all", "any")] + ["p_init", "p_outs", "p_notA"])


def _first_of(groups):
    """For each instance, the index of the first instance in its group."""
    first = {}
    return np.array([first.setdefault(g, i) for i, g in enumerate(groups.tolist())], np.int64)


class Context:
    """A problem's rows, as the gate reads them: x, N, the answer a (of the first answer register), that register, the
    start state, and the instances that share N (or x)."""

    def __init__(self, prob):
        root = prob.root.regs[0].astype(np.int32)
        self.root = root
        self.x, self.N = root[:, brm.X], root[:, brm.N]
        want = np.asarray(prob.want)
        self.a = (want[:, 0] if want.ndim == 2 else want).astype(np.int32)
        self.outs = list(getattr(prob, "outs", [brm.A]))
        self.out = self.outs[0]
        self.fN = _first_of(self.N)
        self.fx = _first_of(self.x)
        want2 = want.reshape(len(want), -1).astype(np.int64)
        self.wcode = np.zeros(len(want2), np.int64)
        for k in range(want2.shape[1]):
            self.wcode = self.wcode * 65536 + want2[:, k]
        init = bool((root[:, [brm.A, brm.T, brm.U]] != 0).any())
        self.prob_sig = problem_signature(self, init)


def problem_signature(c, init):
    a, x, N = c.a, c.x, c.N
    tests = [a < N, a == x, a == 0, a <= x, a >= x, ((a ^ x) & 1) == 0, a == a[c.fN], a == a[c.fx]]
    out = []
    for t in tests:
        out += [t.all(), t.any()]
    out += [init, len(c.outs) > 1, c.out != brm.A]
    return np.array(out, np.float32)


def state_features(regs, c):
    """regs [S, NP, 5] -> [S, N_STATE] float32."""
    S = regs.shape[0]
    out = np.empty((S, N_STATE), np.float32)
    x, N, a = c.x[None, :, None], c.N[None, :, None], c.a[None, :, None]
    root = c.root[None]
    for lo in range(0, S, CHUNK):
        v = regs[lo:lo + CHUNK]
        n = v.shape[0]
        tests = (v == a, v == x, v == N, v == 0, v < N, v <= a, v >= a, ((v ^ a) & 1) == 0, v == root,
                 v == v[:, c.fN, :], v == v[:, c.fx, :])
        j = 0
        for t in tests:
            out[lo:lo + n, j:j + 5] = t.all(axis=1)
            out[lo:lo + n, j + 5:j + 10] = t.any(axis=1)
            j += 10
        eq = tests[0]
        out[lo:lo + n, j] = eq[:, :, c.out].mean(axis=1)
        out[lo:lo + n, j + 1] = eq.mean(axis=1).max(axis=1)
        j += 2
        vv = v.astype(np.int64)
        for r in range(5):
            out[lo:lo + n, j + r] = ~_collide(vv[:, :, r] * 4096 + c.N[None, :], c.wcode)
            out[lo:lo + n, j + 5 + r] = ~_collide(vv[:, :, r], c.wcode)
        code = np.zeros(vv.shape[:2], np.int64)
        for r in range(5):
            code = code * 4096 + vv[:, :, r]
        out[lo:lo + n, j + 10] = ~_collide(code, c.wcode)
    return out


def _collide(code, w):
    """[S] bool: two probes with the same code want different answers."""
    order = np.argsort(code, axis=1, kind="stable")
    cs = np.take_along_axis(code, order, 1)
    ws = w[order]
    return ((cs[:, 1:] == cs[:, :-1]) & (ws[:, 1:] != ws[:, :-1])).any(axis=1)


def features(regs, c):
    """The gate's input for each state: its signature and the problem's, [S, N_STATE + N_PROB]."""
    s = state_features(regs, c)
    return np.concatenate([s, np.broadcast_to(c.prob_sig, (len(s), N_PROB))], axis=1)
