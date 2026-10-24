"""The in-run version (LIBRARY.md §6): the corpus built from the training rows' own numbers, with no answer and no
hand-written arithmetic except the T = 1 labels.

Every probe is made from H1 training rows in base 10. At width W a row gives its modulus's top W digits N' and x's
digits at the same places x' (reading digit strings; x < N makes x' <= N'). Targets come from three places only:
- facts about the rows: x' < N' unless the two digit strings are equal; "x mod N = x"; "N mod N = 0";
- programs the run has already found, run on the rows' numbers ("generator" targets), or run on a candidate's
  output ("inverse" checks: a candidate adder passes if subtracting N back with the found subtraction gives x);
- the T = 1 labels of the full-size rows, for the final x^2 mod N.

Task (what it states, which earlier programs build its probes or targets):
    lt      c = [x' < N'] from the strings: (x', N') borrows iff they differ, (N', x') and (x', x') never  (none)
    modx    x' mod N' = x' (0 if equal), N' mod N' = 0                                                (none)
    add     (x' + N') mod N' = x' with modx's program, x' + N' >= N' with lt's, x' + 0 = x'            (lt, modx)
    modB    (x' + q N') mod N' = x' for q = 0..9, the inputs built with add                              (add)
    mrow    x' * d by the ladder: x' * 0 = 0, x' * (d + 1) = x' * d + x', with add                       (add)
    mac     x' * d + N' = add(mrow(x', d), N'): the row on top of a number                             (mrow, add)
    mul     x' * y' = y' additions of x' (y' another row's x'), with add                                 (add)
    sq      x' * x' with mul                                                                             (mul)
    step0   d mod N' with modB; step: (r B + d) mod N' with modB, r = x' < N', d a digit of the row      (modB)
    reduce  (x' + Q N') mod N' = x', Q another row's x', the inputs built by Q additions of N'            (add)
    sqmod   (x' x') mod N' with mul and reduce; verified on the full-size T = 1 rows against their labels (mul, reduce)

A program is accepted for a task when it passes its laws on fresh rows at widths 1-3 (and, for sqmod, the labels).
The searches, merges and levels are wake_sleep.py's.
"""

from __future__ import annotations

import json
import os

import numpy as np

from library import dsl
from library.dsl import to_digits

H1_DIR = os.environ.get("H1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                "one-layer-deeper", "data", "generated", "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
NEEDS = {"lt": (), "modx": (), "add": ("lt", "modx"), "mac": ("mrow", "add"), "modB": ("add",), "mrow": ("add",), "mul": ("add",), "sq": ("mul",),
         "step0": ("modB",), "step": ("modB",), "reduce": ("add",), "sqmod": ("mul", "reduce")}
INRUN_TASKS = tuple(NEEDS)
# the offline task each in-run task should turn out to be (only for the report; the run never sees these)
OFFLINE = {"lt": "lt", "modx": "csub", "add": "add", "modB": "modB", "mrow": "mrow", "mac": "mac", "mul": "mul",
           "sq": "sq",
           "step0": "step0", "step": "step", "reduce": "reduce", "sqmod": "sqmod"}


def job_data(n_rows=1500, n_verify=800, n_labels=300, seed=0):
    """The rows a run uses: training rows for probes, other training rows for acceptance, T = 1 rows for labels."""
    rows = load_rows(n_rows + n_verify, seed=seed)
    return dict(rows=[list(r) for r in rows[:n_rows]], verify_rows=[list(r) for r in rows[n_rows:]],
                label_rows=[list(r) for r in load_rows(n_labels, seed=seed + 1, T=1)])


def load_rows(n=3000, seed=0, split="train", T=None):
    rows = [json.loads(l) for l in open(os.path.join(H1_DIR, split + ".jsonl"))]
    if T is not None:
        rows = [r for r in rows if r["time_steps"] == T]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))[:n]
    return [(rows[i]["modulus"], rows[i]["x"], rows[i]["time_steps"], rows[i]["result"]) for i in idx]


def top_parts(N, x, W):
    """N's top W digits and x's digits at the same places (digit strings only)."""
    s = str(N)
    t = str(x).zfill(len(s))
    return int(s[:W]), int(t[:W])


def run_prog(st, prog, lib, ins, rng):
    """Run a found program on given inputs {reg: values}; returns (A values, c bits)."""
    n = len(next(iter(ins.values())))
    b = dsl.fresh(st, {r: to_digits(np.asarray(v, dtype=object), st.B, st.P) for r, v in ins.items()}, rng, n)
    dsl.run_items(st, b, prog, lib)
    A = dsl.from_digits(b.regs[0, dsl.A].T, st.B)
    return np.asarray(A, dtype=object), b.c[0].copy()


# how many top digits a task's numbers take at a setting: the facts hold for numbers as wide as the register; a task
# whose numbers grow (a sum, a product) takes fewer so its results still fit in P places
def width(task, st):
    if task in ("lt", "modx"):
        return st.P
    if task in ("add", "modB", "mrow", "step0", "step"):
        return st.P - 1
    return st.W


def build(task, st, n, rng, rows, helpers, lib):
    """Probes of one task at one setting: (inputs {reg: values}, targets {"A": values} / {"c": bits}, inverse check
    or None). helpers: task -> program."""
    W, B = width(task, st), st.B
    pick = [rows[i] for i in rng.integers(0, len(rows), size=4 * n)]
    tp = [top_parts(N, x, W) for N, x, _, _ in pick]
    Nt = np.array([a for a, _ in tp], dtype=object)[:n]
    xt = np.array([b for _, b in tp], dtype=object)[:n]
    other = np.array([b for _, b in tp], dtype=object)[n:2 * n]     # another row's x'
    eq = np.array([a == b for a, b in zip(Nt, xt)])
    zero = np.zeros(n, dtype=object)
    run = lambda name, ins: run_prog(st, helpers[name], lib, ins, rng)
    if task == "lt":
        kind = np.arange(n) % 4               # 0, 1: (x', N'); 2: (N', x'); 3: (x', x')
        X = np.where(kind < 2, xt, np.where(kind == 2, Nt, xt))
        Nr = np.where(kind < 2, Nt, np.where(kind == 2, xt, xt))
        c = np.where(kind < 2, (~eq).astype(np.uint8), 0).astype(np.uint8)
        return {"X": X, "N": Nr}, {"c": c}, None
    if task == "modx":
        kind = np.arange(n) % 3               # 0, 1: x' mod N'; 2: N' mod N'
        X = np.where(kind < 2, xt, Nt)
        A = np.where((kind < 2) & ~eq, xt, zero)
        return {"X": X, "N": Nt}, {"A": A}, None
    if task == "add":
        keep = [i for i, (a, b) in enumerate(tp) if a != b][:n]    # modx is only known below 2N': x' < N' here
        Nt = np.array([tp[i][0] for i in keep], dtype=object)
        xt = np.array([tp[i][1] for i in keep], dtype=object)
        eq, zero = np.zeros(len(keep), bool), np.zeros(len(keep), dtype=object)
        kind = np.arange(len(keep)) % 4       # 0..2: x' + N' (inverse check); 3: x' + 0 = x'
        Nr = np.where(kind < 3, Nt, zero)

        def inverse(A_out):                   # A_out: [S, n] values. (x' + N') mod N' = x' with modx's program,
            S = A_out.shape[0]               # and x' + N' >= N' with lt's (no borrow); x' + 0 = x' as a fact
            ok = np.ones(S, bool)
            sel = np.nonzero(kind < 3)[0]
            flatA = A_out[:, sel].reshape(-1)
            Ns = np.tile(Nt[sel], S)
            back, _ = run("modx", {"X": flatA, "N": Ns})
            want = np.where(eq[sel], zero[sel], xt[sel])
            ok &= (back.reshape(S, -1) == np.tile(want, S).reshape(S, -1)).all(1)
            _, cb = run("lt", {"X": flatA, "N": Ns})
            ok &= (cb.reshape(S, -1) == 0).all(1)
            sel0 = np.nonzero(kind == 3)[0]
            ok &= (A_out[:, sel0] == xt[sel0][None]).all(1)
            return ok
        return {"X": xt, "N": Nr}, {}, inverse
    if task == "modB":
        q = rng.integers(0, B, size=n)
        D = xt.copy()
        for step_ in range(B - 1):
            more = q > step_
            if more.any():
                s_, _ = run("add", {"X": D[more], "N": Nt[more]})
                D[more] = s_
        return {"X": D, "N": Nt}, {"A": np.where(eq, zero, xt)}, None
    if task == "mrow":
        d = rng.integers(0, B, size=n)
        Y = np.array([int(v) for v in d], dtype=object)
        T = zero.copy()
        for step_ in range(B - 1):            # x' * d = d additions of x'
            more = d > step_
            if more.any():
                s_, _ = run("add", {"X": T[more], "N": xt[more]})
                T[more] = s_
        return {"X": xt, "Y": Y}, {"A": T}, None
    if task == "mac":                         # x' * d + N' = add(mrow(x', d), N'): the row on top of a number
        d = rng.integers(0, B, size=n)
        Y = np.array([int(v) for v in d], dtype=object)
        R, _ = run("mrow", {"X": xt, "Y": Y})
        T, _ = run("add", {"X": R, "N": Nt})
        return {"X": xt, "Y": Y, "N": Nt}, {"A": T}, None
    if task == "mul":
        T = zero.copy()
        y = other
        for step_ in range(int(max(y)) if len(y) else 0):
            more = np.array([v > step_ for v in y])
            if more.any():
                s_, _ = run("add", {"X": T[more], "N": xt[more]})
                T[more] = s_
        return {"X": xt, "Y": y}, {"A": T}, None
    if task == "sq":
        T, _ = run("mul", {"X": xt, "Y": xt})
        return {"X": xt}, {"A": T}, None
    if task in ("step0", "step"):
        keep = ~eq
        Nk, xk = Nt[keep][:n], xt[keep][:n]
        m = len(Nk)
        d = np.array([int(v) for v in rng.integers(0, B, size=m)], dtype=object)
        R = xk if task == "step" else np.zeros(m, dtype=object)
        top = d * (B ** (st.P - 1))           # the digit d at the top place of X (placing a digit, no arithmetic
        X = top                               # on the machine's side: built here as a value only to make digits)
        v = R * B + d                         # = the digits of R shifted up with d below: a digit string, too
        T, _ = run("modB", {"X": v, "N": Nk})
        ins = {"X": X, "N": Nk}
        if task == "step":
            ins["Y"] = R
        return ins, {"A": T}, None
    if task == "reduce":                      # x' + Q N' with Q another row's x', Q additions of N' with add
        D = xt.copy()
        for step_ in range(int(max(other)) if len(other) else 0):
            more = np.array([v > step_ for v in other])
            if more.any():
                s_, _ = run("add", {"X": D[more], "N": Nt[more]})
                D[more] = s_
        return {"X": D, "N": Nt}, {"A": np.where(eq, zero, xt)}, None
    if task == "sqmod":
        S2, _ = run("mul", {"X": xt, "Y": xt})
        T, _ = run("reduce", {"X": S2, "N": Nt})
        return {"X": xt, "N": Nt}, {"A": T}, None
    raise ValueError(task)


class LawProblem:
    """search.Problem's interface (settings, roots, goal) with probes and targets from build()."""

    def __init__(self, task, settings, NP, seed, rows, helpers, lib):
        rng = np.random.default_rng(seed)
        self.task, self.settings = task, settings
        self.roots, self.targets, self.inverse = [], [], []
        for st in settings:
            # a one-bit target lets many wrong programs through by chance: twice the probes for lt
            ins, tgt, inv = build(task, st, 2 * NP if task == "lt" else NP, rng, rows, helpers, lib)
            n = len(next(iter(ins.values())))
            self.roots.append(dsl.fresh(st, {r: to_digits(v, st.B, st.P) for r, v in ins.items()}, rng, n))
            t = {}
            if "A" in tgt:
                t["A"] = to_digits(np.asarray(tgt["A"], dtype=object), st.B, st.P).T
            if "c" in tgt:
                t["c"] = np.asarray(tgt["c"], np.uint8)
            self.targets.append(t)
            self.inverse.append(inv)

    def goal(self, batches):
        ok = np.ones(batches[0].S, bool)
        for st, b, t, inv in zip(self.settings, batches, self.targets, self.inverse):
            if "A" in t:
                ok &= (b.regs[:, dsl.A] == t["A"][None]).all(axis=(1, 2))
            if "c" in t:
                ok &= (b.c == t["c"][None]).all(axis=1)
            if inv is not None and ok.any():
                idx = np.nonzero(ok)[0]
                A_out = dsl.from_digits(np.transpose(b.regs[idx, dsl.A], (0, 2, 1)), st.B)
                sub = inv(np.asarray(A_out, dtype=object).reshape(len(idx), -1))
                ok[idx[~sub]] = False
        return ok


def accept(task, prog, lib, helpers, rows, settings, n=60, seed=11, label_rows=None):
    """Does prog pass its task's laws on fresh rows at every setting (and the T = 1 labels at full size)?"""
    out = {}
    for st in settings:
        prob = LawProblem(task, [st], n, seed, rows, helpers, lib)
        b = prob.roots[0].copy()
        dsl.run_items(st, b, prog, lib)
        out[repr(st)] = bool(prob.goal([b])[0])
    if task == "sqmod" and label_rows:
        st = dsl.Setting(10, 8)
        Ns = np.array([r[0] for r in label_rows], dtype=object)
        xs = np.array([r[1] for r in label_rows], dtype=object)
        A, _ = run_prog(st, prog, lib, {"X": xs, "N": Ns}, np.random.default_rng(seed))
        out["labels(10,8)"] = float(np.mean(A == np.array([r[3] for r in label_rows], dtype=object)))
    return out
