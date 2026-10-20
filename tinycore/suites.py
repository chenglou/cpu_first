"""Job lists for tinycore/modal_tinycore.py, and the function a container runs for one job.

Suites:
- fixes_rest  the fix comparison's jobs (fixsweep.jobs) not yet in runs/fixes.jsonl
- b10         microlab's sizes (base 10, odd N of 6, 7, 10 bits, v < N^2): microlab's analogs and the construction
              variants (stopped early; its finished rows are in runs/b10.jsonl)
- b10b        the cell from label-free identities first; construction + staging on every w-digit modulus (control)
- b10c        construction + staging with the register's top digit also scored (stopped early)
- b10d        answers only: shared subtractor + estimator (window 3/2), staged, then the cell's own borrow
- b10e        the same trained at 6 or 7 bits, then run unchanged at 6, 7 and 10 bits
- b10f        the same with a coarse estimator window (2/1), with the width transfer
- grad0       the first gradient at the uniform start, per table entry: toward the truth, away, or zero
- lesson      from the uniform start, how often a small step on the choice's own parameters moves the choice toward
              the true digit, every 25 steps of joint training (learned slots vs constructed slots)
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _name(*parts):
    return "_".join(str(p) for p in parts)


def fixes_rest():
    from .fixsweep import jobs
    path = os.path.join(HERE, "runs", "fixes.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            done.add((r["label"], r["B"], r["w"], r["k"], r["choice"], r["seed"]))
    out = []
    for j in jobs():
        c = j["cfg"]
        key = (j["label"], c["B"], c["w"], c["k"], c["choice"], j["seed"])
        if key in done:
            continue
        out.append({"kind": "fix", "job": j, "name": _name(j["label"], c["B"], c["w"], c["k"], c["choice"], j["seed"])})
    return out


B10_VARIANTS = [
    ("slot", "borrow", "joint", "prefix"),     # microlab longdiv (table cell)
    ("slot", "est", "joint", "prefix"),        # microlab longdiv_est
    ("slot", "borrow", "staged", "prefix"),    # curriculum only
    ("slot", "borrow", "checks", "prefix"),    # label-free cell first
    ("mult", "borrow", "joint", "prefix"),     # construction only
    ("mult", "borrow", "staged", "prefix"),    # construction + curriculum
    ("mult", "est", "staged", "prefix"),
    ("mult", "borrow", "staged", "full"),      # the same from R = 0 (w extra always-0 steps)
    ("repeat", "borrow", "staged", "prefix"),  # construction by repeated subtraction
]


def b10(seeds=3):
    out = []
    for bits in (6, 7, 10):
        for cell, choice, fix, start in B10_VARIANTS:
            for s in range(seeds):
                args = {"bits": bits, "cell": cell, "choice": choice, "fix": fix, "start": start, "seed": s,
                        "steps": 6000, "stage": 2000}
                out.append({"kind": "b10", "args": args, "name": _name(bits, cell, choice, fix, start, s)})
    return out


B10B_VARIANTS = [
    # the cell from label-free identities first, then the choice from answers
    ("slot", "est", "checks", "prefix", "microlab"),
    ("slot", "estc", "checks", "prefix", "microlab"),
    ("mult", "borrow", "checks", "prefix", "microlab"),
    # control: construction + staging on every w-digit modulus (wide leading digits)
    ("mult", "borrow", "staged", "prefix", "wide"),
    ("slot", "borrow", "joint", "prefix", "wide"),
]


def b10b(seeds=3):
    out = []
    for bits in (6, 7, 10):
        for cell, choice, fix, start, mods in B10B_VARIANTS:
            for s in range(seeds):
                args = {"bits": bits, "cell": cell, "choice": choice, "fix": fix, "start": start, "seed": s,
                        "steps": 6000, "stage": 2000, "mods": mods}
                out.append({"kind": "b10", "args": args, "name": _name(bits, cell, choice, fix, start, mods, s)})
    return out


B10C_VARIANTS = [                               # every one also scores the register's top digit (target 0)
    ("mult", "borrow", "staged", "prefix"),      # the candidate
    ("mult", "borrow", "joint", "prefix"),       # without staging
    ("mult", "est", "staged", "prefix"),         # with the estimator choice
    ("mult", "borrow", "staged", "full"),        # from R = 0, as microlab runs (w extra always-0 steps)
    ("slot", "borrow", "staged", "prefix"),      # learned slots (microlab longdiv) with staging
    ("slot", "borrow", "joint", "prefix"),       # microlab longdiv + the top digit only
]


def b10c(seeds=4):
    out = []
    for bits in (6, 7, 10):
        for cell, choice, fix, start in B10C_VARIANTS:
            for s in range(seeds):
                args = {"bits": bits, "cell": cell, "choice": choice, "fix": fix, "start": start, "seed": s,
                        "steps": 6000, "stage": 2000, "score_top": True}
                out.append({"kind": "b10", "args": args, "name": _name(bits, cell, choice, fix, start, "top", s)})
    return out


def b10d(seeds=4):
    """The estimator bootstraps the cell (staged), then the cell's own borrow chooses: answers only."""
    out = []
    for bits in (6, 7, 10):
        for cell in ("mult", "slot"):
            for s in range(seeds):
                args = {"bits": bits, "cell": cell, "choice": "est", "fix": "staged", "start": "prefix", "seed": s,
                        "steps": 4000, "stage": 1000, "switch": "borrow", "switch_steps": 2000}
                out.append({"kind": "b10", "args": args, "name": _name(bits, cell, "est", "staged", "switch", s)})
    return out


def b10e(seeds=3):
    """The answers-only route trained at 6 or 7 bits, then run unchanged at 6, 7 and 10 bits (rule or table?)."""
    out = []
    for bits in (6, 7):
        for s in range(seeds):
            args = {"bits": bits, "cell": "mult", "choice": "est", "fix": "staged", "start": "prefix", "seed": s,
                    "steps": 3000, "stage": 1000, "switch": "borrow", "switch_steps": 2000,
                    "transfer_bits": [6, 7, 10]}
            out.append({"kind": "b10", "args": args, "name": _name(bits, "mult", "est", "staged", "switch", "xfer", s)})
    return out


def b10f(seeds=4):
    """The answers-only route with a coarse estimator window (top 2 digits of 10R + v_t, top 1 of N), which cannot
    store a lookup that covers for subtraction errors; then the cell's own borrow; then the same tables at other
    widths."""
    out = []
    for bits in (6, 7, 10):
        for s in range(seeds):
            args = {"bits": bits, "cell": "mult", "choice": "est", "fix": "staged", "start": "prefix", "seed": s,
                    "steps": 3000, "stage": 1000, "switch": "borrow", "switch_steps": 2000, "kr": 2, "kn": 1,
                    "transfer_bits": [6, 7, 10]}
            out.append({"kind": "b10", "args": args, "name": _name(bits, "mult", "est21", "staged", "switch", s)})
    return out


def grad0():
    out = []
    for B, w, k in [(10, 2, 1), (10, 2, 2), (10, 3, 2)]:
        for cell in ("slot", "mult"):
            out.append({"kind": "grad0", "args": {"B": B, "w": w, "k": k, "cell": cell, "choice": "borrow"},
                        "name": _name(B, w, k, cell, "borrow")})
    return out


def lesson():
    out = []
    for B, w, k in [(2, 4, 1), (2, 4, 3), (3, 2, 1), (3, 2, 2), (4, 2, 1), (4, 2, 2)]:
        for cell in ("slot", "mult"):
            for choice in ("est", "borrow"):
                out.append({"kind": "lesson", "args": {"B": B, "w": w, "k": k, "cell": cell, "choice": choice},
                            "name": _name(B, w, k, cell, choice)})
    return out


SUITES = {"fixes_rest": fixes_rest, "b10": b10, "b10b": b10b, "b10c": b10c, "b10d": b10d, "b10e": b10e,
          "b10f": b10f, "grad0": grad0, "lesson": lesson}


# ----- what a container runs -----
def run_grad0(B, w, k, cell, choice, n=4000, chunk=500):
    """The gradient at the uniform start, accumulated over chunks of a fixed sample (bounded memory)."""
    import torch
    from .core import Config, Core, Data
    from .exact import toward_truth
    d = Data(B, w, k, draw=n, seed=1)
    out = {}
    m0 = Core(Config(B=B, w=w, k=k, cell=cell, choice=choice, init=0.0))
    mv = Core(Config(B=B, w=w, k=k, cell=cell, choice=choice, init=0.5), seed=3)  # which rows the data visits
    for m in (m0, mv):
        for c in m.chunks(d):
            (m.loss(c) * c.X / d.X).backward()
    for n_, T in (("a", m0.TD), ("b", m0.TC)):
        proj = toward_truth(m0.params[n_].grad, T)
        vis = mv.params[n_].grad.abs().sum(-1) > 1e-12
        pv = proj[vis]
        out[n_] = [int((pv > 1e-12).sum()), int((pv < -1e-12).sum()), int((pv.abs() <= 1e-12).sum())]
    return {"B": B, "w": w, "k": k, "cell": cell, "choice": choice, "digit_rows_toward_away_zero": out["a"],
            "carry_rows_toward_away_zero": out["b"]}


def run_lesson(B, w, k, cell, choice, steps=200, every=25, lr=0.05):
    from .lesson import run
    rows = run(B, w, k, cell, choice, steps=steps, every=every, lr=lr)
    return {"B": B, "w": w, "k": k, "cell": cell, "choice": choice, "rows": rows}


def execute(job):
    kind = job["kind"]
    if kind == "fix":
        from .fixsweep import run
        r = run(job["job"])
    elif kind == "b10":
        from .b10 import run
        r = run(**job["args"])
    elif kind == "grad0":
        r = run_grad0(**job["args"])
    elif kind == "lesson":
        r = run_lesson(**job["args"])
    else:
        raise ValueError(kind)
    r["name"] = job["name"]
    return r
