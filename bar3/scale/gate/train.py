"""Training the gate on the learner's own solved practice (GATE.md §3), and judging it before any search.

The data: capture files (capture.py), each solved search's harvest (positives: the states its accepted programs pass
through; negatives: other states of its levels), every state's features read against its own problem's rows
(features.py). Problems are split by where they come from:
- the law book, its depth-2 laws and the dreams pose the same problems for every target ("shared"): always training;
- the rest (restrictions, projections, near misses and remainders, recurrences, constants) are posed from one target's
  rows: they belong to that target's fold.
A gate trained on fold A is tested on fold B's problems, never on its own.

The judgement before search: the AUC on held-out problems, and the delays the gate gives right states (r_t: the share
of positives delayed t) and wrong ones (q_t, each level weighted equally), turned into the toy model's effective
branching factor (toy.py, graded delays): lam solves 1 = sum_t b q_t lam^-(1 + t), b_eff = lam^(1 + sum_t t r_t).
The delay map (kappa, ref, D) is chosen on the training fold's own validation targets by that b_eff.

    cd cpu_first && <python with numba> -m bar3.scale.gate.train CAPTURE.jsonl.gz ... --train T1,T2 --test T3,T4
"""

from __future__ import annotations

import gzip
import json
import math
import sys

import numpy as np

from library.inventing import brm
from bar3.search import Problem
from bar3.scale import jobs as J
from bar3.scale.gate import features as F
from bar3.scale.gate import model as M
from bar3.scale.gate.gjobs import unpack

SHARED = ("law", "law2", "dream")
GROUPS = {                       # feature ablations (GATE.md §4): which features a gate may read
    "full": lambda n: True,
    "noprob": lambda n: not n.startswith("p_"),
    "fitonly": lambda n: n.startswith(("fit", "eq_a")),
    "noanswer": lambda n: not n.startswith(("fit", "eq_a", "le_a", "ge_a", "par_a", "p_", "det", "live")),
    "nodet": lambda n: not n.startswith(("det", "live")),
}
FOLDS = {"A": ("h1", "rho", "minus1", "plus3", "affine", "ten", "tenrho", "cube", "mont", "fib"),
         "B": ("x2x", "x1sq", "three", "x4", "x3x", "pow2", "lucas", "trib", "floorsq", "tri")}
VAL = {"A": ("fib", "mont"), "B": ("lucas", "floorsq")}          # PREREG.md


def mask_of(group):
    return np.array([1.0 if GROUPS[group](n) else 0.0 for n in F.NAMES], np.float32)


def problem_of(job):
    """The job's problem as the search poses it (jobs._posed's first half)."""
    xs, Ns = np.array(job["xs"]), np.array(job["Ns"])
    if "outs" in job or "init" in job:
        return J.RProblem(xs, Ns, job["want"], job.get("init"), job.get("outs", ["A"]))
    return Problem(xs, Ns, np.array(job["want"]))


def target_of(run):
    return run.rsplit("_", 1)[-1]


def load(paths):
    """Every solved, harvested problem once: dict(key, run, target, source, shared, job, rec, harv)."""
    probs, jobs = {}, {}
    for p in paths:
        for line in gzip.open(p, "rt"):
            run, key, job, rec, harv = json.loads(line)
            if job is not None:
                jobs.setdefault(key, job)
            if key in probs or harv is None:
                continue
            job = jobs[key]
            src = job["name"].split("/")[0]
            probs[key] = dict(key=key, run=run, target=target_of(run), source=src, shared=src in SHARED, job=job,
                              rec=rec, harv=harv)
    return list(probs.values())


def dataset(probs):
    """X [n, F], y [n], group (problem index) [n], level [n]."""
    Xs, ys, gs, ls = [], [], [], []
    for i, p in enumerate(probs):
        c = F.Context(problem_of(p["job"]))
        h = p["harv"]
        for part, lab, lev in ((h["pos"], 1, h["plev"]), (h["neg"], 0, h["nlev"])):
            if part is None:
                continue
            regs = unpack(part).astype(np.int32)
            Xs.append(F.features(regs, c))
            ys.append(np.full(len(regs), lab, np.int8))
            gs.append(np.full(len(regs), i, np.int32))
            ls.append(np.asarray(lev, np.int16))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(ls)


def split(probs, train_targets, test_targets):
    tr = [p for p in probs if p["shared"] or p["target"] in train_targets]
    te = [p for p in probs if not p["shared"] and p["target"] in test_targets]
    return tr, te


def delay_shares(gate, logit, y, g, lev):
    """r_t (positives) and q_t (negatives, each (problem, level) weighted equally) for the gate's delay map."""
    t = gate.delays(logit)
    D = gate.D
    r = np.bincount(t[y == 1], minlength=D + 1) / max((y == 1).sum(), 1)
    neg = y == 0
    cell = g[neg].astype(np.int64) * 64 + lev[neg]
    _, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    q = np.bincount(t[neg], weights=w, minlength=D + 1)
    q = q / q.sum()
    return r, q


def b_eff(r, q, b):
    lo, hi = 1e-9, float(b) + 1
    for _ in range(200):
        mid = (lo + hi) / 2
        f = sum(b * qt / mid ** (1 + t) for t, qt in enumerate(q)) - 1
        lo, hi = (mid, hi) if f > 0 else (lo, mid)
    lam = (lo + hi) / 2
    return lam ** (1 + sum(t * rt for t, rt in enumerate(r)))


def tune(gate, logit, y, g, lev, b, log=print):
    """The delay map with the lowest toy b_eff on (logit, y): D, kappa and ref (a quantile of the positives' logits)."""
    pos = logit[y == 1]
    best = None
    for D in (1, 2, 3, 4, 6):
        for kappa in (0.25, 0.5, 1.0, 2.0):
            for qu in (0.05, 0.1, 0.25, 0.5):
                gt = gate.with_delay(kappa=kappa, ref=float(np.quantile(pos, qu)), D=D)
                r, q = delay_shares(gt, logit, y, g, lev)
                be = b_eff(r, q, b)
                if best is None or be < best[0]:
                    best = (be, gt, r, q, qu)
    be, gt, r, q, qu = best
    log(f"  tuned: D={gt.D} kappa={gt.kappa} ref={gt.ref:.2f} (positives' {qu} quantile): toy b_eff {be:.2f} "
        f"(b={b}); r_t {np.round(r, 3).tolist()} q_t {np.round(q, 3).tolist()}")
    return gt


def judge(gate, X, y, g, lev, b, log=print, label=""):
    logit = gate.logits(X)
    a = M.auc(logit, y)
    r, q = delay_shares(gate, logit, y, g, lev)
    be = b_eff(r, q, b)
    log(f"  {label}: AUC {a:.3f}; toy b_eff {be:.2f} (b={b}, plain {b}); per tenfold {math.log(10) / math.log(be):.2f} "
        f"instructions (plain {math.log(10) / math.log(b):.2f}); r_t {np.round(r, 3).tolist()} "
        f"q_t {np.round(q, 3).tolist()}")
    return dict(auc=a, b_eff=be, r=r.tolist(), q=q.tolist())


def train_all(paths, corpus, b, groups, out_dir, log=print):
    """Every gate of one corpus: per fold (trained on it, judged on the other) and feature group. Writes
    <out_dir>/<corpus>_<fold>_<group>.json and returns the judgements."""
    import os
    probs = load(paths)
    log(f"{corpus}: {len(probs)} solved problems")
    out = {}
    for fold, other in (("A", "B"), ("B", "A")):
        tr, te = split(probs, FOLDS[fold], FOLDS[other])
        Xt, yt, gt, lt = dataset(te)
        for group in groups:
            gate = fit(tr, VAL[fold], b=b, group=group, log=log)
            gate.name = f"{corpus}_{fold}_{group}"
            gate.save(os.path.join(out_dir, gate.name + ".json"))
            out[gate.name] = judge(gate, Xt, yt, gt, lt, b, log=log, label=f"test {gate.name} ({len(te)} problems)")
            out[gate.name].update(n_train=len(tr), n_test=len(te), D=gate.D, kappa=gate.kappa, ref=gate.ref)
    return out


def fit(train_probs, val_targets, b=10, group="full", hidden=64, epochs=25, seed=0, log=print):
    """Train on the training fold less its validation targets; choose the delay map on those (tune)."""
    vt = set(val_targets)
    tr = [p for p in train_probs if p["shared"] or p["target"] not in vt]
    va = [p for p in train_probs if not p["shared"] and p["target"] in vt]
    mask = mask_of(group)
    X, y, g, lev = dataset(tr)
    log(f"  fit [{group}]: {len(tr)} problems, {int(y.sum())} positives, {int((y == 0).sum())} negatives")
    gate = M.train(X, y, hidden=hidden, epochs=epochs, seed=seed, mask=mask, log=log)
    del X
    Xv, yv, gv, lv = dataset(va)
    tuned = tune(gate, gate.logits(Xv), yv, gv, lv, b, log=log)
    judge(tuned, Xv, yv, gv, lv, b, log=log, label=f"validation ({len(va)} problems)")
    tuned.name = group
    return tuned


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    probs = load(args)
    train_t, test_t = opt["train"].split(","), opt["test"].split(",")
    val_t = opt.get("val", train_t[-1]).split(",")
    tr, te = split(probs, train_t, test_t)
    print(f"{len(probs)} solved problems; training fold {len(tr)}, test {len(te)}")
    for group in opt.get("groups", "full").split(","):
        gate = fit(tr, val_t, b=float(opt.get("b", 10)), group=group)
        Xt, yt, gt, lt = dataset(te)
        judge(gate, Xt, yt, gt, lt, float(opt.get("b", 10)), label=f"test [{group}] ({len(te)} problems)")
        if "save" in opt:
            gate.save(opt["save"].replace("{group}", group))


if __name__ == "__main__":
    main()
