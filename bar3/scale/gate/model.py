"""The gate (GATE.md §2): a small network, the features' signature in, one logit out: is this state on the path of a
solution? Trained on the learner's own solved practice (harvest.py): the states its found programs pass through
(positives) against other states of the same searches (negatives), the two classes weighted equally, as BUSTLE's
training pairs one positive with one negative. Numpy only, so it runs wherever the search runs.

The delay (BUSTLE's w' = w + 5 - delta, Odena et al. 2021): a state's children are generated `delay` levels later.
delay = clip(round(kappa * (ref - logit)), 0, D): no delay for a state at least as likely as `ref`, one level per
1/kappa nats of evidence against it, at most D. Never infinite, so no state is dropped.
"""

from __future__ import annotations

import json

import numpy as np

from bar3.scale.gate import features as F


class Gate:
    def __init__(self, W1, b1, w2, b2, mask=None, kappa=1.0, ref=0.0, D=3, name=""):
        self.W1, self.b1 = np.asarray(W1, np.float32), np.asarray(b1, np.float32)
        self.w2, self.b2 = np.asarray(w2, np.float32), np.float32(b2)
        self.mask = None if mask is None else np.asarray(mask, np.float32)     # feature ablation: 0 zeroes one
        self.kappa, self.ref, self.D, self.name = float(kappa), float(ref), int(D), name

    def logits(self, X):
        if self.mask is not None:
            X = X * self.mask
        h = np.maximum(X @ self.W1 + self.b1, 0)
        return h @ self.w2 + self.b2

    def delays(self, logit):
        d = np.rint(self.kappa * (self.ref - logit))
        return np.clip(d, 0, self.D).astype(np.int64)

    def for_problem(self, prob):
        """The delay function the gated search calls on each new level: brm.Batch -> delays [S]."""
        c = F.Context(prob)

        def delay(batch):
            out = np.empty(batch.S, np.int64)
            for lo in range(0, batch.S, F.CHUNK):
                out[lo:lo + F.CHUNK] = self.delays(self.logits(F.features(batch.regs[lo:lo + F.CHUNK], c)))
            return out
        return delay

    def with_delay(self, kappa=None, ref=None, D=None):
        return Gate(self.W1, self.b1, self.w2, self.b2, self.mask, self.kappa if kappa is None else kappa,
                    self.ref if ref is None else ref, self.D if D is None else D, self.name)

    def to_json(self):
        return dict(W1=self.W1.tolist(), b1=self.b1.tolist(), w2=self.w2.tolist(), b2=float(self.b2),
                    mask=None if self.mask is None else self.mask.tolist(), kappa=self.kappa, ref=self.ref, D=self.D,
                    name=self.name)

    @staticmethod
    def from_json(d):
        return Gate(**d)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    @staticmethod
    def load(path):
        with open(path) as f:
            return Gate.from_json(json.load(f))


def train(X, y, hidden=64, epochs=30, lr=3e-3, batch=4096, wd=1e-4, seed=0, mask=None, log=print):
    """Balanced logistic loss, Adam. X [n, F] float32, y [n] 0/1. Returns a Gate (delay parameters to be set)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    if mask is not None:
        X = X * np.asarray(mask, np.float32)
    y = np.asarray(y, np.float32)
    n, nf = X.shape
    pos = y.sum()
    w = np.where(y > 0, 0.5 / max(pos, 1), 0.5 / max(n - pos, 1)) * n          # each class weighs half
    params = [rng.normal(0, (2.0 / nf) ** 0.5, (nf, hidden)).astype(np.float32), np.zeros(hidden, np.float32),
              rng.normal(0, (1.0 / hidden) ** 0.5, hidden).astype(np.float32), np.zeros((), np.float32)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        tot = 0.0
        for lo in range(0, n, batch):
            idx = order[lo:lo + batch]
            xb, yb, wb = X[idx], y[idx], w[idx]
            W1, b1, w2, b2 = params
            hpre = xb @ W1 + b1
            h = np.maximum(hpre, 0)
            z = h @ w2 + b2
            p = 1 / (1 + np.exp(-z))
            tot += float((wb * (np.logaddexp(0, z) - yb * z)).sum())
            g = wb * (p - yb) / len(idx)
            gw2 = h.T @ g
            gb2 = g.sum()
            gh = np.outer(g, w2) * (hpre > 0)
            gW1 = xb.T @ gh
            gb1 = gh.sum(axis=0)
            grads = [gW1 + wd * W1, gb1, gw2 + wd * w2, gb2]
            t += 1
            for i, (p_, g_) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * g_
                v[i] = 0.999 * v[i] + 0.001 * g_ * g_
                mh, vh = m[i] / (1 - 0.9 ** t), v[i] / (1 - 0.999 ** t)
                params[i] = (p_ - lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        if log and (ep % 10 == 9 or ep == epochs - 1):
            log(f"  epoch {ep + 1}: loss {tot / n:.4f}")
    return Gate(*params, mask=mask)


def auc(scores, y):
    """The area under the ROC curve (ties counted half)."""
    scores, y = np.asarray(scores, np.float64), np.asarray(y) > 0
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))
