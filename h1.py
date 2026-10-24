"""H1 data in digit form + a faithful certification evaluator.

The competition prompt is `N <N digits LSB-first> X <x digits MSB-first> T <T digits>` with the
answer's digits right-aligned onto the last prompt positions. That is pure plumbing, so experiments here
work on LSB-first digit arrays directly: N_d[i], X_d[i], Y_d[i] = digit i (10^i place). Scoring matches
the evaluator: only the len(str(y)) low digits of y are scored, exact match per example.
"""

from __future__ import annotations

import json
import os

import numpy as np

DATA = os.environ.get(
    "H1_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "one-layer-deeper", "data", "generated",
                 "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"),
)
LADDER = (1, 2, 4, 8, 16, 32, 64)
L = 10  # digit slots: 24-bit N has 8 digits; keep guard digits


def load(split: str) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in open(os.path.join(DATA, split + ".jsonl"))]
    get = lambda k: np.array([r[k] for r in rows], dtype=np.int64)
    return {
        "N": get("modulus"),
        "X": get("x"),
        "T": get("time_steps"),
        "Y": get("result"),
        "bits": get("configured_modulus_bits"),
    }


def digits(v: np.ndarray, width: int = L) -> np.ndarray:
    out = np.zeros((len(v), width), dtype=np.int64)
    w = np.array(v, dtype=np.int64).copy()
    for i in range(width):
        out[:, i] = w % 10
        w //= 10
    return out


def ndigits(v: np.ndarray) -> np.ndarray:
    return np.array([len(str(int(a))) for a in v], dtype=np.int64)


def exact(pred_digits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluator-faithful exact match: compare only the scored (low len(str(y))) digits."""
    yd = digits(y, pred_digits.shape[1])
    scored = np.arange(pred_digits.shape[1])[None, :] < ndigits(y)[:, None]
    return ((pred_digits == yd) | ~scored).all(axis=1)


def certify(predict, profile: str = "depth_t", ladder=LADDER, verbose: bool = True) -> dict:
    """predict(N, X, T) -> LSB-first digit array [B, L]. profile: 'depth_t' or 'depth_ood_n_t'."""
    accs, max_t, broken = {}, 0, False
    for t in ladder:
        d = load(f"{profile}_{t}")
        ok = exact(predict(d["N"], d["X"], d["T"]), d["Y"])
        accs[t] = float(ok.mean())
        if not broken and ok.all():
            max_t = t
        else:
            broken = True
        if verbose:
            print(f"  {profile} T={t:<3d} exact={accs[t]:.4f}", flush=True)
    return {"max_t": max_t, "accs": accs}


def reference_predict(N, X, T):
    """The CPU solution (for testing the harness): T modular squarings, exact integers."""
    out = []
    for n, x, t in zip(N, X, T):
        s = int(x) % int(n)
        for _ in range(int(t)):
            s = s * s % int(n)
        out.append(s)
    return digits(np.array(out))


if __name__ == "__main__":
    for prof in ("depth_t", "depth_ood_n_t"):
        print(prof, certify(reference_predict, prof, verbose=False))
