"""Which whole (reading, digit code) candidates the facts leave on a dataset, and which of them the answers leave
(FIXED_MODULUS.md section 1). Exact counting on the training rows, a diagnostic run after the fact; exact arithmetic
on the labels stands in for an exact machine, as in ../code_search.py. Two differences from code_search.py, so that
the counts are the model's own:

  * the reader's semantics: a reading hypothesis h (which digit run is N and which way it is read, the same for x;
    T is the run left) passes a row when, at the highest place where the two token strings differ (a missing digit
    counts as lower than any digit), x's token ranks below N's; where x has ended there, N's token must not be the
    one coded 0 (code_search padded x with the true 0 token instead). A row where N ends first, or x equals N, fails h.
    T's run must fit the reader's K = 2 slots;
  * v2's answer rows: one-digit T up to max(REAL_MAX_T, the data's smallest one-digit T); code_search used T <= 2,
    which left M1 and M2 (T = 4, 8, 16) with none.

A code is a bijection from the ten digit tokens to the labels 0..9; under (h, code) every field and the answer are
label numbers (the answer read from its last token, the writer's convention), and the candidate fits an answer row
when x'^(2^T') mod N' = y' on the labels, T' being T's own label value. Codes are counted by filtering all 10!
permutations; answers eliminate candidates row by row.

    python fixed/code_census.py DATA_DIR [DATA_DIR ...] [--rounds]     # one JSON line per dataset

--rounds also counts the candidates that fit under some other option of the general file's shared round menu with
long division (the square times 10 or not, plus a constant 0..9, times the round's input and reduced again or not):
impostors a joint tournament over (code, round options) could crown.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

DIGIT0, K, REAL_MAX_T = 7, 2, 2
HYP = [(rn, dn, rx, dx) for rn in range(3) for rx in range(3) if rn != rx for dn in range(2) for dx in range(2)]
TRUE_H = (0, 1, 1, 1)                     # N is the first run, x the second, both read from their right ends
_PERMS = None


def perms() -> np.ndarray:
    """All 10! codes as label arrays [10!, 10] (code[i] = the label of digit token DIGIT0 + i)."""
    global _PERMS
    if _PERMS is None:
        _PERMS = np.array(list(itertools.permutations(range(10))), dtype=np.int8)
    return _PERMS


def runs(ids: list) -> list:
    """The maximal digit-token runs of a prompt, as lists of digit indices 0..9 (token - DIGIT0)."""
    out, cur = [], []
    for t in ids + [-1]:
        if DIGIT0 <= t < DIGIT0 + 10:
            cur.append(t - DIGIT0)
        elif cur:
            out.append(cur)
            cur = []
    return out


def read(run: list, d: int) -> list:
    """A run read LSB first: direction 0 from its left end, 1 from its right end."""
    return list(run) if d == 0 else list(run[::-1])


def constraints(rows: list, h: tuple):
    """The reader's facts under hypothesis h: pairs (a, b) meaning code[a] < code[b], digits that must not be coded 0,
    and whether every row passes at all (None if some row fails whatever the code)."""
    rn, dn, rx, dx = h
    pairs, nonmin = set(), set()
    for r in rows:
        rs = r["runs"]
        if len(rs) != 3 or len(rs[3 - rn - rx]) > K:
            return None
        n, x = read(rs[rn], dn), read(rs[rx], dx)
        L = max(len(n), len(x))
        n, x = n + [-1] * (L - len(n)), x + [-1] * (L - len(x))
        top = max((j for j in range(L) if n[j] != x[j]), default=-1)
        if top < 0 or n[top] < 0:
            return None
        if x[top] < 0:
            nonmin.add(n[top])
        else:
            pairs.add((x[top], n[top]))
    return pairs, nonmin


def codes_passing(pairs: set, nonmin: set) -> np.ndarray:
    P = perms()
    ok = np.ones(len(P), dtype=bool)
    for a, b in pairs:
        ok &= P[:, a] < P[:, b]
    for b in nonmin:
        ok &= P[:, b] != 0
    return P[ok]


def value(codes: np.ndarray, digits: list) -> np.ndarray:
    """The label number of LSB-first digits under every code [n_codes]."""
    v = np.zeros(len(codes), dtype=np.int64)
    for j, d in enumerate(digits):
        v += codes[:, d].astype(np.int64) * 10 ** j
    return v


def step_fn(rmul: int, radd: int, rcube: bool):
    def f(y, x, n):
        s = (y * y * rmul + radd) % n
        return s * x % n if rcube else s
    return f


def fits(codes: np.ndarray, rows: list, h: tuple, fn) -> np.ndarray:
    """Which codes reproduce every answer row under hypothesis h and the round function fn (rows eliminate)."""
    rn, dn, rx, dx = h
    alive = np.arange(len(codes))
    for r in rows:
        if not len(alive):
            break
        c = codes[alive]
        rs = r["runs"]
        n = value(c, read(rs[rn], dn))
        x = value(c, read(rs[rx], dx))
        t = value(c, read(rs[3 - rn - rx], 1))
        y = value(c, r["answer"][::-1])
        good = n > 0
        nn = np.where(good, n, 1)
        cur = x % nn
        for k in range(int(t.max()) if len(t) else 0):
            cur = np.where(t > k, fn(cur, x, nn), cur)
        alive = alive[good & (cur == y)]
    out = np.zeros(len(codes), dtype=bool)
    out[alive] = True
    return out


def census(data: str, rounds: bool = False) -> dict:
    rows = []
    for line in open(Path(data) / "train.jsonl"):
        r = json.loads(line)
        rows.append({"runs": runs(r["input_ids"]), "answer": [t - DIGIT0 for t in r["labels"]], "T": r["time_steps"],
                     "N": r["modulus"]})
    one = [r["T"] for r in rows if r["T"] < 10]
    tmax = max(REAL_MAX_T, min(one)) if one else REAL_MAX_T
    ans = [r for r in rows if r["T"] <= tmax]
    out = {"data": Path(data).name, "moduli": len({r["N"] for r in rows}), "rows": len(rows),
           "answer_T_max": tmax, "answer_rows": len(ans), "readings": {}}
    true_code = np.arange(10, dtype=np.int8)
    fitting = []
    plain = step_fn(1, 0, False)
    for h in HYP:
        c = constraints(rows, h)
        if c is None:
            continue
        codes = codes_passing(*c)
        if not len(codes):
            continue
        f = fits(codes, ans, h, plain)
        key = "".join(map(str, h))
        rec = {"codes": int(len(codes)), "pairs": len(c[0]), "nonmin": sorted(c[1]), "fit": int(f.sum())}
        if rounds:
            other = 0
            for rmul, radd, rcube in itertools.product((1, 10), range(10), (False, True)):
                if (rmul, radd, rcube) != (1, 0, False):
                    other += int(fits(codes, ans, h, step_fn(rmul, radd, rcube)).sum())
            rec["fit_other_rounds"] = other
        out["readings"][key] = rec
        for code in codes[f]:
            fitting.append((key, "".join(map(str, code.tolist()))))
    out["candidates"] = sum(v["codes"] for v in out["readings"].values())
    out["fitting"] = fitting[:20]
    out["n_fitting"] = len(fitting)
    out["true_fits"] = ("".join(map(str, TRUE_H)), "".join(map(str, true_code.tolist()))) in fitting
    return out


if __name__ == "__main__":
    rounds = "--rounds" in sys.argv
    for d in [a for a in sys.argv[1:] if not a.startswith("--")]:
        print(json.dumps(census(d, rounds)), flush=True)
