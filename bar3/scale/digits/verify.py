"""DIGITS.md §5: the outcomes, read from the records, each program rechecked. For every run: the round its target came
in, or that it did not; the program and its size in primitive instructions; and the recheck on the compiled kernel
(machine.evaluate):
- the target's rows at both sizes, on the target's machines;
- the check at big x the learner used (N <= 10, 200 values of x with 6–8 digits, 100,000 steps);
- beyond it, fresh samples (seeded apart): 1,000 values of x below 10^9 at N <= 10 within the same 100,000 steps
  (longer x); and 1,000 at N <= 40 with up to 10^7 steps (bigger N: right, whatever it costs).
Also the round each digit step (source "digit") was posed, learnable and practised, and the search-hours. No search
runs.

    cd cpu_first && <python with numba> -m bar3.scale.digits.verify [records/*.jsonl ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from library import dsl
from library.inventing import brm
from bar3.scale.digits import machine as M
from bar3.scale.digits import targets as DT

HERE = Path(__file__).resolve().parent


def wide(name, top_n, n=1000, seed=1):
    rng = np.random.default_rng(seed + top_n)
    Ns = rng.integers(1, top_n + 1, n).astype(np.int64)
    xs = rng.integers(0, 10 ** 9, n).astype(np.int64)
    return xs, Ns, DT.answers(name, xs, Ns)


def right(prog, lib, data, m):
    xs, Ns, want = data
    got, dead = M.evaluate(prog, lib, xs, Ns, m)
    return "dead" if dead else f"{int((got == want).sum())}/{len(want)}"


def read(path):
    with open(path) as fh:
        return [r for r in (json.loads(line) for line in fh if line.strip()) if "round" in r]


def steps_seen(recs):
    """{digit problem: [round posed, round learnable, round practised]} (None where it never was)."""
    out = {}
    for r in recs:
        for c in r["pool"]:
            if c["source"] == "digit":
                out.setdefault(c["name"], [r["round"], None, None])
        for n, st in r["status"].items():
            if n in out and st.get("status") in ("learnable", "solved in practice") and out[n][1] is None:
                out[n][1] = r["round"]
        for n in r["solved"]:
            if n in out and out[n][2] is None:
                out[n][2] = r["round"]
    return out


def hours(recs):
    return sum(t["seconds"] for t in recs[-1]["totals"].values()) / 3600 if recs else 0.0


def rejected(recs):
    """PREREG prediction 4: the programs the target and guided searches found right on all the target's rows but not
    at big x (the first 10 kept per search). Those with a digit block of base >= 2 in their expansion are rechecked at
    big x with 10^7 steps: right then means too slow for the budget, not wrong."""
    seen, digit = {}, []
    name = recs[0]["target_name"]
    for r in recs:
        lib = {t: dsl.parse(v) for t, v in r["lib"].items()}
        for key in ("target", "guided"):
            got = (r.get(key) or {}).get("right_on_rows_only") or {}
            for p in got.get("first", []):
                if p["program"] in seen:
                    continue
                prim = p["primitive"]
                seen[p["program"]] = prim
                if "D.2[" in prim or "D.10[" in prim:
                    share = DT.bigx_share(dsl.parse(p["program"]), lib, name, brm.Machine(cap=10 ** 9, limit=10 ** 7))
                    digit.append((p["program"], p["general"], share))
    return len(seen), digit


def check(path):
    recs = read(path)
    if not recs:
        return f"{Path(path).name}: no rounds"
    name = recs[0]["target_name"]
    r = next((x for x in recs if x.get("general")), None)
    seen = steps_seen(recs)
    practised = sorted((v[2], n) for n, v in seen.items() if v[2] is not None)
    head = f"{Path(path).name}: {len(recs)} rounds, {hours(recs):.2f} search-hours"
    n_rej, digit = rejected(recs)
    steps = (f"; digit steps posed {len(seen)}, first learnable in round "
             f"{min((v[1] for v in seen.values() if v[1] is not None), default='–')}, practised: "
             + (", ".join(f"{n} (round {rd})" for rd, n in practised) or "none")
             + f"; right on the rows but not at big x: {n_rej} kept, {len(digit)} with a digit loop of base >= 2"
             + (" (" + "; ".join(f"{p}: {g:.0%} in the budget, {s:.0%} with 10^7 steps" for p, g, s in digit[:3]) + ")"
                if digit else ""))
    if r is None:
        return head + "; did not come" + steps
    g = r["general"]
    lib = {t: dsl.parse(v) for t, v in r["lib"].items()}
    prog = dsl.parse(g["program"])
    out = [right(prog, lib, DT.rows(name, size), DT.machine(size)) for size in (7, 10)]
    big = DT.bigx_share(prog, lib, name)
    longer = right(prog, lib, wide(name, 10), brm.Machine(cap=10 ** 9, limit=DT.BIG_STEPS))
    bigger = right(prog, lib, wide(name, 40), brm.Machine(cap=10 ** 9, limit=10 ** 7))
    prim = dsl.cost(dsl.expand(prog, lib))
    return (head + f"; came in round {r['round']}: {g['program']} ({prim} primitive); rows {out[0]} and {out[1]}, "
            f"big x {big:.0%}; x < 10^9 at N <= 10 {longer}, at N <= 40 {bigger}" + steps)


def main():
    paths = sys.argv[1:] or sorted(str(p) for p in (HERE / "records").glob("*.jsonl"))
    for p in paths:
        print(check(p))


if __name__ == "__main__":
    main()
