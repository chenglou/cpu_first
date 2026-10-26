"""The one-more-digit test on an arm's routines, offline (COMPOSE.md §5): what the digits study's source
(bar3/scale/digits/source.py, DIGITS.md §2.3) would pose from each routine on this machine, had it been ported.

For each practised routine (`T.*` in the record's library), its value v in A on the target's inputs, by the machine's
exact cells, in two kinds of placement: the target's own (X = x, N = N: what the source reads), and, for a routine on
two operands, its own binding with x and N in both orders. The other registers start dirty, as every machine here
does. A placement whose value changes with the dirty start (two seeds) is not a function of x and N, so the source
could not pose it; it is reported and skipped. For each N, every x whose 10 children 10 x + d are on the grid gives a
row (v(x), d, N) -> v(10 x + d). v is its own state when those rows are a function on both grids. The pair
(x^2 mod N, v) is tested the same way (x^2 mod N is not its own state; with x mod N beside it, it is). Grids: x < 60
with N of 2 to 60 (the children: x < 600), and x < 30 with N of 2 to 30. They take x past N, as the source's rows
do, since H1's x < N leaves few x with all ten children below N.

A state's step is a piece of the round only when v is x mod N (its step is Horner's, (10 r + d) mod N: the
reduction's), or when the pair holds x mod N (the square's step). Every other state's step is reported as what it is.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose.digit_test bar3/scale/compose/records/cp_bar3.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import torch

from integration.lib import libcore as L

P = 16
GRIDS = ((60, 60), (30, 30))                        # (x below, N up to)


def digits(vals):
    out = torch.zeros(len(vals), P, dtype=torch.long)
    for i, v in enumerate(vals):
        for j in range(P):
            out[i, j] = v % 10
            v //= 10
    return out


def value(d):
    return [sum(int(v) * 10 ** i for i, v in enumerate(r)) for r in d.tolist()]


def values(prog, lib, xs, Ns, cells, place=("X", "N"), seed=1):
    st = L.Setting(10, P // 2, cells, P)
    A, _ = L.run_prog(st, prog, lib, {place[0]: digits(xs), place[1]: digits(Ns)}, L.gen_on("cpu", seed), "cpu")
    return value(A)


def placements(name):
    """The target's placement, and the routine's own binding (its name's last part) with x and N in both orders."""
    out = [("X", "N")]
    b = name.split(".")[-1]
    if len(b) == 2 and all(r in L.RI for r in b):
        out += [p for p in ((b[0], b[1]), (b[1], b[0])) if p not in out]
    return out


def is_state(rows):
    """rows: [(key, next)]: a function of the key?"""
    seen = {}
    for k, v in rows:
        if seen.setdefault(k, v) != v:
            return False
    return True


def test(prog, lib, cells, place):
    out = {}
    for xmax, nmax in GRIDS:
        xs, Ns = [], []
        for N in range(2, nmax + 1):
            for x in range(xmax):
                xs += [x] + [10 * x + d for d in range(10)]
                Ns += [N] * 11
        v = values(prog, lib, xs, Ns, cells, place)
        if v != values(prog, lib, xs, Ns, cells, place, seed=2):
            return dict(clean=False, state=False, pair_state=False, x_mod_N=False, piece=False)
        own, pair, modx = [], [], True
        for i in range(0, len(xs), 11):
            x, N = xs[i], Ns[i]
            y = x * x % N
            modx &= v[i] == x % N
            for d in range(10):
                c = i + 1 + d
                own.append(((v[i], d, N), v[c]))
                pair.append(((y, v[i], d, N), ((xs[c] ** 2) % N, v[c])))
        out[(xmax, nmax)] = (is_state(own), is_state(pair), modx)
    own = all(o for o, _, _ in out.values())
    pair = all(p for _, p, _ in out.values())
    modx = all(m for _, _, m in out.values())
    return dict(clean=True, state=own, pair_state=pair, x_mod_N=modx, piece=modx and (own or pair))


def main():
    rec = json.load(open(sys.argv[1]))
    lib = {k: L.parse(v) for k, v in rec["lib"].items()}
    cells = L.Cells(10, "cpu")
    rows = []
    for name in sorted(k for k in lib if k.startswith("T.")):
        for place in placements(name):
            r = test((name,), lib, cells, place)
            rows.append(dict(routine=name, program=rec["lib"][name], place="".join(place), **r))
            print(f"{name:34s} {rec['lib'][name][:52]:52s} x,N in {''.join(place)} "
                  + (f"state {r['state']!s:5s} pair {r['pair_state']!s:5s} x mod N {r['x_mod_N']!s:5s} piece {r['piece']}"
                     if r["clean"] else "dirty: not a function of x and N"), flush=True)
    clean = [r for r in rows if r["clean"]]
    print(f"{rec.get('tag')}: {len({r['routine'] for r in rows})} routines, {len(rows)} placements, {len(clean)} clean; "
          f"{sum(r['state'] for r in clean)} states, "
          f"{sum(r['pair_state'] and not r['state'] for r in clean)} more in a pair with x^2 mod N, "
          f"{sum(r['piece'] for r in clean)} a piece of the round")
    if len(sys.argv) > 2:
        json.dump(rows, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
