"""The toy model of theory/GUIDED_SEARCH.md §3: what a gate that delays states, never drops them, does to the effective
branching factor of a uniform-cost search.

A tree with branching b, every edge one instruction. The gate passes a state (delay 0) or delays it by D: a delayed
state's children cost D more. Off the solution path a state passes with probability q (the false-pass rate), on it with
probability r (the recall). States of cost <= C then number about lam**C, where lam solves
    1 = b*q/lam + b*(1 - q)/lam**(1 + D),
and the solution, L instructions deep, costs L + D*(L - 1)*(1 - r) on average. So per instruction of the solution the
search pays b_eff = lam ** (1 + D*(1 - r)), and ten times the states buys log(10)/log(b_eff) instructions.

    python3 bar3/scale/gate/toy.py
"""

import math


def lam(b, q, D):
    lo, hi = 1e-9, float(b) + 1
    for _ in range(200):
        mid = (lo + hi) / 2
        f = b * q / mid + b * (1 - q) / mid ** (1 + D) - 1
        lo, hi = (mid, hi) if f > 0 else (lo, mid)
    return (lo + hi) / 2


def b_eff(b, q, r, D):
    return lam(b, q, D) ** (1 + D * (1 - r))


def best(b, q, r, Ds=range(0, 13)):
    return min((b_eff(b, q, r, D), D) for D in Ds)


if __name__ == "__main__":
    print("b, q (false pass), r (recall): best delay D, b_eff, instructions per tenfold (plain: 1/log10(b))")
    for b in (10, 40):
        for q, r in ((1.0, 1.0), (0.5, 0.7), (0.5, 0.9), (0.3, 0.8), (0.2, 0.9), (0.1, 0.9), (0.1, 0.99), (0.05, 0.95)):
            be, D = best(b, q, r)
            print(f"  b={b:>2} q={q:<4} r={r:<4}: D={D:>2} b_eff={be:6.2f}  per tenfold {math.log(10) / math.log(be):.2f}"
                  f" (plain {math.log(10) / math.log(b):.2f}; likelihood ratio r/q = {r / q:.1f})")
