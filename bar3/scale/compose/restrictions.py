"""Which restrictions of the target the oracle arm's source poses (COMPOSE.md §5.1), rebuilt offline with no search:
the climb as core_compose.py builds it (the tiny core's rows, settings and seed), and the restrict source's pose
(oracle.SmallTarget.pose with no near misses). The draws are fixed by the seed and the root's key, so every level
and every arm poses the same restrictions. For each: its predicate t1 < t2, its share of the rows and its probes, and
what the target is on those probes (x'^2 itself where every one has x'^2 < N').

    cd cpu_first && H1_DATA=... ~/github/sotaku/venv/bin/python -m bar3.scale.compose.restrictions
"""

from __future__ import annotations

import torch

from integration.lib import libcore as L
from bar3.scale.compose import climb as CC
from bar3.scale.compose.core_compose import digits, load_rows
from integration.self_practice.oracle import SmallTarget


def main():
    rows = load_rows(4096, seed=0)
    Nd, Xd = digits(rows, "modulus"), digits(rows, "x")
    target = SmallTarget(sources=("restrict",))
    c = CC.ComposeClimb(L.Cells(10, "cpu"), "cpu", 16, chunk=1 << 16, target=target, rule="lpca", NP=8)
    tr = target.root(c, (Nd, Xd))
    for st, d in zip(c.settings, tr.draws):                 # the source measures each predicate on the last draw
        sq = sum(x * x < N for x, N in zip(d.xs, d.Ns))
        print(f"the target's root, setting {st}: {d.n} probes, x'^2 < N' on {sq}; x' {d.xs} N' {d.Ns}")
    for r, goals in target.pose(c, (Nd, Xd), []):
        if r.kind != "restrict":
            continue
        for st, d in zip(c.settings, r.draws):
            sq = sum(x * x < N for x, N in zip(d.xs, d.Ns))
            print(f"{r.key}: {r.pred[0]} < {r.pred[1]}, setting {st}: {d.n} probes, x'^2 < N' on {sq}; "
                  f"x' {d.xs[:8]} N' {d.Ns[:8]} y {d.ys[:8]}")


if __name__ == "__main__":
    main()
