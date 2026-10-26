"""Which restrictions the source poses with RESTRICT_ROWS (COMPOSE2.md §5): each predicate's share and pattern are
measured on the climb's 4,096 rows' top parts at the target's last width (hooks.share_rows), which do not depend on
the device; only each restriction's own probes do (a generator on the device). For each posed predicate: its share
there, and on its own probes (here, CPU draws) how many have y = x'^2 (x'^2 < N').

    cd cpu_first && H1_DATA=... ~/github/sotaku/venv/bin/python -m bar3.scale.compose2.restrictions_c2
"""

from __future__ import annotations

import torch

from integration.lib import libcore as L
from bar3.scale.compose2 import climb_c2 as CC
from bar3.scale.compose2 import hooks as H
from bar3.scale.compose2.core_c2 import digits, load_rows
from bar3.scale.compose2.oracle_c2 import TERMS, SmallTarget


def main():
    rows = load_rows(4096, seed=0)
    R = (digits(rows, "modulus"), digits(rows, "x"))
    t = SmallTarget(sources=("restrict",))
    t.restrict_rows = True
    c = CC.ComposeClimb(L.Cells(10, "cpu"), "cpu", 16, chunk=1 << 16, target=t, rule="lpca", NP=8)
    sr = H.share_rows(t, c, R)
    print(f"share rows: {len(sr.xs)} (the climb's rows' top parts at the target's last width)")
    for r, _ in t.pose(c, R, []):
        if r.kind != "restrict":
            continue
        a, b = r.pred
        share = sum(TERMS[a](x, N) < TERMS[b](x, N) for x, N in zip(sr.xs, sr.Ns)) / len(sr.xs)
        sq = [sum(x * x < N for x, N in zip(d.xs, d.Ns)) for d in r.draws]
        print(f"{r.key}: {a} < {b}, share {share:.3f}; its probes with x'^2 < N': {sq} of {[d.n for d in r.draws]}")
    print("torch", torch.__version__)


if __name__ == "__main__":
    main()
