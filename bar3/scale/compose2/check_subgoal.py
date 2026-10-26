"""subgoal_c2's copies against bar3/scale/subgoal/score.py (COMPOSE2.md §2.3): the operations on a grid, and the table
view and the depth-2 law view on random rows and candidates, including candidates built to pass (laws of g, X and N).
The depth rule is checked on x^2 mod N's own rows: depth 1 on, depth 2 off.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose2.check_subgoal
"""

from __future__ import annotations

import numpy as np

from bar3.scale.compose2 import subgoal_c2 as SG
from bar3.scale.subgoal import score as SC


def main():
    vals = np.arange(0, 12)
    for name in SC.OPS2:
        for a in vals:
            for b in vals:
                assert int(SC.OPS2[name](np.array([a]), np.array([b]))[0]) == int(SG.OPS2[name](np.array([a]),
                                                                                                np.array([b]))[0])
    for name in SC.OPS1:
        assert all(int(SC.OPS1[name](np.array([a]))[0]) == int(SG.OPS1[name](np.array([a]))[0]) for a in vals)
    rng = np.random.default_rng(0)
    n_law = n_tab = 0
    for trial in range(300):
        n = int(rng.integers(8, 40))
        x = rng.integers(0, 30, n)
        N = rng.integers(2, 30, n)
        t = (x * x) % N if trial % 2 else rng.integers(0, 10, n)
        cap = np.full(n, 1000)
        r1 = SC.Rows([(int(a), int(b)) for a, b in zip(x, N)], x, N, t, cap, [])
        r2 = SG.Rows(r1.keys, x, N, t, cap, [])
        for g in (x * x, (x * x) + N, rng.integers(0, 5, n), x, t + 1, np.abs(x - N)):
            assert SC.table_view(g, r1) == SG.table_view(g, r2)
            l1 = SC._laws({"g": g, "X": x, "N": N}, r1, ("g",))
            l2 = SG.laws({"g": g, "X": x, "N": N}, r2, ("g",), 2)
            assert l1 == l2, (l1, l2)
            n_law += l1 is not None
            n_tab += SC.table_view(g, r1)[0] > 0
    x = np.array([a for N in range(2, 12) for a in range(N)])
    N = np.array([N for N in range(2, 12) for a in range(N)])
    rows = SG.Rows(list(zip(x, N)), x, N, (x * x) % N, np.full(len(x), 10 ** 4), [])
    assert SG.law_depths(rows) == [1], SG.law_depths(rows)
    assert SC.inputs_law(SC.Rows(rows.keys, x, N, rows.t, rows.cap, [])) is not None     # score.py: the view off
    assert SG.law_view(x * x, rows, [1]) == "mod(g,N)"
    assert SG.law_view(x, rows, [1]) is None
    print(f"subgoal_c2 agrees with score.py: operations on a 12x12 grid, 1,800 table views and depth-2 laws "
          f"({n_law} laws found, {n_tab} tables that save rows); on x^2 mod N the law view is on at depth 1 only, "
          f"and passes x^2 (mod(g,N)) and not x")


if __name__ == "__main__":
    main()
