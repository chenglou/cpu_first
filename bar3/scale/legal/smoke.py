"""Local checks of the full-size target mode (LEGAL_TEST.md §3), tiny and on the CPU: under a minute, well under 1 GB.

1. x² by the learner's own multiply (`LDA.X STA.Y T.mul.x,o.XY`) on every T = 1 row is x² (checked with Python's
   integers), and clean;
2. the law view by routines on handed values (x², x² + N, 2x, x² + 1; the values handed with Python's integers, the
   test itself the learner's routines) on the harness's 400 answer rows and on all 1,000 T = 1 rows: x² and x² + N
   pass on the rows where the chain is exact, the others do not;
3. the rest's acceptor on the 400 rows, X = x²: s4_x2's own reduction from x² passes, the chain alone does not;
4. the round's held states on 16 rows to cost 2 (the Kept goal keeps the last level), and a state's program;
5. a tiny rest search at full size: it runs.

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 ~/github/sotaku/venv/bin/python -m bar3.scale.legal.smoke
"""

from __future__ import annotations

import time

import torch

from integration.lib.libcore import _OneGoal, parse, programs, show
from bar3.scale.legal import offline as O
from bar3.scale.legal.rowtarget import P, RowTarget, answer_rows, h1_rows, routine_for
from bar3.scale.slice.spcore_s import _Res, bfs_roots


def handed(rt, f):
    """Digits [S = 1, P, n] of a value handed with Python's integers (a check's input, never a target)."""
    xs, Ns = O.ints(rt.Xd).tolist(), O.ints(rt.Nd).tolist()
    vals = [f(x, N) for x, N in zip(xs, Ns)]
    d = torch.tensor([[int(ch) for ch in reversed(str(v))] + [0] * (P - len(str(v))) for v in vals])
    return d.T.to(torch.uint8)[None]


def main():
    t0 = time.time()
    txt, ops = O.library("full")
    lib = {k: parse(v) for k, v in txt.items()}
    chain, compare = routine_for(ops, "red", O.REDUCTIONS), routine_for(ops, "lt", ("pair",))
    print("library", len(lib), "chain", chain, "compare", compare)
    all_rows, answers = h1_rows(O.H1_DIR), answer_rows(O.H1_DIR, 400, 1)
    for name, rows in (("400 answer rows", answers), ("all 1,000 T = 1 rows", all_rows)):
        rt = RowTarget(rows, "cpu")
        v = rt.value(parse(O.X2), lib)
        print(f"1. {name}: x² by the learner's multiply is x²: {O.check_x2(v.T.to(torch.uint8), rt)}; "
              f"clean: {rt.clean(parse(O.X2), lib)}")
        G = torch.cat([handed(rt, f) for f in (lambda x, N: x * x, lambda x, N: x * x + N, lambda x, N: 2 * x,
                                                lambda x, N: x * x + 1)])
        n_exact, ok = rt.law_mod(G, lib, chain, compare)
        print(f"2. {name}: x², x² + N, 2x, x² + 1: exact rows {n_exact.tolist()}, pass {ok.tolist()}")
    rt = RowTarget(answers, "cpu")
    acc = rt.rest_acceptor(rt.value(parse(O.X2), lib))
    progs = [parse("LDA.N P[BRG.X T.red.Bx+d.AN]"), parse("T.red.Bx+d.AN")]
    print(f"3. the rest's acceptor, s4_x2's reduction / the chain alone: {acc(progs, lib)}")
    small = RowTarget(answers[:16], "cpu")
    classes, sizes, stop, levels, parents = small.held(lib, max_cost=2, chunk=1 << 12)
    i = levels[2][0].S - 1
    pr = programs(_OneGoal(_Res(parents, 2, [i]), i), classes, limit=1)
    print(f"4. held on 16 rows to cost 2: sizes {sizes}, stop {stop!r}; level 2 state {i}: {show(pr[0])}")
    v = small.value(parse(O.X2), lib)
    root, goal = small.rest_root(v, torch.arange(8))
    cls = small.comp_classes(lib, max_body=1, chunk=1 << 16)
    t1 = time.time()
    res, sz, stops, *_ = bfs_roots([[root]], [small.st], [(0, goal)], [small.rest_acceptor(v)], cls, lib, max_cost=7,
                                   max_states=20_000, deadline=t1 + 30, chunk=1 << 13)
    print(f"5. a tiny rest search ({len(cls)} classes, 20,000 states): {res[0]['found_cost']}, sizes {sz[0]}, "
          f"{stops[0]!r}, {time.time() - t1:.0f} s")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
