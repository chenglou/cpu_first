"""A local check (CPU, under a minute) of constants from the data (SP_BREADTH.md): on a small library (the routines
of the H1 round, an addition, a link, a subtraction), the search by calls finds the H1 round as a near miss of
y^2 + 1, y^2 + 3 and y^2 - 1, reads the constant off the answer rows, and the rest of the way completes the round,
ending with a reduction mod N; a reduction against a register holding 0 fails that rule."""
import time

import torch

from integration.sp_breadth import offline as O

lib = O.library("integration/sp_breadth/libs/h1_sp2_official.log")
want = {"T.mul.x,o.XY", "T.red.x+on.XA", "T.add.pair.AY", "T.red.pair.AN", "T.lt.pair.AY"}
keep = set()


def add(n):
    if n in keep:
        return
    keep.add(n)
    for tok in O.L.parse(lib[n]):
        for t in O.S.routines_in((tok,), {k: O.L.parse(v) for k, v in lib.items()}, closure=True) | (
                {tok} if isinstance(tok, str) and tok in lib else set()):
            add(t)


for n in want:
    add(n)
small = {k: v for k, v in lib.items() if k in keep or k.startswith("B")}   # with every merge
rows = O.load_rows(64, seed=1, T=1)
for task in ("rho", "plus3", "minus1", "h1"):
    rws = [dict(r, result=O.STEP[task](int(r["x"]), int(r["modulus"]))) for r in rows]
    ans = tuple(O.digits(rws, k) for k in ("modulus", "x", "result"))
    c = O.climb(small, macro_cost=2, macro_after=0, round_budget=200000, round_cost=1, constants=True, ends="red")
    t0 = time.time()
    progs, rec = c.search_round(ans, t=1)
    print(task, rec["found_cost"], rec["const"], rec["programs"][:1], f"{time.time() - t0:.0f}s", flush=True)
    assert progs and all(c.ends_ok(p[-1]) for p in progs), rec["stop"]

# the reductions that fooled the rule's first versions: against a register holding 0 (it does nothing), and a
# "reduction" by 1 whose dividend is routed from X (it subtracts 1)
rws = [dict(r, result=O.STEP["plus3"](int(r["x"]), int(r["modulus"]))) for r in rows]
ans = tuple(O.digits(rws, k) for k in ("modulus", "x", "result"))
c = O.climb(small, constants=True, ends="red")
k3 = torch.tensor([3] + [0] * 15, dtype=torch.uint8)
for text in ("LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA STA.S LDK STA.Y LDA.S T.add.pair.AY "
             "STA.S LDA.X STA.Y LDA.S T.red.pair.AY",):
    p = O.L.parse(text)
    wrapped = p[:-5] + (("REP", O.L.ONE, tuple(p[-5:])),)
    assert O.L.round_ok(p, c.lib, c.cells, ans, 16, "cpu", const=k3), "it does fit the answer rows"
    assert not c.round_ends(wrapped, c.lib, ans, k3), text
print("the fake reductions fail the rule")
