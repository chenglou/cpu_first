"""A local check (CPU, about a minute) of the every-T check (SP_BREADTH.md): y^2 - 1's round as g2 accepted it (the
H1 round, subtract 1, then a reduction) is right on every T = 1 row but wrong when a round's input is 0, which the
training's T = 4 rows reach; deep_ok rejects it on those rows, and the search, with 3 calls for the rest of the way,
finds a round right on both."""
import json
import os
import time

from integration.sp_breadth import offline as O
from integration.sp_breadth import libb as L

lib = O.library("integration/sp_breadth/libs/h1_sp2_official.log")
keep = set()


def add(n):
    if n in keep:
        return
    keep.add(n)
    for tok in L.parse(lib[n]):
        for t in O.S.routines_in((tok,), {k: L.parse(v) for k, v in lib.items()}, closure=True) | (
                {tok} if isinstance(tok, str) and tok in lib else set()):
            add(t)


for n in ("T.mul.x,o.XY", "T.red.x+on.XA", "T.add.pair.AN", "T.red.pair.AN", "T.lt.pair.AY", "T.red.pair.AY"):
    add(n)
small = {k: v for k, v in lib.items() if k in keep or k.startswith("B")}
step = O.STEP["minus1"]


def run(x, n, t):
    for _ in range(t):
        x = step(x, n)
    return x


rows = O.load_rows(64, seed=1, T=1)
ans = tuple(O.digits([dict(r, result=step(int(r["x"]), int(r["modulus"]))) for r in rows], k)
            for k in ("modulus", "x", "result"))
deep = [r for r in (json.loads(l) for l in open(os.path.join(O.H1_DIR if hasattr(O, "H1_DIR") else
                                                             O.load_rows.__globals__["H1_DIR"], "train.jsonl")))
        if r["time_steps"] == 4][:2048]
drows = [dict(r, result=run(int(r["x"]), int(r["modulus"]), 4)) for r in deep]
dans = tuple(O.digits(drows, k) for k in ("modulus", "x", "result"))
c = O.climb(small, macro_cost=2, macro_after=0, round_budget=200000, round_cost=1, constants=True, ends="red",
            rest_cost=3)
c.deep = [(4, dans)]
g2 = L.parse("LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA STA.S LDK STA.Y LDA.S T.lt.pair.AY STA.S LDA.N "
             "STA.Y LDA.S T.red.pair.AY")
one = L.digits_of = None
import torch
k1 = torch.tensor([1] + [0] * 15, dtype=torch.uint8)
print("g2's round on T = 1:", L.round_ok(g2, c.lib, c.cells, ans, 16, "cpu", const=k1),
      "| on T = 4:", c.deep_ok(g2, c.lib, k1))
t0 = time.time()
progs, rec = c.search_round(ans, t=1)
print("search:", rec["found_cost"], rec["const"], rec["programs"][:1], f"{time.time() - t0:.0f}s")
assert progs and all(c.deep_ok(p, c.lib, k1) for p in progs[:4]), rec["stop"]
