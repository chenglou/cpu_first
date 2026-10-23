"""libcore (the consolidated, numpy-free core the stage-3 file inlines) against the tested modules: the same classes,
the same search, the same BPE merges, the same one-hot machine. Local, small budgets (well under 1 GB).

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m integration.lib.test_libcore
"""
import json
import os

import torch

from library import dsl as pdsl, merge, search
from integration.lib import libcore as L, tdsl, tfast, dslste, inclimb
from integration.lib.test_tdsl import to_torch_problem

sets = [pdsl.Setting(2, 3), pdsl.Setting(3, 2)]
st_t = [tdsl.Setting(2, 3, tdsl.Cells(2)), tdsl.Setting(3, 2, tdsl.Cells(3))]
st_l = [L.Setting(2, 3, L.Cells(2)), L.Setting(3, 2, L.Cells(3))]
c1 = tfast.enumerate_items(list(pdsl.PRIMS), {}, st_t)
c2 = L.enumerate_items(list(L.PRIMS), {}, st_l, "cpu")
print("classes same:", [pdsl.show((a.rep,)) for a in c1] == [L.show((b.rep,)) for b in c2], len(c1), len(c2))
prob = search.Problem("shl", sets, NP=8, seed=0)
tp = to_torch_problem(prob, st_t)
lp = L.Problem("shl", st_l, [L.Batch(*(getattr(b, f) for f in ("regs", "c", "h", "k", "p"))) for b in tp.roots],
               tp.targets)
r1 = tfast.bfs(tp, c1, {}, max_cost=4, max_states=60_000)
r2 = L.bfs(lp, c2, {}, max_cost=4, max_states=60_000)
print("bfs:", r1.found_cost, r2.found_cost, r1.level_sizes, r2.level_sizes,
      sorted(pdsl.show(p) for p in tdsl.programs(r1, c1)) == sorted(L.show(p) for p in L.programs(r2, c2)))
here = os.path.dirname(os.path.abspath(__file__))
lib_txt = json.load(open(os.path.join(here, "r11_lib.json")))["lib"]
corpus = {"a": [pdsl.parse("LDA.X CLC P[SBB.N INC]")], "b": [pdsl.parse("LDA.Y CLC P[SBB.N INC]")],
          "c": [pdsl.parse("LDA.X STA.S CLC P[SBB.N INC] SEL.S")]}
o1 = merge.bpe(corpus, {}, min_count=2.0, tiebreak="largest")
o2 = L.bpe({k: [L.parse(pdsl.show(p)) for p in v] for k, v in corpus.items()}, {}, min_count=2.0)
print("bpe same:", {k: pdsl.show(v) for k, v in o1[1].items()} == {k: L.show(v) for k, v in o2[1].items()},
      {k: L.show(v) for k, v in o2[1].items()})
lib = {k: L.parse(v) for k, v in lib_txt.items()}
rows = inclimb.load_rows(16, seed=3, T=1)
Nd, Xd, Yd = inclimb.digits_of_rows(rows)
cells = L.Cells(10)
print("round fit of T.sqmod:", L.round_fit(lib["T.sqmod"], lib, cells, (Nd, Xd, Yd), 16, "cpu"))
SBB, ADC, MUL = dslste.tables_onehot(tdsl.Cells(10))
m = L.SteMachine(SBB, ADC, MUL, 16, damp=lambda t: t)
regs, c, h, k = dslste.start(16, 16, {"X": Xd, "N": Nd})
st_ = m.run(list(regs.unbind(1)), c, h, k, lib["T.sqmod"], lib)
print("one-hot machine vs answers:", float((st_[0][0].argmax(-1)[:, :8] == Yd).all(1).float().mean()))
rows2 = inclimb.load_rows(500, seed=0)
Nd2, Xd2, _ = inclimb.digits_of_rows(rows2)
cl = L.Climb(L.Cells(10), "cpu", 16, budget=60_000, fallback_budget=60_000, max_body=2, max_cost=5, levels=1,
             chunk=1 << 15)
cl.tasks = ["lt", "lt~"]
for _ in range(3):
    print("climb:", cl.step((Nd2, Xd2)))
