"""The lifted climb's reduce search, rerun as the climb ran it at level 5 (records/climb/lifted_b200_x4m: library
search, 4 times s3i's budgets), with what the climb does not record: the goal programs its acceptor rejected
(`aliases`). An argument sets bfs's max_goals (the goal states listed at the cost where goals appear, taken in the
order of their hash; default 50,000, the climb's).

    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/bridge/modal_climb_b.py --script bar3.bridge.checks.reduce_search --tag reduce_search [--args 100000000]
"""
import sys
import time

import torch

from bar3.bridge import climb_b as CB, libcore_b as L
from integration.lib import inclimb

LIB4 = {
 "B0.1": "SUB.N INC",
 "B0.2": "P[B0.1]",
 "B0.3": "CLC B0.2",
 "B0.4": "LT.N INC",
 "B0.5": "P[B0.4]",
 "B0.6": "CLC B0.5",
 "B0.7": "LDA.N CLC",
 "B0.8": "CLC LDA.N",
 "T.lt": "CLC LDA.X B0.5",
 "T.lt~": "CLC LDA.Y B0.5",
 "T.modx": "LDA.X B0.3 P[SELC.X INC]",
 "T.modx~": "LDA.Y B0.3 P[SELC.Y INC]",
 "B2.1": "LDA.N CLC",
 "B2.2": "CLC LDA.N",
 "B2.3": "ADD.N INC",
 "B2.4": "P[B2.3]",
 "B2.5": "CLC B2.4",
 "T.add": "B2.2 P[ADD.X INC]",
 "T.add~": "B2.2 P[ADD.Y INC]",
 "T.modB": "B[T.modx STA.X]",
 "T.modB~": "B[T.modx~ STA.Y]",
 "T.mrow": "LDH.Y B[INC RMULD.X]",
 "T.mrow~": "LDH.X B[INC RMULD.Y]",
 "T.mac": "LDH.Y CLK P[MAC.X INC]",
 "T.mac~": "LDH.X CLK P[MAC.Y INC]",
 "T.step": "LDA.Y BRG.X STA.X T.modB"
}

dev = "cuda" if torch.cuda.is_available() else "cpu"
cells, _ = CB.build_views(CB.SETS["lifted"]["cells"])
L.set_instructions(cells, sel=False)
L.MAC_ADDEND = "A"
L.NEEDS.update(mul=("add", "mac"), reduce=("add", "modB"))
CB.strong_modx()
rows = tuple(t.to(dev) for t in inclimb.digits_of_rows(inclimb.load_rows(4096, seed=0))[:2])
cfg = dict(CB.CLIMB, **{k: int(CB.CLIMB[k] * 4) for k in ("budget", "fallback_budget", "round_budget", "search_s")})
climb = L.Climb(L.Cells(10, dev), dev, 16, chunk=1 << 21, **cfg)
for k, v in LIB4.items():
    climb.lib[k] = L.parse(v)
for t in ("add", "add~", "modB", "modB~", "modx", "modx~", "lt", "lt~"):
    climb.corpus[t] = [L.parse("T." + t)]
helpers = climb.helpers_for("reduce")
prob = L.LawProblem("reduce", climb.settings, climb.NP, 0, rows, helpers, climb.lib, dev)
acc = L.Acceptor("reduce", climb.lib, helpers, rows, climb.verify, dev)
want = L.parse("CLA P[BRG.X STA.Y T.modB~]")
outs = []
for st, root in zip(climb.settings, prob.roots):
    b = root.copy()
    L.run_items(st, b, want, climb.lib)
    outs.append(b)
wkey = int(L.keys_of(outs)[0])
print("CLA P[BRG.X STA.Y T.modB~]: goal", bool(prob.goal(outs)[0]), "acceptor", acc(want), flush=True)
classes = climb.classes()
seen = []
orig_many = acc.many


def many(progs):
    ok = orig_many(progs)
    seen.extend(zip([L.show(p) for p in progs], ok))
    return ok


t0 = time.time()
MAX_GOALS = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000      # bfs's default, which the climb uses
res = L.bfs(prob, classes, climb.lib, max_cost=climb.max_cost, max_states=climb.budget, accept=many, listing="goals",
            chunk=climb.chunk, fuse=climb.fuse, max_goals=MAX_GOALS)
print("max_goals", MAX_GOALS, flush=True)
print(f"bfs: found {res.found_cost}, levels {res.level_sizes}, stop {res.stop!r}, {time.time() - t0:.0f}s", flush=True)
print("accepted:", [L.show(p) for p in (res.accepted or [])][:8], flush=True)
print("goal programs tested:", len(seen), "accepted", sum(o for _, o in seen), flush=True)
for p, o in seen[:12] + [x for x in seen if x[1]][:8]:
    print("   ", "ok " if o else "no ", p, flush=True)
print("aliases (level, first rejected):", [(d, a[:6]) for d, a in res.aliases][:4], flush=True)
