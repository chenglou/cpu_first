"""tdcm (torch) against dcm/dsearch (numpy): the same items applied to random machines give the same registers and
dead flags, and the same searches (the same item classes) give the same level sizes, best fits, found cost and
programs. Local (CPU torch), about a minute.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.test_tdcm
"""
import time

import numpy as np
import torch

from library import dsl
from bar3.bridge import dcm, dsearch, tdcm, dproblems as P, dlearner as DL, djobs
from bar3.bridge import slices as S

t0 = time.time()
m = P.machine_for(S.SEARCH_BASES)
lib = {"T.w": dsl.parse("INC.A MOV.T.N L.A[DEC.T] Z.T[CLR.A INC.C]"), "T.v": dsl.parse("Z.A[MOV.A.N INC.C] DEC.A")}
regs6 = ("A", "R", "C", "H", "N", "T")
toks, kinds = djobs.vocabulary({k: dsl.show(v) for k, v in lib.items()}, regs6)
classes = dsearch.enumerate_items(toks, lib, kinds, m, max_body=2)
print(f"{len(classes)} item classes ({time.time() - t0:.1f}s)")
rng = np.random.default_rng(3)
b = dcm.random_states(64, 6, rng, top=12)
bad = 0
for cl in classes[:: max(1, len(classes) // 3000)]:
    ref = dcm.apply(m, b, cl.rep, lib)
    got, dead = tdcm.apply(m.cap, m.limit, torch.tensor(b.regs), (cl.rep,), lib)
    ok = np.array_equal(ref.dead, dead.numpy()) and np.array_equal(ref.regs[~ref.dead], got.numpy()[~ref.dead])
    bad += not ok
print(f"items on random machines: {bad} differ ({time.time() - t0:.1f}s)")
assert bad == 0
posed, _ = S.book(S.SEARCH_BASES + S.CHECK_BASES)
by = {s.name: s for s in posed}
for name, budget in (("select(C,A,R)", 30_000), ("sub(A,R)", 30_000), ("add(A,R)", 200_000)):
    c = P.from_slice(by[name], "target")
    ins, want = P.sample(c)
    prob = dsearch.Problem(ins, c.outputs, want)
    cls = dsearch.enumerate_items(dcm.prims(c.regs()) + ["T.w", "T.v"], lib, dcm.loop_kinds(c.regs()), m, max_body=2)
    r1 = dsearch.bfs(prob, cls, lib, m, max_states=budget, time_limit=600)
    root = torch.tensor(prob.root.regs)
    r2 = tdcm.bfs(root, [dcm.RI[o] for o in c.outputs], torch.tensor(want), cls, lib, m.cap, m.limit, max_states=budget,
                  time_limit=600)
    same = (r1.level_sizes == r2.level_sizes and r1.best_fit == r2.best_fit and r1.found_cost == r2.found_cost)
    p1 = sorted(dsl.show(p) for p in dsearch.programs(r1, cls)) if r1.found_cost is not None else []
    p2 = sorted(dsl.show(p) for p in tdcm.programs(r2, cls)) if r2.found_cost is not None else []
    print(f"{name}: numpy {r1.level_sizes} {r1.found_cost} {r1.seconds:.1f}s | torch {r2.level_sizes} "
          f"{r2.found_cost} {r2.seconds:.1f}s | programs {len(p1)} vs {len(p2)}: "
          f"{'same' if same and p1 == p2 else 'DIFFERENT'}")
    assert same and p1 == p2, name
print(f"all checks passed ({time.time() - t0:.1f}s)")
