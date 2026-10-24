"""How fast tdcm searches on a GPU against dsearch on a CPU core: a few digit-level problems at 30k and 1M states.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.bench_tdcm
"""
import time

import torch

from library import dsl
from bar3.bridge import dcm, dsearch, tdcm, dproblems as P
from bar3.bridge import slices as S

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = P.machine_for(S.SEARCH_BASES)
posed, _ = S.book(S.SEARCH_BASES + S.CHECK_BASES)
by = {s.name: s for s in posed}
for name in ("sub(A,R)", "acc.mul(A,R,H)"):
    c = P.from_slice(by[name], "target")
    ins, want = P.sample(c)
    prob = dsearch.Problem(ins, c.outputs, want)
    t0 = time.time()
    cls = dsearch.enumerate_items(dcm.prims(c.regs()), {}, dcm.loop_kinds(c.regs()), m, max_body=2)
    t_items = time.time() - t0
    for budget in (30_000, 1_000_000):
        t1 = time.time()
        r2 = tdcm.bfs(torch.tensor(prob.root.regs, device=dev), [dcm.RI[o] for o in c.outputs],
                      torch.tensor(want, device=dev), cls, {}, m.cap, m.limit, max_states=budget, time_limit=1200)
        if dev == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        r1 = dsearch.bfs(prob, cls, {}, m, max_states=budget, time_limit=1200)
        t3 = time.time()
        print(f"{name} {budget}: {len(cls)} classes ({t_items:.1f}s); torch on {dev} {t2 - t1:.1f}s {r2.level_sizes}; "
              f"numpy {t3 - t2:.1f}s {r1.level_sizes}; same {r1.level_sizes == r2.level_sizes}", flush=True)
