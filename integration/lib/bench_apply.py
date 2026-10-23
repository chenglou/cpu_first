"""apply_items' speed on the call shapes a cost-6 practice search makes (s3h): about 2M rows of (item, state) pairs,
with thousands, hundreds or a handful of distinct loop items (the primitives-alone vocabulary, bodies up to cost 3), at
the (10, 2) setting with 8 probes; the lockstep and the item-by-item machine, each timed (GPU synchronized) and checked
equal. Not a lesson: a measurement.

    env: ROWS (2097152), REPS (3)
"""
import os
import time

import torch

from integration.lib import libcore as L

dev = "cuda" if torch.cuda.is_available() else "cpu"
cells = L.Cells(10, dev)
sets = [L.Setting(10, 1, cells), L.Setting(10, 2, cells)]
st = sets[1]
gen = L.gen_on(dev, 5)
rows = int(os.environ.get("ROWS", 1 << 21))
reps = int(os.environ.get("REPS", 3))
t0 = time.time()
cls = L.enumerate_items(list(L.PRIMS), {}, sets, dev, max_body=3)
by_cost = {}
for c in cls:
    by_cost.setdefault(c.cost, []).append(c.rep)
print("device", torch.cuda.get_device_name() if dev == "cuda" else "cpu", "classes", len(cls),
      {c: len(v) for c, v in sorted(by_cost.items())}, f"{time.time() - t0:.1f}s", flush=True)


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


def bench(name, items, st=st):
    k = len(items)
    ns = max(1, rows // k)
    src = L.random_states(st, ns, 8, gen, dev)
    ri = torch.arange(k, device=dev).repeat_interleave(ns)
    rs = torch.arange(ns, device=dev).repeat(k)
    b = src.take(rs)
    lib = {}
    lens = []
    for it in items:
        ops = []
        L.trace(st, (it,), lib, ops)
        lens.append(len(ops))
    out = {}
    for how in ("lockstep", "grouped", "auto"):
        saved = L.LOCK_COST
        if how == "lockstep":
            L.LOCK_COST = -10 ** 9
        fn = L._apply_grouped if how == "grouped" else L.apply_items
        try:
            fn(st, b, ri, items, lib)
            sync()
            t1 = time.time()
            for _ in range(reps):
                r = fn(st, b, ri, items, lib)
            sync()
        finally:
            L.LOCK_COST = saved
        dt = (time.time() - t1) / reps
        out[how] = r
        print(f"{name} {st}: {k} items x {ns} states = {k * ns} rows, {how}: {dt * 1000:.1f} ms, "
              f"{k * ns / dt / 1e6:.1f}M rows/s (sequences {min(lens)}-{max(lens)}, mean {sum(lens) / k:.1f})",
              flush=True)
    same = all(bool(torch.equal(getattr(out["lockstep"], f), getattr(out["grouped"], f)))
               for f in ("regs", "c", "h", "k", "p"))
    print(f"   same: {same}", flush=True)


g2 = torch.Generator().manual_seed(1)
pick = lambda v, n: [v[i] for i in torch.randperm(len(v), generator=g2)[:n].tolist()]
for s_ in sets:
    bench("cost-4 loops (level 2 sources)", pick(by_cost[4], 2600), s_)
    bench("cost-3 loops (level 3 sources)", pick(by_cost[3], 106), s_)
    bench("cost-2 loops (level 4 sources)", pick(by_cost[2], 4), s_)
    bench("cost-2 loops, one", pick(by_cost[2], 1), s_)
    bench("tokens, one", by_cost[1][:1], s_)
if dev == "cuda":
    from torch.profiler import ProfilerActivity, profile
    items = pick(by_cost[4], 2600)
    k = len(items)
    ns = rows // k
    b = L.random_states(st, ns, 8, gen, dev).take(torch.arange(ns, device=dev).repeat(k))
    ri = torch.arange(k, device=dev).repeat_interleave(ns)
    L.LOCK_COST = -10 ** 9
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        L.apply_items(st, b, ri, items, {})
        sync()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)
