"""Where the in-run climb's time goes (INTEGRATION.md §3, s3h): libcore.Climb with the stage-3 template's settings (the
A/B's variant C rules, mac = A + x d, the pieces first), alone on the data's rows and exact cells, every unit timed and
the time split over the search's parts (each part's wall time, the GPU synchronized at its edges, inclusive of the
parts inside it). Not a lesson: a measurement.

    env: SECONDS (3600: the climb's own limit), ROWS (4096), STOP_AT_ROUND (1), SYNC (1: synchronize at the parts'
         edges, for the split; 0: the climb as it runs), MODULE_FILE (a libcore file to load instead, for A/B timing)
"""
import importlib.util
import json
import os
import time
from collections import defaultdict

import torch

from integration.lib import inclimb

dev = "cuda" if torch.cuda.is_available() else "cpu"
if os.environ.get("MODULE_FILE"):
    spec = importlib.util.spec_from_file_location("libcore_x", os.environ["MODULE_FILE"])
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
else:
    from integration.lib import libcore as L
SYNC = os.environ.get("SYNC", "1") == "1"
T = defaultdict(float)
N = defaultdict(int)
depth = [0]


def timed(name, fn):
    def wrap(*a, **k):
        if SYNC and dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        depth[0] += 1
        try:
            return fn(*a, **k)
        finally:
            depth[0] -= 1
            if SYNC and dev == "cuda":
                torch.cuda.synchronize()
            T[name] += time.time() - t0
            N[name] += 1
    return wrap


_apply = L.apply_items


def apply_split(st, b, item_of_row, items, lib):
    loop = any(L.is_loop(it) and it[1] not in lib for it in items)
    return timed("apply_items[loops]" if loop else "apply_items[straight]", _apply)(st, b, item_of_row, items, lib)


L.apply_items = apply_split
for name in ("enumerate_items", "_signatures", "bfs", "keys_of", "_fresh", "programs", "bpe", "run_items"):
    if name == "run_items":
        continue                                     # too fine-grained: called recursively
    setattr(L, name, timed(name, getattr(L, name)))
L.Problem.goal = timed("goal", L.Problem.goal)
L.Acceptor.many = timed("Acceptor.many", L.Acceptor.many)
L.LawProblem.__init__ = timed("LawProblem", L.LawProblem.__init__)
L.MAC_ADDEND = "A"
L.NEEDS.update(mul=("add", "mac"), reduce=("add", "modB"))
CLIMB = dict(budget=5_800_000, fallback_budget=27_000_000, max_body=3, max_cost=7, min_count=2.0, max_merges=12,
             levels=12, NP=8, round_budget=4_000_000, round_cost=5, comp_first=True, comp_body=3, skip_twins=True,
             stall_after=2, search_s=900, listing="goals", occam_steps=True, merged_routines=True,
             fuse=int(os.environ.get("FUSE", 1)))
rows = inclimb.load_rows(int(os.environ.get("ROWS", 4096)), seed=0)
Nd, Xd, _ = (t.to(dev) for t in inclimb.digits_of_rows(rows))
ans = tuple(t.to(dev) for t in inclimb.digits_of_rows(inclimb.load_rows(400, seed=1, T=1)))
c = L.Climb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 18, **CLIMB)
t0 = time.time()
deadline = t0 + float(os.environ.get("SECONDS", 3600))
print("device", torch.cuda.get_device_name() if dev == "cuda" else "cpu", "sync", SYNC, "libcore",
      os.environ.get("MODULE_FILE", "integration/lib/libcore.py"), flush=True)
units = []
while True:
    t1 = time.time()
    before = dict(T)
    line = c.step((Nd, Xd), answers=ans, deadline=deadline)
    if line is None:
        break
    dt = time.time() - t1
    split = {k: round(T[k] - before.get(k, 0.0), 1) for k in T if T[k] - before.get(k, 0.0) > 0.05}
    units.append({"s": round(dt, 1), "line": line[:160], "split": split})
    mem = (f" | peak GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB, reserved {torch.cuda.memory_reserved() / 2 ** 30:.1f} GiB"
           if dev == "cuda" else "")
    print(f"UNIT {time.time() - t0:7.1f}s +{dt:6.1f}s {line[:150]} | {split}{mem}", flush=True)
    if time.time() > deadline or (c.rounds and os.environ.get("STOP_AT_ROUND", "1") == "1"):
        break
total = time.time() - t0
print("TOTAL", round(total, 1), "s; rounds", [L.show(p) for p in c.rounds][:4], flush=True)
print("SPLIT", json.dumps({k: [round(v, 1), N[k]] for k, v in sorted(T.items(), key=lambda kv: -kv[1])}), flush=True)
print("RECORDS", json.dumps([{k: r[k] for k in ("task", "level", "found_cost", "level_sizes", "programs", "stop")}
                             for r in c.records]), flush=True)
print("LIB", json.dumps({k: L.show(v) for k, v in c.lib.items()}), flush=True)
