"""One practice search as the in-run climb runs it (INTEGRATION.md §3, s3h), from a recorded library: the library and
the solved tasks rebuilt from a run's sleep lines (lib/<LIB>_sleeps.txt, the levels before LEVEL), the data's rows and
exact cells, the stage-3 template's settings; then TASK's search with every apply_items call timed (rows, items, the
longest instruction sequence, the GPU synchronized), the slowest kinds of call summed. Not a lesson: a measurement.

    env: LIB (s3g5), LEVEL (4), TASKS (mac,step~), ROWS (4096), CALLS (1)
"""
import json
import os
import time
from collections import defaultdict

import torch

from integration.lib import libcore as L, inclimb

dev = "cuda" if torch.cuda.is_available() else "cpu"
here = os.path.dirname(os.path.abspath(__file__))
L.MAC_ADDEND = "A"
L.NEEDS.update(mul=("add", "mac"), reduce=("add", "modB"))
CLIMB = dict(budget=5_800_000, fallback_budget=27_000_000, max_body=3, max_cost=7, min_count=2.0, max_merges=12,
             levels=12, NP=8, round_budget=4_000_000, round_cost=5, comp_first=True, comp_body=3, skip_twins=True,
             stall_after=2, search_s=900, listing="goals", occam_steps=True, merged_routines=True,
             fuse=int(os.environ.get("FUSE", 1)))
level = int(os.environ.get("LEVEL", 4))
lines = open(os.path.join(here, f"{os.environ.get('LIB', 's3g5')}_sleeps.txt")).read().splitlines()[:level]
lib = {}
for line in lines:
    body = line.split("; ", 1)[1].rsplit(" (", 1)[0]
    for part in body.split("; "):
        if "=" in part:
            k, v = part.split("=", 1)
            lib[k.strip()] = L.parse(v.strip())
rows = inclimb.load_rows(int(os.environ.get("ROWS", 4096)), seed=0)
Nd, Xd, _ = (t.to(dev) for t in inclimb.digits_of_rows(rows))
c = L.Climb(L.Cells(10, dev), dev, 16, chunk=1 << 21, **CLIMB)
c.lib = dict(lib)
c.corpus = {k[2:]: [v] for k, v in lib.items() if k.startswith("T.")}
c.level = level
calls = defaultdict(lambda: [0, 0.0, 0, 0, 0, 0.0])
_apply = L.apply_items


def apply_timed(st, b, item_of_row, items, lib_):
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = _apply(st, b, item_of_row, items, lib_)
    if dev == "cuda":
        torch.cuda.synchronize()
    if hasattr(L, "_trace_of"):
        ent = L._traces(st, lib_)
        lens = [len(L._trace_of(st, it, lib_, 0, ent)[0]) for it in items]
    else:
        lens = [1] * len(items)
    mx = max(lens, default=0)
    top = {}
    for n in lens:
        j = n.bit_length()
        top[j] = max(top.get(j, 0), n)
    loop = any(L.is_loop(it) and it[1] not in lib_ for it in items)
    key = (f"P={st.P}", "loops" if loop else "straight", f"items~{1 << max(0, len(items) - 1).bit_length()}",
           f"rows~{1 << max(0, b.S - 1).bit_length()}", f"longest~{1 << max(0, mx - 1).bit_length()}")
    a = calls[key]
    a[0] += 1
    a[1] += time.time() - t0
    a[2] += b.S
    a[3] = max(a[3], mx)
    a[4] += sum(top.values())
    a[5] += sum(n * b.S / max(1, len(items)) for n in lens)
    return out


if os.environ.get("MODULE_FILE"):                    # another libcore (A/B timing), e.g. work/s3g/libcore.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("libcore_x", os.environ["MODULE_FILE"])
    L2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L2)
    L2.MAC_ADDEND, L2.NEEDS = L.MAC_ADDEND, L.NEEDS
    lib = {k: L2.parse(L.show(v)) for k, v in lib.items()}
    L = L2
    if "fuse" not in L.Climb.__init__.__code__.co_varnames:
        CLIMB.pop("fuse")
    c = L.Climb(L.Cells(10, dev), dev, 16, chunk=1 << 21, **CLIMB)
    c.lib = dict(lib)
    c.corpus = {k[2:]: [v] for k, v in lib.items() if k.startswith("T.")}
    c.level = level
    _apply = L.apply_items
parts_t = defaultdict(float)


def timed(name, fn):
    def wrap(*a, **k):
        if dev == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        try:
            return fn(*a, **k)
        finally:
            if dev == "cuda":
                torch.cuda.synchronize()
            parts_t[name] += time.time() - t1
    return wrap


if os.environ.get("CALLS", "1") == "1":
    L.apply_items = apply_timed
    L.Acceptor.many = timed("Acceptor.many", L.Acceptor.many)
    L.programs = timed("programs", L.programs)
    L.enumerate_items = timed("enumerate_items", L.enumerate_items)
    L.bfs = timed("bfs", L.bfs)
print("device", torch.cuda.get_device_name() if dev == "cuda" else "cpu", "level", level, "routines",
      sorted(k for k in lib if k.startswith("T.")), flush=True)
for task in os.environ.get("TASKS", "mac,step~").split(","):
    calls.clear()
    t0 = time.time()
    progs, rec = c.search(task, (Nd, Xd))
    print("SEARCH", json.dumps({k: rec[k] for k in ("task", "found_cost", "level_sizes", "seconds", "stop", "how",
                                                    "programs", "runs")}), flush=True)
    print(f"PARTS {task}: " + ", ".join(f"{k} {v:.1f}s" for k, v in sorted(parts_t.items(), key=lambda kv: -kv[1])),
          flush=True)
    parts_t.clear()
    tot = sum(v[1] for v in calls.values())
    print(f"CALLS {task}: {sum(v[0] for v in calls.values())} calls, {tot:.1f}s in apply_items", flush=True)
    for k, v in sorted(calls.items(), key=lambda kv: -kv[1][1])[:14]:
        print(f"   {' '.join(k)}: {v[0]} calls, {v[1]:.1f}s, {v[2]} rows, longest {v[3]}, lockstep steps {v[4]}, "
              f"row-instructions {v[5]:.3g}", flush=True)
