"""Diagnostic (INTEGRATION.md §2): the reduction's search with the c5 climbs' level-4 vocabulary (rebuilt from their logged
merges), on the reduce problem, with acceptance, as the climb runs it; then the same goal states listed one spelling
per goal state. Not a lesson: a reproduction of a stall.

    env: ROWS (500; the c5 climbs read 4000), MAX_BODY (3), MAX_STATES (150000; the climbs' cap 3000000), MAX_COST (4)
"""
import os
import time

import torch

from integration.lib import libcore as L, inclimb

dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    torch.set_num_threads(4)
ROWS, MAX_BODY = int(os.environ.get("ROWS", 500)), int(os.environ.get("MAX_BODY", 3))
MAX_STATES, MAX_COST = int(os.environ.get("MAX_STATES", 150_000)), int(os.environ.get("MAX_COST", 4))
rows = inclimb.load_rows(ROWS, seed=0)
Nd, Xd, _ = (t.to(dev) for t in inclimb.digits_of_rows(rows))
cells = L.Cells(10, dev)
txt = {"B0.1": "SBB.N INC", "B0.2": "P[B0.1]", "B0.3": "CLC B0.2", "T.lt": "CLC P[LDA.X B0.1]", "T.lt~": "CLC LDA.Y B0.2",
       "B1.1": "STA.S B0.3", "B1.2": "B1.1 SEL.S", "T.modx": "LDA.X B1.2", "T.modx~": "LDA.Y B1.2",
       "B2.1": "LDA.N CLC", "B2.2": "CLC LDA.N", "B2.3": "ADC.N INC", "B2.4": "P[B2.3]", "B2.5": "CLC B2.4",
       "T.add": "B2.2 P[ADC.X INC]", "T.add~": "B2.2 P[ADC.Y INC]",
       "B3.1": "B[B1.2]", "B3.2": "CLK CLA", "B3.3": "CLA CLK", "T.modB": "LDA.X B3.1", "T.modB~": "LDA.Y B3.1",
       "T.mrow": "LDH.Y B3.2 P[MUL.X INC]", "T.mrow~": "LDH.X B3.2 P[MUL.Y INC]"}
lib = {k: L.parse(v) for k, v in txt.items()}
settings = [L.Setting(10, 1, cells), L.Setting(10, 2, cells)]
t0 = time.time()
classes = L.enumerate_items(list(L.PRIMS) + list(lib), lib, settings, dev, max_body=MAX_BODY)
print("rows", ROWS, "device", dev, "classes", len(classes), f"{time.time() - t0:.1f}s", flush=True)
prob = L.LawProblem("reduce", settings, 8, 0, (Nd, Xd), {"add": lib["T.add"]}, lib, dev)
verify = [L.Setting(10, W, cells) for W in (1, 2, 3)]
from integration.lib import climb_ab as AB
ok = AB.Acceptor("reduce", lib, {"add": lib["T.add"]}, (Nd, Xd), verify, dev)      # the same probes as libcore.accept
_programs = L.programs


def programs(res, classes, limit=200):                 # libcore's listing, recording how many goal states it covers
    out = _programs(res, classes, limit)
    covered, n = 0, 0
    for g in res.goals:
        if n >= limit:
            break
        n += len(AB.spellings(res, g, classes, limit))
        covered += 1
    print("PROGRAMS", {"cost": res.found_cost, "goal_states": len(res.goals), "listed": len(out),
                       "goal_states_the_list_covers": covered}, flush=True)
    return out


L.programs = programs
t0 = time.time()
res = L.bfs(prob, classes, lib, max_cost=MAX_COST, max_states=MAX_STATES, chunk=1 << 21, accept=lambda ps: [ok(p) for p in ps])
L.programs = _programs
print("AS THE CLIMB: found", res.found_cost, res.level_sizes, res.stop, f"{time.time() - t0:.1f}s",
      "aliases", [(d, len(a)) for d, a in res.aliases], "tested", len(ok.cache), flush=True)
print("accepted:", [L.show(p) for p in (res.accepted or [])][:12], flush=True)
t0, n0, glog = time.time(), len(ok.cache), []
with AB.every_goal_state(glog) as (orig, goal_of):
    res = L.bfs(prob, classes, lib, max_cost=MAX_COST, max_states=MAX_STATES, chunk=1 << 21,
                accept=lambda ps: [ok(p) for p in ps])
    acc = AB.more_spellings(res, res.accepted, classes, ok, orig, goal_of) if res.accepted else []
print("ONE PER GOAL STATE: found", res.found_cost, res.level_sizes, res.stop, f"{time.time() - t0:.1f}s", "listings", glog,
      "newly tested", len(ok.cache) - n0, flush=True)
print("accepted:", [L.show(p) for p in (res.accepted or [])][:12], "with more spellings:", [L.show(p) for p in acc][:12],
      flush=True)
