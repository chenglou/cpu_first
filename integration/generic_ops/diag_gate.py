"""Why a law gets no deep probe (GENERIC_OPS.md §4): a tiny-core record's library at the start of a level, rebuilt, and
that level's ordinary probe (the composition vocabulary at the probe budget, as SelfClimb.probe runs it) on the roots
of one pattern; for each law, the cost found, the aliases (programs the probes accept and acceptance rejects) and the
closeness two ways: SelfClimb's (0 for a law with aliases) and over the held states that are not goal states.

    env: REC (a record in diag/ or records/), ISA, LEVEL, PATTERN (x+on), BINDINGS (XA,XY), LAWS (red,div), PROBE (1e6)
    cd cpu_first && PYTHONPATH=. ISA=if,k REC=g_if_k LEVEL=4 ~/github/sotaku/venv/bin/python -m integration.generic_ops.diag_gate
"""
import json
import os
import time
from pathlib import Path

import torch

from integration.generic_ops import isa as I
from integration.generic_ops.core_g import digits, load_rows

env = os.environ.get
L, S = I.load(env("ISA", "if,k"))
HERE = Path(__file__).resolve().parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
name = env("REC", "g_if_k")
path = next(p for p in (HERE / "diag" / f"{name}.json", HERE / "records" / f"{name}.json") if p.exists())
rec = json.load(open(path))
level = int(env("LEVEL", 4))
lib = {k: L.parse(v) for k, v in rec["history"][level - 1]["lib"].items()}
c = S.SelfClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16, rule="lpca", K=4, NP=8,
                deep_by="bindings", deep_focus="closure", fuse=4 if dev == "cuda" else 1,
                verify=tuple((10, int(w)) for w in env("VERIFY", "1,2,3").split(",")),
                pure_first={"0": False, "1": True}.get(env("PURE_FIRST", "0"), env("PURE_FIRST", "0")))
c.lib = lib
for key, pairs in rec["ops"].items():                       # the routines the learner could call, in practice order
    law, pat = key.split("@")
    for tok, bd in pairs:
        if tok in lib:
            c.ops.add(law, pat, tok, tuple(bd))
rows = load_rows(4096, seed=0)
Nd, Xd = digits(rows, "modulus").to(dev), digits(rows, "x").to(dev)
pat = env("PATTERN", "x+on")
laws = env("LAWS", "red,div").split(",")
roots = []
for bd in env("BINDINGS", "XA,XY").split(","):
    r = S.Root(pat, tuple(bd), c.settings, c.NP, (Nd, Xd), c.ops, c.lib, dev, c.seed)
    r.goal_list = r.goals(c.ops, c.lib, laws)
    roots.append(r)
t0 = time.time()
comp = c.comp_classes()
print(f"level {level}: {len(lib)} names, {len(c.routines())} routines; composition vocabulary {len(comp)} items "
      f"({time.time() - t0:.0f}s)", flush=True)
goals, accs = [], []
for ri, r in enumerate(roots):
    for g in r.goal_list:
        goals.append((ri, g))
        accs.append(S.Acceptor(g.name, r, c.verify, (Nd, Xd), c.ops, c.lib, dev).many)
pure = None
if c.pure_first == "scratch":
    pure = [lambda q, r=roots[ri]: S.scratch(q, c.lib, c.settings[-1], r.roots[-1], r.binding) == 0 for ri, _ in goals]
res, sizes, stops, levels, rids, parents = S.bfs_roots([r.roots for r in roots], c.settings, goals, accs, comp, c.lib,
                                                       max_cost=7, max_states=int(float(env("PROBE", 1e6))),
                                                       chunk=c.chunk, fuse=c.fuse, cleans=pure)
prog = L.parse(env("PROG", "STA.Y P[BRG.X T.red.x+dn.AY]"))
for ri, r in enumerate(roots):
    for g, a in zip(r.goal_list, [x for (j, _), x in zip(goals, accs) if j == ri]):
        print(f"  {g.name}/{r.key}: {L.show(prog)} accepted {a([prog], c.lib)}; it is in the vocabulary as items:",
              [L.show((cl.rep,)) for cl in comp if L.show((cl.rep,)) in L.show(prog)][:6], flush=True)
print(f"probe {time.time() - t0:.0f}s; level sizes {sizes}; stops {stops}", flush=True)
for (ri, g), x in zip(goals, res):
    tot = sum(int(t["Am"].sum()) if "Am" in t else int(t["cm"].sum()) for t in g.targets)
    f_all, f_non = [], []
    for d, (lv, rd) in enumerate(zip(levels, rids)):
        sel = torch.nonzero(rd == ri).flatten() if lv is not None else None
        if sel is None or not sel.numel():
            f_all.append(0)
            f_non.append(0)
            continue
        part = [b.take(sel) for b in lv]
        ff = g.fits(part)
        f_all.append(int(ff.max()))
        f_non.append(int(torch.where(g.test(part, c.lib), torch.zeros_like(ff), ff).max()))
    base = max(f_all[:2])
    fit = (max(f_all) - base) / (tot - base) if tot > base and not x["aliases"] else 0.0
    fit2 = (max(f_non) - base) / (tot - base) if tot > base else 0.0
    print(f"{g.name}/{roots[ri].key}: found {x['found_cost']} accepted {[L.show(p) for p in (x['accepted'] or [])][:2]}"
          f"\n  aliases {x['aliases'][:2]}\n  fits per level {f_all} of {tot} (not goal states: {f_non}); "
          f"closeness: SelfClimb's {fit:.3f}, without the goal states {fit2:.3f}", flush=True)
