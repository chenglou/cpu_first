"""libcore.apply_items' lockstep machine (s3h) against the item-by-item one it replaces (_apply_grouped, run_items):
every primitive alone, then real vocabularies (the c5 level-4 library's items and loops, the composition vocabulary of
s3g seed 5's level-6 library, its round vocabulary at 16 places, the library study's r11 routines) on random machines
at every setting the climb uses, with random rows per item; the whole state (registers, flags, pointer) compared
exactly, with the lockstep forced on every bucket and with apply_items' own choice. Then bfs for every fuse,
Acceptor.many, the round's retry, and the round machine (s3i) against run_prog's (every primitive, random programs, the
recorded rounds, round after round on one generator). Local, small (about 30 s, under 1 GB).

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m integration.lib.test_lockstep
"""
import json
import os
import time

import torch

from integration.lib import libcore as L
from integration.lib.profile_search import L4

here = os.path.dirname(os.path.abspath(__file__))
dev = "cuda" if torch.cuda.is_available() else "cpu"
gen = L.gen_on(dev, 3)
t0 = time.time()


def same(a, b):
    return all(bool(torch.equal(getattr(a, f), getattr(b, f))) for f in ("regs", "c", "h", "k", "p"))


def check(name, st, items, lib, S=24, NP=6, rows=4000, forced=True):
    if not items:
        return
    b = L.random_states(st, S, NP, gen, dev)
    n = min(rows, len(items) * S)
    item_of_row = torch.randint(0, len(items), (n,), generator=gen, device=dev)
    src = b.take(torch.randint(0, S, (n,), generator=gen, device=dev))
    keep = src.copy()
    ref = L._apply_grouped(st, src, item_of_row, items, lib)
    saved = L.LOCK_COST
    if forced:
        L.LOCK_COST = -10 ** 9                        # every bucket in lockstep
    try:
        new = L.apply_items(st, src, item_of_row, items, lib)
    finally:
        L.LOCK_COST = saved
    ok = same(ref, new) and same(src, keep)
    print(f"{name}: {len(items)} items, {n} rows, {'forced lockstep' if forced else 'its own choice'}: "
          f"{'same' if ok else 'DIFFERENT'} ({time.time() - t0:.1f}s)", flush=True)
    assert ok, name


def lib_of_sleeps(fname):
    lib = {}
    for line in open(os.path.join(here, fname)):
        body = line.split("; ", 1)[1].rsplit(" (", 1)[0]
        for part in body.split("; "):
            if "=" in part:
                k, v = part.split("=", 1)
                lib[k.strip()] = L.parse(v.strip())
    return lib


cells = L.Cells(10, dev)
sets = [L.Setting(10, 1, cells), L.Setting(10, 2, cells), L.Setting(10, 3, cells), L.Setting(10, 8, cells, 16)]
small = [L.Setting(2, 3, L.Cells(2, dev)), L.Setting(3, 2, L.Cells(3, dev))]
for st in sets + small:                                      # every primitive, alone and mixed
    check(f"primitives {st} P={st.P}", st, list(L.PRIMS), {}, rows=2000)
    check(f"primitives {st} P={st.P}", st, list(L.PRIMS), {}, rows=2000, forced=False)
lib4 = {k: L.parse(v) for k, v in L4.items()}
items = [c.rep for c in L.enumerate_items(list(L.PRIMS) + list(lib4), lib4, sets[:2], dev, max_body=2)]
for st, forced in ((sets[0], True), (sets[1], True), (sets[1], False), (sets[2], True)):
    check(f"c5 L4 vocabulary (body <= 2) {st}", st, items, lib4, rows=1500, forced=forced)
L.LOCK_ROWS, saved_rows = 333, L.LOCK_ROWS                  # a lockstep in slices of rows
check("c5 L4 vocabulary, lockstep in slices of 333 rows", sets[1], items, lib4, rows=1500)
L.LOCK_ROWS = saved_rows
lib6 = lib_of_sleeps("s3g5_sleeps.txt")
routines = sorted(k for k in lib6 if k.startswith("T."))
comp = [c.rep for c in L.enumerate_items(list(L.PRIMS) + routines, lib6, sets[:2], dev, max_body=1)]
for st, forced in ((sets[0], True), (sets[1], True), (sets[1], False), (sets[2], True)):
    check(f"s3g5 L6 composition vocabulary {st}", st, comp, lib6, rows=2000, forced=forced)
rv = [c.rep for c in L.enumerate_items(list(L.PRIMS) + routines, lib6, [sets[3]], dev, max_body=0)]
check("s3g5 L6 round vocabulary (16 places)", sets[3], rv, lib6, S=12, NP=8, rows=600)
check("s3g5 L6 round vocabulary (16 places)", sets[3], rv, lib6, S=12, NP=8, rows=600, forced=False)
r11 = {k: L.parse(v) for k, v in json.load(open(os.path.join(here, "r11_lib.json")))["lib"].items()}
progs = [L.parse(p) for p in ("LDA.X STA.Y T.mul STA.X T.reduce", "T.sqmod", "T.mul", "T.reduce")
         if all(t in r11 or t in L.PRIMS for t in p.split())]
for prog in progs:
    check(f"r11 {L.show(prog)} (16 places)", sets[3], list(prog), r11, S=8, NP=8, rows=64)
print(f"all same ({time.time() - t0:.1f}s)", flush=True)
# bfs with consecutive blocks fused into one apply_items call: the same search for every fuse (small chunks, so that
# levels split into many blocks of different class costs and source chunks)
from integration.lib import inclimb  # noqa: E402

rows = inclimb.load_rows(300, seed=0)
Nd, Xd, _ = (t.to(dev) for t in inclimb.digits_of_rows(rows))
practice = sets[:2]
small_lib = {k: lib4[k] for k in ("B0.1", "B0.2", "B0.3", "T.lt", "T.lt~", "B1.1", "B1.2", "T.modx", "T.modx~")}
vocab = L.enumerate_items(list(L.PRIMS) + list(small_lib), small_lib, practice, dev, max_body=2)
prims = L.enumerate_items(list(L.PRIMS), {}, practice, dev, max_body=2)
lib_lt = {k: small_lib[k] for k in ("B0.1", "B0.2", "B0.3", "T.lt", "T.lt~")}
vocab_lt = L.enumerate_items(list(L.PRIMS) + list(lib_lt), lib_lt, practice, dev, max_body=2)
add_h = {"lt": small_lib["T.lt"], "modx": small_lib["T.modx"]}
for task, helpers, voc, lb, cap in (("lt", {}, prims, {}, 30_000), ("lt~", {}, prims, {}, 4_000),
                                    ("add", add_h, vocab, small_lib, 30_000), ("add", add_h, vocab, small_lib, 3_000),
                                    ("modx", {}, vocab_lt, lib_lt, 100_000)):
    prob = L.LawProblem(task, practice, 8, 0, (Nd, Xd), helpers, lb, dev)
    acc = L.Acceptor(task, lb, helpers, (Nd, Xd), [L.Setting(10, W, cells) for W in (1, 2, 3)], dev)
    outs = []
    for fuse in (1, 2, 4):
        r = L.bfs(prob, voc, lb, max_cost=4, max_states=cap, accept=acc.many, chunk=1 << 13, listing="goals",
                  fuse=fuse)
        par = [None if p is None else [t.tolist() for t in p] for p in r.parents]
        outs.append((r.found_cost, r.level_sizes, r.stop, r.goals, par, [L.show(p) for p in (r.accepted or [])]))
    print(f"bfs {task} cap {cap}: found {outs[0][0]}, levels {outs[0][1]} {outs[0][2]}, accepted {outs[0][5][:2]}; "
          f"fuse 2 and 4 the same: "
          f"{outs[1] == outs[0] and outs[2] == outs[0]} ({time.time() - t0:.1f}s)", flush=True)
    assert outs[1] == outs[0] and outs[2] == outs[0], task
print(f"bfs the same for every fuse ({time.time() - t0:.1f}s)", flush=True)
# Acceptor.many (the programs batched) against Acceptor.__call__ (each program alone, run_items)
helpers_add = {"lt": small_lib["T.lt"], "modx": small_lib["T.modx"]}
verify3 = [L.Setting(10, W, cells) for W in (1, 2, 3)]
cand = [c.rep for c in vocab[:400]]
progs = [(a,) for a in cand[:120]] + [(a, b) for a, b in zip(cand[120:280], cand[160:320])] + \
    [(a, b, c) for a, b, c in zip(cand[200:300], cand[250:350], cand[300:400])] + \
    [L.parse("CLC LDA.N P[ADC.X INC]"), L.parse("T.modx CLC P[ADC.N INC]"), L.parse("LDA.N CLC P[ADC.X INC]")]
for task, hp in (("add", helpers_add), ("modx", {}), ("lt", {})):
    one = L.Acceptor(task, small_lib, hp, (Nd, Xd), verify3, dev)
    alone = [one(p) for p in progs]
    many = L.Acceptor(task, small_lib, hp, (Nd, Xd), verify3, dev).many(progs)
    print(f"Acceptor {task}: {len(progs)} programs, {sum(alone)} pass; many() the same: {many == alone} "
          f"({time.time() - t0:.1f}s)", flush=True)
    assert many == alone, task
# the round's retry (s3h): a finished climb with no round searches again every round_retry calls, then stops once found
r11 = {k: L.parse(v) for k, v in json.load(open(os.path.join(here, "r11_lib.json")))["lib"].items()}
rows_t1 = inclimb.load_rows(16, seed=3, T=1)
ans = tuple(t.to(dev) for t in inclimb.digits_of_rows(rows_t1))
c = L.Climb(L.Cells(10, dev), dev, 16, round_retry=3, round_cost=1, round_budget=10_000)
c.lib, c.done = dict(r11), True                             # T.sqmod alone is the round, at cost 1
lines = [c.step((Nd, Xd), answers=ans) for _ in range(7)]
print("retry:", [None if x is None else x[:60] for x in lines], "| rounds:", [L.show(p) for p in c.rounds], flush=True)
assert lines[:2] == [None, None] and lines[2] is not None and c.rounds and all(x is None for x in lines[3:])
print(f"retry as specified ({time.time() - t0:.1f}s)", flush=True)
# the round machine (s3i) against run_prog's machine (fresh, run_items), as stage 3's run_rounds uses both: rounds on one
# generator, each on the rows of a random go mask; every go row's whole state (registers, c, h, k) compared after every
# round, and the rows' digits carried from round to round (on a GPU: the first run eager, the rest graph replays)


def rounds_check(name, st, prog, lib, m=37, rounds=5):
    W = st.W
    rnd = lambda shape: torch.randint(0, st.B, shape, generator=gen, device=dev)
    s, Nd = rnd((m, W)), rnd((m, W))
    gos = [torch.rand(m, generator=gen, device=dev) < f for f in (0.9, 0.7, 0.5, 0.3, 1.0)][:rounds]
    g_old, g_new = L.gen_on(dev, 0), L.gen_on(dev, 0)
    mach = L.round_machine(st, prog, lib, m, dev)
    s_new, P = s.clone(), st.P
    ok = True
    for go in gos:
        n = int(go.sum())
        if n == 0:
            continue
        b = L.fresh(st, {"X": L.pad(s[go], P), "N": L.pad(Nd[go], P)}, g_old, n, dev)
        L.run_items(st, b, prog, lib)
        A = mach.run(s_new, Nd, go, n, g_new)
        St = mach.St[:, :m][:, go]
        ok &= bool(torch.equal(St[:5 * P].reshape(5, P, n), b.regs[0].long()))
        ok &= bool(torch.equal(St[5 * P:5 * P + 3], torch.cat([b.c, b.h, b.k]).long()))
        s[go] = b.regs[0, L.A_].T.long()[:, :W]
        s_new = torch.where(go[:, None], A[:, :W], s_new)
        ok &= bool(torch.equal(s, s_new))
    print(f"round machine {name} {st} P={P}: {len(gos)} rounds of {m} rows: {'same' if ok else 'DIFFERENT'} "
          f"({time.time() - t0:.1f}s)", flush=True)
    assert ok, name


r16 = L.Setting(10, 8, cells, 16)
for st in [r16] + sets[:3] + small:                          # every primitive, then random straight-line programs
    rounds_check("every primitive", st, tuple(L.PRIMS), {})
    for i in range(3):
        rounds_check(f"random program {i}", st, tuple(L.PRIMS[j] for j in torch.randint(
            0, len(L.PRIMS), (300,), generator=gen, device=dev).tolist()), {})
rounds_check("c5 L4 vocabulary (body <= 2) as one program", sets[1], tuple(items[:300]), lib4)
for fname, m in (("s3h6_lib.json", 130), ("s3f_lib.json", 64), ("s3d_lib.json", 64)):    # recorded libraries, rounds
    rec = json.load(open(os.path.join(here, fname)))
    libr = {k: L.parse(v) for k, v in rec["lib"].items()}
    for p in rec["round"]:
        rounds_check(f"{fname} {p}", r16, L.parse(p), libr, m=m)
for p in ("LDA.X STA.Y T.mul STA.X T.reduce", "T.sqmod", "T.mul", "T.reduce"):
    if all(t in r11 or t in L.PRIMS for t in p.split()):
        rounds_check(f"r11 {p}", r16, L.parse(p), r11)
print(f"the round machine the same ({time.time() - t0:.1f}s)", flush=True)
