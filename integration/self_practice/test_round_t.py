"""A check of the round's search on answers of one T (SELF_PRACTICE.md §1, the smallest-T rule), with s3f's recorded
library (integration/lib/s3f_lib.json): T = 1 answers find s3f's round at cost 5; T = 2 answers (x^4 mod N) find the
round when it is one routine (T.sq, the whole round), run twice, and never accept the one-round program itself.

    run on Modal (a GPU; the full-size search is heavy): modal_sp.py --envs '[{"TAG": "test_round_t",
        "MODULE": "integration.self_practice.test_round_t"}]'
"""
import torch

from integration.lib import libcore as L
from integration.self_practice import spcore as S
from integration.self_practice.core import digits, load_rows

dev = "cuda" if torch.cuda.is_available() else "cpu"
S3F = {
 "B0.1": "SBB.N INC",
 "B0.2": "P[B0.1]",
 "T.lt~": "CLC LDA.Y B0.2",
 "T.modx~": "LDA.Y STA.S T.lt~ SEL.S",
 "B3.1": "CLK CLA",
 "T.modB~": "B[T.modx~ STA.Y]",
 "B4.1": "STA.Y T.modB~",
 "B4.2": "BRG.X B4.1",
 "T.reduce": "B3.1 P[B4.2]",
 "T.mac": "LDH.Y CLK P[MUL.X INC]",
 "T.mul": "CLA W[T.mac SHL.X SHR.Y]"
}
lib = {k: L.parse(v) for k, v in S3F.items()}      # s3f's routines for the round (lib/s3f_lib.json)
for t in (1, 2):
    rows = load_rows(64, seed=1, T=t)
    ans = tuple(digits(rows, k).to(dev) for k in ("modulus", "x", "result"))
    for extra in ({}, {"T.sq": L.parse("LDA.X STA.Y T.mul STA.X T.reduce")}):
        c = S.SelfClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16, fuse=4)
        c.lib = dict(lib, **extra)
        progs, rec = c.search_round(ans, t=t)
        print(f"T={t} {'with T.sq' if extra else 'library only'}: cost {rec['found_cost']} levels {rec['level_sizes']} "
              f"{rec['seconds']}s {rec['stop']} programs {rec['programs'][:4]}", flush=True)

# the round's search by calls (the big-library mode): s3f's routines with their input registers, two calls
rows = load_rows(64, seed=1, T=1)
ans = tuple(digits(rows, k).to(dev) for k in ("modulus", "x", "result"))
c = S.SelfClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16, fuse=4)
c.lib = dict(lib)
for tok, bd in (("T.mul", "XY"), ("T.reduce", "XN"), ("T.mac", "AXY"), ("T.modB~", "YN"), ("T.modx~", "YN"),
                ("T.lt~", "YN")):
    c.ops.add(tok, "s3f", tok, tuple(bd))
prob = L.round_problem(c.cells, ans, 8, 16, dev)
classes = c.macro_classes(c.lib)
res = L.bfs(prob, classes, c.lib, max_cost=2, max_states=4_000_000,
            accept=lambda ps: [L.round_ok(p, c.lib, c.cells, ans, 16, dev) for p in ps], chunk=1 << 19, fuse=4)
print(f"by calls: {len(classes)} items, cost {res.found_cost} levels {res.level_sizes} {res.stop} programs "
      f"{[L.show(p) for p in (res.accepted or [])][:4]}", flush=True)
