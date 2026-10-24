"""Why the 61-instruction lifted climb (records/climb/lifted_b200_x4m) misses the reduction at cost 5 although
`CLA P[BRG.X STA.Y T.modB~]` passes its check (checks/lifted_reduce.py): is that loop in the library vocabulary the
reduce search used, and if its body was merged into another body's class (items that agree on the vocabulary's 24 random
test machines are one item), which one, and does that one's loop pass?

    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/bridge/modal_climb_b.py --script bar3.bridge.checks.reduce_vocab --tag reduce_vocab
"""
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
}   # the climb's library when it searched the reduction (level 4 and 5, before the multiply's merges)

dev = "cuda" if torch.cuda.is_available() else "cpu"
cells, _ = CB.build_views(CB.SETS["lifted"]["cells"])
L.set_instructions(cells, sel=False)
L.MAC_ADDEND = "A"
L.NEEDS.update(mul=("add", "mac"), reduce=("add", "modB"))
CB.strong_modx()
rows = tuple(t.to(dev) for t in inclimb.digits_of_rows(inclimb.load_rows(4096, seed=0))[:2])
climb = L.Climb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16, **CB.CLIMB)
for k, v in LIB4.items():
    climb.lib[k] = L.parse(v)
for t in ("add", "add~", "modB", "modB~", "modx", "modx~", "lt", "lt~"):
    climb.corpus[t] = [L.parse("T." + t)]
helpers = climb.helpers_for("reduce")
prob = L.LawProblem("reduce", climb.settings, climb.NP, 0, rows, helpers, climb.lib, dev)
acc = L.Acceptor("reduce", climb.lib, helpers, rows, climb.verify, dev)


def on_problem(p):
    outs = []
    for st, root in zip(climb.settings, prob.roots):
        b = root.copy()
        L.run_items(st, b, p, climb.lib)
        outs.append(b)
    return bool(prob.goal(outs)[0])


want = L.parse("CLA P[BRG.X STA.Y T.modB~]")
print("CLA P[BRG.X STA.Y T.modB~]: search problem", on_problem(want), "acceptor", acc(want), flush=True)
t0 = time.time()
classes = climb.classes()
print(f"library vocabulary: {len(classes)} classes in {time.time() - t0:.0f}s from {len(L.PRIMS) + len(climb.lib)} tokens",
      flush=True)
loop = want[1]
print("the loop P[BRG.X STA.Y T.modB~] is a class of its own:", any(c.rep == loop for c in classes), flush=True)
body = loop[2]
one = ("REP", L.ONE, body)
cache = climb._sig_cache
sig = cache.get(one)
print("its body was a candidate:", sig is not None, flush=True)
if sig is not None:
    same = [k for k, s in cache.items() if s == sig and k[0] == "REP" and k[1] == L.ONE]
    print("bodies with the same signature:", [L.show(k[2]) for k in same][:12], flush=True)
    for k in same[:12]:
        for kind in L.LOOP_KINDS:
            p = [want[0], ("REP", kind, k[2])]
            if any(c.rep == p[1] for c in classes):
                print("  class", L.show(p), "search problem", on_problem(p), "acceptor", acc(p), flush=True)
