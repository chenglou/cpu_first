"""climb_ab (the A/B's variants B and C) on tiny budgets: the cached acceptance agrees with libcore.accept; C's mac takes
the accumulating row and rejects A's mac program; mul = CLA W[T.mac INC SHL.X] passes mul's laws with that routine;
B's rules (twin skip, stall) fire on a short climb. Local, small budgets (well under 1 GB, under a minute).

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m integration.lib.test_climb_ab
"""
import time

import torch

from integration.lib import libcore as L, inclimb, climb_ab as AB

torch.manual_seed(0)
t0 = time.time()
rows = inclimb.load_rows(300, seed=0)
Nd, Xd, _ = inclimb.digits_of_rows(rows)
cells = L.Cells(10)
verify = [L.Setting(10, W, cells) for W in (1, 2, 3)]
txt = {"T.add": "CLC LDA.N P[ADC.X INC]", "T.add~": "CLC LDA.N P[ADC.Y INC]",
       "T.mrow": "LDH.Y CLK CLA P[MUL.X INC]", "T.mrow~": "LDH.X CLK CLA P[MUL.Y INC]"}
lib = {k: L.parse(v) for k, v in txt.items()}
helpers = {"mrow": lib["T.mrow"], "add": lib["T.add"]}
progs = ["T.mrow CLC P[ADC.N INC]", "LDH.Y CLK P[MUL.X INC]", "LDH.Y CLK LDA.N P[MUL.X INC]", "CLK LDH.Y P[MUL.X INC]"]
acc = AB.Acceptor("mac", lib, helpers, (Nd, Xd), verify, "cpu")
print("A's mac, cached vs libcore:", [acc(L.parse(p)) for p in progs],
      [L.accept("mac", L.parse(p), lib, helpers, (Nd, Xd), verify, "cpu") for p in progs])
L.build = AB.build_c
acc = AB.Acceptor("mac", lib, helpers, (Nd, Xd), verify, "cpu")
print("C's mac (A + x d):           ", [acc(L.parse(p)) for p in progs])
acc2 = AB.Acceptor("mac~", lib, {"mrow": lib["T.mrow~"], "add": lib["T.add~"]}, (Nd, Xd), verify, "cpu")
print("C's mac~ (A + y x_p):        ", acc2(L.parse("LDH.X CLK P[MUL.Y INC]")), acc2(L.parse("LDH.Y CLK P[MUL.X INC]")))
lib["T.mac"] = L.parse("LDH.Y CLK P[MUL.X INC]")
mul = AB.Acceptor("mul", lib, {"add": lib["T.add"]}, (Nd, Xd), verify, "cpu")
print("mul = CLA W[T.mac INC SHL.X]:", mul(L.parse("CLA W[T.mac INC SHL.X]")), "| without CLA:",
      mul(L.parse("W[T.mac INC SHL.X]")), "| with T.mrow:", mul(L.parse("CLA W[T.mrow INC SHL.X]")))
L.build = AB._build
# a short climb with B's rules on tiny caps (lt, lt~ at level 0), then the twin and stall rules on a set state
c = AB.ClimbB(L.Cells(10), "cpu", 16, budget=10_000, fallback_budget=10_000, max_body=2, max_cost=4, levels=1, comp_body=2,
              chunk=1 << 15)
c.tasks = ["lt", "lt~"]
for _ in range(4):
    line = c.step((Nd, Xd))
    if line is None:
        break
    print("climb:", line[:160])
c.corpus.update({"modx": [L.parse("LDA.X STA.S CLC P[SBB.N INC] SEL.S")]})
c.used_up.update({"modx~": 1, "add": 2, "add~": 1})
c.tasks, c.level = ["modx", "modx~", "add", "add~"], 1
c.corpus.update({"lt": [L.parse("CLC LDA.X P[SBB.N INC]")], "lt~": [L.parse("CLC LDA.Y P[SBB.N INC]")]})
print("todo:", c.todo(), "| events:", c.events, "| stalled", c.stalled, "skipped", c.skipped, f"{time.time() - t0:.1f}s")
# C's listing (one spelling per goal state, then more spellings of the accepted ones), with rule 1's compositions:
# modx from the routine T.lt at level 1
c = AB.ClimbB(L.Cells(10), "cpu", 16, budget=30_000, fallback_budget=30_000, max_body=2, max_cost=5, levels=2,
              chunk=1 << 15, every_goal=True, comp_body=2)
c.lib = {"T.lt": L.parse("CLC LDA.X P[SBB.N INC]"), "T.lt~": L.parse("CLC LDA.Y P[SBB.N INC]")}
c.corpus = {"lt": [c.lib["T.lt"]], "lt~": [c.lib["T.lt~"]]}
c.tasks, c.level = ["lt", "lt~", "modx", "modx~"], 1
lines_c = []
for _ in range(3):
    line = c.step((Nd, Xd))
    if line is None:
        break
    lines_c.append(line)
    print("climb C:", line[:200])
print("C's records:", [(r["task"], r["how"], [(x["how"], x["listings"]) for x in r["runs"]]) for r in c.records],
      f"{time.time() - t0:.1f}s")
# libcore's own switches (what stage 3 inlines) reproduce C on the same case
L.MAC_ADDEND = "A"
k = L.Climb(L.Cells(10), "cpu", 16, budget=30_000, fallback_budget=30_000, max_body=2, max_cost=5, levels=2,
            chunk=1 << 15, comp_first=True, comp_body=2, skip_twins=True, stall_after=2, listing="goals")
k.lib = {"T.lt": L.parse("CLC LDA.X P[SBB.N INC]"), "T.lt~": L.parse("CLC LDA.Y P[SBB.N INC]")}
k.corpus = {"lt": [k.lib["T.lt"]], "lt~": [k.lib["T.lt~"]]}
k.tasks, k.level = ["lt", "lt~", "modx", "modx~"], 1
lines_k = [k.step((Nd, Xd)) for _ in range(3)]
import re
strip = lambda x: re.sub(r" [\d.]+s ", " ", x or "")
print("libcore switches = C:", [strip(x) for x in lines_k] == [strip(x) for x in lines_c])
acc_c = L.Acceptor("mac", lib, helpers, (Nd, Xd), verify, "cpu")
print("libcore's mac with MAC_ADDEND A:", [acc_c(L.parse(p)) for p in progs])
L.MAC_ADDEND = "N"
# Occam's tie-break by executed steps (libcore, occam_steps): lt's two cost-5 spellings, and the in-run level-1 modx
# pair whose first is the alias "x if x < N else 0" (with the slower lt as T.lt)
two = [L.parse("CLC P[LDA.X SBB.N INC]"), L.parse("CLC LDA.X P[SBB.N INC]")]
print("Occam, lt:", L.show(L.simplest(two, {})), "| by steps:", L.show(L.simplest(two, {}, by_steps=True)))
lib_r = {"T.lt": two[0], "T.lt~": L.parse("CLC LDA.Y P[SBB.N INC]")}
pair = [L.parse("T.lt LDA.X STA.S CLA SEL.S"), L.parse("LDA.X STA.S STA.Y T.lt~ SEL.S")]
print("Occam, modx:", L.show(L.simplest(pair, lib_r)), "| by steps:", L.show(L.simplest(pair, lib_r, by_steps=True)))
acc_m = L.Acceptor("modx", lib_r, {}, (Nd, Xd), verify, "cpu")
xs = L.pad(torch.tensor([[3, 2]] * 4), 4)          # x = 23 (digits LSB first)
ns = L.pad(torch.tensor([[7, 1]] * 4), 4)          # N = 17
st2 = L.Setting(10, 2, cells)
for p in pair:      # both pass modx's laws; on x in [N, 2N) the alias gives 0, the other x - N = 6
    A, _ = L.run_prog(st2, p, lib_r, {"X": xs, "N": ns}, L.gen_on("cpu", 0), "cpu")
    print("   ", L.show(p), "passes modx:", acc_m(p), "| x = 23, N = 17 ->", A[0].tolist())
