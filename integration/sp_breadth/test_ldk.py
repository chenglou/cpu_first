"""Local checks (seconds, CPU) of this study's machine changes (SP_BREADTH.md): LDK gives the same state on the four
machines (step, the lockstep, the round machine, the one-hot machine); route's spill and constant sources."""
import torch
import torch.nn.functional as F

from integration.sp_breadth import libb as L
from integration.sp_breadth import spbcore as S

torch.manual_seed(0)
cells, P, NP = L.Cells(10), 16, 6
const = torch.tensor([3, 1] + [0] * (P - 2), dtype=torch.uint8)
st = L.Setting(10, P // 2, cells, P, const)
lib = {"T.add.pair.AY": L.parse("CLC P[ADC.Y INC]")}
x = torch.randint(0, 10, (NP, P), dtype=torch.uint8)
x[:, P // 2:] = 0
prog = tuple(S.route(("A", "Y"), ("A", "K"), spill=True)) + ("T.add.pair.AY",)
print("route (A, Y) <- (A, K):", L.show(prog))
assert L.show(prog) == "STA.S LDK STA.Y LDA.S T.add.pair.AY"
assert S.route(("A", "Y"), ("A", "X")) is None and S.route(("A", "Y"), ("A", "X"), spill=True) == ["STA.S", "LDA.X", "STA.Y", "LDA.S"]
assert S.route(("X", "Y"), ("X", "X")) == ["LDA.X", "STA.Y"]           # unchanged without a spill

b0 = L.fresh(st, {"A": x}, torch.Generator().manual_seed(1), NP, "cpu")
b1 = b0.copy()
L.run_items(st, b1, prog, lib)                                            # step()
want = (torch.tensor([int("".join(str(int(d)) for d in reversed(r.tolist()))) for r in x]) + 13)
got = torch.tensor([int("".join(str(int(d)) for d in reversed(r.tolist()))) for r in b1.regs[0, L.A_].T])
assert bool((got == want).all()), (got, want)
L.LOCK_COST = -1e9                                                        # the lockstep, forced
b2 = L.apply_items(st, b0.copy(), torch.zeros(1, dtype=torch.long), [("REP", L.ONE, prog)], lib)
assert bool((b2.regs == b1.regs).all()) and bool((b2.c == b1.c).all())
m = L.round_machine(st, prog, lib, NP, "cpu")                             # the round machine (A from X here: LDA.X)
m2 = L.round_machine(st, ("LDA.X",) + prog, lib, NP, "cpu")
A = m2.run(x, torch.zeros(NP, P // 2, dtype=torch.long), torch.ones(NP, dtype=torch.bool), NP, torch.Generator().manual_seed(1))
assert bool((A[:, :P].to(torch.uint8) == b1.regs[0, L.A_].T).all())
oh = lambda d: F.one_hot(d.long(), 10).float()
regs = [oh(x)] + [oh(torch.zeros(NP, P, dtype=torch.uint8))] * 4
SBB = torch.cat([F.one_hot(cells.sbb[:, 0], 10), F.one_hot(cells.sbb[:, 1], 2)], 1).float()
ADC = torch.cat([F.one_hot(cells.adc[:, 0], 10), F.one_hot(cells.adc[:, 1], 2)], 1).float()
sm = L.SteMachine(SBB, ADC, None, P, lambda t: t, const=oh(const))
z2, z10 = F.one_hot(torch.zeros(NP, dtype=torch.long), 2).float(), F.one_hot(torch.zeros(NP, dtype=torch.long), 10).float()
out = sm.run(regs, z2, z10, z10, prog, lib)
assert bool((out[0][0].argmax(-1).to(torch.uint8) == b1.regs[0, L.A_].T).all())
print("LDK: step, lockstep, round machine and one-hot machine agree; x + 13 on", NP, "rows")
