"""s3i's modx check is weak: its probes test x < N (keep x) and x = N (0) only, never N < x < 2N. The 61-instruction
lifted climb (records/climb/lifted_b200_x4) accepts `T.lt P[BRG.X RSELC.X]` there (T.lt = `CLC LDA.X P[LT.N INC]`,
which leaves A = x), and it zeroes x = N and keeps everything else. This runs it, and the variant's
`T.lt P[SELC.X INC]` (T.lt = `CLC LDA.X P[SUB.N INC]`, which leaves A = x - N), on x < N, x = N and N < x < 2N
(12 random N below 5,000 each).

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python bar3/bridge/checks/modx_overfit.py
"""
import torch
from bar3.bridge import climb_b as CB, libcore_b as L

cells, _ = CB.build_views(CB.SETS["lifted"]["cells"])
L.set_instructions(cells, sel=False)
lib = {k: L.parse(v) for k, v in {
    "B0.4": "LT.N INC", "B0.5": "P[B0.4]", "B0.1": "SUB.N INC", "B0.2": "P[B0.1]",
    "T.lt": "CLC LDA.X B0.5", "T.ltsub": "CLC LDA.X B0.2",
    "B1.1": "BRG.X RSELC.X", "B1.2": "P[B1.1]", "T.modx": "T.lt B1.2",
    "T.modx_inv4": "T.ltsub P[SELC.X INC]"}.items()}
st = L.Setting(10, 6, L.Cells(10, "cpu"))
gen = torch.Generator().manual_seed(0)
def digits(v, P):
    return torch.tensor([[(int(x) // 10 ** j) % 10 for j in range(P)] for x in v])
def val(d):
    return [sum(int(x) * 10 ** j for j, x in enumerate(r)) for r in d]
N = torch.randint(100, 4999, (12,), generator=gen)
x_lo = (torch.rand(12, generator=gen) * N).long()
x_hi = N + (torch.rand(12, generator=gen) * N).long() + 1
for name in ("T.modx", "T.modx_inv4"):
    for label, x in (("x<N", x_lo), ("x=N", N.clone()), ("N<x<2N", x_hi)):
        A, c = L.run_prog(st, lib[name], lib, {"X": digits(x.tolist(), st.P), "N": digits(N.tolist(), st.P)}, gen, "cpu")
        got = val(A.tolist())
        want = [(xi - ni) if xi >= ni else xi for xi, ni in zip(x.tolist(), N.tolist())]
        print(f"{name:12s} {label:7s} right {sum(g == w for g, w in zip(got, want))}/12   e.g. N={N[0].item()} x={x[0].item()} -> {got[0]} (want {want[0]})")
