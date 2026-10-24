"""Why no_step failed on seed 74 (ABLATION_S3.md §3): its reduction search accepted B[P[T.modB~ T.modB]], a repeated
subtraction. This runs that program (with the run's own routines, exact cells) on X = x + q N at growing sizes against
the truth, and estimates how often s3i's reduction check lets it through.

The check (s3i's Acceptor) tests a program on 60 probes at each verify size, W = 1, 2 and 3 digits. At W = 3 the
program performs 10 x 6 x 10 = 600 conditional subtractions, so it passes when no probe's quotient q is above 600. A
probe's q is another pool row's x digits at N's top 3 places, and the probe rows sit at fixed pool positions (the
generator's seed is fixed), so the chance a pool lets the program through is P(q <= 600)^60 over H1's training rows.
Local: CPU, seconds, under 1 GB.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/reduce_check.py
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CPU_FIRST = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(CPU_FIRST, "wider_menus", "local"))
import local_wide as LW

DATA = os.path.expanduser("~/github/one-layer/one-layer-deeper/data/generated/"
                          "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2/train.jsonl")
LIB = ("B0.1=SBB.N INC; B0.2=P[B0.1]; T.lt=CLC LDA.X B0.2; T.lt~=CLC LDA.Y B0.2; T.modx=LDA.X STA.S T.lt SEL.S; "
       "T.modx~=LDA.Y STA.S T.lt~ SEL.S; T.modB=B[T.modx STA.X]; T.modB~=B[T.modx~ STA.Y]")   # no_step's sleep lines
PROGRAM = "B[P[T.modB~ T.modB]]"
LIMIT = 600                     # its subtractions at W = 3: B x P x B, P = 2 W places
N_PROBES = 60                   # the Acceptor's probes per verify size

sc = LW.load_module(os.path.join(CPU_FIRST, "integration", "submission_stage3_s3i.py"))
cells = sc.exact_cells("cpu")
lib = {k: sc.parse(v) for k, v in (d.split("=", 1) for d in LIB.split("; "))}
gen = torch.Generator().manual_seed(0)
digits = lambda v, P: torch.stack([(v // 10 ** j) % 10 for j in range(P)], 1)
print(f"{PROGRAM} against the truth on X = x + q N (q below 10^W, as the reduction's practice problem draws it):")
for W in (1, 2, 3, 4, 8):
    st = sc.Setting(10, W, cells)
    n = 400
    N = torch.randint(10 ** (W - 1) if W > 1 else 2, 10 ** W, (n,), generator=gen)
    x = (torch.rand(n, generator=gen) * N).long()
    q = (torch.rand(n, generator=gen) * 10 ** W).long()
    A, _ = sc.run_prog(st, sc.parse(PROGRAM), lib, {"X": digits(x + q * N, st.P), "N": digits(N, st.P)},
                       sc.gen_on("cpu", 1), "cpu")
    ok = sum(A[:, j].long() * 10 ** j for j in range(st.P)) == x
    wrong = q[~ok]
    print(f"  W = {W}: right on {int(ok.sum())}/{n}" + (f", wrong from q = {int(wrong.min())}" if wrong.numel() else ""))
rows = [json.loads(line) for line in open(DATA)]
Nv = torch.tensor([r["modulus"] for r in rows])
xv = torch.tensor([r["x"] for r in rows])
shift = (torch.floor(torch.log10(Nv.double())).long() + 1 - 3).clamp(min=0)
q = xv // 10 ** shift                                         # x's digits at N's top 3 places
p = float((q <= LIMIT).double().mean())
print(f"H1's {len(rows):,} training rows: q <= {LIMIT} on {p:.3f} of them, so a pool's {N_PROBES} probes at W = 3 "
      f"all pass with chance {p ** N_PROBES:.3f}")
