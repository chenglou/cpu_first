"""Checks the fix of the two weak checks (ABLATION_S3.md §8) locally, with exact cells and s3i's own library and
acceptance code:
1. the fixed file builds the same reduction probes as s3i at the practice and verify sizes (so its climb is s3i's
   wherever s3i's check was right);
2. on pools of H1's training rows, how often s3i's check lets the repeated subtraction B[P[T.modB~ T.modB]] through,
   and, on such pools, that the fixed check rejects it while still accepting s3i's reduction;
3. the residual risk: the chance that a pool's 60 full-size probes all have quotients small enough for a loop of
   subtractions within the search's reach;
4. the conditional subtraction's new probes (N < X < 2N, the answer X - N) in both fixed files, and that the fixed
   check rejects the "x if x < N, else 0" alias s3i's check accepts, while accepting s3i's modx.
Local: CPU, a few seconds, under 1 GB.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/fix/check_fix.py [POOLS=3000]
"""

import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CPU_FIRST = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(CPU_FIRST, "wider_menus", "local"))
import local_wide as LW

DATA = os.path.expanduser("~/github/one-layer/one-layer-deeper/data/generated/"
                          "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2/train.jsonl")
LIB = ("B0.1=SBB.N INC; B0.2=P[B0.1]; B0.3=CLC B0.2; T.lt=CLC LDA.X B0.2; T.lt~=CLC LDA.Y B0.2; "
       "T.modx=LDA.X STA.S T.lt SEL.S; T.modx~=LDA.Y STA.S T.lt~ SEL.S; B2.2=CLC LDA.N; T.add=B2.2 P[ADC.X INC]; "
       "B3.1=CLK CLA; T.modB=B[T.modx STA.X]; T.modB~=B[T.modx~ STA.Y]")      # s3i_official's, from its sleep lines
HELPERS = {"add": "B2.2 P[ADC.X INC]", "modB": "B[T.modx STA.X]"}               # reduce's helpers (needs: add, modB)
REAL, LOOP = "B3.1 P[BRG.X STA.Y T.modB~]", "B[P[T.modB~ T.modB]]"
POOL, N_PROBES, LIMIT3 = 4096, 60, 600     # the climb's pool; the Acceptor's probes per size; LOOP's subtractions at W=3
REACH = 10 * 16 ** 3                       # the most conditional subtractions a program within the search's reach makes
                                           # at W = 8: P[P[P[T.modB]]] (cost 4, bodies of cost 3: the deepest nesting)

args = dict(a.split("=") for a in sys.argv[1:])
pools = int(args.get("POOLS", 3000))
t0 = time.time()
s3i = LW.load_module(os.path.join(CPU_FIRST, "integration", "submission_stage3_s3i.py"))
fix = LW.load_module(os.path.join(HERE, "submission_stage3_s3i_fix.py"))
rows = [json.loads(line) for line in open(DATA)]
Nv = torch.tensor([r["modulus"] for r in rows])
xv = torch.tensor([r["x"] for r in rows])
digits = lambda v, L: torch.stack([(v // 10 ** j) % 10 for j in range(L)], 1)
ND, XD = digits(Nv, 8), digits(xv, 8)                                     # the read rows, LSB first


def setup(mod):
    cells = mod.exact_cells("cpu")
    lib = {k: mod.parse(v) for k, v in (d.split("=", 1) for d in LIB.split("; "))}
    return cells, lib, {k: mod.parse(v) for k, v in HELPERS.items()}


# 1. the same practice machines (inputs, targets and the random starts of every register) at the search's sizes and at
#    each verify size, as the climb builds them: the search's problem (seed 0) and the acceptance's (seed 11)
gen = torch.Generator().manual_seed(1)
pool = torch.randint(0, len(rows), (POOL,), generator=gen)


def machines(mod, Ws, n, seed):
    cells, lib, helpers = setup(mod)
    prob = mod.LawProblem("reduce", [mod.Setting(10, W, cells) for W in Ws], n, seed, (ND[pool], XD[pool]), helpers,
                          lib, "cpu")
    return [b.features() for b in prob.roots] + [t["A"] for t in prob.targets]


same = {"search (W = 1, 2)": all(torch.equal(a, b) for a, b in zip(machines(s3i, (1, 2), 8, 0),
                                                                     machines(fix, (1, 2), 8, 0)))}
for W in (1, 2, 3):
    same[f"acceptance W = {W}"] = all(torch.equal(a, b) for a, b in zip(machines(s3i, (W,), N_PROBES, 11),
                                                                         machines(fix, (W,), N_PROBES, 11)))
print(f"1. the fixed file's reduction machines equal s3i's, random starts included: {same}")

# 2. pools that let LOOP through s3i's check: its W = 3 probes' quotients are all <= 600. The Acceptor draws them with
#    a fresh generator (seed 11), so they sit at fixed pool positions: the second block of 60 picks gives q.
pick = torch.randint(0, POOL, (4 * N_PROBES,), generator=s3i.gen_on("cpu", 11))[N_PROBES:2 * N_PROBES]
shift = (torch.floor(torch.log10(Nv.double())).long() + 1 - 3).clamp(min=0)
q3 = xv // 10 ** shift                                                    # x's digits at N's top 3 places
lucky = []
for i in range(pools):
    p = torch.randint(0, len(rows), (POOL,), generator=gen)
    if int(q3[p[pick]].max()) <= LIMIT3:
        lucky.append(p)
print(f"2. pools whose 60 three-digit probes all have q <= {LIMIT3}: {len(lucky)}/{pools} "
      f"({len(lucky) / pools:.1%}; 0.943^60 = 3.0%)")
for p in lucky[:2]:
    verdicts = {}
    for name, mod in (("s3i", s3i), ("fix", fix)):
        cells, lib, helpers = setup(mod)
        verify = [mod.Setting(10, W, cells) for W in (1, 2, 3)] + ([mod.Setting(10, 8, cells)] if mod is fix else [])
        acc = mod.Acceptor("reduce", lib, helpers, (ND[p], XD[p]), verify, "cpu")
        verdicts[name] = {s: acc(mod.parse(s)) for s in (LOOP, REAL)}
    print(f"   s3i's check: {LOOP} {verdicts['s3i'][LOOP]}, {REAL} {verdicts['s3i'][REAL]}; "
          f"the fix's: {verdicts['fix'][LOOP]}, {verdicts['fix'][REAL]}")

# 3. the residual risk: at W = 8 a probe's q is another row's x itself
p8 = float((xv <= REACH).double().mean())
print(f"3. H1's rows with x <= {REACH:,} (the most subtractions within reach at W = 8): {p8:.3f}; all {N_PROBES} "
      f"full-size probes that small: {p8:.3f}^{N_PROBES} = 10^{math.log10(p8) * N_PROBES:.0f}")

# 4. modx: the new probes, and the alias occam_steps_off's climb accepted
sp2 = LW.load_module(os.path.join(HERE, "submission_stage3_sp2_fix.py"))
MODX, ALIAS = "LDA.X STA.S T.lt SEL.S", "T.lt LDA.X STA.S CLA SEL.S"      # s3i's modx; "x if x < N, else 0"
cells, lib, _ = setup(fix)
for name, mod in (("s3i fix", fix), ("sp2 fix", sp2)):
    right = total = drawn = 0
    for W in (1, 2, 3):
        st = mod.Setting(10, W, mod.exact_cells("cpu"))
        p = torch.randint(0, len(rows), (60,), generator=gen)
        Nt, _ = mod.top_parts(ND[p], XD[p], st.P - 1 if mod is sp2 else st.P)
        X, A, ok = mod.above_probes(mod.pad(Nt, st.P), st, mod.gen_on("cpu", 11), "cpu")
        val = lambda D: sum(D[:, j].long() * 10 ** j for j in range(D.shape[1]))
        n, x, a = val(mod.pad(Nt, st.P)), val(X), val(A)
        right += int(((n < x) & (x < 2 * n) & (x - n == a))[ok].sum())
        drawn += int(ok.sum())
        total += ok.numel()
    print(f"4. {name}: {drawn}/{total} probes drawn at W = 1, 2, 3; N < X < 2N and X - N the answer on {right}/{drawn}")
draw = sp2.Draw("pair", type("S", (), {"P": 6, "B": 10, "W": 3, "cells": sp2.exact_cells("cpu"), "above": True})(),
                60, (ND, XD), sp2.gen_on("cpu", 11), "cpu")
ops_x, ops_n = draw.operands(None, None)
t = sp2.law_target("red", draw, None, None)
val = lambda D: sum(D[:, j].long() * 10 ** j for j in range(D.shape[1]))
agree = bool((val(ops_x) - val(ops_n) == val(t["A"]))[t["Am"]].all())
print(f"   sp2 fix: red on an above draw: operands (N + d B^j, N), target d B^j, {int(t['Am'].sum())} probes, agree: {agree}")
p = torch.randint(0, len(rows), (POOL,), generator=gen)
v = {}
for name, mod in (("s3i", s3i), ("fix", fix)):
    cells, lib, helpers = setup(mod)
    climb = mod.Climb(cells, "cpu", mod.PR, chunk=1 << 13, **dict(mod.CLIMB, budget=20_000, fallback_budget=20_000))
    verify = climb.verify_for("modx") if mod is fix else climb.verify
    acc = mod.Acceptor("modx", lib, {}, (ND[p], XD[p]), verify, "cpu")
    v[name] = {s: acc(mod.parse(s)) for s in (ALIAS, MODX)}
print(f"   modx's check on a pool: s3i accepts the alias {v['s3i'][ALIAS]} and its modx {v['s3i'][MODX]}; "
      f"the fix: {v['fix'][ALIAS]} and {v['fix'][MODX]}")
print(f"({time.time() - t0:.0f} s)")
