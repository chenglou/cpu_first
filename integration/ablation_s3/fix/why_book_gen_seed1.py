"""Why book_gen's seed-1 climb changed with the fix (ABLATION_S3.md §8): on that seed min on the pair is practised a
level before red on the pair, so "T.min.pair.Ar CLA SEL.S" (min leaves x in S and the borrow set; then A := S where
the borrow is set, else 0: "x if x < N, else 0") costs 3, as the conditional subtraction does. The original check
accepts it, so it joined the corpus among each pick's spellings and BPE merged "CLA SEL.S"; the fixed check rejects it.
Both accept the conditional subtraction itself. Run on three pools of H1's rows, with the library of the original
run's round_found line and exact cells. Local: CPU, seconds, under 1 GB.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/fix/why_book_gen_seed1.py
"""

import ast
import json
import os
import sys
import types

import torch

CPU_FIRST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(CPU_FIRST, "wider_menus", "local"))
import local_wide as LW  # noqa: E402

DATA = os.path.expanduser("~/github/one-layer/one-layer-deeper/data/generated/"
                          "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2/train.jsonl")
orig = LW.load_module(os.path.join(CPU_FIRST, "integration/sp_ablation/variants/book_gen.py"))
fix = LW.load_module(os.path.join(CPU_FIRST, "integration/sp_ablation/fix/book_gen_fix.py"))
rec = json.load(open(os.path.join(CPU_FIRST, "integration/sp_ablation/records/book_gen_b200_seed1.json")))
line = next(l for l in rec["stdout"].splitlines() if l.startswith("LR round_found"))
lib_txt = ast.literal_eval(line[line.index("library: ") + 9:].strip())
rows = [json.loads(l) for l in open(DATA)]
digits = lambda v: torch.stack([(v // 10 ** j) % 10 for j in range(8)], 1)
ND = digits(torch.tensor([r["modulus"] for r in rows]))
XD = digits(torch.tensor([r["x"] for r in rows]))
gen = torch.Generator().manual_seed(5)
for trial in range(3):
    p = torch.randint(0, len(rows), (4096,), generator=gen)
    out = []
    for mod in (orig, fix):
        lib = {k: mod.parse(v) for k, v in lib_txt.items()}
        ops = mod.Ops()
        for k in lib_txt:                                      # the routines, in the order practised
            if k.startswith("T.") and k.count(".") >= 3:
                parts = k[2:].split(".")
                ops.add(".".join(parts[:-2]), parts[-2], k, tuple(parts[-1]))
        climb = mod.SelfClimb(mod.exact_cells("cpu"), "cpu", mod.PR)
        res = {}
        for r in "XYN":
            settings = climb.verify + (climb.above if mod is fix else [])
            acc = mod.Acceptor("red", types.SimpleNamespace(pattern="pair", binding=("A", r)), settings,
                               (ND[p], XD[p]), ops, lib, "cpu")
            res[r] = acc.many([mod.parse(f"T.min.pair.A{r} CLA SEL.S"), mod.parse(f"STA.S T.lt.pair.A{r} SEL.S")], lib)
        out.append(res)
    print(f"pool {trial}: red on the pair in A against r = X, Y, N: [the alias T.min.pair.Ar CLA SEL.S, the routine "
          f"STA.S T.lt.pair.Ar SEL.S] accepted by the original {out[0]}, by the fix {out[1]}")
