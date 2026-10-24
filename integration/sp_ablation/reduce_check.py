"""A reduction a run practised, against the truth (SP_ABLATION.md §4): runs PROGRAM with the record's own routines (its
LIBRARY line, exact cells) on the dividend x + q N in one register and N in another, at growing sizes, as the
reduction's practice problem draws them (x < N, q below 10^W), and says where it is right. Adapted from
integration/ablation_s3/reduce_check.py (which runs s3i's no_step program). Local: CPU, seconds, under 1 GB.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/sp_ablation/reduce_check.py RECORD PROGRAM DIVIDEND MODULUS
    e.g. ... records/no_law_mul_h100_seed1.json "CLA P[BRG.X T.add.pair.AN T.red.x+dn.AN]" X N
"""

import ast
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CPU_FIRST = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(CPU_FIRST, "wider_menus", "local"))
import local_wide as LW  # noqa: E402

SP2 = os.path.join(CPU_FIRST, "integration", "self_practice", "submission_stage3_sp2.py")


def library(record):
    """The run's routines: the LIBRARY line its climb prints at the end, or, where that is empty (a run with no
    round keeps no library), every level's sleep line (name=program; ...) in order."""
    rec = json.load(open(record))
    lines = rec["stdout"].splitlines()
    lib = ast.literal_eval(next(l for l in lines if l.startswith("LIBRARY "))[len("LIBRARY "):])
    if lib:
        return lib
    for l in lines:
        m = re.match(r"LR climb step=\d+ L\d+ sleep: (.*) \(\d+\.\d+s\)$", l)
        if m and "=" in m.group(1):
            lib += [tuple(d.split("=", 1)) for d in m.group(1).split("; ")]
    return lib


def main(argv):
    record, program, dividend, modulus = argv
    sc = LW.load_module(SP2)
    cells = sc.exact_cells("cpu")
    lib = {k: sc.parse(v) for k, v in library(record)}
    gen = torch.Generator().manual_seed(0)
    digits = lambda v, P: torch.stack([(v // 10 ** j) % 10 for j in range(P)], 1)
    print(f"{program} on {dividend} = x + q N, {modulus} = N, with {os.path.basename(record)}'s {len(lib)} routines:")
    for W in (1, 2, 3, 4, 8):
        st = sc.Setting(10, W, cells)
        n = 400
        N = torch.randint(10 ** (W - 1) if W > 1 else 2, 10 ** W, (n,), generator=gen)
        x = (torch.rand(n, generator=gen) * N).long()
        q = (torch.rand(n, generator=gen) * 10 ** W).long()
        A, _ = sc.run_prog(st, sc.parse(program), lib, {dividend: digits(x + q * N, st.P), modulus: digits(N, st.P)},
                           sc.gen_on("cpu", 1), "cpu")
        ok = sum(A[:, j].long() * 10 ** j for j in range(st.P)) == x
        wrong = q[~ok]
        print(f"  W = {W}: right on {int(ok.sum())}/{n}" + (f", wrong from q = {int(wrong.min())}" if wrong.numel()
                                                           else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
