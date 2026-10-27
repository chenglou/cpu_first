"""A tiny core's answer programs on both of H1's ladders (WIDE.md §4): the record's library, its rounds as T.round<i>
and each answer program it found, run by the file's own answer machine (outer_machine, as search_outer checks a
program) on every row of depth_t_<T> (the seen moduli) and depth_ood_n_t_<T> (the unseen ones), T = 1 to 64, 768 rows
a rung, against each row's own answer, read. Exact cells. CUDA if there is one, else the CPU.

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 ~/github/sotaku/venv/bin/python integration/legal_wide/checks/ladder_core.py RECORD.json [FILE]
"""
import json
import os
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDE = HERE.parent
CPU = WIDE.parents[1]
sys.path.insert(0, str(CPU / "learned_remainder" / "chain_local" / "stub"))   # the evaluator's API
import torch  # noqa: E402

H1_DIR = os.environ.get("H1_DATA", str(CPU.parent / "one-layer-deeper" / "data" / "generated" /
                                       "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
RUNGS = (1, 2, 4, 8, 16, 32, 64)


def digits(rows, key, L_):
    return torch.tensor([[int(ch) for ch in reversed(str(r[key]))][:L_] + [0] * (L_ - min(L_, len(str(r[key]))))
                         for r in rows])


def main():
    rec = json.load(open(sys.argv[1]))
    path = sys.argv[2] if len(sys.argv) > 2 else str(WIDE / "submission_stage3_wide.py")
    mod = types.ModuleType("combined")
    sys.modules["combined"] = mod
    exec(compile(open(path).read(), path, "exec"), mod.__dict__)
    M = mod
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lib = {k: M.parse(v) for k, v in rec["lib"].items()}
    for name, prog in rec["outer"]["rounds"].items():
        lib[name] = M.parse(prog)
    st = M.round_settings(M.Cells(10, dev), M.PR)
    print(f"{sys.argv[1]}: rounds {rec['outer']['rounds']}", flush=True)
    bad = 0
    for p in rec["outer"]["programs"]:
        prog = M.parse(p)
        for name in ("depth_t", "depth_ood_n_t"):
            for T in RUNGS:
                t0 = time.time()
                rows = [json.loads(line) for line in open(os.path.join(H1_DIR, f"{name}_{T}.jsonl"))]
                Xd, Nd, Yd = (digits(rows, k, M.W).to(dev) for k in ("x", "modulus", "result"))
                Cd = digits(rows, "time_steps", M.K).to(dev)
                m = M.outer_machine(st, prog, lib, Xd.shape[0], dev)
                A = m.run(Xd, Nd, Cd, M.gen_on(dev, 1))
                ok = (A[:, :Yd.shape[1]] == Yd).all(1)
                bad += int((~ok).sum())
                print(f"  {p} {name}_{T}: {int(ok.sum())}/{len(rows)} right ({time.time() - t0:.0f} s)", flush=True)
    print(f"done: {bad} rows wrong", flush=True)


if __name__ == "__main__":
    main()
