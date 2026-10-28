"""DEEP.md §2: a candidate round, or one routine, run with a tiny core's learned library on its answer rows of T = 1
(core_f.py's 400, seed 1, the task's answers by Python's integers), on the file's own machine (run_prog at the
round's places, as round_ok runs it): how many rows it gets right, and the sizes of the rows it gets wrong.

    cd cpu_first && PYTHONPATH=. python3 integration/legal_deep/checks/try_round.py RECORD.json TASK 'PROGRAM' [WANT]
    WANT: a Python expression in x and N for the program's value in A (default: the task's answer)
"""
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU = HERE.parents[2]
sys.path.insert(0, str(CPU / "learned_remainder" / "chain_local" / "stub"))
import torch  # noqa: E402

H1_DIR = os.environ.get("H1_DATA", str(CPU.parent / "one-layer-deeper" / "data" / "generated" /
                                       "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
TASKS = {"three2": "(3*x*x+2)%N", "plus3": "(x*x+3)%N", "h1": "x*x%N", "cube": "x**3%N", "ten": "10*x*x%N",
         "tenrho": "(10*x*x+1)%N", "rho": "(x*x+1)%N", "minus1": "(x*x-1)%N", "plusy": "(x*x+x)%N"}


def load_rows(n, seed, T=None):
    rows = [json.loads(line) for line in open(os.path.join(H1_DIR, "train.jsonl"))]
    if T is not None:
        rows = [r for r in rows if r["time_steps"] == T]
    g = torch.Generator().manual_seed(seed)
    return [rows[i] for i in torch.randperm(len(rows), generator=g)[:n].tolist()]


def digits_of(vals, L_):
    return torch.tensor([[int(ch) for ch in reversed(str(v))][:L_] + [0] * (L_ - min(L_, len(str(v)))) for v in vals])


def value(t):
    return [sum(int(v) * 10 ** i for i, v in enumerate(r)) for r in t.tolist()]


def main():
    rec_path, task, prog = sys.argv[1:4]
    want_expr = sys.argv[4] if len(sys.argv) > 4 else TASKS.get(task)
    if task.startswith("poly_") and want_expr is None:
        a, k, b, c = (int(v) for v in task.split("_")[1:])
        want_expr = f"({a}*x**{k}+{b}*x+{c})%N"
    mod = types.ModuleType("deep")
    sys.modules["deep"] = mod
    path = os.environ.get("FILE", str(HERE.parent / "submission_stage3_deep.py"))
    exec(compile(open(path).read(), path, "exec"), mod.__dict__)
    M = mod
    rec = json.load(open(rec_path))
    lib = {k: M.parse(v) for k, v in rec["lib"].items()}
    rows = load_rows(int(os.environ.get("ANSWERS", 400)), seed=1, T=1)
    xs, Ns = [int(r["x"]) for r in rows], [int(r["modulus"]) for r in rows]
    want = [eval(want_expr, {"x": x, "N": N}) for x, N in zip(xs, Ns)]
    P = M.PR
    st = M.round_settings(M.Cells(10, "cpu"), P)
    A, _ = M.run_prog(st, M.parse(prog), lib, {"X": M.pad(digits_of(xs, 8), P), "N": M.pad(digits_of(Ns, 8), P)},
                      M.gen_on("cpu", 1), "cpu")
    got = value(A)
    ok = [g == w for g, w in zip(got, want)]
    print(f"{task}: {prog}\n  right on {sum(ok)} of {len(ok)} rows (want {want_expr})")
    bad = [(x, N, g, w) for x, N, g, w, o in zip(xs, Ns, got, want, ok) if not o]
    if bad:
        digs = sorted({len(str(x)) for x, *_ in bad})
        good_digs = sorted({len(str(x)) for x, o in zip(xs, ok) if o})
        print(f"  wrong where x has {digs} digits (right where it has {good_digs}); e.g.",
              [(x, N, g, w) for x, N, g, w in bad[:3]])


if __name__ == "__main__":
    main()
