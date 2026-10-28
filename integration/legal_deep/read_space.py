"""DEEP.md §3: the file's law reader on every task of the space y <- a y^k + b y + c mod N (k 2, 3; a 1..9; b, c
0..9), each task's answers made by Python's integers on H1's answer rows of T = 1 (core_f.py's 400, seed 1; the
first ROWS of them). The reader is the file's own (LawReader.read: the chains, the gap constants, then x's
polynomials), with the learner's routines of the legal tiny core at the start of its level 3, as
legal_wider/checks/wider_local.py loads them. Per task: the law read, its rows, the laws tried, the seconds. Writes
OUT/<TAG>.jsonl, a line a task, as it goes.

    env: FILE, ROWS (141), PART (i/n: every n-th task from the i-th), RETRY (a second ROWS for the tasks the first
         misses), TAG, OUT
    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/legal_deep/read_space.py
"""
import json
import os
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU = HERE.parents[1]
sys.path.insert(0, str(CPU / "learned_remainder" / "chain_local" / "stub"))   # the evaluator's API
import torch  # noqa: E402

env = os.environ.get
H1_DIR = env("H1_DATA", str(CPU.parent / "one-layer-deeper" / "data" / "generated" /
                            "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
REC = CPU / "integration" / "combined_legal" / "records" / "core_legal.json"


def load_rows(n, seed, T=None):
    rows = [json.loads(line) for line in open(os.path.join(H1_DIR, "train.jsonl"))]
    if T is not None:
        rows = [r for r in rows if r["time_steps"] == T]
    g = torch.Generator().manual_seed(seed)
    return [rows[i] for i in torch.randperm(len(rows), generator=g)[:n].tolist()]


def digits_of(vals, dev, L_=8):
    return torch.tensor([[int(ch) for ch in reversed(str(v))][:L_] + [0] * (L_ - min(L_, len(str(v)))) for v in vals],
                        device=dev)


def climb_at_level3(M, dev):
    rec = json.load(open(REC))
    c = M.SelfClimb(M.Cells(10, dev), dev, M.PR, chunk=1 << 21 if dev == "cuda" else 1 << 18, **M.SP)
    c.lib = {k: M.parse(v) for k, v in rec["history"][2]["lib"].items()}
    for key, toks in rec["ops"].items():
        law, pat = key.split("@")
        for tok, bd in toks:
            if tok in c.lib:
                c.ops.add(law, pat, tok, tuple(bd))
    return c


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = types.ModuleType("deep")
    sys.modules["deep"] = mod
    path = env("FILE", str(HERE / "submission_stage3_deep.py"))
    exec(compile(open(path).read(), path, "exec"), mod.__dict__)
    M = mod
    c = climb_at_level3(M, dev)
    st = M.round_settings(c.cells, c.P_round)
    arows = load_rows(400, seed=1, T=1)
    i0, n_ = (int(v) for v in env("PART", "0/1").split("/"))
    space = [(a, k, b, cc) for k in (2, 3) for a in range(1, 10) for b in range(10) for cc in range(10)][i0::n_]
    out = Path(env("OUT", str(HERE / "records"))) / f"{env('TAG', 'read_space')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tries = [int(env("ROWS", 141))] + ([int(env("RETRY"))] if env("RETRY") else [])
    print("device", torch.cuda.get_device_name() if dev == "cuda" else "cpu", "tasks", len(space), "rows", tries,
          "READ_POLY", M.READ_POLY, flush=True)
    with open(out, "w") as f:
        for a, k, b, cc in space:
            rec = dict(task=f"poly_{a}_{k}_{b}_{cc}")
            for n in tries:
                rows = arows[:n]
                xs, Ns = [int(r["x"]) for r in rows], [int(r["modulus"]) for r in rows]
                ys = [(a * x ** k + b * x + cc) % N for x, N in zip(xs, Ns)]
                ans = (digits_of(Ns, dev), digits_of(xs, dev), digits_of(ys, dev))
                t0 = time.time()
                r = M.LawReader(c.ops, c.lib, st, dev, ans)
                law, ops, got = r.read()
                if dev == "cuda":
                    torch.cuda.synchronize()
                rec[f"r{n}"] = dict(law=law, rows=got, tried=r.tried, seconds=round(time.time() - t0, 2))
                if law:
                    break
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
