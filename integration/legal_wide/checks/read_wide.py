"""The wide reader on a real climb's state (WIDE.md §3), local, CPU. The legal file's tiny core
(combined_legal/records/core_legal.json) at the start of its level 3, where the comparison, the link and the addition
are routines, loaded into the wide file's SelfClimb. Exact cells.

  MODE=read     the law read on ANSWERS (400) answer rows of T = 1 (core_c.py's rows): its rows, the candidates
                weighed, seconds, and the pairs it poses
  MODE=breadth  the same rows with the answers of other formulas of the family, one T, made here by Python's integers
                (a test of the grammar's reach, outside any training run): which law each reads, the candidates weighed
                and seconds (FORMULAS: a comma list of names, default all)
  MODE=pose     the pose phase on that state, then each posed law's answers on its draws against Python's integers

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 MODE=read ~/github/sotaku/venv/bin/python integration/legal_wide/checks/read_wide.py
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

FILE = os.environ.get("FILE", str(WIDE / "submission_stage3_wide.py"))
REC = CPU / "integration" / "combined_legal" / "records" / "core_legal.json"
H1_DIR = os.environ.get("H1_DATA", str(CPU.parent / "one-layer-deeper" / "data" / "generated" /
                                       "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
env = os.environ.get
FORMULAS = {                         # one round of each (the family's tasks of SP_BREADTH.md §1 on H1's prompts)
    "h1": lambda x, N: x * x % N,
    "cube": lambda x, N: x ** 3 % N,
    "rho": lambda x, N: (x * x + 1) % N,
    "plus3": lambda x, N: (x * x + 3) % N,
    "minus1": lambda x, N: (x * x - 1) % N,
    "ten": lambda x, N: 10 * x * x % N,
    "tenrho": lambda x, N: (10 * x * x + 1) % N,
    "xplusx2": lambda x, N: (x * x + x) % N,
}


def load_rows(n, seed, T=None):
    """core_c.py's rows."""
    rows = [json.loads(line) for line in open(os.path.join(H1_DIR, "train.jsonl"))]
    if T is not None:
        rows = [r for r in rows if r["time_steps"] == T]
    g = torch.Generator().manual_seed(seed)
    return [rows[i] for i in torch.randperm(len(rows), generator=g)[:n].tolist()]


def digits_of(vals, L_=8):
    return torch.tensor([[int(ch) for ch in reversed(str(v))][:L_] + [0] * (L_ - min(L_, len(str(v)))) for v in vals])


def load():
    mod = types.ModuleType("combined")
    sys.modules["combined"] = mod
    exec(compile(open(FILE).read(), FILE, "exec"), mod.__dict__)
    return mod


def climb_at_level3(M, dev):
    rec = json.load(open(REC))
    c = M.SelfClimb(M.Cells(10, dev), dev, M.PR, chunk=1 << 18, **M.SP)
    c.lib = {k: M.parse(v) for k, v in rec["history"][2]["lib"].items()}
    for key, toks in rec["ops"].items():
        law, pat = key.split("@")
        for tok, bd in toks:
            if tok in c.lib:
                c.ops.add(law, pat, tok, tuple(bd))
    for h in rec["history"][:3]:
        for n in h["picks"]:
            c.practiced[n] = h["level"]
    c.level = 3
    return c


def value(t):
    return [sum(int(v) * 10 ** i for i, v in enumerate(r)) for r in t.tolist()]


def main():
    dev = "cpu"
    M = load()
    arows = load_rows(int(env("ANSWERS", 400)), seed=1, T=1)
    xs, Ns = [int(r["x"]) for r in arows], [int(r["modulus"]) for r in arows]
    c = climb_at_level3(M, dev)
    print("book:", c.laws, c.patterns, "| routines on the pair:",
          {law: c.ops.count(law, "pair") for law in ("lt", "red", "add")}, flush=True)
    mode = env("MODE", "read")
    if mode in ("read", "breadth"):
        names = ["h1"] if mode == "read" else env("FORMULAS", ",".join(FORMULAS)).split(",")
        for name in names:
            ys = [FORMULAS[name](x, N) for x, N in zip(xs, Ns)] if name != "h1" else [int(r["result"]) for r in arows]
            ans = (digits_of(Ns), digits_of(xs), digits_of(ys))
            t0 = time.time()
            r = M.LawReader(c, ans)
            law, ops_, n = r.read()
            print(f"{name}: read {law} (operations {ops_}) on {n} of {len(xs)} rows; {r.tried} candidates weighed; "
                  f"{time.time() - t0:.1f} s; poses {M.read_pairs(ops_)}", flush=True)
        return
    rows = load_rows(4096, seed=0)
    rows = (digits_of([int(r["modulus"]) for r in rows]), digits_of([int(r["x"]) for r in rows]))
    ans = (digits_of(Ns), digits_of(xs), digits_of([int(r["result"]) for r in arows]))
    t0 = time.time()
    pool, roots = c.pose(rows, ans)
    print(f"pose: {len(pool)} candidates on {len(roots)} roots, {time.time() - t0:.1f} s", flush=True)
    print("record:", [x for x in c.records if x.get("task") == "read"], flush=True)
    bad = n = 0
    for r_ in roots:
        for g in r_.goal_list:
            for d, t in zip(r_.draws, g.targets):
                if "A" not in t:
                    continue
                ins = [value(o) for o in d.operands(c.ops, c.lib)]
                got, m = value(t["A"]), t["Am"].tolist()
                want = {"red": lambda a, b: a % b, "mul": lambda a, b: a * b}.get(g.name)
                want = [want(*v) for v in zip(*ins)] if want else [a + b * e for a, b, e in zip(*ins)]
                n += 1
                bad += not all(w == x for w, x, k in zip(want, got, m) if k)
    print(f"done in {time.time() - t0:.1f} s; {n} targets checked against Python's integers, {bad} disagree",
          flush=True)


if __name__ == "__main__":
    main()
