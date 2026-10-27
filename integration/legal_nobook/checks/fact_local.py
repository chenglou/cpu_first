"""The no-book file's first readings (NOBOOK.md §3), local, CPU, exact cells. A fresh climb (no routine, an empty book)
with 400 answer rows of T = 1 (core_c.py's rows):
  1. level 0: read_fact's decision and the rows it read, and what the pose poses;
  2. the comparison and the link as routines (the legal tiny core's, combined_legal/records/core_legal.json): the pose;
  3. the addition too: the pose, with the wide reading in it.
For each pose: the laws posed by pattern, and each posed law's answers on its draws against Python's integers.

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 ~/github/sotaku/venv/bin/python integration/legal_nobook/checks/fact_local.py
"""
import json
import os
import sys
import time
import types
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOBOOK = HERE.parent
CPU = NOBOOK.parents[1]
sys.path.insert(0, str(CPU / "learned_remainder" / "chain_local" / "stub"))   # the evaluator's API
import torch  # noqa: E402

FILE = os.environ.get("FILE", str(NOBOOK / "submission_stage3_nobook.py"))
REC = CPU / "integration" / "combined_legal" / "records" / "core_legal.json"
H1_DIR = os.environ.get("H1_DATA", str(CPU.parent / "one-layer-deeper" / "data" / "generated" /
                                       "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))


def load_rows(n, seed, T=None):
    rows = [json.loads(line) for line in open(os.path.join(H1_DIR, "train.jsonl"))]
    if T is not None:
        rows = [r for r in rows if r["time_steps"] == T]
    g = torch.Generator().manual_seed(seed)
    return [rows[i] for i in torch.randperm(len(rows), generator=g)[:n].tolist()]


def digits(rows, key, L_=8):
    return torch.tensor([[int(ch) for ch in reversed(str(r[key]))][:L_] + [0] * (L_ - min(L_, len(str(r[key]))))
                         for r in rows])


def value(t):
    return [sum(int(v) * 10 ** i for i, v in enumerate(r)) for r in t.tolist()]


def check(M, c, roots):
    """Each posed law's answers on its draws against Python's integers: (draws checked, disagreements)."""
    n = bad = 0
    for r_ in roots:
        for g in r_.goal_list:
            for d, t in zip(r_.draws, g.targets):
                ins = [value(o) for o in d.operands(c.ops, c.lib)]
                if "A" in t:
                    got, m = value(t["A"]), t["Am"].tolist()
                    f = {"red": lambda a, b: a % b if a < 2 * b else None, "mul": lambda a, b: a * b,
                         "add": lambda a, b: a + b}.get(g.name, lambda a, b, e: a + b * e)
                    want = [f(*v) for v in zip(*ins)]
                    ok = all(w is None or w == x for w, x, k in zip(want, got, m) if k)
                if "c" in t:
                    got, m = t["c"].tolist(), t["cm"].tolist()
                    ok = all(int(a < b) == x for a, b, x, k in zip(*ins, got, m) if k)
                n += 1
                bad += not ok
    return n, bad


def main():
    dev = "cpu"
    mod = types.ModuleType("combined")
    sys.modules["combined"] = mod
    exec(compile(open(FILE).read(), FILE, "exec"), mod.__dict__)
    M = mod
    arows = load_rows(400, seed=1, T=1)
    ans = tuple(digits(arows, k) for k in ("modulus", "x", "result"))
    rows = load_rows(4096, seed=0)
    rows = (digits(rows, "modulus"), digits(rows, "x"))
    rec = json.load(open(REC))
    lib = {k: M.parse(v) for k, v in rec["history"][2]["lib"].items()}
    c = M.SelfClimb(M.Cells(10, dev), dev, M.PR, chunk=1 << 18, **M.SP)
    print("book:", c.laws, c.patterns, "| STEP_PARTS", M.STEP_PARTS, "STEP_ADD", M.STEP_ADD, flush=True)
    for stage, laws in (("level 0, no routine", ()), ("the comparison and the link", ("lt", "red")),
                        ("and the addition", ("lt", "red", "add"))):
        c.lib = {k: v for k, v in lib.items() if not k.startswith("T.") or k.split(".")[1] in laws}
        c.ops = M.Ops()
        for key, toks in rec["ops"].items():
            law, pat = key.split("@")
            for tok, bd in toks:
                if law in laws and pat == "pair" and tok in c.lib:
                    c.ops.add(law, pat, tok, tuple(bd))
        c.practiced = {f"{law}/pair@{b}": 0 for law in laws for b in ("AX", "AY", "AN", "XY", "XN", "YX", "YN",
                                                                         "NX", "NY", "XA", "YA", "NA")}
        n0 = len(c.records)
        t0 = time.time()
        pool, roots = c.pose(rows, ans)
        new = c.records[n0:]
        for r in new:
            show = dict(r, rows=f"{len(r['rows'])} rows, first {r['rows'][:3]}") if r["task"] == "fact" else r
            print(f"  [{stage}] record: {show}", flush=True)
        print(f"  [{stage}] pose: {len(pool)} candidates on {len(roots)} roots, {time.time() - t0:.1f} s; "
              f"{dict(Counter(p.split('@')[0] for p in pool))}", flush=True)
        n, bad = check(M, c, roots)
        print(f"  [{stage}] {n} draws' answers checked against Python's integers, {bad} disagree", flush=True)


if __name__ == "__main__":
    main()
