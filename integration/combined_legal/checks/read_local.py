"""The reading on a real climb's state (LEGAL_FILE.md §3), local, CPU. fast2's tiny core (combined_fast/records/
core_fast2.json) at the start of its level 3, where the comparison, the link and the addition are routines, loaded
into the legal file's SelfClimb (its three-law book). Exact cells.

  MODE=read  the law read on ANSWERS (400) answer rows of T = 1 (core_c.py's rows): its rows, the laws tried, seconds,
             and the pairs it poses
  MODE=pose  the pose phase on that state, then each posed law's answers on its draws against Python's integers, in
             every binding

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 MODE=read ~/github/sotaku/venv/bin/python integration/combined_legal/checks/read_local.py
"""
import json
import os
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEGAL = HERE.parent
CPU = LEGAL.parents[1]
sys.path.insert(0, str(CPU / "learned_remainder" / "chain_local" / "stub"))   # the evaluator's API
import torch  # noqa: E402

FILE = os.environ.get("FILE", str(LEGAL / "submission_stage3_legal.py"))
REC = CPU / "integration" / "combined_fast" / "records" / "core_fast2.json"
H1_DIR = os.environ.get("H1_DATA", str(CPU.parent / "one-layer-deeper" / "data" / "generated" /
                                       "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
env = os.environ.get


def load_rows(n, seed, T=None):
    """core_c.py's rows."""
    rows = [json.loads(line) for line in open(os.path.join(H1_DIR, "train.jsonl"))]
    if T is not None:
        rows = [r for r in rows if r["time_steps"] == T]
    g = torch.Generator().manual_seed(seed)
    return [rows[i] for i in torch.randperm(len(rows), generator=g)[:n].tolist()]


def digits(rows, key, L_=8):
    return torch.tensor([[int(ch) for ch in reversed(str(r[key]))][:L_] + [0] * (L_ - min(L_, len(str(r[key]))))
                         for r in rows])


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
    ans = tuple(digits(arows, k) for k in ("modulus", "x", "result"))
    rows = load_rows(4096, seed=0)
    rows = (digits(rows, "modulus"), digits(rows, "x"))
    c = climb_at_level3(M, dev)
    print("book:", c.laws, c.patterns, "| routines on the pair:",
          {law: c.ops.count(law, "pair") for law in ("lt", "red", "add")}, flush=True)
    t0 = time.time()
    if env("MODE", "read") == "read":
        r = M.LawReader(c, ans)
        law, ops_, n = r.read()
        xs = value(ans[1])
        print(f"read: {law} (operations {ops_}) on {n} of {len(xs)} answer rows; {r.tried} laws tried; "
              f"{time.time() - t0:.1f} s; rows with x < 1,000: {sum(x < 1000 for x in xs)}", flush=True)
        print("poses:", M.read_pairs(ops_), flush=True)
        return
    pool, roots = c.pose(rows, ans)
    print(f"pose: {len(pool)} candidates on {len(roots)} roots, {time.time() - t0:.1f} s", flush=True)
    print("record:", [x for x in c.records if x.get("task") == "read"], flush=True)
    print("pool:", pool, flush=True)
    bad = 0
    for r_ in roots:
        for g in r_.goal_list:
            if g.name in M.BOOK[0] and r_.pattern == "pair":
                continue
            for d, t in zip(r_.draws, g.targets):
                if "A" not in t:
                    continue
                ins = [value(o) for o in d.operands(c.ops, c.lib)]
                got, m = value(t["A"]), t["Am"].tolist()
                if g.name == "red":
                    want = [a % b for a, b in zip(*ins)]
                elif g.name == "mul":
                    want = [a * b for a, b in zip(*ins)]
                elif g.name == "add.mul":
                    want = [a + b * e for a, b, e in zip(*ins)]
                else:
                    raise ValueError(g.name)
                ok = all(w == x for w, x, k in zip(want, got, m) if k)
                bad += not ok
                print(f"  {g.name} on {r_.key} at {d.st}: {sum(m)} probes answered; the book's construction "
                      f"{'agrees' if ok else 'DISAGREES'} with Python's integers", flush=True)
    print(f"done in {time.time() - t0:.1f} s; {bad} disagreements", flush=True)


if __name__ == "__main__":
    main()
