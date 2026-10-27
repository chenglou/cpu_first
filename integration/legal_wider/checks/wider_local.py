"""The wider file's posing and ROUNDT (WIDER.md §3; legal_family/checks/family_local.py for this file), local, CPU, exact cells. The legal tiny core's state at
the start of its level 3, where the comparison, the link and the addition are routines (combined_legal/records/
core_legal.json), loaded into the family file's SelfClimb.

  MODE=pose    (default) for each task, its answers on H1's 400 answer rows of T = 1 made by Python's integers (a test
               outside any run): the law read, what the reading poses (its cuts and loops, constants and windows), and
               each posed law's answers on its draws against Python's integers, with how many rows each covers
  MODE=roundt  ROUNDT on the training data's rows of T = 2, 4 and 8 (the file's ROUNDT of each, as core_f.py draws
               them): phase B's tiny core's rounds pass, and a wrong answer or a round right at T = 1 only fails
  TASKS        a comma list of task names (default all)

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 ~/github/sotaku/venv/bin/python integration/legal_wider/checks/wider_local.py
"""
import json
import os
import sys
import time
import types
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAM = HERE.parent
CPU = FAM.parents[1]
sys.path.insert(0, str(CPU / "learned_remainder" / "chain_local" / "stub"))   # the evaluator's API
import torch  # noqa: E402

FILE = os.environ.get("FILE", str(FAM / "submission_stage3_wider.py"))
REC = CPU / "integration" / "combined_legal" / "records" / "core_legal.json"
CORE_B = CPU / "integration" / "legal_nobook" / "records" / "t_nobook.json"
H1_DIR = os.environ.get("H1_DATA", str(CPU.parent / "one-layer-deeper" / "data" / "generated" /
                                       "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
env = os.environ.get
TASKS = {                            # one round of each: SP_BREADTH's family, and the two held out (PREREG.md)
    "h1": lambda x, N: x * x % N,
    "cube": lambda x, N: x ** 3 % N,
    "ten": lambda x, N: 10 * x * x % N,
    "rho": lambda x, N: (x * x + 1) % N,
    "plus3": lambda x, N: (x * x + 3) % N,
    "minus1": lambda x, N: (x * x - 1) % N,
    "tenrho": lambda x, N: (10 * x * x + 1) % N,
    "plusy": lambda x, N: (x * x + x) % N,
    "three2": lambda x, N: (3 * x * x + 2) % N,
}
for _n in os.environ.get("TASKS", "").split(","):     # held-out tasks: poly_A_K_B_C, y <- A y^K + B y + C mod N
    if _n.startswith("poly_"):
        _a, _k, _b, _c = (int(v) for v in _n.split("_")[1:])
        TASKS[_n] = lambda x, N, _a=_a, _k=_k, _b=_b, _c=_c: (_a * x ** _k + _b * x + _c) % N
PY = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
      "pow": lambda a, b: a ** b if 1 <= b <= 8 else None, "mod": lambda a, b: a % b if b else None}


def load_rows(n, seed, T=None):
    """core_c.py's rows."""
    rows = [json.loads(line) for line in open(os.path.join(H1_DIR, "train.jsonl"))]
    if T is not None:
        rows = [r for r in rows if r["time_steps"] == T]
    g = torch.Generator().manual_seed(seed)
    return [rows[i] for i in torch.randperm(len(rows), generator=g)[:n].tolist()]


def digits_of(vals, L_=8):
    return torch.tensor([[int(ch) for ch in reversed(str(v))][:L_] + [0] * (L_ - min(L_, len(str(v)))) for v in vals])


def value(t):
    return [sum(int(v) * 10 ** i for i, v in enumerate(r)) for r in t.tolist()]


def load():
    mod = types.ModuleType("family")
    sys.modules["family"] = mod
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


def py_tree(t, env_, n, cap):
    """A tree's value by Python's integers on n rows, and where every value on the way is in [0, cap)."""
    if isinstance(t, str):
        return (env_[t] if t in env_ else [int(t)] * n), [True] * n
    (a, ma), (b, mb) = (py_tree(u, env_, n, cap) for u in t[1])
    v = [PY[t[0]](p, q) if k and l else None for p, q, k, l in zip(a, b, ma, mb)]
    return v, [w is not None and 0 <= w < cap for w in v]


def check_pose(M, c, roots):
    """{law/pattern: [draws, disagreements, rows the file answers, rows Python can]}."""
    out = {}
    for r_ in roots:
        for g in r_.goal_list:
            for d, t in zip(r_.draws, g.targets):
                if "A" not in t:
                    continue
                ins = [value(o) for o in d.operands(c.ops, c.lib)]
                got, m = value(t["A"]), t["Am"].tolist()
                if g.name in M.READ_LAWS:
                    tr, n = M.READ_LAWS[g.name], len(got)
                    env_ = {"a": ins[0], "N": ins[-1]}
                    if len(ins) == 3:
                        env_["b"] = ins[1]
                    want, ok = py_tree(tr, env_, n, 10 ** d.st.P)
                else:
                    want = {"red": lambda a, b: a % b, "mul": lambda a, b: a * b, "add.add": lambda a, b, e: a + b + e,
                            "add.mul": lambda a, b, e: a + b * e}[g.name]
                    want = [want(*v) for v in zip(*ins)]
                    ok = [True] * len(got)
                key = f"{g.name}/{r_.pattern}"
                s = out.setdefault(key, [0, 0, 0, 0])
                s[0] += 1
                s[1] += not all(w == x for w, x, k in zip(want, got, m) if k)
                s[2] += sum(m)
                s[3] += sum(ok)
    return out


def pose_mode(M, dev):
    arows = load_rows(int(env("ANSWERS", 400)), seed=1, T=1)
    xs, Ns = [int(r["x"]) for r in arows], [int(r["modulus"]) for r in arows]
    rows = load_rows(4096, seed=0)
    rows = (digits_of([int(r["modulus"]) for r in rows]), digits_of([int(r["x"]) for r in rows]))
    book0 = dict(M.LAW_BOOK)
    for name in env("TASKS", ",".join(TASKS)).split(","):
        M.READ_LAWS.clear()
        M.LAW_BOOK.clear()
        M.LAW_BOOK.update(book0)
        c = climb_at_level3(M, dev)
        ys = [TASKS[name](x, N) for x, N in zip(xs, Ns)]
        if name == "h1":
            assert ys == [int(r["result"]) for r in arows]
        ans = (digits_of(Ns), digits_of(xs), digits_of(ys))
        t0 = time.time()
        pool, roots = c.pose(rows, ans)
        rec = [x for x in c.records if x.get("task") == "read"][-1]
        fact = [x for x in c.records if x.get("task") == "fact"][-1]
        print(f"{name}: fact {fact['field']} {fact['parts']}; read {rec['law']} on {rec['rows']} of {rec['answers']} "
              f"rows, {rec['tried']} laws, {rec['seconds']} s; its own {rec['more']}", flush=True)
        print(f"  pose: {len(pool)} candidates on {len(roots)} roots, {time.time() - t0:.1f} s; "
              f"{dict(Counter(p.split('@')[0] for p in pool))}", flush=True)
        for key, (n, bad, got, can) in check_pose(M, c, roots).items():
            print(f"  {key}: {n} draws, {bad} disagree with Python; the file answers {got} rows, Python's integers "
                  f"stay in range on {can}", flush=True)


def roundt_mode(M, dev):
    """ROUNDT: phase B's rounds (H1's, from its tiny core) against the data's kept rows of T = 2, 4, 8; then a round
    that is x^2 mod N at T = 1 but drops the reduction where it matters at T > 1 would fail (a corrupted answer)."""
    core = json.load(open(CORE_B))
    c = climb_at_level3(M, dev)
    lib = {k: M.parse(v) for k, v in core["lib"].items()}
    c.lib = lib
    more = []
    for T in (2, 4, 8):                                 # ROUNDT's rows, as core_f.py draws them
        rs = load_rows(M.ROUNDT, seed=10 + T, T=T)
        more.append((T, tuple(digits_of([int(r[k]) for r in rs]) for k in ("modulus", "x", "result"))))
    print("T values of the training data:", sorted(Counter(json.loads(line)["time_steps"]
                                                            for line in open(os.path.join(H1_DIR, "train.jsonl")))
                                                    .items()), flush=True)
    for label, m in (("the data's answers", more),
                     ("one answer of T = 4 changed", [(T, (N, X, Y if T != 4 else torch.cat([(Y[:1] + 1) % 10, Y[1:]])))
                                                      for T, (N, X, Y) in more])):
        c.more = m
        for p in core["rounds"][:4]:
            t0 = time.time()
            ok = c.round_t(M.parse(p), lib)
            print(f"  {label}: {p}: ROUNDT {ok} ({time.time() - t0:.1f} s)", flush=True)
    c.more = more
    wrong = "LDA.X STA.Y CLA T.add.mul.o,x,o.AXY"          # x^2 with no reduction: right only where x^2 < N
    print(f"  x^2 with no reduction: ROUNDT {c.round_t(M.parse(wrong), lib)}", flush=True)


def main():
    M = load()
    print(f"{FILE}: READ_WINDOW {M.READ_WINDOW} CONST_OPS {M.CONST_OPS} ROUNDT {M.ROUNDT}", flush=True)
    (roundt_mode if env("MODE", "pose") == "roundt" else pose_mode)(M, "cpu")


if __name__ == "__main__":
    main()
