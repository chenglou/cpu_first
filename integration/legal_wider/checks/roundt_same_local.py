"""ROUNDT, old against new (WIDER.md §3): the family file's round_t (run_prog, T runs per count) and this file's (the
round machine from run_prog's draws), on the same rounds and the same rows: the verdicts, every count's outputs bit
for bit, the work each counts, and the time. Rounds: the tiny cores' rounds and libraries (phase B's H1, phase C's
cube, rho and minus1), and minus1's plain rounds over its window x·o − 1 with its level-5 library (wrong at 0).
Rows: core_f.py's draw for ROUNDT (the file's ROUNDT rows of each T = 2, 4, 8), each task's answers by Python's
integers. CPU.

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 ~/github/sotaku/venv/bin/python integration/legal_wider/checks/roundt_same_local.py [NEW_FILE]
"""
import json
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU = HERE.parents[2]
sys.path.insert(0, str(CPU / "learned_remainder" / "chain_local" / "stub"))
sys.path.insert(0, str(CPU / "integration" / "legal_family"))
import torch  # noqa: E402

import core_f  # noqa: E402  (load_rows, digits, STEPS; its TASK is not used here)

OLD = CPU / "integration" / "legal_family" / "submission_stage3_family.py"
NEW = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "submission_stage3_wider.py"
REC = CPU / "integration"
CASES = [  # (task, record, history level whose library to use (None: the final one), rounds (None: the record's))
    ("h1", "legal_nobook/records/t_nobook.json", None, None),
    ("cube", "legal_family/records/t_family_cube.json", None, None),
    ("rho", "legal_family/records/t_family_rho.json", None, None),
    ("minus1", "legal_family/records/t_family_minus1.json", None, None),
    ("minus1", "legal_family/records/t_family_minus1.json", 5,
     ["LDA.X STA.Y T.w.sub1.mul.w2.XYN STA.Y T.red.x+on.YN", "LDA.X STA.Y T.w.sub1.mul.w2.YXN STA.Y T.red.x+on.YN"]),
]


def load(path, name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(open(path).read(), str(path), "exec"), mod.__dict__)
    return mod


def answered(rows, task):
    out = []
    for r in rows:
        y, n = int(r["x"]), int(r["modulus"])
        for _ in range(int(r["time_steps"])):
            y = core_f.STEPS[task](y, n)
        out.append(dict(r, result=y))
    return out


def climb(M, more):
    c = M.SelfClimb(M.Cells(10, "cpu"), "cpu", M.PR, chunk=1 << 18, **M.SP)
    c.more = more
    return c


def outputs(M, c, p, lib):
    """Each count's final X, as the module's own round_t computes it (run_prog for the old, the machine for the new)."""
    st = M.round_settings(c.cells, c.P_round)
    out = []
    if hasattr(M, "ROUNDT_SLOTS"):                                  # the new file: the round machine
        ops = M.answer_trace(st, p, lib)
        m = M.RoundMachine(st, ops, -(-max(x[0].shape[0] for _, x in c.more) // M.ROUND_ROWS) * M.ROUND_ROWS, "cpu")
        m.cells(st.cells)
        for T, (Nd, Xd, Yd) in c.more:
            X, go = Xd, torch.ones(Xd.shape[0], dtype=torch.bool)
            for _ in range(T):
                X = m.run(X, Nd, go, go.numel(), M.gen_on("cpu", 1)).clone()
            out.append(X)
    else:
        for T, (Nd, Xd, Yd) in c.more:
            X, N = M.pad(Xd, c.P_round), M.pad(Nd, c.P_round)
            for _ in range(T):
                X = M.run_prog(st, p, lib, {"X": X, "N": N}, M.gen_on("cpu", 1), "cpu")[0]
            out.append(X)
    return out


def main():
    A, B = load(OLD, "old_family"), load(NEW, "new_wider")
    print(f"old {OLD.name}, new {NEW.name}; ROUNDT {A.ROUNDT} / {B.ROUNDT}", flush=True)
    bad = 0
    for task, rec, level, rounds in CASES:
        r = json.load(open(REC / rec))
        libtext = r["lib"] if level is None else next(h for h in r["history"] if h["level"] == level)["lib"]
        rounds = rounds or r["rounds"]
        more = []
        for T in (2, 4, 8):
            rows = answered(core_f.load_rows(A.ROUNDT, seed=10 + T, T=T), task)
            more.append((T, tuple(core_f.digits(rows, k) for k in ("modulus", "x", "result"))))
        for p in rounds:
            res = []
            for M in (A, B):
                lib = {k: M.parse(v) for k, v in libtext.items()}
                c = climb(M, more)
                M.WORK[0] = 0.0                                 # (both counters from 0: the same float sums)
                w0, t0 = M.WORK[0], time.time()
                ok = c.round_t(M.parse(p), lib)
                res.append((ok, M.WORK[0] - w0, time.time() - t0, outputs(M, c, M.parse(p), lib)))
            (oa, wa, ta, xa), (ob, wb, tb, xb) = res
            same = oa == ob and wa == wb and all(torch.equal(x, y) for x, y in zip(xa, xb))
            bad += not same
            print(f"{task}{'' if level is None else f' (level {level})'}: {p}: verdict {oa} / {ob}, work {wa:.6g} / "
                  f"{wb:.6g}, outputs {'equal' if all(torch.equal(x, y) for x, y in zip(xa, xb)) else 'DIFFER'}; "
                  f"{ta:.1f} s / {tb:.1f} s{'' if same else '  <-- NOT THE SAME'}", flush=True)
    print(f"done: {bad} rounds differ", flush=True)


if __name__ == "__main__":
    main()
