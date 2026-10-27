"""ROUNDT in minus1's tiny core (FAMILY_FILE.md §4): with the library its climb had after level 5 (records/
t_family_minus1.json), the plain two-call rounds over its window x·o − 1 and the reduction, on the T = 1 answers the
climb's round search used (400 rows) and on ROUNDT's rows of T = 2, 4 and 8 (core_f.py's draw), with the rows through 0.

    cd cpu_first && TASK=minus1 PYTHONDONTWRITEBYTECODE=1 ~/github/sotaku/venv/bin/python integration/legal_family/checks/roundt_minus1.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import core_f  # noqa: E402
import family_local as FL  # noqa: E402
import torch  # noqa: E402

PROGS = ["LDA.X STA.Y T.w.sub1.mul.w2.XYN STA.Y T.red.x+on.YN", "LDA.X STA.Y T.w.sub1.mul.w2.YXN STA.Y T.red.x+on.YN"]


def main():
    M = FL.load()
    rec = json.load(open(HERE.parent / "records" / "t_family_minus1.json"))
    h5 = next(h for h in rec["history"] if h["level"] == 5)
    lib = {k: M.parse(v) for k, v in h5["lib"].items()}
    c = FL.climb_at_level3(M, "cpu")
    arows = core_f.answered(core_f.load_rows(400, seed=1, T=1))
    ans = tuple(core_f.digits(arows, k) for k in ("modulus", "x", "result"))
    for T in (2, 4, 8):
        mr = core_f.answered(core_f.load_rows(M.ROUNDT, seed=10 + T, T=T))
        c.more.append((T, tuple(core_f.digits(mr, k) for k in ("modulus", "x", "result"))))
    for p in PROGS:
        prog = M.parse(p)
        ok1 = M.round_ok(prog, lib, c.cells, ans, M.PR, "cpu")
        bad = {}
        st = M.round_settings(c.cells, M.PR)
        for T, (Nd, Xd, Yd) in c.more:
            X, N = M.pad(Xd, M.PR), M.pad(Nd, M.PR)
            for _ in range(T):
                X = M.run_prog(st, prog, lib, {"X": X, "N": N}, M.gen_on("cpu", 1), "cpu")[0]
            bad[T] = int((~(X[:, :Yd.shape[1]] == Yd).all(1)).sum())
        print(f"{p}: right on the {len(arows)} answers of T = 1: {ok1}; rows wrong of ROUNDT's: {bad}; "
              f"ROUNDT {c.round_t(prog, lib)}", flush=True)


if __name__ == "__main__":
    main()
