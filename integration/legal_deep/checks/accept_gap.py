"""DEEP.md §2: the check gap, on the file's own acceptance. A tiny core's learned routine for a reading's window (by
default phase D's three2: w.add2.mul3.mul on w2, binding YXN, whose routine multiplies by b's last digit) is put
to the Acceptor the climb builds for that window, with the file's verify settings (60 fresh rows each, seed 11).
Prints, per setting, the rows the window's target says something on, and the verdict; with FILE the file before
(b) and after it.

    cd cpu_first && FILE=... PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/legal_deep/checks/accept_gap.py \
        [RECORD.json TASK LAW BINDING]
"""
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU = HERE.parents[2]
sys.path.insert(0, str(CPU / "learned_remainder" / "chain_local" / "stub"))
sys.path.insert(0, str(CPU / "integration" / "legal_wider" / "checks"))
import torch  # noqa: E402
import wider_local as WL  # noqa: E402  (its rows, tasks and the level-3 climb)


def main():
    args = sys.argv[1:] or [str(CPU / "integration/legal_wider/records/t_wider_three2.json"), "three2",
                            "w.add2.mul3.mul", "YXN"]
    rec_path, task, law, binding = args
    path = os.environ.get("FILE", str(HERE.parent / "submission_stage3_deep.py"))
    WL.FILE = path
    M = WL.load()
    c = WL.climb_at_level3(M, "cpu")
    rec = json.load(open(rec_path))
    lib = {k: M.parse(v) for k, v in rec["lib"].items()}
    prog = M.parse(rec["lib"][f"T.{law}.w2.{binding}"])
    arows = WL.load_rows(400, seed=1, T=1)
    xs, Ns = [int(r["x"]) for r in arows], [int(r["modulus"]) for r in arows]
    ys = [WL.TASKS[task](x, N) for x, N in zip(xs, Ns)]
    rows = WL.load_rows(4096, seed=0)
    rows = (WL.digits_of([int(r["modulus"]) for r in rows]), WL.digits_of([int(r["x"]) for r in rows]))
    ans = (WL.digits_of(Ns), WL.digits_of(xs), WL.digits_of(ys))
    pool, roots = c.pose(rows, ans)
    root = [r_ for r_ in roots if r_.pattern == "w2" and "".join(r_.binding) == binding
            and any(g.name == law for g in r_.goal_list)]
    assert root, ("no root", law, binding, sorted({(r_.pattern, "".join(r_.binding)) for r_ in roots}))
    root = root[0]
    c.lib = dict(c.lib, **lib)
    accs = c.acceptors(root, rows)
    acc = [a for a, g in zip(accs, root.goal_list) if g.name == law][0]
    for st, b, goal in acc.probs:
        t = goal.targets[0]
        n = int(t["Am"].sum()) if "Am" in t else None
        print(f"  setting ({st.B}, {st.W}), {st.P} places: the target says something on {n} of {b.regs.shape[0]} rows")
    print(f"{Path(path).name}: {law}@{binding}, the learned routine {rec['lib'][f'T.{law}.w2.{binding}']}: "
          f"accepted {acc.many([prog], c.lib)[0]}")


if __name__ == "__main__":
    main()
