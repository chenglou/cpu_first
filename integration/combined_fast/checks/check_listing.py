"""The listing (FAST.md §2): first_programs gives, for every goal state, exactly the program the per-state walk gives
(programs(_OneGoal(...), classes, limit=1)[0]). The climb runs on the laptop's CPU at small budgets (the probes
stop early, so it stays small), and every call of first_programs is checked against the per-state walk on the same
parents; the climb stops after UNITS units.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/combined_fast/checks/check_listing.py [FILE]
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import core_c  # noqa: E402

FILE = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent / "submission_stage3_fast.py")
UNITS = int(os.environ.get("UNITS", 12))
BUDGET = int(os.environ.get("BUDGET", 150_000))
M, _ = core_c.load(FILE, [])
calls = [0, 0]
first = M.first_programs


def checked(parents, level, goals, classes):
    got = first(parents, level, goals, classes)
    want = [M.programs(M._OneGoal(M._Res(parents, level, goals), g), classes, limit=1)[0] for g in goals]
    assert got == want, (level, len(goals))
    calls[0] += 1
    calls[1] += len(goals)
    return got


M.first_programs = checked
rows = core_c.load_rows(1024, seed=0)
Nd, Xd = core_c.digits(rows, "modulus"), core_c.digits(rows, "x")
sp = dict(M.SP, probe_budget=BUDGET, fallback_budget=BUDGET, deep_budget=BUDGET, round_budget=BUDGET)
c = M.SelfClimb(M.Cells(10, "cpu"), "cpu", M.PR, chunk=1 << 14, **sp)
for i in range(UNITS):
    line = c.step((Nd, Xd))
    if line is None:
        break
    print(i, line[:150], flush=True)
print(f"first_programs: {calls[0]} calls, {calls[1]} goal states, every program the per-state walk's")
assert calls[1] > 0
