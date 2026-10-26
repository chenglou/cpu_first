"""Two evaluator records' climbs, line by line (FAST.md §5): every `LR climb`, `LR outer`, `LR round_found` and `LR frozen`
line, with its seconds masked (the unit's own, each search's, the bracketed timings); and each run's step, training
clock and work at the round, the answer program and the freeze. For the fast file's official run against round 2's
own run of the same seed (combined_sp2's relaxed run: the same evaluator, a longer budget).

    python3 integration/combined_fast/checks/compare_official.py A.json B.json
"""
import json
import re
import sys

KEEP = ("LR climb", "LR outer", "LR round_found", "LR frozen")


def lines(rec):
    return [l for l in rec["stdout"].splitlines() if l.startswith(KEEP)]


def masked(l):
    l = re.sub(r"\(\d+(\.\d+)?s\)", "(#s)", l)
    l = re.sub(r"\b\d+(\.\d+)?s\b", "#s", l)
    l = re.sub(r"\('([^']*)', [\d.]+\)", r"('\1', #)", l)
    return re.sub(r'\("([^"]*)", [\d.]+\)', r'("\1", #)', l)


def clock(rec, step):
    """The training clock at the evaluator's progress line nearest before step (its lines every 100 steps)."""
    best = None
    for l in rec["stdout"].splitlines():
        m = re.match(r"step=(\d+) .*elapsed=([\d.]+)s", l)
        if m and int(m.group(1)) <= step:
            best = (int(m.group(1)), float(m.group(2)))
    return best


A, B = (json.load(open(p)) for p in sys.argv[1:3])
la, lb = lines(A), lines(B)
n = min(len(la), len(lb))
first = next((i for i in range(n) if masked(la[i]) != masked(lb[i])), None)
print(f"{len(la)} and {len(lb)} climb lines; " + (f"the first {n} the same, seconds masked" if first is None else
                                                    f"the first difference at line {first}:\n  {la[first][:240]}\n  {lb[first][:240]}"))
for rec, name in ((A, sys.argv[1]), (B, sys.argv[2])):
    marks = {}
    for l in lines(rec):
        m = re.match(r"LR (\w+) step=(\d+)", l)
        if m and (m.group(1) in ("round_found", "frozen") or " round: cost " in l and "None" not in l.split("round: cost ")[1][:5]):
            marks.setdefault(m.group(1) if m.group(1) != "climb" else "round", int(m.group(2)))
    seed = rec["result"]["seeds"][0] if rec.get("result") else {}
    print(f"{name}: " + "; ".join(f"{k} at step {v} (clock {clock(rec, v)})" for k, v in marks.items())
          + f"; training {seed.get('training_seconds')} s, {seed.get('completed_training_steps')} steps; "
            f"evaluation {seed.get('evaluation_seconds')} s")
