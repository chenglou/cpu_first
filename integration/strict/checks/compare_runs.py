"""Two evaluator records of stage-3 files, side by side (STRICT.md §4): is the second run the first one's climb, line
for line, with the same round, the same schedule and the same result? Timings and the strict files' work counts are
left out of the comparison; the work is summed instead (per climb unit, per search, in all).

    python3 integration/strict/checks/compare_runs.py A.json B.json [--quiet]
"""
import ast
import json
import re
import sys


def climb_lines(out):
    keep = []
    for l in out.splitlines():
        if l.startswith(("LR climb ", "LR climb_starts", "LR climb_joins", "LR round_found", "LR frozen", "LR switch",
                         "LR cells_join", "LR io_stage", "LR order_digits", "LR mined")):
            l = re.sub(r" \[work \+[^]]*\]", "", l)
            l = re.sub(r"\(\d+(\.\d+)?s\)$", "(s)", l)
            l = re.sub(r"\b\d+(\.\d+)?s\b", "_s", l)
            l = re.sub(r"(['\"]), \d+\.\d+\)", r"\1, _)", l)          # (how, seconds) pairs
            keep.append(l)
    return keep


def schedule(out):
    for l in out.splitlines():
        if l.startswith("SCHEDULE"):
            return l
    return None


def result(rec):
    r = rec["result"]
    sr = (r.get("seeds") or [{}])[0] if isinstance(r, dict) else {}
    dp = sr.get("depth_profile") or {}
    ev = sr.get("evaluation") or {}
    return dict(max_t=dp.get("max_certified_time_steps"), ood_n_max_t=dp.get("ood_n_max_certified_time_steps"),
                rungs=[(g["time_steps"], g["correct_examples"], g["example_count"]) for g in dp.get("rungs", [])],
                ood_rungs=[(g["time_steps"], g["correct_examples"], g["example_count"]) for g in dp.get("ood_n_rungs", [])],
                splits={k: (v.get("correct_examples"), v.get("example_count")) for k, v in ev.items()},
                train_s=sr.get("training_seconds"), eval_s=sr.get("evaluation_seconds"),
                steps=sr.get("completed_training_steps"),
                state=sr.get("model_state_elements"))


def work(out):
    """The strict file's work: per climb unit (the LR climb lines) and per search (SEARCH lines' runs)."""
    units = [float(m.group(1)) for m in re.finditer(r"\[work \+([0-9.e+]+), [0-9.e+]+ in all\]", out)]
    total = [float(m.group(1)) for m in re.finditer(r"\[work \+[0-9.e+]+, ([0-9.e+]+) in all\]", out)]
    searches = []
    for l in out.splitlines():
        if l.startswith("SEARCH "):
            try:
                r = ast.literal_eval(l[7:])
            except Exception:
                continue
            for run in r.get("runs", []) or []:
                if "work" in run:
                    searches.append((run["work"], run.get("seconds"), run.get("how", "")[:40], r.get("level")))
            if "work" in r and not r.get("runs"):
                searches.append((r["work"], r.get("seconds"), r.get("task", "round"), r.get("level")))
    secs = [float(m.group(1)) for m in re.finditer(r"\[work [^]]*\] \((\d+\.\d+)s\)", out)]
    return dict(units=len(units), unit_max=max(units, default=0), total=total[-1] if total else 0,
                unit_secs=sum(secs), searches=sorted(searches, reverse=True)[:5])


def main():
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    la, lb = climb_lines(a["stdout"]), climb_lines(b["stdout"])
    same = la == lb
    print(f"climb and schedule lines: {len(la)} vs {len(lb)}: {'IDENTICAL' if same else 'DIFFERENT'}")
    if not same:
        for i, (x, y) in enumerate(zip(la, lb)):
            if x != y:
                print(f"  first difference at line {i}:\n    A: {x[:400]}\n    B: {y[:400]}")
                break
        else:
            print("  one is a prefix of the other")
    sa, sb = schedule(a["stdout"]), schedule(b["stdout"])
    print("SCHEDULE", "same" if sa == sb else "DIFFERENT")
    print("  A:", (sa or "")[:400])
    if sa != sb:
        print("  B:", (sb or "")[:400])
    ra, rb = result(a), result(b)
    for k in ra:
        if k in ("train_s", "eval_s", "state"):
            print(f"  {k}: {ra[k]} vs {rb[k]}")
        elif ra[k] != rb[k]:
            print(f"  {k} DIFFERS: {ra[k]} vs {rb[k]}")
    print(f"  max T {rb['max_t']} / {rb['ood_n_max_t']}; splits {rb['splits']}")
    if "--quiet" not in sys.argv:
        for name, rec in (("A", a), ("B", b)):
            w = work(rec["stdout"])
            if w["units"]:
                print(f"  work {name}: {w}")


if __name__ == "__main__":
    main()
