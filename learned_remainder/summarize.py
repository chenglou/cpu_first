"""Summaries and the table audit of run records (cpu_first/runs/learned_remainder/<name>.json).

    python summarize.py NAME [NAME ...]      # one block per record
    python summarize.py --table NAME ...     # one markdown row per record

The audit compares the snapped subtractor that the submission printed at its first evaluation batch (TABLE lines) with
true subtraction r - m - c (done here, outside the submission), and lists the training visit counts of every row.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs" / "learned_remainder"


def truth():
    d, b = [], []
    for r in range(10):
        for m in range(10):
            for c in range(2):
                x = r - m - c
                d.append(x % 10)
                b.append(int(x < 0))
    return d, b


def audit(stdout: str) -> dict:
    out = {}
    lines = stdout.splitlines()
    dig = next((l.split("=", 1)[1] for l in lines if l.startswith("TABLE digit=")), None)
    bor = next((l.split("=", 1)[1] for l in lines if l.startswith("TABLE borrow=")), None)
    vis = next((l for l in lines if l.startswith("TABLE train_visits")), None)
    au = [l for l in lines if l.startswith("AUDIT")]
    if dig and bor:
        td, tb = truth()
        wrong = [(i // 20, (i // 2) % 10, i % 2) for i in range(200)
                 if int(dig[i]) != td[i] or int(bor[i]) != tb[i]]
        out["rows_right"] = 200 - len(wrong)
        out["wrong_rows"] = wrong
    if vis:
        counts = [int(c) for c in vis.split("counts=")[1].split(",")]
        out["visits_min"] = min(counts)
        out["visits_median"] = sorted(counts)[100]
        out["rows_never_visited"] = sum(c == 0 for c in counts)
    if au:
        out["audit_last"] = au[-1]
    sched = [l for l in lines if l.startswith("LR switch") or l.startswith("LR stage_done") or l.startswith("LR frozen")]
    out["schedule"] = sched
    return out


def result(rec: dict) -> dict:
    r = rec.get("result") or {}
    seed = (r.get("seeds") or [{}])[0]
    sdp = seed.get("depth_profile", {})
    return {
        "maxT": (r.get("depth_profile") or {}).get("max_certified_time_steps"),
        "oodN_maxT": (r.get("depth_profile") or {}).get("ood_n_max_certified_time_steps"),
        "rungs": [g["correct_examples"] for g in sdp.get("rungs", [])],
        "ood_rungs": [g["correct_examples"] for g in sdp.get("ood_n_rungs", [])],
        "eval": {k: f"{v['correct_examples']}/{v['example_count']}" for k, v in seed.get("evaluation", {}).items()},
        "train_s": round(seed.get("training_seconds", 0), 1),
        "eval_s": round(seed.get("evaluation_seconds", 0), 1),
        "steps": seed.get("completed_training_steps"),
        "seed": seed.get("seed"),
        "state": seed.get("model_state_elements"),
    }


def main(argv):
    table = argv[:1] == ["--table"]
    names = argv[1:] if table else argv
    if table:
        print("| run | seed | max T | OOD-N max T | rungs (seen / OOD) | test, ood_t, ood_n_t | table rows right | "
              "frozen at step | train s | eval s |")
        print("|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        rec = json.loads((RUNS / f"{name}.json").read_text())
        res, aud = result(rec), audit(rec.get("stdout", ""))
        frozen = next((re.search(r"step=(\d+)", l).group(1) for l in aud.get("schedule", [])
                       if l.startswith("LR frozen")), "-")
        if table:
            ev = ", ".join(res["eval"].get(k, "-") for k in ("test", "ood_t", "ood_n_t"))
            print(f"| {name} | {res['seed']} | {res['maxT']} | {res['oodN_maxT']} | "
                  f"{min(res['rungs'] or [0])}+ / {min(res['ood_rungs'] or [0])}+ of 768 | {ev} | "
                  f"{aud.get('rows_right', '-')}/200 | {frozen} | {res['train_s']} | {res['eval_s']} |")
            continue
        print(f"== {name} ({rec.get('gpu') or 'cpu'}, rc={rec['returncode']}, manifest="
              f"{rec['manifest'] if isinstance(rec['manifest'], str) else 'override'})")
        print("  result:", res)
        print("  table audit:", {k: v for k, v in aud.items() if k != "schedule"})
        for l in aud.get("schedule", []):
            print("  ", l[:300])


if __name__ == "__main__":
    main(sys.argv[1:])
