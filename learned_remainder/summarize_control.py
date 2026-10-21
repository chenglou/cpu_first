"""Summaries of submission_control.py's run records (cpu_first/runs/learned_remainder/<name>.json): summarize.py's result
and table audit, plus the commit gate and the countdown's rows.

    python summarize_control.py NAME [NAME ...]      # one block per record
    python summarize_control.py --table NAME ...     # one markdown row per record

The gate audit compares the snapped gate that the submission printed (TABLE gate=, 1 = commit, indexed by the borrow)
with the true one (commit unless the subtraction borrows), outside the submission.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import RUNS, audit, result  # noqa: E402


def control_audit(stdout: str) -> dict:
    lines = stdout.splitlines()
    out = {}
    gate = next((l for l in lines if l.startswith("TABLE gate=")), None)
    if gate:
        g = gate.split("=", 1)[1].split()[0]
        out["gate"] = g
        out["gate_right"] = g == "10"
        out["gate_p_commit"] = gate.split("p_commit=")[1] if "p_commit=" in gate else None
    gv = next((l for l in lines if l.startswith("TABLE gate_train_visits=")), None)
    if gv:
        out["gate_train_visits"] = [int(c) for c in gv.split("=", 1)[1].split(",")]
    for key in ("countdown_rows_used", "gate_rows_used"):
        al = [l for l in lines if l.startswith(f"AUDIT {key}=")]
        if al:
            out[key] = al[-1][len("AUDIT "):]
    rounds = [int(m.group(1)) for l in lines for m in [re.search(r"LR frozen step=(\d+)", l)] if m]
    out["frozen"] = rounds[0] if rounds else None
    sw = [int(m.group(1)) for l in lines for m in [re.search(r"LR switch_to_own_borrow step=(\d+)", l)] if m]
    out["switch"] = sw[0] if sw else None
    sg = [int(m.group(1)) for l in lines for m in [re.search(r"LR switch_to_gate step=(\d+)", l)] if m]
    out["gate_on"] = sg[0] if sg else None
    return out


def main(argv):
    table = argv[:1] == ["--table"]
    names = argv[1:] if table else argv
    if table:
        print("| run | seed | max T | OOD-N max T | rungs (seen / OOD) | test, ood_t, ood_n_t | table rows right | gate | "
              "phase B / C / frozen at step | train s | eval s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        rec = json.loads((RUNS / f"{name}.json").read_text())
        res, aud, ctl = result(rec), audit(rec.get("stdout", "")), control_audit(rec.get("stdout", ""))
        if table:
            ev = ", ".join(res["eval"].get(k, "-") for k in ("test", "ood_t", "ood_n_t"))
            print(f"| {name} | {res['seed']} | {res['maxT']} | {res['oodN_maxT']} | "
                  f"{min(res['rungs'] or [0])}+ / {min(res['ood_rungs'] or [0])}+ of 768 | {ev} | "
                  f"{aud.get('rows_right', '-')}/200 | {ctl.get('gate', '-')} | {ctl.get('switch', '-')} / "
                  f"{ctl.get('gate_on', '-')} / {ctl.get('frozen', '-')} | {res['train_s']} | {res['eval_s']} |")
            continue
        print(f"== {name} ({rec.get('gpu') or 'cpu'}, rc={rec['returncode']}, manifest="
              f"{rec['manifest'] if isinstance(rec['manifest'], str) else 'override'})")
        print("  result:", res)
        print("  table audit:", {k: v for k, v in aud.items() if k != "schedule"})
        print("  control audit:", ctl)
        for l in aud.get("schedule", []):
            print("  ", l[:300])


if __name__ == "__main__":
    main(sys.argv[1:])
