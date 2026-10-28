"""Summaries and audits of submission_combined.py's run records (cpu_first/history/runs/learned_remainder/<name>.json).

    python summarize_combined.py NAME [NAME ...]      # one block per record
    python summarize_combined.py --table NAME ...     # one markdown row per record

As summarize.py and summarize_square.py: the snapped tables the submission printed at its first evaluation batch
(TABLE lines) against true subtraction and multiplication (done here, outside the submission), the gate, the I/O it
reads with (IO lines), the evaluation audits (AUDIT lines) and the schedule.
"""

from __future__ import annotations

import json
import re
import sys

from summarize import RUNS, audit, result
from summarize_square import product_audit


def combined_audit(stdout: str) -> dict:
    lines = stdout.splitlines()
    out = {"io": [l[3:] for l in lines if l.startswith("IO ")]}
    gate = next((l for l in lines if l.startswith("TABLE gate=")), None)
    if gate:
        out["gate"] = gate.split("=", 1)[1]
    out["audits"] = sorted({l for l in lines if l.startswith("AUDIT")}, key=lines.index)
    out["schedule"] = [l for l in lines if l.startswith("LR switch") or l.startswith("LR ladder") or
                       l.startswith("LR io_stage") or l.startswith("LR frozen")]
    sched = next((l for l in lines if l.startswith("TABLE schedule")), "")
    out["final_phase"] = sched.split("phase=")[1].split()[0] if "phase=" in sched else "-"
    return out


def main(argv):
    table = argv[:1] == ["--table"]
    names = argv[1:] if table else argv
    if table:
        print("| run | seed | max T | OOD-N max T | rungs (seen / OOD) | test, ood_t, ood_n_t | subtractor rows right | "
              "product cells right | gate | frozen at step | train s | eval s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        rec = json.loads((RUNS / f"{name}.json").read_text())
        out = rec.get("stdout", "")
        res, aud, pa, ca = result(rec), audit(out), product_audit(out), combined_audit(out)
        frozen = next((re.search(r"step=(\d+)", l).group(1) for l in ca["schedule"] if l.startswith("LR frozen")), "-")
        if table:
            ev = ", ".join(res["eval"].get(k, "-") for k in ("test", "ood_t", "ood_n_t"))
            print(f"| {name} | {res['seed']} | {res['maxT']} | {res['oodN_maxT']} | "
                  f"{min(res['rungs'] or [0])}+ / {min(res['ood_rungs'] or [0])}+ of 768 | {ev} | "
                  f"{aud.get('rows_right', '-')}/200 | {pa.get('cells_right', '-')}/100 | {ca.get('gate', '-')[:2]} | "
                  f"{frozen} | {res['train_s']} | {res['eval_s']} |")
            continue
        print(f"== {name} ({rec.get('gpu') or 'cpu'}, rc={rec['returncode']}, manifest="
              f"{rec['manifest'] if isinstance(rec['manifest'], str) else 'override'})")
        print("  result:", res)
        print("  subtractor:", {k: v for k, v in aud.items() if k not in ("schedule", "audit_last")})
        print("  product cell:", pa)
        print("  gate:", ca.get("gate"), " final phase:", ca["final_phase"])
        for l in ca["io"]:
            print("  IO", l[:200])
        for l in ca["audits"]:
            print("  ", l[:300])
        for l in ca["schedule"]:
            print("  ", l[:300])


if __name__ == "__main__":
    main(sys.argv[1:])
