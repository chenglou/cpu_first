"""Summaries and the table audits of submission_square.py's run records
(cpu_first/history/runs/learned_remainder/<name>.json).

    python summarize_square.py NAME [NAME ...]      # one block per record
    python summarize_square.py --table NAME ...     # one markdown row per record

As summarize.py, plus the product cell: the snapped lo/hi tables the submission printed at its first evaluation batch
(TABLE prod_lo / prod_hi) against true multiplication a * b (done here, outside the submission), and the training
visit counts of every cell (TABLE prod_train_visits).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from summarize import RUNS, audit, result


def product_audit(stdout: str) -> dict:
    out = {}
    lines = stdout.splitlines()
    lo = next((l.split("=", 1)[1] for l in lines if l.startswith("TABLE prod_lo=")), None)
    hi = next((l.split("=", 1)[1] for l in lines if l.startswith("TABLE prod_hi=")), None)
    vis = next((l for l in lines if l.startswith("TABLE prod_train_visits")), None)
    if lo and hi:
        wrong = [(i // 10, i % 10) for i in range(100)
                 if int(lo[i]) != (i // 10) * (i % 10) % 10 or int(hi[i]) != (i // 10) * (i % 10) // 10]
        out["cells_right"] = 100 - len(wrong)
        out["wrong_cells"] = wrong
    if vis:
        counts = [int(c) for c in vis.split("counts=")[1].split(",")]
        out["cell_visits_min"] = min(counts)
        out["cell_visits_median"] = sorted(counts)[50]
        out["cells_never_visited"] = sum(c == 0 for c in counts)
    return out


def main(argv):
    table = argv[:1] == ["--table"]
    names = argv[1:] if table else argv
    if table:
        print("| run | seed | max T | OOD-N max T | rungs (seen / OOD) | test, ood_t, ood_n_t | subtractor rows right | "
              "product cells right | frozen at step | train s | eval s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        rec = json.loads((RUNS / f"{name}.json").read_text())
        res, aud, pa = result(rec), audit(rec.get("stdout", "")), product_audit(rec.get("stdout", ""))
        frozen = next((re.search(r"step=(\d+)", l).group(1) for l in aud.get("schedule", [])
                       if l.startswith("LR frozen")), "-")
        if table:
            ev = ", ".join(res["eval"].get(k, "-") for k in ("test", "ood_t", "ood_n_t"))
            print(f"| {name} | {res['seed']} | {res['maxT']} | {res['oodN_maxT']} | "
                  f"{min(res['rungs'] or [0])}+ / {min(res['ood_rungs'] or [0])}+ of 768 | {ev} | "
                  f"{aud.get('rows_right', '-')}/200 | {pa.get('cells_right', '-')}/100 | {frozen} | "
                  f"{res['train_s']} | {res['eval_s']} |")
            continue
        print(f"== {name} ({rec.get('gpu') or 'cpu'}, rc={rec['returncode']}, manifest="
              f"{rec['manifest'] if isinstance(rec['manifest'], str) else 'override'})")
        print("  result:", res)
        print("  subtractor audit:", {k: v for k, v in aud.items() if k != "schedule"})
        print("  product audit:", pa)
        for l in aud.get("schedule", []):
            print("  ", l[:300])


if __name__ == "__main__":
    main(sys.argv[1:])
