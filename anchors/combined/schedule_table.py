"""The combined records' schedules, one row per run (ANCHORS.md §6): certification, the step of each event, times.

    python combined/schedule_table.py [PREFIX]   (default: cmbanchors_)
"""
import json
import re
import sys
from pathlib import Path

REC = Path(__file__).resolve().parent.parent / "records"
EVENTS = [("io_stage=4", "reading"), ("order_digits", "order lessons"), ("switch_to_facts", "phase B"),
          ("ladder_joins", "ladder"), ("switch_to_gate", "phase C"), ("io_stage=5", "T's direction"),
          ("frozen", "frozen")]


def row(rec):
    lines = rec["stdout"].splitlines()
    res = rec.get("result") or {}
    dp = res.get("depth_profile", {})
    seed = (res.get("seeds") or [{}])[0]
    step = {}
    for l in lines:
        for key, name in EVENTS:
            if l.startswith("LR " + key) and name not in step:
                m = re.search(r"step=(\d+)", l)
                step[name] = m.group(1) if m else "?"
    rungs = [l for l in lines if " depth_t=" in l and "certified=" in l]
    cert = f"{dp.get('max_certified_time_steps')} / {dp.get('ood_n_max_certified_time_steps')}"
    where = "H100" if "H100" in rec.get("gpu", "") else f"CPU seed {(rec.get('meta') or {}).get('seed')}"
    return (f"| {rec['name']} | {where} | {cert} ({sum('certified=True' in l for l in rungs)}/{len(rungs)} rungs) | "
            + " | ".join(step.get(n, "–") for _, n in EVENTS)
            + f" | {seed.get('training_seconds', 0):.0f} / {seed.get('evaluation_seconds', 0):.0f} |")


def main(prefix):
    print("| run | where | max T (seen / unseen) | " + " | ".join(n for _, n in EVENTS) + " | training / evaluation s |")
    print("|---" * (len(EVENTS) + 4) + "|")
    recs = [json.loads(p.read_text()) for p in sorted(REC.glob(f"{prefix}*.json"))]
    order = lambda r: (0 if "H100" in r.get("gpu", "") else 1, [74, 1, 2, 3, 4, 5, 6, 7].index(
        (r.get("meta") or {}).get("seed")) if (r.get("meta") or {}).get("seed") in (74, 1, 2, 3, 4, 5, 6, 7) else 9)
    for r in sorted(recs, key=order):
        print(row(r))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cmbanchors_")
