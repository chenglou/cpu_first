"""One line per evaluator record in records/: certification, the snapped table against true subtraction, the audit,
the schedule's events, times. Every record holds the exact source that ran (its md5 is printed).

    python summarize.py [NAME ...]          (default: every record)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def true_table():
    d, b = [], []
    for r in range(10):
        for m in range(10):
            for c in range(2):
                x = r - m - c
                d.append(x % 10)
                b.append(int(x < 0))
    return d, b


def line(rec: dict) -> str:
    out = rec.get("stdout", "")
    lines = out.splitlines()
    res = rec.get("result") or {}
    dp = res.get("depth_profile", {})
    seed = (res.get("seeds") or [{}])[0]
    rungs = [l for l in lines if " depth_t=" in l and "certified=" in l]
    cert = sum("certified=True" in l for l in rungs)
    ev = {m["split"]: m["exact_accuracy"] for m in res.get("structured_metrics", []) if m.get("type") == "evaluation"}
    td, tb = true_table()
    get = lambda key: next((l.split("=", 1)[1] for l in lines if l.startswith(key)), None)
    dig, bor = get("TABLE digit="), get("TABLE borrow=")
    rows = sum(int(dig[i]) == td[i] and int(bor[i]) == tb[i] for i in range(200)) if dig and bor else None
    audits = [l for l in lines if l.startswith("AUDIT")]
    never = [int(m) for l in audits for m in re.findall(r"used_but_never_trained=(\d+)", l)]
    used = next((l.split()[1] for l in reversed(audits) if "eval_rows_used" in l), "?")
    audit = f"{used} never-trained max {max(never) if never else '?'} over {len(audits)} audit lines"
    ev_names = ("digits_start", "order_digits", "switch_to_facts", "ladder_joins", "switch_to_gate", "frozen",
                "io_stage=4", "io_stage=5")
    events = []
    for l in lines:
        for e in ev_names:
            if l.startswith("LR " + e):
                m = re.search(r"step=(\d+)", l)
                events.append(f"{e.replace('io_stage=', 'io')}@{m.group(1) if m else '?'}")
    where = "H100" if "H100" in rec.get("gpu", "") else f"cpu s{(rec.get('meta') or {}).get('seed')}"
    md5 = hashlib.md5(rec.get("submission", "").encode()).hexdigest()[:8]
    return (f"{rec['name']:24s} {where:9s} md5 {md5} maxT {dp.get('max_certified_time_steps')}/"
            f"{dp.get('ood_n_max_certified_time_steps')} rungs {cert}/{len(rungs)} "
            f"test {ev.get('test')} ood_t {ev.get('ood_t')} ood_n_t {ev.get('ood_n_t')} table {rows}/200 "
            f"[{audit}] train {seed.get('training_seconds', 0):.0f}s ({seed.get('completed_training_steps')}) "
            f"eval {seed.get('evaluation_seconds', 0):.0f}s {' '.join(events)}")


def main(argv):
    paths = [HERE / "records" / f"{n}.json" for n in argv] if argv else sorted((HERE / "records").glob("*.json"))
    for p in paths:
        print(line(json.loads(p.read_text())), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
