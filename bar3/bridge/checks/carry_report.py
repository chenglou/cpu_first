"""The run table of CARRY.md from records/carry/*.jsonl: per run, its rule and flags, the rounds it finished, what it
invented and in which round, and its search-hours (every job's seconds; a job the shared cache answered counts what
it cost when first run, as BRIDGE.md's tables do).

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_report
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "records" / "carry"
ORDER = [f"{f}{r}_{t}_k1" for t in ("adc", "mul", "sbb", "sel") for f in ("", "w") for r in ("lp", "ct2", "all")]


def main():
    print("| run | target | rule | pose rule | rounds finished | invented | program | search-hours |")
    print("|---|---|---|---|---|---|---|---|")
    for name in ORDER:
        p = HERE / f"{name}.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if not recs:
            continue
        last = recs[-1]
        done = sum(1 for r in recs if r.get("target_rec") is not None)
        g = last.get("general")
        hours = sum(t["seconds"] for t in last["totals"].values()) / 3600
        print(f"| {name} | {last['target']} | {last['rule']} | {'widen' if name.startswith('w') else 'none'} | "
              f"{done} | {'round ' + str(last['round']) if g else 'no'} | {'`' + g['program'] + '`' if g else '-'} | "
              f"{hours:.1f} |")


if __name__ == "__main__":
    main()
