"""DIGITS.md §4: the runs' cost. For each of the six apps: the billing report's hours for that app id (as watch_spend.py
reads them). For each run: the search-hours its records carry (a search shared by runs gives each an even share of its
seconds, so the shares add up to the searches' time); every store started empty, so these are all new searches.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/digits/cost.py [records/*_dg1.jsonl ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from modal.billing import workspace_billing_report

HERE = Path(__file__).resolve().parent
APPS = {"ap-CHVZ0z5a6t98ZLQdObyuEj": "hardest first, unary", "ap-oZRxfCSCmFtVGPy00suFzi": "hardest first, binary",
        "ap-1cjwDVG8L3eLXKp4H2OO3A": "hardest first, decimal", "ap-BN2gqKKz2gqdhyjUOUV0O8": "anchored pairs, unary",
        "ap-rrGGwYfdtHYN2KlNKupGua": "anchored pairs, binary", "ap-3RxqlYYwHUMrOpClkdUUwQ": "anchored pairs, decimal"}
START = datetime(2026, 10, 25, 16, tzinfo=timezone.utc)      # the hour the first app was created in (16:59:57 UTC)


def main():
    hours = defaultdict(lambda: defaultdict(float))
    for r in workspace_billing_report(start=START, resolution="h"):
        if r.get("object_id") in APPS:
            hours[r["object_id"]][f"{r['interval_start']:%H:%M}"] += float(r["cost"])
    for app, name in APPS.items():
        h = hours[app]
        print(f"{name} ({app}): " + ", ".join(f"{k} UTC ${v:.2f}" for k, v in sorted(h.items()))
              + f"; ${sum(h.values()):.2f}")
    print(f"all six: ${sum(sum(h.values()) for h in hours.values()):.2f}")
    paths = sys.argv[1:] or sorted(str(p) for p in (HERE / "records").glob("*_dg1.jsonl"))
    total = 0.0
    for p in paths:
        recs = [r for r in (json.loads(line) for line in open(p) if line.strip()) if "round" in r]
        sh = sum(t["seconds"] for t in recs[-1]["totals"].values()) / 3600 if recs else 0.0
        total += sh
        print(f"{Path(p).stem}: {len(recs)} rounds, {sh:.2f} search-hours")
    print(f"all runs: {total:.2f} search-hours")


if __name__ == "__main__":
    main()
