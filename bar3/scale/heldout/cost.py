"""HELDOUT.md §3: the control's cost. For each of its two apps: the billing report's hours for that app id (as
watch_spend.py reads them). For each rule: the searches the control added to its store, which start from the test's
stores joined, and their search-hours (a shared search's records each carry an even share of its seconds). The
records' own search-hours also count the stored searches the control reused, so they are larger.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/heldout/cost.py BEFORE_DIR AFTER_DIR

BEFORE_DIR holds `store_ctl_<rule>_ho1.jsonl` as uploaded before the control; AFTER_DIR the same files fetched after.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-LcCT6mNn4rLbbHHhUmsOHr": "hardest first", "ap-Q5nWfEKrXrB2mgbAGjmp61": "anchored pairs"}
START = datetime(2026, 10, 25, 14, tzinfo=timezone.utc)      # the hour both apps started in


def main():
    before, after = sys.argv[1:3]
    hours = defaultdict(lambda: defaultdict(float))
    for r in workspace_billing_report(start=START, resolution="h"):
        if r.get("object_id") in APPS:
            hours[r["object_id"]][f"{r['interval_start']:%H:%M}"] += float(r["cost"])
    for app, name in APPS.items():
        h = hours[app]
        print(f"{name} ({app}): " + ", ".join(f"{k} UTC ${v:.2f}" for k, v in sorted(h.items()))
              + f"; ${sum(h.values()):.2f}")
    print(f"both: ${sum(sum(h.values()) for h in hours.values()):.2f}")
    total = 0.0
    for rule in ("lp", "ct2a2"):
        old = open(f"{before}/store_ctl_{rule}_ho1.jsonl").read().splitlines()
        new = open(f"{after}/store_ctl_{rule}_ho1.jsonl").read().splitlines()
        assert new[:len(old)] == old, f"{rule}: the store was not only extended"
        added = [json.loads(line)[1] for line in new[len(old):] if line.strip()]
        sh = sum(float(r.get("seconds") or 0) for r in added) / 3600
        total += sh
        print(f"{rule}: {len(added)} new searches, {sh:.2f} search-hours")
    print(f"both: {total:.2f} new search-hours")


if __name__ == "__main__":
    main()
