"""FAMILY.md §9: this study's cost, from the billing report's hours for each of its apps (as
bar3/scale/digits/watch_spend.py reads them). An hour still under way shows what is billed so far.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/family/cost.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-1D1Y8tz0CDZ3CTVtnwy776": "phase 1: the eight arms of PREREG.md",
        "ap-zFQQAYrrPuIR0mUlMStYpA": "addendum 1: the eleven arms of PREREG_ADDENDUM1.md",
        "ap-zKxsiEpvkIa8fXHBFlrAIX": "addendum 2: seven of the ten arms of PREREG_ADDENDUM2.md",
        "ap-wP8PzLLDuSJgGmfl8AyY98": "addendum 2, relaunched: the three arms the watcher cut"}
START = datetime(2026, 10, 27, 10, tzinfo=timezone.utc)        # the hour this study's first app started in


def main():
    hours = defaultdict(lambda: defaultdict(float))
    for r in workspace_billing_report(start=START, resolution="h"):
        if r.get("object_id") in APPS:
            hours[r["object_id"]][f"{r['interval_start']:%H:%M}"] += float(r["cost"])
    total = 0.0
    for app, name in APPS.items():
        h = hours.get(app, {})
        cost = sum(h.values())
        total += cost
        print(f"{app} {name}: ${cost:.2f}  " + ", ".join(f"{k} UTC ${v:.2f}" for k, v in sorted(h.items())))
    print(f"all: ${total:.2f}")


if __name__ == "__main__":
    main()
