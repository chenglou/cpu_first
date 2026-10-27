"""LEGAL_TEST.md §6: this study's cost, from the billing report's hours for its apps (as bar3/scale/slice/cost.py reads
them). The report covers complete hours only, so an hour still under way shows what is billed so far.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/legal/cost.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-Z9QkxEXcTDN3cy04JySmvU": "lt1: the offline test, one B200",
        "ap-4F1UpxcPAYjxJYB1K73GbG": "fetch: the records from the Volume"}
START = datetime(2026, 10, 27, 2, tzinfo=timezone.utc)        # the hour the run started in (19:21 PDT)


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
