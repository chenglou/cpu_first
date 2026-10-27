"""WIDER.md §11: this phase's cost, from the billing report's hours for its apps (as legal_nobook/cost.py reads
them). The report covers complete hours only, so an hour still under way shows what is billed so far.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_wider/cost.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-0ZshSUOtWIDx9OIvxJgRnN": "the 13 tiny cores (B200s): the file on 11 tasks, the READ_POLY = 4 arm on 2",
        "ap-j7cQbKXviC365iqEDT16mS": "addendum 1's arm: the file with phase C's ROUNDT, minus1's tiny core (B200)",
        "ap-HXhUisJ4o3TnnTnNRV13uI": "H1's official run (H100)",
        "ap-GXIkI4b2vi2WJEZxwQwSRT": "the family and the held-out pair, ten tasks (H100s)",
        "ap-fFB4KczSeFGdMQPiPm0Vlb": "H1's seeds 1-7 (H100s)"}
START = datetime(2026, 10, 27, 22, tzinfo=timezone.utc)       # the hour the first app started in (15:01 PDT)


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
