"""COMPOSE.md §9: the runs' cost, from the billing report's hours for each of the study's five apps (as
bar3/scale/digits/watch_spend.py reads them). An hour still under way shows what is billed so far.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/compose/cost.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-2lba6KdE5XuYVdUqEKat2C": "PREREG.md's five arms (06:29 PDT)",
        "ap-ootToK0KizdVnVoR8wEM5s": "cp_bar3_clean (06:40)",
        "ap-PuOQEjmeAGlzrtZ28CxZkk": "cp_bar3_minus_red_clean (06:46)",
        "ap-dHPW3PcxAbLqiVENqDfkC5": "the route check (07:10)", "ap-VjGEPxGPHeXM9e82UM90Mq": "the focused route check (07:14)"}
START = datetime(2026, 10, 26, 13, tzinfo=timezone.utc)      # the hour the first app was created in (13:29 UTC)


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
