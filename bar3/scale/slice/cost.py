"""SLICE.md §9: this study's cost, from the billing report's hours for each of its apps (as
bar3/scale/digits/watch_spend.py reads them). An hour still under way shows what is billed so far.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/slice/cost.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-eAgytteJx73N51JrAItShR": "phase 1: s4_x2, replay_c3_r1, s4_x3, s4_10x2p1, s4_x2_lessons, replay_c2_b2_rd",
        "ap-ylmSJysPFNCBKVfHx41cX3": "phase 2: s4_x2_no_step, s4_x2_less_mul"}
START = datetime(2026, 10, 26, 21, tzinfo=timezone.utc)       # the hour this study's first app can start in


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
