"""LOOPS.md §9: this study's cost, from the billing report's hours for each of its apps (as
bar3/scale/digits/watch_spend.py reads them). An hour still under way shows what is billed so far.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/loops/cost.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-fKOc27yonAqzLDGy6DBghb": "phase 1: l5_x2, replay_s4_x2, l5_x3, l5_10x2p1",
        "ap-E8vaCE0oHLtHmkkNmXdKQc": "phase 2: l5_x2_bare, l5_x2_less_add, l5_x2_less_link, l5_x2_lessons",
        "ap-Rgq3V3rAbOhHqELXv4s5yT": "addendum 1: l5_x2_no_book, l5_x2_all_off, l5_x2_bare_no_redloop, l5_x3_less_mul, l5_10x2p1_less_mul",
        "ap-ReeN6ffw3Jx5sCoKcjCcC8": "addendum 2: s4_x2_no_subgoal, s4_x2_no_lawnear, s4_x2_no_spell, l5_x3_long, l5_x3_less_mul_long, l5_10x2p1_less_mul_long, l5_x2_lessons_long"}
START = datetime(2026, 10, 27, 0, tzinfo=timezone.utc)        # the hour this study's first app can start in


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
