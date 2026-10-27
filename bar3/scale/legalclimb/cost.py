"""LEGALCLIMB.md §7: this study's cost, from the billing report's hours for its apps (as bar3/scale/slice/cost.py reads
them). The report covers complete hours only, so an hour still under way shows what is billed so far.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/legalclimb/cost.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-NPshwTcOAm62vX2IFobnuB": "the tiny tests: rs_b3, rs_b2",
        "ap-hg9dvMh6S1xQvVMCQFpWgS": "the arms: g1, g1_id, g0, c1, g3, g2",
        "ap-UaLkzctSGNRYYxVq81WgvS": "addendum 1: g1_b2, g1_s1",
        "ap-sxl4g6ITJ2M1m7nAm4ptjw": "addendum 2: g2_m3, g1_s2, g1_s3"}
START = datetime(2026, 10, 27, 3, tzinfo=timezone.utc)        # the hour the first app started in (20:23 PDT)


def main():
    hours = defaultdict(lambda: defaultdict(float))
    other = defaultdict(float)
    for r in workspace_billing_report(start=START, resolution="h"):
        if r.get("object_id") in APPS:
            hours[r["object_id"]][f"{r['interval_start']:%H:%M}"] += float(r["cost"])
        elif "legalclimb" in str(r.get("description", "")):
            other[r.get("object_id")] += float(r["cost"])
    total = 0.0
    for app, name in APPS.items():
        h = hours.get(app, {})
        cost = sum(h.values())
        total += cost
        print(f"{app} {name}: ${cost:.2f}  " + ", ".join(f"{k} UTC ${v:.2f}" for k, v in sorted(h.items())))
    for app, cost in other.items():
        total += cost
        print(f"{app} (a fetch): ${cost:.2f}")
    print(f"all: ${total:.2f}")


if __name__ == "__main__":
    main()
