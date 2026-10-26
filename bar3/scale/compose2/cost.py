"""COMPOSE2.md §9: round 2's cost, from the billing report's hours for each of its apps (as
bar3/scale/digits/watch_spend.py reads them). An hour still under way shows what is billed so far.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/compose2/cost.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-DPry7prHfDelGRUXLYVjc1": "spell_test (07:44 PDT)", "ap-wnJWCOromJdTZcgvKQqdPw": "spell_fix (07:53)",
        "ap-sCIzoY3CrMShugxjwu1Jyh": "lawnear_test (08:09)",
        "ap-VdukrJkH3FphVS1bSUmMuk": "the replays and B1, B1lp, B2, B2lp (08:13)",
        "ap-g0qTjnbvnmnzRbQq483tNE": "spell_full (08:36)", "ap-NWMwLMOcrtRHgmq6XCUGWS": "rest_deep_test (08:40)",
        "ap-qZJXhZCdf2hUwpjfbam0zv": "B2rd, B2lp_rd (08:43)", "ap-yflupemrMgMmPD0vbgLDrR": "four flags off (08:57)",
        "ap-S7H7am4iYYIHLWYv7vYKvB": "two more flags off (09:10)"}
START = datetime(2026, 10, 26, 14, tzinfo=timezone.utc)      # the hour the first app was created in (14:44 UTC)


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
