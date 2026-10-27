"""LEGAL_FILE.md §7: this study's cost, from the billing report's hours for its apps (as bar3/scale/slice/cost.py reads
them). The report covers complete hours only, so an hour still under way shows what is billed so far.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/combined_legal/cost.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-p4him667wjHiR6agj9bn0V": "the tiny core (B200)",
        "ap-N6PsewCB8njYPVm7Bp61fQ": "a fetch",
        "ap-QRntMmiDUFfeAOMdLH1hB6": "the official run (H100)",
        "ap-5ZlPSwmeFz1qsCQW12BAFR": "seeds 1-7 (H100s)"}
START = datetime(2026, 10, 27, 5, tzinfo=timezone.utc)        # the hour the first app started in (22:43 PDT)


def main():
    hours = defaultdict(lambda: defaultdict(float))
    other = defaultdict(float)
    for r in workspace_billing_report(start=START, resolution="h"):
        if r.get("object_id") in APPS:
            hours[r["object_id"]][f"{r['interval_start']:%H:%M}"] += float(r["cost"])
        elif "combined-legal" in str(r.get("description", "")):
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
