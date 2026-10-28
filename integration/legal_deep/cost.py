"""DEEP.md §11: this phase's cost, from the billing report's hours for its apps (as legal_nobook/cost.py reads
them). The report covers complete hours only, so an hour still under way shows what is billed so far.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_deep/cost.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-NsrWyFLvQ7xZKxpvqvLlb4": "exploration: 3 tiny cores with no work budget (B200s)",
        "ap-yXWtXxssZi7kRCRl9LjIb4": "the reader on the space, the candidate (6 B200s)",
        "ap-oppsXTCVBm9uBGIYSoifGz": "arms (b) and (c): 7 tiny cores (B200s)",
        "ap-9uTw2tfmYYgAM0LQ4CPWkc": "arm (d): 6 tiny cores (B200s)",
        "ap-Jmn8ASRP54qjCmfIkqM5sk": "the file's 17 tiny cores and the reader on the space (B200s)",
        "ap-a3xoljJyvm3pI1HI7XxOEf": "H1's official run (H100)",
        "ap-SyOPls8n3mUMV6hfYyXfIq": "H1's seeds 1-7 (H100s)",
        "ap-ogpgOFTsCiEosqULmkZptW": "the 16 task runs (H100s)",
        "ap-rf8Js4Far5isrIxwzaz41e": "the md5 six's tiny cores (B200s; PREREG_ADDENDUM1.md)",
        "ap-AT8QWRDJMozSqUdxt79Q8x": "the md5 six's runs (H100s; PREREG_ADDENDUM1.md)"}
START = datetime(2026, 10, 28, 0, tzinfo=timezone.utc)       # the hour the first app started in (17:56 PDT)


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
