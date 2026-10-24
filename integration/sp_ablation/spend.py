"""The workspace's Modal bill so far this month (UTC days; the billing report covers complete hours only), and the last
few hours: checked before every ablation launch. The seed-74 stage launched nothing new at $1,900 or more of the
$2,000 limit (STOP 1900); the user raised the limit before the second stage.

    ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py [STOP]    (STOP: no new launch at or above it)
"""

import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from modal.billing import workspace_billing_report

stop = float(sys.argv[1]) if len(sys.argv) > 1 else None
now = datetime.now(timezone.utc)
today = now.replace(hour=0, minute=0, second=0, microsecond=0)
days = workspace_billing_report(start=today.replace(day=1), resolution="d")    # daily reports span at most 31 days,
hours = defaultdict(float)                                                      # hourly ones 7; today is hourly only
for r in workspace_billing_report(start=min(today, now - timedelta(hours=6)), resolution="h"):
    hours[r["interval_start"]] += float(r["cost"])
total = (sum(float(r["cost"]) for r in days if r["interval_start"] < today)
         + sum(c for h, c in hours.items() if h >= today))
print(f"{now:%Y-%m-%d %H:%M} UTC: ${total:.2f} billed this month")
for h in sorted(hours)[-4:]:
    print(f"  {h:%m-%d %H:00} ${hours[h]:.2f}")
if stop is not None:
    print("LAUNCH OK" if total < stop else f"AT OR OVER ${stop:,.0f}: launch nothing new")
