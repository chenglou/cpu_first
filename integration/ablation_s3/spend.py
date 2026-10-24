"""The workspace's Modal bill so far this month (UTC days; the billing report covers complete hours only), and the last
few hours: checked before every ablation launch (no new launch at $1,900 or more of the $2,000 limit).

    ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from modal.billing import workspace_billing_report

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
print("LAUNCH OK" if total < 1900 else "AT OR OVER $1,900: launch nothing new")
