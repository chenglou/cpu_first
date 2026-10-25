"""A spend limit for one Modal app (HELDOUT.md §3), run beside it. Every few minutes it logs the app's cost so far:
the billing report's complete hours for that app id, plus the hour under way at the last complete hour's rate (the
report lags by up to an hour). At or above STOP it stops the app; the records written so far stay on the Volume. It
exits when the app has stopped, by itself or by this.

    ~/github/sotaku/venv/bin/python bar3/scale/heldout/watch_spend.py APP_ID STOP [MINUTES]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from modal.billing import workspace_billing_report


def app(app_id):
    out = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"], capture_output=True, text=True,
                         check=True).stdout
    return next((a for a in json.loads(out) if a["App ID"] == app_id), None)


def cost(app_id, since):
    """(the complete hours' cost, the estimate with the hour under way) for the app since `since`."""
    hours = defaultdict(float)
    for r in workspace_billing_report(start=since, resolution="h"):
        if r.get("object_id") == app_id:
            hours[r["interval_start"]] += float(r["cost"])
    billed = sum(hours.values())
    if not hours:
        return billed, billed
    last = max(hours)
    under_way = (datetime.now(timezone.utc) - (last + timedelta(hours=1))).total_seconds() / 3600
    return billed, billed + max(0.0, under_way) * hours[last]


def main():
    app_id, stop = sys.argv[1], float(sys.argv[2])
    minutes = float(sys.argv[3]) if len(sys.argv) > 3 else 15
    a = app(app_id)
    since = (datetime.fromisoformat(a["Created at"]).astimezone(timezone.utc).replace(minute=0, second=0,
                                                                                       microsecond=0))
    while True:
        a = app(app_id)
        billed, est = cost(app_id, since)
        print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC {app_id}: {a['State']}, {a['Tasks']} tasks; "
              f"${billed:.2f} billed, about ${est:.2f} with the hour under way (stop at ${stop:.2f})", flush=True)
        if a["Stopped at"] or a["State"].startswith("stopped"):
            print("the app has stopped", flush=True)
            return
        if est >= stop:
            print(f"AT OR OVER ${stop:.2f}: stopping {app_id}", flush=True)
            subprocess.run([sys.executable, "-m", "modal", "app", "stop", app_id], check=False)
            return
        time.sleep(60 * minutes)


if __name__ == "__main__":
    main()
