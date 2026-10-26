"""A spend limit for several Modal apps at once (DIGITS.md §4): bar3/scale/heldout/watch_spend.py for a list of apps,
with one billing report per poll for all of them (six watchers polling together hit the report's rate limit), retried
when it fails. Every few minutes it logs each app's cost so far (the report's complete hours, plus the hour under way
at the last complete hour's rate). It stops an app at STOP, and every app once their sum reaches TOTAL. It exits when
all have stopped.

    ~/github/sotaku/venv/bin/python bar3/scale/digits/watch_spend.py STOP TOTAL APP_ID ...
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from modal.billing import workspace_billing_report


GONE = {"State": "stopped (no longer listed)", "Tasks": 0, "Stopped at": "unlisted"}


def apps(ids):
    """{app id: its row in `modal app list`}; an app the list no longer shows (it drops stopped apps after a while) is
    taken as stopped."""
    out = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"], capture_output=True, text=True,
                         check=True).stdout
    rows = {a["App ID"]: a for a in json.loads(out)}
    return {a: rows.get(a, GONE) for a in ids}


def costs(ids, since, tries=5):
    """{app id: (complete hours' cost, the estimate with the hour under way)}, retrying a failed report."""
    for t in range(tries):
        try:
            hours = defaultdict(lambda: defaultdict(float))
            for r in workspace_billing_report(start=since, resolution="h"):
                if r.get("object_id") in ids:
                    hours[r["object_id"]][r["interval_start"]] += float(r["cost"])
            break
        except Exception as e:  # noqa: BLE001  (a rate limit or a network error: wait and ask again)
            print(f"billing report failed ({type(e).__name__}); retrying", flush=True)
            time.sleep(60 * (t + 1))
    else:
        return None
    now = datetime.now(timezone.utc)
    out = {}
    for a in ids:
        h = hours.get(a, {})
        billed = sum(h.values())
        est = billed
        if h:
            last = max(h)
            est += max(0.0, (now - (last + timedelta(hours=1))).total_seconds() / 3600) * h[last]
        out[a] = (billed, est)
    return out


def stop(a):
    subprocess.run([sys.executable, "-m", "modal", "app", "stop", a], check=False)


def main():
    lim, total, ids = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]
    first = apps(ids)
    since = min(datetime.fromisoformat(first[a]["Created at"]).astimezone(timezone.utc) for a in ids
                if "Created at" in first[a])
    since = since.replace(minute=0, second=0, microsecond=0)
    stopped = set()
    while True:
        st = apps(ids)
        c = costs(ids, since)
        now = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
        if c is None:
            print(f"{now}: no billing report after retries; trying again next poll", flush=True)
        else:
            s = sum(e for _, e in c.values())
            for a in ids:
                print(f"{now} {a}: {st[a]['State']}, {st[a]['Tasks']} tasks; ${c[a][0]:.2f} billed, about "
                      f"${c[a][1]:.2f} with the hour under way (stop at ${lim:.2f})", flush=True)
            print(f"{now} all: about ${s:.2f} (stop all at ${total:.2f})", flush=True)
            for a in ids:
                if a not in stopped and not st[a]["State"].startswith("stopped") and (c[a][1] >= lim or s >= total):
                    print(f"{now} AT OR OVER THE LIMIT: stopping {a}", flush=True)
                    stop(a)
                    stopped.add(a)
        if all(st[a]["State"].startswith("stopped") or st[a]["Stopped at"] for a in ids):
            print("all apps have stopped", flush=True)
            return
        time.sleep(15 * 60)


if __name__ == "__main__":
    main()
