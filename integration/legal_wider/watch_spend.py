"""Phase D's watcher (legal_wider/WIDER.md): phase C's (a stopped app's hour under way ends at its stop), with the
margin counted from the containers still running: it stops every app once the estimate plus one more hour of the
running containers (their number, from `modal app list`, times PRICE a container-hour) reaches CAP, and any app at
STOP. Phase C's description, then the original's: a stopped app's hour under way ends when it stopped. The original
carried every app's last complete hour forward to now, so an app stopped hours ago kept adding to the sum and the
watcher cut live runs (round 7, FAMILY.md).

A spend limit for several Modal apps at once (DIGITS.md §4): bar3/scale/heldout/watch_spend.py for a list of apps,
with one billing report per poll for all of them (six watchers polling together hit the report's rate limit), retried
when it fails. Every few minutes it logs each app's cost so far (the report's complete hours, plus the hour under way
at the last complete hour's rate). It stops an app at STOP, and every app once their sum reaches TOTAL. It exits when
all have stopped.

    ~/github/sotaku/venv/bin/python integration/legal_wider/watch_spend.py STOP CAP PRICE APP_ID ...
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


def stopped_at(row):
    """When an app stopped: None while it runs; the epoch for one the list no longer shows, or whose time does not
    parse (long stopped: no hour under way)."""
    s = row.get("Stopped at")
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def costs(ids, since, ends, tries=5):
    """{app id: (complete hours' cost, the estimate with the hour under way)}, retrying a failed report; ends: {app
    id: when it stopped, or None}, where its hour under way ends."""
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
            end = now if ends.get(a) is None else min(now, ends[a])
            est += max(0.0, (end - (last + timedelta(hours=1))).total_seconds() / 3600) * h[last]
        out[a] = (billed, est)
    return out


def stop(a):
    subprocess.run([sys.executable, "-m", "modal", "app", "stop", a], check=False)


def main():
    lim, cap, price, ids = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
    first = apps(ids)
    since = min(datetime.fromisoformat(first[a]["Created at"]).astimezone(timezone.utc) for a in ids
                if "Created at" in first[a])
    since = since.replace(minute=0, second=0, microsecond=0)
    stopped = set()
    while True:
        st = apps(ids)
        c = costs(ids, since, {a: stopped_at(st[a]) for a in ids})
        now = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
        if c is None:
            print(f"{now}: no billing report after retries; trying again next poll", flush=True)
        else:
            s = sum(e for _, e in c.values())
            run = sum(int(st[a]["Tasks"] or 0) for a in ids if not st[a]["State"].startswith("stopped"))
            total = cap - run * price                           # the cap less one hour of the containers running
            for a in ids:
                print(f"{now} {a}: {st[a]['State']}, {st[a]['Tasks']} tasks; ${c[a][0]:.2f} billed, about "
                      f"${c[a][1]:.2f} with the hour under way (stop at ${lim:.2f})", flush=True)
            print(f"{now} all: about ${s:.2f}; {run} containers running (stop all at ${total:.2f})", flush=True)
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
