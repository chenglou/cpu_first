"""FAMILY_FILE.md §11: this phase's cost, from the billing report's hours for its apps (as legal_nobook/cost.py reads
them). The report covers complete hours only, so an hour still under way shows what is billed so far.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_family/cost.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-OaSkeF0NN94IgRrvJQG77i": "the nine tiny cores (B200s)",
        "ap-47GRV2nMoyyPlRPdJx0P4C": "the official run's first launch (the runner copy failed at import; no job ran)",
        "ap-kMy7lHDCkebWeYGrk61Dng": "a launch the runner refused (its record name was taken)",
        "ap-7MCyQ0iwCFmtLA5gD1KEeT": "H1's official run (H100)",
        "ap-Lbe9N0JDKXkjKMHkOcB3FT": "H1's seeds 1-7 (H100s)",
        "ap-SChXfyNyNUutgLz60AVZPM": "the family, eight tasks (H100s)"}
START = datetime(2026, 10, 27, 18, tzinfo=timezone.utc)       # the hour the first app started in (11:21 PDT)


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
