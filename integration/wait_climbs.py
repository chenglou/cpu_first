"""Wait until the named climb records are on the `integration` Volume (climb/), download them to integration/lib/records/
and print each one's solved tasks and round (polls every 60 s, up to 8 h).

    python wait_climbs.py NAME [NAME ...]
"""
import json
import sys
import time
from pathlib import Path

import modal

OUT = Path(__file__).resolve().parent / "lib" / "records"


def main(names):
    vol = modal.Volume.from_name("integration")
    OUT.mkdir(parents=True, exist_ok=True)
    pending, t0 = set(names), time.time()
    while pending and time.time() - t0 < 8 * 3600:
        try:
            have = {Path(e.path).stem: e for e in vol.listdir("climb")}
        except Exception as exc:
            print("listdir failed:", repr(exc)[:200], flush=True)
            time.sleep(60)
            continue
        for name in sorted(pending & set(have)):
            dest = OUT / f"{name}.json"
            with open(dest, "wb") as f:
                for chunk in vol.read_file(have[name].path):
                    f.write(chunk)
            r = json.loads(dest.read_text())
            print(name, r.get("seconds"), "solved", r.get("solved_at"), "rounds", r.get("rounds"), flush=True)
            pending.discard(name)
        if pending:
            time.sleep(60)
    print("still pending:", sorted(pending), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
