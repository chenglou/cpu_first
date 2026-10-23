"""The A/B climbs' records so far (each run saves its record to the Volume after every unit of work): status, seconds,
what was solved and when, stalls, skipped twins, the last log lines. Copies the records to lib/records/.

    cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python ../cpu_first/integration/lib/peek_ab.py ab_B_exact_h100 ab_C_exact_h100
"""
import json
import sys
from pathlib import Path

import modal

OUT = Path(__file__).resolve().parent / "records"
vol = modal.Volume.from_name("integration")
for name in sys.argv[1:]:
    try:
        data = b"".join(vol.read_file(f"climb/{name}.json"))
    except Exception as exc:
        print(f"== {name}: no record yet ({type(exc).__name__})")
        continue
    (OUT / f"{name}.json").write_bytes(data)
    r = json.loads(data)
    print(f"== {name} [{r.get('status')}] {r.get('seconds')} s; solved (s): {r.get('solved_s')}; stalled {r.get('stalled')}; "
          f"skipped {r.get('skipped')}; rounds {r.get('rounds')}")
    for line in r.get("log", [])[-6:]:
        print("   ", line[:220])
