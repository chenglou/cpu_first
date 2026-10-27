"""FAMILY.md §5.4: two climb records the same apart from times and from ROUNDT's own records (bar3/scale/compose/same.py,
with every "roundt" record dropped first)?

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.family.same_but_roundt A.json B.json
"""

from __future__ import annotations

import json
import sys

from bar3.scale.compose.same import strip


def main():
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    for r in (a, b):
        r["records"] = [x for x in r["records"] if x.get("task") != "roundt"]
    diff = [k for k in ("history", "records", "lib", "rounds", "ops", "near", "units") if strip(a[k]) != strip(b[k])]
    print("same apart from times and ROUNDT's records" if not diff else f"differ in {diff}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
