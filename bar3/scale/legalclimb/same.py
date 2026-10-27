"""Two tiny-core records the same apart from times (LEGALCLIMB.md §3): the copies with no full-size root against round
4's copies, on the same env. Compares every field but the times (units' t and s, seconds, and every "seconds" or
"wall" inside), and the device line.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.legalclimb.same A.json B.json
"""

from __future__ import annotations

import json
import re
import sys

TIMES = {"seconds", "wall", "t", "s"}


def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if k not in TIMES}
    if isinstance(v, list):
        return [strip(x) for x in v]
    if isinstance(v, str):
        return re.sub(r"\d+\.\d+", "#", v)                  # times (and any decimals) inside the units' lines
    return v


def main():
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    for k in ("tag",):
        a.pop(k, None)
        b.pop(k, None)
    sa, sb = strip(a), strip(b)
    if sa == sb:
        print(f"{sys.argv[1]} and {sys.argv[2]}: same apart from times")
        return 0
    for k in sorted(set(sa) | set(sb)):
        if sa.get(k) != sb.get(k):
            print("differs:", k, json.dumps(sa.get(k))[:300], "|", json.dumps(sb.get(k))[:300])
    return 1


if __name__ == "__main__":
    sys.exit(main())
