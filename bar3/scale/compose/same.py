"""Are two climb records the same apart from wall-clock times (COMPOSE.md §2.3)? Compares history, records, lib, rounds,
ops, near and the units' lines, with every "seconds", "t" and "s" field dropped, and in a line every "<n>s" or "+<n>s"
and every phase's time in a list of (phase, seconds) pairs.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose.same A.json B.json
"""

from __future__ import annotations

import json
import re
import sys

TIMES = re.compile(r"\+?\s*[0-9]+(\.[0-9]+)?s\b")
PHASES = re.compile(r"""(['"][^'"]*['"]), [0-9]+(\.[0-9]+)?\)""")


def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if k not in ("seconds", "t", "s", "launch", "elapsed")}
    if isinstance(v, list):
        return [strip(x) for x in v]
    if isinstance(v, str):
        return PHASES.sub(r"\1)", TIMES.sub("", v))
    return v


def main():
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    diff = [k for k in ("history", "records", "lib", "rounds", "ops", "near", "units") if strip(a[k]) != strip(b[k])]
    print("same apart from times" if not diff else f"differ in {diff}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
