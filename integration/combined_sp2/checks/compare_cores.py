"""Two tiny-core records (core_c.py's) unit by unit and search by search, the seconds and the work masked (COMBINED.md
§8): the first unit whose line differs (the lines are kept to 600 characters, so they are compared to the shorter
one's length), the first search whose results differ (its roots, the programs it found for each law and root, its
level sizes and stop), and how much matches before each.

    python3 integration/combined_sp2/checks/compare_cores.py A.json B.json
"""
import ast
import json
import re
import sys


def masked(line):
    line = re.sub(r"\d+(\.\d+)?s\b", "#s", line)                       # a unit's or a search's seconds
    line = re.sub(r"\('([^']*)', [\d.]+\)", r"('\1', #)", line)      # (how, seconds) of each try
    return re.sub(r'\("([^"]*)", [\d.]+\)', r'("\1", #)', line)       # (... when how has quotes of its own)


def result(rec):
    """A search record without its seconds and work: what it searched and found."""
    out = {}
    for k, v in rec.items():
        if k in ("seconds", "work", "runs"):
            continue
        if k == "laws" and isinstance(v, str):
            v = ast.literal_eval(v)
        out[k] = v
    if "runs" in rec:                                                  # each run's roots, sizes, how; not its time
        runs = rec["runs"] if isinstance(rec["runs"], list) else ast.literal_eval(rec["runs"])
        out["runs"] = [{k: v for k, v in r.items() if k not in ("seconds", "work")} for r in runs]
    return json.dumps(out, sort_keys=True, default=str)


A, B = (json.load(open(p)) for p in sys.argv[1:3])
a, b = A["units"], B["units"]
n = min(len(a), len(b))


def same(x, y):
    x, y = masked(x), masked(y)
    k = min(len(x), len(y))
    return x[:k] == y[:k]


first = next((i for i in range(n) if not same(a[i]["line"], b[i]["line"])), None)
if first is None:
    print(f"lines: the first {n} units are the same (of {len(a)} and {len(b)})")
else:
    print(f"lines: units 0-{first - 1} are the same ({first} units, to {a[first - 1]['t']} s and {b[first - 1]['t']} s);"
          f" the first difference, unit {first}:")
    for name, u in (("A", a[first]), ("B", b[first])):
        print(f"  {name} at {u['t']} s: {u['line'][:300]}")
ra, rb = A["records"], B["records"]
m = min(len(ra), len(rb))
first = next((i for i in range(m) if result(ra[i]) != result(rb[i])), None)
if first is None:
    print(f"searches: the first {m} are the same (of {len(ra)} and {len(rb)})")
else:
    x, y = json.loads(result(ra[first])), json.loads(result(rb[first]))
    print(f"searches: 0-{first - 1} are the same ({first} searches); the first difference, search {first} (level "
          f"{x.get('level')}, roots {x.get('roots', x.get('task'))}):")
    for k in sorted(set(x) | set(y)):
        if x.get(k) != y.get(k):
            if isinstance(x.get(k), dict) and isinstance(y.get(k), dict):
                for kk in sorted(set(x[k]) | set(y[k])):
                    if x[k].get(kk) != y[k].get(kk):
                        print(f"  {k}[{kk}]: A {str(x[k].get(kk))[:200]} | B {str(y[k].get(kk))[:200]}")
            else:
                print(f"  {k}: A {str(x.get(k))[:300]} | B {str(y.get(k))[:300]}")
