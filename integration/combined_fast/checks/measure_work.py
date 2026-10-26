"""The climbs' work on the official manifest, from evaluator records (FAST.md §9), and the budgets STRICT.md §2's rule
gives: per record, the climb's work at its round and after the answer program's search (the climb's whole count, as
the file counts it against CLIMB_WORK), the answer search's own, the largest single search measured as round 2 and
STRICT.md measured it (a probe unit's work, from its LR climb line: a unit is one or more runs, and SEARCH_WORK bounds
each run, so this is the larger, safer reading) and the largest run (from the SEARCH lines), with any work-budget stop;
then CLIMB_WORK = 1.22 times the largest climb, rounded up to 5e7, and SEARCH_WORK = twice the largest search,
rounded up to 1e8, as round 2 rounded. EXTRA: other measured largest searches and climbs (name=search,climb).

    python3 integration/combined_fast/checks/measure_work.py RECORD.json ... [--extra tiny_core=3.34e8,2.651e9]
"""
import ast
import json
import math
import re
import sys

args = sys.argv[1:]
extra = {}
if "--extra" in args:
    for kv in args[args.index("--extra") + 1].split(";"):
        k, v = kv.split("=")
        extra[k] = tuple(float(x) for x in v.split(","))
    args = args[:args.index("--extra")]
climbs, searches = dict((k, v[1]) for k, v in extra.items()), dict((k, v[0]) for k, v in extra.items())
for path in args:
    r = json.load(open(path))
    out = r["stdout"].splitlines()
    seed = r["manifest"]["runtime"]["seeds"][0]
    rnd = next((l for l in out if l.startswith("LR climb") and " round: cost " in l and "round: cost None" not in l), None)
    at_round = float(re.search(r"\[work \+\S+, (\S+) in all\]", rnd).group(1)) if rnd else None
    outer = [l for l in out if l.startswith("LR outer")]
    found = [l for l in outer if "answer program: cost None" not in l]
    m = re.search(r"\[work \+(\S+), (\S+) in all\]", found[-1]) if found else None
    total = float(m.group(2)) if m else None
    best, unit = (0.0, ""), (0.0, "")
    for l in out:
        if l.startswith("SEARCH "):
            d = ast.literal_eval(l[len("SEARCH "):])
            for run in d.get("runs", []):
                if run["work"] > best[0]:
                    best = (float(run["work"]), f"level {d['level']} {run['how'][:24]}...")
        u = re.match(r"LR climb step=\d+ (L\d (?:probe|deep \[).{0,30}).*\[work \+(\S+), \S+ in all\]", l)
        if u and float(u.group(2)) > unit[0]:
            unit = (float(u.group(2)), u.group(1))
    stops = sum("work budget" in l for l in out if l.startswith(("LR climb", "LR outer")))
    name = f"seed {seed} ({path.split('/')[-1]})"
    climbs[name], searches[name] = total, unit[0]
    print(f"{name}: round at {at_round:.4g}, answer search +{float(m.group(1)) if m else float('nan'):.4g} "
          f"({len(outer)} searches), climb {total:.4g}; largest search {unit[0]:.4g} ({unit[1]}; its largest run "
          f"{best[0]:.4g}); work-budget stops {stops}")
cmax = max(v for v in climbs.values() if v is not None)
smax = max(searches.values())
cw = math.ceil(1.22 * cmax / 5e7) * 5e7
sw = math.ceil(2 * smax / 1e8) * 1e8
print(f"largest climb {cmax:.4g} ({[k for k, v in climbs.items() if v == cmax][0]}): CLIMB_WORK = 1.22 x = "
      f"{1.22 * cmax:.4g}, rounded up {cw:.4g}")
print(f"largest search {smax:.4g} ({[k for k, v in searches.items() if v == smax][0]}): SEARCH_WORK = 2 x = "
      f"{2 * smax:.4g}, rounded up {sw:.4g}")
