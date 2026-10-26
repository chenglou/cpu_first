"""What compare_cores.py masks or leaves out (FAST.md §4), for two tiny-core records: each unit's work and the running
total, as floats, bit for bit; the history (each level's pool, counts, picks, learnable programs, merges, library),
the final library, rounds and routines by law and pattern (ops); the answer program's record but its seconds and
work; and the seconds, unit by unit, side by side.

    python3 integration/combined_fast/checks/compare_more.py A.json B.json
"""
import json
import sys

A, B = (json.load(open(p)) for p in sys.argv[1:3])
ua, ub = A["units"], B["units"]
n = min(len(ua), len(ub))
work = [i for i in range(n) if ua[i]["work"] != ub[i]["work"] or ua[i]["total"] != ub[i]["total"]]
print(f"units: {len(ua)} and {len(ub)}; work and running total equal as floats on {n - len(work)} of {n}"
      + (f"; first difference at unit {work[0]}" if work else ""))
print(f"total work: {A['work']!r} and {B['work']!r}: {'equal' if A['work'] == B['work'] else 'DIFFER'}")
for key in ("history", "lib", "rounds", "ops"):
    print(f"{key}: {'equal' if A[key] == B[key] else 'DIFFER'}")
oa, ob = A.get("outer"), B.get("outer")
if oa is not None or ob is not None:
    strip = lambda o: {k: v for k, v in (o or {}).items() if k not in ("wall", "seconds", "work")}
    print(f"outer (but its seconds and work): {'equal' if strip(oa) == strip(ob) else 'DIFFER'}; "
          f"work {oa.get('work')!r} and {ob.get('work')!r}; wall {oa.get('wall')} s and {ob.get('wall')} s")
print(f"seconds: {A['seconds']} and {B['seconds']} ({A['seconds'] / max(B['seconds'], 1e-9):.2f} times)")
lv = {}
for tag, units in (("a", ua), ("b", ub)):
    for u in units:
        level = u["line"].split(" ")[0]
        lv.setdefault(level, {"a": 0.0, "b": 0.0})[tag] += u["s"]
print("by level (s): " + "; ".join(f"{k} {v['a']:.0f} -> {v['b']:.0f}" for k, v in lv.items()))
