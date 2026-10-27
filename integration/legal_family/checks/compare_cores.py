"""H1's tiny core against phase B's (FAMILY_FILE.md §4): unit by unit, each unit's line with its seconds masked and its
machine work; where they first differ, and the picks, library, rounds and answer programs of each.

    cd cpu_first && python3 integration/legal_family/checks/compare_cores.py [OLD.json NEW.json]
"""
import json
import re
import sys
from pathlib import Path

CPU = Path(__file__).resolve().parents[3]
OLD, NEW = (sys.argv[1:3] if len(sys.argv) > 2 else
            (CPU / "integration/legal_nobook/records/t_nobook.json", CPU / "integration/legal_family/records/t_family_h1.json"))


def mask(line):
    """Seconds masked: "12.3s", and a probe's ('how', seconds) pairs."""
    return re.sub(r"(\('[^']*', )\d+\.\d+\)", r"\1T)", re.sub(r"\d+\.\d+s\b", "Ts", line))


def main():
    a, b = (json.load(open(p)) for p in (OLD, NEW))
    ua, ub = a["units"], b["units"]
    same = 0
    for x, y in zip(ua, ub):
        if mask(x["line"]) != mask(y["line"]) or x["work"] != y["work"]:
            break
        same += 1
    print(f"{OLD}: {len(ua)} units; {NEW}: {len(ub)} units; the first {same} the same (line and work)")
    for x, y in list(zip(ua, ub))[same:same + 2]:
        print(f"  old: work {x['work']:.6g} {mask(x['line'])[:260]}\n  new: work {y['work']:.6g} {mask(y['line'])[:260]}")
    for k in ("picks",):
        for h, g in zip(a["history"], b["history"]):
            print(f"  level {h['level']} picks {'the same' if h[k] == g[k] else 'differ'}: {g[k]}"
                  + ("" if h[k] == g[k] else f" (old: {h[k]})"))
    la, lb = a["lib"], b["lib"]
    print(f"  library: {len(la)} and {len(lb)} entries; only old {sorted(set(la) - set(lb))}; only new "
          f"{sorted(set(lb) - set(la))}; differing {sorted(k for k in set(la) & set(lb) if la[k] != lb[k])}")
    print(f"  rounds the same: {a['rounds'] == b['rounds']} ({len(a['rounds'])}, {len(b['rounds'])})")
    print(f"  answer programs: {a['outer']['programs']} | {b['outer']['programs']}")
    print(f"  the climb's work: {a['work']:.6g} and {b['work']:.6g}; seconds {a['seconds']} and {b['seconds']}")


if __name__ == "__main__":
    main()
