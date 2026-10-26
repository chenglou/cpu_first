"""A tiny-core record level by level (core_c.py's, or core_g.py's for comparison): each level's seconds and machine work
(the work only in core_c's records), how many candidates it posed, what it practised, and the round's searches.

    cd cpu_first && python3 integration/combined_sp2/summarize_c.py RECORD.json [REFERENCE.json ...]
"""
import json
import re
import sys


def levels(path):
    r = json.load(open(path))
    out, cur = {}, None
    for u in r["units"]:
        m = re.match(r"L(\d+) ", u["line"])
        if not m:
            continue
        lv = int(m.group(1))
        d = out.setdefault(lv, dict(s=0.0, work=0.0, end=0.0, posed="", picks="", round=""))
        d["s"] += u["s"]
        d["work"] += u.get("work", 0.0)
        d["end"] = u["t"]
        line = u["line"]
        if " pose: " in line:
            d["posed"] = line.split(" pose: ")[1].split(" candidates")[0]
        elif " choose " in line:
            d["picks"] = ", ".join(re.findall(r"\('([^']+)', \d+, ", line))
        elif " round: " in line or " round again" in line:
            d["round"] = line.split(": ", 1)[1][:90]
    return r, out


for path in sys.argv[1:]:
    r, lv = levels(path)
    print(f"== {path}: {r['seconds']} s, final {r.get('final')}, work {r.get('work', 0):.4g}; rounds {r['rounds'][:2]}")
    for k in sorted(lv):
        d = lv[k]
        print(f"  L{k}: {d['s']:7.1f} s (ends {d['end']:7.1f} s), work {d['work']:.3g}, posed {d['posed'] or '-'}; "
              f"picks {d['picks'] or '-'}; round {d['round'] or '-'}")
