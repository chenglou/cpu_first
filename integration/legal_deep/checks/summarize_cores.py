"""DEEP.md §2–§4: each tiny core's record (records/<TAG>.json): its file (md5) and settings, the law read, the
level and work of its first round and the round, its answer programs, its levels and work in all, its seconds.

    cd cpu_first && python3 integration/legal_deep/checks/summarize_cores.py [PREFIX ...]   (default: every record)
"""
import json
import re
import sys
from pathlib import Path

REC = Path(__file__).resolve().parents[1] / "records"


def main():
    prefixes = sys.argv[1:] or [""]
    for p in sorted(REC.glob("*.json")):
        if not any(p.stem.startswith(x) for x in prefixes) or p.stem.startswith(("legal_", "deep_")):
            continue
        r = json.loads(p.read_text())
        cfg = r.get("config", {})
        units = r.get("units", [])
        read = next((re.search(r"read (\S+) on (\d+) of (\d+)", u["line"]) for u in units if " pose: " in u["line"]
                     and "read " in u["line"]), None)
        rnd = next((u for u in units if re.search(r"round: cost \d", u["line"])), None)
        levels = max((int(m.group(1)) for u in units for m in [re.search(r" L(\d+) ", " " + u["line"])] if m),
                     default=None)
        o = r.get("outer") or {}
        progs = o.get("programs", []) if isinstance(o, dict) else []
        line = (f"{p.stem}: {cfg.get('task')} file {cfg.get('file_md5', '')[:8]} sets {cfg.get('sets')}; "
                f"read {read.group(1) + ' on ' + read.group(2) + '/' + read.group(3) if read else '-'}; ")
        if rnd:
            lv = re.search(r"L(\d+) round", rnd["line"])
            prog = re.search(r"\[('[^']*')", rnd["line"][rnd["line"].index("round:"):])
            line += (f"round after level {lv.group(1)} at {rnd['total']:.3g} ({prog.group(1) if prog else ''}); "
                     f"answer programs {progs[:2]}; ")
        else:
            line += f"no round (to level {levels}); "
        line += f"work {r.get('work', 0):.3g}, {r.get('seconds')} s{'' if r.get('final') else ' (unfinished)'}"
        print(line)


if __name__ == "__main__":
    main()
