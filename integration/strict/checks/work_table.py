"""The strict files' work against the clock (STRICT.md §2): for each record, the climb's work and seconds in all, its
units' and its searches' largest, and how many seconds a unit of work took (in all, and at worst over the units that
did real work). This is what the budgets SEARCH_WORK and CLIMB_WORK are set from.

    python3 integration/strict/checks/work_table.py REC.json ...
"""
import ast
import json
import re
import sys


def scan(path):
    rec = json.load(open(path))
    out = rec["stdout"]
    units = []
    for l in out.splitlines():
        m = re.search(r"^LR climb step=(\d+) (.*) \[work \+([0-9.e+-]+), ([0-9.e+-]+) in all\] \((\d+\.\d+)s\)$", l)
        if m:
            units.append((int(m.group(1)), float(m.group(3)), float(m.group(5)), m.group(2)[:60]))
    searches = []
    for l in out.splitlines():
        if not l.startswith("SEARCH "):
            continue
        try:
            r = ast.literal_eval(l[7:])
        except Exception:
            continue
        for run in r.get("runs") or []:
            if "work" in run:
                searches.append((run["work"], run.get("seconds", 0.0), run.get("how", "")[:30], r.get("level")))
        if r.get("task") == "round" and "work" in r:
            searches.append((r["work"], r.get("seconds", 0.0), "round", r.get("level")))
    total_w = sum(u[1] for u in units)
    total_s = sum(u[2] for u in units)
    busy = [u for u in units if u[1] > 1e6]
    worst = max((u[2] / u[1] for u in busy), default=0.0)
    sr = (rec["result"].get("seeds") or [{}])[0]
    dp = sr.get("depth_profile") or {}
    hist = next((l for l in out.splitlines() if l.startswith("SCHEDULE")), "")
    rnd = re.search(r"\((\d+), 'round'", hist)
    frz = re.search(r"\((\d+), 'frozen'", hist)
    return dict(name=rec.get("name"), gpu=(rec.get("gpu") or "").split(",")[0], units=len(units), work=total_w,
                secs=total_s, s_per_unit=total_s / total_w if total_w else 0.0, worst_s_per_unit=worst,
                unit_max=max(units, key=lambda u: u[1], default=None),
                search_max=max(searches, default=None), round_step=rnd and int(rnd.group(1)),
                frozen_step=frz and int(frz.group(1)), max_t=(dp.get("max_certified_time_steps"),
                                                           dp.get("ood_n_max_certified_time_steps")),
                train_s=sr.get("training_seconds"), state=sr.get("model_state_elements"))


def main():
    for p in sys.argv[1:]:
        d = scan(p)
        um, sm = d["unit_max"], d["search_max"]
        print(f"{d['name']} ({d['gpu']}): T {d['max_t']}, round at step {d['round_step']}, frozen at {d['frozen_step']}, "
              f"training {d['train_s'] and round(d['train_s'])} s, state {d['state']}; climb {d['units']} units, work "
              f"{d['work']:.4g} in {d['secs']:.0f} s ({d['s_per_unit'] * 1e6:.3g} s per 1e6 units, worst unit "
              f"{d['worst_s_per_unit'] * 1e6:.3g}); largest unit {um and f'{um[1]:.4g} in {um[2]:.0f} s ({um[3]})'}; "
              f"largest search {sm and f'{sm[0]:.4g} in {sm[1]} s ({sm[2]}, level {sm[3]})'}")


if __name__ == "__main__":
    main()
