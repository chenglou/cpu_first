"""DEEP.md §2 and PREREG.md: the climb's seconds per million units of machine work on the H100, from the runs' logs.
For each record: the step the climb starts at and its elapsed time (interpolated between the logged step lines),
the climb's units (each "LR climb"/"LR outer" line: its work and its seconds), its work in all, the step and time
of the answer program and of the freeze, and training's end. Rates: all units together (seconds over work), and
the slowest stretch: the worst rate over any run of consecutive units holding at least MIN_WORK of work.

    cd cpu_first && python3 integration/legal_deep/checks/rates.py [RECORD.json ...]
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = sorted((ROOT / "legal_wider/records").glob("legal_wider_*.json")) + sorted(
    (ROOT / "legal_wider/records").glob("wider_*_h100.json"))
MIN_WORK = 5e8


def elapsed_at(steps, step):
    """Elapsed seconds at a step, by linear interpolation between the logged step lines."""
    lo = max((s for s in steps if s <= step), default=None)
    hi = min((s for s in steps if s >= step), default=None)
    if lo is None or hi is None:
        return None
    return steps[lo] if hi == lo else steps[lo] + (steps[hi] - steps[lo]) * (step - lo) / (hi - lo)


def one(path):
    r = json.loads(Path(path).read_text())
    out = r.get("stdout", "")
    steps = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"^step=(\d+) loss=\S+ \S+ elapsed=([\d.]+)s", out, re.M)}
    units = [(int(m.group(1)), float(m.group(3)), float(m.group(4)), m.group(2)) for m in re.finditer(
        r"^LR climb step=(\d+) (L\d+ \w+(?: again)?).*?\[work \+([\d.e+-]+), [\d.e+-]+ in all\] \(([\d.]+)s\)$",
        out, re.M)] + [(int(m.group(1)), float(m.group(3)), float(m.group(2)), "answer program") for m in re.finditer(
        r"^LR outer step=(\d+) answer program: cost \S+ levels \[[^\]]*\] ([\d.]+)s .*\[work \+([\d.e+-]+),", out, re.M)]
    units.sort()
    start = re.search(r"^LR climb_starts step=(\d+)", out, re.M)
    found = re.search(r"^LR round_found step=(\d+)", out, re.M)
    frozen = re.search(r"^LR frozen step=(\d+)", out, re.M)
    total = [float(x) for x in re.findall(r"\[work \+[\d.e+-]+, ([\d.e+-]+) in all\]", out)]
    work = max(total) if total else 0.0
    secs = sum(t for _, _, t, _ in units)
    kinds = {}
    for _, w, t, k in units:
        k = re.sub(r"^L\d+ ", "", k)
        kw, kt, kmax = kinds.get(k, (0.0, 0.0, 0.0))
        kinds[k] = (kw + w, kt + t, max(kmax, t / (w / 1e6) if w >= 1e6 else 0.0))
    worst = 0.0
    for i in range(len(units)):
        w = t = 0.0
        for j in range(i, len(units)):
            w, t = w + units[j][1], t + units[j][2]
            if w >= MIN_WORK:
                worst = max(worst, t / (w / 1e6))
                break
    res = (r.get("result") or {}).get("seeds") or [{}]
    train = r.get("train_seconds") or (res[0].get("train_seconds") if res else None)
    st = int(start.group(1)) if start else None
    return dict(name=Path(path).stem, start_step=st, start_s=elapsed_at(steps, st) if st else None,
                found_step=int(found.group(1)) if found else None,
                found_s=elapsed_at(steps, int(found.group(1))) if found else None,
                frozen_step=int(frozen.group(1)) if frozen else None,
                frozen_s=elapsed_at(steps, int(frozen.group(1))) if frozen else None,
                end_s=steps.get(max(steps)) if steps else None, work=work, unit_s=secs, units=len(units),
                rate=secs / (work / 1e6) if work else None, worst=worst or None, kinds=kinds)


def main():
    paths = sys.argv[1:] or DEFAULT
    rows = [one(p) for p in paths]
    f = lambda v, fmt: "-" if v is None else format(v, fmt)  # noqa: E731
    print(f"{'record':32s} {'climb from':>16s} {'round':>14s} {'frozen':>14s} {'end s':>7s} {'work':>8s} "
          f"{'unit s':>7s} {'s/M':>6s} {'worst s/M':>9s}")
    for d in rows:
        print(f"{d['name']:32s} {f(d['start_step'], 'd'):>6s} {f(d['start_s'], '.0f'):>7s} s "
              f"{f(d['found_step'], 'd'):>5s} {f(d['found_s'], '.0f'):>6s} s {f(d['frozen_step'], 'd'):>5s} "
              f"{f(d['frozen_s'], '.0f'):>6s} s {f(d['end_s'], '.0f'):>7s} {d['work']:8.3g} {d['unit_s']:7.0f} "
              f"{f(d['rate'], '.3f'):>6s} {f(d['worst'], '.3f'):>9s}")
    kinds = {}
    for d in rows:
        for k, (w, t, m) in d["kinds"].items():
            kw, kt, km = kinds.get(k, (0.0, 0.0, 0.0))
            kinds[k] = (kw + w, kt + t, max(km, m))
    print("by kind of unit, every record: work, seconds, s/M overall, the worst single unit's s/M (units of 1e6+)")
    for k, (w, t, m) in sorted(kinds.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:16s} {w:10.3g} {t:9.0f} s {t / (w / 1e6) if w else 0:7.3f} {m:7.3f}")
    rates = [d["rate"] for d in rows if d["rate"]]
    worsts = [d["worst"] for d in rows if d["worst"]]
    print(f"all units: {min(rates):.3f}-{max(rates):.3f} s per million units; the slowest stretch of at least "
          f"{MIN_WORK:.0e} units: {max(worsts):.3f}")


if __name__ == "__main__":
    main()
