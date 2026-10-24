"""Two evaluator records' training, side by side, up to the round (LEARNED_LOOP.md §5): the schedule's events (reader
stages, the switch, the cells, the climb's start, the round), every climb line with its timings taken out, and the
logged losses; prints where they first differ. Standard library only.

    cd cpu_first && python3 integration/learned_loop/diag/compare_climb.py REF.json NEW.json
"""
import json
import re
import sys


def parts(path):
    lines = json.load(open(path))["stdout"].splitlines()
    ev = [re.sub(r" \([\d.]+s\)$", "", l) for l in lines
          if re.match(r"LR (io_stage|switch_to_facts|cells_join|climb_joins|order_digits)", l)]
    climb, losses = [], []
    for l in lines:
        if l.startswith("LR climb step="):
            l = re.sub(r" \([\d.]+s\)$", "", l)                     # the unit's time
            l = re.sub(r"\b\d+\.\d+s\b", "Ts", l)                    # times inside the line
            l = re.sub(r", [\d.]+\)", ", T)", l)                      # (how, seconds) pairs
            climb.append(l)
        m = re.match(r"LR step=(\d+) phase=(\w) io=\d loss=([\d.-]+)", l)
        if m:
            losses.append((int(m.group(1)), m.group(2), m.group(3)))
    rnd = next((l for l in climb if re.match(r"LR climb step=\d+ L\d+ round(?: again)?.*: cost (?!None)", l)), None)
    return ev, climb, losses, rnd


def first_diff(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None), n


def main(ref, new):
    (ea, ca, la, ra), (eb, cb, lb, rb) = parts(ref), parts(new)
    for name, a, b in (("schedule events", ea, eb), ("logged losses (every 50 steps)", la, lb), ("climb lines", ca, cb)):
        i, n = first_diff(a, b)
        print(f"{name}: {len(a)} and {len(b)}; " + ("identical over the first %d" % n if i is None else
              f"first differ at #{i}:\n    {str(a[i])[:180]}\n    {str(b[i])[:180]}"))
    print("round, reference:", (ra or "none")[:200])
    print("round, new:      ", (rb or "none")[:200])


if __name__ == "__main__":
    main(*sys.argv[1:3])
