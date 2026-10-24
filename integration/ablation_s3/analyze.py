"""Reads ablation records (the runner's, records/) and says, per run: certified or not, when the climb started, when the
round was found and the model froze (the evaluator's elapsed time at the first logged step after each, as
INTEGRATION.md reports s3i; the climb's start: the last logged step before it), the climb's last level and time per
level, what it solved and never solved, and where it stopped. --table prints the markdown table ABLATION_S3.md
carries; --levels the time per level and every search; --json the facts.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/analyze.py [--table|--levels|--json] \
        [RECORD ...]          (default: every record in integration/ablation_s3/records/)
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINALS = ["lt", "modx", "add", "modB", "mrow", "mac", "mul", "step", "reduce"]
DROPPED = {"no_redstep": {"modB", "step"}, "no_step": {"step"}, "no_modB": {"modB"}, "no_mrow": {"mrow"}}
NAME = re.compile(r"^(?P<arm>.+?)_(?P<hw>h100|h200|b200|cpu)(?:_r(?P<budget>\d+))?_seed(?P<seed>\d+)$")
STEP = re.compile(r"^step=(\d+) .*elapsed=([\d.]+)s")
CLIMB = re.compile(r"^LR climb step=(\d+) L(\d+) (.*) \(([\d.]+)s\)$")
TASK = re.compile(r"^(\S+)\s+cost (\S+) \((\w+)\) levels (\[[^\]]*\]) ([\d.]+)s ?(.*?) (\[.*\])$")


def tasks_of(arm):
    arm = arm.removesuffix("_wide")              # a wider-search arm poses its failed arm's tasks
    base = [t for t in ORIGINALS if t not in DROPPED.get(arm, set())]
    return base if arm == "no_twins" else [x for t in base for x in (t, t + "~")]


def facts(path):
    rec = json.loads(Path(path).read_text())
    m = NAME.match(rec["name"])
    arm = m["arm"] if m else rec["name"]
    out = rec.get("stdout") or ""
    lines = out.splitlines()
    res = rec.get("result") or {}
    dp = res.get("depth_profile") or {}
    seed = (res.get("seeds") or [{}])[0]
    runtime = (rec.get("manifest") or {}).get("runtime") or {}
    f = dict(name=rec["name"], arm=arm, seed=int(m["seed"]) if m else (runtime.get("seeds") or [None])[0],
             hardware=rec.get("hardware") or ("h100" if "H100" in (rec.get("gpu") or "") else "?"),
             budget=runtime.get("total_training_time_seconds"), md5=rec.get("submission_md5", "")[:8],
             returncode=rec.get("returncode"), killed=(rec.get("runner") or {}).get("killed"),
             T=(dp.get("max_certified_time_steps"), dp.get("ood_n_max_certified_time_steps")),
             train_s=seed.get("training_seconds"), eval_s=seed.get("evaluation_seconds"),
             steps=seed.get("completed_training_steps"))
    f["certified"] = f["T"] == (64, 64)
    f["error"] = None
    if rec.get("returncode") not in (0, None) or not res:
        err = [l for l in (rec.get("stderr") or "").splitlines() if l.strip()]
        f["error"] = f["killed"] or (err[-1][:160] if err else f"returncode {rec.get('returncode')}")
    steps = [(int(a), float(b)) for a, b in (STEP.match(l).groups() for l in lines if STEP.match(l))]

    def at(step):        # the evaluator's elapsed time at the first logged step at or after `step`
        return next((e for s, e in steps if s >= step), None) if step is not None else None

    def first(pat):
        mm = next((re.search(pat, l) for l in lines if re.search(pat, l)), None)
        return int(mm.group(1)) if mm else None
    f["cells_step"] = first(r"^LR cells_join step=(\d+)")
    f["climb_step"] = first(r"^LR climb_joins step=(\d+)")
    f["round_step"] = first(r"^LR round_found step=(\d+)")
    f["frozen_step"] = first(r"^LR frozen step=(\d+)")
    f["T_step"] = first(r"T_resolved step=(\d+)")
    for k in ("round", "frozen"):
        f[f"{k}_s"] = at(f[f"{k}_step"])
    # the climb's start: the last logged step before it (phase B's steps are fast; the next logged step comes after
    # climb units)
    f["climb_s"] = next((e for s, e in reversed(steps) if s <= f["climb_step"]), None) if f["climb_step"] else None
    f["last_elapsed"] = steps[-1][1] if steps else None
    rp = next((l for l in lines if l.startswith("LR round_found")), "")
    mm = re.search(r"programs: (\[.*?\]) library:", rp)
    f["rounds"] = eval(mm.group(1)) if mm else []
    # the climb
    level_s, solved, searches, events, rounds = defaultdict(float), [], [], [], []
    last_level, done = None, None
    for l in lines:
        mm = CLIMB.match(l)
        if not mm:
            if l.startswith("LR climb step=") and "climb done" in l:
                done = l
            continue
        step, level, body, sec = int(mm[1]), int(mm[2]), mm[3], float(mm[4])
        level_s[level] += sec
        last_level = level
        if body.startswith("climb done"):
            done = body
        elif body.startswith("sleep: solved"):
            solved += eval(re.match(r"sleep: solved (\[.*?\])", body).group(1))
        elif body.startswith(("round:", "round again")):
            rounds.append((level, step, body))
        elif body.startswith(("STALL", "TWIN")):
            events.append((level, body.split(":")[0]))
        else:
            t = TASK.match(body)
            if t:
                searches.append(dict(level=level, step=step, task=t[1], cost=t[2], how=t[3], sizes=t[4],
                                     seconds=float(t[5]), stop=t[6].strip(), programs=t[7]))
    # the climb ends when a level solves and merges nothing, or when nothing is left to search; after that only the
    # round is retried (every ROUND_RETRY steps), so the first retry or the end line marks it
    retries = [(lv, step) for lv, step, b in rounds if b.startswith("round again")]
    ends = [(int(m_.group(1)), int(m_.group(2))) for m_ in (re.match(r"^LR climb step=(\d+) L(\d+) sleep: solved \[\];\s*\(", l)
                                                       for l in lines) if m_]
    if ends or retries or done:
        done = done or "climb ended"
        step_end = ends[0][0] if ends else (retries[0][1] if retries else None)
        f["ended_step"], f["ended_level"] = step_end, (ends[0][1] if ends else last_level)
    else:
        f["ended_step"] = f["ended_level"] = None
    f["ended_s"] = at(f["ended_step"])
    f["retries"] = len(retries)
    # the climb's log without its seconds, to tell a climb identical to the control's (same_climb)
    f["climb_lines"] = [re.sub(r" \(?\d+\.\d+s\)?", "", l) for l in lines if l.startswith("LR climb step=")]
    f.update(level_s=dict(level_s), last_level=last_level, solved=solved, done=done, searches=searches,
             events=events, round_searches=rounds)
    expected = tasks_of(arm)
    f["unsolved"] = [t for t in expected if t not in solved]
    f["deadline"] = any("deadline" in s["stop"] for s in searches) or any("deadline" in b for _, _, b in rounds)
    f["unsolved_stops"] = {t: [(s["level"], s["how"], s["stop"] or "exhausted") for s in searches if s["task"] == t]
                           for t in f["unsolved"]}
    if f["certified"]:
        f["where"] = "certified"
    elif f["error"]:
        f["where"] = f"crashed: {f['error']}"
    elif f["round_step"] is not None:
        f["where"] = "round found, not certified"
    elif f["climb_step"] is None:
        f["where"] = ("cells never joined" if f["cells_step"] is None else "cells joined, climb never started")
    elif done:
        f["where"] = (f"climb ended in L{f['ended_level']} (step {f['ended_step']}, {fmt_s(f['ended_s'])} s), no round; "
                      f"{f['retries']} round retries to the deadline")
    elif f["deadline"]:
        f["where"] = f"out of time in L{last_level}"
    else:
        f["where"] = f"stopped in L{last_level}"
    return f


_CONTROL = {}


def control_of(f):
    """The control's facts for this run's seed, hardware and budget, or None (for the control itself, or none run)."""
    suffix = "" if f["budget"] in (None, 3600) else f"_r{f['budget']}"
    ref = HERE / "records" / f"control_{f['hardware']}{suffix}_seed{f['seed']}.json"
    if f["arm"] == "control" or not ref.exists():
        return None
    if ref not in _CONTROL:
        _CONTROL[ref] = facts(ref)
    return _CONTROL[ref]


def host_factor(f):
    """How much slower this run's host searched than the control's (same seed, hardware and budget): the summed seconds
    of the searches identical to the control's (same level, task, vocabulary and level sizes) over the control's, and
    how many there were; None without a control or with under 30 s of identical searches. The climb is host-bound
    (INTEGRATION.md §3), so single-seed times differ by host; dividing the climb's time by this factor compares the
    work. The primitives-alone fallback is left out: it streams 300-400M programs of single instructions, bound by the
    GPU rather than the host, and its time varied from 0.5x to 5.5x the control's between runs whose other identical
    searches agreed within 30% (ABLATION_S3.md §4)."""
    c = control_of(f)
    if c is None:
        return None
    key = lambda s: (s["level"], s["task"], s["how"], s["sizes"])
    mine = {key(s): s["seconds"] for s in f["searches"]}
    pairs = [(mine[key(s)], s["seconds"]) for s in c["searches"] if key(s) in mine and s["how"] != "primitives"]
    a, b = sum(x for x, _ in pairs), sum(y for _, y in pairs)
    return (a / b, len(pairs)) if b >= 30 else None


def adjust(f):
    """The host factor, the round's time with the climb's part divided by it, and whether the climb's log equals the
    control's once its seconds are removed."""
    c = control_of(f)
    f["same_climb"] = None if c is None else f["climb_lines"] == c["climb_lines"]
    f["host"] = host_factor(f)
    f["round_adj_s"] = None
    if f["host"] and f["round_s"] is not None and f["climb_s"] is not None:
        f["round_adj_s"] = f["climb_s"] + (f["round_s"] - f["climb_s"]) / f["host"][0]
    return f


def fmt_s(x):
    return "-" if x is None else f"{x:,.0f}"


def table(rows):
    print("| run | certified (T seen / unseen) | climb from | round at | frozen at | climb's last level | never "
          "solved (originals) | where it stopped | md5 |")
    print("|---|---|---|---|---|---|---|---|---|")
    def when(f, k):
        return "-" if f[f"{k}_step"] is None else f"step {f[f'{k}_step']} ({fmt_s(f[f'{k}_s'])} s)"
    for f in rows:
        un = [t for t in f["unsolved"] if not t.endswith("~")]
        level = "-" if f["last_level"] is None else f"L{f['last_level']}"
        print(f"| {f['name']} | {'yes' if f['certified'] else 'no'} ({f['T'][0]} / {f['T'][1]}) | {when(f, 'climb')} | "
              f"{when(f, 'round')} | {when(f, 'frozen')} | {level} | {', '.join(un) or 'none'} | {f['where']} | "
              f"{f['md5']} |")


def levels(rows):
    for f in rows:
        print(f"{f['name']}: " + ", ".join(f"L{k} {v:,.0f}s" for k, v in sorted(f["level_s"].items())))
        for s in f["searches"]:
            print(f"   L{s['level']} {s['task']:8s} cost {s['cost']:4s} {s['how']:10s} {s['seconds']:7.1f}s "
                  f"{s['stop'][:40]:40s} {s['programs'][:90]}")
        for lv, step, body in f["round_searches"]:
            print(f"   L{lv} {body[:150]}")
        for lv, e in f["events"]:
            print(f"   L{lv} {e}")


def main(argv):
    mode = next((a for a in argv if a.startswith("--")), "--summary")
    paths = [a for a in argv if not a.startswith("--")] or sorted(str(p) for p in (HERE / "records").glob("*.json"))
    rows = [adjust(facts(p)) for p in paths]
    if mode == "--table":
        table(rows)
    elif mode == "--levels":
        levels(rows)
    elif mode == "--json":
        print(json.dumps([{k: v for k, v in f.items() if k not in ("searches", "climb_lines")} for f in rows], indent=1,
                         default=str))
    else:
        for f in rows:
            un = [t for t in f["unsolved"] if not t.endswith("~")]
            host = f"host x{f['host'][0]:.2f} ({f['host'][1]} searches) round adj {fmt_s(f['round_adj_s'])}s " \
                if f["host"] else ""
            host += "climb = control's " if f["same_climb"] else ""
            print(f"{f['name']:34s} {'CERTIFIED' if f['certified'] else 'no':9s} T={f['T']} climb@{f['climb_step']} "
                  f"({fmt_s(f['climb_s'])}s) round@{f['round_step']} ({fmt_s(f['round_s'])}s) {host}"
                  f"frozen@{f['frozen_step']} ({fmt_s(f['frozen_s'])}s) last L{f['last_level']} "
                  f"train {fmt_s(f['train_s'])}s unsolved {un} | {f['where']}")


if __name__ == "__main__":
    main(sys.argv[1:])
