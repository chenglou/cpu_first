"""Reads this study's records (the runner's, records/) and says, per run: certified or not, when the climb started, when
the round was found and the model froze (the evaluator's elapsed time at the first logged step after each, as
INTEGRATION.md reports s3i; the climb's start: the last logged step before it), the climb's levels (posed, learnable,
picks, seconds), which of the round's pieces it practised and when, how the climb ended, and whether that end was its
own (every probe ran to its state cap: under the 900 s search limit and before the training deadline) or the clock's.
--both prints SP_ABLATION.md's results table (every arm on seeds 74 and 1, and the verdict), --results one row per
run with its verdict against its control, --table a longer one, --levels every level's units, --json the facts.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/sp_ablation/analyze.py [--table|--levels|--json] \
        [RECORD ...]          (default: every record in integration/sp_ablation/records/)
"""

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = re.compile(r"^(?P<arm>.+?)_(?P<hw>h100|h200|b200|cpu)(?:_r(?P<budget>\d+))?_seed(?P<seed>\d+)$")
STEP = re.compile(r"^step=(\d+) .*elapsed=([\d.]+)s")
UNIT = re.compile(r"^LR climb step=(\d+) L(\d+) (.*) \(([\d.]+)s\)$")
SEARCH_S = 900                  # sp2's one-search limit (SEARCH_S)
# The round's pieces in sp2's certified runs (SELF_PRACTICE.md §3.4), as concepts (a law on a pattern), with stage 3's
# names: the comparison, the link (conditional subtraction), the addition (it builds every composite pattern), the
# chain, the multiply-accumulate row, the multiply, the reduction. Stage 3's other two (the step, the plain row) are
# practised too but not called by the round.
PIECES = [("lt", "pair", "lt"), ("red", "pair", "link"), ("add", "pair", "add"), ("red", "x+dn", "chain"),
          ("acc.mul", "o,x,d", "mac"), ("mul", "x,o", "mul"), ("red", "x+on", "reduce"), ("red", "Bx+d", "step"),
          ("mul", "x,d", "row")]
ROUND_PIECES = ["lt", "link", "add", "chain", "mac", "mul", "reduce"]
ALIAS = {"acc.mul": ["add.mul"], "acc.add": ["add.add"]}      # sp2's ternary laws as the generated book names them
ROUTINE = re.compile(r"T\.[^\s\[\]]+")


def round_uses(prog, lib, practised):
    """The routines the round program calls, directly or through other routines and merges, each with the level its
    concept (a law on a pattern) was first practised: [(routine, level)] in the order first met."""
    out, seen, todo = [], set(), [prog]
    while todo:
        text = todo.pop(0)
        for tok in re.findall(r"[^\s\[\]]+", text):
            if tok in seen or tok not in lib:
                continue
            seen.add(tok)
            todo.append(lib[tok])
            if ROUTINE.fullmatch(tok):
                parts = tok[2:].rsplit(".", 2)                  # T.<law>.<pattern>.<binding>
                level = practised.get((parts[0], parts[1]), (None,))[0] if len(parts) == 3 else None
                out.append((tok, level))
    return out


def untimed(body):
    """A climb unit's line without its seconds (the unit's own, "12.3s", and each search's, "('routines', 12.3)")."""
    return re.sub(r"\d+\.\d+s|(?<=['\"], )\d+\.\d+(?=\))", "", body)


def fmt_s(x):
    return "-" if x is None else f"{x:,.0f}"


def literal(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


def facts(path):
    rec = json.loads(Path(path).read_text())
    m = NAME.match(rec["name"])
    arm = m["arm"] if m else rec["name"]
    lines = (rec.get("stdout") or "").splitlines()
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
             steps=seed.get("completed_training_steps"), wall_s=rec.get("wall_seconds"))
    f["certified"] = f["T"] == (64, 64)
    ev = seed.get("evaluation") or {}
    f["splits"] = {k: (v.get("correct_examples"), v.get("example_count")) for k, v in ev.items()}
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
    f["climb_step"] = first(r"^LR climb_starts step=(\d+)")
    f["round_step"] = first(r"^LR round_found step=(\d+)")
    f["frozen_step"] = first(r"^LR frozen step=(\d+)")
    f["T_step"] = first(r"T_resolved step=(\d+)")
    for k in ("round", "frozen"):
        f[f"{k}_s"] = at(f[f"{k}_step"])
    f["climb_s"] = next((e for s, e in reversed(steps) if s <= f["climb_step"]), None) if f["climb_step"] else None
    f["budget_s"] = float(f["budget"] or 3600)
    rp = next((l for l in lines if l.startswith("LR round_found")), "")
    mm = re.search(r"programs: (\[.*?\]) library: (\{.*\})$", rp)
    f["rounds"] = literal(mm.group(1)) if mm else []
    f["round_lib"] = (literal(mm.group(2)) or {}) if mm else {}
    # the climb, unit by unit
    units, levels = [], defaultdict(lambda: dict(seconds=0.0, posed=None, roots=None, counts=None, picks=[],
                                                 probes=[], deep=[], round=None, sleep=None))
    practised = {}                                   # concept (law, pattern) -> (level, [bindings])
    ended = None
    for l in lines:
        cap = re.match(r"^LR climb step=(\d+) climb done at level (\d+)", l)      # the level cap (no level prefix)
        if cap:
            ended = ended or ("the level cap", int(cap[1]), int(cap[2]))
            continue
        u = UNIT.match(l)
        if not u:
            continue
        step, lv, body, sec = int(u[1]), int(u[2]), u[3], float(u[4])
        units.append(dict(step=step, level=lv, body=body, seconds=sec))
        L = levels[lv]
        L["seconds"] += sec
        if body.startswith("pose:"):
            mm = re.match(r"pose: (\d+) candidates on (\d+) roots", body)
            L["posed"], L["roots"] = int(mm[1]), int(mm[2])
        elif body.startswith("probe "):
            mm = re.match(r"probe (\d+) roots ([\d.]+)s solved (\{.*?\}) (\[.*\])", body)
            L["probes"].append(dict(roots=int(mm[1]), seconds=float(mm[2]), solved=literal(mm[3]) or {},
                                    runs=literal(mm[4]) or []))
        elif body.startswith("deep ["):
            mm = re.match(r"deep (\[.*?\]) ([\d.]+)s solved (\{.*?\}) (\[.*\])", body)
            L["deep"].append(dict(roots=literal(mm[1]), seconds=float(mm[2]), solved=literal(mm[3]) or {},
                                  runs=literal(mm[4]) or []))
        elif body.startswith("choose ("):
            mm = re.match(r"choose \((\w+)\): (\{.*?\}); picks (\[.*\])$", body)
            L["counts"] = literal(mm[2])
            L["picks"] = literal(mm[3]) or []
            for name, cost, prog in L["picks"]:
                law, rk = name.split("/")
                pat, bd = rk.split("@")
                if (law, pat) not in practised:
                    practised[(law, pat)] = (lv, [])
                practised[(law, pat)][1].append((bd, cost, prog))
        elif body.startswith("sleep:"):
            L["sleep"] = body[len("sleep: "):]
            if "climb done" in body:
                ended = ended or ("nothing learnable", step, lv)
        elif body.startswith("round:"):
            L["round"] = body
        elif body.startswith("climb done"):
            ended = ended or ("level cap", step, lv)
    f["levels"] = {k: dict(v) for k, v in sorted(levels.items())}
    f["units"] = units
    f["practised"] = practised
    f["pieces"] = {name: next((practised[(l, pat)][0] for l in [law] + ALIAS.get(law, []) if (l, pat) in practised),
                              None) for law, pat, name in PIECES}
    f["round_uses"] = round_uses(f["rounds"][0] if f["rounds"] else "", f["round_lib"], practised)
    f["missing"] = [p for p in ROUND_PIECES if f["pieces"][p] is None]
    f["last_level"] = max(levels) if levels else None
    f["retries"] = sum(1 for u in units if u["body"].startswith("round again"))
    # was the climb cut by the clock? the training deadline (a "deadline" stop, or searches that return at once
    # after it), or one search's limit (a probe of SEARCH_S seconds or more)
    f["deadline_units"] = [u for u in units if "deadline" in u["body"]]
    f["slow_searches"] = [u for u in units if u["body"].startswith(("probe ", "deep [")) and u["seconds"] >= SEARCH_S]
    f["ended"] = ended
    f["ended_s"] = at(ended[1]) if ended else None
    f["climb_lines"] = [untimed(u["body"]) for u in units]
    if f["certified"]:
        f["where"] = "certified"
    elif f["error"]:
        f["where"] = f"crashed: {f['error']}"
    elif f["round_step"] is not None:
        f["where"] = "round found, not certified"
    elif f["climb_step"] is None:
        f["where"] = "the climb never started"
    elif ended:
        cut = f["deadline_units"] and f["deadline_units"][0]["step"] <= ended[1]
        f["where"] = (f"climb ended in L{ended[2]} ({ended[0]}, step {ended[1]}, {fmt_s(f['ended_s'])} s)"
                      f"{': cut by the training deadline' if cut else ''}; {f['retries']} round retries")
    elif f["deadline_units"]:
        f["where"] = f"out of time in L{f['last_level']}"
    else:
        f["where"] = f"stopped in L{f['last_level']}"
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
    of the probe units identical to the control's (same level and the same line once its seconds are removed: the same
    roots, the same results) over the control's, and how many there were; None under 30 s of identical units. The climb
    is host-bound (INTEGRATION.md §3), so single-run times differ by host; dividing the climb's time by this factor
    compares the work."""
    c = control_of(f)
    if c is None:
        return None
    key = lambda u: (u["level"], untimed(u["body"]))
    mine = defaultdict(list)
    for u in f["units"]:
        if u["body"].startswith(("probe ", "deep [")):
            mine[key(u)].append(u["seconds"])
    pairs = []
    for u in c["units"]:
        if u["body"].startswith(("probe ", "deep [")) and mine.get(key(u)):
            pairs.append((mine[key(u)].pop(0), u["seconds"]))
    a, b = sum(x for x, _ in pairs), sum(y for _, y in pairs)
    return (a / b, len(pairs)) if b >= 30 else None


def adjust(f):
    c = control_of(f)
    f["same_climb"] = None if c is None else f["climb_lines"] == c["climb_lines"]
    f["host"] = host_factor(f)
    f["round_adj_s"] = None
    if f["host"] and f["round_s"] is not None and f["climb_s"] is not None:
        f["round_adj_s"] = f["climb_s"] + (f["round_s"] - f["climb_s"]) / f["host"][0]
    return f


def pieces_str(f):
    return " ".join(f"{p}@L{f['pieces'][p]}" if f["pieces"][p] is not None else f"{p}:-" for p in ROUND_PIECES)


def verdict(f):
    """not needed / speed only / needed, against the control of the same seed, hardware and budget: a certified run
    whose round came within 25% of the control's (host-corrected when at least 10 search units equal the control's,
    else raw) is not needed; a later one is speed only; a run that never certified is needed."""
    c = control_of(f)
    if f["arm"] == "control":
        return "control"
    if not f["certified"]:
        return "needed"
    if c is None or c["round_s"] is None:
        return "certified"
    mine = f["round_adj_s"] if f["host"] and f["host"][1] >= 10 else f["round_s"]
    d = mine - c["round_s"]
    return "not needed" if d <= 0.25 * c["round_s"] else f"speed only: +{d:,.0f} s"


def ends(f):
    """How the climb ended, in a few words: the round, or the level and why, and whether the clock cut it."""
    if f["round_step"] is not None:
        return f"round at L{max(f['levels'])}"
    if f["deadline_units"] and (not f["ended"] or f["deadline_units"][0]["step"] <= f["ended"][1]):
        return f"cut by the training deadline in L{f['deadline_units'][0]['level']} (of L0-L{f['last_level']})"
    if not f["ended"]:
        return f"no end logged (last L{f['last_level']})"
    kind, step, lv = f["ended"]
    return f"ended in L{lv}: {kind} (step {step}, {fmt_s(f['ended_s'])} s)"


def results(rows):
    print("| arm | certified | round: step (level), s (host-corrected) | frozen by, s | the climb | missing of the "
          "round's pieces | verdict |")
    print("|---|---|---|---|---|---|---|")
    for f in rows:
        r = "-" if f["round_step"] is None else (f"{f['round_step']} (L{max(f['levels'])}), {fmt_s(f['round_s'])}" +
                                                (f" ({fmt_s(f['round_adj_s'])})" if f["host"] and f["host"][1] >= 10
                                                 else ""))
        print(f"| {f['arm']} | {'yes' if f['certified'] else 'no'} | {r} | {fmt_s(f['frozen_s'])} | {ends(f)} | "
              f"{', '.join(f['missing']) or '-'} | {verdict(f)} |")


def cell(f):
    """One run in a few words: certified with its round's step, level and seconds (host-corrected in brackets), or
    not, with how its climb ended."""
    if f is None:
        return "-"
    if f["certified"]:
        adj = f" ({fmt_s(f['round_adj_s'])})" if f["host"] and f["host"][1] >= 10 else ""
        return f"yes: {f['round_step']} (L{max(f['levels'])}), {fmt_s(f['round_s'])}{adj} s"
    return f"no: {ends(f)}"


def both(rows, arms):
    """SP_ABLATION.md's results table: each arm on seed 74 and seed 1 (H100, the official manifest), and the verdict
    of the two together."""
    by = {(f["arm"], f["seed"]): f for f in rows if f["hardware"] == "h100" and f["budget"] in (None, 3600)}
    print("| arm | seed 74 (certified: round step (level), s (host-corrected)) | seed 1 | verdict |")
    print("|---|---|---|---|")
    for arm in [a for a in arms if not a.startswith("book_")]:      # (the books: SP_ABLATION.md §5)
        a, b = by.get((arm, 74)), by.get((arm, 1))
        if a is None and b is None:
            continue
        vs = [verdict(f) for f in (a, b) if f is not None]
        if all(v == vs[0] for v in vs):
            v = vs[0]
        elif all(v == "not needed" or v.startswith("speed") for v in vs):
            v = "speed only on one seed (" + "; ".join(vs) + ")"
        else:
            v = "differs by seed (" + "; ".join(vs) + ")"
        print(f"| {arm} | {cell(a)} | {cell(b)} | {v} |")


def table(rows):
    print("| run | certified (T seen / unseen) | climb from | round at | frozen at | levels | round's pieces "
          "(level practised) | where it stopped | md5 |")
    print("|---|---|---|---|---|---|---|---|---|")
    def when(f, k):
        return "-" if f[f"{k}_step"] is None else f"step {f[f'{k}_step']} ({fmt_s(f[f'{k}_s'])} s)"
    for f in rows:
        level = "-" if f["last_level"] is None else f"L0-L{f['last_level']}"
        print(f"| {f['name']} | {'yes' if f['certified'] else 'no'} ({f['T'][0]} / {f['T'][1]}) | {when(f, 'climb')} | "
              f"{when(f, 'round')} | {when(f, 'frozen')} | {level} | {pieces_str(f)} | {f['where']} | {f['md5']} |")


def levels(rows):
    for f in rows:
        print(f"{f['name']}: {f['where']}")
        for lv, L in f["levels"].items():
            probe_s = sum(p["seconds"] for p in L["probes"])
            deep_s = sum(p["seconds"] for p in L["deep"])
            print(f"  L{lv} {L['seconds']:7.1f}s posed {L['posed']} on {L['roots']} roots; probes {probe_s:.0f}s "
                  f"(max {max([p['seconds'] for p in L['probes']] or [0]):.0f}), deep {deep_s:.0f}s; {L['counts']}")
            for name, cost, prog in L["picks"]:
                print(f"       pick {name:22s} cost {cost}  {prog}")
            if L["round"]:
                print(f"       {L['round'][:160]}")
        for u in f["deadline_units"][:3]:
            print(f"  deadline: step {u['step']} L{u['level']} {u['body'][:120]}")
        for u in f["slow_searches"]:
            print(f"  search at the {SEARCH_S} s limit: step {u['step']} L{u['level']} {u['seconds']:.0f}s")


def main(argv):
    mode = next((a for a in argv if a.startswith("--")), "--summary")
    paths = [a for a in argv if not a.startswith("--")] or sorted(str(p) for p in (HERE / "records").glob("*.json"))
    rows = [adjust(facts(p)) for p in paths]
    if mode == "--table":
        table(rows)
    elif mode == "--results":
        results(rows)
    elif mode == "--both":
        both(rows, list(json.loads((HERE / "plans" / "arms.json").read_text())))
    elif mode == "--levels":
        levels(rows)
    elif mode == "--json":
        print(json.dumps([{k: v for k, v in f.items() if k not in ("units", "climb_lines", "levels")} for f in rows],
                         indent=1, default=str))
    else:
        for f in rows:
            host = f"host x{f['host'][0]:.2f} ({f['host'][1]} units) round adj {fmt_s(f['round_adj_s'])}s " \
                if f["host"] else ""
            host += "climb = control's " if f["same_climb"] else ""
            print(f"{f['name']:32s} {'CERTIFIED' if f['certified'] else 'no':9s} T={f['T']} climb@{f['climb_step']} "
                  f"({fmt_s(f['climb_s'])}s) round@{f['round_step']} ({fmt_s(f['round_s'])}s) {host}"
                  f"frozen@{f['frozen_step']} ({fmt_s(f['frozen_s'])}s) L0-L{f['last_level']} "
                  f"train {fmt_s(f['train_s'])}s | {pieces_str(f)} | {f['where']}")


if __name__ == "__main__":
    main(sys.argv[1:])
