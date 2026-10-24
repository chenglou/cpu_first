"""The tables of SP_BREADTH.md from this study's evaluator records (records/<prefix>_<task>_<hw>.json): per task, the
evaluator's verdict (T on both ladders, every rung, the splits), training time, the climb (levels, when the round came)
and the round's search (the round found, its constant, or the last search's stop).

    ~/github/sotaku/venv/bin/python integration/sp_breadth/summarize.py u0 [g1 ...] [--lines]
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def after(text, step):
    """Seconds into training at the first progress line at or after this step (the evaluator prints every 100)."""
    for m in re.finditer(r"^step=(\d+) .*? elapsed=([\d.]+)s", text, re.M):
        if int(m.group(1)) >= step:
            return float(m.group(2))
    return None


def row(rec):
    r = rec.get("result") or {}
    dp, s = r.get("depth_profile") or {}, (r.get("seeds") or [{}])[0]
    ev = s.get("evaluation") or {}
    out = rec.get("stdout", "")
    rungs = [p.get("correct_examples") for p in (dp.get("profile") or [])]
    found = re.search(r"^LR round_found step=(\d+) programs: (\[.*?\])(?: constant: (\S+))?", out, re.M)
    frozen = re.search(r"^LR frozen step=(\d+)", out, re.M)
    levels = [int(m) for m in re.findall(r"^LR climb step=\d+ L(\d+) ", out, re.M)]
    rounds = [l for l in re.findall(r"^LR climb step=\d+ L\d+ round(?: again)?[^\n]*", out, re.M)
              if "deadline" not in l]
    done = re.search(r"^LR climb step=(\d+) (L\d+ sleep: nothing learnable[^\n]*|climb done[^\n]*)", out, re.M)
    T = (dp.get("max_certified_time_steps"), dp.get("ood_n_max_certified_time_steps"))
    return dict(name=rec["name"], task=rec.get("task"), T=T, certified=T == (64, 64),
                splits={k: f"{v.get('correct_examples')}/{v.get('example_count')}" for k, v in sorted(ev.items())},
                train=s.get("training_seconds"), eval=s.get("evaluation_seconds"), steps=s.get("completed_training_steps"),
                md5=rec.get("submission_md5", "")[:8], levels=max(levels) if levels else None,
                round_step=int(found.group(1)) if found else None, round_s=after(out, int(found.group(1))) if found else None,
                programs=found.group(2) if found else None, const=found.group(3) if found and found.group(3) not in
                (None, "None") else None, frozen=int(frozen.group(1)) if frozen else None,
                frozen_s=after(out, int(frozen.group(1))) if frozen else None, rungs=rungs,
                last_round=rounds[-1][:700] if rounds else None,
                done=f"done by {after(out, int(done.group(1))) or 0:.0f} s" if done else None)


def main():
    lines = "--lines" in sys.argv
    for prefix in [a for a in sys.argv[1:] if not a.startswith("--")]:
        print(f"## {prefix}")
        print("| task | max T seen / unseen | splits | training s (frozen at step, by s) | climb | the round (step, by s) |")
        print("|---|---|---|---|---|---|")
        for path in sorted(HERE.glob(f"records/{prefix}_*.json")):
            x = row(json.loads(path.read_text()))
            sp = ", ".join(f"{k} {v}" for k, v in x["splits"].items())
            fr = f" ({x['frozen']}, {x['frozen_s']:.0f})" if x["frozen"] and x["frozen_s"] else ""
            rd = (f"{x['programs']}{' + constant ' + x['const'] if x['const'] else ''} ({x['round_step']}, "
                  f"{x['round_s']:.0f})" if x["programs"] and x["round_s"] else (x["programs"] or "none"))
            print(f"| {x['task']} | {x['T'][0]} / {x['T'][1]} | {sp} | {x['train'] or 0:.0f}{fr} | "
                  f"L{x['levels']}{'; ' + x['done'] if x['done'] else ''} | {rd} |")
            if lines and x["last_round"]:
                print(f"    last round search: {x['last_round']}")


if __name__ == "__main__":
    main()
