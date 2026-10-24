"""Tables for SELF_PRACTICE.md from the study's records (records/<tag>.json, written by core.py; the in-run file's logs
in records/runner/): per level the pool, the probes' outcomes, the picks and their programs, the time; the round.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/self_practice/summarize.py TAG [TAG ...] [--picks]
    cd cpu_first && ~/github/sotaku/venv/bin/python integration/self_practice/summarize.py --runs   (the evaluator's runs)
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def level_times(units):
    """Seconds per level, from the units' log lines (the level a unit belongs to is in its line, Lk)."""
    t, last = {}, 0.0
    for u in units:
        m = re.search(r" L(\d+) ", " " + u["line"])
        if m:
            k = int(m.group(1))
            t[k] = t.get(k, 0.0) + u["s"]
        last = u["t"]
    return t, last


# stage 3's designed tasks (INTEGRATION.md §2) and the law-book candidates that pose the same problem (any binding)
PIECES = {"lt": "lt/pair", "modx": "red/pair", "add": "add/pair", "modB": "red/x+dn", "mrow": "mul/x,d",
          "mac": "acc.mul/o,x,d", "mul": "mul/x,o", "step": "red/Bx+d", "reduce": "red/x+on"}


def pieces(r):
    """For each of stage 3's tasks: the level its analog was first posed, first learnable, first practised (with
    the binding and program)."""
    print("| stage 3's task | the law-book candidate | first posed | first learnable | practised (level, binding: program) |")
    print("|---|---|---|---|---|")
    for task, pre in PIECES.items():
        posed = learn = None
        prac = []
        for h in r["history"]:
            names = [n for n in list(h["learnable"]) + h.get("pool_names", []) if n.startswith(pre + "@")]
            if learn is None and any(n.startswith(pre + "@") for n in h["learnable"]):
                learn = h["level"]
            for n in h["picks"]:
                if n.startswith(pre + "@"):
                    prac.append(f"L{h['level']} {n.split('@')[1]}: `{h['learnable'].get(n, ('', '?'))[1]}`")
        for x in r["records"]:
            if posed is None and any(k.startswith(pre + "@") for k in x.get("laws", {})):
                posed = x["level"]
        print(f"| {task} | {pre} | {posed if posed is not None else '-'} | {learn if learn is not None else '-'} | "
              f"{'; '.join(prac) or 'never'} |")


def times(rec):
    """Seconds into training: the climb started after the last progress line before its first unit; the round was
    found and the model froze by the first progress line after each (progress lines come every 100 steps)."""
    out, pending, last = {}, [], 0.0
    for line in rec.get("stdout", "").splitlines():
        if line.startswith("LR climb step=") and "climb" not in out:
            out["climb"] = last
        for key, pat in (("round", "LR round_found"), ("frozen", "LR frozen")):
            if line.startswith(pat) and key not in out and key not in pending:
                pending.append(key)
        m = re.search(r"elapsed=([\d.]+)s", line)
        if m and line.startswith("step="):
            last = float(m.group(1))
            for key in pending:
                out[key] = last
            pending = []
    return out


def runs_table():
    """The evaluator's records (records/runner/*.json, the shared runner's), in INTEGRATION.md's S3_RUNS form."""
    sys.path.insert(0, str(HERE.parent.parent))
    from integration import summarize as S
    print("| record | task | device | seed | max T, seen / unseen moduli | rungs | splits | climb from / round by / "
          "frozen by (s) | training s (frozen at step) | evaluation s | what it chose | md5 |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted((HERE / "records" / "runner").glob("*.json")):
        rec = S.load(f)
        if rec is None:
            continue
        if rec.get("returncode"):
            err = (rec.get("stderr") or "").strip().splitlines()[-1:] or ["?"]
            print(f"| {f.stem} | {rec.get('task')} | {S.device_of(rec)} | {S.seed_in(rec)} | crashed ({err[0][:90]}) "
                  f"| | | | | | | {S.md5_of(rec)} |")
            continue
        sd, t = S.seed_of(rec), times(rec)
        when = " / ".join(f"{t[k]:,.0f}" if k in t else "-" for k in ("climb", "round", "frozen"))
        print(f"| {f.stem} | {rec.get('task')} | {S.device_of(rec)} | {S.seed_in(rec)} | {S.maxT(rec)} | "
              f"{S.rungs(rec)} | {S.splits(rec)} | {when} | {sd.get('training_seconds', 0):.0f} "
              f"({S.frozen_step(rec)}) | {sd.get('evaluation_seconds', 0):.0f} | {S.chosen(rec)} | {S.md5_of(rec)} |")


def main():
    if "--runs" in sys.argv:
        return runs_table()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    picks = "--picks" in sys.argv
    for tag in args:
        r = json.loads((HERE / "records" / f"{tag}.json").read_text())
        t, total = level_times(r["units"])
        print(f"== {tag}: {r['config'].get('rule')} target {r['config'].get('target')} sources "
              f"{r['config'].get('sources')}; {total:.0f} s; final={r['final']}; rounds {r['rounds'][:3]}")
        print("| level | pool | trivial | learnable | duplicate | alias | unsolved | picks (cost, program) | seconds |")
        print("|---|---|---|---|---|---|---|---|---|")
        for h in r["history"]:
            c = h["counts"]
            pk = "; ".join(f"{n} ({h['learnable'][n][0]}: `{h['learnable'][n][1]}`)" if n in h["learnable"] else n
                           for n in h["picks"])
            if not picks:
                pk = "; ".join(h["picks"])
            print(f"| {h['level']} | {h['pool']} | {c.get('trivial', 0)} | {c.get('learnable', 0)} | "
                  f"{c.get('duplicate', 0)} | {c.get('alias', 0)} | {c.get('unsolved', 0)} | {pk} | "
                  f"{t.get(h['level'], 0):.0f} |")
        if "--pieces" in sys.argv:
            pieces(r)
        rounds = [x for x in r["records"] if x.get("task") in ("round", "target")]
        if rounds:
            x = rounds[-1]
            print(f"last {x['task']} search: level {x['level']} cost {x['found_cost']} {x.get('programs', [])[:3]} "
                  f"{x.get('stop')}")


if __name__ == "__main__":
    main()
