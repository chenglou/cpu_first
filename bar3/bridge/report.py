"""Tables for BRIDGE.md from the learner records (bar3/bridge/records/*.jsonl).

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.report --runs           # per run
    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.report --detail lp_sbb_r1  # rounds
    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.report --book book_r1      # untargeted

Search-hours count every job's seconds (a job answered from the shared cache counts what it cost when first run).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    p = HERE / "records" / (name if name.endswith(".jsonl") else name + ".jsonl")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def hours(rec):
    return sum(t["seconds"] for t in rec["totals"].values()) / 3600


def states(rec):
    return sum(t["states"] for t in rec["totals"].values())


def runs(names):
    print("| run | target | rounds | invented | program (as found) | primitive program | picks the program uses | "
          "states | search-hours |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n in names:
        recs = load(n)
        if not recs:
            print(f"| {n} | - | 0 | - | - | - | - | - | - |")
            continue
        last = recs[-1]
        g = last.get("general")
        used = []
        if g:
            for r in recs:
                for e in r["promoted"]:
                    if e["name"] in g["program"] or e["name"] in json.dumps(g):
                        used.append(e["name"])
        print(f"| {n} | {last['target']} | {len(recs)} | {'round ' + str(last['round']) if g else 'no'} | "
              f"{'`' + g['program'] + '`' if g else '-'} | {'`' + g['primitive'] + '`' if g else '-'} | "
              f"{', '.join(sorted(set(used))) or '-'} | {states(last) / 1e6:,.1f}M | {hours(last):.2f} |")


def detail(name):
    for r in load(name):
        st = r["status"]
        print(f"round {r['round']}: pool {len(r['pool'])}; " + ", ".join(f"{k} {v}" for k, v in
                                                                         Counter(x['status'] for x in st.values()).items()))
        print(f"  picked: {r['picked']}")
        for e in r["merges"] + r["promoted"]:
            print(f"    + {e['name']} = {e['def']}   ({e.get('expands_to', '')})")
        tr = r.get("target_rec") or {}
        print(f"  target: cost {tr.get('found_cost')} levels {tr.get('level_sizes')} best {tr.get('best_fit')} "
              f"of {tr.get('NP')}; programs {[p['program'] for p in tr.get('programs', [])][:3]}")
        if r.get("general"):
            print(f"  INVENTED: {r['general']['program']} = {r['general']['primitive']}")
        learn = sorted([(v["cost"], k, v["best"]) for k, v in st.items() if v["status"] == "learnable"],
                       reverse=True)[:12]
        for c, k, b in learn:
            print(f"     learnable cost {c}: {k}: {b}")
        print(f"  search-hours so far {hours(r):.2f}, states {states(r) / 1e6:,.1f}M")


def book(name):
    recs = load(name)
    for r in recs:
        st = r["status"]
        print(f"round {r['round']}: open {len(r['open'])}; pool {len(r['pool'])}; "
              + ", ".join(f"{k} {v}" for k, v in Counter(x['status'] for x in st.values()).items()))
    inv = recs[-1]["invented"]
    print("| slice | invented in round | cost | program | primitive |")
    print("|---|---|---|---|---|")
    for s, v in sorted(inv.items(), key=lambda kv: (kv[1]["round"], kv[0])):
        print(f"| {s} | {v['round']} | {v['cost']} | `{v['program']}` | `{v['primitive']}` |")
    print("not invented:", [s for s in recs[-1]["open"] if s not in inv])
    print(f"search-hours {hours(recs[-1]):.2f}, states {states(recs[-1]) / 1e6:,.1f}M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="store_true")
    ap.add_argument("--detail", default="")
    ap.add_argument("--book", default="")
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--climbs", default="")
    ap.add_argument("--certified", default="")
    a = ap.parse_args()
    if a.runs:
        runs([f"{r}_{t}_{a.tag}" for t in ("sbb", "adc", "mul", "sel") for r in ("lp", "ct2", "all")])
    if a.detail:
        detail(a.detail)
    if a.book:
        book(a.book)
    if a.climbs:
        climbs(a.climbs.split(","))
    if a.certified:
        certified(a.certified.split(","))



def climbs(names):
    """The climbs of climb_b.py (records/climb/<name>.json): the instruction set's size, when each piece came, the
    round."""
    print("| climb | instructions | modx (conditional subtraction) | mac | reduce | mul | round | round after s |")
    print("|---|---|---|---|---|---|---|---|")
    for n in names:
        r = json.loads((HERE / "records" / "climb" / f"{n}.json").read_text())
        lib = r["lib"]
        at = r["solved_at"]
        cell = lambda t: f"L{at[t]} `{lib.get('T.' + t, '?')}`" if t in at else "never"   # noqa: E731
        print(f"| {n} | {len(r['instructions'])} | {cell('modx')} | {cell('mac')} | {cell('reduce')} | {cell('mul')} | "
              f"{'`' + r['rounds'][-1] + '`' if r['rounds'] else 'none'} | {r['seconds'] if r['rounds'] else '-'} |")


def _after(stdout, pattern):
    """(the step a log line names, the evaluator's elapsed seconds at the first progress line after it)."""
    import re
    m = re.search(pattern, stdout, re.M)
    if not m:
        return None, None
    e = re.search(r"^step=\d+ .*?elapsed=([0-9.]+)s", stdout[m.end():], re.M)
    return int(m.group(1)), (float(e.group(1)) if e else None)


def certified(names):
    """The evaluator's records (records/runner/<name>.json): T on both ladders, rungs, splits, when the round came and
    when the model froze (the elapsed seconds at the evaluator's first progress line after each), training and
    evaluation seconds."""
    print("| record | hardware | T seen / unseen | rungs at 768/768 | test, ood_t, ood_n_t | round: step, s | frozen: step, s | "
          "training s / evaluation s | md5 |")
    print("|---|---|---|---|---|---|---|---|---|")
    import hashlib
    for n in names:
        p = HERE / "records" / "runner" / f"{n}.json"
        if not p.exists():
            print(f"| {n} | - | not in | | | | | | |")
            continue
        r = json.loads(p.read_text())
        res = r.get("result") or {}
        dp, sd = res.get("depth_profile") or {}, (res.get("seeds") or [{}])[0]
        rungs = (sd.get("depth_profile") or {}).get("rungs", []) + (sd.get("depth_profile") or {}).get("ood_n_rungs", [])
        full = sum(1 for x in rungs if x.get("correct_examples") == x.get("example_count") == 768)
        ev = sd.get("evaluation") or {}
        splits = ", ".join(f"{ev[k]['correct_examples']:,}/{ev[k]['example_count']:,}" for k in ("test", "ood_t", "ood_n_t")
                           if k in ev)
        rs, rt = _after(r.get("stdout", ""), r"^LR round_found step=(\d+)")
        fs, ft = _after(r.get("stdout", ""), r"^LR frozen step=(\d+)")
        print(f"| {n} | {r.get('hardware')} | {dp.get('max_certified_time_steps')} / {dp.get('ood_n_max_certified_time_steps')} | "
              f"{full} / {len(rungs)} | {splits or '-'} | {rs}, {rt} | {fs}, {ft} | "
              f"{sd.get('training_seconds') or 0:,.1f} / {sd.get('evaluation_seconds') or 0:,.1f} | "
              f"{hashlib.md5(r['submission'].encode()).hexdigest()[:8]} |")


if __name__ == "__main__":
    main()
