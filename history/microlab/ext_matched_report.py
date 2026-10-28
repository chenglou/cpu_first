"""Tables for LADDER_MATCHED.md from microlab/ladder_matched_results.jsonl (plus LADDER.md's plain models).

    python -m microlab.ext_matched_report --tags M3k,M12k --holdout Mho
    python -m microlab.ext_matched_report --tags M3k,M12k,Mrem --ledger

- breaking points: as in LADDER.md (best seed >= 99% exact, unbroken run from the smallest size; a later budget,
  restarted at the earlier breaking point, after an arrow). The last column is the best plain model of LADDER.md.
- rule or table, per task and model:
  - unseen inputs: exact on the test inputs of a 1,000-example pool run that never occurred in training, at the first
    size where most are unseen;
  - seen / unseen at equal lengths: fresh data, exact on seen vs unseen test inputs with the same operand lengths
    (equal = a rule, unseen << seen = a table), at the largest size that has unseen inputs;
  - one step wider: exact on test inputs with a digit past the trained width, one size up, best over trained sizes;
  - held-out moduli: exact on moduli kept out of training.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import ext_ladder
from .ext_ladder_report import by_cell, fmt_top, top_size
from .ext_matched import FOR, RESULTS
from .tasks import TASKS

TASK_ORDER = ["add", "mul", "sq", "mod", "sqmod"]
SHOWN = ["colsum", "longdiv", "abacus", "abacus_loop"]   # the longdiv column holds colsum_longdiv for sqmod


def load(tag=None, path=RESULTS):
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    return [r for r in rows if (tag is None or r["tag"] == tag) and not r.get("train_task")]


def _m(task, m):
    return "colsum_longdiv" if (m == "longdiv" and task == "sqmod") else m


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else float(np.mean(xs))


def breaking_table(tags, plain_tags=("S3k", "S12k")):
    cells = [by_cell(load(tag=t)) for t in tags]
    plain = [by_cell(ext_ladder.load(tag=t)) for t in plain_tags]
    cols = [(m, d) for m in SHOWN for d in ("fresh", "pool1000")]
    head = [f"{m.replace('longdiv', 'longdiv*')} {d.replace('pool1000', 'pool 1k')}" for m, d in cols]
    lines = ["| task (size) | " + " | ".join(head) + " | best plain model, fresh (LADDER.md) |",
             "|---|" + "---|" * (len(cols) + 1)]
    for task in TASK_ORDER:
        sizes = ext_ladder.SIZES[TASKS[task].kind]
        row = []
        for m, d in cols:
            mm = _m(task, m)
            parts = [fmt_top(*top_size(c[(task, mm, d)], sizes), sizes) for c in cells if (task, mm, d) in c]
            row.append(" → ".join(parts) if task in FOR.get(mm, ()) and parts else "·")
        best = []
        for c in plain:
            tops = [top_size(c[(task, pm, "fresh")], sizes) for pm in ("mlp", "xf", "loop") if (task, pm, "fresh") in c]
            if tops:
                best.append(fmt_top(*max(tops, key=lambda tf: -1 if tf[0] is None else tf[0]), sizes))
        unit = "digits" if TASKS[task].kind == "digits" else "bits"
        lines.append(f"| {task} ({unit}) | " + " | ".join(row) + f" | {' → '.join(best) or '·'} |")
    return "\n".join(lines)


def _latest(tags):
    """by_cell over several tags; at a size run under more than one tag, the later tag's cells replace the earlier."""
    c = {}
    for t in tags:
        for key, by in by_cell(load(tag=t)).items():
            c.setdefault(key, {}).update(by)
    return c


def rule_table(tags, holdout_tag=None, min_unseen=0.7):
    c = _latest(tags)
    ho = by_cell(load(tag=holdout_tag)) if holdout_tag else {}
    lines = ["| task | model | unseen inputs, pool 1k (size) | seen / unseen at equal lengths, fresh (size) | "
             "one step wider than trained, fresh (best) | held-out moduli (size) |", "|---|---|---|---|---|---|"]
    for task in TASK_ORDER:
        sizes = ext_ladder.SIZES[TASKS[task].kind]
        for m in SHOWN:
            mm = _m(task, m)
            if task not in FOR.get(mm, ()):
                continue
            by = c.get((task, mm, "pool1000"), {})
            ok = [s for s in sorted(by) if np.mean([1 - r["seen_frac"] for r in by[s]]) >= min_unseen]
            un = f"{_mean([r['exact_unseen'] for r in by[ok[0]]]):.2f} ({ok[0]})" if ok else "·"
            byf = c.get((task, mm, "fresh"), {})
            mt = "·"
            have = [s for s in sorted(byf) if all(r.get("matched") for r in byf[s])]
            if have:                                              # the largest size with unseen inputs to compare
                at = have[-1]
                mt = (f"{_mean([r['matched']['seen'] for r in byf[at]]):.2f} / "
                      f"{_mean([r['matched']['unseen'] for r in byf[at]]):.2f} ({at})")
            wd = "·"
            best = None                                           # best exact one digit past the trained width
            for s, rs in byf.items():
                if rs[0].get("wide"):
                    w = min(rs[0]["wide"], key=int)
                    v = _mean([r["wide"][w]["wider"] for r in rs])
                    if v is not None and (best is None or v > best[2]):
                        best = (s, w, v)
            if best:
                wd = f"{best[0]} → {best[1]}: {best[2]:.2f}"
            hb = {s: rs for (t, m2, d), byh in ho.items() if t == task and m2 == mm for s, rs in byh.items()}
            hs = ", ".join(f"{_mean([r['exact_heldout_moduli'] for r in rs]):.2f} ({s})" for s, rs in sorted(hb.items()))
            lines.append(f"| {task} | {mm} | {un} | {mt} | {wd} | {hs or '·'} |")
    return "\n".join(lines)


def ledger(tags, path=RESULTS):
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    for t in tags:
        g = {}
        for r in rows:
            if r["tag"] == t:
                key = (r["task"], r["model"], r["data"], r.get("train_task") or "", r["steps"], r.get("holdout", 0))
                g.setdefault(key, {}).setdefault(r["size"], []).append(r)
        print(f"## {t}")
        for (task, m, d, tr, st, ho), by in sorted(g.items()):
            parts = []
            for s, rs in sorted(by.items()):
                p = f"{s}:{max(r['exact'] for r in rs):.2f}"
                if rs[0].get("matched"):
                    p += f" m{_mean([r['matched']['seen'] for r in rs]):.2f}/{_mean([r['matched']['unseen'] for r in rs]):.2f}"
                if ho:
                    p += f" ho{_mean([r.get('exact_heldout_moduli') for r in rs]):.2f}"
                if rs[0].get("wide"):
                    p += " w[" + " ".join(f"{w}:{_mean([r['wide'][w]['wider'] for r in rs]) or 0:.2f}"
                                          for w in sorted(rs[0]["wide"], key=int)) + "]"
                parts.append(p)
            label = f"{task:6} {m:14} {d:8}" + (f" train={tr}" if tr else "") + f" {st // 1000}k" + (f" ho{ho}" if ho else "")
            print(label + "  " + "  ".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", default="M3k")
    p.add_argument("--holdout", default=None, help="tag of the held-out-moduli runs")
    p.add_argument("--ledger", action="store_true")
    a = p.parse_args()
    tags = a.tags.split(",")
    if a.ledger:
        return ledger(tags)
    print(breaking_table(tags))
    print()
    print(rule_table(tags, a.holdout))


if __name__ == "__main__":
    main()
