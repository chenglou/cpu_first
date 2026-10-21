"""Summaries of the learned-wiring run records (cpu_first/runs/learned_wiring/<name>.json).

    python summarize_wiring.py NAME [NAME ...]          # one block per record
    python summarize_wiring.py --table NAME [NAME ...]  # one markdown row per record
    python summarize_wiring.py --stage div              # the stage's official run and CPU seeds (div, sq, io3)

From each record: the evaluator's RESULT_JSON (certified depth on both profiles, rungs, splits, times), the snapped
tables the submission printed at its first evaluation batch (TABLE lines) checked against true subtraction and
multiplication here (outside the submission), the division's wiring (WIRING), the squaring's layout (LAYOUT), the
pointer shape (IO shape), the schedule (LR lines), the evaluation audits (AUDIT lines) and the md5 of the exact source.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs" / "learned_wiring"


def truth():
    td = "".join(str((r - m - c) % 10) for r in range(10) for m in range(10) for c in range(2))
    tb = "".join(str(int(r - m - c < 0)) for r in range(10) for m in range(10) for c in range(2))
    plo = "".join(str((a * b) % 10) for a in range(10) for b in range(10))
    phi = "".join(str((a * b) // 10) for a in range(10) for b in range(10))
    return td, tb, plo, phi


def parse(rec: dict) -> dict:
    out = rec.get("stdout", "")
    lines = out.splitlines()
    res = rec.get("result") or {}
    seed = (res.get("seeds") or [{}])[0]
    dp = res.get("depth_profile", {})
    sdp = seed.get("depth_profile", {})
    get = lambda p: next((l.split("=", 1)[1] for l in lines if l.startswith(p)), "")
    td, tb, plo, phi = truth()
    d, b, lo, hi = get("TABLE digit="), get("TABLE borrow="), get("TABLE prod_lo="), get("TABLE prod_hi=")
    rows = sum(d[i] == td[i] and b[i] == tb[i] for i in range(min(len(d), 200))) if d else None
    cells = sum(lo[i] == plo[i] and hi[i] == phi[i] for i in range(min(len(lo), 100))) if lo else None
    wiring = next((l[7:] for l in lines if l.startswith("WIRING ") and "probabilities" not in l), "")
    layout = next((re.search(r"right=(\d+)/(\d+)", l).group(0)[6:] for l in lines if l.startswith("LAYOUT")), "-")
    shape = next((l.split("shape=")[1].split()[0] for l in lines if l.startswith("IO shape=")), "step (designed)")
    frozen = next((re.search(r"step=(\d+)", l).group(1) for l in lines if l.startswith("LR frozen")), "-")
    never = [int(x) for l in lines if l.startswith("AUDIT") for x in re.findall(r"used_but_never_trained=(\d+)", l)]
    return {
        "maxT": dp.get("max_certified_time_steps"), "oodN": dp.get("ood_n_max_certified_time_steps"),
        "rungs": [g["correct_examples"] for g in sdp.get("rungs", [])],
        "ood_rungs": [g["correct_examples"] for g in sdp.get("ood_n_rungs", [])],
        "eval": {k: v.get("exact_accuracy") for k, v in seed.get("evaluation", {}).items()},
        "train_s": round(seed.get("training_seconds", 0), 1), "eval_s": round(seed.get("evaluation_seconds", 0), 1),
        "seed": seed.get("seed"), "params": seed.get("model_state_elements"), "rows": rows, "cells": cells,
        "gate": get("TABLE gate=")[:2], "wiring": wiring, "layout": layout, "shape": shape, "frozen": frozen,
        "never": max(never) if never else None, "md5": hashlib.md5(rec.get("submission", "").encode()).hexdigest(),
        "gpu": rec.get("gpu") or "cpu",
        "schedule": [l for l in lines if l.startswith(("LR switch", "LR ladder", "LR chain", "LR io_wiring",
                                                       "LR io_stage", "LR frozen"))],
        "audits": sorted({l for l in lines if l.startswith("AUDIT")}, key=lines.index),
    }


def stage_table(tag: str) -> None:
    """One markdown row per record of a stage (TAG_official, TAG_cpu_seed74, 1..7): results, tables, wiring, schedule."""
    print("| run | seed | max T seen / unseen | rungs (fewest right of 768, seen / unseen) | test, ood_t, ood_n_t | "
          "subtractor, product cell, gate | division wiring | layout | shape | reading chosen | ladder joins | "
          "chain lesson joins | I/O lesson joins | T settled | frozen | never trained | training s | evaluation s |")
    print("|---|" * 1 + "---|" * 17)
    for name in [f"{tag}_official"] + [f"{tag}_cpu_seed{s}" for s in (74, 1, 2, 3, 4, 5, 6, 7)]:
        p = RUNS / f"{name}.json"
        if not p.exists():
            print(f"| {name} | (no record) |")
            continue
        rec = json.loads(p.read_text())
        r = parse(rec)
        hist = next((l for l in rec.get("stdout", "").splitlines() if l.startswith("LR frozen")), "")
        ev = {}
        for k in ("io4_reading", "ladder", "chain", "wire_io", "T_resolved"):
            m = re.search(r"\((\d+), '" + k + "'", hist)
            ev[k] = m.group(1) if m else "-"
        where = "H100" if "H100" in r["gpu"] else "CPU"
        ev_s = ", ".join(str(r["eval"].get(k)) for k in ("test", "ood_t", "ood_n_t"))
        true = "true" if r["wiring"].endswith("true=True") else r["wiring"]
        print(f"| {name} ({where}) | {r['seed']} | {r['maxT']} / {r['oodN']} | {min(r['rungs'] or [0])} / "
              f"{min(r['ood_rungs'] or [0])} | {ev_s} | {r['rows']}/200, {r['cells']}/100, {r['gate']} | {true} | "
              f"{r['layout']} | {r['shape']} | {ev['io4_reading']} | {ev['ladder']} | {ev['chain']} | {ev['wire_io']} | "
              f"{ev['T_resolved']} | {r['frozen']} | {r['never']} | {r['train_s']} | {r['eval_s']} |")


def main(argv):
    if argv[:1] == ["--stage"]:
        stage_table(argv[1])
        return
    table = argv[:1] == ["--table"]
    names = argv[1:] if table else argv
    if table:
        print("| run | seed | max T seen / unseen | rungs (min of 768, seen / unseen) | test, ood_t, ood_n_t | "
              "subtractor / product / gate | division wiring true | layout right | shape | frozen at | "
              "never trained | train s | eval s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        p = RUNS / f"{name}.json"
        if not p.exists():
            print(f"| {name} | (no record) |" if table else f"== {name}: no record")
            continue
        r = parse(json.loads(p.read_text()))
        if table:
            ev = ", ".join(str(r["eval"].get(k, "-")) for k in ("test", "ood_t", "ood_n_t"))
            true = r["wiring"].split("true=")[-1] if "true=" in r["wiring"] else "-"
            print(f"| {name} | {r['seed']} | {r['maxT']} / {r['oodN']} | {min(r['rungs'] or [0])} / "
                  f"{min(r['ood_rungs'] or [0])} | {ev} | {r['rows']}/200, {r['cells']}/100, {r['gate']} | {true} | "
                  f"{r['layout']} | {r['shape']} | {r['frozen']} | {r['never']} | {r['train_s']} | {r['eval_s']} |")
            continue
        print(f"== {name} ({r['gpu']}) md5={r['md5']} params={r['params']}")
        for k in ("maxT", "oodN", "rungs", "ood_rungs", "eval", "train_s", "eval_s", "rows", "cells", "gate", "wiring",
                  "layout", "shape", "frozen", "never"):
            print(f"  {k}: {r[k]}")
        for l in r["schedule"] + r["audits"]:
            print("  ", l[:300])


if __name__ == "__main__":
    main(sys.argv[1:])
