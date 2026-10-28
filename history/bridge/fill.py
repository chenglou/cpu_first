"""Rewrite BRIDGE.md's generated tables from the records, between <!-- NAME --> and <!-- /NAME --> markers.

    ~/github/sotaku/venv/bin/python -m bridge.fill
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from bridge import summarize as Sm

HERE = Path(__file__).resolve().parent

NOTCH_NAMES = {
    "N1_table": "1a. the subtraction table learned",
    "N1_tables": "1b. all three rules learned as tables",
    "N1_mlp": "1c. all three rules one ReLU MLP",
    "N1_mlp_st": "1c'. the same, straight-through states",
    "N1_mlp_sub": "1d. only the subtraction cell a ReLU MLP",
    "N1_mlp_sub_st": "1d'. the same, straight-through states",
    "N1_mlptab_sub": "1e. the subtraction table an MLP's outputs",
    "N1_mlp_sub_w256": "1d, 256 hidden units",
    "N1_mlptab_sub_w256": "1e, 256 hidden units",
    "N2_narrow_given": "2a. narrow menu (2-3 options), tables given",
    "N2_narrow": "2a. narrow menu, table learned",
    "N2_menu_given": "2b. wide menu (7 options), tables given",
    "N2_menu": "2b. wide menu, table learned",
    "N2_menu_mlptab_sub": "2b. wide menu, the table an MLP's outputs",
    "N3_free_given": "3. free attention, tables given",
    "N3_free": "3. free attention, table learned",
    "N3_free_mlp": "3. free attention, one ReLU MLP",
    "N4_symbols": "4. free symbols (format learned)",
    "N2_rel13_given": "2c. 13-option menu, tables given",
    "N3_free_given_st": "3. free attention, tables given, snapped states",
    "N3_free_given_first": "3. free attention, tables given, prompt at tick 0 only",
    "N3_free_given_first_st": "3. the same, snapped states",
    "N3_free_first": "3. free attention, table learned, prompt at tick 0 only",
    "N4_symbols_untied": "4. free symbols, untied blocks",
    "N5_free": "5. continuous states (residual block)",
    "N6_plain": "6. plain looped transformer",
}


def ladder_table(suite):
    rows = Sm.load(suite)
    if not rows:
        return "(no records yet)"
    for r in rows:
        r["tier"] = Sm.tier_of(r)
    by = defaultdict(list)
    for r in rows:
        by[(r["notch"], r["tier"], r["size"])].append(r)
    sizes = sorted({r["size"] for r in rows})
    out = ["| notch | lessons | " + " | ".join(f"({s})" for s in sizes) + " |", "|" + "---|" * (len(sizes) + 2)]
    seen = []
    for r in rows:
        key = (r["notch"], r["tier"])
        if key not in seen:
            seen.append(key)
    order = list(NOTCH_NAMES)
    seen.sort(key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1]))
    for n, t in seen:
        cells = [Sm.cell(by[(n, t, s)]) for s in sizes]
        out.append(f"| {NOTCH_NAMES.get(n, n)} | {t} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def width_table():
    p = HERE / "runs" / "menu_width_local.jsonl"
    if not p.exists():
        return "(no records yet)"
    rows = [json.loads(l) for l in open(p)]
    by = defaultdict(list)
    for r in rows:
        by[r["options"]].append(r)
    out = ["| options per port | certified | median steps | training share of the others |", "|---|---|---|---|"]
    for k in sorted(by):
        rs = by[k]
        c = [r for r in rs if r["certified"]]
        f = [r["train"] for r in rs if not r["certified"]]
        out.append(f"| {k} | {len(c)}/{len(rs)} | {int(np.median([r['hit'] for r in c])) if c else '·'} | "
                   f"{('%.2f' % np.median(f)) if f else '·'} |")
    return "\n".join(out)


def compile_table():
    r = json.loads((HERE / "records" / "compile.json").read_text())
    out = ["| (B, w, k) | questions | cells | ticks | every setting exact, hard and soft, every cell, from random scratch |"
           " smallest soft probability of the right answer |", "|---|---|---|---|---|---|"]
    for size, row in r["sizes"].items():
        f = row["free"]
        ok = all(row[s]["hard"]["exact"] == 1 and row[s]["soft"]["exact"] == 1 and row[s]["hard"]["cells_digit"] == 1
                 and row[s]["hard"]["cells_borrow"] == 1 and row[s]["hard_from_random_scratch"] == 1
                 for s in r["settings"])
        q = f"{f['questions']:,}" + (" (sample)" if f["sampled"] else "")
        pmin = min(row[s]["soft"]["min_p_answer"] for s in r["settings"])
        out.append(f"| ({size}) | {q} | {f['cells']} | {f['ticks']} | {'yes' if ok else 'NO'} | {pmin:.9f} |")
    return "\n".join(out)


def diag_table():
    """One row per case and seed. The wiring cases: the share of ports (head and kind) whose every cell reads the
    compiled source, the share of cells whose argmax source is an input cell, the first gradient's share toward input
    cells, and the answer loss on the straight line to the compiled weights (start -> highest -> end). The rule cases:
    the rules' entries right. (The straight line is reported only for the wiring: two MLPs that compute the same table
    are different points, so a barrier between them says nothing.)"""
    p = HERE / "records" / "diagnose.json"
    if not p.exists():
        return "(no records yet)"
    recs = json.loads(p.read_text())
    order = list(dict.fromkeys(r["case"] for r in recs))
    recs.sort(key=lambda r: (order.index(r["case"]), r["seed"]))
    out = ["| case | seed | certified | training, hard / soft | ports right | cells reading an input | first gradient "
           "toward an input | loss on the line to the compiled weights | rules right (SUB digits, copy, select) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        ports = inp = g0 = line = "·"
        if "heads" in r:
            hs = r["heads"]
            ports = f"{sum(v.get('right', 0) == 1 for v in hs.values())}/{len(hs)}"
            inp = f"{np.mean([v.get('input', 0) for v in hs.values()]):.2f}"
        if "grad0_start" in r:
            g0 = f"{np.mean([v.get('input', 0) for v in r['grad0_start'].values()]):.2f}"
        if "path" in r and "heads" in r:
            losses = [x[1] for x in r["path"]]
            line = f"{losses[0]:.2f} -> {max(losses):.2f} -> {losses[-1]:.2f}"
        rules = ", ".join(str(r[k]) for k in ("sub_digits_right", "shift_right", "sel_right") if k in r)
        out.append(f"| {r['case']} | {r['seed']} | {'yes' if r['certified'] else 'no'} | {r['train_hard']:.2f} / "
                   f"{r['train_soft']:.2f} | {ports} | {inp} | {g0} | {line} | {rules} |")
    return "\n".join(out)


def census_table():
    rows = []
    for p in sorted((HERE / "records").glob("census_*.json"), key=lambda q: int(q.stem.split("_")[1])):
        r = json.loads(p.read_text())
        c = r["counts"]
        true_ok = any(x["true"] for x in r["passing_both"])
        rows.append(f"| {r['options']} | {c['wirings']:,} | {c['fit']:,} | {c['link']:,} | {c['both']:,} | "
                    f"{c['cert']:,} | {'yes' if true_ok else 'no'} |")
    if not rows:
        return "(no records yet)"
    head = ["| options per port | wirings | fit every training answer | pass the one-link law | both | both and "
            "certify | the true wiring among them |", "|---|---|---|---|---|---|---|"]
    return "\n".join(head + rows)


def h1_seed_row(r):
    import hashlib
    import re
    res = r.get("result")
    if not res:
        return None
    sd = res["seeds"][0]
    dp = sd["depth_profile"]
    out = r["stdout"]
    fz = re.search(r"^LR frozen step=(\d+)", out, re.M)
    lines = {l.split("=")[0]: l.split("=", 1)[1] for l in out.splitlines()
             if l.startswith(("TABLE digit=", "TABLE borrow=", "TABLE prod_lo=", "TABLE prod_hi="))}
    dig = "".join(str((a - m - c) % 10) for a in range(10) for m in range(10) for c in range(2))
    bor = "".join(str(int(a - m - c < 0)) for a in range(10) for m in range(10) for c in range(2))
    lo = "".join(str((a * b) % 10) for a in range(10) for b in range(10))
    hi = "".join(str((a * b) // 10) for a in range(10) for b in range(10))
    exact = (lines.get("TABLE digit") == dig and lines.get("TABLE borrow") == bor and lines.get("TABLE prod_lo") == lo
             and lines.get("TABLE prod_hi") == hi)
    chain = re.search(r"^WIRING .*chain=(\S+)", out, re.M)
    rungs = all(x["correct_examples"] == x["example_count"] for x in dp["rungs"] + dp["ood_n_rungs"])
    ev = sd["evaluation"]
    splits = all(ev[k]["correct_examples"] == ev[k]["example_count"] for k in ("test", "ood_t", "ood_n_t"))
    md5 = hashlib.md5(r["submission"].encode()).hexdigest()[:8]
    return (f"| {sd['seed']} | {dp['max_certified_time_steps']} / {dp['ood_n_max_certified_time_steps']} | "
            f"{'yes' if rungs else 'no'} | {'yes' if splits else 'no'} | {sd['training_seconds']:.0f} "
            f"({fz.group(1) if fz else '·'}) | {sd['evaluation_seconds']:.0f} | {'yes' if exact else 'no'} | "
            f"{chain.group(1).replace('_', ' ') if chain else '·'} | {md5} |")


def h1_cpu_table():
    recs = sorted((HERE / "h1" / "records").glob("wide_mlp_cpu_seed*.json"))
    rows = [h1_seed_row(json.loads(p.read_text())) for p in recs]
    rows = [x for x in rows if x]
    head = ["**CPU seeds** (the same evaluator with `device = cpu`, the file byte for byte; `h1/plans/wide_mlp_cpu.json`):",
            "", "| seed | max T, seen / unseen | every rung 768/768 | splits exact | training s (frozen at step) | "
            "evaluation s | both MLP cells exact | chain | md5 |", "|---|---|---|---|---|---|---|---|---|"]
    if not rows:
        return "(the 8 CPU seeds are running)"
    return "\n".join(head + rows)


def softcert_table():
    rows = Sm.load("softcert")
    if not rows:
        return "(no records yet)"
    by = defaultdict(list)
    for r in rows:
        by[(r["notch"], r["tier"])].append(r)
    out = ["| notch | lessons | runs | fit training, snapped | fit training, soft | median soft share at (3, 2, 2) | at (3, 3, 2) |",
           "|---|---|---|---|---|---|---|"]
    for (n, t), rs in by.items():
        hard = sum(r["train_hard"] == 1.0 for r in rs)
        soft = sum(r["train_soft"] == 1.0 for r in rs)
        b1 = np.median([r["bigger_soft"]["3,2,2"] for r in rs])
        b2 = np.median([r["bigger_soft"]["3,3,2"] for r in rs])
        out.append(f"| {NOTCH_NAMES.get(n, n)} | {t} | {len(rs)} | {hard} | {soft} | {b1:.2f} | {b2:.2f} |")
    out.append("| always answer 0 | | | | | 0.18 | 0.06 |")
    return "\n".join(out)


def wstab_table():
    """Stability from the compiled point: per notch, width and lessons, at each noise level, "c/n (s)": c runs
    certified at the end out of n, s of them exact (hard path) before training."""
    rows = Sm.load("wstab")
    if not rows:
        return "(no records yet)"
    by = defaultdict(list)
    for r in rows:
        width = "widened" if r["name"].split(".")[1] == "wide" else "minimal"
        by[(r["notch"], width, r["cfg"]["lessons"] == {"ans": 1.0}, r["cfg"]["init_noise"])].append(r)
    sig = sorted({k[3] for k in by})
    out = ["| notch | compiled as | lessons | " + " | ".join(f"σ = {x:g}" for x in sig) + " |",
           "|---|---|---|" + "---|" * len(sig)]
    keys = sorted({k[:3] for k in by}, key=lambda k: (list(NOTCH_NAMES).index(k[0]) if k[0] in NOTCH_NAMES else 99,
                                                      k[2], k[1]))
    for n, width, ans in keys:
        cells = []
        for x in sig:
            rs = by.get((n, width, ans, x), [])
            if not rs:
                cells.append("·")
                continue
            c = sum(bool(r.get("certified")) for r in rs)
            st = sum(r.get("start_hard") == 1.0 for r in rs)
            cells.append(f"{c}/{len(rs)} ({st})")
        tier = "the answers" if ans else rows[0]["tier"] and next(r["tier"].split()[0] for r in rows if r["notch"] == n
                                                                    and r["cfg"]["lessons"] != {"ans": 1.0})
        out.append(f"| {NOTCH_NAMES.get(n, n)} | {width} | {tier} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    path = HERE / "BRIDGE.md"
    text = path.read_text()
    blocks = {"COMPILE": compile_table, "LADDER": lambda: ladder_table("ladder"), "WIDTH": width_table,
              "MLPCELL": lambda: ladder_table("mlpcell"), "DIAG": diag_table, "CENSUS": census_table, "SOFTCERT": softcert_table,
              "WSTAB": wstab_table, "WLEARN": lambda: ladder_table("wlearn"), "BYPASS": lambda: ladder_table("bypass"),
              "H1CPU": h1_cpu_table}
    for name, fn in blocks.items():
        pat = re.compile(rf"(<!-- {name} -->\n).*?(<!-- /{name} -->)", re.S)
        if pat.search(text):
            text = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(2), text)
    path.write_text(text)
    print("filled", path)


if __name__ == "__main__":
    main()
