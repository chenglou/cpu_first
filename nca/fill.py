"""Fill NCA.md's tables from the records (between <!-- NAME --> and <!-- /NAME --> markers).

    ~/github/sotaku/venv/bin/python -m nca.fill

Tables: COMPILE (records/compile.json), SITUATIONS (records/situations.json), VERIFY (records/verify.json), and from
nca/runs/*.jsonl: LESSONS (every run taught by lessons, division at (3, 2, 1) unless its row says otherwise), TRACE
(every run taught by the compiled trace), SQUARE and LOGIC. A cell of a run table reads "certified / runs"; the training and bigger-size shares are
medians over the arm's runs, on the hard path, "snapped / soft" for training.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUITES = ("explore", "ladder2", "ladder3", "free2", "sparse1", "causal2", "square1", "logic2", "logic3", "partial1",
          "cover1", "cover2", "cover3", "selfcopy2", "hyper1", "hyper2", "hyper3", "hyper_stab")

# the generated wiring (hyper.py): tag -> (wiring, lessons)
HYPER = {
    "hyper_A": ("softmax", "answers"),
    "hyper_GCL": ("softmax", "GCL"),
    "hyper_GCL_ste": ("softmax, snapped (STE)", "GCL"),
    "hyper_GCL_tf": ("softmax", "GCL, the cell's lessons first and g frozen (the rest from step 300)"),
    "hyper_GCL_link1st": ("softmax", "GCL, the cell's lessons and one link first (the rest from step 600)"),
    "hyper_sig_GCL": ("sigmoid", "GCL"),
    "hyper_causal_GCL": ("softmax, causal mask", "GCL"),
    "hyper_trace": ("softmax", "trace at (3, 2, 1)"),
    "hyper_causal_trace": ("softmax, causal mask", "trace at (3, 2, 1)"),
    "hyper_sig_trace": ("sigmoid", "trace at (3, 2, 1)"),
    "hyper_w3_trace": ("softmax", "trace at (3, 3, 2), batches of 256"),
    "hyper_w3_causal_trace": ("softmax, causal mask", "trace at (3, 3, 2), batches of 256"),
}

# arm tag -> (table, notch, lessons)
ARMS = {
    # lessons, the compiled format, the 3 x 3 with the cell's own state hidden
    "A": ("LESSONS", "noself", "answers"),
    "A_ste": ("LESSONS", "noself, snapped (STE)", "answers"),
    "A_3x3": ("LESSONS", "3x3 (the cell sees its own state)", "answers"),
    "GL": ("LESSONS", "noself", "GL: answers, x mod N = x, one-link law"),
    "GL_indep": ("LESSONS", "noself", "GL + the answer must not depend on the scratch start"),
    "GCL": ("LESSONS", "noself", "GCL: GL + the cell's subtraction lessons"),
    "GCL_ste": ("LESSONS", "noself, snapped (STE)", "GCL"),
    "GCL_tf": ("LESSONS", "noself", "GCL, the cell's lessons first (the rest from step 300)"),
    "GCL_tf_ste": ("LESSONS", "noself, snapped (STE)", "GCL, the cell's lessons first"),
    "GCL_link1st": ("LESSONS", "noself", "GCL, the cell's lessons and one link first (the rest from step 600)"),
    "GCL_lr01": ("LESSONS", "noself, lr 0.01", "GCL"),
    "GCL_ctxrand": ("LESSONS", "noself", "GCL, the cell's lessons on random contexts"),
    "GCL_3x3": ("LESSONS", "3x3", "GCL"),
    "GCL_scr": ("LESSONS", "noself, random scratch starts", "GCL"),
    "split_GCL": ("LESSONS", "noself, one MLP block per channel", "GCL"),
    "k1_sheet": ("LESSONS", "noself", "answers + the worksheet's visible digits"),
    "k2_A": ("LESSONS", "noself, trained at (3, 2, 2)", "answers"),
    "k2_GCL": ("LESSONS", "noself, trained at (3, 2, 2)", "GCL (step law off)"),
    "k2_GCL_ste": ("LESSONS", "noself, STE, trained at (3, 2, 2)", "GCL (step law off)"),
    "k2_sheet": ("LESSONS", "noself, trained at (3, 2, 2)", "answers + the worksheet's visible digits"),
    "ncol_GCL": ("LESSONS", "noself + N's column", "GCL"),
    "ports_A": ("LESSONS", "noself + the bridge's ports, digit-and-borrow cells", "answers"),
    "ports_GCL": ("LESSONS", "noself + the bridge's ports, digit-and-borrow cells", "GCL"),
    "ports_GCL_ste": ("LESSONS", "the same, STE", "GCL"),
    "ports_GCL_tf": ("LESSONS", "noself + the bridge's ports, digit-and-borrow cells", "GCL, the cell's lessons first"),
    "k2_ports_GCL": ("LESSONS", "the same, trained at (3, 2, 2)", "GCL (step law off)"),
    "causal_A": ("LESSONS", "causal", "answers"),
    "causal_GCL": ("LESSONS", "causal", "GCL"),
    "causal_GCL_ste": ("LESSONS", "causal, STE", "GCL"),
    "causal_k2_GCL": ("LESSONS", "causal, trained at (3, 2, 2)", "GCL (step law off)"),
    "w3_causal_GCL": ("LESSONS", "causal, trained at (3, 3, 2)", "GCL (step law off)"),
    "causal_GCL_l1_1.0": ("LESSONS", "causal, L1 1.0 on the input weights", "GCL"),
    "causal_GCL_ste_l1_1.0": ("LESSONS", "causal, STE, L1 1.0", "GCL"),
    "free_A": ("LESSONS", "free format, noself", "answers"),
    "free_GL": ("LESSONS", "free format, noself", "GL"),
    "free_GL_ste": ("LESSONS", "free format, noself, STE", "GL"),
    "free_GL_wide": ("LESSONS", "free format with 6 hidden channels", "GL"),
    "free_GL_indep": ("LESSONS", "free format, noself", "GL + the answer must not depend on the scratch start"),
    "free_sheet": ("LESSONS", "free format, noself", "answers + the worksheet's visible digits"),
    "free_sheet_ste": ("LESSONS", "free format, noself, STE", "answers + the worksheet's visible digits"),
    "free_k2_sheet_causal": ("LESSONS", "free format, causal, trained at (3, 2, 2)",
                             "answers + the worksheet's visible digits"),
    # the trace
    "fixpoint_3x3": ("TRACE", "3x3", "(3, 2, 1), only the settled grid told"),
    "fixpoint_noself": ("TRACE", "noself", "(3, 2, 1), only the settled grid told"),
    "trace": ("TRACE", "noself", "(3, 2, 1)"),
    "trace_3x3": ("TRACE", "3x3", "(3, 2, 1)"),
    "ports_trace": ("TRACE", "noself + the bridge's ports", "(3, 2, 1)"),
    "k2_trace": ("TRACE", "noself", "(3, 2, 2)"),
    "w3_trace": ("TRACE", "noself", "(3, 3, 2), batches of 256"),
    "causal_trace": ("TRACE", "causal", "(3, 2, 1)"),
    "causal_trace_more": ("TRACE", "causal", "(3, 2, 1)"),
    "causal_k2_trace": ("TRACE", "causal", "(3, 2, 2)"),
    "w3_causal_trace": ("TRACE", "causal", "(3, 3, 2), batches of 256"),
    **{f"{h}_trace_l1_{l}": ("TRACE", f"{h}, L1 {l} on the input weights", "(3, 2, 1)")
       for h in ("noself", "causal") for l in (0.3, 1.0, 3.0)},
    **{f"partial_{c}": ("TRACE", "causal", f"(3, 2, 1), only {', '.join(c)} told") for c in ("d", "db", "dbn", "dbnm")},
    # the square
    "sq_A": ("SQUARE", "noself", "the square's answers"),
    "sq_A_ste": ("SQUARE", "noself, STE", "the square's answers"),
    "sq_dGCL_A": ("SQUARE", "noself", "the square's answers + the division's GCL on its own grid"),
    "sq_dGCL_A_prod": ("SQUARE", "noself", "the same + the product x^2 as a label"),
    "sq_trace": ("SQUARE", "noself", "the compiled trace"),
    "sq_causal_trace": ("SQUARE", "causal", "the compiled trace"),
    "sq_ports_trace": ("SQUARE", "noself + the bridge's ports", "the compiled trace"),
    "sq_w3_causal_trace": ("SQUARE", "causal, trained at (3, 3), batches of 64", "the compiled trace (12 ticks a step)"),
    # the bit-level arm
    "logic_A": ("LOGIC", "256 x 4 gates, noself", "answers"),
    "logic_GL": ("LOGIC", "256 x 4 gates, noself", "GL"),
    "logic_GCL": ("LOGIC", "256 x 4 gates, noself", "GCL"),
    "logic_GCL_tf": ("LOGIC", "256 x 4 gates, noself", "GCL, the cell's lessons first"),
    "logic_GCL_ste_wide": ("LOGIC", "1024-1024-512-256 gates, noself, STE", "GCL"),
    "logic_trace": ("LOGIC", "256 x 4 gates, noself", "trace at (3, 2, 1)"),
    "logic_trace_wide": ("LOGIC", "1024-1024-512-256 gates, noself", "trace at (3, 2, 1)"),
    "logic_k2_trace_wide": ("LOGIC", "1024-1024-512-256 gates, noself", "trace at (3, 2, 2)"),
    "logic_ports_trace_wide": ("LOGIC", "1024-1024-512-256 gates + the bridge's ports", "trace at (3, 2, 1)"),
    "logic_causal_trace": ("LOGIC", "1024-1024-512-256 gates, causal", "trace at (3, 2, 1)"),
    "logic_causal_k2_trace": ("LOGIC", "1024-1024-512-256 gates, causal", "trace at (3, 2, 2)"),
    "logic3_k1_narrow": ("LOGIC", "256 x 4 gates, causal, 10,000 steps", "trace (6 ticks a step) at (3, 2, 1)"),
    "logic3_k1_mid": ("LOGIC", "512-512-512-256 gates, causal, 10,000 steps", "trace (6 ticks a step) at (3, 2, 1)"),
    "logic3_k2_mid": ("LOGIC", "512-512-512-256 gates, causal, 10,000 steps",
                      "trace (6 ticks a step) at (3, 2, 2), batches of 64"),
}


def load():
    """Every run record, once: free2 repeats three explore arms with the same seeds, and those reproduce exactly
    (same steps and shares), so a repeat counts once."""
    rows, seen = [], set()
    for s in SUITES:
        p = HERE / "runs" / f"{s}.jsonl"
        if p.exists():
            for l in open(p):
                r = json.loads(l)
                key = (r["cfg"]["tag"], r["cfg"]["seed"], r["steps"], r["train_hard"], r["train_soft"])
                if key not in seen:
                    seen.add(key)
                    rows.append(dict(r, _suite=s))
    return rows


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def f2(x):
    return "·" if x is None else f"{x:.2f}"


def offsets(r):
    """A division run's measured shares by (dw, dk) away from its training size (bigger and extra sizes)."""
    c = r["cfg"]
    out = {}
    for key in ("bigger", "extra"):
        for k, v in (r.get(key) or {}).items():
            b, w, k_ = (int(x) for x in k.split(","))
            out[(w - c["w"], k_ - c["k"])] = v
    return out


def run_table(rows, table):
    """One row per (notch, lessons) label: arms with the same label (the same setting, more seeds) are pooled."""
    by = defaultdict(list)
    first = {}
    for r in rows:
        t = r["cfg"]["tag"]
        if ARMS.get(t, (None,))[0] == table:
            lab = ARMS[t][1:]
            if r["_suite"] in ("cover2", "cover3"):           # certified on every question, not on samples
                lab = (lab[0], lab[1] + ", every question checked")
            by[lab].append(r)
            first.setdefault(lab, t)
    head = {"LESSONS": "| notch | lessons | certified | training, snapped / soft | bigger, snapped | sub table | answers 0 |",
            "TRACE": "| neighbourhood | trained at | certified | training | k + 1 | k + 2 | w + 1, k + 1 | w + 2, k + 2 |",
            "SQUARE": "| notch | lessons | certified | training, snapped / soft | bigger, snapped | the plain division grid (B, w, 1) |",
            "LOGIC": "| rule | lessons | certified | training, snapped / soft | bigger, snapped | gates |"}[table]
    out = [head, "|" + "---|" * (head.count("|") - 1)]
    labels = []
    for a in ARMS:
        for lab in (ARMS[a][1:], (ARMS[a][1], ARMS[a][2] + ", every question checked")):
            if ARMS[a][0] == table and lab in by and lab not in labels:
                labels.append(lab)
    for lab in labels:
        rs = by[lab]
        notch, less = lab
        cert = sum(bool(r.get("certified")) for r in rs)
        hits = [r["hit"] for r in rs if r.get("certified") and r.get("hit")]
        c = f"{cert}/{len(rs)}" + (f" ({int(med(hits))})" if hits else "")
        fit = sum(r["train_hard"] == 1.0 for r in rs)
        tr = f"{f2(med([r['train_hard'] for r in rs]))} / {f2(med([r['train_soft'] for r in rs]))}"
        keys = list(rs[0].get("bigger", {}))
        big = " / ".join(f2(med([r.get("bigger", {}).get(k) for r in rs])) for k in keys)
        if table == "LESSONS":
            p = [r["probe"] for r in rs]
            sub = ("·" if p[0].get("sub_digits_right") is None else
                   f"{int(med([x['sub_digits_right'] for x in p]))} + {int(med([x['sub_borrows_right'] for x in p]))}")
            fits = f" (fit {fit})" if fit and not cert else ""
            out.append(f"| {notch} | {less} | {c}{fits} | {tr} | {big} | {sub} | {f2(med([x['answers_0'] for x in p]))} |")
        elif table == "TRACE":
            hit = [r["hit"] for r in rs if r.get("hit")]
            trn = f"{fit}/{len(rs)} exact" + (f" (step {int(med(hit))})" if hit else "")
            cols = []
            for dw, dk in ((0, 1), (0, 2), (1, 1), (2, 2)):
                vals = [v for v in (offsets(r).get((dw, dk)) for r in rs) if v is not None]
                cols.append("·" if not vals else f"{med(vals):.3f}" + (f" (min {min(vals):.3f})" if min(vals) < med(vals) - 5e-4
                                                                     and min(vals) > 0.9 else ""))
            out.append(f"| {notch} | {less} | {c} | {trn} | {' | '.join(cols)} |")
        elif table == "SQUARE":
            dv = med([r.get("div", {}).get(f"{r['cfg']['B']},{r['cfg']['w']},{r['cfg']['div_k']}") for r in rs])
            out.append(f"| {notch} | {less} | {c} | {tr} | {big} | {f2(dv)} |")
        else:
            out.append(f"| {notch} | {less} | {c} | {tr} | {big} | {rs[0]['params'] // 16} |")
    return "\n".join(out)


def compile_table():
    """One row per grid; "yes" means every cell of every question exact on the hard path from blank and from random
    scratch cells and still at 2x the ticks, and the soft path right on every cell."""
    rec = json.loads((HERE / "records" / "compile.json").read_text())
    by = defaultdict(dict)
    for name, v in rec["sizes"].items():
        j = v["job"]
        key = (j.get("task", "div"), j["B"], j["w"], j["k"])
        variant = "tick0" if not j.get("clamp", True) else j.get("reach", "3x3")
        by[key][variant] = v
    ok = lambda v: "·" if v is None else ("yes" if all(v["hard_exact"].values()) and v["soft_every_cell"] else "NO")
    out = ["| grid | questions | cells | ticks: every cell / the answer | hidden units | exact | prompt at tick 0 only | "
           "+ N's column (units) | + the bridge's ports (units) | smallest soft p(answer) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(by, key=lambda k: (k[0] != "div", k[1], k[2], k[3]))
    for key in order:
        g = by[key]
        v = g.get("3x3")
        if v is None:
            continue
        task, B, w, k = key
        what = f"x² mod N ({B}, {w})" if task == "sq" else f"v mod N ({B}, {w}, {k})"
        q = f"{v['questions']:,}" + (" (sample)" if v["job"].get("sample") else "")
        reach = {}
        for r in ("ncol", "ports"):
            x = g.get(r)
            reach[r] = ok(x) + (f" ({x['hidden_units']})" if x else "")
        minp = min(x["soft_min_p"] for x in g.values())
        out.append(f"| {what} | {q} | {v['cells']} | {v['ticks']} / {v['answer_ticks']} | {v['hidden_units']} | {ok(v)} | "
                   f"{ok(g.get('tick0'))} | {reach['ncol']} | {reach['ports']} | {minp:.9f} |")
    return "\n".join(out)


def situations_table():
    rec = json.loads((HERE / "records" / "situations.json").read_text())
    out = ["| trained at | measured at | cells meeting a new 3x3 kind pattern | a new full neighbourhood | new "
           "inputs of the compiled rule |", "|---|---|---|---|---|"]
    for size, v in rec["bigger"].items():
        out.append(f"| (3, 2, 1) | ({size.replace(',', ', ')}) | {v['kinds']['cells_new']:.1%} | {v['full']['cells_new']:.1%} | "
                   f"{v['relevant']['cells_new']:.1%} |")
    for base in ("3,2,2", "3,3,2"):
        for size, v in rec[f"from_{base}"].items():
            out.append(f"| ({base.replace(',', ', ')}) | ({size.replace(',', ', ')}) | {v['kinds']:.1%} | {v['full']:.1%} | "
                       f"{v['relevant']:.1%} |")
    for size, v in rec.get("square_from_3,3", {}).items():
        out.append(f"| the square at (3, 3) | the square at ({size.replace(',', ', ')}) | {v['kinds']:.1%} | {v['full']:.1%} | "
                   f"{v['relevant']:.1%} |")
    return "\n".join(out)


def hyper_table(rows):
    """The generated wiring's runs: certified / runs, the shares, and what the wiring settled on (the median share
    of cells whose head reads the compiled source, over the machine's 7 ports; the bypass share)."""
    by = defaultdict(list)
    for r in rows:
        if r["cfg"]["tag"] in HYPER:
            by[r["cfg"]["tag"]].append(r)
    out = ["| wiring | lessons | certified | training, snapped / soft | bigger, snapped | ports right | bypass | "
           "sub table |", "|---|---|---|---|---|---|---|---|"]
    for t in [t for t in HYPER if t in by]:
        rs = by[t]
        cert = sum(bool(r.get("certified")) for r in rs)
        hits = [r["hit"] for r in rs if r.get("hit")]
        fit = sum(r["train_hard"] == 1.0 for r in rs)
        c = f"{cert}/{len(rs)}" + (f" (fit {fit})" if fit and not cert else "")
        tr = f"{f2(med([r['train_hard'] for r in rs]))} / {f2(med([r['train_soft'] for r in rs]))}"
        big = " / ".join(f2(med([r.get("bigger", {}).get(k) for r in rs])) for k in rs[0].get("bigger", {}))
        wr = [r["probe"].get("wiring", {}) for r in rs]
        ports = [med([v["right"] for v in w.values() if isinstance(v, dict) and "right" in v]) for w in wr]
        pr = "·" if all(p is None for p in ports) else f2(med(ports))
        byp = "·" if all("bypass" not in w for w in wr) else f2(med([w.get("bypass") for w in wr]))
        p = [r["probe"] for r in rs]
        sub = f"{int(med([x['sub_digits_right'] for x in p]))} + {int(med([x['sub_borrows_right'] for x in p]))}"
        out.append(f"| {HYPER[t][0]} | {HYPER[t][1]} | {c} | {tr} | {big} | {pr} | {byp} | {sub} |")
    return "\n".join(out)


def hyper_stab_table(rows):
    by = defaultdict(list)
    for r in rows:
        if r["cfg"]["tag"].startswith("hstab_"):
            by[r["cfg"]["tag"]].append(r)
    out = ["| lessons | noise sigma | exact at the start, snapped / soft | exact after 1,000 steps, snapped / soft | "
           "certified | ports right at the end |", "|---|---|---|---|---|---|"]
    for t in sorted(by, key=lambda t: (t.split("_")[1], float(t.split("_s")[1]))):
        rs = by[t]
        less, sig = t.split("_")[1], t.split("_s")[1]
        st = f"{f2(med([r['start_hard'] for r in rs]))} / {f2(med([r['start_soft'] for r in rs]))}"
        en = f"{f2(med([r['train_hard'] for r in rs]))} / {f2(med([r['train_soft'] for r in rs]))}"
        wr = [r["probe"].get("wiring", {}) for r in rs]
        ports = [med([v["right"] for v in w.values() if isinstance(v, dict) and "right" in v]) for w in wr]
        out.append(f"| {'answers' if less == 'A' else 'GCL'} | {sig} | {st} | {en} | "
                   f"{sum(bool(r.get('certified')) for r in rs)}/{len(rs)} | {f2(med(ports))} |")
    return "\n".join(out)


def hyper_swaps_table():
    p = HERE / "runs" / "hyper_diag.jsonl"
    rows = [json.loads(l) for l in open(p)] if p.exists() else []
    sw = next((r for r in rows if r["name"] == "hyper_swaps"), None)
    pulls = [r for r in rows if r["name"].startswith("hyper_pull")]
    out = ["| wirings | how many | right on every training answer | median share right | best share right |",
           "|---|---|---|---|---|"]
    if sw:
        t = sw["true"]
        out.append(f"| the compiled wiring | 1 | {int(t['hard'] == 1.0)} | {t['hard']:.2f} | {t['hard']:.2f} |")
        for key, lab in (("one", "one port changed (to another of the 7 rules)"), ("two", "two ports changed"),
                         ("random", "every port drawn at random")):
            v = sw[key]
            h = [x["hard"] for x in v]
            out.append(f"| {lab} | {len(v)} | {sum(x == 1.0 for x in h)} | {np.median(h):.2f} | {max(h):.2f} |")
    if pulls:
        vals = defaultdict(list)
        for r in pulls:
            for k, v in r["pull"].items():
                vals[k].append(v)
        out.append("")
        import textwrap
        out += textwrap.wrap("First pull toward a prompt cell (share of cells and ports, random initializations): " +
                             "; ".join(f"{k}: {min(v):.2f}–{max(v):.2f} over {len(v)} seeds" for k, v in vals.items())
                             + ".", 120)
    return "\n".join(out)


TRIM_ARMS = {"ctl": "plain Adam (the control)", "wd0.1": "AdamW, weight decay 0.1", "wd0.3": "AdamW, weight decay 0.3",
             "wd1": "AdamW, weight decay 1", "and": "AND-mask over 8 groups", "dbd": "delta-bar-delta speeds",
             "and_wd0.3": "AND-mask + weight decay 0.3", "dbd_wd0.3": "delta-bar-delta + weight decay 0.3",
             "and_cols": "AND-mask over the 4 columns",
             "and_cols_wd0.3": "AND-mask over the columns + weight decay 0.3"}
REST_ARMS = {"eps": "Adam, eps 1e-4", "eps_wd0.3": "AdamW, eps 1e-4, weight decay 0.3",
             "sgd0.01": "plain SGD, learning rate 0.01", "sgd0.01_wd": "plain SGD 0.01 + weight decay",
             "sgd0.1": "plain SGD, learning rate 0.1", "sgd0.1_wd": "plain SGD 0.1 + weight decay"}


def trim_rows(suites=("trim1", "trim_cols")):
    return [json.loads(l) for s in suites if (HERE / "runs" / f"{s}.jsonl").exists()
            for l in open(HERE / "runs" / f"{s}.jsonl")]


def trim_table(suites=("trim1", "trim_cols"), arms=TRIM_ARMS):
    """trim1 (§9; rest1, §10), one row per size and arm, medians over seeds: training share snapped / soft at the end
    and its best over the run; one size up at the end and the best any curve point reached (a late jump would show
    there); the weight norm at the first and last curve points; the sub table and the plumbing probe at the end."""
    by = defaultdict(list)
    for r in trim_rows(suites):
        by[r["cfg"]["tag"]].append(r)
    out = ["| trained at | arm | certified | training, snapped / soft | best training, snapped | one size up (k + 1 / "
           "w + 1, k + 1) | its best over the run | weight norm | sub table | plumbing: N's digit / old register / "
           "final borrow |", "|---|---|---|---|---|---|---|---|---|---|"]
    for size in ("w3", "w2"):
        for arm, lab in arms.items():
            rs = by.get(f"{size}_{arm}")
            if not rs:
                continue
            c = rs[0]["cfg"]
            where = f"({c['B']}, {c['w']}, {c['k']}), {rs[0]['steps']:,} steps"
            cert = f"{sum(bool(r.get('certified')) for r in rs)}/{len(rs)}"
            tr = f"{f2(med([r['train_hard'] for r in rs]))} / {f2(med([r['train_soft'] for r in rs]))}"
            best = f2(med([max(x["exact"] for x in r["log"]) for r in rs]))
            up = [k for k in rs[0]["curve"][0] if k.count(",") == 2]
            end = " / ".join(f2(med([r["bigger"][k] for r in rs])) for k in rs[0]["bigger"])
            peak = " / ".join(f2(med([max(x[k] for x in r["curve"]) for r in rs])) for k in up)
            wn = f"{med([r['curve'][0]['wnorm'] for r in rs]):.0f} → {med([r['curve'][-1]['wnorm'] for r in rs]):.0f}"
            p = [r["probe"] for r in rs]
            sub = f"{int(med([x['sub_digits_right'] for x in p]))} + {int(med([x['sub_borrows_right'] for x in p]))}"
            pl = " / ".join(f2(med([r["curve"][-1]["plumbing"][k] for r in rs]))
                            for k in ("n_digit", "old_register", "final_borrow"))
            out.append(f"| {where} | {lab} | {cert} | {tr} | {best} | {end} | {peak} | {wn} | {sub} | {pl} |")
    return "\n".join(out)


def trim_curve_table(suites=("trim1", "trim_cols"), arms=TRIM_ARMS):
    """trim1 (§9; rest1, §10) over training, medians over seeds at a few curve points: training share snapped / soft,
    and one size up ((B, w + 1, k + 1), on the curve's 500-question sample)."""
    by = defaultdict(list)
    for r in trim_rows(suites):
        by[r["cfg"]["tag"]].append(r)
    out = []
    for size, points in (("w3", (1000, 5000, 10000, 20000)), ("w2", (1000, 10000, 25000, 50000))):
        rows = [(arm, lab, by[f"{size}_{arm}"]) for arm, lab in arms.items() if by.get(f"{size}_{arm}")]
        if not rows:
            continue
        c = rows[0][2][0]["cfg"]
        up = f"{c['B']},{c['w'] + 1},{c['k'] + 1}"
        if out:
            out.append("")
        out.append(f"| trained at ({c['B']}, {c['w']}, {c['k']}); one size up is ({up.replace(',', ', ')}) | " +
                   " | ".join(f"step {p:,}" for p in points) + " |")
        out.append("|---|" + "---|" * len(points))
        for arm, lab, rs in rows:
            cells = []
            for p in points:
                at = [next((x for x in r["curve"] if x["step"] == p), None) for r in rs]
                at = [x for x in at if x]
                cells.append("·" if not at else f"{f2(med([x['train_hard'] for x in at]))} / "
                             f"{f2(med([x['train_soft'] for x in at]))} / {f2(med([x[up] for x in at]))}")
            seeds = f" ({len(rs)} seed{'s' if len(rs) > 1 else ''})" if len(rs) < 4 else ""      # 4 per arm, if all ran
            out.append(f"| {lab}{seeds} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def cut_table(which="rest", arms=REST_ARMS):
    """records/<which>_cut.json (§10): the runs that stopped short without a record, one row each: the step reached;
    at its last curve point the training share snapped / soft, one size up (k + 1 / w + 1, k + 1) and the weight norm;
    and the best one size up any of its curve points reached."""
    p = HERE / "records" / f"{which}_cut.json"
    if not p.exists():
        return ""
    arm = lambda pk: pk["tag"].split("_", 1)[1]
    out = ["| arm | seed | stopped at step | at its last curve point: training, snapped / soft | one size up (k + 1 / "
           "w + 1, k + 1) | its best over the run | weight norm |", "|---|---|---|---|---|---|---|"]
    runs = sorted(json.loads(p.read_text()).items(), key=lambda kv: (list(arms).index(arm(kv[1])), kv[0]))
    for name, pk in runs:
        last, up = pk["curve"][-1], [k for k in pk["curve"][0] if k.count(",") == 2]
        out.append(f"| {arms[arm(pk)]} | {name.rsplit('_s', 1)[1]} | {pk['step']:,} | {f2(last['train_hard'])} / "
                   f"{f2(last['train_soft'])} | {' / '.join(f2(last[k]) for k in up)} | "
                   f"{' / '.join(f2(max(x[k] for x in pk['curve'])) for k in up)} | {last['wnorm']:.0f} |")
    return "\n".join(out)


def trim_stab_table(suite="trim_stab", rows=tuple((f"w2_{a}", lab) for a, lab in TRIM_ARMS.items())):
    """trim_stab (§9; rest_stab, §10): each arm started at the compiled rule, one row per (tag, label) in `rows`. Per
    arm, over its seeds: the step at which the training share first fell below 1, snapped (the log, every 100 steps)
    and soft (the curve, every 500); the shares at the end; the weight norm at the first and last curve points; one
    size up and certification at the end."""
    p = HERE / "runs" / f"{suite}.jsonl"
    by = defaultdict(list)
    for l in (open(p) if p.exists() else []):
        r = json.loads(l)
        by[r["cfg"]["tag"]].append(r)
    first = lambda xs, key: next((f"{x['step']:,}" for x in xs if x[key] < 1.0), "never")
    each = lambda xs: " and ".join(xs) if len(xs) < 3 else ", ".join(xs[:-1]) + " and " + xs[-1]
    out = ["| arm | first step below exact, snapped (each seed) | the same, soft | training at the end, snapped / soft "
           "| weight norm | one size up at the end (k + 1 / w + 1, k + 1) | certified | sub table |",
           "|---|---|---|---|---|---|---|---|"]
    for tag, lab in rows:
        rs = sorted(by.get(tag, []), key=lambda r: r["cfg"]["seed"])
        if not rs:
            continue
        hard = each([first(r["log"], "exact") for r in rs])
        soft = each([first(r["curve"], "train_soft") for r in rs])
        tr = f"{f2(med([r['train_hard'] for r in rs]))} / {f2(med([r['train_soft'] for r in rs]))}"
        wn = f"{med([r['curve'][0]['wnorm'] for r in rs]):.0f} → {med([r['curve'][-1]['wnorm'] for r in rs]):.0f}"
        up = " / ".join(f2(med([r["bigger"][k] for r in rs])) for k in rs[0]["bigger"])
        pr = [r["probe"] for r in rs]
        sub = f"{int(med([x['sub_digits_right'] for x in pr]))} + {int(med([x['sub_borrows_right'] for x in pr]))}"
        out.append(f"| {lab} | {hard} | {soft} | {tr} | {wn} | {up} | "
                   f"{sum(bool(r.get('certified')) for r in rs)}/{len(rs)} | {sub} |")
    return "\n".join(out)


def rest_lr_table():
    """rest_lr (§10): plain SGD at each learning rate, one seed. From random: the training share snapped / soft at
    steps 1,000 and 5,000, and the weight norm at 5,000. From the compiled rule (output scale 4): the first step below
    exact, snapped (the log, every 100 steps) and soft (the curve, every 500), and the shares at step 10,000."""
    p = HERE / "runs" / "rest_lr.jsonl"
    rows = [json.loads(l) for l in open(p)] if p.exists() else []
    rand = {r["cfg"]["lr"]: r for r in rows if r["cfg"]["tag"] == "rand_sgd"}
    comp = {r["cfg"]["lr"]: r for r in rows if r["cfg"]["tag"] == "b4_sgd"}
    at = lambda r, s: next((x for x in r["curve"] if x["step"] == s), None) if r else None
    pair = lambda x: "·" if x is None else f"{f2(x['train_hard'])} / {f2(x['train_soft'])}"
    first = lambda xs, key: next((f"{x['step']:,}" for x in xs if x[key] < 1.0), "never")
    out = ["| learning rate | from random: training at step 1,000, snapped / soft | at step 5,000 | weight norm at "
           "step 5,000 | from the compiled rule: first step below exact, snapped / soft | training at step 10,000 |",
           "|---|---|---|---|---|---|"]
    for lr in sorted(set(rand) | set(comp)):
        a, b = rand.get(lr), comp.get(lr)
        norm = f"{at(a, 5000)['wnorm']:.0f}" if at(a, 5000) else "·"
        below = f"{first(b['log'], 'exact')} / {first(b['curve'], 'train_soft')}" if b else "·"
        out.append(f"| {lr:g} | {pair(at(a, 1000))} | {pair(at(a, 5000))} | {norm} | {below} | "
                   f"{pair(at(b, 10000))} |")
    return "\n".join(out)


def rest_stab_table():
    """rest_stab (§10): rest1's arms started at the compiled rule at output scales 4 and 20 (at 20 also plain Adam and
    AdamW 0.3), as in trim_stab_table."""
    return trim_stab_table("rest_stab", [(f"b4_{a}", f"scale 4: {lab}") for a, lab in REST_ARMS.items()] +
                           [(f"b20_{a}", f"scale 20: {lab}") for a, lab in
                            {"ctl": TRIM_ARMS["ctl"], "wd0.3": TRIM_ARMS["wd0.3"], **REST_ARMS}.items()])


REST3_ARMS = {"ctl": TRIM_ARMS["ctl"], "eps": REST_ARMS["eps"], "sgd0.003": "plain SGD, learning rate 0.003",
              "sgd0.01": "plain SGD, learning rate 0.01"}


def arm_label(c):
    """An optimizer's name from a run's cfg (the arms of §10)."""
    if c["optim"] == "sgd":
        return f"plain SGD, learning rate {c['lr']:g}" + (" + weight decay" if c["wd"] else "")
    eps = f", eps {c['eps']:.0e}".replace("e-0", "e-") if c["eps"] != 1e-8 else ""          # 1e-4, as REST_ARMS says
    if c["optim"] == "adamw":
        return f"AdamW{eps}, weight decay {c['wd']:g}"
    return f"Adam{eps}" if eps else "plain Adam"


def rest_up_table(points=(2500, 5000, 7500, 10000)):
    """rest_lr, rest_stab and rest_stab3 (§10): the compiled starts that ended with every training answer right on the
    snapped path, one row per size, output scale and arm: one size up (k + 1 / w + 1, k + 1, on the curve's sample
    of 500 questions) at a few curve points, medians over the seeds."""
    by = defaultdict(list)
    for suite in ("rest_stab", "rest_stab3", "rest_lr"):          # rest_lr only where the others ran nothing
        p = HERE / "runs" / f"{suite}.jsonl"
        got = defaultdict(list)
        for r in map(json.loads, open(p) if p.exists() else []):
            c = r["cfg"]
            if c["init"] == "compiled":
                got[(c["w"], c["init_beta"], arm_label(c))].append(r)
        by.update({key: rs for key, rs in got.items() if key not in by})
    out = ["| started at the compiled rule | seeds | " + " | ".join(f"step {p:,}" for p in points) + " |",
           "|---|---|" + "---|" * len(points)]
    for (w, beta, lab), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[1][0]["cfg"]["lr"])):
        if not all(r["train_hard"] == 1.0 for r in rs):
            continue
        c = rs[0]["cfg"]
        ups = [f"{c['B']},{c['w']},{c['k'] + 1}", f"{c['B']},{c['w'] + 1},{c['k'] + 1}"]
        cells = []
        for p in points:
            at = [x for x in (next((x for x in r["curve"] if x["step"] == p), None) for r in rs) if x]
            cells.append("·" if not at else " / ".join(f2(med([x[u] for x in at])) for u in ups))
        out.append(f"| ({c['B']}, {c['w']}, {c['k']}), output scale {beta:g}: {lab} | {len(rs)} | " +
                   " | ".join(cells) + " |")
    return "\n".join(out)


def erosion_table():
    """records/erosion.json (§10): where a rule that kept its training answers goes wrong one size up. Per record and
    size: the answers right run freely; the share of the machine's cell-ticks whose situation never occurs at the
    training size; the trained rule's one-tick errors in situations met there and never met (count and rate)."""
    p = HERE / "records" / "erosion.json"
    if not p.exists():
        return ""
    recs = {f"{s}/{r['name']}": r for s in ("rest_stab", "rest_stab3")
            if (HERE / "runs" / f"{s}.jsonl").exists() for r in map(json.loads, open(HERE / "runs" / f"{s}.jsonl"))}
    pct = lambda n, of: f"{n:,} ({100 * n / of:.2g}%)" if of else "·"
    out = ["| rule | measured at | answers right | cell-ticks in situations never met at the training size | one-tick "
           "errors in situations met there (rate) | in situations never met (rate) |", "|---|---|---|---|---|---|"]
    for name, e in json.loads(p.read_text()).items():
        c, t = recs[name]["cfg"], e["train"]
        lab = f"{arm_label(c)}, {e['steps']:,} steps from the compiled rule (output scale {c['init_beta']:g})"
        out.append(f"| {lab} | ({', '.join(map(str, t['size']))}), the training size, {t['questions']:,} questions | "
                   f"{f2(t['right'])} | · | {pct(t['errors'], t['cell_ticks'])} | · |")
        for size, x in e.items():
            if size in ("steps", "train"):
                continue
            q = f"every one of {x['questions']:,} questions" if x["every"] else f"a sample of {x['questions']:,}"
            nn = round(x["never_met"] * x["cell_ticks"])
            out.append(f"| | ({size.replace(',', ', ')}), {q} | {f2(x['right'])} | {x['never_met']:.0%} | "
                       f"{pct(x['errors'] - x['errors_never_met'], x['cell_ticks'] - nn)} | "
                       f"{pct(x['errors_never_met'], nn)} |")
    return "\n".join(out)


def sharp_table(suites=("trim1", "trim_cols"), arms=TRIM_ARMS):
    """records/sharpness.json (§9; rest1, §10): the compiled rule at output scales 4 and 20, then every arm of the
    suites at (3, 2, 1) (medians over its seeds): the weight norm, the training share snapped / soft, and per channel
    the median of a state's largest probability over every tick and written cell (the share above 0.9)."""
    p = HERE / "records" / "sharpness.json"
    if not p.exists():
        return ""
    rec = json.loads(p.read_text())
    chans = list(next(iter(rec["runs"].values()))["sharpness"]) if rec["runs"] else ["d", "n", "b", "m", "g"]
    out = ["| rule | weight norm | training, snapped / soft | " + " | ".join(chans) + " |",
           "|---|---|---|" + "---|" * len(chans)]
    for beta in ("4", "20"):
        c = rec["compiled"][beta]
        out.append(f"| the compiled rule, output scale {beta} | {c['norm']:.0f} | {f2(c['exact 3,2,1'][0])} / "
                   f"{f2(c['exact 3,2,1'][1])} | " +
                   " | ".join(f"{m:.2f} ({s:.0%})" for m, s in c["sharpness"].values()) + " |")
    by = defaultdict(list)
    for name, r in rec["runs"].items():
        suite, run = name.split("/")
        if suite in suites:
            by[run.rsplit("_s", 1)[0].removeprefix("w2_")].append(r)
    for arm, lab in arms.items():
        rs = by.get(arm)
        if not rs:
            continue
        cells = [f"{med([r['sharpness'][c][0] for r in rs]):.2f} ({med([r['sharpness'][c][1] for r in rs]):.0%})"
                 for c in chans]
        out.append(f"| {lab} ({len(rs)} seed{'s' if len(rs) > 1 else ''}) | {med([r['norm'] for r in rs]):.0f} | "
                   f"{f2(med([r['exact'][0] for r in rs]))} / {f2(med([r['exact'][1] for r in rs]))} | " +
                   " | ".join(cells) + " |")
    return "\n".join(out)


def verify_table():
    """The certified setting's saved rules (cover2, cover3) checked further out (records/verify.json): right / asked,
    on every question where marked, else a sample of 3,000."""
    rec = json.loads((HERE / "records" / "verify.json").read_text())
    names = sorted(rec, key=lambda n: int(n.rsplit("_s", 1)[1]))
    conds = list(next(iter(rec.values())))
    seeds = [n.rsplit("_s", 1)[1] for n in names]
    out = ["| size, start, ticks | " + " | ".join(f"seed {x}" for x in seeds) + " |", "|---|" + "---|" * len(seeds)]
    for c in conds:
        size, start, mult = c.split()
        lab = f"({size.replace(',', ', ')}), {start}, {mult.replace('x', '')}× ticks"
        cells = []
        for n in names:
            v = rec[n].get(c)
            if v is None:
                cells.append("·")
                continue
            q = f"{v['questions']:,}" + ("" if v["every"] else " sampled")
            cells.append(f"all {q}" if v["exact"] else f"{v['right']:,} of {q}")
        out.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    rows = load()
    doc = (HERE / "NCA.md").read_text()
    tables = {"COMPILE": compile_table, "SITUATIONS": situations_table, "VERIFY": verify_table,
              "HYPER": lambda: hyper_table(rows), "HYPER_STAB": lambda: hyper_stab_table(rows),
              "HYPER_SWAPS": hyper_swaps_table, "TRIM": trim_table, "TRIM_CURVE": trim_curve_table,
              "TRIM_STAB": trim_stab_table, "SHARP": sharp_table, "REST_LR": rest_lr_table,
              "REST": lambda: trim_table(("rest1",), REST_ARMS), "REST_STAB": rest_stab_table,
              "REST_CURVE": lambda: trim_curve_table(("rest1",), REST_ARMS),
              "REST_SHARP": lambda: sharp_table(("rest1",), REST_ARMS), "REST_UP": rest_up_table,
              "EROSION": erosion_table, "REST_CUT": cut_table,
              "REST_STAB3": lambda: trim_stab_table("rest_stab3", [(f"w3_{a}", lab) for a, lab in REST3_ARMS.items()]),
              **{t: (lambda t=t: run_table(rows, t)) for t in ("LESSONS", "TRACE", "SQUARE", "LOGIC")}}
    for name, fn in tables.items():
        pat = re.compile(rf"(<!-- {name} -->\n).*?(<!-- /{name} -->)", re.S)
        if pat.search(doc):
            doc = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(2), doc)
    (HERE / "NCA.md").write_text(doc)
    print("filled", ", ".join(tables))


if __name__ == "__main__":
    main()
