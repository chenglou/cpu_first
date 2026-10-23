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
    "k2_GCL": ("LESSONS", "noself, trained at (3, 2, 2)", "GCL + the step law"),
    "k2_GCL_ste": ("LESSONS", "noself, STE, trained at (3, 2, 2)", "GCL + the step law"),
    "k2_sheet": ("LESSONS", "noself, trained at (3, 2, 2)", "answers + the worksheet's visible digits"),
    "ncol_GCL": ("LESSONS", "noself + N's column", "GCL"),
    "ports_A": ("LESSONS", "noself + the bridge's ports, digit-and-borrow cells", "answers"),
    "ports_GCL": ("LESSONS", "noself + the bridge's ports, digit-and-borrow cells", "GCL"),
    "ports_GCL_ste": ("LESSONS", "the same, STE", "GCL"),
    "ports_GCL_tf": ("LESSONS", "noself + the bridge's ports, digit-and-borrow cells", "GCL, the cell's lessons first"),
    "k2_ports_GCL": ("LESSONS", "the same, trained at (3, 2, 2)", "GCL + the step law"),
    "causal_A": ("LESSONS", "causal", "answers"),
    "causal_GCL": ("LESSONS", "causal", "GCL"),
    "causal_GCL_ste": ("LESSONS", "causal, STE", "GCL"),
    "causal_k2_GCL": ("LESSONS", "causal, trained at (3, 2, 2)", "GCL + the step law"),
    "w3_causal_GCL": ("LESSONS", "causal, trained at (3, 3, 2)", "GCL + the step law"),
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
              "HYPER_SWAPS": hyper_swaps_table,
              **{t: (lambda t=t: run_table(rows, t)) for t in ("LESSONS", "TRACE", "SQUARE", "LOGIC")}}
    for name, fn in tables.items():
        pat = re.compile(rf"(<!-- {name} -->\n).*?(<!-- /{name} -->)", re.S)
        if pat.search(doc):
            doc = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(2), doc)
    (HERE / "NCA.md").write_text(doc)
    print("filled", ", ".join(tables))


if __name__ == "__main__":
    main()
