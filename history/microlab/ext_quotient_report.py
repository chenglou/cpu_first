"""Tables for FINDINGS_quotient.md from microlab/results_quotient.jsonl, with ext_matched's longdiv and modq rows
(ladder_matched_results.jsonl) for reference.

    python -m microlab.ext_quotient_report                  # the main table
    python -m microlab.ext_quotient_report --steps          # the choice per division step
    python -m microlab.ext_quotient_report --findings       # the two tables of FINDINGS_quotient.md

Per variant x bits x steps, over 2 seeds (ranges are min–max over seeds):
- exact: the plain runs (all moduli trained);
- held-out moduli: the holdout=4 runs, exact on the quarter of moduli never trained (trained moduli in brackets);
- one bit wider: 6 -> 7 bits (the `wide` entry) and 7 -> 8 bits (ood, same digit width);
- one digit wider: the first size with one more digit (6 -> 7, 7 -> 10, 10 -> 14 bits), on inputs with a digit past
  the trained width;
- every choice right: the share of test examples whose quotient digits are all chosen right (slots mapped to digits),
  and exact among those (what the subtraction still gets wrong).
"""

from __future__ import annotations

import argparse
import json
import os

from .ext_matched import RESULTS as MATCHED
from .ext_quotient import RESULTS

ORDER = ["longdiv", "longdiv_tr", "longdiv_est", "longdiv_estc", "modq"]


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def span(xs, fmt="{:.2f}"):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return "·"
    lo, hi = fmt.format(xs[0]), fmt.format(xs[-1])
    return lo if lo == hi else f"{lo}–{hi}"


def label(r):
    if r["task"] == "modq":
        return "modq"
    kw = {k: v for k, v in r["model_kw"].items() if k != "task"}
    return (r["model"] + "".join(f" {k}={v}" for k, v in sorted(kw.items()))
            + (f" trained on {r['train_task']}" if r.get("train_task") else ""))


def wider1(r):
    if r.get("train_task") and "choice" not in r:   # ext_matched scored ood on the train task, not on mod
        return None
    if r.get("ood_exact") is not None:
        return r["ood_exact"]
    w = (r.get("wide") or {}).get(str(r["size"] + 1))
    return None if w is None else w["all"]


def digit1(r):
    """Exact one digit wider than trained, on inputs with a digit past the trained width (6 -> 7 bits, 7 -> 10)."""
    w = r.get("wide") or {}
    return None if not w else w[min(w, key=int)]["wider"]


def kept(r):
    """Exact on the trained moduli of a holdout run."""
    f, h = r.get("heldout_frac"), r.get("exact_heldout_moduli")
    return None if f is None or h is None or f >= 1 else (r["exact"] - f * h) / (1 - f)


def rows():
    ref = [r for r in load(MATCHED) if r["model"] == "longdiv" and r["task"] in ("mod", "modq")
           and r.get("train_task") in (None, "modmix") and r["tag"] in ("M3k", "M12k", "Mrem", "Mho")
           and r["size"] in (6, 7)]
    seen, out = set(), []
    for r in ref:                                   # Mrem repeats some M12k cells
        k = (r["task"], r.get("train_task"), r["size"], r["steps"], r["seed"], r.get("holdout", 0))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out + load(RESULTS)


def table(rs):
    g = {}
    for r in rs:
        g.setdefault((label(r), r["size"], r["steps"]), {}).setdefault(r.get("holdout", 0), []).append(r)
    order = lambda k: (ORDER.index(k[0].split()[0]) if k[0].split()[0] in ORDER else 99, k[0], k[1], k[2])
    lines = ["| variant | bits | steps | exact | held-out moduli (trained moduli) | one bit wider | one digit wider "
             "| every choice right | exact when every choice right |", "|---|---|---|---|---|---|---|---|---|"]
    for (lab, size, steps) in sorted(g, key=order):
        by = g[(lab, size, steps)]
        main, ho = by.get(0, []), by.get(4, [])
        ch = [r["choice"] for r in main if r.get("choice")]
        hos = span([r.get("exact_heldout_moduli") for r in ho])
        if ho:
            hos += f" ({span([kept(r) for r in ho])})"
        lines.append(f"| {lab} | {size} | {steps // 1000}k | {span([r['exact'] for r in main])} | {hos} | "
                     f"{span([wider1(r) for r in main])} | {span([digit1(r) for r in main])} | "
                     f"{span([c['choice_mapped']['all'] for c in ch])} | "
                     f"{span([c['choice_mapped']['exact_if_all'] for c in ch])} |")
    return "\n".join(lines)


NAMES = ["longdiv, borrow choice (LADDER_MATCHED)", "longdiv, borrow choice (rerun)", "estimator",
         "estimator + borrow check", "quotient digits given (modq)"]


def name(r):
    """Plain variant names for FINDINGS_quotient.md; None for the rows it leaves out (the frozen fitted MLP cell)."""
    kw = {k: v for k, v in r["model_kw"].items() if k != "task"}
    if kw.get("fit_cell"):
        return None
    if r["task"] == "modq":
        return NAMES[4]
    out = dict(zip(["longdiv", "longdiv_tr", "longdiv_est", "longdiv_estc"], NAMES))[r["model"]]
    if kw.get("kr", 3) != 3 or kw.get("kn", 2) != 2:
        out += f", window {kw.get('kr', 3)}/{kw.get('kn', 2)}"
    if kw.get("est_bias0"):
        out += ", starts at q = 0"
    if r.get("train_task"):
        out += f", trained on {r['train_task']}"
    return out


def findings(rs):
    """The two tables of FINDINGS_quotient.md: everything learned from answers, and the exact digit cell."""
    out = []
    for exact_cell in (False, True):
        g = {}
        for r in rs:
            n = name(r)
            if n and bool(r["model_kw"].get("soft_cell")) == exact_cell:
                g.setdefault((n, r["size"], r["steps"]), {}).setdefault(r.get("holdout", 0), []).append(r)
        mods = ["window", "starts", "trained"]
        base = lambda n: max((b for b in NAMES if n.startswith(b)), key=len)
        key = lambda k: (NAMES.index(base(k[0])),
                         [next((i for i, m in enumerate(mods) if p.startswith(m)), 9)
                          for p in k[0][len(base(k[0])):].split(", ")[1:]], k[1], k[2])
        out += ["| variant | bits | steps | exact | held-out moduli (trained) | one bit wider | one digit wider | "
                "every choice right |", "|---|---|---|---|---|---|---|---|"]
        for k in sorted(g, key=key):
            main, ho = g[k].get(0, []), g[k].get(4, [])
            hos = span([r.get("exact_heldout_moduli") for r in ho]) + (f" ({span([kept(r) for r in ho])})" if ho else "")
            ch = [r["choice"]["choice_mapped"]["all"] for r in main if r.get("choice")]
            out.append(f"| {k[0]} | {k[1]} | {k[2] // 1000}k | {span([r['exact'] for r in main])} | {hos} | "
                       f"{span([wider1(r) for r in main])} | {span([digit1(r) for r in main])} | {span(ch)} |")
        out.append("")
    return "\n".join(out)


def steps_table(rs):
    lines = []
    for r in sorted((r for r in rs if r.get("choice") and not r.get("holdout")),
                    key=lambda r: (label(r), r["size"], r["steps"], r["seed"])):
        c = r["choice"]
        lines.append(f"{label(r):28} {r['size']} {r['steps'] // 1000:>2}k s{r['seed']}  exact {r['exact']:.2f}  "
                     f"choice raw {c['choice_raw']['all']:.2f} mapped {c['choice_mapped']['all']:.2f} "
                     f"[{' '.join(f'{a:.2f}' for a in c['choice_mapped']['per_step'])}]  "
                     f"est raw {c['est_raw']['all']:.2f} mapped {c['est_mapped']['all']:.2f}  map {c['choice_map']}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", action="store_true")
    p.add_argument("--findings", action="store_true")
    a = p.parse_args()
    rs = rows()
    print(steps_table(rs) if a.steps else findings(rs) if a.findings else table(rs))


if __name__ == "__main__":
    main()
