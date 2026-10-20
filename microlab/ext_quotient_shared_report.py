"""Tables for FINDINGS_quotient_shared.md from microlab/results_quotient_shared.jsonl.

    python -m microlab.ext_quotient_shared_report                # every table
    python -m microlab.ext_quotient_shared_report --tags S1,S1L  # only these tags

Ranges are min–max over seeds. "Solved" counts runs exact (>= 0.99) with their own borrow choice after the switch.
- main: the plain runs (holdout 0): the estimator phase (exact with its own choices / given the true digits), then
  after the switch to the own borrow (exact / given / hard datapath), the steps after the switch until the validation
  sample was exact, and the subtractor's 200 one-hot rows;
- held-out moduli: the holdout=4 runs, exact on the quarter of moduli never trained (trained moduli in brackets);
- widths: the same weights at 6, 7, 10, 14 and 20 bits, own borrow (given the true digits in brackets);
- rows: which subtractor rows are wrong; maps: the slot maps (slot i stands mostly for digit map[i]).
"""

from __future__ import annotations

import argparse
import json
import os

from .ext_quotient_shared import RESULTS

PASS = 0.99


def load(path=RESULTS, tags=None):
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    return [r for r in rows if tags is None or r["tag"] in tags]


def span(xs, fmt="{:.2f}"):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return "·"
    lo, hi = fmt.format(xs[0]), fmt.format(xs[-1])
    return lo if lo == hi else f"{lo}–{hi}"


def label(r):
    out = "shared subtractor" if r["slots"] == "shared" else "learned slots (longdiv's cell)"
    if (r["kr"], r["kn"]) != (3, 2):
        out += f", window {r['kr']}/{r['kn']}"
    if not r["staged"]:
        out += ", not staged"
    if r["first"] != "est":
        out += ", own borrow from the start"
    if r["start"] != "tight":
        out += ", always from the top w digits"
    if r.get("hold"):
        out += f", choice held uniform for {r['hold']} steps"
    if r.get("handover"):
        out += f", handover over {r['handover']} steps"
    if r.get("lr_b") and r["lr_b"] != r["lr"]:
        out += f", lr {r['lr_b']:g} after the switch"
    if r.get("from_a"):
        out += ", phase B only, from " + r["from_a"].split("/")[-2] + " weights"
    if r.get("hard_b") is not None and r.get("hard_b") != r.get("hard_borrow"):
        out += f", phase B hard_borrow={r['hard_b']}"
    if r.get("freeze_digits"):
        out += f", digits held out of the lesson for {r['freeze_digits']} steps"
    if r.get("hard_borrow") == "all":
        out += ", every borrow a bit"
    elif r.get("hard_borrow"):
        out += ", borrow a bit between places"
    return out


def groups(rows):
    out = {}
    for r in rows:
        out.setdefault((label(r), r["bits"]), []).append(r)
    return sorted(out.items(), key=lambda kv: (kv[0][0] != "shared subtractor", kv[0][0], kv[0][1]))


def median(xs):
    xs = sorted(xs)
    return None if not xs else xs[len(xs) // 2]


def main_table(rows):
    print("| variant | bits | solved | estimator phase: exact / given the true digits | after the switch: exact / "
          "given / hard | steps after the switch to exact (median) | subtractor rows right (of 200) |")
    print("|---|---|---|---|---|---|---|")
    for (lab, bits), rs in groups([r for r in rows if not r["holdout"]]):
        a, b = [r["after_a"] for r in rs], [r["after_b"] for r in rs]
        solved = sum(x["exact"] >= PASS for x in b)
        hits = [r["hit_b"] for r in rs if r["hit_b"] is not None]
        tab = span([x["table"]["right"] for x in b if "table" in x], "{:.3f}") if "table" in b[0] else "·"
        print(f"| {lab} | {bits} | {solved}/{len(rs)} | {span([x['exact'] for x in a])} / "
              f"{span([x['exact_given'] for x in a])} | {span([x['exact'] for x in b])} / "
              f"{span([x['exact_given'] for x in b])} / {span([x['exact_hard'] for x in b])} | "
              f"{median(hits) if hits else '·'} ({len(hits)} of {len(rs)} reached it) | {tab} |")


def heldout_table(rows):
    print("| variant | bits | solved (all moduli) | estimator phase, held-out moduli (trained) | after the switch, "
          "held-out moduli (trained) | given the true digits, held-out moduli |")
    print("|---|---|---|---|---|---|")
    for (lab, bits), rs in groups([r for r in rows if r["holdout"]]):
        a, b = [r["after_a"] for r in rs], [r["after_b"] for r in rs]
        print(f"| {lab} | {bits} | {sum(x['exact'] >= PASS for x in b)}/{len(rs)} | "
              f"{span([x['exact_heldout_moduli'] for x in a])} ({span([x['exact_unseen_kept_moduli'] for x in a])}) | "
              f"{span([x['exact_heldout_moduli'] for x in b])} ({span([kept(x) for x in b])}) | "
              f"{span([x['exact_given_heldout_moduli'] for x in b])} |")


def kept(s):
    f, h = s.get("heldout_frac"), s.get("exact_heldout_moduli")
    return None if f is None or h is None or f >= 1 else (s["exact"] - f * h) / (1 - f)


def widths_table(rows, phase="after_b"):
    sizes = sorted({int(b) for r in rows for b in r[phase]["transfer"]})
    print("| variant | trained at | holdout | " + " | ".join(f"{s} bits" for s in sizes) + " |")
    print("|---|---|---|" + "---|" * len(sizes))
    for (lab, bits), rs in groups(rows):
        for ho in sorted({r["holdout"] for r in rs}):
            sel = [r[phase]["transfer"] for r in rs if r["holdout"] == ho]
            cells = [f"{span([t[str(s)]['exact'] for t in sel])} ({span([t[str(s)]['exact_given'] for t in sel])})"
                     for s in sizes]
            print(f"| {lab} | {bits} | {ho} | " + " | ".join(cells) + " |")


def rows_table(rows):
    print("| variant | bits | seed | holdout | rows right | true-path rows: visited, right | over-subtracting trials' "
          "rows: borrow right | wrong rows (r, m, borrow in) |")
    print("|---|---|---|---|---|---|---|---|")
    for (lab, bits), rs in groups([r for r in rows if r["slots"] == "shared"]):
        for r in sorted(rs, key=lambda r: (r["holdout"], r["seed"])):
            t = r["after_b"]["table"]
            wrong = " ".join(f"({a},{b},{c})" for a, b, c in t["wrong"][:12]) + (" …" if len(t["wrong"]) > 12 else "")
            print(f"| {lab} | {bits} | {r['seed']} | {r['holdout']} | {t['right']:.3f} | {t['path_rows']}, "
                  f"{t['right_path']:.3f} | {t['borrow_right_over']:.3f} | {wrong or '–'} |")


def maps_table(rows):
    print("| variant | bits | seed | holdout | estimator phase: choice map (all steps right, mapped) | after the "
          "switch: choice map (all steps right) |")
    print("|---|---|---|---|---|---|")
    for (lab, bits), rs in groups(rows):
        for r in sorted(rs, key=lambda r: (r["holdout"], r["seed"])):
            a, b = r["after_a"]["choice"], r["after_b"]["choice"]
            print(f"| {lab} | {bits} | {r['seed']} | {r['holdout']} | {a['choice_map']} ({a['choice_mapped']['all']:.2f}) "
                  f"| {b['choice_map']} ({b['choice_raw']['all']:.2f}) |")


def stages_table(rows):
    print("| variant | bits | seed | holdout | stages: steps / exact on the stage's validation sample | "
          "own borrow after the switch, every 100 steps (first 6) |")
    print("|---|---|---|---|---|---|")
    for (lab, bits), rs in groups(rows):
        for r in sorted(rs, key=lambda r: (r["holdout"], r["seed"])):
            st = " ".join(f"k{s['k']}: {s['steps']}/{s['exact']:.2f}" for s in r["stages"])
            cb = " ".join(f"{e:.2f}" for _, e, _ in r["curve_b"][:6])
            print(f"| {lab} | {bits} | {r['seed']} | {r['holdout']} | {st} | {cb} |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", default="")
    a = p.parse_args()
    rows = load(tags=set(a.tags.split(",")) if a.tags else None)
    for name, fn in (("main", main_table), ("held-out moduli", heldout_table), ("widths", widths_table),
                     ("stages", stages_table), ("rows", rows_table), ("maps", maps_table)):
        print(f"\n### {name}\n")
        fn(rows)


if __name__ == "__main__":
    main()
