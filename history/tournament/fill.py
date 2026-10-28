"""Rewrite TOURNAMENT.md's generated tables from the records, between <!-- NAME --> and <!-- /NAME --> markers.

    ~/github/sotaku/venv/bin/python -m tournament.fill
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

VARIANTS = {   # tag: (what changes against the recipe, size)
    "main": ("the recipe: MLP subtraction cell learned, the one-link laws and every answer, gate on, lr 0.5", "(3, 2, 1)"),
    "given": ("the truth tables given: the wiring alone", "(3, 2, 1)"),
    "k2": ("two quotient digits", "(3, 2, 2)"),
    "b4": ("base 4 (three links per step)", "(4, 2, 1)"),
    "b5": ("base 5 (four links per step)", "(5, 2, 1)"),
    "lab10": ("10% of the questions labelled (10 of 99)", "(3, 2, 1)"),
    "lab02": ("2% labelled (2 of 99)", "(3, 2, 1)"),
    "nolabel": ("no answers: label-free laws only", "(3, 2, 1)"),
    "h1like": ("minibatches of 32, 5% labelled", "(3, 2, 1)"),
    "mb16": ("minibatches of 16 questions", "(3, 2, 1)"),
    "nogate": ("the wiring's lessons from step 1 (no waiting for the table)", "(3, 2, 1)"),
    "row": ("verdicts per row (WIRING.md's mixture) instead of per batch", "(3, 2, 1)"),
    "lr02": ("lr 0.2 on the combinations' logits", "(3, 2, 1)"),
    "lr005": ("lr 0.05 (the H1 file's switch rate)", "(3, 2, 1)"),
    "mlptab": ("the H1 file's cell form: the table an MLP's outputs (bridge notch 1e)", "(3, 2, 1)"),
    "w52": ("a wider window: rows -5..0, columns -4..4 (52 options per port, 1.0e12 wirings)", "(3, 2, 1)"),
    "w61": ("rows -6..0, columns -4..4 (61 options, 3.1e12 wirings)", "(3, 2, 1)"),
}


def load(suite):
    p = HERE / "runs" / f"{suite}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def med(xs):
    return f"{statistics.median(xs):g} ({min(xs)}-{max(xs)})" if xs else "·"


def failure(r):
    """What a failed run ended on: the table, or the kind of wiring (tiny.Tiny.trap's classes)."""
    if r["table_right"] < r["table_entries"] and "gate_open" not in r["events"]:
        return f"table never learned ({r['table_right']}/{r['table_entries']})"
    if not r["trap"]:
        return "true wiring"
    kinds = set()
    for t in r["trap"].values():
        kinds.add("bypass" if t["input"] > 0 else "constant" if t["zero"] == 1.0 else "impostor")
    return ("bypass" if "bypass" in kinds else "constant" if "constant" in kinds else "impostor") + \
        f" ({r['ports_right']}/7 ports right, {r['train_exact']:.0%} of training answers)"


def tiny_table():
    rows = load("main") + load("slow") + load("wide")
    if not rows:
        return "(no records yet)"
    by = defaultdict(list)
    for r in rows:
        by[r["cfg"]["tag"]].append(r)
    out = ["| variant | size | certified | true wiring | steps from the gate to the true wiring | steps to the first exact "
           "network | wall s, median | grad test | failures |", "|---|---|---|---|---|---|---|---|---|"]
    for tag, (what, size) in VARIANTS.items():
        rs = by.get(tag)
        if not rs:
            continue
        cert = sum(r["certified"] for r in rs)
        true = sum(r["true_wiring"] for r in rs)
        gap = [r["events"]["true_wiring"] - r["events"].get("gate_open", 0) for r in rs if "true_wiring" in r["events"]]
        hits = [r["hit"] for r in rs if r["hit"]]
        grad = sum(bool(r.get("grad_test", {}).get("grad_sub") and r["grad_test"].get("grad_sel")
                        and r["grad_test"].get("grad_cell") and r["grad_test"].get("answers_equal_hard")) for r in rs)
        fails = defaultdict(int)
        for r in rs:
            if not r["certified"]:
                fails[failure(r).split(" (")[0]] += 1
        f = ", ".join(f"{v} {k}" for k, v in fails.items()) or "-"
        gt = "·" if all(r["cfg"].get("tables") == "given" for r in rs) else f"{grad}/{len(rs)}"
        out.append(f"| {what} | {size} | {cert}/{len(rs)} | {true}/{len(rs)} | {med(gap)} | {med(hits)} | "
                   f"{statistics.median(r['wall'] for r in rs):.0f} | {gt} | {f} |")
    return "\n".join(out)


def traps_table():
    p = HERE / "records" / "traps.json"
    if not p.exists():
        return "(no record yet)"
    r = json.loads(p.read_text())
    k1, k2, fit = r["sub_kinds"], r["link_kinds"], r["link_fit"]
    out = ["| what passes | combinations | true | bypass | constant source | other impostor | fit every training answer |",
           "|---|---|---|---|---|---|---|",
           f"| block \"sub\"'s law (the subtraction borrows exactly when x < N, one link) | {r['sub_pass']:,} of "
           f"{r['sub_blocks']:,} | {k1.get('true', 0)} | {k1.get('bypass', 0)} | {k1.get('constant', 0)} | "
           f"{k1.get('impostor', 0)} | · |",
           f"| + the one-link law (x below N unchanged, N to 0), over every select | {r['link_pass']:,} of "
           f"{r['link_pairs']:,} | {k2.get('true', 0)} | {k2.get('bypass', 0)} | {k2.get('constant', 0)} | "
           f"{k2.get('impostor', 0)} | {sum(v for k, v in fit.items() if k.endswith('fits'))} (the true one) |"]
    return "\n".join(out)


def census_table():
    p = HERE / "records" / "h1_census.json"
    if not p.exists():
        return "(no record yet)"
    r = json.loads(p.read_text())
    out = [f"| the chain's lessons on {r['rows']} training rows (the link law passes {r['link_pass']} of "
           f"{r['link_combinations']:,} link combinations) | chain wirings passing every lesson (of 7.9e9) | their step "
           f"right on all {r['ood_rows_8digit']} 8-digit moduli | wrong there | the gate as the loop needs it (commit on "
           f"no borrow) |", "|---|---|---|---|---|"]
    for key, label in (("plain", "the plain lessons"), ("with_loop", "+ the loop law"),
                       ("with_norm", "+ the moved-up rows"), ("with_norm_loop", "+ both (the file)")):
        c = r.get(key)
        if not c:
            continue
        out.append(f"| {label} | {c['pass_training']} | {c['right_on_8digit']} | "
                   f"{c['pass_training'] - c['right_on_8digit']} | {'all' if 'loop' in key else 'half'} |")
    return "\n".join(out)


def h1_row(r):
    res = r.get("result")
    if not res:
        return f"| {r['name']} | {r.get('gpu') or 'cpu'} | failed (return code {r.get('returncode')}) | | | | | | | |"
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
    w = re.search(r"^WIRING (op=.*) true=(\S+) chain=(\S+)", out, re.M)
    rungs = all(x["correct_examples"] == x["example_count"] for x in dp["rungs"] + dp["ood_n_rungs"])
    worst = min(x["correct_examples"] for x in dp["ood_n_rungs"]) if dp["ood_n_rungs"] else 0
    ev = sd["evaluation"]
    splits = ", ".join(f"{ev[k]['correct_examples']:,}/{ev[k]['example_count']:,}" for k in ("test", "ood_t", "ood_n_t"))
    md5 = hashlib.md5(r["submission"].encode()).hexdigest()[:8]
    wiring = "·"
    if w:
        wd = dict(kv.split("=", 1) for kv in w.group(1).split() if "=" in kv)
        chain = w.group(3).replace("_", " ")
        ab = (wd.get("a"), wd.get("b"))
        algo = {("('D',0)", "('M',0)"): "repeated subtraction", ("('Rp',0)", "('Mq',0)"): "trial multiples"}.get(ab)
        if chain == "other" and algo and wd.get("cv") == "('D',0)" and wd.get("hv") == "('E',0)" and wd.get("gate") == "HC":
            chain = f"{algo}, the gate flipped"
        elif (chain == "other" and algo and wd.get("cv") == "('E',0)" and wd.get("hv") == "('D',0)"
              and wd.get("gate") == "CH" and re.fullmatch(r"b\d", wd.get("flag", ""))):
            chain = f"{algo}, the gate reading the borrow out of place {wd['flag'][1]}"
        wiring = (f"{chain}; flag {wd.get('flag')}, a {wd.get('a')}, b {wd.get('b')}, "
                  f"commit {wd.get('cv')}, hold {wd.get('hv')}, gate {wd.get('gate')}").replace("'", "")
    return (f"| {r['name']} | {'H100' if 'H100' in (r.get('gpu') or '') else 'cpu'} | {sd['seed']} | "
            f"{dp['max_certified_time_steps']} / {dp['ood_n_max_certified_time_steps']} | "
            f"{'768/768 every rung' if rungs else f'not all (worst unseen rung {worst}/768)'} | {splits} | "
            f"{sd['training_seconds']:.0f} ({fz.group(1) if fz else '·'}) | {sd['evaluation_seconds']:.0f} | "
            f"{'yes' if exact else 'no'} | {wiring} | {md5} |")


def h1_table():
    """The H1 runs, one table per version of the file (by the md5 of the source each record embeds)."""
    recs = [json.loads(p.read_text()) for p in sorted((HERE / "h1_records").glob("*.json"))]
    if not recs:
        return "(the runs are on Modal)"
    head = ["| record | device | seed | max T, seen / unseen moduli | rungs | test, ood_t, ood_n_t | training s (frozen "
            "at step) | evaluation s | both MLP cells exact | the chain it chose | md5 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    mask = lambda src: hashlib.md5(re.sub(r"^CHAIN_NORM = .*$", "CHAIN_NORM", src, flags=re.M).encode()).hexdigest()
    final_src = (HERE / "submission_tournament.py").read_text()
    final = mask(final_src)
    groups = defaultdict(list)
    for r in recs:                       # a variant's source differs only in its CHAIN_NORM line
        groups[mask(r["submission"])].append(r)
    order = lambda r: (not r["name"].endswith("official"), "nonorm" in r["name"], r["name"])
    out = []
    for key in sorted(groups, key=lambda m: m != final):
        rs = sorted(groups[key], key=order)
        base = [r for r in rs if "nonorm" not in r["name"]]
        md5 = hashlib.md5((base or rs)[0]["submission"].encode()).hexdigest()
        if key == final:
            title = f"**The file** (md5 `{md5[:8]}`; a `nonorm` record is the same source with `CHAIN_NORM = False`):"
        else:
            title = f"**The first version, without the loop law** (md5 `{md5[:8]}`):"
        out += ["", title, ""]
        out += head + [h1_row(r) for r in rs]
    return "\n".join(out[1:])


def main():
    path = HERE / "TOURNAMENT.md"
    text = path.read_text()
    blocks = {"TINY": tiny_table, "TRAPS": traps_table, "CENSUS": census_table, "H1": h1_table}
    for name, fn in blocks.items():
        pat = re.compile(rf"(<!-- {name} -->\n).*?(<!-- /{name} -->)", re.S)
        if pat.search(text):
            text = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(2), text)
    path.write_text(text)
    print("filled", path)


if __name__ == "__main__":
    print(tiny_table())
    print(traps_table())
    print(census_table())
    print(h1_table())
