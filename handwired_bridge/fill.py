"""Rewrite HANDWIRED_BRIDGE.md's generated tables from the records, between <!-- NAME --> and <!-- /NAME --> markers.

    ~/github/sotaku/venv/bin/python -m handwired_bridge.fill
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from handwired_bridge import summarize as Sm

HERE = Path(__file__).resolve().parent
DOC = HERE / "HANDWIRED_BRIDGE.md"

NAMES = {
    "H1_table": "1. the subtraction table learned (softmax rows)",
    "H1_table_raw": "1'. the same, plain affine rows",
    "H2_gates_sub": "2. the subtraction cell's product circuit, values learned",
    "H2_gates_sub_i1": "2. the same, values started at N(0, 1)",
    "H2_gates": "2'. every cell's circuit, values learned",
    "H2_dense_sub": "2d. the subtraction cell a dense ReGLU network",
    "H2_dense_sub_full": "2d'. the same, reading every port entry",
    "H2_dense": "2d''. every cell a dense ReGLU network",
    "H2_densetab_sub": "2t. the subtraction table a dense ReGLU network's outputs (the H1 form)",
    "H2_gates_sub_x4": "2 x4. the product circuit, every unit in 4 copies",
    "H2_dense_sub_b3": "2d b3. the dense subtraction cell, write-back softmax(3 x)",
    "H2_dense_sub_x4": "2d x4. the dense subtraction cell, 4 times wider",
    "H2_gates_sub_do": "2-do. the product circuit, output rows dense (the table's pattern learned)",
    "H2_gates_sub_do_x4": "2-do x4. the same, every unit in 4 copies",
    "H2_gates_sub_lift20": "2 lifted. the product circuit through a random projection, d = 20",
    "H2_gates_sub_lift40": "2 lifted. the same, d = 40",
    "H4_untied_table": "4. untied ticks, the table learned",
    "H4_untied_dense_sub": "4. untied ticks, the dense subtraction cell",
    "H4_untied_menu_given_softmax": "4. untied ticks, 7-option menu (softmax), tables given",
    "H4_untied_menu_given_relu": "4. untied ticks, 7-option menu (normalized ReLU), tables given",
}
MUXNAME = {"softmax": "softmax", "relu": "normalized ReLU", "relu_free": "ReLU, no competition",
           "sigmoid_free": "sigmoid, no competition", "raw": "plain ReLU multiplexer, host normalizes",
           "softplus": "normalized softplus (a ReLU that never dies)"}


def notch_name(n):
    if n in NAMES:
        return NAMES[n]
    m = re.match(r"H3_(menu|dense|rel\d+)(_given)?_(\w+)$", n)
    if m:
        kind, given, mux = m.groups()
        what = {"menu": "7-option menu", "dense": "dense routing"}.get(kind, f"{kind[3:]}-option menu")
        return f"3. {what} ({MUXNAME[mux]}){', tables given' if given else ', table learned'}"
    return n


def compile_table():
    rows = json.loads((HERE / "records" / "compile.json").read_text())
    out = ["| (B, w, k) | questions | cells | ticks | live operations | layers | ReGLU gates | stored weights | "
           "handwired's evaluators (tensor / reference / random start) | PyTorch, every answer and cell, f64 and f32, "
           "also from random scratch |", "|" + "---|" * 10]
    for r in rows:
        c, j = r["counts"], r["js_checks"]
        js = (f"{j['tensorRight']}/{j['checked']}, {j['referenceRight']}/{j['reference']}, "
              f"{j['randomRight']}/{j['randomStart']}")
        ok = "exact" if r["exact"] else "WRONG"
        q = f"{r['questions']:,}" + (" (sample)" if r["questions"] < j["questions"] else "")
        out.append(f"| ({','.join(map(str, r['size']))}) | {q} | {c['cells']} | {c['ticks']} | "
                   f"{c['liveOperations']:,} | {c['layers']} | {c['reglu']:,} | {c['storedWeights']:,} | {js} | {ok} |")
    return "\n".join(out)


def ladder_table(suite="ladder"):
    rs = Sm.rows(suite)
    if not rs:
        return "(no records yet)"
    sizes = sorted({r["size"] for r in rs})
    groups = defaultdict(list)
    keys = []
    for r in rs:
        k = (r["notch"], r["tier"])
        groups[(k, r["size"])].append(r)
        if k not in keys:
            keys.append(k)
    out = ["| notch | lessons | " + " | ".join(f"({s})" for s in sizes) + " |", "|" + "---|" * (len(sizes) + 2)]
    for k in keys:
        out.append(f"| {notch_name(k[0])} | {k[1]} | " + " | ".join(Sm.cell(groups[(k, s)]) for s in sizes) + " |")
    return "\n".join(out)


def width_table():
    rs = Sm.rows("width") + [r for r in Sm.rows("width2") if "6k" not in r["name"]]
    if not rs:
        return "(no records yet)"
    n = lambda r: 7 if r["notch"].startswith("H3_menu_given") else int(r["notch"].split("_")[1][3:])
    mux = lambda r: r["notch"].split("_given_")[1]
    widths = sorted({n(r) for r in rs})
    out = ["| multiplexer | lessons | " + " | ".join(f"{w} options" for w in widths) + " |",
           "|" + "---|" * (len(widths) + 2)]
    for tier in ("AL", "A"):
        for m in MUXNAME:
            g = [r for r in rs if r["tier"] == tier and mux(r) == m]
            if g:
                out.append(f"| {MUXNAME[m]} | {tier} | " + " | ".join(Sm.cell([r for r in g if n(r) == w])
                                                                    for w in widths) + " |")
    return "\n".join(out)


def diag_table():
    rs = Sm.rows("diagnose")
    if not rs:
        return "(no records yet)"
    share = lambda d, cls: float(np.mean([v.get(cls, 0.0) for v in d.values()])) if d else float("nan")
    out = ["| case | multiplexer | seed | certified | training, hard / soft | ports right | sources: prompt / zero / "
           "other computed | first gradient toward the prompt | loss on the line to the compiled weights | "
           "subtraction digits right |", "|" + "---|" * 10]
    for r in sorted(rs, key=lambda r: r["name"]):
        case, _, rest = r["name"].partition(".")
        if case.startswith("H2"):
            continue
        mux, seed = rest.rsplit(".s", 1)
        ports = r.get("ports", {})
        right = sum(1 for v in ports.values() if v.get("right", 0) == 1.0)
        path = r.get("path") or []
        p = (f"{path[0][1]:.2f} -> {max(x[1] for x in path):.2f} -> {path[-1][1]:.2f}" if path else "·")
        out.append(f"| {case} | {MUXNAME.get(mux, mux)} | {seed} | {'yes' if r.get('certified') else 'no'} | "
                   f"{r['train_hard']:.2f} / {r['train_soft']:.2f} | {right}/7 | {share(ports, 'input'):.2f} / "
                   f"{share(ports, 'zero'):.2f} / {share(ports, 'computed'):.2f} | {share(r.get('grad0', {}), 'input'):.2f}"
                   f" | {p} | {r.get('sub_digits_right', '·')}/{r.get('sub_entries', '·')} |")
    return "\n".join(out)


def cells_table():
    rs = [r for r in Sm.rows("diagnose") if r["name"].startswith("H2")]
    if not rs:
        return "(no records yet)"
    out = ["| notch | lessons | seed | certified | training, hard / soft | subtraction digits / borrows right | "
           "outputs changed by the entries the compiled cell ignores | scratch digits right, renamed / as is |",
           "|" + "---|" * 8]
    for r in sorted(rs, key=lambda r: r["name"]):
        notch, tier, seed = r["name"].split(".")
        rl = r.get("relabel", {})
        ren = np.mean([v[0] for v in rl.values()]) if rl else float("nan")
        asis = np.mean([v[1] for v in rl.values()]) if rl else float("nan")
        irr = r.get("irrelevant")
        out.append(f"| {notch_name(notch)} | {tier} | {seed[1:]} | {'yes' if r.get('certified') else 'no'} | "
                   f"{r['train_hard']:.2f} / {r['train_soft']:.2f} | {r.get('sub_digits_right')}/"
                   f"{r.get('sub_borrows_right')} of {r.get('sub_entries')} | "
                   f"{'·' if irr is None else f'{irr:.2f}'} | {ren:.2f} / {asis:.2f} |")
    return "\n".join(out)


def generic_table(suite, key, col, cols=None):
    """Rows by key(r) (a tuple of strings), columns by col(r)."""
    rs = Sm.rows(suite)
    if not rs:
        return "(no records yet)"
    groups, keys = defaultdict(list), []
    for r in rs:
        groups[(key(r), col(r))].append(r)
        if key(r) not in keys:
            keys.append(key(r))
    cols = cols or sorted({col(r) for r in rs}, key=str)
    return "\n".join(["| " + " | ".join(["notch", "lessons", *[""] * 0]) + " | " + " | ".join(map(str, cols)) + " |",
                      "|" + "---|" * (len(cols) + 2)]
                     + [f"| {notch_name(k[0])} | {k[1]} | " + " | ".join(Sm.cell(groups[(k, c)]) for c in cols) + " |"
                        for k in keys])


def width6k_table():
    rs = [r for r in Sm.rows("width2") if "6k" in r["name"]]
    if not rs:
        return "(no records yet)"
    n = lambda r: int(r["notch"].split("_")[1][3:])
    mux = lambda r: r["notch"].split("_given_")[1]
    widths = sorted({n(r) for r in rs})
    out = ["| multiplexer | " + " | ".join(f"{w} options" for w in widths) + " |", "|" + "---|" * (len(widths) + 1)]
    for m in ("softmax", "softplus"):
        g = [r for r in rs if mux(r) == m]
        out.append(f"| {MUXNAME[m]} | " + " | ".join(Sm.cell([r for r in g if n(r) == w]) for w in widths) + " |")
    return "\n".join(out)


def stab_cell(rs):
    """certified at the end / runs (runs exact on training before training)."""
    if not rs:
        return "·"
    return f"{sum(bool(r.get('certified')) for r in rs)}/{len(rs)} ({sum(r.get('start_hard') == 1.0 for r in rs)})"


def wstab_table():
    rs = Sm.rows("wstab")
    if not rs:
        return "(no records yet)"
    def sigma(r):                                       # names: <notch>.<min|wide>.<tier>.s<sigma>.321.s<seed>
        part = r["name"].split(".")
        return float(f"{part[3][1:]}.{part[4]}")
    rs = [r for r in rs if r["notch"].startswith("H3")]
    groups, keys = defaultdict(list), []
    for r in rs:
        notch, wide, tier = r["name"].split(".")[:3]
        k = (notch, wide, tier)
        groups[(k, sigma(r))].append(r)
        if k not in keys:
            keys.append(k)
    sigmas = sorted({s for (_, s) in groups})
    out = ["| notch | start | lessons | " + " | ".join(f"sigma = {s:g}" for s in sigmas) + " |",
           "|" + "---|" * (len(sigmas) + 3)]
    for k in sorted(keys, key=lambda k: (k[0], k[2] == "A", k[1])):
        out.append(f"| {notch_name(k[0])} | {'widened x4' if k[1] == 'wide' else 'minimal'} | "
                   f"{'the answers' if k[2] == 'A' else k[2]} | " + " | ".join(stab_cell(groups[(k, s)]) for s in sigmas)
                   + " |")
    return "\n".join(out)


def overparam_table():
    """The compressed product circuit, minimal against widened (Net2Net copies) against projected (lifted through a
    random projection): near the compiled point (certified at the end / runs, with the runs exact at the start in
    brackets) and from random."""
    stab = [r for r in Sm.rows("wstab") if r["notch"] == "H2_gates_sub"] + [r for r in Sm.rows("lift") if ".proj." in r["name"]]

    def form(r):
        part = r["name"].split(".")
        return {"min": "minimal", "wide": "widened x4 (Net2Net)", "proj": "projected, d = 20"}[part[1]]

    def sigma(r):
        part = r["name"].split(".")
        return float(f"{part[3][1:]}.{part[4]}")
    groups = defaultdict(list)
    for r in stab:
        groups[(form(r), r["name"].split(".")[2], sigma(r))].append(r)
    sigmas = sorted({k[2] for k in groups})
    out = []
    if stab:
        out = ["**Near the compiled point** (noise sigma on every learned number, then 1,000 steps):", "",
               "| circuit | lessons | " + " | ".join(f"sigma = {s:g}" for s in sigmas) + " |",
               "|" + "---|" * (len(sigmas) + 2)]
    for f in ("minimal", "widened x4 (Net2Net)", "projected, d = 20"):
        for t in ("GC", "A"):
            if any((f, t, s_) in groups for s_ in sigmas):
                out.append(f"| {f} | {'the answers' if t == 'A' else t} | "
                           + " | ".join(stab_cell(groups.get((f, t, s_), [])) for s_ in sigmas) + " |")
    rnd = ([r for r in Sm.rows("ladder") if r["notch"] == "H2_gates_sub" and r["tier"] in ("GC", "AC")]
           + [r for r in Sm.rows("wlearn") if r["notch"] == "H2_gates_sub"]
           + [r for r in Sm.rows("lift") if ".proj." not in r["name"]]
           + [r for r in Sm.rows("over") if r["notch"].startswith("H2_gates_sub")])

    def label(r):
        if "lift" in r["notch"]:
            return "projected, d = " + r["notch"].split("lift")[1]
        wide = "wide" in r["tier"] or r["notch"].endswith("_x4")
        do = "_do" in r["notch"]
        return ("output rows dense, " if do else "table's pattern given, ") + ("widened x4" if wide else "minimal")
    if rnd:
        sizes = sorted({r["size"] for r in rnd})
        out += ["", "**From random:**", "",
                "| circuit | lessons | " + " | ".join(f"({s})" for s in sizes) + " |", "|" + "---|" * (len(sizes) + 2)]
        for f in ("table's pattern given, minimal", "table's pattern given, widened x4", "output rows dense, minimal",
                  "output rows dense, widened x4", "projected, d = 20", "projected, d = 40"):
            for t in ("AC", "GC"):
                g = [r for r in rnd if label(r) == f and r["tier"].split()[0] == t]
                if g:
                    out.append(f"| {f} | {t} | " + " | ".join(Sm.cell([r for r in g if r["size"] == s]) for s in sizes)
                               + " |")
    return "\n".join(out) or "(no records yet)"


def h1_table():
    import glob
    import hashlib
    rows = ["| run | device | seed | max T, seen / unseen | every rung 768/768 | test, ood_t, ood_n_t | training s (steps) | "
            "evaluation s | md5 |", "|" + "---|" * 9]
    for f in sorted(glob.glob(str(HERE / "h1" / "records" / "*.json"))):
        r = json.loads(Path(f).read_text())
        res = r.get("result")
        name = Path(f).stem
        if not res:
            rows.append(f"| {name} | · | · | crashed (returncode {r['returncode']}) | | | | | |")
            continue
        s_ = res["seeds"][0]
        dp = s_["depth_profile"]
        every = all(x["correct_examples"] == 768 for x in dp["rungs"] + dp["ood_n_rungs"])
        ev = ", ".join(f"{s_['evaluation'][k]['correct_examples']:,}/{s_['evaluation'][k]['example_count']:,}"
                       for k in ("test", "ood_t", "ood_n_t"))
        dev = "H100" if "H100" in (r.get("gpu") or "") else "CPU"
        mt = f"{dp['max_certified_time_steps'] or 'none'} / {dp['ood_n_max_certified_time_steps'] or 'none'}"
        rows.append(f"| {name} | {dev} | {s_['seed']} | {mt} | {'yes' if every else 'no'} | {ev} | "
                    f"{s_['training_seconds']:.0f} ({s_['completed_training_steps']:,}) | {s_['evaluation_seconds']:.0f} | "
                    f"{hashlib.md5(r['submission'].encode()).hexdigest()[:8]} |")
    return "\n".join(rows)


def fill(text, name, table):
    pat = re.compile(rf"(<!-- {name} -->\n).*?(<!-- /{name} -->)", re.S)
    if not pat.search(text):
        return text
    return pat.sub(lambda m: m.group(1) + table + "\n" + m.group(2), text)


def main():
    text = DOC.read_text()
    tables = (("COMPILE", compile_table), ("LADDER", ladder_table), ("WIDTH", width_table), ("DIAG", diag_table),
              ("CELLS", cells_table), ("WSTAB", wstab_table),
              ("WLEARN", lambda: generic_table("wlearn", lambda r: (r["notch"], r["tier"]), lambda r: r["size"])),
              ("PROJ", lambda: generic_table("proj", lambda r: (r["notch"], r["tier"]), lambda r: r["size"])),
              ("BASES", lambda: generic_table("bases", lambda r: (r["notch"], r["tier"]), lambda r: r["size"])),
              ("WIDTH6K", width6k_table), ("OVERPARAM", overparam_table),
              ("UPDOWN", lambda: generic_table("updown", lambda r: (r["notch"], r["tier"]), lambda r: r["size"])),
              ("H1RUNS", h1_table))
    for name, fn in tables:
        try:
            text = fill(text, name, fn())
        except FileNotFoundError:
            pass
    DOC.write_text(text)


if __name__ == "__main__":
    main()
