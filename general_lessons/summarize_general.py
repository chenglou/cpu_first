"""The general-lessons records (cpu_first/general_lessons/records/*.json) as rows of facts and as LESSONS.md's tables.

    python summarize_general.py                  # one line per record
    python summarize_general.py fill LESSONS.md  # rewrite the <!-- NAME --> ... <!-- /NAME --> tables in place

Per record: everything litmus/summarize_litmus.py reads (score, splits, certified T, rungs, steps, seconds, schedule,
tables, reader), and from this file's own lines: the selected lesson set and the selector's weights, each set's wiring
and its algorithm, the round's options, the mined facts in force at the end (the MINED line) and when the schedule's
events happened (each set's I/O lesson, the freeze).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORDS = HERE / "records"
_spec = importlib.util.spec_from_file_location("summarize_litmus", HERE.parent / "litmus" / "summarize_litmus.py")
SL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SL)
FILES = {}                                                  # md5 -> short name, filled from the files on disk
for p in [HERE / "submission_general.py", HERE / "submission_general_v2.py", *sorted((HERE / "variants").glob("*.py"))]:
    if p.exists():
        import hashlib
        FILES[hashlib.md5(p.read_bytes()).hexdigest()] = "the file" if p.name == "submission_general.py" else \
            p.stem.replace("submission_general_", "")


def facts(rec: dict) -> dict:
    out = SL.facts(rec)
    so = rec.get("stdout", "")
    out["file"] = FILES.get(out["md5"], out["md5"][:8])
    if "H200" in rec.get("gpu", ""):                   # Modal may run a gpu="H100" job on an H200
        out["gpu"] = "H200"
    les = SL.line(so, "LESSONS selected=")
    if les:
        out["selected"] = re.search(r"selected=(\S+)", les).group(1)
        out["selector"] = re.search(r"selector=(\S+)", les).group(1)
        out["round"] = " ".join(re.findall(r"(rmul=\S+|radd=\S+|rcube=\S+)", les.split("round_probabilities")[0]))
    out["sets"] = {}
    for l in so.splitlines():
        m = re.match(r"WIRING set=(\S+) (.*) true=(\S+) chain=(\S+)", l)
        if m and m.group(1) not in out["sets"]:
            out["sets"][m.group(1)] = {"wiring": m.group(2), "true": m.group(3) == "True",
                                       "chain": m.group(4).replace("_", " ")}
    mined = SL.line(so, "MINED ")
    if mined is not None:
        out["mined"] = [re.sub(r"\(since.*", "", f) for f in mined.removeprefix("MINED ").split(";") if f.strip()]
    first = {}
    for l in so.splitlines():
        m = re.match(r"LR mined step=(\d+) fact=(\S+)", l)
        if m:
            first.setdefault(m.group(2), int(m.group(1)))
    out["mined_first"] = first
    ev = {}
    for l in so.splitlines():
        m = re.match(r"LR io_wiring_lesson_joins set=(\S+) step=(\d+)", l)
        if m:
            ev[f"io_{m.group(1)}"] = int(m.group(2))
        m = re.match(r"LR (order_digits|switch_to_facts|ladder_joins|chain_lesson_joins|pieces_join) step=(\d+)", l)
        if m:
            ev.setdefault(m.group(1), int(m.group(2)))
        m = re.match(r"LR io_stage=(\d) .*?step=(\d+)", l) or re.match(r"LR io_stage=(\d) step=(\d+)", l)
        if m:
            ev.setdefault(f"io{m.group(1)}", int(m.group(2)))
        m = re.match(r"LR frozen step=(\d+)", l)
        if m:
            ev["frozen"] = int(m.group(1))
    out["gl_events"] = ev
    last = [l for l in so.splitlines() if l.startswith("LR step=")]
    if last:
        fit = re.search(r" fit=(\S+)", last[-1])
        out["fit"] = fit.group(1) if fit else None
    return out


def records() -> list:
    out = []
    for p in sorted(RECORDS.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:                   # still being downloaded
            continue
        out.append(facts(rec))
    return out


def stopped(f: dict) -> str:
    ev = f.get("gl_events", {})
    if "frozen" in ev:
        return f"froze at step {ev['frozen']}"
    last = f.get("last") or {}
    if f.get("max_t"):
        return "certified without freezing" if f.get("max_t") == 64 else f"max T {f.get('max_t')}"
    if "io4" not in ev and int(last.get("io") or 0) < 4:   # the frozen-reader variant starts at io stage 5
        return f"reading never chosen (io stage {last.get('io')})"
    if f.get("sub_right") is not None and f["sub_right"] < 200:
        return f"subtraction table never exact ({f['sub_right']}/200 entries right)"
    if f.get("fit") and all(float(v) == 0 for v in f["fit"].split(",")):
        why = "no row has T <= 2" if float(last.get("ans_ok") or 0) == 1 else "answers 0"   # 1.0 here is vacuous
        return f"stopped in phase {last.get('phase')}: hard checks {last.get('hard_exact')}, no set-and-round fits ({why})"
    return f"stopped in phase {last.get('phase')}: hard checks {last.get('hard_exact')}, answers {last.get('ans_ok')}"


def row(f: dict) -> str:
    sp = ", ".join(f"{k} {v * 100:.1f}%" for k, v in sorted(f.get("splits", {}).items()))
    sets = "; ".join(f"{k}: {v['chain']}" for k, v in f.get("sets", {}).items())
    sc = f"{f['score'] * 100:.1f}%" if f.get("score") is not None else f"failed (rc {f['rc']})"
    joined = any(k.startswith("io_") for k in f.get("gl_events", {}))
    sel = f"{f.get('selected')} ({f.get('selector')})" if joined else "- (no set's wiring lessons joined)"
    rnd = f.get("round") if joined else "-"
    sets = sets if joined else "-"
    return (f"| {f['name']} | {f.get('task')} | {f.get('file')} | {f['gpu']} | {sc} | {sp} | {f.get('max_t') or 0} / "
            f"{f.get('ood_max_t') or 0} | {f.get('steps')} | {f.get('train_s') or 0:.0f} | {f.get('eval_s') or 0:.0f} | "
            f"{stopped(f)} | {sel} | {rnd} | {sets} |")


HEAD = ("| run | task | file | device | score | splits | max T seen / unseen | steps | training s | evaluation s | "
        "where it stopped | lesson set chosen (selector) | round options | each set's chain |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def mined_row(f: dict) -> str:
    return f"| {f['name']} | {f.get('task')} | {', '.join(f.get('mined') or []) or '-'} |"


def tiny_gen_table() -> str:
    """The tiny kind-3 core (tiny_general.py) runs: one row per task and variant, seeds pooled."""
    groups = {}
    for p in sorted((HERE / "tiny").glob("gen_*.json")):
        r = json.loads(p.read_text())
        if "error" in r:
            continue
        c = r["cfg"]
        key = (c["task"], c.get("cands", "below,lowzero"), c.get("gates", "answers"), c.get("label_frac", 0.02))
        groups.setdefault(key, []).append(r)
    out = ["| task | candidates | selector trained by | answered rows (share) | seeds certified (T up to 8, every width) "
           "| set chosen (its weight) | round options chosen | each set's chain, by seed |", "|---|---|---|---|---|---|---|---|"]
    for (task, cands, gates, lf), rs in groups.items():
        cert = sum(r["certified"] for r in rs)
        chosen = ", ".join(sorted({f"{r['selected']} ({max(r['selector']):.3f})" for r in rs}))
        rnd = ", ".join(sorted({" ".join(f"{k}={v}" for k, v in r["round"].items()) for r in rs}))
        chains = "; ".join(f"s{r['cfg']['seed']}: " + ", ".join(
            f"{k} {v['a']}-{v['b']} {v['op']} {v['flag']}" for k, v in r["wirings"].items() if k != "none") for r in rs)
        out.append(f"| {task} | {cands} | {'answers only' if gates == 'answers' else 'lesson loss (soft gates)'} | "
                   f"{lf} | {cert}/{len(rs)} | {chosen} | {rnd} | {chains} |")
    return "\n".join(out)


def tiny_table_table() -> str:
    """The tiny kind-1 core (tiny_table.py): the subtraction table from a random start, reader given."""
    groups = {}
    for p in sorted((HERE / "tiny").glob("table_*.json")):
        r = json.loads(p.read_text())
        if "error" in r:
            continue
        c = r["cfg"]
        groups.setdefault((c["fixed"] or "many (4-5 digits)", c["pairs"], c.get("early", 0)), []).append(r)
    out = ["| moduli | laws applied to | when the algebraic laws join | seeds exact | step exact | entries right at the end "
           "(digit, borrow, both) |", "|---|---|---|---|---|---|"]
    for (fx, pairs, early), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
        ex = [r["first_exact"] for r in rs]
        fin = ", ".join(f"{r['final'][1]}/{r['final'][2]}/{r['final'][3]}" for r in rs)
        out.append(f"| {fx} | {'a register and N' if pairs == 'N' else 'every pair (ALL_PAIRS)'} | "
                   f"{'from step 1' if early else '100 steps after the digit lessons'} | "
                   f"{sum(e is not None for e in ex)}/{len(rs)} | {', '.join('-' if e is None else str(e) for e in ex)} | {fin} |")
    return "\n".join(out)


def preview_table() -> str:
    pv = json.loads((HERE / "preview.json").read_text())
    order = ["h1"] + [f"e{i}" for i in range(1, 11)] + [f"m{i}" for i in range(1, 11)] + \
        ["xfirst", "mont", "ten", "rho", "cube", "xy", "tenrho", "plus3", "minus1"]
    out = ["| task | moduli | digit codes passing x < N (N as written / N reversed) | adjacent digit pairs ordered | "
           "facts (every field read right) |", "|---|---|---|---|---|"]
    for t in order:
        if t in pv:
            f = pv[t]
            out.append(f"| {t} | {f['moduli']} | {f['codes']['codes']:,} / {f['codes_n_reversed']['codes']:,} | "
                       f"{f['codes']['adjacent_pinned']}/9 | {', '.join(f['facts'])} |")
    return "\n".join(out)


LITMUS_FIX = {   # LITMUS.md's smallest change per failing task (its SUMMARY table, abridged)
    "rho": "one switch, +1 before the reduction (17 lines)", "ten": "one switch, x10 before the reduction (13 lines)",
    "cube": "one round option, times the input, and a second multiply operand (21 lines)",
    "mont": "its three division lessons rewritten for Montgomery (2/4 seeds), plus a gate check (4/4)",
    "m5": "one tie: T read the way the writer writes (4/4 seeds)",
}
GLANCE_TASKS = ["h1", "rho", "ten", "cube", "mont", "tenrho", "plus3", "minus1", "xfirst", "m5", "e3", "m4"] + \
    [f"e{i}" for i in (1, 2, 6, 7, 8, 9, 10)] + [f"m{i}" for i in (1, 2, 6, 7, 8, 9, 10)]


def glance_table() -> str:
    lit = {}
    for p in sorted((HERE.parent / "litmus" / "records").glob("*.json")):
        f = SL.facts(json.loads(p.read_text()))
        if f["md5"] == SL.BEST_MD5 and (p.stem.endswith("_s74") or p.stem.endswith("_relaxed")):
            lit.setdefault(f["task"], f)
    mine, mine2 = {}, {}
    for f in records():
        if "steps3000" in f["name"]:                       # another budget: §4 reports it
            continue
        if f.get("file") in ("the file", "v2"):
            (mine if f["file"] == "the file" else mine2).setdefault(f["task"], []).append(f)
    out = ["| task | the best model unchanged (LITMUS.md) | the smallest change litmus found | the general file, unchanged: "
           "runs certified (T = 64 on both ladders) | its score, max T seen / unseen | lesson set and round options chosen |",
           "|---|---|---|---|---|---|"]
    for t in GLANCE_TASKS:
        L = lit.get(t)
        lt = (f"{L['score'] * 100:.1f}%, max T {L['max_t'] or 0} / {L['ood_max_t'] or 0}" if L and L.get("score") is not None
              else ("certified (WIDE_ANCHORED.md)" if t == "h1" else "-"))
        cert, sc, ch = glance_cells(mine.get(t, []))
        if t in mine2:                                     # v2 (§11) after v1
            cert2, sc2, ch2 = glance_cells(mine2[t])
            cert, sc, ch = f"{cert}; v2: {cert2}", f"{sc}; v2: {sc2}", f"{ch}; v2: {ch2}"
        fix = LITMUS_FIX.get(t, "none needed" if t in ("h1", "xfirst") else
                             ("held out: not a litmus task" if t in ("tenrho", "plus3", "minus1") else
                              "none found (LITMUS.md §4)"))
        if t in ("e3", "m4"):
            fix = "none needed (learned unchanged)"
        out.append(f"| {t} | {lt} | {fix} | {cert} | {sc} | {ch} |")
    return "\n".join(out)


def glance_cells(fs: list) -> tuple:
    """One file's runs of one task: runs certified, the scores, the lesson set and round options chosen."""
    from collections import Counter
    cert = sum(1 for f in fs if f.get("max_t") == 64 and f.get("ood_max_t") == 64)
    cnt = Counter(f"{f['score'] * 100:.1f}%, {f.get('max_t') or 0} / {f.get('ood_max_t') or 0}"
                  if f.get("score") is not None else f"failed (rc {f['rc']})" for f in fs)
    sc = "; ".join(f"{k} (×{v})" if v > 1 else k for k, v in cnt.items())

    def chose(f: dict) -> str:
        ev = f.get("gl_events", {})
        if not any(k.startswith("io_") for k in ev):
            return "- (no set's wiring lessons joined)"
        if f.get("fit") and all(float(v) == 0 for v in f["fit"].split(",")) and f.get("max_t") != 64:
            return "- (no set-and-round combination fits the answers)"
        sel = max(float(v.split(":")[1]) for v in (f.get("selector") or "x:0").split(","))
        return f"{f.get('selected')} ({sel:.4f}), {f.get('round')}"
    groups = {}
    for f in fs:
        c = chose(f)
        m = re.match(r"(\S+) \(([\d.]+)\), (.*)", c)
        if m:
            groups.setdefault((m.group(1), m.group(3)), []).append(float(m.group(2)))
        else:
            groups.setdefault((c, ""), [])
    ch = "; ".join(f"{k[0]} ({min(v):.4f}" + (f"–{max(v):.4f}" if max(v) > min(v) else "") + f"), {k[1]}" if v else k[0]
                   for k, v in groups.items())
    return f"{cert}/{len(fs) if fs else '-'}", sc or "-", ch or "-"


def codes_table() -> str:
    cj = json.loads((HERE / "codes.json").read_text())
    pv = json.loads((HERE / "preview.json").read_text())
    order = [f"e{i}" for i in (1, 2, 6, 7, 8, 9, 10)] + [f"m{i}" for i in (1, 2, 6, 7, 8, 9, 10)]
    out = ["| task | N | codes passing x < N (N as written) | N reversed also passes? | answered rows the file uses (T <= 2) "
           "| codes that also reproduce every one of those answers | the true code among them |", "|---|---|---|---|---|---|---|"]
    for t in order:
        c = cj[t]
        n = re.search(r"fixed_n_(\d+)", c["data"]).group(1)
        rev = pv[t]["codes_n_reversed"]["codes"]
        out.append(f"| {t} | {n} | {c['codes_passing_x_lt_n']:,} | {'yes, ' + format(rev, ',') + ' codes' if rev else 'no'} | "
                   f"{c['answered_rows']:,} | {c['codes_fitting_answers'] if c['answered_rows'] else '- (no answered rows)'} | "
                   f"{'yes' if c['true_code_fits'] else '-'} |")
    return "\n".join(out)


def times_table() -> str:
    """Seconds to every hundredth step (the evaluator's progress lines), against the best model's official run."""
    base = json.loads((HERE.parent / "runs" / "wider_menus" / "wide_anchored_official.json").read_text())
    recs = [("wide_anchored_official (the best model, H100)", base)] + \
        [(f"{p.stem}", json.loads(p.read_text())) for p in sorted(RECORDS.glob("*.json"))
         if p.stat().st_size > 1000 and not re.match(r"gen_[em]\d+_", p.stem)]   # the tiers stall early: not timed
    out = ["| run | device | frozen at step | s to step 200 | 300 | 400 | 500 | 600 | 700 | training s | evaluation s |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, rec in recs:
        so = rec.get("stdout", "")
        st = {int(a): float(b) for a, b in re.findall(r"^step=(\d+) .*elapsed=([\d.]+)s", so, flags=re.M)}
        fz = re.search(r"^LR frozen step=(\d+)", so, flags=re.M)
        seed = ((rec.get("result") or {}).get("seeds") or [{}])[0]
        dev = "H100" if "H100" in rec.get("gpu", "") or name.startswith("wide_anchored") else "CPU"
        cells = " | ".join(f"{st[k]:.0f}" if k in st else "-" for k in (200, 300, 400, 500, 600, 700))
        out.append(f"| {name} | {dev} | {fz.group(1) if fz else '-'} | {cells} | {seed.get('training_seconds', 0):.0f} | "
                   f"{seed.get('evaluation_seconds', 0):.0f} |")
    return "\n".join(out)


def fill(path: str) -> None:
    fs = records()
    tables = {"RUNS": HEAD + "\n" + "\n".join(row(f) for f in fs),
              "MINED": "| run | task | facts in force at the end (the model's own miner) |\n|---|---|---|\n"
                       + "\n".join(mined_row(f) for f in fs),
              "TINY_GEN": tiny_gen_table(), "TINY_TABLE": tiny_table_table(), "PREVIEW": preview_table(),
              "GLANCE": glance_table(), "CODES": codes_table(), "TIMES": times_table()}
    text = Path(path).read_text()
    for name, body in tables.items():
        text = re.sub(rf"<!-- {name} -->.*?<!-- /{name} -->", f"<!-- {name} -->\n{body}\n<!-- /{name} -->", text,
                      flags=re.S)
    Path(path).write_text(text)


if __name__ == "__main__":
    if sys.argv[1:2] == ["fill"]:
        fill(sys.argv[2])
    else:
        for f in records():
            print(json.dumps({k: f.get(k) for k in ("name", "task", "file", "score", "max_t", "ood_max_t", "steps",
                                                     "train_s", "eval_s", "selected", "selector", "round", "sets",
                                                     "gl_events", "sub_right", "mined")}), flush=True)
