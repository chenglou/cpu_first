"""The fixed-modulus study's records (general_lessons/fixed/records/*.json) as rows of facts and as FIXED_MODULUS.md's
tables. Everything general_lessons/summarize_general.py reads per record, plus v3's own lines: when the reader's
digit-order facts left the code ambiguous (LR digit_codes), the census (LR code_census), the survivors (LR
code_survivors), the code the answers chose (LR code_chosen), when the reader read with it (LR code_adopted), and the
reader's code and reading at evaluation (IO code, IO digit_tokens).

    python fixed/summarize_fixed.py                     # one line per record
    python fixed/summarize_fixed.py same REF.json NEW.json   # do two runs' training logs agree line for line
                                                        # (timings and v3's own lines left out)?
    python fixed/summarize_fixed.py fill FIXED_MODULUS.md     # rewrite the <!-- NAME --> ... <!-- /NAME --> tables
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORDS = HERE / "records"
_spec = importlib.util.spec_from_file_location("summarize_general", HERE.parent / "summarize_general.py")
SG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SG)
SG.FILES[hashlib.md5((HERE.parent / "submission_general_v3.py").read_bytes()).hexdigest()] = "v3"
N_OF = {"e1": 323, "e2": 899, "e6": 247, "e7": 287, "e8": 287, "e9": 299, "e10": 403, "m1": 10403, "m2": 38021,
        "m6": 1517, "m7": 1763, "m8": 1333, "m9": 1927, "m10": 1739}
V3_LINE = re.compile(r"^(LR (digit_codes|code_|mined)|CODES )")
TIMING = re.compile(r"(elapsed|seconds|t|time|s)=[0-9.]+s?|\d+\.\d+s")


def facts(rec: dict) -> dict:
    out = SG.facts(rec)
    so = rec.get("stdout", "")
    out["manifest_seeds"] = ((rec.get("manifest") or {}).get("runtime") or {}).get("seeds")
    g = rec.get("gpu") or ""
    out["gpu"] = next((k for k in ("B200", "H200", "H100") if k in g), "CPU" if not g else g.split(",")[0])
    ev = {}
    for l in so.splitlines():
        m = re.match(r"LR digit_codes ambiguous=True step=(\d+) readings=(\{.*\})", l)
        if m and "ambiguous" not in ev:
            ev["ambiguous"] = (int(m.group(1)), m.group(2))
        m = re.match(r"LR digit_codes pinned_by_facts step=(\d+) reading=(.*)", l)
        if m:
            ev["pinned"] = (int(m.group(1)), m.group(2))
        m = re.match(r"LR code_census step=(\d+) contestants=(\d+) readings=(\{.*\})", l)
        if m:
            ev.setdefault("census", []).append((int(m.group(1)), int(m.group(2)), m.group(3)))
        m = re.match(r"LR code_none_fit step=(\d+)", l)
        if m:
            ev.setdefault("none_fit", []).append(int(m.group(1)))
        m = re.match(r"LR code_chosen step=(\d+) reading=\((.*?)\) code=\[(.*?)\] survivors=(\d+) tie=(\S+) "
                     r"machine_runs=(\d+)", l)
        if m:
            ev["chosen"] = {"step": int(m.group(1)), "reading": m.group(2).replace(", ", ""),
                            "code": m.group(3).replace(", ", ""), "survivors": int(m.group(4)), "tie": m.group(5),
                            "runs": int(m.group(6))}
        m = re.match(r"LR code_adopted step=(\d+)", l)
        if m:
            ev["adopted"] = int(m.group(1))
    out["v3"] = ev
    shp = next((l for l in so.splitlines() if l.startswith("IO shape=")), None)
    out["shape"] = shp.split()[1].split("=")[1] if shp else None
    return out


def records(names=None) -> list:
    out = []
    for p in sorted(RECORDS.glob("*.json")):
        if names and p.stem not in names:
            continue
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out.append(facts(rec))
    return out


def show(f: dict) -> str:
    v = f.get("v3", {})
    ch = v.get("chosen") or {}
    sc = f"{f['score'] * 100:.1f}%" if f.get("score") is not None else f"failed (rc {f['rc']})"
    return (f"{f['name']} {f.get('task')} {f.get('file')} {f['gpu']} score={sc} maxT={f.get('max_t') or 0}/"
            f"{f.get('ood_max_t') or 0} steps={f.get('steps')} train={f.get('train_s') or 0:.0f}s "
            f"eval={f.get('eval_s') or 0:.0f}s events={f.get('gl_events')} ambiguous={v.get('ambiguous')} "
            f"census={v.get('census')} chosen={ch} adopted={v.get('adopted')} reading={f.get('reading')} "
            f"code={f.get('code')} code_right={f.get('code_right')} stopped={SG.stopped(f)}")


def log_lines(rec: dict) -> list:
    """The training log without timings and without v3's own lines (for `same`)."""
    out = []
    for l in rec.get("stdout", "").splitlines():
        if not l.startswith(("LR ", "TABLE ", "WIRING ", "LESSONS ", "MINED ", "LAYOUT ", "IO ", "AUDIT ")):
            continue
        if V3_LINE.match(l):
            continue
        out.append(TIMING.sub("", l))
    return out


def same(ref: str, new: str) -> None:
    a, b = (log_lines(json.loads(Path(p).read_text())) for p in (ref, new))
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print(f"lines: {len(a)} vs {len(b)}; differing: {len(diff)}")
    for i, x, y in diff[:10]:
        print(f"  {i}:\n    ref {x[:240]}\n    new {y[:240]}")


def tier_row(f: dict) -> str:
    v = f.get("v3", {})
    ch = v.get("chosen")
    cen = v.get("census") or []
    cens = f"{cen[0][1]:,} at step {cen[0][0]}" if cen else "-"
    if ch:
        tie = {"none": "", "readings": ", a tie of readings", "codes": ", a tie of codes"}[ch["tie"]]
        chose = (f"step {ch['step']}: reading {ch['reading']}, code {'true' if ch['code'] == '0123456789' else 'WRONG'}"
                 f" ({ch['survivors']} survivor{'s' if ch['survivors'] > 1 else ''}{tie})")
    else:
        chose = "-"
    sc = f"{f['score'] * 100:.1f}%" if f.get("score") is not None else f"failed (rc {f['rc']})"
    sp = ", ".join(f"{k} {x * 100:.1f}%" for k, x in sorted(f.get("splits", {}).items()))
    seed = (f.get("manifest_seeds") or [74])[0]
    return (f"| {f.get('task')} | {N_OF.get(f.get('task'), '-')} | {f.get('file')} | {seed} | {cens} | {chose} | "
            f"{v.get('adopted') or '-'} | {sc} | {sp} | {f.get('max_t') or 0} / {f.get('ood_max_t') or 0} | "
            f"{f.get('steps')} | {f.get('train_s') or 0:.0f} | {f.get('eval_s') or 0:.0f} | {f['gpu']} |")


TIER_HEAD = ("| task | N | build | seed | contestants (census step) | the answers chose | reader read with it at step | "
             "score | splits | max T seen / unseen | steps | training s | evaluation s | device |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


TASK_OF = {"323_t123": "e1", "899_t124": "e2", "247_t1234": "e6", "287_t123_ood4": "e7", "287_t123": "e8",
           "299_t123": "e9", "403_t123": "e10", "10403_t4816": "m1", "38021_t4816": "m2", "1517_t124": "m6",
           "1763_t124": "m7", "1333_t124": "m8", "1927_t124": "m9", "1739_t124": "m10"}


def task_of(data: str) -> str:
    return next(t for k, t in TASK_OF.items() if data.endswith("fixed_n_" + k))


def census_table() -> str:
    """fixed/census.jsonl (code_census.py): per dataset, the readings the facts leave and their codes, and what the
    answers leave."""
    old = json.loads((HERE.parent / "codes.json").read_text())
    rows = []
    for l in (HERE / "census.jsonl").read_text().splitlines():
        d = json.loads(l)
        t = task_of(d["data"])
        reads = ", ".join(f"{k}: {v['codes']:,}" for k, v in sorted(d["readings"].items(), key=lambda kv: kv[0]))
        fit = ", ".join(f"{k} + {''.join(c) if c != '0123456789' else 'true code'}" for k, c in d["fitting"])
        rows.append((list(N_OF).index(t), f"| {t} | {N_OF[t]} | {d['rows']:,} | {reads} | {d['candidates']:,} | "
                     f"{old[t]['codes_passing_x_lt_n']:,} | T <= {d['answer_T_max']}: {d['answer_rows']:,} | "
                     f"{d['n_fitting']} ({fit}) | {'yes' if d['true_fits'] else 'no'} |"))
    head = ("| task | N | training rows | readings the facts leave (N's run and direction, x's: codes) | whole candidates "
            "| code_search.py's count (N as written, x padded with the 0 token) | answer rows (v2's rule) | candidates "
            "that reproduce every answer row | the true one among them |\n|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(r for _, r in sorted(rows))


def tie_table() -> str:
    """fixed/tiny/tie_*_s*.json (tiny_tie.py): seeds exact, the step, and the entries right at the end."""
    groups = {}
    for p in sorted((HERE / "tiny").glob("tie_*_s[0-9].json")):
        d = json.loads(p.read_text())
        c = d["cfg"]
        key = (c["fixed"] or 0, c["tie"], c["code"])
        groups.setdefault(key, []).append(d)
    rows = []
    for (n, tie, code), ds in sorted(groups.items(), key=lambda kv: (kv[0][0] == 0, kv[0][0], kv[0][1], kv[0][2])):
        ex = [d["first_exact"] for d in ds]
        fin = ", ".join(f"{d['final'][1]}/{d['final'][2]}/{d['final'][3]}" for d in ds)
        rows.append(f"| {n or 'many (4-5 digits)'} | {'the reader ties the borrows' if tie else 'the facts only'} | "
                    f"{'true' if code == 'true' else 'a wrong one that passes x < N'} | {sum(e is not None for e in ex)}"
                    f"/{len(ds)} | {', '.join(str(e) if e else '-' for e in ex)} | {fin} |")
    head = ("| modulus | the table's borrow lessons | the reader's code | seeds exact | step exact | entries right at the "
            "end (digit, borrow, both) |\n|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows)


def tourn_table() -> str:
    """fixed/tiny/tourn_{hard,soft}_*.json (tiny_tournament.py)."""
    rows = []
    for p in sorted((HERE / "tiny").glob("tourn_*.json")):
        d = json.loads(p.read_text())
        t = task_of(d["data"])
        st = d["steps"]
        if d["mode"] == "hard":
            left = st[-1]["contestants_alive"]
        else:
            left = st[-1]["contestants_active"]
        when = d.get("steps_to_choose") or next((x["step"] for x in st if x.get("contestants_active", 9) <= 2), "-")
        alive = st[0].get("sets_rounds_alive") if d["mode"] == "hard" else None
        sr = ", ".join(f"{a} + {'/'.join(map(str, b))}" for a, b in alive) if alive else "-"
        code = "".join(map(str, d.get("leader_code") or [])) or "-"
        rows.append((d["mode"], list(N_OF).index(t),
                     f"| {t} | {d['mode']} | {d['census']:,} | {left} | {when} | {d.get('leader_reading', '-')} | "
                     f"{'true' if code == '0123456789' else code} | {sr} | {d['machine_runs']:,} | {d['seconds']} |"))
    head = ("| task | weights | contestants (census) | contestants left | batch it ended at | reading chosen | code chosen "
            "| (set, round) left after batch 1 | machine runs | s (laptop CPU) |\n|---|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(r for _, _, r in sorted(rows))


def at_seconds(rec: dict, step) -> int | None:
    """Training seconds at a step, interpolated between the evaluator's progress lines (every 100 steps)."""
    if step is None:
        return None
    el = {int(m.group(1)): float(m.group(2)) for m in
          re.finditer(r"^step=(\d+) loss=\S+ accuracy=\S+ elapsed=([\d.]+)s", rec.get("stdout", ""), re.M)}
    lo = max((k for k in el if k <= step), default=None)
    hi = min((k for k in el if k >= step), default=None)
    if lo is None or hi is None:
        return None
    return round(el[lo] if hi == lo else el[lo] + (el[hi] - el[lo]) * (step - lo) / (hi - lo))


def times_table() -> str:
    """When each final fixed-modulus run reached each stage (step, training seconds on its GPU)."""
    rows = []
    for p in sorted(RECORDS.glob("*.json")):
        rec = json.loads(p.read_text())
        f = facts(rec)
        if f.get("task") not in N_OF or f.get("file") != "v3":
            continue
        ev, v = f.get("gl_events", {}), f.get("v3", {})
        stages = [ev.get("io4"), ev.get("order_digits"), ev.get("chain_lesson_joins"),
                  (v.get("chosen") or {}).get("step"), v.get("adopted"), ev.get("frozen")]
        cells = " | ".join("-" if st is None else f"{st} ({at_seconds(rec, st)} s)" for st in stages)
        seed = (f.get("manifest_seeds") or [74])[0]
        rows.append((list(N_OF).index(f["task"]), seed, f"| {f['task']} | {seed} | {cells} | {f.get('train_s') or 0:.0f} |"))
    head = ("| task | seed | reading chosen | digit lessons | phase C | code chosen | reader read with it | frozen | "
            "training s |\n|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(r for _, _, r in sorted(rows))


def fill(doc: Path) -> None:
    fs = records()
    key = lambda f: (list(N_OF).index(f["task"]), (f.get("manifest_seeds") or [74])[0] != 74, f["name"])
    tables = {"TIERS": TIER_HEAD + "\n" + "\n".join(tier_row(f) for f in sorted(
        (f for f in fs if f.get("task") in N_OF and f.get("file") == "v3"), key=key)),
              "TIERS_EARLIER": TIER_HEAD + "\n" + "\n".join(tier_row(f) for f in sorted(
        (f for f in fs if f.get("task") in N_OF and f.get("file") != "v3"), key=key)),
              "CENSUS": census_table(), "TIE": tie_table(), "TOURN": tourn_table(), "TIMES": times_table(),
              "H1": SG.HEAD + "\n" + "\n".join(SG.row(f) for f in fs if f.get("task") == "h1"),
              "OTHER": SG.HEAD + "\n" + "\n".join(SG.row(f) for f in fs
                                                    if f.get("task") not in N_OF and f.get("task") != "h1")}
    text = doc.read_text()
    for k, v in tables.items():
        text = re.sub(rf"<!-- {k} -->.*?<!-- /{k} -->", f"<!-- {k} -->\n{v}\n<!-- /{k} -->", text, flags=re.S)
    doc.write_text(text)


if __name__ == "__main__":
    if sys.argv[1:2] == ["same"]:
        same(sys.argv[2], sys.argv[3])
    elif sys.argv[1:2] == ["fill"]:
        fill(Path(sys.argv[2]))
    else:
        for f in records(sys.argv[1:] or None):
            print(show(f))
