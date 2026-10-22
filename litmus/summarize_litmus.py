"""The litmus records (cpu_first/litmus/records/*.json) as rows of facts and as LITMUS.md's tables.

    python summarize_litmus.py                 # one line per record
    python summarize_litmus.py fill LITMUS.md  # rewrite the <!-- NAME --> ... <!-- /NAME --> tables in place

Per record: the score (mean exact accuracy over the scored splits), every split's accuracy, the certified T on seen and
unseen moduli and each rung's count, training and evaluation seconds and steps; from the log: the schedule (IO.md's
stages, phase B, the digit lessons, the ladder, the chain lesson, the I/O lesson, T's direction, the freeze), the last
progress line (phase, the hard checks, the answers of the T <= 2 rows), the snapped tables against true subtraction and
multiplication, the wiring chosen, the reader's reading, code and T pointer, and the coverage audit.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORDS = HERE / "records"
BEST_MD5 = "c82b11fa8a93b09c0724c3931cf8de98"
TD = [(r - m - c) % 10 for r in range(10) for m in range(10) for c in range(2)]
TB = [int(r - m - c < 0) for r in range(10) for m in range(10) for c in range(2)]
PLO = [(a * b) % 10 for a in range(10) for b in range(10)]
PHI = [(a * b) // 10 for a in range(10) for b in range(10)]
VARIANTS = {"c82b11fa8a93b09c0724c3931cf8de98": "the file"}


def line(stdout: str, prefix: str) -> str | None:
    return next((l for l in stdout.splitlines() if l.startswith(prefix)), None)


def facts(rec: dict) -> dict:
    out = {"name": rec["name"], "task": rec.get("task"), "rc": rec["returncode"], "gpu": "H100" if "H100" in rec.get("gpu", "")
           else "CPU", "md5": hashlib.md5(rec["submission"].encode()).hexdigest()}
    m = rec.get("manifest") or {}
    out["budget"] = m.get("runtime", {}).get("total_training_time_seconds")
    out["max_steps"] = m.get("runtime", {}).get("max_steps")
    out["official_manifest"] = rec.get("manifest_file") is not None
    r = rec.get("result") or {}
    seed = (r.get("seeds") or [{}])[0]
    out["score"] = (r.get("score") or {}).get("mean_exact_accuracy")
    out["splits"] = {k: v["exact_accuracy"] for k, v in seed.get("evaluation", {}).items()}
    dp = r.get("depth_profile") or {}
    out["max_t"], out["ood_max_t"] = dp.get("max_certified_time_steps"), dp.get("ood_n_max_certified_time_steps")
    sdp = seed.get("depth_profile") or {}
    out["rungs"] = [(g["time_steps"], g["correct_examples"], g["example_count"]) for g in sdp.get("rungs", [])]
    out["ood_rungs"] = [(g["time_steps"], g["correct_examples"], g["example_count"]) for g in sdp.get("ood_n_rungs", [])]
    so = rec.get("stdout", "")
    steps = re.findall(r"^step=(\d+) .*elapsed=([\d.]+)s", so, flags=re.M)
    out["steps"] = seed.get("completed_training_steps") or (int(steps[-1][0]) if steps else None)
    out["train_s"], out["eval_s"] = seed.get("training_seconds"), seed.get("evaluation_seconds")
    if out["train_s"] is None and steps:
        out["train_s"] = float(steps[-1][1])
    sched = line(so, "TABLE schedule")
    hist = []
    if sched:
        try:
            hist = ast.literal_eval(sched.split("history=", 1)[1])
        except (ValueError, SyntaxError):
            hist = []
    ev = {}
    for h in hist:
        ev.setdefault(h[1], h[0])
        if h[1] == "io3_ranked":
            out["ranked"] = h[2]
    out["events"] = ev
    out["phase_end"] = re.search(r"phase=(\w+) io=(\d)", sched).groups() if sched else None
    if sched and not seed.get("completed_training_steps"):                 # no result: the model's own step count
        out["steps"] = int(re.search(r" step=(\d+)", sched).group(1))
    last = [l for l in so.splitlines() if l.startswith("LR step=")]
    if last:
        kv = dict(re.findall(r"(\w+)=([^\s]+)", last[-1].split(" layout:")[0]))
        out["last"] = {k: kv.get(k) for k in ("step", "phase", "io", "hard_exact", "table_exact", "ladder_ok", "div_ok",
                                               "law_ok", "step_ok", "ans_ok", "io_exact", "chosen")}
    for key, truth in (("digit", TD), ("borrow", TB), ("prod_lo", PLO), ("prod_hi", PHI)):
        l = line(so, f"TABLE {key}=")
        if l:
            v = l.split("=", 1)[1].strip()
            out[f"{key}_right"] = sum(int(a) == b for a, b in zip(v, truth))
    sub = (min(out.get("digit_right", 0), out.get("borrow_right", 0)) if "digit_right" in out else None)
    if "digit_right" in out:
        dv = line(so, "TABLE digit=").split("=", 1)[1].strip()
        bv = line(so, "TABLE borrow=").split("=", 1)[1].strip()
        out["sub_right"] = sum(int(a) == b and int(c) == d for a, b, c, d in zip(dv, TD, bv, TB))
        lv = line(so, "TABLE prod_lo=").split("=", 1)[1].strip()
        hv = line(so, "TABLE prod_hi=").split("=", 1)[1].strip()
        out["prod_right"] = sum(int(a) == b and int(c) == d for a, b, c, d in zip(lv, PLO, hv, PHI))
    g = line(so, "TABLE gate=")
    out["gate"] = g.split("=", 1)[1].split()[0] if g else None
    w = line(so, "WIRING a=") or line(so, "WIRING op=")
    if w:
        out["wiring"] = w.removeprefix("WIRING ").split(" true=")[0]
        out["chain"] = w.split("chain=")[-1].replace("_", " ")
        out["wiring_true"] = "true=True" in w
        added = re.search(r"\b(plus1|cube|times10)=(\S+)", w)
        out["switch"] = f"{added.group(1)}={added.group(2)}" if added else None
    lay = line(so, "LAYOUT right=")
    if lay:
        out["layout_right"] = lay.split()[1].split("=")[1]
        out["numbers"] = re.search(r"numbers=(\S+)", lay).group(1).replace("_", " ")
        out["layout"] = re.search(r"(lo=\S+)\s", lay).group(1)
    io = line(so, "IO digit_tokens=")
    if io:
        rd = re.search(r"reading=(\(.*?\)|None)", io).group(1)
        out["reading"] = rd
    code = line(so, "IO code=")
    if code:
        pairs = [tuple(map(int, p.split(":"))) for p in code.split("=", 1)[1].split(",") if p]
        out["code"] = "".join(str(v) for _, v in sorted(pairs))
        out["code_right"] = sum(v == t - 7 for t, v in pairs)
    for f in ("N", "x", "T", "writer"):
        l = line(so, f"IO field={f} ")
        if l:
            out[f"field_{f}"] = " ".join(l.split()[2:4])
    shp = line(so, "IO shape=")
    out["shape"] = shp.split()[1].split("=")[1] if shp else None
    a = line(so, "AUDIT eval_rows_used=")
    if a and "used_but_never_trained" in a:
        out["never_trained"] = int(re.search(r"used_but_never_trained=(\d+)", a).group(1))
    fz = line(so, "LR frozen")
    out["frozen"] = int(re.search(r"step=(\d+)", fz).group(1)) if fz else None
    out["error"] = (rec.get("stderr") or "").strip().splitlines()[-1][:160] if rec["returncode"] else ""
    out["counts"] = {k: round(v["exact_accuracy"] * v["example_count"]) for k, v in seed.get("evaluation", {}).items()
                     if "example_count" in v}
    out["counts"].update({f"depth_t_{t}": c for t, c, _ in out["rungs"]})
    out["counts"].update({f"depth_ood_n_t_{t}": c for t, c, _ in out["ood_rungs"]})
    return out


def as_h1_machine(f: dict) -> str | None:
    """Whether the run's right rows equal, split by split and rung by rung, those of an exact H1 machine (x^(2^T) mod N,
    scored on the answer's own digits; agreement.json)."""
    path = HERE / "agreement.json"
    if not path.exists() or f.get("task") not in json.loads(path.read_text()) or not f["counts"]:
        return None
    a = json.loads(path.read_text())[f["task"]]
    diff = {k: (c, a[k][0]) for k, c in f["counts"].items() if k in a and a[k][0] != c}
    return "yes, on every split and rung" if not diff else f"no: {diff}"


def load(names=None) -> list[dict]:
    paths = sorted(RECORDS.glob("*.json"))
    return [facts(json.loads(p.read_text())) for p in paths if names is None or p.stem in names]


def show(f: dict) -> str:
    e = f.get("events", {})
    return (f"{f['name']}: rc={f['rc']} score={f['score']} maxT={f['max_t']}/{f['ood_max_t']} steps={f['steps']} "
            f"train_s={f['train_s']} eval_s={f['eval_s']} phase,io={f.get('phase_end')} events={e} "
            f"sub={f.get('sub_right')} prod={f.get('prod_right')} gate={f.get('gate')} chain={f.get('chain')} "
            f"numbers={f.get('numbers')} code={f.get('code')} reading={f.get('reading')} T={f.get('field_T')} "
            f"last={f.get('last')} splits={ {k: round(v, 4) for k, v in f['splits'].items()} } {f['error']}")


def pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%" if v < 0.9995 or v == 1 else f"{100 * v:.2f}%"


def stopped(f: dict) -> str:
    """Where a run's schedule got to (the stages of the file's docstring), in words."""
    e, last = f.get("events", {}), f.get("last") or {}
    if f["rc"] and not e:
        return "nothing learned before the deadline"
    if "frozen" in e:
        return f"froze at step {e['frozen']}"
    ph, io = (f.get("phase_end") or ("?", "?"))
    if f.get("steps") and int(last.get("step") or 0) == 1 < f["steps"]:  # only step 1 logged (a line every 250 steps)
        stage = {"1": "finding three numbers", "2": "ranking the digits by x < N", "3": "choosing a reading",
                 "4": "reading chosen"}.get(io, io)
        return f"deadline at step {f['steps']}: phase {ph}, the reader {stage}"
    hard, ans = float(last.get("hard_exact") or 0), float(last.get("ans_ok") or 0)
    if ph == "C" and hard == 1 and ans == 1:
        return "exact on every check and answer; T's direction never settled, so no freeze"
    if ph == "C" and hard == 1:
        return f"exact on its own checks; {pct(ans)} of the T <= 2 answers right"
    if ph == "C":
        return f"wiring lessons on, checks {pct(hard)} exact"
    if ph == "B":
        return f"subtraction table never exact ({pct(float(last.get('table_exact') or 0))} of its checks pass)"
    if io == "1":
        return "the reader never found exactly three numbers in a prompt (its first stage)"
    if io == "2":
        return "no reading passed x < N on every row (the reader's second stage)"
    if io == "3":
        return f"reading never chosen (posterior {float(last.get('chosen') or 0):.3f} < 0.99)"
    return f"stage {io} of the reader"


def chain_name(f: dict) -> str:
    """The run's own label for its chain, with Montgomery's two forms named (the file labels them "other"; the operands
    of an addition may come in either order)."""
    w = dict(kv.split("=", 1) for kv in (f.get("wiring") or "").split() if "=" in kv)
    ab, op, flag = {w.get("a"), w.get("b")}, w.get("op"), w.get("flag")
    if op == "add" and w.get("keep") == "gated":
        if ab == {"D", "M"} and flag == "zD":
            return "Montgomery, repeated addition"
        if ab == {"Rp", "Mq"} and flag == "zE":
            return "Montgomery, trial multiples (the sum's digit tested)"
        if ab == {"Rp", "Mq"} and flag == "zD":
            return "Montgomery, trial multiples (the register's digit tested)"
    return f["chain"]


def chose(f: dict) -> str:
    if not f.get("chain") or "chain" not in f.get("events", {}):           # the wiring lessons never joined
        return "-"
    bits = [chain_name(f), f.get("numbers") or ""]
    return ", ".join(b for b in bits if b)


def rungs(f: dict) -> str:
    return f"{f['max_t'] or 0} / {f['ood_max_t'] or 0}"


def splits(f: dict) -> str:
    return ", ".join(f"{k} {pct(v)}" for k, v in sorted(f["splits"].items())) or "-"


def table(head: list, rows: list) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fill(doc: Path) -> None:
    recs = {f["name"]: f for f in load()}
    audit = json.loads((HERE / "audit.json").read_text())
    tiers = [f"e{i}" for i in range(1, 11)] + [f"m{i}" for i in range(1, 11)]
    blocks = {}
    rows = []
    for t in tiers + ["h1", "xfirst", "mont", "ten", "rho", "cube", "xy"]:
        a = audit[t]
        Ts = ", ".join(a["train_T"])
        mod = "1 (fixed)" if a["moduli"] == 1 else f"{a['moduli']:,}"
        rows.append([t.upper() if t in tiers else t, f"{a['train_rows']:,}", mod,
                     ", ".join(f"{k}: {v}" for k, v in a["modulus_digits"].items()), a["layout"],
                     f"{a['adjacent_pairs']}/9", "yes" if a["rows_rejecting_rev_n"] else "no",
                     "yes" if a["rows_rejecting_rev_x"] else "no", Ts, f"{a['answer_rows']:,}",
                     f"{a['t_two_way_rows']:,}"])
    blocks["AUDIT"] = table(["task", "training rows", "moduli", "their digits (digits: moduli)", "layout",
                             "digit order x < N pins (adjacent pairs)", "N's direction pinned", "x's direction pinned",
                             "T in training", "rows with answers used (T = 1, 2)", "rows with a two-digit T"], rows)
    rows = []
    for t in tiers:
        a, o, r = audit[t], recs.get(f"{t}_official"), recs.get(f"{t}_relaxed")
        off = "-" if o is None else ("failed: " + o["error"].split(":")[0] + f" ({o['steps']} steps)" if o["rc"]
                                     else f"{pct(o['score'])} ({o['steps']} steps)")
        rel = "-" if r is None else (f"failed: {r['error']}" if r["rc"] else pct(r["score"]))
        rows.append([t.upper(), "fixed" if a["moduli"] == 1 else f"{a['moduli']} moduli", ", ".join(a["train_T"]),
                     off, rel, "-" if r is None else splits(r), "-" if r is None else rungs(r),
                     "-" if r is None else stopped(r), "-" if r is None else chose(r)])
    blocks["TIERS"] = table(["task", "modulus", "T in training", "official manifest: score", "relaxed: score",
                             "relaxed: splits", "relaxed: max T seen / unseen", "relaxed: where it stopped",
                             "relaxed: chose"], rows)
    rows = []
    for name in sorted(recs):
        f = recs[name]
        if f["task"] in tiers and name.endswith(("_official", "_relaxed")):
            continue
        rows.append([name, f["task"], VARIANT_FILES.get(f["md5"], f["md5"][:8]), pct(f["score"]), splits(f), rungs(f),
                     f["steps"], f"{f['train_s']:.0f}" if f["train_s"] else "-",
                     f"{f['eval_s']:.0f}" if f["eval_s"] else "-", stopped(f), chose(f),
                     f.get("switch") or "-"])
    blocks["RUNS"] = table(["run", "task", "file", "score", "splits", "max T seen / unseen", "steps", "training s",
                            "evaluation s", "where it stopped", "chose", "added switch"], rows)
    rows = []
    for t, unchanged, change, fixed in SUMMARY:
        u = recs.get(unchanged)
        fx = [recs[n] for n in fixed if n in recs]
        res = lambda f: "-" if f is None else (f"failed ({f['error'].split(':')[0]})" if f["rc"] else
                                              f"{pct(f['score'])}, max T {rungs(f)}")
        rows.append([t, res(u), "-" if u is None else stopped(u), change,
                     "; ".join(f"{res(f)} ({f['name']})" for f in fx) or "-"])
    blocks["SUMMARY"] = table(["task", "unchanged: score, max T seen / unseen", "unchanged: where it stopped",
                               "smallest change that made it work", "runs with that change (fixed moduli: the diagnostics of §4)"],
                              rows)
    text = doc.read_text()
    for name, body in blocks.items():
        text, n = re.subn(rf"<!-- {name} -->.*?<!-- /{name} -->", f"<!-- {name} -->\n{body}\n<!-- /{name} -->", text,
                          flags=re.S)
        if n == 0:
            print("no block", name)
    doc.write_text(text)


VARIANT_FILES = {}
# task, the unchanged run, the smallest change found (in words), the runs with that change
FIXED = "none found: not even a given, frozen reader with hand-written arithmetic (§4)"
SAMPLED = "none (more time than the tier's budget)"
SUMMARY = [
    ("E1", "e1_relaxed", FIXED, ["e1_reader_frozen", "e1_warm_frozen"]),
    ("E2", "e2_relaxed", FIXED, ["e2_reader_given", "e2_reader_frozen", "e2_warm_frozen"]),
    ("E3", "e3_relaxed", SAMPLED, []), ("E4", "e4_relaxed", SAMPLED, []), ("E5", "e5_relaxed", SAMPLED, []),
    ("E6", "e6_relaxed", FIXED, []), ("E7", "e7_relaxed", FIXED, []), ("E8", "e8_relaxed", FIXED, []),
    ("E9", "e9_relaxed", FIXED, []), ("E10", "e10_relaxed", FIXED, ["e10_reader_frozen"]),
    ("M1", "m1_relaxed", FIXED, ["m1_reader_given", "m1_reader_frozen", "m1_warm_frozen"]), ("M2", "m2_relaxed", FIXED, []),
    ("M3", "m3_relaxed", SAMPLED, []), ("M4", "m4_relaxed", SAMPLED, []),
    ("M5", "m5_relaxed", "one tie: T's pointer reads the way the writer writes, its anchor and step "
     "(variants/submission_tie_writer.py), 4 of 4 seeds; tying only its step to x's (variants/submission_tie.py) fixed 3 "
     "of 4", ["m5_tiew_s74", "m5_tiew_s1", "m5_tiew_s2", "m5_tiew_s3", "tiew_on_h1"]),
    ("M6", "m6_relaxed", FIXED, ["m6_reader_frozen", "m6_warm_frozen"]), ("M7", "m7_relaxed", FIXED, []), ("M8", "m8_relaxed", FIXED, []),
    ("M9", "m9_relaxed", FIXED, []), ("M10", "m10_relaxed", FIXED, []),
    ("xfirst", "xfirst_s74", "none (seeds 1-3 too)", ["xfirst_s1", "xfirst_s2", "xfirst_s3"]),
    ("mont", "mont_s74", "its three division lessons restated for Montgomery's reduction (variants/submission_mont.py: "
     "2 of 4 seeds), plus a fourth: the division ladder also checks the gate's role in the final correction "
     "(variants/submission_mont_gate.py)", ["mont_fix2_s74", "mont_fix2_s1", "mont_fix2_s2", "mont_fix2_s3",
                                             "mont_gate_s74", "mont_gate_s1", "mont_gate_s2", "mont_gate_s3"]),
    ("ten", "ten_s74", "one switch: the square times 10 before the reduction (variants/submission_ten.py)",
     ["ten_fix_s74", "ten_fix_on_h1"]),
    ("rho", "rho_s74", "one switch: the square plus 1 before the reduction (variants/submission_rho.py)",
     ["rho_fix_s74", "rho_fix_on_h1"]),
    ("cube", "cube_s74", "one round option: times the round's input, reduced again; the multiply takes a second "
     "operand (variants/submission_cube.py)", ["cube_fix_s74", "cube_fix_on_h1"]),
    ("xy", "xy_s74", "a redesign, not tried: a reader for four numbers and a round that multiplies by y", []),
]


def variant_md5s() -> None:
    root = HERE.parent
    for p in [root / "wider_menus" / "submission_wide_anchored.py", *sorted((HERE / "variants").glob("*.py"))]:
        VARIANT_FILES[hashlib.md5(p.read_bytes()).hexdigest()] = (
            "the file" if p.parent.name == "wider_menus" else p.name.removeprefix("submission_").removesuffix(".py"))


if __name__ == "__main__":
    variant_md5s()
    if len(sys.argv) > 1 and sys.argv[1] == "fill":
        fill(Path(sys.argv[2]))
        raise SystemExit
    for f in load(sys.argv[1:] or None):
        print(show(f))
