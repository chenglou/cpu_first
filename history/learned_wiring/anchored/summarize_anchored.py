"""Tables of ANCHORED.md from the run records (cpu_first/history/runs/learned_wiring/<name>.json) and the diagnostic
logs (anchored/logs/<name>.log).

    python summarize_anchored.py runs [TAG ...]     # one row per record: TAG_official and TAG_cpu_seed74, 1..7
                                                    # (default: anchored and io3v2, stage 3's certified file)
    python summarize_anchored.py diag NAME [...]    # one row per diagnostic log (local_anchored.py's lines)
    python summarize_anchored.py audit [TAG ...]    # the coverage audit of each record
    python summarize_anchored.py times              # seconds to a training step in the official runs
    python summarize_anchored.py tlesson NAME [...] # T's lesson against phase B's reset (logs with gradient norms)
    python summarize_anchored.py fill ../ANCHORED.md   # rewrite the write-up's generated tables

From each record (summarize_wiring.parse): the evaluator's RESULT_JSON, the snapped tables against true subtraction and
multiplication, the division's wiring, the layout, the shape, the evaluation audits, the md5 of the exact source; and
the schedule from the `LR frozen ... history=` line (the step of each event).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
import summarize_wiring as sw  # noqa: E402

EVENTS = ("io4_reading", "order_digits", "switch", "ladder", "pieces", "chain", "wire_io", "T_resolved")


def schedule(stdout: str) -> dict:
    hist = next((l for l in stdout.splitlines() if l.startswith("LR frozen")), "")
    out = {}
    for k in EVENTS:
        m = re.search(r"\((\d+), '" + k + "'", hist)
        out[k] = m.group(1) if m else "-"
    return out


def runs_table(tags) -> None:
    print("| run | seed | max T seen / unseen | rungs (fewest right of 768, seen / unseen) | test, ood_t, ood_n_t | "
          "subtraction, product, gate | division wiring | layout | shape | reading chosen | digit lessons | phase B | "
          "ladder | pieces' lessons | chain lesson | I/O lesson | T settled | frozen | never trained | training s | "
          "evaluation s |")
    print("|---" * 21 + "|")
    for tag in tags:
        for name in [f"{tag}_official"] + [f"{tag}_cpu_seed{s}" for s in (74, 1, 2, 3, 4, 5, 6, 7)]:
            p = sw.RUNS / f"{name}.json"
            if not p.exists():
                print(f"| {name} | (no record) |")
                continue
            rec = json.loads(p.read_text())
            r = sw.parse(rec)
            ev = schedule(rec.get("stdout", ""))
            where = "H100" if "H100" in r["gpu"] else "CPU"
            ev_s = ", ".join(str(r["eval"].get(k)) for k in ("test", "ood_t", "ood_n_t"))
            true = "true" if r["wiring"].endswith("true=True") else r["wiring"]
            print(f"| {name} ({where}) | {r['seed']} | {r['maxT']} / {r['oodN']} | {min(r['rungs'] or [0])} / "
                  f"{min(r['ood_rungs'] or [0])} | {ev_s} | {r['rows']}/200, {r['cells']}/100, {r['gate']} | {true} | "
                  f"{r['layout']} | {r['shape']} | " + " | ".join(ev[k] for k in EVENTS) +
                  f" | {r['frozen']} | {r['never']} | {r['train_s']} | {r['eval_s']} |")


def audit_table(tags) -> None:
    """The coverage audit (CHAIN.md): how often training's checks used each table row and cell (the TABLE lines),
    and the largest count of rows, cells and gate rows that evaluation used but training never did (the AUDIT
    lines, every rung)."""
    print("| run | subtraction rows used at evaluation | fewest training visits of a row | product cells used at "
          "evaluation | fewest training visits of a cell | count-down rows used | gate rows used | used at evaluation "
          "but never trained (rows, cells, count-down, gate) |")
    print("|---" * 8 + "|")
    for tag in tags:
        for name in [f"{tag}_official"] + [f"{tag}_cpu_seed{s}" for s in (74, 1, 2, 3, 4, 5, 6, 7)]:
            p = sw.RUNS / f"{name}.json"
            if not p.exists():
                continue
            lines = json.loads(p.read_text()).get("stdout", "").splitlines()
            grab = lambda pat: [int(x) for l in lines if l.startswith("AUDIT") for x in re.findall(pat, l)]
            vmin = lambda key: next((re.search(r"min=(\d+)", l).group(1) for l in lines
                                     if l.startswith(f"TABLE {key} ")), "-")
            rows_used = max(grab(r"eval_rows_used=(\d+)") or [0])
            cells_used = max(grab(r"product_cells_used=(\d+)") or [0])
            never_rows = max(grab(r"eval_rows_used=\d+/200 used_but_never_trained=(\d+)") or [0])
            never_cells = max(grab(r"product_used_but_never_trained=(\d+)") or [0])
            never_count = max(grab(r"countdown_rows_used=\d+ used_but_never_trained=(\d+)") or [0])
            never_gate = max(grab(r"gate_rows_used=\d/2 used_but_never_trained=(\d+)") or [0])
            count_used = max(grab(r"countdown_rows_used=(\d+)") or [0])
            gate_used = max(grab(r"gate_rows_used=(\d)") or [0])
            print(f"| {name} | {rows_used}/200 | {vmin('train_visits')} | {cells_used}/100 | "
                  f"{vmin('prod_train_visits')} | {count_used} | {gate_used}/2 | "
                  f"{never_rows}, {never_cells}, {never_count}, {never_gate} |")


def md5s(tags) -> None:
    """The md5 of every record's embedded source, against the files."""
    files = {hashlib.md5((HERE.parent / f).read_bytes()).hexdigest(): f
             for f in ("submission_anchored.py", "submission_io3.py")}
    for tag in tags:
        for p in sorted(sw.RUNS.glob(f"{tag}_*.json")):
            rec = json.loads(p.read_text())
            h = hashlib.md5(rec.get("submission", "").encode()).hexdigest()
            print(f"{p.stem}: md5 {h} = {files.get(h, 'no file')}")


def diag_row(name: str) -> str:
    """A diagnostic log: when each event happened and what the reader and the table were then."""
    lines = (HERE / "logs" / f"{name}.log").read_text().splitlines()
    comp = [l for l in lines if l.startswith("s ")]                    # every 5 steps, before the step's lessons
    num = lambda l, k: re.search(k + r"[ =](\S+)", l).group(1)
    step = lambda l: int(l.split()[1])
    first = lambda pred: next((step(l) for l in comp if pred(l)), None)
    hist = next((l for l in lines if l.startswith("history")), "")
    ev = {k: (re.search(r"\((\d+), '" + k + "'", hist) or [None, "-"])[1] for k in EVENTS + ("frozen",)}
    io4 = int(ev["io4_reading"]) if ev["io4_reading"] != "-" else 10 ** 9
    n0 = max(float(num(l, "N_read_0")) for l in comp if step(l) < io4)   # before the table's first lesson
    n0_lines = sum(float(num(l, "N_read_0")) >= 0.9 for l in comp if step(l) < io4)
    misread = [l for l in comp if float(num(l, "N,x")) < 1]
    right_from = step(misread[-1]) + 5 if misread else 1                 # every logged row right from here on
    worst = max([1 - float(num(l, "N,x")) for l in comp if step(l) >= io4] or [0])
    b200 = first(lambda l: num(l, "borrows") == "200/200")
    d200 = first(lambda l: num(l, "digits") == "200/200")
    ex = first(lambda l: num(l, "table_exact") == "1.0000")             # the table's hard checks all pass
    ex_fail = sum(step(l) > (ex or 10 ** 9) and num(l, "table_exact") != "1.0000" and num(l, "phase") != "frozen"
                  for l in comp)
    rungs = [l for l in lines if l.startswith("depth_")]
    exact = sum("index=1.0000 ste=1.0000 identical_logits=True" in l for l in rungs)
    grad = next((l.split("every_parameter_gets_gradient=")[1] for l in lines if l.startswith("GRADTEST splits")), "-")
    return (f"| {name} | {n0:.2f} ({n0_lines}) | {right_from} | {ev['io4_reading']} | {worst:.4f} | {b200} | "
            f"{ev['order_digits']} | {d200} | {ex} ({ex_fail}) | {ev['switch']} | {ev['ladder']} | {ev['pieces']} | "
            f"{ev['chain']} | {ev['wire_io']} | {ev['T_resolved']} | {ev['frozen']} | {exact}/{len(rungs)} | {grad} |")


def diag_table(names) -> None:
    print("| run | most rows read as N = 0 before the reading (logged steps with 90% or more) | N and x read right on "
          "every row from | reading chosen: the table's first lesson | most rows misread after it | borrows 200/200 | "
          "digit lessons | digits 200/200 | the table's checks all pass from (logged steps failing after) | phase B | "
          "ladder | pieces' lessons | chain lesson | I/O lesson | T settled | frozen | rungs exact on both paths, "
          "identical logits | gradient test |")
    print("|---" * 18 + "|")
    for name in names:
        print(diag_row(name))


def times_table() -> None:
    """Seconds to a training step (the evaluator's progress lines) in the official runs."""
    recs = [("anchored_official", sw.RUNS / "anchored_official.json"),
            ("io3v2_official", sw.RUNS / "io3v2_official.json"),
            ("cmbanchors_official", HERE.parents[2] / "anchors" / "records" / "cmbanchors_official.json")]
    marks = (100, 200, 300, 400, 500, 600, 1000)
    print("| run | frozen at step | T settled at step | " + " | ".join(f"s to step {m}" for m in marks)
          + " | training s | evaluation s |")
    print("|---" * (len(marks) + 5) + "|")
    for name, p in recs:
        rec = json.loads(p.read_text())
        out = rec["stdout"].splitlines()
        prog = [re.match(r"\s*step=(\d+) .*elapsed=([\d.]+)s", l) for l in out]
        el = {int(m.group(1)): m.group(2) for m in prog if m}
        frozen = next((re.search(r"step=(\d+)", l).group(1) for l in out if l.startswith("LR frozen")), "-")
        tset = next((re.search(r"step=(\d+)", l).group(1) for l in out if l.startswith("LR io_stage=5")), "-")
        seed = rec["result"]["seeds"][0]
        print(f"| {name} | {frozen} | {tset} | " + " | ".join(el.get(m, "-") for m in marks)
              + f" | {seed['training_seconds']:.1f} | {seed['evaluation_seconds']:.1f} |")


def log_path(name: str) -> Path:
    """A diagnostic log: anchored/logs/, or WIRING.md's local/logs/ (its d13 runs of stage 3)."""
    p = HERE / "logs" / f"{name}.log"
    return p if p.exists() else HERE.parent / "local" / "logs" / f"{name}.log"


def t_row(name: str) -> str:
    """T's lesson in a diagnostic log with gradient norms: when phase B cleared Adam's state, T's pointer gradient
    (anchor and step logits) around it and when T's lesson starts, and how long the lesson took."""
    lines = log_path(name).read_text().splitlines()
    hist = next((l for l in lines if l.startswith("history")), "")
    ev = {k: int(m.group(1)) for k in ("io4_reading", "switch", "wire_io", "T_resolved")
          for m in [re.search(r"\((\d+), '" + k + "'", hist)] if m}
    g = {int(l.split()[1]): float(re.search(r"T_pointer_grad (\S+)", l).group(1)) for l in lines
         if l.startswith("s ") and "T_pointer_grad" in l}
    near = lambda s: f"{g[min(g, key=lambda k: abs(k - s))]:.1e}" if g else "–"
    start = ev["wire_io"] + 20                                         # T's lesson needs 20 exact steps after it
    return (f"| {name} | {ev['io4_reading']} | {ev['switch']} | {near(ev['switch'] - 5)} | "
            f"{near(ev['switch'] + 20)} | {near(start)} | {ev['wire_io']} | {ev['T_resolved']} | "
            f"{ev['T_resolved'] - ev['wire_io']} |")


def t_cpu_table() -> None:
    """T's lesson on the evaluator's CPU seeds, this file against stage 3: phase B (Adam's state cleared), the I/O
    lesson, T settled."""
    print("| seed | this file: reading, phase B, I/O lesson, T settled | steps from the I/O lesson to T settled | "
          "stage 3: reading, phase B, I/O lesson, T settled | steps |")
    print("|---" * 5 + "|")
    for seed in (74, 1, 2, 3, 4, 5, 6, 7):
        cells = []
        for tag in ("anchored", "io3v2"):
            ev = schedule(json.loads((sw.RUNS / f"{tag}_cpu_seed{seed}.json").read_text())["stdout"])
            n = int(ev["T_resolved"]) - int(ev["wire_io"])
            cells += [f"{ev['io4_reading']}, {ev['switch']}, {ev['wire_io']}, {ev['T_resolved']}",
                      f"{n}" + (" (T started the right way)" if n <= 24 else "")]
        print(f"| {seed} | " + " | ".join(cells) + " |")


def t_table(names) -> None:
    print("| run | reading chosen | phase B: Adam state cleared | T's pointer gradient 5 steps before | 20 steps after "
          "| when T's lesson can start | I/O lesson | T settled | steps from the I/O lesson to T settled |")
    print("|---" * 9 + "|")
    for name in names:
        print(t_row(name))


DIAG = [f"a1_s{s}" for s in (74, 1, 2, 4)] + [f"a2_s{s}" for s in (3, 5, 6, 7)]
ABLATION = ["f1_s4", "f1_s74"]
TLESSON = ["d3_s5", "a3_s5", "h1_s5", "k1_s5", "d13_s74", "a1_s74", "k1_s74"]


def fill(doc: Path) -> None:
    """Rewrite the generated tables of ANCHORED.md (between <!-- NAME --> and <!-- /NAME --> markers)."""
    import contextlib
    import io

    def capture(fn, *args) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*args)
        return buf.getvalue().strip()

    text = doc.read_text()
    have = lambda names: [n for n in names if log_path(n).exists()]
    parts = {"RUNS": capture(runs_table, ["anchored", "io3v2"]), "AUDIT": capture(audit_table, ["anchored"]),
             "TIMES": capture(times_table), "DIAG": capture(diag_table, have(DIAG)),
             "ABLATION": capture(diag_table, have(ABLATION)), "TCPU": capture(t_cpu_table),
             "TLESSON": capture(t_table, have(TLESSON))}
    for key, table in parts.items():
        text, n = re.subn(rf"<!-- {key} -->.*?<!-- /{key} -->", lambda _: f"<!-- {key} -->\n{table}\n<!-- /{key} -->",
                          text, flags=re.S)
        print(key, "filled" if n else "(no marker)")
    doc.write_text(text)


def main(argv):
    if argv[:1] == ["diag"]:
        diag_table(argv[1:])
        return
    if argv[:1] == ["tlesson"]:
        t_table(argv[1:])
        return
    if argv[:1] == ["times"]:
        times_table()
        return
    if argv[:1] == ["fill"]:
        fill(Path(argv[1]))
        return
    tags = argv[1:] or ["anchored", "io3v2"]
    if argv[:1] == ["audit"]:
        audit_table(tags)
        return
    runs_table(tags)
    md5s(tags)


if __name__ == "__main__":
    main(sys.argv[1:])
