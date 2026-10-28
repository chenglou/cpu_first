"""Tables of WIDE_ANCHORED.md from the run records (cpu_first/history/runs/wider_menus/<name>.json) and the diagnostic
logs (wider_menus/anchored/logs/<name>.log): history/learned_wiring/anchored/summarize_anchored.py's tables for the wide
file, on wider_menus/summarize_wide.py's parse (which also names the division chain and the multiply grouping).

    python summarize_wide_anchored.py runs              # the anchored file's records, then submission_wide.py's
    python summarize_wide_anchored.py audit             # the coverage audit of each record
    python summarize_wide_anchored.py times             # seconds to a training step in the official runs
    python summarize_wide_anchored.py tcpu              # T's lesson on the CPU seeds, against submission_wide.py
    python summarize_wide_anchored.py diag NAME [...]   # one row per diagnostic log
    python summarize_wide_anchored.py fill ../WIDE_ANCHORED.md   # rewrite the write-up's generated tables
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent / "history" / "learned_wiring" / "anchored")]
import summarize_wide as sw  # noqa: E402
import summarize_anchored as sa  # noqa: E402

sa.HERE = HERE                                   # its diagnostic-log helpers read this folder's logs/
EVENTS = ("io4_reading", "order_digits", "switch", "ladder", "pieces", "chain", "wire_io", "T_resolved")
SEEDS = (74, 1, 2, 3, 4, 5, 6, 7)
DIAG = [f"w1_s{s}" for s in (74, 1, 2, 4)]


def names(tag: str) -> list:
    return [f"{tag}_official"] + [f"{tag}_cpu_seed{s}" for s in SEEDS]


def runs_table() -> None:
    print("| run | max T seen / unseen | rungs (fewest right of 768, seen / unseen) | test, ood_t, ood_n_t | "
          "subtraction, product, gate | division chain | multiply grouping | layout | shape | reading chosen | "
          "digit lessons | phase B | ladder | pieces' lessons | chain lesson | I/O lesson | T settled | frozen | "
          "never trained | training s | evaluation s |")
    print("|---" * 21 + "|")
    for tag in ("wide_anchored", "wide"):
        for name in names(tag):
            p = sw.RUNS / f"{name}.json"
            if not p.exists():
                print(f"| {name} | (no record) |")
                continue
            rec = json.loads(p.read_text())
            r = sw.parse(rec)
            ev = sa.schedule(rec.get("stdout", ""))
            where = "H100" if "H100" in r["gpu"] else "CPU"
            ev_s = ", ".join(str(r["eval"].get(k)) for k in ("test", "ood_t", "ood_n_t"))
            print(f"| {name} ({where}) | {r['maxT']} / {r['oodN']} | {min(r['rungs'] or [0])} / "
                  f"{min(r['ood_rungs'] or [0])} | {ev_s} | {r['rows']}/200, {r['cells']}/100, {r['gate']} | "
                  f"{r['chain']} | {r['numbers']} | {r['layout']} | {r['shape']} | "
                  + " | ".join(ev[k] for k in EVENTS) + f" | {r['frozen']} | {r['never']} | {r['train_s']} | "
                  f"{r['eval_s']} |")


def md5s() -> None:
    files = {hashlib.md5((HERE.parent / f).read_bytes()).hexdigest(): f
             for f in ("submission_wide_anchored.py", "submission_wide.py")}
    for name in names("wide_anchored"):
        p = sw.RUNS / f"{name}.json"
        if p.exists():
            h = hashlib.md5(json.loads(p.read_text()).get("submission", "").encode()).hexdigest()
            print(f"{name}: md5 {h} = {files.get(h, 'no file')}")


def audit_table() -> None:
    sa.sw = sw                                   # read the wide records
    sa.audit_table(["wide_anchored"])


def times_table() -> None:
    marks = (100, 200, 300, 400, 500, 600, 1000)
    print("| run | frozen at step | T settled at step | " + " | ".join(f"s to step {m}" for m in marks)
          + " | training s | evaluation s |")
    print("|---" * (len(marks) + 5) + "|")
    for name in ("wide_anchored_official", "wide_official"):
        rec = json.loads((sw.RUNS / f"{name}.json").read_text())
        out = rec["stdout"].splitlines()
        prog = [re.match(r"\s*step=(\d+) .*elapsed=([\d.]+)s", l) for l in out]
        el = {int(m.group(1)): m.group(2) for m in prog if m}
        frozen = next((re.search(r"step=(\d+)", l).group(1) for l in out if l.startswith("LR frozen")), "-")
        tset = next((re.search(r"step=(\d+)", l).group(1) for l in out if l.startswith("LR io_stage=5")), "-")
        seed = rec["result"]["seeds"][0]
        print(f"| {name} | {frozen} | {tset} | " + " | ".join(el.get(m, "-") for m in marks)
              + f" | {seed['training_seconds']:.1f} | {seed['evaluation_seconds']:.1f} |")


def t_cpu_table() -> None:
    """T's lesson on the CPU seeds, this file against submission_wide.py: the reading, phase B (Adam's state
    cleared), the I/O lesson, T settled."""
    print("| seed | this file: reading, phase B, I/O lesson, T settled | steps from the I/O lesson to T settled | "
          "submission_wide.py: reading, phase B, I/O lesson, T settled | steps |")
    print("|---" * 5 + "|")
    for seed in SEEDS:
        cells = []
        for tag in ("wide_anchored", "wide"):
            p = sw.RUNS / f"{tag}_cpu_seed{seed}.json"
            if not p.exists():
                cells += ["(no record)", "-"]
                continue
            ev = sa.schedule(json.loads(p.read_text())["stdout"])
            n = int(ev["T_resolved"]) - int(ev["wire_io"])
            cells += [f"{ev['io4_reading']}, {ev['switch']}, {ev['wire_io']}, {ev['T_resolved']}", f"{n}"]
        print(f"| {seed} | " + " | ".join(cells) + " |")


def diag_table(names) -> None:
    """summarize_anchored.py's diagnostic rows, with local_wide.py's second (deeper) gradient test as a last column."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sa.diag_table(names)
    lines = buf.getvalue().strip().splitlines()
    print(lines[0].replace("| gradient test |", "| gradient test, T = 1 and 2 | T = 8 and OOD-N T = 4 |"))
    print(lines[1] + "---|")
    for name, row in zip(names, lines[2:]):
        log = (HERE / "logs" / f"{name}.log").read_text().splitlines()
        tests = [l for l in log if l.startswith("GRADTEST splits")]
        deep = tests[1].split("every_parameter_gets_gradient=")[1] if len(tests) > 1 else "-"
        zero = [l.split()[1].rstrip(":") for l in log if l.startswith("GRADTEST  ") and "max|grad|=0.000e+00" in l]
        print(row + f" {deep}{' (0 for ' + ', '.join(zero) + ')' if zero else ''} |")


def fill(doc: Path) -> None:
    """Rewrite the generated tables of WIDE_ANCHORED.md (between <!-- NAME --> and <!-- /NAME --> markers)."""
    def capture(fn, *args) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*args)
        return buf.getvalue().strip()

    text = doc.read_text()
    have = [n for n in DIAG if (HERE / "logs" / f"{n}.log").exists()]
    parts = {"RUNS": capture(runs_table), "AUDIT": capture(audit_table), "TIMES": capture(times_table),
             "TCPU": capture(t_cpu_table), "DIAG": capture(diag_table, have)}
    for key, table in parts.items():
        text, n = re.subn(rf"<!-- {key} -->.*?<!-- /{key} -->", lambda _: f"<!-- {key} -->\n{table}\n<!-- /{key} -->",
                          text, flags=re.S)
        print(key, "filled" if n else "(no marker)")
    doc.write_text(text)


def main(argv):
    mode = argv[0] if argv else "runs"
    if mode == "runs":
        runs_table()
        md5s()
    elif mode == "audit":
        audit_table()
    elif mode == "times":
        times_table()
    elif mode == "tcpu":
        t_cpu_table()
    elif mode == "diag":
        diag_table(argv[1:])
    elif mode == "fill":
        fill(Path(argv[1]))


if __name__ == "__main__":
    main(sys.argv[1:])
