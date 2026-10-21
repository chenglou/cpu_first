"""Rewrite MENUS.md's generated tables (between <!-- NAME --> markers) from the tiny rows and the H1 records.

    ~/github/sotaku/venv/bin/python wider_menus/fill_menus.py
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import summarize as st                                                    # noqa: E402
import summarize_wide as sw                                               # noqa: E402


def capture(fn, *args) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue().strip()


def tiny(suite, key, title):
    return lambda: st.cell_table(st.rows(suite), key, title)


level = lambda r: f"division {r['cfg']['div']}, multiply {r['cfg']['mul']}"
variant = lambda r: r["name"].rsplit("_s", 1)[0]

def cpu_rows():
    import json
    out = []
    for k in (74, 1, 2, 3, 4, 5, 6, 7):
        f = sw.RUNS / f"wide_cpu_seed{k}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            out.append((k, rec, sw.parse(rec)))
    return out


def cpu_choices() -> str:
    """The CPU seeds' chosen wiring: chain, aliases, grouping (from each record's WIRING and LAYOUT lines)."""
    rows = ["| CPU seed | certified | division chain | aliases | multiply grouping | frozen at |", "|---|---|---|---|---|---|"]
    for k, rec, r in cpu_rows():
        w = dict(kv.split("=", 1) for kv in r["wiring"].split() if "=" in kv)
        al = []
        if w.get("bring") == "(1,0,'add')":
            al.append("digit added")
        if w.get("final") in ("csub", "csub2"):
            al.append("final subtraction" + (" twice" if w["final"] == "csub2" else ""))
        if w.get("flag") == "lt":
            al.append('stops on "D < N"')
        ok = r["maxT"] == 64 and r["oodN"] == 64
        rows.append(f"| {k} | {'yes' if ok else 'no'} | {r['chain']} | {', '.join(al) or 'none (io3' + chr(39) + 's own wiring)'} | "
                    f"{r['numbers']} | {r['frozen']} |")
    return "\n".join(rows)


def cpu_time() -> str:
    rs = cpu_rows()
    if not rs:
        return "CPU seeds: (running)"
    cert = [r for _, _, r in rs if r["maxT"] == 64 and r["oodN"] == 64]
    tr = [r["train_s"] for _, _, r in rs]
    ev = [r["eval_s"] for _, _, r in rs]
    fr = [int(r["frozen"]) for _, _, r in rs if r["frozen"] != "-"]
    return (f"- **CPU seeds** ({len(cert)} of {len(rs)} certified{'' if len(rs) == 8 else ', ' + str(8 - len(rs)) + ' running'}): "
            f"training {min(tr):.0f}–{max(tr):.0f} s, frozen at steps {min(fr)}–{max(fr)}, evaluation {min(ev):.0f}–{max(ev):.0f} s "
            f"(io3's CPU seeds: 580–1,060 s and 380–559 s). On a CPU the wide file's phase C costs about 5–7 s a step, "
            f"mostly the chain lesson's 9,408 combinations, which run every step until the freeze.")


BLOCKS = {
    "H1_CPU_CHOICES": cpu_choices,
    "H1_CPU_TIME": cpu_time,
    "TINY_GROW": tiny("grow", level, "levels"),
    "TINY_IO": tiny("io", variant, "variant"),
    "TINY_LAWS": tiny("laws", variant, "variant"),
    "TINY_W4": tiny("w4", variant, "variant"),
    "H1_RUNS": lambda: capture(sw.stage_table, "wide", ["wide_official", "wide_try2_h100", "wide_d4m3_try1_h100",
                                                        "wide_joint_d1m2_h100", "wide_enum_d3m2_h100"]),
}


def main():
    p = HERE / "MENUS.md"
    s = p.read_text()
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- {name} -->\n)(?:.*?\n)?(<!-- /{name} -->)", re.S)
        if pat.search(s):
            s = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(2), s)
    p.write_text(s)


if __name__ == "__main__":
    main()
