"""Rewrite WIRING.md's generated tables (between <!-- NAME --> markers) from the run records and the tiny rows.

    ~/github/sotaku/venv/bin/python fill_wiring.py
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import summarize_wiring as sw                                              # noqa: E402
from learned_wiring.tiny import summarize_tiny as st                       # noqa: E402


def capture(fn, *args) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue().strip()


def stage(tags) -> str:
    parts = [capture(sw.stage_table, t) for t in tags]
    return parts[0] + "\n" + "\n".join("\n".join(p.splitlines()[2:]) for p in parts[1:]) if len(parts) > 1 else parts[0]


def results() -> str:
    """One row per stage file: its official run and its CPU seeds."""
    import json
    out = ["| stage | file (md5) | official H100 run: max T seen / unseen | training s (frozen at step) | evaluation s | "
           "CPU seeds certified (T = 64 on both profiles) | CPU seeds: frozen at step | CPU training s | CPU evaluation s |",
           "|---|---|---|---|---|---|---|---|---|"]
    stages = [("1 (first version)", "div", "submission_div_v1.py"), ("1", "div2", "submission_div.py"),
              ("2", "sq", "submission_sq.py"), ("3 (first version)", "io3", "submission_io3_v1.py"),
              ("3", "io3v2", "submission_io3.py")]
    for label, tag, fname in stages:
        off = sw.RUNS / f"{tag}_official.json"
        if not off.exists():
            out.append(f"| {label} | `{fname}` | (running) | | | | | | |")
            continue
        r = sw.parse(json.loads(off.read_text()))
        seeds = [sw.RUNS / f"{tag}_cpu_seed{k}.json" for k in (74, 1, 2, 3, 4, 5, 6, 7)]
        got = [sw.parse(json.loads(p_.read_text())) for p_ in seeds if p_.exists()]
        cert = [g for g in got if g["maxT"] == 64 and g["oodN"] == 64]
        span = lambda v: f"{min(v)}-{max(v)}" if v else "-"
        fr = [int(g["frozen"]) for g in cert if g["frozen"] != "-"]
        failed = [str(g["seed"]) for g in got if g not in cert]
        note = f" (seed {', '.join(failed)} failed)" if failed else ""
        out.append(f"| {label} | `{fname}` (`{r['md5'][:8]}`) | {r['maxT']} / {r['oodN']} | {r['train_s']} ({r['frozen']}) | "
                   f"{r['eval_s']} | {len(cert)}/{len(got)}{note}{'' if len(got) == 8 else ' (' + str(8 - len(got)) + ' running)'} | "
                   f"{span(fr)} | {span([round(g['train_s']) for g in cert])} | {span([round(g['eval_s']) for g in cert])} |")
    return "\n".join(out)


BLOCKS = {
    "RESULTS": results,
    "STAGE1": lambda: stage(["div", "div2"]),
    "STAGE2": lambda: stage(["sq"]),
    "STAGE3": lambda: stage(["io3", "io3v2"]),
    "TINY_S2": lambda: capture(st.s2) + "\n\nTrained on numbers one digit shorter than at certification (`s2b`):\n\n"
                       + capture(st.s2b),
    "TINY_S3": lambda: "One shape per field (`s3`):\n\n" + capture(st.s3) + "\n\nOne shape switch for all four fields "
                       "(`s3b`, the file):\n\n" + capture(st.s3, "s3b"),
}


def main():
    p = HERE / "WIRING.md"
    s = p.read_text()
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"<!-- {name} -->.*?<!-- /{name} -->", re.S)
        if pat.search(s):
            s = pat.sub(lambda _: f"<!-- {name} -->\n{fn()}\n<!-- /{name} -->", s)
    p.write_text(s)
    print("filled", ", ".join(BLOCKS))


if __name__ == "__main__":
    main()
