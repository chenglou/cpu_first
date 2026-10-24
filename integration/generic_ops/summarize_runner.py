"""An evaluator record's climb (GENERIC_OPS.md §5): per level, what was practised (from the `LR climb ... choose` and
`sleep` lines of the record's stdout), the round's searches, the freeze, and the result (T on both ladders, the
splits, the times). Standard library only.

    cd cpu_first && python3 integration/generic_ops/summarize_runner.py records/runner/NAME.json [--routines]
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(path, routines=False):
    r = json.load(open(path if Path(path).exists() else HERE / path))
    out = r["stdout"]
    print(f"{Path(path).name}: md5 {r.get('submission_md5', '')[:8]}, {r.get('hardware') or r.get('gpu')}, "
          f"returncode {r.get('returncode')}")
    for line in out.splitlines():
        m = re.match(r"LR climb step=(\d+) (L\d+) (choose \(\w+\)|round(?: again)?|sleep|climb done)(.*)", line)
        if m and (m.group(3) != "sleep" or routines):
            body = m.group(4)
            if m.group(3).startswith("choose"):
                picks = re.findall(r"\('([^']+)', (\d+), '([^']*)'\)", body)
                body = "; ".join(f"{n} ({c}) {p}" for n, c, p in picks)
            print(f"  step {m.group(1)} {m.group(2)} {m.group(3)}: {body[:600]}")
        elif line.startswith(("LR round_found", "LR frozen")) or "climb done" in line:
            print("  " + line[:400])
    res = r.get("result") or {}
    sm = res.get("structured_metrics") or {}
    print("  result:", json.dumps({k: res.get(k) for k in ("score",)})[:300])
    for line in out.splitlines():
        if line.startswith(("step=", "RESULT_JSON")) and ("budget" in line):
            last = line
    print("  last progress:", last[:200] if 'last' in locals() else "-")
    try:
        dp = res["depth_profile"]
        print("  depth profile:", json.dumps(dp)[:600])
    except (KeyError, TypeError):
        pass


if __name__ == "__main__":
    main(sys.argv[1], "--routines" in sys.argv)
