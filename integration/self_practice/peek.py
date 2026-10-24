"""Check-ins on the study's Modal runs (SELF_PRACTICE.md): each run's latest level, picks and round from its log on the
Volume `self-practice` (no app needed).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/self_practice/peek.py TAG [TAG ...] [--lines N]
"""
import re
import subprocess
import sys

PY = sys.executable


def log_of(tag):
    out = subprocess.run([PY, "-m", "modal", "volume", "get", "self-practice", f"runs/{tag}.log", "-"],
                         capture_output=True, text=True, timeout=120)
    return out.stdout


def main():
    args = sys.argv[1:]
    n = 3
    if "--lines" in args:
        i = args.index("--lines")
        n = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    for tag in args:
        text = log_of(tag)
        lines = [l for l in text.splitlines() if l.startswith("UNIT") or l.startswith(("TOTAL", "EXIT", "DEADLINE",
                                                                                           "Traceback"))]
        last = lines[-1] if lines else "(no log yet)"
        keep = [l for l in lines if re.search(r" L\d+ (choose|round|target|sleep)| climb done|TOTAL|EXIT|Traceback", l)]
        print(f"== {tag}: {len(lines)} units; last: {last[:220]}")
        for l in keep[-n:]:
            print("   ", l[:900])
        if "Traceback" in text:
            print(text[text.index("Traceback"):][:3000])


if __name__ == "__main__":
    main()
