"""DEEP.md §3: read_space.py's sweep of the space (records/read_space_*.jsonl): how many tasks are read with the
first ROWS answer rows, how many only with the retry's, and which not at all; by k, the stage that read them (the
chains and gap constants, or x's polynomials by Horner's steps, counted from the law's text), the laws tried and the
seconds per try.

    cd cpu_first && python3 integration/legal_deep/checks/summarize_space.py [read_space_deep_]
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REC = Path(__file__).resolve().parents[1] / "records"


def steps(law):
    """A polynomial reading's Horner steps: the operations on x's chain inside mod(...,N), less the gap's."""
    inner = re.sub(r"^mod\((?:add|sub)\((.*),\d+\),N\)$", r"\1", law)
    inner = re.sub(r"^mod\((.*),N\)$", r"\1", inner)
    return inner.count("(")


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "read_space_"  # read_space_ (the candidate), read_space_deep_
    rows = [json.loads(line) for f in sorted(REC.glob(prefix + "[0-9]*.jsonl")) for line in f.read_text().splitlines()
            if line.strip()]
    tasks = {r["task"]: r for r in rows}
    print(f"{len(tasks)} tasks of 1,800")
    first = lambda r: next(iter(k for k in r if k.startswith("r")))  # noqa: E731
    out = Counter()
    by_steps = Counter()
    tried, secs = {}, {}
    missing = []
    for name, r in sorted(tasks.items()):
        a, k, b, c = (int(v) for v in name.split("_")[1:])
        tries = [key for key in r if key.startswith("r")]
        got = [key for key in tries if r[key]["law"]]
        where = got[0] if got else None
        out[(k, where or "none")] += 1
        if not got:
            missing.append(name)
        else:
            law = r[where]["law"]
            by_steps[(k, "N" in law[:-3] and "x" in law and law.count("N") > 1, steps(law))] += 1
        for key in tries:
            tried.setdefault(key, []).append(r[key]["tried"])
            secs.setdefault(key, []).append(r[key]["seconds"])
    for (k, where), n in sorted(out.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        print(f"k = {k}: read with {where}: {n}")
    print("not read:", missing[:40], "..." if len(missing) > 40 else "")
    for key in sorted(tried):
        t, s = tried[key], secs[key]
        print(f"{key}: laws tried per try {min(t):,}-{max(t):,} (median {int(statistics.median(t)):,}); seconds "
              f"{min(s):.2f}-{max(s):.2f} (median {statistics.median(s):.2f}); {len(t)} tries")
    hard = [(r["task"], r[next(k for k in r if k.startswith("r") and r[k]["law"])]["rows"])
            for r in tasks.values() if any(k.startswith("r") and r[k]["law"] for k in r)]
    hard.sort(key=lambda x: x[1])
    print("fewest rows among the read:", hard[:12])


if __name__ == "__main__":
    main()
