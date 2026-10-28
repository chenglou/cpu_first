"""combined_sp2's local checks on the deep file against the wider file's (DEEP.md), part by part (the two
runs took the parts in different orders), each part's summary lines with times masked; and the fix part's detail lines.

    cd cpu_first && python3 integration/legal_deep/checks/compare_local.py
"""
import difflib
import re
from pathlib import Path

CPU = Path(__file__).resolve().parents[3]
OLD, NEW = (CPU / "integration" / p for p in ("legal_wider/checks/logs/local_wider.log",
                                                 "legal_deep/checks/logs/local_deep.log"))
PART = r"^(policy|grad|machines|isa|designed|buffers|replay|climb \(tiny budgets\)|climb|fix|book):"


def parts(p):
    out = {}
    for line in open(p):
        m = re.match(PART, line)
        if m:
            line = re.sub(r"\d+\.\d+s\b", "Ts", re.sub(r"\(\d+(\.\d+)?s\)", "", line))
            out.setdefault(m.group(1), []).append(line.rstrip())
    return out


def fix_details(p):
    L = open(p).read().splitlines()
    i = next(k for k, line in enumerate(L) if line.startswith("fix: check_fix_variants"))
    j = next(k for k, line in enumerate(L) if line.startswith("fix: above probes"))
    return [re.sub(r"\(\d+(\.\d+)?s\)", "", line) for line in L[i + 1:j]]


a, b = parts(OLD), parts(NEW)
for k in dict.fromkeys(list(a) + list(b)):
    if a.get(k) == b.get(k):
        print(f"{k}: the same ({len(a[k])} lines)")
        continue
    print(f"{k}: differs")
    for line in difflib.unified_diff(a.get(k, []), b.get(k, []), "wider", "deep", n=0, lineterm=""):
        if not line.startswith(("@@", "---", "+++")):
            print("   ", line[:220])
fa, fb = fix_details(OLD), fix_details(NEW)
print(f"fix's details: {'the same' if fa == fb else 'differ'} ({len(fa)} lines)")
