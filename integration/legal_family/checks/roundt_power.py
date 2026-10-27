"""ROUNDT's reach on minus1 (FAMILY_FILE.md): round 7's wrong minus1 rounds are wrong exactly where a round's input is 0
(FAMILY.md §5.4). Of H1's training prompts of T = 2, 4 and 8 with minus1's answers (Python's integers), how many pass
through 0 before their last round: in all, among ROUNDT's rows as core_f.py draws them (the file's ROUNDT of each,
seeded), and among 64 (the answer program's pool). A round wrong at 0 passes ROUNDT only if none of its rows does.

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 ~/github/sotaku/venv/bin/python integration/legal_family/checks/roundt_power.py
"""
import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import core_f  # noqa: E402

ROUNDT = int(re.search(r"^ROUNDT = (\d+)", (HERE.parent / "submission_stage3_family.py").read_text(), re.M).group(1))


def through_zero(r):
    y, n = int(r["x"]), int(r["modulus"])
    for _ in range(int(r["time_steps"])):
        if y == 0:
            return True
        y = (y * y - 1) % n
    return False


def main():
    rows = [json.loads(line) for line in open(os.path.join(core_f.H1_DIR, "train.jsonl"))]
    print(f"ROUNDT = {ROUNDT} rows of each count", flush=True)
    for T in (2, 4, 8):
        allT = [r for r in rows if r["time_steps"] == T]
        hits = sum(map(through_zero, allT))
        kept = core_f.load_rows(ROUNDT, seed=10 + T, T=T)
        k = sum(map(through_zero, kept))
        p64 = 1 - math.comb(len(allT) - hits, 64) / math.comb(len(allT), 64)
        pk = 1 - math.comb(len(allT) - hits, min(ROUNDT, len(allT))) / math.comb(len(allT), min(ROUNDT, len(allT)))
        print(f"T = {T}: {hits} of {len(allT)} rows pass through 0; {k} of the {len(kept)} drawn here; the chance that "
              f"{min(ROUNDT, len(allT))} random rows hold one: {pk:.4f}, that 64 do: {p64:.3f}", flush=True)


if __name__ == "__main__":
    main()
