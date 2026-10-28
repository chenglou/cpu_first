"""DEEP.md §8: REVIEW.md's line numbers in the wider file, and the same lines in this file. Only REVIEW's sections
about the file to judge are read (§2 up to the legal file, §5 up to where the legal file differs, §6). Lines the
squeeze left unchanged but for their comment's padding are matched by difflib; the few that changed are named by
hand (CHANGED). Writes checks/logs/review_lines.log.

    cd cpu_first && python3 integration/legal_deep/checks/review_lines.py
"""
import difflib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / "integration/legal_wider/submission_stage3_wider.py"
WIDER = ROOT / "integration/legal_deep/submission_stage3_deep.py"
SECTIONS = [("## 2. The file to judge", "**The earlier file to judge, the family file:**"),
            ("## 5. The gray areas", "**Where the family file differs**"),
            ("## 6. An hour's reading order", "## 7. ")]
CHANGED = {1: "1 (the docstring, lines 1-10; its first line names DEEP.md and make_deep.py)",
           79: "79 (CLIMB_WORK = 13_000_000_000)",
           103: "103 (READ_POLY = 4)",
           2993: "2991 (windows(t, least=2): reduced1's one-operation windows)"}
CITE = re.compile(r"(?:`(?P<sym>[^`]+)`[^`(]{0,24}?)?\blines?\s+(?P<nums>\d{1,4}(?:[–-]\d{1,4})?"
                  r"(?:(?:,? and|,)\s+\d{1,4}(?:[–-]\d{1,4})?)*)|`(?P<sym2>[^`]+)` at (?P<at>\d{3,4})"
                  r"|`(?P<sym3>[^`]+)` \((?P<paren>\d{2,4}(?:[–-]\d{2,4})?)\b|\((?P<prng>\d{2,4}[–-]\d{2,4})[;)]"
                  r"|\bat (?P<atr>\d{2,4}[–-]\d{2,4})\b")


def norm(line):
    s = line.rstrip()
    return s.strip() if s.lstrip().startswith("#") else re.sub(r"\s+#", "  #", s)


def main():
    fam, wid = FAMILY.read_text().splitlines(), WIDER.read_text().splitlines()
    sm = difflib.SequenceMatcher(None, [norm(x) for x in fam], [norm(x) for x in wid], autojunk=False)
    same = {a + k + 1: b + k + 1 for a, b, n in sm.get_matching_blocks() for k in range(n)}
    review = (ROOT / "REVIEW.md").read_text().splitlines()
    keep = set()
    for start, end in SECTIONS:
        lo = next(i for i, x in enumerate(review) if x.startswith(start))
        hi = next(i for i, x in enumerate(review) if i > lo and x.startswith(end))
        keep.update(range(lo, hi))
    text, where = "", []
    for i in sorted(keep):  # joined, so a citation wrapped over two lines is found; where[k]: REVIEW's line of char k
        if i - 1 not in keep:
            text, where = text + "\n", where + [i + 1]
        text += review[i] + " "
        where += [i + 1] * (len(review[i]) + 1)
    out, pairs = [], {}
    for m in CITE.finditer(text):
        sym = m.group("sym") or m.group("sym2") or m.group("sym3") or ""
        cited = next(m.group(g) for g in ("nums", "at", "paren", "prng", "atr") if m.group(g))
        nums = [int(x) for x in re.findall(r"\d{1,4}", cited)]
        for n in nums:
            new = CHANGED.get(n) or (str(same[n]) if n in same else "not matched")
            pairs[n] = new
            code = wid[same[n] - 1].strip()[:64] if n in same and n not in CHANGED else ""
            out.append(f"REVIEW {where[m.start()]:4d}  {sym[:28]:28s} {n:5d} -> {new}" + (f"   | {code}" if code else ""))
    out.append("\nEvery family-file line REVIEW cites there, and this file's line:")
    out += [f"{n:5d} -> {pairs[n]}" for n in sorted(pairs)]
    log = Path(__file__).resolve().parent / "logs" / "review_lines.log"
    log.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"wrote {log}: {len(pairs)} lines cited")


if __name__ == "__main__":
    main()
