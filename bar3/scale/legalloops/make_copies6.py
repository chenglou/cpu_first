"""Writes this study's copies (LEGALLOOPS.md §2): test 3's committed copies (bar3/scale/legalclimb/, ef10ac1, sha256
asserted), module paths moved to this folder, then this study's replacements, each asserted to match exactly once:
- hooks.py: with READ set, the pose phase's laws are the book's plus the cuts and loops of the target's law read
  legally (lread.pose);
- core_k.py: the climb keeps the answer rows for the reader (`answer_rows`); READ in the run's config.
Everything else is test 3's with the paths moved. With READ unset the copies run as test 3's ran.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.legalloops.make_copies6
"""

from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "legalclimb"
SOURCES = {                                       # test 3's copies at ef10ac1
    "spcore_l.py": ("spcore_k.py", "1e6bc49a406d24b4721efb1ce9d4e34418369eafe39ecf375778709860121a4b"),
    "climb_l.py": ("climb_k.py", "acdac65f63554c1b714977456c0477fe0bc5a6d61a7221e318dba2a29196bb75"),
    "core_l.py": ("core_k.py", "ce47e06fb40d9a03e1c6aafce8ac9688baec56ddb24739f5b8ba2a32aa8d01a1"),
    "hooks.py": ("hooks.py", "3c287ea5438f23f99eae35d26b76c23dc12f8e0409a4828e38e33fcad11f2a4a"),
    "oracle_l.py": ("oracle_k.py", "02e1e2a6b86a735fcd742a1cf8efc68e3ee8b4117058d7ff00d860659cb26736"),
    "clean_l.py": ("clean_k.py", "a2a6c8470fcaf4268615aeefdd467e8ee5e2c4352648292a5ce1ef0a3a58eb00"),
    "rowclimb.py": ("rowclimb.py", "ab8679ba190121877294aeea3cb8f09924d7f5c1a89462dfe04f7df109908f9d"),
    "lclimb.py": ("lclimb.py", "d6182d0c1d772ba93a85a93704d3163a5e18fff9a585f182e14f0b179e3c30ed"),
}


def move(s):
    """Test 3's module paths, moved to this folder."""
    for a, b in (("bar3.scale.legalclimb.", "bar3.scale.legalloops."),
                 ("bar3.scale.legalclimb import", "bar3.scale.legalloops import"),
                 ("spcore_l", "spcore_k"), ("oracle_l", "oracle_k"), ("climb_l", "climb_k"), ("clean_l", "clean_k"),
                 ("core_l", "core_k")):
        s = s.replace(a, b)
    return s


HOOKS6 = [
    ("SLICE = os.environ.get(\"SLICE\", \"\")              # slice: the cuts posed (\"1\": every cut of CUTS); \"\" off\n",
     "SLICE = os.environ.get(\"SLICE\", \"\")              # slice: the cuts posed (\"1\": every cut of CUTS); \"\" off\n"
     "READ = os.environ.get(\"READ\", \"\")                # legalloops: the cuts and loops of the law read legally\n"),
    ("    book = [(pat, [lw for lw in climb.laws if pat in LAW_BOOK[lw][1]]) for pat in climb.patterns]\n"
     "    if not SLICE:\n",
     "    book = [(pat, [lw for lw in climb.laws if pat in LAW_BOOK[lw][1]]) for pat in climb.patterns]\n"
     "    if READ:                                          # legalloops: the cuts and loops of the law read legally\n"
     "        from bar3.scale.legalloops import lread\n"
     "        return lread.pose(climb, book, LAW_BOOK)\n"
     "    if not SLICE:\n"),
]
CORE6 = [
    ("    c = LC.LegalClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == \"cuda\" else 1 << 16, target=target, **cfg)\n",
     "    c = LC.LegalClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == \"cuda\" else 1 << 16, target=target, **cfg)\n"
     "    c.answer_rows = ans                                           # legalloops: the legal reader's rows\n"),
    ("    if H.SLICE:                                                   # slice: the cuts posed\n",
     "    if H.READ:                                                    # legalloops: the cuts and loops read legally\n"
     "        flags[\"read\"] = H.READ\n"
     "    if H.SLICE:                                                   # slice: the cuts posed\n"),
]
EDITS = {"hooks.py": HOOKS6, "core_k.py": CORE6}


def main():
    for src, (dst, want) in SOURCES.items():
        text = (SRC / src).read_text()
        got = hashlib.sha256(text.encode()).hexdigest()
        assert got == want, (src, got)
        text = move(text)
        for a, b in EDITS.get(dst, []):
            assert text.count(a) == 1, (dst, a[:70], text.count(a))
            text = text.replace(a, b)
        first, rest = text.split("\n", 1)
        head = f'"""LEGALLOOPS.md §2: written by bar3/scale/legalloops/make_copies6.py from bar3/scale/legalclimb/' \
               f'{src}, {len(EDITS.get(dst, []))} asserted replacements\n(this study\'s, marked "legalloops"). The ' \
               f'rest of this docstring is the copy\'s.\n\n'
        assert first.startswith('"""'), (src, first[:40])
        (HERE / dst).write_text(head + first[3:] + "\n" + rest)
        print("wrote", dst, len(EDITS.get(dst, [])), "replacements")


if __name__ == "__main__":
    main()
