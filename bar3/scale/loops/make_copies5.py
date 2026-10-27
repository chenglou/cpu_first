"""Writes this study's copies (LOOPS.md §2): round 4's (bar3/scale/slice/make_copies4.py: round 3's lists, then
round 4's replacements and its slicing hooks), their module paths moved to this folder, then this study's replacements,
each asserted to match exactly once, behind the flag LOOPS:
- spcore_l.py: a draw's x + o N is built by Python when the reduction's loop is posed (hooks.xon_term), not by the
  learner's own addition;
- core_l.py: the flag in the run's config;
- hooks.py: round 4's hooks with the loops: LOOP_OF (each cut's loop over the places), the loops added where their cuts
  are posed (slice_read), and their answers (slice_target).
With LOOPS off (empty) the copies run as round 4's ran.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.loops.make_copies5
"""

from __future__ import annotations

from pathlib import Path

from bar3.scale.compose3 import make_copies3 as M3
from bar3.scale.slice import make_copies4 as M4

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent.parent


def move(s):
    """Round 3's module paths (round 2's moved by round 3's move), moved to this folder."""
    for a, b in (("bar3.scale.compose3.", "bar3.scale.loops."), ("bar3.scale.compose3 import", "bar3.scale.loops import"),
                 ("spcore_c3", "spcore_l"), ("oracle_c3", "oracle_l"), ("climb_c3", "climb_l"), ("clean_c3", "clean_l"),
                 ("core_c3", "core_l")):
        s = s.replace(a, b)
    return s


SPCORE5 = [
    ("        elif name == \"x+on\":\n"
     "            v = repeat_add(ops, st, lib, self.x, self.N, count_of(self.o), gen, dev)\n",
     "        elif name == \"x+on\" and (\"red\", \"x+on\") in H.SLICE_PAIRS:      # loops: x + o N by Python\n"
     "            v = H.xon_term(self)\n"
     "        elif name == \"x+on\":\n"
     "            v = repeat_add(ops, st, lib, self.x, self.N, count_of(self.o), gen, dev)\n"),
]
CORE5 = [
    ("        flags[\"slice\"] = H.SLICE\n",
     "        flags[\"slice\"] = H.SLICE\n"
     "    if H.LOOPS:                                                   # loops: the cuts' loops posed\n"
     "        flags[\"loops\"] = H.LOOPS\n"),
]
HOOKS5 = [
    ("SLICE_DEPTH = 5                                   # the law reader's depth (lawread.read)\n",
     "SLICE_DEPTH = 5                                   # the law reader's depth (lawread.read)\n"
     "LOOPS = os.environ.get(\"LOOPS\", \"\")              # loops: each posed cut's loop ('1': all; or mul, red, add)\n"
     "LOOP_OF = {                                       # a cut -> its loop over the digit places, the operation itself\n"
     "    \"row\": (\"mul\", \"x,o\", \"mul\"),                 #   the row over the multiplier's digits: the multiply\n"
     "    \"xd\": (\"mul\", \"x,o\", \"mul\"),                  #   x d likewise\n"
     "    \"step\": (\"red\", \"x+on\", \"red\"),               #   Horner's step over the dividend's digits: the reduction\n"
     "    \"add\": (\"add\", \"pair\", \"add\"),                #   x + d over the other operand's digits: the addition\n"
     "}\n"),
    ("    climb.records.append(dict(task=\"slice\", level=climb.level, law=law, ops=ops, depth=depth,\n",
     "    lnames = {n for _, _, n in LOOP_OF.values()} if LOOPS == \"1\" else set(LOOPS.split(\",\"))   # loops\n"
     "    for (lw, pat), cut in list(SLICE_PAIRS.items()):\n"
     "        if cut in LOOP_OF and LOOP_OF[cut][2] in lnames:\n"
     "            llw, lpat, lname = LOOP_OF[cut]\n"
     "            if not (llw in climb.laws and lpat in climb.patterns and lpat in LAW_BOOK[llw][1]):\n"
     "                SLICE_PAIRS.setdefault((llw, lpat), lname)\n"
     "    climb.records.append(dict(task=\"slice\", level=climb.level, law=law, ops=ops, depth=depth,\n"),
    ("    elif key == (\"add\", \"x,d\"):\n",
     "    elif key == (\"mul\", \"x,o\"):                      # loops: the multiply\n"
     "        ys = [x * o for x, o in zip(val(d.x), val(d.o))]\n"
     "    elif key == (\"red\", \"x+on\"):                     # loops: the reduction\n"
     "        ys = [v % n for v, n in zip(val(d.term(\"x+on\", None, None)), val(d.N))]\n"
     "    elif key == (\"add\", \"pair\"):                     # loops: the addition, on the pair's kinds\n"
     "        a, b = d.operands(None, None)\n"
     "        ys = [p + q for p, q in zip(val(a), val(b))]\n"
     "    elif key == (\"add\", \"x,d\"):\n"),
]
XON = '''

def xon_term(d):
    """LOOPS: the reduction's instance x + o N on a draw (x < N or x = N, o a number of the draw's width), by Python's
    integers, as digits [n, P]: the book's x + o N, which the book builds by the learner's own addition."""
    B, P = d.st.B, d.st.P
    val = lambda t: [sum(int(v) * B ** i for i, v in enumerate(r)) for r in t.tolist()]   # noqa: E731
    vs = [x + o * n for x, o, n in zip(val(d.x), val(d.o), val(d.N))]
    assert all(0 <= v < B ** P for v in vs)
    return torch.tensor([[(v // B ** i) % B for i in range(P)] for v in vs], dtype=torch.long, device=d.device)
'''


def main():
    for src, dst, repl in M3.JOBS:
        text = src.read_text()
        r4 = {"spcore_c3.py": M4.SPCORE4, "core_c3.py": M4.CORE4}.get(dst, [])
        r5 = {"spcore_c3.py": SPCORE5, "core_c3.py": CORE5}.get(dst, [])
        for old, new in repl + r4 + r5:
            n = text.count(old)
            assert n == 1, (dst, n, old)
            text = text.replace(old, new)
        text = move(text)
        out = move(dst)
        rel = src.relative_to(CPU)
        head = (f'"""LOOPS.md §2: written by bar3/scale/loops/make_copies5.py from {rel}, {len(repl)} + {len(r4)} + '
                f'{len(r5)} asserted\nreplacements (rounds 2 and 3\'s, marked "compose2" and "compose3", round 4\'s, '
                f'marked "slice", then this study\'s,\nmarked "loops"). The rest of this docstring is the original\'s.\n\n')
        assert text.startswith('"""')
        (HERE / out).write_text(head + text[3:])
        print(f"wrote {out}: {len(repl)} + {len(r4)} + {len(r5)} replacements")
    text = M3.move((CPU / "bar3" / "scale" / "compose2" / "hooks.py").read_text())
    for old, new in M3.HOOKS3:
        assert text.count(old) == 1, ("hooks.py", old)
        text = text.replace(old, new)
    text = text + M4.HOOKS4
    for old, new in HOOKS5:
        n = text.count(old)
        assert n == 1, ("hooks.py", n, old)
        text = text.replace(old, new)
    text = move(text).replace("bar3.scale.slice.spcore_s", "bar3.scale.loops.spcore_l").replace("spcore_s", "spcore_l")
    assert "spcore_s" not in text and "oracle_s" not in text
    text += XON
    head = ('"""LOOPS.md §2: written by bar3/scale/loops/make_copies5.py from bar3/scale/compose2/hooks.py, its module\n'
            f'paths moved here, round 3\'s {len(M3.HOOKS3)} asserted replacements, round 4\'s slicing source appended, then '
            f'this\nstudy\'s {len(HOOKS5)} replacements and xon_term (LOOPS). The rest of this docstring is the '
            'original\'s.\n\n')
    (HERE / "hooks.py").write_text(head + text[3:])
    print(f"wrote hooks.py: {len(M3.HOOKS3)} + SLICE + {len(HOOKS5)} replacements and xon_term")


if __name__ == "__main__":
    main()
