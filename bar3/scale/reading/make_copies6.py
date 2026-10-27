"""Writes this study's copies (READING.md §2): round 5's (bar3/scale/loops/make_copies5.py: round 3's lists, round 4's
replacements and slicing hooks, round 5's loops), their module paths moved to this folder, then this study's
replacements, each asserted to match exactly once, behind four flags:
- DOWN: the posed step's own cuts, down to the comparison: the link (one conditional subtraction) and the comparison;
- SUBLAW: each two-operation sub-law outer(inner(a, b), N) that the target's reading holds twice or more, as a law of
  its own (the cube's mod(mul(a, b), N): (x o) mod N on the pattern x,o,n);
- CONST: each constant of the reading, made from N (const.c on n), and each of the reading's operations with it as an
  operand (op.c on x,n: 10·x² + 1's x + 1 and x·10);
- PRIOR: under anchored pairs (ct2a2), the reading's own concepts first among tied gains.
spcore_r.py gets the new patterns (with SUBLAW or CONST on) and PRIOR's tie-break; core_r.py the flags in the run's
config; hooks.py round 5's hooks with reading_pose after the loops and the new laws' answers (reading_target). With
every flag off the copies run as round 5's ran.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.reading.make_copies6
"""

from __future__ import annotations

from pathlib import Path

from bar3.scale.compose3 import make_copies3 as M3
from bar3.scale.slice import make_copies4 as M4
from bar3.scale.loops import make_copies5 as M5

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent.parent


def move(s):
    """Round 3's module paths (round 2's moved by round 3's move), moved to this folder."""
    for a, b in (("bar3.scale.compose3.", "bar3.scale.reading."), ("bar3.scale.compose3 import", "bar3.scale.reading import"),
                 ("spcore_c3", "spcore_r"), ("oracle_c3", "oracle_r"), ("climb_c3", "climb_r"), ("clean_c3", "clean_r"),
                 ("core_c3", "core_r")):
        s = s.replace(a, b)
    return s


SPCORE6 = [
    ("    \"o,x,d\": (((\"o\", \"x\", \"d\"),), \"carry\", \"\"),\n}\n",
     "    \"o,x,d\": (((\"o\", \"x\", \"d\"),), \"carry\", \"\"),\n}\n"
     "if H.SUBLAW or H.CONST:                     # reading: the sub-law's and the constants' patterns\n"
     "    PATTERNS.update(H.PATTERNS6)\n"),
    ("                           key=lambda p: -p[0])\n",
     "                           key=lambda p: (-p[0],) + H.prior_rank(self, p[1]))     # reading: PRIOR\n"),
    ("            for n in sorted(singles, key=lambda n: -singles[n]):\n",
     "            for n in sorted(singles, key=lambda n: (-singles[n],) + H.prior_rank(self, (n,))):   # reading: PRIOR\n"),
]
CORE6 = [
    ("        flags[\"loops\"] = H.LOOPS\n",
     "        flags[\"loops\"] = H.LOOPS\n"
     "    for _k in (\"DOWN\", \"SUBLAW\", \"CONST\", \"PRIOR\"):                 # reading: its flags\n"
     "        if getattr(H, _k):\n"
     "            flags[_k.lower()] = getattr(H, _k)\n"),
]
HOOKS6 = [
    ("    climb.records.append(dict(task=\"slice\", level=climb.level, law=law, ops=ops, depth=depth,\n",
     "    reading_pose(climb, law, LAW_BOOK)              # reading: the step's cuts, the repeated sub-laws, the constants\n"
     "    climb.records.append(dict(task=\"slice\", level=climb.level, law=law, ops=ops, depth=depth,\n"),
    ("    key = (law, d.pattern)\n",
     "    key = (law, d.pattern)\n"
     "    if key in READING_PAIRS:                        # reading: the new laws' answers\n"
     "        return reading_target(law, d)\n"),
]
READING = '''

# ---------------------------------------------------------------------------------------------------- READING (round 6)
DOWN = os.environ.get("DOWN", "")                # down: the step's own cuts ("1": the link and the comparison; or a list)
STEP_CUTS = {                                    # a cut -> its own cuts, down to the comparison
    "step": (("red", "pair", "link"),),          #   Horner's step is B conditional subtractions (B[link]): one is the link
    "link": (("lt", "pair", "cmp"),),            #   the link is a comparison and a select (STA.S lt SEL.S): the comparison
}
SUBLAW = os.environ.get("SUBLAW", "")            # sublaw: the reading's repeated two-operation sub-laws, as laws
SUBLAWS = {("mod", "mul"): ("mod.mul", "x,o,n")}  # outer(inner(a, b), N) -> (law, pattern): (a b) mod N on (x, o, n)
CONST = os.environ.get("CONST", "")              # const: the reading's constants, made from N, and its operations on them
CONST_OPS = ("add", "mul")                       # the reading's operations posed with a constant operand (x + c, x c)
PRIOR = os.environ.get("PRIOR", "")              # prior: under ct2a2, the reading's own concepts first among tied gains
PATTERNS6 = {                                    # the new patterns (spcore_r.PATTERNS, with SUBLAW or CONST on)
    "x,o,n": ((("x", "o", "n"),), "double", ""),  #   the sub-law: x o fits the registers before its reduction
    "n": ((("n",),), "same", ""),                #   a constant made from N (N >= 1 gives the borrow of 0 - N)
    "x,n": ((("x", "n"),), "carry", ""),         #   x + c and x c, N beside them (to make c)
}
READING_PAIRS = set()                            # the (law, pattern)s reading_pose posed, answered by reading_target


def _tree(s):
    """A reading (lawread's string) as nested (op, [args]); leaves as strings."""
    s = s.strip()
    i = s.find("(")
    if i < 0:
        return s
    body, args, depth, start = s[i + 1:-1], [], 0, 0
    for j, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(_tree(body[start:j]))
            start = j + 1
    args.append(_tree(body[start:]))
    return (s[:i], args)


def sublaws(t, out=None):
    """{(outer, inner): count} of the two-operation sub-laws outer(inner(a, b), N) in a reading's tree."""
    out = {} if out is None else out
    if isinstance(t, str):
        return out
    op, args = t
    if len(args) == 2 and args[1] == "N" and not isinstance(args[0], str) and len(args[0][1]) == 2:
        out[(op, args[0][0])] = out.get((op, args[0][0]), 0) + 1
    for a in args:
        sublaws(a, out)
    return out


def constant_ops(t, out=None):
    """{c: [ops]} for each constant leaf c of a reading's tree, with the operations it is an operand of."""
    out = {} if out is None else out
    if isinstance(t, str):
        return out
    op, args = t
    for a in args:
        if isinstance(a, str) and a.isdigit():
            ops = out.setdefault(int(a), [])
            if op not in ops:
                ops.append(op)
        constant_ops(a, out)
    return out


def reading_pose(climb, law, book_laws):
    """READING: what the target's reading poses beyond the cuts and their loops, added to SLICE_PAIRS (and to the law
    book, so that a concept practised in one binding counts as done), each unless the book poses it itself:
    - DOWN: the posed step's own cuts that DOWN names, down to the comparison (STEP_CUTS);
    - SUBLAW: each sub-law of SUBLAWS the reading holds at least twice;
    - CONST: each constant leaf c, made from N (const.c on n), and the reading's operations with c as an operand
      (op.c on x,n, for op in CONST_OPS)."""
    def book(lw, pat):
        return lw in climb.laws and pat in climb.patterns and pat in book_laws.get(lw, ("", ()))[1]

    def pose(lw, pat, name):
        book_laws.setdefault(lw, ("A", (pat,)))
        if not book(lw, pat) and (lw, pat) not in SLICE_PAIRS:
            SLICE_PAIRS[(lw, pat)] = name
            READING_PAIRS.add((lw, pat))

    if DOWN:
        names = {n for cs in STEP_CUTS.values() for _, _, n in cs} if DOWN == "1" else set(DOWN.split(","))
        todo, seen = list(SLICE_PAIRS.values()), set()
        while todo:
            for lw, pat, name in STEP_CUTS.get(todo.pop(0), ()):
                if name not in seen:
                    seen.add(name)
                    todo.append(name)
                    if name in names:
                        pose(lw, pat, name)
    tree = _tree(law) if law else None
    if SUBLAW and tree is not None:
        for k, n in sublaws(tree).items():
            if n >= 2 and k in SUBLAWS:
                pose(*SUBLAWS[k], "sublaw")
    if CONST and tree is not None:
        for c, ops in constant_ops(tree).items():
            pose(f"const.{c}", "n", "const")
            for op in ops:
                if op in CONST_OPS:
                    pose(f"{op}.{c}", "x,n", "const")


def reading_target(law, d):
    """READING: the new laws' answers on a draw (spcore_r.Draw), by Python's integers, in the law target's form:
    - the comparison (lt on pair): the carry a < b;
    - the link (red on pair): a - b if a >= b, else a, where that is one conditional subtraction (b > 0, a < 2 b);
    - the sub-law (mod.mul on x,o,n): (x o) mod N;
    - a constant (const.c on n): c on every row, which may not vary;
    - an operation with a constant (op.c on x,n): x + c, x c.
    None when the answers do not vary (but for a constant), or fewer than two rows are in the law's range."""
    B, P, dev = d.st.B, d.st.P, d.device
    val = lambda t: [sum(int(v) * B ** i for i, v in enumerate(r)) for r in t.tolist()]   # noqa: E731
    key, n = (law, d.pattern), d.n
    ok = [True] * n
    if key == ("lt", "pair"):
        a, b = (val(t) for t in d.operands(None, None))
        c = torch.tensor([int(p < q) for p, q in zip(a, b)], dtype=torch.long, device=dev)
        if bool((c == c[:1]).all()):
            return None
        return {"c": c, "cm": torch.ones(n, dtype=torch.bool, device=dev)}
    if key == ("red", "pair"):
        a, b = (val(t) for t in d.operands(None, None))
        ok = [q > 0 and p < 2 * q for p, q in zip(a, b)]
        ys = [p - q if p >= q else p for p, q in zip(a, b)]
    elif key == ("mod.mul", "x,o,n"):
        ys = [x * o % m for x, o, m in zip(val(d.x), val(d.o), val(d.N))]
    elif law.startswith("const.") and d.pattern == "n":
        ys = [int(law.split(".")[1])] * n
    elif law.split(".")[0] in CONST_OPS and d.pattern == "x,n":
        op, c = law.split(".")
        ys = [x + int(c) if op == "add" else x * int(c) for x in val(d.x)]
    else:
        raise ValueError(key)
    ok = [k and 0 <= y < B ** P for k, y in zip(ok, ys)]
    ys = [y if k else 0 for k, y in zip(ok, ys)]
    A = torch.tensor([[(y // B ** i) % B for i in range(P)] for y in ys], dtype=torch.long, device=dev)
    m = torch.tensor(ok, dtype=torch.bool, device=dev)
    if not law.startswith("const."):
        if int(m.sum()) < 2 or bool((A[m] == A[m][:1]).all()):
            return None
    return {"A": A, "Am": m}


def prior_rank(climb, names):
    """PRIOR: a sort key's tail among tied gains (ct2a2's pairs and singles): how many of names are not the reading's
    own (its cuts and their loops, the step's cuts, its sub-laws and constants); () with PRIOR off, so that pool order
    breaks the ties as before."""
    if not PRIOR:
        return ()
    own = {f"{lw}/{pat}" for lw, pat in SLICE_PAIRS}
    return (sum(nm.split("@")[0] not in own for nm in names),)
'''


def main():
    for src, dst, repl in M3.JOBS:
        text = src.read_text()
        r4 = {"spcore_c3.py": M4.SPCORE4, "core_c3.py": M4.CORE4}.get(dst, [])
        r5 = {"spcore_c3.py": M5.SPCORE5, "core_c3.py": M5.CORE5}.get(dst, [])
        r6 = {"spcore_c3.py": SPCORE6, "core_c3.py": CORE6}.get(dst, [])
        for old, new in repl + r4 + r5 + r6:
            n = text.count(old)
            assert n == 1, (dst, n, old)
            text = text.replace(old, new)
        text = move(text)
        out = move(dst)
        rel = src.relative_to(CPU)
        head = (f'"""READING.md §2: written by bar3/scale/reading/make_copies6.py from {rel}, {len(repl)} + {len(r4)} + '
                f'{len(r5)} + {len(r6)}\nasserted replacements (rounds 2 and 3\'s, marked "compose2" and "compose3", '
                f'round 4\'s, marked "slice", round 5\'s,\nmarked "loops", then this study\'s, marked "reading"). The rest '
                f'of this docstring is the original\'s.\n\n')
        assert text.startswith('"""')
        (HERE / out).write_text(head + text[3:])
        print(f"wrote {out}: {len(repl)} + {len(r4)} + {len(r5)} + {len(r6)} replacements")
    text = M3.move((CPU / "bar3" / "scale" / "compose2" / "hooks.py").read_text())
    for old, new in M3.HOOKS3:
        assert text.count(old) == 1, ("hooks.py", old)
        text = text.replace(old, new)
    text = text + M4.HOOKS4
    for old, new in M5.HOOKS5 + HOOKS6:
        n = text.count(old)
        assert n == 1, ("hooks.py", n, old)
        text = text.replace(old, new)
    text = move(text).replace("bar3.scale.slice.spcore_s", "bar3.scale.reading.spcore_r").replace("spcore_s", "spcore_r")
    assert "spcore_s" not in text and "oracle_s" not in text and "bar3.scale.loops" not in text
    text += M5.XON + READING
    head = ('"""READING.md §2: written by bar3/scale/reading/make_copies6.py from bar3/scale/compose2/hooks.py, its\n'
            f'module paths moved here, round 3\'s {len(M3.HOOKS3)} asserted replacements, round 4\'s slicing source '
            f'appended, round 5\'s\n{len(M5.HOOKS5)} replacements and xon_term (LOOPS), then this study\'s '
            f'{len(HOOKS6)} replacements and its reading source\n(DOWN, SUBLAW, CONST, PRIOR). The rest of this '
            'docstring is the original\'s.\n\n')
    (HERE / "hooks.py").write_text(head + text[3:])
    print(f"wrote hooks.py: {len(M3.HOOKS3)} + SLICE + {len(M5.HOOKS5)} + {len(HOOKS6)} replacements, xon_term and READING")


if __name__ == "__main__":
    main()
