"""Writes this study's copies (SLICE.md §2): round 3's (bar3/scale/compose3/make_copies3.py: the committed files with
rounds 2 and 3's replacement lists), their module paths moved to this folder, then this study's replacements, each
asserted to match exactly once, behind the flag SLICE:
- spcore_s.py: the pose phase takes its (pattern, laws) from hooks.pose_laws (the book's, then the target's cuts),
  and law_target answers a cut by hooks.slice_target;
- core_s.py: the flag in the run's config;
- hooks.py: round 3's hooks (round 2's with round 3's replacements), the slicing source appended.
oracle_s.py, climb_s.py and clean_s.py are round 3's with the paths moved. Round 2's subgoal score stays in its folder.
With SLICE off (empty) the copies run as round 3's ran.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.slice.make_copies4
"""

from __future__ import annotations

from pathlib import Path

from bar3.scale.compose3 import make_copies3 as M3

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent.parent


def move(s):
    """Round 3's module paths (round 2's moved by round 3's move), moved to this folder."""
    for a, b in (("bar3.scale.compose3.", "bar3.scale.slice."), ("bar3.scale.compose3 import", "bar3.scale.slice import"),
                 ("spcore_c3", "spcore_s"), ("oracle_c3", "oracle_s"), ("climb_c3", "climb_s"), ("clean_c3", "clean_s"),
                 ("core_c3", "core_s")):
        s = s.replace(a, b)
    return s


SPCORE4 = [
    ("        for pat in self.patterns:\n"
     "            laws = [l for l in self.laws if pat in LAW_BOOK[l][1]]\n"
     "            if not laws:\n"
     "                continue\n",
     "        for pat, laws in H.pose_laws(self, rows):          # slice: the book's laws by pattern, then the cuts\n"
     "            if not laws:\n"
     "                continue\n"),
    ("    k, eq, P, dev, st, gen = d.kind, d.eq, d.st.P, d.device, d.st, d.gen\n"
     "    allm = torch.ones(d.n, dtype=torch.bool, device=dev)\n"
     "    x, N, z = d.x, d.N, d.z\n",
     "    if (law, d.pattern) in H.SLICE_PAIRS:               # slice: a cut of the target's law, answered by Python\n"
     "        return H.slice_target(law, d)\n"
     "    k, eq, P, dev, st, gen = d.kind, d.eq, d.st.P, d.device, d.st, d.gen\n"
     "    allm = torch.ones(d.n, dtype=torch.bool, device=dev)\n"
     "    x, N, z = d.x, d.N, d.z\n"),
]
CORE4 = [
    ("        flags[\"target_fn\"] = env(\"TARGET_FN\")\n"
     "    cfg = dict(cfg_rec, round_too=c.round_too, **flags)\n",
     "        flags[\"target_fn\"] = env(\"TARGET_FN\")\n"
     "    if H.SLICE:                                                   # slice: the cuts posed\n"
     "        flags[\"slice\"] = H.SLICE\n"
     "    cfg = dict(cfg_rec, round_too=c.round_too, **flags)\n"),
]
HOOKS4 = '''

# ---------------------------------------------------------------------------------------------------- SLICE
SLICE = os.environ.get("SLICE", "")              # slice: the cuts posed ("1": every cut of CUTS); "" off
CUTS = {                                          # an operation of the target's law -> its cuts at one digit place,
    "mul": (("acc.mul", "o,x,d", "row"),          #   as (law, pattern) on this machine: one operand at one digit,
            ("mul", "x,d", "xd")),                #   the rest carried (the row a + x d), or nothing carried (x d);
    "mod": (("red", "Bx+d", "step"),),            #   the dividend at one digit from the top: (B r + d) mod N;
    "add": (("add", "x,d", "add"),),              #   one operand at one digit: x + d
}
SLICE_PAIRS = {}                                  # (law, pattern) -> cut, once the target's law is read
SLICE_DEPTH = 5                                   # the law reader's depth (lawread.read)


def _target_rows(climb, rows):
    """(x, N, t, capacity) per row: the target's probes and acceptance rows, as subgoal_scores reads them."""
    t = climb.target
    tr = t.root(climb, rows)
    ar = _acc_root(t, climb, rows)
    sts = list(climb.settings) + list(climb.verify)
    draws = list(tr.draws) + list(ar.draws)
    xs = [x for d in draws for x in d.xs]
    Ns = [n for d in draws for n in d.Ns]
    ys = [y for d in draws for y in d.ys]
    caps = [st.B ** st.P for st, d in zip(sts, draws) for _ in range(d.n)]
    return xs, Ns, ys, caps


def slice_read(climb, rows):
    """SLICE: the target's law read once on its rows (lawread.read), SLICE_PAIRS set to the cuts of its operations
    that SLICE names and the book does not pose itself; recorded as a "slice" record."""
    if getattr(climb, "_slice_read", False) or climb.target is None:
        return
    climb._slice_read = True
    import time as _time
    from bar3.scale.slice import lawread
    from bar3.scale.slice.spcore_s import LAW_BOOK
    t0 = _time.time()
    xs, Ns, ys, caps = _target_rows(climb, rows)
    law, ops, depth = lawread.read(xs, Ns, ys, caps, depth=SLICE_DEPTH)
    names = {c for cs in CUTS.values() for _, _, c in cs} if SLICE == "1" else set(SLICE.split(","))
    for op in ops:
        for lw, pat, cut in CUTS.get(op, ()):
            book = lw in climb.laws and pat in climb.patterns and pat in LAW_BOOK[lw][1]
            if cut in names and not book:
                SLICE_PAIRS[(lw, pat)] = cut
    climb.records.append(dict(task="slice", level=climb.level, law=law, ops=ops, depth=depth,
                              constants=lawread.constants(xs, Ns, ys, caps), rows=len(set(zip(xs, Ns))),
                              cuts=[f"{lw}/{pat}={c}" for (lw, pat), c in SLICE_PAIRS.items()],
                              seconds=round(_time.time() - t0, 1)))


def pose_laws(climb, rows):
    """[(pattern, laws)] the pose phase poses: the book's laws by pattern, then (SLICE) the target's cuts."""
    from bar3.scale.slice.spcore_s import LAW_BOOK
    book = [(pat, [lw for lw in climb.laws if pat in LAW_BOOK[lw][1]]) for pat in climb.patterns]
    if not SLICE:
        return book
    slice_read(climb, rows)
    out = dict(book)
    for (lw, pat) in SLICE_PAIRS:
        ls = out.setdefault(pat, [])
        if lw not in ls:
            ls.append(lw)
    return list(out.items())


def slice_target(law, d):
    """SLICE: a cut's answers on a draw (spcore_s.Draw: x, o, the digits d, N), by Python's integers, in the law
    target's form ({"A": digits [n, P], "Am": mask}); None when they do not vary."""
    B, P, dev = d.st.B, d.st.P, d.device
    val = lambda t: [sum(int(v) * B ** i for i, v in enumerate(r)) for r in t.tolist()]   # noqa: E731
    dv = d.dv.tolist()
    key = (law, d.pattern)
    if key == ("acc.mul", "o,x,d"):
        ys = [o + x * e for o, x, e in zip(val(d.o), val(d.x), dv)]
    elif key == ("mul", "x,d"):
        ys = [x * e for x, e in zip(val(d.x), dv)]
    elif key == ("red", "Bx+d"):
        ys = [v % n for v, n in zip(val(d.term("Bx+d", None, None)), val(d.N))]
    elif key == ("add", "x,d"):
        ys = [x + e for x, e in zip(val(d.x), dv)]
    else:
        raise ValueError(key)
    assert all(0 <= y < B ** P for y in ys), key
    A = torch.tensor([[(y // B ** i) % B for i in range(P)] for y in ys], dtype=torch.long, device=dev)
    if bool((A == A[:1]).all()):
        return None
    return {"A": A, "Am": torch.ones(d.n, dtype=torch.bool, device=dev)}
'''


def main():
    for src, dst, repl in M3.JOBS:
        text = src.read_text()
        mine = {"spcore_c3.py": SPCORE4, "core_c3.py": CORE4}.get(dst, [])
        for old, new in repl + mine:
            n = text.count(old)
            assert n == 1, (dst, n, old)
            text = text.replace(old, new)
        text = move(text)
        out = move(dst)
        rel = src.relative_to(CPU)
        head = (f'"""SLICE.md §2: written by bar3/scale/slice/make_copies4.py from {rel}, {len(repl)} + {len(mine)} '
                f'asserted\nreplacements (rounds 2 and 3\'s, marked "compose2" and "compose3", then this study\'s, '
                f'marked "slice"). The rest of\nthis docstring is the original\'s.\n\n')
        assert text.startswith('"""')
        (HERE / out).write_text(head + text[3:])
        print(f"wrote {out}: {len(repl)} + {len(mine)} replacements")
    text = M3.move((CPU / "bar3" / "scale" / "compose2" / "hooks.py").read_text())
    for old, new in M3.HOOKS3:
        n = text.count(old)
        assert n == 1, ("hooks.py", n, old)
        text = text.replace(old, new)
    text = move(text) + HOOKS4
    head = ('"""SLICE.md §2: written by bar3/scale/slice/make_copies4.py from bar3/scale/compose2/hooks.py, its module\n'
            f'paths moved here, round 3\'s {len(M3.HOOKS3)} asserted replacements (TARGET_FN, TIEFOLLOW) and the slicing '
            'source\nappended (SLICE). The rest of this docstring is the original\'s.\n\n')
    (HERE / "hooks.py").write_text(head + text[3:])
    print(f"wrote hooks.py: {len(M3.HOOKS3)} replacements and SLICE")


if __name__ == "__main__":
    main()
