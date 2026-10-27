"""Writes this study's copies (FAMILY.md §2): round 6's (bar3/scale/reading/make_copies6.py: rounds 3 to 5's lists,
then round 6's reading source), their module paths moved to this folder, then this study's replacements, each
asserted to match exactly once:
- the family's targets (SP_BREADTH.md §1): ten, rho, plus3 and minus1 beside x2, x3 (the cube) and 10x2p1 (tenrho);
- SUBTERMS: a rule that names no task in place of round 6's table of sub-laws (SUBLAW). Every window of the target's
  reading, a connected run of 2 to SUBTERM_OPS operations whose other children are cut off as holes (N and the
  constants stay), is posed as a law of its own, with at most two holes, smallest first. A window's holes are its
  operands and its modulus is bound to N (the patterns w1 and w2); its answers are the law reader's own arithmetic.
spcore_f.py gets the windows' patterns and the binding rule, core_f.py the flag in the run's config, hooks.py the
targets, the windows (pose_subterms, called from reading_pose) and their answers (in reading_target). With SUBTERMS
off the copies run as round 6's ran.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.family.make_copies7
"""

from __future__ import annotations

from pathlib import Path

from bar3.scale.compose3 import make_copies3 as M3
from bar3.scale.slice import make_copies4 as M4
from bar3.scale.loops import make_copies5 as M5
from bar3.scale.reading import make_copies6 as M6

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent.parent


def move(s):
    """Round 3's module paths (round 2's moved by round 3's move), moved to this folder."""
    for a, b in (("bar3.scale.compose3.", "bar3.scale.family."), ("bar3.scale.compose3 import", "bar3.scale.family import"),
                 ("spcore_c3", "spcore_f"), ("oracle_c3", "oracle_f"), ("climb_c3", "climb_f"), ("clean_c3", "clean_f"),
                 ("core_c3", "core_f")):
        s = s.replace(a, b)
    return s


SPCORE7 = [
    ("if H.SUBLAW or H.CONST:                     # reading: the sub-law's and the constants' patterns\n"
     "    PATTERNS.update(H.PATTERNS6)\n",
     "if H.SUBLAW or H.CONST:                     # reading: the sub-law's and the constants' patterns\n"
     "    PATTERNS.update(H.PATTERNS6)\n"
     "if H.SUBTERMS:                              # family: the windows' patterns\n"
     "    PATTERNS.update(H.PATTERNS7)\n"),
    ("            for bd in bindings(arity):\n",
     "            for bd in bindings(arity):\n"
     "                if pat in H.PATTERNS7 and bd[-1] != \"N\":     # family: a window's modulus stays in N\n"
     "                    continue\n"),
    ("        self.o = pad(xt[3 * n:3 * n + self.n], P)\n",
     "        self.o = pad(xt[3 * n:3 * n + self.n], P)\n"
     "        self.p = pad(xt[2 * n:2 * n + self.n], P)        # family: a third row's x' (HOLES)\n"),
    ("        if name in (\"x\", \"z\", \"o\", \"d\"):\n",
     "        if name in (\"x\", \"z\", \"o\", \"d\", \"p\"):                # family: p (HOLES)\n"),
]
CORE7 = [
    ("    c.round_too = env(\"ROUND_TOO\", \"0\") == \"1\"                    # compose: the round's search too\n",
     "    c.round_too = env(\"ROUND_TOO\", \"0\") == \"1\"                    # compose: the round's search too\n"
     "    if H.ROUNDT:                                                  # family: the answers at T = 2, 4 and 8\n"
     "        c.more_answers = H.more_answers(load_rows, digits, dev)\n"),
    ("    for _k in (\"DOWN\", \"SUBLAW\", \"CONST\", \"PRIOR\"):                 # reading: its flags\n",
     "    for _k in (\"DOWN\", \"SUBLAW\", \"CONST\", \"PRIOR\", \"SUBTERMS\", \"CONST1\", \"HOLES\",\n"
     "               \"ROUNDT\"):                                       # reading, family: their flags\n"),
]
CLIMB7 = [
    ("from bar3.scale.family.spcore_f import SelfClimb                    # compose2: the copy\n",
     "from bar3.scale.family.spcore_f import SelfClimb                    # compose2: the copy\n"
     "from bar3.scale.family import hooks as H                            # family: ROUNDT\n"),
    ("        if progs:\n            self.rounds = progs\n            self.done = True\n            return line\n",
     "        progs = H.round_filter(self, progs, answers)                # family: ROUNDT\n"
     "        if progs:\n            self.rounds = progs\n            self.done = True\n            return line\n"),
    ("            if rprogs:\n                self.rounds = rprogs\n                self.done = True\n",
     "            rprogs = H.round_filter(self, rprogs, answers)          # family: ROUNDT\n"
     "            if rprogs:\n                self.rounds = rprogs\n                self.done = True\n"),
]
HOOKS7 = [
    ("            \"10x2p1\": lambda x, N: (10 * x * x + 1) % N}[TARGET_FN]\n",
     "            \"10x2p1\": lambda x, N: (10 * x * x + 1) % N, \"ten\": lambda x, N: 10 * x * x % N,  # family:\n"
     "            \"rho\": lambda x, N: (x * x + 1) % N, \"plus3\": lambda x, N: (x * x + 3) % N,    # SP_BREADTH's\n"
     "            \"minus1\": lambda x, N: (x * x - 1) % N}[TARGET_FN]                          # formulas\n"),
    ("                    pose(f\"{op}.{c}\", \"x,n\", \"const\")\n",
     "                    pose(f\"{op}.{c}\", \"x,n\", \"const\")\n"
     "    if CONST1 and tree is not None:                 # family: the 1 of the reader's succ and pred\n"
     "        pose_const1(tree, pose)\n"
     "    if SUBTERMS and tree is not None:               # family: the reading's windows, smallest first\n"
     "        pose_subterms(tree, pose)\n"),
    ("    elif key == (\"mod.mul\", \"x,o,n\"):\n",
     "    elif law in SUBTERM_TREES:                      # family: a window, by the law reader's arithmetic\n"
     "        return subterm_target(law, d)\n"
     "    elif key == (\"mod.mul\", \"x,o,n\"):\n"),
    ("    elif law.split(\".\")[0] in CONST_OPS and d.pattern == \"x,n\":\n",
     "    elif law in (\"succ\", \"pred\") and d.pattern == \"x,n\":        # family: CONST1, the reader's own\n"
     "        ys = [x + 1 if law == \"succ\" else max(x - 1, 0) for x in val(d.x)]\n"
     "    elif law.split(\".\")[0] in CONST_OPS and d.pattern == \"x,n\":\n"),
    ("    climb.records.append(dict(task=\"slice\", level=climb.level, law=law, ops=ops, depth=depth,\n",
     "    climb.records.append(dict(task=\"slice\", level=climb.level, law=law, ops=ops, depth=depth,\n"
     "                              **({\"subterms\": dict(SUBTERM_SHAPES)} if SUBTERMS else {}),\n"),
]
FAMILY = '''

# ---------------------------------------------------------------------------------------------------- SUBTERMS (round 7)
import itertools                                 # noqa: E402

SUBTERMS = os.environ.get("SUBTERMS", "")        # subterms: the reading's windows, smallest first, as laws of their own
SUBTERM_OPS = int(os.environ.get("SUBTERM_OPS", 3))   # a window's most operations
SUBTERM_HOLES = 2                                # a window's most holes (in A, X and Y; N holds the modulus)
CONST1 = os.environ.get("CONST1", "")            # const1: the 1 of the reader's succ and pred (needs CONST's patterns)
HOLES = os.environ.get("HOLES", "")              # holes: a window's holes drawn from other rows than its modulus's
ROUNDT = os.environ.get("ROUNDT", "")            # roundt: a round must also give the data's answers at T = 2, 4, 8
ROUNDT_TS = (2, 4, 8)                            # the training data's one-digit T's past 1
PATTERNS7 = {                                    # the windows' patterns (spcore_f.PATTERNS, with SUBTERMS on)
    "w1": ((("x", "n"),), "double", ""),         #   one hole, the modulus beside it (x <= N)
    "w2": ((("x", "o", "n"),), "double", ""),    #   two holes, the modulus beside them
    "w1o": ((("o", "n"),), "double", ""),        #   HOLES: one hole from another row (so above N as often as not)
    "w2o": ((("o", "p", "n"),), "double", ""),   #   HOLES: two holes from two other rows
}
SUBTERM_SHAPES = {}                              # a window's law -> its shape (holes a and b)
SUBTERM_TREES = {}                               # a window's law -> its tree (holes "a" and "b")


def show_tree(t):
    """A tree (_tree's form) as a law string."""
    return t if isinstance(t, str) else f"{t[0]}({','.join(show_tree(a) for a in t[1])})"


def windows(t, most_ops=3, most_holes=2):
    """The reading's windows: every connected run of 2 to most_ops of its operations, each operation's children off
    the run cut off as holes, but N (the modulus) and the constants, which stay. Holes are renamed a, b, ... in the
    order they appear; at most most_holes of them. [(shape string, tree, operations)], each shape once, smallest
    first, then top-down as found."""
    def cut(node, k):                           # [(tree, ops)]: node cut off (a hole), or a run of <= k ops from it
        if isinstance(node, str):
            return [(node if node == "N" or node.isdigit() else "?", 0)]
        res = [("?", 0)]
        if k >= 1:
            for combo in itertools.product(*[cut(a, k - 1) for a in node[1]]):
                ops = 1 + sum(o for _, o in combo)
                if ops <= k:
                    res.append(((node[0], [c for c, _ in combo]), ops))
        return res

    def name_holes(t, names):
        if isinstance(t, str):
            if t != "?":
                return t
            names.append("ab"[len(names)] if len(names) < 2 else f"h{len(names)}")
            return names[-1]
        return (t[0], [name_holes(a, names) for a in t[1]])

    found, seen = [], set()

    def walk(node):
        if isinstance(node, str):
            return
        for tree, ops in cut(node, most_ops):
            if ops >= 2:
                names = []
                named = name_holes(tree, names)
                s = show_tree(named)
                if len(names) <= most_holes and s not in seen:
                    seen.add(s)
                    found.append((s, named, ops))
        for a in node[1]:
            walk(a)

    walk(t)
    return sorted(found, key=lambda w: w[2])      # stable: top-down within a size


def subterm_name(t, taken):
    """A window's law name: its operations top-down, each with its constant operands (w.mod.add1.mul10)."""
    parts = []

    def go(n):
        if isinstance(n, str):
            return
        parts.append(n[0] + "".join(a for a in n[1] if isinstance(a, str) and a.isdigit()))
        for a in n[1]:
            go(a)

    go(t)
    name = "w." + ".".join(parts)
    k = 2
    while name in taken:
        name = f"w.{'.'.join(parts)}.{k}"
        k += 1
    return name


def pose_subterms(tree, pose):
    """SUBTERMS: the reading's windows (at most SUBTERM_OPS operations, SUBTERM_HOLES holes), posed smallest first
    through reading_pose's pose, each on w1 or w2 by its holes."""
    for s, t, _ in windows(tree, SUBTERM_OPS, SUBTERM_HOLES):
        holes = len({h for h in ("a", "b") if _has_hole(t, h)})
        name = subterm_name(t, SUBTERM_TREES)
        SUBTERM_TREES[name], SUBTERM_SHAPES[name] = t, s
        pose(name, f"w{holes}" + ("o" if HOLES else ""), "subterm")


def more_answers(load_rows, digits, device):
    """ROUNDT: every training row with T in ROUNDT_TS, as (T, (N, x, answer) digits). For x2 the answer is the data's
    own; for another target, its function applied T times by Python's integers (as the T = 1 answers are)."""
    out = []
    for T in ROUNDT_TS:
        rows = load_rows(10 ** 6, seed=1, T=T)
        if TARGET_FN != "x2":
            def run(x, N, T=T):
                for _ in range(T):
                    x = TARGET_F(x, N)
                return x
            rows = [dict(r, result=run(int(r["x"]), int(r["modulus"]))) for r in rows]
        out.append((T, tuple(digits(rows, k).to(device) for k in ("modulus", "x", "result"))))
    return out


def round_filter(climb, progs, answers):
    """ROUNDT: of the programs found for the round, the ones right at full size on the T = 1 answers and, run T
    times, on every T of more_answers, in their order; the programs unchanged with ROUNDT off."""
    if not ROUNDT or not progs or answers is None:
        return progs
    from integration.lib.libcore import gen_on, pad, round_ok, round_settings, run_prog
    st = round_settings(climb.cells, climb.P_round)
    keep = []
    for p in progs:
        ok = bool(round_ok(p, climb.lib, climb.cells, answers, climb.P_round, climb.device))
        for T, (Nd, Xd, Yd) in getattr(climb, "more_answers", []) if ok else []:
            X = pad(Xd, climb.P_round)
            for _ in range(T):
                X, _c = run_prog(st, p, climb.lib, {"X": X, "N": pad(Nd, climb.P_round)}, gen_on(climb.device, 1),
                                 climb.device)
            ok = ok and bool((X[:, :Yd.shape[1]] == Yd).all())
        if ok:
            keep.append(p)
    climb.records.append(dict(task="roundt", level=climb.level, found=len(progs), kept=len(keep)))
    return keep


def unary_ops(t, out=None):
    """The one-operand operations of a reading's tree (the law reader's succ and pred)."""
    out = set() if out is None else out
    if not isinstance(t, str):
        if len(t[1]) == 1:
            out.add(t[0])
        for a in t[1]:
            unary_ops(a, out)
    return out


def pose_const1(tree, pose):
    """CONST1: succ and pred are an addition and a subtraction of 1, a constant the reading holds without a leaf.
    When the reading has one, its 1 is made from N (const.1 on n) and each of them is posed (on x,n), as CONST poses a
    constant and the reading's operations on it."""
    ops = sorted(unary_ops(tree) & {"succ", "pred"})
    if ops:
        pose("const.1", "n", "const")
        for op in ops:
            pose(op, "x,n", "const")


def _has_hole(t, h):
    return t == h if isinstance(t, str) else any(_has_hole(a, h) for a in t[1])


def _eval(t, env):
    """A window's value on numpy arrays (the law reader's operations), and a mask of the rows where every value on the
    way is in [0, cap)."""
    import numpy as np
    from bar3.scale.compose2.subgoal_c2 import OPS1, OPS2
    if isinstance(t, str):
        v = env[t] if t in env else np.full_like(env["N"], int(t))
        return v, np.ones(v.shape, bool)
    vals = [_eval(a, env) for a in t[1]]
    f = OPS2.get(t[0]) or OPS1[t[0]]
    v = f(*[a for a, _ in vals])
    ok = np.logical_and.reduce([m for _, m in vals]) & (v >= 0) & (v < env["cap"])
    return v, ok


def subterm_target(law, d):
    """SUBTERMS: a window's answers on a draw (w1: holes x; w2: x and o; N the modulus), by the law reader's own
    arithmetic; rows where a value on the way leaves [0, B^P) are masked. None when they do not vary."""
    import numpy as np
    B, P, dev = d.st.B, d.st.P, d.device
    val = lambda t: np.array([sum(int(v) * B ** i for i, v in enumerate(r)) for r in t.tolist()], np.int64)  # noqa
    kinds = PATTERNS7[d.pattern][0][0]                 # the holes' terms: x and o, or (HOLES) o and p
    env = {"a": val(d.term(kinds[0], None, None)), "N": val(d.N), "cap": np.int64(B ** P)}
    if len(kinds) == 3:
        env["b"] = val(d.term(kinds[1], None, None))
    v, ok = _eval(SUBTERM_TREES[law], env)
    ys = [int(y) if k else 0 for y, k in zip(v.tolist(), ok.tolist())]
    A = torch.tensor([[(y // B ** i) % B for i in range(P)] for y in ys], dtype=torch.long, device=dev)
    m = torch.tensor(ok.tolist(), dtype=torch.bool, device=dev)
    if int(m.sum()) < 2 or bool((A[m] == A[m][:1]).all()):
        return None
    return {"A": A, "Am": m}
'''


def main():
    for src, dst, repl in M3.JOBS:
        text = src.read_text()
        r4 = {"spcore_c3.py": M4.SPCORE4, "core_c3.py": M4.CORE4}.get(dst, [])
        r5 = {"spcore_c3.py": M5.SPCORE5, "core_c3.py": M5.CORE5}.get(dst, [])
        r6 = {"spcore_c3.py": M6.SPCORE6, "core_c3.py": M6.CORE6}.get(dst, [])
        r7 = {"spcore_c3.py": SPCORE7, "core_c3.py": CORE7}.get(dst, [])
        for old, new in repl + r4 + r5 + r6 + r7:
            n = text.count(old)
            assert n == 1, (dst, n, old)
            text = text.replace(old, new)
        text = move(text)
        if dst == "climb_c3.py":                       # family: ROUNDT, on the moved text
            for old, new in CLIMB7:
                n = text.count(old)
                assert n == 1, (dst, n, old)
                text = text.replace(old, new)
            r7 = CLIMB7
        out = move(dst)
        rel = src.relative_to(CPU)
        head = (f'"""FAMILY.md §2: written by bar3/scale/family/make_copies7.py from {rel}, {len(repl)} + {len(r4)} + '
                f'{len(r5)} + {len(r6)} + {len(r7)}\nasserted replacements (rounds 2 and 3\'s, marked "compose2" and '
                f'"compose3", round 4\'s, "slice", round 5\'s, "loops",\nround 6\'s, "reading", then this study\'s, '
                f'"family"). The rest of this docstring is the original\'s.\n\n')
        assert text.startswith('"""')
        (HERE / out).write_text(head + text[3:])
        print(f"wrote {out}: {len(repl)} + {len(r4)} + {len(r5)} + {len(r6)} + {len(r7)} replacements")
    text = M3.move((CPU / "bar3" / "scale" / "compose2" / "hooks.py").read_text())
    for old, new in M3.HOOKS3:
        assert text.count(old) == 1, ("hooks.py", old)
        text = text.replace(old, new)
    text = text + M4.HOOKS4
    for old, new in M5.HOOKS5 + M6.HOOKS6:
        n = text.count(old)
        assert n == 1, ("hooks.py", n, old)
        text = text.replace(old, new)
    text = text + M5.XON + M6.READING
    for old, new in HOOKS7:
        n = text.count(old)
        assert n == 1, ("hooks.py", n, old)
        text = text.replace(old, new)
    text = move(text).replace("bar3.scale.slice.spcore_s", "bar3.scale.family.spcore_f").replace("spcore_s", "spcore_f")
    assert "spcore_s" not in text and "oracle_s" not in text and "bar3.scale.loops" not in text
    assert "bar3.scale.reading" not in text
    text += FAMILY
    head = ('"""FAMILY.md §2: written by bar3/scale/family/make_copies7.py from bar3/scale/compose2/hooks.py, its\n'
            f'module paths moved here, round 3\'s {len(M3.HOOKS3)} asserted replacements, round 4\'s slicing source, round '
            f'5\'s {len(M5.HOOKS5)}\nreplacements and xon_term, round 6\'s {len(M6.HOOKS6)} and its reading source, then '
            f'this study\'s {len(HOOKS7)} replacements and\nits windows (SUBTERMS). The rest of this docstring is the '
            'original\'s.\n\n')
    (HERE / "hooks.py").write_text(head + text[3:])
    print(f"wrote hooks.py: {len(M3.HOOKS3)} + SLICE + {len(M5.HOOKS5)} + {len(M6.HOOKS6)} + {len(HOOKS7)} "
          "replacements, xon_term, READING and FAMILY")


if __name__ == "__main__":
    main()
