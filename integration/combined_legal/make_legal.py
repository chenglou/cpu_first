"""Builds the legal file (integration/combined_legal/LEGAL_FILE.md): fast2 (combined_fast's submission_stage3_fast2.py,
its committed bytes at f237a61, md5 asserted) with its generated law book replaced by three laws on the pair (the
comparison, the link and the addition) and the target's law read from the answer rows by the learner's own routines,
whose operations' cuts and loops it then poses (bar3/scale/legalloops/LEGALLOOPS.md), as named groups of exact edits,
each asserted to match exactly once. Writes submission_stage3_legal.py; NAME=VALUE sets a constant (research builds,
written to research/).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/combined_legal/make_legal.py [--diff] [--dev] [NAME=VALUE ...]

(--diff: the edits' effect as a unified diff against fast2; --dev: a working build that may be over the size limit,
written as *_dev.py; --md5s: each cut's md5 printed instead of asserted, for writing a new cut.)
"""

from __future__ import annotations

import ast
import builtins
import difflib
import hashlib
import re
import subprocess
import symtable
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU = HERE.parents[1]
OUT = HERE / "submission_stage3_legal.py"
LIMIT = 256 * 1024                          # the evaluator's limit on a submission's size
BASE = ("f237a61", "integration/combined_fast/submission_stage3_fast2.py", "41b9a013ae9436d7b7a53e9e594a18db")


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def base() -> str:
    commit, path, want = BASE
    src = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=CPU, capture_output=True, text=True,
                         check=True).stdout
    assert md5(src) == want, (path, md5(src))
    return src


EDITS = []          # ("edit", group, old, new) or ("cut", group, start, end, md5, new): applied in order, each old
                    # text (or anchor) found exactly once in the text of the moment


def edit(group: str, old: str, new: str) -> None:
    EDITS.append(("edit", group, old, new))


def cut(group: str, start: str, end: str, digest: str, new: str = "") -> None:
    """The text from start (included) to end (excluded), its md5 digest, replaced by new."""
    EDITS.append(("cut", group, start, end, digest, new))


# ================================================================================================ doc
edit("doc", '''"""Stage 3, self-practising, with every variant of sp2 that certified on its own, round 2 (cpu_first/integration/
combined_sp2/COMBINED.md §8; make_combined.py --round 2 writes it from the certified files): sp2 (SELF_PRACTICE.md)
with the loop over T learned (LEARNED_LOOP.md: the count loop C.r[body], T in S, OUTER_*), generic instructions
(GENERIC_OPS.md: COND, SHIFT, BRING, g3's SP_* rules), the generated law book (SP_ABLATION.md 5.2), the fixed checks
(ABLATION_S3.md 8: FULL_CHECK, ABOVE_CHECK) and the strict build (STRICT.md: work budgets, never the clock; the kept
rows and the answer programs in buffers; Adam's own reset; no late lessons); new here, SHARED_ACCEPT, DOUBLE_COUNTS and
budgets set from the combined climbs (combined_fast/FAST.md §9). The rest is sp2's, as LEARNED_LOOP.md describes it.
"""''', '''"""Stage 3, self-practising, legal (cpu_first/integration/combined_legal/LEGAL_FILE.md; make_legal.py writes it from
fast2, combined_fast/FAST.md §9): sp2 (SELF_PRACTICE.md) with the loop over T learned (LEARNED_LOOP.md), generic
instructions (GENERIC_OPS.md), the fixed checks (ABLATION_S3.md 8) and the strict build (STRICT.md), made fast
(FAST.md); new here, a law book of three laws on the pair (BOOK: the comparison, the link and the addition) in place of
the generated one, whose grammar stays as the forms a law can take; the rest is read from the target: its law, read
from the answer rows by the learner's own routines (Reader), poses its operations' cuts and loops (read_pairs).
"""''')

# ================================================================================================ book
# The climb's book: three laws on the pair. The generated grammar (PATTERNS, LAW_BOOK, law_target) stays: the forms a
# law can take and how each is answered (facts, identities, the learner's routines); the reading picks among them.
edit("book", '''FUSE = 4               # a search runs up to this many chunks of its work in one batch
''', '''FUSE = 4               # a search runs up to this many chunks of its work in one batch
BOOK = (("lt", "red", "add"), ("pair",))  # the law book: the comparison, the link and the addition, on the pair
READ_MIN = 4           # the target's law, read (Reader): exact on every answer row it can be computed on, at least
READ_LINKS = 1000      # ... this many; a reduction there takes at most this many links; the reading is tried at
READ_TRIES = 3         # ... this many levels at most, from the first with the comparison, the link and the addition
''')
edit("book", '''          fuse=FUSE, round_retry=ROUND_RETRY)
''', '''          fuse=FUSE, round_retry=ROUND_RETRY, laws=BOOK[0], patterns=BOOK[1])
''')

# ================================================================================================ reader
# The reading (bar3/scale/legalloops/lread.py, in this file's terms): the learner's routines on the answer rows at
# full size; then the cuts (a digit's count, d) and loops (another row's number, o) of the read law's operations.
edit("reader", """

class SelfClimb:
""", """

READ_OPS = {"add": "add", "mul": "mul", "mod": "red"}  # the reading's operations, as the book's laws


class LawReader:
    \"\"\"The target's law, read from answer rows of one T (Nd, Xd, Yd) at full size by the learner's routines and the
    book's constructions only: left-deep chains of add, mul and mod over x and N, depth 1 then 2; the first that gives
    every row where it is computed (READ_MIN at least) its answer is read. add: the learner's addition (where the sum
    fits); mul: repeat_add (where both factors fit half the places); mod: the learner's link until its comparison
    says below (at most READ_LINKS links; the divisor not zero).\"\"\"

    def __init__(self, climb, answers):
        self.c, self.dev = climb, answers[0].device
        self.st = round_settings(climb.cells, climb.P_round)
        self.N, self.X, self.Y = (pad(v, self.st.P) for v in answers)
        self.gen, self.tried = gen_on(self.dev, 3), 0

    def run(self, law, ins, m):
        A, c = torch.zeros_like(ins[0]), torch.zeros_like(m, dtype=torch.long)
        i = torch.nonzero(m).flatten()
        if i.numel():
            A[i], c[i] = self.c.ops.run(law, "pair", self.st, self.c.lib, [v[i] for v in ins], self.gen, self.dev)
        return A, c

    def apply(self, op, a, b, m):
        \"\"\"op(a, b), and the rows where it was computed.\"\"\"
        h = self.st.P // 2
        if op == "add":
            m = m & (a[:, -1] == 0) & (b[:, -1] == 0)
            return self.run("add", (a, b), m)[0], m
        if op == "mul":
            m = m & (a[:, h:] == 0).all(1) & (b[:, h:] == 0).all(1)
            out, i = torch.zeros_like(a), torch.nonzero(m).flatten()
            if i.numel():
                out[i] = repeat_add(self.c.ops, self.st, self.c.lib, out[i], a[i], count_of(b[i]), self.gen, self.dev)
            return out, m
        m = m & (b != 0).any(1)
        v, left = a.clone(), m.clone()         # left: not yet below b
        for _ in range(READ_LINKS + 1):
            left &= self.run("lt", (v, b), left)[1] == 0
            if not bool(left.any()):
                break
            v = torch.where(left[:, None], self.run("red", (v, b), left)[0], v)
        return v, m & ~left

    def passes(self, v, m):
        return int(m.sum()) >= READ_MIN and bool(((v == self.Y).all(1) | ~m).all())

    def read(self):
        \"\"\"(the law read, its operations, its rows), or (None, (), 0).\"\"\"
        every = torch.ones(self.N.shape[0], dtype=torch.bool, device=self.dev)
        leaves, first = (("x", self.X), ("N", self.N)), []
        for (an, a), (bn, b), op in itertools.product(leaves, leaves, READ_OPS):
            v, m = self.apply(op, a, b, every)
            law, v, self.tried = f"{op}({an},{bn})", v * m[:, None], self.tried + 1
            if self.passes(v, m):
                return law, (op,), int(m.sum())
            if int(m.sum()) >= READ_MIN and not any(torch.equal(v, u) and torch.equal(m, w) for _, _, u, w in first):
                first.append((law, op, v, m))
        for (l1, o1, c, cm), op, (ln, l), side in itertools.product(first, READ_OPS, leaves, (0, 1)):
            v, m = self.apply(op, *((c, l) if side == 0 else (l, c)), cm)
            self.tried += 1
            if self.passes(v, m):
                law = f"{op}({l1},{ln})" if side == 0 else f"{op}({ln},{l1})"
                return law, tuple(dict.fromkeys((o1, op))), int(m.sum())
        return None, (), 0


def read_pairs(ops_):
    \"\"\"What a reading poses: each of its operations' laws (f, and f accumulated: add.f) on every pattern whose count
    is a digit (the operation cut to one digit, d) or another row's number (its loop over the digits, o).\"\"\"
    q = dict(NFORM, **{p: p[-1] for p in PATTERNS if "," in p})
    return [(l, p) for f in ops_ for l in (READ_OPS[f], "add." + READ_OPS[f]) if l in LAW_BOOK
            for p in LAW_BOOK[l][1] if q.get(p) in ("d", "o")]


class SelfClimb:
""")
edit("reader", """        self._focus, self._focus_classes = {}, {}        # root -> law -> its closest attempt's routines
        self.log = log
""", """        self._focus, self._focus_classes = {}, {}        # root -> law -> its closest attempt's routines
        self.log = log
        self._read, self._read_tries = None, 0  # the reading: {law, pairs} once one is read; the tries so far
""")
edit("reader", """    def pose(self, rows):
        pool, roots = [], []
        for pat in self.patterns:
            laws = [l for l in self.laws if pat in LAW_BOOK[l][1]]
""", """    def read_law(self, answers):
        \"\"\"The target's law (LawReader), what it poses (read_pairs), its record.\"\"\"
        t0, r = _clock(), LawReader(self, answers)
        law, ops_, n = r.read()
        if law is not None:
            self._read = dict(law=law, pairs=read_pairs(ops_))
        self.records.append(dict(task="read", level=self.level, law=law, ops=list(ops_), rows=n, tried=r.tried,
                                 answers=int(answers[0].shape[0]), seconds=round(_clock() - t0, 1)))

    def pose(self, rows, answers=None):
        pool, roots = [], []
        if self._read is None and answers is not None and self._read_tries < READ_TRIES and all(
                self.ops.has(l, "pair") for l in ("lt", "red", "add")):
            self._read_tries += 1
            self.read_law(answers)
        read = {}                               # pattern -> the laws the reading poses on it
        for l, p in (self._read or {}).get("pairs", ()):
            read.setdefault(p, []).append(l)
        for pat in self.patterns + [p for p in read if p not in self.patterns]:
            book = [l for l in self.laws if pat in LAW_BOOK[l][1]] if pat in self.patterns else []
            laws = list(dict.fromkeys(book + read.get(pat, [])))
""")
edit("reader", """            self.pool, roots = self.pose(rows)
            self.results, self._fit, self._focus = {}, {}, {}
            self.queue, self._level_roots = self.groups(roots, self.group), list(roots)
            self.phase = "probe"
            return f"L{self.level} pose: {len(self.pool)} candidates on {len(roots)} roots"
""", """            n = len(self.records)
            self.pool, roots = self.pose(rows, answers)
            self.results, self._fit, self._focus = {}, {}, {}
            self.queue, self._level_roots = self.groups(roots, self.group), list(roots)
            self.phase = "probe"
            return f"L{self.level} pose: {len(self.pool)} candidates on {len(roots)} roots" + "".join(
                f"; read {r['law']} on {r['rows']} of {r['answers']} answer rows, {r['tried']} laws tried, "
                f"{r['seconds']}s" for r in self.records[n:] if r.get("task") == "read")
""")

# ================================================================================================ room
# The new code is paid for by the generated book's rules for laws that neither the book (lt, red and add on the pair)
# nor a reading (f and add.f on d and o counts, f its operations: add, mul and red) ever poses: eq, min, max, div and
# sub, mul by zero, red on a base form. The grammar keeps the operations law_target answers.
edit("room", """OPS2 = {"lt": "c", "eq": "c", "min": "A", "max": "A", "red": "A", "div": "A", "add": "A", "sub": "A", "mul": "A"}
""", """OPS2 = {"lt": "c", "red": "A", "add": "A", "mul": "A"}
""")
edit("room", """            if law in ("lt", "eq", "min", "max") and na and nb:       # the order of r + q N forms, by the facts
                (ra, qa, _, _), (rb, qb, _, _) = na, nb
                lt = (qa < qb) | ((qa == qb) & ~ra & rb)
                same = (qa == qb) & (ra == rb)
                if law in ("lt", "eq"):
                    got = ("c", (lt if law == "lt" else same).long(), sel)
                else:
                    small = lt | same
                    pick = small if law == "min" else ~small
                    got = ("A", torch.where(pick[:, None], val(a), val(b)), sel)
            elif law in ("red", "div") and a == b:                   # red.zero; a div a = 1 (a != 0)
                if law == "red":
                    got = ("A", d.z, sel)
                elif a in ("x", "o"):
                    got = ("A", digit_string(torch.ones(d.n, dtype=torch.long, device=dev), P),
                           sel & ~(val(a) == 0).all(1))
            elif law == "red" and b == "n" and na:                    # the division law: red(r + q N) = r
                got = ("A", torch.where(na[0][:, None], d.x, d.z), sel)
            elif law == "div" and b == "n" and na:                    # ... and its quotient q (where it is read off)
                got = ("A", na[2], sel & na[3])
            elif law == "red" and b == "n" and BFORM.get(a) == "x":   # d + x B < B N for x' < N': the chain's range
                src = "x+dn" if ops.has("red", "x+dn") else "x+on"
                got = ("A", ops.run("red", src, st, lib, [val(a), d.N], gen, dev)[0], sel)
            elif law == "add" and b == "z":                           # add.identity
""", """            if law == "lt" and na and nb:                             # the order of r + q N forms, by the facts
                (ra, qa), (rb, qb) = na, nb
                got = ("c", ((qa < qb) | ((qa == qb) & ~ra & rb)).long(), sel)
            elif law == "red" and a == b:                             # red.zero
                got = ("A", d.z, sel)
            elif law == "red" and b == "n" and na:                    # the division law: red(r + q N) = r
                got = ("A", torch.where(na[0][:, None], d.x, d.z), sel)
            elif law == "add" and b == "z":                           # add.identity
""")
edit("room", """            elif law == "sub" and a == b:                             # sub.self
                got = ("A", d.z, sel)
            elif law == "sub" and b == "z":                           # sub.identity
                got = ("A", val(a), sel)
            elif law == "sub" and b == "n" and a in NFORM:            # (x + q N) - N = x + (q - 1) N, q >= 1
                q = d.count(NFORM[a])
                got = ("A", repeat_add(ops, st, lib, d.x, d.N, (q - 1).clamp(min=0), gen, dev), sel & (q >= 1))
            elif law == "mul" and b == "z":                           # mul.zero
                got = ("A", d.z, sel)
            elif law == "mul" and b in""", """            elif law == "mul" and b in""")

edit("room", """# The book, generated (integration/sp_ablation/SP_ABLATION.md §5). A pattern: the instance kinds (one data term per
# operand; probe i is of kind i mod len), how many digits its numbers need beyond the operands' width w ("carry":
# w + 1; "double": 2 w, wherever a number multiplies: the widest w that fits the registers), and a row filter ("ne":
# only rows with x' != N', where the base form's bound needs x' < N').
QS = ("1", "d", "o", "x")                 # the counts q of r + q b: one, a digit, another row's x, x itself
NFORM = {"x+n": "1", "x+dn": "d", "x+on": "o", "x+xn": "x"}       # r + q N with r = x (x' <= N')
BFORM = {"Bx+d": "x", "Bo+d": "o"}                                # r + q B with r = d (d < B)
PATTERNS = {"pair": ((("x", "n"), ("n", "x"), ("x", "x"), ("x", "z")), "carry", "")}
PATTERNS.update({t: (((t, "n"),), "carry" if q in ("1", "d") else "double", "") for t, q in NFORM.items()})
PATTERNS.update({t: (((t, "n"),), "carry", "ne" if q == "x" else "") for t, q in BFORM.items()})
""", """# The book's grammar (integration/sp_ablation/SP_ABLATION.md §5; its base forms d + q B cut, LEGAL_FILE.md §2). A
# pattern: the instance kinds (one data term per operand; probe i is of kind i mod len) and how many digits its
# numbers need beyond the operands' width w ("carry": w + 1; "double": 2 w, wherever a number multiplies: the widest w
# that fits the registers).
QS = ("1", "d", "o", "x")                 # the counts q of r + q b: one, a digit, another row's x, x itself
NFORM = {"x+n": "1", "x+dn": "d", "x+on": "o", "x+xn": "x"}       # r + q N with r = x (x' <= N')
PATTERNS = {"pair": ((("x", "n"), ("n", "x"), ("x", "x"), ("x", "z")), "carry", "")}
PATTERNS.update({t: (((t, "n"),), "carry" if q in ("1", "d") else "double", "") for t, q in NFORM.items()})
""")
edit("room", """        idx = torch.arange(n, device=device)
        if filt == "ne":
            idx = torch.nonzero(~eq[:3 * n]).flatten()[:n]
""", """        idx = torch.arange(n, device=device)
""")
edit("room", """        elif name in BFORM:                    # d + q B: q's digits moved up one place, d below
            q = self.x if BFORM[name] == "x" else self.o
            v = torch.cat([self.dv[:, None], q[:, :st.P - 1]], 1)
        else:
""", """        else:
""")
edit("room", """        (r is x, q, the digits of the quotient by N, where those are read off), or None for a term of no such form
        (o, a digit, a base form).\"\"\"
        zero = torch.zeros(self.n, dtype=torch.long, device=self.device)
        one, every = digit_string(zero + 1, self.st.P), torch.ones(self.n, dtype=torch.bool, device=self.device)
        if name == "z":
            return zero.bool(), zero, self.z, every
        if name == "n":
            return zero.bool(), zero + 1, one, every
        if name == "x":                        # x div N: 0, or 1 where x' = N'
            return ~self.xz & ~self.eq, self.eq.long(), torch.where(self.eq[:, None], one, self.z), every
        if name in NFORM:                      # (x + q N) div N: q's digits, but q + 1 where x' = N' (not read off)
            q = self.count(NFORM[name])
            qd = {"1": one, "d": self.d, "o": self.o, "x": self.x}[NFORM[name]]
            return ~self.xz & ~self.eq, q + self.eq.long(), qd, ~self.eq
        return None
""", """        (r is x, q), or None for a term of no such form (o, a digit).\"\"\"
        zero = torch.zeros(self.n, dtype=torch.long, device=self.device)
        if name == "z":
            return zero.bool(), zero
        if name == "n":
            return zero.bool(), zero + 1
        if name == "x":                        # x div N: 0, or 1 where x' = N'
            return ~self.xz & ~self.eq, self.eq.long()
        if name in NFORM:                      # (x + q N) div N: q, but q + 1 where x' = N'
            return ~self.xz & ~self.eq, self.count(NFORM[name]) + self.eq.long()
        return None
""")
edit("room", """        if name in BFORM:
            return (self.st.B - 1) + self.st.B * big
        raise ValueError(name)
""", """        raise ValueError(name)
""")
edit("room", """SP_LOOK = 300_000      # a lookahead's cap (ct2a2)
""", "")
edit("room", """pure_first=SP_PURE_FIRST, look_budget=SP_LOOK,
""", """pure_first=SP_PURE_FIRST,
""")
edit("room", """round_cost=5, look_budget=300_000, settings=""", """round_cost=5, settings=""")


# ================================================================================================ the build
def undefined_globals(src: str) -> set:
    """Names a function reads as globals that neither the module's top level nor builtins define (symtable)."""
    top = symtable.symtable(src, "legal", "exec")
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    out = set()

    def walk(t):
        for s in t.get_symbols():
            if s.is_global() and s.get_name() not in defined and not hasattr(builtins, s.get_name()):
                out.add(s.get_name())
        for c in t.get_children():
            walk(c)
    walk(top)
    return out


SHOW_MD5 = "--md5s" in sys.argv          # (prints each cut's md5 instead of asserting it: writing a new cut)


def build(sets: dict | None = None, dev: bool = False) -> tuple[str, list]:
    """The legal file's text and the log of the edit groups applied; dev: over the size limit is allowed (a working
    build, never the submission)."""
    src = base()
    log = []
    for e in EDITS:
        if e[0] == "edit":
            _, group, old, new = e
            n = src.count(old)
            assert n == 1, (group, n, old[:120])
            src = src.replace(old, new)
        else:
            _, group, start, end, digest, new = e
            assert src.count(start) == 1 and src.count(end) == 1, (group, start[:80], end[:80])
            i = src.index(start)
            j = src.index(end, i)
            got = md5(src[i:j])
            if SHOW_MD5:
                print(f"cut {got} {start[:60]!r}")
            else:
                assert got == digest, (group, start[:80], got)
            src = src[:i] + new + src[j:]
        log.append(group)
    for name, value in (sets or {}).items():                  # research builds only
        src, n = re.subn(rf"^{name} = .*$", f"{name} = {value}", src, count=1, flags=re.M)
        assert n == 1, name
    ast.parse(src)
    compile(src, OUT.name, "exec")
    undef = undefined_globals(src) - {"big", "name", "self"}  # (symtable's false alarm on book_gen's Draw.top, as
    assert not undef, undef                                   #  make_fast.py has it)
    sys.path.insert(0, str(CPU.parent / "one-layer-deeper"))
    from submission_validation import validate_submission_source
    validate_submission_source("submission.py", src, 2 * LIMIT if dev else LIMIT)
    return src, log


def main(argv):
    sets = dict(a.split("=", 1) for a in argv if "=" in a and not a.startswith("--"))
    src, log = build(sets, dev="--dev" in argv)
    if "--diff" in argv:
        sys.stdout.writelines(difflib.unified_diff(base().splitlines(True), src.splitlines(True), "fast2", "legal",
                                                   n=1))
        return
    for g in dict.fromkeys(log):
        print(f"{log.count(g):4d}  {g}")
    out = OUT if not sets else HERE / "research" / (OUT.stem + "_" + "_".join(
        f"{k}-{v}" for k, v in sets.items()).replace("/", "") + ".py")
    if "--dev" in argv:
        out = out.with_name(out.stem + "_dev.py")
    out.parent.mkdir(exist_ok=True)
    out.write_text(src)
    print(f"wrote {out}: {len(src.splitlines())} lines, {len(src.encode()):,} bytes "
          f"({LIMIT - len(src.encode()):,} under the limit), md5 {md5(src)}")


if __name__ == "__main__":
    main(sys.argv[1:])
