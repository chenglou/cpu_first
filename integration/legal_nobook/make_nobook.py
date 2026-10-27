"""Builds the no-book file (integration/legal_nobook/NOBOOK.md): the wide file (legal_wide's submission_stage3_wide.py,
md5 asserted, phase A) with no law left in its book. The cheapest reading, made before any routine exists, poses the
first laws: if every answer row is below one prompt field F (digit strings compared from the top), the law ends in a
reduction by F, and the reduction's step x + d F is posed cut to its parts on the pair (the comparison, the link and
the addition); the wide reading follows once those are routines. Named groups of exact edits, each asserted to match
exactly once. Writes submission_stage3_nobook.py; NAME=VALUE sets a constant (research builds, written to research/).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_nobook/make_nobook.py [--diff] [--dev] [NAME=VALUE ...]

(--diff: the edits' effect as a unified diff against the wide file; --dev: a working build that may be over the size
limit, written as *_dev.py.)
"""

from __future__ import annotations

import ast
import builtins
import difflib
import hashlib
import re
import symtable
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU = HERE.parents[1]
OUT = HERE / "submission_stage3_nobook.py"
LIMIT = 256 * 1024                          # the evaluator's limit on a submission's size
BASE = (CPU / "integration" / "legal_wide" / "submission_stage3_wide.py", "3bb5ceebf1de115501efc3398af8a457")


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def base() -> str:
    path, want = BASE
    src = path.read_text()
    assert md5(src) == want, (path, md5(src))
    return src


EDITS = []          # ("edit", group, old, new): applied in order, each old text found exactly once in the text of
                    # the moment


def edit(group: str, old: str, new: str) -> None:
    EDITS.append(("edit", group, old, new))


# ================================================================================================ doc
edit("doc", '''(FAST.md); new here, a law book of three laws on the pair (BOOK: the comparison, the link and the addition) in place of
the generated one, whose grammar stays as the forms a law can take; the rest is read from the target (legal_wide/
WIDE.md): its law, read from the answer rows by the learner's own routines over a wide grammar (LawReader), poses its
operations' cuts and loops (read_pairs).
"""''', '''(FAST.md); new here, no law in the book (legal_nobook/NOBOOK.md): every law practised is read from the target. First
the cheapest reading (read_fact: every answer below N, so the law ends in a reduction by N, whose step is posed cut to
its parts); then its law, read from the answer rows by the learner's own routines over a wide grammar (LawReader),
poses its operations' cuts and loops (read_pairs). The grammar stays as the forms a law can take.
"""''')

# ================================================================================================ book
# The book: empty. The step's parts: the designed cut table of a reduction's step (round 6's STEP_CUTS, the link and
# the comparison, with the addition, since the step's x + d N is a sum).
edit("book", '''BOOK = (("lt", "red", "add"), ("pair",))  # the law book: the comparison, the link and the addition, on the pair
''', '''BOOK = ((), ())        # the law book: no law; read_fact and the reading pose every one
STEP_PARTS = ("lt", "red", "add")  # a reduction's step x + d F cut to its parts, on the pair: the comparison, the link
STEP_ADD = True        # ... and (x + d F is a sum) the addition, posed with the step (False: once the reader needs it)
''')

# ================================================================================================ fact
edit("fact", '''        self._read, self._read_tries = None, 0  # the reading: {law, pairs} once one is read; the tries so far
''', '''        self._read, self._read_tries = None, 0  # the reading: {law, pairs} once one is read; the tries so far
        self._fact = None                       # the cheapest reading: {field, parts} once made
''')
edit("fact", '''    def pose(self, rows, answers=None):
        pool, roots = [], []
        if self._read is None and answers is not None and self._read_tries < READ_TRIES and all(
                self.ops.has(l, "pair") for l in ("lt", "red", "add")):
            self._read_tries += 1
            self.read_law(answers)
        read = {}                               # pattern -> the laws the reading poses on it
        for l, p in (self._read or {}).get("pairs", ()):
''', '''    def read_fact(self, answers):
        """The cheapest reading, before any routine (NOBOOK.md §2): every answer row below a prompt field F, as digit
        strings from their top place (compare), reads a reduction by F; by N, its step's parts are posed."""
        N, X, Y = answers
        F = next((f for f, v in (("N", N), ("x", X)) if bool((compare(Y, v) < 0).all())), None)
        s = lambda d: "".join(map(str, d.flip(0).tolist())).lstrip("0") or "0"           # noqa: E731
        self._fact = dict(field=F, parts=[p for p in STEP_PARTS if STEP_ADD or p != "add"] if F == "N" else [])
        self.records.append(dict(task="fact", level=self.level, rows=[f"{s(n)}:{s(y)}" for n, y in zip(N, Y)],
                                 **self._fact))

    def pose(self, rows, answers=None):
        pool, roots = [], []
        if self._fact is None and answers is not None:
            self.read_fact(answers)
        if self._read is None and answers is not None and self._read_tries < READ_TRIES and all(
                self.ops.has(l, "pair") for l in ("lt", "red", "add")):
            self._read_tries += 1
            self.read_law(answers)
        parts = list((self._fact or {}).get("parts", ()))
        if not STEP_ADD and self.ops.has("lt", "pair") and self.ops.has("red", "pair"):
            parts.append("add")                 # the reader's construction asks for the addition
        read = {"pair": parts} if parts else {}  # pattern -> the laws the readings pose on it
        for l, p in (self._read or {}).get("pairs", ()):
''')
edit("fact", '''            self.pool, roots = self.pose(rows, answers)
            self.results, self._fit, self._focus = {}, {}, {}
            self.queue, self._level_roots = self.groups(roots, self.group), list(roots)
            self.phase = "probe"
            return f"L{self.level} pose: {len(self.pool)} candidates on {len(roots)} roots" + "".join(
                f"; read {r['law']} on {r['rows']} of {r['answers']} answer rows, {r['tried']} laws tried, "
                f"{r['seconds']}s" for r in self.records[n:] if r.get("task") == "read")
''', '''            self.pool, roots = self.pose(rows, answers)
            if not roots and self._fact is None:
                return None                     # (no answer rows to read yet: posed again at the next unit)
            self.results, self._fit, self._focus = {}, {}, {}
            self.queue, self._level_roots = self.groups(roots, self.group), list(roots)
            self.phase = "probe"
            return f"L{self.level} pose: {len(self.pool)} candidates on {len(roots)} roots" + "".join(
                f"; read {r['law']} on {r['rows']} of {r['answers']} answer rows, {r['tried']} laws tried, "
                f"{r['seconds']}s" if r["task"] == "read" else f"; fact: answers below {r['field']} on "
                f"{len(r['rows'])} rows (N:y {' '.join(r['rows'])}): pose {r['parts']}"
                for r in self.records[n:] if r.get("task") in ("read", "fact"))
''')

# ================================================================================================ room
# Paid for by branches the file never takes with its settings: deep_union's other mode (SP_DEEP_UNION is False), the
# deep probe without a focus (SP_DEEP_FOCUS is "closure"), pure_first's other mode ("scratch"), and the guard for
# choosing rules other than the concept rules (lpca).
edit("room", '''                        if self.deep_by == "bindings" and self.deep_union:   # what the closest attempts used at any
                                                                            # of its bindings (else at this root)
                            for k, fo in self._focus.items():
                                if k.split("@")[0] == r.pattern:
                                    fs |= fo.get(name, set())
                        else:
                            fs |= self._focus.get(r.key, {}).get(name, set())
''', '''                        fs |= self._focus.get(r.key, {}).get(name, set())   # what the closest attempt used here
''')
edit("room", '''            elif deep:
                fit = run("deep", comp, self.lib, self.deep_budget, [(r, open_of(r)) for r in group])
                for r in group:
                    self._fit[r.key] = fit[r.key]
            else:
''', '''            else:
''')
edit("room", '''            elif self.pure_first:                   # a program that changes no register but A
                pure = [lambda q, r=r: side_effects(q, lib, self.settings[-1], r.roots[-1], r.binding)[1] == 0
                        for r in owner]
''', "")
edit("room", '''                        if self.alias_fit:      # (a goal state here is an alias: acceptance rejected it)
                            ff = torch.where(g.test(part, lib), torch.zeros_like(ff), ff)
''', '''                        ff = torch.where(g.test(part, lib), torch.zeros_like(ff), ff)   # (an alias: rejected)
''')
edit("room", '''                    fit[r.key][g.name] = ((max(f) - base) / (tot - base) if tot > base and (self.alias_fit or  # of reach)
                                                                                              not x["aliases"])
                                          else 0.0)
''', '''                    fit[r.key][g.name] = (max(f) - base) / (tot - base) if tot > base else 0.0   # of reach)
''')
edit("room", '''    """The target's law, read from answer rows of one T (N, x, y) at full size by the learner's routines and the
    book's constructions only (legal_wide/WIDE.md §2): left-deep chains of add, sub, mul, pow and mod over x, N and
    the data's constants (a gap by which a reduction mod N misses every answer), depth by depth; the first exact on
    every row where it is computed (READ_MIN at least). mul and mod repeat the addition and the link by doubling."""
''', '''    """The target's law, read from answer rows of one T (N, x, y) by the learner's routines and the book's
    constructions only (legal_wide/WIDE.md §2): chains of add, sub, mul, pow and mod over x, N and the data's
    constants, depth by depth; the first exact on every row where it is computed."""
''')
edit("room", '''        if self.rule not in ("lpc", "lpca"):
            raise ValueError(self.rule)
''', "")


# ================================================================================================ the build
def undefined_globals(src: str) -> set:
    """Names a function reads as globals that neither the module's top level nor builtins define (symtable)."""
    top = symtable.symtable(src, "nobook", "exec")
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


def build(sets: dict | None = None, dev: bool = False) -> tuple[str, list]:
    """The no-book file's text and the log of the edit groups applied; dev: over the size limit is allowed (a
    working build, never the submission)."""
    src = base()
    log = []
    for _, group, old, new in EDITS:
        n = src.count(old)
        assert n == 1, (group, n, old[:120])
        src = src.replace(old, new)
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
        sys.stdout.writelines(difflib.unified_diff(base().splitlines(True), src.splitlines(True), "wide", "nobook",
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
