"""Builds the deep file (integration/legal_deep/DEEP.md): the wider file (legal_wider's submission_stage3_wider.py,
its committed bytes at 80082c0, md5 asserted) with phase E's changes: the law reader's fourth Horner step, a bigger
work budget, and the docstring's first line. Named groups of exact edits, each asserted to match exactly once.
Writes submission_stage3_deep.py; NAME=VALUE sets a constant (research builds, written to research/).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_deep/make_deep.py [--diff] [--groups=G,...] [NAME=VALUE ...]

(--diff: the edits' effect as a unified diff against the wider file.)
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
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU = HERE.parents[1]
OUT = HERE / "submission_stage3_deep.py"
LIMIT = 256 * 1024                          # the evaluator's limit on a submission's size
BASE = ("80082c0", "integration/legal_wider/submission_stage3_wider.py", "59060b5cd7a7b536e25302e1d5be7ef7")


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def base() -> str:
    commit, path, want = BASE
    src = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=CPU, capture_output=True, text=True,
                         check=True).stdout
    assert md5(src) == want, (path, md5(src))
    return src


EDITS = []          # (group, a, b): applied in order, old text a found exactly once in the text of the moment


def edit(group: str, old: str, new: str) -> None:
    EDITS.append((group, old, new))


# ================================================================================================ doc
edit("doc", """\"\"\"Stage 3, self-practising, legal (cpu_first/integration/legal_wider/WIDER.md; make_wider.py writes it from the family
file, legal_family/FAMILY_FILE.md, the last of a line of builds from fast2, combined_fast/FAST.md §9): sp2""",
     """\"\"\"Stage 3, self-practising, legal (cpu_first/integration/legal_deep/DEEP.md; make_deep.py writes it from the wider
file, legal_wider/WIDER.md, the last of a line of builds from fast2, combined_fast/FAST.md §9): sp2""")

# ================================================================================================ reader
edit("reader", "READ_POLY = 3          # ... values checked READ_CHUNK at a time",
     "READ_POLY = 4          # ... values checked READ_CHUNK at a time")

# ================================================================================================ research arms
# (b) check: a reading's own law (read_target) and x's polynomials (poly) multiply wherever the product fits the
# places, not only where both factors fit half of them: a window such as 3 a b + 2 is then checked on draws with
# multi-digit a b at every size (with half the places it said something only at one digit, and three2's routine for
# it multiplied by b's last digit).
edit("check", """    def __init__(self, ops, lib, st, dev, answers=None):
        self.ops, self.lib, self.st, self.dev = ops, lib, st, dev""",
     """    def __init__(self, ops, lib, st, dev, answers=None, full=False):
        self.ops, self.lib, self.st, self.dev, self.full = ops, lib, st, dev, full  # full: a product where it fits""")
edit("check", """        fits = lambda v: (v[:, self.st.P // 2:] == 0).all(1)  # noqa: E731""",
     """        fits = lambda v: (v[:, self.st.P // 2:] == 0).all(1)  # noqa: E731
        places = lambda v: ((v != 0) * torch.arange(1, v.shape[1] + 1, device=v.device)).amax(1)  # noqa: E731""")
edit("check", """            m = m & fits(a) & fits(b)""",
     """            m = m & ((places(a) + places(b) <= self.st.P) if self.full else fits(a) & fits(b))""")
edit("check", """    v, m = LawReader(ops, lib, d.st, d.device).value(t, {"a": d.x, "b": d.o, "N": d.N}, m)""",
     """    v, m = LawReader(ops, lib, d.st, d.device, full=True).value(t, {"a": d.x, "b": d.o, "N": d.N}, m)""")
edit("check", """        one = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        cs = [(str(c), self.value(str(c), {}, one)[0]) for c in range(1, self.st.B + 1)]""",
     """        one, self.full = torch.ones(self.n, dtype=torch.bool, device=self.dev), True
        cs = [(str(c), self.value(str(c), {}, one)[0]) for c in range(1, self.st.B + 1)]""")

# (c1) reduced: each window not already reduced is also posed reduced mod N (a round may then chain them: Horner's
# rule mod N stays exact, and every value keeps to the places).
edit("reduced", """        pose(name, f"w{h}", w)
    return out""", """        pose(name, f"w{h}", w)
    for w, h in windows(t):  # each also reduced mod N, after them all (one the same as a window is posed once)
        if w[0] != "mod":
            r_, name = ("mod", [w, "N"]), "w.mod." + ".".join(ops_of(w))
            while name in READ_LAWS and READ_LAWS[name] != r_:
                name += "2"
            pose(name, f"w{h}", r_)
    return out""")

# (c2) order: x's polynomials try the constants before x at each step, so the first law read is Horner's form of
# a x^k + b x + c, (a x + b) x or ((a x) x + b) x, whose windows a round can chain.
edit("order", """        steps = [(f, l, 0) for f in ("mul", "add") for l in [("x", self.X)] + cs[f == "mul":]]""",
     """        steps = [(f, l, 0) for f in ("mul", "add") for l in cs[f == "mul":] + [("x", self.X)]]""")
# (c') horner: x's polynomials by Horner's steps only, x x first as before but no "+ x" (Horner's rule never adds x),
# so a x^3 + b x + c is read ((x x) a + b) x + c, never as (x^3 + x) a when a = b; reduced1: the reduced windows also
# of one operation ((a b) mod N), so a x^3 + c has its two calls too.
edit("horner", """        steps = [(f, l, 0) for f in ("mul", "add") for l in [("x", self.X)] + cs[f == "mul":]]""",
     """        steps = [(f, l, 0) for f in ("mul", "add") for l in [("x", self.X)][:f == "mul"] + cs[f == "mul":]]""")
edit("reduced1", """def windows(t):""", """def windows(t, least=2):""")
edit("reduced1", """                if o >= 2 and 0 < len(hs) <= READ_WINDOW[1] and law_text(w) not in seen:""",
     """                if o >= least and 0 < len(hs) <= READ_WINDOW[1] and law_text(w) not in seen:""")
edit("reduced1", """        pose(name, f"w{h}", w)
    return out""", """        pose(name, f"w{h}", w)
    for w, h in windows(t, 1):  # each also reduced mod N, after them all (one the same as a window is posed once)
        if w[0] != "mod":
            r_, name = ("mod", [w, "N"]), "w.mod." + ".".join(ops_of(w))
            while name in READ_LAWS and READ_LAWS[name] != r_:
                name += "2"
            pose(name, f"w{h}", r_)
    return out""")
# ================================================================================================ budget
edit("budget", "CLIMB_WORK = 4_300_000_000    # ... and the whole climb's, from its start (0: none): combined_fast/FAST.md §9 sets both",
     "CLIMB_WORK = 8_000_000_000    # ... and the whole climb's, from its start (0: none): FAST.md §9, legal_deep/DEEP.md §2")
# (d) keepx: when the round by calls finds none and the law read uses x again above a deeper operation (Horner's
# x (...) + c), the same search once more with x kept in S (the round starts LDA.X STA.S, S a routed input): a learned
# routine may leave X and Y as scratch, and the round's second call needs x.
edit("keepx", """    def macro_classes(self, lib):""", """    def macro_classes(self, lib, sources=SOURCES):""")
edit("keepx", """            for srcs in itertools.product(SOURCES, repeat=len(bd)):""",
     """            for srcs in itertools.product(sources, repeat=len(bd)):""")
edit("keepx", """                if mres.found_cost is not None:
                    found, accepted = f"{mres.found_cost} calls", [unwrap(p) for p in mres.accepted]
            if found is None:""", """                if mres.found_cost is not None:
                    found, accepted = f"{mres.found_cost} calls", [unwrap(p) for p in mres.accepted]
                if found is None and self._read and len(xdepths(tree(self._read["law"]))) > 1:  # x kept in S
                    pre, p2 = tuple(parse("LDA.X STA.S")), round_problem(self.cells, answers, 8, self.P_round, self.device)
                    run_items(p2.settings[0], p2.roots[0], pre, lib)
                    m2 = bfs(p2, self.macro_classes(lib, SOURCES + ("S",)), lib, max_cost=self.macro_cost,
                             max_states=budget or self.round_budget, accept=lambda ps: ok([pre + unwrap(p) for p in ps]),
                             stop_at=stop_at, chunk=max(1, self.chunk // 4), fuse=self.fuse)
                    stop += f"; x kept: levels {m2.level_sizes} {m2.stop}"
                    if m2.found_cost is not None:
                        found, accepted = f"{m2.found_cost} calls, x kept", [pre + unwrap(p) for p in m2.accepted]
            if found is None:""")
edit("keepx", """def tree(s):""", '''def xdepths(t, d=0):
    """The depths x occurs at in a law's tree."""
    return {d} if t == "x" else set() if isinstance(t, str) else set().union(*(xdepths(a, d + 1) for a in t[1]))


def tree(s):''')
edit("budget13", "CLIMB_WORK = 4_300_000_000    # ... and the whole climb's, from its start (0: none): combined_fast/FAST.md §9 sets both",
     "CLIMB_WORK = 13_000_000_000   # ... and the whole climb's, from its start (0: none): FAST.md §9, legal_deep/DEEP.md §2")

# ================================================================================================ room
# (for keepx and reduced1: seven docstrings that a design doc explains cut to their first words and the pointer)
DOCS = [("SelfClimb.search_outer", "The answer's program (LEARNED_LOOP.md §3): returns (programs, the library with the "
         "rounds, a record)."),
        ("bfs_roots", "bfs_multi for many roots in one search (SELF_PRACTICE.md §2.3)."),
        ("law_target", "The law's target on a draw (SELF_PRACTICE.md §2.2), or None where the law says nothing on it."),
        ("answer_rows", "The rows whose answers the lessons read: the batch's smallest nonzero T, at most ANS_WORK // "
         "(T + 1) of them (general_lessons/LESSONS.md)."),
        ("SelfClimb.outer_classes", "The answer program's vocabulary (LEARNED_LOOP.md §3)."),
        ("SteMachine", "A found program on one-hot digits, the prediction path (LEARNED_LOOP.md §2; INTEGRATION.md)."),
        ("LibraryMachine.outer_ste", "The answer's program on one-hot digits (LEARNED_LOOP.md §4).")]
OFFICIAL_AB = ("doc", "reader", "check", "budget")  # the plan without keepx (8e9)
OFFICIAL = ("doc", "reader", "check", "budget13", "reduced1", "horner", "keepx", "room")


def docstring_at(src: str, name: str) -> tuple[str, str]:
    """(the docstring's source lines, their indentation) for the def or class with that dotted name, found once."""
    found = []

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.ClassDef)):
                if prefix + ch.name == name:
                    found.append(ch)
                walk(ch, prefix + ch.name + ".")
    walk(ast.parse(src), "")
    assert len(found) == 1, (name, len(found))
    d = found[0].body[0]
    assert isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant) and isinstance(d.value.value, str), name
    lines = src.splitlines(True)
    return "".join(lines[d.lineno - 1:d.end_lineno]), " " * d.col_offset


def undefined_globals(src: str) -> set:
    """Names a function reads as globals that neither the module's top level nor builtins define (symtable)."""
    top = symtable.symtable(src, "deep", "exec")
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


def build(sets: dict | None = None, groups=OFFICIAL) -> tuple[str, list]:
    """The deep file's text and the log of the edit groups applied (research arms: more groups, over the limit)."""
    src = base()
    log = []
    for group, old, new in EDITS:
        if group not in groups:
            continue
        n = src.count(old)
        assert n == 1, (group, n, old[:120])
        src = src.replace(old, new)
        log.append(group)
    if "room" in groups:
        for name, new in DOCS:
            old, ind = docstring_at(src, name)
            new = "\n".join(textwrap.wrap(new + '"""', 120, initial_indent=ind + '"""', subsequent_indent=ind,
                                          break_long_words=False, break_on_hyphens=False)) + "\n"
            assert src.count(old) == 1, name
            src = src.replace(old, new)
            log.append("room")
    for name, value in (sets or {}).items():                  # research builds only
        src, n = re.subn(rf"^{name} = .*$", f"{name} = {value}", src, count=1, flags=re.M)
        assert n == 1, name
    ast.parse(src)
    compile(src, OUT.name, "exec")
    undef = undefined_globals(src) - {"big", "name", "self"}  # (symtable's false alarm on book_gen's Draw.top)
    assert not undef, undef
    sys.path.insert(0, str(CPU.parent / "one-layer-deeper"))
    from submission_validation import validate_submission_source
    validate_submission_source("submission.py", src, LIMIT if set(groups) <= set(OFFICIAL) else 2 * LIMIT)
    return src, log


def main(argv):
    sets = dict(a.split("=", 1) for a in argv if "=" in a and not a.startswith("--"))
    more = [g for a in argv if a.startswith("--groups=") for g in a.split("=", 1)[1].split(",")]
    src, log = build(sets, OFFICIAL + tuple(more))
    if "--diff" in argv:
        sys.stdout.writelines(difflib.unified_diff(base().splitlines(True), src.splitlines(True), "wider", "deep",
                                                   n=1))
        return
    for g in dict.fromkeys(log):
        print(f"{log.count(g):4d}  {g}")
    out = OUT if not (sets or more) else HERE / "research" / (OUT.stem + "_" + "_".join(
        more + [f"{k}-{v}" for k, v in sets.items()]).replace("/", "") + ".py")
    out.parent.mkdir(exist_ok=True)
    out.write_text(src)
    print(f"wrote {out}: {len(src.splitlines())} lines, {len(src.encode()):,} bytes "
          f"({LIMIT - len(src.encode()):,} under the limit), md5 {md5(src)}")


if __name__ == "__main__":
    main(sys.argv[1:])
