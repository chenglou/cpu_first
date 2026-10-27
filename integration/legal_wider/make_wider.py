"""Builds the wider file (integration/legal_wider/WIDER.md): the family file (legal_family's
submission_stage3_family.py, its committed bytes at ce9f118, md5 asserted) with phase D's changes: ROUNDT's rows in
registered buffers; ROUNDT on the round machine (run_prog's draws and work, so the same verdicts, faster); a third
stage of the law reader, x's polynomials with small coefficients by Horner's rule; each constant read posed with the
constants it is built from; the stale first line of the docstring; room. Named groups of exact edits, each asserted
to match exactly once. Writes submission_stage3_wider.py; NAME=VALUE sets a constant (research builds, written to
research/).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_wider/make_wider.py [--diff] [--dev] [--skip=GROUP] [NAME=VALUE ...]

(--diff: the edits' effect as a unified diff against the family file; --dev: a working build that may be over the
size limit, written as *_dev.py.)
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
OUT = HERE / "submission_stage3_wider.py"
LIMIT = 256 * 1024                          # the evaluator's limit on a submission's size
BASE = ("ce9f118", "integration/legal_family/submission_stage3_family.py", "457216d746d7a21e854a2d2d3673c9d6")


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def base() -> str:
    commit, path, want = BASE
    src = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=CPU, capture_output=True, text=True,
                         check=True).stdout
    assert md5(src) == want, (path, md5(src))
    return src


EDITS = []          # (kind, group, a, b): applied in order. "edit": old text a found exactly once in the text of the
                    # moment, replaced by b; "doc": the docstring of the def or class named a (dotted, as
                    # Class.method) replaced by b, wrapped at 120 columns; "sub": the regex a replaced by b, its count
                    # of matches asserted (the group's name ends in it)


def edit(group: str, old: str, new: str) -> None:
    EDITS.append(("edit", group, old, new))


def doc(name: str, new: str) -> None:
    EDITS.append(("doc", "room", name, new))


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


# ================================================================================================ doc
edit("doc", """\"\"\"Stage 3, self-practising, legal (cpu_first/integration/combined_legal/LEGAL_FILE.md; make_legal.py writes it from
fast2, combined_fast/FAST.md §9): sp2 (SELF_PRACTICE.md) with the loop over T learned (LEARNED_LOOP.md), generic
instructions (GENERIC_OPS.md), the fixed checks (ABLATION_S3.md 8) and the strict build (STRICT.md), made fast
(FAST.md); new here, no law in the book (legal_nobook/NOBOOK.md): every law practised is read from the target. First
the cheapest reading (read_fact: every answer below N, so the law ends in a reduction by N, whose step is posed cut to
its parts); then its law, read from the answer rows by the learner's own routines over a wide grammar (LawReader),
poses its operations' cuts and loops (read_pairs), its constants and its windows (read_more, legal_family/
FAMILY_FILE.md); a round must give the answers of every count kept (ROUNDT).
\"\"\"""", """\"\"\"Stage 3, self-practising, legal (cpu_first/integration/legal_wider/WIDER.md; make_wider.py writes it from the family
file, legal_family/FAMILY_FILE.md, the last of a line of builds from fast2, combined_fast/FAST.md §9): sp2
(SELF_PRACTICE.md) with the loop over T learned (LEARNED_LOOP.md), generic instructions (GENERIC_OPS.md), the fixed
checks (ABLATION_S3.md 8) and the strict build (STRICT.md), made fast (FAST.md), with no law in its book
(legal_nobook/NOBOOK.md): every law practised is read from the target. First the cheapest reading (read_fact: every
answer below N, so the law ends in a reduction by N, whose step is posed cut to its parts); then its law, read from the
answer rows by the learner's own routines over a wide grammar (LawReader, with x's polynomials by Horner's rule), poses
its operations' cuts and loops (read_pairs), its constants with those they are built from, and its windows
(read_more); a round must give the answers of every count kept (ROUNDT, its rows in the buffers).
\"\"\"""")

# ================================================================================================ buffers: ROUNDT's rows
edit("buffers", """ROUNDT = 4096          # a round must also give the answers of every count T > 1, run T times: the newest this many
                       # rows of each count the batches showed (keep_counts; 0: off)
""", """ROUNDT = 4096          # a round must also give the answers of every count T > 1, run T times: the newest this many
ROUNDT_SLOTS = 4       # ... rows of each of at most this many counts the batches showed (buffers, keep_counts; 0: off)
""")
edit("buffers", """        self.register_buffer("count_n", torch.zeros(10 ** K, dtype=torch.long))  # (N, x, count, y) by count (keep_counts)
""", """        self.register_buffer("count_n", torch.zeros(10 ** K, dtype=torch.long))  # (N, x, count, y) by count (keep_counts)
        self.register_buffer("more_rows", torch.zeros(ROUNDT_SLOTS, ROUNDT, 4, W, dtype=torch.long))  # ROUNDT's, as
        self.register_buffer("more_n", torch.zeros(ROUNDT_SLOTS, dtype=torch.long))    # count_rows', by slot: their
        self.register_buffer("more_s", torch.zeros(ROUNDT_SLOTS, dtype=torch.long))    # number, their count's slot
""")
edit("buffers", """        self.more: dict = {}                  # ROUNDT: the newest answer rows of each count (keep_counts)
""", "")
edit("buffers", """            for s in rows:                                  # ROUNDT's: the newest ROUNDT of each count
                ctl.more[s] = torch.cat([ctl.more.get(s, new[:0]), new[slot == s]])[-ROUNDT:]
""", """            for s in rows:                                  # ROUNDT's (buffers): the newest ROUNDT of each count
                j = [j for j in range(ROUNDT_SLOTS) if int(self.more_s[j]) == s] + [   # T > 1, in its slot or the
                    j for j in range(ROUNDT_SLOTS) if not int(self.more_n[j])]          # first free one
                r = new[slot == s]
                if j and int(count_of(r[:1, 2, :K])) > 1:
                    r = torch.cat([self.more_rows[j[0], :int(self.more_n[j[0]])], r])[-ROUNDT:]
                    self.more_rows[j[0], :r.shape[0]], self.more_n[j[0]], self.more_s[j[0]] = r, r.shape[0], s
""")
edit("buffers", """        ctl.climb.more = [(T, (r[:, 0], r[:, 1], r[:, 3])) for r in ctl.more.values()      # ROUNDT's rows
                          for T in [int(count_of(r[:1, 2, :K]))] if T > 1]
""", """        ctl.climb.more = [(int(count_of(r[:1, 2, :K])), (r[:, 0], r[:, 1], r[:, 3]))       # ROUNDT's rows (buffers)
                          for r in (self.more_rows[j, :int(self.more_n[j])] for j in range(ROUNDT_SLOTS)) if len(r)]
""")

# ================================================================================================ ROUNDT on the round machine
edit("roundt", """    def round_t(self, p, lib):
        \"\"\"ROUNDT (round 7): the round, run T times at full size, gives the answers of every count T > 1 kept.\"\"\"
        st = round_settings(self.cells, self.P_round)
        for T, (Nd, Xd, Yd) in self.more if ROUNDT else ():
            X, N = pad(Xd, self.P_round), pad(Nd, self.P_round)
            for _ in range(T):
                X = run_prog(st, p, lib, {"X": X, "N": N}, gen_on(self.device, 1), self.device)[0]
            if not bool((X[:, :Yd.shape[1]] == Yd).all()):
                return False
        return True
""", """    def round_t(self, p, lib):
        \"\"\"ROUNDT (round 7): the round, run T times at full size, gives the answers of every count T > 1 kept, the
        first count it misses ending it; on the round machine from run_prog's fresh draws, its work counted as run_prog
        counts it (legal_wider/WIDER.md).\"\"\"
        if not (ROUNDT and self.more):
            return True
        st = round_settings(self.cells, self.P_round)
        ops = answer_trace(st, p, lib)
        n = sum(o < NOP for o in ops)
        m = RoundMachine(st, ops, -(-max(x[0].shape[0] for _, x in self.more) // ROUND_ROWS) * ROUND_ROWS, self.device)
        m.cells(st.cells)
        for T, (Nd, Xd, Yd) in self.more:
            X, go = Xd, torch.ones(Xd.shape[0], dtype=torch.bool, device=self.device)
            for _ in range(T):
                X = m.run(X, Nd, go, go.numel(), gen_on(self.device, 1)).clone()
                for _ in range(n):
                    WORK[0] += PRIM_STEP + PRIM_ROW
            if not bool((X[:, :Yd.shape[1]] == Yd).all()):
                return False
        return True
""")

# ================================================================================================ the reader: polynomials
edit("reader", """READ_TRIES = 3         # ... addition
""", """READ_TRIES = 3         # ... addition; then x's polynomials (LawReader.poly): at most READ_POLY Horner steps, their
READ_POLY = 3          # ... values checked READ_CHUNK at a time
READ_CHUNK = 512
""")
edit("reader", """    constants, depth by depth; the first exact on every row where it is computed.\"\"\"
""", """    constants, depth by depth, then x's polynomials (poly); the first exact on every row where it is computed.\"\"\"
""")
edit("reader", """    def layer(self, prev, leaves, seen):
        \"\"\"The next depth's new distinct values [(name, ops, v, m)], computed operation by operation, rows stacked.\"\"\"
        cands, names = [], set()
        for (cn, co, cv, cm), f, (ln, lv), side in itertools.product(prev, READ_OPS, leaves, (0, 1)):
""", """    def layer(self, prev, steps, seen):
        \"\"\"The next depth's new distinct values [(name, ops, v, m)], computed operation by operation, rows stacked;
        steps: [(operation, (leaf, its value), side)].\"\"\"
        cands, names = [], set()
        for (cn, co, cv, cm), (f, (ln, lv), side) in itertools.product(prev, steps):
""")
edit("reader", """        seen, made = set(self.keys(prev)[0]), []
        for _ in range(READ_DEPTH):
            prev = self.layer(prev, leaves, seen)
""", """        seen, made, steps = set(self.keys(prev)[0]), [], list(itertools.product(READ_OPS, leaves, (0, 1)))
        for _ in range(READ_DEPTH):
            prev = self.layer(prev, steps, seen)
""")
edit("reader", """        return law or (None, (), 0)


def read_pairs(ops_):
""", """        return law or self.poly() or (None, (), 0)

    def poly(self):
        \"\"\"Stage 3 (legal_wider/WIDER.md): x's polynomials with small coefficients by Horner's rule. From x, up to
        READ_POLY steps, each multiplying by x or a constant 2..B or adding x or a constant 1..B (the digits and the
        base); each value reduced mod N, exact or off by one constant either way round (the gap). As read(), or None.\"\"\"
        one = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        cs = [(str(c), self.value(str(c), {}, one)[0]) for c in range(1, self.st.B + 1)]
        steps = [(f, l, 0) for f in ("mul", "add") for l in [("x", self.X)] + cs[f == "mul":]]
        prev = [("x", (), self.X, one)]
        seen = set(self.keys(prev)[0])
        for _ in range(READ_POLY):
            prev = self.layer(prev, steps, seen)
            for i in range(0, len(prev), READ_CHUNK):
                got = self.tail(prev[i:i + READ_CHUNK])
                if got:
                    return got
        return None

    def tail(self, vals):
        \"\"\"The first of vals whose reduction mod N is the answer, or off from it by one constant either way round:
        (its law, operations, rows) or None.\"\"\"
        k, n = len(vals), self.n
        Ns, Ys = self.N.repeat(k, 1), self.Y.repeat(k, 1)
        r, rm = self.op("mod", torch.cat([x[2] for x in vals]), Ns, torch.cat([x[3] for x in vals]))
        for f, (a, b) in (("add", (Ys, r)), ("sub", (r, Ys))):            # Y = r + g, or r = Y + g, mod N
            d, ok = self.op("sub", a, b, rm)
            s, sm = self.op("add", a, Ns, rm & ~ok)
            e, ok2 = self.op("sub", s, b, sm)
            g, gm = torch.where(ok[:, None], d, e).view(k, n, -1), (ok | ok2).view(k, n)
            ref = g[torch.arange(k, device=self.dev), gm.long().argmax(1)]
            hit = torch.nonzero(((g == ref[:, None]).all(-1) | ~gm).all(1) & (gm.sum(1) >= READ_MIN)).flatten()
            if hit.numel():
                q = int(hit[0])
                c = "".join(map(str, ref[q].flip(0).tolist())).lstrip("0")
                law = f"mod({f}({vals[q][0]},{c}),N)" if c else f"mod({vals[q][0]},N)"
                return law, tuple(dict.fromkeys(vals[q][1] + (f,) * bool(c) + ("mod",))), int(gm[q].sum())
        return None


def read_pairs(ops_):
""")
edit("reader", """def read_more(law):
""", """def builds(c, r):
    \"\"\"The constants c is built from: 1 and its doublings below c, made by the learner's addition while its comparison
    says below c (the reader r's routines).\"\"\"
    one = torch.ones(1, dtype=torch.bool, device=r.dev)
    v, cv, out = r.value("1", {}, one)[0], r.value(c, {}, one)[0], []
    while bool(r.run("lt", v, cv, one)[1][0]):
        out.append("".join(map(str, v[0].flip(0).tolist())).lstrip("0"))
        v = r.run("add", v, v, one)[0]
    return out


def read_more(law, r):
""")
edit("reader", """    def pose(name, pat, tr):
        READ_LAWS[name] = tr
        LAW_BOOK.setdefault(name, ("A", (pat,)))
        out.append((name, pat))
    for c, fs in constants(t, {}).items():
        pose(f"const.{c}", "n", c)
""", """    def pose(name, pat, tr):
        READ_LAWS[name] = tr
        LAW_BOOK.setdefault(name, ("A", (pat,)))
        out.extend([(name, pat)] * ((name, pat) not in out))
    for c, fs in constants(t, {}).items():
        for k in builds(c, r) + [c]:                    # (each made from N; those below c first)
            pose(f"const.{k}", "n", k)
""")
edit("reader", """            more = read_more(law)
""", """            more = read_more(law, r)
""")

edit("reader", """    if int(m.sum()) < 2 or (not isinstance(t, str) and bool((v[m] == v[m][:1]).all())):
        return None
    return {"A": v * m[:, None], "Am": m}
""", """    if int(m.sum()) < 2:                            # a size too small for it (as 10 for 1-digit numbers): it says
        return {"A": v * 0, "Am": m & False}        # nothing there (Root.goals: it must say something somewhere)
    if not isinstance(t, str) and bool((v[m] == v[m][:1]).all()):
        return None
    return {"A": v * m[:, None], "Am": m}
""")
edit("reader", """            if all(t is not None for t in ts):
                out.append(Goal(law, ts))
""", """            if all(t is not None for t in ts) and any("check" in t or bool(t.get("Am", t.get("cm")).any()) for t in ts):
                out.append(Goal(law, ts))
""")

# ================================================================================================ room
# Bytes for phase D, with no change in behaviour: in the code after the settings, the padding before a trailing
# comment squeezed to two spaces (a comment continued on the lines below moves with it; no text changes, the AST is
# asserted unchanged); three of phase C's docstrings that FAMILY_FILE.md explains cut to a pointer; a long line split.
edit("room", """            if all(t is not None for t in ts) and any("check" in t or bool(t.get("Am", t.get("cm")).any()) for t in ts):
""", """            if all(t is not None for t in ts) and any("check" in t or bool(t.get("Am", t.get("cm")).any())
                                                      for t in ts):
""")
doc("windows", "A reading's windows (round 7's SUBTERMS; legal_family/FAMILY_FILE.md §2): [(tree, holes)], each shape "
    "once, smallest first.")
doc("read_more", "What a reading poses beyond its cuts and loops, each once, as a law of the book: its constants and "
    "its operations on them, then its windows (legal_family/FAMILY_FILE.md §2; the constants each built from, "
    "legal_wider/WIDER.md).")
doc("read_target", "A reading's own law on a draw: its tree computed by the reader's operations at the draw's size, "
    "the holes a and b the draw's x and o, the modulus its N (legal_family/FAMILY_FILE.md §2).")
doc("above_probes", "Probes with N < X < 2N for the conditional subtraction's check (the fix, ABLATION_S3.md §8): X "
    "is N with its digit at a place j raised by d, from one entry of the learned addition table with no carry, and "
    "the answer X - N is d at place j; ok marks the probes where such a d was drawn (one try each).")
doc("SelfClimb.deep_roots", "The deep probes' roots (deep_by=\"bindings\", SELF_PRACTICE.md §2.3): the closest open "
    "concepts whole, every binding where each is still open, concept by concept until deep_k roots are taken.")
doc("_ustep", "One instruction on every machine, each its own (op [S]), exactly as step() runs it (FAST.md §2): regs "
    "[S, 5 P, NP] uint8 (flat, contiguous), c, h, k [S, NP] uint8, p [S], ps [S, D] the routines' saved pointers (ar, "
    "ad: aranges of S and D), off [S, NP] uint8 the C blocks each probe is off in (None: no C block runs). Returns the "
    "new state.")
doc("link_soft", "D - M on digit distributions D [X, L, 10] minus exact digits M [X, L], the cell applied "
    "multilinearly (learned_remainder/ADDER.md); TDC [20, 120]. Returns the digits [X, L, 10] and each place's borrow "
    "out [X, L, 2].")
doc("LibraryMachine.set_round", "The answer programs into the buffers (combined_sp2/COMBINED.md): their instructions "
    "(answer_trace) in round_ops and round_len, their text with the definitions they use in round_src.")
doc("_run_count", "A count loop on batch b in place (LEARNED_LOOP.md §2): each probe runs the body, as a routine, as "
    "many times as its count register held when the loop began.")
edit("room", """FULL_CHECK = (("red", "x+on"), ("red", "x+xn"))  # fix: laws on patterns whose acceptance also runs at the round's size
                                 # (W digits; both reductions whose quotient is another row's x or x itself):
                                 # their quotients reach the round's, so no loop of subtractions in reach passes
ABOVE_CHECK = (("red", "pair"),)  # fix: laws on patterns also checked with N < a < 2N (above_probes): red's
                                 # own probes on pair are a < N and a = N only
""", """FULL_CHECK = (("red", "x+on"), ("red", "x+xn"))  # fix: laws on patterns whose acceptance also runs at the round's size
ABOVE_CHECK = (("red", "pair"),)  # fix: laws also checked with N < a < 2N (above_probes): both fixes, ABLATION_S3.md §8
""")
EDITS.append(("squeeze", "room", None, None))

# ================================================================================================ the build
def undefined_globals(src: str) -> set:
    """Names a function reads as globals that neither the module's top level nor builtins define (symtable)."""
    top = symtable.symtable(src, "wider", "exec")
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


def squeeze(src: str) -> str:
    """In the code after the settings (from the first def or class), the padding before each trailing comment cut to
    two spaces; comment-only lines just below it at its old column move to its new one. The AST is unchanged."""
    import io
    import tokenize
    lines = src.splitlines(True)
    first = next(i for i, l in enumerate(lines) if l.startswith(("def ", "class ")))
    cols = {}                                               # line -> its trailing comment's column
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.COMMENT and t.start[0] - 1 >= first and lines[t.start[0] - 1][:t.start[1]].strip():
            cols[t.start[0] - 1] = t.start[1]
    for i, col in cols.items():
        code = lines[i][:col].rstrip()
        new = len(code) + 2
        if new >= col:
            continue
        lines[i] = code + "  " + lines[i][col:]
        j = i + 1
        while j < len(lines) and lines[j][:col].strip() == "" and lines[j][col:col + 1] == "#" and j not in cols:
            lines[j] = " " * new + lines[j][col:]
            j += 1
    out = "".join(lines)
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(src))
    return out


def build(sets: dict | None = None, dev: bool = False) -> tuple[str, list]:
    """The wider file's text and the log of the edit groups applied; dev: over the size limit is allowed (a
    working build, never the submission)."""
    src = base()
    log = []
    for kind, group, old, new in EDITS:
        if kind == "squeeze":
            src = squeeze(src)
            log.append(group)
            continue
        if kind == "sub":
            src, n = re.subn(old, new, src, flags=re.M)
            assert n == int(group.split()[-1]), (group, n)
            log.append(group)
            continue
        if kind == "doc":
            old, ind = docstring_at(src, old)
            new = "\n".join(textwrap.wrap(new + '"""', 120, initial_indent=ind + '"""', subsequent_indent=ind,
                                          break_long_words=False, break_on_hyphens=False)) + "\n"
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
    global EDITS
    skip = [a.split("=", 1)[1] for a in argv if a.startswith("--skip=")]   # research: an edit group left out
    EDITS = [e for e in EDITS if e[1] not in skip]
    sets = dict(a.split("=", 1) for a in argv if "=" in a and not a.startswith("--"))
    src, log = build(sets, dev="--dev" in argv)
    if "--diff" in argv:
        sys.stdout.writelines(difflib.unified_diff(base().splitlines(True), src.splitlines(True), "family", "wider",
                                                   n=1))
        return
    for g in dict.fromkeys(log):
        print(f"{log.count(g):4d}  {g}")
    out = OUT if not (sets or skip) else HERE / "research" / (OUT.stem + "_" + "_".join(
        [f"{k}-{v}" for k, v in sets.items()] + [f"no-{g}" for g in skip]).replace("/", "") + ".py")
    if "--dev" in argv:
        out = out.with_name(out.stem + "_dev.py")
    out.parent.mkdir(exist_ok=True)
    out.write_text(src)
    print(f"wrote {out}: {len(src.splitlines())} lines, {len(src.encode()):,} bytes "
          f"({LIMIT - len(src.encode()):,} under the limit), md5 {md5(src)}")


if __name__ == "__main__":
    main(sys.argv[1:])
