"""Builds the wide file (integration/legal_wide/WIDE.md): the legal file (combined_legal's submission_stage3_legal.py,
its committed bytes at 8e3c7fe, md5 asserted) with its reader's grammar widened: subtract, power, constants read from
the data and depth 3, beside add, mul and mod, every value still computed by the learner's own routines; each depth's
candidates computed together. Named groups of exact edits, each asserted to match exactly once (a cut between two
anchors pinned by the md5 of the text it removes). Writes submission_stage3_wide.py; NAME=VALUE sets a constant
(research builds, written to research/).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_wide/make_wide.py [--diff] [--dev] [NAME=VALUE ...]

(--diff: the edits' effect as a unified diff against the legal file; --dev: a working build that may be over the size
limit, written as *_dev.py; --md5s: each cut's md5 printed instead of asserted, for writing a new cut.)
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
OUT = HERE / "submission_stage3_wide.py"
LIMIT = 256 * 1024                          # the evaluator's limit on a submission's size
BASE = ("8e3c7fe", "integration/combined_legal/submission_stage3_legal.py", "e2fe7a681bc981ce844fc8c826132d47")


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
edit("doc", '''new here, a law book of three laws on the pair (BOOK: the comparison, the link and the addition) in place of
the generated one, whose grammar stays as the forms a law can take; the rest is read from the target: its law, read
from the answer rows by the learner's own routines (Reader), poses its operations' cuts and loops (read_pairs).
"""''', '''new here, a law book of three laws on the pair (BOOK: the comparison, the link and the addition) in place of
the generated one, whose grammar stays as the forms a law can take; the rest is read from the target (legal_wide/
WIDE.md): its law, read from the answer rows by the learner's own routines over a wide grammar (LawReader), poses its
operations' cuts and loops (read_pairs).
"""''')

# ================================================================================================ grammar
edit("grammar", '''READ_MIN = 4           # the target's law, read (Reader): exact on every answer row it can be computed on, at least
READ_LINKS = 1000      # ... this many; a reduction there takes at most this many links; the reading is tried at
READ_TRIES = 3         # ... this many levels at most, from the first with the comparison, the link and the addition
''', '''READ_MIN = 4           # the target's law, read (LawReader): exact on every answer row it can be computed on, at
READ_POW = 8           # ... least this many; a power's count at most this; laws of at most READ_DEPTH operations;
READ_DEPTH = 3         # ... tried at READ_TRIES levels at most, from the first with the comparison, the link and the
READ_TRIES = 3         # ... addition
''')

# ================================================================================================ reader
# The reader, widened (WIDE.md §2): the operations add, sub, mul, pow and mod; the leaves x, N and the data's constants
# (gaps read off the answers); depth 1 to READ_DEPTH; each depth's candidates computed together, operation by
# operation, their rows stacked, and only the first chain of each distinct value kept.
cut("reader", '''READ_OPS = {"add": "add", "mul": "mul", "mod": "red"}''', '''

class SelfClimb:
''', "6ac21d225c7d13943643d4670420ceb9", '''READ_OPS = {"add": "add", "sub": "sub", "mul": "mul", "pow": "mul", "mod": "red"}  # as the book's laws


class LawReader:
    """The target's law, read from answer rows of one T (N, x, y) at full size by the learner's routines and the
    book's constructions only (legal_wide/WIDE.md §2): left-deep chains of add, sub, mul, pow and mod over x, N and
    the data's constants (a gap by which a reduction mod N misses every answer), depth by depth; the first exact on
    every row where it is computed (READ_MIN at least). mul and mod repeat the addition and the link by doubling."""

    def __init__(self, climb, answers):
        self.c, self.dev = climb, answers[0].device
        self.st = round_settings(climb.cells, climb.P_round)
        self.N, self.X, self.Y = (pad(v, self.st.P) for v in answers)
        self.n, self.gen, self.tried = self.N.shape[0], gen_on(self.dev, 3), 0

    def run(self, law, a, b, m):
        A, c = torch.zeros_like(a), torch.zeros_like(m, dtype=torch.long)
        i = torch.nonzero(m).flatten()
        if i.numel():
            A[i], c[i] = self.c.ops.run(law, "pair", self.st, self.c.lib, [a[i], b[i]], self.gen, self.dev)
        return A, c

    def op(self, f, a, b, m):
        """f(a, b) on stacked rows, and the rows m where it is computed."""
        fits = lambda v: (v[:, self.st.P // 2:] == 0).all(1)          # noqa: E731
        if f == "add":
            m = m & (a[:, -1] == 0) & (b[:, -1] == 0)
            return self.run("add", a, b, m)[0], m
        if f == "sub":
            d, c = self.run("lt", a, b, m)
            m = m & (c == 0)
            return d, m & (self.run("add", b, d, m)[0] == a).all(1)
        if f == "mul":
            m = m & fits(a) & fits(b)
            v, i = torch.zeros_like(a), torch.nonzero(m).flatten()
            if i.numel():
                v[i] = repeat_add(self.c.ops, self.st, self.c.lib, v[i], a[i], count_of(b[i]), self.gen, self.dev)
            return v, m
        if f == "pow":
            k = torch.where(fits(b), count_of(b), 0)
            m, v = m & (k >= 1) & (k <= READ_POW), a.clone()
            for j in range(2, READ_POW + 1):
                more = m & (k >= j)
                if not bool(more.any()):
                    break
                w, ok = self.op("mul", v, a, more)
                v, m = torch.where(more[:, None], w, v), m & (ok | ~more)
            return v, m
        m = m & (b != 0).any(1) & (a[:, -1] == 0)  # mod: the link by doubling (as repeat_add adds): b, 2b, 4b, ...
        ds, d, ok = [], b, m                   # by the addition while below a, then the link with each, largest first
        while bool(ok.any()):
            ds.append((d, ok))
            ok = ok & (self.run("lt", d, a, ok)[1] == 1)
            d = torch.where(ok[:, None], self.run("add", d, d, ok)[0], d)
        v = a
        for d, ok in reversed(ds):
            v = torch.where(ok[:, None], self.run("red", v, d, ok)[0], v)
        return v, m

    def layer(self, prev, leaves, seen):
        """The next depth's new distinct values [(name, ops, v, m)], computed operation by operation, rows stacked."""
        cands, names = [], set()
        for (cn, co, cv, cm), f, (ln, lv), side in itertools.product(prev, READ_OPS, leaves, (0, 1)):
            name = f"{f}({cn},{ln})" if side == 0 else f"{f}({ln},{cn})"
            if name not in names:
                names.add(name)
                cands.append((name, tuple(dict.fromkeys(co + (f,))), *((cv, lv) if side == 0 else (lv, cv)), cm, f))
        out, n = [None] * len(cands), self.n
        for f in READ_OPS:
            idx = [i for i, x in enumerate(cands) if x[5] == f]
            if idx:
                v, m = self.op(f, *(torch.cat([cands[i][j] for i in idx]) for j in (2, 3, 4)))
                for k, i in enumerate(idx):
                    s = slice(k * n, (k + 1) * n)
                    out[i] = (cands[i][0], cands[i][1], v[s] * m[s, None], m[s])
        self.tried += len(cands)
        keys, _, rows = self.keys(out)
        new = []
        for k, r, x in zip(keys, rows, out):
            if r >= READ_MIN and k not in seen:
                seen.add(k)
                new.append(x)
        return new

    def keys(self, vals):
        """Per value: a hash of it and its rows, whether it is exact, its rows."""
        V, M = torch.stack([x[2] for x in vals]), torch.stack([x[3] for x in vals])
        ok = ((V == self.Y).all(-1) | ~M).all(-1)
        return (hash_rows(torch.cat([V.flatten(1), M.long()], 1).to(torch.uint8)).tolist(), ok.tolist(),
                M.sum(1).tolist())

    def chains(self, leaves):
        """(the first exact law, its operations, rows) or None, and every value made."""
        one = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        prev = [(n_, (), v, one) for n_, v in leaves]
        seen, made = set(self.keys(prev)[0]), []
        for _ in range(READ_DEPTH):
            prev = self.layer(prev, leaves, seen)
            made += prev
            if prev:
                for x, ok, r in zip(prev, *self.keys(prev)[1:]):
                    if ok:
                        return (x[0], x[1], r), made
        return None, made

    def read(self):
        """(the law read, its operations, its rows), or (None, (), 0)."""
        leaves = [("x", self.X), ("N", self.N)]
        law, made = self.chains(leaves)
        if law is None:
            red = [x for x in made if x[0].startswith("mod(") and x[0].endswith(",N)") and x[0].count("(") <= 2]
            consts = {}
            for r in red:                                     # the gap either way round, mod N
                for a, b in ((self.Y, r[2]), (r[2], self.Y)):
                    d, ok = self.op("sub", a, b, r[3])
                    s, sm = self.op("add", a, self.N, r[3] & ~ok)
                    e, ok2 = self.op("sub", s, b, sm)
                    g, gm = torch.where(ok[:, None], d, e), ok | ok2
                    if int(gm.sum()) >= READ_MIN and (g[gm] != 0).any() and bool((g[gm] == g[gm][:1]).all()):
                        c = g[gm][0]
                        consts.setdefault("".join(map(str, c.flip(0).tolist())).lstrip("0"), c)
            if consts:
                law, _ = self.chains(leaves + [(k, v.expand(self.n, -1)) for k, v in consts.items()])
        return law or (None, (), 0)


def read_pairs(ops_):
    """What a reading poses: each operation's laws (f, add.f) where the count is a digit (its cut) or a number (loop)."""
    q = dict(NFORM, **{p: p[-1] for p in PATTERNS if "," in p})
    return list(dict.fromkeys((l, p) for f in ops_ for l in (READ_OPS[f], "add." + READ_OPS[f]) if l in LAW_BOOK
                              for p in LAW_BOOK[l][1] if q.get(p) in ("d", "o")))
''')


# ================================================================================================ room
# The wider reader is paid for by code this file never runs with its settings (counts always by doubling, one
# acceptance root shared, sp2's settings' other modes), by the miner's report-only parts (its facts about the answers,
# and divisibility, never a lesson's), and by the settings' history notes (COMBINED.md and FAST.md keep them).
edit("room", """SP_RULE = "lpca"       # the choosing rule: "lp" (hardest learnable first), "ct2a2" (the cheaper pairs signal), "all",
                       # "lpca" (concepts, hardest by their cheapest binding, each practised once in every binding tied
                       # for cleanest; SELF_PRACTICE.md §3)
""", """SP_RULE = "lpca"       # the choosing rule: concepts, hardest by their cheapest binding, each practised once in every
                       # binding tied for cleanest (SELF_PRACTICE.md §3)
""")
edit("room", """SP_DEEP_K = 16         # ... for this many roots a level: those whose closest open law is closest (the best fit;
                       # with SP_DEEP_BY "bindings", whole concepts until this many roots: two concepts of 12
                       # bindings. sp2: 8, one concept)
SP_DEEP_ROTATE = "gain"        # a concept that had a deep probe goes after the others unless its closeness grew
                               # since (False, sp2: by closeness alone)
SP_DEEP_BY = "bindings"        # ... taken concept by concept, every binding where it is still open ("root": the
                               # closest roots alone, as submission_stage3_sp.py, md5 f0ef5b06)
SP_DEEP_FOCUS = "closure"      # their vocabulary: the primitives, the routines the concept's closest attempts used, and
                               # the routines those call (True: that root's attempt's routines alone, as f0ef5b06)
SP_DEEP_UNION = False          # ... the closest attempts at the root itself (True, sp2: at every binding of its concept)
SP_PURE_FIRST = "scratch"  # a law whose programs all write a register besides A is searched one cost deeper for one
                       # that does not (False, sp2: the first cost's programs)
SP_ALIAS_FIT = True    # a law whose probe found only aliases (programs acceptance rejects) keeps its closeness, from
                       # its states that are not goal states (False, sp2: it counts as reached, closeness 0)
ROUND_STATES = 2_000_000       # the round's search: states stored per level (sp2: 4M; 2M stops the flat search at cost
                               # 4, the search by calls having come first) ...
""", """SP_DEEP_K = 16         # ... for this many roots a level, whole concepts (the closest first) until this many roots
SP_DEEP_ROTATE = "gain"        # a concept that had a deep probe goes after the others unless its closeness grew since
SP_DEEP_BY = "bindings"        # ... taken concept by concept, every binding where it is still open
SP_DEEP_FOCUS = "closure"      # their vocabulary: the primitives, the routines the concept's closest attempts used, and
                               # the routines those call
SP_DEEP_UNION = False          # ... the closest attempts at the root itself
SP_PURE_FIRST = "scratch"  # a law whose programs all write a register besides A is searched one cost deeper for one
                       # that does not
SP_ALIAS_FIT = True    # a law whose probe found only aliases (programs acceptance rejects) keeps its closeness, from
                       # its states that are not goal states
ROUND_STATES = 2_000_000       # the round's search: states stored per level ...
""")
edit("room", """SP_CALLS_FIRST = True  # ... and that search by calls runs first, the flat one only if it finds nothing (False, sp2:
                       # flat first)
""", """SP_CALLS_FIRST = True  # ... and that search by calls runs first, the flat one only if it finds nothing
""")
edit("room", """SHARED_ACCEPT = True   # a probed root's laws share one acceptance Root a call (COMBINED.md §8)
DOUBLE_COUNTS = True   # the book's counts by doubling, times_add (COMBINED.md §8)
""", "")
edit("room", """    \"\"\"D + times v by the learner's own addition (times a count per row; the additions repeated, mul.ladder).\"\"\"
    if DOUBLE_COUNTS:
        return times_add(ops, st, lib, D, v, times, gen, device)
    D = D.clone()
    for k in range(int(times.max()) if times.numel() else 0):
        more = torch.nonzero(times > k).flatten()
        if more.numel():
            D[more] = ops.run("add", "pair", st, lib, [D[more], v[more]], gen, device)[0]
    return D
""", """    \"\"\"D + times v by the learner's own addition (times a count per row: mul.ladder), by doubling (times_add).\"\"\"
    return times_add(ops, st, lib, D, v, times, gen, device)
""")
edit("room", """        if box is None:
            r = Root(root.pattern, root.binding, verify, n, rows, ops, lib, device, seed)
        else:                                  # SHARED_ACCEPT
            if "root" not in box:
                box["root"] = Root(root.pattern, root.binding, box["sets"], n, rows, ops, lib, device, seed)
            r = box["root"]
""", """        box = {"sets": verify} if box is None else box     # one acceptance root a call, shared by the root's laws
        if "root" not in box:
            box["root"] = Root(root.pattern, root.binding, box["sets"], n, rows, ops, lib, device, seed)
        r = box["root"]
""")
edit("room", """        box = {"sets": top} if SHARED_ACCEPT else None
""", """        box = {"sets": top}
""")
cut("room", """    def mine_answers(self, aux: dict, yd: Tensor) -> None:
""", """    # ------------------------------------------------------------------ the forward pass
""", "3f2ef40d9a02e1a921108b9d3e3ecacc")
edit("room", """        model.mine_answers(aux, yd)
""", "")
edit("room", """    row: order, equality, constant across rows, nonzero, range (a < 10^k) and divisibility by 2, 3, 5 (from the digit
    labels: the lowest digit for 2 and 5, the digit sum for 3; the miner's bookkeeping, never a lesson's target).\"\"\"
""", """    row: order, equality, constant across rows, nonzero and range (a < 10^k).\"\"\"
""")
edit("room", """            out[f"{a} < 10^{k}"] = (A[:, k:] == 0).all(1)
        for d, v in ((2, A[:, 0] % 2), (5, A[:, 0] % 5), (3, A.sum(1) % 3)):
            out[f"{a} mod {d} != 0"] = v != 0
            for r in range(d):
                out[f"{a} mod {d} = {r}"] = v == r
""", """            out[f"{a} < 10^{k}"] = (A[:, k:] == 0).all(1)
""")
edit("room", """        \"\"\"The relations in force, without those a stronger one implies (a < 10^k when a < 10^(k-1) holds; a mod d != 0
        when a mod d = r holds).\"\"\"
""", """        \"\"\"The relations in force, without those a stronger one implies (a < 10^k when a < 10^(k-1) holds).\"\"\"
""")
edit("room", """            if k.endswith(" != 0") and " mod " in k:
                a, d = k.split(" mod ")[0], k.split(" mod ")[1].split()[0]
                if any(f"{a} mod {d} = {r}" in f for r in range(1, 5)):
                    continue
""", "")


# ================================================================================================ the build
def undefined_globals(src: str) -> set:
    """Names a function reads as globals that neither the module's top level nor builtins define (symtable)."""
    top = symtable.symtable(src, "wide", "exec")
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
    """The wide file's text and the log of the edit groups applied; dev: over the size limit is allowed (a working
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
        sys.stdout.writelines(difflib.unified_diff(base().splitlines(True), src.splitlines(True), "legal", "wide", n=1))
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
