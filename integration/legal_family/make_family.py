"""Builds the family file (integration/legal_family/FAMILY_FILE.md): the no-book file (legal_nobook's
submission_stage3_nobook.py, its committed bytes at f7dc5f8, md5 asserted) with round 7's four general pieces
(bar3/scale/family/FAMILY.md) ported legally: the reading's windows as laws of their own (SUBTERMS), its constants
made from N with its operations on them (round 6's CONST, which here also carries CONST1's 1), and a round kept only if
it also gives the kept answers of every count T > 1 (ROUNDT); every answer from the learner's routines through the
reader. Named groups of exact edits, each asserted to match exactly once. Writes submission_stage3_family.py;
NAME=VALUE sets a constant (research builds, written to research/).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_family/make_family.py [--diff] [--dev] [NAME=VALUE ...]

(--diff: the edits' effect as a unified diff against the no-book file; --dev: a working build that may be over the
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
OUT = HERE / "submission_stage3_family.py"
LIMIT = 256 * 1024                          # the evaluator's limit on a submission's size
BASE = ("f7dc5f8", "integration/legal_nobook/submission_stage3_nobook.py", "b42ec30422c182c0972aebca5be0c470")


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
edit("doc", '''(FAST.md); new here, no law in the book (legal_nobook/NOBOOK.md): every law practised is read from the target. First
the cheapest reading (read_fact: every answer below N, so the law ends in a reduction by N, whose step is posed cut to
its parts); then its law, read from the answer rows by the learner's own routines over a wide grammar (LawReader),
poses its operations' cuts and loops (read_pairs). The grammar stays as the forms a law can take.
"""''', '''(FAST.md); new here, no law in the book (legal_nobook/NOBOOK.md): every law practised is read from the target. First
the cheapest reading (read_fact: every answer below N, so the law ends in a reduction by N, whose step is posed cut to
its parts); then its law, read from the answer rows by the learner's own routines over a wide grammar (LawReader),
poses its operations' cuts and loops (read_pairs), its constants and its windows (read_more, legal_family/
FAMILY_FILE.md); a round must give the answers of every count kept (ROUNDT).
"""''')

# ================================================================================================ port
edit("port", '''STEP_ADD = True        # ... and (x + d F is a sum) the addition, posed with the step (False: once the reader needs it)
''', '''STEP_ADD = True        # ... and (x + d F is a sum) the addition, posed with the step (False: once the reader needs it)
READ_WINDOW = (3, 2)   # a reading's windows (read_more): at most this many operations and holes
CONST_OPS = ("add", "sub", "mul")  # its operations posed with each of its constants: x + c, x - c, x c
ROUNDT = 4096          # a round must also give the answers of every count T > 1, run T times: the newest this many
                       # rows of each count the batches showed (keep_counts; 0: off)
''')
edit("port", '''GROW = {"same": 0, "carry": 1, "double": 2}
''', '''GROW = {"same": 0, "carry": 1, "double": 2}
PATTERNS.update({"n": ((("n",),), "same", ""), "x,n": ((("x", "n"),), "carry", ""),   # a reading's own (read_more,
                 "w1": ((("x", "n"),), "double", ""), "w2": ((("x", "o", "n"),), "double", "")})   # not the book's)
NBOUND = ("n", "x,n", "w1", "w2")   # (their last term, the modulus, bound to N)
''')
edit("port", '''    if d.above:                                                 # fix: red(N + d B^j) = d B^j, where one was drawn
''', '''    if law in READ_LAWS:                                        # a reading's own law (read_target)
        return read_target(law, d, ops, lib)
    if d.above:                                                 # fix: red(N + d B^j) = d B^j, where one was drawn
''')
edit("port", '''    def __init__(self, climb, answers):
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
''', '''    def __init__(self, ops, lib, st, dev, answers=None):
        self.ops, self.lib, self.st, self.dev = ops, lib, st, dev
        self.gen, self.tried = gen_on(dev, 3), 0
        if answers is not None:
            self.N, self.X, self.Y = (pad(v, st.P) for v in answers)
            self.n = self.N.shape[0]

    def run(self, law, a, b, m):
        A, c = torch.zeros_like(a), torch.zeros_like(m, dtype=torch.long)
        i = torch.nonzero(m).flatten()
        if i.numel():
            A[i], c[i] = self.ops.run(law, "pair", self.st, self.lib, [a[i], b[i]], self.gen, self.dev)
        return A, c

    def value(self, t, env, m):
        """A tree's value on rows (holes and N from env, a constant from its digits) and the rows it is computed on."""
        if isinstance(t, str):
            if t in env:
                return env[t], m
            P = self.st.P
            c = torch.tensor([int(ch) for ch in reversed(t)][:P] + [0] * (P - len(t)), device=self.dev)
            return c.expand(m.shape[0], -1).clone(), m & (len(t) <= P)
        (a, ma), (b, mb) = (self.value(u, env, m) for u in t[1])
        return self.op(t[0], a, b, ma & mb)
''')
edit("port", '''                v[i] = repeat_add(self.c.ops, self.st, self.c.lib, v[i], a[i], count_of(b[i]), self.gen, self.dev)
''', '''                v[i] = repeat_add(self.ops, self.st, self.lib, v[i], a[i], count_of(b[i]), self.gen, self.dev)
''')
edit("port", '''

class SelfClimb:
''', '''

READ_LAWS = {}         # a reading's own laws (read_more) -> their trees: holes a and b, N, constants


def tree(s):
    """A law read (its text) as (operation, [operands]); leaves as text."""
    i = s.find("(")
    if i < 0:
        return s
    out, depth, j0 = [], 0, i + 1
    for j in range(i + 1, len(s) - 1):
        depth += (s[j] == "(") - (s[j] == ")")
        if s[j] == "," and not depth:
            out.append(tree(s[j0:j]))
            j0 = j + 1
    return s[:i], out + [tree(s[j0:-1])]


def law_text(t):
    return t if isinstance(t, str) else f"{t[0]}({','.join(law_text(a) for a in t[1])})"


def windows(t):
    """A reading's windows (round 7's SUBTERMS): every connected run of 2 to READ_WINDOW[0] of its operations, each
    child off the run cut off as a hole, named a, b in order (at most READ_WINDOW[1]), N and the constants kept;
    [(tree, holes)], each shape once, smallest first."""
    def cut(n, k):
        if isinstance(n, str):
            return [(n if n == "N" or n.isdigit() else "?", 0)]
        out = [("?", 0)]
        for combo in itertools.product(*[cut(a, k - 1) for a in n[1]]) if k else ():
            o = 1 + sum(c for _, c in combo)
            if o <= k:
                out.append(((n[0], [c for c, _ in combo]), o))
        return out

    def name(n, hs):
        if isinstance(n, str):
            if n == "?":
                hs.append("ab"[len(hs)] if len(hs) < 2 else "h")
                return hs[-1]
            return n
        return n[0], [name(a, hs) for a in n[1]]

    found, seen = [], set()

    def walk(n):
        if not isinstance(n, str):
            for w, o in cut(n, READ_WINDOW[0]):
                hs = []
                w = name(w, hs)
                if o >= 2 and 0 < len(hs) <= READ_WINDOW[1] and law_text(w) not in seen:
                    seen.add(law_text(w))
                    found.append((o, w, len(hs)))
            for a in n[1]:
                walk(a)
    walk(t)
    return [(w, h) for _, w, h in sorted(found, key=lambda f: f[0])]


def constants(t, out):
    """{c: {operations}}: a tree's constant leaves and the operations they are operands of."""
    if not isinstance(t, str):
        for a in t[1]:
            if isinstance(a, str) and a.isdigit():
                out.setdefault(a, set()).add(t[0])
            constants(a, out)
    return out


def read_more(law):
    """What a reading poses beyond its cuts and loops, each once, as a law of the book: each constant c, made from N,
    and the reading's operations on it (round 6's constants: const.c on n, f.c on x,n); then its windows, smallest
    first (round 7's SUBTERMS), on w1 or w2 by their holes, named by their operations and constants."""
    t, out = tree(law), []

    def pose(name, pat, tr):
        READ_LAWS[name] = tr
        LAW_BOOK.setdefault(name, ("A", (pat,)))
        out.append((name, pat))
    for c, fs in constants(t, {}).items():
        pose(f"const.{c}", "n", c)
        for f in sorted(fs & set(CONST_OPS)):
            pose(f"{f}.{c}", "x,n", (f, ["a", c]))
    ops_of = lambda n: [] if isinstance(n, str) else [n[0] + "".join(a for a in n[1] if str(a).isdigit())] + [
        o for a in n[1] for o in ops_of(a)]                                                  # noqa: E731
    for w, h in windows(t):
        name = "w." + ".".join(ops_of(w))
        while name in READ_LAWS and READ_LAWS[name] != w:
            name += "2"
        pose(name, f"w{h}", w)
    return out


def read_target(law, d, ops, lib):
    """A reading's own law (read_more) on a draw: its tree computed by the reader's operations at the draw's size (the
    learner's routines), the holes a and b the draw's x and o, the modulus its N; rows where a value on the way is not
    computed are masked. A constant's answers do not vary."""
    t, m = READ_LAWS[law], torch.ones(d.n, dtype=torch.bool, device=d.device)
    v, m = LawReader(ops, lib, d.st, d.device).value(t, {"a": d.x, "b": d.o, "N": d.N}, m)
    if int(m.sum()) < 2 or (not isinstance(t, str) and bool((v[m] == v[m][:1]).all())):
        return None
    return {"A": v * m[:, None], "Am": m}


class SelfClimb:
''')
edit("port", '''        self._fact = None                       # the cheapest reading: {field, parts} once made
''', '''        self._fact = None                       # the cheapest reading: {field, parts} once made
        self.more = []                          # ROUNDT: [(T, (N, x, y))], the kept answer rows of each count T > 1
''')
edit("port", '''        t0, r = _clock(), LawReader(self, answers)
        law, ops_, n = r.read()
        if law is not None:
            self._read = dict(law=law, pairs=read_pairs(ops_))
        self.records.append(dict(task="read", level=self.level, law=law, ops=list(ops_), rows=n, tried=r.tried,
                                 answers=int(answers[0].shape[0]), seconds=round(_clock() - t0, 1)))
''', '''        t0, more = _clock(), []
        r = LawReader(self.ops, self.lib, round_settings(self.cells, self.P_round), answers[0].device, answers)
        law, ops_, n = r.read()
        if law is not None:
            more = read_more(law)
            self._read = dict(law=law, pairs=read_pairs(ops_) + more)
        self.records.append(dict(task="read", level=self.level, law=law, ops=list(ops_), rows=n, tried=r.tried,
                                 answers=int(answers[0].shape[0]), seconds=round(_clock() - t0, 1),
                                 more=[f"{a}/{p}" for a, p in more]))
''')
edit("port", '''            for bd in bindings(arity):
                key = f"{pat}@{binding_name(bd)}"
''', '''            for bd in bindings(arity):
                if pat in NBOUND and bd[-1] != "N":     # a reading's own pattern: its modulus in N
                    continue
                key = f"{pat}@{binding_name(bd)}"
''')
edit("port", '''                f"; read {r['law']} on {r['rows']} of {r['answers']} answer rows, {r['tried']} laws tried, "
                f"{r['seconds']}s" if r["task"] == "read" else f"; fact: answers below {r['field']} on "
''', '''                f"; read {r['law']} on {r['rows']} of {r['answers']} answer rows, {r['tried']} laws tried, "
                f"{r['seconds']}s, poses {r['more']}" if r["task"] == "read" else f"; fact: answers below {r['field']} "
                f"on "
''')
edit("port", '''            ok = lambda ps: [round_ok(p, lib, self.cells, answers, self.P_round, self.device) for p in ps]
''', '''            ok = lambda ps: [round_ok(p, lib, self.cells, answers, self.P_round, self.device) and self.round_t(p, lib)
                             for p in ps]
''')
edit("port", '''    def macro_classes(self, lib):
''', '''    def round_t(self, p, lib):
        """ROUNDT (round 7): the round, run T times at full size, gives the answers of every count T > 1 kept."""
        st = round_settings(self.cells, self.P_round)
        for T, (Nd, Xd, Yd) in self.more if ROUNDT else ():
            X, N = pad(Xd, self.P_round), pad(Nd, self.P_round)
            for _ in range(T):
                X = run_prog(st, p, lib, {"X": X, "N": N}, gen_on(self.device, 1), self.device)[0]
            if not bool((X[:, :Yd.shape[1]] == Yd).all()):
                return False
        return True

    def macro_classes(self, lib):
''')
edit("port", '''        t0, w0 = _clock(), WORK[0]
        dl = ctl.climb_work0 + CLIMB_WORK if CLIMB_WORK else None            # the climb's work budget
''', '''        ctl.climb.more = [(T, (r[:, 0], r[:, 1], r[:, 3])) for r in ctl.more.values()      # ROUNDT's rows
                          for T in [int(count_of(r[:1, 2, :K]))] if T > 1]
        t0, w0 = _clock(), WORK[0]
        dl = ctl.climb_work0 + CLIMB_WORK if CLIMB_WORK else None            # the climb's work budget
''')

edit("port", '''        self.round_fit: list = []             # each answer program's share of this step's answer rows
''', '''        self.round_fit: list = []             # each answer program's share of this step's answer rows
        self.more: dict = {}                  # ROUNDT: the newest answer rows of each count (keep_counts)
''')
edit("port", '''            rows = {s: torch.cat([self.count_rows[s, :int(self.count_n[s])], new[slot == s]])
                    for s in torch.unique(slot).tolist()}
''', '''            rows = {s: torch.cat([self.count_rows[s, :int(self.count_n[s])], new[slot == s]])
                    for s in torch.unique(slot).tolist()}
            for s in rows:                                  # ROUNDT's: the newest ROUNDT of each count
                ctl.more[s] = torch.cat([ctl.more.get(s, new[:0]), new[slot == s]])[-ROUNDT:]
''')


# ================================================================================================ room
# Bytes for the port, with no change in behaviour: the phase-B arm's switch (STEP_ADD, True here), docstrings and a
# comment a design doc already explains cut to what they say plus a pointer to that doc, and shorter rules between a
# class's methods (combined_sp2's book check reads the section rules, which stay).
edit("room", """STEP_PARTS = ("lt", "red", "add")  # a reduction's step x + d F cut to its parts, on the pair: the comparison, the link
STEP_ADD = True        # ... and (x + d F is a sum) the addition, posed with the step (False: once the reader needs it)
""", """STEP_PARTS = ("lt", "red", "add")  # a reduction's step x + d F cut to its parts, on the pair: the comparison, the link
                                   # and (x + d F is a sum) the addition
""")
edit("room", """parts=[p for p in STEP_PARTS if STEP_ADD or p != "add"] if F == "N" else [])""",
     """parts=list(STEP_PARTS) if F == "N" else [])""")
edit("room", """        parts = list((self._fact or {}).get("parts", ()))
        if not STEP_ADD and self.ops.has("lt", "pair") and self.ops.has("red", "pair"):
            parts.append("add")                 # the reader's construction asks for the addition
""", """        parts = list((self._fact or {}).get("parts", ()))
""")
doc("SelfClimb", "The self-practising climb as a state machine a training loop drives (step: one unit of work), "
    "libcore.Climb's interface: each level poses, probes, chooses, practises and searches the round (SELF_PRACTICE.md "
    "§2.3-2.4).")
doc("bfs_roots", "bfs_multi for many roots in one search (SELF_PRACTICE.md §2.3): each root its own salt, budget and "
    "stop, a level's work one apply_items. Returns ([per goal: dict(found_cost, accepted, aliases)], [per root: level "
    "sizes], [per root: stop], held levels, their root ids, parents); fits, a list, gets the best fit of goals[0] (on "
    "its root) at every level.")
doc("answer_rows", "The rows whose answers the lessons read (the selector, the round, the reading's answers, the "
    "freeze's check): those of the batch's smallest nonzero T as read (digits Cd LSB first, the reader's comparison), "
    "at most ANS_WORK // (T + 1) of them; nothing about the data's T is assumed (general_lessons/LESSONS.md). tcount "
    "is for general_lessons/local_check.py's call.")
doc("LibraryMachine.table_lessons", "The subtraction table's lessons (general_lessons/LESSONS.md, kinds 1 and 2): "
    "sub.order on every mined order fact, then the digit laws; no late lessons (ABLATION_S3.md's no_late). Returns "
    "the loss and the hard checks' share passing (what gates the next phase).")
doc("bfs", "Breadth-first by cost, states merged over every probe of every setting, every minimal-cost parent kept; "
    "stops at the first cost whose goal programs (the first `limit` spellings) pass `accept`, when a level would pass "
    "max_states, or past stop_at. chunk and fuse batch a level's work (FAST.md §2); the result is the same for any "
    "fuse.")
doc("order_pairs", "The pairs of table entries the two order laws compare (columns and rows), each oriented (lower, "
    "upper) by the learned borrows; entries flat (r, m, c) -> r * 20 + m * 2 + c, only those in `used` "
    "(anchors/ANCHORS.md).")
doc("apply_items", "A new batch: row r of b with items[item_of_row[r]] applied, exactly as run_items would: traced, "
    "then run in lockstep or item by item, whichever launches fewer kernels (FAST.md §2); count loops go to "
    "_apply_counts.")
doc("Ops", "The routines the learner can call to build problems: (law, pattern) -> [(token, binding)], in the order "
    "practised. run() runs the first (data is built with it) or the i-th (a check passes when any confirms it, as a "
    "pattern may not tell a routine from its alias) on a fresh machine.")
doc("SelfClimb.probe", "A group of roots' probe (bfs_roots, SELF_PRACTICE.md §2.3): their laws over the composition "
    "vocabulary at probe_budget, a root's first time the open laws over the primitives alone at fallback_budget, with "
    "deep the open laws at deep_budget; records each open law's closeness, the deep probes' gate.")
doc("LibraryMachine.outer_ste", "The answer's program on one-hot digits (LEARNED_LOOP.md §4): x = Sx, N = Sn and T's "
    "digits Cnt in X, N and S, the selected program run on libcore.SteMachine; A's low W places [X, W, 10]. With "
    "autograd every program runs (straight-through over the tournament's), without only the argmax.")
doc("SelfClimb.search_round", "The round by composition (libcore.Climb.search_round): the routines and the "
    "straight-line primitives at the task's full size, the goal the answers of the smallest T as they are (the "
    "answer's program is search_outer's); ROUNDT in the acceptance.")
doc("SelfClimb.search_outer", "The answer's program (LEARNED_LOOP.md §3): each round a routine T.round<i>, composed "
    "with the library's routines, the whole-register instructions and count loops, breadth-first by cost against rows "
    "of every T (Nd, Xd, Cd, Yd; x in X, N in N, the count in S), accepted if it reproduces every row (OuterMachine). "
    "Returns (programs, the library with the rounds, a record).")
doc("OuterMachine", "An answer's program (count loops at its top level, plain items between them) on `rows` machines "
    "at once: each stretch and each loop body a RoundMachine on one shared state, a loop replaying its body while any "
    "row's count lasts (LEARNED_LOOP.md §2). Returns run_items' result.")
doc("SelfClimb.outer_classes", "The answer program's vocabulary (LEARNED_LOOP.md §3): the whole-register instructions "
    "and every routine (vocab \"all\"; \"rounds\": the rounds only), each one item, and a count loop on every register "
    "over every distinct body of up to max_body of those items (bodies deduped as straight-line code).")
doc("SteMachine._count", "A count loop on register r (LEARNED_LOOP.md §2): the count counted down by the subtraction "
    "cell; each round the rows whose count down did not borrow run the body and take its state, the others keep "
    "theirs (a straight-through blend); rows above LOOP_CAP end cleared.")
doc("LibraryMachine.run_outer", "Answer program i (from the buffers) on digits, for training's checks "
    "(LEARNED_LOOP.md §4): fresh machines with x = Sd, N = Nd and the count's digits C in X, N and S, run with the "
    "snapped cells on libcore.OuterMachine. Returns A's low W places [X, W].")
doc("_apply_counts", "apply_items with count loops among the items (LEARNED_LOOP.md §2): the loops' rows run their "
    "bodies together, one apply_items call per iteration, each probe keeping the result while its count lasts; above "
    "LOOP_CAP it ends cleared.")
doc("count_down", "How many times a count loop runs on each count C [..., P] (digits, LSB first), counting down with "
    "the subtraction cell (LEARNED_LOOP.md §2): the counts [...] and ok [...], the count at most LOOP_CAP.")
doc("LibraryMachine.keep_counts", "The answer rows whose count reads the same both ways (a count of one digit, or a "
    "palindrome: LEARNED_LOOP.md §6) join the answer program's pool (phase C), at most COUNT_EACH of each count and "
    "COUNT_POOL in all, and ROUNDT's, the newest ROUNDT of each.")
doc("tourney_nll", "The tournament's loss, -log sum_c w_c [c passes every observation] (tournament/TOURNAMENT.md "
    "§5): lw [K] the log weights of K whole combinations, ok [K] their verdicts; None when none passes.")
doc("LibraryMachine.round_lesson", "The answer programs' tournament on the batch's answer rows (tourney_nll): a "
    "program passes when it reproduces every one; the loss is -log of the passing programs' total weight, nothing "
    "when none passes.")
doc("SelfClimb.pick", "lpca: concepts (a law on a pattern), hardest by their cheapest binding, each practised once in "
    "every binding tied for cleanest (SELF_PRACTICE.md §2.4); a practised concept's other bindings are posed only "
    "when no new concept is learnable.")
doc("Control", "The training schedule (Python-side, as a scheduler's state is): the reader's stages, the phases A-D "
    "and frozen, the climb and the audits' counters; the rows kept and the round programs are LibraryMachine's "
    "buffers.")
doc("LibraryMachine.outer_unit", "The answer's program (phase C, once the round is found; LEARNED_LOOP.md §3): "
    "search_outer with the simplest rounds as routines, against the kept rows of every count once they show "
    "COUNT_MIN counts; again after OUTER_RETRY steps or a new count, while none is found.")
doc("_grouped_work", "WORK as _apply_grouped counts it on items with cnt[i] rows each, without running it (FAST.md "
    "§2, grouped): each group's steps in _apply_grouped's order.")
doc("SteMachine", "A found program on one-hot digits, the prediction path (LEARNED_LOOP.md §2; INTEGRATION.md): every "
    "cell a straight-through table, the pointer a Python int, registers immutable tensors (autograd keeps every "
    "earlier value); every cell's output and count loop's blend passes `damp` (its gradient scaled).")
doc("enumerate_items", "Every distinct item (tokens, then loops over bodies of cost up to max_body), the cheapest "
    "spelling first, items that agree on S_test random machines of every setting being one (FAST.md §2). Raises "
    "OutOfWork past stop_at; cache: {item: signature} kept across calls with the same settings.")
doc("SelfClimb.concepts", "The k hardest concepts not yet practised in any binding (a concept: a law on a pattern, as "
    "hard as its cheapest learnable binding), each in its cleanest learnable binding (SELF_PRACTICE.md §2.4).")
doc("SelfClimb.macro_classes", "The round's search by calls (SP_BREADTH.md): one item per routine and routing of its "
    "inputs (each from A, X, Y, N or zero, by route), and the whole-register instructions; every item costs 1.")
doc("LibraryMachine.which_run", "The whole run field f's snapped pointer reads (learned_wiring/WIRING.md): its index "
    "[X], its direction [X] (0: from the run's left end) and whether it is one [X].")
doc("LibraryMachine.shape_logprobs", "shape_probs in log space, no clamp (INTEGRATION.md): log P(field f reads pos as "
    "a whole number of length digits) under each shape [X, K, 3]; T's direction lesson needs it (a clamped weight's "
    "log has gradients up to 1e30).")
doc("RoundMachine.run", "One run from fresh machines (GENERIC_OPS.md §1.1): x digits Xd in X and Nd in N (rows m <= "
    "rows), the rest fresh's random draws for the rows of go, exactly as run_prog draws them. Returns A's digits [m, "
    "P] (a view, until the next run).")
edit("room", """# the round machine (s3i): one program on many rows from fresh machines, as run_prog runs it. Control never depends on
# data, so the pointer is resolved once on the host (INC and the routines' saves and restores cost nothing) and every
# other instruction becomes one or two fixed slice operations on one state tensor; on a GPU the whole program is one
# CUDA graph, captured once and replayed
""", """# the round machine (s3i; GENERIC_OPS.md §1.1): one program on many rows from fresh machines, as run_prog runs it, the
# pointer resolved on the host, on a GPU the whole program one CUDA graph
""")
EDITS.append(("sub", "room 14", r"^( +)# -{19,} ", r"\1# --- "))   # the method rules (no check reads them): 3 characters

# ================================================================================================ the build
def undefined_globals(src: str) -> set:
    """Names a function reads as globals that neither the module's top level nor builtins define (symtable)."""
    top = symtable.symtable(src, "family", "exec")
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
    """The family file's text and the log of the edit groups applied; dev: over the size limit is allowed (a
    working build, never the submission)."""
    src = base()
    log = []
    for kind, group, old, new in EDITS:
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
    sets = dict(a.split("=", 1) for a in argv if "=" in a and not a.startswith("--"))
    src, log = build(sets, dev="--dev" in argv)
    if "--diff" in argv:
        sys.stdout.writelines(difflib.unified_diff(base().splitlines(True), src.splitlines(True), "nobook", "family",
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
