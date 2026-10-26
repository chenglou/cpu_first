"""Builds the combined self-practising file (integration/combined_sp2/COMBINED.md): sp2 with every variant of it that
certified on its own (OVERVIEW.md §4.8), in one file. The certified files are only read (each md5 asserted), and so
are the two study builders whose edit lists this one reuses (make_strict.py, make_fix_variants.py, md5s asserted, and
each replayed first: its edits must rebuild its certified file byte for byte). Writes submission_stage3_combined.py.

The pieces, in the order they are applied (each a named group of exact edits, every edit asserted to match):
  ll      the learned loop over T (learned_loop/submission_stage3_ll.py, LEARNED_LOOP.md): no countdown, no gate, no
          loop law; the count loop C.r[body]; T in S; the answer program's search after the round
  g3      generic instructions (generic_ops/submission_stage3_g3.py, GENERIC_OPS.md): the if block C[...] for SEL.S, the
          shifts through k for BRG.X, and g3's search rules (SP_ALIAS_FIT, SP_CALLS_FIRST, SP_DEEP_K 16,
          SP_DEEP_ROTATE, SP_DEEP_UNION, SP_PURE_FIRST, SP_VERIFY to four digits, ROUND_STATES 2M)
  book    the generated law book (sp_ablation/variants/book_gen.py, SP_ABLATION.md §5.2)
          (these three were built from sources, not as edits of sp2: their edits are the hunks of each certified
          file's diff against sp2, applied to sp2 together; where two of them change the same lines of sp2, MERGES
          below holds the hand-merged text, and every hunk there must lie inside it)
  fix     the two weak checks fixed (ablation_s3/fix, ABLATION_S3.md §8), as make_fix_variants.py carried them into
          book_gen (its own Draw and law_target, both reductions at full size): FULL_CHECK, ABOVE_CHECK
  strict  the strict build (strict/make_strict.py, STRICT.md): work budgets for the clock, the kept rows and the
          programs in buffers, Adam's own reset, no late lessons, no wrap check; its edits are make_strict.py's,
          and where the learned loop changed their text, ADAPT below holds the combined edit
  glue    what the pieces need from each other once together (GLUE below): the answer programs' count loops and
          the if blocks in the buffers' instructions and their machines, the count rows as buffers, the work of the
          answer program's acceptance counted

Left out, as the brief says: the bridge's invented select (made before training) and sp_breadth's round search.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/combined_sp2/make_combined.py [--diff PIECE] [NAME=VALUE ...]

(--diff PIECE prints one piece's effect as a unified diff; NAME=VALUE sets a constant, for research builds only.)
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import re
import symtable
import sys
import builtins
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTEG = HERE.parent
sys.path.insert(0, str(INTEG / "strict"))
sys.path.insert(0, str(INTEG / "ablation_s3" / "fix"))
OUT = HERE / "submission_stage3_combined.py"
LIMIT = 256 * 1024                          # the evaluator's limit on a submission's size

SP2 = (INTEG / "self_practice" / "submission_stage3_sp2.py", "afc70fe91069a85c416eba18b48ee30e")
CERTIFIED = {   # piece: (its certified file, md5) (OVERVIEW.md §4.8)
    "fix": (INTEG / "ablation_s3" / "fix" / "submission_stage3_sp2_fix.py", "a92d62812e8445789e14c86e61b0164c"),
    "strict": (INTEG / "strict" / "submission_stage3_sp2_strict.py", "1a1f15fb231d6da5fd53d530bc9bdcb6"),
    "ll": (INTEG / "learned_loop" / "submission_stage3_ll.py", "419c15e70ccf6483ff68b04b08f52bdc"),
    "book": (INTEG / "sp_ablation" / "variants" / "book_gen.py", "8bbac4a1c631a08c8b72179de37fcb26"),
    "g3": (INTEG / "generic_ops" / "submission_stage3_g3.py", "e6bd2a1f9bfd9fd2119b0d44483c7100"),
}
BUILDERS = {    # the study builders whose edit lists are reused, and what each must rebuild
    "make_strict": (INTEG / "strict" / "make_strict.py", "5d1af920ac2088ad1400c3b0a8ff49d0"),
    "make_fix": (INTEG / "ablation_s3" / "fix" / "make_fix.py", "dd7f009a798acc5191fef54d601fae79"),
    "make_fix_variants": (INTEG / "ablation_s3" / "fix" / "make_fix_variants.py", "7f690d5f403ff0009e5de8a675167848"),
    "make_variants": (INTEG / "ablation_s3" / "make_variants.py", "9c23d836eb6536291584883d33beebab"),
}
BOOK_GEN_FIX_MD5 = "8b701e2825db56ccc0df65805469f417"     # sp_ablation/fix/book_gen_fix.py, certified (ABLATION_S3 §8)


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def read(path: Path, want: str) -> str:
    data = path.read_bytes()
    got = hashlib.md5(data).hexdigest()
    if got != want:
        raise SystemExit(f"{path}: md5 {got} != {want}: an input changed")
    return data.decode()


for _name, (_path, _want) in BUILDERS.items():
    read(_path, _want)
import make_strict as MS        # noqa: E402  (the strict build's edits)
import make_fix as MF           # noqa: E402  (the weak checks' fix)
import make_fix_variants as MFV # noqa: E402  (... carried into book_gen)


def replay() -> None:
    """Each reused edit list rebuilds its certified file byte for byte, so the edits used here are the certified
    builds' own."""
    src, out = MF.build("submission_stage3_sp2_fix")
    assert md5(out) == CERTIFIED["fix"][1], "make_fix.py no longer rebuilds the fixed sp2"
    src, out = MFV.build("book_gen_fix")
    assert md5(out) == BOOK_GEN_FIX_MD5, "make_fix_variants.py no longer rebuilds book_gen_fix"
    out, _ = MS.build("sp2", {})
    assert md5(out) == CERTIFIED["strict"][1], "make_strict.py no longer rebuilds the strict sp2"


# ============================================================================================ ll, g3, book: the hunks
def hunks(base: list, other: list) -> list:
    """The changes of other against base as (i1, i2, lines): base[i1:i2] becomes lines (difflib's opcodes)."""
    sm = difflib.SequenceMatcher(None, base, other, autojunk=False)
    return [(i1, i2, other[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


# Where two of ll, g3 and book change the same lines of sp2: (name, sp2's text there, the pieces whose hunks lie in it,
# the merged text). Every hunk touching the text must lie inside it; the merged text replaces it.
MERGES = []


def merge(name, old, pieces, new):
    MERGES.append((name, old, tuple(sorted(pieces)), new))


merge("count: a count loop has no size; a C block runs once (ll, g3)", '''    def count(self, kind):
        return 1 if kind == ONE else {"P": self.P, "B": self.B, "W": self.W}[kind]
''', ("ll", "g3"), '''    def count(self, kind):
        assert kind not in COUNT_KINDS, "a count loop's count is data, not a size"
        return 1 if kind in (ONE, "C") else {"P": self.P, "B": self.B, "W": self.W}[kind]
''')

merge("run_items: the if block (g3) and the count loop (ll), and their helpers", '''        raise ValueError(tok)


def run_items(st, b, items, lib):
    for it in items:
        if is_loop(it):
            _, kind, body = it
            if kind in lib:
                run_items(st, b, fill(lib[kind], body[0]), lib)
            else:
''', ("ll", "g3"), '''        raise ValueError(tok)


def _unless(b, old, keep):
    """Where keep [S, NP]: the registers and flags as in old (a C block's probes whose borrow was clear)."""
    b.regs = torch.where(keep[:, None, None, :], old.regs, b.regs)
    b.c, b.h, b.k = (torch.where(keep, o, n) for o, n in ((old.c, b.c), (old.h, b.h), (old.k, b.k)))


def count_down(cells, C):
    """How many times a count loop runs on each count C [..., P] (digits, LSB first): C counted down by one with the
    subtraction cell (the digit 1 at place 0, the borrows rippling up) until a count down borrows, at most LOOP_CAP
    times. Returns the counts [...] and ok [...]: the count is at most LOOP_CAP (else the loop's error)."""
    D, B = C.long(), cells.B
    cnt = torch.zeros(D.shape[:-1], dtype=torch.long, device=D.device)
    done = torch.zeros_like(cnt, dtype=torch.bool)
    for _ in range(LOOP_CAP + 1):
        c, out = torch.zeros_like(cnt), []
        for j in range(D.shape[-1]):
            o = cells.sbb[(D[..., j] * B + int(j == 0)) * 2 + c]
            out.append(o[..., 0])
            c = o[..., 1]
        go = ~done & (c == 0)
        D = torch.where(go[..., None], torch.stack(out, -1), D)
        cnt, done = cnt + go.long(), done | (c == 1)
        if bool(done.all()):
            break
    return cnt, done


def _merge(b, nb, live):
    """b's probes where live [S, NP] take nb's registers and flags (in place)."""
    b.regs = torch.where(live[:, None, None, :], nb.regs, b.regs)
    b.c, b.h, b.k = (torch.where(live, getattr(nb, f), getattr(b, f)) for f in ("c", "h", "k"))


def _clear(b, bad):
    """The count loop's error: the probes where bad [S, NP] get every register and flag 0 (in place)."""
    b.regs = b.regs.masked_fill(bad[:, None, None, :], 0)
    b.c, b.h, b.k = (getattr(b, f).masked_fill(bad, 0) for f in ("c", "h", "k"))


def _run_count(st, b, it, lib):
    """A count loop on batch b in place: each probe runs the body as many times as its count register held when the
    loop began (count_down), the body as a routine (it starts at the loop's pointer, which is restored after it)."""
    _, kind, body = it
    cnt, ok = count_down(st.cells, b.regs[:, RI[kind[2:]]].transpose(1, 2))
    p = b.p.clone()
    for i in range(int(torch.where(ok, cnt, 0).max()) if cnt.numel() else 0):
        nb = b.copy()
        run_items(st, nb, body, lib)
        _merge(b, nb, ok & (cnt > i))
        b.p = p.clone()
    _clear(b, ~ok)


def run_items(st, b, items, lib):
    for it in items:
        if is_count(it):
            _run_count(st, b, it, lib)
        elif is_loop(it):
            _, kind, body = it
            if kind in lib:
                run_items(st, b, fill(lib[kind], body[0]), lib)
            elif kind == "C":                        # if the borrow: the body runs; where the borrow was clear at
                old = b.copy()                       # its start, its effects are undone (the pointer moves anyway)
                run_items(st, b, body, lib)
                _unless(b, old, old.c == 0)
            else:
''')

merge("RoundMachine: a stretch starts at its pointer (ll); a C block's buffers (g3)", '''        fns, p, saved = [], 0, [0] * MAX_DEPTH
        for o in ops:
''', ("ll", "g3"), '''        fns, p, saved = [], p0, [0] * MAX_DEPTH
        bufs, depth = [], 0                          # a C block: its start's state and borrow, one buffer per nesting
        for o in ops:
''')

merge("s3i's practice tasks: gone in both (ll drops more of them: none of it is called)", '''# ===================================================================================================== practice tasks
NEEDS = {"lt": (), "modx": (), "add": ("lt", "modx"), "modB": ("add",), "mrow": ("add",), "mac": ("mrow", "add"),
         "mul": ("add",), "step": ("modB",), "reduce": ("add",)}
SWAP = {"X": "Y", "Y": "X", "N": "N"}
MAC_ADDEND = "N"           # mac's addend: N (x d + N) or the running sum in A (A + x d, the A/B's variant C)


def task_names(variants=True):
    out = []
    for t in NEEDS:
        out.append(t)
        if variants:
            out.append(t + "~")
    return out


def needs(task):
    return tuple(n + ("~" if task.endswith("~") else "") for n in NEEDS[task.rstrip("~")])


''', ("ll", "g3"), '''# ===================================================================================================== the practice problems' helpers
''')

# The hand merges too long to write out: a span of sp2's lines named by its first and last lines, its text's md5
# asserted.
SPANS = []


def span(name, first, last, text_md5, pieces, new):
    SPANS.append((name, first, last, text_md5, tuple(sorted(pieces)), new))


span("s3i's practice problems (build, LawProblem, Acceptor, accept): ll drops them all, g3 the Acceptor; nothing calls "
     "them", ["\n", "\n", "def build(task, st, n, gen, rows, helpers, lib, device):\n"],
     ["        if not bool(prob.goal([b])[0]):\n", "            return False\n", "    return True\n"],
     None, ("ll", "g3"), "")
span("s3i's climb (simplest, Climb): gone in both, with the same text left", ["\n", "\n",
     "def simplest(progs, lib, by_steps=False):\n"], ["        self.queue = None\n", "        return line\n", "\n", "\n"],
     None, ("ll", "g3"), "\n\n")
span("parse: kept (g3 dropped it as unused; the strict build reads the round programs' text back with it)",
     ["\n", "\n", "def parse(s):\n"], ["    return tuple(seq())\n"], None, ("g3",), None)       # None: sp2's text
span("the round's search: no t (ll: the answers of the smallest T as they are), by calls first (g3)",
     ["            if t == 1:\n"],
     ["                found, accepted, sizes, stop = self._round_many(prob, classes, lib, answers, t, budget, deadline)\n"],
     None, ("ll", "g3"), "@G3_ROUND@")
span("the docstring: the combined file's own (doc.py)", ['"""Stage 3 with self-posed practice (cpu_first/integration/SELF_PRACTICE.md): '
     'integration stage 3 (s3i,\n'], ['"""\n'], None, ("book", "g3", "ll"), "@DOC@")


def g3_round(g3_lines: list) -> str:
    """g3's round search (its `if t == 1:` block) one indent out, as ll has no t."""
    i = g3_lines.index("            if t == 1:\n")
    j = g3_lines.index("            else:\n", i)
    assert g3_lines[j + 1].lstrip().startswith("found, accepted, sizes, stop = self._round_many(")
    body = g3_lines[i + 1:j]
    assert all(l.startswith("                ") or l == "\n" for l in body)
    return "".join(l[4:] for l in body)


def own_ranges(base: list) -> list:
    """MERGES and SPANS as line ranges of sp2: (lo, hi, name, pieces, new text or None for sp2's own)."""
    text = "".join(base)
    out = []
    for name, old, pieces, new in MERGES:
        assert text.count(old) == 1 and old.endswith("\n"), name
        lo = text[:text.index(old)].count("\n")
        assert text.index(old) == len("".join(base[:lo])), name           # it starts at a line's start
        out.append((lo, lo + old.count("\n"), name, pieces, new))
    for name, first, last, want, pieces, new in SPANS:
        at = [i for i in range(len(base)) if base[i:i + len(first)] == first]
        assert len(at) == 1, (name, len(at))
        i = at[0]
        j = next(j for j in range(i + len(first) - 1, len(base)) if base[j:j + len(last)] == last) + len(last)
        if want is not None:
            assert md5("".join(base[i:j])) == want, (name, md5("".join(base[i:j])))
        out.append((i, j, name, pieces, new))
    out.sort()
    assert all(a[1] <= b[0] for a, b in zip(out, out[1:])), "two hand merges overlap"
    return out


def merge_pieces(base_text: str, texts: dict, subs: dict) -> tuple[str, list]:
    """sp2 with every hunk of ll, g3 and book (texts: piece -> its certified text), the hand merges where they meet.
    Returns the text and a log of (what, count)."""
    base = base_text.splitlines(True)
    own = own_ranges(base)
    seen = {o[2]: set() for o in own}
    edits, log = [], []
    for piece, text in texts.items():
        n = 0
        for i1, i2, lines in hunks(base, text.splitlines(True)):
            ins = i1 == i2
            home = [o for o in own if (o[0] < i1 < o[1] or (i1 == o[0] == 0)) if ins] or \
                   [o for o in own if not ins and i1 < o[1] and i2 > o[0]]
            if home:
                o = home[0]
                assert ins or (o[0] <= i1 and i2 <= o[1]), (piece, i1, i2, o[2])
                seen[o[2]].add(piece)
                continue
            edits.append((i1, i2, "".join(lines), piece))
            n += 1
        log.append((f"{piece}: hunks applied as they are", n))
    for lo, hi, name, pieces, new in own:
        assert tuple(sorted(seen[name])) == pieces, (name, sorted(seen[name]), pieces)
        new = "".join(base[lo:hi]) if new is None else subs.get(new, new)
        edits.append((lo, hi, new, "merge"))
        log.append((f"merge: {name}", 1))
    edits.sort(key=lambda e: (e[0], e[1]))

    def keeps_edge(m, at):                      # a hand merge whose line next to an insertion is sp2's own
        lines = m[2].splitlines(True)
        return bool(lines) and (lines[-1] == base[m[1] - 1] if at == m[1] else lines[0] == base[m[0]])
    for a, b in zip(edits, edits[1:]):          # no two pieces' hunks meet outside the hand merges
        ok = a[1] < b[0] or (a[1] == b[0] and a[0] < a[1] and b[0] < b[1]) or (a[1] == b[0] and a[3] == b[3])
        for m, x in ((a, b), (b, a)):
            if m[3] == "merge" and x[0] == x[1] and x[0] in (m[0], m[1]) and keeps_edge(m, x[0]):
                ok = True
        assert ok, ("two pieces change the same lines", a[:2], a[3], b[:2], b[3])
    out = list(base)
    for i1, i2, new, _ in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        out[i1:i2] = new.splitlines(True)
    return "".join(out), log


# ============================================================================================== strict: make_strict.py
def key(old: str) -> str:
    """A make_strict.py edit's name here: its old text's first two lines (unique among its 52)."""
    return "".join(old.splitlines(True)[:2])


# make_strict.py's edits whose text the other pieces changed: the start of its old text -> ("gone", why) (the code it
# edits is not here) or (why, the combined edits [(what, old, new, count)] that do its job on this text).
ADAPT = {}


def gone(old_start: str, why: str) -> None:
    ADAPT[old_start] = ("gone", why)


def adapt(old_start: str, why: str, edits: list) -> None:
    ADAPT[old_start] = (why, edits)


for _k in ("# the deadline passed while a vocabulary was being built",
           "      search_s: one search's time limit (its stop counts as a used-up cap);",
           "    def search(self, task, rows, deadline=None):\n        t0 = time.time()\n",
           "        def run(how, classes, lib, cap):\n            t1 = time.time()\n",
           "        if res.found_cost is None and not (climb_deadline is not None and time.time() > climb_deadline):",
           "                   seconds=round(time.time() - t0, 1), stop=res.stop, how=how,",
           "        t0 = time.time()\n        routines = [k for k in self.lib if k.startswith(\"T.\")]\n",
           "                   seconds=round(time.time() - t0, 1), stop=res.stop, routines=len(routines),"):
    gone(_k, "s3i's climb (libcore.Climb), which ll and g3 drop: nothing calls it")

adapt('''        round programs and their library are given by the check (ctl.progs, ctl.lib)."""''',
      "ll: designed's docstring (no gate; answer programs)", [("round", '''the answer
        programs and their library are given by the check (ctl.progs, ctl.lib)."""''', '''the answer
        programs are given by the check (set_round)."""''', 1)])

adapt("    def round_ste(self, s: Tensor, Nr: Tensor, SBB: Tensor, ADC: Tensor, MUL: Tensor) -> Tensor:\n",
      "ll: the prediction runs the answer program (outer_ste), its instructions from the buffers", [("round",
          '''    def outer_ste(self, Sx: Tensor, Sn: Tensor, Cnt: Tensor, SBB: Tensor, ADC: Tensor, MUL: Tensor) -> Tensor:
        """The answer's program on one-hot digits: a clean machine with x = Sx in X, N = Sn in N and T's digits Cnt in
        S, the selected program run on it (libcore.SteMachine; a count loop counts down by the subtraction table);
        returns A's low W places [X, W, 10]. Straight-through over the tournament's programs: with autograd every
        program runs (the gradient reaches the tournament's logits); without, the argmax program only."""
''', '''    def outer_ste(self, Sx: Tensor, Sn: Tensor, Cnt: Tensor, SBB: Tensor, ADC: Tensor, MUL: Tensor,
                  ops: list) -> Tensor:
        """The answer's program on one-hot digits: a clean machine with x = Sx in X, N = Sn in N and T's digits Cnt in
        S, the selected program (ops: each program's instructions, round_trace) run on it (libcore.SteMachine; a
        count loop counts down by the subtraction table); returns A's low W places [X, W, 10]. Straight-through over
        the tournament's programs: with autograd every program runs (the gradient reaches the tournament's logits);
        without, the argmax program only."""
''', 1)])

adapt("        s = Sx\n        if ctl.progs:\n", "ll: no countdown; the answer program, from the buffers", [("round",
      "        s = self.outer_ste(Sx, Sn, Cnt, SBB, ADC, MUL) if ctl.progs else Sx\n",
      "        ops = [self.round_trace(i) for i in range(self.n_progs())]     # the answer programs, from the buffers\n"
      "        s = self.outer_ste(Sx, Sn, Cnt, SBB, ADC, MUL, ops) if ops else Sx\n", 1)])
gone("                R = self.round_ste(s, Sn, SBB, ADC, MUL)\n", "ll: no countdown, so no round_ste call")

adapt("    def run_rounds(self, Sd: Tensor, C: Tensor, Nd: Tensor, prog, cap: int | None = None) -> Tensor:",
      "ll: training's checks run the answer program (run_outer), program i from the buffers", [("round",
          "    def run_outer(self, Sd: Tensor, C: Tensor, Nd: Tensor, prog) -> Tensor:\n",
          "    def run_outer(self, Sd: Tensor, C: Tensor, Nd: Tensor, i: int) -> Tensor:\n", 1)])
adapt('''        """The countdown and the round program on digits: rounds until no row commits (the snapped subtraction for
        C - 1, the gate on its borrow), each committing row running the program (libcore's machine at PR places, the''',
      "ll: run_outer's docstring", [("round", '''        """The answer's program on digits (training's checks): fresh machines (run_prog's dirty starts) with x = Sd in X,
        N = Nd in N and the count's digits C in S, the program run with the snapped cells (libcore.OuterMachine: every
        row at once, on a GPU each stretch and each count loop's body one CUDA graph). Returns A's low W places
        [X, W]."""''', '''        """Answer program i (its instructions from the buffers) on digits (training's checks): fresh machines
        (run_prog's dirty starts) with x = Sd in X, N = Nd in N and the count's digits C in S, the program run with the
        snapped cells (libcore.OuterMachine: every row at once, on a GPU each stretch and each count loop's body one CUDA
        graph). Returns A's low W places [X, W]."""''', 1)])
adapt("        m = round_machine(st, prog, self.ctl.lib, Sd.shape[0], Sd.device)\n",
      "ll: the outer machine, from the buffers' instructions", [("round",
          "        m = outer_machine(round_settings(cells, PR), prog, self.ctl.lib, Sd.shape[0], Sd.device)\n",
          "        m = outer_machine_ops(round_settings(cells, PR), self.round_trace(i), Sd.shape[0], Sd.device)\n", 1)])
adapt('''        sel = torch.nonzero(aux["small"]).flatten()
        n = len(ctl.progs)
''', "ll: the tournament runs the answer programs (no cap: no countdown)", [("round", '''        sel = torch.nonzero(aux["small"]).flatten()
        n = len(ctl.progs)
        if sel.numel() == 0 or n == 0:
            return self.prog.sum() * 0.0
        with torch.no_grad():
            ok = torch.stack([(self.run_outer(aux["Xd"][sel], aux["Cd"][sel], aux["Nd"][sel], p) == yd[sel]).all(1)
                              for p in ctl.progs])                           # [programs, rows]
''', '''        sel = torch.nonzero(aux["small"]).flatten()
        n = self.n_progs()
        if sel.numel() == 0 or n == 0:
            return self.prog.sum() * 0.0
        with torch.no_grad():
            ok = torch.stack([(self.run_outer(aux["Xd"][sel], aux["Cd"][sel], aux["Nd"][sel], i) == yd[sel]).all(1)
                              for i in range(n)])                            # [programs, rows]
''', 1)])
adapt('''        self.pool: list = []                  # read rows (N, x digits) for the practice problems
        self.ans_pool: list = []              # answer rows (N, x, y) of the smallest T seen, for the round's search
''', "ll: Control's fields (the count rows go to buffers too; the outer search's retry stays the scheduler's)",
      [("pools", '''        self.pool: list = []                  # read rows (N, x digits) for the practice problems
        self.ans_pool: list = []              # answer rows (N, x, y) of the smallest T seen, for the round's search
        self.ans_t: int | None = None         # ... that T
        self.count_pool: dict = {}            # count digits -> answer rows (N, x, count, y) whose count reads the
                                              #   same both ways, for the answer's program (the newest kept)
        self.outer_tried: tuple | None = None # (step, counts) of the last search for the answer's program
        self.climb = None                     # spcore.SelfClimb, from phase C on
        self.cells = None                     # the snapped cells the climb runs with (libcore.Cells)
        self.lib: dict = {}                   # the library the answer programs use
        self.progs: list = []                 # the tournament's answer programs (items)
        self.climb_log: list = []
        self.deadline: float | None = None    # the training budget's end (build_optimizer), for the searches
        self.round_fit: list = []             # each answer program's share of this step's answer rows
''', '''        self.outer_tried: tuple | None = None # (step, counts) of the last search for the answer's program
        self.climb = None                     # spcore.SelfClimb, from phase C on
        self.climb_work0 = 0.0                # WORK when the climb started (its budget counts from there)
        self.cells = None                     # the snapped cells the climb runs with (libcore.Cells)
        self.climb_log: list = []
        self.round_fit: list = []             # each answer program's share of this step's answer rows
''', 1)])
adapt("        pool = torch.cat([torch.stack([n, x]) for n, x in ctl.pool], 1) if ctl.pool else None\n        if pool is None:\n",
      "ll: the climb unit (no t; once the climb is done with a round, the answer program's unit)", [("pools", '''        pool = torch.cat([torch.stack([n, x]) for n, x in ctl.pool], 1) if ctl.pool else None
        if pool is None:
            return
        ans = None
        if len(ctl.ans_pool) and sum(a[0].shape[0] for a in ctl.ans_pool) >= ANS_MIN:
            ans = tuple(torch.cat([a[i] for a in ctl.ans_pool]) for i in range(3))
        t0 = time.time()
        dl = None if ctl.deadline is None else ctl.deadline - CLIMB_MARGIN
        line = ctl.climb.step((pool[0], pool[1]), answers=ans, deadline=dl)
        if line is not None:
            ctl.climb_log.append(line)
            print(f"LR climb step={ctl.step} {line} ({time.time() - t0:.1f}s)", flush=True)
        elif ctl.climb.rounds and not ctl.progs:
            self.outer_unit(dl)
''', '''        pool = self.kept(self.pool_rows, self.pool_sizes)                    # the rows kept (buffers)
        if pool is None:
            return
        ans = self.kept(self.ans_rows, self.ans_sizes)
        ans = None if ans is None or ans.shape[0] < ANS_MIN else tuple(ans[:, i].contiguous() for i in range(3))
        t0, w0 = _clock(), WORK[0]
        dl = ctl.climb_work0 + CLIMB_WORK if CLIMB_WORK else None            # the climb's work budget
        line = ctl.climb.step((pool[:, 0].contiguous(), pool[:, 1].contiguous()), answers=ans, stop_at=dl)
        if line is not None:
            ctl.climb_log.append(line)
            print(f"LR climb step={ctl.step} {line} [work +{WORK[0] - w0:.4g}, {WORK[0] - ctl.climb_work0:.4g} in all] "
                  f"({_clock() - t0:.1f}s)", flush=True)
        elif ctl.climb.rounds and not self.n_progs():
            self.outer_unit(dl)
''', 1)])


def swap(text: str, pairs: list) -> str:
    """text with each (old, new) of pairs replaced, each old asserted to occur exactly once."""
    for a, b in pairs:
        assert text.count(a) == 1, a[:80]
        text = text.replace(a, b)
    return text


def _strict_edit(start: str) -> tuple:
    hit = [e for e in MS.MODEL if e[1].startswith(start)]
    assert len(hit) == 1, start
    return hit[0]


_what, _old, _new, _count = _strict_edit("    def program(self, i: int | None = None):\n")
adapt("    def program(self, i: int | None = None):\n",
      "ll: the buffers hold the answer programs, a count loop's body between markers (answer_trace)", [(_what, swap(_old, [(
          '''        """Round program i (default: the tournament's argmax), or None before the round is found."""''',
          '''        """Answer program i (default: the tournament's argmax), or None before one is found."""''')]),
          swap(_new, [
              ("# ------------------------------------------------------------------ the buffers: rows kept, the round programs",
               "# ------------------------------------------------------------------ the buffers: rows kept, the answer programs"),
              ('''        """The round programs into the buffers: each one's instructions as both machines run them (trace, at the round
        machine's size) in round_ops and round_len, and their text with the definitions they use in round_src."""''',
               '''        """The answer programs into the buffers: each one's instructions as both machines run them (answer_trace, at
        the round machine's size: a count loop's body between its markers) in round_ops and round_len, and their text
        with the definitions they use in round_src."""'''),
              ("        ops = [_trace_seq(st, p, lib, 0, _traces(st, lib))[0] for p in progs]\n",
               "        ops = [answer_trace(st, p, lib) for p in progs]\n"),
              ('''        """The round programs as written and the definitions they use, read back from round_src: (lib, programs)."""''',
               '''        """The answer programs as written and the definitions they use, read back from round_src: (lib, programs)."""'''),
              ('''        """Round program i's instructions, read from round_ops (array('h') of trace's opcodes)."""''',
               '''        """Answer program i's instructions, read from round_ops (array('h') of answer_trace's opcodes)."""'''),
              ('''        """Round program i's slot (default: the tournament's argmax), or None before the round is found."""''',
               '''        """Answer program i's slot (default: the tournament's argmax), or None before one is found."""''')]),
          _count)])


def strict_edits() -> tuple[list, list]:
    """make_strict.py's sp2 edits, each as it is or as ADAPT says; and the log of what was adapted or dropped."""
    out, notes = [], []
    edits = MS.LIBCORE + MS.SPCORE + MS.MODEL + MS.kind_patches("sp2") + MS.NOWRAP
    keys = [key(o) for _, o, _, _ in edits]
    assert len(set(keys)) == len(keys)
    used = set()
    for (what, old, new, count), k in zip(edits, keys):
        hit = [a for a in ADAPT if k.startswith(a) or a.startswith(k)]
        assert len(hit) <= 1, hit
        if not hit:
            out.append((what, old, new, count))
            continue
        used.add(hit[0])
        why, repl = ADAPT[hit[0]]
        if why == "gone":
            notes.append((f"strict ({what}): dropped, {repl}", 1))
            continue
        out += repl
        notes.append((f"strict ({what}): adapted, {why}", len(repl)))
    assert used == set(ADAPT), set(ADAPT) - used
    return out, notes


def apply_strict(src: str, log: list) -> str:
    """make_strict.build's steps for sp2 on this text: the late lessons' rewrite, the edits (strict_edits), the
    clock's renames and the budgets."""
    tail = MS.OLD_TABLE_LESSONS_TAIL.replace("def cell_lessons(self, Cd: Tensor)", "def cell_lessons(self)")
    new_tl = MS.NEW_TABLE_LESSONS.replace("        Td, Tb, _ = self.tables()\n", "        Td, Tb = self.tables()\n")
    assert tail != MS.OLD_TABLE_LESSONS_TAIL and new_tl != MS.NEW_TABLE_LESSONS      # (ll: no gate)
    head, end = src.index(MS.OLD_TABLE_LESSONS_HEAD), src.index(tail)
    body = src[head:end]
    assert src.count(MS.OLD_TABLE_LESSONS_HEAD) == 1 and src.count(tail) == 1 and body.count(MS.TABLE_CHECKS_START) == 1
    assert body.count("if pieces:") == 5, body.count("if pieces:")
    src = src[:head] + new_tl + body[body.index(MS.TABLE_CHECKS_START):] + src[end:]
    log.append(("strict (late): table_lessons without the late lessons (make_strict's rewrite)", 1))
    edits, notes = strict_edits()
    log += notes
    src = MS.apply(src, edits, log)
    for pat, new in MS.RENAMES:
        src, n = re.subn(pat, new, src)
        log.append((f"strict (clock): rename {pat}", n))
    for name, value in MS.BUDGETS.items():
        assert src.count(f"@{name}@") == 1, name
        src = src.replace(f"@{name}@", value)
    return src


# ================================================================================================== glue
# What the pieces need from each other once together, applied after strict's renames (so in its names): (what, old,
# new, count). ll x strict: the buffers hold the answer programs (count loops at their top), so their instructions mark
# a count loop's body (CLOOP + r ... CLEND), and the outer machine and the one-hot machine run from those; the count
# rows are buffers too; the answer program's acceptance counts its machine work. g3 x strict: the one-hot machine's
# run from the buffers runs C blocks (CBEG ... CEND) as its run from the items does.
GLUE = []


def glue(what, old, new, count=1):
    GLUE.append((what, old, new, count))


glue("buffers' opcodes", '''CBEG = SAVE + 2 * MAX_DEPTH      # a C block's start and end (each probe counts the blocks it is off in: a block starts
CEND = CBEG + 1                  #   off where the borrow is clear, or inside a block that is off)
''', '''CBEG = SAVE + 2 * MAX_DEPTH      # a C block's start and end (each probe counts the blocks it is off in: a block starts
CEND = CBEG + 1                  #   off where the borrow is clear, or inside a block that is off)
CLOOP = CEND + 1                 # a count loop on register r: CLOOP + r, its body's instructions, CLEND (only in an
CLEND = CLOOP + len(REGS)        #   answer program's instructions, answer_trace: the model's buffers)
''')

glue("outer machine from instructions", '''def round_machine(st, prog, lib, rows, device):
    """prog's RoundMachine at setting st (its cells loaded) for rows machines (rounded up to a multiple of ROUND_ROWS),
    made once per instruction sequence and size and kept (ROUND_KEEP at most)."""
    return round_machine_ops(st, _trace_seq(st, prog, lib, 0, _traces(st, lib))[0], rows, device)


def round_machine_ops(st, ops, rows, device):
    """round_machine for a program given as its instructions (trace's opcodes, array('h'))."""
    R = -(-max(rows, 1) // ROUND_ROWS) * ROUND_ROWS
    key = (ops.tobytes(), st.P, st.B, R, str(device))
    m = _ROUND.get(key)
    if m is None:
        while len(_ROUND) >= ROUND_KEEP:
            _ROUND.pop(next(iter(_ROUND)))
        m = _ROUND[key] = RoundMachine(st, ops, R, device)
    m.cells(st.cells)
    return m


''', '''def answer_trace(st, prog, lib) -> array:
    """An answer program's instructions as the model's buffers keep them (array('h')): each stretch of plain items as
    _trace_seq traces it, each count loop on register r as CLOOP + r, its body's trace, CLEND."""
    ent, ops, run = _traces(st, lib), array("h"), []
    for it in tuple(prog) + (None,):
        if it is not None and not is_count(it):
            run.append(it)
            continue
        ops += _trace_seq(st, tuple(run), lib, 0, ent)[0]
        run = []
        if it is not None:
            ops += array("h", [CLOOP + RI[it[1][2:]]]) + _trace_seq(st, it[2], lib, 0, ent)[0] + array("h", [CLEND])
    return ops


def answer_parts(ops) -> list:
    """answer_trace's instructions as the program's parts, in order: (None, a stretch's instructions) or (r, a count
    loop's register and its body's instructions); empty stretches left out (count loops do not nest)."""
    parts, run, i = [], array("h"), 0
    while i < len(ops):
        if CLOOP <= ops[i] < CLEND:
            j = i + 1
            while ops[j] != CLEND:
                j += 1
            if run:
                parts.append((None, run))
                run = array("h")
            parts.append((ops[i] - CLOOP, ops[i + 1:j]))
            i = j + 1
        else:
            run.append(ops[i])
            i += 1
    if run:
        parts.append((None, run))
    return parts


''')

glue("outer machine from instructions", '''    def __init__(self, st, prog, lib, rows, device):
        ent, self.parts, self.base, p, run = _traces(st, lib), [], None, 0, []
        for it in tuple(prog) + (None,):
            if it is not None and not is_count(it):
                run.append(it)
                continue
            if run or self.base is None:
                m = RoundMachine(st, _trace_seq(st, tuple(run), lib, 0, ent)[0], rows, device, self.base, p)
                self.base, p = self.base or m, m.p_end
                if run:
                    self.parts.append((None, m))
                run = []
            if it is not None:
                body = RoundMachine(st, _trace_seq(st, it[2], lib, 0, ent)[0], rows, device, self.base, p)
                self.parts.append((RI[it[1][2:]], body))
        self.P, self.st, self.dev = st.P, self.base.st, self.base.dev
''', '''    def __init__(self, st, parts, rows, device):
        """parts: answer_parts' (None or a count loop's register, instructions); the first part's machine holds the
        state (an empty one when the program starts with a count loop)."""
        self.parts, self.base, p = [], None, 0
        for r, ops in parts:
            if self.base is None and r is not None:
                self.base = RoundMachine(st, array("h"), rows, device, None, p)
            m = RoundMachine(st, ops, rows, device, self.base, p)
            self.base = self.base or m
            if r is None:
                p = m.p_end
            self.parts.append((r, m, len(ops)))
        if self.base is None:
            self.base = RoundMachine(st, array("h"), rows, device, None, p)
        self.P, self.st, self.dev = st.P, self.base.st, self.base.dev
''')

glue("outer machine from instructions", '''        with torch.no_grad():
            for r, mach in self.parts:
                if r is None:
                    mach._program()
                    continue
                cnt, ok = count_down(self.cl, St[r * P:(r + 1) * P].T)
                for i in range(int(torch.where(ok, cnt, 0).max())):
                    old = St.clone()
                    mach._program()
                    St.copy_(torch.where((ok & (cnt > i))[None], St, old))
''', '''        self.ran = 0                     # instructions replayed (the acceptance's work: search_outer counts it)
        with torch.no_grad():
            for r, mach, n in self.parts:
                if r is None:
                    mach._program()
                    self.ran += n
                    continue
                cnt, ok = count_down(self.cl, St[r * P:(r + 1) * P].T)
                for i in range(int(torch.where(ok, cnt, 0).max())):
                    old = St.clone()
                    mach._program()
                    self.ran += n
                    St.copy_(torch.where((ok & (cnt > i))[None], St, old))
''')

glue("outer machine from instructions", '''_OUTER = {}                      # (instructions, P, B, rows, device) -> OuterMachine


def outer_machine(st, prog, lib, rows, device):
    """prog's OuterMachine (rows rounded up as round_machine's), its cells loaded; made once per program and size."""
    ent = _traces(st, lib)
    key = tuple((it[1], _trace_seq(st, it[2], lib, 0, ent)[0].tobytes()) if is_count(it)
                else _trace_of(st, it, lib, 0, ent)[0].tobytes() for it in prog)
    R = -(-max(rows, 1) // ROUND_ROWS) * ROUND_ROWS
    key = (key, st.P, st.B, R, str(device))
    m = _OUTER.get(key)
    if m is None:
        while len(_OUTER) >= ROUND_KEEP:
            _OUTER.pop(next(iter(_OUTER)))
        m = _OUTER[key] = OuterMachine(st, prog, lib, R, device)
    m.cells(st.cells)
    return m
''', '''_OUTER = {}                      # (instructions, P, B, rows, device) -> OuterMachine


def outer_machine(st, prog, lib, rows, device):
    """prog's OuterMachine (rows rounded up to a multiple of ROUND_ROWS), its cells loaded; made once per program and
    size."""
    return outer_machine_ops(st, answer_trace(st, prog, lib), rows, device)


def outer_machine_ops(st, ops, rows, device):
    """outer_machine for a program given as its instructions (answer_trace's, as the model's buffers keep them)."""
    R = -(-max(rows, 1) // ROUND_ROWS) * ROUND_ROWS
    key = (ops.tobytes(), st.P, st.B, R, str(device))
    m = _OUTER.get(key)
    if m is None:
        while len(_OUTER) >= ROUND_KEEP:
            _OUTER.pop(next(iter(_OUTER)))
        m = _OUTER[key] = OuterMachine(st, answer_parts(ops), R, device)
    m.cells(st.cells)
    return m
''')
glue("outer machine from instructions", "_ROUND = {}                      # (instructions, P, B, rows, device) -> RoundMachine\n",
     "")

glue("one-hot machine from instructions", '''    def run_ops(self, regs, c, h, k, ops, p=0):
        """run on a program's instructions (trace's opcodes: a routine between its pointer's save and restore): the
        same steps in the same order."""
        st, saved = [list(regs), c, h, k, p], [0] * MAX_DEPTH
        for o in ops:
            if o >= SAVE:
                d, out = divmod(o - SAVE, 2)
                if out:
                    st[4] = saved[d]
                else:
                    saved[d] = st[4]
            else:
                self._step(st, PRIMS[o])
        return st
''', '''    def run_ops(self, regs, c, h, k, ops, p=0):
        """run on a program's instructions (answer_trace's opcodes: a routine between its pointer's save and restore, a
        C block between CBEG and CEND, a count loop's body between CLOOP + r and CLEND): the same steps in the same
        order."""
        st = [list(regs), c, h, k, p]
        self._ops(st, list(ops), [0] * MAX_DEPTH)
        return st

    def _ops(self, st, ops, saved):
        i, blocks = 0, []
        while i < len(ops):
            o = ops[i]
            if CLOOP <= o < CLEND:                  # a count loop: its body is the instructions up to CLEND
                j = ops.index(CLEND, i)
                self._count(st, o - CLOOP, lambda new, body=ops[i + 1:j]: self._ops(new, body, saved))
                i = j
            elif o == CBEG:                         # a C block: the values at its start (its borrow decides)
                blocks.append((st[0], st[1], st[2], st[3]))
            elif o == CEND:
                self._unless(st, blocks.pop())
            elif o >= SAVE:
                d, out = divmod(o - SAVE, 2)
                if out:
                    st[4] = saved[d]
                else:
                    saved[d] = st[4]
            else:
                self._step(st, PRIMS[o])
            i += 1
''')
glue("one-hot machine from instructions", '''    def _count(self, st, it, lib):
        """A count loop: the count (register r as the loop begins) counted down by the subtraction cell, place by
        place; each round the rows whose count down did not borrow run the body (as a routine, from the loop's
        pointer) and take its state, the others keep theirs (a straight-through blend, as stage 1's countdown); rows
        whose count is above LOOP_CAP end cleared."""
        _, kind, body = it
        C, p, B = st[0][RI[kind[2:]]], st[4], self.B
''', '''    def _count(self, st, r, body):
        """A count loop on register r (body: runs the loop's body on a state): the count (r as the loop begins) counted
        down by the subtraction cell, place by place; each round the rows whose count down did not borrow run the body
        (as a routine, from the loop's pointer) and take its state, the others keep theirs (a straight-through blend,
        as stage 1's countdown); rows whose count is above LOOP_CAP end cleared."""
        C, p, B = st[0][r], st[4], self.B
''')
glue("one-hot machine from instructions", '''            new = [list(st[0]), st[1], st[2], st[3], p]
            self._items(new, body, lib)
''', '''            new = [list(st[0]), st[1], st[2], st[3], p]
            body(new)
''')
glue("one-hot machine from instructions", '''            if is_count(it):
                self._count(st, it, lib)
            elif is_loop(it):
                _, kind, body = it
                if kind in lib:
                    self._items(st, fill(lib[kind], body[0]), lib)
                elif kind == "C":                    # each value the body changed: the new where the borrow was set
                    old = (st[0], st[1], st[2], st[3])                     # at the start, else the old (soft: by the
                    self._items(st, body, lib)                             # borrow's one-hot, as SEL.S's select)
                    on = old[1][:, 1]
                    mix = lambda new, prev: new if new is prev else self.damp(
                        on.view(-1, *[1] * (new.dim() - 1)) * new + (1 - on).view(-1, *[1] * (new.dim() - 1)) * prev)
                    st[0] = [mix(r, q) for r, q in zip(st[0], old[0])]
                    st[1], st[2], st[3] = (mix(st[i], old[i]) for i in (1, 2, 3))
''', '''            if is_count(it):
                self._count(st, RI[it[1][2:]], lambda new, body=it[2]: self._items(new, body, lib))
            elif is_loop(it):
                _, kind, body = it
                if kind in lib:
                    self._items(st, fill(lib[kind], body[0]), lib)
                elif kind == "C":                    # each value the body changed: the new where the borrow was set
                    old = (st[0], st[1], st[2], st[3])                     # at the start, else the old (_unless)
                    self._items(st, body, lib)
                    self._unless(st, old)
''')
glue("one-hot machine from instructions", '''    @staticmethod
    def _joint(a, flags):''', '''    def _unless(self, st, old):
        """A C block's end: each value the body changed (st), the new where the borrow was set at the block's start
        (old's), else the old; soft, by the borrow's one-hot, as SEL.S's select."""
        on = old[1][:, 1]
        mix = lambda new, prev: new if new is prev else self.damp(
            on.view(-1, *[1] * (new.dim() - 1)) * new + (1 - on).view(-1, *[1] * (new.dim() - 1)) * prev)
        st[0] = [mix(r, q) for r, q in zip(st[0], old[0])]
        st[1], st[2], st[3] = (mix(st[i], old[i]) for i in (1, 2, 3))

    @staticmethod
    def _joint(a, flags):''')

glue("the answer program's acceptance counts its work", '''        def ok(p):
            A = outer_machine(st, p, lib, Xd.shape[0], self.device).run(Xd, Nd, Cd, gen_on(self.device, 1))
            return bool((A[:, :Yd.shape[1]] == Yd).all())
''', '''        def ok(p):                              # (its work counted as the cost model counts instructions run alone)
            m = outer_machine(st, p, lib, Xd.shape[0], self.device)
            A = m.run(Xd, Nd, Cd, gen_on(self.device, 1))
            WORK[0] += m.ran * (PRIM_STEP + Xd.shape[0] * PRIM_ROW)
            return bool((A[:, :Yd.shape[1]] == Yd).all())
''')

glue("count rows as buffers", '''        self.register_buffer("round_src", torch.zeros(1, dtype=torch.uint8))              # ... and text
''', '''        self.register_buffer("round_src", torch.zeros(1, dtype=torch.uint8))              # ... and text
        self.register_buffer("count_rows", torch.zeros(10 ** K, COUNT_EACH, 4, W, dtype=torch.long))  # answer rows
        self.register_buffer("count_n", torch.zeros(10 ** K, dtype=torch.long))  # (N, x, count, y) by count (keep_counts)
''')
glue("count rows as buffers", '''            Cs = aux["Cd"][sel]
            for key in {tuple(r) for r in Cs.tolist()}:
                idx = sel[(Cs == torch.tensor(key, device=Cs.device)).all(1)]
                new = (aux["Nd"][idx], aux["Xd"][idx], aux["Cd"][idx], yd[idx])
                old = ctl.count_pool.get(key)
                ctl.count_pool[key] = new if old is None else tuple(torch.cat([a, b]) for a, b in zip(old, new))
            share = min(COUNT_EACH, max(1, COUNT_POOL // len(ctl.count_pool)))
            for key, v in ctl.count_pool.items():
                ctl.count_pool[key] = tuple(t[-share:] for t in v)
''', '''            Cs = aux["Cd"][sel]
            slot = (Cs * 10 ** torch.arange(K - 1, -1, -1, device=Cs.device)).sum(1)    # a count's slot (by its digits)
            new = torch.stack([t.long() for t in (aux["Nd"][sel], aux["Xd"][sel], F.pad(Cs, (0, W - K)), yd[sel])], 1)
            rows = {s: torch.cat([self.count_rows[s, :int(self.count_n[s])], new[slot == s]])
                    for s in torch.unique(slot).tolist()}
            keys = sorted(set(rows) | set(torch.nonzero(self.count_n).flatten().tolist()))
            share = min(COUNT_EACH, max(1, COUNT_POOL // len(keys)))
            for s in keys:
                r = rows.get(s, self.count_rows[s, :int(self.count_n[s])])[-share:].clone()
                self.count_rows[s, :r.shape[0]] = r
                self.count_n[s] = r.shape[0]

    def counts_kept(self) -> dict:
        """The count rows kept (buffers): how many of each count, the count written as T is (most significant first)."""
        return {"".join(str(s // 10 ** (K - 1 - j) % 10) for j in reversed(range(K))): int(self.count_n[s])
                for s in torch.nonzero(self.count_n).flatten().tolist()}
''')
glue("count rows as buffers", '''        keys = sorted(ctl.count_pool)
''', '''        keys = torch.nonzero(self.count_n).flatten().tolist()              # the counts kept (buffers), in order
''')
glue("count rows as buffers", '''        rows = tuple(torch.cat([ctl.count_pool[k][i] for k in keys]) for i in range(4))
        progs, lib, rec = ctl.climb.search_outer(rows, rounds, stop_at, OUTER_STATES, OUTER_COST, OUTER_BODY,
                                                 OUTER_NP)
''', '''        R = torch.cat([self.count_rows[s, :int(self.count_n[s])] for s in keys])
        rows = (R[:, 0].contiguous(), R[:, 1].contiguous(), R[:, 2, :K].contiguous(), R[:, 3].contiguous())
        w0 = WORK[0]
        progs, lib, rec = ctl.climb.search_outer(rows, rounds, stop_at, OUTER_STATES, OUTER_COST, OUTER_BODY,
                                                 OUTER_NP)
''')
glue("answer programs into the buffers", '''        print(f"LR outer step={ctl.step} {line}", flush=True)
        if progs:
            ctl.lib = lib
            ctl.progs = [tuple(p) for p in sorted(progs, key=lambda p: (cost(expand(p, lib)), show(p)))[:PROG_SLOTS]]
            ctl.phase, ctl.in_phase = "D", 0
            ctl.history.append((ctl.step, "answer_program", len(ctl.progs)))
            print(f"LR round_found step={ctl.step} programs: {[show(p) for p in ctl.progs]} rounds: "
                  f"{ {f'T.round{i + 1}': show(p) for i, p in enumerate(rounds)} } library: "
                  f"{ {k: show(v) for k, v in ctl.lib.items()} }", flush=True)
''', '''        print(f"LR outer step={ctl.step} {line} [work +{WORK[0] - w0:.4g}, {WORK[0] - ctl.climb_work0:.4g} in all]",
              flush=True)
        if progs:
            progs = [tuple(p) for p in sorted(progs, key=lambda p: (cost(expand(p, lib)), show(p)))[:PROG_SLOTS]]
            self.set_round(lib, progs)
            ctl.phase, ctl.in_phase = "D", 0
            ctl.history.append((ctl.step, "answer_program", len(progs)))
            print(f"LR round_found step={ctl.step} programs: {[show(p) for p in progs]} rounds: "
                  f"{ {f'T.round{i + 1}': show(p) for i, p in enumerate(rounds)} } library: "
                  f"{ {k: show(v) for k, v in lib.items()} }", flush=True)
''')
glue("count rows as buffers", '''f"counts={ {''.join(map(str, k[::-1])): v[0].shape[0] for k, v in sorted(ctl.count_pool.items())} } "''',
     '''f"counts={self.counts_kept()} "''')
GONE = MS.GONE + [r"ctl\.count_pool", r"round_machine\b"]

# Dead once the pieces are together (no code names them: asserted), dropped for room under 256 KiB, as make_ll.py and
# make_g.py dropped the inlined core's dead code: sub_soft (the countdown's and the late lessons' soft subtraction) and
# bfs_multi (the round's search run t times, sp2's _round_many).
DEAD = ("sub_soft", "bfs_multi")


def drop_dead(src: str, names: tuple, log: list) -> str:
    """src without the top-level functions names (each, with the two blank lines before it), each asserted unnamed
    anywhere else in src."""
    for name in names:
        tree = ast.parse(src)
        uses = [n for n in ast.walk(tree) if (isinstance(n, ast.Name) and n.id == name)
                or (isinstance(n, ast.Attribute) and n.attr == name)]
        assert not uses, (name, len(uses))
        node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        assert len(node) == 1, name
        lines = src.splitlines(True)
        lo, hi = node[0].lineno - 1, node[0].end_lineno
        assert lines[lo - 2:lo] == ["\n", "\n"], name
        src = "".join(lines[:lo - 2] + lines[hi:])
        log.append((f"dead: {name} dropped", 1))
    return src


DOC = '''"""Stage 3, self-practising, with every variant of sp2 that certified on its own (cpu_first/integration/combined_sp2/
COMBINED.md; make_combined.py writes it from the certified files): sp2 (SELF_PRACTICE.md) with
  * the loop over T learned (LEARNED_LOOP.md): no countdown, no gate, no loop law; the count loop C.r[body]; T in S;
    once the round is found, the answer's program searched (OUTER_*);
  * generic instructions (GENERIC_OPS.md): an if block C[...] for SEL.S and shifts through k for BRG.X (COND, SHIFT,
    BRING), with g3's search rules (SP_ALIAS_FIT, SP_CALLS_FIRST, SP_DEEP_*, SP_PURE_FIRST, SP_VERIFY, ROUND_STATES);
  * the law book generated by a grammar (SP_ABLATION.md 5.2: PATTERNS, LAW_BOOK, Draw, law_target);
  * the two weak checks fixed (ABLATION_S3.md 8: FULL_CHECK, ABOVE_CHECK);
  * the strict build (STRICT.md): budgets of machine work, never the clock (SEARCH_WORK, CLIMB_WORK); the kept rows
    and the answer programs in buffers; Adam's own reset; no late lessons, no wrap check.
Phases A-D, the cells, the climb and the prediction path are sp2's as LEARNED_LOOP.md describes them.
"""
'''


# ================================================================================================== the build
def undefined_globals(src: str) -> set:
    """Names a function reads as globals that neither the module's top level nor builtins define (symtable)."""
    top = symtable.symtable(src, "combined", "exec")
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


def build(sets: dict | None = None) -> tuple[str, list, list]:
    """The combined file's text, the log of every edit group applied (what, how many) and the text after each stage
    (for --diff)."""
    sp2 = read(*SP2)
    texts = {k: read(*v) for k, v in CERTIFIED.items()}
    replay()
    subs = {"@G3_ROUND@": g3_round(texts["g3"].splitlines(True)), "@DOC@": DOC}
    src, log = merge_pieces(sp2, {k: texts[k] for k in ("ll", "g3", "book")}, subs)
    stages = [("sp2", sp2), ("ll,g3,book", src)]
    # the fix: make_fix_variants.py's edits for book_gen (the generated book's Draw and law_target; sp2's other edits)
    for old, new in MFV.BOOK_GEN:
        n = src.count(old)
        assert n == 1, ("fix", n, old[:80])
        src = src.replace(old, new)
    log.append(("fix: make_fix_variants.BOOK_GEN's edits", len(MFV.BOOK_GEN)))
    stages.append(("fix", src))
    src = apply_strict(src, log)
    stages.append(("strict", src))
    src = MS.apply(src, [(f"glue: {w}", o, n, c) for w, o, n, c in GLUE], log)
    src = drop_dead(src, DEAD, log)
    stages.append(("glue", src))
    for name, value in (sets or {}).items():                  # research builds only
        src, n = re.subn(rf"^{name} = .*$", f"{name} = {value}", src, count=1, flags=re.M)
        assert n == 1, name
    for pat in GONE:                                          # nothing of the clock, the countdown or the Python-side
        left = [l for l in src.splitlines() if re.search(pat, l)]          # rows and programs is left
        assert not left, (pat, left[:5])
    assert "#@" not in src and "@SEARCH_WORK@" not in src and "@CLIMB_WORK@" not in src and "@DOC@" not in src
    assert src.count("time.monotonic()") == 1 and src.count("_clock()") >= 5
    ast.parse(src)
    compile(src, OUT.name, "exec")
    undef = undefined_globals(src) - {"big", "name", "self"}  # (symtable's false alarm on book_gen's Draw.top, as in
    assert not undef, undef                                   #  book_gen.py itself)
    assert len(src.encode()) <= LIMIT, f"{len(src.encode()):,} bytes, over the evaluator's 256 KiB"
    return src, log, stages


def main(argv):
    sets = dict(a.split("=", 1) for a in argv if "=" in a and not a.startswith("--"))
    src, log, stages = build(sets)
    if argv[:1] == ["--diff"]:
        names = [n for n, _ in stages]
        i = names.index(argv[1])
        a, b = stages[i - 1], stages[i]
        sys.stdout.writelines(difflib.unified_diff(a[1].splitlines(True), b[1].splitlines(True), a[0], b[0], n=1))
        return
    for what, n in log:
        print(f"{n:4d}  {what}")
    out = OUT if not sets else HERE / "research" / ("submission_stage3_combined_" + "_".join(
        f"{k}-{v}" for k, v in sets.items()).replace("/", "") + ".py")
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
    if out == OUT and OUT.exists() and OUT.read_text() != src and "--force" not in argv:
        raise SystemExit(f"{OUT} differs from this build (pass --force to replace it)")
    out.parent.mkdir(exist_ok=True)
    out.write_text(src)
    print(f"wrote {out}: {len(src.splitlines())} lines, {len(src.encode()):,} bytes, md5 {md5(src)}")


if __name__ == "__main__":
    main(sys.argv[1:])
