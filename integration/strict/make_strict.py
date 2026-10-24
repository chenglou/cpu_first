"""Builds the strict files (STRICT.md): a certified stage-3 file with the gray points of REVIEW.md §5 that can go
without changing what is learned taken out, as exact edits of the certified source.

    ~/github/sotaku/venv/bin/python integration/strict/make_strict.py [sp2|s3i ...] [NAME=VALUE ...] [--keep-wrap] [--r1] [--out DIR]

(--r1: round 1's builds, without budgets, with --keep-wrap the gate's wrap check kept: they land in r1/ by default.)

Each edit names what it is for; each must match the certified text exactly as many times as it says, or the build
stops. The certified files are only read (their md5s are asserted). What changes:
  clock    every search and the climb stop on budgets of machine work (WORK, counted by the cost model apply_items
           already uses to pick its path), never on the wall clock or the evaluator's time budget; the clock is read
           for the logs' seconds only (_clock)
  pools    the rows kept across batches (the practice problems' read rows, the round's answer rows) are registered
           buffers, written in place in training
  round    the round programs are registered buffers: their instructions as the machines run them (round_ops,
           round_len) and their text (round_src); the forward pass and the training's hard machine run them from
           there (no Python-side library or program list)
  adam     the optimizer resets its own moments after the step that switches to phase B (the scheduler used to)
  late     the late table lessons are gone (ABLATION_S3.md: not needed; the table's checks, which gate, stay)
  wrap     the gate's check of the register-width fact the late lessons taught is gone too (--keep-wrap keeps it)
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTEG = HERE.parent
CERT = {"sp2": (INTEG / "self_practice" / "submission_stage3_sp2.py", "afc70fe91069a85c416eba18b48ee30e"),
        "s3i": (INTEG / "submission_stage3_s3i.py", "0e8131440e1842546c67e7378caf472d")}
OUT = {"sp2": "submission_stage3_sp2_strict.py", "s3i": "submission_stage3_s3i_strict.py"}
# the work budgets (WORK's units; 0: none), set from round 1's work (STRICT.md §2): SEARCH_WORK is twice the largest
# search seen (sp2 seed 3's, 3.5e8) and takes 660-970 s on the H100s measured, about the 900 s it replaces;
# CLIMB_WORK is 1.22 times the largest climb seen (sp2 seed 3's, 1.81e9), at most about 3,060 s of climb at the
# slowest H100 rate seen (1.39 s per 1e6 units)
BUDGETS = {"SEARCH_WORK": "700_000_000", "CLIMB_WORK": "2_200_000_000"}
R1_BUDGETS = {"SEARCH_WORK": "0", "CLIMB_WORK": "0"}       # round 1: no budgets (the work only counted, STRICT.md §4)

STRICT_DOC = '''

The strict build (cpu_first/integration/strict/STRICT.md; strict/make_strict.py writes it from the certified file):
  * no clock: each search and the climb stop on budgets of machine work (WORK: the cost model apply_items already
    uses to choose its path, a count that is the same on every host), never on the time or the evaluator's budget;
  * the rows kept across batches (the practice problems' and the round's answers) and the round programs (their
    instructions as the machines run them, and their text) are registered buffers: in the state dict, counted, on
    the device; the forward pass runs the programs from there;
  * the optimizer starts its own moments afresh at the switch to phase B (the scheduler only sets the learning rate);
  * no late table lessons (ABLATION_S3.md: not needed; the table's checks stay).'''
NOWRAP_DOC = '''
  * the table's gate has no register-width fact ("a wrapped x - N is large"): the check is gone with the lesson.'''

# ------------------------------------------------------------------------------------------------------ the library
LIBCORE = [
    ("clock", '''class Deadline(Exception):
    """The training's deadline passed inside a climb unit (the unit returns at once; nothing is cached)."""


def _late(deadline):
    return deadline is not None and time.time() > deadline
''', '''class OutOfWork(Exception):
    """A work budget ran out inside a climb unit (the unit returns at once; nothing is cached)."""


WORK = [0.0]       # the machine work done so far, in the units of apply_items' cost model (LOCK_STEP and the rest):
                   # what every search's and the climb's budgets count, the same count on every host


def _spent(stop_at):
    """Has the work passed stop_at (a WORK value; None: no budget)?"""
    return stop_at is not None and WORK[0] > stop_at


def _clock():
    """The wall clock, for the logs' seconds only: nothing the model computes reads it."""
    return time.monotonic()
''', 1),
    ("clock", '''def step(st, b, tok):
    """One primitive on batch b in place."""
    cl, B = st.cells, st.B
''', '''def step(st, b, tok):
    """One primitive on batch b in place (its work counted as the cost model counts an instruction run alone)."""
    WORK[0] += PRIM_STEP + b.S * PRIM_ROW
    cl, B = st.cells, st.B
''', 1),
    ("clock", '''    lock = {j for j in range(1, nb) if per[j] and LOCK_COST + mx[j] * (LOCK_STEP + len(per[j]) * rows_per * LOCK_ROW)
            < sum(lens[i] * PRIM_STEP + groups[i] * rows_per * PRIM_ROW + GROUP_FIX for i in per[j])}
    if b.S == 0 or not lock:
        return _apply_grouped(st, b, item_of_row, items, lib)
''', '''    lc = [mx[j] * (LOCK_STEP + len(per[j]) * rows_per * LOCK_ROW) for j in range(nb)]
    lock = {j for j in range(1, nb) if per[j] and LOCK_COST + lc[j]
            < sum(lens[i] * PRIM_STEP + groups[i] * rows_per * PRIM_ROW + GROUP_FIX for i in per[j])}
    if b.S == 0 or not lock:
        return _apply_grouped(st, b, item_of_row, items, lib)
    WORK[0] += sum(lc[j] for j in lock)              # the locksteps' work (the other buckets' is counted by step)
''', 1),
    ("round", '''def round_machine(st, prog, lib, rows, device):
    """prog's RoundMachine at setting st (its cells loaded) for rows machines (rounded up to a multiple of ROUND_ROWS),
    made once per instruction sequence and size and kept (ROUND_KEEP at most)."""
    ops, _ = _trace_seq(st, prog, lib, 0, _traces(st, lib))
    R = -(-max(rows, 1) // ROUND_ROWS) * ROUND_ROWS
''', '''def round_machine(st, prog, lib, rows, device):
    """prog's RoundMachine at setting st (its cells loaded) for rows machines (rounded up to a multiple of ROUND_ROWS),
    made once per instruction sequence and size and kept (ROUND_KEEP at most)."""
    return round_machine_ops(st, _trace_seq(st, prog, lib, 0, _traces(st, lib))[0], rows, device)


def round_machine_ops(st, ops, rows, device):
    """round_machine for a program given as its instructions (trace's opcodes, array('h'))."""
    R = -(-max(rows, 1) // ROUND_ROWS) * ROUND_ROWS
''', 1),
    ("round", '''    def run(self, regs, c, h, k, items, lib, p=0):
        st = [list(regs), c, h, k, p]
        self._items(st, items, lib)
        return st
''', '''    def run(self, regs, c, h, k, items, lib, p=0):
        st = [list(regs), c, h, k, p]
        self._items(st, items, lib)
        return st

    def run_ops(self, regs, c, h, k, ops, p=0):
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
''', 1),
    # the clock's words in docstrings and stop messages (the identifiers are renamed below)
    ("clock", "Raises Deadline if the deadline passes.", "Raises OutOfWork past stop_at.", 1),
    ("clock", "(that level is finished\n    streaming goal states only), or at the deadline.",
     "(that level is finished\n    streaming goal states only), or past stop_at.", 1),
    ("clock", 'f"deadline at level {d}"', 'f"work budget at level {d}"', None),
    ("clock", '"deadline (building the vocabulary)"', '"work budget (building the vocabulary)"', None),
    ("clock", "# the deadline passed while a vocabulary was being built",
     "# the work budget ran out while a vocabulary was being built", 1),
    ("clock", "      search_s: one search's time limit (its stop counts as a used-up cap);",
     "      search_work: one search's work budget (its stop counts as a used-up cap);", 1),
    # libcore.Climb: one search's budget, and its records' work
    ("clock", '''    def search(self, task, rows, deadline=None):
        t0 = time.time()
        climb_deadline = deadline
        if self.search_s:
            deadline = min(deadline or float("inf"), t0 + self.search_s)
''', '''    def search(self, task, rows, stop_at=None):
        t0, w0 = _clock(), WORK[0]
        climb_stop_at = stop_at
        if self.search_work:
            stop_at = min(stop_at or float("inf"), WORK[0] + self.search_work)
''', 1),
    ("clock", '''        def run(how, classes, lib, cap):
            t1 = time.time()
            r = bfs(prob, classes, lib, max_cost=self.max_cost, max_states=cap,
                    accept=ok.many if self.listing == "goals" else (lambda ps: [ok(p) for p in ps]),
                    deadline=deadline, chunk=self.chunk, listing=self.listing, fuse=self.fuse)
            runs.append(dict(how=how, found_cost=r.found_cost, level_sizes=r.level_sizes, stop=r.stop,
                             seconds=round(time.time() - t1, 1)))
''', '''        def run(how, classes, lib, cap):
            t1, w1 = _clock(), WORK[0]
            r = bfs(prob, classes, lib, max_cost=self.max_cost, max_states=cap,
                    accept=ok.many if self.listing == "goals" else (lambda ps: [ok(p) for p in ps]),
                    stop_at=stop_at, chunk=self.chunk, listing=self.listing, fuse=self.fuse)
            runs.append(dict(how=how, found_cost=r.found_cost, level_sizes=r.level_sizes, stop=r.stop,
                             seconds=round(_clock() - t1, 1), work=round(WORK[0] - w1)))
''', 1),
    ("clock", "        if res.found_cost is None and not (climb_deadline is not None and time.time() > climb_deadline):",
     "        if res.found_cost is None and not _spent(climb_stop_at):", 1),
    ("clock", "                   seconds=round(time.time() - t0, 1), stop=res.stop, how=how,",
     "                   seconds=round(_clock() - t0, 1), work=round(WORK[0] - w0), stop=res.stop, how=how,", 1),
    ("clock", '''        t0 = time.time()
        routines = [k for k in self.lib if k.startswith("T.")]
        prob = round_problem(self.cells, answers, self.NP, self.P_round, self.device)
''', '''        t0, w0 = _clock(), WORK[0]
        routines = [k for k in self.lib if k.startswith("T.")]
        prob = round_problem(self.cells, answers, self.NP, self.P_round, self.device)
''', 1),
    ("clock", "                   seconds=round(time.time() - t0, 1), stop=res.stop, routines=len(routines),",
     "                   seconds=round(_clock() - t0, 1), work=round(WORK[0] - w0), stop=res.stop, routines=len(routines),",
     1),
]

# ------------------------------------------------------------------------------------- the self-practising climb
SPCORE = [
    ("clock", '''        t0 = time.time()
        dl = deadline
        if self.search_s:
            dl = min(deadline or float("inf"), t0 + self.search_s)
''', '''        t0 = _clock()
        dl = stop_at
        if self.search_work:
            dl = min(stop_at or float("inf"), WORK[0] + self.search_work)
''', 1),
    ("clock", 'min(deadline or float("inf"), time.time() + self.search_s)',
     'min(stop_at or float("inf"), WORK[0] + self.search_work)', 3),
    ("clock", '''            """members: [(root, [goal indices])]."""
            t1 = time.time()
''', '''            """members: [(root, [goal indices])]."""
            t1, w1 = _clock(), WORK[0]
''', 1),
    ("clock", '''            runs.append(dict(how=how, roots=[r.key for r, _ in members], seconds=round(time.time() - t1, 1),
''', '''            runs.append(dict(how=how, roots=[r.key for r, _ in members], seconds=round(_clock() - t1, 1),
                             work=round(WORK[0] - w1),
''', 1),
    ("clock", "            comp = self.comp_classes(deadline)          #  training's deadline; one search's limit starts after)",
     "            comp = self.comp_classes(stop_at)           #  climb's budget; one search's budget starts after)", 1),
    ("clock", '''        t0 = time.time()
        lib = dict(self.lib, **(extra or {}))
''', '''        t0, w0 = _clock(), WORK[0]
        lib = dict(self.lib, **(extra or {}))
''', 1),
    ("clock", "                   seconds=round(time.time() - t0, 1), stop=stop, routines=len(routines),",
     "                   seconds=round(_clock() - t0, 1), work=round(WORK[0] - w0), stop=stop, routines=len(routines),",
     1),
]

# --------------------------------------------------------------------------------------------- the model (template)
OLD_TABLE_LESSONS_HEAD = '''    def table_lessons(self, Nd, Xd, Cd, terms, orders, nonzero) -> tuple[Tensor, float]:'''
OLD_TABLE_LESSONS_TAIL = '''        return loss, table

    def cell_lessons(self, Cd: Tensor) -> Tensor:'''
NEW_TABLE_LESSONS = '''    def table_lessons(self, Nd, Xd, Cd, terms, orders, nonzero) -> tuple[Tensor, float]:
        """The subtraction table's lessons (general_lessons' kind 1 and 2, as in stage 1, on registers made of the rows'
        x and other rows' x: there is no machine square here): sub.order on every mined order fact, then the digit
        laws (sub.order on the entries the batch's subtractions read, sub.bijective). The strict build has no late
        lessons on the pieces' laws (ABLATION_S3.md's no_late). Returns the loss and the hard checks' share passing
        (sub.inverse with the learned adder a - (0 - b), sub.identity, add.commute, the facts' borrows, sub.wrap,
        sub.zero_minus: what gates the next phase)."""
        ctl = self.ctl
        dev = Xd.device
        Td, Tb, _ = self.tables()
        TDC = self.soft_table()
        X = Xd.shape[0]
        M = F.pad(Nd, (0, P - W))
        hot = lambda v: F.one_hot(v, 10).float()
        log = lambda p: p.clamp_min(1e-12).log()
        with torch.no_grad():
            Rg = torch.stack([F.pad(Xd.roll(r, 0), (0, P - W)) for r in range(W)], 1).reshape(X * W, P)
            MW = M.repeat_interleave(W, 0)
            links = [Rg]
            for _ in range(8):                                               # each register minus N again and again
                links.append(sub_hard(links[-1], MW, Td, Tb)[0])
            NL = W * 9
            D = torch.stack(links, 1).reshape(X * NL, P)
            Mr = M.repeat_interleave(NL, 0)
            Z = torch.zeros_like(M)
            R2 = Rg.view(X, W, P).roll(1, 0).reshape(X * W, P)
        facts = bool(orders)
        digits_on = ctl.gated_on is not None
        zero = torch.zeros_like(Rg)
        loss = self.zero()
        for a_, b_ in orders:
            A_, B_ = terms[a_], terms[b_]
            for j in range(P):
                differ = (A_[:, j:] != B_[:, j:]).any(1)
                p1 = link_soft(hot(A_[:, j:]), B_[:, j:], TDC)[1][:, -1, 1]
                p2 = link_soft(hot(B_[:, j:]), A_[:, j:], TDC)[1][:, -1, 1]
                loss = loss - (torch.where(differ, log(p1), log(1 - p1)).mean() + log(1 - p2).mean()) / P
        if facts:
            with torch.no_grad():
                okf = torch.ones(X, dtype=torch.bool, device=dev)
                for a_, b_ in orders:
                    okf = okf & (sub_hard(terms[a_], terms[b_], Td, Tb)[1] == 1) & \\
                        (sub_hard(terms[b_], terms[a_], Td, Tb)[1] == 0)
                ctl.facts_streak = ctl.facts_streak + 1 if bool(okf.all()) else 0
        if digits_on:
            with torch.no_grad():
                used = torch.bincount(sub_hard(D, Mr, Td, Tb)[2].flatten(), minlength=200) > 0
                E2, _, i2 = sub_hard(Rg, R2, Td, Tb)
                C2, _, i3 = sub_hard(zero, R2, Td, Tb)
                i4 = sub_hard(E2, C2, Td, Tb)[2]
                for ii in (i2, i3, i4):
                    used = used | (torch.bincount(ii.flatten(), minlength=200) > 0)
            pa = self.a.float().softmax(-1)
            pd = pa.reshape(200, 10)
            for lo, hi in order_pairs(Tb, used):
                loss = loss - log((pd[lo] * (1 - pd[hi].cumsum(-1))).sum(-1)).mean()
            loss = loss - log(10 * pa.mean(0)).mean() - log(10 * pa.mean(1)).mean()
        if facts and not digits_on and ctl.facts_streak >= GATE_WIN:
            ctl.gated_on = ctl.step
            ctl.history.append((ctl.step, "order_digits"))
            print(f"LR order_digits step={ctl.step}", flush=True)
        checks = [(Rg, zero, Rg, 0), (Rg, Rg, zero, 0)]
        for a_, b_ in orders:
            checks.append((terms[a_], terms[b_], None, 1))
            checks.append((terms[b_], terms[a_], None, 0))
        for a_ in nonzero:
            checks.append((torch.zeros_like(terms[a_]), terms[a_], None, 1))
'''
# the hard checks' block is kept as it is: it starts here in the certified file
TABLE_CHECKS_START = '''        with torch.no_grad():
            Eh, beh, idx = sub_hard(D, Mr, Td, Tb)
'''

MODEL = [
    ("late", '''ID_PLACE0 = 5.0        # weight of the lowest place in the check (it tells the swapped code)
ID_BORROW = 1.0        # weight of the final borrow in the check: the complete identity
GATE_WIN = 10          # the digit lessons start once the borrow facts hold on every row of this many batches in a row
LATE_AFTER = 100       # the pieces' own table lessons join this many steps after the digit lessons start
STE_CIN = "anchors"
ANCHOR_ZERO = ANCHOR_SELF = ANCHOR_NEG2 = ANCHOR_COMM = ANCHOR_LESS = ANCHOR_MORE = ANCHOR_WRAP = ANCHOR_NEG = 1.0
''', '''GATE_WIN = 10          # the digit lessons start once the borrow facts hold on every row of this many batches in a row
''', 1),
    ("pools", '''class Control:
    """The training schedule (Python-side, not model state): the reader's stages, the phases A-D and frozen, the
    library's state, the audits' counters."""
''', '''def lib_names(lib: dict, items) -> set:
    """The library names items use (routines, merges, loop patterns), through their definitions."""
    out, todo = set(), list(items)
    while todo:
        it = todo.pop()
        if is_loop(it):
            todo.extend(it[2])
            it = it[1]
        if it in lib and it not in out:
            out.add(it)
            todo.extend(lib[it])
    return out


class Control:
    """The training schedule (Python-side, as a scheduler's state is): the reader's stages, the phases A-D and frozen,
    the climb (the search's working state), the audits' counters. The rows kept across batches and the round programs
    are the model's buffers (LibraryMachine)."""
''', 1),
    ("pools", '''        self.absp = nn.Parameter(INIT * torch.randn(4, W, self.LPM))
        self.ctl = Control()
''', '''        self.absp = nn.Parameter(INIT * torch.randn(4, W, self.LPM))
        # the model's state besides its parameters (registered buffers: in the state dict, counted, on the device)
        self.register_buffer("pool_rows", torch.zeros(POOL, 2, W, dtype=torch.long))      # read rows (N, x digits)
        self.register_buffer("pool_sizes", torch.zeros(POOL, dtype=torch.long))           # for the practice problems
        self.register_buffer("ans_rows", torch.zeros(ANS_POOL, 3, W, dtype=torch.long))   # answer rows (N, x, y)
        self.register_buffer("ans_sizes", torch.zeros(ANS_POOL, dtype=torch.long))        # for the round's search
#@ANS_T@
        self.register_buffer("round_ops", torch.full((PROG_SLOTS, 1), NOP, dtype=torch.int16))   # the round programs:
        self.register_buffer("round_len", torch.zeros(PROG_SLOTS, dtype=torch.long))      # instructions (set_round)
        self.register_buffer("round_src", torch.zeros(1, dtype=torch.uint8))              # ... and text
        self.ctl = Control()
''', 1),
    ("round", "        round programs and their library are given by the check (ctl.progs, ctl.lib).\"\"\"",
     "        round programs are given by the check (set_round).\"\"\"", 1),
    ("round", '''    def program(self, i: int | None = None):
        """Round program i (default: the tournament's argmax), or None before the round is found."""
        if not self.ctl.progs:
            return None
        n = len(self.ctl.progs)
        return self.ctl.progs[int(self.prog[:n].argmax()) if i is None else i]

    def prog_names(self) -> str:
        n = len(self.ctl.progs)
        if not n:
            return "none"
        pr = self.prog[:n].float().softmax(-1).tolist()
        return " | ".join(f"{show(p)} ({w:.3f})" for p, w in zip(self.ctl.progs, pr))
''', '''    # ------------------------------------------------------------------ the buffers: rows kept, the round programs
    @staticmethod
    @torch.no_grad()
    def keep(rows: Tensor, sizes: Tensor, new: Tensor) -> None:
        """A batch's rows (new [n, fields, W]) join a buffer of whole batches, the newest last, at most rows.shape[0]
        rows in all (the oldest batches leave first): rows [cap, fields, W] and sizes [cap] (each kept batch's row
        count, the oldest first, then 0)."""
        if new.shape[0] == 0:
            return
        sz = [s for s in sizes.tolist() if s]
        old = sum(sz)
        sz.append(new.shape[0])
        drop = 0
        while drop < len(sz) and sum(sz[drop:]) > rows.shape[0]:
            drop += 1
        if drop < len(sz):
            body = torch.cat([rows[sum(sz[:drop]):old], new.to(rows.dtype)])
            rows[:body.shape[0]] = body
        sizes.zero_()
        sizes[:len(sz) - drop] = torch.tensor(sz[drop:], dtype=sizes.dtype, device=sizes.device)

    @staticmethod
    def kept(rows: Tensor, sizes: Tensor):
        """The rows a buffer keeps, the oldest first ([n, fields, W], a copy), or None when it is empty."""
        n = int(sizes.sum())
        return rows[:n].clone() if n else None

    @torch.no_grad()
    def set_round(self, lib: dict, progs: list) -> None:
        """The round programs into the buffers: each one's instructions as both machines run them (trace, at the round
        machine's size) in round_ops and round_len, and their text with the definitions they use in round_src."""
        progs = [tuple(p) for p in progs[:PROG_SLOTS]]
        st = round_settings(None, PR)
        ops = [_trace_seq(st, p, lib, 0, _traces(st, lib))[0] for p in progs]
        buf = torch.full((PROG_SLOTS, max(len(o) for o in ops)), NOP, dtype=torch.int16)
        for i, o in enumerate(ops):
            buf[i, :len(o)] = torch.tensor(o.tolist(), dtype=torch.int16)
        dev = self.prog.device
        self.round_ops = buf.to(dev)
        self.round_len = torch.tensor([len(o) for o in ops] + [0] * (PROG_SLOTS - len(ops)), device=dev)
        used = lib_names(lib, [it for p in progs for it in p])
        text = "\\n".join([f"{k}={show(v)}" for k, v in lib.items() if k in used] + ["#" + show(p) for p in progs])
        self.round_src = torch.tensor(list(text.encode()), dtype=torch.uint8, device=dev)

    def round_text(self):
        """The round programs as written and the definitions they use, read back from round_src: (lib, programs)."""
        lib, progs = {}, []
        for line in bytes(self.round_src.tolist()).decode().split("\\n"):
            if line.startswith("#"):
                progs.append(parse(line[1:]))
            elif "=" in line:
                k, v = line.split("=", 1)
                lib[k] = parse(v)
        return lib, progs

    def n_progs(self) -> int:
        return int((self.round_len > 0).sum())

    def round_trace(self, i: int) -> array:
        """Round program i's instructions, read from round_ops (array('h') of trace's opcodes)."""
        return array("h", self.round_ops[i, :int(self.round_len[i])].tolist())

    def program(self, i: int | None = None):
        """Round program i's slot (default: the tournament's argmax), or None before the round is found."""
        n = self.n_progs()
        if not n:
            return None
        return int(self.prog[:n].argmax()) if i is None else i

    def prog_names(self) -> str:
        n = self.n_progs()
        if not n:
            return "none"
        pr = self.prog[:n].float().softmax(-1).tolist()
        return " | ".join(f"{show(p)} ({w:.3f})" for p, w in zip(self.round_text()[1], pr))
''', 1),
    ("round", '''    def round_ste(self, s: Tensor, Nr: Tensor, SBB: Tensor, ADC: Tensor, MUL: Tensor) -> Tensor:
        """One round on one-hot digits: the selected program from x = s in X and N in N (a clean start) on the round
        machine; the new state is A's low W places [X, W, 10]. Straight-through over the tournament's programs: with
        autograd every program runs (the gradient reaches the tournament's logits); without, the argmax program only."""
        ctl = self.ctl
        X = s.shape[0]
''', '''    def round_ste(self, s: Tensor, Nr: Tensor, SBB: Tensor, ADC: Tensor, MUL: Tensor, ops: list) -> Tensor:
        """One round on one-hot digits: the selected program (ops: each program's instructions, round_trace) from
        x = s in X and N in N (a clean start) on the round machine; the new state is A's low W places [X, W, 10].
        Straight-through over the tournament's programs: with autograd every program runs (the gradient reaches the
        tournament's logits); without, the argmax program only."""
        X = s.shape[0]
''', 1),
    ("round", '''        n = len(ctl.progs)
        sel = ste(F.one_hot(self.prog[:n].argmax(), n).float(), self.prog[:n].float().softmax(-1))
        run = lambda i: m.run(regs, z2, z10, z10, ctl.progs[i], ctl.lib)[0][0][:, :W]
''', '''        n = len(ops)
        sel = ste(F.one_hot(self.prog[:n].argmax(), n).float(), self.prog[:n].float().softmax(-1))
        run = lambda i: m.run_ops(regs, z2, z10, z10, ops[i])[0][0][:, :W]
''', 1),
    ("round", '''        s = Sx
        if ctl.progs:
''', '''        s = Sx
        ops = [self.round_trace(i) for i in range(self.n_progs())]     # the round programs, from the buffers
        if ops:
''', 1),
    ("round", "                R = self.round_ste(s, Sn, SBB, ADC, MUL)\n",
     "                R = self.round_ste(s, Sn, SBB, ADC, MUL, ops)\n", 1),
    ("round", "    def run_rounds(self, Sd: Tensor, C: Tensor, Nd: Tensor, prog, cap: int | None = None) -> Tensor:",
     "    def run_rounds(self, Sd: Tensor, C: Tensor, Nd: Tensor, i: int, cap: int | None = None) -> Tensor:", 1),
    ("round", '''        """The countdown and the round program on digits: rounds until no row commits (the snapped subtraction for
        C - 1, the gate on its borrow), each committing row running the program (libcore's machine at PR places, the
        snapped cells, a fresh machine as run_prog makes it); at most `cap` rounds (default 10^K). Every row runs every
        round, with fixed shapes (a row that does not commit keeps its state), on the program's round machine (on a GPU
        one CUDA graph, captured once per program and row count); its dirty starts are run_prog's own draws for the
        committing rows. Returns the state [X, W]."""''', '''        """The countdown and round program i (its instructions from the buffers) on digits: rounds until no row
        commits (the snapped subtraction for C - 1, the gate on its borrow), each committing row running the program
        (libcore's machine at PR places, the snapped cells, a fresh machine as run_prog makes it); at most `cap` rounds
        (default 10^K). Every row runs every round, with fixed shapes (a row that does not commit keeps its state), on
        the program's round machine (on a GPU one CUDA graph, captured once per program and row count); its dirty
        starts are run_prog's own draws for the committing rows. Returns the state [X, W]."""''', 1),
    ("round", "        m = round_machine(st, prog, self.ctl.lib, Sd.shape[0], Sd.device)\n",
     "        m = round_machine_ops(st, self.round_trace(i), Sd.shape[0], Sd.device)\n", 1),
    ("pools", '''        if ctl.phase in ("B", "C", "D") and ctl.io >= 4:                    # the read rows: the practice problems'
            ctl.pool.append((Nd.detach(), Xd.detach()))
            while sum(n.shape[0] for n, _ in ctl.pool) > POOL:
                ctl.pool.pop(0)
''', '''        if ctl.phase in ("B", "C", "D") and ctl.io >= 4:                    # the read rows: the practice problems'
            self.keep(self.pool_rows, self.pool_sizes, torch.stack([Nd, Xd], 1).detach())
''', 1),
    ("round", '''        sel = torch.nonzero(aux["small"]).flatten()
        n = len(ctl.progs)
        if sel.numel() == 0 or n == 0:
            return self.prog.sum() * 0.0
        with torch.no_grad():
            cap = answer_cap(aux["Cd"][sel])
            ok = torch.stack([(self.run_rounds(aux["Xd"][sel], aux["Cd"][sel], aux["Nd"][sel], p, cap) == yd[sel])
                              .all(1) for p in ctl.progs])                   # [programs, rows]
''', '''        sel = torch.nonzero(aux["small"]).flatten()
        n = self.n_progs()
        if sel.numel() == 0 or n == 0:
            return self.prog.sum() * 0.0
        with torch.no_grad():
            cap = answer_cap(aux["Cd"][sel])
            ok = torch.stack([(self.run_rounds(aux["Xd"][sel], aux["Cd"][sel], aux["Nd"][sel], i, cap) == yd[sel])
                              .all(1) for i in range(n)])                    # [programs, rows]
''', 1),
    ("round", "        if rows.numel() == 0 or not ctl.progs:\n", "        if rows.numel() == 0 or not self.n_progs():\n", 1),
    ("pools", '''f"pool={sum(n.shape[0] for n, _ in ctl.pool)} answers={sum(a[0].shape[0] for a in ctl.ans_pool)} "''',
     '''f"pool={int(self.pool_sizes.sum())} answers={int(self.ans_sizes.sum())} "''', 1),
    ("round", '''        if ctl.climb is not None:
            print(f"LIBRARY {[(k, show(v)) for k, v in ctl.lib.items()]}", flush=True)
''', '''        if ctl.climb is not None:
            print(f"LIBRARY {[(k, show(v)) for k, v in ctl.climb.lib.items()]}", flush=True)
            print(f"ROUND_SOURCE (the buffers) {[(k, show(v)) for k, v in self.round_text()[0].items()]}", flush=True)
''', 1),
    ("adam", '''class Schedule:
    """Scheduler: a fresh Adam state at the switch to phase B; learning rate 0 once the model is frozen."""

    def __init__(self, optimizer: torch.optim.Optimizer, ctl: Control) -> None:
        self.optimizer, self.ctl = optimizer, ctl

    def step(self) -> None:
        if self.ctl.reset:
            self.optimizer.state.clear()
            self.ctl.reset = False
        if self.ctl.phase == "frozen":
''', '''class Adam(torch.optim.Adam):
    """Adam that starts its moments afresh after the step in which the schedule switched to phase B (the certified
    file's scheduler cleared them at the same point: the same updates)."""

    def __init__(self, params, ctl: Control, **kw) -> None:
        super().__init__(params, **kw)
        self.ctl = ctl

    def step(self, closure=None):
        out = super().step(closure)
        if self.ctl.reset:
            self.state.clear()
            self.ctl.reset = False
        return out


class Schedule:
    """Scheduler: learning rate 0 once the model is frozen."""

    def __init__(self, optimizer: torch.optim.Optimizer, ctl: Control) -> None:
        self.optimizer, self.ctl = optimizer, ctl

    def step(self) -> None:
        if self.ctl.phase == "frozen":
''', 1),
    ("clock", '''    model.ctl.deadline = time.time() + spec.training_time_seconds          # the searches stop before the budget ends
    opt = torch.optim.Adam(model.parameters(), lr=LR, fused=spec.device_type in ("cuda", "mps"))
''', '''    opt = Adam(model.parameters(), model.ctl, lr=LR, fused=spec.device_type in ("cuda", "mps"))
''', 1),
]

CLIMB_UNIT_TAIL = '''        pool = self.kept(self.pool_rows, self.pool_sizes)                    # the rows kept (buffers)
        if pool is None:
            return
        ans = self.kept(self.ans_rows, self.ans_sizes)
        ans = None if ans is None or ans.shape[0] < ANS_MIN else tuple(ans[:, i].contiguous() for i in range(3))
        t0, w0 = _clock(), WORK[0]
        dl = ctl.climb_work0 + CLIMB_WORK if CLIMB_WORK else None            # the climb's work budget
        line = ctl.climb.step((pool[:, 0].contiguous(), pool[:, 1].contiguous()), answers=ans, stop_at=dl#@T@)
        if line is not None:
            ctl.climb_log.append(line)
            print(f"LR climb step={ctl.step} {line} [work +{WORK[0] - w0:.4g}, {WORK[0] - ctl.climb_work0:.4g} in all] "
                  f"({_clock() - t0:.1f}s)", flush=True)
        if ctl.climb.rounds and not self.n_progs():
            lib = dict(ctl.climb.lib)
            progs = [tuple(p) for p in sorted(ctl.climb.rounds, key=lambda p: (cost(expand(p, lib)), show(p)))
                     ][:PROG_SLOTS]
            self.set_round(lib, progs)
            ctl.phase, ctl.in_phase = "D", 0
            ctl.history.append((ctl.step, "round", len(progs)))
            print(f"LR round_found step={ctl.step} programs: {[show(p) for p in progs]} library: "
                  f"{ {k: show(v) for k, v in lib.items()} }", flush=True)
'''
OLD_CLIMB_UNIT_TAIL = '''        pool = torch.cat([torch.stack([n, x]) for n, x in ctl.pool], 1) if ctl.pool else None
        if pool is None:
            return
        ans = None
        if len(ctl.ans_pool) and sum(a[0].shape[0] for a in ctl.ans_pool) >= ANS_MIN:
            ans = tuple(torch.cat([a[i] for a in ctl.ans_pool]) for i in range(3))
        t0 = time.time()
        dl = None if ctl.deadline is None else ctl.deadline - CLIMB_MARGIN
        line = ctl.climb.step((pool[0], pool[1]), answers=ans, deadline=dl#@T@)
        if line is not None:
            ctl.climb_log.append(line)
            print(f"LR climb step={ctl.step} {line} ({time.time() - t0:.1f}s)", flush=True)
        if ctl.climb.rounds and not ctl.progs:
            ctl.lib = dict(ctl.climb.lib)
            ctl.progs = [tuple(p) for p in sorted(ctl.climb.rounds, key=lambda p: (cost(expand(p, ctl.lib)),
                                                                                      show(p)))[:PROG_SLOTS]]
            ctl.phase, ctl.in_phase = "D", 0
            ctl.history.append((ctl.step, "round", len(ctl.progs)))
            print(f"LR round_found step={ctl.step} programs: {[show(p) for p in ctl.progs]} library: "
                  f"{ {k: show(v) for k, v in ctl.lib.items()} }", flush=True)
'''

CONTROL_FIELDS_NEW = '''        self.climb = None                     # #@CLIMB@, from phase C on
        self.climb_work0 = 0.0                # WORK when the climb started (its budget counts from there)
        self.cells = None                     # the snapped cells the climb runs with (libcore.Cells)
        self.climb_log: list = []
        self.round_fit: list = []             # each program's share of this step's answer rows
'''


def kind_patches(kind: str) -> list:
    """The edits that differ between the two certified files."""
    if kind == "sp2":
        climb, ans_comment, ans_t = "spcore.SelfClimb", "of the smallest T seen, for the round's search", ", t=ctl.ans_t or 1"
        out = [
            ("clock", '''SEARCH_S = 900         # one search's time limit in seconds (0: none; the training's deadline always holds)
CLIMB_MARGIN = 120.0   # the climb stops this many seconds before the training budget ends
''', '''SEARCH_WORK = @SEARCH_WORK@   # one search's work budget, in WORK's units (0: none; the climb's always holds) ...
CLIMB_WORK = @CLIMB_WORK@    # ... and the whole climb's, from its start (0: none): STRICT.md §2 sets both
''', 1),
            ("clock", "search_s=SEARCH_S or None, fuse=FUSE", "search_work=SEARCH_WORK or None,\n          fuse=FUSE", 1),
            ("pools", '''        t = answer_t(aux["Cd"][sel[:1]])
        if ctl.ans_t is None or t < ctl.ans_t:
            ctl.ans_pool, ctl.ans_t = [], t
        if t != ctl.ans_t:
            return
        ctl.ans_pool.append((aux["Nd"][sel].detach(), aux["Xd"][sel].detach(), yd[sel].detach()))
        while sum(a[0].shape[0] for a in ctl.ans_pool) > ANS_POOL:
            ctl.ans_pool.pop(0)
''', '''        t = answer_t(aux["Cd"][sel[:1]])
        if int(self.ans_t) == 0 or t < int(self.ans_t):
            self.ans_sizes.zero_()
            self.ans_t.fill_(t)
        if t != int(self.ans_t):
            return
        self.keep(self.ans_rows, self.ans_sizes, torch.stack([aux["Nd"][sel], aux["Xd"][sel], yd[sel]], 1).detach())
''', 1),
            ("pools", "#@ANS_T@\n",
             '''        self.register_buffer("ans_t", torch.zeros((), dtype=torch.long))                  # their T (0: none yet)\n''', 1),
            ("pools", '''        self.pool: list = []                  # read rows (N, x digits) for the practice problems
        self.ans_pool: list = []              # answer rows (N, x, y) of the smallest T seen, for the round's search
        self.ans_t: int | None = None         # ... that T
        self.climb = None                     # spcore.SelfClimb, from phase C on
        self.cells = None                     # the snapped cells the climb runs with (libcore.Cells)
        self.lib: dict = {}                   # the library the round programs use
        self.progs: list = []                 # the tournament's round programs (items)
        self.climb_log: list = []
        self.deadline: float | None = None    # the training budget's end (build_optimizer), for the searches
        self.round_fit: list = []             # each program's share of this step's answer rows
''', CONTROL_FIELDS_NEW.replace("#@CLIMB@", climb), 1),
            ("pools", '''            ctl.climb = SelfClimb(ctl.cells, Xd.device, PR, chunk=1 << 21 if Xd.device.type == "cuda" else 1 << 18,
                                  **SP)
''', '''            ctl.climb = SelfClimb(ctl.cells, Xd.device, PR, chunk=1 << 21 if Xd.device.type == "cuda" else 1 << 18,
                                  **SP)
            ctl.climb_work0 = WORK[0]
''', 1),
            ("pools", OLD_CLIMB_UNIT_TAIL.replace("#@T@", ans_t),
             CLIMB_UNIT_TAIL.replace("#@T@", ",\n                              t=int(self.ans_t) or 1"), 1),
        ]
        return out
    return [
        ("clock", '''SEARCH_S = 900         # one search's time limit in seconds (0: none; the training's deadline always holds)
''', '''SEARCH_WORK = @SEARCH_WORK@   # one search's work budget, in WORK's units (0: none; the climb's always holds)
''', 1),
        ("clock", '''CLIMB_MARGIN = 120.0   # the climb stops this many seconds before the training budget ends (a piece of a unit, such
                       # as one batch of a search, cannot be cut and can take tens of seconds)
''', '''CLIMB_WORK = @CLIMB_WORK@    # the whole climb's work budget, from its start (0: none): STRICT.md §2 sets both
''', 1),
        ("clock", "search_s=SEARCH_S or None", "search_work=SEARCH_WORK or None", 1),
        ("pools", '''        if answer_t(aux["Cd"][sel[:1]]) != 1:
            return
        ctl.ans_pool.append((aux["Nd"][sel].detach(), aux["Xd"][sel].detach(), yd[sel].detach()))
        while sum(a[0].shape[0] for a in ctl.ans_pool) > ANS_POOL:
            ctl.ans_pool.pop(0)
''', '''        if answer_t(aux["Cd"][sel[:1]]) != 1:
            return
        self.keep(self.ans_rows, self.ans_sizes, torch.stack([aux["Nd"][sel], aux["Xd"][sel], yd[sel]], 1).detach())
''', 1),
        ("pools", "#@ANS_T@\n", "", 1),
        ("pools", '''        self.pool: list = []                  # read rows (N, x digits) for the practice problems
        self.ans_pool: list = []              # answer rows (N, x, y) of T = 1 for the round's search
        self.climb = None                     # libcore.Climb, from phase C on
        self.cells = None                     # the snapped cells the climb runs with (libcore.Cells)
        self.lib: dict = {}                   # the library the round programs use
        self.progs: list = []                 # the tournament's round programs (items)
        self.climb_log: list = []
        self.deadline: float | None = None    # the training budget's end (build_optimizer), for the searches
        self.round_fit: list = []             # each program's share of this step's answer rows
''', CONTROL_FIELDS_NEW.replace("#@CLIMB@", "libcore.Climb"), 1),
        ("pools", '''            ctl.climb = Climb(ctl.cells, Xd.device, PR, chunk=1 << 21 if Xd.device.type == "cuda" else 1 << 18,
                              **CLIMB)
''', '''            ctl.climb = Climb(ctl.cells, Xd.device, PR, chunk=1 << 21 if Xd.device.type == "cuda" else 1 << 18,
                              **CLIMB)
            ctl.climb_work0 = WORK[0]
''', 1),
        ("pools", OLD_CLIMB_UNIT_TAIL.replace("#@T@", ""), CLIMB_UNIT_TAIL.replace("#@T@", ""), 1),
    ]


# an option (--nowrap): the gate's check of "a wrapped x - N is large" (the register-width fact the late lessons taught)
NOWRAP = [
    ("wrap", '''            for a_, b_ in orders:
                Ewh, _, idw = sub_hard(terms[a_], terms[b_], Td, Tb)
                tally(ctl.visits, idw, 200)
                table = min(table, float((sub_hard(Ewh, terms[b_], Td, Tb)[1] == 0).float().mean()))
        return loss, table
''', '''        return loss, table
''', 1),
    ("wrap", "add.commute, the facts' borrows, sub.wrap,\n        sub.zero_minus: what gates the next phase)",
     "add.commute, the facts' borrows,\n        sub.zero_minus: what gates the next phase; no register-width fact)", 1),
]

# the identifiers of the clock, renamed once the edits above are in (the new code uses the new names already)
RENAMES = [(r"\bDeadline\b", "OutOfWork"), (r"\b_late\b", "_spent"), (r"\bdeadline\b", "stop_at"),
           (r"\bsearch_s\b", "search_work"), (r"time\.time\(\)", "_clock()")]
# what must be gone from a strict file
GONE = [r"(?i)deadline", r"time\.time", r"\bSEARCH_S\b", r"\bCLIMB_MARGIN\b", r"ctl\.progs", r"ctl\.lib\b",
        r"ctl\.pool", r"ctl\.ans_pool", r"ctl\.ans_t", r"\bLATE_AFTER\b", r"\bANCHOR_", r"\bSTE_CIN\b", r"\bID_PLACE0\b",
        r"\bID_BORROW\b", r"if pieces", r"optimizer\.state\.clear"]


def apply(src: str, edits: list, log: list) -> str:
    for what, old, new, count in edits:
        n = src.count(old)
        if count is None:
            assert n >= 1, (what, old[:200])
        else:
            assert n == count, (what, n, count, old[:200])
        src = src.replace(old, new)
        log.append((what, n))
    return src


def build(kind: str, sets: dict, nowrap: bool = True, r1: bool = False) -> str:
    path, md5 = CERT[kind]
    data = path.read_bytes()
    assert hashlib.md5(data).hexdigest() == md5, f"{path} is not the certified file"
    src = data.decode()
    log = []
    anchor = '\n"""\n\nfrom __future__ import annotations\n'
    assert src.count(anchor) == 1
    src = src.replace(anchor, STRICT_DOC + (NOWRAP_DOC if nowrap and not r1 else "") + anchor)
    # the late lessons: table_lessons up to its hard checks is rewritten; the checks are kept as they are
    head, tail = src.index(OLD_TABLE_LESSONS_HEAD), src.index(OLD_TABLE_LESSONS_TAIL)
    body = src[head:tail]
    assert src.count(OLD_TABLE_LESSONS_HEAD) == 1 and body.count(TABLE_CHECKS_START) == 1
    assert body.count("if pieces:") == 5, body.count("if pieces:")
    checks = body[body.index(TABLE_CHECKS_START):]
    src = src[:head] + NEW_TABLE_LESSONS + checks + src[tail:]
    log.append(("late", 1))
    src = apply(src, LIBCORE + (SPCORE if kind == "sp2" else []) + MODEL + kind_patches(kind)
                + (NOWRAP if nowrap else []), log)
    for pat, new in RENAMES:
        src, n = re.subn(pat, new, src)
        log.append((f"rename {pat}", n))
    for name, value in {**(R1_BUDGETS if r1 else BUDGETS), **sets}.items():
        if f"@{name}@" in src:
            src = src.replace(f"@{name}@", value)
        else:
            src, n = re.subn(rf"^{name} = .*$", f"{name} = {value}", src, count=1, flags=re.M)
            assert n == 1, name
    for pat in GONE:
        left = [l for l in src.splitlines() if re.search(pat, l)]
        assert not left, (pat, left[:5])
    assert "#@" not in src and "@SEARCH_WORK@" not in src and "@CLIMB_WORK@" not in src
    assert src.count("time.monotonic()") == 1 and src.count("_clock()") >= 5
    ast.parse(src)
    assert len(src.encode()) <= 256 * 1024, f"over the 256 KiB limit: {len(src.encode())}"
    return src, log


def main() -> None:
    args = sys.argv[1:]
    kinds = [a for a in args if a in CERT] or ["sp2"]
    sets = dict(a.split("=", 1) for a in args if "=" in a and not a.startswith("--"))
    out_dir = Path(args[args.index("--out") + 1]) if "--out" in args else HERE / "r1" if "--r1" in args else HERE
    nowrap, r1 = "--keep-wrap" not in args, "--r1" in args
    for kind in kinds:
        src, log = build(kind, sets, nowrap, r1)
        name = OUT[kind]
        if r1:                                  # round 1's builds (r1/): no budgets; the wrap check kept or not
            name = name.replace(".py", ("_nowrap" if nowrap else "") + "_r1.py")
        elif not nowrap:
            name = name.replace(".py", "_wrap.py")
        out = out_dir / name
        out.write_text(src)
        print(f"{kind}: wrote {out} ({len(src.splitlines())} lines, {len(src.encode())} bytes, md5 "
              f"{hashlib.md5(src.encode()).hexdigest()}); edits: "
              + ", ".join(f"{w}x{n}" for w, n in log))


if __name__ == "__main__":
    main()
