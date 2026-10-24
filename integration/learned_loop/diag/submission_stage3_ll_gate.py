"""Stage 3 with self-posed practice and the countdown learned (cpu_first/integration/learned_loop/LEARNED_LOOP.md): sp2
(integration/self_practice/submission_stage3_sp2.py, SELF_PRACTICE.md) with its hand-written countdown over T taken
out. The machine gains one instruction kind, the count loop C.r[body]: the body run as a routine as many times as
register r holds, counted down by the subtraction table. The prompt's third number goes into register S, as x goes
into X and N into N. Once the round is found (a program for the answers of the smallest T, as they are), a second
search finds the answer's program from the round, the library, the whole-register instructions and count loops,
against answers of every T. The file has no countdown, no gate and no loop law; everything else is sp2's.

What the model holds:
  * a reader and a writer (stage 1's, line for line: IO.md, WIRING.md stage 3): which tokens are digits and which
    digit each one is, where each field starts and which way it runs;
  * three digit cells, each a table of logits learned from random: the subtraction (a, b, borrow) -> (digit, borrow),
    the addition (a, b, carry) -> (digit, carry), the multiply-accumulate (a, x, h, k) -> low and high digit of
    a + x h + k;
  * a digit machine (libcore: library/dsl.py's 5 registers, a shared pointer, a borrow/carry, a multiplier digit and
    a multiply carry; 32 primitive instructions that each read the cells at the pointer or move whole registers; loops
    that run their body P, B or W times; count loops that run it as many times as a register holds) and a library of
    programs that grows during training;
  * one logit per answer program the training found (PROG_SLOTS), a tournament the answers decide.

How it learns, in one training run (one evaluator step = one forward, one loss, one backward):
  A  the reader learns to read (IO.md) and the subtraction table its borrows and digits from the mined order facts and
     the laws (general_lessons' table lessons, on the registers x and other rows' x; there is no square yet);
  B  once the subtraction table passes its own checks: the addition table by sub.inverse ((a + b + c) - b - c = a, on
     every entry, with the snapped subtraction) and the multiply-accumulate table by mul.zero and mul.ladder (on every
     entry, with the snapped addition);
  C  once every cell passes its own checks, the self-practising climb (spcore.SelfClimb), one unit per step: each
     level it poses every law of its book on the read rows' own digits in every binding (a law whose construction
     needs a routine it has not practised waits for it), probes them all (one search per pattern and binding), keeps
     the trivial and the unsolved aside, picks K learnable ones by its rule (SP_RULE), practises them (BPE, each
     pick's routine without side effects on its inputs), then searches the round by composition at full size with the
     answers of the smallest T the batches showed as its goal. Once a round is found: the answer's program
     (SelfClimb.search_outer), the rounds as routines, against the answers of rows of every T whose count reads the
     same both ways (so T's direction is not needed yet);
  D  the answer programs that reproduce them become the tournament's contestants; the answers keep scoring them; T's
     direction by speculation (IO.md, each reading run through the answer's program); the freeze once everything has
     been exact FREEZE_WIN steps.
The prediction (rule 8): the reader, then the selected answer program on one-hot digits with straight-through tables
(libcore.SteMachine: x in X, N in N, T's digits in S; a count loop's count down by the subtraction table), then the
writer.

What is designed is listed in LEARNED_LOOP.md (sp2's list in SELF_PRACTICE.md, without the countdown, its gate and
the loop law; with the count loop, the place of T in the machine and the answer program's search space).
"""

from __future__ import annotations

import itertools
import math
import time
import zlib
from array import array
from collections import Counter, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, TokenLossBatch, assert_model_state

MODE = "learned"       # "learned" | "designed" (the reader and every cell at the truth, the library given: plumbing)
PATH = "ste"           # the prediction path: one-hot digits through straight-through tables (the only one here)
GRAD_DAMP = 0.1        # the prediction path's gradient is multiplied by this after every cell and blend
W = 8                  # digits of the widest modulus (a 24-bit N is below 10^8)
P = W + 1              # the table lessons' registers (stage 1's)
PR = 2 * W             # the round machine's register places (a product of two W-digit numbers)
K = 2                  # T's field: its slots (the count's digits)
SLOTS = (W, W, K, W)   # the fields' slots: N, x, T; the writer
INIT = 0.1             # std of the initial logits
LR = 0.05              # Adam on the logits
# the subtraction table's lessons (general_lessons, ANCHORS.md)
ID_SWITCH = 0.8        # phase B once this share of the table's checks passes, averaged ...
STAGE_WIN = 20         # ... over this many steps,
ID_STEPS = 300         # ... or after this many steps (and in both cases only once the reading is chosen)
ID_PLACE0 = 5.0        # weight of the lowest place in the check (it tells the swapped code)
ID_BORROW = 1.0        # weight of the final borrow in the check: the complete identity
GATE_WIN = 10          # the digit lessons start once the borrow facts hold on every row of this many batches in a row
LATE_AFTER = 100       # the pieces' own table lessons join this many steps after the digit lessons start
STE_CIN = "anchors"
ANCHOR_ZERO = ANCHOR_SELF = ANCHOR_NEG2 = ANCHOR_COMM = ANCHOR_LESS = ANCHOR_MORE = ANCHOR_WRAP = ANCHOR_NEG = 1.0
ALL_PAIRS = True       # the table's laws also on a register and another row's register
MINE_WIN = 10          # a mined relation is in force while it has held on every row of this many batches in a row
# the cells after the subtraction (stage 3)
CELL_WIN = 20          # a cell's next stage joins once its own checks have held this many steps in a row
# the self-practising climb (spcore.SelfClimb)
SP_RULE = "lpca"       # the choosing rule: "lp" (hardest learnable first), "ct2a2" (the cheaper pairs signal), "all",
                       # "lpca" (concepts, hardest by their cheapest binding, each practised once in every binding tied
                       # for cleanest; SELF_PRACTICE.md §3)
SP_K = 4               # picks a level
SP_NP = 8              # probes per setting of a practice problem
SP_PROBE = 1_000_000   # one probe search's cap: states stored per level
SP_FALLBACK = 1_000_000        # a root's first probe over the primitives alone (cost-5 programs of primitives)
SP_DEEP = 5_800_000    # the deep probes' cap (INTEGRATION.md §2.1's measured 10-minute search on the H100) ...
SP_DEEP_K = 8          # ... for this many roots a level: those whose closest open law is closest (the best fit)
SP_DEEP_BY = "bindings"        # ... taken concept by concept, every binding where it is still open ("root": the
                               # closest roots alone, as submission_stage3_sp.py, md5 f0ef5b06)
SP_DEEP_FOCUS = "closure"      # their vocabulary: the primitives, the routines the concept's closest attempts used, and
                               # the routines those call (True: that root's attempt's routines alone, as f0ef5b06)
SP_LOOK = 300_000      # a lookahead's cap (ct2a2)
ROUND_STATES = 4_000_000       # the round's search: states stored per level ...
ROUND_COST = 5         # ... and its programs' cost at most
SP_MACRO_AFTER = 12    # past this many routines, when that search finds nothing, the round is searched by calls: up
SP_MACRO_COST = 2      # to this many routine calls, each with its inputs routed from A, X, Y, N or zero
SP_LEVELS = 16         # the climb's levels at most
SEARCH_S = 900         # one search's time limit in seconds (0: none; the training's deadline always holds)
CLIMB_MARGIN = 120.0   # the climb stops this many seconds before the training budget ends
ROUND_RETRY = 10       # once the climb is done with no round, the round is searched again every this many steps
OUTER_COST = 3         # the answer's program (SelfClimb.search_outer): its cost at most, ...
OUTER_BODY = 2         # ... a count loop's body at most this many items, ...
OUTER_STATES = 1_000_000       # ... states stored per level, ...
OUTER_NP = 8           # ... probes (taken in turn from each count)
OUTER_RETRY = 50       # a search that found no answer program is run again after this many steps, or once a new count
COUNT_POOL = 512       # answer rows kept for the answer's program (rows whose count reads the same both ways): at most
COUNT_EACH = 64        # ... this many of each count, the newest, and COUNT_POOL in all (each count an equal share)
COUNT_MIN = 2          # the answer's program is searched once the kept rows show this many counts
FUSE = 4               # a search runs up to this many chunks of its work in one batch
SP = dict(rule=SP_RULE, K=SP_K, NP=SP_NP, probe_budget=SP_PROBE, fallback_budget=SP_FALLBACK, deep_budget=SP_DEEP,
          deep_k=SP_DEEP_K, deep_by=SP_DEEP_BY, deep_focus=SP_DEEP_FOCUS, look_budget=SP_LOOK,
          round_budget=ROUND_STATES, macro_after=SP_MACRO_AFTER, macro_cost=SP_MACRO_COST,
          round_cost=ROUND_COST, levels=SP_LEVELS, search_s=SEARCH_S or None, fuse=FUSE, round_retry=ROUND_RETRY)
POOL = 4096            # read rows kept for the practice problems
ANS_POOL = 512         # answer rows kept for the round's search (rows of the smallest T the batches showed)
ANS_MIN = 64           # the round is searched once this many answer rows are kept
PROG_SLOTS = 4         # the answer programs the tournament holds: the simplest of those the answers accept (Occam,
                       # by expansion into primitives)
# the reader (stage 1's)
SHAPES = ("step", "offsets", "absolute")
OFF = W + 1
BT_TAU = 1.0
PICK_PASS = 0.999
PICK_WIN = 20
CHOSEN = 0.99
T_AFTER = 20
T_ROWS = 64
T_DONE = 0.999
ANS_WORK = 96          # the answers used: the rows with the batch's smallest nonzero T (answer_rows), at most
                       # ANS_WORK // (T + 1) of them
FREEZE_WIN = 200       # phase D: freeze after this many consecutive steps with every check exact
BATCH = 512
EVAL_BATCH = 2048
MAX_STEPS = 20000
LOG_EVERY = 50
EPS = 1e-6

PERMS = list(itertools.permutations(range(3)))
HYP = [(rn, dn, rx, dx) for rn in range(3) for rx in range(3) if rn != rx for dn in range(2) for dx in range(2)]
HIDX = {h: i for i, h in enumerate(HYP)}

# the law library (general_lessons/LESSONS.md, reduced to what this model uses): each law is about an operation of
# the machine; the table lessons apply them to the registers, the practice problems (libcore.build) to the rows' digits
LAWS = {
    "sub.order": ("a < b exactly when a - b borrows, on every top part; subtracting keeps, reverses or wraps the order",
                  "every mined order fact; the subtraction table"),
    "sub.bijective": ("every row and column of the subtraction uses each digit once", "the table"),
    "sub.inverse": ("(a - b) + b = a", "the registers; the addition table on every entry; the adder's practice"),
    "sub.identity": ("a - 0 = a, a - a = 0", "the registers"),
    "sub.zero_minus": ("0 - a borrows when a != 0", "every mined nonzero fact"),
    "add.commute": ("a + b = b + a", "the registers"),
    "mul.zero": ("a + x 0 + k = a + k", "the multiply-accumulate table on every entry"),
    "mul.ladder": ("a + x (h + 1) + k = (a + x h + k) + x", "the multiply-accumulate table; the row's practice"),
    "mul.distribute": ("x y = y additions of x", "the multiply's practice"),
    "red.identity": ("red(a) = a for a < N; red(N) = 0", "the link's practice (modx)"),
    "red.step": ("one step moves the register by multiples of N until it is below N", "the chain's and the step's "
                 "practice (modB, step)"),
    "red.invariant": ("red(a + Q N) = a for a < N", "the reduction's practice (reduce), Q another row's x"),
}
ORDER_TERMS = ("N", "x", "T")

# ===================================================================================================== programs
REGS = ("A", "X", "Y", "N", "S")
RI = {r: i for i, r in enumerate(REGS)}
A_, X_, Y_, N_, S_ = range(5)
PRIMS = ([f"SBB.{r}" for r in "XYNS"] + [f"ADC.{r}" for r in "XYNS"] + [f"MUL.{r}" for r in "XYN"]
         + ["SEL.S"] + [f"LDA.{r}" for r in "XYNS"] + [f"STA.{r}" for r in "XYS"] + ["CLA", "CLC", "CLK"]
         + [f"LDH.{r}" for r in "XY"] + ["INC"] + [f"SHL.{r}" for r in "AXY"] + [f"SHR.{r}" for r in "AXY"]
         + ["BRG.X"])
LOOP_KINDS = ("P", "B", "W")
ONE = "ONE"                                          # a pseudo loop kind run once: a straight-line body
COUNT_KINDS = tuple(f"C.{r}" for r in REGS)          # count loops: the body runs as many times as register r holds
LOOP_CAP = 100                                       # a count loop runs its body at most this many times; a larger
                                                     #   count is an error that clears the probe's machine


def is_loop(it):
    return isinstance(it, tuple)


def is_count(it):
    return isinstance(it, tuple) and it[1] in COUNT_KINDS


def fill(pattern, arg):
    out = []
    for it in pattern:
        if is_loop(it):
            out.append(("REP", it[1], fill(it[2], arg)))
        else:
            out.append(arg if it == "_" else it)
    return tuple(out)


def cost(items):
    return sum(1 + cost(it[2]) if is_loop(it) else 1 for it in items)


def expand(items, lib):
    out = []
    for it in items:
        if is_loop(it) and it[1] in lib:
            out.extend(expand(fill(lib[it[1]], it[2][0]), lib))
        elif is_loop(it):
            out.append(("REP", it[1], tuple(expand(it[2], lib))))
        elif it in lib:
            out.extend(expand(lib[it], lib))
        else:
            out.append(it)
    return tuple(out)


def show(items):
    return " ".join(f"{it[1]}[{show(it[2])}]" if is_loop(it) else it for it in items)


def parse(s):
    toks = s.replace("[", " [ ").replace("]", " ] ").split()
    pos = 0

    def seq():
        nonlocal pos
        out = []
        while pos < len(toks) and toks[pos] != "]":
            t = toks[pos]
            if pos + 1 < len(toks) and toks[pos + 1] == "[":
                pos += 2
                body = seq()
                pos += 1
                out.append(("REP", t, tuple(body)))
            else:
                out.append(t)
                pos += 1
        return out
    return tuple(seq())


# ===================================================================================================== BPE
def seqs(items):
    yield items
    for it in items:
        if is_loop(it):
            yield from seqs(it[2])


def rewrite(items, fn):
    out = []
    for it in items:
        out.append(("REP", it[1], rewrite(it[2], fn)) if is_loop(it) else it)
    return tuple(fn(tuple(out)))


def _name(lib, prefix):
    i = 1
    while f"{prefix}{i}" in lib:
        i += 1
    return f"{prefix}{i}"


def _pairs(seq):
    out, i = [], 0
    while i < len(seq):
        a = seq[i]
        if is_loop(a):
            if len(a[2]) == 1 and not is_loop(a[2][0]):
                out.append(("LOOP", a[1], a[2][0]))
            i += 1
            continue
        if i + 1 < len(seq) and not is_loop(seq[i + 1]):
            out.append(("PAIR", a, seq[i + 1]))
            if a == seq[i + 1]:
                i += 2
                continue
        i += 1
    return out


def pair_counts(corpus):
    """A task counts a pair as often as its best program uses it (library/merge.py, weighting "max")."""
    cnt = Counter()
    for progs in corpus.values():
        if not progs:
            continue
        per = []
        for p in progs:
            c = Counter()
            for s in seqs(p):
                c.update(_pairs(s))
            per.append(c)
        for pr in set().union(*per):
            cnt[pr] += max(c[pr] for c in per)
    return cnt


def _apply_pair(pr, name):
    kind = pr[0]

    def fn(seq):
        out, i = [], 0
        while i < len(seq):
            it = seq[i]
            if kind == "LOOP" and is_loop(it) and it[1] == pr[1] and it[2] == (pr[2],):
                out.append(name)
                i += 1
            elif kind == "PAIR" and i + 1 < len(seq) and it == pr[1] and seq[i + 1] == pr[2]:
                out.append(name)
                i += 2
            else:
                out.append(it)
                i += 1
        return out
    return fn


def _pair_size(pr, lib):
    items = (("REP", pr[1], (pr[2],)),) if pr[0] == "LOOP" else (pr[1], pr[2])
    return cost(expand(items, lib))


def bpe(corpus, lib, max_merges=12, min_count=2.0, prefix="B", log=None):
    """Greedy pair merges, ties toward the pair standing for the most primitives (library/merge.py's "largest")."""
    lib, new = dict(lib), []
    for _ in range(max_merges):
        cnt = pair_counts(corpus)
        if not cnt:
            break
        pr, w = max(cnt.items(), key=lambda kv: (round(kv[1], 9), _pair_size(kv[0], lib), str(kv[0])))
        if w < min_count:
            break
        name = _name(lib, prefix)
        lib[name] = (("REP", pr[1], (pr[2],)),) if pr[0] == "LOOP" else (pr[1], pr[2])
        fn = _apply_pair(pr, name)
        corpus = {t: [rewrite(p, fn) for p in progs] for t, progs in corpus.items()}
        new.append(name)
        if log is not None:
            log.append({"name": name, "def": show(lib[name]), "weight": round(w, 3)})
    return corpus, lib, new


def _apply_routine(pat, name):
    def fn(seq):
        out, i = [], 0
        while i < len(seq):
            if tuple(seq[i:i + len(pat)]) == tuple(pat):
                out.append(name)
                i += len(pat)
            else:
                out.append(seq[i])
                i += 1
        return out
    return fn


# ===================================================================================================== the machine
class Cells:
    """The digit cells as tensors: sbb, adc [B, B, 2] -> digit, flag; mul [B, B, B, B] (a, x, h, k) -> low, high.
    Exact by default (for checks); the model's learned tables replace them (set)."""

    def __init__(self, B=10, device="cpu"):
        self.B = B
        a, b, c = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
        d = a - b - c
        self.sbb_d, self.sbb_c = d % B, (d < 0).long()
        d = a + b + c
        self.adc_d, self.adc_c = d % B, (d >= B).long()
        a4, x4, h4, k4 = torch.meshgrid(*(torch.arange(B),) * 4, indexing="ij")
        t = a4 + x4 * h4 + k4
        self.mul_lo, self.mul_hi = t % B, t // B
        self.to(device)

    def to(self, device):
        self.device = device
        for n in ("sbb_d", "sbb_c", "adc_d", "adc_c", "mul_lo", "mul_hi"):
            setattr(self, n, getattr(self, n).to(device))
        self._flat()
        return self

    def set(self, **tabs):
        for n, t in tabs.items():
            setattr(self, n, t.long().to(self.device).reshape(getattr(self, n).shape))
        self._flat()
        return self

    def _flat(self):
        B = self.B
        self.sbb = torch.stack([self.sbb_d, self.sbb_c], -1).reshape(B * B * 2, 2)
        self.adc = torch.stack([self.adc_d, self.adc_c], -1).reshape(B * B * 2, 2)
        self.mul = torch.stack([self.mul_lo, self.mul_hi], -1).reshape(B ** 4, 2)
        self.alu = torch.cat([self.sbb, self.adc, self.mul]).to(torch.uint8)    # the lockstep machine's one lookup


class Setting:
    """A machine size: base B, operands of W digits, registers of P places (2 W unless given)."""

    def __init__(self, B, W, cells=None, P=0):
        self.B, self.W, self.cells = B, W, cells
        self.P = P or 2 * W

    def count(self, kind):
        assert kind not in COUNT_KINDS, "a count loop's count is data, not a size"
        return 1 if kind == ONE else {"P": self.P, "B": self.B, "W": self.W}[kind]

    def __repr__(self):
        return f"({self.B},{self.W})"


class Batch:
    """S machine states over NP probes: regs [S, 5, P, NP] uint8, c / h / k [S, NP], p [S]."""

    def __init__(self, regs, c, h, k, p):
        self.regs, self.c, self.h, self.k, self.p = regs, c, h, k, p

    @property
    def S(self):
        return self.regs.shape[0]

    def copy(self):
        return Batch(self.regs.clone(), self.c.clone(), self.h.clone(), self.k.clone(), self.p.clone())

    def take(self, idx):
        return Batch(self.regs[idx], self.c[idx], self.h[idx], self.k[idx], self.p[idx])

    @staticmethod
    def concat(bs):
        return Batch(*(torch.cat([getattr(b, f) for b in bs], 0) for f in ("regs", "c", "h", "k", "p")))

    def features(self):
        return torch.cat([self.regs.reshape(self.S, -1), self.c, self.h, self.k, self.p.to(torch.uint8)[:, None]], 1)


def gen_on(device, seed):
    """A random generator on the device (the machines' dirty starts and the probes' picks stay on it)."""
    return torch.Generator(device=device).manual_seed(seed)


def fresh(st, inputs, gen, NP, device):
    """One fresh machine (S = 1) over NP probes: inputs {reg: digits [NP, P]}; everything else random (a dirty
    start, so that every found program sets up what it reads)."""
    B, P = st.B, st.P
    rnd = lambda hi, shape: torch.randint(0, hi, shape, generator=gen, device=device, dtype=torch.uint8)
    regs = rnd(B, (1, 5, P, NP))
    for r, v in inputs.items():
        regs[0, RI[r]] = v.to(device=device, dtype=torch.uint8).T
    return Batch(regs, rnd(2, (1, NP)), rnd(B, (1, NP)), rnd(B, (1, NP)), torch.zeros(1, dtype=torch.long,
                                                                                         device=device))


def random_states(st, S, NP, gen, device):
    B, P = st.B, st.P
    rnd = lambda hi, shape: torch.randint(0, hi, shape, generator=gen, device=device, dtype=torch.uint8)
    return Batch(rnd(B, (S, 5, P, NP)), rnd(2, (S, NP)), rnd(B, (S, NP)), rnd(B, (S, NP)),
                 torch.randint(0, P, (S,), generator=gen, device=device))


def _shl(b, r):
    b.regs[:, r, 1:] = b.regs[:, r, :-1].clone()
    b.regs[:, r, 0] = 0


def _shr(b, r):
    b.regs[:, r, :-1] = b.regs[:, r, 1:].clone()
    b.regs[:, r, -1] = 0


def step(st, b, tok):
    """One primitive on batch b in place."""
    cl, B = st.cells, st.B
    op, _, arg = tok.partition(".")
    if op in ("SBB", "ADC", "MUL"):
        ar = torch.arange(b.S, device=b.regs.device)
        a, x = b.regs[ar, A_, b.p].long(), b.regs[ar, RI[arg], b.p].long()
        if op == "MUL":
            out = cl.mul[((a * B + x) * B + b.h.long()) * B + b.k.long()]
            b.k = out[..., 1].to(torch.uint8)
        else:
            out = (cl.sbb if op == "SBB" else cl.adc)[(a * B + x) * 2 + b.c.long()]
            b.c = out[..., 1].to(torch.uint8)
        b.regs[ar, A_, b.p] = out[..., 0].to(torch.uint8)
    elif op == "SEL":
        b.regs[:, A_] = torch.where(b.c[:, None, :].bool(), b.regs[:, S_], b.regs[:, A_])
    elif op == "LDA":
        b.regs[:, A_] = b.regs[:, RI[arg]]
    elif op == "STA":
        b.regs[:, RI[arg]] = b.regs[:, A_]
    elif op == "CLA":
        b.regs[:, A_] = 0
    elif op == "CLC":
        b.c = torch.zeros_like(b.c)
    elif op == "CLK":
        b.k = torch.zeros_like(b.k)
    elif op == "LDH":
        ar = torch.arange(b.S, device=b.regs.device)
        b.h = b.regs[ar, RI[arg], b.p].clone()
    elif op == "INC":
        b.p = (b.p + 1) % st.P
    elif op == "SHL":
        _shl(b, RI[arg])
    elif op == "SHR":
        _shr(b, RI[arg])
    elif op == "BRG":
        top = b.regs[:, X_, -1].clone()
        _shl(b, A_)
        b.regs[:, A_, 0] = top
        _shl(b, X_)
    else:
        raise ValueError(tok)


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
            else:
                for _ in range(st.count(kind)):
                    run_items(st, b, body, lib)
        elif it in lib:
            if it.startswith("T."):                      # a routine restores the pointer it was called with
                p = b.p.clone()
                run_items(st, b, lib[it], lib)
                b.p = p
            else:
                run_items(st, b, lib[it], lib)
        else:
            step(st, b, it)


# ===================================================================================================== batched search
_GEN = torch.Generator().manual_seed(12345)
_MULT = torch.randint(-(1 << 62), 1 << 62, (8192,), generator=_GEN, dtype=torch.int64) * 2 + 1
_MULT_ON = {}


def _s64(v):
    return v - (1 << 64) if v >= 1 << 63 else v


_M2 = _s64(0x94D049BB133111EB)


def _lsr(h, s):
    return (h >> s) & ((1 << (64 - s)) - 1)


def hash_rows(F):
    """A 64-bit hash (int64, wrapping arithmetic) of every row of a uint8 matrix: a random odd linear combination of
    its 64-bit words, then a mix."""
    S, n = F.shape
    pad = (-n) % 8
    if pad:
        F = torch.cat([F, torch.zeros(S, pad, dtype=torch.uint8, device=F.device)], 1)
    Wd = F.contiguous().view(torch.int64)
    if F.device not in _MULT_ON:
        _MULT_ON[F.device] = _MULT.to(F.device)
    h = (Wd * _MULT_ON[F.device][:Wd.shape[1]][None]).sum(1)
    h = h ^ _lsr(h, 31)
    h = h * _M2
    return h ^ _lsr(h, 29)


def keys_of(batches):
    return hash_rows(torch.cat([b.features() for b in batches], 1))


def _put(out, idx, sub):
    out.regs[idx] = sub.regs
    out.c[idx] = sub.c
    out.h[idx] = sub.h
    out.k[idx] = sub.k
    out.p[idx] = sub.p


def _groups(ids):
    keep = torch.nonzero(ids >= 0).flatten()
    if keep.numel() == 0:
        return []
    order = keep[torch.sort(ids[keep], stable=True).indices]
    uniq, counts = torch.unique_consecutive(ids[order], return_counts=True)
    out, start = [], 0
    for u, c in zip(uniq.tolist(), counts.tolist()):
        out.append((u, order[start:start + c]))
        start += c
    return out


def _apply_grouped(st, b, item_of_row, items, lib):
    """apply_items item by item: tokens run once per distinct token, loops run their iterations together with the
    body's elements grouped position by position."""
    out = b.copy()
    dev = b.regs.device
    loop = [is_loop(it) and it[1] not in lib for it in items]
    is_l = torch.tensor(loop, dtype=torch.bool, device=dev)[item_of_row]
    for u, idx in _groups(torch.where(~is_l, item_of_row, torch.full_like(item_of_row, -1))):
        sub = out.take(idx)
        run_items(st, sub, (items[u],), lib)
        _put(out, idx, sub)
    if not any(loop):
        return out
    elems, eid = {}, []
    L = max((len(it[2]) for it, lp in zip(items, loop) if lp), default=0)
    cnt = torch.zeros(len(items), dtype=torch.long)
    table = torch.full((len(items), max(L, 1)), -1, dtype=torch.long)
    for i, (it, lp) in enumerate(zip(items, loop)):
        if lp:
            cnt[i] = st.count(it[1])
            for j, e in enumerate(it[2]):
                if e not in elems:
                    elems[e] = len(elems)
                    eid.append(e)
                table[i, j] = elems[e]
    rcnt = torch.where(is_l, cnt.to(dev)[item_of_row], torch.zeros_like(item_of_row))
    rtab = table.to(dev)[item_of_row]
    for itn in range(int(rcnt.max())):
        live = rcnt > itn
        for j in range(L):
            for u, idx in _groups(torch.where(live, rtab[:, j], torch.full_like(item_of_row, -1))):
                sub = out.take(idx)
                run_items(st, sub, (eid[u],), lib)
                _put(out, idx, sub)
    return out


# the lockstep machine: every row runs its own primitive instructions, one per step, all rows in the same kernels
OPCODE = {t: i for i, t in enumerate(PRIMS)}
NOP = len(PRIMS)                 # no instruction (a sequence shorter than the others it runs with)
SAVE = NOP + 1                   # a routine's entry at depth d (SAVE + 2 d) keeps the pointer, its exit (SAVE + 2 d + 1)
MAX_DEPTH = 32                   #   restores it
_TRACES = {}                     # (P, B, W, id(lib)) -> (lib, {item: (instructions, depth)})
_OPS = {}                        # (P, device) -> _Ops
# apply_items' time model (microseconds; measured on an H100, s3h): a lockstep step, plus per row; on the item-by-item
# path an instruction, a group's rows (taken and put back once per group: an item, or each element of a loop body at
# each iteration), and a group's split and merge
LOCK_STEP, LOCK_ROW, PRIM_STEP, PRIM_ROW, GROUP_FIX = 200.0, 0.0025, 25.0, 0.004, 60.0
LOCK_COST = 0.0                  # (tests: a large negative value forces the lockstep on every bucket)
LOCK_ROWS = 1 << 22              # a lockstep runs at most this many rows at once (its step's temporaries: ~1 KB a row)


def trace(st, items, lib, out, depth=0):
    """The primitive instructions run_items executes for items at setting st, in order, as opcodes (a routine between
    its pointer's save and restore); returns the deepest routine nesting."""
    deep = depth
    for it in items:
        if is_loop(it):
            _, kind, body = it
            if kind in lib:
                deep = max(deep, trace(st, fill(lib[kind], body[0]), lib, out, depth))
            else:
                for _ in range(st.count(kind)):
                    deep = max(deep, trace(st, body, lib, out, depth))
        elif it in lib:
            if it.startswith("T."):
                assert depth < MAX_DEPTH, it
                out.append(SAVE + 2 * depth)
                deep = max(deep, trace(st, lib[it], lib, out, depth + 1))
                out.append(SAVE + 2 * depth + 1)
            else:
                deep = max(deep, trace(st, lib[it], lib, out, depth))
        else:
            out.append(OPCODE[it])
    return deep


TRACE_KEEP = 50_000_000         # instructions the trace memo holds before it starts over (int16: 100 MB)


def _traces(st, lib):
    """The memo of traces at st's sizes under lib, kept while lib is (a lib only ever gains names; a name's definition
    never changes): {(item, depth): (instructions as array('h'), deepest nesting)}."""
    key = (st.P, st.B, st.W, id(lib) if lib else 0)
    ent = _TRACES.get(key)
    if ent is None or (lib and ent[0] is not lib) or ent[2][0] > TRACE_KEEP:
        while len(_TRACES) >= 6:
            _TRACES.pop(next(iter(_TRACES)))
        ent = _TRACES[key] = (lib, {}, [0])
    return ent


def _trace_of(st, it, lib, depth, ent):
    """trace() of one item at a routine depth, memoized: array('h') of opcodes and the deepest nesting."""
    memo = ent[1]
    got = memo.get((it, depth))
    if got is not None:
        return got
    if is_loop(it):
        _, kind, body = it
        if kind in lib:
            got = _trace_seq(st, fill(lib[kind], body[0]), lib, depth, ent)
        else:
            ops, deep = _trace_seq(st, body, lib, depth, ent)
            got = (ops * st.count(kind), deep)
    elif it in lib:
        if it.startswith("T."):
            assert depth < MAX_DEPTH, it
            ops, deep = _trace_seq(st, lib[it], lib, depth + 1, ent)
            got = (array("h", [SAVE + 2 * depth]) + ops + array("h", [SAVE + 2 * depth + 1]), deep)
        else:
            got = _trace_seq(st, lib[it], lib, depth, ent)
    else:
        got = (array("h", [OPCODE[it]]), depth)
    memo[(it, depth)] = got
    ent[2][0] += len(got[0])
    return got


def _trace_seq(st, items, lib, depth, ent):
    ops, deep = array("h"), depth
    for it in items:
        o, d = _trace_of(st, it, lib, depth, ent)
        ops += o
        deep = max(deep, d)
    return ops, deep


class _Ops:
    """Every opcode's effect as table rows, for one register size P: where each register place comes from (rm; 5 P is
    a zero), the cell family and its register (SBB 1, ADC 2, MUL 3), and the other instructions' flags."""

    def __init__(self, P, device):
        n, Z = SAVE + 2 * MAX_DEPTH, 5 * P
        ar = torch.arange(P)
        rm = torch.arange(5 * P).repeat(n, 1)
        fam, ralu, rldh, dep = (torch.zeros(n, dtype=torch.long) for _ in range(4))
        flag = {f: torch.zeros(n, dtype=torch.bool) for f in ("sel", "clc", "clk", "ldh", "inc", "save", "rest")}
        for tok, o in OPCODE.items():
            op, _, arg = tok.partition(".")
            r = RI.get(arg, 0)
            if op in ("SBB", "ADC", "MUL"):
                fam[o], ralu[o] = {"SBB": 1, "ADC": 2, "MUL": 3}[op], r
            elif op == "LDA":
                rm[o, :P] = r * P + ar
            elif op == "STA":
                rm[o, r * P:(r + 1) * P] = ar
            elif op == "CLA":
                rm[o, :P] = Z
            elif op == "LDH":
                flag["ldh"][o], rldh[o] = True, r
            elif op == "SHL":
                rm[o, r * P] = Z
                rm[o, r * P + 1:(r + 1) * P] = r * P + ar[:-1]
            elif op == "SHR":
                rm[o, (r + 1) * P - 1] = Z
                rm[o, r * P:(r + 1) * P - 1] = r * P + ar[1:]
            elif op == "BRG":
                rm[o, 0] = X_ * P + P - 1
                rm[o, 1:P] = ar[:-1]
                rm[o, X_ * P] = Z
                rm[o, X_ * P + 1:(X_ + 1) * P] = X_ * P + ar[:-1]
            else:
                flag[{"SEL": "sel", "CLC": "clc", "CLK": "clk", "INC": "inc"}[op]][o] = True
        for d in range(MAX_DEPTH):
            flag["save"][SAVE + 2 * d], flag["rest"][SAVE + 2 * d + 1] = True, True
            dep[SAVE + 2 * d] = dep[SAVE + 2 * d + 1] = d
        self.P, self.Z = P, Z
        self.rm = rm.to(device)
        self.ri = torch.stack([ralu * P, rldh * P], 1).to(device)       # the register a cell or LDH reads, as a place
        self.dep = dep.to(device)
        self.flags = torch.stack([flag[f] for f in ("sel", "clc", "clk", "ldh", "inc", "save", "rest")]
                                 + [fam > 0, (fam == 1) | (fam == 2), fam == 3], 1).to(device)
        # the cell table's row (Cells.alu: the subtraction's 2 B^2 rows, the addition's, then the multiply's B^4) as
        # off + a ca + x cx + c cc + h ch + k ck over (a, x, c, h, k)
        self.B = None

    def coefs(self, B):
        if self.B != B:
            n = self.flags.shape[0]
            co = torch.zeros(n, 6, dtype=torch.int32)
            fam = torch.zeros(n, dtype=torch.long)
            for tok, o in OPCODE.items():
                op = tok.partition(".")[0]
                if op in ("SBB", "ADC"):
                    co[o] = torch.tensor([2 * B, 2, 1, 0, 0, 0 if op == "SBB" else 2 * B * B], dtype=torch.int32)
                elif op == "MUL":
                    co[o] = torch.tensor([B ** 3, B * B, 0, B, 1, 4 * B * B], dtype=torch.int32)
            self.co, self.B = co.to(self.rm.device), B
        return self.co


def _ops(P, device):
    key = (P, str(device))
    if key not in _OPS:
        _OPS[key] = _Ops(P, device)
    return _OPS[key]


def _ustep(o, alu, co, regs, c, h, k, p, ps, op, ar, ad):
    """One instruction on every machine, each its own (op [S]), exactly as step() runs it: regs [S, 5 P, NP] uint8
    (flat, contiguous), c, h, k [S, NP] uint8, p [S], ps [S, D] the routines' saved pointers (ar, ad: aranges of S and
    D). Returns the new state. Register moves go by words of the probes' bytes (8, 4 or 2 at a time, as NP allows);
    the three cells by one table (Cells.alu) at one row index per probe."""
    S, P = op.shape[0], o.P
    NP = regs.shape[2]
    fl = o.flags[op]
    m = o.rm[op]
    wd = torch.int64 if NP % 8 == 0 else torch.int32 if NP % 4 == 0 else torch.int16 if NP % 2 == 0 else torch.uint8
    w = regs.view(wd)
    g = w.gather(1, m.clamp(max=o.Z - 1)[:, :, None].expand(S, o.Z, w.shape[2]))
    g = g.masked_fill_((m == o.Z)[:, :, None], 0).view(torch.uint8)
    sel = fl[:, 0:1] & (c > 0)
    g[:, :P] = torch.where(sel[:, None, :], g[:, 4 * P:], g[:, :P])
    ri = o.ri[op]
    a = g[ar, p]
    v = torch.stack([a, g[ar, ri[:, 0] + p], c, h, k], -1).int()
    cf = co[op]
    out = alu[((v * cf[:, None, :5]).sum(-1) + cf[:, 5:6]).long()]
    g[ar, p] = torch.where(fl[:, 7:8], out[..., 0], a)
    c = torch.where(fl[:, 8:9], out[..., 1], c).masked_fill_(fl[:, 1:2], 0)
    k = torch.where(fl[:, 9:10], out[..., 1], k).masked_fill_(fl[:, 2:3], 0)
    h = torch.where(fl[:, 3:4], g[ar, ri[:, 1] + p], h)
    if ps.shape[1]:
        d = o.dep[op].clamp(max=ps.shape[1] - 1)[:, None]
        ps = torch.where(fl[:, 5:6] & (ad == d), p[:, None], ps)
        p = torch.where(fl[:, 6], ps.gather(1, d)[:, 0], p)
    p = torch.where(fl[:, 4], (p + 1) % P, p)
    return g, c, h, k, p, ps


def _columns(arrs, n, dev, block=512):
    """The instructions of the sequences arrs (array('h') each, at most n long), NOP-padded, as columns on the device:
    [n, len(arrs)] (built block by block of steps, so that no index tensor is larger than block x len(arrs))."""
    flat = array("h")
    offs = []
    for a in arrs:
        offs.append(len(flat))
        flat += a
    flat.append(NOP)
    f = torch.frombuffer(flat, dtype=torch.int16).to(dev)
    off = torch.tensor(offs, dtype=torch.long, device=dev)
    ln = torch.tensor([len(a) for a in arrs], dtype=torch.long, device=dev)
    out = torch.empty(n, len(arrs), dtype=torch.long, device=dev)
    for t0 in range(0, n, block):
        t = torch.arange(t0, min(n, t0 + block), device=dev)[:, None]
        out[t0:t0 + block] = f[torch.where(t < ln[None], off[None] + t, len(flat) - 1)]
    return out


GRAPH_STEPS = 32                 # on a GPU, a long lockstep replays its steps from a captured CUDA graph of this many
GRAPH_MIN = 192                  #   steps (0: never), from this many steps on (the same kernels, one launch per graph)


def _lockstep(st, b, cols, lr, depth):
    """b's rows (a Batch, in place) run their sequences, one step per column of cols [n, items] (opcodes, NOP-padded;
    lr [S]: each row's column), all rows at once."""
    S, _, P, NP = b.regs.shape
    dev = b.regs.device
    o = _ops(P, dev)
    co = o.coefs(st.B)
    state = [b.regs.reshape(S, 5 * P, NP).contiguous(), b.c, b.h, b.k, b.p,
             torch.zeros(S, depth, dtype=torch.long, device=dev)]
    ar, ad = torch.arange(S, device=dev), torch.arange(depth, device=dev)[None]
    n = cols.shape[0]
    t0 = 0
    if dev.type == "cuda" and GRAPH_STEPS and n >= GRAPH_MIN:
        state, t0 = _graph_steps(o, st.cells.alu, co, state, cols, lr, ar, ad)
    for t in range(t0, n):
        state = list(_ustep(o, st.cells.alu, co, *state, cols[t][lr], ar, ad))
    regs, c, h, k, p, _ = state
    b.regs, b.c, b.h, b.k, b.p = regs.view(S, 5, P, NP), c, h, k, p


def _graph_steps(o, alu, co, state, cols, lr, ar, ad):
    """As many whole blocks of GRAPH_STEPS steps as cols has, by one CUDA graph: the first block runs eagerly on static
    buffers (it also warms up), the second is captured (capture records, it does not run), then the graph is replayed
    for the rest. Returns the state and the steps done."""
    with torch.no_grad():
        return _graph_steps_(o, alu, co, state, cols, lr, ar, ad)


def _graph_steps_(o, alu, co, state, cols, lr, ar, ad):
    K = GRAPH_STEPS
    nb = cols.shape[0] // K
    buf = [x.clone() for x in state]
    t = torch.zeros(1, dtype=torch.long, device=cols.device)

    def block():
        cur = buf
        for _ in range(K):
            cur = _ustep(o, alu, co, *cur, cols.index_select(0, t)[0][lr], ar, ad)
            t.add_(1)
        for x, y in zip(buf, cur):
            x.copy_(y)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        block()
    torch.cuda.current_stream().wait_stream(side)
    if nb > 1:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            block()
        for _ in range(nb - 1):
            g.replay()
        del g
    return buf, nb * K


def apply_items(st, b, item_of_row, items, lib):
    """A new batch: row r of b with items[item_of_row[r]] applied. Each distinct item is traced into the instructions
    run_items would execute (trace); the rows then run their own instructions in lockstep, one per step, every row in
    the same kernels (_ustep), grouped by the length of their sequences (powers of 2) so that short ones do not wait
    for long ones. When few distinct items share the steps, item by item (_apply_grouped) launches fewer kernels and
    is used instead. Both give exactly run_items' result. Rows whose item is a count loop go to _apply_counts."""
    if any(is_count(it) for it in items):
        return _apply_counts(st, b, item_of_row, items, lib)
    ent = _traces(st, lib)
    seqs = [_trace_of(st, it, lib, 0, ent) for it in items]
    lens = [len(s[0]) for s in seqs]
    bk = [n.bit_length() if n else 0 for n in lens]                   # 0; 1; 2-3; 4-7; ...: one lockstep per bucket
    nb = max(bk, default=0) + 1
    mx, per = [0] * nb, [[] for _ in range(nb)]
    loc = [0] * len(items)
    for i, (j, n) in enumerate(zip(bk, lens)):
        mx[j] = max(mx[j], n)
        loc[i] = len(per[j])
        per[j].append(i)
    S, k = b.S, max(1, len(items))
    rows_per = S / k
    groups = [st.count(it[1]) * len(it[2]) if is_loop(it) and it[1] not in lib else 1 for it in items]
    lock = {j for j in range(1, nb) if per[j] and LOCK_COST + mx[j] * (LOCK_STEP + len(per[j]) * rows_per * LOCK_ROW)
            < sum(lens[i] * PRIM_STEP + groups[i] * rows_per * PRIM_ROW + GROUP_FIX for i in per[j])}
    if b.S == 0 or not lock:
        return _apply_grouped(st, b, item_of_row, items, lib)
    out = b.copy()
    dev = b.regs.device
    loc_row = torch.tensor(loc, dtype=torch.long, device=dev)[item_of_row]
    rest = [j for j in range(1, nb) if per[j] and j not in lock]
    if len(lock) == 1 and not rest and not per[0]:
        parts = [(next(iter(lock)), None)]
    else:
        bucket = torch.tensor(bk, dtype=torch.long, device=dev)[item_of_row]
        order = torch.sort(bucket, stable=True).indices
        counts = torch.bincount(bucket, minlength=nb).tolist()
        starts = [sum(counts[:j]) for j in range(nb)]
        parts = [(j, order[starts[j]:starts[j] + counts[j]]) for j in range(1, nb) if per[j] and counts[j]]
    for j, idx in parts:
        sub = out if idx is None else out.take(idx)
        lr = loc_row if idx is None else loc_row[idx]
        if j in lock:
            cols = _columns([seqs[i][0] for i in per[j]], mx[j], dev)
            depth = max(seqs[i][1] for i in per[j])
            if sub.S <= LOCK_ROWS:
                _lockstep(st, sub, cols, lr, depth)
            else:                                    # in slices of rows: a step's temporaries stay bounded
                for r0 in range(0, sub.S, LOCK_ROWS):
                    part = sub.take(slice(r0, r0 + LOCK_ROWS)).copy()
                    _lockstep(st, part, cols, lr[r0:r0 + LOCK_ROWS], depth)
                    _put(sub, slice(r0, r0 + LOCK_ROWS), part)
        else:
            sub = _apply_grouped(st, sub, lr, [items[i] for i in per[j]], lib)
        if idx is not None:
            _put(out, idx, sub)
    return out


def _apply_counts(st, b, item_of_row, items, lib):
    """apply_items with count loops among the items: the other rows as apply_items runs them; the count loops' rows run
    their bodies together, one apply_items call per iteration (each body one item, as a routine), every probe keeping
    the result while its own count lasts; a probe whose count is above LOOP_CAP ends cleared. run_items' result."""
    dev = b.regs.device
    flag = torch.tensor([is_count(it) for it in items], dtype=torch.bool, device=dev)[item_of_row]
    out = b.copy()
    rest = torch.nonzero(~flag).flatten()
    if rest.numel():
        _put(out, rest, apply_items(st, b.take(rest), item_of_row[rest],
                                    [("REP", ONE, ()) if is_count(it) else it for it in items], lib))
    rows = torch.nonzero(flag).flatten()
    if rows.numel() == 0:
        return out
    sub, ior = b.take(rows), item_of_row[rows]
    reg = torch.tensor([RI[it[1][2:]] if is_count(it) else 0 for it in items], dtype=torch.long, device=dev)[ior]
    cnt, ok = count_down(st.cells, sub.regs[torch.arange(rows.numel(), device=dev), reg].transpose(1, 2))
    bodies = [("REP", ONE, it[2] if is_count(it) else ()) for it in items]
    for i in range(int(torch.where(ok, cnt, 0).max())):
        live = ok & (cnt > i)
        act = torch.nonzero(live.any(1)).flatten()
        part = sub.take(act)
        new = apply_items(st, part, ior[act], bodies, lib)
        _merge(part, new, live[act])
        _put(sub, act, part)
    _clear(sub, ~ok)
    _put(out, rows, sub)
    return out


# the round machine (s3i): one program on many rows from fresh machines, as run_prog runs it. Control never depends on
# data, so the pointer is resolved once on the host (INC and the routines' saves and restores cost nothing) and every
# other instruction becomes one or two fixed slice operations on one state tensor; on a GPU the whole program is one
# CUDA graph, captured once and replayed
ROUND_ROWS = 128                 # a round machine's rows: a multiple of this (fewer shapes, so fewer graphs)
ROUND_KEEP = 8                   # round machines kept (each with its graph)
ROUND_GRAPH = True               # on a GPU, replay the program from a CUDA graph (False: eagerly, as on a CPU)
_ROUND = {}                      # (instructions, P, B, rows, device) -> RoundMachine


class RoundMachine:
    """One program's instructions (a trace) on `rows` machines at once: state St [5 P + 4, rows] int64, the registers'
    places (A, X, Y, N, S), then c, h, k and a row that stays 0; the cells as tables [2 (digit, flag), inputs...]."""

    def __init__(self, st, ops, rows, device, share=None, p0=0):
        P, B = st.P, st.B
        self.st, self.P, self.dev = Setting(B, P // 2, None, P), P, torch.device(device)
        if share is None:
            St = self.St = torch.zeros(5 * P + 4, rows, dtype=torch.long, device=device)
            self.sbb = torch.zeros(2, B, B, 2, dtype=torch.long, device=device)
            self.adc = torch.zeros_like(self.sbb)
            self.mul = torch.zeros(2, B, B, B, B, dtype=torch.long, device=device)
        else:                             # a stretch of a longer program (OuterMachine): its state and its tables
            St = self.St = share.St
            self.sbb, self.adc, self.mul = share.sbb, share.adc, share.mul
        c, h, k, Z = St[5 * P], St[5 * P + 1], St[5 * P + 2], 5 * P + 3
        A, S = St[:P], St[S_ * P:(S_ + 1) * P]
        idx = {}

        def rows_of(*ix):
            if ix not in idx:
                idx[ix] = torch.tensor(ix, dtype=torch.long, device=device)
            return idx[ix]
        fns, p, saved = [], p0, [0] * MAX_DEPTH
        for o in ops:
            if o >= SAVE:                                                   # a routine's entry or exit
                d = (o - SAVE) // 2
                if (o - SAVE) % 2 == 0:
                    saved[d] = p
                else:
                    p = saved[d]
                continue
            op, _, arg = PRIMS[o].partition(".")
            r = RI.get(arg, 0)
            R = St[r * P:(r + 1) * P]
            if op == "INC":
                p = (p + 1) % P
            elif op in ("SBB", "ADC"):                                      # A[p], c = the cell at (A[p], R[p], c)
                fns.append(lambda tab=(self.sbb if op == "SBB" else self.adc), a=A[p], x=R[p], to=rows_of(p, 5 * P):
                           St.index_copy_(0, to, tab[:, a, x, c]))
            elif op == "MUL":                                               # A[p], k = the cell at (A[p], R[p], h, k)
                fns.append(lambda a=A[p], x=R[p], to=rows_of(p, 5 * P + 2):
                           St.index_copy_(0, to, self.mul[:, a, x, h, k]))
            elif op == "SEL":
                fns.append(lambda: torch.where(c != 0, S, A, out=A))
            elif op == "LDA":
                fns.append(lambda R=R: A.copy_(R))
            elif op == "STA":
                fns.append(lambda R=R: R.copy_(A))
            elif op in ("CLA", "CLC", "CLK"):
                fns.append({"CLA": A, "CLC": c, "CLK": k}[op].zero_)
            elif op == "LDH":
                fns.append(lambda x=R[p]: h.copy_(x))
            else:                                    # SHL, SHR, BRG: places moved within registers (Z: a 0 place)
                q = list(range(r * P, (r + 1) * P))
                src = {"SHL": [Z] + q[:-1], "SHR": q[1:] + [Z],
                       "BRG": [X_ * P + P - 1] + list(range(P - 1)) + [Z] + list(range(X_ * P, X_ * P + P - 1))}[op]
                lo = 0 if op == "BRG" else r * P
                fns.append(lambda dst=St[lo:lo + len(src)], src=rows_of(*src): dst.copy_(St.index_select(0, src)))
        self.fns, self.graph, self.p_end = fns, None, p

    def cells(self, cells):
        """The cells the program reads (copied into the machine's tables, which a graph reads in place)."""
        B = self.st.B
        self.sbb.copy_(cells.sbb.T.reshape(2, B, B, 2))
        self.adc.copy_(cells.adc.T.reshape(2, B, B, 2))
        self.mul.copy_(cells.mul.T.reshape(2, B, B, B, B))

    def run(self, Xd, Nd, go, n, gen):
        """One run from fresh machines: x digits Xd in X and Nd in N (rows m <= rows [m, <= P]); everything else fresh's
        random draws for the n rows of go [m] alone, in their order, exactly as run_prog draws them for those rows.
        Every row runs; returns A's digits [m, P] (a view, until the next run; rows outside go: anything)."""
        P, St, m = self.P, self.St, Xd.shape[0]
        b = fresh(self.st, {}, gen, n, self.dev)
        at = (go.long().cumsum(0) - 1).clamp(min=0)                          # each row's place among go's rows
        St[:5 * P, :m] = b.regs[0].reshape(5 * P, n)[:, at]
        St[X_ * P:(X_ + 1) * P, :m] = pad(Xd, P).T
        St[N_ * P:(N_ + 1) * P, :m] = pad(Nd, P).T
        St[5 * P:5 * P + 3, :m] = torch.cat([b.c, b.h, b.k])[:, at]
        with torch.no_grad():
            self._program()
        return St[:P, :m].T

    def _program(self):
        if self.dev.type != "cuda" or not ROUND_GRAPH:
            for f in self.fns:
                f()
        elif self.graph is None:                     # the first run eagerly on a side stream (it also warms up), then
            side = torch.cuda.Stream()               # the program captured (capture records, it does not run)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for f in self.fns:
                    f()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                for f in self.fns:
                    f()
        else:
            self.graph.replay()


def round_machine(st, prog, lib, rows, device):
    """prog's RoundMachine at setting st (its cells loaded) for rows machines (rounded up to a multiple of ROUND_ROWS),
    made once per instruction sequence and size and kept (ROUND_KEEP at most)."""
    ops, _ = _trace_seq(st, prog, lib, 0, _traces(st, lib))
    R = -(-max(rows, 1) // ROUND_ROWS) * ROUND_ROWS
    key = (ops.tobytes(), st.P, st.B, R, str(device))
    m = _ROUND.get(key)
    if m is None:
        while len(_ROUND) >= ROUND_KEEP:
            _ROUND.pop(next(iter(_ROUND)))
        m = _ROUND[key] = RoundMachine(st, ops, R, device)
    m.cells(st.cells)
    return m


class OuterMachine:
    """An answer's program (count loops at its top level, plain items between them) on `rows` machines at once: each
    stretch of plain items and each count loop's body is a RoundMachine on one shared state (the body's from the loop's
    pointer); a count loop replays its body's machine while any row's count lasts, each row keeping the result while
    its own does (count_down on the count register's digits, as the loop began). Returns run_items' result."""

    def __init__(self, st, prog, lib, rows, device):
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

    def cells(self, cells):
        self.base.cells(cells)
        self.cl = cells

    def run(self, Xd, Nd, Cd, gen):
        """From fresh machines (fresh's draws for all m rows, in order): x digits Xd in X, Nd in N and the count's Cd
        in S ([m, <= P] each). Returns A's digits [m, P] (a view, until the next run)."""
        P, St, m = self.P, self.base.St, Xd.shape[0]
        b = fresh(self.st, {}, gen, m, self.dev)
        St.zero_()
        St[:5 * P, :m] = b.regs[0].reshape(5 * P, m)
        for r, d in ((X_, Xd), (N_, Nd), (S_, Cd)):
            St[r * P:(r + 1) * P, :m] = pad(d, P).T
        St[5 * P:5 * P + 3, :m] = torch.cat([b.c, b.h, b.k])
        with torch.no_grad():
            for r, mach in self.parts:
                if r is None:
                    mach._program()
                    continue
                cnt, ok = count_down(self.cl, St[r * P:(r + 1) * P].T)
                for i in range(int(torch.where(ok, cnt, 0).max())):
                    old = St.clone()
                    mach._program()
                    St.copy_(torch.where((ok & (cnt > i))[None], St, old))
                St[:5 * P + 3] *= ok[None]
        return St[:P, :m].T


_OUTER = {}                      # (instructions, P, B, rows, device) -> OuterMachine


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


class ItemClass:
    def __init__(self, rep, cost):
        self.rep, self.cost = rep, cost


class Deadline(Exception):
    """The training's deadline passed inside a climb unit (the unit returns at once; nothing is cached)."""


def _late(deadline):
    return deadline is not None and time.time() > deadline


def _signatures(settings, tests, items, lib, chunk=1 << 21, deadline=None):
    S = tests[0].S
    out = []
    step_ = max(1, chunk // S)
    for i0 in range(0, len(items), step_):
        if _late(deadline):
            raise Deadline()
        part = items[i0:i0 + step_]
        k = len(part)
        dev = tests[0].regs.device
        ri = torch.arange(k, device=dev).repeat_interleave(S)
        rs = torch.arange(S, device=dev).repeat(k)
        outs = [apply_items(st, t.take(rs), ri, part, lib) for st, t in zip(settings, tests)]
        keys = keys_of(outs).view(k, S)
        out.append(hash_rows(keys.contiguous().view(torch.uint8).view(k, S * 8)))
    return torch.cat(out).tolist() if out else []


def enumerate_items(tokens, lib, settings, device, max_body=2, max_arity=3, seed=7, S_test=24, NP_test=6,
                    deadline=None, cache=None, chunk=1 << 21):
    """Every distinct item (tokens, then loops over bodies of cost up to max_body built from the distinct items so
    far), the cheapest spelling first: items that agree on S_test random machines of every setting are one item.
    Raises Deadline if the deadline passes. cache: {item: signature} kept across calls with the same settings (the
    test machines are the same every call and an item's meaning never changes, so only new items are run)."""
    gen = gen_on(device, seed)
    tests = [random_states(st, S_test, NP_test, gen, device) for st in settings]

    def sigs(items):
        if cache is None:
            return _signatures(settings, tests, items, lib, chunk=chunk, deadline=deadline)
        miss = list(dict.fromkeys(it for it in items if it not in cache))
        if miss:
            for it, sg in zip(miss, _signatures(settings, tests, miss, lib, chunk=chunk, deadline=deadline)):
                cache[it] = sg
        return [cache[it] for it in items]
    ident = sigs([("REP", ONE, ())])[0]
    classes, by_sig = [], {ident}

    def add_all(cands, c):
        for item, s in zip(cands, sigs(cands)):
            if s not in by_sig:
                by_sig.add(s)
                classes.append(ItemClass(item, c))

    add_all(list(tokens), 1)
    body_sig = {ident}
    for bc in range(1, max_body + 1):
        reps, cands = list(classes), []

        def extend(prefix, pc, n):
            if pc == bc:
                cands.append(prefix)
                return
            if n == max_arity:
                return
            for cl in reps:
                if pc + cl.cost <= bc:
                    extend(prefix + (cl.rep,), pc + cl.cost, n + 1)
        extend((), 0, 0)
        found = []
        for body, s in zip(cands, sigs([("REP", ONE, b) for b in cands])):
            if s not in body_sig:
                body_sig.add(s)
                found.append(body)
        add_all([("REP", kind, body) for body in found for kind in LOOP_KINDS], bc + 1)
    return classes


class Problem:
    """A task on some settings: one fresh machine per setting, the targets ({"A": digits [P, NP]} and/or {"c": [NP]})
    and an optional extra check per setting (a function of the candidates' A digits [S, NP, P] -> bool [S])."""

    def __init__(self, task, settings, roots, targets, inverse=None):
        self.task, self.settings, self.roots, self.targets = task, settings, roots, targets
        self.inverse = inverse or [None] * len(settings)

    def goal(self, batches):
        ok = torch.ones(batches[0].S, dtype=torch.bool, device=batches[0].regs.device)
        for b, t, inv in zip(batches, self.targets, self.inverse):
            if "A" in t:
                ok &= (b.regs[:, A_] == t["A"][None]).all(-1).all(-1)
            if "c" in t:
                ok &= (b.c == t["c"][None]).all(-1)
            if inv is not None and bool(ok.any()):
                idx = torch.nonzero(ok).flatten()
                ok[idx[~inv(b.regs[idx, A_].permute(0, 2, 1))]] = False
        return ok


class Result:
    def __init__(self, task):
        self.task, self.found_cost, self.goals, self.level_sizes, self.stop = task, None, None, [], ""
        self.parents, self.aliases, self.accepted = None, [], None


def _fresh(seen, keys):
    if seen.numel() == 0:
        return torch.ones(keys.numel(), dtype=torch.bool, device=keys.device)
    pos = torch.searchsorted(seen, keys)
    pos[pos == seen.numel()] = 0
    return seen[pos] != keys


def bfs(prob, classes, lib, max_cost=8, max_states=400_000, accept=None, limit=200, deadline=None, chunk=1 << 21,
        listing="spellings", max_goals=50_000, spell=8, batch=256, fuse=1):
    """Breadth-first by cost, states merged over every probe of every setting, every minimal-cost parent kept. Stops
    at the first cost whose goal programs pass `accept`, when a level would pass max_states (that level is finished
    streaming goal states only), or at the deadline. The goal programs tested: the first `limit` spellings
    (listing "spellings"), or one spelling per goal state (listing "goals"), tested in batches, fewest top-level items
    first (then fewest primitives), until a batch has one that passes; the accepted goal states then give up to `spell`
    spellings each that pass. A level's work is blocks of (items, source states) of at most `chunk` rows, taken in
    order; up to `fuse` chunks' worth of consecutive blocks run in one apply_items call (fewer lockstep steps per row;
    every block is then treated as alone, so the result is the same for any fuse)."""
    settings = prob.settings
    dev = prob.roots[0].regs.device
    levels = [prob.roots]
    seen = torch.sort(keys_of(prob.roots)).values
    parents = [None]
    res = Result(prob.task)
    res.level_sizes.append(1)
    if bool(prob.goal(prob.roots)[0]):
        res.found_cost, res.goals, res.parents = 0, [0], parents
        return res
    by_cost = {}
    for ci, cl in enumerate(classes):
        by_cost.setdefault(cl.cost, []).append(ci)
    by_cost = {c: torch.tensor(v, dtype=torch.long, device=dev) for c, v in by_cost.items()}
    for d in range(1, max_cost + 1):
        last = d == max_cost
        ck, cc, cs, ci_, chunks, gk = [], [], [], [], [], []
        n_cand = n_fresh = 0
        late = False
        def blocks(d=d):
            for c, cis in by_cost.items():
                sl = d - c
                if sl < 0 or levels[sl] is None:
                    continue
                for s0 in range(0, levels[sl][0].S, chunk):
                    srcs = [b.take(slice(s0, s0 + chunk)) for b in levels[sl]]
                    i_step = max(1, chunk // srcs[0].S)
                    for i0 in range(0, cis.numel(), i_step):
                        yield cis[i0:i0 + i_step], sl, s0, srcs
        work = blocks()
        nxt = next(work, None)
        while nxt is not None:
            if _late(deadline):
                late = True
                break
            group, n_rows = [], 0
            while nxt is not None and (not group or n_rows + nxt[0].numel() * nxt[3][0].S <= fuse * chunk):
                group.append(nxt)
                n_rows += nxt[0].numel() * nxt[3][0].S
                nxt = next(work, None)
            parts, its, srcrows = [], [], [[] for _ in settings]
            for ii, sl, s0, srcs in group:
                ns, k = srcs[0].S, ii.numel()
                its.append(torch.arange(k, device=dev).repeat_interleave(ns) + len(parts))
                parts.extend(classes[i].rep for i in ii.tolist())
                rs = torch.arange(ns, device=dev).repeat(k)
                for si, b in enumerate(srcs):
                    srcrows[si].append(b.take(rs))
            ri_all = its[0] if len(its) == 1 else torch.cat(its)
            outs_all = [apply_items(st, r[0] if len(r) == 1 else Batch.concat(r), ri_all, parts, lib)
                        for st, r in zip(settings, srcrows)]
            del srcrows
            off = 0
            for ii, sl, s0, srcs in group:
                ns, k = srcs[0].S, ii.numel()
                outs = outs_all if len(group) == 1 else [o.take(slice(off, off + k * ns)) for o in outs_all]
                off += k * ns
                ri = torch.arange(k, device=dev).repeat_interleave(ns)
                rs = torch.arange(ns, device=dev).repeat(k)
                keys = keys_of(outs)
                m = _fresh(seen, keys)
                if not bool(m.any()):
                    continue
                idx = torch.nonzero(m).flatten()
                n_fresh += idx.numel()
                sub = [b.take(idx) for b in outs]
                del outs
                g = prob.goal(sub)
                if bool(g.any()):
                    gk.append(keys[idx][g])
                if last:
                    if not bool(g.any()):
                        continue
                    gi = torch.nonzero(g).flatten()
                    idx, sub = idx[gi], [b.take(gi) for b in sub]
                ck.append(keys[idx])
                cc.append(ii[ri[idx]])
                cs.append(torch.full((idx.numel(),), sl, dtype=torch.long, device=dev))
                ci_.append(rs[idx] + s0)
                chunks.append(sub)
                n_cand += idx.numel()
                if not last and n_cand > 1.5 * max_states:
                    last = True
                    res.stop = f"budget: level {d} streamed (over {max_states} states)"
                    new = [[], [], [], [], []]
                    for j in range(len(chunks)):
                        gm = torch.nonzero(prob.goal(chunks[j])).flatten()
                        if gm.numel():
                            for lst, t in zip(new, (ck[j], cc[j], cs[j], ci_[j])):
                                lst.append(t[gm])
                            new[4].append([b.take(gm) for b in chunks[j]])
                    ck, cc, cs, ci_, chunks = new
            del outs_all
        if late or _late(deadline):
            res.stop = f"deadline at level {d}"
        if ck:
            allk = torch.cat(ck)
            uk, inv = torch.unique(allk, sorted=True, return_inverse=True)
            pos = torch.arange(allk.numel(), device=dev)
            first = torch.full((uk.numel(),), allk.numel(), dtype=torch.long, device=dev).scatter_reduce(
                0, inv, pos, reduce="amin")
            sfirst = torch.sort(first).values
            rank = torch.searchsorted(sfirst, first)
            child = rank[inv]
            order = torch.sort(child, stable=True).indices
            parents.append((child[order], torch.cat(cc)[order], torch.cat(cs)[order], torch.cat(ci_)[order]))
            if not last and not res.stop:
                parts, off = [[] for _ in settings], 0
                for j in range(len(chunks)):
                    ch, n = chunks[j], chunks[j][0].S
                    lo_, hi_ = int(torch.searchsorted(sfirst, off)), int(torch.searchsorted(sfirst, off + n))
                    rows = sfirst[lo_:hi_] - off
                    for si in range(len(settings)):
                        parts[si].append(ch[si].take(rows))
                    off += n
                levels.append([Batch.concat(p) for p in parts])
            else:
                levels.append(None)
            chunks = None
        else:
            uk = torch.zeros(0, dtype=torch.int64, device=dev)
            rank = torch.zeros(0, dtype=torch.long, device=dev)
            parents.append(None)
            levels.append(None)
        res.level_sizes.append(int(uk.numel()) if not last else int(n_fresh))
        if uk.numel() > max_states and not res.stop:
            res.stop = f"budget: level {d} has {uk.numel()} > {max_states} states"
        if gk:
            g = torch.unique(torch.cat(gk))
            res.found_cost = d
            res.goals = rank[torch.searchsorted(uk, g)].tolist()
            res.parents = parents
            if listing == "goals":
                cand = []
                for gi, g in enumerate(res.goals[:max_goals]):
                    if gi % 256 == 0 and _late(deadline):
                        res.stop = res.stop or f"deadline at level {d}"
                        break
                    f = programs(_OneGoal(res, g), classes, limit=1)
                    if f:
                        cand.append((len(f[0]), cost(expand(f[0], lib)), f[0], g))
                cand.sort(key=lambda c: (c[0], c[1]))
                progs = [c[2] for c in cand]
                if accept is None:
                    res.accepted = progs
                    break
                good = []
                for i0 in range(0, len(cand), batch):
                    if _late(deadline):
                        res.stop = res.stop or f"deadline at level {d}"
                        break
                    good = [c for c, o in zip(cand[i0:i0 + batch], accept(progs[i0:i0 + batch])) if o]
                    if good:
                        break
                if good:
                    res.accepted = [c[2] for c in good]
                    if spell > 1:                              # more spellings of the accepted goal states, for BPE
                        more = [q for c in good for q in programs(_OneGoal(res, c[3]), classes, limit=spell)[1:]]
                        res.accepted += [q for q, o in zip(more, accept(more) if more else []) if o]
                    break
                res.aliases.append((d, [show(p) for p in progs[:20]]))
                res.found_cost, res.goals = None, None
                if res.stop:
                    break
                seen = torch.sort(torch.cat([seen, uk])).values
                continue
            progs = programs(res, classes, limit=limit)
            if accept is None:
                res.accepted = progs
                break
            ok = accept(progs)
            if any(ok):
                res.accepted = [p for p, o in zip(progs, ok) if o]
                break
            res.aliases.append((d, [show(p) for p in progs[:20]]))
            res.found_cost, res.goals = None, None
        if res.stop:
            break
        seen = torch.sort(torch.cat([seen, uk])).values
    if res.found_cost is None and not res.stop:
        res.stop = f"exhausted cost {max_cost}"
    res.parents = parents
    return res


class _OneGoal:
    def __init__(self, res, g):
        self.parents, self.found_cost, self.goals = res.parents, res.found_cost, [g]


def programs(res, classes, limit=200):
    """Up to `limit` minimal programs reaching the goal states."""
    memo = {}

    def paths(level, idx):
        if level == 0:
            return [()]
        if (level, idx) in memo:
            return memo[(level, idx)]
        out = []
        child, pcl, psl, psi = res.parents[level]
        key = torch.tensor([idx], device=child.device)
        lo, hi = int(torch.searchsorted(child, key)), int(torch.searchsorted(child, key, right=True))
        for ci, sl, si in zip(pcl[lo:hi].tolist(), psl[lo:hi].tolist(), psi[lo:hi].tolist()):
            for p in paths(sl, si):
                out.append(p + (classes[ci].rep,))
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        memo[(level, idx)] = out
        return out

    progs = []
    for g in res.goals:
        progs.extend(paths(res.found_cost, g)[:limit - len(progs)])
        if len(progs) >= limit:
            break
    return progs


# ===================================================================================================== the practice problems' helpers
def top_parts(Nd, Xd, w):
    """N's top w digits N' and x's digits at the same places x' [n, w] (digit strings only)."""
    L = Nd.shape[1]
    top = torch.where(Nd != 0, torch.arange(L, device=Nd.device)[None], -1).max(1).values
    lo = (top - w + 1).clamp(min=0)
    idx = lo[:, None] + torch.arange(w, device=Nd.device)[None]
    valid = idx <= top[:, None]
    idx = idx.clamp(max=L - 1)
    return Nd.gather(1, idx) * valid, Xd.gather(1, idx) * valid


def pad(d, P):
    if d.shape[1] >= P:
        return d[:, :P]
    return torch.cat([d, torch.zeros(d.shape[0], P - d.shape[1], dtype=d.dtype, device=d.device)], 1)


def run_prog(st, prog, lib, ins, gen, device):
    """A found program on inputs {reg: digits [n, P]}: (A digits [n, P], c bits [n]), by the machine's cells."""
    n = next(iter(ins.values())).shape[0]
    b = fresh(st, ins, gen, n, device)
    run_items(st, b, prog, lib)
    return b.regs[0, A_].T.long(), b.c[0].long()


def count_of(d):
    """A digit string's count (how many times to repeat an addition): read off its digits."""
    v = torch.zeros(d.shape[0], dtype=torch.long, device=d.device)
    for j in reversed(range(d.shape[1])):
        v = v * 10 + d[:, j]
    return v


# ===================================================================================================== the round
def round_settings(cells, P):
    return Setting(10, P // 2, cells, P)


def round_problem(cells, rows, NP, P, device, seed=0):
    """Probes at the task's size: x in X, N in N, the target the row's answer y in A (rows: Nd, Xd, Yd)."""
    Nd, Xd, Yd = rows
    st = round_settings(cells, P)
    gen = gen_on(device, seed)
    root = fresh(st, {"X": pad(Xd[:NP], P), "N": pad(Nd[:NP], P)}, gen, min(NP, Nd.shape[0]), device)
    return Problem("round", [st], [root], [{"A": pad(Yd[:NP], P).T.to(torch.uint8)}])


def round_fit(prog, lib, cells, rows, P, device, seed=1):
    """The share of answer rows (Nd, Xd, Yd) the program reproduces at the task's size (a report: on a GPU the mean is
    a float32 sum times 1/n, below 1.0 for some n although every row is right; round_ok is the exact test)."""
    Nd, Xd, Yd = rows
    st = round_settings(cells, P)
    A, _ = run_prog(st, prog, lib, {"X": pad(Xd, P), "N": pad(Nd, P)}, gen_on(device, seed), device)
    return float((A[:, :Yd.shape[1]] == Yd).all(1).float().mean())


def round_ok(prog, lib, cells, rows, P, device, seed=1):
    """Does the program reproduce every answer row (Nd, Xd, Yd) at the task's size? (round_fit's machine and seed.)"""
    Nd, Xd, Yd = rows
    st = round_settings(cells, P)
    A, _ = run_prog(st, prog, lib, {"X": pad(Xd, P), "N": pad(Nd, P)}, gen_on(device, seed), device)
    return bool((A[:, :Yd.shape[1]] == Yd).all())


# ===================================================================================================== Occam's step count
def steps(items, lib, P=4, B=10, W=2):
    """How many primitive instructions a program executes at a setting (base B, W-digit operands, P places)."""
    n = 0
    for it in items:
        if is_count(it):                  # its count is data: the body once
            n += steps(it[2], lib, P, B, W)
        elif is_loop(it) and it[1] in lib:
            n += steps(fill(lib[it[1]], it[2][0]), lib, P, B, W)
        elif is_loop(it):
            n += {"P": P, "B": B, "W": W, ONE: 1}[it[1]] * steps(it[2], lib, P, B, W)
        elif it in lib:
            n += steps(lib[it], lib, P, B, W)
        else:
            n += 1
    return n


# ===================================================================================================== the one-hot machine
class SteMachine:
    """A found program on one-hot digits (the prediction path): every cell a straight-through table (rows (a, b, c) ->
    digit 10 + flag 2; (a, x, h, k) -> low 10 + high 10), the pointer a Python int, registers immutable tensors (so
    autograd keeps every earlier value); every cell's output and every count loop's blend passes `damp` (its gradient
    scaled)."""

    def __init__(self, SBB, ADC, MUL, P, damp, B=10, used=None):
        self.SBB, self.ADC, self.MUL, self.P, self.B, self.damp = SBB, ADC, MUL, P, B, damp
        self.used = used                 # the subtraction's rows (a, b, c) the count loops' count downs meet (a tally)

    def count(self, kind):
        return {"P": self.P, "B": self.B, "W": self.P // 2}[kind]

    def run(self, regs, c, h, k, items, lib, p=0):
        st = [list(regs), c, h, k, p]
        self._items(st, items, lib)
        return st

    def _count(self, st, it, lib):
        """A count loop: the count (register r as the loop begins) counted down by the subtraction cell, place by
        place; each round the rows whose count down did not borrow run the body (as a routine, from the loop's
        pointer) and take its state, the others keep theirs (a straight-through blend, as stage 1's countdown); rows
        whose count is above LOOP_CAP end cleared."""
        _, kind, body = it
        C, p, B = st[0][RI[kind[2:]]], st[4], self.B
        one = torch.zeros_like(C)
        one[..., 0] = 1
        one[:, 0] = 0
        one[:, 0, 1] = 1
        for i in range(LOOP_CAP + 1):
            c = torch.zeros_like(C[:, 0, :2])
            c[:, 0] = 1
            out = []
            for j in range(C.shape[1]):
                jt = self._joint(C[:, j], (one[:, j], c))
                if self.used is not None:
                    self.used += jt.detach().sum(0)
                o = self.damp(jt @ self.SBB)
                out.append(o[:, :B])
                c = o[:, B:]
            go = c[:, 0]                 # the count down did not borrow: the count lasts
            if not bool((go > 0.5).any()):
                break
            g3, g2 = go[:, None, None], go[:, None]
            if i == LOOP_CAP:            # the error: a count above LOOP_CAP clears the row's machine
                zero = torch.zeros_like(C)
                zero[..., 0] = 1
                st[0] = [g3 * zero + (1 - g3) * a for a in st[0]]
                st[1], st[2], st[3] = (g2 * zero[:, 0, :a.shape[1]] + (1 - g2) * a for a in st[1:4])
                break
            new = [list(st[0]), st[1], st[2], st[3], p]
            self._items(new, body, lib)
            st[0] = [self.damp(g3 * a + (1 - g3) * b) for a, b in zip(new[0], st[0])]
            st[1], st[2], st[3] = (self.damp(g2 * a + (1 - g2) * b) for a, b in zip(new[1:4], st[1:4]))
            C = self.damp(g3 * torch.stack(out, 1) + (1 - g3) * C)
        st[4] = p

    def _items(self, st, items, lib):
        for it in items:
            if is_count(it):
                self._count(st, it, lib)
            elif is_loop(it):
                _, kind, body = it
                if kind in lib:
                    self._items(st, fill(lib[kind], body[0]), lib)
                else:
                    for _ in range(self.count(kind)):
                        self._items(st, body, lib)
            elif it in lib:
                if it.startswith("T."):
                    p = st[4]
                    self._items(st, lib[it], lib)
                    st[4] = p
                else:
                    self._items(st, lib[it], lib)
            else:
                self._step(st, it)

    @staticmethod
    def _joint(a, flags):
        j = a
        for f in flags:
            j = (j[:, :, None] * f[:, None, :]).reshape(a.shape[0], -1)
        return j

    def _step(self, st, tok):
        regs, c, h, k, p = st
        B = self.B
        op, _, arg = tok.partition(".")
        regs = list(regs)
        zero = torch.zeros_like(regs[0][:, :1])
        zero[..., 0] = 1
        put = lambda R, d: torch.cat([R[:, :p], d[:, None], R[:, p + 1:]], 1)
        if op in ("SBB", "ADC"):
            out = self.damp(self._joint(regs[0][:, p], (regs[RI[arg]][:, p], c)) @ (self.SBB if op == "SBB" else self.ADC))
            regs[0] = put(regs[0], out[:, :B])
            st[0], st[1] = regs, out[:, B:]
        elif op == "MUL":
            out = self.damp(self._joint(regs[0][:, p], (regs[RI[arg]][:, p], h, k)) @ self.MUL)
            regs[0] = put(regs[0], out[:, :B])
            st[0], st[3] = regs, out[:, B:]
        elif op == "SEL":
            sel = c[:, 1][:, None, None]
            regs[0] = self.damp(sel * regs[4] + (1 - sel) * regs[0])
            st[0] = regs
        elif op == "LDA":
            regs[0] = regs[RI[arg]]
            st[0] = regs
        elif op == "STA":
            regs[RI[arg]] = regs[0]
            st[0] = regs
        elif op == "CLA":
            regs[0] = zero.expand_as(regs[0])
            st[0] = regs
        elif op == "CLC":
            st[1] = torch.zeros_like(c)
            st[1][:, 0] = 1
        elif op == "CLK":
            st[3] = torch.zeros_like(k)
            st[3][:, 0] = 1
        elif op == "LDH":
            st[2] = regs[RI[arg]][:, p]
        elif op == "INC":
            st[4] = (p + 1) % self.P
        elif op == "SHL":
            regs[RI[arg]] = torch.cat([zero, regs[RI[arg]][:, :-1]], 1)
            st[0] = regs
        elif op == "SHR":
            regs[RI[arg]] = torch.cat([regs[RI[arg]][:, 1:], zero], 1)
            st[0] = regs
        elif op == "BRG":
            top = regs[1][:, -1:]
            regs[0] = torch.cat([top, regs[0][:, :-1]], 1)
            regs[1] = torch.cat([zero, regs[1][:, :-1]], 1)
            st[0] = regs
        else:
            raise ValueError(tok)

# ===================================================================================================== the law book
OPERAND_REGS = ("A", "X", "Y", "N")       # a law's operands go to these registers in every injective way

# A pattern: the instance kinds (one data term per operand; probe i is of kind i mod len), how many digits its
# numbers need beyond the operands' width w ("same": w; "carry": w + 1; "double": 2 w; the generic width rule picks
# the widest w that fits the registers), and a row filter ("ne": only rows with x' != N', where a term would leave the
# law's range).
PATTERNS = {
    "pair": ((("x", "n"), ("n", "x"), ("x", "x"), ("x", "z")), "carry", ""),     # room for add's carry
    "x+n": ((("x+n", "n"),), "carry", ""),
    "x+dn": ((("x+dn", "n"),), "carry", ""),
    "x+on": ((("x+on", "n"),), "double", ""),
    "Bx+d": ((("Bx+d", "n"),), "carry", "ne"),
    "x,d": ((("x", "d"),), "carry", ""),
    "x,o": ((("x", "o"),), "double", ""),
    "o,x,d": ((("o", "x", "d"),), "carry", ""),
}
# The operations, each with the patterns it is posed on; a law on a pattern abstains on the kinds where neither a fact
# nor a routine gives its value.
LAW_BOOK = {
    "lt": ("c", ("pair", "x+dn")),           # sub.order: a < b exactly when a - b borrows
    "eq": ("c", ("pair",)),                  # a = b
    "min": ("A", ("pair",)),
    "max": ("A", ("pair",)),
    "red": ("A", ("pair", "x+n", "x+dn", "x+on", "Bx+d")),   # red.identity, red.zero, red.invariant, red.step
    "div": ("A", ("x+n", "x+dn", "x+on")),   # the division law's quotient
    "add": ("A", ("pair",)),                 # sub.inverse (checked by red and lt), add.identity
    "sub": ("A", ("pair", "x+n")),           # sub.identity, sub.self, sub.inverse
    "mul": ("A", ("x,d", "x,o")),            # mul.zero, mul.ladder, mul.distribute (as additions)
    "acc.mul": ("A", ("o,x,d",)),            # mul.ladder with its accumulator: a + b c
    "acc.add": ("A", ("o,x,d",)),            # a + b + c
}
GROW = {"same": 0, "carry": 1, "double": 2}


class NotPosable(Exception):
    """A construction needs a routine the learner does not have yet (or its instance leaves the law's range)."""


def width_for(grow, st):
    """The widest operand width whose instance fits the registers (P places)."""
    if grow == "double":
        return st.P // 2
    return st.P - GROW[grow]


def binding_name(b):
    return "".join(b)


def bindings(arity):
    return list(itertools.permutations(OPERAND_REGS, arity))


def digit_string(v, P):
    """A digit (one per row, below the base) as a digit string [n, P]: the digit at place 0, zeros above."""
    out = torch.zeros(v.shape[0], P, dtype=torch.long, device=v.device)
    out[:, 0] = v
    return out


class Ops:
    """The routines the learner can call to build problems: (law, pattern) -> [(token, binding)], every practised
    routine of that law on that pattern in the order practised. run() places operands in a routine's registers and
    runs it on a fresh machine: the first one (data is built with it), or the i-th (a check passes when any of them
    confirms it: a law's own pattern may not tell its routine from an alias, as "x if x < N, else 0" passes red's
    pair pattern; one routine that is right is then enough)."""

    def __init__(self):
        self.by_key, self.version = {}, 0

    def add(self, law, pattern, token, binding):
        self.by_key.setdefault((law, pattern), []).append((token, binding))
        self.version += 1

    def has(self, law, pattern):
        return (law, pattern) in self.by_key

    def count(self, law, pattern):
        return len(self.by_key.get((law, pattern), ()))

    def run(self, law, pattern, st, lib, operands, gen, device, i=0):
        """(A digits [n, P], c [n]) of routine i for law on pattern, operands placed in its bound registers."""
        if (law, pattern) not in self.by_key:
            raise NotPosable(f"{law}@{pattern}")
        tok, bd = self.by_key[(law, pattern)][i]
        ins = {r: v for r, v in zip(bd, operands)}
        n = operands[0].shape[0]
        b = fresh(st, ins, gen, n, device)
        run_items(st, b, (tok,), lib)
        return b.regs[0, A_].T.long(), b.c[0].long()


def repeat_add(ops, st, lib, D, v, times, gen, device):
    """D + times v by the learner's own addition (times a count per row; the additions repeated, mul.ladder)."""
    D = D.clone()
    for k in range(int(times.max()) if times.numel() else 0):
        more = torch.nonzero(times > k).flatten()
        if more.numel():
            D[more] = ops.run("add", "pair", st, lib, [D[more], v[more]], gen, device)[0]
    return D


class Draw:
    """One draw of the data's numbers for a pattern at a setting: n rows' top parts at the pattern's width (x' and N',
    so x' <= N'), another row's x' (o), the digits 0..B-1 over the probes (d), zero; the composite terms are built on
    demand by the learner's routines (NotPosable when one is missing)."""

    def __init__(self, pattern, st, n, rows, gen, device):
        kinds, grow, filt = PATTERNS[pattern]
        P, B = st.P, st.B
        Nd, Xd = rows
        w = width_for(grow, st)
        pick = torch.randint(0, Nd.shape[0], (4 * n,), generator=gen, device=device)
        Nt, xt = top_parts(Nd[pick], Xd[pick], w)
        eq = (Nt == xt).all(1)
        idx = torch.arange(n, device=device)
        if filt == "ne":
            idx = torch.nonzero(~eq[:3 * n]).flatten()[:n]
        self.n = idx.numel()
        self.x, self.N = pad(xt[idx], P), pad(Nt[idx], P)
        self.o = pad(xt[3 * n:3 * n + self.n], P)
        self.eq = eq[idx]
        self.dv = torch.arange(self.n, device=device) % B
        self.d = digit_string(self.dv, P)
        self.z = torch.zeros_like(self.x)
        self.kind = torch.arange(self.n, device=device) % len(kinds)
        self.st, self.gen, self.device, self.pattern = st, gen, device, pattern
        self.cache = {}

    def term(self, name, ops, lib):
        if name in self.cache:
            return self.cache[name]
        st, gen, dev = self.st, self.gen, self.device
        if name in ("x", "z", "o", "d"):
            v = getattr(self, name)
        elif name == "n":
            v = self.N
        elif name == "x+n":
            v = ops.run("add", "pair", st, lib, [self.x, self.N], gen, dev)[0]
        elif name == "x+dn":
            v = repeat_add(ops, st, lib, self.x, self.N, self.dv, gen, dev)
        elif name == "x+on":
            v = repeat_add(ops, st, lib, self.x, self.N, count_of(self.o), gen, dev)
        elif name == "Bx+d":
            v = torch.cat([self.dv[:, None], self.x[:, :st.P - 1]], 1)      # x's digits moved up one place, d below
        else:
            raise ValueError(name)
        self.cache[name] = v
        return v

    def operands(self, ops, lib):
        """The operands' digits [n, P] per operand position, each probe of its kind."""
        kinds = PATTERNS[self.pattern][0]
        out = []
        for j in range(len(kinds[0])):
            v = torch.zeros_like(self.x)
            for k, kd in enumerate(kinds):
                m = self.kind == k
                if bool(m.any()):
                    v[m] = self.term(kd[j], ops, lib)[m]
            out.append(v)
        return out


def law_target(law, d, ops, lib):
    """The law's target on a draw: {"A": digits [n, P], "Am": mask [n]} and/or {"c": [n], "cm": mask} and/or a "check"
    (a function of the candidates' A digits [S, n, P] -> bool [S]), from facts and the learner's routines; None when
    the law says nothing on this draw (or its answers do not vary)."""
    k, eq, P, dev, st, gen = d.kind, d.eq, d.st.P, d.device, d.st, d.gen
    allm = torch.ones(d.n, dtype=torch.bool, device=dev)
    x, N, z = d.x, d.N, d.z
    pat = d.pattern
    t = None
    if pat == "pair":                          # kinds 0 (x, N), 1 (N, x), 2 (x, x), 3 (x, 0)
        xz = (x == 0).all(1)
        if law == "lt":
            t = {"c": torch.where(k == 0, (~eq).long(), 0), "cm": allm}
        elif law == "eq":
            t = {"c": torch.where(k <= 1, eq.long(), torch.where(k == 2, 1, xz.long())), "cm": allm}
        elif law == "min":
            t = {"A": torch.where((k == 3)[:, None], z, x), "Am": allm}
        elif law == "max":
            t = {"A": torch.where((k <= 1)[:, None], N, x), "Am": allm}
        elif law == "red":
            t = {"A": torch.where(((k == 0) & ~eq)[:, None], x, z), "Am": (k == 0) | (k == 2)}
        elif law == "sub":
            t = {"A": torch.where((k == 3)[:, None], x, z), "Am": k >= 2}
        elif law == "add":
            if not (ops.has("red", "pair") and ops.has("lt", "pair")):
                raise NotPosable("add needs red and lt")
            sel = torch.nonzero((k == 0) & ~eq).flatten()

            def check(A_out, lib):             # sub.inverse: red(a + b, b) = a and a + b >= b, for a < b
                S = A_out.shape[0]
                if sel.numel() == 0:
                    return torch.ones(S, dtype=torch.bool, device=dev)
                flat = A_out[:, sel].reshape(-1, P).long()
                Ns = N[sel].repeat(S, 1)
                ok = torch.zeros(S, dtype=torch.bool, device=dev)
                for i in range(ops.count("red", "pair")):     # any practised red that confirms it
                    r = ops.run("red", "pair", st, lib, [flat, Ns], gen, dev, i)[0].view(S, -1, P)
                    ok |= (r == x[sel][None]).all(-1).all(-1)
                c = ops.run("lt", "pair", st, lib, [flat, Ns], gen, dev)[1].view(S, -1)
                return ok & (c == 0).all(-1)
            t = {"A": x, "Am": k == 3, "check": check}
    elif pat in ("x+n", "x+dn", "x+on"):
        q = {"x+n": digit_string(torch.ones(d.n, dtype=torch.long, device=dev), P), "x+dn": d.d, "x+on": d.o}[pat]
        d.term(pat, ops, lib)                  # the pattern itself needs the addition
        if law == "red":
            t = {"A": torch.where(eq[:, None], z, x), "Am": allm}
        elif law == "div":
            t = {"A": q, "Am": ~eq}
        elif law == "sub" and pat == "x+n":
            t = {"A": x, "Am": allm}
        elif law == "lt" and pat == "x+dn":
            t = {"c": ((d.dv == 0) & ~eq).long(), "cm": allm}
    elif pat == "Bx+d":
        if law == "red":
            v = d.term("Bx+d", ops, lib)
            src = "x+dn" if ops.has("red", "x+dn") else "x+on"
            t = {"A": ops.run("red", src, st, lib, [v, N], gen, dev)[0], "Am": allm}      # red.step, by the chain
    elif pat == "x,d":
        if law == "mul":
            t = {"A": repeat_add(ops, st, lib, z, x, d.dv, gen, dev), "Am": allm}
    elif pat == "x,o":
        if law == "mul":
            t = {"A": repeat_add(ops, st, lib, z, x, count_of(d.o), gen, dev), "Am": allm}
    elif pat == "o,x,d":
        if law == "acc.mul":
            m = repeat_add(ops, st, lib, z, x, d.dv, gen, dev)
            t = {"A": ops.run("add", "pair", st, lib, [m, d.o], gen, dev)[0], "Am": allm}
        elif law == "acc.add":
            s = ops.run("add", "pair", st, lib, [x, d.d], gen, dev)[0]
            t = {"A": ops.run("add", "pair", st, lib, [s, d.o], gen, dev)[0], "Am": allm}
    if t is None:
        return None
    if "A" in t:                               # the answers must vary where the law speaks (else nothing to learn)
        m = t["Am"]
        if int(m.sum()) < 2 and "check" not in t:
            return None
        if "check" not in t and bool((t["A"][m] == t["A"][m][:1]).all()):
            return None
    if "c" in t:
        cm = t["cm"]
        if bool((t["c"][cm] == t["c"][cm][:1]).all()):
            return None
    return t


class Goal:
    """One law's goal on a root's probes (one machine per setting): per setting its targets with masks."""

    def __init__(self, name, targets):
        self.name, self.targets = name, targets           # targets: one dict per setting (law_target's)

    def test(self, batches, lib=None):
        """Which states reach the goal (a check runs the learner's routines: lib, the library of the moment)."""
        dev = batches[0].regs.device
        ok = torch.ones(batches[0].S, dtype=torch.bool, device=dev)
        for b, t in zip(batches, self.targets):
            if "A" in t:
                m = (b.regs[:, A_] == t["A"].T.to(torch.uint8)[None]).all(1)          # [S, n]
                ok &= (m | ~t["Am"][None]).all(-1)
            if "c" in t:
                ok &= ((b.c.long() == t["c"][None]) | ~t["cm"][None]).all(-1)
            if "check" in t and bool(ok.any()):
                idx = torch.nonzero(ok).flatten()
                ok[idx[~t["check"](b.regs[idx, A_].permute(0, 2, 1), lib)]] = False
        return ok

    def fits(self, batches):
        """How many probes each state gets right (A on the masked probes; c too), over every setting."""
        dev = batches[0].regs.device
        f = torch.zeros(batches[0].S, dtype=torch.long, device=dev)
        for b, t in zip(batches, self.targets):
            r = torch.ones(b.S, t["Am"].numel() if "Am" in t else t["cm"].numel(), dtype=torch.bool, device=dev)
            if "A" in t:
                r &= (b.regs[:, A_] == t["A"].T.to(torch.uint8)[None]).all(1) & t["Am"][None]
            if "c" in t:
                r &= (b.c.long() == t["c"][None]) & t["cm"][None]
            f += r.sum(-1)
        return f


class Root:
    """A pattern bound to registers at the search settings: the probes every law on it shares, and the goals."""

    def __init__(self, pattern, binding, settings, NP, rows, ops, lib, device, seed):
        self.pattern, self.binding = pattern, binding
        self.key = f"{pattern}@{binding_name(binding)}"
        gen = gen_on(device, seed + zlib.crc32(self.key.encode()))
        self.draws, self.roots = [], []
        for st in settings:
            d = Draw(pattern, st, NP, rows, gen, device)
            ins = {r: v for r, v in zip(binding, d.operands(ops, lib))}
            self.draws.append(d)
            self.roots.append(fresh(st, ins, gen, d.n, device))

    def goals(self, ops, lib, laws):
        out = []
        for law in laws:
            try:
                ts = [law_target(law, d, ops, lib) for d in self.draws]
            except NotPosable:
                continue
            if all(t is not None for t in ts):
                out.append(Goal(law, ts))
        return out


class Acceptor:
    """A candidate's acceptance: the same law, pattern and binding on fresh rows (60 per setting, seed 11) at the
    verify settings; a program passes if it reaches the goal at every one. Built once per candidate."""

    def __init__(self, law, root, verify, rows, ops, lib, device, n=60, seed=11):
        self.cache = {}
        self.probs = []
        r = Root(root.pattern, root.binding, verify, n, rows, ops, lib, device, seed)
        g = r.goals(ops, lib, [law])
        self.ok = bool(g)
        self.ops, self.version = ops, ops.version          # a check's verdicts change when routines join
        self.checked = bool(g) and any("check" in t for t in g[0].targets)
        if g:
            for st, b, t in zip(verify, r.roots, g[0].targets):
                self.probs.append((st, b, Goal(law, [t])))

    def many(self, progs, lib):
        if self.checked and self.version != self.ops.version:
            self.cache, self.version = {}, self.ops.version
        keys = [show(p) for p in progs]
        live, seen = [], set()
        for i, k in enumerate(keys):
            if k not in self.cache and k not in seen:
                seen.add(k)
                live.append(i)
        if not self.ok:
            for i in live:
                self.cache[keys[i]] = False
            live = []
        for st, root, goal in self.probs:
            if not live:
                break
            dev = root.regs.device
            b = root.take(torch.zeros(len(live), dtype=torch.long, device=dev))
            b = apply_items(st, b, torch.arange(len(live), device=dev), [("REP", ONE, tuple(progs[i])) for i in live],
                            lib)
            ok = goal.test([b], lib).tolist()
            for i, o in zip(live, ok):
                if not o:
                    self.cache[keys[i]] = False
            live = [i for i, o in zip(live, ok) if o]
        for i in live:
            self.cache[keys[i]] = True
        return [self.cache.get(k, False) for k in keys]


# ===================================================================================================== many goals
class _Res:
    def __init__(self, parents, found_cost, goals):
        self.parents, self.found_cost, self.goals = parents, found_cost, goals


WHOLE = ("LDA.X", "LDA.Y", "LDA.N", "LDA.S", "STA.X", "STA.Y", "STA.S", "CLA", "CLC", "CLK", "SHL.A", "SHL.X",
         "SHL.Y", "SHR.A", "SHR.X", "SHR.Y", "SEL.S", "BRG.X")    # the instructions that act on whole registers
SOURCES = ("A", "X", "Y", "N", "Z")       # a routed input: a register as it was before the call, or zero


def route(binding, srcs):
    """Straight-line code that fills a routine's input registers (binding, one per operand) from sources (srcs: a
    register as it was before the code, or Z for zero), A as the carrier, the A input last; None when no order of the
    moves does it on this machine (N is read-only; a source overwritten before it is read)."""
    rest = [i for i in range(len(binding)) if binding[i] != "A"]
    for order in itertools.permutations(rest):
        code, held = [], {r: r for r in "AXYNS"}
        ok = True
        for i in list(order) + [i for i in range(len(binding)) if binding[i] == "A"]:
            dst, src = binding[i], srcs[i]
            if held[dst] == src:
                continue
            if dst == "N":
                ok = False
                break
            if held["A"] != src:
                where = [r for r in "AXYNS" if held[r] == src]
                if src == "Z":
                    code.append("CLA")
                elif where:
                    code.append(f"LDA.{where[0]}")
                else:
                    ok = False
                    break
                held["A"] = src
            if dst != "A":
                code.append(f"STA.{dst}")
                held[dst] = src
        if ok and all(held[b] == s_ for b, s_ in zip(binding, srcs)):
            return code
    return None


def routines_in(items, lib, out=None, closure=False):
    """The routines a program calls (through merges and loops); with closure, also the routines those call."""
    out = set() if out is None else out
    for it in items:
        if isinstance(it, tuple):
            if it[1] in lib:
                routines_in(lib[it[1]], lib, out, closure)
            routines_in(it[2], lib, out, closure)
        elif it.startswith("T."):
            if closure and it not in out and it in lib:
                out.add(it)
                routines_in(lib[it], lib, out, closure)
            out.add(it)
        elif it in lib:
            routines_in(lib[it], lib, out, closure)
    return out


def unwrap(prog):
    """A program with its ONE[...] items (the search by calls' routed calls) written out: the same instructions, in a
    form every machine runs (the prediction path's has no ONE)."""
    out = []
    for it in prog:
        if isinstance(it, tuple) and it[1] == ONE:
            out.extend(unwrap(it[2]))
        else:
            out.append(it)
    return tuple(out)


_SALT = torch.randint(-(1 << 62), 1 << 62, (8192,), generator=torch.Generator().manual_seed(4242), dtype=torch.int64)
_SALT_ON = {}


def bfs_multi(roots, settings, goals, accepts, classes, lib, keep=False, fits=None, **kw):
    """bfs_roots with one root: libcore.bfs with many goals sharing it (each goal's goal states listed one spelling
    each, fewest top-level items first, then fewest primitives, tested by its acceptor at the first cost where it has
    any; the search goes on while any goal is open). Returns ([per goal: dict(found_cost, accepted, aliases)], level
    sizes, stop[, held levels, parents]); fits, a list, gets the best fit of goals[0] at every level."""
    out, sizes, stops, levels, _, parents = bfs_roots([roots], settings, [(0, g) for g in goals], accepts, classes, lib,
                                                      fits=fits, **kw)
    return (out, sizes[0], stops[0], levels, parents) if keep else (out, sizes[0], stops[0])


def bfs_roots(roots, settings, goals, accepts, classes, lib, max_cost=7, max_states=1_000_000, deadline=None,
              chunk=1 << 21, fuse=1, batch=256, spell=8, max_goals=50_000, fits=None):
    """bfs_multi for many roots in one search: roots [R][per setting Batch, S = 1] (the same probe shapes), goals
    [(root index, Goal)], accepts per goal. Every root keeps its own search: its states' keys carry a salt of its own,
    so no two roots' states merge; it has its own budget (a level whose candidates would pass 1.5 max_states streams that
    root's goal states only, and the root stops after it; a level that stores more than max_states also ends it); it
    stops once its goals are closed. Each level's work is shared: one apply_items over every live root's rows.
    Returns ([per goal: dict(found_cost, accepted, aliases)], [per root: level sizes], [per root: stop], held levels,
    their root ids, parents); fits, a list, gets the best fit of goals[0] (on its root) at every level."""
    dev = roots[0][0].regs.device
    R = len(roots)
    if dev not in _SALT_ON:
        _SALT_ON[dev] = _SALT.to(dev)
    salt = _SALT_ON[dev][:R]
    lv0 = [Batch.concat([roots[r][si] for r in range(R)]) for si in range(len(settings))]
    rid0 = torch.arange(R, device=dev)
    levels, rids = [lv0], [rid0]
    seen = torch.sort(keys_of(lv0) ^ salt[rid0]).values
    parents = [None]
    out = [dict(found_cost=None, accepted=None, aliases=[]) for _ in goals]
    sizes, stops = [[1] for _ in range(R)], [""] * R
    open_by = {r: [] for r in range(R)}
    for j, (r, g) in enumerate(goals):
        if bool(g.test([b.take(torch.tensor([r], device=dev)) for b in lv0], lib)[0]):
            out[j].update(found_cost=0, accepted=[()])
        else:
            open_by[r].append(j)
    stopped = torch.tensor([not open_by[r] for r in range(R)], dtype=torch.bool, device=dev)
    if fits is not None:
        fits.append(int(goals[0][1].fits([b.take(torch.tensor([goals[0][0]], device=dev)) for b in lv0]).max()))
    by_cost = {}
    for ci, cl in enumerate(classes):
        by_cost.setdefault(cl.cost, []).append(ci)
    by_cost = {c: torch.tensor(v, dtype=torch.long, device=dev) for c, v in by_cost.items()}
    late = False
    for d in range(1, max_cost + 1):
        if bool(stopped.all()) or late:
            break
        lastr = torch.full((R,), d == max_cost, dtype=torch.bool, device=dev)
        n_cand = torch.zeros(R, dtype=torch.long, device=dev)
        n_fresh = torch.zeros(R, dtype=torch.long, device=dev)
        ck, cc, cs, ci_, crid, chunks = [], [], [], [], [], []
        gk = {j: [] for r in range(R) for j in open_by[r]}
        best = 0
        live_src = {}
        for sl in range(d):                          # each held level's rows of roots still searching
            if levels[sl] is not None:
                live_src[sl] = torch.nonzero(~stopped[rids[sl]]).flatten()

        def blocks(d=d):
            for c, cis in by_cost.items():
                sl = d - c
                if sl < 0 or sl not in live_src or live_src[sl].numel() == 0:
                    continue
                al = live_src[sl]
                for s0 in range(0, al.numel(), chunk):
                    rows_ = al[s0:s0 + chunk]
                    srcs = [b.take(rows_) for b in levels[sl]]
                    i_step = max(1, chunk // rows_.numel())
                    for i0 in range(0, cis.numel(), i_step):
                        yield cis[i0:i0 + i_step], sl, rows_, srcs

        def goal_hits(sub, sub_rid, keys_sub):
            hit = torch.zeros(sub_rid.numel(), dtype=torch.bool, device=dev)
            for r in torch.unique(sub_rid).tolist():
                if not open_by[r]:
                    continue
                sel = torch.nonzero(sub_rid == r).flatten()
                part = [b.take(sel) for b in sub]
                for j in open_by[r]:
                    g = goals[j][1].test(part, lib)
                    if bool(g.any()):
                        if keys_sub is not None:
                            gk[j].append(keys_sub[sel][g])
                        hit[sel[g]] = True
            return hit
        work = blocks()
        nxt = next(work, None)
        while nxt is not None:
            if _late(deadline):
                late = True
                break
            group, n_rows = [], 0
            while nxt is not None and (not group or n_rows + nxt[0].numel() * nxt[2].numel() <= fuse * chunk):
                group.append(nxt)
                n_rows += nxt[0].numel() * nxt[2].numel()
                nxt = next(work, None)
            parts, its, srcrows = [], [], [[] for _ in settings]
            for ii, sl, rows_, srcs in group:
                ns, k = rows_.numel(), ii.numel()
                its.append(torch.arange(k, device=dev).repeat_interleave(ns) + len(parts))
                parts.extend(classes[i].rep for i in ii.tolist())
                rs = torch.arange(ns, device=dev).repeat(k)
                for si, b in enumerate(srcs):
                    srcrows[si].append(b.take(rs))
            ri_all = its[0] if len(its) == 1 else torch.cat(its)
            outs_all = [apply_items(st, r[0] if len(r) == 1 else Batch.concat(r), ri_all, parts, lib)
                        for st, r in zip(settings, srcrows)]
            del srcrows
            off = 0
            for ii, sl, rows_, srcs in group:
                ns, k = rows_.numel(), ii.numel()
                outs = outs_all if len(group) == 1 else [o.take(slice(off, off + k * ns)) for o in outs_all]
                off += k * ns
                ri = torch.arange(k, device=dev).repeat_interleave(ns)
                rs = torch.arange(ns, device=dev).repeat(k)
                rid_rows = rids[sl][rows_][rs]
                keys = keys_of(outs) ^ salt[rid_rows]
                m = _fresh(seen, keys)
                if not bool(m.any()):
                    continue
                idx = torch.nonzero(m).flatten()
                sub = [b.take(idx) for b in outs]
                sub_rid = rid_rows[idx]
                n_fresh += torch.bincount(sub_rid, minlength=R)
                del outs
                if fits is not None:
                    sel = torch.nonzero(sub_rid == goals[0][0]).flatten()
                    if sel.numel():
                        best = max(best, int(goals[0][1].fits([b.take(sel) for b in sub]).max()))
                hit = goal_hits(sub, sub_rid, keys[idx])
                keep = ~lastr[sub_rid] | hit
                if not bool(keep.all()):
                    kk = torch.nonzero(keep).flatten()
                    idx, sub, sub_rid = idx[kk], [b.take(kk) for b in sub], sub_rid[kk]
                if idx.numel() == 0:
                    continue
                ck.append(keys[idx])
                cc.append(ii[ri[idx]])
                cs.append(torch.full((idx.numel(),), sl, dtype=torch.long, device=dev))
                ci_.append(rows_[rs[idx]])
                crid.append(sub_rid)
                chunks.append(sub)
                n_cand += torch.bincount(sub_rid, minlength=R)
                over = torch.nonzero(~lastr & (n_cand > 1.5 * max_states)).flatten()
                if over.numel():                     # these roots stream the rest of this level: goal states only
                    lastr[over] = True
                    for r in over.tolist():
                        stops[r] = f"budget: level {d} streamed (over {max_states} states)"
                    for jj in range(len(chunks)):
                        o_ = torch.isin(crid[jj], over)
                        if not bool(o_.any()):
                            continue
                        h = goal_hits(chunks[jj], crid[jj], None)
                        kk = torch.nonzero(~o_ | h).flatten()
                        ck[jj], cc[jj], cs[jj], ci_[jj], crid[jj] = (t[kk] for t in (ck[jj], cc[jj], cs[jj], ci_[jj],
                                                                                    crid[jj]))
                        chunks[jj] = [b.take(kk) for b in chunks[jj]]
                    n_cand = torch.bincount(torch.cat(crid), minlength=R) if crid else n_cand
            del outs_all
        if late or _late(deadline):
            late = True
            for r in range(R):
                if not bool(stopped[r]):
                    stops[r] = stops[r] or f"deadline at level {d}"
        if ck:
            allk = torch.cat(ck)
            allr = torch.cat(crid)
            uk, inv = torch.unique(allk, sorted=True, return_inverse=True)
            pos = torch.arange(allk.numel(), device=dev)
            first = torch.full((uk.numel(),), allk.numel(), dtype=torch.long, device=dev).scatter_reduce(
                0, inv, pos, reduce="amin")
            sfirst = torch.sort(first).values
            rank = torch.searchsorted(sfirst, first)
            child = rank[inv]
            order = torch.sort(child, stable=True).indices
            parents.append((child[order], torch.cat(cc)[order], torch.cat(cs)[order], torch.cat(ci_)[order]))
            urid = allr[sfirst]                        # each stored state's root, in rank order
            per_root = torch.bincount(urid, minlength=R)
            parts_, offs = [[] for _ in settings], 0
            for jj in range(len(chunks)):
                ch, n = chunks[jj], chunks[jj][0].S
                lo_, hi_ = int(torch.searchsorted(sfirst, offs)), int(torch.searchsorted(sfirst, offs + n))
                rows = sfirst[lo_:hi_] - offs
                for si in range(len(settings)):
                    parts_[si].append(ch[si].take(rows))
                offs += n
            levels.append([Batch.concat(p) for p in parts_])
            rids.append(urid)
            chunks = None
        else:
            uk = torch.zeros(0, dtype=torch.int64, device=dev)
            rank = torch.zeros(0, dtype=torch.long, device=dev)
            per_root = torch.zeros(R, dtype=torch.long, device=dev)
            parents.append(None)
            levels.append(None)
            rids.append(torch.zeros(0, dtype=torch.long, device=dev))
        if fits is not None:
            fits.append(best)
        for r in range(R):
            if bool(stopped[r]):
                continue
            sizes[r].append(int(n_fresh[r]) if bool(lastr[r]) else int(per_root[r]))
            if bool(lastr[r]) and not stops[r]:
                stops[r] = f"exhausted cost {max_cost}"
            if int(per_root[r]) > max_states and not stops[r]:
                stops[r] = f"budget: level {d} has {int(per_root[r])} > {max_states} states"
        for r in range(R):
            if bool(stopped[r]):
                continue
            for j in list(open_by[r]):
                if not gk.get(j):
                    continue
                gs = torch.unique(torch.cat(gk[j]))
                gidx = rank[torch.searchsorted(uk, gs)].tolist()
                res = _Res(parents, d, gidx)
                cand = []
                for gi, g in enumerate(gidx[:max_goals]):
                    if gi % 256 == 0 and _late(deadline):
                        break
                    f = programs(_OneGoal(res, g), classes, limit=1)
                    if f:
                        cand.append((len(f[0]), cost(expand(f[0], lib)), f[0], g))
                cand.sort(key=lambda c: (c[0], c[1]))
                progs = [c[2] for c in cand]
                good = []
                for i0 in range(0, len(cand), batch):
                    if _late(deadline):
                        break
                    good = [c for c, o in zip(cand[i0:i0 + batch], accepts[j](progs[i0:i0 + batch], lib)) if o]
                    if good:
                        break
                if good:
                    acc = [c[2] for c in good]
                    if spell > 1:
                        more = [q for c in good for q in programs(_OneGoal(res, c[3]), classes, limit=spell)[1:]]
                        acc += [q for q, o in zip(more, accepts[j](more, lib) if more else []) if o]
                    out[j].update(found_cost=d, accepted=acc)
                    open_by[r].remove(j)
                else:
                    out[j]["aliases"].append((d, [show(p) for p in progs[:5]]))
            if stops[r] or not open_by[r] or bool(lastr[r]):
                stopped[r] = True
        if ck:
            seen = torch.sort(torch.cat([seen, uk])).values
    for r in range(R):
        if open_by[r] and not stops[r]:
            stops[r] = f"exhausted cost {max_cost}"
    return out, sizes, stops, levels, rids, parents


# ===================================================================================================== the learner
def side_effects(prog, lib, st, root, binding):
    """Does prog leave its input registers (the binding's, but A, the answer) as they were, and how many registers
    other than A does it change, on these probes? (bar3/learner.py's rule for routines: no side effects on inputs.)"""
    b = root.copy()
    run_items(st, b, prog, lib)
    changed = [(b.regs[0, i] != root.regs[0, i]).any().item() for i in range(5)]
    keeps = all(not changed[RI[r]] for r in binding if r != "A")
    return keeps, sum(changed[1:])


class SelfClimb:
    """The self-practising climb as a state machine a training loop drives (step: one unit of work), libcore.Climb's
    interface. Each level (bar 3's round):
      pose    every law of the book on every pattern and binding whose construction works now (the routines it needs
              exist), minus those already practised: the pool, grouped by root;
      probe   one breadth-first search per root (all its laws at once) over the composition vocabulary (the routines
              and the primitives, loops of bodies up to comp_body), at probe_budget; a root posed for the first time is
              searched once more over the primitives alone at fallback_budget for its open laws;
      choose  trivial (cost <= 1), learnable (cost >= 2, not an alias of a token, not a duplicate of an earlier
              learnable by what it does on random machines), unsolved; then the rule: "lp" (the K costliest learnable,
              ties in pool order), "ct2a2" (the cheaper pairs signal: the target probed with every learnable's routine,
              and with every pair whose member is among the 2 hardest; the best pairs' members), "all";
      sleep   the picks' programs join the corpus, BPE merges, each pick's routine (among its shortest programs, one
              that leaves its inputs as they were and changes the fewest registers, then Occam by expansion and steps);
      target  the round: libcore's search by composition at full size with the answers as its goal.
    """

    def __init__(self, cells, device, P_round, rule="lp", K=4, NP=16, probe_budget=1_000_000,
                 fallback_budget=1_000_000, deep_budget=5_800_000, deep_k=8, group=16, deep_group=4, max_body=3,
                 comp_body=3, max_cost=7,
                 min_count=2.0, max_merges=12,
                 levels=12, round_budget=4_000_000, round_cost=5, look_budget=300_000, settings=((10, 1), (10, 2)),
                 verify=((10, 1), (10, 2), (10, 3)), chunk=1 << 21, fuse=1, search_s=None, seed=0, log=None,
                 patterns=None, laws=None, round_retry=0, target=None, macro_after=12, macro_cost=2,
                 deep_by="root", deep_focus=True):
        self.cells, self.device, self.P_round = cells, device, P_round
        self.rule, self.K, self.NP = rule, K, NP
        self.probe_budget, self.fallback_budget = probe_budget, fallback_budget
        self.deep_budget, self.deep_k, self._fit = deep_budget, deep_k, {}
        self.group, self.deep_group = group, deep_group
        self._variants = rule not in ("lpc", "lpca")    # pose practised concepts' other bindings (concept rules: no)
        self._want_variants = False
        self.max_body, self.comp_body, self.max_cost = max_body, comp_body, max_cost
        self.min_count, self.max_merges, self.levels = min_count, max_merges, levels
        self.round_budget, self.round_cost, self.look_budget = round_budget, round_cost, look_budget
        self.settings = [Setting(B, W, cells) for B, W in settings]
        self.verify = [Setting(B, W, cells) for B, W in verify]
        self.chunk, self.fuse, self.search_s, self.seed = chunk, fuse, search_s, seed
        self.patterns = list(PATTERNS) if patterns is None else list(patterns)
        self.laws = list(LAW_BOOK) if laws is None else list(laws)
        self.lib, self.corpus, self.ops = {}, {}, Ops()
        self.practiced = {}                     # candidate name -> level
        self.level, self.phase, self.queue = 0, "pose", []
        self.records, self.history, self.events = [], [], []
        self.rounds, self.done = [], False
        self.fallback_seen = set()              # roots whose primitives-alone search ran (it never changes)
        self._roots, self._accs = {}, {}        # key -> Root, (law, key) -> Acceptor (the same probes every level)
        self._sig_cache, self._comp, self._prims = {}, None, None
        self.pool, self.results, self.picks = [], {}, []
        self.tokens_sig = None
        self.looks, self._base_look, self._scores = [], None, {}
        self.round_retry, self._idle = round_retry, 0
        self.macro_after, self.macro_cost = macro_after, macro_cost     # the round by calls: past this many routines
        self.deep_by, self.deep_focus = deep_by, deep_focus
        self._focus, self._focus_classes = {}, {}        # root -> law -> its closest attempt's routines
        self.target, self._near, self.near_log = target, [], []      # a small target (the tiny core's oracle arm)
        self.log = log

    # ------------------------------------------------------------------ vocabularies
    def routines(self):
        return tuple(sorted(k for k in self.lib if k.startswith("T.")))

    def comp_classes(self, deadline=None):
        r = self.routines()
        if self._comp is None or self._comp[0] != r:
            self._comp = (r, enumerate_items(list(PRIMS) + list(r), self.lib, self.settings, self.device,
                                             max_body=self.comp_body, deadline=deadline, cache=self._sig_cache,
                                             chunk=self.chunk * self.fuse))
        return self._comp[1]

    def prim_classes(self, deadline=None):
        if self._prims is None:
            self._prims = enumerate_items(list(PRIMS), {}, self.settings, self.device, max_body=self.max_body,
                                          deadline=deadline, cache=self._sig_cache, chunk=self.chunk * self.fuse)
        return self._prims

    def signature(self, prog, lib):
        if self.tokens_sig is None or self.tokens_sig[0] != tuple(sorted(lib)):
            gen = gen_on(self.device, 7)
            tests = [random_states(st, 24, 6, gen, self.device) for st in self.settings]
            toks = list(PRIMS) + [k for k in lib if k.startswith("T.")]
            sigs = _signatures(self.settings, tests, [("REP", ONE, (t,)) for t in toks], lib, chunk=self.chunk)
            self.tokens_sig = (tuple(sorted(lib)), tests, set(sigs))
        tests = self.tokens_sig[1]
        return _signatures(self.settings, tests, [("REP", ONE, tuple(prog))], lib, chunk=self.chunk)[0]

    # ------------------------------------------------------------------ pose
    def pose(self, rows):
        pool, roots = [], []
        for pat in self.patterns:
            laws = [l for l in self.laws if pat in LAW_BOOK[l][1]]
            if not laws:
                continue
            arity = len(PATTERNS[pat][0][0])
            for bd in bindings(arity):
                key = f"{pat}@{binding_name(bd)}"
                r = self._roots.get(key)
                if r is None:
                    try:
                        r = Root(pat, bd, self.settings, self.NP, rows, self.ops, self.lib, self.device, self.seed)
                    except NotPosable:
                        continue
                    self._roots[key] = r
                goals = [g for g in r.goals(self.ops, self.lib, laws) if f"{g.name}/{r.key}" not in self.practiced
                         and (self._variants or (g.name, pat) not in self.concepts_done())]
                if not goals:
                    continue
                r.goal_list = goals
                roots.append(r)
                for g in goals:
                    pool.append(f"{g.name}/{r.key}")
        if self.target is not None:                     # the target's restrictions, projections, near misses
            for r, goals in self.target.pose(self, rows, self._near):
                goals = [g for g in goals if f"{g.name}/{r.key}" not in self.practiced]
                if goals:
                    r.goal_list = goals
                    roots.append(r)
                    pool.extend(f"{g.name}/{r.key}" for g in goals)
        return pool, roots

    # ------------------------------------------------------------------ probe
    def acceptors(self, root, rows):
        accs = []
        for g in root.goal_list:
            if (g.name, root.key) not in self._accs:
                self._accs[(g.name, root.key)] = (
                    self.target.acceptor(self, rows, root, g.name) if hasattr(root, "kind") else
                    Acceptor(g.name, root, self.verify, rows, self.ops, self.lib, self.device))
            accs.append(self._accs[(g.name, root.key)])
        return accs

    def probe(self, group, rows, deadline=None, deep=False):
        """A group of roots' probe (bfs_roots: one search, every root its own budget): their laws at once over the
        composition vocabulary at probe_budget, then (a root's first time, once routines exist) the open laws over the
        primitives alone at fallback_budget (that search never changes); with deep, the open laws again at
        deep_budget. Records each open law's closeness (how much of the gap left by the best single instruction the
        closest held state closes): the gate for the deep probes."""
        t0 = time.time()
        dl = deadline
        if self.search_s:
            dl = min(deadline or float("inf"), t0 + self.search_s)
        accs = {r.key: self.acceptors(r, rows) for r in group}
        out = {r.key: {} for r in group}
        if deep:
            for r in group:
                for g in r.goal_list:
                    prev = self.results.get(f"{g.name}/{r.key}")
                    if prev is not None:
                        out[r.key][g.name] = (prev, prev["how"])
        runs = []

        def run(how, classes, lib, cap, members):
            """members: [(root, [goal indices])]."""
            t1 = time.time()
            goals, gacc, owner = [], [], []
            for ri, (r, idx) in enumerate(members):
                for i in idx:
                    goals.append((ri, r.goal_list[i]))
                    gacc.append(accs[r.key][i].many)
                    owner.append(r)
            res, sizes, stops, levels, rids, parents = bfs_roots([r.roots for r, _ in members], self.settings, goals,
                                                              gacc, classes, lib, max_cost=self.max_cost,
                                                              max_states=cap, deadline=dl, chunk=self.chunk,
                                                              fuse=self.fuse)
            fit = {r.key: {} for r, _ in members}
            for (ri, g), x, r in zip(goals, res, owner):
                if x["found_cost"] is None or g.name not in out[r.key]:
                    if x["found_cost"] is not None or g.name not in out[r.key]:
                        out[r.key][g.name] = (x, how)
                if x["found_cost"] is None:             # closeness: the share of the gap single instructions leave
                    tot = sum(int(t["Am"].sum()) if "Am" in t else int(t["cm"].sum()) for t in g.targets)
                    f, best = [], (-1, None, None)
                    for d, (lv, rd) in enumerate(zip(levels, rids)):
                        sel = torch.nonzero(rd == ri).flatten() if lv is not None else None
                        if sel is None or not sel.numel():
                            f.append(0)
                            continue
                        ff = g.fits([b.take(sel) for b in lv])
                        f.append(int(ff.max()))
                        if d and f[-1] > best[0]:
                            best = (f[-1], d, int(sel[int(ff.argmax())]))
                    if best[1] is not None:              # the closest held state's program: its routines are the
                        pr = programs(_OneGoal(_Res(parents, best[1], [best[2]]), best[2]), classes, limit=1)
                        self._focus.setdefault(r.key, {})[g.name] = (
                            routines_in(pr[0], lib, closure=self.deep_focus == "closure") if pr else set())
                    base = max(f[:2])                   # (a law whose goal states acceptance rejected is not out
                    fit[r.key][g.name] = ((max(f) - base) / (tot - base) if tot > base and not x["aliases"]  # of reach)
                                          else 0.0)
                elif x["found_cost"] is not None:
                    out[r.key][g.name] = (x, how)
            del levels
            runs.append(dict(how=how, roots=[r.key for r, _ in members], seconds=round(time.time() - t1, 1),
                             sizes={r.key: sz for (r, _), sz in zip(members, sizes)},
                             stops={r.key: st for (r, _), st in zip(members, stops)},
                             fit={k: {g: round(v, 3) for g, v in fv.items()} for k, fv in fit.items()}))
            return fit
        open_of = lambda r: [i for i, g in enumerate(r.goal_list)
                             if g.name not in out[r.key] or out[r.key][g.name][0]["found_cost"] is None]
        try:                                            # (the vocabularies are shared by every root: built under the
            comp = self.comp_classes(deadline)          #  training's deadline; one search's limit starts after)
            if self.search_s:
                dl = min(deadline or float("inf"), time.time() + self.search_s)
            if deep and self.deep_focus:                # deeper where the closest attempt was, with what it used:
                by = {}                                 # the primitives and its routines (a big library leaves even
                for r in group:                         # a 5.8M search no reach past cost 4)
                    fs = set()
                    for i in open_of(r):
                        name = r.goal_list[i].name
                        if self.deep_by == "bindings":  # what the closest attempts used at any of its bindings
                            for k, fo in self._focus.items():
                                if k.split("@")[0] == r.pattern:
                                    fs |= fo.get(name, set())
                        else:
                            fs |= self._focus.get(r.key, {}).get(name, set())
                    by.setdefault(tuple(sorted(fs)), []).append(r)
                for fkey, rs in by.items():
                    if fkey not in self._focus_classes:
                        if len(self._focus_classes) > 16:
                            self._focus_classes.clear()
                        self._focus_classes[fkey] = enumerate_items(list(PRIMS) + list(fkey), self.lib, self.settings,
                                                                    self.device, max_body=self.comp_body,
                                                                    deadline=deadline, cache=self._sig_cache,
                                                                    chunk=self.chunk * self.fuse)
                    if self.search_s:
                        dl = min(deadline or float("inf"), time.time() + self.search_s)
                    fit = run(f"deep {list(fkey)}", self._focus_classes[fkey], self.lib, self.deep_budget,
                              [(r, open_of(r)) for r in rs])
                    for r in rs:
                        self._fit[r.key] = fit[r.key]
            elif deep:
                fit = run("deep", comp, self.lib, self.deep_budget, [(r, open_of(r)) for r in group])
                for r in group:
                    self._fit[r.key] = fit[r.key]
            else:
                fit = run("routines", comp, self.lib, self.probe_budget,
                          [(r, list(range(len(r.goal_list)))) for r in group])
                fb = [(r, open_of(r)) for r in group if open_of(r) and r.key not in self.fallback_seen]
                if fb and self.fallback_budget and self.routines():
                    prims = self.prim_classes(deadline)
                    if self.search_s:
                        dl = min(deadline or float("inf"), time.time() + self.search_s)
                    fit2 = run("primitives", prims, {}, self.fallback_budget, fb)
                    for r, _ in fb:
                        fit[r.key] = {k: max(v, fit2[r.key].get(k, 0.0)) for k, v in fit[r.key].items()}
                for r in group:
                    self.fallback_seen.add(r.key)
                    self._fit[r.key] = {k: v for k, v in fit[r.key].items() if out[r.key][k][0]["found_cost"] is None}
        except Deadline:
            pass
        rec = dict(roots=[r.key for r in group], level=self.level, seconds=round(time.time() - t0, 1), runs=runs,
                   deep=deep, laws={f"{g}/{k}": dict(found_cost=x["found_cost"], how=how,
                                                    programs=[show(p) for p in (x["accepted"] or [])][:8])
                                    for k, o in out.items() for g, (x, how) in o.items()})
        for r in group:
            for g, (x, how) in out[r.key].items():
                self.results[f"{g}/{r.key}"] = dict(x, how=how, root=r)
        return rec

    def groups(self, roots, size):
        """Roots in groups for one batched search each: the same probe shapes, at most size roots, pool order."""
        by = {}
        for r in roots:
            by.setdefault(tuple(b.regs.shape for b in r.roots), []).append(r)
        out = []
        for rs in by.values():
            out += [rs[i:i + size] for i in range(0, len(rs), size)]
        return out

    def deep_roots(self, roots):
        """The deep probes' roots: the deep_k roots whose closest open law is closest (ties in pool order), among those
        with an open law that the probe brought closer than one instruction does. With deep_by="concept", for each
        open concept (a law on a pattern) its closest binding's root, the deep_k closest such concepts' roots. With
        deep_by="bindings", the closest concepts whole: every binding where the concept is still open, concept by
        concept until deep_k roots are taken (the closest binding may be one no short program fits, as an operand that
        must first be moved out of A)."""
        if self.deep_by == "bindings":
            best = {}
            for i, r in enumerate(roots):
                for law, f in self._fit.get(r.key, {}).items():
                    k = (law, r.pattern)
                    if f > 0 and (k not in best or f > best[k][0]):
                        best[k] = (f, i)
            out = []
            for (law, pat), _ in sorted(best.items(), key=lambda kv: (-kv[1][0], kv[1][1])):
                if len(out) >= self.deep_k:
                    break
                out += [r for r in roots if r.pattern == pat and law in self._fit.get(r.key, {}) and r not in out]
            return out
        if self.deep_by == "concept":
            best = {}
            for i, r in enumerate(roots):
                for law, f in self._fit.get(r.key, {}).items():
                    k = (law, r.pattern)
                    if f > 0 and (k not in best or f > best[k][0]):
                        best[k] = (f, i, r)
            out = []
            for f, i, r in sorted(best.values(), key=lambda c: (-c[0], c[1])):
                if r not in out:
                    out.append(r)
            return out[:self.deep_k]
        cand = [(max(self._fit[r.key].values()), i, r) for i, r in enumerate(roots)
                if self._fit.get(r.key) and max(self._fit[r.key].values()) > 0]
        cand.sort(key=lambda c: (-c[0], c[1]))
        return [r for _, _, r in cand[:self.deep_k]]

    # ------------------------------------------------------------------ choose
    def classify(self):
        """status per pool candidate: trivial / learnable / alias / duplicate / unsolved, with its chosen program."""
        status, learn_sig = {}, {}
        for name in self.pool:
            r = self.results.get(name)
            if r is None or r["found_cost"] is None:
                status[name] = dict(status="unsolved")
                continue
            lib = {} if r["how"] == "primitives" else self.lib
            progs = self.choose_routine(r["accepted"], lib, r["root"])
            st = dict(status="trivial" if r["found_cost"] <= 1 else "learnable", cost=r["found_cost"],
                      prog=progs[0], progs=progs, lib_used=lib is self.lib)
            if st["status"] == "learnable":
                s = self.signature(progs[0], self.lib)
                if s in self.tokens_sig[2]:
                    st["status"] = "alias"
                elif s in learn_sig:
                    st["status"], st["of"] = "duplicate", learn_sig[s]
                else:
                    learn_sig[s] = name
            status[name] = st
        return status

    def choose_routine(self, progs, lib, root):
        """Among the shortest programs: one that leaves its inputs as they were and changes the fewest registers, then
        Occam (fewest primitives written, then fewest executed)."""
        st, r0 = self.settings[-1], root.roots[-1]

        def key(p):
            keeps, touched = side_effects(p, lib, st, r0, root.binding)
            return (not keeps, touched, cost(expand(p, lib)), steps(p, lib))
        best = min(progs, key=key)
        return [best] + [p for p in progs if p != best]

    def learnable(self, status):
        return [n for n in self.pool if status[n]["status"] == "learnable"]

    def hardest(self, status, k):
        """The k costliest learnable candidates (bar 3's lp order: ties in pool order)."""
        return sorted(self.learnable(status), key=lambda n: -status[n]["cost"])[:k]

    def look_jobs(self, status):
        """The cheaper pairs signal's lookaheads (bar3/breadth/cheap.py, anchor_cost 2): every learnable alone, and
        every pair with a member among the 2 hardest."""
        learn = self.learnable(status)
        top = set(self.hardest(status, 2))
        jobs = [(n,) for n in learn]
        jobs += [(a, b) for a, b in itertools.combinations(learn, 2) if a in top or b in top]
        return jobs

    def look(self, rows, answers, deadline, extra):
        """One lookahead: the target searched with extra routines (the small target, or the round), at look_budget."""
        if self.target is not None:
            return self.target.search(self, rows, self.look_budget, deadline, extra=extra)[1]
        return self.search_round(answers, deadline, extra=extra, budget=self.look_budget)[1]

    def mdl(self, rec, n_symbols):
        """Bits to describe the target's answers with the best program found (the small target: bar 3's mdl, from
        each level's best fit; the round: every probe wrong when nothing is found, since its search records only
        whether a level reaches every answer)."""
        if "fits" in rec:
            return self.target.mdl(self, rec, n_symbols)
        exc = math.log2(8) + self.P_round * math.log2(10)
        if rec["found_cost"] is not None:
            return rec["found_cost"] * math.log2(n_symbols)
        return 8 * exc

    def unlockers(self, status, rows):
        """The laws whose practice would let the learner pose a kind of problem (a law on a pattern) it cannot pose
        now: for each law and pattern with a learnable candidate and no routine yet, its routine is tried in the
        constructions of every law and pattern not posed now. Returns {(law, pattern): [what it unlocks]}."""
        posed = {(n.split("/")[0], n.split("/")[1].split("@")[0]) for n in list(self.pool) + list(self.practiced)}
        want = [(law, pat) for pat in self.patterns for law in self.laws
                if pat in LAW_BOOK[law][1] and (law, pat) not in posed]
        out = {}
        for n in self.learnable(status):
            law, rk = n.split("/")
            pat, bd = rk.split("@")
            if self.ops.has(law, pat) or (law, pat) in out or not want:
                continue
            ops = Ops()
            ops.by_key = {k: list(v) for k, v in self.ops.by_key.items()}
            ops.add(law, pat, "T.trial", self.results[n]["root"].binding)
            lib = dict(self.lib, **{"T.trial": status[n]["prog"]})
            got = []
            for l2, p2 in want:
                try:
                    r = Root(p2, bindings(len(PATTERNS[p2][0][0]))[0], self.settings, self.NP, rows, ops, lib,
                             self.device, self.seed)
                    if r.goals(ops, lib, [l2]):
                        got.append(f"{l2}@{p2}")
                except NotPosable:
                    pass
            if got:
                out[(law, pat)] = got
        return out

    def concepts_done(self):
        """The concepts (a law on a pattern) practised in some binding."""
        return {tuple(n.split("@")[0].split("/")) for n in self.practiced if n.split("/")[0] in LAW_BOOK}

    def concepts(self, status, k, ties=False):
        """The k hardest concepts not yet practised in any binding (a concept: a law on a pattern; its hardness: the
        cost of its cheapest learnable binding, so that an awkward binding does not make a concept look hard), each
        in its cleanest learnable binding (the fewest registers touched besides A, then the cheapest, then pool
        order)."""
        done = {tuple(n.split("@")[0].split("/")) for n in self.practiced}
        by = {}
        for i, n in enumerate(self.learnable(status)):
            law, rk = n.split("/")
            pat = rk.split("@")[0]
            if (law, pat) in done:
                continue
            r = self.results[n]["root"]
            lib = {} if self.results[n]["how"] == "primitives" else self.lib
            _, touched = side_effects(status[n]["prog"], lib, self.settings[-1], r.roots[-1], r.binding)
            by.setdefault((law, pat), []).append((touched, status[n]["cost"], i, n))
        ranked = sorted(by.values(), key=lambda v: (-min(c for _, c, _, _ in v), min(i for _, _, i, _ in v)))
        if not ties:
            return [min(v)[3] for v in ranked[:k]]
        return [n for v in ranked[:k] for t, c, _, n in sorted(v) if (t, c) == min(v)[:2]]

    def pick(self, status, rows=None):
        if self.rule == "all":
            return self.learnable(status)
        if self.rule == "lp":
            return self.hardest(status, self.K)
        if self.rule in ("lpc", "lpca"):              # concepts: a law on a pattern, hardest by its cheapest binding,
            got = self.concepts(status, self.K, ties=self.rule == "lpca")     # practised once, in its cleanest
            if got or not self._variants:             # binding (lpca: in every binding tied for cleanest). A practised
                self._want_variants = not got         # concept's other bindings are not posed; when no new concept is
                return got                            # learnable, the next level poses them, and picks the hardest
            return self.hardest(status, self.K)
        if self.rule == "lpu":                        # unlockers first (each its hardest candidate), then hardest
            un = self.unlockers(status, rows)
            self._scores = {"unlocks": {f"{k[0]}@{k[1]}": v for k, v in un.items()}}
            first = []
            for n in self.hardest(status, len(self.learnable(status))):
                law, rk = n.split("/")
                key = (law, rk.split("@")[0])
                if key in un and all(not m.startswith(f"{law}/{key[1]}@") for m in first):
                    first.append(n)
            return (first + [n for n in self.hardest(status, self.K + len(first)) if n not in first])[:max(self.K,
                                                                                                         len(first))]
        if self.rule == "ct2a2":
            n_sym = len(PRIMS) + len(self.routines())
            base = self.mdl(self._base_look, n_sym)
            singles = {j[0]: base - self.mdl(r, n_sym + 1) for j, r in self.looks if len(j) == 1}
            pairs = sorted(((base - self.mdl(r, n_sym + 2), j) for j, r in self.looks if len(j) == 2),
                           key=lambda p: -p[0])
            chosen = []
            for g, j in pairs:
                for n in j:
                    if n not in chosen and len(chosen) < self.K:
                        chosen.append(n)
                if len(chosen) >= self.K:
                    break
            for n in sorted(singles, key=lambda n: -singles[n]):
                if len(chosen) < self.K and n not in chosen:
                    chosen.append(n)
            self._scores = dict(singles=singles, pairs=[(round(g, 1), list(j)) for g, j in pairs[:20]])
            return chosen
        raise ValueError(self.rule)

    # ------------------------------------------------------------------ sleep
    def sleep(self, picks, status):
        mlog = []
        for n in picks:
            self.corpus[n] = list(status[n]["progs"][:8])
            self.practiced[n] = self.level
        for n in self.pool:                          # a pick is not posed again; nor is a trivial problem
            if status[n]["status"] == "trivial":
                self.practiced.setdefault(n, self.level)
        self.corpus, self.lib, new = bpe(self.corpus, self.lib, max_merges=self.max_merges, min_count=self.min_count,
                                         prefix=f"B{self.level}.", log=mlog)
        for n in picks:                              # each pick's routine: its chosen program, rewritten by the merges
            prog = self.corpus[n][0]
            name = "T." + n.replace("/", ".").replace("@", ".")
            if cost(expand(prog, self.lib)) >= 2 and name not in self.lib:
                self.lib[name] = prog
                fn = _apply_routine(prog, name)
                self.corpus = {u: [rewrite(p, fn) for p in ps] for u, ps in self.corpus.items()}
                mlog.append({"name": name, "def": show(prog), "routine": True})
                law, rk = n.split("/")
                pat, _ = rk.split("@")
                self.ops.add(law, pat, name, self.results[n]["root"].binding)
        return mlog

    # ------------------------------------------------------------------ the target (the round)
    def search_round(self, answers, deadline=None, extra=None, budget=None):
        """The round by composition (libcore.Climb.search_round): the routines and the straight-line primitives at
        the task's full size, the goal the answers. The answers are rows of one T, the smallest the batches showed:
        the search's goal is that T's answer as it is (sp2 ran each candidate t times when t > 1: the countdown's
        assumption, gone here; the answer's program is search_outer's)."""
        t0 = time.time()
        lib = dict(self.lib, **(extra or {}))
        routines = [k for k in lib if k.startswith("T.")]
        prob = round_problem(self.cells, answers, 8, self.P_round, self.device)
        try:
            classes = enumerate_items(list(PRIMS) + routines, lib, prob.settings, self.device, max_body=0,
                                      deadline=deadline)
            res = bfs(prob, classes, lib, max_cost=self.round_cost, max_states=budget or self.round_budget,
                      accept=lambda ps: [round_ok(p, lib, self.cells, answers, self.P_round, self.device)
                                         for p in ps], deadline=deadline, chunk=max(1, self.chunk // 4),
                      fuse=self.fuse)
            found, accepted, sizes, stop = res.found_cost, res.accepted, res.level_sizes, res.stop
            if found is None and len(routines) > self.macro_after and not extra:
                mres = bfs(prob, self.macro_classes(lib), lib, max_cost=self.macro_cost,
                           max_states=budget or self.round_budget,
                           accept=lambda ps: [round_ok(p, lib, self.cells, answers, self.P_round, self.device)
                                              for p in ps], deadline=deadline, chunk=max(1, self.chunk // 4),
                           fuse=self.fuse)
                stop = f"{stop}; by calls: levels {mres.level_sizes} {mres.stop}"
                if mres.found_cost is not None:
                    found, accepted = f"{mres.found_cost} calls", [unwrap(p) for p in mres.accepted]
        except Deadline:
            found, accepted, sizes, stop = None, None, [], "deadline (building the vocabulary)"
        rec = dict(task="round", level=self.level, found_cost=found, level_sizes=sizes,
                   seconds=round(time.time() - t0, 1), stop=stop, routines=len(routines),
                   answers=int(answers[0].shape[0]), programs=[show(p) for p in (accepted or [])][:8])
        return accepted or [], rec

    def macro_classes(self, lib):
        """The round's search by calls (a big library leaves the instruction-by-instruction search no reach): one item
        per routine and per routing of its inputs (each from A, X, Y, N or zero, by route), and the instructions that
        act on whole registers; every item costs 1."""
        bind = {tok: "".join(bd) for v in self.ops.by_key.values() for tok, bd in v}
        items = [ItemClass(w, 1) for w in WHOLE]
        seen = set()
        for tok in sorted(k for k in lib if k.startswith("T.") and k in bind):
            bd = bind[tok]
            for srcs in itertools.product(SOURCES, repeat=len(bd)):
                code = route(bd, srcs)
                if code is None:
                    continue
                it = ("REP", ONE, tuple(code) + (tok,))
                if it not in seen:
                    seen.add(it)
                    items.append(ItemClass(it, 1))
        return items

    # ------------------------------------------------------------------ the answer's program
    def outer_classes(self, lib, settings, deadline=None, max_body=2, vocab="all"):
        """The answer program's vocabulary: the whole-register instructions and every routine (vocab "all"; "rounds":
        the rounds only), each one item (deduped on random machines, enumerate_items'), and a count loop on every
        register over every distinct body of up to max_body of those items (the bodies deduped as straight-line code;
        the loops are not: on random machines every count is above the cap, so every count loop would look alike)."""
        tokens = list(WHOLE) + sorted(k for k in lib if k.startswith("T.round" if vocab == "rounds" else "T."))
        classes = enumerate_items(tokens, lib, settings, self.device, max_body=0, deadline=deadline)
        gen = gen_on(self.device, 7)
        tests = [random_states(st, 24, 6, gen, self.device) for st in settings]
        reps = [c.rep for c in classes]
        seen = set(_signatures(settings, tests, [("REP", ONE, ())], lib, deadline=deadline))
        bodies = []
        for n in range(1, max_body + 1):
            cands = list(itertools.product(reps, repeat=n))
            for b, s in zip(cands, _signatures(settings, tests, [("REP", ONE, b) for b in cands], lib,
                                               chunk=self.chunk, deadline=deadline)):
                if s not in seen:
                    seen.add(s)
                    bodies.append(b)
        return classes + [ItemClass(("REP", kind, b), 1 + len(b)) for b in bodies for kind in COUNT_KINDS], len(bodies)

    def search_outer(self, rows, rounds, deadline=None, budget=1_000_000, max_cost=3, max_body=2, NP=8, vocab="all"):
        """The answer's program: each round (a program of the smallest T's answers) a routine T.round<i>, composed
        with the library's routines, the whole-register instructions and count loops, breadth-first by cost against
        the answers of rows of every T (rows: Nd, Xd, Cd, Yd with the count's digits Cd; the machine starts with x in
        X, N in N and the count in S; the probes: NP rows taken in turn from each T); accepted if it reproduces every
        row (OuterMachine). Returns (programs, the library with the rounds, a record)."""
        t0 = time.time()
        lib = dict(self.lib)
        for i, p in enumerate(rounds):
            lib[f"T.round{i + 1}"] = tuple(p)
        _, grp = torch.unique(rows[2], dim=0, return_inverse=True)       # rows grouped by their count's digits
        by_t = [torch.nonzero(grp == g).flatten() for g in range(int(grp.max()) + 1)]
        idx = torch.stack([g[i] for i in range(max(g.numel() for g in by_t)) for g in by_t if i < g.numel()])
        Nd, Xd, Cd, Yd = (r[idx] for r in rows)
        st = round_settings(self.cells, self.P_round)
        n = min(NP, idx.numel())
        root = fresh(st, {"X": pad(Xd[:n], st.P), "N": pad(Nd[:n], st.P), "S": pad(Cd[:n], st.P)},
                     gen_on(self.device, 0), n, self.device)
        prob = Problem("outer", [st], [root], [{"A": pad(Yd[:n], st.P).T.to(torch.uint8)}])

        def ok(p):
            A = outer_machine(st, p, lib, Xd.shape[0], self.device).run(Xd, Nd, Cd, gen_on(self.device, 1))
            return bool((A[:, :Yd.shape[1]] == Yd).all())
        try:
            classes, nb = self.outer_classes(lib, [st], deadline, max_body, vocab)
            res = bfs(prob, classes, lib, max_cost=max_cost, max_states=budget, accept=lambda ps: [ok(p) for p in ps],
                      deadline=deadline, chunk=max(1, self.chunk // 4), fuse=self.fuse)
            found, accepted, sizes, stop = res.found_cost, res.accepted, res.level_sizes, res.stop
        except Deadline:
            found, accepted, sizes, stop, classes, nb = None, None, [], "deadline (building the vocabulary)", [], 0
        rec = dict(task="outer", level=self.level, found_cost=found, level_sizes=sizes, stop=stop,
                   seconds=round(time.time() - t0, 1), items=len(classes), bodies=nb, rows=int(idx.numel()),
                   counts=len(by_t), programs=[show(p) for p in (accepted or [])][:8])
        return accepted or [], lib, rec

    # ------------------------------------------------------------------ one unit
    def step(self, rows, answers=None, deadline=None):
        """One unit of work; answers (Nd, Xd, Yd) are rows of one T (the smallest the batches showed)."""
        if not self.events:
            line = self._step(rows, answers, deadline)
            if line is not None:
                self.events.append(line)
        return self.events.pop(0) if self.events else None

    def _step(self, rows, answers, deadline):
        if self.done:
            if not (self.round_retry and answers is not None and not self.rounds and self.routines()):
                return None
            self._idle += 1
            if self._idle < self.round_retry:
                return None
            self._idle = 0
            progs, rec = self.search_round(answers, deadline)
            rec["retry"] = True
            self.records.append(rec)
            self.rounds = progs
            return (f"L{self.level} round again ({rec['answers']} answer rows): cost {rec['found_cost']} "
                    f"{rec['seconds']}s {rec['stop']} {rec['programs'][:3]}")
        if self.phase == "pose":
            if self.level >= self.levels:
                self.done = True
                return f"climb done at level {self.level} (the level cap)"
            if self.target is not None:
                _, rec, self._near = self.target.search(self, rows, self.look_budget, deadline, near=12)
                self.near_log.append(dict(level=self.level, fits=rec["fits"], near=rec["near"]))
                self.records.append(rec)
            self.pool, roots = self.pose(rows)
            self.results, self._fit, self._focus = {}, {}, {}
            self.queue, self._level_roots = self.groups(roots, self.group), list(roots)
            self.phase = "probe"
            return f"L{self.level} pose: {len(self.pool)} candidates on {len(roots)} roots"
        if self.phase == "probe":
            if self.queue:
                group = self.queue.pop(0)
                rec = self.probe(group, rows, deadline)
                self.records.append(rec)
                solved = {g: v["found_cost"] for g, v in rec["laws"].items() if v["found_cost"] is not None}
                return (f"L{self.level} probe {len(group)} roots {rec['seconds']}s solved {solved} "
                        f"{[(r['how'], r['seconds']) for r in rec['runs']]}")
            if self.deep_k and self.deep_budget and self.routines():
                deep = self.deep_roots(self._level_roots)
                self.queue = self.groups(deep, self.deep_group)
                self.phase = "deep"
                return f"L{self.level} deep: {[(r.key, self._fit[r.key]) for r in deep]}"
            self.phase = "classify"
        if self.phase == "deep":
            if self.queue:
                group = self.queue.pop(0)
                rec = self.probe(group, rows, deadline, deep=True)
                self.records.append(rec)
                solved = {g: v["found_cost"] for g, v in rec["laws"].items() if v["found_cost"] is not None}
                return (f"L{self.level} deep {[r.key for r in group]} {rec['seconds']}s solved {solved} "
                        f"{[(r['how'], r['seconds']) for r in rec['runs']]}")
            self.phase = "classify"
        if self.phase == "classify":
            self._status = self.classify()
            self.phase = "choose"
            if self.rule == "ct2a2" and (answers is not None or self.target is not None) and self.learnable(self._status):
                self.queue = self.look_jobs(self._status)
                self.looks = []
                self._base_look = self.look(rows, answers, deadline, {})
                self.phase = "look"
                return (f"L{self.level} look: {len(self.queue)} lookaheads; the target alone: cost "
                        f"{self._base_look['found_cost']}")
        if self.phase == "look":
            if self.queue:
                job = self.queue.pop(0)
                extra = {f"T.look{i}": self._status[n]["prog"] for i, n in enumerate(job)}
                rec = self.look(rows, answers, deadline, extra)
                self.looks.append((job, rec))
                return f"L{self.level} look {job}: cost {rec['found_cost']} {rec['seconds']}s"
            self.phase = "choose"
        if self.phase == "choose":
            status = self._status
            if self.rule == "ct2a2" and not self.looks:
                picks = self.hardest(status, self.K)
            else:
                picks = self.pick(status, rows)
            counts = Counter(v["status"] for v in status.values())
            self.history.append(dict(level=self.level, pool=len(self.pool), counts=dict(counts), picks=list(picks),
                                     learnable={n: (status[n]["cost"], show(status[n]["prog"])) for n in self.pool
                                                if status[n]["status"] == "learnable"},
                                     scores=self._scores if self.rule in ("ct2a2", "lpu") else {}))
            self.picks = picks
            self.phase = "sleep"
            self.looks = []
            return (f"L{self.level} choose ({self.rule}): {dict(counts)}; picks "
                    f"{[(n, status[n]['cost'], show(status[n]['prog'])) for n in picks]}")
        if self.phase == "sleep":
            mlog = self.sleep(self.picks, self._status)
            self.history[-1]["merges"] = mlog
            self.history[-1]["lib"] = {k: show(v) for k, v in self.lib.items()}
            self.phase = "target"
            self._variants = self.rule not in ("lpc", "lpca") or self._want_variants
            self._want_variants = False
            if not self.picks and not self._variants:
                self.done = True
                return f"L{self.level} sleep: nothing learnable to practise; climb done"
            if not self.picks:
                return f"L{self.level} sleep: no new concept learnable; the next level poses practised ones' variants"
            return f"L{self.level} sleep: " + "; ".join(f"{m['name']}={m['def']}" for m in mlog)
        if self.phase == "target":
            self.phase = "pose"
            lvl = self.level
            self.level += 1
            if self.target is not None:
                progs, rec, _ = self.target.search(self, rows, self.round_budget, deadline)
                if progs and answers is not None:           # is the small target's program the round at full size?
                    rec["full_size"] = [bool(round_ok(p, self.lib, self.cells, answers, self.P_round, self.device))
                                        for p in progs[:8]]
                self.records.append(rec)
                if progs:
                    self.rounds = progs
                    self.done = True
                return (f"L{lvl} target: cost {rec['found_cost']} levels {rec['level_sizes']} fits {rec['fits']} "
                        f"{rec['seconds']}s {rec['stop']} {rec['programs'][:3]} full size {rec.get('full_size')}")
            if answers is None or not self.routines():
                return f"L{lvl} target: no answers yet"
            progs, rec = self.search_round(answers, deadline)
            self.records.append(rec)
            if progs:
                self.rounds = progs
                self.done = True
            return (f"L{lvl} round: cost {rec['found_cost']} levels {rec['level_sizes']} {rec['seconds']}s "
                    f"{rec['stop']} {rec['programs'][:3]}")
        raise ValueError(self.phase)


def compare(A: Tensor, B: Tensor) -> Tensor:
    """The reader's comparison of two read numbers (digit labels [X, L], LSB first): -1, 0, 1 per row, decided at the
    highest place where they differ (as the reader's ranking decides x < N)."""
    L = A.shape[1]
    top = torch.where(A != B, torch.arange(L, device=A.device), -1).max(-1).values
    a = A.gather(1, top.clamp(min=0)[:, None])[:, 0]
    b = B.gather(1, top.clamp(min=0)[:, None])[:, 0]
    return torch.where(top < 0, 0, torch.where(a < b, -1, 1))


def relations(terms: dict, pairs: bool = True) -> dict:
    """The miner's candidate relations among read numbers (name -> digits [X, L] of the same width), each a bool per
    row: order, equality, constant across rows, nonzero, range (a < 10^k) and divisibility by 2, 3, 5 (from the digit
    labels: the lowest digit for 2 and 5, the digit sum for 3; the miner's bookkeeping, never a lesson's target)."""
    out = {}
    names = list(terms)
    if pairs:
        for a, b in itertools.permutations(names, 2):
            out[f"{a} < {b}"] = compare(terms[a], terms[b]) < 0
        for a, b in itertools.combinations(names, 2):
            out[f"{a} = {b}"] = compare(terms[a], terms[b]) == 0
    for a in names:
        A = terms[a]
        out[f"{a} constant"] = (A == A[:1]).all(1)
        out[f"{a} != 0"] = (A != 0).any(1)
        for k in range(1, A.shape[1]):
            out[f"{a} < 10^{k}"] = (A[:, k:] == 0).all(1)
        for d, v in ((2, A[:, 0] % 2), (5, A[:, 0] % 5), (3, A.sum(1) % 3)):
            out[f"{a} mod {d} != 0"] = v != 0
            for r in range(d):
                out[f"{a} mod {d} = {r}"] = v == r
    return out


class Miner:
    """Relations kept while they hold on every row of the last MINE_WIN batches (a batch where one fails pauses it);
    `fails` counts the batches where each failed since it was first evaluated."""

    def __init__(self) -> None:
        self.streak: dict = {}
        self.fails: dict = {}
        self.first: dict = {}
        self.announced: set = set()

    def update(self, rels: dict, step: int) -> list:
        new = []
        for k, ok in rels.items():
            good = bool(ok.all())
            self.streak[k] = self.streak.get(k, 0) + 1 if good else 0
            self.fails[k] = self.fails.get(k, 0) + (not good)
            if self.streak[k] >= MINE_WIN and k not in self.announced:
                self.announced.add(k)
                self.first[k] = step
                new.append(k)
        return new

    def holds(self, k: str) -> bool:
        return self.streak.get(k, 0) >= MINE_WIN

    def facts(self) -> list:
        """The relations in force, without those a stronger one implies (a < 10^k when a < 10^(k-1) holds; a mod d != 0
        when a mod d = r holds)."""
        f = {k for k in self.streak if self.holds(k)}
        out = []
        for k in sorted(f):
            if " < 10^" in k:
                a, e = k.split(" < 10^")
                if f"{a} < 10^{int(e) - 1}" in f:
                    continue
            if k.endswith(" != 0") and " mod " in k:
                a, d = k.split(" mod ")[0], k.split(" mod ")[1].split()[0]
                if any(f"{a} mod {d} = {r}" in f for r in range(1, 5)):
                    continue
            out.append(k)
        return out

    def describe(self) -> str:
        return ";".join(f"{k}(since {self.first.get(k, '?')}, fails {self.fails.get(k, 0)})" for k in self.facts())


def answer_t(Cs: Tensor) -> int:
    """The count T of the digits Cs [n, K] (LSB first) of the first row, as a Python int (the hard machine's round
    bound only)."""
    return sum(int(d) * 10 ** j for j, d in enumerate(Cs[0].tolist())) if Cs.shape[0] else 0


def answer_rows(Cd: Tensor, tcount: Tensor | None = None) -> Tensor:
    """The rows whose answers the lessons read (the selector, the round, the reading's answers, the freeze's check):
    the rows whose T (as read, digits Cd LSB first) is the batch's smallest nonzero T, compared the reader's way (the
    highest place where two counts differ decides), at most ANS_WORK // (T + 1) of them. Nothing about the data's T is
    assumed: on H1's batches that is T = 1 (about 5 rows of 512), on data whose every T is 8 it is T = 8 (general
    lessons v1 read the one-digit T <= 2 only, and so never saw an answer on such data). tcount (T's digit count) is
    not used; it is there for general_lessons/local_check.py's call."""
    keep = (Cd != 0).any(1)
    if not bool(keep.any()):
        return keep
    for j in reversed(range(Cd.shape[1])):                                  # the least digit at each place, top down
        keep = keep & (Cd[:, j] == Cd[keep, j].min())
    t = answer_t(Cd[keep][:1])
    keep[torch.nonzero(keep).flatten()[max(1, ANS_WORK // (t + 1)):]] = False
    return keep


def tally(acc: Tensor | None, idx: Tensor, n: int) -> None:
    """Training / evaluation audits: how often each table row is used."""
    if acc is not None:
        acc += torch.bincount(idx.flatten(), minlength=n)


def ste(hard: Tensor, soft: Tensor) -> Tensor:
    """Straight-through: the value is exactly `hard` (soft - soft.detach() is exactly 0), the gradient is soft's."""
    return hard + (soft - soft.detach())


def sub_hard(D: Tensor, M: Tensor, Td: Tensor, Tb: Tensor):
    """D - M place by place from the lowest with the snapped table (flat Td, Tb [200] indexed r * 20 + m * 2 + c), any
    width and batch shape: digits [..., L], LSB first. Returns the digits, the final borrow and the table rows."""
    c = torch.zeros_like(D[..., 0])
    outs, rows = [], []
    for j in range(D.shape[-1]):
        idx = (D[..., j] * 10 + M[..., j]) * 2 + c
        outs.append(Td[idx])
        c = Tb[idx]
        rows.append(idx)
    return torch.stack(outs, -1), c, torch.stack(rows, -1)


def add_learned(A: Tensor, B: Tensor, Td: Tensor, Tb: Tensor, visits: Tensor | None = None):
    """The learned adder (ADDER.md): A + B mod 10^L as A - (0 - B), both subtractions the table (in ten's complement
    0 - B is 10^L - B), and the final carry borrow(0 - B) - borrow(A - (0 - B))."""
    C, bc, r1 = sub_hard(torch.zeros_like(B), B, Td, Tb)
    Y, by, r2 = sub_hard(A, C, Td, Tb)
    tally(visits, r1, 200)
    tally(visits, r2, 200)
    return Y, bc - by


def onehot0(n: int, L: int, device, dtype=torch.float32) -> Tensor:
    """n numbers of L exact 0 digits [n, L, 10]."""
    z = torch.zeros(n, L, 10, device=device, dtype=dtype)
    z[..., 0] = 1
    return z


def zpad(D: Tensor, low: int, high: int) -> Tensor:
    """Digit distributions [X, L, 10] padded with exact 0 digits below (low) and above (high)."""
    return torch.cat([onehot0(D.shape[0], low, D.device, D.dtype), D, onehot0(D.shape[0], high, D.device, D.dtype)], 1)


def carry_on(c: Tensor, ste_cin: bool) -> Tensor:
    """The borrow passed to the next place: the soft one, or (ste_cin) the snapped borrow of the entry used, with the
    soft one's gradient. With ste_cin a lesson's next place uses exactly one table entry, so a lesson trains only the
    entries it uses (ABLATION.md rule 3); the borrow returned for the lesson itself stays the soft one."""
    if not ste_cin:
        return c
    return ste(F.one_hot(c.argmax(-1), 2).to(c.dtype), c)


def link_soft(D: Tensor, M: Tensor, TDC: Tensor, ste_cin: bool = False):
    """D - M on digit distributions D [X, L, 10] minus exact digits M [X, L], the cell applied multilinearly. TDC
    [20, 120]: rows (r, c), columns (m, 10 digit + 2 borrow probabilities). Returns the digits [X, L, 10] and the
    borrow out of every place [X, L, 2] (the last is the final borrow)."""
    X = D.shape[0]
    c = torch.zeros(X, 2, device=D.device)
    c[:, 0] = 1
    out, borrows = [], []
    for j in range(D.shape[1]):
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = (joint @ TDC).view(X, 10, 12)                                       # every m at once
        Y = Y.gather(1, M[:, j, None, None].expand(X, 1, 12)).squeeze(1)        # M's own digit
        out.append(Y[:, :10])
        borrows.append(Y[:, 10:])
        c = carry_on(Y[:, 10:], ste_cin)
    return torch.stack(out, 1), torch.stack(borrows, 1)


def sub_soft(D: Tensor, M: Tensor, TDC: Tensor, used: Tensor | None = None, ste_cin: bool = False):
    """D - M with both operands digit distributions [X, L, 10] (multilinear). Returns the digits [X, L, 10] and the
    borrow out of every place [X, L, 2]. With one-hot operands and a straight-through table this is the prediction
    path's subtraction: every value is exactly the snapped table's. `used` accumulates the rows (r, m, c) met."""
    X = D.shape[0]
    c = torch.zeros(X, 2, device=D.device, dtype=D.dtype)
    c[:, 0] = 1
    out, borrows = [], []
    for j in range(D.shape[1]):
        if used is not None:
            used += torch.einsum("xr,xm,xc->rmc", D[:, j], M[:, j], c).detach().flatten()
        joint = (D[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        Y = torch.einsum("xmk,xm->xk", (joint @ TDC).view(X, 10, 12), M[:, j])   # expected over the subtrahend's digit
        out.append(Y[:, :10])
        borrows.append(Y[:, 10:])
        c = carry_on(Y[:, 10:], ste_cin)
    return torch.stack(out, 1), torch.stack(borrows, 1)


def tourney_nll(lw: Tensor, ok: Tensor) -> Tensor | None:
    """The tournament's loss: -log sum_c w_c [c passes every observation], lw [K] the log weights of K whole
    combinations (normalized over every combination of their blocks, so the ones left out count as failing), ok [K]
    their verdicts. None (no signal) when none passes."""
    if not bool(ok.any()):
        return None
    return -torch.logsumexp(lw[ok], 0)


def order_pairs(Tb: Tensor, used: Tensor) -> list:
    """The pairs of table entries the two order laws compare, each oriented (lower, upper) by the learned borrows
    (anchors/submission_anchors.py). Entries are flat (r, m, c) -> r * 20 + m * 2 + c; only entries in `used` take
    part. An entry wraps when its borrow out is 1, and r < r' when r - r' borrows.
      * columns: (r, m, c) and (r', m, c), r < r': the digits keep their order, unless exactly one wraps, which then
        comes out larger;
      * rows: (r, m, c) and (r, m', c), m < m': subtracting more gives less, unless exactly one wraps, which then comes
        out larger."""
    dev = Tb.device
    ent = torch.arange(200, device=dev)
    r, m, c = ent // 20, (ent // 2) % 10, ent % 2
    e0 = ent.repeat_interleave(10)
    k = torch.arange(10, device=dev).repeat(200)
    out = []
    e1 = k * 20 + (e0 % 20)                                                # columns
    ok = (r[e0] < k) & used[e0] & used[e1]
    a, b = e0[ok], e1[ok]
    wa, wb = Tb[a], Tb[b]
    lt = Tb[(r[a] * 10 + r[b]) * 2]                                        # r < r'
    first = (wa < wb) | ((wa == wb) & (lt == 1))
    out.append((torch.where(first, a, b), torch.where(first, b, a)))
    e1 = (r[e0] * 10 + k) * 2 + c[e0]                                      # rows
    ok = (m[e0] < k) & used[e0] & used[e1]
    a, b = e0[ok], e1[ok]
    wa, wb = Tb[a], Tb[b]
    lt = Tb[(m[a] * 10 + m[b]) * 2]                                        # m < m'
    first = (wa < wb) | ((wa == wb) & (lt == 0))                           # a (less taken) comes out smaller
    out.append((torch.where(first, a, b), torch.where(first, b, a)))
    return out


def pick(w: Tensor, opts: list) -> Tensor:
    """A switch on the prediction path: w [n] its straight-through one-hot (values exactly 0 or 1), opts the options'
    digit distributions. With autograd the blend sum_o w_o opts_o (its value is the chosen option exactly, its gradient
    reaches every option's logit); without autograd, as at evaluation, the chosen option itself (the same value)."""
    if not torch.is_grad_enabled():
        return opts[int(w.argmax())]
    return sum(w[i] * o for i, o in enumerate(opts))


def shift(A: Tensor, d: int) -> Tensor:
    """Position distributions [X, Lp] moved by d places (mass at q goes to q + d; zero-filled)."""
    if d >= 0:
        return F.pad(A, (d, 0))[:, :A.shape[1]]
    return F.pad(A, (0, -d))[:, -d:]


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len



def damp(x: Tensor) -> Tensor:
    """The prediction path's value, its gradient times GRAD_DAMP (the certified files' damp; here after every cell
    and every count loop's blend of the machine). Without autograd, nothing to scale."""
    if GRAD_DAMP == 1.0 or not torch.is_grad_enabled():
        return x
    return x.detach() + GRAD_DAMP * (x - x.detach())


def exact_cells(device):
    """The true tables (designed mode and the reports' "right" counts only)."""
    return Cells(10, device)


def adc_targets(Td: Tensor, Tb: Tensor) -> tuple[Tensor, Tensor]:
    """sub.inverse on every entry of the addition: for (a, b, c) the digit d whose snapped subtraction (d, b, c) gives
    back a, and that subtraction's borrow as the carry ([10, 10, 2] each; -1 where no d gives a back: a subtraction
    that is not yet a bijection in its first operand)."""
    T3, B3 = Td.view(10, 10, 2), Tb.view(10, 10, 2)
    tgt_d = torch.full((10, 10, 2), -1, dtype=torch.long, device=Td.device)
    tgt_c = torch.full((10, 10, 2), -1, dtype=torch.long, device=Td.device)
    bb, cc = torch.meshgrid(torch.arange(10, device=Td.device), torch.arange(2, device=Td.device), indexing="ij")
    for d in range(10):
        tgt_d[T3[d], bb, cc] = d
        tgt_c[T3[d], bb, cc] = B3[d]
    return tgt_d, tgt_c


def mul_targets(Ml: Tensor, Mh: Tensor, Ad: Tensor, Ac: Tensor) -> tuple[Tensor, Tensor]:
    """mul.zero and mul.ladder on every entry of the multiply-accumulate, with the snapped addition Ad, Ac [10, 10, 2]
    and the snapped multiply Ml, Mh [10, 10, 10, 10] over (a, x, h, k): row h = 0 is a + k (its carry the high digit);
    row h + 1 is row h plus x (the low digit plus x with carry, the carry added into the high digit)."""
    dev = Ml.device
    a, x, k = torch.meshgrid(*(torch.arange(10, device=dev),) * 3, indexing="ij")
    lo = torch.empty(10, 10, 10, 10, dtype=torch.long, device=dev)
    hi = torch.empty_like(lo)
    lo[:, :, 0, :] = Ad[a, k, 0]
    hi[:, :, 0, :] = Ac[a, k, 0]
    for h in range(9):
        l0, h0 = Ml[:, :, h, :], Mh[:, :, h, :]
        lo[:, :, h + 1, :] = Ad[l0, x, 0]
        hi[:, :, h + 1, :] = Ad[h0, 0, Ac[l0, x, 0]]
    return lo, hi


def bijective(Td: Tensor) -> bool:
    """sub.bijective on the snapped table: every column (fixed b, c) and row (fixed a, c) uses each digit once."""
    T3 = Td.view(10, 10, 2)
    ar = torch.arange(10, device=Td.device)
    return bool((torch.sort(T3, 0).values == ar[:, None, None]).all() and
                (torch.sort(T3, 1).values == ar[None, :, None]).all())


class Control:
    """The training schedule (Python-side, not model state): the reader's stages, the phases A-D and frozen, the
    library's state, the audits' counters."""

    def __init__(self) -> None:
        designed = MODE == "designed"
        self.step = 0
        self.phase = "A"
        self.io = 5 if designed else 1
        self.in_phase = 0
        self.streak = 0
        self.pick = 0
        self.best = 0
        self.rows3 = 0.0
        self.chosen = 0.0
        self.io_exact = False
        self.ans_ok = 0.0
        self.facts_streak = 0
        self.gated_on: int | None = None
        self.reset = False
        self.reset_params: list = []
        self.win: deque = deque(maxlen=STAGE_WIN)
        self.table_streak = 0                 # steps with the subtraction's checks exact and the table a bijection
        self.cells_on = False                 # the addition and multiply lessons have joined
        self.cell_streak = 0                  # steps with every cell passing its own checks
        self.adc_ok = self.mul_ok = self.gate_ok = False
        self.miner = Miner()
        self.visits: Tensor | None = None
        self.used: Tensor | None = None
        self.used_count: Tensor | None = None
        self.reported = None
        self.history: list = []
        self.t_log = ""
        self.io_parts = ""
        self.pool: list = []                  # read rows (N, x digits) for the practice problems
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


class LibraryMachine(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        V = spec.vocab_size
        self.config = Config(V, spec.max_seq_len)
        self.V, self.END, self.OUT = V, V, V + 1
        self.a = nn.Parameter(INIT * torch.randn(10, 10, 2, 10))           # subtraction: (a, b, borrow) -> digit
        self.b = nn.Parameter(INIT * torch.randn(10, 10, 2, 2))            # ... -> borrow
        self.aa = nn.Parameter(INIT * torch.randn(10, 10, 2, 10))          # addition: (a, b, carry) -> digit
        self.ab = nn.Parameter(INIT * torch.randn(10, 10, 2, 2))           # ... -> carry
        self.ml = nn.Parameter(INIT * torch.randn(10, 10, 10, 10, 10))     # multiply-accumulate: (a, x, h, k) -> low
        self.mh = nn.Parameter(INIT * torch.randn(10, 10, 10, 10, 10))     # ... -> high
        self.g = nn.Parameter(INIT * torch.randn(2, 2))                    # sp2's gate: dead weight (diag)
        self.prog = nn.Parameter(INIT * torch.randn(PROG_SLOTS))           # the answer programs' tournament
        self.dg = nn.Parameter(INIT * torch.randn(V))
        self.u = nn.Parameter(INIT * torch.randn(len(HYP), V))
        self.wc = nn.Parameter(INIT * torch.randn(10, V))
        self.anc = nn.Parameter(INIT * torch.randn(4, V + 1))
        self.strd = nn.Parameter(INIT * torch.randn(4, 2))
        self.shape = nn.Parameter(INIT * torch.randn(len(SHAPES)))
        self.off = nn.Parameter(INIT * torch.randn(4, W, 2 * OFF + 1))
        self.LPM = spec.max_seq_len + 1 + 2 * (W + 1)
        self.absp = nn.Parameter(INIT * torch.randn(4, W, self.LPM))
        self.ctl = Control()
        if MODE == "designed":
            self.designed()

    @torch.no_grad()
    def designed(self) -> None:
        """The plumbing check only: the reader (stage 1's designed reading) and every cell at the truth; the answer
        programs and their library are given by the check (ctl.progs, ctl.lib)."""
        hot = lambda t, n: 10.0 * F.one_hot(t, n).float() - 5.0
        c = exact_cells("cpu")
        self.a.copy_(hot(c.sbb_d, 10))
        self.b.copy_(hot(c.sbb_c, 2))
        self.aa.copy_(hot(c.adc_d, 10))
        self.ab.copy_(hot(c.adc_c, 2))
        self.ml.copy_(hot(c.mul_lo, 10))
        self.mh.copy_(hot(c.mul_hi, 10))
        DIGIT, MARK_N, MARK_T = 7, 2, 4
        self.dg.fill_(-5.0)
        self.dg[DIGIT:DIGIT + 10] = 5.0
        self.u.copy_(torch.arange(self.V, dtype=torch.float)[None].expand_as(self.u))
        self.wc.fill_(-5.0)
        self.wc[torch.arange(10), DIGIT + torch.arange(10)] = 5.0
        self.anc.fill_(-5.0)
        self.strd.fill_(-5.0)
        self.shape.copy_(hot(torch.tensor(0), len(SHAPES)))
        for f, (tok, step) in enumerate([(MARK_N, 0), (MARK_T, 1), (self.END, 1), (self.END, 1)]):
            self.anc[f, tok] = 5.0
            self.strd[f, step] = 5.0
        self.ctl.phase = "frozen"

    def zero(self) -> Tensor:
        return sum(p.sum() for p in self.parameters()) * 0.0

    # ------------------------------------------------------------------ the cells
    def tables(self):
        """The snapped subtraction (digit, borrow [200])."""
        return self.a.argmax(-1).flatten(), self.b.argmax(-1).flatten()

    def snapped_cells(self) -> Cells:
        """The snapped cells as the machine's tables (libcore.Cells)."""
        return Cells(10, self.a.device).set(sbb_d=self.a.argmax(-1), sbb_c=self.b.argmax(-1), adc_d=self.aa.argmax(-1),
                                            adc_c=self.ab.argmax(-1), mul_lo=self.ml.argmax(-1),
                                            mul_hi=self.mh.argmax(-1))

    def soft_table(self) -> Tensor:
        """The subtraction's softmax, laid out [20 (r, c), 120 (m, 10 digit + 2 borrow)] (stage 1's layout)."""
        T = torch.cat([self.a.float().softmax(-1), self.b.float().softmax(-1)], -1)
        return T.permute(0, 2, 1, 3).reshape(20, 120)

    def ste_cells(self):
        """The prediction path's straight-through tables: the machine's SBB, ADC [200, 12] and MUL [10^4, 20] (rows
        flat over the inputs)."""
        hot = lambda p: ste(F.one_hot(p.argmax(-1), p.shape[-1]).float(), p.float().softmax(-1))
        SBB = torch.cat([hot(self.a), hot(self.b)], -1).reshape(200, 12)
        ADC = torch.cat([hot(self.aa), hot(self.ab)], -1).reshape(200, 12)
        MUL = torch.cat([hot(self.ml), hot(self.mh)], -1).reshape(10 ** 4, 20)
        return SBB, ADC, MUL

    def program(self, i: int | None = None):
        """Answer program i (default: the tournament's argmax), or None before one is found."""
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

    def ext(self, ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Token ids with END at the prompt's end and OUT beyond, padded with OUT by W + 1 on each side (so every
        pointer stays inside): [X, L + 1 + 2 (W + 1)], and the prompt's length [X]."""
        end = mask.long().sum(1)
        p = torch.arange(ids.shape[1] + 1, device=ids.device)[None]
        e = torch.where(p < end[:, None], F.pad(ids, (0, 1)),
                        torch.where(p == end[:, None], self.END, self.OUT))
        return F.pad(e, (W + 1, W + 1), value=self.OUT), end

    def isdig(self, e: Tensor) -> Tensor:
        return (e < self.V) & (self.dg[e.clamp(max=self.V - 1)] > 0)

    def ranks(self, k: Tensor) -> Tensor:
        """The code of hypothesis k [X] -> [X, V]: each digit token's rank among the digit tokens (others 0)."""
        isd = self.dg > 0
        u = self.u[k.clamp(min=0)]
        below = ((u[..., None, :] < u[..., :, None]) & isd).sum(-1)
        return torch.where(isd, below, 0).clamp(max=9)

    def wsm(self) -> Tensor:
        """The writer's code P(token | value) [10, V]: only tokens believed to be digits can be written."""
        return (self.wc.float() + F.logsigmoid(self.dg.float())[None]).softmax(-1)

    def valid_anchor(self, e: Tensor, end: Tensor) -> Tensor:
        p = torch.arange(e.shape[1], device=e.device)[None]
        return (p >= W + 1) & (p <= W + 1 + end[:, None])

    def anchor_logits(self, f: int, e: Tensor, end: Tensor) -> Tensor:
        return self.anc[f].float()[e.clamp(max=self.END)].masked_fill(~self.valid_anchor(e, end), -1e9)

    def anchor(self, f: int, e: Tensor, end: Tensor) -> Tensor:
        """Field f's soft anchor over the prompt's positions and its end [X, Lp] (zero elsewhere)."""
        return self.anchor_logits(f, e, end).softmax(-1)

    def shape_logp(self) -> Tensor:
        return self.shape.float().log_softmax(-1)

    def hard_pos(self, f: int, e: Tensor, end: Tensor) -> Tensor:
        """Field f's snapped pointer: its slots' positions [X, n] under the argmax shape and its argmax parameters."""
        n, Lp = SLOTS[f], e.shape[1]
        m = int(self.shape.argmax())
        j = torch.arange(n, device=e.device)
        if m == 0:
            q = self.anchor_logits(f, e, end).argmax(-1)
            s = 1 if int(self.strd[f].argmax()) == 0 else -1
            pos = q[:, None] + (j[None] + 1) * s
        elif m == 1:
            q = self.anchor_logits(f, e, end).argmax(-1)
            pos = q[:, None] + (self.off[f, :n].argmax(-1) - OFF)[None]
        else:
            pos = self.absp[f, :n, :Lp].argmax(-1)[None].expand(e.shape[0], n)
        return pos.clamp(0, Lp - 1)

    def shape_probs(self, f: int, e: Tensor, end: Tensor, pos: Tensor, length: Tensor) -> Tensor:
        """The probability that field f reads the positions pos [X, K, W + 1] (reading order) as a whole number of
        length [X, K] digits under each shape: slots 0..len-1 on pos[..., :len] and, when the field has more slots,
        slot len on pos[..., len] (right past the number, where the read stops). Returns [X, K, 3]."""
        X, K, _ = pos.shape
        n, Lp = SLOTS[f], e.shape[1]
        dev = e.device
        A = self.anchor(f, e, end)                                           # [X, Lp]
        j = torch.arange(n, device=dev)
        use = (j[None, None] <= length[..., None]).float()                   # [X, K, n]
        pos = pos[..., :n].clamp(0, Lp - 1)
        d = (pos[..., 1] - pos[..., 0]) if n > 1 else torch.ones_like(pos[..., 0])
        sl = self.strd[f].float().softmax(-1)
        step = A.gather(1, (pos[..., 0] - d).clamp(0, Lp - 1)) * torch.where(d > 0, sl[0], sl[1])
        lo = self.off[f, :n].float().log_softmax(-1)                         # [n, 2 OFF + 1]
        rel = pos[..., None] - torch.arange(Lp, device=dev)                  # [X, K, n, Lp]: slot j relative to q
        inr = (rel >= -OFF) & (rel <= OFF)
        lp = lo[j[None, None, :, None].expand_as(rel), (rel + OFF).clamp(0, 2 * OFF)].masked_fill(~inr, -1e4)
        offs = (A.clamp_min(1e-30).log()[:, None] + (lp * use[..., None]).sum(2)).logsumexp(-1).exp()
        la = self.absp[f, :n, :Lp].float().log_softmax(-1)                   # [n, Lp]
        absl = (la[j[None, None].expand(X, K, n), pos] * use).sum(-1).exp()
        return torch.stack([step, offs, absl], -1)

    def shape_rho(self, fields: tuple, e: Tensor, end: Tensor, si: Tensor, ti: Tensor, length: Tensor,
                  hold_shape: bool = False) -> Tensor:
        """P(field f reads run r whole in direction d) [X, len(fields), 3 (runs), 2 (directions: from the run's left
        end, from its right end)], the shapes mixed by the shape switch (held fixed with hold_shape: the lesson then
        teaches each shape's own pointer and leaves the choice of shape to the reading lessons)."""
        X = e.shape[0]
        j = torch.arange(W + 1, device=e.device)
        pos = torch.stack([si[..., None] + j, ti[..., None] - j], 2).view(X, 6, W + 1)
        ln = length[..., None].expand(X, 3, 2).reshape(X, 6)
        w = self.shape_logp().exp()
        w = w.detach() if hold_shape else w
        out = []
        for f in fields:
            fits = (length <= SLOTS[f])[..., None]
            out.append((self.shape_probs(f, e, end, pos, ln) * w).sum(-1).view(X, 3, 2) * fits)
        return torch.stack(out, 1)

    def runs(self, e: Tensor, end: Tensor):
        """The snapped digit runs: rows with exactly three; each run's first and last position and its length."""
        p = torch.arange(e.shape[1], device=e.device)[None]
        isd = (p >= W + 1) & (p < W + 1 + end[:, None]) & self.isdig(e)
        start = isd & ~F.pad(isd[:, :-1], (1, 0))
        stop = isd & ~F.pad(isd[:, 1:], (0, 1))
        ok = start.sum(1) == 3
        si = torch.where(start, p, e.shape[1] - 2).sort(1).values[:, :3]
        ti = torch.where(stop, p, e.shape[1] - 2).sort(1).values[:, :3]
        return ok, si, ti, ti - si + 1

    def which_run(self, f: int, e: Tensor, end: Tensor, si: Tensor, ti: Tensor, ok: Tensor, length: Tensor):
        """The whole run field f's snapped pointer reads: its index [X], its direction [X] (0: from the run's left end)
        and whether it is one [X]: slots 0..len-1 on the run's digits in reading order and, if the field has more
        slots, slot len right past the run in that order (where the read stops; it also tells the direction of a
        one-digit run)."""
        X = e.shape[0]
        pos = self.hard_pos(f, e, end)                                       # [X, n]
        n = SLOTS[f]
        j = torch.arange(n, device=e.device)
        exp = torch.stack([si[..., None] + j, ti[..., None] - j], 2)        # [X, 3, 2, n]
        use = j <= length[..., None, None]                                   # slots 0..len (len < n: the stop slot)
        on = ((pos[:, None, None] == exp) | ~use).all(-1)
        match = (on & (length <= n)[..., None] & ok[:, None, None]).view(X, 6)
        idx = match.float().argmax(1)
        return idx // 2, idx % 2, match.any(1)

    def selected(self, e: Tensor, end: Tensor) -> Tensor:
        """Per row, the hypothesis the snapped N and x pointers make (-1: not a reading of two whole runs)."""
        ok, si, ti, length = self.runs(e, end)
        rn, dn, okn = self.which_run(0, e, end, si, ti, ok, length)
        rx, dx, okx = self.which_run(1, e, end, si, ti, ok, length)
        lut = torch.full((3, 2, 3, 2), -1, dtype=torch.long, device=e.device)
        for h, i in HIDX.items():
            lut[h] = i
        return torch.where(okn & okx & (rn != rx), lut[rn, dn, rx, dx], -1)

    def read(self, f: int, e: Tensor, end: Tensor, code: Tensor):
        """Field f read by its snapped pointer with the code [X, V]: digits [X, n] (LSB first, 0 after the first
        non-digit) and the number of digits read [X]."""
        pos = self.hard_pos(f, e, end)
        t = e.gather(1, pos)
        live = self.isdig(t).long().cumprod(-1).bool()
        return torch.where(live, code.gather(1, t.clamp(max=self.V - 1)), 0), live.sum(1)

    def label_seq(self, e: Tensor, labels: Tensor, valid: Tensor, tpos: Tensor) -> Tensor:
        """The answer's tokens placed at their prompt positions (the evaluator's target positions), OUT elsewhere."""
        labe = torch.full_like(e, self.OUT)
        labe.scatter_(1, torch.where(valid, tpos + W + 1, 0), torch.where(valid, labels, self.OUT))
        labe[:, 0] = self.OUT
        return labe

    def read_answer(self, labe: Tensor, e: Tensor, end: Tensor, code: Tensor) -> Tensor:
        """The answer read at the writer's positions with the reader's code: its digits [X, W], LSB first."""
        pos = self.hard_pos(3, e, end)
        t = labe.gather(1, pos)
        live = self.isdig(t).long().cumprod(-1).bool()
        return torch.where(live, code.gather(1, t.clamp(max=self.V - 1)), 0)

    def reading_weights(self, e: Tensor, end: Tensor) -> Tensor:
        """Per row, the soft weight of each reading hypothesis [X, H]: the N and x pointers' probabilities of reading
        that hypothesis's whole runs (io_lessons' mixture), normalized. The gradient side of the selected reading."""
        _, si, ti, length = self.runs(e, end)
        rho = self.shape_rho((0, 1), e, end, si, ti, length)                               # [X, field, run, dir]
        hyp = torch.tensor(HYP, device=e.device)
        w = rho[:, 0, hyp[:, 0], hyp[:, 1]] * rho[:, 1, hyp[:, 2], hyp[:, 3]] + EPS
        return w / w.sum(-1, keepdim=True)

    def codes_ste(self) -> Tensor:
        """Every hypothesis's code as straight-through one-hots [H, V, 10]: the value is the rank among the digit tokens
        (ranks), the gradient a soft rank's (each token's expected number of digit tokens scored below it)."""
        u, isd = self.u.float(), self.dg > 0
        below = ((u[:, None, :] < u[:, :, None]) & isd[None, None, :]).sum(-1)
        hard = F.one_hot(torch.where(isd[None], below, 0).clamp(max=9), 10).float()
        pd = torch.sigmoid(self.dg.float())
        sr = (torch.sigmoid((u[:, :, None] - u[:, None, :]) / BT_TAU) * pd[None, None, :]).sum(-1) - 0.5 * pd[None]
        soft = (-(torch.arange(10, device=u.device)[None, None] - sr[..., None]) ** 2).softmax(-1)
        return ste(hard, soft)

    def io_lessons(self, e: Tensor, end: Tensor, k: Tensor, code: Tensor, vals: Tensor) -> Tensor:
        """The I/O's label-free lessons (IO.md): the rank codes (Bradley-Terry on x < N, every hypothesis, from stage
        2), the reading (the normalized mixture over complete readings, from stage 3), the tiling (from stage 4) and the
        writer's code round trip on the machine's own numbers `vals` [X, W] (from stage 3)."""
        ctl = self.ctl
        X, Lp = e.shape
        dev = e.device
        loss = self.u.sum() * 0.0
        ok, si, ti, length = self.runs(e, end)
        ctl.rows3 = float(ok.float().mean())
        if ctl.io < 2 or not ok.any():
            return loss
        okf = ok.float()
        j = torch.arange(W, device=dev)
        # each run read in each direction: tokens [X, run, dir, P] (-1 after the run's end)
        tokp = e.gather(1, (si[..., None] + j).clamp(max=Lp - 1).reshape(X, -1)).view(X, 3, W)
        tokm = e.gather(1, (ti[..., None] - j).clamp(min=0).reshape(X, -1)).view(X, 3, W)
        liv = j[None, None] < length[..., None]
        R = F.pad(torch.stack([torch.where(liv, tokp, -1), torch.where(liv, tokm, -1)], 2), (0, P - W), value=-1)
        hyp = torch.tensor(HYP, device=dev)
        Tn, Tx = R[:, hyp[:, 0], hyp[:, 1]], R[:, hyp[:, 2], hyp[:, 3]]     # [X, H, P]: N's and x's readings
        place = torch.where(Tx != Tn, torch.arange(P, device=dev), -1).max(-1).values   # the decisive place
        ta = Tx.gather(2, place.clamp(min=0)[..., None])[..., 0]
        tb = Tn.gather(2, place.clamp(min=0)[..., None])[..., 0]
        u = self.u.float()
        isd = (self.dg > 0).detach()
        umin = u.masked_fill(~isd[None], 1e9).min(-1).values.detach()     # the score of each code's 0
        ua = u[None].expand(X, -1, -1).gather(2, ta.clamp(min=0)[..., None])[..., 0]
        ub = u[None].expand(X, -1, -1).gather(2, tb.clamp(min=0)[..., None])[..., 0]
        both = (ta >= 0) & (tb >= 0) & (place >= 0)                       # two digits differ there
        padx = (ta < 0) & (tb >= 0) & (place >= 0)                        # x' ended first: N's digit must not be 0
        bt = -(F.logsigmoid((ub - ua) / BT_TAU) * both * okf[:, None]).sum() / (both * okf[:, None]).sum().clamp_min(1)
        loss = loss + bt
        hp = torch.where(both, ub > ua, torch.where(padx, ub > umin[None], torch.zeros_like(both)))  # exact verdicts
        rest = 3 - hyp[:, 0] - hyp[:, 2]                                   # the run left for T's two slots
        tfits = length.gather(1, rest[None].expand(X, -1)) <= SLOTS[2]
        passes = ((hp & tfits).float() * okf[:, None]).sum(0) / okf.sum()  # [H]
        ctl.pick = ctl.pick + 1 if float(passes.max()) >= PICK_PASS else 0
        ctl.best = int(passes.argmax())
        if ctl.io >= 3:
            rho = self.shape_rho((0, 1, 2), e, end, si, ti, length)                      # [X, field, run, dir]
            Lk = hp.float() + EPS
            lik, norm = torch.zeros(X, device=dev), torch.zeros(X, device=dev)
            post = torch.zeros(X, len(HYP), device=dev)
            for sN, sx, sT in PERMS:
                for dn in range(2):
                    for dx in range(2):
                        w = rho[:, 2, sT].sum(-1) * rho[:, 0, sN, dn] * rho[:, 1, sx, dx]
                        h = HIDX[(sN, dn, sx, dx)]
                        lik = lik + w * Lk[:, h]
                        norm = norm + w
                        post[:, h] = post[:, h] + w.detach()
            loss = loss - ((lik.clamp_min(1e-30).log() - norm.clamp_min(1e-30).log()) * okf).sum() / okf.sum()
            ctl.chosen = float((post / norm.detach()[:, None].clamp_min(1e-30)).max(1).values[ok].mean())
            if ctl.io >= 4:                                                # whole runs, distinct fields
                loss = loss - (norm.clamp_min(EPS).log() * okf).sum() / okf.sum()
            # the writer's code: write the machine's own numbers, read them back
            with torch.no_grad():
                inv = torch.zeros(X, 10, dtype=torch.long, device=dev)
                for t in torch.nonzero(isd).flatten().tolist():
                    inv.scatter_(1, code[:, t:t + 1], t)                    # the token whose code is v
            rv = self.wsm()[vals, inv.gather(1, vals)]
            loss = loss - rv.clamp_min(EPS).log().mean()
        return loss

    def mine(self, terms: dict) -> None:
        """The miner's step on the fields as read (terms: name -> digits [X, P]): every candidate relation of
        `relations`; newly mined facts are logged."""
        ctl = self.ctl
        for k in ctl.miner.update(relations(terms), ctl.step):
            print(f"LR mined step={ctl.step} fact={k.replace(' ', '')}", flush=True)

    def mine_answers(self, aux: dict, yd: Tensor) -> None:
        """The miner on the answers (reported only): the relations between the answer y and N, x, T."""
        ctl = self.ctl
        if ctl.io < 3:
            return
        with torch.no_grad():
            terms = {"N": aux["M"], "x": F.pad(aux["Xd"], (0, P - W)), "T": F.pad(aux["Cd"], (0, P - K)),
                     "y": F.pad(yd, (0, P - W))}
            rels = {k: v for k, v in relations(terms).items() if "y" in k.split()}
            for k in ctl.miner.update(rels, ctl.step):
                print(f"LR mined step={ctl.step} fact={k.replace(' ', '')}", flush=True)


    # ------------------------------------------------------------------ the forward pass
    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        if attention_mask is None:
            attention_mask = input_ids != 0
        with torch.autocast(input_ids.device.type, enabled=False):
            return self._forward(input_ids, attention_mask)

    def _forward(self, input_ids: Tensor, attention_mask: Tensor):
        dev = input_ids.device
        e, end = self.ext(input_ids, attention_mask)
        k = self.selected(e, end)
        rows, L, V = input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
        if self.training:
            code = self.ranks(k)
            Nd, width = self.read(0, e, end, code)
            Xd, _ = self.read(1, e, end, code)
            Cd, tcount = self.read(2, e, end, code)
            small = answer_rows(Cd)
            aux = self.lessons(Nd, Xd, Cd, small, width, e, end, k, code)
            return torch.zeros(rows, L, V, device=dev), aux
        ctl = self.ctl
        if ctl.used is None:
            ctl.used = torch.zeros(200, device=dev)
            ctl.used_count = torch.zeros(200, device=dev)
            self.report_tables(e, end)
        logits = self.predict_ste(e, end, k, L)
        self.audit()
        return logits, None

    def outer_ste(self, Sx: Tensor, Sn: Tensor, Cnt: Tensor, SBB: Tensor, ADC: Tensor, MUL: Tensor) -> Tensor:
        """The answer's program on one-hot digits: a clean machine with x = Sx in X, N = Sn in N and T's digits Cnt in
        S, the selected program run on it (libcore.SteMachine; a count loop counts down by the subtraction table);
        returns A's low W places [X, W, 10]. Straight-through over the tournament's programs: with autograd every
        program runs (the gradient reaches the tournament's logits); without, the argmax program only."""
        ctl = self.ctl
        X = Sx.shape[0]
        zero = onehot0(X, PR, Sx.device)
        regs = [zero, zpad(Sx, 0, PR - W), zero, zpad(Sn, 0, PR - W), zpad(Cnt, 0, PR - K)]
        z2 = torch.zeros(X, 2, device=Sx.device)
        z2[:, 0] = 1
        z10 = torch.zeros(X, 10, device=Sx.device)
        z10[:, 0] = 1
        m = SteMachine(SBB, ADC, MUL, PR, damp, used=ctl.used_count)
        n = len(ctl.progs)
        sel = ste(F.one_hot(self.prog[:n].argmax(), n).float(), self.prog[:n].float().softmax(-1))
        run = lambda i: m.run(regs, z2, z10, z10, ctl.progs[i], ctl.lib)[0][0][:, :W]
        if not torch.is_grad_enabled():
            return run(int(sel.argmax()))
        return sum(sel[i] * run(i) for i in range(n))

    def predict_ste(self, e: Tensor, end: Tensor, k: Tensor, L: int) -> Tensor:
        """The prediction path inside autograd: the reader, the selected answer program, and the writer, on one-hot
        digits with straight-through tables (stage 1's reader and writer, line for line)."""
        ctl, V = self.ctl, self.V
        X, Lp = e.shape
        E1 = F.one_hot(e, V + 2).float()
        DG = ste((self.dg > 0).float(), torch.sigmoid(self.dg.float()))
        isdv = F.pad(DG, (0, 2))
        hot = lambda lg: ste(F.one_hot(lg.argmax(-1), lg.shape[-1]).float(), lg.softmax(-1))
        anchors = [hot(self.anchor_logits(f, e, end)) for f in range(4)]
        steps = [hot(self.strd[f].float()) for f in range(4)]
        H = len(HYP)
        Wh = ste(F.one_hot(k.clamp(min=0), H).float(), self.reading_weights(e, end))
        code = torch.einsum("xh,hvk->xvk", Wh, self.codes_ste())
        shp = hot(self.shape.float())

        def slot(f: int, j: int) -> Tensor:
            off = hot(self.off[f, j].float())
            ab = hot(self.absp[f, j, :Lp].float())[None].expand(X, Lp)
            opts = [lambda: steps[f][0] * shift(anchors[f], j + 1) + steps[f][1] * shift(anchors[f], -(j + 1)),
                    lambda: (pick(off, [shift(anchors[f], o - OFF) for o in range(2 * OFF + 1)])
                             if torch.is_grad_enabled() else shift(anchors[f], int(off.argmax()) - OFF)),
                    lambda: ab]
            if not torch.is_grad_enabled():
                return opts[int(shp.argmax())]()
            return sum(shp[m] * opts[m]() for m in range(len(SHAPES)))

        def field_(f: int) -> Tensor:
            live, out = torch.ones(X, device=e.device), []
            zero = F.one_hot(torch.zeros(X, dtype=torch.long, device=e.device), 10).float()
            for j in range(SLOTS[f]):
                tok = torch.einsum("xp,xpv->xv", slot(f, j), E1)
                live = live * (tok @ isdv)
                v = torch.einsum("xv,xvk->xk", tok[:, :V], code)
                out.append(live[:, None] * v + (1 - live)[:, None] * zero)
            return torch.stack(out, 1)

        SBB, ADC, MUL = self.ste_cells()
        Sx, Cnt, Sn = field_(1), field_(2), field_(0)
        s = self.outer_ste(Sx, Sn, Cnt, SBB, ADC, MUL) if ctl.progs else Sx
        wcode = hot(self.wc.float() + F.logsigmoid(self.dg.float())[None])
        out = torch.zeros(X, Lp, V, device=e.device)
        for j in range(W):
            out = out + slot(3, j)[:, :, None] * (s[:, j] @ wcode)[:, None, :]
        return 20.0 * out[:, W + 1:W + 1 + L] - 20.0

    # ------------------------------------------------------------------ the hard machine (training's checks)
    def run_outer(self, Sd: Tensor, C: Tensor, Nd: Tensor, prog) -> Tensor:
        """The answer's program on digits (training's checks): fresh machines (run_prog's dirty starts) with x = Sd in X,
        N = Nd in N and the count's digits C in S, the program run with the snapped cells (libcore.OuterMachine: every
        row at once, on a GPU each stretch and each count loop's body one CUDA graph). Returns A's low W places
        [X, W]."""
        cells = self.ctl.cells or self.snapped_cells()
        m = outer_machine(round_settings(cells, PR), prog, self.ctl.lib, Sd.shape[0], Sd.device)
        return m.run(Sd, Nd, C, gen_on(Sd.device, 0))[:, :W].clone().to(Sd.dtype)

    # ------------------------------------------------------------------ the lessons
    def table_lessons(self, Nd, Xd, Cd, terms, orders, nonzero) -> tuple[Tensor, float]:
        """The subtraction table's lessons (general_lessons' kind 1 and 2, as in stage 1, on registers made of the rows'
        x and other rows' x: there is no machine square here): sub.order on every mined order fact, then the digit
        laws (sub.order on the entries the batch's subtractions read, sub.bijective), then the pieces' laws
        (sub.inverse with the learned adder a - (0 - b), sub.identity, add.commute, the facts' borrows, sub.wrap,
        sub.zero_minus). Returns the loss and the hard checks' share passing."""
        ctl = self.ctl
        dev = Xd.device
        Td, Tb = self.tables()
        TDC = self.soft_table()
        X = Xd.shape[0]
        M = F.pad(Nd, (0, P - W))
        hot = lambda v: F.one_hot(v, 10).float()
        log = lambda p: p.clamp_min(1e-12).log()
        digits_nll = lambda Y, target: -log(Y).gather(-1, target[..., None]).squeeze(-1).sum(-1)
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
        pieces = digits_on and ctl.step - ctl.gated_on >= LATE_AFTER
        sc, sa = False, True
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
                    okf = okf & (sub_hard(terms[a_], terms[b_], Td, Tb)[1] == 1) & \
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
        weight = torch.ones(P, device=dev)
        weight[0] = ID_PLACE0

        def inverse_soft(Dv, Bv, Cs, bc):
            Es, be = link_soft(hot(Dv), Bv, TDC, sc)
            back, bf = sub_soft(Es, Cs, TDC, None, sc)
            pe, pf, pc = be[:, -1, 1], bf[:, -1, 1], bc[:, -1, 1]
            agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc
            nll = (-log(back).gather(-1, Dv[..., None]).squeeze(-1) * weight).sum(-1) - ID_BORROW * log(agree)
            return nll.mean() / (weight.sum() + ID_BORROW)
        if pieces:
            Cs, bc = link_soft(hot(Z), M, TDC, sc)
            loss = loss + inverse_soft(D, Mr, Cs.repeat_interleave(NL, 0), bc.repeat_interleave(NL, 0))
            C2s, bc2 = link_soft(hot(zero), R2, TDC, sc)
            loss = loss + inverse_soft(Rg, R2, C2s, bc2)
            bc = bc[:, -1]
        checks = []
        for A, B, target in ((Rg, zero, Rg), (Rg, Rg, zero)):
            if pieces:
                Y, b = link_soft(hot(A), B, TDC, sa)
                loss = loss + (digits_nll(Y, target) - log(b[:, -1, 0])).mean() / (P + 1)
            checks.append((A, B, target, 0))
        if pieces:
            Y2, b2 = sub_soft(hot(Z), Cs, TDC, None, sc)
            agree2 = bc[:, 1] * b2[:, -1, 1] + bc[:, 0] * b2[:, -1, 0]
            loss = loss + (digits_nll(Y2, M) - log(agree2)).mean() / (P + 1)

            def commute_soft(A, B, CB, bcb):
                A1, b1 = sub_soft(hot(A), CB, TDC, None, sc)
                CA, bca = link_soft(hot(torch.zeros_like(A)), A, TDC, sc)
                A2, b2_ = sub_soft(hot(B), CA, TDC, None, sc)
                c1 = bcb * b1[:, -1, 0]
                c2 = bca[:, -1, 1] * b2_[:, -1, 0]
                return -(log((A1 * A2).sum(-1)).sum(-1) + log(c1 * c2 + (1 - c1) * (1 - c2))).mean() / (P + 1)
            loss = loss + commute_soft(Rg, MW, Cs.repeat_interleave(W, 0), bc[:, 1].repeat_interleave(W, 0))
            loss = loss + commute_soft(Rg, R2, C2s, bc2[:, -1, 1])
        for a_, b_ in orders:
            A_, B_ = terms[a_], terms[b_]
            if pieces:
                loss = loss - log(link_soft(hot(A_), B_, TDC, sc)[1][:, -1, 1]).mean()
                loss = loss - log(link_soft(hot(B_), A_, TDC, sc)[1][:, -1, 0]).mean()
                Ew = link_soft(hot(A_), B_, TDC, sc)[0]
                loss = loss - log(link_soft(Ew, B_, TDC, sc)[1][:, -1, 0]).mean()
            checks.append((A_, B_, None, 1))
            checks.append((B_, A_, None, 0))
        for a_ in nonzero:
            if pieces:
                loss = loss - log(link_soft(hot(torch.zeros_like(terms[a_])), terms[a_], TDC, sc)[1][:, -1, 1]).mean()
            checks.append((torch.zeros_like(terms[a_]), terms[a_], None, 1))
        with torch.no_grad():
            Eh, beh, idx = sub_hard(D, Mr, Td, Tb)
            tally(ctl.visits, idx, 200)
            Ch, bch, idc = sub_hard(Z, M, Td, Tb)
            backh, bfh, idf = sub_hard(Eh, Ch.repeat_interleave(NL, 0), Td, Tb)
            tally(ctl.visits, idc, 200)
            tally(ctl.visits, idf, 200)
            table = float(((backh == D).all(-1) & (beh == bch.repeat_interleave(NL, 0) - bfh)).float().mean())
            E2h, be2, i2 = sub_hard(Rg, R2, Td, Tb)
            C2h, bc2h, i3 = sub_hard(zero, R2, Td, Tb)
            back2, bf2, i4 = sub_hard(E2h, C2h, Td, Tb)
            for ii in (i2, i3, i4):
                tally(ctl.visits, ii, 200)
            table = min(table, float(((back2 == Rg).all(-1) & (be2 == bc2h - bf2)).float().mean()))
            A1h, c1h = add_learned(Rg, R2, Td, Tb, ctl.visits)
            A2h, c2h = add_learned(R2, Rg, Td, Tb, ctl.visits)
            table = min(table, float(((A1h == A2h).all(-1) & (c1h == c2h)).float().mean()))
            Y2h, b2h, id2 = sub_hard(Z, Ch, Td, Tb)
            tally(ctl.visits, id2, 200)
            table = min(table, float(((Y2h == M).all(-1) & (b2h == bch)).float().mean()))
            A1h, c1h = add_learned(Rg, MW, Td, Tb, ctl.visits)
            A2h, c2h = add_learned(MW, Rg, Td, Tb, ctl.visits)
            table = min(table, float(((A1h == A2h).all(-1) & (c1h == c2h)).float().mean()))
            for A, B, target, borrow in checks:
                Yh, bh, ida = sub_hard(A, B, Td, Tb)
                tally(ctl.visits, ida, 200)
                okc = bh == borrow
                if target is not None:
                    okc = okc & (Yh == target).all(-1)
                table = min(table, float(okc.float().mean()))
            for a_, b_ in orders:
                Ewh, _, idw = sub_hard(terms[a_], terms[b_], Td, Tb)
                tally(ctl.visits, idw, 200)
                table = min(table, float((sub_hard(Ewh, terms[b_], Td, Tb)[1] == 0).float().mean()))
        return loss, table

    def cell_lessons(self, Cd: Tensor) -> Tensor:
        """The addition by sub.inverse, the multiply-accumulate by mul.zero and mul.ladder (every entry, the targets from
        the snapped tables before them). Each cell's own check is that its snapped table equals its law's targets."""
        ctl = self.ctl
        Td, Tb = self.tables()
        loss = self.zero()
        with torch.no_grad():
            td, tc = adc_targets(Td, Tb)
            m = td >= 0
            Ad, Ac = self.aa.argmax(-1), self.ab.argmax(-1)
            ctl.adc_ok = bool(m.all()) and bool((Ad == td).all() and (Ac == tc).all())
        if bool(m.any()):
            loss = loss + F.cross_entropy(self.aa.float()[m], td[m]) + F.cross_entropy(self.ab.float()[m], tc[m])
        with torch.no_grad():
            Ml, Mh = self.ml.argmax(-1), self.mh.argmax(-1)
            lo, hi = mul_targets(Ml, Mh, Ad, Ac)
            ctl.mul_ok = bool((Ml == lo).all() and (Mh == hi).all())
        loss = loss + F.cross_entropy(self.ml.float().reshape(-1, 10), lo.flatten()) + \
            F.cross_entropy(self.mh.float().reshape(-1, 10), hi.flatten())
        Gc = self.g.argmax(-1)                                              # sp2's loop law (diag: nothing reads it)
        with torch.no_grad():
            one = F.one_hot(torch.zeros_like(Cd[:, 0]), Cd.shape[1])
            b_run = sub_hard(Cd, one, Td, Tb)[1]                           # a count of at least 1, minus 1
            b_stop = sub_hard(torch.zeros_like(Cd), one, Td, Tb)[1]        # 0 - 1
            live = (Cd != 0).any(1)
        lg = self.g.float().log_softmax(-1)
        loss = loss - lg[b_stop, 0].mean()
        if bool(live.any()):            # (a batch whose counts all read 0, as when T's reading turns, has no run row)
            loss = loss - lg[b_run[live], 1].mean()
        ctl.gate_ok = bool((Gc[b_run[live]] == 1).all() and (Gc[b_stop] == 0).all())
        return loss

    def climb_unit(self, Nd: Tensor, Xd: Tensor) -> None:
        """One unit of the climb (phase C): the read rows join the pool, then the next search or sleep."""
        ctl = self.ctl
        if ctl.climb is None:
            ctl.cells = self.snapped_cells()
            ctl.climb = SelfClimb(ctl.cells, Xd.device, PR, chunk=1 << 21 if Xd.device.type == "cuda" else 1 << 18,
                                  **SP)
            print(f"LR climb_starts step={ctl.step} cells: subtraction/addition/multiply snapped, "
                  f"right vs the truth (report only): {self.cells_right()}", flush=True)
        pool = torch.cat([torch.stack([n, x]) for n, x in ctl.pool], 1) if ctl.pool else None
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

    def outer_unit(self, deadline) -> None:
        """The answer's program (phase C, once the round is found): one search (spcore.SelfClimb.search_outer) with the
        simplest rounds as routines, against the kept rows of every count, once they show COUNT_MIN counts; again
        after OUTER_RETRY steps or once a new count has come, while none is found."""
        ctl = self.ctl
        keys = sorted(ctl.count_pool)
        if len(keys) < COUNT_MIN or (ctl.outer_tried is not None and ctl.outer_tried[1] == keys and
                                     ctl.step - ctl.outer_tried[0] < OUTER_RETRY):
            return
        ctl.outer_tried = (ctl.step, keys)
        rounds = sorted(ctl.climb.rounds, key=lambda p: (cost(expand(p, ctl.climb.lib)), show(p)))[:PROG_SLOTS]
        rows = tuple(torch.cat([ctl.count_pool[k][i] for k in keys]) for i in range(4))
        progs, lib, rec = ctl.climb.search_outer(rows, rounds, deadline, OUTER_STATES, OUTER_COST, OUTER_BODY,
                                                 OUTER_NP)
        ctl.climb.records.append(rec)
        line = (f"answer program: cost {rec['found_cost']} levels {rec['level_sizes']} {rec['seconds']}s "
                f"{rec['stop']} {rec['programs'][:3]} ({rec['rows']} rows of {rec['counts']} counts, {rec['items']} "
                f"items, {rec['bodies']} loop bodies)")
        ctl.climb_log.append(line)
        print(f"LR outer step={ctl.step} {line}", flush=True)
        if progs:
            ctl.lib = lib
            ctl.progs = [tuple(p) for p in sorted(progs, key=lambda p: (cost(expand(p, lib)), show(p)))[:PROG_SLOTS]]
            ctl.phase, ctl.in_phase = "D", 0
            ctl.history.append((ctl.step, "answer_program", len(ctl.progs)))
            print(f"LR round_found step={ctl.step} programs: {[show(p) for p in ctl.progs]} rounds: "
                  f"{ {f'T.round{i + 1}': show(p) for i, p in enumerate(rounds)} } library: "
                  f"{ {k: show(v) for k, v in ctl.lib.items()} }", flush=True)

    def cells_right(self) -> str:
        c, t = self.snapped_cells(), exact_cells(self.a.device)
        return ",".join(f"{n}:{int((getattr(c, n) == getattr(t, n)).sum())}/{getattr(t, n).numel()}"
                        for n in ("sbb_d", "sbb_c", "adc_d", "adc_c", "mul_lo", "mul_hi"))

    def lessons(self, Nd, Xd, Cd, small, width, e, end, k, code) -> dict:
        """The label-free lessons (the label-dependent ones run in token_training_loss on what this returns)."""
        ctl = self.ctl
        ctl.step += 1
        dev = Xd.device
        if ctl.visits is None:
            ctl.visits = torch.zeros(200, dtype=torch.long, device=dev)
        common = {"ctl": ctl, "model": self, "e": e, "end": end, "k": k, "code": code, "Xd": Xd, "Nd": Nd, "Cd": Cd,
                  "M": F.pad(Nd, (0, P - W)), "small": small, "width": width, "exact": 0.0, "table_exact": 0.0}
        if ctl.phase == "frozen":
            return {"loss": self.zero(), **common}
        M = F.pad(Nd, (0, P - W))
        terms = {"N": M, "x": F.pad(Xd, (0, P - W)), "T": F.pad(Cd, (0, P - K))}
        if ctl.io >= 3:
            with torch.no_grad():
                self.mine(terms)
        usable = ctl.phase != "A" or ctl.io >= 4
        orders = [(a, b) for a in ORDER_TERMS for b in ORDER_TERMS if a != b and usable and ctl.miner.holds(f"{a} < {b}")]
        nonzero = [a for a in ORDER_TERMS if usable and ctl.miner.holds(f"{a} != 0")]
        loss, table = self.table_lessons(Nd, Xd, Cd, terms, orders, nonzero)
        exact = table
        if ctl.cells_on:
            loss = loss + self.cell_lessons(Cd)
            exact = min(exact, float(ctl.adc_ok and ctl.mul_ok and ctl.gate_ok))
        if ctl.phase in ("B", "C", "D") and ctl.io >= 4:                    # the read rows: the practice problems'
            ctl.pool.append((Nd.detach(), Xd.detach()))
            while sum(n.shape[0] for n, _ in ctl.pool) > POOL:
                ctl.pool.pop(0)
        if ctl.phase == "C":
            self.climb_unit(Nd, Xd)
        if MODE == "learned":
            loss = loss + self.io_lessons(e, end, k, code, Xd)
        common.update(exact=exact, table_exact=table)
        return {"loss": loss, **common}

    def round_lesson(self, aux: dict, yd: Tensor) -> Tensor:
        """The answer programs' tournament, by the answers: every program runs on the batch's answer rows (the hard
        machine); a program passes when it reproduces every one of them; the loss is -log of the passing programs'
        total weight (tourney_nll; nothing when none passes)."""
        ctl = self.ctl
        sel = torch.nonzero(aux["small"]).flatten()
        n = len(ctl.progs)
        if sel.numel() == 0 or n == 0:
            return self.prog.sum() * 0.0
        with torch.no_grad():
            ok = torch.stack([(self.run_outer(aux["Xd"][sel], aux["Cd"][sel], aux["Nd"][sel], p) == yd[sel]).all(1)
                              for p in ctl.progs])                           # [programs, rows]
            ctl.round_fit = ok.float().mean(1).tolist()
            ctl.ans_ok = float(ok[int(self.prog[:n].argmax())].float().mean())
        t = tourney_nll(self.prog[:n].float().log_softmax(-1), ok.all(1))
        return t if t is not None else self.prog.sum() * 0.0

    def keep_answers(self, aux: dict, yd: Tensor) -> None:
        """The answer rows (the batch's smallest T, as read: answer_rows) join the round's pool (phases C and D); the
        pool holds the smallest T the batches have shown (general lessons v2's rule; on H1 that is T = 1)."""
        ctl = self.ctl
        sel = torch.nonzero(aux["small"]).flatten()
        if sel.numel() == 0 or ctl.phase not in ("C", "D") or ctl.io < 4:
            return
        t = answer_t(aux["Cd"][sel[:1]])
        if ctl.ans_t is None or t < ctl.ans_t:
            ctl.ans_pool, ctl.ans_t = [], t
        if t != ctl.ans_t:
            return
        ctl.ans_pool.append((aux["Nd"][sel].detach(), aux["Xd"][sel].detach(), yd[sel].detach()))
        while sum(a[0].shape[0] for a in ctl.ans_pool) > ANS_POOL:
            ctl.ans_pool.pop(0)

    def keep_counts(self, aux: dict, yd: Tensor) -> None:
        """The answer rows whose count reads the same both ways (T's two readings agree, and the reader's count is that
        one: a count of one digit, or a palindrome) join the answer program's pool (phase C), at most COUNT_EACH of
        each count and COUNT_POOL in all, the newest kept. T's direction is not needed to read them."""
        ctl = self.ctl
        if ctl.phase != "C" or ctl.io < 4:
            return
        with torch.no_grad():
            ok, _, _, _, _, Cp, Cm = self.t_readings(aux)
            sel = torch.nonzero(ok & (Cp == Cm).all(1) & (Cp == aux["Cd"]).all(1)).flatten()
            if sel.numel() == 0:
                return
            Cs = aux["Cd"][sel]
            for key in {tuple(r) for r in Cs.tolist()}:
                idx = sel[(Cs == torch.tensor(key, device=Cs.device)).all(1)]
                new = (aux["Nd"][idx], aux["Xd"][idx], aux["Cd"][idx], yd[idx])
                old = ctl.count_pool.get(key)
                ctl.count_pool[key] = new if old is None else tuple(torch.cat([a, b]) for a, b in zip(old, new))
            share = min(COUNT_EACH, max(1, COUNT_POOL // len(ctl.count_pool)))
            for key, v in ctl.count_pool.items():
                ctl.count_pool[key] = tuple(t[-share:] for t in v)

    def io_label_lessons(self, aux: dict, labels: Tensor, valid: Tensor, tpos: Tensor):
        """The I/O's label lessons (IO.md): the answer round trip (the writer's pointer, digit-ness), the token-side code
        round trip (the writer's code) and, once the machine is exact, T's direction by speculation."""
        ctl = self.ctl
        e, end, code = aux["e"], aux["end"], aux["code"]
        X, Lp = e.shape
        dev = e.device
        labe = self.label_seq(e, labels, valid, tpos)
        wsm = self.wsm()
        # the writer's pointer: its slots 0..m-1 on the answer's own positions, LSB first, and slot m right past them
        m = valid.long().sum(1)
        last = tpos.gather(1, (m - 1).clamp(min=0)[:, None])[:, 0] + W + 1   # the answer's last position: its LSB
        apos = last[:, None] - torch.arange(W + 1, device=dev)[None]
        pw = (self.shape_probs(3, e, end, apos[:, None], m[:, None])[:, 0] * self.shape_logp().exp()).sum(-1)
        loss = -pw.clamp_min(1e-30).log().mean()
        lv = labe < self.V
        rows_l, tl = torch.nonzero(lv)[:, 0], labe[lv]
        loss = loss - wsm[code[rows_l, tl], tl].clamp_min(EPS).log().mean()   # read an answer token, write it back
        # the answer's tokens are digits, read slot by slot: IO.md's round trip's liveness term on the answer's own
        # positions (slot j is live only if slots 0..j read digits)
        atok = labe.gather(1, apos[:, :W].clamp(0, Lp - 1))
        dl = torch.where(atok < self.V, torch.sigmoid(self.dg.float()[atok.clamp(max=self.V - 1)]),
                         torch.zeros((), device=dev))
        inans = torch.arange(W, device=dev)[None] < m[:, None]
        loss = loss - (dl.clamp_min(EPS).log().cumsum(-1) * inans).sum(-1).mean()
        with torch.no_grad():                                               # the hard round trips
            yd = self.read_answer(labe, e, end, code)
            pos = self.hard_pos(3, e, end)
            back = self.wsm().argmax(-1)[yd]
            o = torch.full_like(e, -1).scatter_(1, pos, back)
            ans = (torch.where(valid, o.gather(1, torch.where(valid, tpos + W + 1, 0)), labels) == labels).all(1)
            isd = self.dg > 0
            inv_ok = bool((self.wsm().argmax(-1)[code[:, isd]] == torch.nonzero(isd).flatten()[None]).all())
            ctl.io_exact = bool(ans.all()) and inv_ok and int(isd.sum()) == 10 and bool((aux["k"] >= 0).all())
            ctl.io_parts = (f"answers_ok={float(ans.float().mean()):.4f} code_inverse={int(inv_ok)} "
                            f"digit_tokens={int(isd.sum())} rows_read={float((aux['k'] >= 0).float().mean()):.4f}")
            if not ctl.io_exact and ctl.phase == "D":
                c0 = code[0, isd].tolist()
                wa = self.wsm().argmax(-1).tolist()
                print(f"LR io_not_exact step={ctl.step} {ctl.io_parts} digit_tokens={torch.nonzero(isd).flatten().tolist()} "
                      f"code_row0={c0} writer={wa} codes_differ={int((code[:, isd] != code[:1, isd]).any())}", flush=True)
        if ctl.io == 4 and ctl.phase == "D" and ctl.ans_ok == 1.0 and ctl.streak >= T_AFTER:
            loss = loss + self.t_lesson(aux, yd)
        return loss, yd


    def shape_logprobs(self, f: int, e: Tensor, end: Tensor, pos: Tensor, length: Tensor) -> Tensor:
        """shape_probs in log space (the same three terms, no clamp): log P(field f reads pos as a whole number of
        length digits) under each shape [X, K, 3]. T's direction lesson needs it: a confidently wrong reading leaves the
        right one a weight far below float32's range, and the log of a clamped weight then has a gradient up to 1e30."""
        X, K, _ = pos.shape
        n, Lp = SLOTS[f], e.shape[1]
        dev = e.device
        lA = self.anchor_logits(f, e, end).log_softmax(-1)                   # [X, Lp]
        j = torch.arange(n, device=dev)
        use = (j[None, None] <= length[..., None]).float()                   # [X, K, n]
        pos = pos[..., :n].clamp(0, Lp - 1)
        d = (pos[..., 1] - pos[..., 0]) if n > 1 else torch.ones_like(pos[..., 0])
        lsl = self.strd[f].float().log_softmax(-1)
        step = lA.gather(1, (pos[..., 0] - d).clamp(0, Lp - 1)) + torch.where(d > 0, lsl[0], lsl[1])
        lo = self.off[f, :n].float().log_softmax(-1)                         # [n, 2 OFF + 1]
        rel = pos[..., None] - torch.arange(Lp, device=dev)                  # [X, K, n, Lp]: slot j relative to q
        inr = (rel >= -OFF) & (rel <= OFF)
        lp = lo[j[None, None, :, None].expand_as(rel), (rel + OFF).clamp(0, 2 * OFF)].masked_fill(~inr, -1e4)
        offs = (lA[:, None] + (lp * use[..., None]).sum(2)).logsumexp(-1)
        la = self.absp[f, :n, :Lp].float().log_softmax(-1)                   # [n, Lp]
        absl = (la[j[None, None].expand(X, K, n), pos] * use).sum(-1)
        return torch.stack([step, offs, absl], -1)

    def shape_logrho(self, f: int, e: Tensor, end: Tensor, si: Tensor, ti: Tensor, length: Tensor) -> Tensor:
        """log shape_rho of one field with the shape held fixed: log P(field f reads run r whole in direction d)
        [X, 3 (runs), 2 (directions)], -inf where the run does not fit the field's slots."""
        X = e.shape[0]
        j = torch.arange(W + 1, device=e.device)
        pos = torch.stack([si[..., None] + j, ti[..., None] - j], 2).view(X, 6, W + 1)
        ln = length[..., None].expand(X, 3, 2).reshape(X, 6)
        lw = self.shape_logp().detach()
        lr = (self.shape_logprobs(f, e, end, pos, ln) + lw).logsumexp(-1).view(X, 3, 2)
        return lr.masked_fill(~(length <= SLOTS[f])[..., None], float("-inf"))

    def t_readings(self, aux: dict):
        """T's two readings (IO.md): the run left over by N's and x's runs, read from its left end (Cp) and from its
        right end (Cm), as digits [X, K]; with the rows whose runs are read (ok), the runs (si, ti, length) and T's run."""
        e, end, code = aux["e"], aux["end"], aux["code"]
        Lp = e.shape[1]
        ok, si, ti, length = self.runs(e, end)
        rn, _, _ = self.which_run(0, e, end, si, ti, ok, length)
        rx, _, _ = self.which_run(1, e, end, si, ti, ok, length)
        rt = (3 - rn - rx).clamp(0, 2)
        a0, b0 = si.gather(1, rt[:, None])[:, 0], ti.gather(1, rt[:, None])[:, 0]
        j = torch.arange(K, device=e.device)
        tp = e.gather(1, (a0[:, None] + j).clamp(max=Lp - 1))
        tm = e.gather(1, (b0[:, None] - j).clamp(min=0))
        lp = j[None] < (b0 - a0 + 1)[:, None]
        Cp = torch.where(lp, code.gather(1, tp.clamp(max=self.V - 1)), 0)
        Cm = torch.where(lp, code.gather(1, tm.clamp(max=self.V - 1)), 0)
        return ok, si, ti, length, rt, Cp, Cm

    def t_lesson(self, aux: dict, yd: Tensor) -> Tensor:
        """T's direction by speculation (IO.md; stage 1's lesson with the answer program's machine): on rows whose two
        readings of T differ, the answer's program runs from both, and the answers keep the reading that reproduces
        them."""
        ctl = self.ctl
        e, end, Xd, Nd = aux["e"], aux["end"], aux["Xd"], aux["Nd"]
        ok, si, ti, length, rt, Cp, Cm = self.t_readings(aux)
        rows = torch.nonzero(ok & (Cp != Cm).any(1)).flatten()[:T_ROWS]
        if rows.numel() == 0 or not ctl.progs:
            return self.strd.sum() * 0.0
        with torch.no_grad():
            n = rows.numel()
            s2 = self.run_outer(Xd[rows].repeat(2, 1), torch.cat([Cp[rows], Cm[rows]]), Nd[rows].repeat(2, 1),
                                self.program())
            Lpl, Lmi = (s2[:n] == yd[rows]).all(1).float(), (s2[n:] == yd[rows]).all(1).float()
        # -log of the reproducing readings' share of the two weights, in log space: the weights' logs (shape_logrho)
        # and logsumexp, so that a weight far below float32's range keeps a finite, bounded gradient
        lw = self.shape_logrho(2, e[rows], end[rows], si[rows], ti[rows], length[rows])
        lw = lw.gather(1, rt[rows][:, None, None].expand(-1, 1, 2))[:, 0]           # [n, 2]: from the left, the right
        good = torch.stack([Lpl, Lmi], 1) > 0                                         # the readings that reproduce
        hit = good.any(1)
        keep = torch.nonzero((good & (lw > float("-inf"))).any(1)).flatten()
        lk, gk = lw[keep], good[keep]
        lt = -(lk.masked_fill(~gk, float("-inf")).logsumexp(1) - lk.logsumexp(1)).sum() / hit.sum().clamp_min(1)
        with torch.no_grad():
            pT = self.hard_pos(2, e, end)[rows]
            hs = int(((pT[:, 1] - pT[:, 0]) > 0).float().mean() > 0.5) * 2 - 1
            mine = Lpl if hs > 0 else Lmi
            share = float((lw[:, 0 if hs > 0 else 1] - lw.logsumexp(1)).exp().nan_to_num(0.0).mean())
            ctl.t_log = f"rows={n} hits={int(hit.sum())} snapped_right={float(mine.mean()):.3f} share={share:.4f}"
            if bool(mine.all()) and share >= T_DONE:
                ctl.io = 5
                ctl.history.append((ctl.step, "T_resolved", n))
                print(f"LR io_stage=5 T_resolved step={ctl.step} {ctl.t_log}", flush=True)
        return lt

    def control(self, aux: dict, loss: float) -> None:
        """The schedule, once per step (after the label lessons): the reader's stages, the phases, the freeze."""
        ctl = self.ctl
        exact, table = aux["exact"], aux["table_exact"]
        before = ctl.io
        if ctl.io == 1 and ctl.rows3 >= 0.99:
            ctl.io = 2
            ctl.history.append((ctl.step, "io2_digits"))
        elif ctl.io == 2 and ctl.pick >= PICK_WIN:
            ctl.io = 3
            ctl.history.append((ctl.step, "io3_ranked", HYP[ctl.best]))
        elif ctl.io == 3 and ctl.chosen >= CHOSEN:
            ctl.io = 4
            ctl.history.append((ctl.step, "io4_reading", round(ctl.chosen, 4)))
        if ctl.io != before:
            print(f"LR io_stage={ctl.io} step={ctl.step} history={ctl.history[-1]}", flush=True)
        ctl.win.append(table)
        ctl.in_phase += 1
        mean = sum(ctl.win) / len(ctl.win)
        if ctl.step % LOG_EVERY == 0 or ctl.step == 1:
            print(f"LR step={ctl.step} phase={ctl.phase} io={ctl.io} loss={loss:.4f} exact={exact:.4f} "
                  f"table={table:.4f} win={mean:.4f} cells_on={int(ctl.cells_on)} adc={int(ctl.adc_ok)} "
                  f"mul={int(ctl.mul_ok)} io_exact={ctl.io_exact} chosen={ctl.chosen:.4f} "
                  f"ans_ok={ctl.ans_ok:.4f} streak={ctl.streak} facts_streak={ctl.facts_streak} "
                  f"order_digits={ctl.gated_on} rows_visited={int((ctl.visits > 0).sum())}/200 "
                  f"pool={sum(n.shape[0] for n, _ in ctl.pool)} answers={sum(a[0].shape[0] for a in ctl.ans_pool)} "
                  f"counts={ {''.join(map(str, k[::-1])): v[0].shape[0] for k, v in sorted(ctl.count_pool.items())} } "
                  f"climb_level={ctl.climb.level if ctl.climb else '-'} solved="
                  f"{sorted(ctl.climb.corpus) if ctl.climb else []} round: {self.prog_names()} fit={ctl.round_fit} "
                  f"{ctl.t_log} facts={ctl.miner.describe()}", flush=True)
        if ctl.phase == "A":
            if ctl.io >= 4 and ((len(ctl.win) == STAGE_WIN and mean >= ID_SWITCH) or ctl.in_phase >= ID_STEPS):
                ctl.history.append((ctl.step, "switch", ctl.in_phase, round(mean, 4)))
                print(f"LR switch_to_facts step={ctl.step} win={mean:.4f}", flush=True)
                ctl.phase, ctl.reset, ctl.in_phase = "B", True, 0
                ctl.win.clear()
        elif ctl.phase == "B":
            Td = self.a.argmax(-1).flatten()
            ctl.table_streak = ctl.table_streak + 1 if (table == 1.0 and bijective(Td)) else 0
            if not ctl.cells_on and ctl.table_streak >= CELL_WIN:
                ctl.cells_on = True
                ctl.history.append((ctl.step, "cells", ctl.in_phase))
                print(f"LR cells_join step={ctl.step} (the subtraction passes its checks and is a bijection)", flush=True)
            ok = ctl.cells_on and table == 1.0 and ctl.adc_ok and ctl.mul_ok and ctl.gate_ok
            ctl.cell_streak = ctl.cell_streak + 1 if ok else 0
            if ctl.cell_streak >= CELL_WIN:
                ctl.history.append((ctl.step, "climb", ctl.in_phase))
                print(f"LR climb_joins step={ctl.step} cells right (report only): {self.cells_right()}", flush=True)
                ctl.phase, ctl.in_phase = "C", 0
        elif ctl.phase == "D":
            ok = exact == 1.0 and ctl.io_exact and ctl.ans_ok == 1.0
            ctl.streak = ctl.streak + 1 if ok else 0
            if ctl.streak >= FREEZE_WIN and ctl.io == 5:
                ctl.history.append((ctl.step, "frozen", ctl.in_phase))
                print(f"LR frozen step={ctl.step} history={ctl.history} round: {self.prog_names()}", flush=True)
                ctl.phase = "frozen"

    # ------------------------------------------------------------------ reports
    def report_tables(self, e: Tensor, end: Tensor) -> None:
        ctl = self.ctl
        print(f"CELLS right (vs the truth, report only): {self.cells_right()}", flush=True)
        print(f"ROUND {self.prog_names()}", flush=True)
        if ctl.climb is not None:
            print(f"LIBRARY {[(k, show(v)) for k, v in ctl.lib.items()]}", flush=True)
            for line in ctl.climb_log:
                print(f"CLIMB {line}", flush=True)
            for r in ctl.climb.records:
                print(f"SEARCH {r}", flush=True)
        print(f"MINED {ctl.miner.describe()}", flush=True)
        print(f"SCHEDULE phase={ctl.phase} io={ctl.io} step={ctl.step} history={ctl.history}", flush=True)
        print("LAWS " + "; ".join(f"{k}: {v[0]} [{v[1]}]" for k, v in LAWS.items()), flush=True)
        with torch.no_grad():
            k = self.selected(e, end)
            Hs = HYP[int(torch.bincount(k[k >= 0], minlength=len(HYP)).argmax())] if (k >= 0).any() else None
            print(f"IO shape={SHAPES[int(self.shape.argmax())]} reading={Hs} digit_tokens="
                  f"{torch.nonzero(self.dg > 0).flatten().tolist()}", flush=True)

    def audit(self) -> None:
        ctl = self.ctl
        used = ctl.used_count > 0
        key = int(used.sum())
        if key == ctl.reported:
            return
        ctl.reported = key
        if ctl.visits is None:
            print(f"AUDIT count_loop_rows_used={key} (no training ran)", flush=True)
            return
        print(f"AUDIT count_loop_rows_used={key} used_but_never_trained={int((used & (ctl.visits == 0)).sum())}",
              flush=True)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    aux = batch.auxiliary
    with torch.autocast(batch.labels.device.type, enabled=False):
        loss = aux["loss"]
        ctl, model = aux["ctl"], aux["model"]
        if ctl.phase == "frozen":
            return loss
        labels, valid = batch.labels, batch.valid_mask
        tpos = batch.target_positions
        if tpos is None:
            cnt = valid.long().sum(1)
            kk = torch.arange(labels.shape[1], device=labels.device)[None]
            tpos = torch.where(kk < cnt[:, None], aux["end"][:, None] - cnt[:, None] + kk, -1)
        lio, yd = model.io_label_lessons(aux, labels, valid, tpos)
        loss = loss + lio
        model.mine_answers(aux, yd)
        model.keep_answers(aux, yd)
        model.keep_counts(aux, yd)
        if ctl.phase == "D":
            loss = loss + model.round_lesson(aux, yd)
        model.control(aux, float(loss.detach()))
        return loss


class Schedule:
    """Scheduler: a fresh Adam state at the switch to phase B; learning rate 0 once the model is frozen."""

    def __init__(self, optimizer: torch.optim.Optimizer, ctl: Control) -> None:
        self.optimizer, self.ctl = optimizer, ctl

    def step(self) -> None:
        if self.ctl.reset:
            self.optimizer.state.clear()
            self.ctl.reset = False
        if self.ctl.phase == "frozen":
            for group in self.optimizer.param_groups:
                group["lr"] = 0.0


def build_model(spec: ModelSpec) -> LibraryMachine:
    model = LibraryMachine(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    model.ctl.deadline = time.time() + spec.training_time_seconds          # the searches stop before the budget ends
    opt = torch.optim.Adam(model.parameters(), lr=LR, fused=spec.device_type in ("cuda", "mps"))
    return OptimizerBundle(opt, scheduler=Schedule(opt, model.ctl))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=BATCH,
    max_steps=MAX_STEPS,
    eval_batch_size=EVAL_BATCH,
)
