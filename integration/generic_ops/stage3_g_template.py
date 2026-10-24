"""Stage 3 with self-posed practice on a generic instruction set (cpu_first/integration/generic_ops/GENERIC_OPS.md):
self_practice/submission_stage3_sp2.py with the digit machine's division-shaped instructions replaced by generic ones as
the constants COND, SHIFT and BRING below say: SEL.S (keep or discard on the borrow) by a generic conditional, and,
unless BRING, BRG.X (bring X's next digit in) by shifts that pass the digit they move through a carry. The SP_* search
settings say where they differ from sp2's. Everything else is sp2's.

sp2's own summary: stage 3 with self-posed practice (cpu_first/integration/SELF_PRACTICE.md): integration stage 3 (s3i,
INTEGRATION.md §3) with its designed curriculum taken out. The climb's practice problems are posed and chosen by the
learner itself during the run (self_practice/spcore.py: a law book applied to the rows' own numbers in every binding
of the operands to the registers, targets from facts and the learner's own routines, bar 3's choosing rules); no task
list, helper order, register variant, mac form or answer constant of T = 1 is in the file. Everything else is s3i's.

What the model holds:
  * a reader and a writer (stage 1's, line for line: IO.md, WIRING.md stage 3): which tokens are digits and which
    digit each one is, where each field starts and which way it runs;
  * three digit cells, each a table of logits learned from random: the subtraction (a, b, borrow) -> (digit, borrow),
    the addition (a, b, carry) -> (digit, carry), the multiply-accumulate (a, x, h, k) -> low and high digit of
    a + x h + k; and the countdown's 2 x 2 gate;
  * a digit machine (libcore_g: library/dsl.py's 5 registers, a shared pointer, a borrow/carry, a multiplier digit
    and a multiply carry; primitive instructions that each read the cells at the pointer or move whole registers, with
    no select and no bring-in; loops that run their body P, B or W times, and with COND = "if" a block C[...] that
    keeps its effects only where the borrow was set) and a library of programs that grows during training;
  * one logit per round program the training found (PROG_SLOTS), a tournament the answers decide.

How it learns, in one training run (one evaluator step = one forward, one loss, one backward):
  A  the reader learns to read (IO.md) and the subtraction table its borrows and digits from the mined order facts and
     the laws (general_lessons' table lessons, on the registers x and other rows' x; there is no square yet);
  B  once the subtraction table passes its own checks: the addition table by sub.inverse ((a + b + c) - b - c = a, on
     every entry, with the snapped subtraction), the multiply-accumulate table by mul.zero and mul.ladder (on every
     entry, with the snapped addition), and the gate by the loop law (a count of at least 1 runs its round, 0 stops);
  C  once every cell passes its own checks, the self-practising climb (spcore.SelfClimb), one unit per step: each
     level it poses every law of its book on the read rows' own digits in every binding (a law whose construction
     needs a routine it has not practised waits for it), probes them all (one search per pattern and binding), keeps
     the trivial and the unsolved aside, picks K learnable ones by its rule (SP_RULE: hardest first, or the cheaper
     pairs signal), practises them (BPE, each pick's routine without side effects on its inputs), then searches the
     round by composition at full size with the answers as its goal (the rows of the smallest T the batches showed);
  D  the round programs that reproduce the answers become the tournament's contestants; the answers keep scoring them;
     T's direction by speculation (IO.md); the freeze once everything has been exact FREEZE_WIN steps.
The prediction (rule 8): the reader, then the countdown (the subtraction table and the gate) running the selected
round program on one-hot digits with straight-through tables (libcore.SteMachine), then the writer.

What is designed is listed in SELF_PRACTICE.md (the instruction set, the sizes, the law book and its patterns, the
choosing rule, the search's budgets, BPE and the routine rule, the round's search space).
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
GRAD_DAMP = 0.1        # the prediction path's gradient is multiplied by this after every cell and gate
W = 8                  # digits of the widest modulus (a 24-bit N is below 10^8)
P = W + 1              # the table lessons' registers (stage 1's)
PR = 2 * W             # the round machine's register places (a product of two W-digit numbers)
K = 2                  # the countdown's places: T's field
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
SP_VERIFY = ((10, 1), (10, 2), (10, 3))  # acceptance: the practice problem's law at these sizes (base, digits)
SP_PROBE = 1_000_000   # one probe search's cap: states stored per level
SP_FALLBACK = 1_000_000        # a root's first probe over the primitives alone (cost-5 programs of primitives)
SP_DEEP = 5_800_000    # the deep probes' cap (INTEGRATION.md §2.1's measured 10-minute search on the H100) ...
SP_DEEP_K = 16         # ... for this many roots a level: those whose closest open law is closest (the best fit;
                       # with SP_DEEP_BY "bindings", whole concepts until this many roots: two concepts of 12
                       # bindings. sp2: 8, one concept)
SP_DEEP_ROTATE = "gain"        # a concept that had a deep probe goes after the others unless its closeness grew
                               # since (False, sp2: by closeness alone)
SP_DEEP_BY = "bindings"        # ... taken concept by concept, every binding where it is still open ("root": the
                               # closest roots alone, as submission_stage3_sp.py, md5 f0ef5b06)
SP_DEEP_FOCUS = "closure"      # their vocabulary: the primitives, the routines the concept's closest attempts used, and
                               # the routines those call (True: that root's attempt's routines alone, as f0ef5b06)
SP_DEEP_UNION = False          # ... the closest attempts at the root itself (True, sp2: at every binding of its concept)
SP_LOOK = 300_000      # a lookahead's cap (ct2a2)
SP_PURE_FIRST = False  # a law whose programs all write a register besides A is searched one cost deeper for one
                       # that does not (False, sp2: the first cost's programs)
SP_ALIAS_FIT = True    # a law whose probe found only aliases (programs acceptance rejects) keeps its closeness, from
                       # its states that are not goal states (False, sp2: it counts as reached, closeness 0)
ROUND_STATES = 2_000_000       # the round's search: states stored per level (sp2: 4M; 2M stops the flat search at cost
                               # 4, the search by calls having come first) ...
ROUND_COST = 5         # ... and its programs' cost at most
SP_MACRO_AFTER = 12    # past this many routines, when that search finds nothing, the round is searched by calls: up
SP_MACRO_COST = 2      # to this many routine calls, each with its inputs routed from A, X, Y, N or zero
SP_CALLS_FIRST = True  # ... and that search by calls runs first, the flat one only if it finds nothing (False, sp2:
                       # flat first)
SP_LEVELS = 16         # the climb's levels at most
SEARCH_S = 900         # one search's time limit in seconds (0: none; the training's deadline always holds)
CLIMB_MARGIN = 120.0   # the climb stops this many seconds before the training budget ends
ROUND_RETRY = 10       # once the climb is done with no round, the round is searched again every this many steps
FUSE = 4               # a search runs up to this many chunks of its work in one batch
SP = dict(rule=SP_RULE, K=SP_K, NP=SP_NP, probe_budget=SP_PROBE, fallback_budget=SP_FALLBACK, deep_budget=SP_DEEP,
          deep_k=SP_DEEP_K, deep_by=SP_DEEP_BY, deep_focus=SP_DEEP_FOCUS, deep_rotate=SP_DEEP_ROTATE, deep_union=SP_DEEP_UNION,
          alias_fit=SP_ALIAS_FIT, verify=SP_VERIFY, pure_first=SP_PURE_FIRST, look_budget=SP_LOOK,
          round_budget=ROUND_STATES, macro_after=SP_MACRO_AFTER, macro_cost=SP_MACRO_COST, calls_first=SP_CALLS_FIRST,
          round_cost=ROUND_COST, levels=SP_LEVELS, search_s=SEARCH_S or None, fuse=FUSE, round_retry=ROUND_RETRY)
POOL = 4096            # read rows kept for the practice problems
ANS_POOL = 512         # answer rows kept for the round's search (rows of the smallest T the batches showed)
ANS_MIN = 64           # the round is searched once this many answer rows are kept
PROG_SLOTS = 4         # the round programs the tournament holds: the simplest of those the answers accept (Occam,
                       # by expansion into primitives); the round's search returns every minimal program, often dozens
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
    "loop.count": ("a count of at least 1 runs its round; a count of 0 stops the loop", "the countdown's gate"),
}
ORDER_TERMS = ("N", "x", "T")

#@LIBCORE@
#@SPCORE@

#@STAGE1_TOP: compare relations Miner answer_t answer_rows answer_cap tally ste sub_hard add_learned onehot0 zpad carry_on link_soft sub_soft mixture_nll tourney_nll order_pairs pick pickl shift Config@


def damp(x: Tensor) -> Tensor:
    """The prediction path's value, its gradient times GRAD_DAMP (the certified files' damp; here after every cell
    and gate of the round machine). Without autograd, nothing to scale."""
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
        self.cells_on = False                 # the addition, multiply and gate lessons have joined
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
        self.climb = None                     # spcore.SelfClimb, from phase C on
        self.cells = None                     # the snapped cells the climb runs with (libcore.Cells)
        self.lib: dict = {}                   # the library the round programs use
        self.progs: list = []                 # the tournament's round programs (items)
        self.climb_log: list = []
        self.deadline: float | None = None    # the training budget's end (build_optimizer), for the searches
        self.round_fit: list = []             # each program's share of this step's answer rows


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
        self.g = nn.Parameter(INIT * torch.randn(2, 2))                    # the countdown's gate: borrow -> hold, commit
        self.prog = nn.Parameter(INIT * torch.randn(PROG_SLOTS))           # the round programs' tournament
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
        """The plumbing check only: the reader (stage 1's designed reading), every cell and the gate at the truth; the
        round programs and their library are given by the check (ctl.progs, ctl.lib)."""
        hot = lambda t, n: 10.0 * F.one_hot(t, n).float() - 5.0
        c = exact_cells("cpu")
        self.a.copy_(hot(c.sbb_d, 10))
        self.b.copy_(hot(c.sbb_c, 2))
        self.aa.copy_(hot(c.adc_d, 10))
        self.ab.copy_(hot(c.adc_c, 2))
        self.ml.copy_(hot(c.mul_lo, 10))
        self.mh.copy_(hot(c.mul_hi, 10))
        self.g.copy_(hot(torch.tensor([1, 0]), 2))
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
        """The snapped subtraction (digit, borrow [200]) and the gate [2] (1 = commit)."""
        return self.a.argmax(-1).flatten(), self.b.argmax(-1).flatten(), self.g.argmax(-1)

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
        """The prediction path's straight-through tables: the round machine's SBB, ADC [200, 12] and MUL [10^4, 20]
        (rows flat over the inputs), the countdown's subtraction in stage 1's layout [20, 120], the gate [2, 2]."""
        hot = lambda p: ste(F.one_hot(p.argmax(-1), p.shape[-1]).float(), p.float().softmax(-1))
        SBB = torch.cat([hot(self.a), hot(self.b)], -1).reshape(200, 12)
        ADC = torch.cat([hot(self.aa), hot(self.ab)], -1).reshape(200, 12)
        MUL = torch.cat([hot(self.ml), hot(self.mh)], -1).reshape(10 ** 4, 20)
        TDC = torch.cat([hot(self.a), hot(self.b)], -1).permute(0, 2, 1, 3).reshape(20, 120)
        return SBB, ADC, MUL, TDC, hot(self.g)

    def program(self, i: int | None = None):
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

#@STAGE1_METHODS: ext isdig ranks wsm valid_anchor anchor_logits anchor shape_logp hard_pos shape_probs shape_rho runs which_run selected read label_seq read_answer reading_weights codes_ste io_lessons mine mine_answers@

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

    def round_ste(self, s: Tensor, Nr: Tensor, SBB: Tensor, ADC: Tensor, MUL: Tensor) -> Tensor:
        """One round on one-hot digits: the selected program from x = s in X and N in N (a clean start) on the round
        machine; the new state is A's low W places [X, W, 10]. Straight-through over the tournament's programs: with
        autograd every program runs (the gradient reaches the tournament's logits); without, the argmax program only."""
        ctl = self.ctl
        X = s.shape[0]
        zero = torch.zeros(X, PR, 10, device=s.device)
        zero[..., 0] = 1
        regs = [zero, zpad(s, 0, PR - W), zero, zpad(Nr, 0, PR - W), zero]
        z2 = torch.zeros(X, 2, device=s.device)
        z2[:, 0] = 1
        z10 = torch.zeros(X, 10, device=s.device)
        z10[:, 0] = 1
        m = SteMachine(SBB, ADC, MUL, PR, damp)
        n = len(ctl.progs)
        sel = ste(F.one_hot(self.prog[:n].argmax(), n).float(), self.prog[:n].float().softmax(-1))
        run = lambda i: m.run(regs, z2, z10, z10, ctl.progs[i], ctl.lib)[0][0][:, :W]
        if not torch.is_grad_enabled():
            return run(int(sel.argmax()))
        return sum(sel[i] * run(i) for i in range(n))

    def predict_ste(self, e: Tensor, end: Tensor, k: Tensor, L: int) -> Tensor:
        """The prediction path inside autograd: the reader, the countdown with the selected round program, and the
        writer, on one-hot digits with straight-through tables (stage 1's reader and writer, line for line)."""
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

        SBB, ADC, MUL, TDC, G = self.ste_cells()
        Sx, Cnt, Sn = field_(1), field_(2), field_(0)
        one = onehot0(X, K, e.device)
        one[:, 0] = F.one_hot(torch.tensor(1, device=e.device), 10).float()
        s = Sx
        if ctl.progs:
            for _ in range(10 ** K):
                Cn, bor = sub_soft(Cnt, one, TDC, ctl.used_count)
                kk = (bor[:, -1] * G[:, 1]).sum(-1)
                if not bool((kk > 0.5).any()):
                    break
                R = self.round_ste(s, Sn, SBB, ADC, MUL)
                s = damp(kk[:, None, None] * R + (1 - kk)[:, None, None] * s)
                Cnt = damp(kk[:, None, None] * Cn + (1 - kk)[:, None, None] * Cnt)
        wcode = hot(self.wc.float() + F.logsigmoid(self.dg.float())[None])
        out = torch.zeros(X, Lp, V, device=e.device)
        for j in range(W):
            out = out + slot(3, j)[:, :, None] * (s[:, j] @ wcode)[:, None, :]
        return 20.0 * out[:, W + 1:W + 1 + L] - 20.0

    # ------------------------------------------------------------------ the hard machine (training's checks)
    def run_rounds(self, Sd: Tensor, C: Tensor, Nd: Tensor, prog, cap: int | None = None) -> Tensor:
        """The countdown and the round program on digits: rounds until no row commits (the snapped subtraction for
        C - 1, the gate on its borrow), each committing row running the program (libcore's machine at PR places, the
        snapped cells, a fresh machine as run_prog makes it); at most `cap` rounds (default 10^K). Every row runs every
        round, with fixed shapes (a row that does not commit keeps its state), on the program's round machine (on a GPU
        one CUDA graph, captured once per program and row count); its dirty starts are run_prog's own draws for the
        committing rows. Returns the state [X, W]."""
        Td, Tb, Gc = self.tables()
        cells = self.ctl.cells or self.snapped_cells()
        st = round_settings(cells, PR)
        gen = gen_on(Sd.device, 0)
        m = round_machine(st, prog, self.ctl.lib, Sd.shape[0], Sd.device)
        s = Sd.clone()
        one = F.one_hot(torch.zeros_like(C[:, 0]), C.shape[1])
        for _ in range(10 ** C.shape[1] if cap is None else cap):
            Cn, b, _ = sub_hard(C, one, Td, Tb)
            go = Gc[b] == 1
            n = int(go.sum())
            if n == 0:
                break
            s = torch.where(go[:, None], m.run(s, Nd, go, n, gen)[:, :W].to(s.dtype), s)
            C = torch.where(go[:, None], Cn, C)
        return s

    # ------------------------------------------------------------------ the lessons
    def table_lessons(self, Nd, Xd, Cd, terms, orders, nonzero) -> tuple[Tensor, float]:
        """The subtraction table's lessons (general_lessons' kind 1 and 2, as in stage 1, on registers made of the rows'
        x and other rows' x: there is no machine square here): sub.order on every mined order fact, then the digit
        laws (sub.order on the entries the batch's subtractions read, sub.bijective), then the pieces' laws
        (sub.inverse with the learned adder a - (0 - b), sub.identity, add.commute, the facts' borrows, sub.wrap,
        sub.zero_minus). Returns the loss and the hard checks' share passing."""
        ctl = self.ctl
        dev = Xd.device
        Td, Tb, _ = self.tables()
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
        the snapped tables before them), the gate by the loop law (the batch's own counts: a count of at least 1 runs
        its round, the gate committing on the borrow of C - 1; a count of 0 stops it, the gate holding on 0 - 1).
        Each cell's own check is that its snapped table equals its law's targets."""
        ctl = self.ctl
        Td, Tb, Gc = self.tables()
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
        line = ctl.climb.step((pool[0], pool[1]), answers=ans, deadline=dl, t=ctl.ans_t or 1)
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
        """The round programs' tournament, by the answers: every program runs the countdown on the batch's answer rows
        (the hard machine); a program passes when it reproduces every one of them; the loss is -log of the passing
        programs' total weight (tourney_nll; nothing when none passes)."""
        ctl = self.ctl
        sel = torch.nonzero(aux["small"]).flatten()
        n = len(ctl.progs)
        if sel.numel() == 0 or n == 0:
            return self.prog.sum() * 0.0
        with torch.no_grad():
            cap = answer_cap(aux["Cd"][sel])
            ok = torch.stack([(self.run_rounds(aux["Xd"][sel], aux["Cd"][sel], aux["Nd"][sel], p, cap) == yd[sel])
                              .all(1) for p in ctl.progs])                   # [programs, rows]
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

#@STAGE1_METHODS: io_label_lessons@

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

    def t_lesson(self, aux: dict, yd: Tensor) -> Tensor:
        """T's direction by speculation (IO.md; stage 1's lesson with the round program's machine): on rows whose two
        readings of T differ, the countdown runs from both, and the answers keep the reading that reproduces them."""
        ctl = self.ctl
        e, end, code, Xd, Nd = aux["e"], aux["end"], aux["code"], aux["Xd"], aux["Nd"]
        X, Lp = e.shape
        dev = e.device
        ok, si, ti, length = self.runs(e, end)
        rn, _, _ = self.which_run(0, e, end, si, ti, ok, length)
        rx, _, _ = self.which_run(1, e, end, si, ti, ok, length)
        rt = (3 - rn - rx).clamp(0, 2)
        a0, b0 = si.gather(1, rt[:, None])[:, 0], ti.gather(1, rt[:, None])[:, 0]
        j = torch.arange(K, device=dev)
        tp = e.gather(1, (a0[:, None] + j).clamp(max=Lp - 1))
        tm = e.gather(1, (b0[:, None] - j).clamp(min=0))
        lp = j[None] < (b0 - a0 + 1)[:, None]
        Cp = torch.where(lp, code.gather(1, tp.clamp(max=self.V - 1)), 0)
        Cm = torch.where(lp, code.gather(1, tm.clamp(max=self.V - 1)), 0)
        rows = torch.nonzero(ok & (Cp != Cm).any(1)).flatten()[:T_ROWS]
        if rows.numel() == 0 or not ctl.progs:
            return self.strd.sum() * 0.0
        with torch.no_grad():
            n = rows.numel()
            s2 = self.run_rounds(Xd[rows].repeat(2, 1), torch.cat([Cp[rows], Cm[rows]]), Nd[rows].repeat(2, 1),
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
                  f"mul={int(ctl.mul_ok)} gate={int(ctl.gate_ok)} io_exact={ctl.io_exact} chosen={ctl.chosen:.4f} "
                  f"ans_ok={ctl.ans_ok:.4f} streak={ctl.streak} facts_streak={ctl.facts_streak} "
                  f"order_digits={ctl.gated_on} rows_visited={int((ctl.visits > 0).sum())}/200 "
                  f"pool={sum(n.shape[0] for n, _ in ctl.pool)} answers={sum(a[0].shape[0] for a in ctl.ans_pool)} "
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
        print(f"GATE {self.g.argmax(-1).tolist()} p_commit={self.g.float().softmax(-1)[:, 1].tolist()}", flush=True)
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
            print(f"AUDIT countdown_rows_used={key} (no training ran)", flush=True)
            return
        print(f"AUDIT countdown_rows_used={key} used_but_never_trained={int((used & (ctl.visits == 0)).sum())}",
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
