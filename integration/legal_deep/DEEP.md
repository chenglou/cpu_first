# The deep file (phase E): four Horner steps, rounds that keep x, H1, the family and twelve random tasks

<!-- PLAIN -->
## In plain words

**What changed** (`make_deep.py`, from the wider file's committed bytes at 80082c0):
- **The reader's fourth step.** x's polynomials up to four Horner steps (READ_POLY = 4), by Horner's steps only: ×x,
  ×c and +c, x·x first, never "+ x". a·x³ + b·x + c is read ((x·x)·a + b)·x + c.
- **The check fix.** A reading's own law, and the reader's polynomials, multiply wherever the product fits the
  places, not only where both factors fit half of them. Phase D's three2 failed on this gap, not on its budget.
- **Reduced windows.** Each window of a reading, and each single operation of it, is also posed reduced mod N.
  Horner's rule mod N stays exact, and every value keeps to the places.
- **x kept.** When the round's search by calls finds none and the law uses x again above a deeper operation, the
  search runs once more with x kept in S: the round starts `LDA.X STA.S`, and S is a routed input. The learned
  routines use X and Y as scratch, and a cubic's second call needs x.
- **A bigger budget:** CLIMB_WORK = 13e9 (was 4.3e9). The worst case of a run that learns: pre-registered at
  3,030 s, about 3,390 s with the fixed costs measured here (below).
- **The docstring's first line** names this doc and builder; seven docstrings that a design doc explains were cut to
  a pointer for the room.

The brief asked for the fourth step and a bigger budget. The rest goes further, because a budget alone brought no
depth-4 round (below). The coordinator had the evidence and this default at 19:05 PDT; with no objection by 19:25,
the file was fixed then, before either set of held-out tasks was drawn.

**Does four steps cover the space?** Yes, by the grammar and by the reader. Every a·y^k + b·y + c of the space
(k 2, 3; a 1..9; b, c 0..9) is a Horner chain of at most four steps, the last constant read as a gap. This file's
reader read all 1,800 tasks from the first 141 answer rows a full-size run has, at its first try: 200 to 50,560 laws a
try (median 6,260), 0.68–6.38 s on a B200 (median 2.36).

**Why the budget alone was not enough** (found before the design was fixed):
- With no budget at all, phase D's pair and three2 climbed to levels 9–10 (up to 9.2e9 units of work) with no round.
- three2's window 3·x·o + 2 had been learned multiplying by o's last digit: wrong on 400 of 400 full-size rows. The
  window was checked only where 3·(a·b) fit half the places, so almost only on one-digit operands.
- A cubic's round needs x again in its second call. The routines overwrite X with x² and Y with 0, and the round
  search could route inputs only from the registers the last call left.

**The bound on training time.** Pre-registered: phase D's fixed costs at their maxima (609 s), plus the whole budget
at phase D's slowest long H100 climb (0.149 s per million units) with a 25% margin: 3,030 s. Measured here: the
climbs ran no slower (0.080–0.191 s per million), but the longer rounds made the fixed costs larger (up to 967 s:
559 s from the answer program to the freeze for 8y³ + 8y + 3). With those, the bound is about 3,390 s, inside the
hour by about 210 s. And a search can pass the budget by its last group of work: 7y³ + 6y + 4's tiny core ended its
answer-program search 1.5e9 units past it. That is the cheapest work (at most 0.080 s per million on this phase's
H100s), so a pass like it adds about 120 s: about 3,510 s, inside the hour by about 90 s. The slowest learned run
trained 1,806 s.

**H1 recertifies, unchanged.**
- The official H100 run certifies: T = 64 on both ladders, every rung 768/768, test 9,999/9,999, unseen T and
  unseen moduli 10,002/10,002; training 519 s. Its climb is phase D's line for line (all 61 climb lines, seconds
  masked), its tiny core unit for unit (all 52 units, line and work).
- So do seeds 1–7 ("8 of 8 records check out"), each finding its answer program and freezing at phase D's step;
  training 502–741 s (phase D's 509–777 s).

**The tasks at full size**, one H100 each, seed 74, the file byte for byte:

| task | learned? | training | the round |
|---|---|---|---|
| plusy, y² + y | **yes** | 616 s | after level 4 |
| rho, y² + 1 | **yes** | 747 s | after level 5 |
| the cube, y³ | **yes** | 798 s | after level 5 |
| minus1, y² − 1 | **yes** | 674 s | after level 5 |
| plus3, y² + 3 | **yes** | 637 s | after level 5 |
| ten, 10y² | **yes** | 1,406 s | after level 8 |
| tenrho, 10y² + 1 | **yes** | 871 s | after level 6 |
| three2, 3y² + 2 | **yes** (phase D: no) | 772 s | after level 5 |
| 8y³ + 8y + 3 (phase D's pair) | **yes** (phase D: no) | 1,806 s | after level 7, x kept |
| 2y³ + 9y + 9 (phase D's pair) | **yes** (phase D: no) | 1,542 s | after level 7, x kept |
| 4y² + 3y + 9 (HEAD six) | **yes** | 782 s | after level 6 |
| 9y³ + 7y (HEAD six, depth 4) | **yes** | 1,575 s | after level 9, x kept |
| y² + 6 (HEAD six) | **yes** | 905 s | after level 6 |
| 5y² + 2y + 6 (HEAD six) | **yes** | 1,187 s | after level 6 |
| y³ + 5y + 1 (HEAD six) | **yes** | 1,423 s | after level 7 |
| y² + 7 (HEAD six) | **yes** | 838 s | after level 5 |
| 7y² + 9y + 5 (md5 six) | **yes** | 1,282 s | after level 7 |
| 2y³ + 6y + 8 (md5 six, depth 4) | **yes** | 1,120 s | after level 6, x kept |
| 4y³ + 4y + 5 (md5 six, depth 4) | **yes** | 1,327 s | after level 6, x kept |
| 2y² + 9y + 3 (md5 six) | **yes** | 947 s | after level 7 |
| 7y³ + 6y + 4 (md5 six, depth 4) | **yes** | 1,686 s | after level 7, x kept |
| 7y³ + y (md5 six, depth 4) | **yes** | 1,223 s | after level 6, x kept |

Each learned task certifies as H1 does: T = 64 on both ladders, every rung 768/768, test 9,999/9,999, unseen T and
unseen moduli 10,002/10,002. All eight family tasks are learned, three2 for the first time; both of phase D's pair are
learned, the cubics with x kept (known tasks now: the design was developed on them).

**The held-out tasks, two sets:**
- **The clean test: six drawn by the file's md5** (PREREG_ADDENDUM1.md), seed `int('56acca73', 16)`, from the 1,786
  tasks not used before, four of them depth 4: **6 of 6.**
- **The HEAD six** (PREREG.md), seed `int('cb623f9', 16)`, one of them depth 4: **6 of 6.** Their draw was logged at
  19:25:16, six seconds after the file was fixed at 19:25:10, but their seed was knowable from 17:48, while the design
  was being made; so they are reported apart.
- **In all: 12 of 12.**
- **Depth 4, predicted before the runs:** the addendum listed it for each set before any run of the md5 six: four of
  them, and of the HEAD six only 9y³ + 7y (which PREREG.md had named as four steps before any run). Each run's
  reading agreed: those five in four steps, the other seven in at most three.

**What it means.**
- **The clean test.** All six tasks drawn by the file's md5 are learned and certify, the four depth-4 cubics among
  them, each with x kept. Their seed did not exist until the file was fixed, so the design could not be tuned toward
  them. From 6 of 6, the share of the space this file learns is at least 61% (95%, one-sided); counting the HEAD six,
  12 of 12, at least 78%. It learns tasks drawn at random from this space: most of the space, at least.
- **What it does not show:** that every task of the space is learned. The heaviest come close to the budget:
  7y³ + 6y + 4's answer program came at 1.273e10 units of the 1.3e10, so a task needing 2% more than it would run out.
  And a task's run can fail where another succeeds: 8y³ + 8y + 3's and 2y³ + 6y + 8's tiny cores ran out of budget at
  level 9, but their full-size runs found the round after levels 7 and 6 (8y³ + 8y + 3's no-budget arm found it only
  at 2.9e10 units, its full-size run at 1.21e10).
- **What made the rounds come:** the check fix (three2, and every window with a constant times a product), the
  reduced windows and the search with x kept (the cubics and the a·x² + b·x tasks), Horner's grammar (8y³ + 8y + 3's
  form, a·x³ + b·x + c), and the budget (the learned runs used up to 1.27e10 units).
- **Inside the hour:** the learned runs trained in 502–1,806 s. The pre-registered bound, 3,030 s, took phase D's
  fixed costs; this phase's longer rounds took up to 559 s from the answer program to the freeze (283 s in phase D)
  and 258 s of frozen steps (190), and with the measured maxima the bound is about 3,390 s, inside the hour by about
  210 s; with a search's pass beyond the budget like 7y³ + 6y + 4's tiny core's, about 3,510 s, by about 90 s. That
  is thinner than the brief's "well inside". A smaller budget would not widen it for free: 7y³ + 6y + 4's run needed
  1.273e10 of its 1.3e10, 8y³ + 8y + 3's 1.21e10.
- **H1 is untouched:** its tiny core and its official climb are phase D's, unit for unit and line for line.
- **Newly designed** (§10): the Horner grammar, the product rule, the reduced windows, the search with x kept and
  its trigger, the budget.
- **Not covered:** SP_BREADTH's mont, M4, xfirst and xy, and tasks outside the space (other degrees and forms).

**Cost:** about $125 of the soft $400:
- the research before the freeze (exploration, three arms, the reader on the space), $35.49;
- the file's 17 tiny cores and its reader on the space, $31.09;
- H1's eight H100 runs, $8.83;
- the 16 task runs, $29.29;
- the md5 six: their tiny cores $11.02, their H100 runs $6.68 billed and about $2.81 for their last minutes, not yet
  in the billing report.

Every app has stopped, and none is left running in Modal's listing. The month stood at $4,262 billed afterwards
(21:30 PDT).
<!-- /PLAIN -->

## Details

### 1. Pre-registered

`PREREG.md`, frozen 2026-10-27 19:26:36 PDT (read-only, sha256
`74178d2b5485d9a8fa3b49f230a890e68ce5f11068cefb0919be8cb892b8a3e0`), before any GPU run of the file. The file was fixed
(md5 `56acca73`) at 19:25:10 PDT (`checks/logs/file_fixed.txt`), before the six tasks were drawn, and has not changed
since. Hashes of the code: `checks/logs/prereg_sha256.txt`. §8 sets each prediction against the outcome.

`PREREG_ADDENDUM1.md`: six more held-out tasks, seeded by the file's own md5, as the coordinator's queued approval
asked. It was written after every run of PREREG.md and before any run of these six, and frozen at 20:41:17 PDT. Its
written and draw times, estimated in its text, were corrected to the logs' (20:41, 20:40:02) and it was frozen again
at 20:41:30, before any run (sha256 `caf002f9…`; both hashes and the reason in `checks/logs/prereg_sha256.txt`).

**The order of events.**
- 17:48:55: HEAD at the start of phase E, cb623f9, recorded (`checks/logs/head_at_start.txt`): the draw's seed.
- 17:56–19:10: exploration and three research arms on B200 tiny cores, and the reader on the whole space; four
  messages to the coordinator with the evidence and the choices. The coordinator's one message in this time raised
  the budget to soft $400, cap $550 and asked for six fresh tasks in place of two; it did not choose among the
  designs.
- 19:05: the default announced (this design, CLIMB_WORK 12e9; raised to 13e9 before the freeze because 2y³ + 9y + 9's
  answer-program search needs it), with 19:25 for an objection. None came.
- 19:25:10 the file fixed; 19:25:16 the draw (HEAD's seed); 19:26:36 PREREG.md frozen; 19:26:52 the first run.
- After the runs: the coordinator's queued messages asked for the held-out seed from the file's md5 (HEAD's having
  been knowable while the design was made). 20:40:02 the second draw (the md5's seed); 20:41:30 the addendum frozen;
  20:41:42 its first run (the tiny cores); 20:43:10 the six H100 runs, all done by 21:25.

### 2. The changes (`make_deep.py`)

From the wider file's committed bytes at 80082c0 (md5 `59060b5c` asserted), groups of exact edits, each asserted to
match once (the room group's docstrings found by name):

| group | edits | what |
|---|---|---|
| doc | 1 | the docstring's first line names DEEP.md and make_deep.py |
| reader | 1 | READ_POLY = 4 |
| check | 5 | `LawReader(..., full)`: with full, a product is computed wherever `places(a) + places(b) <= P` (the digits it needs), not only where both factors fit half the places; `read_target` reads a reading's own law with full; `poly` sets full for x's polynomials |
| horner | 1 | `poly`'s steps: ×x, ×c (2..10) and +c (1..10); no "+ x" |
| reduced1 | 3 | `windows(t, least)`; `read_more` poses, after the windows, each window and each single operation of the reading reduced mod N (`w.mod.…`), on the window's pattern; one the same as a window is posed once |
| keepx | 4 | `xdepths`; `macro_classes(lib, sources)`; in `search_round`, when the search by calls finds none and the law read has x at more than one depth, the same search from the probes after `LDA.X STA.S`, with S among the routed sources; a round found so starts `LDA.X STA.S` |
| budget13 | 1 | CLIMB_WORK = 13,000,000,000 |
| room | 7 | the docstrings of `SelfClimb.search_outer`, `bfs_roots`, `law_target`, `answer_rows`, `SelfClimb.outer_classes`, `SteMachine` and `LibraryMachine.outer_ste` cut to their first words and the design doc's pointer |

The file: `submission_stage3_deep.py`, md5 `56acca7358c74c65c72b5f33e3b9aafa`, 261,679 bytes (465 under the limit);
the contest's source check passes.

**Choices, and why:**
- **Horner's steps only.** Horner's rule never adds x, and "+ x" let the reader find (x³ + x)·8 + 3 for 8y³ + 8y + 3
  first: a form no two windows can compute (x³ + x needs three holes).
- **x·x first, as before.** An arm that tried the constants first (`xc_*`) read ten as (10x)·x, whose window
  (10a·b) mod N its climb never learned; with x·x first, ten, tenrho and three2 read as in phase D.
- **Reduced windows of one operation too.** a·x³ + c needs (x·x) mod N as its first call; the cost is one more law,
  (a·b) mod N, posed for plusy, rho, minus1 and plus3.
- **x kept only when the plain search finds none, and only for a law that uses x again.** H1's law has x at one depth,
  so H1 never runs it: its climb is phase D's, line for line.
- **S for x.** The routines learned use X and Y as scratch and leave S alone; every machine reads a count loop's count
  once, when the loop begins (`_run_count`, `SteMachine._count`, `OuterMachine`), so a round that writes S inside
  `C.S[...]` keeps the count.
- **13e9.** 2y³ + 9y + 9's tiny core needed 1.07e10 for its round and 1.25e10 with its answer programs; 8y³ + 8y + 3's
  no-budget arm needed 2.89e10, beyond any budget the hour allows (its full-size run, later, needed 1.21e10).

### 3. Checks

| check | result |
|---|---|
| combined_sp2's local checks (`checks/logs/local_deep.log`, `checks/compare_local.py`) | the wider file's part for part, but the size, the fix part's file path, and the law book's diff region (170 lines against 165: a docstring cut); check_books part 2 the same |
| the posing (`legal_wider/checks/wider_local.py`, `checks/logs/pose.log`) | every posed law agrees with Python's integers (186 checks); H1 and the cube pose as in phase D line for line; plusy, rho, minus1 and plus3 one more law, (a·b) mod N; three2, ten and tenrho read phase D's laws on 400 of 400 rows (were 158); phase D's pair read in Horner's form |
| the reader on the space (`read_space.py`; `checks/logs/read_space.log`, `read_space_deep.log`) | the reader with only the fourth step: 1,800 of 1,800 tasks read at 141 rows, 200–60,087 laws a try (median 6,726), 0.40–5.21 s (B200), the thinnest reading on 6 rows; this file's reader: 1,800 of 1,800, 200–50,560 laws (median 6,260), 0.68–6.38 s (median 2.36), the thinnest on 51 rows |
| the check gap (`checks/accept_gap.py`) | three2's window on the Acceptor's 60 rows a size: the target said something on 38, 5, 0 and 0 rows at sizes 1–4 before the fix, 38, 16, 25 and 18 after |
| phase D's three2 routine (`checks/try_round.py`) | 3·x·o + 2 computed as 3·x·(o mod 10) + 2: right on 0 of 400 full-size rows |
| the x overwrite (`checks/try_round.py`, arm (c)'s library) | (8x² + 8) mod N right on 400 of 400 rows; after it X holds x², Y holds 0; the 2-call round right on 0 of 400 |
| the H1 runs against phase D's | the official run's 61 climb lines the same (seconds masked); the tiny core's 52 units the same (line and work) |
| the room (`checks/logs/room_ast.log`) | the file built without the room group (263,483 bytes) and with it (261,679): the ASTs the same with every docstring blanked; the file rebuilds byte for byte (`56acca73`) |

### 4. Exploration and the arms (B200 tiny cores, no climb budget; `checks/summarize_cores.py`)

| arm | what | task: the round, or not |
|---|---|---|
| `x_deep_*` | the fourth step only | phase D's pair and three2: no round to levels 9–10 (7.4e9–9.2e9); stopped |
| `xb_three2` | + the check fix | three2: after level 7 at 4.88e9; both ladders right |
| `xc_*` | + reduced windows, the constants first | H1 unit for unit; three2 after level 6 (4.90e9), tenrho after 7 (4.92e9), ten after 8 (5.67e9); the pair: no round to levels 10–11 (1.4e10–1.7e10), their windows learned and right, x overwritten |
| `xd_*` | this file's design | H1 unit for unit; plusy after level 4 (1.29e9), rho after 5 (2.85e9), three2 after 5 (3.94e9); 2y³ + 9y + 9 after level 7 with x kept (1.07e10, answer programs at 1.25e10); 8y³ + 8y + 3 after level 10 with x kept (2.89e10); every answer program right on both ladders |

### 5. The tiny cores (the file, its own budget; `records/t_deep_*`)

Every answer program found was run on both ladders, T = 1 to 64, seen and unseen moduli
(`checks/logs/ladder_t_deep_<task>.log`).

| task | the round | work: round, in all | both ladders |
|---|---|---|---|
| H1 | after level 4, phase D's 12 rounds | 1.21e9, 1.32e9 | phase D's answer programs |
| plusy | after level 4: x + x·x, then the reduction | 1.29e9, 1.40e9 | 56 rungs 768/768 |
| rho | after level 5: (x·x) mod N, then (a + 1) mod N | 2.85e9, 3.65e9 | 42 rungs 768/768 |
| the cube | after level 5: two calls of (x·o) mod N (phase D's) | 2.22e9, 3.69e9 | 14 rungs 768/768 |
| minus1 | after level 6: x moved by N first, then (x·o − 1) mod N | 3.68e9 | two right on all 28; the third wrong on 151 rows (as phase D's) |
| plus3 | after level 5: x·o + 3, then the reduction | 3.03e9, 3.83e9 | 28 rungs 768/768 |
| ten | after level 8: a call of add.mul with a digit, then (a·b) mod N | 8.53e9, 9.32e9 | the seen moduli right (21 rungs); the unseen moduli wrong (1,506 rows) |
| tenrho | none: the budget out at level 12 | 1.30e10 | – |
| three2 | after level 5: (x·x) mod N, then (3a + 2) mod N | 3.94e9, 4.87e9 | 14 rungs 768/768 |
| 8y³ + 8y + 3 | none: the budget out at level 9 | 1.30e10 | – |
| 2y³ + 9y + 9 | after level 7, x kept: (2x² + 9) mod N, then (t·x + 9) mod N | 1.07e10, 1.25e10 | 28 rungs 768/768 |
| 4y² + 3y + 9 (HEAD six) | after level 7: 4x + 3, then (t·x + 9) mod N | 7.91e9, 8.83e9 | 14 rungs 768/768 |
| 9y³ + 7y (HEAD six) | after level 7, x kept: (9x²) mod N, then ((t + 7)·x) mod N | 6.73e9 | 28 rungs 768/768 |
| y² + 6 (HEAD six) | after level 6, one call: (x·o + 6) mod N | 3.79e9, 4.68e9 | 14 rungs 768/768 |
| 5y² + 2y + 6 (HEAD six) | after level 6: 5x, then ((a + 2)·x + 6) mod N | 9.10e9 | 28 rungs 768/768 |
| y³ + 5y + 1 (HEAD six) | after level 6, x kept: (x² + 5) mod N, then (t·x + 1) mod N | 6.84e9 | 28 rungs 768/768 |
| y² + 7 (HEAD six) | after level 5: (x·x) mod N, then (a + 7) mod N | 3.04e9, 3.88e9 | 14 rungs 768/768 |
| 7y² + 9y + 5 (md5 six) | after level 7: 7x + 9, then (x·t + 5) mod N | 1.12e10, 1.22e10 | 14 rungs 768/768 |
| 2y³ + 6y + 8 (md5 six) | none: the budget out at level 9 | 1.30e10 | – |
| 4y³ + 4y + 5 (md5 six) | after level 6, x kept: (4x² + 4) mod N, then (t·x + 5) mod N | 7.65e9, 9.67e9 | 28 rungs 768/768 |
| 2y² + 9y + 3 (md5 six) | after level 7: x + x, then ((a + 9)·x + 3) mod N | 8.77e9, 9.65e9 | 42 rungs 768/768 |
| 7y³ + 6y + 4 (md5 six) | after level 7, x kept: (7x² + 6) mod N, then (t·x + 4) mod N | 1.26e10, 1.45e10 (below) | 28 rungs 768/768 |
| 7y³ + y (md5 six) | after level 7, x kept: (7x²) mod N, then ((t + 1)·x) mod N | 9.86e9, 1.14e10 | 14 rungs 768/768 |

**The md5 six's tiny cores:** five rounds, every answer program right on both ladders (0 rows wrong). 2y³ + 6y + 8's
ran out at level 9, as 8y³ + 8y + 3's did; its full-size run found the round after level 6. 7y³ + 6y + 4's found its
round at 1.26e10 and its answer programs in a search that ended at 1.45e10, 1.5e9 units past the budget, in 35 s:
`search_outer` checks the budget between groups of its work, so its last group can pass it. The climb's searches
stopped within 4e7 units of the budget (the three tiny cores that ran out: 1.302e10, 1.302e10, 1.304e10).

**Against phase D's tiny cores** (`legal_wider/checks/compare_cores.py`): H1 the same in all 52 units; the cube the
same picks at every level, library, round and answer program, its round searches at levels 3–4 longer by the search
with x kept (+6e6 units).

### 6. H1: the official run and seeds 1–7

`modal_runner_deep.py --mode official` (H1's manifest untouched, seed 74, one H100) and `--mode seeds` (seeds 1–7,
one H100 each). `tools/verify_records.py` on the eight: "8 of 8 records check out" (`checks/logs/verify_h1.log`).
Every one: T = 64 on both ladders, every rung 768/768, test 9,999/9,999, unseen T and unseen moduli 10,002/10,002;
the model state at the start 1,019,008 numbers, as phase D's.

| seed | the answer program, step (phase D) | frozen (phase D) | the round search after level 4, s (phase D) | training s (phase D) | evaluation s |
|---|---|---|---|---|---|
| 74 (official) | 632 (632) | 831 (831) | 23.8 (24.1) | 519 (522) | 271 |
| 1 | 653 (653) | 852 (852) | 7.5 (6.4) | 538 (509) | 320 |
| 2 | 649 (649) | 848 (848) | 24.7 (30.7) | 610 (777) | 290 |
| 3 | 676 (676) | 875 (875) | 19.0 (25.9) | 623 (720) | 311 |
| 4 | 716 (716) | 915 (915) | 25.3 (25.9) | 516 (581) | 295 |
| 5 | 673 (673) | 872 (872) | 53.3 (51.0) | 741 (690) | 344 |
| 6 | 687 (687) | 886 (886) | 19.2 (21.1) | 502 (581) | 309 |
| 7 | 632 (632) | 831 (831) | 19.3 (21.3) | 560 (579) | 353 |

### 7. The tasks at full size

`modal_runner_deep.py --mode plan` (`plans/runs.json`; the md5 six `plans/runs2.json`): each task on one H100, seed
74, the file byte for byte (md5 `56acca73` in every record), H1's manifest with only the name and `data.data_root`
changed (`litmus_<task>`). The data: SP_BREADTH's for its six tasks, phase C's for plusy and three2, and
`legal_wider/tasks_wider.py` in the runner's image for the fourteen poly_A_K_B_C. Summary: `checks/summarize_runs.py`
(`checks/logs/summarize_runs.log`; the md5 six's `checks/logs/summarize_runs2.log`).

| task | learned? | T, seen / unseen moduli | test | training s | the round (from the run's log) | climb work |
|---|---|---|---|---|---|---|
| H1 (official, §6) | **yes** | 64 / 64 | 9,999/9,999 | 519 | after level 4: 2 calls | 1.45e+09 |
| plusy | **yes** | 64 / 64 | 9,999/9,999 | 616 | after level 4: 2 calls | 1.42e+09 |
| rho | **yes** | 64 / 64 | 9,999/9,999 | 747 | after level 5: 2 calls | 3.66e+09 |
| the cube | **yes** | 64 / 64 | 9,999/9,999 | 798 | after level 5: 2 calls | 3.71e+09 |
| minus1 | **yes** | 64 / 64 | 9,999/9,999 | 674 | after level 5: 2 calls | 2.7e+09 |
| plus3 | **yes** | 64 / 64 | 9,999/9,999 | 637 | after level 5: 2 calls | 3.18e+09 |
| ten | **yes** | 64 / 64 | 9,999/9,999 | 1,406 | after level 8: the flat search, cost 4 | 8.95e+09 |
| tenrho | **yes** | 64 / 64 | 9,999/9,999 | 871 | after level 6: 2 calls | 4.47e+09 |
| three2 | **yes** | 64 / 64 | 9,999/9,999 | 772 | after level 5: 2 calls | 4.84e+09 |
| 8y³ + 8y + 3 (phase D's) | **yes** | 64 / 64 | 9,999/9,999 | 1,806 | after level 7: 2 calls, x kept | 1.21e+10 |
| 2y³ + 9y + 9 (phase D's) | **yes** | 64 / 64 | 9,999/9,999 | 1,542 | after level 7: 2 calls, x kept | 8.59e+09 |
| 4y² + 3y + 9 (HEAD six) | **yes** | 64 / 64 | 9,999/9,999 | 782 | after level 6: the flat search, cost 4 | 2.44e+09 |
| 9y³ + 7y (HEAD six) | **yes** | 64 / 64 | 9,999/9,999 | 1,575 | after level 9: 2 calls, x kept | 9.04e+09 |
| y² + 6 (HEAD six) | **yes** | 64 / 64 | 9,999/9,999 | 905 | after level 6: 1 call | 4.88e+09 |
| 5y² + 2y + 6 (HEAD six) | **yes** | 64 / 64 | 9,999/9,999 | 1,187 | after level 6: 2 calls | 7.69e+09 |
| y³ + 5y + 1 (HEAD six) | **yes** | 64 / 64 | 9,999/9,999 | 1,423 | after level 7: 2 calls | 1.09e+10 |
| y² + 7 (HEAD six) | **yes** | 64 / 64 | 9,999/9,999 | 838 | after level 5: 2 calls | 3.69e+09 |
| 7y² + 9y + 5 (md5 six) | **yes** | 64 / 64 | 9,999/9,999 | 1,282 | after level 7: 2 calls | 1.09e+10 |
| 2y³ + 6y + 8 (md5 six) | **yes** | 64 / 64 | 9,999/9,999 | 1,120 | after level 6: 2 calls, x kept | 1.00e+10 |
| 4y³ + 4y + 5 (md5 six) | **yes** | 64 / 64 | 9,999/9,999 | 1,327 | after level 6: 2 calls, x kept | 7.61e+09 |
| 2y² + 9y + 3 (md5 six) | **yes** | 64 / 64 | 9,999/9,999 | 947 | after level 7: 2 calls | 4.76e+09 |
| 7y³ + 6y + 4 (md5 six) | **yes** | 64 / 64 | 9,999/9,999 | 1,686 | after level 7: 2 calls, x kept | 1.27e+10 |
| 7y³ + y (md5 six) | **yes** | 64 / 64 | 9,999/9,999 | 1,223 | after level 6: 2 calls, x kept | 8.02e+09 |

Every learned task: every rung 768/768 on both ladders, unseen T and unseen moduli 10,002/10,002. Every record's
data md5 is SP_BREADTH's for its six tasks (ten `f02bdb5a`, rho `b19d39e7`, plus3 `a9d9e672`, minus1 `3089dc1e`,
tenrho `0b18439b`, the cube `75a99480`), phase C's for plusy (`feb26ee2`) and three2 (`343745f1`), phase D's for
2y³ + 9y + 9 (`96cf55ca`); the HEAD six's (`21e3fb4e`, `b4a999ac`, `4fe74d51`, `a024eaf0`, `afc21617`,
`e5f78adc`) and the md5 six's (`b148c9e9`, `79aba3cf`, `a162dba7`, `06e760fb`, `bc15d740`, `3ebac911`) were made in
the runner's image, and each run read its own polynomial from its data's answers (for 4y² + 3y + 9,
`mod(add(mul(add(mul(x,4),3),x),9),N)`; for 9y³ + 7y, `mod(mul(add(mul(mul(x,x),9),7),x),N)`; for 7y³ + 6y + 4,
`mod(add(mul(add(mul(mul(x,x),7),6),x),4),N)`).

**How the new ones were learned** (the runs' logs):
- **The cubics with x kept.** 2y³ + 9y + 9: `LDA.X STA.S`, then (2·x·x + 9) mod N, x back from S, then (t·x + 9)
  mod N. 9y³ + 7y the same way with (9·x·x) mod N and ((t + 7)·x) mod N. Their runs' rounds came after levels 7 and 9,
  at 8.6e9 and 9.0e9 units.
- **y³ + 5y + 1:** (x·x) mod N, then ((a + 5)·b + 1) mod N with b still x: its first routine left Y alone, so the
  plain search found it.
- **4y² + 3y + 9:** the flat search found (4x) mod N, then ((a + 3)·b + 9), then the reduction: three routine calls
  at cost 4. 5y² + 2y + 6: 5x, then ((a + 2)·x + 6) mod N.
- **y² + 6, y² + 7:** as plus3, their constants made from 1, 2 and 4: y² + 6 in one call, (x·o + 6) mod N.
- **three2:** (x·x) mod N, then (3a + 2) mod N; its window 3·x·o + 2 is now checked where the product fits.
- **ten and tenrho at full size,** though not in their tiny cores: ten's round after level 8 (x by a digit, shifted,
  then (x·o) mod N), right on both ladders; tenrho's after level 6.
- **8y³ + 8y + 3 at full size, though not in its tiny core:** `LDA.X STA.S`, then (8·x·x + 8) mod N, x back from S,
  then (t·x + 3) mod N, after level 7 at 1.21e10 units (its tiny core's budget ran out at level 9 with no round; the
  no-budget arm found it after level 10 at 2.89e10). Its level-7 round search took 246 s.
- **The md5 six.** The four cubics with x kept, as 2y³ + 9y + 9: (a·x·x + b) mod N, x back from S, then (t·x + c)
  mod N; 7y³ + y as 9y³ + 7y, (7·x·x) mod N, then ((t + 1)·x) mod N. 7y² + 9y + 5: 7x + 9, then (x·t + 5) mod N.
  2y² + 9y + 3: (2x) mod N, then ((a + 9)·x + 3) mod N. 7y³ + 6y + 4's round came after level 7 at 1.08e10 units
  (its level-7 round search took 235 s), its answer program at 1.273e10, 2% inside the budget.

**The hour, measured** (`checks/rates.py`, `checks/logs/rates_phaseE.log`): the climbs ran at 0.096–0.201 s per
million units (the long ones, 2.4e9 units or more, at 0.096–0.191; the bound took 0.149). The fixed costs were larger
than phase D's, because the rounds are longer: the climb started by 150 s (phase D: 136), the 199 steps from the
answer program to the freeze took up to 559 s (8y³ + 8y + 3; phase D: 283), the frozen steps up to 258 s (phase D:
190). With these maxima, the pre-registered bound (3,030 s) becomes about 3,390 s: 967 s + 13,000 × 0.186. The
slowest learned run trained 1,806 s. The md5 six (`checks/logs/rates_md5six.log`) stayed inside all of these: their
climbs at 0.080–0.130 s per million, started by 137 s, 177–509 s from the answer program to the freeze, 201–246 s of
frozen steps; the slowest trained 1,686 s (7y³ + 6y + 4). Beyond the bound: a search can pass the budget by its last
group (§5: 7y³ + 6y + 4's tiny core, 1.5e9 units in its answer-program search). At that search's slowest rate on
this phase's H100s (0.080 s per million), a pass like it adds about 120 s: about 3,510 s.

### 8. Predictions against outcomes

| what | prediction (confidence) | outcome |
|---|---|---|
| H1's tiny core | phase D's, unit for unit (high) | **right**: all 52 units, line and work |
| the cube's | phase D's picks, library, rounds and answer programs; more work (medium) | **right**: +6e6 units at levels 3–4 |
| plusy's, rho's, minus1's, plus3's | each finds its round and answer programs (medium-high) | **right**: after levels 4–6 |
| ten's, tenrho's, three2's | each finds its round and answer programs (medium) | **partly**: three2 right; ten's round, after level 8, is right on the seen moduli but wrong on 1,506 unseen-moduli rows; tenrho none, the budget out at level 12 |
| 2y³ + 9y + 9's | its round with x kept and its answer programs within the budget (medium) | **right**: after level 7, 1.25e10 in all |
| 8y³ + 8y + 3's | no round within the budget (medium) | **right**: the budget out at level 9 |
| every answer program found | 768/768 on every rung of both ladders (high) | **wrong for two**: ten's (above) and minus1's third (151 rows, as phase D's); every other right |
| H1, official run | certifies; its climb phase D's line for line; training 450–800 s (high) | **right**: 61 lines the same; 519 s |
| H1, seeds 1–7 | 7 of 7 certify (high) | **right** |
| the eight family tasks and phase D's pair at full size | as their tiny cores (medium) | **better**: all eight family tasks learned, ten and tenrho too; both of phase D's pair learned, 8y³ + 8y + 3 too (its tiny core: no round) |
| y² + 6, y² + 7 (HEAD six) | learned (medium) | **right** |
| 4y² + 3y + 9, 5y² + 2y + 6 (HEAD six) | learned (medium-low) | **right** |
| y³ + 5y + 1 (HEAD six) | learned (low) | **right** |
| 9y³ + 7y (HEAD six) | not learned within the budget (medium) | **wrong**: learned, after level 9 with x kept, 9.0e9 units |
| the six | 2 to 4 learned; not all six (medium) | **wrong**: all six |
| the reader on the space, this file | every task read at the first try's 141 rows (medium-high) | **right**: 1,800 of 1,800 |
| learned runs' training | under 2,000 s each; the bound 3,030 s (medium) | **right**: 502–1,806 s. **The bound's fixed costs were too low:** this phase's longer rounds took up to 559 s from the answer program to the freeze (phase D: 283) and 258 s of frozen steps (190); with the measured maxima the bound is about 3,390 s, inside the hour by about 210 s |
| money, phase E | $150–250 (medium) | **lower**: about $103 before the addendum, about $125 with it |

**The addendum's predictions** (`PREREG_ADDENDUM1.md`, written before any run of the md5 six):

| what | prediction (confidence) | outcome |
|---|---|---|
| depth 4 | the md5 six: 2y³ + 6y + 8, 4y³ + 4y + 5, 7y³ + 6y + 4, 7y³ + y; the HEAD six: 9y³ + 7y only | **right**: each run read those five in four steps, the other seven in at most three |
| 7y² + 9y + 5, 2y² + 9y + 3 | learned, as 4y² + 3y + 9 and 5y² + 2y + 6 were (high) | **right**: both after level 7 |
| 2y³ + 6y + 8, 7y³ + y | learned with x kept (medium-high) | **right**: both after level 6, x kept |
| 4y³ + 4y + 5 | learned, as 8y³ + 8y + 3 (medium) | **right**: after level 6, x kept, 7.61e9 units |
| 7y³ + 6y + 4 | learned (medium) | **right**, with 2% of the budget to spare: 1.273e10 of 1.3e10 |
| the six | 5 or 6 learned (medium) | **right**: 6 |
| the learned runs' training | under 2,000 s each (medium) | **right**: 947–1,686 s |
| the tiny cores | some depth-4 cores run out of budget where their H100 runs do not (medium) | **right**: 2y³ + 6y + 8's, at level 9; its H100 run's round after level 6 |

### 9. What REVIEW needs to switch to this file

- **The file to judge:** `integration/legal_deep/submission_stage3_deep.py`, md5
  `56acca7358c74c65c72b5f33e3b9aafa`, 261,679 bytes (465 under 262,144), written by `make_deep.py` from the wider
  file's committed bytes at 80082c0; the contest's source check passes.
- **Its records:** `records/legal_deep_official.json` (seed 74) and `records/legal_deep_h100_seed<1..7>.json`.
  `python3 tools/verify_records.py <each>=integration/legal_deep/submission_stage3_deep.py`: "8 of 8 records check
  out" (`checks/logs/verify_h1.log`). `tools/verify_records.py`'s own default list (its JUDGE and records) still names
  the wider file; it needs the same switch cb623f9 made.
- **Rules 4 and 5: unchanged from the wider file.** The same buffers (lines 4107–4119), the same rows kept
  (`keep_answers` line 4831, `keep_counts` line 4846, `climb_unit` line 4718 handing ROUNDT's rows to the climb at
  4733), 207,370 parameters, 1,019,008 numbers of model state at the start in every record (ten's and tenrho's
  1,018,944: their sequences are two tokens shorter). combined_sp2's buffers part is the wider file's.
- **What the file keeps outside the state dict:** the wider file's list (WIDER.md §8): the climb's working state on
  `Control` (`ctl.climb`, line 4077) and Control's schedule; module-level `READ_LAWS` (2965, now also the reduced
  windows' trees), the law book's added entries (`LAW_BOOK`, 2025), `WORK` (1424), the caches `_TRACES` (703) and
  `_OUTER` (1393); `ctl.climb.more`, views of the buffers; ROUNDT's round machines, built per check and dropped.
  New here, and transient: the search with x kept builds its probes and its vocabulary per search and drops them.
- **Rule 7, what is newly designed:** the Horner grammar of the reader's third stage (×x, ×2..10, +1..10, four steps,
  x·x first, no "+ x"); the product rule for a reading's own law (`places(a) + places(b) <= P`); the reduced windows
  (each window and each operation of a reading, mod N); the search with x kept (when the plain one finds none and the
  law has x at more than one depth; the prefix `LDA.X STA.S`; S a routed input); CLIMB_WORK = 13e9. Phase B's, C's
  and D's lists stand (NOBOOK.md §7, FAMILY_FILE.md §9, WIDER.md §9).
- **Rule 11, the hour:** the bound for a run that learns, 3,030 s pre-registered, about 3,390 s with this phase's
  measured fixed costs (§7), about 3,510 s with a search's pass beyond the budget like 7y³ + 6y + 4's tiny core's
  (§5); every run of this phase learned, in 502–1,806 s (H1's 502–741 s). A run that does not learn trains to the
  hour, as phase D's pair did.
- **Generality:** the clean test is the six drawn by the file's own md5 (`PREREG_ADDENDUM1.md`): 6 of 6 learned,
  four of them depth 4. The HEAD six, drawn six seconds after the file was fixed but from a seed knowable while it was
  designed: 6 of 6. In all, 12 of 12. The family's tasks and phase D's pair were known when the design was made (the
  arms were run on them).
- **REVIEW's line numbers:** REVIEW's §2, §5 and §6 cite 114 lines of the wider file, 172 times.
  `checks/review_lines.py` gives this file's line for each (`checks/logs/review_lines.log`). Four lines changed: the
  docstring's first line; CLIMB_WORK (79); READ_POLY (103); `windows` (2993 → 2991, its new `least`). New code REVIEW
  may cite: `LawReader`'s `full` (2791) and the product rule (`places`, 2818); `poly` (2920); `xdepths` (2968);
  `windows` (2991); `read_more`'s reduced windows (3049); the search with x kept (3538, in `search_round`, 3515);
  `macro_classes`' sources (3582).
- **The tasks' records:** `records/deep_<task>_h100.json` (22), §7.

### 10. What is still designed

Phase B's, C's and D's lists stand (NOBOOK.md §7, FAMILY_FILE.md §9, WIDER.md §9). This phase adds:
- **The Horner grammar:** x's polynomials by ×x, ×c and +c only, x·x first, up to four steps.
- **The product rule:** a reading's own law is computed wherever each product's digits fit the places.
- **The reduced windows:** each window and each single operation of a reading also posed mod N.
- **The search with x kept:** its trigger (no round by calls, and a law that uses x again), the register (S) and the
  prefix (`LDA.X STA.S`).
- **The budget:** 13e9, set from this phase's measured climbs and bounded by the hour.

### 11. Where it still touches Python

As the wider file (WIDER.md §10), plus:
- **The product rule** counts each factor's places (the index of its highest nonzero digit), as `fits` reads the top
  places; the product itself is the learner's addition, doubled.
- **`xdepths`** reads the depths of x in the law's tree, as `windows` reads the tree.
- **The search with x kept** starts from the probes after two fixed instructions; every round it finds is the
  learner's routines and moves, checked on the answer rows and by ROUNDT as any other.

### 12. Cost

`cost.py` (`checks/logs/cost.log`; the billing report's complete hours, read at 21:29 PDT):

| app | what | billed |
|---|---|---|
| ap-NsrWyFLvQ7xZKxpvqvLlb4 | exploration: 3 tiny cores with no work budget (B200s) | $5.30 |
| ap-yXWtXxssZi7kRCRl9LjIb4 | the reader on the space, the candidate (6 B200s) | $7.42 |
| ap-oppsXTCVBm9uBGIYSoifGz | arms (b) and (c): 7 tiny cores (B200s) | $11.98 |
| ap-9uTw2tfmYYgAM0LQ4CPWkc | arm (d): 6 tiny cores (B200s) | $10.79 |
| ap-Jmn8ASRP54qjCmfIkqM5sk | the file's 17 tiny cores and the reader on the space (B200s) | $31.09 |
| ap-a3xoljJyvm3pI1HI7XxOEf | H1's official run (H100) | $1.02 |
| ap-SyOPls8n3mUMV6hfYyXfIq | H1's seeds 1–7 (H100s) | $7.81 |
| ap-ogpgOFTsCiEosqULmkZptW | the 16 task runs (H100s) | $29.29 |
| ap-rf8Js4Far5isrIxwzaz41e | the md5 six's tiny cores (B200s) | $11.02 |
| ap-AT8QWRDJMozSqUdxt79Q8x | the md5 six's runs (H100s) | $6.68 |
| all | | **$122.39**, with about $2.81 more for the md5 six's runs in the 04:00 UTC hour (the watcher's estimate) |

**Limits.** Soft $400, cap $550 (raised from $100 and $150 by the user's update during the phase). The month stood at
$4,140 when phase E began and at $4,262 billed afterwards (21:30 PDT). `spend.py 4750` passed before every launch.

**The spend watchers** (`logs/watch_x*.log`) watched every app of the phase, restarted as apps were added: they stop
everything once the estimate plus one more hour of the containers still running ($6.25 a container-hour) reaches the
cap ($150, then $550; the last, for the md5 six's two apps, $447: the cap less what the phase had already spent),
and any app at half the cap. The estimate never came near either. Every app has stopped, and so
has the last watcher.

### 13. Files and how to rerun

Everything is in `cpu_first/integration/legal_deep/`; nothing outside it was written.

| file | what |
|---|---|
| `submission_stage3_deep.py` | the file, md5 `56acca73` |
| `make_deep.py` | the builder (§2); `--groups=G,...` and `NAME=VALUE` for research builds (`research/`) |
| `research/submission_stage3_deep_*.py` | the arms, each built with the default groups of its time (the tiny cores set CLIMB_WORK = 0): `_check` (b, md5 `264cf6e3`), `_check_reduced_order` (c, `4b322206`), `_check_reduced1_horner` (c′, `e971c931`), `_reduced1_horner_keepx` (d, the file's design with CLIMB_WORK 8e9 and the docstrings whole, `2e971ce6`) |
| `PREREG.md` | frozen; `checks/logs/prereg_sha256.txt` holds its hash and the code's |
| `heldout.json`, `checks/draw_heldout.py` | the HEAD six and how they were drawn; `checks/logs/head_at_start.txt` the seed's commit |
| `PREREG_ADDENDUM1.md`, `heldout2.json`, `checks/draw_heldout2.py` | frozen; the md5 six and how they were drawn (the seed from the file's md5, asserted) |
| `core_f.py`, `modal_deep.py` | the tiny core and its launcher (app and Volume `legal-deep`; `SCRIPT=read_space.py` runs the reader) |
| `read_space.py` | the reader on every task of the space |
| `make_runner.py`, `modal_runner_deep.py`, `make_plan.py`, `plans/runs.json`, `plans/runs2.json` | the shared runner with the poly tasks, the 16 runs and the md5 six's |
| `watch_spend.py` | the spend watcher |
| `cost.py` | §12 |
| `checks/rates.py` | the climbs' rates from the H100 records (`checks/logs/rates_phaseD.log`, `rates_phaseE.log`, `rates_md5six.log`) |
| `checks/summarize_space.py`, `checks/summarize_cores.py`, `checks/summarize_runs.py` | the sweeps, the tiny cores, the H100 runs |
| `checks/try_round.py`, `checks/accept_gap.py` | a round or routine at full size; a window's acceptance |
| `checks/compare_local.py`, `checks/review_lines.py` | combined_sp2's checks against the wider file's; REVIEW's line numbers |
| `checks/logs/`, `logs/`, `records/` | the logs, the spend watchers, every record |

From `cpu_first/`, with `PY=~/github/sotaku/venv/bin/python` and `PYTHONDONTWRITEBYTECODE=1` exported:

```bash
$PY integration/legal_deep/make_deep.py
$PY integration/legal_deep/make_deep.py --groups=check,reduced,order
$PY integration/legal_deep/checks/draw_heldout.py
$PY integration/legal_deep/checks/draw_heldout2.py
$PY integration/legal_deep/make_runner.py
python3 integration/legal_deep/make_plan.py
python3 integration/legal_deep/make_plan.py --md5six
COMBINED_FILE=integration/legal_deep/submission_stage3_deep.py PYTHONPATH=. $PY integration/combined_sp2/checks/local_combined.py policy grad machines isa designed buffers replay climb climb_tiny fix book
python3 integration/legal_deep/checks/compare_local.py
FILE=integration/legal_deep/submission_stage3_deep.py TASKS=h1,plusy,rho,cube,minus1,plus3,three2,ten,tenrho,poly_8_3_8_3,poly_2_3_9_9 $PY integration/legal_wider/checks/wider_local.py
FILE=integration/legal_deep/submission_stage3_deep.py PYTHONPATH=. $PY integration/legal_deep/checks/accept_gap.py
$PY integration/ablation_s3/spend.py 4750
$PY -m modal run --detach integration/legal_deep/modal_deep.py --envs '[{"TAG": "t_deep_h1", "TASK": "h1"}, ...]'
$PY -m modal run --detach integration/legal_deep/modal_deep.py --envs '[{"TAG": "read_space_deep_0", "SCRIPT": "read_space.py", "PART": "0/6", "ROWS": "141", "RETRY": "400"}, ...]'
TASK=poly_2_3_9_9 $PY integration/legal_wider/checks/ladder_wider.py integration/legal_deep/records/t_deep_poly_2_3_9_9.json integration/legal_deep/submission_stage3_deep.py
python3 integration/legal_wider/checks/compare_cores.py integration/legal_wider/records/t_wider_h1.json integration/legal_deep/records/t_deep_h1.json
$PY -m modal run --detach integration/legal_deep/modal_runner_deep.py --mode official --file integration/legal_deep/submission_stage3_deep.py --name legal_deep_official --out integration/legal_deep/records
$PY -m modal run --detach integration/legal_deep/modal_runner_deep.py --mode seeds --file integration/legal_deep/submission_stage3_deep.py --seeds 1,2,3,4,5,6,7 --hardware h100 --name legal_deep --out integration/legal_deep/records
$PY -m modal run --detach integration/legal_deep/modal_runner_deep.py --mode plan --plan integration/legal_deep/plans/runs.json --out integration/legal_deep/records
$PY -m modal run --detach integration/legal_deep/modal_runner_deep.py --mode plan --plan integration/legal_deep/plans/runs2.json --out integration/legal_deep/records
python3 tools/verify_records.py integration/legal_deep/records/legal_deep_official.json=integration/legal_deep/submission_stage3_deep.py integration/legal_deep/records/legal_deep_h100_seed{1,2,3,4,5,6,7}.json=integration/legal_deep/submission_stage3_deep.py
python3 integration/legal_deep/checks/summarize_runs.py
python3 integration/legal_deep/checks/summarize_runs.py integration/legal_deep/records/deep_poly_{7_2_9_5,2_3_6_8,4_3_4_5,2_2_9_3,7_3_6_4,7_3_1_0}_h100.json
python3 integration/legal_deep/checks/rates.py integration/legal_deep/records/deep_poly_{7_2_9_5,2_3_6_8,4_3_4_5,2_2_9_3,7_3_6_4,7_3_1_0}_h100.json
python3 integration/legal_deep/checks/review_lines.py
$PY integration/legal_deep/cost.py
```
