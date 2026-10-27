# For a reviewer: what is claimed, the file to judge, and what to rule on

*Written 2026-10-23 for an outside judge of One Layer Deeper H1 (learn x^(2^T) mod N from prompt/answer pairs),
rewritten 2026-10-25/26 for fast2, the five-piece stage-3 file made fast, and on 2026-10-27 for the legal file, fast2
with its generated law book replaced by three laws and bar 3's legal reader, and then for the family file, the legal
file with no law in its book and round 7's general rules. It points into [OVERVIEW.md](OVERVIEW.md),
[FAMILY_FILE.md](integration/legal_family/FAMILY_FILE.md), [NOBOOK.md](integration/legal_nobook/NOBOOK.md),
[WIDE.md](integration/legal_wide/WIDE.md), [LEGAL_FILE.md](integration/combined_legal/LEGAL_FILE.md),
[FAST.md](integration/combined_fast/FAST.md) and [COMBINED.md](integration/combined_sp2/COMBINED.md) instead of
repeating their tables. The record checks of §3 and the builders and source checks of §4.3 were rerun on 2026-10-27.
The hosted evaluator has scored nothing here (§1).*

<!-- To swap the file to judge: its path is in "In plain words", §2, §3, §4.1-§4.3, §6 and §7; its md5 in §2; its bytes
and lines in §2 and §5 (rule 1); its records in §3, §4.1 and §4.2; its seconds, memory and seeds in "In plain words",
§2, §3, §4.1, §4.2, §5 (rules 10, 11, 13, 14) and §7; its work budgets in §1, §2 and §5 (rules 4, 11); its line
numbers in §2, §4.1, §5 and §6; the README's top paragraph; and tools/verify_records.py (JUDGE, JUDGE_RECORDS; the
earlier files' records follow them). The file it replaces moves to §2's "earlier file", §3's and §4's second places,
§5's last paragraphs (with the commit whose REVIEW.md takes its rules) and §7. -->


The rules are the competition's own: [one-layer-deeper/README.md §Rules](../one-layer-deeper/README.md#rules), at
commit `87e5c26` ("Releasing H1"). Everything here that reads the competition's files expects that clone at
`../one-layer-deeper`, checked out at `87e5c26` (§4.1 has the commands): the links into it, the record checker (§3),
the builders and the source check (§4.3), and our runner (§4.2).

## In plain words

**The task.** H1 asks for x^(2^T) mod N, that is x squared T times with the remainder by N kept each time, learned from
prompt and answer pairs in one hour of training on one H100. A run certifies when every prompt is exactly right up to
T = 64, on moduli seen in training and on unseen ones.

**The entry.** One file, [`submission_stage3_family.py`](integration/legal_family/submission_stage3_family.py) (§2). Its
model is a small digit machine whose digit tables (subtract, add, multiply-accumulate) start random and are learned from
general laws. Inside the training run it poses itself practice problems, searches for short programs of the machine's
instructions that solve them, and keeps the solved ones as routines. Its law book is empty. Before any routine exists,
one designed rule, the residue rule, reads the answer rows it keeps: every answer is below N, so the law ends in a
reduction by N, and the reduction's step is posed cut to its parts, the comparison, the link (subtract N unless that
borrows) and the addition. Once those are routines, it reads the target's law from the same rows, computing every
candidate with those routines, and finds mod(mul(x, x), N). That law poses the rest: the multiply and the reduction, one
digit at a time and then looped over the digits, and the law's window (x·o) mod N as a law of its own. From its routines
it composes a round (copy x, multiply, reduce by N), kept only if it also gives the kept answers at T = 2, 4 and 8, and
an answer program that runs the round T times. It answers by running that program on its learned tables.

**The evidence.** On the contest's unmodified evaluator and manifest, on one H100, it certifies on the manifest's seed
and, with only the seed changed, on seven more: 8 runs of 8, T = 64 on both ladders, every test split exact, trained
in 562–986 s of the 3,600 (§3). The same file, byte for byte, one H100 each, learns 4 of 8 other tasks of H1's family
at full size: y² + y, y² + 1, the cube and y² − 1. It does not learn y² + 3, 10y², 10y² + 1 or 3y² + 2. Only y² + y and
3y² + 2 were held out, chosen before any run; round 7's rules, which the file uses, were found on the other six (§2).
The file it was built from, the legal file, certified 8 of 8 the same way and stays as the earlier file to judge, with
fast2 before it (§2).

**What to rule on.** Whether a training run that searches for its own program is within the contest's rules, above all
rules 4, 7, 8 and 14 (§5). And, for the project's own bar, whether what people still chose (the instruction set, the
residue rule, the reader's grammar, the rules for windows, constants and rounds, the search's rules, the prompt's
reader and the schedule) amounts to writing the algorithm by hand.

## 1. What is claimed

The project measures itself against three bars
([OVERVIEW §4](OVERVIEW.md#4-where-it-stands-three-bars-and-the-models-at-the-frontier)):

1. **Certify H1 with learned parts: done.** Every step of the removal ladder certifies T = 64 on both ladders of the
   untouched official manifest on an H100. From step 11 on, no hand-written arithmetic runs in training or in the
   answer: every table that computes is learned from a random start, though the datapath around those tables is
   still written by hand before step 17, and in step 18, which branches from step 15 (OVERVIEW §5).
2. **Nothing a strict judge would call the hand-written algorithm: the entry is the family file, and the call is the
   judge's.** In the family file the program the forward pass runs ("copy x, multiply, reduce", T times) is found by
   search inside the training run and kept in the model's buffers; no datapath and no menu of wirings is in the file.
   Against s3i, the first file to judge, four more kinds of hand-made pieces are gone. Each was first removed in a
   study of its own, whose file certified alone on the official manifest (OVERVIEW
   [§4.1](OVERVIEW.md#41-the-models-at-the-frontier) and [§4.8](OVERVIEW.md#48-variants-of-stage-3-that-go-further)),
   and fast2 put them in one file:
   - the hand-picked practice problems and their order: the self-practising file, sp2, poses its own from a book of
     laws ([SELF_PRACTICE.md](integration/SELF_PRACTICE.md)), and fast2 generates that book from a grammar
     ([SP_ABLATION.md](integration/sp_ablation/SP_ABLATION.md) §5.2);
   - the countdown over T: the run now finds it, as a use of a generic count loop
     ([LEARNED_LOOP.md](integration/learned_loop/LEARNED_LOOP.md));
   - SEL.S and BRG.X, long division's step written as two instructions: an if block on the borrow and shifts through
     the multiply's carry digit take their place, and the run builds the step from them
     ([GENERIC_OPS.md](integration/generic_ops/GENERIC_OPS.md));
   - the clock, and the rows and programs held in Python: work budgets and buffers take their place
     ([STRICT.md](integration/strict/STRICT.md)). The family file keeps one set of rows in Python again, ROUNDT's
     (§5, rule 4).

   The legal file went one step further on the first of these. Its book was three laws on the pair, and the rest of what
   it practised was posed by the law the run read from its own answer rows, with its own routines: bar 3's legal reader
   ([LEGALLOOPS.md](bar3/scale/legalloops/LEGALLOOPS.md)), which the legal file was the first to certify
   ([LEGAL_FILE.md](integration/combined_legal/LEGAL_FILE.md)). The family file takes the book to nothing, in three
   phases, each certified 8 of 8: a wider grammar for the reader ([WIDE.md](integration/legal_wide/WIDE.md)); the three
   laws replaced by the residue rule, which reads from the answers that the law ends in a reduction by N and poses the
   reduction's step cut to its parts ([NOBOOK.md](integration/legal_nobook/NOBOOK.md)); and round 7's general rules,
   ported legally: the reading's windows and constants posed as laws of their own, and a round kept only if it also
   gives the kept answers at T = 2, 4 and 8 ([FAMILY_FILE.md](integration/legal_family/FAMILY_FILE.md)). Every practice
   answer comes from a fact, an identity or the learner's own routines; every value the law reader compares comes from
   the learner's routines, and the residue rule compares the answers' own digits with N's and x's. Its two weak practice
   checks are fixed too ([ABLATION_S3.md §8](integration/ABLATION_S3.md#8-fixing-the-two-weak-checks)). Still designed,
   knowing the task family: the instruction set, the residue rule and the step's cut table, the reader's grammar (x and
   N as inputs, constants only as gaps; add, sub, mul, pow and mod; depth 3), the window and constants rules, ROUNDT,
   the cut and loop rule and the catalogue of law forms that answer what is posed, the choosing rule, the search's sizes
   and rules (several found by watching whether the round came), the count loop and T's 2-digit field, the prompt's
   reader, the cells' shapes and the schedule (§5, rule 7). An ablation on the official manifest shows what s3i's result
   rests on (§7); none of fast2, the legal file and the family file has an ablation of its own.
3. **A general algorithm learner: results at a tiny size, and first steps at full size.** A learner that picks its own
   practice problems invents x² mod N for x < N ≤ 7 (and ≤ 26) on a bare counting machine; unchanged, it invents 4 of 9
   other formulas, and 2 more once its solved formulas carry over ([PRACTICE.md](bar3/PRACTICE.md),
   [BREADTH.md](bar3/BREADTH.md)). Aiming one deeper search at the piece the target factors through, found from the rows
   alone, brings x³ and 10·x² from scratch ([SUBGOAL.md](bar3/scale/subgoal/SUBGOAL.md)). Frozen with that rule, it
   invents 5 of 10 formulas that none of its rules were written for, with its predictions written down before the runs
   ([HELDOUT.md](bar3/scale/heldout/HELDOUT.md)). Given a loop over x's digits and a source that poses one more digit's
   step, it invents x mod N in base 2 by long division's step, right up to nine digits
   ([DIGITS.md](bar3/scale/digits/DIGITS.md)). With its duplicate check made on each problem's own rows, its ties broken
   by the subgoal score and the two-source study's offset posing the piece, it also invents 10·x² + 1 from scratch
   ([PICKS.md](bar3/scale/picks/PICKS.md)). Run on one digit place, it invents all four of stage 3's digit cells, each
   under at least one of its choosing rules; multiply-accumulate needs two new ways of posing its piece and a deeper
   search aimed at it the same way ([BRIDGE.md](bar3/bridge/BRIDGE.md), [CARRY.md](bar3/bridge/CARRY.md)). At H1's
   size, the self-practising stage 3 learns 8 of 11 tasks of the family after three general changes to its round's
   search ([SP_BREADTH.md](integration/sp_breadth/SP_BREADTH.md)). Bar 3's legal reader, which reads a target's law
   from its answer rows with the learner's own routines, now runs inside the file to judge (2. above), and that file,
   byte for byte, learns 4 of 8 family tasks at full size: y² + y (held out), y² + 1, the cube and y² − 1. It does not
   learn y² + 3, 10y², 10y² + 1 or 3y² + 2 (held out). Only the two held-out tasks, chosen before any run, are a fair
   test of its generality, since round 7's rules were found on the other six; one of the two is learned
   ([FAMILY_FILE.md](integration/legal_family/FAMILY_FILE.md) §7).

Four rules are **gray** for the family file, as they were for the legal file, fast2 and s3i: 4, 7, 8 and 14 (§5). The
write-ups flag them; they do not argue them away. Rules 2 and 11 also need a word for its Triton kernel, and rules 5
and 9 for what stays in Python.

Not claimed:
- a result from the hosted evaluator: the project's one attempt to submit, s3i after the deadline on Oct 24, was
  refused ("Submissions are now closed"; OVERVIEW §6, §7);
- the leaderboard count in [REPORT.md](REPORT.md) §0 (3 of 146 entries certified) comes from the public leaderboard at
  the time and has no record behind it;
- runs on B200s, CPUs or relaxed budgets are research; only official-manifest H100 runs count as certification here.
  The family runs use H1's manifest on their own data, so they are research too: they show what the file learns (§3);
- that the family result shows generality beyond its two held-out tasks: round 7's rules were found on the other six
  (§2);
- the family file's work budgets are fast2's, set from fast2's eight climbs; its own eight climbs use at most 39% of
  `CLIMB_WORK`, and that the budgets hold on every other seed is not shown (§2);
- that the family file's climb repeats line for line in a second run of the same seed: fast2's did, in four runs
  (§2), but the family file has one run a seed.

## 2. The file to judge, and the others

**[`integration/legal_family/submission_stage3_family.py`](integration/legal_family/submission_stage3_family.py)**,
md5 `457216d746d7a21e854a2d2d3673c9d6` ("the family file"). 5,113 lines, 261,651 bytes (493 under the 256 KiB limit),
207,370 learned numbers in 15 parameter tensors, Adam. Its docstring (lines 1–9) says what it is and where it is
written up. Its first line is still the legal file's: it names LEGAL_FILE.md and says that `make_legal.py` writes the
file, where `make_family.py` does (below).

**What it is.** The legal file (below) taken to an empty law book in three phases, each a file of its own, built by
asserted edits of the one before and certified 8 of 8 on H100s (§7):
- **phase A, a wider reader** ([WIDE.md](integration/legal_wide/WIDE.md) §2): the law reader (`LawReader`, line 2795;
  `READ_OPS`, line 2792) tries chains of add, sub, mul, pow and mod over x and N, depth 1 to 3, and, when none fits,
  with the data's constants among the leaves, read as gaps: a number by which a reduction mod N misses every answer.
  Every value is still the learner's routines': its addition; its comparison's difference, kept where its addition
  gives the minuend back; repeated addition by doubling; and its link by doubling. On H1's rows it weighs 200
  candidates where the legal file's reader weighed 35;
- **phase B, no law in the book** (`BOOK`, line 94; [NOBOOK.md](integration/legal_nobook/NOBOOK.md) §2): before any
  routine exists, the residue rule (`read_fact`, line 3141) compares each kept answer with N and with x, digit by digit
  from the top place, and makes no number. If every answer is below N, the law ends in a reduction by N, so the
  reduction's step x + d·N is posed cut to its parts on the pair: the comparison, the link and the addition
  (`STEP_PARTS`, line 95). The wide reading follows once those three are routines (`read_law` and `pose`, lines 3129
  and 3151);
- **phase C, round 7's pieces** ([FAMILY_FILE.md](integration/legal_family/FAMILY_FILE.md) §2): every window of the
  reading, a connected run of 2 or 3 of its operations, becomes a law of its own, its holes (at most two) its operands
  and its modulus N (`windows` and `read_more`, lines 2957 and 3005; `READ_WINDOW`, line 97). Each constant of the
  reading is made from N, and x + c, x − c and x·c are posed (`CONST_OPS`, line 98); that also gives round 7's 1 for
  x − 1, which the legal reader writes as a leaf. And a round is kept only if it also gives the newest 4,096 kept
  answers of every count T > 1, run T times; on H1's data those are T = 2, 4 and 8 (ROUNDT: `round_t`, line 3503;
  `ROUNDT`, line 99). The windows' and the constants' answers are the reader's own arithmetic, the learner's routines
  (`read_target`, line 3029).

Everything else is the legal file's, and through it fast2's: the cut and loop rule (`read_pairs`, line 2929), the
grammar's target rules, the weak-check fix, the strict build, the learned loop over T, the generic instructions, the
Triton kernel and the work budgets. Against the no-book file, an AST comparison with the docstrings left out finds 13
functions changed and 12 new, all the port's (FAMILY_FILE.md §2).

**What its runs show** (FAMILY_FILE.md §4–§7).
- **H1, 8 of 8.** Every run reads every kept answer below N at level 0 (64–69 rows) and practises the comparison, the
  link and the addition at levels 0–2. At level 3 it reads mod(mul(x, x), N) on every answer row kept by then (140–168
  rows, 200 laws tried in under a second) and poses its cuts and loops and its window (x·o) mod N. It finds its round at
  level 4 by two routine calls, and its answer program, `C.S[T.round<i> STA.X]`, the next step, within 3 steps of the
  no-book file's on every seed: by about 322–639 s of training, against the no-book file's 256–344 s and the legal
  file's 272–349 s. Training ends at the step cap in 562–986 s (the official run: 783.3 s, then 367.6 s of evaluation),
  against the no-book file's 472–634 s. Almost all the extra time is ROUNDT's: the round search after level 4 takes
  187 s on the official run and 344 s on seed 5, against the no-book file's 14 s and 21 s, since it accepts 12 rounds at
  once and runs each 14 times at full size on up to 4,096 rows of each count (FAMILY_FILE.md §6). In the tiny core,
  which runs the climb alone on fixed rows on a B200, H1's climb is the no-book file's unit for unit for its first 17
  units, and every level's picks, the library, the 12 rounds and the answer programs are the same; the window's probes
  add 8% work (FAMILY_FILE.md §4).
- **The family at full size, 4 of 8** (FAMILY_FILE.md §7). The same file, byte for byte, one H100 each, seed 74, on H1's
  manifest with only its name and data root changed; each task's data is SP_BREADTH's, and the two held-out tasks' were
  made by `tasks_family.py` with H1's generator. Learned, each passing H1's checks on its own data (T = 64 on both
  ladders, every rung 768/768, every split exact), in 631–1,534 s of training: y² + y (held out; H1's round with x in
  place of 0), y² + 1 (its window x·o + 1, with a 1 it made from N), the cube (two calls of its window (x·o) mod N) and
  y² − 1 (ROUNDT threw out the rounds that are wrong at 0, and the round kept adds N to x first). Not learned: y² + 3
  (it never makes the constant 3, and its probes spend the climb's work budget), and 10y², 10y² + 1 and 3y² + 2 (held
  out), where the reader reads nothing: 10 is a factor, not a gap, and the other two need depth 4.
- **What that shows.** Only the two held-out tasks, y² + y and 3y² + 2, chosen before any run, are a fair test of
  generality, and the file learns one of them. Round 7's rules were found on the other six, offline
  ([FAMILY.md](bar3/scale/family/FAMILY.md)), so the three of those six that are learned show that the port works, not
  that the rules generalize. The four failures are this file's own, not round 7's: three at the reader's grammar
  (round 7 read 10y² and 10y² + 1 offline with a numpy reading) and y² + 3 at the work budget (round 7 made its 3
  offline; FAMILY_FILE.md, "In plain words").

**How it is built.** It is a generated file, built in six steps; each asserts its input's md5 and rebuilds its file
byte for byte (§4.3):
- [`make_family.py`](integration/legal_family/make_family.py) reads the no-book file from git (commit `f7dc5f8`, md5
  `b42ec304` asserted) and applies 53 exact edits, each asserted to match once: 1 for the docstring, 15 for the port
  and 37 for room (docstrings and one comment that a design doc explains, the rules between a class's methods, and the
  no-book file's STEP_ADD switch, always on), none of the room edits changing behaviour. It then checks that the file
  parses, compiles, reads no undefined global and passes the contest's source policy;
- [`make_nobook.py`](integration/legal_nobook/make_nobook.py) builds the no-book file from the wide file (md5
  `3bb5ceeb` asserted) by 12 edits, and [`make_wide.py`](integration/legal_wide/make_wide.py) the wide file from the
  legal file (from git at `8e3c7fe`, md5 `e2fe7a68` asserted) by 16;
- `make_legal.py`, `make_fast.py --fast2` and `make_combined.py --round 2` build the legal file, fast2 and round 2's
  file (below).

**How it got here.** The legal file (below) showed that bar 3's legal reader could carry a contest file with a book of
three laws. Phase A widened the reader's grammar, and phase B emptied the book (WIDE.md, NOBOOK.md). Round 7, offline,
then found four general pieces that bring SP_BREADTH's family on stage 3's tiny core, with practice answers from
Python (FAMILY.md), and phase C ported them legally. Phase C's predictions were written down before any of its GPU
runs ([PREREG.md](integration/legal_family/PREREG.md), read-only, its hash recorded). Every one about H1 held; of the
family's, y² + 3 (predicted learned, at low confidence) and y² − 1 (predicted not learned, at medium) came out the
other way (FAMILY_FILE.md §8). H1's tiny core found the round, and H1's official run and seeds 1–7, each run once,
all certified (FAMILY_FILE.md §4–§6).

**The work budgets.** The file never reads the clock. Each search, and the climb as a whole, stops after a budget of
machine work, counted by the search's own cost model (STRICT.md §2). The readings' routine calls, the answers of the
laws they pose and ROUNDT's runs of the round are counted the same way, inside the climb's budget: level 3's pose,
the reading included, did 4.9e5 units on the official run. The budgets are fast2's, unchanged: `CLIMB_WORK` = 4.3e9
and `SEARCH_WORK` = 8e8 (lines 77–78), set from fast2's eight climbs (below).
- The family file's eight H1 climbs spend 1.37e9–1.68e9 units, at most 39% of `CLIMB_WORK` (seed 1), and their
  largest search 2.78e8, 35% of `SEARCH_WORK` (seed 4), measured as fast2's were
  (`integration/combined_fast/checks/measure_work.py` on the eight records). No run stops on a work budget.
- A climb that spent the whole budget would still end inside the hour. The eight climbs ran at 0.16–0.40 s per
  million units, ROUNDT's checks included (from the climb's start to the answer program, by the times
  `integration/combined_fast/checks/summarize_seeds.py` reads off the eight records), so 4.3e9 units take at most
  about 1,720 s; with the latest start (about 64 s) and the longest wait from the answer program to the freeze (about
  130 s), such a run would freeze by about 1,900 s. On the family, y² + 3's climb did spend its whole budget, and that
  run went on, never frozen, to its 20,000 steps in 3,183 s (FAMILY_FILE.md §7).

**The earlier file to judge, the legal file:**
**[`integration/combined_legal/submission_stage3_legal.py`](integration/combined_legal/submission_stage3_legal.py)**,
md5 `e2fe7a681bc981ce844fc8c826132d47`, the file to judge on 2026-10-27 until the family file. 5,000 lines, 261,216
bytes. It is fast2 (below) with its generated law book replaced, by 18 asserted edits of fast2's committed bytes
(`make_legal.py`, from commit `f237a61`; [LEGAL_FILE.md](integration/combined_legal/LEGAL_FILE.md) §2):
- **the book** (its `BOOK`, line 101): three laws on the pair, posed from the start: the comparison, the link and the
  addition;
- **the law reader** (its `LawReader`, line 2818): once the three laws are routines, the climb reads the target's law
  from the kept answer rows of the smallest T, at most 3 times. Its candidates are chains of add, mul and mod over x
  and N, depth 1, then 2, every value computed by the learner's own routines: its addition, repeated addition by
  doubling, and its link repeated until its comparison says "below", at most 1,000 times;
- **the cut and loop rule** (its `read_pairs`, line 2882), which the family file keeps: each operation of the law read,
  f and f added into a sum, on the grammar's patterns whose count is a digit (the operation cut to one place) or
  another row's number (its loop over the digits). For mod(mul(x, x), N): the multiply-add row o + x·d, x·d, their
  loops, Horner's step red(x + d·N) and the reduction red(x + o·N).

Every run read mod(mul(x, x), N) at level 3, on 11–23 of the 69–95 answer rows kept by then (35 laws tried), and found
its round at level 4 by two routine calls and its answer program by about 272–349 s of training. It certifies 8 of 8
on H100s, training in 465–721 s (§3), and its eight climbs spend 0.99e9–1.96e9 units, at most 45% of `CLIMB_WORK`. Its
docstring and one comment (its line 102) still call the law reader `Reader`; the class is `LawReader` (LEGAL_FILE.md
§2). It came from bar 3's audit of its practice sources against rule 14 ([LEGAL_SOURCES.md](bar3/LEGAL_SOURCES.md)),
which found that most read a target table that a training run does not have, and from three studies on stage 3's
machine that followed: a law test at full size on H1's own rows ([LEGAL_TEST.md](bar3/scale/legal/LEGAL_TEST.md)), a
fully legal climb ([LEGALCLIMB.md](bar3/scale/legalclimb/LEGALCLIMB.md)), and the legal reader, with which three laws
were enough ([LEGALLOOPS.md](bar3/scale/legalloops/LEGALLOOPS.md)).

**The file to judge before it, fast2:**
**[`integration/combined_fast/submission_stage3_fast2.py`](integration/combined_fast/submission_stage3_fast2.py)**,
md5 `41b9a013ae9436d7b7a53e9e594a18db`, the file to judge from 2026-10-26 until the legal file. 4,940 lines, 258,349
bytes. It is sp2, the self-practising stage-3 file, with the five variants of it that each certified alone: the
weak-check fix, the strict build, the learned loop over T, the generated law book and the generic instructions
([COMBINED.md](integration/combined_sp2/COMBINED.md) §1–§2 say where they meet). Then engineering that changes nothing
it learns ([FAST.md](integration/combined_fast/FAST.md)), all of which the legal file and the family file keep:
- a Triton kernel runs each row's whole instruction sequence of the search's machine in one GPU launch, step for step
  as the old machine did, on integers;
- the short sequences the search ran one instruction at a time go through the same kernel, with their machine work
  counted exactly as before;
- one walk on the GPU lists every goal state's program at once;
- room for this code comes from code the file can never run with its settings, and from the CUDA-graph path the kernel
  replaces.

Its law book is the grammar posed whole: every law of the grammar on every pattern whose construction worked, from the
start (fast2's lines 2026–2043, `Draw` at 2154, `law_target` at 2259). It certifies 8 of 8 on H100s too, training in
759–883 s (§3). In the tiny core the fast file's climb reaches the round 7.7 times sooner than round 2's file did
(416 s against 3,206 s on a B200), with every search, pick and unit of machine work the same (FAST.md §4), and fast2's
tiny core is the same climb again (FAST.md §9). On the evaluator its official climb is the same 72 lines, seconds
masked, in four runs on the manifest's seed: fast2's, the fast file's, the fast file with no budgets, and round 2's
file given two hours (FAST.md §5, §9; rerun on 2026-10-26 with `integration/combined_fast/checks/compare_official.py`).
Its docstring still says that `make_combined.py --round 2` writes the file; `make_fast.py --fast2` does.

Its work budgets come from STRICT.md §2's rule, applied to the eight official climbs measured with no budget: research
runs of the fast file's budget-free build, all eight certified too
(`integration/combined_fast/records/research/fast_budget0_h100_seed<N>.json`; FAST.md §9). `CLIMB_WORK` = 4.3e9 is
1.22 times the largest climb, 3.492e9 on seed 6, rounded up from 4.26e9; `SEARCH_WORK` = 8e8 is twice the largest
search, 3.913e8 on seed 3, rounded up from 7.83e8 (`integration/combined_fast/checks/logs/measure_work.log`). fast2's
largest climb uses 81% of `CLIMB_WORK`, and its eight climbs equal those budget-free runs line for line, with no
work-budget stop anywhere (`integration/combined_fast/checks/logs/compare_fast2.log`). The fast file,
[`submission_stage3_fast.py`](integration/combined_fast/submission_stage3_fast.py) (md5
`91bb30b9d131791d165025d682b3f458`), the first certified version, differs from fast2 only in round 2's budgets,
3.25e9 and 7e8, and a docstring line (fast2's lines 7, 84 and 85); it certifies 8 of 8 on H100s too. On four of its
seeds the climb passed that `CLIMB_WORK` in the last block of the answer program's last search level. The budget is
read only between a level's blocks, so it cut nothing: every certified climb of the fast file equals its budget-free run
line for line (FAST.md §5, §8; `integration/combined_fast/checks/logs/compare_budget0.log`).

**The first file to judge: [`integration/submission_stage3_s3i.py`](integration/submission_stage3_s3i.py)**, md5
`0e8131440e1842546c67e7378caf472d` ("s3i", step 17 of the removal ladder). 3,437 lines, 173,817 bytes, 207,374 learned
numbers in 16 parameter tensors, Adam. It is a generated file: [`make_stage3.py`](integration/make_stage3.py) builds it
from [`stage3_template.py`](integration/stage3_template.py) (1,008 lines),
[`lib/libcore.py`](integration/lib/libcore.py) (1,877 lines) and stage 1's reader, and rebuilding it from those sources
gives the same bytes (§4.3). Its docstring (lines 1–35) is the best one-page summary of stage 3.
- It still has what fast2 removed, and the legal and family files with it: hand-picked practice problems, the
  countdown, SEL.S and BRG.X, the clock, and rows and programs held in Python (the end of §5 lists where it differs).
- It stays here because it has the most evidence about what the line rests on: the official run, seven more H100
  seeds, and an ablation on the official manifest that removes each hand-made part in turn (§7).
- Two of its practice checks are weak: on about 3% of runs, none of its 8 certified seeds among them, the reduction's
  accepts a shortcut that fails at full size, and the conditional subtraction's never tests x between N and 2N. A fixed
  copy, [`fix/submission_stage3_s3i_fix.py`](integration/ablation_s3/fix/submission_stage3_s3i_fix.py) (md5
  `09a8263cf09321fef9ba3d4ffbc2aea6`), only adds the missing probes and replays all 8 certified runs field for field
  (§7). fast2, the legal file and the family file have the fix.

**For context** (OVERVIEW [§4.1](OVERVIEW.md#41-the-models-at-the-frontier) and §4.8 compare them):
- [`integration/self_practice/submission_stage3_sp2.py`](integration/self_practice/submission_stage3_sp2.py), md5
  `afc70fe91069a85c416eba18b48ee30e` ("sp2"): s3i with its curriculum replaced by self-posed practice from a book of 11
  laws ([SELF_PRACTICE.md](integration/SELF_PRACTICE.md)). 250,643 bytes. It certifies on the official manifest and on
  all 8 B200 seeds. Its parent, [`submission_stage3_sp.py`](integration/self_practice/submission_stage3_sp.py) (md5
  `f0ef5b069ef2e1bd9dc337798d246ba4`), certifies on the official manifest and on 7 of 8 seeds. fast2 is sp2 with
  five variants, the legal file is fast2 with three laws and the law reader, and the family file is the legal file
  with a wider reader, no law in its book and round 7's rules.
- The family file's two steps from the legal file, each certified 8 of 8 on H100s (§7): the wide file,
  [`integration/legal_wide/submission_stage3_wide.py`](integration/legal_wide/submission_stage3_wide.py) (md5
  `3bb5ceebf1de115501efc3398af8a457`), and the no-book file,
  [`integration/legal_nobook/submission_stage3_nobook.py`](integration/legal_nobook/submission_stage3_nobook.py) (md5
  `b42ec30422c182c0972aebca5be0c470`).
- The five variants, each certified alone on the official manifest on an H100, with its md5 and record in OVERVIEW
  §4.8: the fixed checks (`a92d6281`), the strict build (`1a1f15fb`), the learned loop (`419c15e7`), the generated book
  (`8bbac4a1`) and the generic instructions (`e6bd2a1f`). Round 3's two four-piece files certify too (COMBINED.md §9).
- [`integration/submission_stage1.py`](integration/submission_stage1.py), md5 `cfd7b61d7d3d4b76c606606ba8b8d9db`:
  the tournament and the general lessons in one file. Its forward pass is still the hand-written H1 datapath, with
  menus that hold the right wirings (rule 7 is gray as for its parents).
- [`general_lessons/submission_general_v3.py`](general_lessons/submission_general_v3.py), md5
  `0805b7d38f9fe4fa15b182c1b481886e`: general lessons v3, where the answers choose whole digit codes. Its 8 H1 seeds
  ran an earlier build, `v3f` (md5 `32a1d154`), on B200s; the final file's own H1 run is the official one.

Every earlier certified file is a step of the removal ladder ([OVERVIEW §5](OVERVIEW.md#5-the-removal-ladder)), each
with its md5 and record.

## 3. The official records, and how to check them

The official records, one JSON per evaluator run on the untouched official manifest (`h100_hard_h1.json`), and the
seed runs on that manifest with only `runtime.seeds` changed:
- **The family file:** [`legal_family_official.json`](integration/legal_family/records/legal_family_official.json) and
  `legal_family_h100_seed1.json` … `seed7.json`, all in `integration/legal_family/records/`, all on H100s.
- **The legal file:** [`combined_legal_official.json`](integration/combined_legal/records/combined_legal_official.json)
  and `combined_legal_h100_seed1.json` … `seed7.json`, all in `integration/combined_legal/records/`, all on H100s.
- **fast2:** [`combined_fast2_official.json`](integration/combined_fast/records/combined_fast2_official.json) and
  `combined_fast2_h100_seed1.json` … `seed7.json`, all in `integration/combined_fast/records/`, all on H100s. The
  fast file's are beside them, named `combined_fast_…`.
- **s3i:** [`s3i_official.json`](integration/records/s3i_official.json) and `s3i_h100_seed1.json` … `seed7.json`, in
  `integration/records/`, all on H100s.
- **Stage 1:** [`s1_official.json`](integration/records/s1_official.json); its other seeds ran on CPUs
  (`integration/records/s1_cpu_seed*.json`).
- **General v3:** [`v3k_h1_official.json`](general_lessons/fixed/records/v3k_h1_official.json); its other H1 seeds ran
  the `v3f` build on B200s ([FIXED_MODULUS.md](general_lessons/fixed/FIXED_MODULUS.md)).

Careful with one name: `runs/learned_remainder/s1_official.json` is a different, failed run (the removal ladder's
first recipe for step 1), not stage 1's.

**What a record holds.** One JSON per evaluator run: `submission` (the exact source sent, as `submission.py`),
`manifest` (as passed; the shared runner's records also name `manifest_file`), the evaluator's full `stdout`, whose
last `RESULT_JSON=` line is the evaluator's own result, `result` (that line, parsed), `gpu`, `returncode` and
`wall_seconds`, and for records written by [`tools/modal_runner.py`](tools/modal_runner.py) or its copy (below) also
`submission_md5` (checked inside the container before the run), `runner.data_md5`, an md5 over the generated data
folder the run read, and `usage`, whose `peak_session_rss_mb` is the peak host memory quoted in §4 and §5.

**The runner's copy.** Every family-file record, H1's included, ran through a copy of the shared runner,
[`integration/legal_family/modal_runner_family.py`](integration/legal_family/modal_runner_family.py) (md5
`36a999e066e2cb369fba7342e85e98fc`, each record's `runner.tool_md5`). `make_runner.py` writes it from
`tools/modal_runner.py` (md5 asserted) by five asserted edits: the two held-out tasks built into its image and its task
list, the tool's name in the record, its paths and its docstring. H1's eight runs went through it, the official one with
the official manifest untouched, and read the same H1 folder as every shared-runner record (data md5 `a6414ec9…`). One
fix was made after the pre-registration froze, outside the submission: the copy's path line, which failed at import in
the container on its first launch, before any job ran. The official run then went out with `--force`, since that launch
had registered the record's name (FAMILY_FILE.md §7).

**One command** (standard library only, from this folder, with the competition clone next to it):

```bash
python3 tools/verify_records.py
```

For each of the 34 records above (the family file's eight, the legal file's eight, fast2's eight, s3i's eight, stage
1's and general v3's) it prints: the md5 of the embedded source against the file's md5 now; the manifest against the
competition's `h100_hard_h1.json` (identical, or differing only in `runtime.seeds` for the seed runs); whether the
`RESULT_JSON=` line in `stdout` equals `result`; T on both ladders; every rung's count; the three splits; the times. On
2026-10-27 all 34 checked out ("34 of 34 records check out"): md5s equal, T = 64 on both ladders, every rung 768/768,
test 9,999/9,999, ood_t and ood_n_t 10,002/10,002. Any other record: `python3 tools/verify_records.py REC.json=FILE.py`.
For example, the wide and no-book files' eight each, the fast file's eight, and the self-practice files' official
records (all check out the same way):

```bash
W=integration/legal_wide; R=$W/records/legal_wide
python3 tools/verify_records.py ${R}_{official,h100_seed{1,2,3,4,5,6,7}}.json=$W/submission_stage3_wide.py
B=integration/legal_nobook; R=$B/records/legal_nobook
python3 tools/verify_records.py ${R}_{official,h100_seed{1,2,3,4,5,6,7}}.json=$B/submission_stage3_nobook.py
F=integration/combined_fast; R=$F/records/combined_fast
python3 tools/verify_records.py ${R}_{official,h100_seed{1,2,3,4,5,6,7}}.json=$F/submission_stage3_fast.py
S=integration/self_practice
python3 tools/verify_records.py $S/records/runner/submission_stage3_sp2_official.json=$S/submission_stage3_sp2.py
python3 tools/verify_records.py $S/records/runner/submission_stage3_sp_official.json=$S/submission_stage3_sp.py
```

Or by hand, for one record:

```bash
python3 - integration/legal_family/records/legal_family_official.json <<'EOF'
import hashlib, json, sys
print(hashlib.md5(json.load(open(sys.argv[1]))["submission"].encode()).hexdigest())
EOF
md5 integration/legal_family/submission_stage3_family.py      # Linux: md5sum
```

**The family's eight records** (`family_<task>_h100.json`, in `integration/legal_family/records/`: plusy, rho, cube,
minus1, plus3, ten, tenrho and three2) are not in the default. Each ran on its task's own data, so the checker flags
its manifest, whose `data.data_root` and `name` differ from H1's (`litmus_<task>`, `litmus-<task>`), and prints BAD.
Check them with the same command and read the lines it prints (with no `=FILE.py`, the checker takes the file each
record names, here the family file):

```bash
R=integration/legal_family/records/family
python3 tools/verify_records.py ${R}_{plusy,rho,cube,minus1,plus3,ten,tenrho,three2}_h100.json
```

In every record the embedded source's md5 is the family file's and the `RESULT_JSON=` line equals `result`. The four
learned tasks (plusy, rho, cube, minus1) show T = 64 on both ladders, every rung 768/768 and every split exact; the
others show no certified T and at most 12 of 9,999 test rows. The data md5 on each record's last line is its task's
folder's: for the six tasks that are not held out it equals their SP_BREADTH records'
(`integration/sp_breadth/records/g3_<task>_h100.json`), and plusy's and three2's folders were made by
`tasks_family.py` in the runner copy's image (FAMILY_FILE.md §7).

**What to look for in the family file's log** (`stdout` of `legal_family_official`):
- `LR climb_starts step=563`: the climb starts, with every cell right;
- `LR climb step=576 L0 pose: 22 candidates on 12 roots; fact: answers below N on 65 rows (N:y …)`: the residue rule,
  once 64 answer rows are kept, with every row it read; it poses the comparison, the link and the addition;
- `LR climb step=593 L3 pose: 102 candidates on 102 roots; read mod(mul(x,x),N) on 141 of 141 answer rows, 200 laws
  tried, 0.6s, poses ['w.mod.mul/w2']`: the law reader, at the start of level 3, the window it poses, and its cuts and
  loops, in every binding;
- `LR climb step=631 L4 round: cost 2 calls`: the round, found at level 4 as two routine calls; most of its 186.8 s
  are ROUNDT's checks;
- `LR outer step=632 answer program: cost 3`: the answer program's search, the programs it found
  (`C.S[T.round<i> STA.X]`), and the climb's work so far (`1.448e+09 in all`, 34% of `CLIMB_WORK`);
- `LR round_found step=632`: the three answer programs, the four rounds found and the library; the programs and the
  three rounds they call go into the buffers (`ROUND_SOURCE` below reads them back);
- `LR frozen step=831 history=[…]`: the freeze, with every phase change and its step;
- at evaluation: `ROUND` (the tournament's weights), `ROUND_SOURCE (the buffers)` (the programs as read back from the
  buffers), and `AUDIT` (how many rows of the subtraction table the count downs used, and how many of those training
  never visited: 0).

In the legal file's log: `LR climb step=580 L3 pose: 96 candidates on 96 roots; read mod(mul(x,x),N) on 23 of 79
answer rows, 35 laws tried` (its law reader, with no residue rule before it), `LR climb step=613 L4 round: cost 2
calls`, `LR outer step=614 answer program: cost 3` (`1.278e+09 in all`, 30% of `CLIMB_WORK`) and
`LR frozen step=813`. In fast2's log: `LR climb step=629 L4 round: cost 2 calls` (its round),
`LR outer step=630 answer program: cost 3` (three answer programs, and `3.254e+09 in all`, 76% of `CLIMB_WORK`) and
`LR frozen step=829`. In s3i's log:
`LR climb step=671 L6 round: cost 5` (the round found, as one of several equivalent spellings), `LR frozen step=870`
(the freeze), the `ROUND` line (the four programs the tournament holds, with their weights), and the
`SCHEDULE … history=[…]` line (every phase change, with its step).

Three things that look odd but are not:
- `exact_accuracy` reads 0.9999999892 on ood_t and ood_n_t in all four files' official records, although
  `correct_examples` is 10,002 of 10,002. The evaluator sums float32 batch means, and the shortfall is exactly one
  float32 rounding (2⁻²⁴) on the last batch of 1,810 rows. The counts are what certification uses.
- `final_train_loss` is −0.0: after the freeze the loss is the parameters' sum times 0.0, exactly zero with the sum's
  sign.
- Training reports 783.3 s for the family file (465.3 s for the legal file, 759.3 s for fast2, 1,340.3 s for s3i),
  though the model froze at step 831 (813, 829, 870): the evaluator keeps stepping, at learning rate 0, to the file's
  20,000 steps. The progress line just after the freeze (step 900) reads 559.1 s (308.1 s, 545.7 s, 1,143.0 s).

[OVERVIEW §9](OVERVIEW.md#9-where-docs-and-records-disagree) lists every place a write-up and a record disagree.

## 4. How to reproduce

### 4.1 With the competition's own evaluator (one H100, no Modal)

```bash
git clone https://github.com/tilde-research/one-layer-deeper.git && cd one-layer-deeper && git checkout 87e5c26
uv venv .venv && source .venv/bin/activate && uv sync
bash scripts/generate_datasets.sh        # every tier; the H1 block alone (lines 282-301) is enough
python3 ../cpu_first/tools/verify_records.py --data \
  data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2
mkdir -p /tmp/family && cp ../cpu_first/integration/legal_family/submission_stage3_family.py /tmp/family/submission.py
CUDA_VISIBLE_DEVICES=0 python -m benchmark.runner --manifest benchmark/manifests/h100_hard_h1.json \
  --submission-file /tmp/family/submission.py --include-structured-metrics | tee family_run.log
python3 ../cpu_first/tools/verify_records.py --log family_run.log
```

- **The family file needs Triton and a C compiler at run time.** On Linux, `uv sync` installs the evaluator's pinned
  torch 2.12.1 and, with it, Triton 3.7.1 (both in the clone's `uv.lock`). Triton builds a small C launcher with the
  host's C compiler the first time it runs a kernel. The evaluator's own image has one (`build-essential`, in the
  clone's `modal_runner.py`, line 49), and so does our runner's (§4.2). The file checks only that Triton imports
  (`_fused`, line 949).
- **Without Triton** the file runs its search's machine one instruction at a time (`_ustep`). That gives the same
  states (FAST.md §3), but slowly: round 2's file, whose same path replayed CUDA graphs, needed about 3,792 s of
  training to reach its round on an H100 (COMBINED.md §9), and the fallback has no graphs (FAST.md §6). The family
  file's official climb does 0.44 times fast2's work (1.448e9 units against 3.254e9); the fallback was not run on it.
- The `--data` line should print `a6414ec9fc6a5efce6414895b52d4eb7`, the H1 folder every H1 record of the shared
  runner and its copy read (this checkout's own H1 folder gives it too).
- The copy only gives the file rule 1's name; the bytes are unchanged.
- The run takes about 14 to 23 minutes (the eight records' wall times: 856–1,360 s).
- `--log` compares your `RESULT_JSON=` line with `legal_family_official`: the certified T, every rung and the
  splits. Times will differ: the eight H100 runs trained in 562–986 s.
- No decision reads the clock: the searches stop by work budgets, the same count on every host (STRICT.md §2), and the
  readings' routine calls and ROUNDT's runs count in the climb's. A slower host takes longer for the same climb.
  fast2's official climb was the same 72 lines in four separate H100 runs (§2); the family file has one run a seed,
  so that is not shown for it.
- For the legal file, copy its file instead, and compare with
  `--log legal_run.log integration/combined_legal/records/combined_legal_official.json`. Its run took about 13 to 19
  minutes (wall times 766–1,130 s), and its eight H100 runs trained in 465–721 s.
- For fast2, copy its file instead, and compare with
  `--log fast2_run.log integration/combined_fast/records/combined_fast2_official.json`. Its run took about 19 to 21
  minutes (wall times 1,143–1,273 s), and its eight H100 runs trained in 759–883 s.
- For s3i, copy its file instead, and compare with `--log s3i_run.log integration/records/s3i_official.json`. Its run
  took about 26 minutes (the official record: 1,557 s of wall time), and its eight H100 runs trained in 1,133–1,350 s.
  Its climb reads the wall clock (§5): a much slower host can change how far a search gets before its time limit; on
  an H100 its searches end well inside their limits (the round's took 26.6 s).

### 4.2 With our runner (Modal)

[`tools/modal_runner.py`](tools/modal_runner.py) builds the competition's image (CUDA 12.9.1, Python 3.13,
`build-essential`, `uv sync --frozen` on the clone's lock file), regenerates every dataset with
`scripts/generate_datasets.sh`, sends the file as bytes, checks its md5 in the container, and runs the untouched
manifest on an H100. It needs a Modal account, the Modal client 1.2.6, and the competition clone at
`../one-layer-deeper` (or `ONE_LAYER_DEEPER=<path>`). The family file's records came from its copy,
`integration/legal_family/modal_runner_family.py`, which differs only in the two held-out tasks built into its image
and its task list, its paths, its docstring and the tool's name in its records (§3):

```bash
R=integration/legal_family/modal_runner_family.py
python -m modal run --detach $R --mode official --file integration/legal_family/submission_stage3_family.py \
  --name family_judge --out /tmp/judge_records --run family-judge
python $R --mode fetch --out /tmp/judge_records --run family-judge --wait
python $R --mode compare integration/legal_family/records/legal_family_official.json \
  /tmp/judge_records/family_judge.json
python -m modal run --detach $R --mode plan --plan integration/legal_family/plans/family.json \
  --out /tmp/judge_records --run family-tasks
```

The last command runs the eight family tasks, one H100 each, with a cap of 48 GB each (`plans/family.json`); they peaked
at 17.1 GB. H1's eight runs had the runner's default cap, 24 GB, and peaked at 13.8 GB. For the legal file or fast2, run
`tools/modal_runner.py`, which wrote their records, with its file and record and `--memory-gb 48`, the cap their runs
had; for s3i, give its file and record and leave the cap at its default. [tools/README.md](tools/README.md) has the
other modes, the limits, and the runner's own validation against stored records. The write-ups' commands say
`~/github/sotaku/venv/bin/python`: that is the author's virtualenv, and any Python with torch, numpy and that Modal
client does.

### 4.3 Cheap checks (CPU)

- **The family file is built by its builder** (about a second; it reads the no-book file from this checkout's git
  history, and the contest's source policy from the clone next to it). This is `make_family.py`, without writing
  anything:

  ```bash
  python3 - <<'EOF'
  import hashlib, importlib.util as u
  s = u.spec_from_file_location("m", "integration/legal_family/make_family.py")
  m = u.module_from_spec(s); s.loader.exec_module(m)
  print(hashlib.md5(m.build()[0].encode()).hexdigest())
  EOF
  ```

  It prints the family file's md5. With `integration/legal_nobook/make_nobook.py` the same lines print the no-book
  file's (`b42ec30422c182c0972aebca5be0c470`), with `integration/legal_wide/make_wide.py` the wide file's
  (`3bb5ceebf1de115501efc3398af8a457`), and with `integration/combined_legal/make_legal.py` the legal file's. With
  `integration/combined_fast/make_fast.py` and `m.build(fast2=True)[0]` they print fast2's (`m.build()[0]` the fast
  file's), and with `integration/combined_sp2/make_combined.py` and `m.build(round_=2)[0]` round 2's,
  `4e0e3f09b1ab55d1bfedbbed4fb0afc2`: that one is built from the certified files, every md5 asserted.
- **s3i is built by its sources.** Without writing anything:

  ```bash
  python3 - <<'EOF'
  import hashlib, importlib.util as u
  s = u.spec_from_file_location("m", "integration/make_stage3.py")
  m = u.module_from_spec(s); s.loader.exec_module(m)
  print(hashlib.md5(m.build().encode()).hexdigest())
  EOF
  ```

  It prints s3i's md5. (`python3 integration/make_stage3.py s3h` checks the previous build, s3h, the same way.)
- **The competition's static source policy** (the check the CLI and the evaluator run before importing a file):

  ```bash
  python3 - integration/legal_family/submission_stage3_family.py <<'EOF'
  import sys
  sys.path.insert(0, "../one-layer-deeper")
  from submission_validation import validate_submission_source
  print(validate_submission_source("submission.py", open(sys.argv[1]).read(), 256 * 1024))
  EOF
  ```

  It prints `submission.py` (a failure raises) for the family file, the legal file, fast2, the fast file and the three
  other frontier files.
- **The local checks** (CPU, about 5 minutes and 4.6 GiB; they need torch and the H1 data): combined_sp2's
  [`checks/local_combined.py`](integration/combined_sp2/checks/local_combined.py) runs each piece's own checks on a
  file: the source policy, the machines, the instruction set, the designed path, the buffers and the state dict, the
  gradient test of rule 8, the fix, the book, a tiny-budget climb and a replay. FAMILY_FILE.md §12 has the command
  (`COMBINED_FILE` picks the file), and `integration/legal_family/checks/logs/local_family.log` its output on the
  family file. [`checks/compare_local.py`](integration/legal_family/checks/compare_local.py) sets that beside the
  no-book file's output: every part gives the same results but the size, the fix's line naming the file, and the book,
  whose code differs from book_gen_fix's by 105 lines against 92 (the port's patterns and dispatch), as they should
  (FAMILY_FILE.md §3). On a CPU the kernel never runs; its own checks need a GPU (FAST.md §3).
- **The readings and ROUNDT alone:** [`checks/family_local.py`](integration/legal_family/checks/family_local.py) runs
  the readings and the pose after them on the legal file's tiny-core state at its level 3, with each task's answers on
  H1's 400 answer rows of T = 1, made by Python's integers outside any run. H1's law is read on all 400 rows, the
  family's as FAMILY_FILE.md §3 lists them (none for 10y², 10y² + 1 and 3y² + 2), and every posed law's answers equal
  Python's integers on every draw: 60 law-pattern pairs, 0 disagreements. `MODE=roundt` and `checks/roundt_rows.py`
  check ROUNDT: the no-book file's four H1 rounds pass it on 1,000, 4,096 and 4,096 rows of T = 2, 4 and 8, and each
  fails with one answer of T = 4 changed (FAMILY_FILE.md §3).
- **For s3i**, the plumbing and gradient checks are in [INTEGRATION §7](integration/INTEGRATION.md#7-reproduce).

## 5. The gray areas: what a judge should rule on

Each of the contest's 16 rules in turn, for the family file. "Gray" means a reasonable reader could object; the point
is flagged, not argued away. Line numbers are the family file's. REVIEW.md as of commit `afbd36e` takes every rule for
the legal file, at the legal file's line numbers, and as of commit `43da60f` for fast2, at fast2's; where they differ
from the family file, the end of this section says so.
[OVERVIEW §6](OVERVIEW.md#6-rules-116-for-the-best-model) does the same for s3i, and
[STRICT.md](integration/strict/STRICT.md) §1 takes the gray points one at a time, with what the strict build changed
and why the rest stayed. Where s3i differs, the end of this section says so too.

1. **One `submission.py` exporting one `Submission`: yes.** The file is sent as `submission.py`, in UTF-8. It has one
   `SUBMISSION` (line 5106): the model and optimizer factories, a token loss, batch sizes of 512 and 2,048, and a cap
   of 20,000 steps (lines 129–131). It is 261,651 bytes of the 262,144 allowed, and the contest's static source policy
   passes (§4.3).
2. **Self-contained: yes, with a word on Triton.** It imports the standard library (`itertools`, `math`, `time`,
   `zlib`, `array`, `collections`), `torch`, the public `benchmark` API and `triton` (lines 11–31). No numpy, no files,
   no network.
   - Triton is not named in the evaluator's `pyproject.toml`. It is pinned in its `uv.lock` (3.7.1) as a Linux
     dependency of the pinned torch 2.12.1, and `uv sync --frozen` installs it.
   - The import sits in a `try` (lines 25–29): without Triton the file runs the old machine, which gives the same
     states but slowly (§4.1).
   - Triton builds a small C launcher with the host's C compiler, which the evaluator's image installs
     (`build-essential`).

   *Does a dependency pinned through torch count as a "pinned evaluator dependency"?*
3. **The participant defines the model, optimizer, loss, batch sizes and steps: yes.** Adam (`LR` = 0.05, line 42),
   which clears its moments once, in its own step, when the schedule enters phase B (`Adam`, line 5067); a scheduler
   that sets the learning rate to 0 once the model is frozen (`Schedule`, line 5083); and the loss
   (`token_training_loss`, line 5044). The answer program runs its round as many times as T says, at most 100
   (`LOOP_CAP`, line 179): recurrence and adaptive computation, which rule 3 allows.
4. **The evaluator owns the loop, data order, backward passes, clipping, cadence, seeds and deadline: gray.**
   - The evaluator is untouched: one forward, one loss and one backward a step, no extra passes, no batch reuse.
   - No decision reads the clock or the budget. `build_optimizer` (line 5101) reads only the device type, and the clock
     (`_clock`, line 1435) only stamps the log's seconds, the readings' records among them. Each search stops after
     `SEARCH_WORK` = 8e8 units of machine work, and the climb, the readings' routine calls and ROUNDT's runs included,
     after `CLIMB_WORK` = 4.3e9 (lines 77–78), counted by the search's own cost model, the same on every host
     (STRICT.md §2).
   - But the model keeps its own schedule as Python state (`Control`, line 3982): the prompt reader's stages, phases
     A–D (the subtraction table, the other cells, the climb, the tournament), and a freeze that sets the learning rate
     to 0 once every check has held for 200 steps (`FREEZE_WIN`, line 128). That is early stopping on the model's own
     checks. The loss advances it (`control`, line 4952, called at line 5063).
   - And it keeps rows across batches, as buffers (lines 4048–4057): up to 4,096 read rows for its practice problems
     (`POOL`), 512 answered rows of the smallest T for the round's search and the readings (`ANS_POOL`), and the
     newest 64 answered rows of each count for the answer program's search (`count_rows`, `COUNT_EACH`). It draws its
     practice probes from them with its own seeded generator. The answered rows come in through the loss, which reads
     the labels (`keep_answers` and `keep_counts`, lines 4772 and 4787).
   - ROUNDT keeps more, and not as buffers. `keep_counts` also keeps the newest 4,096 answered rows of each count it
     keeps (on H1, T = 1, 2, 4 and 8) as tensors on the GPU in a Python dict on `Control`, `more` (lines 4022 and 4805),
     and `climb_unit` hands those of T > 1 to the round search (line 4674). They are outside the state dict, where the
     strict build had put the rows the climb keeps (STRICT.md); FAMILY_FILE.md does not say so. The readings keep no
     rows of their own, but for the residue rule's record of the rows it read, as text for the log (rule 14).

   *Is a model that keeps its own schedule and a memory of past rows, and stops its own learning, within rule 4?*
5. **At most 500,000,000 parameters; buffers count: yes.** 207,370 parameters in 15 tensors (lines 4031–4046). Every
   record counts 494,712 numbers of model state at the start: the parameters, and 287,342 numbers of buffers for the
   kept rows. The buffers that hold the answer programs grow once those are found: 336,487 buffer numbers in all in the
   local check on the family file, with the generic instructions' official rounds
   (`integration/legal_family/checks/logs/local_family.log`). There, a fresh model loaded with the state dict gives
   the same logits.
   - While it searches, the climb's working state (its routines, merges and probes, and the laws read with what they
     pose) stays in Python, outside the state dict, as an optimizer's state would (STRICT.md §1); what the model answers
     with is in the buffers. ROUNDT's kept rows are outside the state dict too (rule 4): at most 4,096 rows of four
     8-digit numbers for each count kept, so at most 524,288 numbers on H1's four counts, far below the limit if they
     were counted.

   *Should the climb's working state count?*
6. **No hard-coded weights: yes.** All 15 parameter tensors start at 0.1 times a random normal (`INIT`, line 41), and
   the buffers start empty. No `torch.load`. The file also holds the true digit tables (`Cells`, line 361), used only by
   the "designed" plumbing mode (`MODE` is "learned", line 33) and by the log's report-only "right vs the truth" counts
   (`exact_cells`, line 3924); nothing the model computes reads them. Rule 6 also says trainable weights must "be
   updated during training". No record shows which of the 15 tensors moved before the freeze.
7. **No hard-coded algorithm in the forward pass: gray, less than for s3i.**
   - The forward pass runs the answer program the training run found, such as `C.S[T.round1 STA.X]`, from the buffers,
     on one-hot digits through the learned tables (`predict_ste`, line 4485). Its rounds, such as
     `LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.X T.red.x+on.XN`, call routines the climb solved and kept: the
     multiply-add row looped over the digits (a multiply of two numbers), then the reduction, Horner's loop over long
     division's step, which is "ten times: subtract N, and add it back if that borrowed", built from the comparison,
     the addition, the if block and the shifts. Both were posed by the law the run read, and the comparison and the
     addition by the residue rule. No datapath, no menu and no countdown is in the file. SEL.S and BRG.X are not in
     its instruction set; their code is still in the machines, switched off by `COND` and `BRING` (lines 164 and 171).
   - Still designed, knowing the task family:
     - the instruction set (lines 164–179): five registers of base-10 digits with one pointer, a borrow, a multiplier
       digit and its carry; 30 instructions that read the learned tables at the pointer, move whole registers, shift
       through the carry, clear the flags, load the multiplier digit or move the pointer; loops by the register's
       length, the base and the operand's width; the if block on the borrow; and the count loop, whose stop rule is the
       old countdown's (LEARNED_LOOP.md §7);
     - the residue rule (`read_fact`, line 3141): if every kept answer is below one prompt field, compared digit by
       digit from the top place (`compare`, line 3696), the law ends in a reduction by that field, and the reduction's
       step is posed cut to its parts. It names no task, but it is family knowledge: every task of the family ends in
       mod N, so on the family it fires as on H1, which is not generality. It decides what to practise and computes no
       target, which is the open question LEGAL_SOURCES.md §6 raises for a judge (rule 14; NOBOOK.md §7);
     - the step's cut table (`STEP_PARTS`, line 95): the comparison, the link and the addition, round 6's cuts of the
       step with the addition for its sum;
     - the reader's grammar (`LawReader`, line 2795; `READ_OPS`, line 2792; `READ_MIN`, `READ_POW`, `READ_DEPTH` and
       `READ_TRIES`, lines 101–104): x and N as its leaves, not T; constants only as gaps by which a reduction mod N
       misses every answer; add, sub, mul, pow and mod, each with its construction; depth 3; the order of its tries;
       and where a value counts as computable (WIDE.md §2). It widens the legal file's (add, mul and mod, depth 2),
       which was chosen knowing that H1's law is a depth-2 chain of those operations. A law outside it is not read:
       10y² (a factor, not a gap), 10y² + 1 and 3y² + 2 (depth 4), on which the climb practises the step's three laws
       only (FAMILY_FILE.md §7);
     - the window rule (`windows` and `read_more`, lines 2957 and 3005; `READ_WINDOW`, line 97): runs of 2 or 3 of the
       reading's operations with at most 2 holes, named a and b in order, N and the constants kept, posed smallest
       first on the reading's own patterns (`n`, `x,n`, `w1` and `w2`, lines 2032–2034) with the modulus bound to N;
     - the constants rule (`constants` and `read_more`, lines 2995 and 3005; `CONST_OPS`, line 98): each constant of
       the reading made from N, and the reading's add, sub and mul with it;
     - ROUNDT (`round_t`, line 3503; `ROUNDT`, line 99): a round must also give the newest 4,096 kept answers of every
       count T > 1, run T times; which counts are kept (those that read the same both ways, `keep_counts`, line 4787)
       and how many rows. Round 7 found it on y² − 1, whose first round was wrong where an input is 0 (FAMILY.md);
     - the cut and loop rule (`read_pairs`, line 2929): an operation's laws f and add.f, on the patterns whose count is
       a digit (the cut) or another row's number (the loop); mod is posed as the book's red;
     - the catalogue of law forms, which now answers only what the readings pose (lines 2014–2030, `Draw` at 2134,
       `law_target` at 2228): numbers r + q·b from the division law, the counts 1, d, o and x, the compositions of +
       and ×, and targets from facts, identities and the learner's own routines (SP_ABLATION.md §5.2);
     - what fast2 designs: the choosing rule (`SP_RULE` = "lpca", line 53), picked after bar 3's own rules failed, by
       watching whether the round came (SELF_PRACTICE.md §6); the search's sizes and rules (lines 53–93 and 105–114),
       among them seven that the generic instructions needed, each found by watching whether the round came: five in
       how the search picks and orders its probes, acceptance at four digits, and routines that leave the other
       registers alone first (GENERIC_OPS.md §7); where the round and the answer program are looked for (lines 71–75
       and 85–92), and T's 2-digit field (`K`, line 39), which is what tells T from x (LEARNED_LOOP.md §6, §7); stage
       1's reader of the prompt (24 reading hypotheses, 3 pointer shapes), the cells' shapes (three tables) and the
       schedule.
   - A judge may say `SHL.X SHL.A` is BRG.X in two pieces. GENERIC_OPS.md §7 agrees, and notes that it is the form a
     place-by-place law gives BRG.X. A judge may also say the count loop is the countdown made into an instruction
     (LEARNED_LOOP.md §7), that the reader's grammar is the family's laws written as a search space, and that the
     residue rule is the family's "mod N" written as a rule.
   - The Triton kernel adds no algorithm. It runs the old machine's instructions, to byte-for-byte the same states
     (FAST.md §3), and every digit it writes is a lookup in the learned tables or a move (`_lock_k`, line 1013). It runs
     only the climb's searches; the answer is computed on the one-hot machine (`SteMachine`, line 1833).

   *Did the algorithm move into the instruction set, the residue rule, the reader's grammar, the rules for windows,
   constants and rounds, the cut and loop rule and the search's rules, and does that count as hard-coding it?*
8. **Final logits from the model, inside autograd, with an unbroken gradient path: yes at evaluation, with gray
   edges.**
   - At evaluation the forward runs the prompt's reader, the answer program from the buffers and the writer on one-hot
     digits through straight-through tables (`predict_ste`, line 4485, and `SteMachine`, line 1833). The tournament
     (`prog`, one logit a program) is a straight-through choice among up to four answer programs.
   - The gradient test on the family file (combined_sp2's `checks/local_combined.py grad`, CPU; its output, the same
     as the no-book file's, is in `integration/legal_family/checks/logs/local_family.log`): the reader and the cells at
     the truth, two answer programs over the generic instructions' official rounds, and the right tokens' logits on 2
     rows of depth_t_2 as the objective. 14 of the 15 tensors get a nonzero gradient. The tournament's gets exactly 0,
     since its two programs give the same answer and a softmax's derivatives sum to zero (LEARNED_LOOP.md §4). The
     multiply tables' gradients are about 1e-33: damped.
   - Gray (STRICT.md §1, §3 and §5):
     - (a) in training the forward returns zero logits (line 4454), so the lessons teach, not the final logits. The
       real answer inside autograd on a training batch does not fit in an H100's 80 GB (512 rows at T = 16). On the
       batch's answer rows, where it would fit, it is exact once the round is found, so it would teach nothing
       (measured on the strict sp2 file, STRICT.md §5);
     - (b) the gradient is damped by 0.1 at each lookup (`GRAD_DAMP`, line 35) so that it does not overflow, and on the
       deepest paths it all but vanishes: about 1e-33 on the multiply tables above, and exactly 0 in float32 in some
       earlier files' tests (OVERVIEW §6, LEARNED_LOOP.md §4). The count loop's count down over 16 places makes the
       path deeper than sp2's countdown;
     - (c) the programs are a discrete result of search, kept in buffers; only the choice among them is a parameter;
     - (d) which reading hypothesis a prompt uses is decided by designed run-detection logic (stage 1's reader:
       `runs`, `which_run` and `selected`, lines 4286–4319).
   - The kernel is outside autograd, but not on the prediction's path: it runs the climb's searches, on integers. So
     do the readings and ROUNDT: they run inside the climb, the readings choosing what to practise and ROUNDT which
     rounds the search accepts, not what to answer.

   *Are the final logits "produced entirely by the submitted model from its inputs and learned PyTorch state, with
   all input-dependent computation inside the autograd graph and an unbroken gradient path from the loss to the
   parameters responsible for the prediction"? The middle clause bears on (d).*
9. **Everything on the GPU: yes, with a word.** The parameters, the buffers and every machine are on the GPU: the
   climb's searches (the Triton kernel, line 1013, or the old machine), the readings' routine calls, ROUNDT's runs and
   its kept rows, training's checks (replayed from CUDA graphs) and the prediction's one-hot machine. There is no
   `.cpu()` and no numpy in the file. It does build a few constant tables on the CPU and copy them to the GPU before
   use: two hash tables drawn at import (`_MULT` and `_SALT`, lines 601 and 2517), the instruction set's opcode tables
   (`_Ops`, line 775), and the true digit tables each `Cells` starts with (line 361), which `snapped_cells` (line 4096)
   replaces with the learned ones. Python holds the bookkeeping: the schedule, the climb's programs as tuples, its
   library and BPE over program text, the readings' loops over their candidates, the laws' text and the windows' trees
   parsed from it. The host reads small results back from the GPU (program indices and counts, the readings' bits,
   row counts and constants, and ROUNDT's counts and bits: rule 14) to steer the search, and traces each program's
   instructions from its text into opcode columns on the GPU. That is s3i's arrangement (OVERVIEW §6). *Does that
   bookkeeping count as computation off the GPU?*
10. **OOM or timeout fails: ok.** Neither happened on the eight runs. Their peak host memory was 12.7–13.8 GB. Rule 10
    is about GPU memory, and the records give no peak for it.
11. **3,600 s of H100 training, compilation included: yes, on every seed.** The eight runs trained 562–986 s, to the
    file's 20,000-step cap, and froze by 368–786 s (the progress line just after each freeze). That is slower than
    the no-book file's 472–634 s, almost all of it ROUNDT's checks in the round search (§2). Triton compiles the
    kernel inside the run, once for each set of its compile-time constants, so that time is in these figures; it was
    not measured apart. A climb that spent its whole work budget would still freeze by about 1,900 s (§2). On the
    family, the three runs that read no law trained until the hour ran out, when the evaluator stopped them
    (FAMILY_FILE.md §7).
12. **A custom loss returns one finite differentiable scalar: yes.** `token_training_loss` (line 5044). It also keeps
    answered rows and advances the schedule (rule 4).
13. **Evaluation within half the training allowance; certification: yes.** Evaluation took 289–394 s of the 1,800
    allowed; T = 64 on both ladders on all eight runs, every rung 768 of 768.
14. **No data inspection, augmentation, task-specific solvers, custom training loops, participant backward passes or
    hidden training work: gray, and the main question.**
    - None of the named mechanisms: no `backward` or `grad` call, no custom autograd function, no file read, no number
      made up. No practice answer comes from Python's arithmetic. Every value the law reader compares is the learner's
      routines', and the residue rule compares the data's own digits (below).
    - **How every practice answer is made.** From facts, identities and the learner's own routines, the same rules
      for every law (`law_target`, line 2228; SELF_PRACTICE.md §2.2):
      - the comparison's, by the order facts of r + q·N forms; the link's, by the division law's identities and the
        fix's probes N < X < 2N, made from one entry of the learned addition table; the addition's, by its identity
        and a check run by the learner's link and comparison on the candidate's own output (NOBOOK.md §2);
      - the cuts and loops (the multiply-add row, x·d, their loops, Horner's step and the reduction), by repeated
        addition by doubling over the learner's addition, and by the division law with x + d·N made by the learner's
        addition (LEGAL_FILE.md §2);
      - the windows' and constants' (`read_target`, line 3029): the law's tree computed by the reader's operations at
        the draw's size, which are the learner's routines (add by its addition; sub by its comparison's difference,
        kept where its addition gives the minuend back; mul by repeated addition by doubling; mod by its link by
        doubling); rows where a value on the way is not computed are masked, and a constant's answers are its digits
        as the reading's text holds them (FAMILY_FILE.md §2, §10; PREREG.md).
    - The loss calls the model's lesson methods, never the model itself, and starts no optimizer or scheduler step.
      But its `control` (line 4952) sets flags that both read: `ctl.reset` (line 4986), which makes Adam clear its
      moments in its own step (line 5077), and `ctl.phase`, whose "frozen" the scheduler turns into a learning rate
      of 0 (line 5090).
    - But the climb is a program search inside the training forward pass (`climb_unit`, line 4659, called from
      `lessons`, line 4723), one unit a step. On the official run it ran from step 563 to step 632, from about 54 s to
      about 506 s of the 783 s of training. What it found, the programs, carries from step to step in buffers.
    - Its practice problems are made from the batch rows' own digits and what the readings pose. The readings and the
      round's search read the answered rows of the smallest T, ROUNDT the answered rows of every count T > 1, and the
      answer program's search those of each count, all kept by the loss from the labels.
    - **The readings** (`read_fact` and `read_law`, lines 3141 and 3129, inside `pose`) test the labels and let what
      they find decide what the climb practises next. Where Python still touches them (NOBOOK.md §8, WIDE.md §8,
      LEGAL_FILE.md §8 and FAMILY_FILE.md §10):
      - the residue rule's decision is `compare` on the GPU: per row, the highest place where an answer's digits and
        a field's differ, and the two digits there. One bit per field is read back, and no number is made from the
        digits. The rows it read are read back as text (N:y) for the record and the log only;
      - every value the law reader compares is an output of the learner's addition, comparison or link, run by the
        machine on its learned cells, or of repeated addition or the link by doubling over them; its test is digit
        equality with the answer rows. Python runs the loop over the candidates, depth by depth, and keeps the law's
        text, the count of laws tried and the (law, pattern) names it poses. Distinct values are found by hashing
        each candidate's digits and rows on the GPU, and the hashes, row counts and exactness bits are read back, a
        small result per candidate; no digit is read back but for the constants' names, which also give the
        constants' digits;
      - a window's tree is parsed from the reading's text (`tree`, line 2939), and its name and the constants' digits
        are made from that text; every value of the tree is computed by the learner's routines through the reader;
      - beyond selections (a value's digits are zeroed on the rows it is not computed on) and counts of rows, the
        designed arithmetic on the rows is counts read off digits: the multiply's loop count, which `count_of` (line
        1784) reads off the multiplier's digits and whose bits `times_add` (line 2098) takes for the doubling; a
        power's count, read the same way and compared with `READ_POW`; and ROUNDT's T (below). Each is read with
        torch's integers on the GPU, as fast2's practice constructions already read theirs, and each counts operations;
        none is compared with an answer. Which rows a value counts on is set by digit facts: the operands' top places
        are zero, the divisor is not.
    - **ROUNDT** (`round_t`, line 3503) decides which rounds the search accepts, not what to practise, against labels
      the loss kept. Its T is read off each kept count's digits by `count_of` and passed to the host as a Python int
      (`climb_unit`, line 4674), and the round runs T times on the GPU; one bit per candidate (right on every row, or
      not) is read back (FAMILY_FILE.md §10).
    - A miner computes relations over each batch's read numbers N, x and T inside the model (`relations` and `Miner`,
      lines 3706–3763). Only its order and nonzero facts feed the lessons (lines 4740–4741); the others, equality,
      constancy and ranges, are only logged. The legal file's miner also ran on the answers and tested divisibility by
      2, 3 and 5, all only logged; the wide file cut both (WIDE.md §2).
    - The rules call "parameter-free work" valid and allow iterative mechanisms (rule 4, and the note after the
      rules). That note is about depth ("Depth is deliberately unconstrained."), so a judge may read "parameter-free
      work" as depth in the forward pass, not as a search whose result carries from step to step.

    *Is a search inside the forward pass, whose result carries from step to step, "hidden training work"? Are a
    residue rule, a reader's grammar and the rules for windows, constants, cuts and loops a task-specific solver
    written as lessons? Is a law read from the labels, by a designed rule or by the learner's own routines, "data
    inspection"?*
15. **Repeated rule-breaking: does not apply.** The write-ups flag their own gray areas, and the hosted evaluator has
    scored nothing: the one attempt, after the deadline, was refused (OVERVIEW §7).
16. **The Hard metric recorder: yes.** It is not touched.

**The questions, in the order [LEGAL_PLAN.md](LEGAL_PLAN.md) §3 asks them, then
[LEGAL_SOURCES.md §6](bar3/LEGAL_SOURCES.md#6-a-question-for-the-judge)'s.** Ask the first one first: if it goes
against the entry, the whole stage-3 line is out, and nothing else here matters.
1. **Rules 8 and 14:** the climb is a program search inside the training forward pass, its result kept in buffers. The
   rules call "parameter-free work" valid and allow iterative mechanisms, though in a note on depth (rule 14 above).
   Is a search whose result carries from step to step "hidden training work"?
2. **Rule 8:** training returns zero logits, and the damped gradient all but vanishes on the deepest paths. The batch's
   answer rows would fit through the real path, at 200 to 400 s a run, but would teach nothing once the round
   reproduces them; the whole batch does not fit in an H100 (STRICT.md §5).
3. **Rule 4:** the model keeps rows across batches, as buffers and, for ROUNDT, as tensors held in Python, and its own
   schedule: phases A–D and the freeze.
4. **Rule 7:** the instruction set, the residue rule, the reader's grammar, the rules for windows, constants and
   rounds, and the search's rules were chosen knowing the task family. When does what is left count as general? On the
   family the same file learns 4 of 8 tasks at full size, but only its two held-out tasks test that, and it learns one
   of them (§2).
5. **Rule 14, LEGAL_SOURCES.md's question:** may a decision about practice use Python's arithmetic on the training rows,
   if no target does? In the family file, what to practise is decided by two readings of the labels. The residue rule is
   a designed rule that compares the answers' digits with N's and x's and makes no number. The law reader's values are
   all the learner's routines' on the learned cells, and it decides by digit equality with the labels. Beyond selections
   and counts of rows, designed arithmetic touches the rows only in counts read off digits on the GPU, the multiply's, a
   power's and ROUNDT's T, as fast2's practice constructions already read theirs; Python runs the readings' loops, keeps
   the laws' text, parses the windows from it and reads back a few bits, row counts and the constants' digits (rule 14
   above). What is left to rule on is those counts, and whether a law read from the labels, by a designed rule or by the
   learner's own routines, may choose what to practise at all.
6. **Bar 2** (the project's own bar, not a rule): of what is still designed, which parts would you call the algorithm?
   Rule 7 above lists the family file's. [LEGAL_PLAN.md](LEGAL_PLAN.md) §2 lists sp2's, with a cheapest test where
   there is one. Of those, fast2 replaces the 11-law list with a grammar (item 1), SEL.S and BRG.X with an if block and
   shifts (item 3) and the countdown with the count loop (item 4), and has neither the late table lessons (item 7) nor
   the clock (item 8). The legal file goes on with item 1: three laws, and what the law reader poses, in place of the
   grammar's whole book; the family file takes item 1 to an empty book, its first laws posed by the residue rule.

**Where the legal file differs** (line numbers are the legal file's):
- Rule 7: its book is three laws on the pair, the comparison, the link and the addition, posed from the start
  (`BOOK`, line 101); it has no residue rule, no step's cut table, no window or constants rule and no ROUNDT. Its
  reader's grammar is narrower (`LawReader`, line 2818; `READ_OPS`, line 2815; `READ_MIN`, `READ_LINKS` and
  `READ_TRIES`, lines 102–104): add, mul and mod over x and N, depth 2, no constants, its mod the link repeated at
  most 1,000 times. Its cut and loop rule is the same (`read_pairs`, line 2882).
- Rules 4 and 5: every row it keeps is a buffer (lines 3916–3925); it has no ROUNDT rows.
- Rule 14: what it practises beyond the three laws is decided by the law reader alone, which read H1's law on 11–23
  of the 69–95 answer rows kept by then; its miner also ran on the answers (`mine_answers`, line 4305) and tested
  divisibility, only logged.
- Rules 10, 11, 13 and 14, in numbers: its peak host memory was 12.6–14.0 GB; its eight runs trained 465–721 s and
  froze by 308–496 s; its evaluations took 295–403 s; its official climb ran from step 563 to step 614, from about
  42 s to about 272 s.

**Where fast2 differs as well** (line numbers are fast2's):
- Rule 7: its book is the grammar posed whole (lines 2026–2043, `Draw` at 2154, `law_target` at 2259): every law of the
  grammar on every pattern whose construction worked, from the start. It has no three-law book, no law reader and no
  cut and loop rule; its tiny core practised 15 concepts, 7 of which the legal file never poses (LEGAL_FILE.md §7).
- Rule 14: what it practises is decided by the grammar alone, and nothing reads a law from the labels; the answer
  program's search and the round's still read the answered rows the loss kept.
- Rules 10, 11, 13 and 14, in numbers: its peak host memory was 13.8–14.7 GB; its eight runs trained 759–883 s and
  froze by 546–658 s; its evaluations took 334–407 s; its official climb ran from step 563 to step 630, about 492 s.

**Where s3i differs** (OVERVIEW §6 has its table; line numbers are s3i's):
- Rule 4: it reads the clock. `build_optimizer` (line 3425) sets a deadline from the budget the evaluator passes in;
  each search stops after `SEARCH_S` = 900 s (line 92); the climb stops `CLIMB_MARGIN` = 120 s before the budget ends
  (line 97). Its kept rows (`POOL` and `ANS_POOL`, lines 110–111) are Python lists.
- Rules 5, 8(c) and 9: its found programs are held in Python (`model.ctl.lib` and `model.ctl.progs`), outside the state
  dict, and traced on the host from their text.
- Rule 7: its practice problems are a hand-picked list, made from the law library (`LAWS`, line 140) in the order
  `NEEDS` sets (line 1413), and two of them are definitions of long division's step. It has SEL.S and BRG.X, and a
  hand-written countdown over T.
- Rule 8: its gradient test reached all 16 tensors on the earlier build s3g, whose prediction path s3i keeps
  (INTEGRATION §3, §5). With s3i's own four programs, 14 of 16 get a gradient: the addition table's two get exactly 0,
  since the chosen program calls no addition (STRICT.md §3).
- Rule 14: its lessons also include the old table lessons, joining `LATE_AFTER` = 100 steps after the digit laws
  (line 68), with the width-dependent fact "a wrapped x − N is large" among them (`sub.wrap`, in `table_lessons` at
  line 2881; the `ANCHOR_WRAP` constant on line 70 is never read, ABLATION_S3.md §3).
- Rule 11: it froze by 982–1,149 s on its eight H100 runs and trained 1,133–1,350 s.

**What to do about them.** [LEGAL_PLAN.md](LEGAL_PLAN.md) §4 puts these questions first, then orders the rest by cost.
Its step 2, one file with every certified variant, is fast2. The legal file goes past it on the law book by bar 3's
legal path, and the family file takes that book to nothing (§2). Its steps 3 to 5 have not run: the reduction's search
without its cap, the first step toward letting the climb take its digit instructions from every one-place slice of a law
(at most about $8); bar 3's subgoal score, as a near-miss test and then as the choosing rule; and one official run with
the freeze removed, training to the deadline (about $4).

## 6. An hour's reading order

- **0–5 minutes:** this file, "In plain words" and §1–§3: the claims and the evidence.
- **5–15:** OVERVIEW's "In one minute", [§1](OVERVIEW.md#1-the-task-in-plain-words) and
  [§2](OVERVIEW.md#2-why-it-is-hard): the task, and why exact answers defeat ordinary training.
- **15–25:** OVERVIEW [§4](OVERVIEW.md#4-where-it-stands-three-bars-and-the-models-at-the-frontier),
  [§4.7](OVERVIEW.md#47-discovering-the-skeleton-library-learning-then-stage-3) and
  [§4.8](OVERVIEW.md#48-variants-of-stage-3-that-go-further): the bars, stage 3, and the variants the family file is
  made of.
- **25–35:** FAMILY_FILE.md's "In plain words", §2, §6–§7 and §9–§10, then NOBOOK.md's "In plain words" and §7–§8:
  what the family file changes, its runs on H1 and on the family, what it designs, where it touches Python, and the
  empty book it starts from.
- **35–45:** the family file itself: the docstring (lines 1–9), the constants (33–133; the book, the step's parts, the
  windows, the constants, ROUNDT and the reader's at 94–104), `LAWS` (141), the instruction set (164–179), the kernel
  `_lock_k` (1013), the grammar (`PATTERNS` at 2022, `Draw` at 2134, `law_target` at 2228), the law reader
  (`LawReader` at 2795, `read_pairs` at 2929, `read_more` at 3005, `read_target` at 3029), `SelfClimb` (3040:
  `read_law` at 3129, `read_fact` at 3141, `pose` at 3151, `round_t` at 3503, `search_outer` at 3553), `Control`
  (3982), `LibraryMachine` (4025: `_forward` at 4442, `climb_unit` at 4659, `lessons` at 4723, `keep_counts` at 4787,
  `control` at 4952), and the loss, the optimizer and `SUBMISSION` (5044–5113).
- **45–50:** `python3 tools/verify_records.py`, and the log lines of §3: the records against the file.
- **50–60:** §5 here, then [LEGAL_PLAN.md](LEGAL_PLAN.md) §2–§3,
  [LEGAL_SOURCES.md §6](bar3/LEGAL_SOURCES.md#6-a-question-for-the-judge) and [STRICT.md](integration/strict/STRICT.md)
  §1: what is still designed, the judge's questions, and each gray point item by item.

With more time: INTEGRATION's ["In plain words"](integration/INTEGRATION.md#in-plain-words),
[§3](integration/INTEGRATION.md#3-stage-3-no-skeleton-no-menus) to [§6](integration/INTEGRATION.md#6-is-bar-2-cleared)
(every stage-3 build, including the failures, and the bar-2 call for s3i); [ABLATION_S3.md](integration/ABLATION_S3.md)
(what s3i's result rests on); the studies of each piece, [SELF_PRACTICE.md](integration/SELF_PRACTICE.md),
[LEARNED_LOOP.md](integration/learned_loop/LEARNED_LOOP.md), [GENERIC_OPS.md](integration/generic_ops/GENERIC_OPS.md),
[SP_ABLATION.md](integration/sp_ablation/SP_ABLATION.md) §5 and [LEGALLOOPS.md](bar3/scale/legalloops/LEGALLOOPS.md);
LEGAL_FILE.md, WIDE.md and round 7's [FAMILY.md](bar3/scale/family/FAMILY.md) (the steps from fast2 to the family file);
FAST.md's "In plain words" and §9 (the speedups and budgets it keeps); COMBINED.md (how fast2's five pieces were put
together); OVERVIEW [§5](OVERVIEW.md#5-the-removal-ladder) (each hand-written piece removed in turn),
[§6](OVERVIEW.md#6-rules-116-for-the-best-model), [§7](OVERVIEW.md#7-status-and-open-questions) and
[§9](OVERVIEW.md#9-where-docs-and-records-disagree); and bar 3's [PRACTICE.md](bar3/PRACTICE.md) and
[BREADTH.md](bar3/BREADTH.md). [README.md](README.md) indexes every file and folder.

## 7. Later studies

Three studies were still running when this file was first written. All three have finished and are committed, and
later ones follow them as they land. None changes a certified file or record. fast2 is built from five of them: the
strict builds, the learned loop, the law ablation's generated book, the generic instructions and the s3i ablation's
fix. The legal file adds a sixth: bar 3's legal reader and its three laws. The family file, now the file to judge,
takes the legal file three steps on: the wider grammar, no law in the book and round 7's rules, legally.

- **The s3i ablation** ([ABLATION_S3.md](integration/ABLATION_S3.md), code and records in
  [`integration/ablation_s3/`](integration/ablation_s3/)). Each hand-made part was removed in turn and the model
  retrained on the official manifest, one H100 per arm, on seed 74 and on seed 1. The unchanged file, run beside the
  arms, replays s3i's own runs on both seeds exactly.
  - **Needed on both seeds** (without it the round is never found, each time at the same level and missing the same
    piece): SEL.S (keep or discard on the borrow); BRG.X (bring X's next digit in); the two practice problems that
    define long division's step; mac posed with the running sum; the register twins; routines.
  - **Needed only because a check was weak:** Occam's tie-break toward fewer executed steps. It kept the search off
    a wrong conditional subtraction that the weak check let through; with the check fixed, s3i certifies without it
    on both seeds.
  - **Needed on one seed of two:** the chain problem alone. Without it the reduction depends on a tie between two
    equal-cost spellings, which the search's listing order breaks.
  - **Speed only** (it still certifies on both seeds, frozen by 1,584 s at the latest): the curriculum's order
    (+277 s and +321 s on seeds 74 and 1); not re-searching solved twins (+364 s, +416 s); BPE merges (+282 s,
    +291 s; an earlier +546 s had counted a search whose time swings 0.5× to 5.5× between runs); the T = 1 filter on
    the round's answers (+82 steps on seed 74, none on seed 1).
  - **Not needed:** the search's routines-first order; the late table lessons, the wrap-around fact (`sub.wrap`) among
    them; the step problem alone; the multiply's row problem (one seed); a round cost limit above 5 (at 4 there is no
    round).
  - **How firm:** every failed run of the seven needed parts searched its whole cost-5 reach without the missing
    piece, so more training time would not rescue it (ABLATION_S3.md §4). A wider search, every program up to cost
    6, found nothing for any of the six parts it applies to. Occam's tie-break was not a search's reach but the weak
    check below.
  - **Two weak checks in s3i, and a fix**
    ([ABLATION_S3.md §8](integration/ABLATION_S3.md#8-fixing-the-two-weak-checks)).
    - The reduction's check can accept a repeated subtraction that is right only for small quotients. It slips
      through when none of the check's 60 three-digit probes has a quotient above 600: about 3% of runs on H1's rows
      ([`reduce_check.py`](integration/ablation_s3/reduce_check.py)). Such a run cannot certify. None of s3i's 8
      certified seeds drew it; the step problem's arm did, on seed 74.
    - The conditional subtraction's check never tests x between N and 2N, so "x if x < N, else 0" passes it. Occam's
      tie-break is what kept s3i off that program.
    - The fix only adds the missing probes: full-size ones for the reduction, built by doubling with the learner's
      own addition, and ones between N and 2N for the conditional subtraction. It is in new copies of s3i and sp2,
      [`integration/ablation_s3/fix/`](integration/ablation_s3/fix/). Both copies certify on the official manifest
      and on their originals' seeds, 17 runs in all, and every record equals the original's field for field. The
      reduction check's 3% falls to about 10^-11. With the fix, s3i needs neither Occam's tie-break nor the step
      problem: both arms certify.
    - The fix is in copies only. `lib/libcore.py` and `self_practice/spcore.py`, which s3i and sp2 are built from,
      keep the old checks. Eight more certified files share the two checks: the strict builds, the learned loop, the
      bridge, sp_breadth's g3, the law ablation's two books and generic_ops' g3. Copies of them with the same fix
      certify wherever the originals did, over 78 runs, and 75 of those runs equal the original's. In one of the
      other three, book_gen's seed 1, the fixed check rejected a wrong conditional subtraction that the old one had
      let into the corpus BPE merges from; its routines, round and every evaluator field are the same. The other two
      ended their hour at a different step, on hosts of a different speed.
- **Stage-3 self-practice** ([SELF_PRACTICE.md](integration/SELF_PRACTICE.md), code and records in
  [`integration/self_practice/`](integration/self_practice/)). The nine hand-picked practice problems, their order,
  the twins and mac's posing are replaced by a book of 11 laws on 8 patterns of the data's numbers. Every law is posed
  in every binding to the registers, and one general rule picks what to practise. The problems the ablation found
  needed come out of that book; nothing marks them.
  - `submission_stage3_sp.py` certifies on the official H100 manifest (round by 746 s, frozen by 785 s) and on 7 of 8
    B200 seeds.
  - `submission_stage3_sp2.py` certifies on the official manifest (round by 1,029 s) and on all 8 seeds.
  - **Still designed:** the law book, the choosing rule (picked after bar 3's own rules failed, by watching whether
    the round came), the instruction set, the search sizes, where the round is looked for, and the rest of stage 3
    (the reader, the cells, the countdown, the tournament).
  - The other formulas are still not learned.
- **The shader follow-up** ([NCA.md](nca/NCA.md) §9, code and records in [`nca/`](nca/)). The NCA probe is one
  identical tiny cell repeated over a grid, each wired only to its neighbours, trained from random weights on lessons
  alone. Two known cures for memorization were tried on it, far past the point where it fits at (3, 2, 1) (at
  (3, 3, 2) no run fits even the soft path): long training with weight decay ("trimming", the setting in which
  networks that had memorized modular arithmetic later found the rule), and agreement (a weight moves only where
  groups of questions, or the grid's columns, push it the same way; or it speeds up while its gradient keeps its
  sign). 72 runs from random weights, 4 seeds per arm (50,000 steps at (3, 2, 1), 20,000 at (3, 3, 2)), and 20
  started at the compiled rule.
  - **Result:** none certifies, none fits the training questions on snapped digits (the best any run reaches is 62
    of the 99 at (3, 2, 1)), and one size up every run stays at chance from start to end: no late jump.
  - **Why:** decay holds the weights small, small weights pass blurred digits, and a blurred code fits the training
    answers on the soft path, unsteadily, without being the machine. The exact rule needs larger weights than decay
    allows. Started at the compiled rule, every arm loses it within a few hundred steps.
  - **Optimizers that can rest at the rule** (§10, 104 runs). Adam moves every weight by about half its learning rate
    even where every lesson is met, so §10 tried Adam with eps 1e-4 and plain SGD. Started at the compiled rule with
    every lesson met, both keep it and certify, 4 of 4 seeds each, and every arm with weight decay leaves it. Where
    the lessons still pull, only plain SGD at a small learning rate keeps the training answers, and it drifts one size
    up, 99.7–99.9% of its errors in situations the training grid never shows. From random, every arm still
    memorizes: one size up no run gets beyond chance. 13 runs from random, cut off by Modal's spend limit, were
    finished from their checkpoints once it was raised: the verdict held, and only numbers read at a run's end moved.
- **The loop over T, learned** ([LEARNED_LOOP.md](integration/learned_loop/LEARNED_LOOP.md), code and records in
  [`integration/learned_loop/`](integration/learned_loop/)). sp2's hand-written countdown, its gate, and the round
  search's assumption that the answer at T = t is the round run t times are removed. The machine gets one generic
  count loop, `C.r[body]` (run the body as many times as register r holds), and T goes into a register as x and N
  do. Once the round is found, a second search finds `C.S[T.round1 STA.X]`: run the round T times, feeding each answer
  back as x.
  - `submission_stage3_ll.py` (md5 `419c15e70ccf6483ff68b04b08f52bdc`) certifies on the official H100 manifest
    (frozen by 1,002 s) and on 7 of 8 B200 seeds. Seed 2 runs out of time in sp2's unchanged climb before any loop is
    searched; with the budget doubled (a research run) it certifies. The slow path comes from the random start, not
    the loop: a diagnostic copy that starts from sp2's own random draws replays sp2's climb line for line on seeds 74
    and 2, and the learned loop then certifies both, seed 2 frozen by 1,062 s.
  - **Still designed:** the count loop itself (a strict judge can call it the countdown made into an instruction),
    T's place in the machine and its 2-digit field, the answer program's search space, and what sp2 designs.
- **Which laws self-practice needs** ([SP_ABLATION.md](integration/sp_ablation/SP_ABLATION.md), code and records in
  [`integration/sp_ablation/`](integration/sp_ablation/)). sp2's law book and choosing rule taken apart one piece at
  a time: 30 arms on the official manifest, each on seeds 74 and 1.
  - **Needed:** the multiply-add law and its pattern (without the accumulating row there is no multiply); compare,
    remainder, add and the pair pattern, which every other problem is built or checked with; the round's two
    routines, the multiply and the reduction; and, in the choosing rule, rating a law on a pattern by its cheapest
    placement. Two more parts of the rule, hardest concepts first and every placement tied for cleanest, failed on
    seed 74 and certified on seed 1: they make the result reliable rather than possible.
  - **Speed only on seed 74, needed on seed 1:** the x + dN pattern (+441 s on seed 74; on seed 1 the reduction never
    comes, although the program seed 74 found is right there too), and choosing at all (practising everything
    certifies on seed 74 at the edge of the budget, frozen by 3,249 s, and runs out of time a level short on seed 1).
  - **Not needed:** six of the 11 laws, three of the 8 patterns, and preferring the cleanest placements over the
    cheapest. Seed 1 agrees on every one.
  - **A smaller book**, 5 laws on 4 patterns ([`variants/book_min.py`](integration/sp_ablation/variants/book_min.py),
    md5 `e4732f09e8e5b46a8cfa5a1236771bd2`), certifies on the official manifest sooner than sp2 (the round by 571 s
    against 977 s) and on 8 of 8 B200 seeds. It leaves out x + dN too, and on seed 1 writes the chain inline, which
    sp2's book without x + dN did not: what a removal costs depends on what else is in the book.
  - **A generated book** ([`variants/book_gen.py`](integration/sp_ablation/variants/book_gen.py), md5
    `8bbac4a1c631a08c8b72179de37fcb26`): a grammar instead of a list (numbers r + q·b from the division law, every
    composition of + and ×, one set of target rules). It holds sp2's book and 24 concepts more, and certifies on the
    official manifest (frozen by 2,228 s) and on 8 of 8 seeds.
  - **What it says:** the particular list of laws is not what the result rests on. Still designed: the grammar, the
    target rules and the choosing rule. Rerun at three times the budget, the rule without its ties still finds no
    round on seed 74: the climb reaches its 16-level cap with no reduction the round can use.
  - **Found in passing:** sp2's searches can run past the training deadline. One failing arm trained 3,673 s of its
    3,600; every certified run in the study trained inside the budget.
- **Generic instructions in place of SEL.S and BRG.X** ([GENERIC_OPS.md](integration/generic_ops/GENERIC_OPS.md),
  code and records in [`integration/generic_ops/`](integration/generic_ops/)). **Certified.**
  - SEL.S becomes an if block, `C[...]`, whose body's effects stay only where the borrow was set. BRG.X becomes
    shifts that pass the digit they move through the multiply's carry digit. No single instruction does either old
    job, so the learner builds the conditional subtraction, the chain and long division's "bring down" itself.
  - Replacing SEL.S alone needs no other change. Replacing BRG.X too makes long division one step longer, and it
    took three rules, each stated for every law but found by watching whether the round came: the gate's closeness
    counted without the reduction's fakes, with deep probes rotated; acceptance at four digits; and a preference for
    routines that leave the other registers alone.
  - `submission_stage3_g3.py` (md5 `e6bd2a1f9bfd9fd2119b0d44483c7100`) certifies on the official H100 manifest
    (`submission_stage3_g3_official.json`: training 2,770 s, the round at step 698 by 2,507 s, evaluation 386 s) and
    on 8 of 8 B200 seeds (74 and 1–7, training 2,290–2,790 s). That is 2.2 times sp2's training on the official
    run, with 830 s of the budget to spare. A first official run, stopped by Modal's spend limit before it wrote a
    record, had taken the same steps. Its acceptance at four digits rejects the ablation's 600-subtraction fake,
    but not a loop of 10,000, which costs 5 there, as long division does (GENERIC_OPS.md §2.1). Its fixed copy
    rejects that loop too, and its 9 runs equal g3's (ABLATION_S3.md §8).
  - **Still designed:** the if block and the shifts, generic in kind but chosen, and the rule changes. A judge may say
    `SHL.X SHL.A` is BRG.X in two pieces; GENERIC_OPS.md §7 agrees, and notes it is the form a place-by-place law
    gives BRG.X.
- **Strict builds of s3i and sp2** ([STRICT.md](integration/strict/STRICT.md), code and records in
  [`integration/strict/`](integration/strict/)). The gray points of §5 that can go without changing what is learned
  are removed, by exact edits to the certified sources:
  - No decision reads the clock: each search, and the climb as a whole, gets a budget of machine work, counted by the
    search's own cost model, the same on every host.
  - The rows the climb remembers and the found round program are buffers: in the state dict, counted, on the GPU.
    The forward pass runs the round from there.
  - The optimizer resets itself at phase B instead of the scheduler doing it, and the late table lessons and the
    wrap-around check in the table's gate are gone.
  - Both strict files certify on the official H100 manifest and on every seed their originals ran (s3i: seeds 1–7 on
    H100s; sp2: 74 and 1–7 on B200s), 17 runs. Each run's climb and schedule are the original's line for line.
  - **Unchanged, and why:** zero logits in training (the real answer inside autograd on a training batch does not fit
    in an H100's memory); the damped gradient; the reader; the climb, the instruction set and the curriculum, which
    are the learning method. The strict files keep the old weak checks; their fixed copies certify on all 17 runs,
    each equal to the original's (ABLATION_S3.md §8).
- **Breadth at full size** ([SP_BREADTH.md](integration/sp_breadth/SP_BREADTH.md), code and records in
  [`integration/sp_breadth/`](integration/sp_breadth/)). Does sp2 learn the family's other formulas at H1's size?
  - Unchanged, 2 of 11 tasks: H1 and H1 with the prompt's fields reordered. The other formulas fail in the round's
    search, which composes at most two routine calls and cannot make a constant.
  - Three general changes, all in the round's search, none naming a formula: three calls instead of two; a constant
    read off the answers when every answer differs from a program's result by the same amount mod N; and a round must
    also reproduce the answers of the other one-digit T's.
  - With all three (`submission_spb_g3.py`, md5 `30efec64d6e4f0a7417938e9370a363e`), 8 of 11 are learned, each in one
    run on one H100 with every split exact: H1 (also on 8 of 8 B200 seeds), the reordered prompt, 10y², y² + 1,
    y² + 3, 10y² + 1, y² − 1, and the competition's Medium data (H1's formula, every T = 8). Not learned: y³ (four
    calls), Montgomery's y²·10⁻⁸ (no law poses its reduction), and a prompt with a fourth number (the reader takes
    three).
  - **How general:** more than before, less than bar 3 asks. Each change is stated without a formula, but each was
    written after watching a formula fail, with the family in view.
- **Every certified variant in one file** ([COMBINED.md](integration/combined_sp2/COMBINED.md),
  [FAST.md](integration/combined_fast/FAST.md), code and records in
  [`integration/combined_sp2/`](integration/combined_sp2/) and
  [`integration/combined_fast/`](integration/combined_fast/)). The weak-check fix, the strict build, the learned loop,
  the generated book and the generic instructions, built into one file by exact, asserted edits of the certified files.
  - **Rounds 1 and 2:** the climb runs out of time before the reduction; with its acceptance setup shared and its counts
    built by doubling, it finds the round, but past the hour.
  - **Round 3:** each four-piece file certifies on the official H100, one without the generated book and one without the
    generic instructions. Either piece alone fits; together they are late.
  - **Made fast:** a Triton kernel runs every row's instruction sequence in one launch, and the climb gets 7.7 times
    faster with every search, pick and unit of work unchanged, in the tiny core and line for line on the evaluator.
    `submission_stage3_fast.py` certifies on the official H100 in 751 s of training, and on seeds 1–7 on H100s, 8
    runs of 8.
  - **Budgets with margin:** measured with no budget on the same eight seeds, the climbs show that round 2's budgets
    never cut anything. `submission_stage3_fast2.py` sets them again from those climbs, by STRICT.md §2's rule; its
    climbs are the same line for line, and it certifies 8 runs of 8, training in 759–883 s. It was the file to judge
    until the legal file (below; §2).
  - **Still designed:** what sp2 still designs, less the pieces each variant took out (§5, rule 7); the judge's
    questions of §5 stand. About $90 at list price in all (COMBINED.md §6, §8 and §9; FAST.md §6, §8 and §9).
- **Bar 3's learner, cheaper and deeper** ([SCALE.md](bar3/scale/SCALE.md), code and records in
  [`bar3/scale/`](bar3/scale/)). The same learner on the bare counting machine, made cheaper and then given new kinds
  of practice problems.
  - **Cheaper, with the same choices:** the compiled search alone cut the breadth study from 515 search-hours to 51,
    and 50 of its 54 runs make the same choices in every round; the other 4 differ only where the original had stopped
    a search at its time limit. With searches shared between problems and the cheaper choosing rule, round 0 on the
    ten formulas went from 67.8 search-hours to 3.95.
  - **That buys about one instruction:** each extra instruction multiplies the distinct machine states by 9–10 (by
    about 40 over a carried library), and the missed formulas need 13–22 instructions from the first pieces.
  - **But what became affordable reached further.** Run along a sequence of formulas, each solved one kept as a
    routine, the pairs rule invents 10·x², 10·x² + 1 and x³, each in the first round of its step: 10·x² is "x² mod N
    (carried), then the rest of the way". A new practice source, steps that move two numbers at once, gives Fibonacci
    from scratch. Unchanged, the learner invents x² mod N, x² ± 1, 2x + 1 and Fibonacci at N ≤ 26 (not x² + 3 in
    2 rounds), and x² mod N, 2x + 1 and Fibonacci at N ≤ 50, each by the algorithm it found at N ≤ 7. Hardest
    first's own sequence, finished after the spend-limit cut, misses 10·x² + 1 but then invents x³, Montgomery's
    (round 3) and Fibonacci (round 1).
  - **Not reached:** Montgomery's and Fibonacci in the anchored-pairs sequences (5 rounds; with the new sources their
    Fibonacci programs are right up to N = 40 but take more steps than the general check allows, and with the breadth
    study's sources nothing poses the step), and 10·x², 10·x² + 1 and x³ from scratch, until the subgoal study
    (x³ and 10·x²) and the picks study (10·x² + 1) below.
  - **Bounds on the search space:** fewer instances with counterexamples, merging symmetric registers, dead
    registers, meet in the middle and a lower value cap each save at most 1.7×, against the 9–10× each instruction
    costs. The runs Modal's spend limit cut off that had stores were finished from them, each the same as before in
    every round they share (184 search-hours, $21.92); the depth-2 law book's runs, with no store, were not rerun.
- **A signal that sees a piece before it is solved** ([SUBGOAL.md](bar3/scale/subgoal/SUBGOAL.md), code and records
  in [`bar3/scale/subgoal/`](bar3/scale/subgoal/)). The learner scores each problem it poses by whether the target
  factors through it, from the rows alone: the target is a function of the problem's answer and N, or one law of the
  book away from it. It aims one deeper search (10M states) at the problem the score singles out.
  - **From scratch at N ≤ 7:** x³ in round 1 under both choosing rules, and 10·x² in round 1 under hardest first.
    The same learner without the score gets neither in 5 rounds. Every program found is right on every pair up to
    N = 120.
  - **Nothing regressed:** x² mod N, x² ± 1, x² + 3, 2x + 1 and Fibonacci still come, and hardest first's x² + 3,
    lost with the bigger law book this study uses, came back.
  - **Not reached:** 10·x² + 1, whose rest after x² mod N is a token beyond the deeper search, and Montgomery's,
    whose 16 instances at N ≤ 7 are too few to pin down halving mod N. A second level, the subgoal's own subgoal,
    tried small, would not reach 10·x² + 1 either: no source poses its piece, 10X mod N (SUBGOAL.md §8).
  - **What did the work:** the deeper search, aimed by the score (1.6 of 47 search-hours). Breaking ties by the
    score changed some picks but no outcome. About $12.
- **Tested on formulas none of its rules were written for** ([HELDOUT.md](bar3/scale/heldout/HELDOUT.md), code and
  records in [`bar3/scale/heldout/`](bar3/scale/heldout/)). The learner was built and tuned on a handful of formulas
  (HELDOUT.md). So, frozen with the subgoal, it was run from scratch on ten new formulas under both
  choosing rules, with a prediction for each written down before any run (`PREREG.md`, its hash recorded).
  - **What came:** 5 of 10, in 9 of 20 runs, all in round 0 or 1: (x² + x), (x + 1)², 3·x² and x⁴ mod N under both
    rules, and 2^x mod N under anchored pairs. Every program is right on every pair up to N = 120.
  - **The predictions:** 16 of 20 held, including all 8 made with medium or high confidence.
  - **The subgoal's part:** a control without it, added after the test, loses 3·x² under hardest first and x⁴ under
    anchored pairs, and gets x⁴ under hardest first two rounds later.
  - **Not reached:** (x³ + x), Lucas, Tribonacci, floor(x²/N) and the triangular numbers. Two need a step no source
    poses: the running sum's y(x + 1) from y(x) and x, and a step on three numbers. Three need a program longer than the
    search reaches. About $11.
- **x as digits** ([DIGITS.md](bar3/scale/digits/DIGITS.md), code and records in
  [`bar3/scale/digits/`](bar3/scale/digits/)). A loop over x's units cannot reach x of 6–8 digits. So the frozen learner
  was given a loop over x's digits in base B and a source that asks whether the answer for x·B + d is a function of the
  answer for x, the digit d and N, and if so poses that step. A program counts only if it also passes a check at big x.
  36 runs, with the predictions written down first (`PREREG.md`, its hash recorded).
  - **What came:** x mod N in base 2 with the source: round 1 under anchored pairs, round 3 under hardest first. Both
    programs loop over x's bits with long division's base-2 step as the body, and are right on 1,000 x below 10⁹.
  - **The predictions:** all 5 general ones held, and 35 of 36 runs came out as predicted.
  - **Not reached:** anything in unary, whose programs all count and run out of steps; base 10's step; 2^x mod N, whose
    square-and-multiply step was never learnable; and x² mod N. The subgoal found its hidden state, x mod N on long x,
    but the alias check counted that a twin of the law book's x mod N, and where the twin was practised the pick rule's
    score saw no gain over counting programs. About $51.
- **Two sources from Newton's differences** ([NEWTON.md](bar3/scale/newton/NEWTON.md), code and records in
  [`bar3/scale/newton/`](bar3/scale/newton/)). The held-out test and the subgoal study each named a piece no source
  poses. The frozen learner was given two sources, behind flags: a running sum's step, y(x + 1) from y(x) and x, and the
  offset, each rest of the way less its value at X = 0.
  - **What came:** the triangular numbers in round 1 under anchored pairs; that result was seen in a replay before the
    predictions were written. Under hardest first the step loses to pool order.
  - **Not reached:** 10·x² + 1. The offset poses its piece, 10X mod N, and the learner solves it, but the duplicate
    check, run on random machine states on which both programs die, calls it a copy of 3X.
  - The nine earlier held-out winning runs stand. Run locally from the earlier stores, $0.
- **The duplicate check, on each problem's own rows** ([ROWCHECK.md](bar3/scale/rowcheck/ROWCHECK.md), code and records
  in [`bar3/scale/rowcheck/`](bar3/scale/rowcheck/)). The learner had thrown 10·x² + 1's piece, 10X mod N, away as a
  copy of 3X: its duplicate check compares programs on 24 random machine states, and both run out of steps on all of
  them. The check now counts a problem as a copy only when the other program is right on the problem's own rows.
  - **What changed:** 10X mod N is learnable in rounds 2 and 3; x² mod N's twin stays a copy, rightly, since one program
    solves both.
  - **What came:** nothing new. Hardest first never picks the piece, since 24 and then 12 problems of the same cost
    stand ahead of it in pool order. All 11 earlier winning runs stand (the nine held-out ones, and binary x mod N
    under both rules).
  - **The predictions:** 6 of 7 held. No outcome would change, and none did; the one that failed said some picks would
    change. $0, run locally from the earlier stores.
- **What the learner picks** ([PICKS.md](bar3/scale/picks/PICKS.md), code and records in
  [`bar3/scale/picks/`](bar3/scale/picks/)). Three changes to how the frozen learner picks its practice, each behind its
  own flag, with predictions written down first.
  - **What came:** with the row check on, breaking hardest first's ties by the subgoal score picks 10X mod N in round 2,
    and 10·x² + 1 comes from scratch, for the first time in any study. Its program is right on every pair to N = 120
    given 10⁹ steps; on the standard machine for N = 120, 2,199 of the 7,260 pairs run out of steps.
  - **What didn't:** passing a subgoal's priority to its twin does not bring x² mod N in base 2, and it loses hardest
    first's binary x mod N. A pick score that sees the big-x check cannot act: no look reaches the five-token program
    that passes at big x.
  - **The earlier wins:** all 11 winning runs stand under the tie-break. About $5.
- **How far its looks reach** ([REACH.md](bar3/scale/reach/REACH.md), code and records in
  [`bar3/scale/reach/`](bar3/scale/reach/)). Two more changes aimed at x² mod N at big x.
  - **What came:** nothing new. A deeper look finds the five-token program that passes at big x, but anchored pairs
    picks from pairs whose looks stay shallow. The twin as a fifth pick keeps hardest first's binary x mod N, but the
    square is never practised in time.
  - **The earlier wins:** all 12 winning runs stand: those 11, and 10·x² + 1's. About $7.
- **Composition on stage 3's machine** ([COMPOSE.md](bar3/scale/compose/COMPOSE.md), code and records in
  [`bar3/scale/compose/`](bar3/scale/compose/)). Stage 3's tiny core with its law book removed and bar 3's sources
  posing the practice: can they replace the 5-law book?
  - **What came:** only the 5-law book's control gets the round. Bar 3's sources pose no piece in any of 5 arms.
  - **Why:** the digit machine's registers start dirty, so near misses' values are not functions of x and N; the
    multiply's route is never posed at this width; and the climb's searches accept no reduction even when handed that
    problem. About $11.
- **Composition, round 2** ([COMPOSE2.md](bar3/scale/compose2/COMPOSE2.md), code and records in
  [`bar3/scale/compose2/`](bar3/scale/compose2/)). Bar 3's remaining pieces ported, and the fixes round 1 named.
  - **What came:** with book_min less its reduction, bar 3's set gets the round at level 5, right on every rung to
    T = 64. It needs the subgoal score's law view, a deeper probe aimed at the subgoal's rest, and a search that tests
    more than one spelling per state.
  - **What didn't:** bar 3's set alone, since the multiply is out of reach without the multiply-add row. About $30.
- **Composition, round 3** ([COMPOSE3.md](bar3/scale/compose3/COMPOSE3.md), code and records in
  [`bar3/scale/compose3/`](bar3/scale/compose3/)). Without the book's multiply-add row, bar 3's set never gets the
  multiply; and round 2's winning setup fails on x³ and 10·x² + 1 mod N, so its route is shaped to x². About $8.
- **The search on a GPU** ([GPU_SEARCH.md](bar3/gpu_search/GPU_SEARCH.md), code and records in
  [`bar3/gpu_search/`](bar3/gpu_search/)). Identical results on 18,708 stored searches and two whole runs. Its big
  searches run about 95 times faster on an L40S or an H100 but only 3–10 times cheaper, and a whole learner run is
  1.6–1.9 times cheaper: about half to one more instruction of depth for the same money. About $8.
- **Consolidation** ([CONSOLIDATE.md](bar3/scale/consolidate/CONSOLIDATE.md), code and records in
  [`bar3/scale/consolidate/`](bar3/scale/consolidate/)). Its counts are formulas: the 16 won so far, each run once
  from scratch. One score for every choice keeps 1 of the 16; two general rules, one uniform search depth for every
  candidate and anchored pairs' picks, bring back 13 of 16, and most special rules were needed for no win. About $50.
- **The target's laws cut to one digit place** ([SLICE.md](bar3/scale/slice/SLICE.md), code and records in
  [`bar3/scale/slice/`](bar3/scale/slice/)). A source reads the target's law and poses each operation cut at one place.
  On stage 3's machine with the book less its reduction and its multiply-add row, x² mod N's round comes at level 5,
  right to T = 64: bar 3's pieces supply two of the book's laws. x³ and 10·x² + 1 get no round, and without the book's
  multiply the multiply never comes. About $25.
- **Each cut's loop, offline** ([LOOPS.md](bar3/scale/loops/LOOPS.md)). A source poses each cut's loop: the multiply
  from the row, the whole reduction from Horner's step, with practice answers from Python. x² mod N's round then comes
  at level 2 with the book down to the comparison alone, right on both ladders, and with every one of round 2's search
  fixes off. With no book at all the multiply still comes, but nothing poses the comparison. x³ and 10·x² + 1 get the
  whole reduction but no round: the cube needs four calls where the round's search reaches two, and 10·x² + 1 has no
  constant 1. About $93.
- **Bar 3's studies on the GPU** ([GPU_SEARCH.md](bar3/gpu_search/GPU_SEARCH.md) §8–§13). One runner runs a bar-3 study
  on the GPU or the CPU; seven whole runs reproduce record for record, in minutes instead of hours. About $17.
- **Bar 3's sources inside a training run** ([LEGAL_SOURCES.md](bar3/LEGAL_SOURCES.md)). Most of bar 3's 14 practice
  sources read a target table that a training run does not have; the audit says which are legal as they are, which need
  a redesign, and which stay offline. It adds a question for a judge: may Python arithmetic choose what to practise if
  it computes no target? No runs.
- **A legal law test at full size** ([LEGAL_TEST.md](bar3/scale/legal/LEGAL_TEST.md)). The audit's cheapest decisive
  test, offline on one B200 from slice round 4's level-5 library. With H1's own T = 1 rows as the target and a law test
  that runs only the learner's own routines, x² is picked alone (1 of 41,007 values), and the rest of the way finds the
  reduction with the regular probe. The pre-registered round took the first of four accepted spellings, a look-alike
  that fails on some unseen 24-bit moduli (757–764 of 768), which no training row can catch; the climb's own rule of
  fewest steps, and a legal identity check, both keep a correct one. A legal climb from scratch is the next test. About
  $3.
- **A fully legal climb** ([LEGALCLIMB.md](bar3/scale/legalclimb/LEGALCLIMB.md)). The same full-size mode inside the
  climb's level loop, so that every practice answer is a data row, a fact, an identity or the learner's own routine, and
  every law test runs the learner's routines. With book_min's reduction cut to one step, bar 3's sources bring H1's
  round at level 5 in seven arms, seeds 1–3 among them, each right on both ladders, unseen moduli included. With no
  reduction at all it comes only when the law test accepts 3 exact rows (one run). Without bar 3's sources, or without
  the book's multiply-add row, nothing comes: the comparison, the link, the addition, the row and the multiply have no
  legal source but the book. So legally, bar 3 supplies the reduction; offline it also supplies the row and the
  multiply. About $20.
- **Legally, three laws** ([LEGALLOOPS.md](bar3/scale/legalloops/LEGALLOOPS.md)). A legal reader tests candidate laws on
  H1's answer rows with the learner's own routines (a product by repeated addition, a reduction by the link repeated)
  and reads x² mod N as mod(mul(X, X), N). It then poses the multiply-add row and Horner's step, and the multiply and
  the reduction as their loops, each answered by the book's own constructions. With the book cut to three laws, the
  comparison, the link and the addition, H1's round comes at level 4 in every arm with the reader, a level sooner than
  with book_min, and every round is right on both ladders, unseen moduli included. Still designed: the reader's grammar
  and its table of what each operation cuts to and loops over. About $7.
- **The legal file** ([LEGAL_FILE.md](integration/combined_legal/LEGAL_FILE.md), code and records in
  [`integration/combined_legal/`](integration/combined_legal/)). fast2 with its generated law book replaced by those
  three laws and that reader, built by 18 asserted edits of fast2's bytes. Every run reads mod(mul(x, x), N) at level
  3 and finds its round at level 4. `submission_stage3_legal.py` certifies on the official H100 and on seeds 1–7 on
  H100s, 8 runs of 8, training in 465–721 s against fast2's 759–883 s. It was the file to judge until the family file
  (§2). Still designed: the three laws, the reader's grammar, the posing rule, and what fast2 designs. About $10.
- **No book, offline** ([READING.md](bar3/scale/reading/READING.md)). Round 6 poses the comparison by cutting Horner's
  step down to the link and the comparison, with practice answers from Python. x² mod N's round then comes at level 2
  with no law in the book at all, right on both ladders: the comparison at level 0, the step, the link and the multiply
  at level 1, and the reduction from its loop at level 2. It also comes with the comparison alone and with every one of
  round 2's search fixes off, but not with the link alone, which is 6 tokens over the primitives. Posed as a sub-law,
  x·o mod N comes at level 3 and the cube's round follows as two calls of it, with the book and without. 10·x² + 1
  learns its constant 1 and x + 1 but gets no round even in a 7,200 s run, and anchored pairs' prior brings the row to
  level 0 without mending their picks. Still hand-written: the source's tables of cuts, loops and the step's cuts, and
  which sub-law to pose. About $72.
- **The wider grammar** ([WIDE.md](integration/legal_wide/WIDE.md)). The legal file's reader with subtract, power,
  constants read as gaps and depth 3 beside add, mul and mod: 200 candidate laws instead of 35, tried in about half a
  second. It still reads mod(mul(x, x), N) on every answer row at level 3 and finds the round at level 4.
  `submission_stage3_wide.py` (`3bb5ceeb`) certifies on 8 of 8 official-manifest H100 runs, training in 478–658 s. About
  $9.
- **No law in the book** ([NOBOOK.md](integration/legal_nobook/NOBOOK.md)). The wide file with its three laws removed.
  Before any routine exists, one designed rule reads the answers: every answer row is below N, compared digit by digit
  with no number made, so the law ends in a reduction by N, and the reduction's step is posed cut to its parts, the
  comparison, the link and the addition, each answered by facts, identities and the learner's own routines. The levels
  are then the legal file's. `submission_stage3_nobook.py` (`b42ec304`) certifies on 8 of 8 official-manifest H100 runs,
  training in 472–634 s. Still designed: that rule, which is family knowledge since every family task ends in mod N, the
  step's cut table, the reader's grammar, the posing rule and what fast2 designs (NOBOOK.md §7). About $10.
- **The family with no book, offline** ([FAMILY.md](bar3/scale/family/FAMILY.md)). Round 7 replaces round 6's
  hand-written sub-law with a rule that names no task: every window of the target's reading, two or three operations,
  becomes a law of its own, smallest first. With no book and nothing tuned per task, 6 of 7 family tasks come on stage
  3's tiny core, with practice answers from Python: x² at level 2, the cube at 3, 10y² at 2, y² + 1 at 3, y² + 3 at 4
  and 10y² + 1 at 3, each right on both ladders and at x = 0. y² − 1 comes with two more general changes: posing the 1
  implied by the reader's pred and succ, and keeping a round only if it also matches the data's own answers at T = 2, 4
  and 8, without which the first round kept is wrong whenever an input is 0. The round search never needed more than two
  calls. The changes were found on these same tasks, so this is not a held-out test. About $85.
- **Round 7's rules, legally** ([FAMILY_FILE.md](integration/legal_family/FAMILY_FILE.md), code and records in
  [`integration/legal_family/`](integration/legal_family/)). The no-book file with round 7's general pieces, ported by
  asserted edits, the new laws' answers made by the learner's own routines through the reader: every window of two or
  three of the reading's operations posed as a law of its own, each constant of the reading made from N with x + c,
  x − c and x·c, and a round kept only if it also gives the kept answers of every count T > 1, run T times (ROUNDT).
  `submission_stage3_family.py` (`457216d7`) certifies on 8 of 8 official-manifest H100 runs, training in 562–986 s,
  slower than the no-book file's 472–634 s because of ROUNDT, and is now the file to judge (§2). The same file, byte for
  byte, one H100 each, on SP_BREADTH's data for six tasks and new data for two held-out ones, learns 4 of 8 family tasks
  at full size: y² + y (held out), y² + 1, the cube and y² − 1, each right to T = 64 on both ladders with every split
  exact. It does not learn y² + 3, which never makes its 3 within the climb's work budget, or 10y², 10y² + 1 and 3y² + 2
  (held out), which the reader cannot read: 10 is a factor, not a gap, and the other two need depth 4. Only the two
  held-out tasks, chosen before any run, test generality: round 7's rules were found on the other six. Still designed:
  the window rule, the constants rule, ROUNDT's rows and the reading's own patterns, beside the no-book file's list
  (FAMILY_FILE.md §9). About $40.
- **Bar 3's learner invents stage 3's digit instructions** ([BRIDGE.md](bar3/bridge/BRIDGE.md), code and records in
  [`bar3/bridge/`](bar3/bridge/)). A digit instruction is one place of a law: the operands' digits and what the place
  below passed up go in, a digit and what to pass on come out. Cut to one place, the self-practice law book gives 18
  such slices, and 11 of them compose back into their law when looped over the places.
  - Bar 3's learner, unchanged, on a counting machine with no digits, carries or wrap-around, invented the select (what
    SEL.S does at one place) and subtract-with-borrow, and, in the round that Modal's spend limit had cut off,
    add-with-carry: it adds, compares with the base and subtracts it once ([CARRY.md](bar3/bridge/CARRY.md)).
    Multiply-accumulate, whose carry must count, came once two new pose rules (the conjunctions and the widening) posed
    its piece and a deeper search was aimed at it as in the subgoal study: a score read off the rows names the piece
    the target loops over, and in that search a clean program beats a cheaper one that uses up N. Hardest first then
    invents it in round 1; the pairs signal does not in 2 rounds (CARRY.md §12). Each invented program, compiled to a
    base-10 table, is exactly stage 3's cell.
  - `submission_stage3_bridge.py` (md5 `85ff74c5faf902c1ede7a9f07d059026`) is s3i with 36 instructions: subtract, add
    and multiply-accumulate at the pointer against each register, their tables learned as s3i learns them, and the
    invented select in place of SEL.S. It certifies on the official H100 manifest (training 1,839 s, against s3i's
    1,340 s) and on 8 B200 seeds (74 and 1–7). The ablation found SEL.S needed; here the learner's own select takes
    its place.
  - With all 11 slices (61 instructions) there is no round. At s3i's budgets the search runs out of states; at 4
    times them, wrong programs pass the two weak checks. The run that would confirm the second explanation was
    refused by Modal's spend limit.
  - **A reduction with nothing estimated** (CARRY.md §13; local, nothing built): Montgomery's quotient digit is a table
    of one place, t's lowest digit and N's last (40 rows at base 10), where long division's has to be estimated and
    corrected. Run from that table and the invented multiply-accumulate, subtract and select alone, Montgomery's
    reduction is right on 300 of 300 random cases at every base from 2 to 16.
  - **Still designed:** the law book, the frame (moves, the pointer, shifts, BRG.X, the loops) and which slices become
    instructions. The instruction list is made before training; the run does not invent its cells.

## 8. About this checkout

[HANDOVER.md](HANDOVER.md) has the hygiene pass: what is tracked and what is on disk only, the scan for secrets and
private data (no credentials anywhere; the author's home path and Modal workspace name appear in run logs), the link
check, the checksums, and the files proposed for removal. It was last brought up to date on 2026-10-26.
