# One Layer Deeper H1, from the CPU up: where everything stands

**Reviewing or judging the project from outside? Start with [REVIEW.md](REVIEW.md)**: the claims, the file to judge,
how to check its records, and the rules to rule on. It points back into this file.

*The entry point for `cpu_first/`. First written 2026-10-21 at 03:45 Pacific and refreshed as studies landed. The
refresh of 2026-10-24 at about 08:45 added the four studies that Modal's spend limit cut short: optimizers that can rest
at the shader grid's rule, which laws self-practice needs, a cheaper and deeper bar-3 learner, and generic instructions
in place of SEL.S and BRG.X. This one, at about 17:30 the same day, adds what the reruns after the limit was raised have
brought so far: that last file certified (§4.8), the law ablation's seed-1 arms (§4, bar 2) and its rerun at three times
the budget, and the NCA's 13 cut-off runs, finished (§4.5); bar 3's replays are still going (§7). A refresh on Oct 25 at
about 03:30 adds the weak-check fix in copies of eight more certified files (§4.7), bar 3's cut runs, finished, and a
signal that brings x³ and 10·x² from scratch, and add-with-carry invented at one digit place (§4, bar 3). The last, on
Oct 25 at about 10:30, adds the held-out test of the frozen bar-3 learner, which invents 5 of 10 formulas none of its
rules were written for, and multiply-accumulate invented at one digit place, so that all four of stage 3's digit cells
are now invented (§4, bar 3). At about 13:00 the same day it adds a pointer to a plan from bar 3 to a contest-legal
entry, written without runs (§7), and at about 17:50 the plan's first test: one file with every certified variant of
sp2, which does not certify (§4.8). At about 19:00 it adds two bar-3 studies: given x as digits, the learner invents x
mod N in base 2 by long division's step, right up to nine digits; and a running sum's step, posed as practice, brings
the triangular numbers (§4, bar 3). Later that night, a second round of the combined file reaches the round, but past
the hour (§4.8), and a duplicate check on each problem's own rows frees 10·x² + 1's piece but changes no outcome (§4,
bar 3). Late that night both four-piece files certify, and the five-piece file, made fast without changing anything it
learns, certifies in 751 s of training (§4.8). Early on Oct 26 its rebuild with work budgets set from eight climbs,
fast2, certifies 8 of 8 and becomes the file to judge ([REVIEW.md](REVIEW.md) §2). Then bar 3's tie-break by the subgoal
score brings 10·x² + 1 under hardest first for the first time, and a first composition test finds that bar 3's sources
do not yet replace stage 3's law book on its digit machine (§4, bar 3). Its second round replaces the book's last law,
the reduction, with bar 3's own pieces (§4, bar 3). On Oct 27 a file built from fast2 with the book cut to three laws
and bar 3's legal reader certifies 8 of 8 (§4.8). Later on Oct 27 the reader gets a wider grammar and the book is
emptied, each certifying 8 of 8, and with round 7's general rules the file with no book certifies 8 of 8 again and
learns 4 of 8 family tasks at full size, one of two held-out tasks among them. That evening a tidier, wider version,
with no data rows outside the state dict, certifies 8 of 8 and learns 7 of 8 family tasks but neither of two new
held-out ones. Late that night a reader of four steps certifies 8 of 8 again and learns all 22 other tasks it is run on,
among them six drawn at random with the file's own md5 as the seed; it is now the file to judge (§4.8).*
- *What was checked against the run records (`history/runs/`, `history/ablation/runs/`, `anchors/records/`,
  `history/bridge/h1/records/`, `history/tournament/h1_records/`, `general_lessons/records/`,
  `general_lessons/fixed/records/`, `integration/records/`, `integration/ablation_s3/records/`,
  `integration/self_practice/records/runner/`, `history/handwired_bridge/h1/records/`, `litmus/records/`,
  `library/records/`): every certification, run time, freeze step, parameter count and seed count quoted here. The
  source embedded in each record was hashed and compared with the file's md5.*
- *For the NCA follow-up (§4.5), the run, seed and step counts, the certifications, the best training and
  one-size-up shares and the weight norms quoted were checked against `history/nca/runs/trim1.jsonl`, `trim_cols.jsonl`,
  `trim_stab.jsonl` and `history/nca/records/sharpness.json`.*
- *For the ablation's second batch (§4, §4.7), each run's verdict, seed and budget was read from its record (35
  runs), and the rounds, freezes and speed costs come from rerunning ABLATION_S3.md's `analyze.py` on them; BPE's
  host correction on seed 74 was redone by hand. The two controls were compared with s3i's own records by the shared
  runner's `--mode compare`. The weak check's 3%, and the example in §4.7, were rerun locally with
  `reduce_check.py`'s code.*
- *For the five studies of the morning of Oct 24 (§4.7, §4.8), every verdict, step and time quoted was read from
  their 125 records, and their seven new official records pass `tools/verify_records.py`. Each study's own compare
  script was rerun: the 17 fixed runs equal their originals field for field, the 17 strict runs' climbs equal theirs
  line for line, and the learned loop's diagnostic copy equals sp2 up to the round on both seeds it ran.*
- *For the four studies written up after the stop: NCA §10's certifications and one-size-up shares were read from
  `history/nca/runs/rest_*.jsonl` (91 runs that finished) and `history/nca/records/rest_cut.json` (13 cut off); once
  those 13 were finished, each one's curve was checked equal to `rest_cut.json` up to its stop, and `nca.fill`
  regenerates NCA.md's tables unchanged from the 104 records; the law ablation's
  verdicts and times from its 61 records, the two books' official ones through `tools/verify_records.py`; and the
  bar-3 scale study's same choices and search-hours were rerun with its own `compare.py`. The law ablation's 17 seed-1
  records added after the limit was raised: its results table, regenerated by `analyze.py --both`, equals
  SP_ABLATION.md's cell for cell, and each record's md5 is its arm's.*
- *For the generic-instructions file (§4.8): its nine records, written after the limit was raised, pass
  `tools/verify_records.py`, and each run's round and freeze, step and clock, were reread from its log, as were
  sp2's round times and training times on the same seeds.*
- *For the weak-check fix in eight more certified files (§4.7): all 8 official records pass
  `tools/verify_records.py` against their fixed copies, and so do 62 of the 63 seed records (the learned loop's seed 2
  fails, as its original does); `check_fix_variants.py` and `compare_fix_variants.py` were rerun, and 75 of the 78
  runs equal their originals'.*
- *For bar 3's finished cut runs and the subgoal study (§4, bar 3): `compare.py --old` was rerun on each replayed
  record against its version from before the cut, and on the subgoal study's eight controls against the cut depth-2
  runs; every round they share is the same. Every program the finished sequences and the subgoal study's 40 records
  found was rechecked on every pair up to N = 120 (Montgomery's: the 3,600 odd-N pairs), and the score's ranking test
  reproduces its record. The subgoal learner's copy check needs numba and was not rerun.*
- *For the carry study (§4, bar 3): its tiny check, its run table and its table check (each invented add against the
  variant's ADC table at bases 2–16) were rerun, and round 0 of three of BRIDGE.md's runs was replayed from the search
  cache with the new flags off, field for field.*
- *For the held-out test (§4, bar 3): the pre-registration's hash and file time (05:12 PT, before the early check at
  05:14 and the launch at 05:19), the frozen learner's six file hashes against `97f30b9`, and every program of its 20
  runs and 8 controls, rechecked up to N = 120 by rerunning `verify.py`, with output identical to its logs. The replays
  on the search stores (`attribute.py`) need the stores from Modal and were not rerun.*
- *For multiply-accumulate and Montgomery's quotient digit (§4, bar 3): every program the study invented was compiled to
  its table and looped over the places again (`lift.py`): multiply-accumulate equals the variant's MUL cell on 10,000 of
  10,000 rows at base 10, and `carry_tables.py` finds every row equal at bases 2–16. `montgomery.py` and the score's
  tiny check were rerun with identical output, and round 0 of all twelve BRIDGE.md runs was replayed from the search
  cache with every flag off, field for field.*
- *For the digits and two-source studies (§4, bar 3): each pre-registration's hash and file time against its runs (the
  digits study's at 09:40 PT, before its launches at 10:00–10:01; the two sources' at 13:28 PT, before its four runs,
  with the one result seen before it marked so), each learner rebuilt byte for byte from the frozen learner at `97f30b9`
  by its `make_learner.py`, and each study's `verify.py` rerun, with output identical to its log. The searches
  themselves were not rerun.*
- *For the duplicate-check study (§4, bar 3): its pre-registration's hash and file time (20:04 PT, before the runs with
  the check on), both learners rebuilt byte for byte from `97f30b9` and the committed study builders, and `verify.py`
  rerun, with output identical to its log. Its replays with the check off need the stores from Modal and were not
  rerun.*
- *For the picks study (§4, bar 3): its pre-registration's hash and file time (20:48 PT, before the runs with its flags
  on), both learners rebuilt byte for byte from `97f30b9` and the committed study builders, and `verify.py` rerun, with
  output identical to its log.*
- *For the reach study (§4, bar 3): its pre-registration's hash and file time (01:24 PT, before the runs with its flags
  on), both learners rebuilt byte for byte from `97f30b9` and the committed study builders, and `verify.py` rerun, with
  output identical to its log.*
- *For the composition test (§4, bar 3): its pre-registration's and two addenda's hashes and file times (06:19, 06:40
  and 06:46 PT, each before the runs it names), the copied stage-3 driver's record against the committed driver's (equal
  but for timings), `verify.py summary` rerun with output identical to its log, and the control's round rerun on both
  ladders, 768/768 on every rung.*
- *For the composition test's second round (§4, bar 3): its pre-registration's and two addenda's hashes and file times
  (07:41, 07:53 and 08:43 PT, each before the runs it covers, by the launch logs), its three replays against round 1's
  records (equal but for timings), and the round rerun on both ladders, 768/768 on every rung.*
- *The data counts in §1–2 were computed from the released H1 files, and the commits come from `git log`.*
- *The one-line findings of the timeline (§3) and the summaries of how each piece learns, including the NCA and bar-3
  studies, summarize their write-ups, which are linked; they were not re-run.*
- *The older write-ups are left as they were. §9 lists where a write-up and a record disagree.*
- *Everything described is committed. One study is running, bar 3 on numbers given as digits (§7); its folder is not
  committed yet.*

**In one minute.**
- The task is exact repeated modular squaring. Almost every entry on the leaderboard scored near 0%, because the
  remainder erases partial credit: an almost-right model gets random-looking answers, so gradients have nothing to
  follow (§1–2).
- Designs built like a CPU certify: discrete digits, one small cell reused everywhere, and labels that choose between
  whole machines. Four lines of work certified through the official evaluator: SpecBit, UnfoldB, UnfoldS and the
  learned digit machine (§3).
- The project is measured against three bars (§4):
  1. **Certify H1 with learned parts: done.**
  2. **Nothing a strict judge would call the hand-written algorithm: partly cleared.** The stage-3 model
     ([`integration/submission_stage3_s3i.py`](integration/submission_stage3_s3i.py)) has no hand-written skeleton
     and no menus. In one training run it learns its digit tables from laws, builds a library of small programs on a
     general digit machine, and discovers the round, "copy x, multiply, reduce". It certifies on all 8 seeds on the
     official manifest, training in at most 1,350 s of the 3,600. What is left is narrower than before:
     - removing s3i's hand-made parts one at a time, on two seeds, shows what the result rests on: two task-shaped
       instructions, the practice problems that define long division's step, how the multiply's row and the
       registers are posed, and routines;
     - variants that also certify on the official manifest go further: one poses its own practice problems from a
       general law book instead of the hand-picked list, and that book can shrink to 5 laws or be generated from a
       grammar; one learns the loop over T, one never reads the clock, and one replaces a task-shaped instruction
       with one that bar 3's learner invented (§4.7, §4.8);
     - one more replaces both task-shaped instructions with generic ones, an if block and shifts, so that the learner
       builds long division's step itself. It certifies too, training 2.2 times as long as sp2 but inside the
       budget, after search rules found by watching whether the round came (§4.8).

     The law book or its grammar, the choosing rule and most of the instruction set are still designed.
  3. **A general algorithm learner: results at a tiny size, and first steps at full size.** A learner that picks its own
     practice problems invents x² mod N for x < N ≤ 7, and, changed in nothing, four of nine other formulas, two more
     once its solved formulas carry over. Made ten times cheaper with the same choices, it also invents 10·x² and
     10·x² + 1 along a sequence of formulas, and Fibonacci from a new kind of practice problem. Scoring which pieces the
     target factors through, and searching one of them deeper, brings x³ and 10·x² from scratch. Frozen, it invents 5 of
     10 formulas none of its rules were written for, with 16 of 20 pre-registered predictions right. Run on one digit
     place, it invents all four of stage 3's digit cells. At full size, the self-practising stage 3 learns 8 of 11 tasks
     of the family after three general changes to its round's search. Within the task family, general lessons v3 now
     learns all 14 practice datasets that use one fixed modulus.
- It got there by removing hand-written parts one at a time, with certification as the gate at each step (§5).

## 1. The task, in plain words

A prompt gives three numbers: a modulus **N**, a start **x** and a count **T**. The model must square x, keep the
remainder after dividing by N, and repeat that T times, then write the result's digits. In short: x^(2^T) mod N.

**A real training row.** The prompt reads `N 9 8 1 4 X 1 3 8 5 T 1`. N's digits are written lowest first, so:
- N = 4189, x = 1385, T = 1;
- 1385² = 1,918,225 = 457 × 4189 + 3852;
- so the answer is `3852`.

A T = 2 row squares twice. With x = 2605 and the same N, 2605² leaves 4034, and 4034² leaves 3080.

**What training shows** (counted from the released data):
- 101,999 rows from only 430 different moduli of 13, 16 and 20 bits;
- every modulus is a Blum integer: N = p × q, with p and q each 3 more than a multiple of 4 (4189 = 59 × 71);
- T = 4, 8 and 16 have 33,333 rows each, and T = 1 and T = 2 only 1,000 each;
- no training number has more than 7 digits.

**What certification asks.**
- There are seven rungs, T = 1, 2, 4, 8, 16, 32, 64, of 768 fresh prompts each.
- A rung counts only if all 768 answers are exactly right, and only as part of an unbroken run from T = 1.
- There are two ladders. One uses moduli from the training pool. The other uses 505 moduli never seen in training, at
  14, 18, 22 and 24 bits; 110 prompts of each of its rungs have an 8-digit N, a size training never shows.
- The evaluator also scores three splits: test (9,999 rows, held-out moduli, T = 4, 8 and 16), and ood_t and
  ood_n_t (10,002 rows each, T = 32).
- Budgets: 3,600 s of training on one H100, and 1,800 s of evaluation.

The rules are [one-layer-deeper/README.md §Rules](../one-layer-deeper/README.md#rules). The Hard tier ranks entries by
the certified T on the two ladders.

## 2. Why it is hard

**The remainder erases partial credit.** Suppose a model squares 1385 but gets one high digit wrong: 2,918,225 instead
of 1,918,225.
- Without the remainder, the answer would be off in one digit.
- With it, the answer is 2681 instead of 3852: no digit in common, and nothing that says "almost".

So every wrong digit fact before the remainder turns into a random-looking answer. Gradient descent from answers gets
no sense of direction until the arithmetic is already right, and one more squaring scrambles any error left.

**There are few easy rows, and the test goes further than training.**
- Only 5 of the 1,000 T = 1 rows have x² < N, the only rows where no remainder is taken at all.
- T = 32 and 64, the tens digits 2–6 of T, and 8-digit moduli never appear in training.
- Certification is all or nothing: one wrong prompt out of 768 loses that rung and every rung above it.

**What happened on the leaderboard.** [REPORT.md](history/REPORT.md) §0 counted 3 of 146 entries certified to T = 64.
The other 143 sat at T = 1 with at most 12.8% (most below 1%). That count comes from the public leaderboard at the time;
it is not in a run record and was not re-checked here.

**The diagnosis** ([REPORT.md](history/REPORT.md) §2–3): labels can *check* the arithmetic but not *teach* it. What
works is a CPU's own structure: digits snapped to exact values, one small cell reused at every place, and lessons that
either check the machine against itself or choose among complete machines that the labels then verify.

## 3. Timeline: what failed, what certified, and why

Commit times are Pacific. Everything before the first commit is in the snapshot `36cac5a` (Oct 19, 23:50).

### Oct 19: the diagnosis, then the first certifications

| approach | result | write-up |
|---|---|---|
| generic networks (MLP, transformer, looped transformer) on small versions of the task | addition is learned as a rule; multiplication, squaring and the remainder only as lookup tables. On moduli the network never trained on, the remainder is at chance | [history/microlab/LADDER.md](history/microlab/LADDER.md), [LADDER_MATCHED.md](history/microlab/LADDER_MATCHED.md), [FINDINGS_pre_remainder.md](history/microlab/FINDINGS_pre_remainder.md) |
| learned digit tables composed into arithmetic, from answers | one table works; deep compositions (schoolbook multiplication, about 72 lookups in a row) stall even with no remainder | [FINDINGS_substrate.md](history/FINDINGS_substrate.md) (first part) |
| looping to large T | not the obstacle, once the state is snapped to digits and the step is exact | [history/microlab/FINDINGS_looping.md](history/microlab/FINDINGS_looping.md) |
| relaxing one rule at a time | traces that show the work certify at 8–24 bits, but only when every trace token is a local operation, and traces are extra supervision. Generic looped networks stop at 6–8 bits | [RELAXED.md](history/RELAXED.md) |
| designed machines with self-calibrated cells (`history/cadet.py`, `history/menu.py`) | labels choose among complete machines; certify T = 64 in a local evaluator | [REPORT.md](history/REPORT.md) §4 |
| **SpecBit**: a designed binary netlist with one learned 3-in/2-out gate, run as a mixture over all 65,536 possible gates at once | **certified** through the official evaluator. The labels pick the full adder | [REPORT.md](history/REPORT.md) §9 |
| search over whole machines in bit lanes on a GPU | a grammar of 2^63.9 machines was searched in 20 minutes on one H100 (576 of its 5,184 survivors certify). Later, laws that need no labels left only machines that certify | [REPORT.md](history/REPORT.md) §9–14 |
| exact arithmetic blocks with learned wiring | with exact multiply and remainder blocks the step is a 2-instruction program, and one label picks it by enumeration. Soft (gradient) selection of the same program fails | [history/microlab/FINDINGS_exact_blocks.md](history/microlab/FINDINGS_exact_blocks.md), [FINDINGS_loop_csub.md](history/microlab/FINDINGS_loop_csub.md) |
| **UnfoldB**: 4 learned numbers (a digit base B and V = w · (1, s, s²)), fitted with a loss that unfolds the remainder by itself | **certified**, but it needs a batch of 100,000 rows | [UNFOLD.md](history/UNFOLD.md), [history/unfold_sub/NOTE.md](history/unfold_sub/NOTE.md) |
| an unknown recurrence | lattice reduction recovers a hidden polynomial step from a few rows | [UNFOLD_FAMILY.md](history/UNFOLD_FAMILY.md) |

### Night of Oct 19–20: a small-batch fix, and the quotient-digit puzzle

| approach | result | write-up |
|---|---|---|
| **UnfoldS**: UnfoldB at the manifest's batch of 512, carrying the fit's information in the optimizer state | **certified** | [history/unfold_small/NOTE.md](history/unfold_small/NOTE.md) |
| long division with a learned subtraction and a learned choice of quotient digit, from answers | each half learns when the other is given; together they stall (about half right at 6 bits, near 0 at 7). Hard EM, a "clock" loss and curricula do not fix it | [history/microlab/FINDINGS_quotient.md](history/microlab/FINDINGS_quotient.md) and its `_em`, `_clock`, `_shared` siblings |
| the same puzzle, shrunk until it can be computed exactly | explained: identical trial slots sit on a symmetric saddle and split the wrong way. Fix: give every slot a fixed meaning (one shared subtractor fed exact multiples of N), teach it with a chooser that starts undecided, then let the subtraction's own borrow choose | [history/tinycore/TINYCORE.md](history/tinycore/TINYCORE.md) |

### Oct 20: learned parts in the real machine, one at a time

| commit | what became learned | write-up |
|---|---|---|
| 05:38 `824d073` | the subtraction table, inside a designed long division | [LEARNED_REMAINDER.md](learned_remainder/LEARNED_REMAINDER.md) |
| 14:51 `e1dd000` | the multiples of N, now built by repeated subtraction | [CHAIN.md](learned_remainder/CHAIN.md) |
| 22:16 `c0c86a2`, `c6c9597` | the times table (squaring); the countdown of T and the division's stop rule | [SQUARE.md](learned_remainder/SQUARE.md), [CONTROL.md](learned_remainder/CONTROL.md) |
| 22:55 `550f60e`, `f342a91` | the adder in the self-check; the prompt reader and answer writer | [ADDER.md](learned_remainder/ADDER.md), [IO.md](learned_remainder/IO.md) |

### Oct 21: one model, then simpler lessons, learned wiring, and both together

| commit | what | write-up |
|---|---|---|
| 00:55 `9959997` | **every learned piece in one model**, with a prediction path inside autograd | [COMBINED.md](learned_remainder/COMBINED.md) |
| 00:55 `9332360` | wiring and tables learned together in tiny machines | [history/tinyrouting/TINYROUTING.md](history/tinyrouting/TINYROUTING.md) |
| 05:19 `f498805` | an ablation of the training recipe; a simplified recipe certified | [history/ablation/ABLATION.md](history/ablation/ABLATION.md) |
| 05:35 `5d2b917` | **digit meanings from the order of the numbers**: the subtraction table learned with no hand-written adder at any step | [anchors/ANCHORS.md](anchors/ANCHORS.md) |
| 10:25 `36fc156` | **the machine's wiring learned**, in three certified stages | [history/learned_wiring/WIRING.md](history/learned_wiring/WIRING.md) |
| 13:02 `b76f2cd` | the learned wiring with no hand-written arithmetic anywhere | [history/learned_wiring/ANCHORED.md](history/learned_wiring/ANCHORED.md) |
| 13:10 `c166f8d` | **wide menus**: the wiring chosen from about 4·10^17 whole machines | [wider_menus/MENUS.md](wider_menus/MENUS.md) |
| 14:29 `d8856a8` | **the wide menus with no hand-written arithmetic anywhere** | [wider_menus/WIDE_ANCHORED.md](wider_menus/WIDE_ANCHORED.md) |

### Oct 21 evening to Oct 22: generic parts, other tasks, and discovering the skeleton

| commit | what | write-up |
|---|---|---|
| Oct 21, 20:42 `16b66dd` | **the machine is one setting of a generic looped transformer**; loosened notch by notch. Generic MLP digit cells certify at H1; soft wiring learns only from short menus | [history/bridge/BRIDGE.md](history/bridge/BRIDGE.md) |
| Oct 22, 02:04 `ec35027` | **the litmus test**: the best model run unchanged on 26 other tasks | [litmus/LITMUS.md](litmus/LITMUS.md) |
| Oct 22, 02:04 `25d6404` | **library learning**: merging the operations of solved programs rediscovers the hand-written skeleton | [library/LIBRARY.md](library/LIBRARY.md) |
| Oct 22, 06:15 `90f245f` | **the tournament**: the division's wiring chosen inside training, at free attention's width | [history/tournament/TOURNAMENT.md](history/tournament/TOURNAMENT.md) |
| Oct 22, 06:15 `1613c28` | **general lessons**: one unchanged file learns H1 and seven other formulas | [general_lessons/LESSONS.md](general_lessons/LESSONS.md) |
| Oct 22, 06:15 `63f1613` | **inventing**: which of our inputs a search-and-compress learner cannot do without | [library/INVENTING.md](library/INVENTING.md) |
| Oct 22, 09:32 `22df2de` | the machine compiled by handwired into affine + ReGLU layers, loosened the same way | [history/handwired_bridge/HANDWIRED_BRIDGE.md](history/handwired_bridge/HANDWIRED_BRIDGE.md) |

### Oct 22 afternoon to Oct 24: integration, one fixed modulus, a shader, and bar 3

| commit | what | write-up |
|---|---|---|
| Oct 22, 14:29 `6f6408d` | general lessons v2: the answers read at the batch's smallest T, which fixes M4 | [general_lessons/LESSONS.md](general_lessons/LESSONS.md) §11 |
| Oct 22, 22:43 to Oct 23, 01:05 `893b8c0`, `d1c8490`, `83f364c` | one shared, preemption-safe Modal runner, and a measured seed policy | [tools/README.md](tools/README.md) |
| Oct 23, 01:06 `b5f1740` | **integration**: the tournament and the general lessons in one file (stage 1); then **the round discovered inside training** with no skeleton or menus (stage 3), certified on the official budget | [integration/INTEGRATION.md](integration/INTEGRATION.md) |
| Oct 23, 04:28 `5f39a31` | **bar 3, practice**: the learner picks its own practice problems and invents x² mod N, at a tiny size | [bar3/PRACTICE.md](bar3/PRACTICE.md) |
| Oct 23, 12:08 `8e7422e` | stage 3 certifies on all 8 seeds within the hour (s3i) | [integration/INTEGRATION.md](integration/INTEGRATION.md) §3 |
| Oct 23, 12:08 `66304ba` | **general lessons v3**: the answers choose whole digit codes, and all 14 fixed-modulus practice sets are learned | [general_lessons/fixed/FIXED_MODULUS.md](general_lessons/fixed/FIXED_MODULUS.md) |
| Oct 23, 12:08 `3b7fd0f` | **the NCA shader probe**: identical local cells hold long division exactly but do not learn it from lessons | [history/nca/NCA.md](history/nca/NCA.md) |
| Oct 23, 12:27 `5c31dd0` | s3i is the final stage-3 file | [integration/INTEGRATION.md](integration/INTEGRATION.md) |
| Oct 23, 14:46 `90dc084` | **bar 3, breadth**: the unchanged learner invents 4 of 9 new formulas, 2 more by carry-over | [bar3/BREADTH.md](bar3/BREADTH.md) |
| Oct 23, 20:44 `d5e0bc5`, 20:47 `40a399c` | the handover for an outside judge: an entry point with the claims and the gray rules, a hygiene pass, and a record checker (`tools/verify_records.py`) | [REVIEW.md](REVIEW.md), [HANDOVER.md](history/HANDOVER.md) |
| Oct 23, 20:45 `aad38c9` | **the s3i ablation**: each hand-made part removed in turn and retrained on the official manifest | [integration/ABLATION_S3.md](integration/ABLATION_S3.md) |
| Oct 23, 20:45 `8f6b3e1` | **stage 3 with self-posed practice**: the hand-picked curriculum replaced by a law book, and it still certifies | [integration/SELF_PRACTICE.md](integration/SELF_PRACTICE.md) |
| Oct 24, 01:27 `6e31164` | **the NCA follow-up**: trained long with weight decay, or moving weights only where the questions agree, the shader grid still memorizes | [history/nca/NCA.md](history/nca/NCA.md) §9 |
| Oct 24, 03:18 `2cbbdf0` | **the s3i ablation, second batch**: seed 1 gives all 13 arms the same verdicts; five more arms; and a weak reduction check in s3i that about 3% of runs would draw | [integration/ABLATION_S3.md](integration/ABLATION_S3.md) |
| Oct 24, 05:02 `69a5fee` | the ablation's wider search: every program up to cost 6 still holds no missing piece | [integration/ABLATION_S3.md](integration/ABLATION_S3.md) §7 |
| Oct 24, 05:03 `8cc8151`, 07:33 `e26fb2b` | **the loop over T, learned**: a generic count loop replaces the countdown, and a second search finds "run the round T times". Seed 2's time-out comes from the random start | [integration/learned_loop/LEARNED_LOOP.md](integration/learned_loop/LEARNED_LOOP.md) |
| Oct 24, 07:32 `3d6909a` | **the two weak checks fixed** in copies of s3i and sp2, which replay all 17 original runs exactly; with the fix, Occam's tie-break and the step problem are not needed | [integration/ABLATION_S3.md](integration/ABLATION_S3.md) §8 |
| Oct 24, 07:35 `6da40e6` | **strict builds**: s3i and sp2 with work budgets instead of the clock and their state in buffers; 17 runs replay line for line | [integration/strict/STRICT.md](integration/strict/STRICT.md) |
| Oct 24, 07:35 `bbb9a77` | **breadth at full size**: after three general changes to its round's search, sp2 learns 8 of 11 tasks | [integration/sp_breadth/SP_BREADTH.md](integration/sp_breadth/SP_BREADTH.md) |
| Oct 24, 07:36 `1ebf7f2` | **the bridge**: bar 3's learner invents stage 3's select and subtract-with-borrow at one digit place; s3i with the invented select certifies | [bar3/bridge/BRIDGE.md](bar3/bridge/BRIDGE.md) |
| Oct 24, 08:00 `d96888b` | **the NCA, with optimizers that can rest at the rule**: they keep the compiled rule where every lesson is met, but from random every arm still memorizes | [history/nca/NCA.md](history/nca/NCA.md) §10 |
| Oct 24, 08:07 `c6922d0` | **the law ablation**: sp2's law book and choosing rule taken apart; a 5-law book and a book generated from a grammar each certify | [integration/sp_ablation/SP_ABLATION.md](integration/sp_ablation/SP_ABLATION.md) |
| Oct 24, 08:21 `5d108a7` | **bar 3, cheaper and deeper**: 515 search-hours down to 51 with the same choices; along a sequence of formulas it invents 10·x², 10·x² + 1 and x³, and Fibonacci comes from scratch | [bar3/scale/SCALE.md](bar3/scale/SCALE.md) |
| Oct 24, 08:23 `597440e` | **generic instructions** in place of SEL.S and BRG.X: the round is found inside training, but the official run was stopped before it wrote a record, so the build is not certified | [integration/generic_ops/GENERIC_OPS.md](integration/generic_ops/GENERIC_OPS.md) |
| Oct 24, 16:10 `10d9882` | **the law ablation on seed 1**: the 17 arms the spend limit cut off; two parts that only cost time on seed 74, the x + d·N pattern and choosing at all, are needed on seed 1 | [integration/sp_ablation/SP_ABLATION.md](integration/sp_ablation/SP_ABLATION.md) |
| Oct 24, 18:01 `29efe74` | **the law ablation's rerun at three times the budget**: without lpca's ties, seed 74 still finds no round; its climb runs out of levels, not time | [integration/sp_ablation/SP_ABLATION.md](integration/sp_ablation/SP_ABLATION.md) §4 |
| Oct 24, 17:25 `23a5534` | **the NCA's 13 cut-off runs, finished**: from random every arm still memorizes; only numbers read at a run's end moved | [history/nca/NCA.md](history/nca/NCA.md) §10 |
| Oct 24, 15:45 `3a9d9e0` | **generic instructions, certified**: rerun once Modal's limit was raised, the official H100 run (training 2,770 s) and 8 of 8 B200 seeds certify | [integration/generic_ops/GENERIC_OPS.md](integration/generic_ops/GENERIC_OPS.md) §0 |
| Oct 25, 03:12 `6d2d60d` | **the weak-check fix in eight more certified files**: each fixed copy certifies wherever its original did, over 78 runs, and 75 runs equal the original's; in one of the other three, the fixed check rejects a wrong program the old one let through | [integration/ABLATION_S3.md](integration/ABLATION_S3.md) §8 |
| Oct 25, 03:30 `74c8c32` | **bar 3's cut runs, finished**: hardest first's sequence invents Montgomery's and Fibonacci; the anchored-pairs sequences miss both in 5 rounds | [bar3/scale/SCALE.md](bar3/scale/SCALE.md) §6 |
| Oct 25, 03:31 `97f30b9` | **bar 3 sees a piece before solving it**: scored by whether the target factors through it, one deeper search brings x³ and 10·x² from scratch | [bar3/scale/subgoal/SUBGOAL.md](bar3/scale/subgoal/SUBGOAL.md) |
| Oct 25, 03:55 `b5c82a2` | **bar 3 invents add-with-carry**: the learner's round 1, lost to the spend limit and finished, invents it by reducing the sum once; multiply-accumulate still not, even with a rule that poses the counting piece | [bar3/bridge/CARRY.md](bar3/bridge/CARRY.md) |
| Oct 25, 05:48 `a495bce` | **a theory note on large numbers**: what classical, learning and neural theory already say, mapped onto the project. H1 is Rivest, Shamir and Wagner's time-lock function, with no known shortcut over T squarings without N's factors; digits, repeated squaring and clock features are the exp/log reparametrizations that work | [theory/LARGE_NUMBERS.md](theory/LARGE_NUMBERS.md) |
| Oct 25, 05:53 `fae297b` | the subgoal's own subgoal, tried small: no source poses 10·x² + 1's piece, 10X mod N, so the study stopped before any run | [bar3/scale/subgoal/SUBGOAL.md](bar3/scale/subgoal/SUBGOAL.md) §8 |
| Oct 25, 10:13 `307ab93` | **bar 3 on formulas no rule was written for**: frozen, with its predictions written down first, the learner invents 5 of 10, all in round 0 or 1, and 16 of 20 predictions hold; the misses need a step no source poses or a longer program | [bar3/scale/heldout/HELDOUT.md](bar3/scale/heldout/HELDOUT.md) |
| Oct 25, 10:22 `9b640ef` | **bar 3 invents multiply-accumulate**: a deeper search aimed at its piece, where a clean program beats a cheaper one that uses up N, so all four of stage 3's digit cells are now invented; Montgomery's quotient digit is a table of one place | [bar3/bridge/CARRY.md](bar3/bridge/CARRY.md) §12–13 |
| Oct 25, 12:59 `160bd0f` | a plan from bar 3 to a contest-legal entry: what has to move inside the training run, what is still designed in the self-practising file, the questions only a judge can settle, and the cheapest test of each; nothing run | [LEGAL_PLAN.md](history/LEGAL_PLAN.md) |
| Oct 25, 17:51 `c185410` | **every certified variant of sp2 in one file, not certified**: it builds, and every study's local check passes on it, but on the official H100 the hour runs out in level 4 of the climb, before the reduction. The generated book and the generic instructions together make level 3 cost 11.5 times sp2's | [integration/combined_sp2/COMBINED.md](integration/combined_sp2/COMBINED.md) |
| Oct 25, 19:02 `305ac14` | **bar 3 with x as digits**: given a loop over x's digits and a source that asks one question of the rows, the learner invents x mod N in base 2 by long division's step, right on x below 10⁹; nothing else of 36 runs comes | [bar3/scale/digits/DIGITS.md](bar3/scale/digits/DIGITS.md) |
| Oct 25, 19:02 `2da752e` | **bar 3 with a running sum's step as a source**: the triangular numbers come in round 1 under anchored pairs; a second source poses 10·x² + 1's piece, but the learner's duplicate check throws it away | [bar3/scale/newton/NEWTON.md](bar3/scale/newton/NEWTON.md) |
| Oct 25, 20:07 `914952f` | **the combined file, round 2**: three fixes to its acceptance setup and work budget make level 3 six times cheaper, and the climb reaches the round, but at 3,206 s in the tiny core, past what fits in the hour; no official run | [integration/combined_sp2/COMBINED.md](integration/combined_sp2/COMBINED.md) §8 |
| Oct 25, 20:29 `a3acc38` | **bar 3's duplicate check on each problem's own rows**: it frees 10·x² + 1's piece, 10X mod N, but no rerun's picks change; hardest first's pool order still buries the piece, and x² mod N's twin is a true copy | [bar3/scale/rowcheck/ROWCHECK.md](bar3/scale/rowcheck/ROWCHECK.md) |
| Oct 25, 22:48 `1f46707` | **two four-piece files certify**: the combined file without the generated book, and without the generic instructions, each on the official H100; given two hours, the five-piece file learns H1 but freezes 531 s past the hour | [integration/combined_sp2/COMBINED.md](integration/combined_sp2/COMBINED.md) §9 |
| Oct 25, 22:48 `5f78072` | **every certified variant in one file, certified**: a Triton kernel and two other speedups make the five-piece climb 7.7 times faster with nothing learned changed, and it certifies in 751 s of training | [integration/combined_fast/FAST.md](integration/combined_fast/FAST.md) |
| Oct 25, 23:15 `a2e4bac` | the fast five-piece file's seeds 1–7 certify on H100s too: 8 of 8, in 733–838 s of training; on three of them the answer program's search crossed the work budget after finding its programs | [integration/combined_fast/FAST.md](integration/combined_fast/FAST.md) §8 |
| Oct 26, 00:15 `f237a61` | **fast2**: the fast file with its work budgets set by STRICT.md's rule from eight climbs measured with no budget; 8 of 8 on H100s, and the measurements show the old budget never cut anything | [integration/combined_fast/FAST.md](integration/combined_fast/FAST.md) §9 |
| Oct 26, 00:32 `238437d` | **REVIEW.md rewritten around fast2**, now the file to judge; the record checker's default run checks its eight records first; the hygiene pass covers the 28 new commits, with no credentials, emails or key found | [REVIEW.md](REVIEW.md), [HANDOVER.md](history/HANDOVER.md) |
| Oct 26, 01:17 `577fb72` | **bar 3 gets 10·x² + 1 under hardest first**: with the row check, breaking ties by the subgoal score picks its piece, 10X mod N, in round 2; passing priority to a subgoal's twin costs a win, and a score that sees big x cannot act | [bar3/scale/picks/PICKS.md](bar3/scale/picks/PICKS.md) |
| Oct 26, 06:03 `8e0126b` | bar 3's deeper look finds the program that passes at big x, but the pick rule never picks its step; the twin as a fifth pick keeps every win; x² mod N at big x still does not come | [bar3/scale/reach/REACH.md](bar3/scale/reach/REACH.md) |
| Oct 26, 07:32 `9e24be4` | **a first composition test**: on stage 3's digit machine with its law book removed, bar 3's sources pose no piece of the round, in 5 arms; the 5-law book's control gets the round, 768/768 to T = 64 | [bar3/scale/compose/COMPOSE.md](bar3/scale/compose/COMPOSE.md) |
| Oct 26, 09:43 `ad77ad4` | **bar 3's pieces replace the law book's reduction**: on stage 3's machine, with the 5-law book less its reduction, bar 3's ported sources, subgoal score and a search that tests more than one spelling per state get the round at level 5, right to T = 64; bar 3's set alone still does not | [bar3/scale/compose2/COMPOSE2.md](bar3/scale/compose2/COMPOSE2.md) |
| Oct 26, 11:17 `0d7be26` | composition round 3: bar 3's sources do not pose the multiply-add row, and round 2's winning setup fails on x³ and 10·x² + 1 mod N, so its route is shaped to x² | [bar3/scale/compose3/COMPOSE3.md](bar3/scale/compose3/COMPOSE3.md) |
| Oct 26, 13:21 `6a8c758` | bar 3's search on a GPU: identical results on 18,708 stored searches and two whole runs, about 95 times faster, but only 3–10 times cheaper, which buys about half to one more instruction of depth | [bar3/gpu_search/GPU_SEARCH.md](bar3/gpu_search/GPU_SEARCH.md) |
| Oct 26, 14:30 `f2a1621` | **bar 3's rules consolidated**: one score for every choice keeps 1 of 16 earlier wins, but two general rules, a uniform search depth for every candidate and anchored pairs' picks, bring back 13 of 16; the twin, the fifth pick, the deeper look and the big-x score were needed for none | [bar3/scale/consolidate/CONSOLIDATE.md](bar3/scale/consolidate/CONSOLIDATE.md) |
| Oct 26 `b49e265` | REVIEW.md checked by a strict outside read: 27 errors and overclaims fixed, the record checker still 18 of 18 | [REVIEW.md](REVIEW.md) |
| Oct 26 `05810dc` | **bar 3 supplies the multiply-add row too**: a source that cuts the target's laws to one digit place poses the row, and on stage 3's machine with the book less its reduction and row, x² mod N's round comes at level 5, right to T = 64; not yet for x³ or 10·x² + 1, and not without the book's multiply | [bar3/scale/slice/SLICE.md](bar3/scale/slice/SLICE.md) |
| Oct 26 `827e784` | bar 3's studies run on the GPU search through one runner, seven whole runs record for record, runs in minutes instead of hours | [bar3/gpu_search/GPU_SEARCH.md](bar3/gpu_search/GPU_SEARCH.md) |
| Oct 26 `5048c75` | an audit of bar 3's 14 practice sources against the rule that no training target may come from Python's arithmetic: which can run inside a training run, which need a redesign | [bar3/LEGAL_SOURCES.md](bar3/LEGAL_SOURCES.md) |
| Oct 26 `a81712c` | the audit's decisive test: at full size, a law test on the learner's own routines picks x² alone and the reduction is found; the first spelling taken is a look-alike that fails on some unseen moduli, and the climb's own rule keeps a correct one | [bar3/scale/legal/LEGAL_TEST.md](bar3/scale/legal/LEGAL_TEST.md) |
| Oct 26 `ef10ac1` | **a fully legal climb gets H1's round**: every practice answer from the data, identities or the learner's own routines; with the book's reduction cut to one step, bar 3's sources bring the round in seven arms, right to T = 64 on unseen moduli too; the row and the multiply still come only from the book | [bar3/scale/legalclimb/LEGALCLIMB.md](bar3/scale/legalclimb/LEGALCLIMB.md) |
| Oct 26 `ec312b6` | **legally, the book shrinks to three laws**: a legal reader poses the row, the step, the multiply and the reduction from x² mod N's law, and H1's round comes at level 4 with only the comparison, the link and the addition left in the book | [bar3/scale/legalloops/LEGALLOOPS.md](bar3/scale/legalloops/LEGALLOOPS.md) |
| Oct 27 `355f62f` | offline, posing each cut's loop takes the book down to the comparison alone: x² mod N's round at level 2, right to T = 64; x³ and 10·x² + 1 still get no round | [bar3/scale/loops/LOOPS.md](bar3/scale/loops/LOOPS.md) |
| Oct 27 `8e3c7fe` | **the legal file certifies 8 of 8**: fast2 with its generated law book replaced by three laws and bar 3's legal reader, every practice answer from the learner's own routines; training in 465–721 s | [integration/combined_legal/LEGAL_FILE.md](integration/combined_legal/LEGAL_FILE.md) |
| Oct 27 `6be8a0e` | **offline, no book at all**: cutting Horner's step down to the link and the comparison poses the comparison, and x² mod N's round comes at level 2 with no law in the book, right to T = 64; the cube comes as two calls of x·o mod N; 10·x² + 1 learns its constant but gets no round | [bar3/scale/reading/READING.md](bar3/scale/reading/READING.md) |
| Oct 27 `22b859c` | the legal file's reader gets a wider grammar, subtract, power, constants and depth 3 (200 candidate laws instead of 35), still reads x² mod N and certifies 8 of 8 | [integration/legal_wide/WIDE.md](integration/legal_wide/WIDE.md) |
| Oct 27 `f7dc5f8` | **a legal file with no law in its book certifies 8 of 8**: a designed rule reads off the answers that the law ends in a reduction by N and poses that step's parts, the comparison, the link and the addition; then the wide reading and the round, every practice answer from the learner's own routines; training 472–634 s | [integration/legal_nobook/NOBOOK.md](integration/legal_nobook/NOBOOK.md) |
| Oct 27 `fe35373` | offline with no book, a rule that poses every window of the target's reading brings 6 of 7 family tasks unchanged (x², the cube, 10y², y² + 1, y² + 3, 10y² + 1), and y² − 1 with two more general changes | [bar3/scale/family/FAMILY.md](bar3/scale/family/FAMILY.md) |
| Oct 27 `ce9f118` | **bar 3's finish line**: with no law in its book and round 7's general rules, one legal file certifies H1 8 of 8 and, byte for byte, learns 4 of 8 family tasks at full size, one of two held-out tasks among them | [integration/legal_family/FAMILY_FILE.md](integration/legal_family/FAMILY_FILE.md) |
| Oct 27 `80082c0` | ROUNDT's rows moved into buffers, ROUNDT made cheaper and the reader widened: H1 8 of 8 again, 7 of 8 family tasks learned, but neither of two held-out tasks drawn at random | [integration/legal_wider/WIDER.md](integration/legal_wider/WIDER.md) |
| Oct 27 `a88fc45` | **four steps deep**: H1 8 of 8 again, and byte for byte all 22 other tasks it is run on, among them 12 held out, six of them drawn with the fixed file's md5 as the seed (6 of 6); now the file to judge | [integration/legal_deep/DEEP.md](integration/legal_deep/DEEP.md) |

### The lessons that came out of the failures

- **Generic networks memorize instead of computing the remainder**, and on a new modulus they are at chance.
- **Soft mixtures of choices learn nothing here.** After the remainder, the loss stays flat until the wrong options
  hold almost no weight.
- **Answers alone, pushed through a learned machine, do not teach it.** Untrained parts get bypassed, or trial slots
  co-adapt into a lookup arrangement. Even with every training row answered, answers did not start the subtraction
  table from random digits within that study's budget ([ANCHORS.md](anchors/ANCHORS.md) §4.3).
- **A check made only of the learned table can be passed by whole families of wrong tables**: digits relabeled,
  borrows swapped, "copy the first number". Meaning has to come from outside the table. First that was a hand-written
  adder for a short while ([ABLATION.md](history/ablation/ABLATION.md) §7). Now the order of the numbers does it alone:
  x < N teaches every borrow, and two laws of subtraction, applied one digit at a time, leave only the true table
  ([ANCHORS.md](anchors/ANCHORS.md)). With one fixed modulus that order is not enough; the answers then choose the
  whole digit code ([FIXED_MODULUS.md](general_lessons/fixed/FIXED_MODULUS.md)).
- **Wiring is learned by trying whole options and scoring each pass or fail**, not by blending options' digits on the
  wire. Blended switches settle on a machine that does nothing ([WIRING.md](history/learned_wiring/WIRING.md),
  [TINYROUTING.md](history/tinyrouting/TINYROUTING.md)). The same held for whole digit codes and whole column rules.
- **The traps of loosened wiring** ([BRIDGE.md](history/bridge/BRIDGE.md),
  [HANDWIRED_BRIDGE.md](history/handwired_bridge/HANDWIRED_BRIDGE.md)):
  - **the bypass**: free soft wiring reads the prompt's clean digits instead of the machine's computed ones. The pull
    comes from the data, not from attention's softmax: a handwired network with no attention felt the same pull;
  - **the near miss**: a 7-option menu taught by answers lands on one nearby wrong wiring on every seed, though the
    answers alone single out the right one;
  - **the constant source**: wiring that cannot drift to the prompt settles on reading a constant zero;
  - **ties**: an MLP digit cell settles on two entries giving the same digit, until "every digit once per row and
    column" breaks the tie.
- **Copies have no local check.** In a grid of identical local cells the arithmetic cells learn from their laws, but
  the long chains of copies that carry N and the borrow never form: nothing checks a copy where it happens
  ([NCA.md](history/nca/NCA.md)).
- **The usual cures for memorizing assume the general rule is the small one.** Weight decay favours small weights;
  on one account of grokking, that is how it wore a memorized lookup away until the general rule was left. In the
  shader grid the exact rule needs large weights to keep its digits sharp tick after tick, while a blurred code that
  fits the training answers is cheap, so decay favours the memorizer. Agreement between questions only slowed the
  fit ([NCA.md](history/nca/NCA.md) §9).
- **Lessons hold the rule only where the training grid goes.** With an optimizer that can rest, the shader grid keeps
  the compiled rule where every lesson is met. Where the lessons still pull, plain SGD keeps every training answer
  but drifts one size up, and 99.7–99.9% of its one-tick errors are in situations the training grid never shows
  ([NCA.md](history/nca/NCA.md) §10).
- **With wide menus, the checks still identify; the search is what can fail.** In every menu counted exactly, every
  wiring that passes all the checks is a correct one. Learning failed only where the search could not see the right
  combination: shown only a few sampled combinations instead of every one, a switch was caught by impostors in 14 of
  16 runs ([MENUS.md](wider_menus/MENUS.md)). The tournament found the identified wiring among 5.3·10^10 inside
  training; where the checks let impostors through, it crowned one ([TOURNAMENT.md](history/tournament/TOURNAMENT.md)).
- **Inventing needs good practice problems**: problems whose shortest solutions are the pieces of the answer. Without
  curated ones the loop never invents x² mod N; a learner can pose them itself from a law book and from what its best
  wrong attempts leave undone ([INVENTING.md](library/INVENTING.md), [PRACTICE.md](bar3/PRACTICE.md)). A source that
  poses steps moving two numbers at once gave Fibonacci, which no earlier source could pose
  ([SCALE.md](bar3/scale/SCALE.md)).
- **Speed buys about one instruction, and sometimes a better rule.** Each extra instruction multiplies bar 3's search
  by 9–10, so a search ten times cheaper reaches about one instruction further, and no bound on the search space
  saved more than 1.7×. But the cheaper search made a dearer choosing rule affordable along a whole sequence of
  formulas, and that rule reached 10·x², 10·x² + 1 and x³ ([SCALE.md](bar3/scale/SCALE.md)).
- **Some hand-made parts carry the result, others only save time.** Removed one at a time from s3i, two instructions,
  the practice problems that define long division's step, the posing of the multiply's row and of the registers, and
  routines each stopped the round from ever being found, on both seeds tried. Each of those runs had searched its
  whole reach, and a search one cost level wider found nothing either. Occam's tie-break seemed needed too, but it
  was only making up for a weak check. The curriculum's order, BPE and two search rules only cost time
  ([ABLATION_S3.md](integration/ABLATION_S3.md)).
- **The list of laws is not what self-practice rests on.** Removed one at a time from sp2's book, six of its 11 laws
  and three of its 8 patterns were not needed. A book of 5 laws on 4 patterns certifies, and so does one generated
  from a grammar, with no law picked by hand. What it needs is what any such grammar must have: a composition of +
  and × that accumulates, the division law's form, and target rules
  ([SP_ABLATION.md](integration/sp_ablation/SP_ABLATION.md)).
- **A check on small numbers can pass a shortcut that fails at full size.** s3i checks a candidate reduction on
  numbers of 1 to 3 digits. A program that just subtracts N over and over, 600 times at that size, passes whenever
  none of the check's 60 three-digit numbers needs more. It is wrong on every full-size product, so about 3% of runs
  would keep it and never certify. s3i's check of "x mod N for x below 2N" never tries x between N and 2N, so
  "x if x < N, else 0" passes it. Adding the missing probes, at full size and between N and 2N, changed nothing in
  the 17 runs that had certified ([ABLATION_S3.md](integration/ABLATION_S3.md) §3, §8). With generic instructions in
  place of SEL.S and BRG.X, more fakes of the reduction fooled both the search's probes and the three-digit check,
  until closeness was counted without them and the check moved to four digits
  ([GENERIC_OPS.md](integration/generic_ops/GENERIC_OPS.md)).
- **A run is one random sample of its climb.** The learned loop dropped sp2's countdown gate, 4 numbers it no longer
  uses. That moved every later random draw, and on seed 2 the climb needed 10 levels instead of 4 for the round's
  last piece, and ran out of time. With the gate put back as dead weight, seed 2 replays sp2's climb line for line,
  and the learned loop certifies on it ([LEARNED_LOOP.md](integration/learned_loop/LEARNED_LOOP.md) §5).
- **The other formulas failed in the last search, not in the practice.** On H1's prompts, the self-practising stage
  3 builds every piece of H1's round whatever the formula; what most formulas lacked was a third routine call or a
  way to make a constant. Three general changes to the round's search, none naming a formula, learn 8 of 11 tasks.
  Each was written after watching a formula fail, with the family in view
  ([SP_BREADTH.md](integration/sp_breadth/SP_BREADTH.md)).
- **What worked every time:** digits snapped to exact values, one shared cell, checks on each training row's own
  numbers that need no answer, and pass/fail scoring of whole choices.

## 4. Where it stands: three bars, and the models at the frontier

**Bar 1: certify H1 with learned parts. Done.** The first learned table certified on Oct 20 (§5, step 1). From step 11
on, every certified model learns every part that computes, and no hand-written arithmetic runs anywhere, in training or
in the answer.

**Bar 2: nothing a strict judge would call the hand-written algorithm. Partly cleared** (INTEGRATION.md §6).
- **Cleared for the forward pass.** The stage-3 model, s3i, has no hand-written skeleton and no menus. The program that
  answers was discovered inside one training run, and that run certifies on all 8 seeds tried within the official
  budget (§4.1, §4.7).
- **Not cleared for the training recipe**, where the algorithm's decomposition still lives. s3i's recipe has:
  - **an instruction set of digit arithmetic**: subtract-with-borrow, add-with-carry and multiply-accumulate at a
    pointer, keep-or-discard on the borrow, bring the top digit in, and loops by the number's length. The instructions
    are general, but they were chosen knowing the task family. Bar 3's learner, run on one digit place, invents two
    of them: subtract-with-borrow, and a select that can stand in for keep-or-discard (bar 3 below). And generic
    instructions can replace both task-shaped ones: the round is still found, though not yet certified (§4.8);
  - **a curriculum** that walks the search up the skeleton's pieces. Which laws become practice problems, and in which
    order, is a designer's choice. Two of them are definitions of long division's step, and one (mac) was posed so
    that the multiply's row becomes a problem of its own;
  - **the rest**: the round's search space (short compositions of routines) and the countdown over T, which a
    variant learns as a generic count loop (§4.8).
- **What the result rests on** ([ABLATION_S3.md](integration/ABLATION_S3.md); each part removed in turn and retrained
  on the official manifest, one H100 per arm, on seed 74 and on seed 1; the unchanged file replays s3i's own runs on
  both seeds exactly):
  - **needed on both seeds**, since without it the round is never found: SEL.S (keep or discard on the borrow) and
    BRG.X (bring X's next digit in), the two practice problems that define long division's step, mac posed with the
    running sum, the register twins, and routines;
  - **needed only because a check was weak**: Occam's tie-break toward the program that runs fewer steps. It kept
    the search off a wrong conditional subtraction that the weak check let through; with the check fixed, s3i's
    copy certifies without it on both seeds (§4.7);
  - **needed on one seed of two**: the chain problem alone. Without it the reduction depends on which of two equally
    cheap programs the search lists first;
  - **speed only**, since it still certifies on both seeds, frozen by 1,584 s at the latest: the curriculum's order
    (+277 s and +321 s on seeds 74 and 1), not searching solved twins again (+364 s, +416 s), BPE merges (+282 s,
    +291 s; an earlier +546 s came from a faulty host correction, §9), and the T = 1 filter on the round's answers
    (+82 steps on seed 74, none on seed 1);
  - **not needed**: the search's routines-first order; the late table lessons, the wrap-around fact among them; the
    step problem alone (its seed-74 run failed on the weak reduction check, and certifies with the fix, §4.7); the
    multiply's row problem (one seed); a cost limit above 5 on the round's search (at 4 there is no round: every
    round found costs 5);
  - every failed run of a needed part had searched its whole cost-5 reach without the missing piece, so more time
    would not rescue it. A wider search, every program up to cost 6, found nothing for any of the six
    (ABLATION_S3.md §7).
- **The curriculum itself can be posed by the learner** ([SELF_PRACTICE.md](integration/SELF_PRACTICE.md)). The nine
  hand-picked problems, their order, the twins and mac's posing are replaced by a book of 11 laws on 8 patterns of the
  data's numbers, each posed in every binding of its operands to the registers, and one general rule picks what to
  practise. The problems the ablation found needed come out of that book; nothing marks them. Stage 3 still finds its
  round and certifies on the official manifest (§4.1, step 19). Still designed: the law book, the choosing rule (picked
  after bar 3's own rules failed, by watching whether the round came), the instruction set and the search sizes.
- **What self-practice rests on** ([SP_ABLATION.md](integration/sp_ablation/SP_ABLATION.md); each law, pattern and
  part of the choosing rule removed in turn, 30 arms on the official manifest, each on seeds 74 and 1):
  - **needed**: the multiply-add law and its pattern, since the multiply is a loop over the row they pose; compare,
    remainder, add and the pair pattern, which every other problem is built or checked with; the round's two
    routines, the multiply and the reduction; and, in the choosing rule, rating a law on a pattern by its cheapest
    placement. Two more parts of the rule each failed on seed 74 and certified on seed 1, one of them still failing
    on seed 74 at three times the budget: they make the result reliable rather than possible;
  - **speed only on seed 74, needed on seed 1**: the x + d·N pattern (+441 s on seed 74, corrected for host speed;
    on seed 1 the reduction never comes), and choosing at all (practising everything certifies on seed 74 at the edge
    of the budget, frozen by 3,249 s, and runs out of time a level short on seed 1);
  - **not needed**: six of the 11 laws, three of the 8 patterns, and the cleanest placements over the cheapest. Seed
    1 agrees on every one;
  - **so the list itself can go**: a book of 5 laws on 4 patterns, and a book generated from a grammar, each certify
    on the official manifest and on 8 of 8 seeds (§4.8). Still designed: the grammar, the target rules and the
    choosing rule.
- **Variants that go further** (§4.8), each certified on the official manifest: the two books above; the loop over T
  learned as a generic count loop; s3i and sp2 with no decision read off the clock and their state in buffers; and
  s3i with SEL.S replaced by a select that bar 3's learner invented.

**Bar 3: a general algorithm learner. Results at a tiny size, and first steps at full size.**
- **Choosing its own practice** ([PRACTICE.md](bar3/PRACTICE.md)): with no task list from us, the learner poses
  candidate problems from a general law book applied to the data, from parts of the target, and from its own best wrong
  attempts. It chooses which to practise, most reliably by asking which pair of them, made into single instructions,
  makes the target shortest. In its first round it invents x² mod N on every pair x < N ≤ 7, and again at N ≤ 26.
  Choosing costs up to 10 times the search of our curated list (7 to 16 times at N ≤ 26).
- **Breadth** ([BREADTH.md](bar3/BREADTH.md)): changed in nothing, it invents 4 of 9 new formulas from scratch
  (x² + 1, x² − 1, x² + 3, 2x + 1), and 2 more once its solved formulas carry over as routines (x³ and Montgomery's
  x²·2⁻¹). It misses 10·x², 10·x² + 1 and Fibonacci. Its limits are a search that reaches about 6 instructions, practice
  problems with one answer from x and N, and a cost that grows with the library: about 520 search-hours in all.
- **Cheaper, then deeper** ([SCALE.md](bar3/scale/SCALE.md)). A compiled search, searches shared between problems and
  the cheaper choosing rule cut the breadth study from 515 search-hours to 51; 50 of its 54 runs make the same
  choices in every round, and the other 4 differ only where the original had stopped a search at its time limit.
  That buys about one instruction, since each extra one multiplies the search by 9–10 (about 40 over a carried
  library). But it made the pairs rule affordable along a sequence of formulas, each solved one kept as a routine,
  and there it invents 10·x², 10·x² + 1 and x³, each in the first round of its step. A new practice source, steps
  that move two numbers at once, gives Fibonacci from scratch. At N ≤ 26 and N ≤ 50, what the unchanged learner
  invents is the program it found at N ≤ 7. Hardest first's own sequence, finished after the spend-limit cut, also
  invents Montgomery's and Fibonacci; the anchored-pairs sequences miss both. Not reached: 10·x², 10·x² + 1 and x³
  from scratch. No bound on the search space tried saves more than 1.7×.
- **Seeing a piece before solving it** ([SUBGOAL.md](bar3/scale/subgoal/SUBGOAL.md)). The learner scores each
  problem it poses by whether the target factors through it, from the rows alone (the target a function of the
  problem's answer and N, or one law of the book away), and aims one deeper search at the problem the score singles
  out. From scratch, x³ then comes in round 1 under both choosing rules and 10·x² in round 1 under hardest first,
  where the same learner without the score gets neither in 5 rounds. Nothing regressed. 10·x² + 1 and Montgomery's
  still do not come: the rest of the way after x² mod N is one token beyond the deeper search, and Montgomery's 16
  instances at N ≤ 7 are too few to pin it down. A second level of subgoals, tried small, would not reach 10·x² + 1
  either: nothing poses its piece, 10X mod N (SUBGOAL.md §8).
- **Formulas no rule was written for** ([HELDOUT.md](bar3/scale/heldout/HELDOUT.md)). Every rule so far was written
  after watching one of the tracked formulas fail, so the learner, frozen with the subgoal, was run from scratch on ten
  new formulas, with its predictions written down first. It invents 5 of them in 9 of 20 runs, all in round 0 or 1:
  (x² + x), (x + 1)², 3·x² and x⁴ mod N under both choosing rules, and 2^x mod N under anchored pairs. 16 of the 20
  predictions hold, all 8 made with medium or high confidence. Without the subgoal, in a control added afterwards, two
  of the nine wins do not come and one comes two rounds later. The misses need a step no source poses (a running sum's
  y(x + 1) from y(x) and x, or Tribonacci's step on three numbers) or a program longer than the search reaches (Lucas,
  floor(x²/N) and (x³ + x)).
- **x as digits** ([DIGITS.md](bar3/scale/digits/DIGITS.md)). A loop over x's units cannot reach x of 6–8 digits, so the
  frozen learner was given a loop over x's digits in base B and one new source, which asks whether the answer for
  x·B + d is a function of the answer for x, the digit d and N, and if so poses that step. A program counted only if it
  also passed a check at big x. Of 36 runs (x mod N, x² mod N and 2^x mod N; unary, base 2 and base 10; both rules; with
  and without the source), only x mod N in base 2 with the source came, in round 1 under anchored pairs and round 3
  under hardest first: a loop over x's bits whose body is long division's base-2 step, right on 1,000 x below 10⁹.
  Unary's programs all count and run out of steps, and base 10's step and square-and-multiply were never learnable. x²
  mod N's hidden state, x mod N on long x, was found by the subgoal, but the alias check counted it a twin of the law
  book's x mod N, and where that twin was practised the pick rule's score saw no gain over counting programs. 35 of the
  36 runs came out as pre-registered. About $51.
- **Two sources from Newton's differences** ([NEWTON.md](bar3/scale/newton/NEWTON.md)). The held-out test and the
  subgoal study each named a piece no source poses. Given a running sum's step, y(x + 1) from y(x) and x, the frozen
  learner gets the triangular numbers in round 1 under anchored pairs (seen in a replay before the predictions were
  written); under hardest first the step loses to pool order. Given the offset, each rest of the way less its value at
  X = 0, it poses 10·x² + 1's piece, 10X mod N, and solves it, but its duplicate check, run on random machine states on
  which both programs die, calls it a copy of 3X, so 10·x² + 1 still does not come. The nine earlier held-out wins
  stand. Run locally from the earlier stores, $0.
- **The duplicate check, on each problem's own rows** ([ROWCHECK.md](bar3/scale/rowcheck/ROWCHECK.md)). The learner had
  thrown 10X mod N away as a copy of 3X because both programs run out of steps on all 24 random machine states its
  duplicate check uses. With the check made on each problem's own rows instead (observational equivalence, as program
  synthesis uses it), 10X mod N is learnable in rounds 2 and 3, but hardest first never picks it: 24 and then 12
  problems of the same cost stand ahead of it in pool order. x² mod N's twin stays a copy, rightly, since one program
  solves both. No rerun's picks change, and all 11 earlier wins stand, as pre-registered. $0.
- **What the learner picks** ([PICKS.md](bar3/scale/picks/PICKS.md)). Three changes, each behind its own flag. Breaking
  hardest first's ties by the subgoal score, with the row check on, brings 10·x² + 1 in round 2, for the first time in
  any study: the score singles out 10X mod N, and a merge makes the target four tokens, right on every pair to N = 120.
  Passing a subgoal's priority to its twin does not bring x² mod N in base 2, and it loses hardest first's binary x mod
  N. A pick score that sees the big-x check cannot act, since no look reaches the five-token program that passes at big
  x. Under the tie-break all 11 earlier wins stand. About $5.
- **Reach** ([REACH.md](bar3/scale/reach/REACH.md)). Two more changes aimed at x² mod N at big x. A deeper look at the
  posed steps finds the five-token program that passes at big x, and the big-x score values it at 2,189 bits, but
  anchored pairs picks from its best pairs, whose looks stay shallow, so the step is never picked. The subgoal's twin as
  a fifth pick keeps hardest first's binary x mod N, which the replacing twin had lost, but x² mod N still does not
  come: the square waits at cost 4, and the five-token target search runs out of time. All 12 earlier wins stand. About
  $7.
- **Composition on stage 3's machine** ([COMPOSE.md](bar3/scale/compose/COMPOSE.md)). Stage 3's tiny core with its law
  book removed, and the practice posed by bar 3's sources instead. Only the control with book_min's 5 laws gets the
  round (stage 3's own, 768/768 on every rung to T = 64). Bar 3's sources pose no piece in any of 5 arms, not even the
  reduction when the book supplies everything else. The causes are specific: the digit machine starts its registers
  dirty, so its near misses' values are not functions of x and N; the multiply's route, the rows where x² < N, is never
  posed at this width; and the reduction's route fails twice, since x² is never a near miss and the climb's searches
  accept nothing even when handed that problem. About $11.
- **Composition, round 2** ([COMPOSE2.md](bar3/scale/compose2/COMPOSE2.md)). With bar 3's subgoal score, row check and
  tie-break ported, clean values, restrictions posed where they hold, and the search fixed to test up to 16 spellings
  per state (it had reached the reduction's state and tested one look-alike), bar 3's set replaces book_min's reduction:
  with the other four laws, the round comes at level 5 as x² then the reduction it learned, right on every rung of both
  ladders to T = 64. The subgoal score, its law view, a deeper probe aimed at the subgoal's rest and the spelling fix
  are each needed; the tie-break never acted. Bar 3's set alone still gets nothing: without the multiply-add row, the
  multiply is 9 tokens over the primitives. About $30.
- **Composition, round 3** ([COMPOSE3.md](bar3/scale/compose3/COMPOSE3.md)). Without the book's multiply-add row, bar
  3's set learns the comparison, link and addition but never the multiply, which is 9 tokens over the primitives; none
  of its sources poses a + b·c. Round 2's winning setup, unchanged, gets no round on x³ or 10·x² + 1 mod N: its route
  needs a value one law from the target that is cheap and small, which x² is and the others are not. So round 2's
  success is shaped to x². The next step is a consolidation: fewer, more general rules for where the learner spends its
  search, tested on every earlier win and on formulas no rule was written for. About $8.
- **The search on a GPU** ([GPU_SEARCH.md](bar3/gpu_search/GPU_SEARCH.md)). The same program search as a GPU backend: of
  18,719 stored searches rerun, 18,708 are identical and none differ (11 had hit the CPU's time limit), and two whole
  learner runs replay record for record. Big searches run about 95 times faster, but they cost only 3–10 times less, and
  a whole run only 1.6–1.9 times less, since a third of it waits on CPU work. The same money buys about half to one more
  instruction, and the next level, about 370M states, is past one GPU's memory. So speed shortens the loop, and depth
  still has to come from aiming the search. About $8.
- **Consolidation** ([CONSOLIDATE.md](bar3/scale/consolidate/CONSOLIDATE.md)). Asked whether ablation, general
  mechanisms and scaling could replace the fixes accumulated by watching failures. One score for every choice (bits
  saved per state searched, best first, no fixed pick count) keeps 1 of 16 earlier wins: it stalls, the known failure of
  myopic metareasoning. An ablation over 114 runs finds that two general rules carry 13 of the 16 at their old rounds:
  probe every candidate at one uniform depth, then practise four a round ranked by anchored pairs. The twin, the fifth
  pick, the deeper look and the big-x score were needed for no win, and the subgoal's tie-break only for 10·x² + 1 under
  hardest first. About $50.
- **The target's laws cut to one digit place** ([SLICE.md](bar3/scale/slice/SLICE.md)). A source reads the target's law
  (x² mod N as mod(mul(X, X), N)) and poses each operation cut at one place: the multiply as the row a + x·d, the
  reduction as Horner's step. On stage 3's machine with the book less its reduction and its row, x² mod N then gets the
  row at level 0 and the round at level 5, right to T = 64, with neither the book's row nor round 2's deeper probe. So
  bar 3's pieces now supply two of the book's laws. x³ and 10·x² + 1 get the row and the multiply but no round, since
  both need the whole reduction as a routine first; and without the book's multiply, the multiply never comes, because
  it is the row's loop, not a cut. About $25.
- **Each cut's loop, offline** ([LOOPS.md](bar3/scale/loops/LOOPS.md)). A source poses each cut's loop: the multiply
  from the row, the whole reduction from Horner's step, with practice answers from Python. x² mod N's round then comes
  at level 2 with the book down to the comparison alone, right on both ladders, and with every one of round 2's search
  fixes off. With no book at all the multiply still comes, but nothing poses the comparison. x³ and 10·x² + 1 get the
  whole reduction but no round: the cube needs four calls where the round's search reaches two, and 10·x² + 1 has no
  constant 1. About $93.
- **Bar 3's studies on the GPU** ([GPU_SEARCH.md](bar3/gpu_search/GPU_SEARCH.md), §8–§13). One runner runs any
  registered bar-3 study on the GPU or the CPU. Seven whole runs reproduce record for record, both digits inventions
  among them, and a run takes 5–34 minutes instead of 2.5–3.8 hours at about three quarters of the cost. A cache key
  that ignored the library's order is fixed behind a flag; no committed record was affected. About $17.
- **Bar 3's sources inside a training run** ([LEGAL_SOURCES.md](bar3/LEGAL_SOURCES.md)). An audit of the 14 practice
  sources against the rule that no training target may come from Python's arithmetic. Most read a target table that a
  run does not have: H1's training rows are 1,000 random prompts at full size, never x = 0 and almost never two rows one
  apart. Near-miss values, the row check and the choosing rules are legal as they are; the slice cuts and the rests of
  the way need redesigns that take their answers from the book's constructions or the T = 1 rows. It also raises a new
  question for a judge: may Python arithmetic decide what to practise if it computes no target? No runs.
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
- **No book, offline** ([READING.md](bar3/scale/reading/READING.md)). Round 6 poses the comparison by cutting Horner's
  step down to the link and the comparison, with practice answers from Python. x² mod N's round then comes at level 2
  with no law in the book at all, right on both ladders: the comparison at level 0, the step, the link and the multiply
  at level 1, and the reduction from its loop at level 2. It also comes with the comparison alone and with every one of
  round 2's search fixes off, but not with the link alone, which is 6 tokens over the primitives. Posed as a sub-law,
  x·o mod N comes at level 3 and the cube's round follows as two calls of it, with the book and without. 10·x² + 1
  learns its constant 1 and x + 1 but gets no round even in a 7,200 s run, and anchored pairs' prior brings the row to
  level 0 without mending their picks. Still hand-written: the source's tables of cuts, loops and the step's cuts, and
  which sub-law to pose. About $72.
- **The family with no book, offline** ([FAMILY.md](bar3/scale/family/FAMILY.md)). Round 7 replaces round 6's
  hand-written sub-law with a rule that names no task: every window of the target's reading, two or three operations,
  becomes a law of its own, smallest first. With no book and nothing tuned per task, 6 of 7 family tasks come on stage
  3's tiny core, with practice answers from Python: x² at level 2, the cube at 3, 10y² at 2, y² + 1 at 3, y² + 3 at 4
  and 10y² + 1 at 3, each right on both ladders and at x = 0. y² − 1 comes with two more general changes: posing the 1
  implied by the reader's pred and succ, and keeping a round only if it also matches the data's own answers at T = 2, 4
  and 8, without which the first round kept is wrong whenever an input is 0. The round search never needed more than two
  calls. The changes were found on these same tasks, so this is not a held-out test. About $85.
- **Stage 3's instructions, invented** ([BRIDGE.md](bar3/bridge/BRIDGE.md)). Counting at H1's size would take about
  10^16 steps, but one place of a digit machine is tiny: two digits and what the place below passed up. Cut to one
  place, the self-practice law book gives 18 such slices, and 11 compose back into their law when looped over the
  places. On a counting machine with no digits, carries or wrap-around, the learner, unchanged, invents the select (what
  SEL.S does at one place) and subtract-with-borrow (under two of its three choosing rules), each exactly stage 3's cell
  once compiled to a base-10 table. In the round that Modal's spend limit had cut off, finished later, it also invents
  add-with-carry, by adding, comparing with the base and subtracting once ([CARRY.md](bar3/bridge/CARRY.md)).
  Multiply-accumulate, whose carry must count, comes once a deeper search is aimed at its piece as in the subgoal study:
  a score read off the rows names the piece the target loops over, and in that search a clean program beats a cheaper
  one that uses up N. Hardest first then invents it in round 1, exactly stage 3's cell; the pairs signal does not in 2
  rounds (CARRY.md §12). Montgomery's quotient digit turns out to be a table of one place, so a reduction with nothing
  estimated could be built from these cells and that one table (CARRY.md §13; not built). s3i with the invented select
  in place of SEL.S certifies at H1 (§4.8).
- **Breadth at full size** ([SP_BREADTH.md](integration/sp_breadth/SP_BREADTH.md)). The self-practising stage 3,
  unchanged, learns 2 of 11 tasks of the family; after three general changes to its round's search, 8 of 11 (§4.8).
  More general than before, less than bar 3 asks: each change is stated without a formula, but each was written after
  watching one fail.
- **Within the task family** ([FIXED_MODULUS.md](general_lessons/fixed/FIXED_MODULUS.md)): general lessons v3 learned
  all 14 practice datasets with one fixed modulus, which every earlier file failed, by letting the answers choose the
  whole digit code. These are relaxed-budget runs, and a different sense of "general" from bar 3's: one file across
  datasets, not a learner of new algorithms.

### 4.1 The models at the frontier

Each certified on the untouched official H1 manifest on an H100, with the file run byte for byte: T = 64 on both
ladders, every rung 768/768, test 9,999/9,999, ood_t and ood_n_t 10,002/10,002.

| file (md5) | what it is | learned numbers | official record: training s (frozen at step) / evaluation s | seeds | found or chose |
|---|---|---|---|---|---|
| [`integration/submission_stage3_s3i.py`](integration/submission_stage3_s3i.py) (`0e813144`) | **stage 3**: no skeleton, no menus; the round discovered in training (§4.7) | 207,374 | `s3i_official`: 1,340.3 (870) / 209.9 | 8 of 8 on H100s, official manifest (74 and 1–7): trained in 1,133–1,350 s, frozen by 982–1,149 s | "copy x, multiply, reduce", at step 671 on seed 74 |
| [`integration/self_practice/submission_stage3_sp2.py`](integration/self_practice/submission_stage3_sp2.py) (`afc70fe9`) | **stage 3 with self-posed practice**: s3i with its curriculum replaced by a law book and a general choosing rule (§4.7) | 207,374 | `submission_stage3_sp2_official`: 1,276.7 (887) / 225.5 | 8 of 8 on B200s (74 and 1–7); its first version `submission_stage3_sp.py` (`f0ef5b06`) certified officially and on 7 of 8 | the same round, at step 688, by 1,029 s |
| [`integration/combined_fast/submission_stage3_fast.py`](integration/combined_fast/submission_stage3_fast.py) (`91bb30b9`) | **every certified variant of sp2 in one file, made fast**: the weak-check fix, the strict build, the learned loop over T, the generated law book and the generic instructions (§4.8) | 207,370 | `combined_fast_official`: 751.4 (829) / 362.0 | 8 of 8 on H100s, official manifest (74 and 1–7) | the generated book's round, at step 629, by about 460 s |
| [`integration/combined_fast/submission_stage3_fast2.py`](integration/combined_fast/submission_stage3_fast2.py) (`41b9a013`) | **the same, with its work budgets set from eight climbs**: the file to judge before the legal file ([REVIEW.md](REVIEW.md) §2) | 207,370 | `combined_fast2_official`: 759.3 (829) / 377.3 | 8 of 8 on H100s, official manifest (74 and 1–7) | the same round, at step 629, by about 492 s |
| [`integration/combined_legal/submission_stage3_legal.py`](integration/combined_legal/submission_stage3_legal.py) (`e2fe7a68`) | **fast2 with three laws and bar 3's legal reader**: the generated book replaced by the comparison, the link and the addition, the rest posed from the law the reader finds in the data (§4.8) | 207,370 | `combined_legal_official`: 465.3 (813) / 295.3 | 8 of 8 on H100s, official manifest (74 and 1–7) | the round at level 4, the answer program by 272–349 s |
| [`integration/legal_wide/submission_stage3_wide.py`](integration/legal_wide/submission_stage3_wide.py) (`3bb5ceeb`) | **the legal file with a wider grammar for its reader**: subtract, power, constants and depth 3 beside add, mul and mod (§4.8) | 207,370 | `legal_wide_official`: 554.1 (813) / 391.9 | 8 of 8 on H100s, official manifest (74 and 1–7) | the round at level 4, the answer program by 244–326 s |
| [`integration/legal_nobook/submission_stage3_nobook.py`](integration/legal_nobook/submission_stage3_nobook.py) (`b42ec304`) | **no law in the book**: the wide file with its three laws replaced by a rule that reads from the answers what to practise first (§4.8) | 207,370 | `legal_nobook_official`: 573.9 (829) / 398.3 | 8 of 8 on H100s, official manifest (74 and 1–7) | the round at level 4, the answer program by 256–344 s |
| [`integration/legal_family/submission_stage3_family.py`](integration/legal_family/submission_stage3_family.py) (`457216d7`) | **no book and round 7's rules**: the file to judge before the wider file ([REVIEW.md](REVIEW.md) §2): the no-book file with windows of the reading posed as laws, constants made from N and rounds checked at T = 2, 4 and 8 (§4.8) | 207,370 | `legal_family_official`: 783.3 (831) / 367.6 | 8 of 8 on H100s, official manifest (74 and 1–7) | the round at level 4; byte for byte it also learns y² + y (held out), y² + 1, the cube and y² − 1 at full size |
| [`integration/legal_wider/submission_stage3_wider.py`](integration/legal_wider/submission_stage3_wider.py) (`59060b5c`) | **tidier and wider**: the file to judge before the deep file ([REVIEW.md](REVIEW.md) §2): the family file with ROUNDT's rows in buffers, a cheaper ROUNDT and a reader of polynomials up to three steps (§4.8) | 207,370 | `legal_wider_official`: 522.1 (831) / 271.0 | 8 of 8 on H100s, official manifest (74 and 1–7) | byte for byte it also learns 7 of 8 family tasks, none of the 2 held out |
| [`integration/legal_deep/submission_stage3_deep.py`](integration/legal_deep/submission_stage3_deep.py) (`56acca73`) | **four steps deep, and the file to judge** ([REVIEW.md](REVIEW.md) §2): the wider file with a four-step reader, windows reduced mod N and x kept between calls (§4.8) | 207,370 | `legal_deep_official`: 519.3 (831) / 270.6 | 8 of 8 on H100s, official manifest (74 and 1–7) | byte for byte it also learns all 22 other tasks it was run on, 12 of them held out |
| [`integration/submission_stage1.py`](integration/submission_stage1.py) (`cfd7b61d`) | **stage 1**: the tournament and the general lessons in one file | 222,113 | `s1_official`: 1,860.7 (716) / 56.1 | 8 of 8 on CPU | also learns y² + 1, 10·y², cubing, Montgomery's, 10·y² + 1, y² + 3 and M4 |
| [`general_lessons/submission_general_v3.py`](general_lessons/submission_general_v3.py) (`0805b7d3`) | **general lessons v3**: the answers choose whole digit codes (§4.6) | 11,589 | `v3k_h1_official`: 1,190.5 (645) / 42.3 | H1: 8 of 8 on B200s, run with an earlier v3 build (`v3f`) | also all 14 fixed-modulus practice sets, on relaxed budgets |

The file to judge is now the deep file ([REVIEW.md](REVIEW.md) §2): the legal file taken to an empty law book, with a
reader of four steps, round 7's general rules, and no data rows outside the state dict. It certifies on the official run
and seven more H100 seeds, and the same file, byte for byte, learns all 22 other tasks it was run on at full size, among
them six held out and drawn with its own md5 as the seed (§4.8). Before it, the wider file was the file to judge,
learning 7 of 8 family tasks but neither of its two held-out ones; before that, the family file, learning 4 of 8 and one
of its two held-out tasks; before that, the legal file: fast2 with its generated law book replaced by three laws and bar
3's legal reader, also 8 of 8. Before that, fast2: every certified variant of sp2 in one file, made fast, also 8 of 8.
Before it, s3i was the file to judge; it keeps the most evidence on what each hand-made part does, with its ablation.
sp2's seeds ran on B200s, which count as research, not certification. Two of s3i's checks are weak: in about 3% of runs
its reduction check would accept a shortcut that fails at full size, and such a run cannot certify. None of its 8 seeds
drew it. Copies of s3i and sp2 with both checks fixed replay all 17 of their runs exactly (§4.7), and the variants of
§4.8 go further.

The earlier frontier files are now steps of the ladder (§5): the wide model (step 11), its MLP-cell version (12), the
tournament (13), and general lessons v1 and v2 (14, 15).

**What s3i still writes by hand** (INTEGRATION.md §4):
- **the instruction set**: a general digit machine with 5 registers of digits, a shared pointer, a borrow or carry, and
  32 instructions, plus loops that repeat their body by the number's length or the base;
- **the alphabet and sizes**: base-10 digits, 8, 8 and 2 digits read and 8 written, 16-place registers for the round's
  machine;
- **the cells' shapes**: three tables of logits (subtraction, addition, and "a + x times h + k") and the countdown's gate;
- **the reader and writer**: stage 1's, line for line (24 reading hypotheses, 3 pointer shapes);
- **the lessons**: a subset of the law library, the template's step laws (now practice problems), and the curriculum:
  which laws become practice problems, their order, and their variants;
- **the search and its rules**: breadth-first search by cost with measured caps, BPE merges, Occam's ties, routines
  for solved problems, and the round's search space;
- **the countdown over T and the schedule** (phases A–D, then the freeze).

### 4.2 The calculator underneath: the combined model

**[`learned_remainder/submission_combined.py`](learned_remainder/submission_combined.py)**: md5
`4f5cecc780894499a462499c3cd7d67f`, committed in `9959997`. Write-up: [COMBINED.md](learned_remainder/COMBINED.md).
It was the first model in which every computing part is learned. Its wiring and its warm start were still written by
hand. Every later model up to stage 1 is this calculator with more of it learned; stage 3 keeps its reader, its cells'
idea and its countdown, and replaces its machine with programs found in training.

**What it is.** A small calculator, wired the way a person does this arithmetic on paper. All 5,079 learned numbers
start random and are learned together, in one training run:

| learned part | numbers | what it does |
|---|---|---|
| subtraction table: (digit, digit, borrow in) → (digit, borrow out), 200 rows | 2,400 | every subtraction, and every addition too, as a + b = a − (0 − b) |
| times table: (digit, digit) → (low digit, high digit), 100 cells | 2,000 | schoolbook squaring |
| commit gate: borrow → keep or discard | 4 | every keep-or-discard decision, in the division and in the countdown |
| reader and writer | 675 | which tokens are digits and which digit each one is; where each number starts and which way to read it; where and how to write the answer |

**The example row, run through it.**
1. **Read.** The reader learned to start at the N marker and step right, since N's digits come lowest first, giving
   N = 4189. It reads x by starting at the T marker and stepping left (1385), and T from the end of the prompt (1).
2. **Square.** It writes one row of digit products per digit of 1385, using the times table, and adds the rows with
   the subtraction table: 1,918,225.
3. **Divide.** It brings down one digit at a time and subtracts 4189 again and again, with the subtraction table.
   The gate discards the subtraction that goes below zero, which it sees as a final borrow. In the early steps 4189
   does not fit at all. Then it fits into 19182 four times (2426 left over), into 24262 five times (3317), and into
   33175 seven times (3852).
4. **Count down.** It subtracts 1 from T's digits with the same table. 1 − 1 = 0 leaves no borrow, so the round runs.
   In the next round 0 − 1 borrows, so the gate holds. Once no prompt in the batch wants another round, the loop ends.
5. **Write.** The writer turns 3852 back into tokens.

**How it learns.** Mostly not from answers. Most lessons are checks on each training row's own numbers that need no
answer:
- subtract N, add it back, and the start must come back;
- x < N, so x − N must borrow and N − x must not;
- x × (b + 1) = x × b + x;
- round trips through the reader and writer.

The answers of the T ≤ 2 rows join late, and only along the machine's own path.

**Its official runs** (Modal H100, the official manifest untouched, seed 74; records in
`history/runs/learned_remainder/`):

| record | what ran | max T, seen / unseen moduli | rungs | test, ood_t, ood_n_t | training s (frozen at step) | evaluation s |
|---|---|---|---|---|---|---|
| `combined_official` | the file byte for byte | **64 / 64** | 768/768 at every rung of both ladders | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 666.6 (545) | 48.2 |
| `combined_index_official` | the file with `PATH = "index"` (the older integer prediction path), the only line changed | 64 / 64 | all 768/768 | all exact | 566.5 (545) | 23.9 |
| `combined_s0_official` | `MODE = "designed"`: every table set to the truth, a plumbing check only | 64 / 64 | all 768/768 | all exact | 143.0 (frozen from the start) | 34.3 |

- **CPU seeds:** the file byte for byte, through the same evaluator with `device = cpu`, certified on 8 of 8 seeds (74
  and 1–7), each on both ladders, every rung 768/768.
- **Inside autograd (rule 8).** At evaluation every digit is a one-hot vector, and every table acts as a
  straight-through one-hot matrix. So the logits are exactly the calculator's, while a gradient from the final logits
  reaches all 10 learned tensors (tested in [COMBINED.md](learned_remainder/COMBINED.md) §3).

### 4.3 The wiring, learned: short menus, wide menus, then a tournament

**Short menus** ([WIRING.md](history/learned_wiring/WIRING.md), steps 7a–7c in §5). Every connection the combined model
wired by hand became a switch: a small menu of options, one of which is right. The division got 7 switches (864
wirings), the squaring's layout 2 placement rules of 24 each (576 layouts), and the reader and writer one shape switch
of 3. Each switch is taught by a check that only the right wiring passes, run on the rows' own numbers. For the
division, every combination of options is tried on the snapped machine, and the combinations that pass gain weight.

**Wide menus** ([MENUS.md](wider_menus/MENUS.md), step 10). 4.1 billion division wirings instead of 3,456, and 33
million multiply layouts instead of 576: about 4·10^17 whole machines with the reader's 3 shapes. The menus hold
genuinely different algorithms, and the machine picks among those that work: two long divisions (repeated
subtraction, and trial multiples as on paper), six groupings of the products, and a Montgomery reduction that the
checks reject.

**The tournament** ([TOURNAMENT.md](history/tournament/TOURNAMENT.md), step 13). The chain's connections stop being a
menu of meaningful parts: each operand may be any of 7 registers moved up or down by up to 3 places, and so on, 7.9·10^9
chain wirings. Inside ordinary training the model holds one weight per whole combination of a block of connections,
scores each pass or fail every step, and takes one gradient step on −log (the total weight of those that passed). In
tiny machines it found the one right wiring among 5.3·10^10 in 8 of 8 runs; at H1 two new lessons closed two gaps the
training data cannot see (the register's unused top place, and a gate whose meaning could flip).

### 4.4 Digit meanings from the order of the numbers

Write-up: [ANCHORS.md](anchors/ANCHORS.md). The subtraction table used to learn what its digits mean from a
hand-written adder, used for its first steps; the ablation showed that this warm start was load-bearing. The anchors
replace it:
1. **The borrows, from x < N on every "top part".** Take x's digits from some place upward, and N's digits from the
   same place. The first number is at most the second, and strictly smaller exactly when the two digit strings differ.
   So taking N's part from x's part must borrow exactly then, and taking x's part from N's never borrows.
2. **The digits, from two laws of subtraction**, applied to one digit at a time once the borrows are right: taking the
   same digit away from two digits keeps their order; taking more away from the same digit leaves less; except when
   exactly one of the two goes below zero, which wraps round and comes out on top. In each column the laws put the ten
   digits in one strict order, and ten digits in a strict order can only be 0 to 9 (ANCHORS.md §2.3).

This was certified on the single-piece machine (step 6b) and the combined model (step 8), then put into the wiring
model (step 9) and the wide-menu model (step 11), and every later file inherits it.

### 4.5 Generic cells: MLPs certify, ReGLU only sometimes, local shader cells not from lessons

**The bridge** ([BRIDGE.md](history/bridge/BRIDGE.md)). The calculator is one weight setting of a generic looped
transformer: attention heads copy digits between the tokens of a worksheet, one small MLP does the digit arithmetic, and
the block runs until every token settles. Loosened one notch at a time and trained from random: the digit tables can be
generic MLPs if the cell gets the anchors' lessons (31 of 32 tiny runs certified); one MLP doing all three rules
certified in 4 of 16; the wiring learns only from short menus (up to 10 options per connection), and never with free
attention, which falls for the bypass. At H1, the wide model with both tables as generic ReLU MLPs (step 12) certified.

**The handwired bridge** ([HANDWIRED_BRIDGE.md](history/handwired_bridge/HANDWIRED_BRIDGE.md)). The same machine
compiled by the user's handwired project into weighted sums and ReGLU products, with no attention: its table rows train,
its switches do not; the wiring learns only from short menus (7, 10 or 13 options, by switchboard); and the bypass is
just as strong, so its pull comes from the data. At H1 the wide model with ReGLU cells did not certify on the official
run (seed 74 fails deterministically); on CPU it certified on 1 of 4 seeds.

**The NCA: long division as a shader** ([NCA.md](history/nca/NCA.md)). A grid of identical tiny cells, each running the
same small learned rule on its 3 × 3 neighbourhood, tick after tick, so that the loops come from repetition alone.
- **The format can do it exactly**: one 56-unit rule at base 3 computes the division at every size checked, and 242
  units do x² mod N.
- **Taught by lessons, it memorizes**: 84 runs in 41 variants, with every lesson the format can express, and none
  certifies. The cells' own arithmetic laws make their table exact, but the chains of copies that carry N, the old
  register and the borrow through the grid never form: no local check reaches a copy.
- **Trained long with weight decay, or with agreement, it still memorizes** (NCA.md §9). Two known cures for
  memorizing were tried, training far longer than the first probe did: *trimming*, weight decay over a long run, the
  setting in which networks that had memorized modular arithmetic later found the general rule ("grokking"); and
  *agreement*: the AND-mask moves a weight only where groups of questions (or the grid's columns) push it the same
  way, and delta-bar-delta speeds a weight up while its gradient keeps its sign. The arms: plain Adam as the control;
  AdamW at decays 0.1, 0.3 and 1; the AND-mask over 8 groups of questions and delta-bar-delta, each alone and with
  decay 0.3; and at the smallest size the AND-mask over the grid's 4 columns, alone and with decay 0.3. That is 72
  runs from random weights, 4 seeds per arm: 50,000 steps at the smallest size (base 3, two-digit N, one quotient
  digit: 99 questions) and 20,000 at the next (three-digit N, two quotient digits: 2,835 questions, the grid in which
  every local situation occurs). 20 more runs, 2 per arm at the smallest size, started at the exact rule.
  - **None certifies**, and none fits its training questions with the digits snapped: the best any run reaches is 62
    of the 99, and about a seventh of the 2,835. On a grid one size bigger, every run stays at chance from the first
    check to the last: the most any check gets above "always answer 0" is 0.07 (0.25, where always answering 0 gets
    0.18). There is no late jump.
  - **Why decay fails here.** Training runs on blurred digits, each a mix of its possible values. Decay holds the
    weights small (a norm of about 17–20, 10–13 and 6–7 at decays 0.1, 0.3 and 1), small weights keep the digits
    blurred, and at the smallest size a blurred code fits the training answers, unsteadily, without being the
    machine. The exact rule needs a norm of about 37 there to keep its digits sharp tick after tick. In grokking the
    general rule was the one with the smaller weights; here it has the larger ones, so decay pushes away from it.
  - **Started at the exact rule, every arm loses it** within a few hundred steps, the plain control included: a rule
    the training cannot hold is not one it can find. NCA.md finds two pushes off it: blurred neighbours of the rule
    meet the lessons as well as the rule does, and Adam, under every arm, keeps taking full-size steps even where
    every lesson is met.
  - **Agreement does not build the copies either**: the AND-mask lets a few percent of the gradient through, and
    delta-bar-delta leaves most weights near its slowest speed. At the larger size no run fits even the blurred
    answers (0.26 at best), so trimming's "fit first, then decay" never starts.
- **With an optimizer that can rest at the rule, it keeps the rule but still does not find it** (NCA.md §10). Adam
  moves every weight by about half its learning rate even where every lesson is met; Adam with eps 1e-4 and plain SGD
  do not. They were tried in §9's setting at the smallest size, 4 seeds per arm, with and without weight decay:
  - **started at the compiled rule where every lesson is met**, both keep it for the 10,000 steps and certify, 4 of 4
    seeds each. Plain Adam leaves it within 100 steps, and so does every arm with weight decay, within 1,100;
  - **where the lessons still pull**, plain SGD at 0.01 keeps every training answer, but one size up it slips to 0.70
    of the answers, and 99.7–99.9% of its one-tick errors are in situations the training grid never shows. At the next
    size, where almost every situation occurs, SGD holds the rule until it loses all of it within 100 steps, on every
    seed;
  - **from random, every arm still memorizes**: one size up no run gets beyond chance. 13 of these 24 runs, cut off
    by Modal's spend limit at 18,000–46,500 of their 50,000 steps, were finished from their checkpoints once it was
    raised; only numbers read at a run's end moved.
- **Told every move** (the exact rule's next state at every cell and tick), it learns the training grid in a few
  hundred steps. With information flowing one way and a training grid that shows every local situation, it also
  certifies on bigger sizes. That help is out of bounds for the contest: producing the moves needs the algorithm.
- **The square** does not extrapolate even when told every move. The logic-gate arm and the all-to-all generator arm
  fail from lessons, and the generator arm brings the bypass back.

### 4.6 General lessons, the litmus test, and one fixed modulus

**The litmus test** ([LITMUS.md](litmus/LITMUS.md)). The wide model, unchanged, on 26 other tasks: it learned H1 with
its prompt's fields reordered and 5 of the 6 practice datasets whose moduli vary; it failed the 14 practice datasets
with one fixed modulus, four other formulas, and a prompt with four numbers. Its verdict: "a method within its family;
an engineered solution across families".

**General lessons** ([LESSONS.md](general_lessons/LESSONS.md), steps 14–15). One file, unchanged on every task: laws
stated once about an operation; facts the model finds in its own rows ("x < N" is found, not told); one template for a
division step in two versions, long division and Montgomery's reduction, each training its own machine while the
answers pick which one answers; and one menu of extra round steps. With zero per-task changes it learned y² + 1, 10·y²,
cubing, Montgomery's squaring, 10·y² + 1 and y² + 3. v2 reads the answers at the batch's smallest T, which fixed M4,
whose rows all have T = 8.

**One fixed modulus** ([FIXED_MODULUS.md](general_lessons/fixed/FIXED_MODULUS.md), step 18). With one modulus the fact
"x < N" only ever compares a digit with one of N's own, so 16,800 to 2,963,520 whole candidates (a reading and a digit
code) fit every fact. The answers single out the true one in 13 of the 14 datasets; in E1, whose N = 323 reads the same
both ways, they pin the code but not which way N is written. They do so only for a model that tries whole candidates.
v3 does:
- the reader's own ranking teaches the table's borrows on pairs from different rows, so the machine is exact under any
  provisional code;
- then a tournament over whole (reading, code) pairs, with every lesson set and round option, keeps only the
  contestants that reproduce every answer of a batch;
- choices made under the provisional code are kept revisable, and the multiply's column rules are scored whole.

All 14 fixed-modulus datasets learned, on relaxed budgets, 17 of 17 runs: 100% on each dataset's own test and
out-of-distribution splits. E1, E10, M1, M2, M7 and M8 also certify T = 64 on both depth ladders. The other eight
reach T = 8: their training T always has one digit, so nothing tells "16" from "61". The README calls that ladder a
diagnostic for the practice tiers, so it changes no score. On H1, v3 runs step for step as v2 does and certifies.

### 4.7 Discovering the skeleton: library learning, then stage 3

**Library learning** ([LIBRARY.md](library/LIBRARY.md)). Offline, search for short programs over 32 digit instructions
on 26 practice tasks, merge the groups of instructions that keep appearing in working programs into new instructions
(BPE, or minimum description length), and search again. Level by level it rediscovered the skeleton's parts (the
ripple, the restoring link, the chain, the division step, the reduction loop, the multiply rows), then x² mod N itself,
22 primitives long, exact on every rung of both H1 ladders.

**The integration** ([INTEGRATION.md](integration/INTEGRATION.md), steps 16–17).
- **Stage 1** put the tournament and the general lessons in one file, and certified.
- **Stage 2** ran the library inside one training run, with learned tables. As first built its climb never found the
  multiply; two fixes (test every candidate's result, and pose the multiply's row as a problem of its own) did.
- **Stage 3** has no skeleton and no menus. In one training run the model reads the prompt, learns three digit tables
  from laws, climbs a library of programs on a general digit machine from practice problems made of its rows' own
  digits, and finds the round among short combinations of its routines that reproduce the training answers:
  - the multiply, as a loop of its own accumulating row;
  - the reduction, as a loop of its own long-division step;
  - x² mod N as "copy x, multiply, reduce", at step 671 on seed 74.

  Then it runs that program T times. The first build to discover the round in-run (s3d) needed 9,142 s and certified
  only on a relaxed budget. The final file, s3i, certifies on all 8 seeds tried within the hour, its freeze by 1,149 s
  at the latest. How hard each build was pushed is in §5.

**The ablation** ([ABLATION_S3.md](integration/ABLATION_S3.md), records in `integration/ablation_s3/records/`). Each of
s3i's hand-made parts was removed in turn, 13 arms and the unchanged control, each retrained on the official manifest
on one H100, on seed 74 and again on seed 1. On all 13 the two seeds agree. Seven arms never found the round:
without SEL.S or BRG.X; without the step's two practice problems; with mac posed as "x times a digit, plus N"; without
the twins; without routines; without Occam's tie-break. Each failure is the climb running out of things to try,
between 2 and 34 minutes into training, after every search for the missing piece had covered every program up to
cost 5. For the first six, a search one cost level wider found nothing either (ABLATION_S3.md §7); the seventh
failed on a weak check, and certifies once it is fixed (below). The other six arms certified, frozen by 1,584 s at
the latest. Five more arms followed on seed 74, two of them also on seed 1:
- **without the chain problem alone**, the run certified on seed 1 but not on seed 74. The long-division step then
  has two equally cheap spellings, one leaving its result in X and one in Y, and the search keeps whichever it lists
  first. Only the one on Y lets the reduction loop over it, since X still holds the digits to bring in. With the
  chain problem, the reduction does not depend on that tie;
- **without the step problem alone**, it certified on seed 1, three steps early, and failed on seed 74 on the weak
  check below; with the check fixed, seed 74 certifies too, three steps early;
- **without the multiply's row problem**, it certified a step early;
- **with the round's cost limit at 6**, nothing changed; **at 4**, there was no round, since every round program
  found costs 5.

**Two weak checks, and a fix** (ABLATION_S3.md §8). s3i accepts a reduction once it is right on 60 numbers at each
of 1, 2 and 3 digits. A program that subtracts N over and over, 600 times at 3 digits, passes whenever none of those
60 three-digit numbers needs more. With N = 123 it brings 5 + 600 × 123 = 73,805 back to 5, but leaves
5 + 601 × 123 = 73,928 at 128, and it is wrong on every full-size product. On H1's rows a run's check lets it through
about 3% of the time ([`reduce_check.py`](integration/ablation_s3/reduce_check.py)). Nothing can be built on it, so
that run cannot certify. None of s3i's 8 seeds drew it; the step problem's arm did, on seed 74. The conditional
subtraction's check is weak too: it tries x below N and x = N, never x between N and 2N, so "x if x < N, else 0"
passes it. Occam's tie-break is what kept s3i off that program.
- **The fix** only adds the probes each check lacked: full-size ones for the reduction, built by doubling with the
  learner's own addition, and ones between N and 2N for the conditional subtraction. It is in new copies of s3i and
  sp2 ([`ablation_s3/fix/`](integration/ablation_s3/fix/)).
- **Both copies certify** on the official manifest and on their originals' seeds, 17 runs, every record equal to the
  original's field for field: the old checks had never let a wrong program through in those runs. The reduction
  check's 3% falls to about 10^-11.
- **On the fixed copy**, the arm without Occam's tie-break certifies on both seeds, at the control's steps, and the
  arm without the step problem on seed 74. So neither part is needed.
- The fix is in copies only: s3i and sp2 themselves, and the files built the same way, keep the old checks.
- **Eight more certified files** share the two checks: the strict builds, the learned loop, the bridge, sp_breadth's
  g3, the law ablation's two books and generic_ops' g3. Copies of them with the same fix certify wherever the
  originals did, over 78 runs, and 75 runs equal the original's. In book_gen's seed 1 the fixed check rejected a
  wrong conditional subtraction that the old one had let into the corpus BPE merges from: three fewer merges, and
  the same routines, round and evaluator fields. The learned loop's seed 2 and sp_breadth's M4 ran on hosts of a
  different speed and stopped at a different step. g3's acceptance at four digits had not closed the reduction's
  check: a loop of 10,000 subtractions passes it at cost 5, as long division does. Its fixed copy rejects the loop.

**Self-posed practice** ([SELF_PRACTICE.md](integration/SELF_PRACTICE.md), step 19). The learner poses its own
practice problems: every law of an 11-law book (compare, equal, min, max, remainder, quotient, add, subtract, multiply,
multiply-add, add three) applied to numbers built from the data's own N and x in 8 ways, with the inputs in every
placement in the registers. That is about 70 problems at the start and 36–180 at each later level, most of them
useless. The rule that works treats a law on one pattern as one concept, rates it by its cheapest placement, and
practises it once in its cleanest placements. Bar 3's own choosing rules (hardest first, the cheaper pairs signal)
climbed 7 to 13 levels without the round. The certified run practised all nine of s3i's problems by level 4, in the
order their constructions allow, plus some s3i never had. It found its round sooner than s3i did on the same day:
by 746 s (`sp`) and 1,029 s (`sp2`), against the control's 1,047 s. The other formulas are still not learned: the
limit is in the round's search and the law book, not the practice. All told it took 43 tiny-core runs, 6 relaxed
in-run attempts and 27 evaluator records, about 40 GPU-hours. Which of its laws, patterns and rule parts it needs is
in §4, bar 2 ([SP_ABLATION.md](integration/sp_ablation/SP_ABLATION.md)).

**The inventing test** ([INVENTING.md](library/INVENTING.md)) took our inputs away from the search-and-compress loop
one at a time, on the tiniest x² mod N. Without our instruction set, on bare counters, it still invented x² mod N, but
only with practice tasks whose short solutions are the pieces; without the practice tasks it never did. Given the one
step "add one, wrap around at N", the search found x² mod N in about a second. Bar 3's two studies (§4, bar 3) start
from there.

### 4.8 Variants of stage 3 that go further

Studies on Oct 24 changed s3i or sp2 one more way each. Each file in the table certified on the untouched official
manifest on an H100, byte for byte: T = 64 on both ladders, every rung 768/768, every split exact. Their B200 seeds
count as research, as sp2's do.

| file (md5) | what changes | official record: training s (frozen at step, by s) / evaluation s | other seeds certified |
|---|---|---|---|
| [`integration/ablation_s3/fix/submission_stage3_s3i_fix.py`](integration/ablation_s3/fix/submission_stage3_s3i_fix.py) (`09a8263c`) | s3i with its two weak checks fixed (§4.7) | `submission_stage3_s3i_fix_official`: 1,280.9 (870, by 1,091) / 189.5 | 7 of 7 more H100 seeds, each record equal to s3i's |
| [`integration/ablation_s3/fix/submission_stage3_sp2_fix.py`](integration/ablation_s3/fix/submission_stage3_sp2_fix.py) (`a92d6281`) | the same fix in sp2 | `submission_stage3_sp2_fix_official`: 1,264.1 (887, by 1,061) / 213.2 | 8 of 8 on B200s, each equal to sp2's |
| [`integration/learned_loop/submission_stage3_ll.py`](integration/learned_loop/submission_stage3_ll.py) (`419c15e7`) | sp2 with the loop over T learned | `ll2_official`: 1,191.6 (810, by 1,002) / 195.7 | 7 of 8 on B200s |
| [`integration/strict/submission_stage3_s3i_strict.py`](integration/strict/submission_stage3_s3i_strict.py) (`11baebb4`) | s3i with work budgets instead of the clock, and its state in buffers | `s3is_official`: 1,278.0 (870, by 1,069) / 202.1 | 7 of 7 more H100 seeds, each climb s3i's line for line |
| [`integration/strict/submission_stage3_sp2_strict.py`](integration/strict/submission_stage3_sp2_strict.py) (`1a1f15fb`) | the same for sp2 | `sp2s_official`: 1,377.4 (887, by 1,174) / 328.6 | 8 of 8 on B200s, line for line |
| [`bar3/bridge/submission_stage3_bridge.py`](bar3/bridge/submission_stage3_bridge.py) (`85ff74c5`) | s3i with 36 instructions: a select bar 3's learner invented, in place of SEL.S | `submission_stage3_bridge_official2`: 1,838.6 (870, by 1,690) / 281.1 | 8 of 8 on B200s |
| [`integration/sp_breadth/submission_spb_g3.py`](integration/sp_breadth/submission_spb_g3.py) (`30efec64`) | sp2 with three general changes to the round's search | `g3_h1_h100`: 1,242.0 (887, by 1,012) / 194.3 | 8 of 8 on B200s; 7 more tasks learned (one H100 each) |
| [`integration/sp_ablation/variants/book_min.py`](integration/sp_ablation/variants/book_min.py) (`e4732f09`) | sp2 with a book of 5 laws on 4 patterns | `book_min_h100_seed74`: 798.8 (888, by 614) / 179.4 | 8 of 8 on B200s |
| [`integration/sp_ablation/variants/book_gen.py`](integration/sp_ablation/variants/book_gen.py) (`8bbac4a1`) | sp2 with its book generated from a grammar | `book_gen_h100_seed74`: 2,428.0 (915, by 2,228) / 182.7 | 8 of 8 on B200s |
| [`integration/generic_ops/submission_stage3_g3.py`](integration/generic_ops/submission_stage3_g3.py) (`e6bd2a1f`) | sp2 with an if block and carry-passing shifts in place of SEL.S and BRG.X | `submission_stage3_g3_official`: 2,770.1 (897, by 2,572) / 386.0 | 8 of 8 on B200s |

**The loop over T, learned** ([LEARNED_LOOP.md](integration/learned_loop/LEARNED_LOOP.md)). sp2 repeats its round by a
hand-written countdown: subtract 1 from T, and while that does not go below zero, run the round again on its own
answer. The learned-loop file drops the countdown, its gate, and the round search's assumption that the answer at
T = t is the round run t times. Instead the machine gets one generic instruction, `C.r[body]` (run the body as many
times as register r holds), and T goes into a register as x and N do. Once the round is found, a second search,
against answers at several values of T, finds `C.S[T.round1 STA.X]`: run the round T times, each answer fed back as
x. That is the countdown, found rather than written.
- It certifies on the official manifest and on 7 of 8 B200 seeds. Seed 2 runs out of time in sp2's climb, before any
  loop is searched, and certifies on a doubled budget.
- The time-out is the random start's, not the loop's (§3): a diagnostic copy with sp2's gate put back as dead weight
  replays sp2's climb line for line on seeds 74 and 2, and certifies on both, seed 2 frozen by 1,062 s.
- Still designed: the count loop itself (a strict judge can call it the countdown made into an instruction), T's
  2-digit field, which is what tells T apart from x, and the answer program's search space.

**Strict builds** ([STRICT.md](integration/strict/STRICT.md)). The gray points that can go without changing what is
learned are gone, by exact edits to the certified sources:
- no decision reads the clock: each search, and the climb as a whole, gets a budget of machine work, counted by the
  search's own cost model and the same on every host;
- the rows the climb remembers and the round program it finds are buffers: in the state dict, counted, on the GPU.
  The forward pass runs the round from there;
- the optimizer resets itself at phase B, instead of the scheduler doing it, and the late table lessons and the
  gate's check of the wrap-around fact are gone.

All 17 runs certify (s3i's official run and seven H100 seeds; sp2's official run and eight B200 seeds), and none hit a
budget. Each climb is the original's line for line, so the changes alter nothing that is learned. Unchanged: training
still returns zero logits, because the real answer inside autograd on a training batch does not fit in an H100's
memory (512 rows at T = 16 ran out of 80 GB); the damped gradient; the reader; and the climb, which is the learning
method. The strict files keep the old weak checks; their fixed copies replay all 17 runs (§4.7).

**s3i on an invented instruction** ([BRIDGE.md](bar3/bridge/BRIDGE.md) §5). The bridge file is s3i with 36 instructions:
subtract, add and multiply-accumulate at the pointer against X, Y, N or S, their tables learned as s3i learns them, and
the select bar 3's learner invented (§4, bar 3), its table the compiled counting program, in place of SEL.S. Only the
conditional subtraction changes: compare x with N, then at every place take x's digit back if the compare borrowed. The
round comes at s3i's step on 7 of the 8 seeds and one step earlier on the eighth. Training takes 37% longer on the
official H100 (1,839 s against 1,340 s), since each search holds more states. The ablation found SEL.S needed; here a
select the learner invented takes its place. Add-with-carry and multiply-accumulate were invented later too (CARRY.md),
each exactly its cell's table, but the instruction list is made before training: the run does not invent its cells.

**Breadth at full size** ([SP_BREADTH.md](integration/sp_breadth/SP_BREADTH.md)). sp2, changed in nothing, learns 2 of
11 tasks of the family: H1, and H1 with the prompt's fields reordered. On the seven other formulas on H1's prompts, its
climb builds every piece of H1's round, but the round's search composes at most two routine calls and cannot make a
number. Three general changes, all in that search and none naming a formula:
1. three routine calls instead of two;
2. a constant read off the answers, when every answer differs from a program's result by the same number mod N;
3. a round must also reproduce the answers of the other one-digit T's.

With all three, 8 of 11 tasks are learned, each in one run on one H100 with every split exact: H1, the reordered
prompt, 10·y², y² + 1, y² + 3, 10·y² + 1, y² − 1, and M4 (H1's formula, every T = 8). Not learned: y³ (four calls,
past the search's three), Montgomery's y²·10⁻⁸ (no law poses its reduction), and a prompt with a fourth number (the
reader takes three).

**A smaller book, and a generated one** ([SP_ABLATION.md](integration/sp_ablation/SP_ABLATION.md) §5). The smaller book
keeps only what the law ablation found needed on seed 74: 5 laws on 4 patterns. It leaves out x + d·N, which sp2's book
needs on seed 1, and still certifies on seed 1 by writing the chain inline. Its climb is nothing but the round's pieces,
one a level: compare, subtract-if-it-fits, add, the multiply-add row, the multiply, the reduction. It found its round
sooner than sp2 on the official manifest (571 s against the same study's 977 s). The generated book is a grammar instead
of a list: numbers r + q·b from the division law, the machine's arithmetic and every composition of + and ×, every
operation on every pattern, and one set of target rules. It holds sp2's book and 24 concepts more, and finds the round
by routes sp2's book does not offer, the square as "0 plus x times x" among them. It is slower (frozen by 2,228 s), but
within the budget.

**Generic instructions** ([GENERIC_OPS.md](integration/generic_ops/GENERIC_OPS.md)). SEL.S becomes
an if block, `C[...]`, whose body's effects stay only where the borrow was set; BRG.X becomes shifts that pass the
digit they move through the multiply's carry digit. No single instruction does either old job, so the learner builds
the conditional subtraction, the chain and long division's "bring down" itself.
- Replacing SEL.S alone needs no other change. Replacing BRG.X too makes long division one step longer, and it took
  three rules, each stated for every law but found by watching whether the round came: closeness counted without the
  reduction's fakes, acceptance at four digits, and a preference for routines that leave the other registers alone.
- `submission_stage3_g3.py` (`e6bd2a1f`) certifies on the official manifest and on 8 of 8 B200 seeds, each run finding
  its round after level 4. It is the slowest file in the table: on the official H100 the round came by 2,507 s and
  training took 2,770 s, against sp2's 1,029 s and 1,277 s, which leaves 830 s of the budget. A first official run,
  stopped by Modal's spend limit before it wrote a record, had taken the same steps.
- What the learner built: the conditional subtraction as "subtract, and reload the input if it borrowed", the chain as
  "subtract, and add back if it borrowed", long division with the two shifts as its "bring down", and, new to stage 3,
  a shift-and-add multiply.
- Still designed: the if block and the shifts, which are generic in kind but chosen, and the rule changes. A judge may
  say `SHL.X SHL.A` is BRG.X in two pieces; the write-up agrees, and notes that it is the form a place-by-place law
  gives BRG.X (GENERIC_OPS.md §7).

**All of them in one file** ([COMBINED.md](integration/combined_sp2/COMBINED.md), Oct 25, step 2 of
[LEGAL_PLAN.md](history/LEGAL_PLAN.md) §4). `integration/combined_sp2/submission_stage3_combined.py` (`11886419`) has
five of the variants above: the weak-check fix, the strict build, the learned loop, the generated book and the generic
instructions. It is built from the certified files by exact, asserted edits, a rebuild gives the same bytes, and every
study's local checks pass on it. It is 869 bytes under the 256 KiB limit.
- Round 1 does not certify. On the official H100 the hour ran out in level 4 of the climb, in the deep probes that look
  for the reduction (training 3,713 s, 623 steps; test 7 of 9,999). In the tiny core the climb ran out of the strict
  build's work budget at the same stage.
- The time goes to two pieces together. In the tiny core, level 3 takes 225 s in sp2, 559 s with the generic
  instructions and their rules, 363 s with the generated book, and 2,585 s with both: 2.9 times what multiplying the two
  costs would give. Most of it is setting up acceptance checks. The book poses 9 laws on each of x + o·N and x + x·N, in
  12 bindings each; at four digits each probe takes up to 9,999 of the learner's own additions to build; and each law
  builds the same probes again.
- Without the generic instructions it fits, barely. With their settings set back to sp2's, the tiny core reaches the
  round after level 5, at 1,695 s, and the learned loop finds `C.S[T.round1 STA.X]`. But that climb uses 93% of the
  strict work budget, which was set from sp2's smaller climbs.
- Round 2 ([COMBINED.md](integration/combined_sp2/COMBINED.md) §8): those three fixes, in
  `submission_stage3_combined2.py` (`4e0e3f09`). Sharing the setup changes nothing in the climb, doubling changes its
  random draws, and the budget comes from the combined climb. Level 3 drops from 2,585 s to 440 s, and the tiny core
  reaches the round after level 4, at 3,206 s, with the learned loop's `C.S[T.round1 STA.X]` right on every unseen row.
  That projects to 3,400–3,700 s on the H100, past the hour, so there was no official run. Level 4's second deep probe
  now spends about 800 s on the book's second reduction, which the round does not use; whether to change that rule is a
  judge's question. About $21 for both rounds.
- Round 3 (COMBINED.md §9): two files of four pieces each certify on the official H100, one without the generated book
  (`c7c711ea`, frozen by 2,113 s) and one without the generic instructions (`682f2e85`, by 2,035 s). Given two hours,
  round 2's five-piece file learns H1, T = 64 with every split exact, but freezes 531 s past the hour.
- **Made fast, it certifies** ([FAST.md](integration/combined_fast/FAST.md)). A profile showed the climb's time going to
  launching tiny GPU kernels, not to arithmetic. A Triton kernel that runs every row's whole instruction sequence in one
  launch, the one-primitive groups run through the same kernel, and one GPU walk to list the goal states make the tiny
  core's climb 7.7 times faster, with every search, pick and unit of work unchanged.
  `integration/combined_fast/submission_stage3_fast.py` (`91bb30b9`) certifies on the official H100 in 751 s of
  training, and its climb is the five-piece file's own, line for line. Seeds 1–7 certify too, 8 of 8 on H100s. On three
  of them the answer program's search crossed the climb's work budget after it had found its programs, so it was rebuilt
  with the budget set from all eight official climbs. That rebuild, fast2 (`41b9a013`), certifies 8 of 8 too and was the
  file to judge until the legal file (REVIEW.md §2); the measurements showed the old budget had never cut anything.
- **The legal file** ([LEGAL_FILE.md](integration/combined_legal/LEGAL_FILE.md)). fast2 with its generated law book
  replaced by three laws on the pair, the comparison, the link and the addition, plus bar 3's legal reader: once those
  three are routines, the reader finds the target's law, mod(mul(x, x), N), in the answer rows using only the learner's
  own routines, and poses the multiply-add row, Horner's step and their loops. `submission_stage3_legal.py` (`e2fe7a68`)
  certifies on 8 of 8 official-manifest H100 runs and trains in 465–721 s, faster than fast2. Still designed: the
  reader's grammar, the posing rule and the three laws.
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
- **Round 7's rules, legally** ([FAMILY_FILE.md](integration/legal_family/FAMILY_FILE.md)). The no-book file with round
  7's general pieces, each answered by the learner's own routines: every window of two or three operations in the
  reading posed as a law, constants made from N with add, sub and mul, and a round kept only if it also gives the data's
  answers at T = 2, 4 and 8. `submission_stage3_family.py` (`457216d7`) certifies on 8 of 8 official-manifest H100 runs,
  training in 562–986 s, most of the extra time going to that check, and was the file to judge until the wider file
  (REVIEW.md §2). The same file, byte for byte, one H100 each on its own data, learns y² + y, y² + 1, the cube and
  y² − 1, each right to T = 64 on both ladders with every split exact. It does not learn y² + 3, 10y², 10y² + 1 or
  3y² + 2: the reader cannot read 10y² (a factor, not a gap) or the depth-4 laws 10y² + 1 and 3y² + 2, and y² + 3 never
  learns to make 3 within the climb's work budget. y² + y and 3y² + 2 were held out, chosen before any run; round 7's
  rules were found on the other six. About $40.
- **Tidier and wider** ([WIDER.md](integration/legal_wider/WIDER.md)). The family file with ROUNDT's rows in registered
  buffers, so that no data row is kept outside the state dict, a cheaper ROUNDT that keeps the same rounds, and a wider
  reader: x's polynomials up to three steps, each constant posed with 1 and its doublings, and a law too big for a size
  counted as saying nothing there. `submission_stage3_wider.py` (`59060b5c`) certifies on 8 of 8 official-manifest H100
  runs, training in 509–777 s, and was the file to judge until the deep file (REVIEW.md §2). Byte for byte it learns 7
  of the 8 family tasks, y² + 3, 10y² and 10y² + 1 among them, but not 3y² + 2, whose climb runs out of work. All eight
  were known when the reader was designed, so they show no generality. The two held-out tasks, drawn at random from a
  space stated before any run, 8y³ + 8y + 3 and 2y³ + 9y + 9, are neither read nor learned: the reader stops at three
  steps and they need four. Across both rounds, 1 of the 4 held-out tasks is learned. About $43.
- **Four steps deep** ([DEEP.md](integration/legal_deep/DEEP.md)). The wider file with a reader of four steps, which
  reads all 1,800 tasks of the space y ← a·y^k + b·y + c mod N (k 2 or 3, a 1 to 9, b and c 0 to 9); windows checked
  wherever a multi-digit product fits; windows reduced mod N in Horner order; a second round search with x kept in S
  when a law reuses x; and a work budget of 13e9 units. `submission_stage3_deep.py` (`56acca73`) certifies on 8 of 8
  official-manifest H100 runs, training in 502–741 s, and is now the file to judge (REVIEW.md §2). Byte for byte, one
  H100 each, it learns all 22 other tasks it was run on, each right to T = 64 on both ladders with every split exact:
  the 8 family tasks, phase D's pair, and 12 tasks held out. Six of those were drawn with the fixed file's own md5 as
  the seed, so the design could not be tuned toward them: the clean test, 6 of 6, four of them needing four steps, which
  puts at least 61% of the space within reach (95%, one-sided). The other six used a seed known while the design was
  made. The margins are thin: the heaviest answer search came at 1.273e10 of its 1.3e10 units, and a run that used the
  whole budget would train for about 3,510 s of the 3,600. About $125.

## 5. The removal ladder

Each step removed one hand-written piece. To count, it had to certify on the untouched official manifest on an H100,
with the file run byte for byte: T = 64 on both ladders, every rung 768/768. Then it was repeated on other seeds
through the same evaluator.

The ladder branches:
- steps 4a–4d each started from step 3's machine, and step 5 put them together;
- steps 6a and 6b went back to 4c's single-piece machine, whose reading, squaring and loop are still designed;
- steps 7a–7c stack on step 5, and so does step 8;
- step 9 is 7c with step 8's lessons, step 10 widens 7c's menus, and step 11 is 10 with step 9's change;
- steps 12–13 stack on 11; steps 14–15 start again from 11, and step 18 stacks on 15;
- step 16 joins 13 and 15, step 17 replaces 16's machine with programs found in training, and step 19 replaces 17's
  curriculum with practice problems the learner poses itself;
- the variants of §4.8 branch from steps 17 and 19, and are not numbered.

| step | builds on | what became learned (the hand-written piece removed) | file (md5) | official record | training s (frozen at step) / evaluation s | other seeds certified | commit |
|---|---|---|---|---|---|---|---|
| 0 | — | nothing: the designed subtraction table, a plumbing check | an earlier `learned_remainder/submission.py` (`9732a11f`) | `s0_official` | 3.3 (100 steps) / 5.5 | 1 of 1 | `824d073` |
| 1 | 0 | the 200-entry subtraction table, taught with made-up practice divisions (gray under rule 14) | `submission.py` with `LESSON = "madeup"` | `final_s1_official` | 154.5 (484) / 4.8 | 8 of 8 (`s1v2` recipe) | `824d073` |
| 2 | 1 | no made-up numbers: the table checks itself on the rows' own numbers | [`submission.py`](learned_remainder/submission.py) (`bd477b9a`) | `final_s2_official` | 181.9 (499) / 4.9 | 8 of 8 (`s2v2` recipe), and 4 of 4 that never see an answer | `824d073` |
| 3 | 2 | the multiples of N: repeated subtraction with the learned table | [`submission_chain.py`](learned_remainder/submission_chain.py) (`ed360cb1`) | `chain_v2_official` | 271.8 (417) / 15.4 | 8 of 8, and 4 of 4 | `e1dd000` |
| 4a | 3 | integer squaring: a learned 100-entry times table, schoolbook | [`submission_square.py`](learned_remainder/submission_square.py) (`37765051`) | `square_official` | 270.5 (499) / 20.5 | 8 of 8, and 4 of 4 | `c0c86a2` |
| 4b | 3 | the integer loop over T and the stop rule: a countdown on the learned table, and a learned 2 × 2 gate | [`submission_control.py`](learned_remainder/submission_control.py) (`ec79a312`) | `control_official` | 239.7 (469) / 12.8 | 8 of 8, and 4 of 4 | `c6c9597` |
| 4c | 3 | the hand-written adder in the self-check, after a 70-step warm start | [`submission_adder.py`](learned_remainder/submission_adder.py) (`44e658ab`) | `adder_warm_official` | 77.8 (343) / 12.8 | 8 of 8, and 4 of 4 | `550f60e` |
| 4d | 3 | hand-written prompt parsing and answer writing: a learned reader and writer | [`submission_io.py`](learned_remainder/submission_io.py) (`45cc6e1b`) | `io_official` | 220.0 (440) / 12.8 | 8 of 8 | `f342a91` |
| 5 | 4a–4d | all of 4a–4d in one model, predicting inside autograd | [`submission_combined.py`](learned_remainder/submission_combined.py) (`4f5cecc7`) | `combined_official` | 666.6 (545) / 48.2 | 8 of 8 | `9959997` |
| 6a | 4c | most of the training recipe: three facts and the check remain; no answers, phases or freeze | [`history/ablation/submission_simple.py`](history/ablation/submission_simple.py) (`f374ec70`) | `simple_official` | 300.7 (3,000 steps, no freeze) / 23.1 | 8 of 8 (a bit-identical twin), and the file itself 4 of 4 | `f498805` |
| 6b | 6a | the warm start: digit meanings from the order of the numbers; no hand-written adder, no answers | [`anchors/submission_anchors.py`](anchors/submission_anchors.py) (`a9e20f66`) | `anchors_official` | 207.7 (3,000 steps, no freeze) / 16.0 | 8 of 8 | `5d2b917` |
| 7a | 5 | the division's wiring: 7 switches, 864 wirings | [`history/learned_wiring/submission_div.py`](history/learned_wiring/submission_div.py) (`11cccf13`); first version `submission_div_v1.py` (`a0050a2f`) | `div2_official`; first version: `div_official` | 313.6 (615) / 36.8; first version: 393.4 (640) / 48.5 | 8 of 8; first version: 7 of 8 | `36fc156` |
| 7b | 7a | the squaring's layout: 576 layouts | [`history/learned_wiring/submission_sq.py`](history/learned_wiring/submission_sq.py) (`781928ea`) | `sq_official` | 377.7 (479) / 49.0 | 8 of 8 | `36fc156` |
| 7c | 7b | the reader's and writer's shape | [`history/learned_wiring/submission_io3.py`](history/learned_wiring/submission_io3.py) (`c2331362`); first version `submission_io3_v1.py` (`070d2ee5`) | `io3v2_official`; first version: `io3_official` | 728.6 (532) / 51.8; first version: 729.4 (532) / 48.9 | 8 of 8; first version: 6 of 8 | `36fc156` |
| 8 | 5 | the warm start in the combined model, with step 6b's lessons | [`anchors/submission_combined_anchors.py`](anchors/submission_combined_anchors.py) (`8887fdae`) | `cmbanchors_official` | 454.0 (553) / 40.6 | 8 of 8, and 4 of 4 with the old table lessons off | `5d2b917` |
| 9 | 7c, with step 8's lessons | the warm start in the wiring model: no hand-written arithmetic anywhere | [`history/learned_wiring/submission_anchored.py`](history/learned_wiring/submission_anchored.py) (`6328a7d2`) | `anchored_official` | 466.1 (522) / 50.3 | 8 of 8 | `b76f2cd` |
| 10 | 7c | the menus' narrowness: the wiring chosen from about 4·10^17 whole machines instead of about 6 million | [`wider_menus/submission_wide.py`](wider_menus/submission_wide.py) (`43db5606`) | `wide_official` | 591.8 (566) / 42.9 | 8 of 8 | `c166f8d` |
| 11 | 10, with step 9's change | the wide menus with no hand-written arithmetic anywhere | [`wider_menus/submission_wide_anchored.py`](wider_menus/submission_wide_anchored.py) (`c82b11fa`) | `wide_anchored_official` | 558.0 (588) / 41.4 | 8 of 8 | `d8856a8` |
| 12 | 11 | the tables' form: both digit tables as generic ReLU MLPs | [`history/bridge/h1/submission_wide_mlp.py`](history/bridge/h1/submission_wide_mlp.py) (`22bd34d0`) | `wide_mlp_official` | 1,047.7 (743) / 98.0 | 8 of 8 | `16b66dd` |
| 13 | 12 | the chain's menu: its wiring chosen inside training among 7.9·10^9 relative-position wirings | [`history/tournament/submission_tournament.py`](history/tournament/submission_tournament.py) (`7ae547d9`); first version `312d9422` (not kept as a file) | `tourney2_official` | 2,114.0 (705) / 89.8 | 8 of 8; first version: 3 of 8 | `90f245f` |
| 14 | 11 | task-specific lessons: laws stated once, mined facts, lesson sets chosen by the answers | [`general_lessons/submission_general.py`](general_lessons/submission_general.py) (`ddfdc065`) | `gen_h1_official` | 1,855.6 (645) / 97.2 | 8 of 8 | `1613c28` |
| 15 | 14 | the constant "answers from rows with T ≤ 2": the answers read at the batch's smallest T | [`general_lessons/submission_general_v2.py`](general_lessons/submission_general_v2.py) (`30d15786`) | `gen2_h1_official` | 1,290.5 (645) / 48.1 | M4 (relaxed budget): 2 of 2 | `6f6408d` |
| 16 | 13 and 15 | separate files: the tournament and the general lessons in one model | [`integration/submission_stage1.py`](integration/submission_stage1.py) (`cfd7b61d`) | `s1_official` | 1,860.7 (716) / 56.1 | 8 of 8 on CPU | `b5f1740` |
| **17** | 16, with the library inside training | **the skeleton and the menus: the machine that answers is a program found in training** | [`integration/submission_stage3_s3i.py`](integration/submission_stage3_s3i.py) (`0e813144`) | `s3i_official` | 1,340.3 (870) / 209.9 | 7 of 7 more H100 seeds, official manifest (1–7) | `8e7422e`, `5c31dd0` |
| 18 | 15 | the digit code's dependence on many moduli: the answers choose whole digit codes | [`general_lessons/submission_general_v3.py`](general_lessons/submission_general_v3.py) (`0805b7d3`) | `v3k_h1_official` | 1,190.5 (645) / 42.3 | H1 8 of 8 on B200s with build `v3f` (`32a1d154`); 14 of 14 fixed-modulus sets, 17 of 17 runs | `66304ba` |
| 19 | 17 | the curriculum: the practice problems, their order, the twins and mac's posing, replaced by a law book of 11 laws on 8 patterns posed in every register binding, and one general choosing rule | [`integration/self_practice/submission_stage3_sp2.py`](integration/self_practice/submission_stage3_sp2.py) (`afc70fe9`); first version `submission_stage3_sp.py` (`f0ef5b06`) | `submission_stage3_sp2_official`; first version: `submission_stage3_sp_official` | 1,276.7 (887) / 225.5; first version: 938.3 (883) / 154.0 | 8 of 8 on B200s; first version: 7 of 8 | `8f6b3e1` |

**How hard stage 3 was pushed** (INTEGRATION.md §3; every build's records are in `integration/records/`):

| build | what changed | H100 runs | certified |
|---|---|---|---|
| s3 | the first stage-3 file, with the climb as first built (variant A) | `s3_official`; two relaxed runs of 18,000 s, stopped | 0: the climb never found the multiply, even with the relaxed budget |
| s3c | the A/B's variant C rules | a relaxed run, stopped at level 3 | 0: two in-run bugs (an Occam tie took a slower spelling, and skipping twins killed a chain) |
| s3d | those two fixes | `s3d_official`; `s3d_relaxed_h100` (18,000 s) | **the first in-run discovery of the round**, certified only on the relaxed budget, in 9,142 s; the official run ended inside the climb |
| s3e | goal states tested fewest pieces first, the deadline checked inside the search | `s3e_official` | 0: the climb reached level 5 at the deadline |
| s3f | the pieces practised first (the multiply after its row, the reduction after its step), with a cache across levels | official and seeds 1–7 | 4 of 8, the official run in 3,133.7 s |
| s3g | every solved problem becomes a routine (`submission_stage3.py`, `70bac83c`) | official and seeds 1–7 | 5 of 8, frozen by 2,749–3,379 s: seed 5's acceptance test rejected the right round, seed 6 hit a NaN, seed 7 ran out of time |
| s3h | an exact acceptance test, a guard in one lesson, the search run in lockstep on the GPU (`3b9d9bcf`) | official and seeds 1–7 | 8 of 8, frozen by 1,341–2,086 s |
| **s3i** | the countdown's round program on all rows at once from a CUDA graph, and two faster search parts (`0e813144`) | official and seeds 1–7 | **8 of 8, frozen by 982–1,149 s** |

Notes:
- Training time includes the steps after the freeze, when the learning rate is 0. "Never see an answer" means the same
  file with one constant changed (`REAL_MAX_T = 0`). Freeze times ("frozen by") are read from the evaluator's progress
  line just after the freeze.
- Records live in `history/runs/learned_remainder/`, `history/runs/learned_wiring/`, `history/runs/wider_menus/`,
  `history/ablation/runs/`, `anchors/records/`, `history/bridge/h1/records/`, `history/tournament/h1_records/`,
  `general_lessons/records/`, `general_lessons/fixed/records/`, `integration/records/`,
  `integration/ablation_s3/records/` (the fix's in `fix/`, and each fixed copy's in its study's `fix/records/`),
  `integration/self_practice/records/runner/`, `integration/learned_loop/records/`, `integration/strict/records/`,
  `integration/sp_breadth/records/`, `integration/sp_ablation/records/`, `integration/generic_ops/records/runner/` and
  `bar3/bridge/records/runner/`, one `<name>.json` each. Each holds the full log, the result and the exact source.
  `tools/verify_records.py` checks the official records against the files they certify.
- Other certified runs worth knowing: MENUS.md's four variant H100 runs; the general files' other formulas
  (`general_lessons/records/gen_*`, `integration/records/s1_*_s74`); and the practice datasets of v3
  (`general_lessons/fixed/records/v3k_*`).
- Records of failures are kept too:
  - `s1_official`: Stage 1's first recipe, with 8 of 200 rows right and no rung certified;
  - `adder_official`: a learned adder from a random start with no warm start, with no rung certified;
  - the first versions' failed CPU seeds: `div_cpu_seed7`, `io3_cpu_seed2`, `io3_cpu_seed5`, and five
    `tourney_cpu_seed*` (the chain's gate flipped, so no round ever ran);
  - `tourney2_nonorm_cpu_seed*`: the tournament without the moved-up rows, 4 of 8 certified;
  - `wide_try1_h100`: a first wide-menu build that crashed on a bug when its chain lesson joined (MENUS.md §4.2);
  - `history/handwired_bridge/h1/records/wide_reglu_official` and `wide_reglu_official2`: the wide model with ReGLU
    cells, nothing right on either official run, and 1 of 4 CPU seeds certified;
  - stage 3's early builds above, and `gen_minus1` and `s1_minus1_s74` (y² − 1, outside every round menu);
  - the ablation's 17 runs that never found the round (`integration/ablation_s3/records/`): the seven needed parts on
    both seeds, and, on seed 74, the arms without the chain problem, without the step problem, and with the round's
    cost limit at 4;
  - self-practice's seed 7 of `sp`, and its eight runs on the other formulas
    (`integration/self_practice/records/runner/sp_*_h100`), none learned;
  - the learned loop's seed 2 on the official budget (`ll2_b200_seed2`, and `ll_b200_seed2` of its first build), and
    two of three runs that read T as wide as x (`llk8_*`);
  - breadth's tasks not learned (`integration/sp_breadth/records/`): nine of the ten run with sp2 unchanged, then
    fewer after each change, down to y³, Montgomery's and the fourth number;
  - the bridge's first launch, which crashed on a bug in its CUDA graph (`bar3/bridge/records/runner_bug1/`);
  - the law ablation's 34 runs that never found the round, 17 on each seed (`integration/sp_ablation/records/`);
  - five earlier generic-instructions runs (`integration/generic_ops/records/runner/`), none certified:
    two killed by the runner's memory cap, two that ran their 3,600 s, and one earlier build on a B200.

  `chain_official`, the chain's first recipe, certified, but one of its CPU seeds that never sees an answer did not.

## 6. Rules 1–16 for the best model

The rules are [one-layer-deeper/README.md §Rules](../one-layer-deeper/README.md#rules). "Gray" means a reasonable
reader could object; the point is flagged, not argued away. This covers the stage-3 file,
[`integration/submission_stage3_s3i.py`](integration/submission_stage3_s3i.py). It follows INTEGRATION.md §5 for rules
7, 8, 9, 11 and 14 and the parents' tables (COMBINED.md, WIRING.md, MENUS.md, TOURNAMENT.md, LESSONS.md) for the rest.
The imports, the initialization, the host transfers and the absence of backward calls were re-checked in the source.

| rule | status | why |
|---|---|---|
| 1. one `submission.py` exporting one `Submission` | yes | the file is sent as `submission.py`; it has one `SUBMISSION` with model, optimizer, loss, batch sizes and a step cap |
| 2. self-contained | yes | it imports only standard-library modules (`itertools`, `time`, `array`, `collections`), `torch` and the public `benchmark` API; no numpy, no files, no network |
| 3. the participant defines model, optimizer, loss, batch sizes, steps | yes | Adam; the rounds run until no prompt wants another (adaptive computation, capped) |
| 4. the evaluator owns the loop, data order, backward, clipping, cadence, seeds, deadline | **gray** | the evaluator is untouched: one pass per step, no batch reuse. But the schedule lives in the model as Python state: the reader's stages, phases A–D (cells, the climb, the round), and a freeze that sets the learning rate to 0 once every check has passed for 200 steps, which is early stopping on the model's own checks. The climb also reads the clock: `build_optimizer` sets a deadline from the budget the evaluator passes in, each search stops after 900 s, and the climb stops 120 s before the budget ends, so how far the climb gets depends on the host's speed (s3g's slowest host ran out of time). And the model keeps rows across batches: up to 4,096 read rows for its practice problems and 512 answered rows for the round's search |
| 5. at most 500M parameters | yes | 207,374 in 16 tensors. The library of found programs lives in Python, outside the state dict, so it is not counted; a strict reader may want that said explicitly |
| 6. no hard-coded weights | yes | all 16 parameter tensors start at 0.1 × random normal: the three digit tables, the countdown's gate, the round programs' tournament, and the reader's and writer's parameters. The "designed" mode is only a plumbing check, never the submission |
| 7. no hard-coded algorithm in the forward pass | **gray, much less than before** | the forward pass runs a program the training run found (search, merges, routines, and composition against the answers) on learned tables; no datapath and no menu is in the file. Still gray: the instruction set leans toward digit arithmetic (cells at a pointer against a register, keep-on-borrow, bring the top digit in, loops by the number's length and the base); the practice problems walk the search up the same pieces the old skeleton had, and two of them are definitions of long division's step; the countdown over T is hand-written. A strict reader can say the algorithm moved into the instruction set and the curriculum |
| 8. final logits from the model inside autograd, with an unbroken gradient path | yes, with gray edges | the prediction path is one-hot digits through straight-through tables, with the round program a straight-through choice over the tournament's contestants. In the gradient test, run on the earlier build s3g, whose prediction path s3i keeps, every one of the 16 parameter tensors is in the graph of the final logits, and all 16 get a nonzero gradient with the right tokens' logits as the objective. Gray: (a) in training the forward returns zero logits, so the lessons teach, not the final logits; (b) the gradient is damped per lookup so it does not overflow, and on deep paths it underflows to 0: with cross-entropy at a perfect prediction, the tournament's logits and the addition table's carry get exactly 0; (c) the program itself is a discrete result of search, held in Python, and only the choice among the four found programs is a parameter (one logit each); (d) which reading hypothesis a prompt uses is decided by designed run-detection logic |
| 9. everything on the GPU | yes | the search, the probes and the machines run as GPU tensors, with no `.cpu()` or numpy in the file. Python holds the bookkeeping: programs as tuples, BPE over program text, and the schedule. Each program's instructions are traced on the host from its text and copied to the GPU as opcode columns; long runs are replayed from CUDA graphs |
| 10. OOM or timeout fails | ok | neither happened on s3i's eight official-manifest runs |
| 11. 3,600 s of training, compilation included | yes, on every seed | frozen by 982–1,149 s, trained 1,133–1,350 s. Earlier builds needed more: s3d a relaxed 9,142 s, s3g up to 3,379 s and out of time on one seed |
| 12. a custom loss returns one finite differentiable scalar | yes | `token_training_loss` |
| 13. evaluation within 1,800 s; certification | yes | 154.9–219.1 s on the eight runs; T = 64 on both ladders |
| 14. no data inspection, augmentation, task-specific solvers, custom loops or participant backward passes | **gray, in a new place** | no backward or grad call, no custom autograd function, no data file read, no number made up, and no hand-written arithmetic at any step of training. But the climb is a program search inside the training forward pass, over many steps. Its practice problems are made from the batch rows' own digits and the law library, and the curriculum (which laws become practice problems, in which order, the register variants, mac) and the round's search space are designed: a strict reader could call the curriculum a task-specific solver written as lessons. The lessons also still carry the anchors' facts, the step laws, and the old table lessons (which join late and include "a wrapped x − N is large", an assumption about the register's width). The ablation shows the late table lessons are not needed: without them s3i certifies on both seeds tried, its climb identical line for line |
| 15. repeated rule-breaking | n/a | the write-ups flag their own gray areas, and the hosted evaluator has scored nothing: the one attempt, after the deadline, was refused (§7) |
| 16. do not exploit the Hard metric recorder | yes | not touched |

**The other frontier files, where they differ.**
- **Stage 3 with self-posed practice** (`integration/self_practice/submission_stage3_sp2.py`): s3i's status on every
  rule, with two differences.
  - Under rule 14, the hand-picked curriculum is gone. A law book (11 laws on 8 patterns of the data's numbers) and a
    choosing rule take its place. Both are designed, and the rule was picked after bar 3's own rules failed, by
    watching whether the round came.
  - Under rule 1, the file is 250,643 bytes, within the 256 KiB limit (s3i: 173,817).

  It imports two more standard-library modules (`math`, `zlib`). Its seeds ran on B200s, which count as research.
- **The strict builds** (`integration/strict/`, §4.8): s3i's or sp2's status, with three points cleared.
  - Rule 4: no decision reads the clock or the budget, and the remembered rows are buffers.
  - Rules 5, 8(c) and 9: the round program is held in buffers, counted and on the GPU.
  - Rule 14: the late table lessons are gone.

  Rule 8(a) is unchanged: training still returns zero logits. sp2's strict file is 254,804 bytes, within 256 KiB.
- **The learned loop** (`integration/learned_loop/submission_stage3_ll.py`, §4.8): under rule 7, the countdown over T
  is no longer written in the file; a generic count loop is one instruction, and the search finds how to use it.
- **The bridge variant** (`bar3/bridge/submission_stage3_bridge.py`, §4.8): under rule 7, SEL.S is replaced by a select
  that bar 3's learner invented before training. All four of its digit cells are ones the learner invents (BRIDGE.md,
  CARRY.md), multiply-accumulate only with a deeper search aimed at its piece.
- **The smaller and generated books** (`integration/sp_ablation/variants/book_min.py` and `book_gen.py`, §4.8): sp2's
  status, with less of rule 14's designed curriculum: 5 laws instead of 11, or a grammar that picks no law by hand.
  The grammar, the target rules and the choosing rule are still designed. The files are 251,101 and 255,195 bytes,
  within 256 KiB.
- **Stage 1** (`integration/submission_stage1.py`): rule 7 is gray as for its parents. Its forward pass is the
  hand-written H1 datapath, and the menus that plug into it, the tournament's included, hold the right wirings
  (INTEGRATION.md §5).
- **General lessons v3** (`general_lessons/submission_general_v3.py`): rule 7 as for the wide model, plus its round
  menu and two lesson sets. Under rule 14 it adds a designer's decision about what a borrow means: the table's borrows
  are tied to the reader's ranking, so that the answers can choose the whole digit code (FIXED_MODULUS.md).

**The earlier certified models, in one line each.**
- **SpecBit**: only one gate's truth table is learned, and the netlist around it is the algorithm. Under a strict
  reading of rule 7, the design panel's critics rated every one of the panel's designs fatal
  ([REPORT.md](history/REPORT.md) §6).
- **UnfoldB and UnfoldS**: 4 learned numbers, with the snapping, the remainder and the loop built into the model.
  Their notes discuss rules 7, 8 and 14 ([history/unfold_sub/NOTE.md](history/unfold_sub/NOTE.md),
  [history/unfold_small/NOTE.md](history/unfold_small/NOTE.md)).

## 7. Status and open questions

**What is running.** At about 07:25 PT on Oct 24, Modal's workspace spend limit stopped every task, at about $3,000
billed this month. The user raised it to $4,000 that afternoon and to $5,000 that evening, and the four cut-off studies
were relaunched at 14:42–14:47 PT. All four have finished: the generic-instructions file certifies (§4.8), the NCA's 13
runs left its verdict as it was (§4.5), the law ablation has all its seed-1 arms and its rerun at three times the budget
(§4), and bar 3's cut runs are finished (§4, bar 3). Oct 25's two bar-3 studies have finished too: numbers given as
digits, testing the theory note's predictions ([theory/LARGE_NUMBERS.md](theory/LARGE_NUMBERS.md) §5), and the two
practice sources the held-out test found missing (§4, bar 3). So have the combined file's second and third rounds, its
fast version and fast2, the file to judge before the legal file (§4.8), and bar 3's duplicate check on each problem's
own rows (§4, bar 3). So have bar 3's three changes to what it picks (§4, bar 3). So have the next two, a deeper look
and a fifth pick (§4, bar 3). So has the composition test (§4, bar 3). So have its second and third rounds (§4, bar 3).
So has the GPU port of its search (§4, bar 3). So have the consolidation of bar 3's rules, the one-place cut of the
target's laws, the GPU runner and the audit of bar 3's sources (§4, bar 3). The legal path's tests are done: a legal
climb gets H1's round with the book cut to three laws (§4, bar 3). So has the offline loop study (§4, bar 3). So has the
legal file, which certifies 8 of 8 (§4.8). So has bar 3's round 6, which takes the book to nothing offline (§4, bar 3).
Bar 3's finish line, which the user set on Oct 27, is a legal file with no practice book that certifies H1 8 of 8 and,
unchanged, learns at least a couple of other tasks of the family, with the reader's grammar widened beyond what H1
needs. Its first two steps are done: the wider grammar and the empty book each certify 8 of 8 (§4.8), and offline the
family's seven tasks come with no book (§4, bar 3). The third is done too: round 7's general rules brought into the
no-book file, which certifies H1 8 of 8 again and, byte for byte, learns 4 of 8 family tasks at full size, one of the
two held-out ones among them. A last round, phase D, moved ROUNDT's rows into buffers and widened the reader: H1 8 of 8
again and 7 of 8 family tasks, but neither of two held-out tasks drawn at random. Phase E read four steps deep and kept
x between calls: H1 8 of 8, and all 22 other tasks it was run on, the six drawn with its own md5 as the seed among them;
its file is now the file to judge (§4.8). Nothing is running.

Everything else described here is committed.

**A plan from here to a contest-legal entry** ([LEGAL_PLAN.md](history/LEGAL_PLAN.md), Oct 25, no runs): what of bar 3
has to move inside a training run, what is still designed in the self-practising file, four questions only a judge can
settle, and the cheapest test of each, in order. No training target may come from Python's arithmetic, so what moves
from bar 3 is a rule rebuilt on stage 3's machine, not bar 3's code. Its first test has run: one file with every
certified variant of sp2 does not certify, because the hour runs out before the climb finds the reduction (§4.8). A
second round with three fixes finds the round in the tiny core, but too late for the hour. In the third, both four-piece
files certify, and the five-piece file, made fast without changing what it learns, certifies in 751 s (§4.8).

**Answered on Oct 23 and 24.**
- **Which hand-made parts carry the result** (ABLATION_S3.md): two instructions, the step's two practice problems,
  the posing of the multiply's row and of the registers, and routines are needed on both seeds tried, and the chain
  problem alone on one seed of two. Occam's tie-break only stood in for a weak check. The curriculum's order, BPE and
  two search rules only cost time (§4, bar 2).
- **Whether the weak checks can be fixed without changing what is learned** (ABLATION_S3.md §8): yes. Copies of s3i
  and sp2 with the missing probes replay all 17 runs field for field, and on them neither Occam's tie-break nor the
  step problem is needed. Copies of eight more certified files with the same fix certify wherever their originals
  did, and 75 of their 78 runs equal the original's.
- **Whether the loop over T can be learned** (LEARNED_LOOP.md): yes, as a generic count loop and a second search. It
  certifies on the official manifest; seed 2's time-out is the random start's.
- **Which gray points can go without changing what is learned** (STRICT.md): the clock, the rows and the round
  program held in Python, the scheduler's reset of the optimizer, and the late table lessons. 17 runs replay line for
  line. The zero logits in training stay: the real answer on a training batch does not fit in an H100's memory, and
  on the rows where it fits it would teach nothing.
- **Whether stage 3 learns the family's other formulas at full size** (SP_BREADTH.md): unchanged, no (2 of 11
  tasks); after three general changes to the round's search, 8 of 11.
- **Whether bar 3's learner can invent stage 3's instructions** (BRIDGE.md, CARRY.md): yes, all four digit cells: the
  select, subtract-with-borrow, add-with-carry (in the round the spend limit had cut off) and multiply-accumulate (once
  a deeper search is aimed at its piece). s3i with the invented select certifies.
- **Whether generic instructions can replace SEL.S and BRG.X** (GENERIC_OPS.md): yes. With an if block and shifts
  that carry their digit, the learner builds the conditional subtraction, the chain and long division's "bring down"
  itself, and the file certifies on the official manifest and 8 of 8 seeds, at 2.2 times sp2's training time. The
  search rules it needed were found by watching whether the round came.
- **Whether a cheaper bar-3 learner reaches further** (SCALE.md): ten times cheaper, with the same choices, buys
  about one instruction. But it made the pairs rule affordable along a sequence of formulas, which then invents
  10·x², 10·x² + 1 and x³, and a new practice source gives Fibonacci.
- **Whether the learner can aim its search before it solves a piece** (SUBGOAL.md): yes. Scored from the rows by
  whether the target factors through it, x² mod N ranks first in round 0 for all four targets built on it, and one
  deeper search aimed there brings x³ and 10·x² from scratch.
- **Whether bar 3's rules hold on formulas none was written for** (HELDOUT.md): for half of them. Frozen, with its
  predictions written down first, the learner invents 5 of 10 new formulas, all in round 0 or 1, and 16 of 20
  predictions hold. The misses need a step no source poses or a program longer than the search reaches.
- **Whether the curriculum can go** (SELF_PRACTICE.md): yes, for H1. With a law book and a general choosing rule in
  its place, stage 3 still certifies on the official manifest (step 19).
- **Which laws self-practice needs** (SP_ABLATION.md): the multiply-add law and its pattern, the parts every problem
  is built or checked with, the round's two routines, and rating a concept by its cheapest placement; on seed 1 also
  the x + d·N pattern and choosing at all, which only cost time on seed 74. Six laws and three patterns are not
  needed, and a 5-law book and a book generated from a grammar both certify.
- **Whether s3i needs the late table lessons**, the width-dependent wrap-around fact among them: no. Without them it
  certifies, its climb identical line for line. The strict builds drop them, and the gate's check of the same fact.
- **Whether the usual cures for memorizing teach the shader grid** (NCA.md §9): no. Neither long training with
  weight decay nor agreement between questions or columns turns its memorizing into the rule, and no arm can even
  hold the rule when started there. Not tried in §9: another learning rate, more seeds, agreement across seeds or
  sizes.
- **Whether an optimizer that can rest at the rule changes that** (NCA.md §10): no. Adam with eps 1e-4 and plain SGD
  keep the compiled rule where every lesson is met, 4 of 4 seeds each, but from random every arm still memorizes.

**Open questions for bar 2** (nothing a strict judge would call the hand-written algorithm):
1. **The instruction set.** s3i's 32 digit instructions are general but chosen knowing the family, and two of them,
   SEL.S (keep or discard on the borrow) and BRG.X (bring the next digit in), are needed: without either the round is
   never found. The inventing test shows bare counters suffice in principle, at sizes far below H1; at H1 a counting
   algorithm would take about 10^16 steps. At one digit place, bar 3's learner invents a select that can take SEL.S's
   place, subtract-with-borrow, add-with-carry and multiply-accumulate; BRG.X, the moves and the loops are not one
   place's work (§4, bar 3). Given all 11 one-place slices of the law book as instructions, s3i's climb finds no round:
   wrong programs pass its weak checks, or crowd out the right one in a capped search. The run that would confirm the
   second was refused by the spend limit (BRIDGE.md §5). Generic instructions, an if block and carry shifts, can stand
   in for SEL.S and BRG.X: the learner then builds the conditional subtraction and long division's "bring down" itself,
   and the file certifies (§4.8). That took new search rules, found by watching whether the round came, and 2.2 times
   sp2's training time.
2. **The law book and the choosing rule.** Self-posed practice replaces the curriculum with a book of 11 laws on 8
   patterns and one choosing rule. A book of 5 laws, and one generated from a grammar, also yield the round and
   certify (§4, bar 2). Still designed: the grammar and its target rules, and the choosing rule, which was found by
   watching whether the round came and whose rating of a concept by its cheapest placement is needed.
3. **How firm the ablation is.** The two seeds agree on every main arm, and every failed run of a needed part had
   exhausted its cost-5 search, so more time would not help; a search up to cost 6 found nothing either. Cost 7 is
   untested (about 50 times more programs). Occam's tie-break was the exception: not a search's reach but a weak
   check, and with the check fixed it is not needed. The row problem's arm and the cost limits ran on seed 74 only.
4. **The round's search space and the count loop** are designed. With three general changes to the round's search,
   sp2 learns 8 of 11 tasks at full size (§4.8), but each change was written with the family in view. Still not
   learned: y³ (four routine calls), Montgomery's (a reduction the law book never poses) and a fourth input (the
   reader takes three).
5. **Rules 4, 5, 8, 9 and 14** (REVIEW.md §5 states them as questions for a judge). In s3i the climb reads the clock
   and keeps past rows, and the library of found programs lives outside the state dict; the strict builds clear those
   points without changing what is learned (§4.8). Left: training never sends the loss through the final logits, the
   gradient is damped, and the climb is a search inside the forward pass.

**Open questions for bar 3** (a general algorithm learner):

6. **Scale.** Bar 3's learner works on a bare counting machine at x < N ≤ 7 (up to N ≤ 50 for x² mod N, 2x + 1 and
   Fibonacci), and on one digit place, where it invents all four of stage 3's digit cells. Frozen, it invents 5 of 10
   formulas none of its rules were written for (HELDOUT.md). Given a loop over x's digits and a source that asks one
   question of the rows, it invents x mod N in base 2 by long division's step, right up to nine digits; x² mod N and 2^x
   mod N do not come (DIGITS.md). On stage 3's digit machine, its pieces replace the law book's reduction but not yet
   the four laws below it (COMPOSE.md, COMPOSE2.md). At H1's size there is the self-practising stage 3: one target at
   first, now 8 of 11 tasks, after changes written with the family in view (§4.8).
7. **What its sources cannot pose.** A new source poses steps that move two numbers at once, and Fibonacci is now
   invented from scratch. 10·x², 10·x² + 1 and x³ need x² mod N as a piece, which the base law book never poses; the
   depth-2 book poses it, and a score of whether the target factors through it, with one deeper search aimed there,
   brings x³ and 10·x² from scratch (SUBGOAL.md). 10·x² + 1 needs a second step: the rest of the way after x² mod N
   would have to be split the same way, and tried small that fails: no source poses 10X mod N (SUBGOAL.md §8). The
   held-out test names two more such gaps: no source poses a running sum's step, y(x + 1) from y(x) and x, or a step on
   three numbers (HELDOUT.md §7). Posed as a source, the running sum's step brings the triangular numbers under anchored
   pairs; a source that takes away the constant term poses 10X mod N, but the learner's duplicate check throws it away
   (NEWTON.md). Checked on its own rows, the piece is kept, and hardest first still does not pick it: problems of the
   same cost stand ahead of it (ROWCHECK.md). Breaking those ties by the subgoal score picks the piece, and 10·x² + 1
   comes (PICKS.md).
8. **The cost of searching.** A compiled search, shared searches and the cheaper choosing rule cut the breadth study
   from 515 search-hours to 51 with the same choices. But each extra instruction still multiplies the search by 9–10
   (about 40 over a carried library), and no bound on the search space tried saves more than 1.7× (SCALE.md §8).
   What brought x³ and 10·x² was aiming one deeper search by a score, not searching faster (SUBGOAL.md).
9. **What the data cannot say.** On data whose T always has one digit, nothing tells "16" from "61" (eight of v3's
   fixed-modulus sets stop at T = 8 for that reason). A prior, such as reading T the way the answer is written, would
   settle it, but it is a designer's choice.
10. **Formulas of another shape.** Breadth's constants are near misses by one number mod N; a gap that is a factor
    (y³ against y², Montgomery's 10⁻⁸) is not looked for (SP_BREADTH.md §6). The general files still fail y² − 1,
    since no round option of theirs computes "subtract 1"; the bar-3 learner invents x² − 1 at its tiny size, and
    breadth's sp2 learns y² − 1 at full size.

**And:** submitting to the hosted evaluator after the deadline, just to see a response, was tried on Oct 24 with
s3i: the service refused it ("Submissions are now closed"), so there is no hosted result.

## 8. Where everything is

| path | what |
|---|---|
| [`OVERVIEW.md`](OVERVIEW.md) | this file |
| [`REVIEW.md`](REVIEW.md) | the entry point for an outside reviewer: the claims, the file to judge, how to check and reproduce its records, the questions to rule on |
| [`history/LEGAL_PLAN.md`](history/LEGAL_PLAN.md) | a plan from bar 3 to a contest-legal entry: what has to move inside the training run, what is still designed in the self-practising file, the questions only a judge can settle, and the cheapest test of each |
| [`README.md`](README.md) | what this is, the entry and its results, where to start, how to check, the layout |
| [`history/`](history/) | the early work, folded in on 2026-10-27 with its layout kept: every path below that starts with `history/` ([history/README.md](history/README.md) indexes it file by file) |
| [`history/REPORT.md`](history/REPORT.md) | the first-day report: the analysis (§0–8), SpecBit and the grammar search (§9–14) |
| [`learned_remainder/`](learned_remainder/) | the removal ladder, steps 0–5: every `submission*.py`, its write-up (LEARNED_REMAINDER, CHAIN, SQUARE, CONTROL, ADDER, IO, COMBINED), the Modal launchers, plans and local checks |
| [`history/ablation/`](history/ablation/) | which training ingredients matter ([ABLATION.md](history/ablation/ABLATION.md)), and `submission_simple.py` (step 6a) |
| [`anchors/`](anchors/) | digit meanings from the order of the numbers ([ANCHORS.md](anchors/ANCHORS.md)): `submission_anchors.py` (step 6b), `submission_combined_anchors.py` (step 8), records in `anchors/records/` |
| [`history/learned_wiring/`](history/learned_wiring/) | the wiring learned in three stages ([WIRING.md](history/learned_wiring/WIRING.md)): `submission_div.py`, `submission_sq.py`, `submission_io3.py` (steps 7a–7c) and the first versions `*_v1.py`; then with no warm start ([ANCHORED.md](history/learned_wiring/ANCHORED.md)): `submission_anchored.py` (step 9) |
| [`wider_menus/`](wider_menus/) | the wide menus ([MENUS.md](wider_menus/MENUS.md)): `submission_wide.py` (step 10); with no warm start ([WIDE_ANCHORED.md](wider_menus/WIDE_ANCHORED.md)): `submission_wide_anchored.py` (step 11) |
| [`history/bridge/`](history/bridge/) | the machine inside a generic looped transformer, loosened notch by notch ([BRIDGE.md](history/bridge/BRIDGE.md)); `h1/submission_wide_mlp.py` (step 12) |
| [`history/handwired_bridge/`](history/handwired_bridge/) | the machine compiled by handwired into affine + ReGLU layers, loosened the same way ([HANDWIRED_BRIDGE.md](history/handwired_bridge/HANDWIRED_BRIDGE.md)); `h1/submission_wide_reglu.py`, which did not certify officially |
| [`history/tournament/`](history/tournament/) | the wiring chosen inside training by tournaments ([TOURNAMENT.md](history/tournament/TOURNAMENT.md)): `submission_tournament.py` (step 13), the tiny machine `tiny.py`, records in `history/tournament/h1_records/` |
| [`general_lessons/`](general_lessons/) | a law library, mined facts and answer-chosen lessons ([LESSONS.md](general_lessons/LESSONS.md)): `submission_general.py`, `submission_general_v2.py` and `submission_general_v3.py` (steps 14, 15 and 18); one fixed modulus in `fixed/` ([FIXED_MODULUS.md](general_lessons/fixed/FIXED_MODULUS.md)); records in `records/` and `fixed/records/` |
| [`integration/`](integration/) | the tournament, the general lessons and the library in one model ([INTEGRATION.md](integration/INTEGRATION.md)): `submission_stage1.py` (step 16), `submission_stage3_s3i.py` (step 17, the best model) and the earlier stage-3 builds, the climb's library in `lib/`, records in `records/`. The ablation of s3i's hand-made parts ([ABLATION_S3.md](integration/ABLATION_S3.md), in `ablation_s3/`, its fixed checks in `ablation_s3/fix/`, and fixed copies of eight more certified files in a `fix/` folder of each one's study), and stage 3 with self-posed practice ([SELF_PRACTICE.md](integration/SELF_PRACTICE.md), in `self_practice/`: `submission_stage3_sp.py` and `submission_stage3_sp2.py`, step 19). The variants of §4.8: the loop over T learned ([LEARNED_LOOP.md](integration/learned_loop/LEARNED_LOOP.md), in `learned_loop/`), strict builds ([STRICT.md](integration/strict/STRICT.md), in `strict/`) and breadth at full size ([SP_BREADTH.md](integration/sp_breadth/SP_BREADTH.md), in `sp_breadth/`); which laws self-practice needs, with the smaller and generated books ([SP_ABLATION.md](integration/sp_ablation/SP_ABLATION.md), in `sp_ablation/`); and generic instructions in place of SEL.S and BRG.X ([GENERIC_OPS.md](integration/generic_ops/GENERIC_OPS.md), in `generic_ops/`); five of them in one file ([COMBINED.md](integration/combined_sp2/COMBINED.md), in `combined_sp2/`), certified once made fast ([FAST.md](integration/combined_fast/FAST.md), in `combined_fast/`) |
| [`history/HANDOVER.md`](history/HANDOVER.md) | the hygiene pass before handing the repo over: secrets and private paths, links, checksums, what is tracked and what is on disk only |
| [`litmus/`](litmus/) | the wide model run unchanged on 26 other tasks ([LITMUS.md](litmus/LITMUS.md)), with the per-task variants that fixed some |
| [`library/`](library/) | search and compression that rediscovers the skeleton ([LIBRARY.md](library/LIBRARY.md)), and the inventing test ([INVENTING.md](library/INVENTING.md), code in `library/inventing/`) |
| [`bar3/`](bar3/) | the self-practising learner: choosing its own practice ([PRACTICE.md](bar3/PRACTICE.md)) and breadth over nine formulas ([BREADTH.md](bar3/BREADTH.md)); records in `records/` and `breadth/records/`. Stage 3's digit instructions invented at one place, and the s3i variant that uses one ([BRIDGE.md](bar3/bridge/BRIDGE.md), in `bridge/`); add-with-carry, from the round the spend limit had cut off, and multiply-accumulate, with a deeper search aimed at its piece ([CARRY.md](bar3/bridge/CARRY.md)). The learner made ten times cheaper, then given new kinds of practice problems ([SCALE.md](bar3/scale/SCALE.md), in `scale/`), and a signal that sees a piece before it is solved ([SUBGOAL.md](bar3/scale/subgoal/SUBGOAL.md), in `scale/subgoal/`), tested on ten formulas no rule was written for ([HELDOUT.md](bar3/scale/heldout/HELDOUT.md), in `scale/heldout/`); then x as digits ([DIGITS.md](bar3/scale/digits/DIGITS.md), in `scale/digits/`) and two sources from Newton's differences ([NEWTON.md](bar3/scale/newton/NEWTON.md), in `scale/newton/`); and the duplicate check on each problem's own rows ([ROWCHECK.md](bar3/scale/rowcheck/ROWCHECK.md), in `scale/rowcheck/`); and changes to what the learner picks ([PICKS.md](bar3/scale/picks/PICKS.md), in `scale/picks/`) and how far its looks reach ([REACH.md](bar3/scale/reach/REACH.md), in `scale/reach/`); and a first composition test on stage 3's machine ([COMPOSE.md](bar3/scale/compose/COMPOSE.md), in `scale/compose/`) and its second round ([COMPOSE2.md](bar3/scale/compose2/COMPOSE2.md), in `scale/compose2/`) and third ([COMPOSE3.md](bar3/scale/compose3/COMPOSE3.md), in `scale/compose3/`); and the search on a GPU ([GPU_SEARCH.md](bar3/gpu_search/GPU_SEARCH.md), in `gpu_search/`); and its rules consolidated ([CONSOLIDATE.md](bar3/scale/consolidate/CONSOLIDATE.md), in `scale/consolidate/`); the target's laws cut to one digit place ([SLICE.md](bar3/scale/slice/SLICE.md), in `scale/slice/`); and an audit of its sources against rule 14 ([LEGAL_SOURCES.md](bar3/LEGAL_SOURCES.md)) |
| [`theory/`](theory/) | what is already known about computing and learning with large numbers: classical (digit automata, repeated squaring, Montgomery reduction, the time-lock puzzle H1 is), learning theory and neural theory (extrapolation, log space, clock and discrete-log features), mapped onto this project, with two local checks in `checks/` ([LARGE_NUMBERS.md](theory/LARGE_NUMBERS.md)); and why a search and gradient training keep the same candidates, every comparison-based search is a ReLU network and every ReLU network is if-else, with a tiny check ([SEARCH_AND_GRADIENTS.md](theory/SEARCH_AND_GRADIENTS.md)) |
| [`history/nca/`](history/nca/) | long division as a shader: identical local cells ([NCA.md](history/nca/NCA.md)), the follow-up trained long with weight decay or agreement (NCA.md §9), and with optimizers that can rest at the rule (NCA.md §10); records in `runs/` and `records/` |
| [`tools/`](tools/) | the shared Modal runner and the measured seed policy ([tools/README.md](tools/README.md)) |
| [`history/tinyrouting/`](history/tinyrouting/) | wiring and tables learned together in tiny machines |
| [`history/tinycore/`](history/tinycore/) | the quotient-digit chicken-and-egg, computed exactly |
| [`history/microlab/`](history/microlab/) | a small harness for one-minute experiments, and its findings ([history/microlab/README.md](history/microlab/README.md)) |
| [`history/unfold_sub/`](history/unfold_sub/), [`history/unfold_small/`](history/unfold_small/) | the UnfoldB and UnfoldS submissions |
| [`history/submission.py`](history/submission.py) | SpecBit |
| [`history/panel/`](history/panel/), [`history/panel_synthesis.md`](history/panel_synthesis.md) | the five-design panel and its synthesis |
| [`history/FINDINGS_substrate.md`](history/FINDINGS_substrate.md), [`history/RELAXED.md`](history/RELAXED.md), [`history/UNFOLD.md`](history/UNFOLD.md), [`history/UNFOLD_FAMILY.md`](history/UNFOLD_FAMILY.md) | the day-1 topic write-ups |
| `history/runs/` | run records. Official ones: `history/runs/learned_remainder/*_official.json`, `history/runs/learned_wiring/*_official.json`, `history/runs/wider_menus/*_official.json`, `history/runs/unfold_sub/official_h100.json`, `history/runs/unfold_small/official_h100.json`, `history/runs/h100_h1.json`. Later folders keep their own: `history/ablation/runs/`, `anchors/records/`, `history/bridge/h1/records/`, `history/tournament/h1_records/`, `general_lessons/records/`, `general_lessons/fixed/records/`, `integration/records/`, `integration/ablation_s3/records/`, `integration/self_practice/records/`, `integration/learned_loop/records/`, `integration/strict/records/`, `integration/sp_breadth/records/`, `history/handwired_bridge/h1/records/`, `litmus/records/`, `library/records/`, `bar3/records/`, `bar3/breadth/records/`, `bar3/bridge/records/`, `bar3/scale/records/`, `integration/sp_ablation/records/`, `integration/generic_ops/records/`, `history/nca/runs/`, `history/nca/records/` |
| `../one-layer-deeper/` | the competition clone at `87e5c26` ("Releasing H1"), with the H1 data under `data/generated/` |
| `~/github/handwired-lab/montgomery-search` | outside this repo: search over a Montgomery squaring program in the handwired project (branch `lab/montgomery-search`) |

**The certified submissions, with their records.** Steps 0–19 are in the §5 table, and the variants in §4.8's. The
earlier three:

| model | file (md5) | official record | training s / evaluation s | learned numbers |
|---|---|---|---|---|
| SpecBit | [`history/submission.py`](history/submission.py) (`a432bb5e`; the record embeds no source) | `history/runs/h100_h1.json` | 16.5 (20 steps) / 26.9 | 65,536 (one logit per possible gate) |
| UnfoldB | [`history/unfold_sub/submission.py`](history/unfold_sub/submission.py) (`17299657`) | `history/runs/unfold_sub/official_h100.json` | 384.3 (1,000 steps, batch 100,000) / 3.7 | 4 |
| UnfoldS | [`history/unfold_small/submission.py`](history/unfold_small/submission.py) (`eb6aedfc`) | `history/runs/unfold_small/official_h100.json` | 1,624.9 (200,000 steps, batch 512) / 3.7 | 4 |

All three reached T = 64 on both ladders, with every rung 768/768 and every split exact.

**How to rerun.** Each write-up ends with its commands. New runs go through the shared runner
[`tools/modal_runner.py`](tools/modal_runner.py) (its README has the commands and the seed policy); older folders keep
their own launchers (`learned_remainder/modal_chain.py --file <submission>` for steps 0–5, and each later folder's
copy). Each builds the competition's image, regenerates H1 with the official generator, and runs the untouched
manifest.

## 9. Where docs and records disagree

What was checked, for every number above:
- each official record's certified T, rung counts, split counts, training and evaluation seconds, parameter count and
  freeze step, against the write-ups that cite them;
- the md5 of the source each record embeds, against the files. Where they differ, the difference is the one `--set`
  line the write-up names, or a first version the write-up keeps by its md5;
- the seed claims, by tallying the records;
- the "frozen by" times, from the evaluator's progress line just after each freeze;
- for the frontier files up to step 15, the final tables printed in the official logs, against true subtraction and
  multiplication (s3i's log prints no tables; its certification and its checks stand for them);
- for the litmus, general-lessons and fixed-modulus tasks, the certified T and score of each record the write-ups'
  tables cite.

The numbers checked this way all match their records in the write-ups that cite them, from LEARNED_REMAINDER through
INTEGRATION and FIXED_MODULUS, and in the two Unfold notes and REPORT.md §9. The exceptions:
- **COMBINED.md calls `combined_official` and `combined_index_official` "the same training run".** They are two runs
  with identical training: the same file apart from the `PATH` line, the same seed and data order, and both froze at
  step 545 with the same tables. They were launched together, and their training times differ (666.6 s and 566.5 s).
- **The leaderboard count** (3 of 146 certified, REPORT.md §0) has no record behind it and was not re-checked.
- **SP_BREADTH.md's summary** says the learned tasks found their rounds "by 904-1,343 s into training". The reordered
  prompt's came by 787 s, as the same file's g3 table says.

Counts that need a word, though none is wrong:
- **v3's eight H1 seeds** ("8 of 8 seeds certified too", FIXED_MODULUS.md) ran an earlier v3 build, `v3f`
  (md5 `32a1d154`), on B200s; the final file's own H1 record is the official one. FIXED_MODULUS.md's table says so.
- **The fixed-modulus runs** used a relaxed budget (3,600 s) on B200s: their tiers' own budgets are 60 s (Easy) and
  600 s (Medium). FIXED_MODULUS.md says so.
- **The tournament's machine count.** Its official log prints 2.8·10^22 machines; TOURNAMENT.md's "about 3·10^23" also
  counts the gate table's 4 settings and the reader's 3 shapes.
- **The ablation's speed costs** compare round times corrected for host speed: ABLATION_S3.md's `analyze.py` times
  the searches that match the control's and scales the climb by the ratio. Rerun on the records, it gives every cost
  quoted, and BPE's on seed 74 was redone by hand. One is uncorrected, the order's +277 s on seed 74, whose run
  shares too few searches with the control's. Earlier refreshes of this file quoted +546 s for BPE, from an earlier
  correction that counted one search whose time swings from 0.5× to 5.5× the control's between runs
  (ABLATION_S3.md §4); without that search the cost is +282 s.
- **The practice tiers' rungs** have anywhere from 14 to 768 prompts, so their records were checked by certified T and
  score.

The first version of this file also flagged WIRING.md for contradicting itself on stage 1's CPU seeds. The finished
WIRING.md says 7 of 8 throughout, and the record of the failed seed (`div_cpu_seed7`) is in
`history/runs/learned_wiring/`.
