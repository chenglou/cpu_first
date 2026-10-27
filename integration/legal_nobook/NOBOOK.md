# The no-book file: no law in the book (phase B)

*Written 2026-10-27. Folder `cpu_first/integration/legal_nobook/`. The input is the wide file (phase A,
`integration/legal_wide/submission_stage3_wide.py`, md5 `3bb5ceeb`, asserted); nothing outside this folder was
written.*

## In plain words

<!-- PLAIN -->
**What was asked.** The wide file (phase A) still kept three laws in its book (the comparison, the link and the
addition), and its reader needs all three before it can read anything. Round 6 (bar3/scale/reading/READING.md)
emptied the book offline, by cutting Horner's step down to the link and the comparison, but its step came from a numpy
reading at level 0. The brief: empty the book in the contest file, every practice answer still legal.

**What changed** (§2), in one new file, [`submission_stage3_nobook.py`](submission_stage3_nobook.py) (md5 `b42ec304`,
261,893 bytes, 251 under the limit), built from the wide file by asserted edits ([`make_nobook.py`](make_nobook.py)):
- **the book is empty;**
- **the cheapest reading** comes first, before any routine exists: if every answer row is below one prompt field F,
  compared as digit strings from the top place down, the law ends in a reduction by F. So the reduction's step,
  x + d·F, is posed cut to its parts on the pair: the comparison and the link (round 6's cuts), and the addition, since
  x + d·N is a sum. The rule names no task; its decision and every row it read go to the run's log;
- **the wide reading** follows, once those three are routines, as in phase A.

**Its answers are the legal file's.** The comparison's are order facts, the link's are identities, the addition's are
its identity and a check run by the learner's own link and comparison; none is a sum made by Python (§2). The
cheapest reading decides what to practise and computes no target.

**The same climb, from nothing** (§4–§6). In training the climb waits for its first 64 answer rows (10–17 steps), reads
every answer below N, practises the comparison, the link and the addition at levels 0–2, reads `mod(mul(x,x),N)` at
level 3, and finds the round at level 4: the legal file's levels, with no law in the book. A tiny-core arm that poses
the addition only when the reader asks for it runs the same climb unit for unit.

**It certifies, 8 of 8** (§5, §6). On the contest's unmodified evaluator, one H100 each, the official manifest (seed
74) and seeds 1–7: T = 64 on both ladders, every rung 768 of 768, every split exact; `tools/verify_records.py`: "8 of 8
records check out". Training 472–634 s; the answer program by 256–344 s.

**Not "nothing designed"** (§7). What decides the practice now is designed: the residue rule, the step's cut table
(the comparison, the link, the addition), the reader's grammar, the cut and loop rule, the catalog of law forms, and
fast2's search and schedule. The residue rule is family knowledge: every task of the family ends in mod N, so on the
family (phase C) it will fire as it does here. That is not generality. Every pre-registered prediction held; about $10
(§9).
<!-- /PLAIN -->

## 1. Pre-registered

PREREG.md was frozen (read-only) at 03:07:05 PDT, after phase A's official run had reported and before any run of
this file on a GPU; the first tiny core was spawned at 03:07:52. Its sha256 is `e914afb3…2153`;
`checks/logs/prereg_sha256.txt` holds it with the code's hashes. Its header's time, 03:07, is the freeze time. Its
text, unchanged:

<!-- PREREG -->
# The no-book file (phase B): pre-registration

Written 2026-10-27 03:07 PDT, before any run of this file on a GPU, its tiny cores included, and after phase A's
official run had reported. NOBOOK.md §1 repeats this text unchanged and records the time and hash it was frozen with.

## The question

The wide file (phase A, integration/legal_wide/WIDE.md) still keeps three laws in its book: the comparison, the link and
the addition, on the pair. Its reader needs all three as routines before it can read anything. Round 6
(bar3/scale/reading/READING.md) emptied the book offline by cutting Horner's step down to the link and the
comparison, but its step came from a numpy reading at level 0, which the contest file cannot make. The brief: empty
the book in the phase-A file, every practice answer still from facts, identities or the learner's routines.

## Design (`make_nobook.py`, from the wide file, md5 `3bb5ceeb` asserted)

- **The book is empty** (`BOOK = ((), ())`).
- **The cheapest reading** (`read_fact`), before any routine exists:
  - the rule, which names no task: if every answer row is below one prompt field F, the law ends in a reduction by F,
    so the reduction's step x + d·F is posed, cut to its parts;
  - the check compares the answer and the field as digit strings, from their top place down (`compare`, the file's
    reader's comparison): no number is made;
  - the catalog's reductions are by N, so for F = N the step's parts are posed on the pair: the comparison (lt), the
    link (red), and, since x + d·N is a sum, the addition (add) (`STEP_PARTS`, a designed cut table, as round 6's
    `STEP_CUTS`);
  - the decision and every row it was read from (N:y) go to the climb's record and the log.
- **Their answers** are the legal file's: the comparison's by the order facts of r + q·N forms, the link's by the
  division law's identities (and the fix's probes from the learned addition table), the addition's by its identity and
  the sub.inverse check run by the learner's link and comparison. None is a sum or difference made by Python.
- **The wide reading** follows once the three are routines, unchanged from phase A.
- **Training:** the climb is handed answer rows once 64 are kept (`ANS_MIN`); until the cheapest reading is made, an
  empty pose is posed again at the next unit instead of ending the climb.
- **Room:** branches the file never takes with its settings (`deep_union`'s other mode, the deep probe without a
  focus, `pure_first`'s other mode, `alias_fit` off, the guard for other choosing rules), and shorter docstrings.
- **The arm `STEP_ADD=False`** (a research build, tiny core only): the addition is posed only when the reader asks for
  it, that is once the comparison and the link are routines and it is not.

## Runs

1. Two tiny cores on B200s: `t_nobook` (the file) and `t_nobook_addreader` (`STEP_ADD=False`). Each then has its answer
   programs checked on both ladders, T = 1 to 64, seen and unseen moduli (legal_wide's `checks/ladder_core.py`).
2. One official H100 run of the file, checked by `tools/verify_records.py`.
3. If it certifies, seeds 1–7 on H100s.

## Seen before this was written (local, CPU)

- **The file:** `submission_stage3_nobook.py`, md5 `b42ec304`, 261,893 bytes (251 under the limit); the arm's research
  build `research/submission_stage3_nobook_STEP_ADD-False.py`, md5 `49ff1146`. The contest's source check passes.
- **The first readings** (`checks/fact_local.py`; a fresh climb, an empty book, 400 answer rows of T = 1):
  - level 0, no routine: the cheapest reading reads every answer below N, on all 400 rows (the rows logged), and poses
    the comparison and the link (23 candidates); the addition is not posable yet (its check needs the link and the
    comparison). All 46 draws' answers equal Python's integers;
  - with the comparison and the link as routines (the legal tiny core's): the addition is posed (12 candidates), 24 of
    24 draws right;
  - with the addition too: the wide reading, `mod(mul(x,x),N)` on all 400 rows, 200 laws, 0.3 s, and its six laws,
    192 of 192 draws right.
- **combined_sp2's local checks** (`checks/logs/local_nobook.log`: 305 s, peak 3.1 GiB): the wide file's results part
  for part (`checks/compare_local.py`), but the climb's two parts. The toy climb waits for answer rows until step 27,
  then reads every answer below N on the 9 rows kept (the log shows them) and practises the comparison at level 0. The
  tiny-budget climb never has 8 answer rows in its 12 steps, so it never poses (and spends no work).

## Predictions

| what | prediction | confidence |
|---|---|---|
| t_nobook: level 0 | the cheapest reading: every answer below N, on all 400 rows; the comparison practised | high |
| t_nobook: levels 1–2 | the link at level 1, the addition at level 2 | high |
| t_nobook: level 3 | the wide reading, `mod(mul(x,x),N)` on all 400 rows; the row and the step | high |
| t_nobook: the round | level 4, then the answer program `C.S[T.round1 STA.X]` | high |
| t_nobook: both ladders | every answer program 768/768 on all 14 rungs | high |
| t_nobook_addreader | the same levels and round as t_nobook (the addition is posable only once the link and the comparison are routines, either way) | high |
| the official run: the first pose | waits for 64 answer rows, then reads every answer below N on all of them | high |
| the official run | the same levels; certifies: T = 64 on both ladders, every split exact | medium-high |
| the official run: training | 400–700 s | medium |
| seeds 1–7 | 7 of 7 certify | medium-high |
| money, phase B | about $10–15 | medium |

## Money

Phase A and B together: soft $150, cap $220. `integration/ablation_s3/spend.py 4750` runs before each launch. One
spend watcher for every app of phases A and B, its total the cap less one hour of eight GPUs (about $34): it stops
everything at $185. Every app is stopped when its runs end.
<!-- /PREREG -->

## 2. The changes

`make_nobook.py` reads the wide file (md5 asserted) and applies four groups of edits, each asserted to match exactly
once; then the same checks as phase A's builder (parse, compile, no undefined global, the contest's source policy at
256 KiB).

| group | edits | what |
|---|---|---|
| doc | 1 | the docstring |
| book | 1 | `BOOK` empty; `STEP_PARTS` and `STEP_ADD` |
| fact | 3 | `read_fact`; the readings' poses in `pose`; the first pose waits for answer rows; the log line |
| room | 7 | branches the file never takes; shorter docstrings |

**The cheapest reading** (`SelfClimb.read_fact`). Before any routine exists, the climb compares each kept answer row's
answer with each of its prompt fields, N and x, as digit strings: the file's own `compare` decides each row at the
highest place where the two strings differ, by that place's digits, and makes no number. If every answer is below one
field F, the law is read as ending in a reduction by F. The catalog's reductions are all by N (its patterns are
r + q·N); for F = N, the reduction's step, x + d·N, is posed cut to its parts on the pair (`STEP_PARTS`):
- the comparison (lt) and the link (red), round 6's `STEP_CUTS` (the step is B links; the link is a comparison and a
  select);
- the addition (add), because the step's own pattern x + d·N is a sum.

The decision (the field, the parts) and every row it was read from (N:y) go to the climb's record and to the log, so
the run's record holds them.

**This decides what to practise; it computes no target.** No practice answer comes from it. The answers of the three
laws are the legal file's (`law_target`, unchanged):
- the comparison's: the order facts of r + q·N forms (which of two terms is smaller, by their q and r, as the draws
  were built);
- the link's: the division law's identities (red(x, N) = x below N, red(x, x) = 0) and the fix's probes N < X < 2N,
  made from one entry of the learned addition table;
- the addition's: its identity (x + 0 = x) and the sub.inverse check, red(a + b, b) = a, run by the learner's link
  and comparison on the candidate's own output. No sum or difference is made by Python.

It is the open question LEGAL_SOURCES.md raises for a judge: a decision of what to practise, read from the answers,
by a designed rule.

**Waiting for answer rows.** The climb is handed the kept answer rows once 64 are kept (`ANS_MIN`). Until the cheapest
reading is made, an empty pose is posed again at the next unit, instead of ending the climb as an empty level would.

**The wide reading** follows unchanged once the three are routines.

**`STEP_ADD`** (True in the file): False poses the addition only when the reader asks for it, once the comparison and
the link are routines and it is not. Either way, the addition's own answers need the link and the comparison
(`law_target` raises NotPosable without them), so it can be practised no sooner.

**Room.** The new code needed about 1.2 KB; the wide file had 679 bytes to spare. Freed: branches the file never
takes with its settings (`deep_union`'s other mode; the deep probe without a focus, since `SP_DEEP_FOCUS` is
"closure"; `pure_first`'s other mode; `alias_fit` off; the guard for choosing rules other than lpca), and shorter
docstrings for the reader. The file: `submission_stage3_nobook.py`, md5 `b42ec30422c182c0972aebca5be0c470`, 5,021
lines, 261,893 bytes (251 under the limit).

## 3. Checks (local, CPU)

| check | result |
|---|---|
| [`checks/fact_local.py`](checks/fact_local.py): a fresh climb, an empty book, 400 answer rows of T = 1 | level 0, no routine: every answer below N on all 400 rows; the comparison and the link posed (23 candidates), the addition not yet posable; 46 draws' answers equal Python's integers. With the comparison and the link as routines: the addition posed (12), 24 of 24 right. With the addition too: the wide reading, `mod(mul(x,x),N)` on 400 rows, 200 laws, 0.3 s, and its six laws, 192 of 192 right |
| combined_sp2's `checks/local_combined.py`, every part ([`checks/logs/local_nobook.log`](checks/logs/local_nobook.log): 305 s, peak 3.1 GiB) | the wide file's results part for part ([`checks/compare_local.py`](checks/compare_local.py)), but the climb's two parts: the toy climb waits for answer rows until step 27, then reads every answer below N on the 9 rows kept (logged) and practises the comparison at level 0; the tiny-budget climb never has 8 answer rows in its 12 steps, so it never poses |

## 4. The tiny cores

[`core_c.py`](core_c.py) (combined_fast's, byte for byte: md5 `2e9c6a52`) on one B200 each through
[`modal_nobook.py`](modal_nobook.py) (app and Volume `legal-nobook`): the file's own climb with its own budgets,
offline, on 4,096 read rows and 400 answer rows of T = 1 with exact cells, then the answer program's search on rows
of T = 1, 2, 4 and 8. Records `records/t_nobook.json` and `records/t_nobook_addreader.json`.

| | the wide file (phase A) | `t_nobook` (no book) | `t_nobook_addreader` (`STEP_ADD=False`) |
|---|---|---|---|
| level 0 | the book's comparison and link (24 candidates); the comparison | the cheapest reading: every answer below N on all 400 rows (logged); its parts posed (24 candidates); the comparison | the same; the parts posed: the comparison and the link |
| level 1 | the link (6 bindings) | the same | the same |
| level 2 | the addition (3 bindings) | the same | the same: posed when the reader asks for it |
| level 3 | the reading, `mod(mul(x,x),N)` on 400 rows, 200 laws, 0.6 s; the row and the step | the same, 0.5 s | the same |
| level 4 and the round | the reduction, the row's loop, x·d; the round by calls at 249 s, work 1.12e9 | the same; 206 s, 1.12e9 | the same; 222 s, 1.12e9 |
| the answer program | `C.S[T.round1 STA.X]`, `C.S[T.round3 STA.X]`, 5.6 s; 224 unseen rows right | the same, 5.4 s | the same |
| the climb in all | 269 s, work 1.222e9 | 223 s, 1.222e9 | 241 s, 1.222e9 |
| both ladders | both answer programs 768/768 on all 14 rungs | the same | the same |

The two no-book cores are the same climb unit for unit: 51 units, each with the same machine work, the same picks,
library, rounds and answer programs; only the first decision's parts differ (with the addition, or without it until
the reader asks). The addition comes at level 2 either way, because its own answers need the link and the comparison.
The seconds differ by host.

## 5. The official run

One H100 through the shared runner (`tools/modal_runner.py --mode official`), the official manifest untouched, the
memory cap 48 GB. Record [`records/legal_nobook_official.json`](records/legal_nobook_official.json).

**It certifies.** `tools/verify_records.py`: "1 of 1 records check out". The embedded source's md5 is the file's
(`b42ec304`); the manifest is the official file; an H100 80GB HBM3 (driver 610.57.04); return code 0; the RESULT_JSON
line equals the result. T = 64 on both ladders, every rung 768 of 768; test 9,999 of 9,999, unseen T 10,002 of 10,002,
unseen moduli 10,002 of 10,002.

**The climb, from an empty book** (the record's log lines):

| step | level | what |
|---|---|---|
| 563 | | the climb starts (about 52 s); no answer row handed to it yet, so its pose waits |
| 576 | 0 | 65 answer rows kept: the cheapest reading, every answer below N on all 65 (each row logged, N:y); the step's parts posed, 22 candidates; the comparison |
| 581, 587 | 1, 2 | the link; the addition |
| 593 | 3 | the wide reading: `mod(mul(x,x),N)` on all 141 rows kept by then, 200 laws, 0.6 s; the row and the step |
| 613 | 4 | the reduction, the row's loop, x·d |
| 629 | 4 | the round by calls (2 calls) |
| 630 | | the answer programs `C.S[T.round1 STA.X]`, `C.S[T.round2 STA.X]`, `C.S[T.round3 STA.X]` (about 318 s) |
| 829 | | frozen (about 362 s) |

Against the wide file's official record (phase A): the same levels, 13 steps later (the wait for answer rows); the
round at step 629 against 613; frozen at 829 against 813; the climb's work 1.865e9 against 1.274e9 (the answer
program's search, over a library from other rows, 7.2e8 against 1.1e8); training 573.9 s against 554.1 s; evaluation
398.3 s against 391.9 s. T.round1 is `LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.X T.red.x+on.XN`.

## 6. Seeds 1–7

The file byte for byte (`b42ec304`), once for each of seeds 1 to 7: the official manifest with only `runtime.seeds`
changed, one H100 each, all seven at once, through the shared runner (`--mode seeds --hardware h100`, memory cap 48
GB). Records `records/legal_nobook_h100_seed<N>.json`.

**Every seed certifies.** `tools/verify_records.py` on the official record and the seven: "8 of 8 records check out",
each embedding the file's md5, its manifest the official one but for the seed, return code 0, the RESULT_JSON line
equal to the result, T = 64 on both ladders with every rung 768 of 768, and test 9,999 of 9,999, unseen T 10,002 of
10,002, unseen moduli 10,002 of 10,002.

`summarize_seeds.py` and the climbs' lines ([`checks/logs/summarize_seeds.log`](checks/logs/summarize_seeds.log)):

| seed | climb starts | the cheapest reading, level 0: below N on | the wide reading, level 3 | the round, level 4 | the answer program | frozen | climb work | training | evaluation |
|---|---|---|---|---|---|---|---|---|---|
| 74 (official) | step 563 | step 576: 65 rows | step 593: 141 of 141, 0.6 s | step 629 | step 630, about 318 s | step 829, about 362 s | 1.865e9 | 574 s | 398 s |
| 1 | 584 | 599: 65 | 617: 168 of 168, 0.3 s | 650 | 651, 256 s | 850, 289 s | 1.137e9 | 472 s | 295 s |
| 2 | 584 | 597: 66 | 614: 140 of 140, 0.4 s | 650 | 651, 275 s | 850, 402 s | 1.266e9 | 582 s | 296 s |
| 3 | 612 | 626: 66 | 643: 146 of 146, 0.5 s | 674 | 675, 280 s | 874, 406 s | 1.276e9 | 595 s | 303 s |
| 4 | 653 | 667: 66 | 684: 157 of 157, 0.6 s | 718 | 719, 344 s | 918, 397 s | 1.695e9 | 634 s | 395 s |
| 5 | 609 | 619: 64 | 636: 141 of 141, 0.4 s | 670 | 671, 282 s | 870, 408 s | 1.525e9 | 585 s | 290 s |
| 6 | 623 | 635: 66 | 652: 145 of 145, 0.4 s | 688 | 689, 270 s | 888, 299 s | 1.316e9 | 477 s | 291 s |
| 7 | 565 | 582: 69 | 599: 150 of 150, 0.8 s | 632 | 633, 343 s | 832, 389 s | 1.680e9 | 616 s | 626 s |

**The same climb on every seed.** Each waits 10 to 17 steps for its first 64 answer rows; reads every answer below N
on all of them and poses the comparison, the link and the addition; practises them at levels 0, 1 and 2; reads
`mod(mul(x,x),N)` at level 3 on every row kept (200 laws); finds its round at level 4 by calls; and its answer program
`C.S[T.round1 STA.X]` the next step. The rounds vary by seed in their bindings: the row's loop or x·d's, the reduction
in X or Y. Seed 7's evaluation took 626 s of the 1,800 allowed.

Against the wide file's eight (phase A): the answer program by 256–344 s against 244–326 s; frozen by 289–408 s
against 304–429 s; training 472–634 s against 478–658 s; the climb's work 1.14e9–1.87e9 against 0.99e9–1.95e9.

## 6a. Predictions against outcomes

| prediction | outcome | |
|---|---|---|
| t_nobook: level 0, every answer below N on all 400 rows; the comparison | so | held |
| t_nobook: the link at level 1, the addition at level 2 | so | held |
| t_nobook: level 3, the wide reading on all 400 rows; the row and the step | so | held |
| t_nobook: the round at level 4; `C.S[T.round1 STA.X]` | so | held |
| t_nobook: both ladders 768/768 on all 14 rungs | so | held |
| t_nobook_addreader: the same levels and round | so: the same climb unit for unit | held |
| the official run: the first pose waits for 64 answer rows, then reads every answer below N on all | so: 65 rows | held |
| the official run: the same levels; it certifies | so | held |
| the official run: training 400–700 s | 574 s | held |
| seeds 1–7: 7 of 7 certify | so | held |
| money, phase B: about $10–15 | about $10 | held |

Every pre-registered outcome held.

## 7. What is still designed

No law is posed because a list names it: the book is empty. That is not "nothing designed". What decides the practice
now, and what answers it:
- **The residue rule** (`read_fact`): if every answer row is below one prompt field F, the law ends in a reduction by
  F, so the reduction's step x + d·F is posed, cut to its parts. It names no task and reads its decision off the
  answers, but it is a designed rule, and family knowledge: the family's answers are residues mod N. It decides what to
  practise and computes no target. That is the open question LEGAL_SOURCES.md raises for a judge: may a run decide its
  practice from the answers by a designed rule? **Every task of the family ends in mod N**, so on the family's other
  tasks (phase C) the rule fires as it does on H1. That is not generality.
- **The step's cut table** (`STEP_PARTS`): the comparison, the link and the addition, round 6's `STEP_CUTS` with the
  addition for the step's sum.
- **The reader's grammar** (phase A, WIDE.md §2): the leaves x and N, the constants as gaps, five operations, depth 3,
  the order of the tries.
- **The cut and loop rule** (`read_pairs`, the legal file's): an operation's laws f and add.f, on the patterns whose
  count is a digit or another number; mod as the book's red.
- **The catalog:** the grammar's forms (the patterns, the counts 1, d, o and x, r + q·N) and `law_target`'s rules,
  which answer every law from facts, identities and the learner's routines.
- **fast2's:** the instruction set, the search's sizes and rules, the choosing rule, the count loop and T's field,
  stage 1's reader of the prompt, the cells' shapes and the schedule (REVIEW.md §5, rule 7).

## 8. Where it still touches Python

As the wide file (WIDE.md §8) and the legal file (LEGAL_FILE.md §8), plus the cheapest reading:
- **Its decision** is `compare` on the GPU: per row, the highest place where the answer's digits and the field's
  differ, and the two digits there. One bit per field (every row below, or not) is read back to the host. No number is
  made from the digits.
- **The rows it was read from** are read back as text (N:y) for the record and the log only; nothing is decided from
  that text.
- **The parts posed** are names (lt, red, add on the pair), Python's bookkeeping, as the reading's (law, pattern)
  names are.
- **Waiting** for answer rows is a test of whether the climb has been handed any (`ANS_MIN`, the file's own), read
  each unit until the first reading is made.

## 9. Cost

- **Billed:** nothing of phase B was in the billing report at 10:57 UTC (its runs fell in the 10:00 UTC hour);
  `cost.py` reads it once it is.
- **By the runs' own times:** the eight H100 jobs' wall time, 7,474 s, at $3.95 an hour is $8.20; with the two tiny
  cores on B200s (about $1.20), the drivers and the fetches, about $10.
- **The budget:** phases A and B together, soft $150, cap $220: about $19 by the runs' times. The spend watcher's
  stop-all was $185; it never acted. `spend.py 4750` said LAUNCH OK before every launch; the month's bill was $3,953 at
  10:57 UTC (the morning's runs not in it yet), under the stop of $4,750.
- **Apps:** all stopped.


## 10. Files and how to rerun

Everything is in `cpu_first/integration/legal_nobook/`; nothing outside it was written.

| file | what |
|---|---|
| `submission_stage3_nobook.py` | the file, md5 `b42ec30422c182c0972aebca5be0c470`, 261,893 bytes (251 under the limit), written by `make_nobook.py` |
| `research/submission_stage3_nobook_STEP_ADD-False.py` | the arm's build (`make_nobook.py STEP_ADD=False`), md5 `49ff1146` |
| `make_nobook.py` | the builder (§2): the wide file, md5 asserted, then the doc, book, fact and room edits |
| `PREREG.md` | frozen; `checks/logs/prereg_sha256.txt` holds its hash and the code's |
| `core_c.py`, `modal_nobook.py` | the tiny core (combined_fast's, byte for byte) and its launcher (app and Volume `legal-nobook`) |
| `cost.py` | §9: the billing report's hours for this phase's apps |
| `checks/fact_local.py` | the first readings on a fresh climb (§3) |
| `checks/compare_local.py` | combined_sp2's local checks on this file against the wide file's, part by part |
| `checks/logs/` | the local checks, the ladder checks, the spend watchers, the fetches |
| `records/` | the tiny cores (`t_nobook*.json`, `.log`, `_app.log`), the official record (`legal_nobook_official.json`), seeds 1–7's (`legal_nobook_h100_seed<N>.json`) |

The ladder check is the wide folder's `checks/ladder_core.py`. From `cpu_first/`, with
`PY=~/github/sotaku/venv/bin/python` and `PYTHONDONTWRITEBYTECODE=1` exported:

```bash
$PY integration/legal_nobook/make_nobook.py
$PY integration/legal_nobook/make_nobook.py STEP_ADD=False
$PY integration/legal_nobook/checks/fact_local.py
COMBINED_FILE=integration/legal_nobook/submission_stage3_nobook.py PYTHONPATH=. $PY integration/combined_sp2/checks/local_combined.py policy grad machines isa designed buffers replay climb climb_tiny fix book
python3 integration/legal_nobook/checks/compare_local.py
$PY integration/ablation_s3/spend.py 4750
$PY -m modal run --detach integration/legal_nobook/modal_nobook.py --envs '[{"TAG": "t_nobook"}]'
$PY -m modal run --detach integration/legal_nobook/modal_nobook.py --envs '[{"TAG": "t_nobook_addreader", "FILE": "research/submission_stage3_nobook_STEP_ADD-False.py"}]'
$PY -m modal run integration/legal_nobook/modal_nobook.py --mode fetch
$PY integration/legal_wide/checks/ladder_core.py integration/legal_nobook/records/t_nobook.json integration/legal_nobook/submission_stage3_nobook.py
$PY -m modal run --detach tools/modal_runner.py --mode official --file integration/legal_nobook/submission_stage3_nobook.py --memory-gb 48 --name legal_nobook_official --out integration/legal_nobook/records --run legal-nobook-official
$PY -m modal run --detach tools/modal_runner.py --mode seeds --file integration/legal_nobook/submission_stage3_nobook.py --seeds 1,2,3,4,5,6,7 --hardware h100 --memory-gb 48 --name legal_nobook --out integration/legal_nobook/records --run legal-nobook-h100-seeds
python3 tools/verify_records.py integration/legal_nobook/records/legal_nobook_official.json=integration/legal_nobook/submission_stage3_nobook.py integration/legal_nobook/records/legal_nobook_h100_seed{1,2,3,4,5,6,7}.json=integration/legal_nobook/submission_stage3_nobook.py
$PY integration/legal_nobook/cost.py
```
