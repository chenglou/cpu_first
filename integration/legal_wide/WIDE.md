# The wide file: the legal file with a wider reader (phase A)

*Written 2026-10-27. Folder `cpu_first/integration/legal_wide/`. The input is the legal file,
`integration/combined_legal/submission_stage3_legal.py`, read from its committed bytes (commit `8e3c7fe`, md5
`e2fe7a68`, asserted); nothing outside this folder was written, and the legal file and its records are unchanged.*

## In plain words

<!-- PLAIN -->
**What was asked.** The legal file (LEGAL_FILE.md, certified 8 of 8) reads the target's law with a grammar made for H1:
chains of add, mul and mod over x and N, depth 2. The brief: widen it, with subtract, power, constants read from the
data and depth 3, every value still computed by the learner's own routines; show it still reads x² mod N's law, how
many candidates it now weighs and what that costs; certify it.

**What changed** (§2), in one new file, [`submission_stage3_wide.py`](submission_stage3_wide.py) (md5 `3bb5ceeb`,
261,465 bytes, 679 under the limit), built from the legal file's committed bytes by asserted edits
([`make_wide.py`](make_wide.py)):
- **five operations:** add; sub (the difference the learner's comparison leaves, kept where the learner's addition
  gives the minuend back); mul (repeated addition, the book's own); pow (repeated mul); mod (the learner's link by
  doubling);
- **constants read from the data:** when no law of x and N is read, a gap by which a reduction mod N misses every
  answer, either way round, joins the leaves;
- **depth 1 to 3,** each depth's candidates computed together, one call per routine, and only the first chain of each
  distinct value kept;
- **room:** code the file never runs, the miner's report-only facts about the answers, and history notes.

**It still reads H1's law, and costs nothing** (§3–§5, §7). On H1's rows it weighs 200 candidates and reads
`mod(mul(x,x),N)` at depth 2, exact on every answer row (the legal file's reader: 35 candidates, about a fifth of the
rows). It takes 0.5 s of the official run and a tenth of the legal file's reading's machine work. Each depth goes
through each routine in one call, and the link by doubling replaces up to 1,000 single links. The climb is the legal
file's, step for step: the same levels, rounds and answer program.

**It is broader than H1 needs** (§7). Given other formulas' answers on H1's rows (a test outside any run), it reads x³,
x² + 1, x² + 3, x² − 1 and x² + x mod N, the constants as gaps. It does not read 10 x² (a factor is not a gap) or
10 x² + 1 (depth 4). Reading a law is not learning it: phase C will run the family.

**It certifies, 8 of 8** (§5, §6). On the contest's unmodified evaluator, one H100 each, the official manifest (seed
74) and seeds 1–7: T = 64 on both ladders, every rung 768 of 768, every split exact; `tools/verify_records.py`: "8 of 8
records check out". Training 478–658 s; the answer program by 244–326 s.

**Still designed** (§8): the grammar (its leaves, five operations, depth 3, the gap rule for constants, the order of
the tries), beside what the legal file designs: the three laws in the book, the cut and loop rule, and fast2's
instruction set, search and schedule. Every pre-registered prediction held; about $9 (§9).
<!-- /PLAIN -->

## 1. Pre-registered

PREREG.md was frozen (read-only) at 02:00:33 PDT, before any run on a GPU; the tiny core was spawned at 02:44:28 (the
network was down from 02:00 to 02:26). Its sha256 is `50454cf9…4094`; `checks/logs/prereg_sha256.txt` holds it with
the code's hashes. Its header's time, 02:00, is the freeze time. Its text, unchanged:

<!-- PREREG -->
# The wide file (phase A): pre-registration

Written 2026-10-27 02:00 PDT, before any run on a GPU, the tiny core included. WIDE.md §1 repeats this text
unchanged and records the time and hash it was frozen with.

## The question

The legal file (integration/combined_legal/LEGAL_FILE.md, certified 8 of 8) reads the target's law with a grammar made
for H1: chains of add, mul and mod over x and N, depth 2. The brief: widen it, with subtract, power, constants read
from the data and depth 3, beside add, mul and mod, every value still computed by the learner's own routines; show it
still reads x² mod N's law, how many candidate laws it now weighs, and what the extra time costs; then certify the
file.

## Design (`make_wide.py`, from the legal file's committed bytes at 8e3c7fe, md5 asserted)

- **The grammar** (`LawReader`): left-deep chains over the leaves x and N, depth 1 to 3, of five operations:
  - add: the learner's addition, where the sum fits the places;
  - sub: the difference the learner's comparison leaves, where it says "not below" and the learner's addition gives
    the minuend back;
  - mul: repeated addition by doubling (`repeat_add`, the book's own), where both factors fit 8 places;
  - pow: repeated mul, for a count of 1 to 8;
  - mod: the learner's link by doubling: b, 2b, 4b, ... made by the learner's addition while the comparison says below
    the dividend, then the link with each, largest first (the legal file repeated the link one at a time, at most
    1,000 times).
- **The data's constants**: when no law of x and N is read, a number c by which the reduction mod N of a leaf or a
  one-operation chain misses every answer, either way round (mod N, by the same operations), joins the leaves, and
  the chains are made again.
- **The order**: depth by depth; each depth's candidates computed together, operation by operation, their rows
  stacked; only the first chain of each distinct value (its digits on its rows, and its rows) kept; the first exact on
  every row where it is computed, at least 4 rows, is read.
- **What it poses**: as before, each operation's laws (f and add.f; sub has none in the book, pow is mul's) on the
  patterns whose count is a digit or another number.
- **Room** (the file stays under 256 KiB): code the file never runs with its settings (counts always by doubling; one
  acceptance root shared, kept for a check's own use), the miner's report-only parts (its facts about the answers, and
  divisibility), and the settings' history notes.
- Everything else is the legal file's.

## Runs

1. The tiny core on one B200 (`core_c.py`, combined_fast's), then its answer programs on both ladders, T = 1 to 64,
   seen and unseen moduli (`checks/ladder_core.py`).
2. One official H100 run (`tools/modal_runner.py --mode official`, the manifest untouched), checked by
   `tools/verify_records.py`.
3. If it certifies, seeds 1–7 on H100s.

## Seen before this was written (local, CPU)

- **The file**: `submission_stage3_wide.py`, md5 `3bb5ceeb`, 261,465 bytes (679 under the limit); the contest's
  source check passes.
- **The reading on H1** (`checks/read_wide.py` MODE=read; the legal tiny core at the start of its level 3, 400 answer
  rows of T = 1): `mod(mul(x,x),N)` on all 400 rows (the legal file's reader: 83), 200 candidates weighed (its: 35),
  0.3 s (its: 2.2 s). The link by doubling reduces every row; the repeated link stopped at 1,000 links.
- **Its reach** (MODE=breadth; the same rows, the answers of other formulas made by Python's integers, a test outside
  any run):

  | formula | read | rows | candidates | seconds |
  |---|---|---|---|---|
  | x³ mod N | `mod(mul(mul(x,x),x),N)` | 158 of 400 | 1,140 | 1.0 |
  | x² + 1 mod N | `mod(add(mul(x,x),1),N)`, the constant 1 read | 400 | 3,825 | 3.1 |
  | x² + 3 mod N | `mod(add(mul(x,x),3),N)`, the constant 3 read | 400 | 8,025 | 5.6 |
  | x² − 1 mod N | `mod(sub(mul(x,x),1),N)`, the constant 1 read | 400 | 3,825 | 3.2 |
  | x² + x mod N | `mod(add(mul(x,x),x),N)` | 400 | 1,140 | 1.0 |
  | 10 x² mod N | none (10 is no gap) | | 1,140 | 1.0 |
  | 10 x² + 1 mod N | none (depth 4) | | 1,140 | 1.0 |

- **The pose** (MODE=pose): the six laws in every binding, 96 candidates; 192 of 192 targets equal Python's integers.
- **combined_sp2's local checks** (`checks/logs/local_wide.log`: 358 s, peak 3.8 GiB): the legal file's results part
  for part (`checks/compare_local.py`), but the book's code (92 lines from `book_gen_fix.py`'s: the room edits);
  check_books part 2 the same, 6,204 targets equal the laws' values. A first build cut the acceptance's unshared path,
  which `check_fix_variants` uses; it is kept, made compact.
- **The ladder check** (`checks/ladder_core.py`) on the legal file's own tiny core: its two answer programs 768/768 on
  all 14 rungs, T = 1 to 64, seen and unseen moduli.

## Predictions

| what | prediction | confidence |
|---|---|---|
| the tiny core: the reading | level 3, `mod(mul(x,x),N)`, on all 400 answer rows, about 200 candidates, under 30 s | high |
| the tiny core: the climb | the legal tiny core's levels; the round at level 4; the answer program `C.S[T.round1 STA.X]` | high |
| the tiny core: both ladders | every answer program 768/768 on all 14 rungs | high |
| the official run | certifies: T = 64 on both ladders, every split exact | medium-high |
| the official run: the reading | level 3, on every answer row kept by then, under 30 s | medium-high |
| the official run: training | 400–650 s (the legal file's: 465 s) | medium |
| seeds 1–7 | 7 of 7 certify | medium-high |
| money | about $10–15 | medium |

## Money

Phase A and B together: soft $150, cap $220. `integration/ablation_s3/spend.py 4750` runs before each launch. One
spend watcher for this study's apps, its total the cap less one hour of eight GPUs (about $34): it stops everything at
$185. Every app is stopped when its runs end.
<!-- /PREREG -->

## 2. The changes

`make_wide.py` reads the legal file from git (`8e3c7fe`), asserts its md5, and applies four groups of edits in order,
each asserted to match exactly once (a cut between two anchors pinned by the md5 of the text it removes). It then
checks that the file parses, compiles, reads no undefined global and passes the contest's source policy at 256 KiB.

| group | edits | what |
|---|---|---|
| doc | 1 | the docstring points here |
| grammar | 1 | the reading's constants: `READ_POW` and `READ_DEPTH` in, `READ_LINKS` out |
| reader | 1 | `READ_OPS`, `LawReader` and `read_pairs`, rewritten (a cut) |
| room | 13 | code the file never runs, the miner's report-only parts, the settings' history notes |

**The grammar** (`LawReader`). Left-deep chains over the leaves x and N, of depth 1 to 3, of five operations. Every
value is computed on the GPU by the learner's own routines (`Ops.run`), the way the book's constructions compute:
- add: the learner's addition, where the top place of both operands is zero, so the sum fits;
- sub: the difference the learner's comparison leaves in A, where its flag says "not below" and the learner's addition
  of the subtrahend gives the minuend back, digit for digit. The comparison's routine is a subtractor whose flag is its
  borrow; the addition confirms its difference on every row it is used;
- mul: repeated addition by doubling (`repeat_add`, the book's own), where both factors fit 8 places;
- pow: repeated mul, for a count (read off the exponent's digits) of 1 to `READ_POW` = 8;
- mod: the learner's link by doubling, as `repeat_add` repeats the addition by doubling: b, 2b, 4b, ... made by the
  learner's addition while the learner's comparison says below the dividend; then the link with each, largest first.
  Exact wherever the dividend's top place is zero. The legal file repeated the link one at a time, at most 1,000
  times, so a row whose quotient passed 1,000 could not be reduced.

**The data's constants.** When no law of x and N alone is read, the reader looks at the reductions it made, mod N, of
the leaves and one-operation chains. A number c by which one of them misses every answer row, either way round, is a
constant of the data: the gap, (y − r) mod N or (r − y) mod N, computed by the same sub and add, the same digits on
every row where it is computed (at least 4), and not zero. The constants join the leaves, and the chains are made
again. This is the near miss of SP_BREADTH.md's g2 ("constants taken from the data"), read here by the learner's
routines instead of in the round's search. Taking the gap mod N is family knowledge: the answers are residues.

**The order.** Depth by depth. Each depth's candidates are every operation on each kept chain of the depth before and
a leaf, either side; they are computed together, operation by operation, their rows stacked into one call of each
routine. Only the first chain of each distinct value (its digits on its rows, and its rows, hashed on the GPU by the
file's own `hash_rows`) is kept, and only where it is computed on at least `READ_MIN` = 4 rows. The first kept chain
exact on every row where it is computed is the law read.

**What a reading poses** (`read_pairs`) is the legal file's rule: each operation's laws, f and add.f, on the patterns
whose count is a digit or another number. sub has no law in the book, so it poses nothing; pow poses mul's. A read law
of mul and mod poses the same six laws as before.

**Room.** The wider reader is 2,600 bytes more than the legal file's. The room edits free about 2,400:
- code the file never runs with its settings: `repeat_add`'s one-at-a-time loop (its counts are always by doubling)
  and the switch that chose it; the switch that shares one acceptance root (always on; the unshared path is kept,
  made compact, because `check_fix_variants` builds an acceptance of its own);
- the miner's report-only parts: `mine_answers`, the relations between the answers and N, x and T, which only the log
  ever showed (REVIEW.md §5 rule 14: "every fact about the answers, are only logged"), and the relations by
  divisibility by 2, 3 and 5, which no lesson reads;
- the settings' history notes in the header (sp2's and earlier files' values; COMBINED.md and FAST.md keep them).

The file: `submission_stage3_wide.py`, md5 `3bb5ceebf1de115501efc3398af8a457`, 5,018 lines, 261,465 bytes (679 under
the limit). The learned parts, the climb, the book of three laws and everything else are the legal file's.

## 3. Checks (local, CPU)

| check | result |
|---|---|
| [`checks/read_wide.py`](checks/read_wide.py) MODE=read: the legal tiny core at the start of its level 3 (the comparison, the link and the addition as routines), 400 answer rows of T = 1 | `mod(mul(x,x),N)` on all 400 rows (the legal file's reader: 83), 200 candidates weighed (its: 35), 0.3 s (its: 2.2 s) |
| read_wide.py MODE=breadth: the same rows with the answers of other formulas of the family, made by Python's integers (a test outside any run) | 6 of 8 read (§7) |
| read_wide.py MODE=pose | the six laws in every binding, 96 candidates; 192 of 192 targets equal Python's integers |
| combined_sp2's `checks/local_combined.py`, every part ([`checks/logs/local_wide.log`](checks/logs/local_wide.log): 358 s, peak 3.8 GiB) | the legal file's results part for part ([`checks/compare_local.py`](checks/compare_local.py)), but the policy (261,465 bytes passes) and the book's code (92 lines from `book_gen_fix.py`'s: the room edits); check_books part 2 the same, 6,204 targets equal the laws' values |
| [`checks/ladder_core.py`](checks/ladder_core.py) on the legal file's own tiny core, as a check of the check | its two answer programs 768/768 on all 14 rungs |

The first build cut the unshared acceptance path; `local_combined.py fix` failed on it (`check_fix_variants` builds an
acceptance with no shared root), and the path is kept, made compact. The pre-registration came after.

## 4. The tiny core

[`core_c.py`](core_c.py) (combined_fast's, byte for byte: md5 `2e9c6a52`) on one B200 through
[`modal_wide.py`](modal_wide.py) (app and Volume `legal-wide`): the file's own climb with its own budgets, offline, on
4,096 read rows and 400 answer rows of T = 1 with exact cells, then the answer program's search on rows of T = 1, 2,
4 and 8. Record `records/core_wide.json`; the legal file's is `combined_legal/records/core_legal.json`.

| | the legal file | this file |
|---|---|---|
| levels 0–2 | the comparison, the link, the addition; level 2 ends at 53.4 s | the same; 53.0 s |
| level 3's pose, the reading in it | 16.1 s, work 4.78e6 | 2.0 s, work 4.43e5 |
| the reading | `mod(mul(x,x),N)`: 35 laws, exact on 83 of 400 rows, 14.7 s | the same law: 200 laws, all 400 rows, 0.6 s |
| level 3 | the row and the step; ends at 164.4 s | the same; 148.3 s |
| level 4 and the round | the reduction, the row's loop, x·d; the round at 270.3 s, work 1.12e9 | the same; 249.2 s, 1.12e9 |
| the answer program | `C.S[T.round1 STA.X]`, `C.S[T.round3 STA.X]`, 5.8 s; 224 unseen rows right | the same, 5.6 s; the same |
| the climb in all | 292 s, work 1.226e9 | 269 s, work 1.222e9 |
| both ladders (`checks/ladder_core.py`) | both answer programs 768/768 on all 14 rungs | the same |

The wider grammar costs no time here: its 200 candidates take 0.6 s on the B200, against 14.7 s for the legal file's
35, because each depth's candidates go through each routine in one call, and because the link by doubling takes about
50 links where the repeated link took up to 1,000.

## 5. The official run

One H100 through the shared runner (`tools/modal_runner.py --mode official`), the official manifest untouched, the
memory cap 48 GB. Record [`records/legal_wide_official.json`](records/legal_wide_official.json).

**It certifies.** `tools/verify_records.py`: "1 of 1 records check out". The embedded source's md5 is the file's
(`3bb5ceeb`); the manifest is the official file; an H100 80GB HBM3 (driver 610.57.04); return code 0; the RESULT_JSON
line equals the result. T = 64 on both ladders, every rung 768 of 768; test 9,999 of 9,999, unseen T 10,002 of 10,002,
unseen moduli 10,002 of 10,002.

Against the legal file's official record (`integration/combined_fast/checks/summarize_seeds.py`: the clock at a step
is the evaluator's progress line before it plus the climb's own unit seconds, so "about"):

| | the legal file | this file |
|---|---|---|
| the climb starts | step 563, about 42 s | step 563, about 54 s |
| the reading, level 3 | step 580: 35 laws, exact on 23 of the 79 rows kept, 8.8 s | step 580: 200 laws, all 79 rows, 0.5 s |
| the round, level 4 | step 613, by calls | step 613, by calls |
| the answer program | step 614, about 272 s | step 614, about 300 s |
| frozen | step 813, about 306 s | step 813, about 340 s |
| the climb's work | 1.278e9 of 4.3e9 | 1.274e9 of 4.3e9 |
| training (to the 20,000-step cap) | 465.3 s | 554.1 s |
| evaluation | 295.3 s | 391.9 s |
| peak host memory | 12.6 GB | 12.6 GB |
| T.round1, the answer programs | `LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.Y T.red.x+on.YN`; `C.S[T.round1 STA.X]`, `C.S[T.round3 STA.X]` | the same |

The climb is the legal file's step for step: the same levels at the same steps, the same rounds and answer programs.
The reading takes 0.5 s of the climb's level 3 pose (1.7 s) and 4.2e5 work units, against 8.8 s and 4.4e6. The
seconds differ by host: the legal file's run trained in 465 s, its seven seeds in 563–721 s, on the same steps.

## 6. Seeds 1–7

The file byte for byte (`3bb5ceeb`), once for each of seeds 1 to 7: the official manifest with only `runtime.seeds`
changed, one H100 each, all seven at once, through the shared runner (`--mode seeds --hardware h100`, memory cap 48
GB). Records `records/legal_wide_h100_seed<N>.json`.

**Every seed certifies.** `tools/verify_records.py` on the official record and the seven: "8 of 8 records check out",
each embedding the file's md5, its manifest the official one but for the seed, return code 0, the RESULT_JSON line
equal to the result, T = 64 on both ladders with every rung 768 of 768, and test 9,999 of 9,999, unseen T 10,002 of
10,002, unseen moduli 10,002 of 10,002.

`summarize_seeds.py` and the climbs' lines ([`checks/logs/summarize_seeds.log`](checks/logs/summarize_seeds.log)):

| seed | the reading, level 3: rows exact of rows kept | the round, level 4 | the answer program | frozen | climb work | training | evaluation |
|---|---|---|---|---|---|---|---|
| 74 (official) | step 580: 79 of 79, 0.5 s | step 613 | step 614, about 300 s | step 813, about 340 s | 1.274e9 | 554 s | 392 s |
| 1 | 601: 75 of 75, 0.5 s | 634 | 635, 313 s | 834, 363 s | 1.305e9 | 587 s | 353 s |
| 2 | 602: 95 of 95, 0.4 s | 633 | 634, 244 s | 833, 374 s | 0.989e9 | 546 s | 297 s |
| 3 | 629: 74 of 74, 0.5 s | 662 | 663, 303 s | 862, 429 s | 1.952e9 | 658 s | 354 s |
| 4 | 670: 85 of 85, 0.4 s | 704 | 705, 289 s | 904, 339 s | 1.215e9 | 514 s | 330 s |
| 5 | 626: 88 of 88, 0.4 s | 657 | 658, 287 s | 857, 414 s | 1.473e9 | 593 s | 306 s |
| 6 | 640: 86 of 86, 0.4 s | 673 | 674, 269 s | 873, 304 s | 1.279e9 | 478 s | 313 s |
| 7 | 582: 69 of 69, 0.5 s | 618 | 619, 326 s | 818, 374 s | 1.306e9 | 615 s | 416 s |

Every run reads `mod(mul(x,x),N)` at level 3 with 200 laws weighed, on every answer row kept by then, and reaches its
round at level 4 and its answer program `C.S[T.round1 STA.X]` the next step, at the legal file's own steps on every
seed. Seeds 1, 2, 4 and 5 multiply in the round by x·d's loop (`LDA.X T.mul.x,o.AX STA.X T.red.x+on.XN`), the others
by the row's, as with the legal file. Against the legal file's eight: the answer program by 244–326 s of training
against 272–349 s; training 478–658 s against 465–721 s; the climb's work 0.99e9–1.95e9 against 0.99e9–1.96e9.

## 6a. Predictions against outcomes

| prediction | outcome | |
|---|---|---|
| the tiny core: the reading at level 3, `mod(mul(x,x),N)`, all 400 rows, about 200 candidates, under 30 s | so: 200 candidates, 0.6 s | held |
| the tiny core: the legal tiny core's levels; the round at level 4; `C.S[T.round1 STA.X]` | so | held |
| the tiny core: every answer program 768/768 on all 14 rungs | so | held |
| the official run certifies | so | held |
| the official run: the reading at level 3, every kept row, under 30 s | so: 79 of 79, 0.5 s | held |
| the official run: training 400–650 s | 554 s | held |
| seeds 1–7: 7 of 7 certify | so | held |
| money: about $10–15 | about $9 | below the range |

Every pre-registered outcome held; the money came in a little under its range.

## 7. What the wider grammar weighs, what it reads, and what it costs

**What it weighs.** On H1's answer rows the reader weighs 200 candidates (20 of depth 1 and 180 of depth 2, over x and
N) and stops at depth 2, where `mod(mul(x,x),N)` is the first exact chain. A law outside depth 2 is looked for at
depth 3 (940 more candidates over x and N, 1,140 in all), and, with no law over x and N, with the data's constants
among the leaves (3,825 to 8,025 in all, in the tests below). The legal file's reader weighed 35 before it read H1's
law, one candidate at a time.

**What it reads** (`checks/read_wide.py` MODE=breadth: H1's 400 answer rows of T = 1, the other formulas' answers made
by Python's integers, outside any run; the legal tiny core's routines at its level 3; CPU):

| formula | read | rows exact | candidates | seconds |
|---|---|---|---|---|
| x² mod N (H1) | `mod(mul(x,x),N)` | 400 of 400 | 200 | 0.3 |
| x³ mod N | `mod(mul(mul(x,x),x),N)` | 158 | 1,140 | 1.0 |
| x² + 1 mod N | `mod(add(mul(x,x),1),N)`, the constant 1 read as a gap | 400 | 3,825 | 3.1 |
| x² + 3 mod N | `mod(add(mul(x,x),3),N)`, the constant 3 | 400 | 8,025 | 5.6 |
| x² − 1 mod N | `mod(sub(mul(x,x),1),N)`, the constant 1, the other way round | 400 | 3,825 | 3.2 |
| x² + x mod N | `mod(add(mul(x,x),x),N)` | 400 | 1,140 | 1.0 |
| 10 x² mod N | none: 10 is a factor, not a gap | | 1,140 | 1.0 |
| 10 x² + 1 mod N | none: depth 4 | | 1,140 | 1.0 |

So the grammar is broader than H1 needs: of the family's formulas on H1's prompts (SP_BREADTH.md §1), it reads the
cube and the three constant ones, and x² + x, and misses the two with a factor 10. x³'s law is exact on the 158 rows
where x² fits 8 places; the others' on every row. Reading a law is not learning it: a run on those tasks would still
need the climb to practise the posed laws and the round's search to find the round (phase C).

**What it costs.** Nothing measurable in a run: 0.6 s in the tiny core and 0.5 s in the official run, against 14.7 s and
8.8 s for the legal file's narrower reader. Level 3's pose, the reading in it, did 4.2–4.4e5 units of machine work, a
tenth of the legal file's, and 0.01% of the climb's budget. The candidates of a depth go through each routine in one
call, and the link by doubling makes a reduction about 50 links instead of up to 1,000. The breadth tests' largest, x² +
3 with its constants, took 5.6 s on a CPU.

## 8. Where it still touches Python

As the legal file (LEGAL_FILE.md §8), with these differences:
- **sub** takes the difference the learner's comparison leaves in A, a routine's output, and keeps it only where the
  learner's addition gives the minuend back, digit for digit.
- **pow** counts its multiplications with a count read off the exponent's digits (`count_of`), compared with
  `READ_POW` (8): a count, as `repeat_add` reads its own.
- **mod** decides how far to double by the learner's comparison's flag, and the link by doubling subtracts with the
  learner's link. No quotient is made.
- **The constants** are gaps made by the same sub and add, kept when the same digits on every row; their names (the
  digits read as text) are only for the log.
- **Distinct values** are found by hashing each candidate's digits and rows on the GPU (`hash_rows`), and the hashes,
  row counts and exactness bits are read back to the host, a small result per candidate, to steer the reading. No
  digit is read back but for the constants' names.
- **Removed:** the miner's facts about the answers (REVIEW.md §5, rule 14: "every fact about the answers, are only
  logged") are no longer computed, and its divisibility relations are gone.

## 9. Cost

- **Billed** (`cost.py`, at 10:26 UTC): $1.25 through the 09:00 UTC hour: the tiny core on a B200 $0.59, the official
  run's first part $0.66. The 10:00 UTC hour (the official run's end and the seven seeds) was not in the report yet;
  `cost.py` reads it once it is.
- **By the runs' own times:** the eight H100 jobs' wall time, 7,353 s, at $3.95 an hour is $8.07; with the tiny core,
  the drivers and the fetches, about $9 in all.
- **The budget:** phases A and B together, soft $150, cap $220. The spend watcher's stop-all was $185 (the cap less an
  hour of eight GPUs); it never acted. `spend.py 4750` said LAUNCH OK before every launch; the month's bill was
  $3,953 at 10:17 UTC, under the stop of $4,750.
- **Apps:** all stopped.


## 10. Files and how to rerun

Everything is in `cpu_first/integration/legal_wide/`; nothing outside it was written.

| file | what |
|---|---|
| `submission_stage3_wide.py` | the file, md5 `3bb5ceebf1de115501efc3398af8a457`, 261,465 bytes (679 under the limit), written by `make_wide.py` |
| `make_wide.py` | the builder (§2): the legal file from git at `8e3c7fe`, md5 asserted, then the doc, grammar, reader and room edits |
| `PREREG.md` | frozen; `checks/logs/prereg_sha256.txt` holds its hash and the code's |
| `core_c.py`, `modal_wide.py` | the tiny core (combined_fast's, byte for byte) and its launcher (app and Volume `legal-wide`) |
| `cost.py` | §9: the billing report's hours for this study's apps |
| `checks/read_wide.py` | the reading (MODE=read), its reach (MODE=breadth) and its pose (MODE=pose) on the legal tiny core's state |
| `checks/ladder_core.py` | a tiny core's answer programs on both ladders, with the file's own answer machine |
| `checks/compare_local.py` | combined_sp2's local checks on this file against the legal file's, part by part |
| `checks/logs/` | the local checks, the readings, the ladder checks, the spend watchers, the fetches |
| `records/` | the tiny core (`core_wide.json`, `.log`, `_app.log`), the official record (`legal_wide_official.json`), seeds 1–7's (`legal_wide_h100_seed<N>.json`) |

From `cpu_first/`, with `PY=~/github/sotaku/venv/bin/python` and `PYTHONDONTWRITEBYTECODE=1` exported:

```bash
$PY integration/legal_wide/make_wide.py
for m in read breadth pose; do MODE=$m $PY integration/legal_wide/checks/read_wide.py; done
COMBINED_FILE=integration/legal_wide/submission_stage3_wide.py PYTHONPATH=. $PY integration/combined_sp2/checks/local_combined.py policy grad machines isa designed buffers replay climb climb_tiny fix book
python3 integration/legal_wide/checks/compare_local.py
$PY integration/ablation_s3/spend.py 4750
$PY -m modal run --detach integration/legal_wide/modal_wide.py --envs '[{"TAG": "core_wide"}]'
$PY -m modal run integration/legal_wide/modal_wide.py --mode fetch
$PY integration/legal_wide/checks/ladder_core.py integration/legal_wide/records/core_wide.json
$PY -m modal run --detach tools/modal_runner.py --mode official --file integration/legal_wide/submission_stage3_wide.py --memory-gb 48 --name legal_wide_official --out integration/legal_wide/records --run legal-wide-official
$PY -m modal run --detach tools/modal_runner.py --mode seeds --file integration/legal_wide/submission_stage3_wide.py --seeds 1,2,3,4,5,6,7 --hardware h100 --memory-gb 48 --name legal_wide --out integration/legal_wide/records --run legal-wide-h100-seeds
python3 tools/verify_records.py integration/legal_wide/records/legal_wide_official.json=integration/legal_wide/submission_stage3_wide.py integration/legal_wide/records/legal_wide_h100_seed{1,2,3,4,5,6,7}.json=integration/legal_wide/submission_stage3_wide.py
$PY integration/legal_wide/cost.py
```
