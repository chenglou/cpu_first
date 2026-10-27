# The legal file: fast2 with a three-law book and a reading of the target

*Written 2026-10-26/27. Folder `cpu_first/integration/combined_legal/`. The input is fast2,
`integration/combined_fast/submission_stage3_fast2.py`, read from its committed bytes (commit `f237a61`, md5
`41b9a013`, asserted); nothing outside this folder was written.*

## In plain words

<!-- PLAIN -->
**What was asked.** "But can it do the contest's x^(2^T) mod N?" Test 4 (bar3/scale/legalloops/LEGALLOOPS.md) showed
in bar 3's tiny core that a law book of three laws on the pair (the comparison, the link and the addition) is enough:
the target's law, read from its answer rows by the learner's own routines, poses the rest. Here the same is put into a
real contest entry: fast2, the five-piece stage-3 file made fast and certified 8 of 8 on the official H100, with its
generated law book replaced by those three laws and that reading. Everything else is fast2's: the weak-check fix, the
strict build, the learned loop over T, the generic instructions, the Triton kernel.

**What changed** (§2), in one new file, [`submission_stage3_legal.py`](submission_stage3_legal.py) (md5 `e2fe7a68`,
261,216 bytes, 928 under the limit), built from fast2's bytes by 18 asserted edits ([`make_legal.py`](make_legal.py)):
1. *The book:* the climb poses three laws on the pair, not every law of the grammar.
2. *The reading:* once those three are routines, the climb reads the target's law from the answer rows it already
   keeps, at full size. It tries chains of add, mul and mod over x and N, depth 1 then 2. Every value is computed by
   the learner's own routines: its addition; repeated addition, the book's own construction; and its link, repeated
   until its comparison says "below". The first chain that is right, digit for digit, on every row where it can be
   computed is the law read.
3. *What it poses:* each operation of the law read, on the grammar's patterns whose count is a digit (the operation
   cut to one place) or another number (its loop over the digits). For mod(mul(x,x),N) that is the multiply-add row
   and x·d, their loops, Horner's step and the reduction. The book's existing rules answer each.
4. *Room:* the grammar's rules for laws that nothing now poses (eq, min, max, div, sub, the base forms) are cut.

**Checks** (§3): combined_sp2's local checks give fast2's results part for part, and the book's targets equal the laws'
values. On fast2's own climb, the reading reads mod(mul(x,x),N) and poses six laws whose answers all equal Python's
integers.

**The tiny core** (§4, one B200): the same route as fast2's, with less. The comparison, the link and the addition at
levels 0–2; the reading, the row and the step at level 3; the multiply and the reduction at level 4, and the round. It
took 270 s and 1.1e9 work units, against fast2's 348 s and 2.5e9. The answer program `C.S[T.round1 STA.X]` came 6 s
later, right on 224 rows it never saw.

**It certifies, 8 of 8** (§5, §6). On the contest's unmodified evaluator, one H100, the manifest untouched (seed 74)
and then seeds 1–7: T = 64 on both ladders, every rung 768 of 768, test 9,999 of 9,999, unseen T and unseen moduli
10,002 of 10,002 each. `tools/verify_records.py`: "8 of 8 records check out". Every run reads the law at level 3, on
11–23 of the 69–95 answer rows kept by then, and finds its round at level 4. Its answer program comes by 272–349 s of
training, against fast2's 492–555 s; training ends at the step cap in 465–721 s, against 759–883 s.

**What it designs that fast2 did not, and the reverse** (§7). New: the three laws, the reading's grammar (leaves,
operations, depth, its limits) and the posing rule. The reading's grammar was chosen knowing the task family: H1's law
is a depth-2 chain of its operations. Gone: the grammar's book posed whole, which had the climb practise 7 concepts
this file never poses, and five operations and the base forms, cut for room. What the climb practises is now decided
by three laws and the target, read by the learner's routines.

**Where it touches Python** (§8): as fast2 does (REVIEW.md §5), plus the reading's control: its loops, the law's text
and names, and small results read back to steer it (row counts, a bit per link, a bit per comparison). No value it
compares is made by Python's arithmetic. Its decisions are digit equality with the answer rows and digit facts. Its
counts are read off digits, as fast2's practice constructions already read theirs.

**Cost** (§9): about $10 of the soft $100.
<!-- /PLAIN -->

## 1. What was asked

The user asked, of test 4 (bar3/scale/legalloops/LEGALLOOPS.md): "but can it do the contest's x^2^T mod N?" Test 4
showed, in bar 3's tiny core, that a book of three laws on the pair (the comparison, the link and the addition) is
enough: the target's law, read from its answer rows by the learner's own routines, poses the rest. The brief: put that
into fast2, the five-piece stage-3 file made fast (certified 8 of 8 on the official H100), in place of its generated
law book; keep everything else (the weak-check fix, the strict build, the learned loop over T and its answer program
`C.S[T.round1 STA.X]`, the generic instructions, the Triton kernel and the other speedups); build it as asserted
edits of fast2's committed bytes; keep the rules (no practice target from Python's arithmetic, everything on the GPU,
256 KiB and the contest's source check, the hour); check it as combined_sp2's local checks, a tiny core on one B200,
one official H100 run with `tools/verify_records.py`, and if it certifies, seeds 1–7 on H100s.

## 2. The changes

`make_legal.py` reads fast2 from git (`f237a61`), asserts its md5, and applies four groups of edits in order, each
asserted to match exactly once. It then checks that the file parses, compiles, reads no undefined global and passes
the contest's source policy at 256 KiB. `NAME=VALUE` writes a research build to `research/` (none was needed).

| group | edits | what |
|---|---|---|
| doc | 1 | the docstring: what the file is and where it is written up |
| book | 2 | `BOOK` (three laws on the pair) and the reading's constants; `SP` passes the book to the climb |
| reader | 4 | `LawReader`, `read_pairs` and `READ_OPS`; the climb's reading state; `read_law` and the reading in `pose`; the pose's log line names the law read |
| room | 11 | the grammar's rules for laws neither the book nor a reading poses |

**The book** (`BOOK`, line 101). The climb poses lt, red and add on the pair, and nothing else of the grammar. Before,
it posed every law of the grammar on every pattern whose construction worked (the generated book: 9 operations on 10
two-operand patterns, 4 compositions on 3 three-operand ones). The grammar stays as the forms a law can take
(`PATTERNS`, `LAW_BOOK`) and the rules that answer them (`law_target`): the reading picks among them.

**The reading** (`LawReader`, line 2818; `SelfClimb.read_law` and `pose`, lines 2988–3008). At the start of a level,
once the comparison, the link and the addition are routines on the pair and the climb has answer rows (the kept rows of
the smallest T, which on H1 is 1: the buffers `ans_rows` the round's search already reads), the climb reads the
target's law, at most `READ_TRIES` = 3 times:
- the law's candidates are left-deep chains of add, mul and mod over x and N, depth 1, then 2 (12, then at most 144);
- each value is computed at the task's full size (16 places) on the rows where it can be, by the learner's own routines
  and the book's own constructions:
  - add: the learner's addition, where the top place of both operands is zero;
  - mul: `repeat_add`, the book's own construction (the learner's addition, by doubling: `times_add`), where both
    factors fit 8 places;
  - mod: the learner's link, repeated until the learner's comparison says the value is below, at most
    `READ_LINKS` = 1,000 links, where the divisor is not zero;
- the first candidate that gives every row where it was computed its answer, digit for digit, on at least
  `READ_MIN` = 4 rows, is the law read.

**What a reading poses** (`read_pairs`, line 2882). Each operation of the law read is a law of the book (`READ_OPS`:
add, mul, and red for mod). For each, the law itself and the law accumulated into a sum (add.f), on every pattern of
the grammar whose count is a digit (d: the operation cut to one place) or another row's number (o: its loop over the
digits). `law_target` answers each where it can, as it answers any law; where it cannot, the law poses nothing. For
mod(mul(x,x),N) that is six laws, test 4's five and one more:

| law | pattern | what | answered by |
|---|---|---|---|
| add.mul | o,x,d | the multiply-add row, o + x·d | repeated addition (the learner's addition) |
| mul | x,d | x·d | the same |
| add.mul | o,x,o | the row's loop, o + x·o | the same |
| mul | x,o | x·d's loop, x·o | the same |
| red | x+dn | Horner's step, red(x + d·N) | the division law red(x + q·N) = x; x + d·N by the learner's addition |
| red | x+on | the step's loop, the reduction, red(x + o·N) | the same |

mul on x+dn and x+on, and red on x,d and x,o, are in the list too; `law_target` has no rule for them, so they pose
nothing. A law the reading poses on the pair would join the book's there, once.

**Room.** The book and the reader add 5,837 bytes, and fast2 had 3,795 to spare. The room edits free 2,845 (and the
shorter docstring 125), from the grammar's rules for laws that neither the book nor a reading ever poses:
- the operations eq, min, max, div and sub (`OPS2` keeps lt, red, add and mul), and their branches in `law_target`;
- mul by zero, which only the pair's zero term gave;
- the base forms d + q·B (Bx+d, Bo+d: their patterns, their terms in `Draw`, the row filter they needed, and red's
  branch for them), whose count is neither d nor o;
- the quotient digits `Draw.nform` returned for div;
- `look_budget`, a setting passed to the climb and never read.

The file: `submission_stage3_legal.py`, md5 `e2fe7a681bc981ce844fc8c826132d47`, 5,000 lines, 261,216 bytes (928
under the limit); against fast2, 118 lines added and 70 removed. Its docstring and one comment still name the reader
`Reader`; the class is `LawReader` (renamed late; left as built, since the official run had started on these bytes).

## 3. Checks (local, CPU)

| check | result |
|---|---|
| [`checks/read_local.py`](checks/read_local.py) MODE=read: fast2's tiny-core climb at the start of its level 3 (the comparison, the link and the addition as routines), the three-law book, 400 answer rows of T = 1 | `mod(mul(x,x),N)` read on 83 of the 400 rows (the multiply is computed on all 400; the reduction on the 83 whose x² has a quotient by N of at most 1,000, so that the links end), 35 laws tried, 2.2 s |
| read_local.py MODE=pose: the pose phase on that state | 96 candidates on 96 roots: the six laws in every binding (24 of each three-operand law, 12 of each two-operand one); each one's answers on its draws, by the book's constructions, equal Python's integers: 192 of 192 |
| combined_sp2's `checks/local_combined.py` (md5 `6f98f631`, as FAST.md ran it), every part ([`checks/logs/local_legal.log`](checks/logs/local_legal.log): 340 s, peak 3.8 GiB) | policy (261,216 bytes) passes; machines, isa, designed, buffers, grad (14 of 15 tensors), replay and fix's details give the fast file's results line for line ([`checks/compare_local.py`](checks/compare_local.py)) |
| local_combined.py book | the book's code differs from `book_gen_fix.py`'s by 87 lines (the room cuts); check_books part 2: every target the book gives equals the law's value, 6,204 probes on 17 concepts (the fast file: 12,382 on 40) |
| local_combined.py climb (toy budgets, two levels, inside training) | level 0: the comparison (3 bindings); level 1: the link, learnable in 8 bindings, by the if block (`LDA.X T.lt.pair.AY C[LDA.X]`) |

The first build posed the book's three laws on the reading's patterns too, and red twice on x + d·N; MODE=pose showed
it, and `pose` now poses the book's laws on the book's patterns only, each law once.

## 4. The tiny core

[`core_c.py`](core_c.py) (combined_fast's, byte for byte: md5 `2e9c6a52`) on one B200 through
[`modal_legal.py`](modal_legal.py) (app and Volume `combined-legal`): the file's own climb with its own budgets,
offline, on 4,096 read rows and 400 answer rows of T = 1 with exact cells, then the answer program's search on rows
of T = 1, 2, 4 and 8, as the file runs it. Record `records/core_legal.json`; fast2's is
`combined_fast/records/core_fast2.json`.

| level | fast2 (the generated book) | this file (three laws and the reading) |
|---|---|---|
| 0 | sub, lt (107 candidates) | lt (24) |
| 1 | min, max, div, red (60) | red (12) |
| 2 | add (24) | add (12) |
| 3 | the row (add.mul on o,x,d), min on x+on, red on x+n, the step (red on x+dn) (435 candidates) | the reading: `mod(mul(x,x),N)` on 83 of 400 rows, 35 laws tried, 14.7 s; the row, the step (96 candidates) |
| 4 | the reduction (red on x+on), the row's loop (add.mul on o,x,o), red on Bx+d, x·d (243) | the reduction, the row's loop, x·d (60) |
| the round, level 4 | 348 s, work 2.53e9 (by calls, 2 calls) | 270 s, work 1.12e9 (by calls, 2 calls) |
| the answer program | `C.S[T.round1 STA.X]`, cost 3, 6.6 s; 224 unseen rows right | the same, 5.8 s; 224 unseen rows right |
| the climb in all | 367 s, work 2.65e9 | 292 s, work 1.23e9 |

The routines are spelled as fast2's: the comparison `CLC P[SBB.N INC]`; the link `LDA.X T.lt.pair.AN C[LDA.X]` (if
the subtraction borrowed, put x back); the addition `CLC P[ADC.N INC]`; the row `LDH.Y CLK P[MUL.X INC]`; the step
`B[T.lt.pair.AN C[T.add.pair.AN]]` (ten times: subtract N, and add it back if that borrowed); the reduction
`LDA.N P[SHL.Y SHL.A T.red.x+dn.AN]` (Horner's loop over the step); the round
`LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.Y T.red.x+on.YN`. The reading's machine work is counted as every instruction's
is (`step`), inside the climb's budget: level 3's pose, the reading included, did 4.8e6 units of the 4.3e9.

## 5. The official run

One H100 through the shared runner (`tools/modal_runner.py --mode official`), the official manifest untouched, the
memory cap 48 GB as fast2's. Record [`records/combined_legal_official.json`](records/combined_legal_official.json).

**It certifies.** `tools/verify_records.py`: "1 of 1 records check out". The embedded source's md5 is the file's
(`e2fe7a68`); the manifest is the official file; the GPU an H100 80GB HBM3 (driver 610.57.04); return code 0; the
RESULT_JSON line equals the result. T = 64 on both ladders, every rung 768 of 768; test 9,999 of 9,999, unseen T 10,002
of 10,002, unseen moduli 10,002 of 10,002.

`integration/combined_fast/checks/summarize_seeds.py` (the clock at a step: the evaluator's progress line before it
plus the climb's own unit seconds, so "about"), against fast2's official record:

| | fast2 | this file |
|---|---|---|
| the climb starts | step 563, about 57 s | step 563, about 42 s |
| the reading | — | step 580, level 3: `mod(mul(x,x),N)` on 23 of the 79 answer rows kept by then, 35 laws tried, 8.8 s |
| the round | level 4 | level 4, step 613, by calls (2 calls) |
| the answer program | step 630, about 492 s | step 614, about 272 s |
| frozen | step 829, about 538 s | step 813, about 306 s |
| the climb's work, the answer program's search in it | 3.254e9 of 4.3e9 | 1.278e9 of 4.3e9 |
| training (to the 20,000-step cap) | 759.3 s | 465.3 s |
| evaluation | 377.3 s | 295.3 s |
| peak host memory | 13.8 GB | 12.6 GB |
| T.round1 | `LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.X T.red.x+xn.XN` | `LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.Y T.red.x+on.YN` |
| the answer programs | `C.S[T.round1 STA.X]`, `C.S[T.round2 STA.X]`, `C.S[T.round3 STA.X]` | `C.S[T.round1 STA.X]`, `C.S[T.round3 STA.X]` |

The climb inside training takes the tiny core's levels: the comparison (level 0, AX, AY, AN); the link (level 1, XY, XN,
YN, NY); the addition (level 2, AX, AY, AN); the reading, then the row (4 bindings) and the step (level 3); the
reduction and the row's loop (level 4), and the round. The reading came 17 steps into the climb, with 79 answer rows
kept (the climb is handed answer rows once 64 are: `ANS_MIN`); 23 of them could be computed, and the law was exact on
all 23. fast2's reduction in the round, red on x + x·N, is one the reading does not pose (its count is x itself, neither
a digit nor another row's number); red on x + o·N does the same work.

## 6. Seeds 1–7

The file byte for byte (`e2fe7a68`), once for each of seeds 1 to 7: the official manifest with only `runtime.seeds`
changed, one H100 each, all seven at once, through the shared runner as fast2's were run (`--mode seeds --hardware
h100`, memory cap 48 GB). Records `records/combined_legal_h100_seed<N>.json`.

**Every seed certifies.** `tools/verify_records.py` on the official record and the seven: "8 of 8 records check out".
Each embeds the file's md5; its manifest is the official one but for the seed; return code 0; the RESULT_JSON line
equals the result; T = 64 on both ladders with every rung 768 of 768; test 9,999 of 9,999, unseen T 10,002 of 10,002,
unseen moduli 10,002 of 10,002.

`summarize_seeds.py` and the climbs' lines ([`checks/logs/summarize_seeds.log`](checks/logs/summarize_seeds.log)):

| seed | climb starts | the reading, level 3: rows exact of rows kept | the round, level 4 | the answer program | frozen | climb work | training | evaluation |
|---|---|---|---|---|---|---|---|---|
| 74 (official) | step 563, about 42 s | step 580: 23 of 79, 8.8 s | step 613 | step 614, about 272 s | step 813, about 306 s | 1.278e9 | 465 s | 295 s |
| 1 | 584, 49 s | 601: 15 of 75, 9.7 s | 634 | 635, 314 s | 834, 361 s | 1.308e9 | 563 s | 322 s |
| 2 | 584, 48 s | 602: 17 of 95, 10.4 s | 633 | 634, 275 s | 833, 409 s | 0.993e9 | 616 s | 343 s |
| 3 | 612, 69 s | 629: 17 of 74, 13.6 s | 662 | 663, 349 s | 862, 481 s | 1.956e9 | 721 s | 403 s |
| 4 | 653, 64 s | 670: 11 of 85, 12.0 s | 704 | 705, 335 s | 904, 389 s | 1.220e9 | 630 s | 358 s |
| 5 | 609, 61 s | 626: 16 of 88, 9.9 s | 657 | 658, 319 s | 857, 451 s | 1.477e9 | 677 s | 348 s |
| 6 | 623, 62 s | 640: 17 of 86, 11.9 s | 673 | 674, 312 s | 873, 354 s | 1.283e9 | 587 s | 368 s |
| 7 | 565, 53 s | 582: 17 of 69, 10.8 s | 618 | 619, 313 s | 818, 358 s | 1.309e9 | 567 s | 343 s |

**The same climb on every seed.** Each reads `mod(mul(x,x),N)` at level 3 (35 laws tried) and practises the same
concepts at the same levels: the comparison; the link; the addition; the row and the step; then the reduction, the
row's loop and x·d, with x·d's loop too on seeds 1, 2, 4 and 5. Each finds its round at level 4, by calls (2 calls),
and its answer program `C.S[T.round1 STA.X]` the next step. Seeds 1, 2, 4 and 5 multiply in the round by x·d's loop
(`LDA.X T.mul.x,o.AX STA.X T.red.x+on.XN`), the others by the row's (`LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.X
T.red.x+on.XN`).

**Against fast2's eight** (the same script on its records): the answer program by 272–349 s of training against
492–555 s; frozen by 306–481 s against 538–641 s; training 465–721 s against 759–883 s; evaluation 295–403 s against
334–407 s; the climb's work 0.99e9–1.96e9 against 2.77e9–3.49e9. fast2's work budgets are kept as they were: the
largest climb here (seed 3) uses 45% of `CLIMB_WORK`, fast2's largest 81%.

## 7. What this file designs that fast2 did not, and the reverse

**Designed here, not in fast2:**
- **the book's three laws** (`BOOK`): the comparison, the link and the addition on the pair, the floor test 4 found
  (LEGALLOOPS.md). Without the addition the reading never runs (test 4's k1_less_add); without the reading the three
  laws give nothing more (its k0);
- **the reading's grammar**: leaves x and N (not T: the answer rows are rows of one T), the operations add, mul and mod
  with their constructions, depth 2, the order of the tries (depth, then the inner chain, then the operation, the
  leaf, the side), `READ_MIN` (4 rows), `READ_LINKS` (1,000), `READ_TRIES` (3), and where a value counts as computable
  (it fits the places; the links end). It was chosen knowing the task family: H1's law is a chain of depth 2 of these
  operations over these leaves. A law outside the grammar would not be read, and the climb would have the three laws
  only;
- **the posing rule** (`read_pairs`): an operation's laws are f and add.f, posed on the patterns whose count is a digit
  (the cut) or another row's number (the loop); mod is the book's red (`READ_OPS`).

**Designed in fast2, not here:**
- **the generated book posed whole**: every law of the grammar on every pattern whose construction worked, from the
  start (13 operations; fast2's tiny core practised 15 concepts, 7 of them this file never poses: sub, min, max and div
  on the pair, min on x + o·N, red on x + N and on the base form);
- **five operations and the base forms**, with their rules in `law_target` (eq; min and max by the order; div's
  quotient; sub's self, identity and (x + q·N) − N; d + q·B and the chain's range on it): cut here for room, since
  nothing poses them.

**The same in both** (REVIEW.md §5, rule 7): the instruction set; the grammar's forms (the patterns, the counts 1, d, o
and x, r + q·N) and `law_target`'s rules for lt, red, add and mul and their compositions; the choosing rule (lpca); the
search's sizes and rules; the count loop and T's 2-digit field; stage 1's reader of the prompt; the cells' shapes; the
schedule. What the climb practises is now decided by three laws and the target, read by the learner's routines; before,
by the grammar alone.

## 8. Where it still touches Python

REVIEW.md §5 (rules 9 and 14) for fast2 holds for this file unchanged: constant tables built on the CPU and copied to
the GPU (the two hash tables, the opcode tables, the true digit tables each `Cells` starts with and the learned ones
replace), Python's bookkeeping (the schedule, the climb's programs as tuples, its library and BPE over program text),
small results read back from the GPU to steer the search, and each program traced from its text into opcode columns.
No practice target is computed with Python's arithmetic: every one is a fact, an identity or the learner's own routine.

New here, all in `LawReader`, `read_pairs`, `read_law` and `pose`:
- **Its values** are the learner's: every number it compares is an output of the learner's addition, link or
  comparison, run on the GPU by the machine's learned cells (`Ops.run`), or of `repeat_add` over the learner's addition.
  No value is made by Python's arithmetic.
- **Its decisions** are digit equality with the answer rows (`(v == Y).all(1)`, on the GPU) and digit facts (an
  operand's top places are zero, so the result fits; a divisor's digits are not all zero). The rows a value is not
  computed on are zeroed by multiplying its digits by the 0/1 mask: a selection.
- **Counts**: the multiply reads its count off the multiplier's digits (`count_of`) and the doubling reads that count's
  bits (`times_add`: `t % 2`, `t // 2`), on the GPU, as fast2's practice constructions already do (`mul.distribute`,
  the terms x + q·N).
- **Host reads**, small results that steer it: whether any row still needs a link (`bool(left.any())`, at most 1,001
  times a reduction), how many rows a value was computed on and whether it passes (`int(m.sum())`, a `bool`), the size
  of a row selection (`i.numel()`), and whether two candidates' values are the same (`torch.equal`, a bit). No digit
  is read back.
- **Python's bookkeeping**: the law's text, the count of laws tried, the (law, pattern) names it poses, its record.
- **The rows** it reads are the answer rows fast2 already keeps for the round's search (`ans_rows`, kept by the loss
  from the labels: REVIEW.md §5, rule 4); nothing new is kept.
- **The work budget**: its routine calls run instruction by instruction (`step`), whose work is counted as every
  instruction's is, so the reading spends the climb's budget like any search: 4.4e6 units at level 3's pose in the
  official run, of the 4.3e9.
- **No clock**: `_clock` only stamps the reading's record (seconds), as it stamps every log line.

## 9. Cost

- **Billed** (`cost.py`, at 06:26 UTC): $1.14 through the 05:00 UTC hour: the tiny core on a B200 $0.58, the official
  run's first 11 minutes $0.56. The 06:00 UTC hour (the official run's last minutes and the seven seeds) was not in
  the report yet; `cost.py` reads it once it is.
- **By the runs' own times:** the eight H100 jobs' wall time, 7,653 s, at $3.95 an hour is $8.40; with their
  start-up, the runners' drivers and the tiny core, about $10 in all.
- **The budget:** soft $100, cap $150. The spend watcher's stop-all was $115 (the cap less an hour of eight GPUs); it
  never acted. `spend.py 4750` said LAUNCH OK before each of the three launches; the month's bill was $3,870.26 at
  06:26 UTC, under the stop of $4,750.
- **Apps:** all stopped: the tiny core's and its fetch (`combined-legal`), the official run's and the seeds'
  (`cpu-first-runner`).

## 10. Files and how to rerun

Everything is in `cpu_first/integration/combined_legal/`; nothing outside it was written.

| file | what |
|---|---|
| `submission_stage3_legal.py` | the file, md5 `e2fe7a681bc981ce844fc8c826132d47`, 261,216 bytes (928 under the limit), written by `make_legal.py` |
| `make_legal.py` | the builder (§2): fast2 from git at `f237a61`, md5 asserted, then the doc, book, reader and room edits |
| `core_c.py` | combined_fast's tiny core, copied byte for byte (md5 `2e9c6a52`) |
| `modal_legal.py` | the tiny core's launcher on Modal (app and Volume `combined-legal`) |
| `cost.py` | §9: the billing report's hours for this study's apps |
| `checks/read_local.py` | the reading and its pose on fast2's tiny-core state (§3) |
| `checks/compare_local.py` | combined_sp2's local checks on this file against the fast file's, part by part (§3) |
| `checks/logs/` | `local_legal.log` (the local checks), `compare_local.log`, the spend watchers, the fetches |
| `records/` | the tiny core (`core_legal.json`, `.log`, `_app.log`), the official record (`combined_legal_official.json`), seeds 1–7's (`combined_legal_h100_seed<N>.json`) |

From `cpu_first/`, with `PY=~/github/sotaku/venv/bin/python`:

```bash
$PY integration/combined_legal/make_legal.py
PYTHONDONTWRITEBYTECODE=1 MODE=read $PY integration/combined_legal/checks/read_local.py
PYTHONDONTWRITEBYTECODE=1 MODE=pose $PY integration/combined_legal/checks/read_local.py
COMBINED_FILE=integration/combined_legal/submission_stage3_legal.py PYTHONPATH=. $PY integration/combined_sp2/checks/local_combined.py policy grad machines isa designed buffers replay climb climb_tiny fix book
python3 integration/combined_legal/checks/compare_local.py
$PY integration/ablation_s3/spend.py 4750
$PY -m modal run --detach integration/combined_legal/modal_legal.py --envs '[{"TAG": "core_legal"}]'
$PY -m modal run integration/combined_legal/modal_legal.py --mode fetch
$PY -m modal run --detach tools/modal_runner.py --mode official --file integration/combined_legal/submission_stage3_legal.py --memory-gb 48 --name combined_legal_official --out integration/combined_legal/records --run combined-legal-official
$PY tools/modal_runner.py --mode fetch --run combined-legal-official --out integration/combined_legal/records --wait
python3 tools/verify_records.py integration/combined_legal/records/combined_legal_official.json=integration/combined_legal/submission_stage3_legal.py
$PY -m modal run --detach tools/modal_runner.py --mode seeds --file integration/combined_legal/submission_stage3_legal.py --seeds 1,2,3,4,5,6,7 --hardware h100 --memory-gb 48 --name combined_legal --out integration/combined_legal/records --run combined-legal-h100-seeds
$PY tools/modal_runner.py --mode fetch --run combined-legal-h100-seeds --out integration/combined_legal/records --wait
python3 tools/verify_records.py integration/combined_legal/records/combined_legal_h100_seed{1,2,3,4,5,6,7}.json=integration/combined_legal/submission_stage3_legal.py
python3 integration/combined_fast/checks/summarize_seeds.py integration/combined_legal/records/combined_legal_official.json integration/combined_legal/records/combined_legal_h100_seed{1,2,3,4,5,6,7}.json
$PY integration/combined_legal/cost.py
```
