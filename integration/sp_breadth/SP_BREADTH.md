# Breadth at full size: does the self-practising stage 3 learn the family's other formulas?

*Written 2026-10-24. Code, plans and records in `cpu_first/integration/sp_breadth/`. Nothing submitted, no shared or
certified file touched (the builder asserts the certified files' md5s and writes only here).*

## In plain words

<!-- PLAIN -->
**The question.** sp2, the self-practising stage 3, learns H1 (square x, keep the remainder after dividing by N,
repeat T times) at full size in one training run. It poses its own practice problems from a book of 11 arithmetic
laws, keeps the solutions as routines, and then finds the round "copy x, multiply, reduce" by searching short
combinations of its routines against the answers. Bar 3's tiny learner invents several formulas unchanged, but only on
a counting machine with N up to 26. Does sp2, changed in nothing, learn the family's other formulas at H1's size? And
where it does not, do general changes (none naming a formula) fix that without costing H1?

**Unchanged: 2 of 11 tasks.** H1 and xfirst (H1's data with the prompt's fields reordered) certify. The seven other
formulas on H1's prompts all fail the same way: the climb reads only N and x, so it is the same run as H1's and has
every piece of H1's round by level 4; but the round's search composes at most two routine calls and has no way to make
a number, so it finds nothing. 10 y^2 needs a third call; y^2 + 1, y^2 + 3 and y^2 - 1 need a constant; 10 y^2 + 1
needs both; y^3 needs four calls and a reduction that leaves x in a register; Montgomery's y^2 10^-8 needs a reduction
the law book never poses. M4 (H1's formula, every row T = 8) fails because answers of 8 rounds are searched only over
short flat programs, one at a time; xy (four numbers in the prompt) never gets past the reader.

**Three general changes, all in the round's search.** None names a formula; after each, every task was rerun (xy,
whose reader stops before any search, only after the second and third).
1. *A wider search (g1).* Three calls instead of two; and for answers of more than one round, every program of up to
   two calls runs all its rounds, in batches. 10 y^2 becomes `[copy x, multiply] SHL.A [reduce]`.
2. *Constants taken from the data (g2).* When no round is found, a program whose result differs from every answer by
   the same number c (mod N) is a near miss, and c is read off the answer rows with the learned subtraction table;
   for y^2 + 3 the H1 round is the near miss and c = 3. A new instruction loads c, and a second short search finds
   the rest of the way from the near miss. A round must also end with a true reduction mod N (given its result plus
   N, it gives the result again), because the shortest rest of the way, "add 3", is right on every training row but
   wrong on about 3 inputs in N.
3. *The answers of every T (g3).* g2's round for y^2 - 1 subtracted 1 without wrapping round, which is wrong exactly
   when a round's input is 0. The data's x are never 0, but a later round's input can be (whenever y^2 = 1 mod N), and
   training's own rows with T = 4 and T = 8 show it, in 24 and 61 of their 33,333 each. So a round must also reproduce
   the answers of the other one-digit T's the batches show, and the rest of the way may take a third call: subtract
   1, add N, reduce.

**Results** (each on one H100, the competition's evaluator untouched, seed 74; "learned" = T = 64 on both ladders,
every rung 768/768):
- *Learned with the final file (g3), 8 of 11:* H1 (official manifest, and 8 of 8 seeds: §5), xfirst, 10 y^2, M4,
  y^2 + 1, y^2 + 3, 10 y^2 + 1 and y^2 - 1, each in one training run, its round found at the climb's level 4 (M4:
  level 5), by 904-1,343 s into training and frozen by 1,410 s at the latest (M4: the round by about 1,800-2,300 s;
  it never freezes, as T is always 8).
- *What each change bought:* the wider search, 10 y^2 and M4; constants, y^2 + 1, y^2 + 3 and 10 y^2 + 1 (and y^2 - 1
  right on all but the rows that pass through 0); the answers of every T, y^2 - 1. H1 and xfirst certified after
  every change.
- *Not learned, 3 of 11:* y^3 (a round needs four calls, past the search's three, and a reduction that leaves x in
  a register, which the cube runs' climb did not practise), Montgomery's y^2 10^-8 (no law poses its reduction, and
  its step is past the probes' reach), and x y^T with a fourth number (the reader takes three).

**How general is it?** More than before, less than bar 3 asks. The climb, the law book and the reader did not change;
the new parts are all in how the round is searched and accepted, and each is stated without a formula. But each was
written after watching a formula fail, with the family in view: the gap is taken mod N because the answers are
residues, LDK is a 33rd instruction, and the reduction rule is a type rule for modular answers. What stays out of reach
is outside the family's "something of y^2 plus a constant, then reduce" shape: y^3 needs four calls (past the
search's three) and a reduction that leaves x in a register (practised on some seeds, not in the cube runs),
Montgomery's needs a reduction the law book never poses, and a fourth input needs another reader.
<!-- /PLAIN -->

## 1. The question and the formulas

sp2 ([`submission_stage3_sp2.py`](../self_practice/submission_stage3_sp2.py), md5 `afc70fe9`) learns H1, x^(2^T) mod N,
in one training run: it practises problems it poses from an 11-law book, builds routines, and then finds the round
("copy x, multiply, reduce") by searching short compositions of its routines against the answers. Bar 3's tiny
learner invents several formulas unchanged, but only on a counting machine at x < N <= 26 ([BREADTH.md](../../bar3/BREADTH.md)).
Here: the same question at H1's size, on the family's other formulas. Every task uses H1's prompts (the same rows, the
same moduli) with another answer, except M4 (the competition's own Medium dataset, every T = 8) and xfirst and xy
(other prompt layouts):

| task | answer | what it was expected to need (SELF_PRACTICE.md §4.5, OVERVIEW §7 questions 4 and 10) |
|---|---|---|
| H1 | x^(2^T) mod N | (certified) |
| xfirst | H1's, prompt fields reordered (X x N n T t) | nothing |
| ten | y <- 10 y^2 mod N, T times | a third routine call |
| rho | y <- y^2 + 1 mod N | a constant |
| plus3 | y <- y^2 + 3 mod N | a constant |
| minus1 | y <- y^2 - 1 mod N | a constant, subtracted ("no round option computes subtract 1") |
| tenrho | y <- 10 y^2 + 1 mod N | a constant and a third call |
| cube | x^(3^T) mod N | a fourth call (multiply, reduce, multiply, reduce) |
| mont | y <- y^2 10^-8 mod N | another reduction (Montgomery's) |
| M4 | x^(2^T) mod N, every row T = 8, N written forwards | answers at T = 8 |
| xy | x y^T mod N, a fourth field Y y | a reader for four numbers |

Each run: the competition's evaluator, untouched, on one H100, H1's manifest with only `data_root` (and the name)
changed for the litmus tasks (M4: its own manifest with H1's 3,600 s), seed 74, through the shared runner.
"Learned" means certified: T = 64 on both ladders, every rung 768/768.

## 2. sp2 unchanged

sp2 byte for byte (md5 `afc70fe9`), plan `plans/u0_unchanged.json`, records `records/u0_<task>_h100.json`:

<!-- U0 -->
| task | learned? | max T seen / unseen | test | training s | why |
|---|---|---|---|---|---|
| H1 | yes (SELF_PRACTICE.md §4.2) | 64 / 64 | 9,999 / 9,999 | 1,277 (round by 1,029 s) | |
| xfirst | **yes** | 64 / 64 | 9,999 / 9,999 | 1,067 (round at step 610 by 855 s, frozen at step 809) | the same round as H1; the reader takes the reordered fields as they come |
| ten | no | - / - | 6 / 9,999 | 3,600 | the round is 3 calls; the search by calls stops at 2 |
| rho | no | - / - | 8 / 9,999 | 3,600 | needs the constant 1; nothing in the search makes a constant |
| plus3 | no | - / - | 6 / 9,999 | 3,600 | needs the constant 3 |
| minus1 | no | - / - | 7 / 9,999 | 3,600 | needs the constant 1, subtracted |
| tenrho | no | - / - | 12 / 9,999 | 3,600 | needs a constant and a third call |
| cube | no | - / - | 11 / 9,999 | 3,600 | needs 4 calls, and a reduction that leaves x in a register |
| mont | no | - / - | 10 / 9,999 | 3,600 | needs Montgomery's reduction, which nothing in the law book poses |
| M4 | no | - / - | 8 / 9,000 | 3,600 | every row has T = 8, and for answers of T > 1 sp2 searches only flat programs (cost 5; the round is 2 calls, cost 6), running each listed one 8 times, one at a time: 103-654 s a level, so the climb itself only got to level 7 |
| xy | no | - / - | 2 / 9,999 | 741 (the file's 20,000 steps) | the reader's first stage never completes on a prompt of four numbers (phase A, reader stage 1, all 20,000 steps); nothing after the reader runs |
<!-- /U0 -->

What happened in the seven failing runs on H1's prompts:
- **The climb is the same run on every formula.** It reads only the rows' N and x, which are H1's, so up to its last
  level each run poses, practises and keeps the same routines, in the same order, as the others: the libraries left
  at the end are identical except for depth (four runs ended after 10 levels, three after 12; the hosts differed in
  speed). By level 4, about 825-1,050 s of climbing in, it has every piece of H1's round: the multiply
  `CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]` and the long division `STA.Y P[BRG.X T.red.x+dn.AY]`.
- **Only the round's search sees the answers, and it finds nothing.** After every level the round is searched as sp2
  searches it (the flat search to cost 5, then by calls to 2 calls): at level 7, 706 call items and 493,651 states of
  two calls, none of them the answer.
- **The climb then runs to the deadline.** Levels 5-11 practise the remaining bindings of a + b c, a + b + c, eq and
  max (58 routines in all, against H1's 47 at its round); the last levels are cut by the climb's deadline, 120 s
  before the budget ends. No new reduction is practised: `red` on x + o N exists only in the binding XA (dividend in
  X, modulus in A, which it copies into Y).
- The answers on the test split are what a wrong machine gets by chance: 6-12 of 9,999.

## 3. The general changes

All of them are in the round's search (spbcore.py's `search_round` and what it calls); the climb, the law book, the
reader, the cells, the countdown and the tournament are sp2's, byte for byte in what they compute. None names a
formula. Each is a setting at the top of the file, so one source builds every file of the study.

**g1: a wider round search** (`submission_spb_g1.py`).
- *A third call.* sp2's search by calls composes up to 2 routine calls (each with its inputs routed from A, X, Y, N or
  zero), so it finds H1's `LDA.X STA.Y T.mul STA.X LDA.N T.red` and nothing longer. `SP_MACRO_COST = 3` lets it compose
  3. Example: 10 y^2 mod N is `[copy x, multiply] SHL.A [reduce]`, one whole-register instruction between the two calls.
- *Answers of more than one round.* When the smallest T the batches show is above 1 (M4: every row has T = 8), no
  state of one round shows the answer. sp2 then listed at most 4,096 programs a level of its flat search (cost 5 at
  most, no routine calls composed) and ran each T times, one at a time; in the earlier sp run on M4 that took 124-432 s
  a level and never reached the round, which is a 2-call program. Now every program of up to `SP_MACRO_T = 2` calls is
  held, and each one whose round leaves A below N on every probe runs T - 1 more rounds, all of them at once, a call
  position at a time; the flat search's listed programs are run the same batched way (`t_rounds`). The filter "below N
  after every round" is the data's own fact: every round's result is the next round's input, and every input is below
  N.

**g2: constants taken from the data** (`submission_spb_g2.py`: g1 with `SP_CONST = True`, `SP_ROUND_ENDS = "red"`).
- *Near misses.* While the search by calls runs, a state whose A is below N on every probe and differs from every
  probe's answer by one and the same number c mod N (either way: answer = A + c or A - c) is a near miss. The
  difference is computed with the learned subtraction and addition tables, place by place (the machine's own SBB and
  ADC), never with Python's integers. Example, y^2 + 3: the H1 round leaves x^2 mod N; on every answer row the
  answer is 3 more, mod N, so c = 3. The constant is read off the answers; nothing in the file says 3.
- *An instruction that loads the constant.* LDK puts c's digits in A. It is not one of the climb's 32 instructions
  (the climb never sees it); only the search for the rest of the way can use it, as a routing source like zero. It is
  implemented on all four machines (the search's step and lockstep machines, the round machine's CUDA graph, the one-hot
  prediction machine), and a local test checks they agree (`test_ldk.py`).
- *The rest of the way.* From the near miss's end state, the search by calls runs again, up to `SP_REST_COST = 2`
  calls, with c as a source of every routed input (and A kept in S when a routing would lose it); the round is the
  near miss followed by the rest. Example, y^2 + 1: `[the H1 round] STA.Y LDK T.add.pair.AY T.red.pair.AN`: add c,
  then subtract N unless that borrows.
- *A round ends with a reduction mod N.* Needed once constants are in: the shortest rest of the way for y^2 + 3 was
  "add 3" alone, right on every answer row the run keeps (512), but wrong whenever x^2 mod N >= N - 3 (about 3 rows in
  N), which the deep rungs would hit. Every answer is below N (a mined fact), and in the law book only a reduction's
  result is; so a round must end with a call to a red routine that really reduces mod N: on every answer row its
  result r is below N, and given r + N as its dividend it gives r again (the book's own red.invariant, red(a + N) = a
  for a < N, at a = r). Getting this rule right took four versions; local tests found each earlier one's flaw: a
  syntactic version was fooled by a reduction against a register holding 0 (it does nothing); adding N to A before
  the call was fooled by a call whose dividend is routed from X; and adding N to the dividend itself (rather than to
  the result) left the law's range, so it rejected the right last step of y^2 - 1 (`test_deep.py`). g2 ran with the
  third version; g3 has the fourth. The rule also prunes the search's last level to the calls that can end a round.

**g3: the answers of every T** (`submission_spb_g3.py`: g2 with `ANS_DEEP = 2048`, `SP_REST_COST = 3` and the last
version of the reduction rule).
- *Why.* g2's round for y^2 - 1 was "x^2 mod N, subtract 1, reduce", right on every T = 1 row but wrong when a round's
  input is 0: then x^2 mod N is 0 and "subtract 1" wraps. The data's x are never 0, but a later round's input can be,
  whenever y^2 = 1 mod N (§4.4).
- *The change.* The file keeps, besides the rows of the smallest T, up to 2,048 answer rows of every other T the
  batches show whose reading is settled (one digit: on H1's data 2, 4 and 8; a two-digit T may still be read
  backwards when the round is searched). A round must also reproduce those, run T times on the round machine.
  Training's own T = 4 and T = 8 rows pass through 0 in 24 and 61 of 33,333 each, so 2,048 of each hold about 5 such
  rows. The rest of the way may take 3 calls (the right last step of y^2 - 1 needs a third: subtract 1, add N,
  reduce).

## 4. What each change bought

### 4.1 Checked offline first

Before any training run, g2's round search alone (`offline.py`, one B200, log `offline/off_g2.log`):
sp2's library at its round in the certified H1 run (47 routines, the climb's level 4, which is where every
formula's climb stands at that time: §2), each formula's answers on 512 of H1's T = 1 rows, the search as g2 runs it.

| formula | found | the round (one of the accepted spellings) | seconds |
|---|---|---|---|
| H1 | 2 calls | `LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA` | 12 |
| ten | 3 calls | `LDA.X STA.Y T.mul.x,o.XY SHL.A STA.X LDA.N T.red.x+on.XA` | 32 |
| rho | near miss + 1, then the rest | `[the H1 round] STA.S LDK STA.X STA.Y LDA.S T.acc.mul.o,x,d.AXY STA.Y T.red.Bx+d.YN` (add 1 x 1 with the multiply-accumulate row, then one long-division step against N) | 203 |
| plus3 | near miss + 3, then the rest | `[the H1 round] STA.Y LDK T.add.pair.AY STA.S LDA.N STA.X LDA.S T.red.pair.AX` (3 into Y and add it, then N into X and subtract it unless that borrows) | 137 |
| minus1 | near miss - 1, then the rest | `[the H1 round] STA.S LDK STA.X LDA.S T.lt.pair.AX STA.S LDA.Y STA.X LDA.S STA.Y T.red.Bx+d.YX` (1 into X and subtract it, then N into X and one long-division step) | 183 |
| tenrho | near miss + 1, then the rest | `[the ten round] STA.S LDK STA.X STA.Y LDA.S T.acc.mul.o,x,d.AXY T.red.pair.AN` | 199 |
| cube | none (3 calls exhausted) | | 28 |
| mont | none (3 calls exhausted) | | 28 |

The near misses are what the formulas' structure suggests: the H1 round for the three "y^2 + c" forms, the ten round
for 10 y^2 + 1 (found at the third call). The 3-call level held 3.6M states after the last-level pruning (14.3M
without it, in SELF_PRACTICE.md's own offline check). Most of the constant formulas' time (about 170 s of their
roughly 200 s) went to checking candidate programs one at a time on all 512 answer rows; g2's build checks them in
batches on the lockstep machine instead (the same verdicts, checked locally by `test_near.py`).

### 4.2 g1 in the evaluator: a wider round search

`submission_spb_g1.py` (md5 `c8422b2f`), plan `plans/g1_sweep.json`, records `records/g1_<task>_h100.json`. xy was
not rerun here (its reader stops before any round search runs; it is rerun with the final file).

<!-- G1 -->
| task | learned? | max T seen / unseen | test | training s | the round (found at step, by s) |
|---|---|---|---|---|---|
| H1 (the official manifest, untouched) | **yes** | 64 / 64 | 9,999 / 9,999 | 1,275 (frozen at step 887, by 1,069 s) | `LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA` and its X/Y twin, 2 calls (688, 1,024): sp2's own round at sp2's own step |
| ten | **yes** | 64 / 64 | 9,999 / 9,999 | 1,263 (887, 1,070) | `LDA.X STA.Y T.mul.x,o.XY SHL.A STA.X LDA.N T.red.x+on.XA`, 3 calls, and three spellings (`BRG.X` for `SHL.A`, the X/Y twin) (688, 1,018) |
| M4 (its own data, H1's 3,600 s) | **yes** | 64 / 64 | 9,000 / 9,000 | 3,600 (never frozen: T is always 8, one digit, so its reading direction never settles; it started the right way) | `LDA.X STA.Y T.mul.x,o.XY STA.X T.red.x+on.XN` and its X/Y twin, 2 calls, found by running every 2-call program 8 times (632, by 2,266 s); the climb on M4's own rows practised the reduction with the modulus in N, `CLA P[BRG.X T.add.pair.AN T.red.x+dn.AN]`, at level 5 |
| rho | no | - / - | 8 / 9,999 | 3,600 | no round: 3 calls exhausted at every level (no constant) |
| plus3 | no | - / - | 6 / 9,999 | 3,600 | likewise |
| minus1 | no | - / - | 7 / 9,999 | 3,600 | likewise (at level 7: 56M states of three calls, 75 s) |
| tenrho | no | - / - | 12 / 9,999 | 3,600 | likewise |
| cube | no | - / - | 11 / 9,999 | 3,600 | no round: 3 calls exhausted |
| mont | no | - / - | 10 / 9,999 | 3,600 | likewise |
| xfirst | **yes** | 64 / 64 | 9,999 / 9,999 | 1,314 (frozen at step 809) | H1's round, 2 calls |
<!-- /G1 -->

The 3-call level is cheap in the run: at level 4, 16.9M states of three calls in 37 s on the H100 (the whole round
search), and 8 s at level 3, where nothing can be found yet. H1's climb, round and freeze come at the same steps as in
sp2's official run (688 and 887). In the ten run, `BRG.X` (bring X's top digit into A) does what `SHL.A` does: the
multiply has shifted x up by 8 places, so X's top place holds x's 8th digit, which is 0 for every x below 10^7; ten's
moduli stay below 10^7 by construction (its unseen moduli stop at 22 bits, about 4 million).

### 4.3 g2 in the evaluator: constants taken from the data

`submission_spb_g2.py` (md5 `371c3d67`: g1's source with `SP_CONST = True`, `SP_ROUND_ENDS = "red"`, and the
candidates checked in batches), plans `plans/g2a_sweep.json` and `plans/g2b_sweep.json`, records
`records/g2_<task>_h100.json`.

<!-- G2 -->
| task | learned? | max T seen / unseen | test | training s | the round (found at step, by s) |
|---|---|---|---|---|---|
| H1 (the official manifest, untouched) | **yes** | 64 / 64 | 9,999 / 9,999 | 1,222 (frozen at step 887, by 1,028 s) | sp2's round, 2 calls (688, 983) |
| rho: y^2 + 1 | **yes** | 64 / 64 | 9,999 / 9,999 | 1,417 (887, 1,234) | near miss: the H1 round, constant 1; then `STA.X LDK T.add.pair.AX T.red.pair.AY` and three other spellings (688, 1,184) |
| plus3: y^2 + 3 | **yes** | 64 / 64 | 9,999 / 9,999 | 1,354 (887, 1,131) | near miss: the H1 round, constant 3; the same four programs (688, 1,084) |
| tenrho: 10 y^2 + 1 | **yes** | 64 / 64 | 9,999 / 9,999 | 1,312 (887, 1,121) | near miss: the ten round (3 calls), constant 1; then `STA.Y LDK T.add.pair.AY T.red.pair.AN` and three other spellings (688, 1,074) |
| ten | **yes** | 64 / 64 | 9,999 / 9,999 | 1,206 (887, 1,026) | as in g1 (688, 976) |
| minus1: y^2 - 1 | **no**: right except where a round's input is 0 | 4 / 2 | 9,995 / 9,999 (ood_t 9,978 / 10,002) | 1,371 (887, 1,190) | near miss: the H1 round, constant 1, subtracted; then `STA.S LDK STA.X LDA.S T.lt.pair.AX T.red.pair.AN`: subtract 1, then subtract N unless that borrows (688, 1,143) |
| M4 (its own data, H1's 3,600 s) | **yes** | 64 / 64 | 9,000 / 9,000 | 3,600 (never frozen, as in g1) | as in g1, 2 calls by running every 2-call program 8 times (632, 1,776) |
| cube | no | - / - | 11 / 9,999 | 3,600 | no round, no near miss: 3 calls exhausted at every level |
| mont | no | - / - | 10 / 9,999 | 3,600 | likewise |
| xy | no | - / - | 2 / 9,999 | 601 (the file's 20,000 steps) | the reader, as unchanged |
| xfirst | **yes** | 64 / 64 | 9,999 / 9,999 | 990 (frozen at step 809) | H1's round |
<!-- /G2 -->

### 4.4 y^2 - 1: right except at 0

g2's round for y^2 - 1 is "x^2 mod N, subtract 1, then subtract N unless that borrows". On every input whose square is
not 0 mod N it is exact. When a round's input is 0, x^2 mod N is 0, "subtract 1" wraps round to 9999999999999999,
and the last step leaves that (it is above N, no borrow, so N is subtracted once): the answer should be N - 1. The
data's x are never 0 (they are units), so the T = 1 rows the round is searched and accepted on can never show it, nor
can the tournament's rows (also the smallest T). But a later round's input is 0 whenever the round before gave
y^2 = 1 mod N: computing the true answers of the evaluator's rows with Python's integers, exactly 4 of the test
split's 9,999 rows pass through 0 (the run got exactly 4 wrong), and 26 of ood_t's 10,002 (it got 24 wrong). On the
depth ladders the first such row is on rung T = 8 for the seen moduli (1 of 768) and on rung T = 4 for the unseen
ones (1 of 768), which is exactly where the certified T stops (4 and 2). So the round's own data never refutes the
short round; deeper data does: among training's rows, 24 of the 33,333 with T = 4, 61 of those with T = 8 and 74 of
those with T = 16 pass through 0, and none of those with T = 1 or 2. g3 (§3) reads those.

### 4.5 g3 in the evaluator: the answers of every T

`submission_spb_g3.py` (md5 `30efec64`), plans `plans/g3a_sweep.json` and `plans/g3b_sweep.json`, records
`records/g3_<task>_h100.json`.

<!-- G3 -->
| task | learned? | max T seen / unseen | test | training s | the round (found at step) |
|---|---|---|---|---|---|
| H1 (the official manifest, untouched) | **yes** | 64 / 64 | 9,999 / 9,999 | 1,242 (frozen at step 887, by 1,012 s) | sp2's round, 2 calls (688, by 966 s), its search 12.5 s with the every-T check |
| minus1: y^2 - 1 | **yes** | 64 / 64 | 9,999 / 9,999 (ood_t, ood_n_t 10,002 / 10,002) | 1,560 (887, 1,334) | near miss: the H1 round, constant 1, subtracted; then `STA.S LDK STA.Y LDA.S T.acc.mul.o,x,d.ANY STA.X T.sub.x+n.XY T.red.x+dn.AN`: add N (the multiply-accumulate row, N times the loaded 1), subtract 1, reduce mod N, which is (v - 1) mod N for every v below N, 0 included (688, by 1,283 s; the search 313 s) |
| ten | **yes** | 64 / 64 | 9,999 / 9,999 | 1,110 (887, 947) | as in g1 and g2, 3 calls (688, by 904 s) |
| plus3 | **yes** | 64 / 64 | 9,999 / 9,999 | 1,359 (887, 1,127) | as in g2: the H1 round + 3, then reduce against N (688, by 1,074 s) |
| tenrho | **yes** | 64 / 64 | 9,999 / 9,999 | 1,363 (887, 1,146) | the ten round + 1, then reduce (688, by 1,094 s) |
| rho | **yes** | 64 / 64 | 9,999 / 9,999 | 1,644 (887, 1,410) | the H1 round + 1, then reduce against N (688, by 1,343 s; the search 188 s) |
| xfirst | **yes** | 64 / 64 | 9,999 / 9,999 | 961 (809, 807) | H1's round (610, by 787 s) |
| M4 (its own data, H1's 3,600 s) | **yes** | 64 / 64 | 9,000 / 9,000 | 3,600 (never frozen) | as in g1 and g2 (632, by 1,877 s) |
| xy | no | - / - | 2 / 9,999 | 703 (the file's 20,000 steps) | the reader, as unchanged |
| cube | no | - / - | 11 / 9,999 | 3,600 | no round and no near miss in 6 round searches (levels 2-7): 3 calls exhausted (at level 7, 9.7M states of three calls in 62 s) |
| mont | no | - / - | 10 / 9,999 | 3,600 | likewise |
<!-- /G3 -->

### 4.6 All the changes side by side

"yes" = certified (T = 64 on both ladders, every rung 768/768); training seconds in brackets. Each change was rerun on
every task (xy only unchanged, in g2 and in g3: its reader stops before any round search, and no change touches it).

<!-- ALL -->
| task | sp2 unchanged | g1: a wider search | g2: + constants from the data | g3: + the answers of every T (final) |
|---|---|---|---|---|
| H1 (official manifest) | yes (1,277) | yes (1,275) | yes (1,222) | yes (1,242) |
| xfirst | yes (1,067) | yes (1,314) | yes (990) | yes (961) |
| ten: 10 y^2 | no | **yes** (1,263) | yes (1,206) | yes (1,110) |
| M4: every T = 8 | no | **yes** (3,600) | yes (3,600) | yes (3,600) |
| rho: y^2 + 1 | no | no | **yes** (1,417) | yes (1,644) |
| plus3: y^2 + 3 | no | no | **yes** (1,354) | yes (1,359) |
| tenrho: 10 y^2 + 1 | no | no | **yes** (1,312) | yes (1,363) |
| minus1: y^2 - 1 | no | no | no: T = 4 / 2, wrong only where a round's input is 0 | **yes** (1,560) |
| cube: y^3 | no | no | no | no |
| mont: y^2 10^-8 | no | no | no | no |
| xy: x y^T, four numbers | no (the reader) | - | no (the reader) | no (the reader) |
<!-- /ALL -->

## 5. H1 with the final changes

The final file is g3, `submission_spb_g3.py` (md5 `30efec64`, 253,321 bytes).
- **Official run** (`records/g3_h1_h100.json`): the competition's `h100_hard_h1.json` untouched, one H100, seed 74,
  the file byte for byte: T = 64 on both ladders, every rung 768/768, test 9,999/9,999, ood_t and ood_n_t
  10,002/10,002; 1,242 s of training (the round at step 688, frozen at 887), 194 s of evaluation.
  `tools/verify_records.py records/g3_h1_h100.json=submission_spb_g3.py` checks it out.
- **8 seeds** (`records/submission_spb_g3_b200_seed*.json`, run `spb-g3-seeds`, B200s, the official manifest with
  only `runtime.seeds` changed; B200 runs count as research here, as for sp2's seeds): **8 of 8 certify**, each T = 64
  on both ladders, every rung 768/768, every split exact, and each record checks out against the file
  (`tools/verify_records.py`).

<!-- SEEDS -->
| seed | training s | the round (step, by s) | frozen (step, by s) | evaluation s |
|---|---|---|---|---|
| 74 | 828 | 688, 678 | 887, 705 | 93 |
| 1 | 1,077 | 630, 842 | 829, 868 | 164 |
| 2 | 856 | 683, 702 | 882, 725 | 89 |
| 3 | 1,991 | 610, 1,751 | 809, 1,778 | 194 |
| 4 | 798 | 696, 656 | 895, 682 | 89 |
| 5 | 754 | 637, 623 | 836, 641 | 92 |
| 6 | 1,198 | 600, 909 | 799, 990 | 164 |
| 7 | 913 | 684, 760 | 883, 788 | 88 |
<!-- /SEEDS -->

Every seed's round is H1's own, "copy x, multiply, reduce", with no constant. Seed 3 ran on a slow host (its reader's
phases took 692 s to step 600, where the others took 45-260 s) and its climb practised the other long division, with
the modulus in N (`T.red.x+on.XN`); its round came by 1,751 s, still half the budget. sp2's own 8 seeds (SELF_PRACTICE
§4.2) found their rounds by 650-1,775 s on B200s; g3's by 623-1,751 s.

## 6. What failed, what is still designed, and the cost

**Not learned, and why.**
- **cube, x^(3^T):** a round needs four calls (multiply, reduce, multiply by x, reduce: x^3 does not fit 16 places, so
  the reduction must come between the multiplies), and x must survive the first reduction in a register. In the cube
  runs (seed 74) the climb's one reduction on x + o N copies its modulus into Y and uses S as scratch, and the multiply
  consumes X, so x is gone; long division can only take its dividend from X (`BRG.X` is the one "bring a digit"
  instruction). A reduction that keeps Y exists and is learnable: `CLA P[BRG.X T.red.x+dn.AN]` (cost 4, dividend in X,
  modulus in N) passes its practice problem (checked locally), though at seed 74 its probe never reached it (at level
  4 it streamed its last level and closed none of the gap); and `CLA P[BRG.X T.add.pair.AN T.red.x+dn.AN]` (cost 5)
  was practised at level 5 by M4's climb and by H1's seed 3, whose level 4 had not practised that reduction in any
  binding. Once the X-A
  reduction is practised at level 4, as at seed 74, the concept rule practises no other binding of it until the
  variants' levels, where costlier concepts come first. A 4-call search would also have to hold the 3-call level (about 17M states at level 4,
  against the 4M budget) and apply some 2.5 billion calls after it, over an hour on an H100 at the 3-call level's
  measured speed. No near miss helps: x^3 and x^2 mod N differ by a factor of x, not by a constant. Tried: g1-g3 in the evaluator (each exhausted 3 calls at every level);
  the 4-call search was not tried.
- **mont, y^2 10^-8 mod N:** no law of the book poses Montgomery's reduction (a division by the base mod N), and its
  one step ("add N until the low digit is 0, then shift right") is about 8-10 instructions on this machine, past the
  probes' reach of 5-7. Its gap to x^2 mod N is a factor of 10^-8 mod N, which differs from row to row, so no constant
  near miss exists. Tried: g1-g3 in the evaluator; no law added.
- **xy, x y^T mod N with a fourth number:** stage 1's reader takes exactly three fields, and its first stage never
  completes on four (all 20,000 steps in phase A). No change here touches the reader; a reader for any number of
  fields is a redesign, not tried.

**The practice check's weak spot (from the s3i ablation, `integration/ablation_s3/reduce_check.py`).** Stage 3's
acceptance of a reduction can let through a repeated subtraction that is right only for quotients up to 600 (about 3%
of runs on H1's rows). It touches nothing here: across this study's 50 records the climb practised only three
reductions on x + o N, all of them long division, right for any quotient (each place brings one digit in and reduces
a number below 11 N with at most 10 conditional subtractions): `STA.Y P[BRG.X T.red.x+dn.AY]` (every H1-prompt run but
one), `CLA P[BRG.X T.add.pair.AN T.red.x+dn.AN]` (M4's runs and H1's seed 3) and `CLA P[BRG.X T.add.pair.AY
T.red.x+dn.AY]` (seed 3). No failure here is at the reduction: cube and mont find no round, xy stops at the reader,
and g2's y^2 - 1 fails only on the rows that pass through 0 (§4.4).

**Still designed** (on top of SELF_PRACTICE.md §6's list for sp2, unchanged):
- the round search's sizes: 3 calls, 2 calls for T > 1, 3 calls for the rest of the way, at most 4 near misses kept,
  2,048 answer rows per other T;
- the near-miss test's form: a gap that is one constant mod N, either way. It uses the data's own N and the mined fact
  that every answer is below N; a gap of another shape (a factor, as for cube and mont) is not looked for;
- LDK, a 33rd instruction that only the rest of the way can use, and the constant it loads, read off the answer rows;
- "a round ends with a reduction mod N", tested with the book's red.invariant: a type rule written with the family's
  modular answers in view. Without it the shortest rounds for y^2 + 3 and 10 y^2 + 1 would be wrong on about 3 inputs
  in N; with it, they reduce;
- "the answers of every T", restricted to one-digit T because a two-digit T can still be read backwards when the
  round is searched.

**The cost.** About $139 of Modal in all, within the $400 soft budget (`--mode cost`; the last two runs' latest hours
were not billed yet, so their list-price estimates): the unchanged sweep $35.6, g1 $34.5, g2 $25.2, g3 $25.5 (list),
H1's 8 seeds $16.6 (list), the offline check about $1.6. Each hour-long failing H100 run costs about $4.1, each
certifying one about $1.5-2.

## 7. Files and how to rerun

Everything is in `cpu_first/integration/sp_breadth/`; no shared file changed (the shared ones were copied here first).
- `spbcore.py`: `integration/self_practice/spcore.py` with the round search's changes (§3): `call_seqs`, `below`,
  `add_digits`, `mod_diff`, `constant_gap`, `NearRound`, `route` (the constant and the spill), and in `SelfClimb`:
  `search_round`, `ends_ok` / `final` / `round_ends`, `deep_ok`, `_accept_round` (batched), `rest_of_way`,
  `macro_classes`, `_round_many` with `t_rounds`, `_calls_many`; and `bfs_roots(hold_last=)`.
- `libb.py`: `integration/lib/libcore.py` with LDK on its four machines (`step`, `_Ops` / `_ustep`, `RoundMachine`,
  `SteMachine`), a constant in `Setting`, `round_settings` / `round_problem` / `round_ok`, and `bfs(final=)`.
- `template_b.py`: `integration/self_practice/stage3_sp_template.py` with the settings (`SP_MACRO_COST`, `SP_MACRO_T`,
  `SP_CONST`, `SP_NEAR_K`, `SP_REST_COST`, `SP_ROUND_ENDS`, `ANS_DEEP`), the answer rows of the other one-digit T's
  (`keep_answers`, passed to the climb in `climb_unit`), and the round's constant carried to the round machine and the
  prediction path.
- `make_spb.py`: the builder (make_sp.py's, from these copies), with `CONSTANT=VALUE` arguments for a build's settings
  and a pass that drops the top-level functions and classes the file never names (libcore's curriculum climb and task
  builders are dead in a self-practice file), which keeps the file under 256 KiB.
- `offline.py`, `modal_spb.py`: the round's search alone on recorded libraries and each formula's answers (a study
  check, with Python's integers for the answers), on a Modal GPU. `libs/` holds the libraries it reads.
- `test_ldk.py`, `test_near.py`, `test_deep.py`, `smoke_phase_c.py`: local checks (a minute or less each; the smoke
  test needs about 10 GB, so it is not for a laptop). `test_near.py` also checks that a reduction against a register
  holding 0 fails the reduction rule; `test_deep.py`, that g2's y^2 - 1 round fails the every-T check on training's
  T = 4 rows and that the search then finds a right one.
- `lib_of.py`: a record's library for `offline.py`; `summarize.py`: the tables here, from `records/`.
- `plans/`: the evaluator runs (`u0_unchanged.json`, `g1_sweep.json`, `g2a_sweep.json` and `g2b_sweep.json`,
  `g3a_sweep.json` and `g3b_sweep.json`; `g2_sweep.json` is the two g2 plans in one); `records/`: their records, and
  the 8 seeds (`--mode seeds`, run `spb-g3-seeds`).

The sources as they stand rebuild the final file byte for byte (md5 `30efec64`). g1 (`c8422b2f`) and g2 (`371c3d67`)
were built from earlier states of the same sources (g1 before the batched candidate check; g2 before the fourth
version of the reduction rule and the every-T check) and are kept as built, their sources also embedded in every one
of their records.

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python integration/sp_breadth/make_spb.py submission_spb_g3.py SP_CONST=True 'SP_ROUND_ENDS="red"' ANS_DEEP=2048 SP_REST_COST=3
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode plan --plan integration/sp_breadth/plans/g3b_sweep.json --out integration/sp_breadth/records --cap 8
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/sp_breadth/submission_spb_g3.py --seeds 8 --memory-gb 48 --out integration/sp_breadth/records
```

```bash
cd cpu_first && python3 tools/verify_records.py integration/sp_breadth/records/g3_h1_h100.json=integration/sp_breadth/submission_spb_g3.py
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m integration.sp_breadth.test_ldk
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m integration.sp_breadth.test_deep
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python integration/sp_breadth/summarize.py u0 g1 g2 g3
```
