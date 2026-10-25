# Bar 3 to bar 2 from below: stage 3's digit instructions, invented by bar 3's learner

*2026-10-24. Code and records in `cpu_first/bar3/bridge/`. Nothing submitted; no file outside this
folder changed (the builders assert the md5s of the files they copy).*

*Update, 2026-10-25: [CARRY.md](CARRY.md) finished the learner's lost round 1. Unchanged, the learner invents
add-with-carry there by adding, comparing with the base and subtracting it once, a route §3 did not consider.
Multiply-accumulate is still not invented. So the variant's ADC cell is an invented instruction, and its MUL cell
still rests on a hand-written piece.*

## In plain words

<!-- PLAIN -->
**The question.** Stage 3 (s3i) learns x^(2^T) mod N at H1's size because its digit machine has 32 instructions a
person chose: subtract with borrow, add with carry and multiply-accumulate at a pointer, keep or discard a register on
the borrow (SEL.S), bring X's next digit in (BRG.X), moves, shifts, and loops by the number's length. A strict judge can
call those the algorithm. Bar 3's learner invents formulas on a bare counting machine, but counting at H1's size would
take about 10^16 steps. One place of a digit machine is tiny, though: two digits below the base and a carry. Can bar
3's learner invent stage 3's instructions there, and does stage 3 still find its round with the invented ones?

**What a digit instruction is.** A digit machine computes a law on whole numbers one place at a time: at each place it
sees the operands' digits and what the place below passed up, and it writes a digit and passes something on. That step
is the law's *slice*, read off the law itself. Example: 1385 − 4189 at place 2 sees 3 and 1 and a borrow from below
(85 < 89), writes (3 − 1 − 1) mod 10 = 1 and passes no borrow (385 ≥ 189): row (3, 1, 1) → (1, 0) of stage 3's
subtraction table. Cutting bar 3's law book (SELF_PRACTICE's 11 laws, succ, pred and select) to one place, and keeping a
slice only when it is a function of what the place sees, leaves 18 slices. Eleven of them compose back into their law
when looped over the places from a cleared carry: subtract (both orders), compare (both), add, multiply by a digit
(both), multiply-accumulate (both) and select (both). Min, max, the remainder, the quotient and the full product drop
out, since one place cannot know them. Stage 3's four cell functions are among the eleven.

**What the learner invented.** Bar 3's learner, unchanged in its rules and sources, ran on a counting machine with a
register for each thing the place sees or writes. Its instructions are add one, subtract one, clear, copy, "repeat r
times" and "if r is 0": no digits, no carry, no wrap-around.
- **Select** (what SEL.S does at one place) was invented in round 0 under every rule: `L.C[MOV.A.R]`, "if the flag is
  set, take the operand's digit".
- **Subtract with borrow** (SBB) was invented in round 0 under the pairs signal and under practise-everything, as
  `L.R[INC.C] L.C[piece]`. The piece is the learner's own practice problem: the target restricted to the rows where R < C,
  meaning "a borrow comes in and nothing is subtracted". Its program is `Z.A[INC.C MOV.A.N] DEC.C DEC.A`: take one off,
  wrapping at the base. C first counts how many to take off, then holds the borrow. Hardest first did not invent it
  in 2 rounds: it practised the costliest problems instead (absdiff, compare, equal; then gcd, succ, a square).
- **Add with carry** (ADC) and **multiply-accumulate** (MUL) were not invented in round 0 under any rule, after 3 to
  12 search-hours a run; their round 1 was cut short when Modal's spend limit stopped every run. Both need the mirror-image
  piece, "add where R < C" (add one, wrap at the base, carry). That piece takes 8 counting instructions, and no probe
  reaches past 6: testing "reached the base" costs three instructions more than testing "reached zero". Given the piece,
  the search finds add as `L.R[INC.C] L.C[piece]` and multiply-accumulate as `L.H[L.R[INC.C]] L.C[piece]`.
  *Later* ([CARRY.md](CARRY.md)): the finished round 1 invents add-with-carry under the pairs signal and
  practise-everything, by another route that reduces the sum once. Multiply-accumulate is still not invented.
- Each invented program, compiled to a table at base 10, is exactly stage 3's cell (200 of 200 rows), and so is each
  program built on the hand-written add piece (10,000 of 10,000 rows for multiply-accumulate).
- **The rest** of stage 3's instructions are not functions of one place, or are trivial there. Moves and clears are
  the counting machine's copy and clear. SHL is a slice of x·B, but the book has no law on the base. SHR and BRG.X move
  digits down or across registers, which a step running up from place 0 cannot do. The pointer and the loops belong to
  the frame.

**Stage 3 on the invented instructions.** s3i's climb (exact tables, H1's rows, one GPU) finds the round:
- with s3i's own 32 instructions, in 1,030 s;
- with the invented subtract plus add and multiply-accumulate against every register, SEL.S kept, in 1,385 s;
- with SEL.S replaced by the invented select at the pointer (36 instructions), in 1,606 s. Only the conditional
  subtraction changes: compare x with N, then at every place take x's digit back if the compare borrowed
  (`T.lt P[SELC.X INC]`, 4 instructions, the same count as s3i's);
- with all eleven slices that lift (61 instructions), no round. At s3i's budgets the first search runs out of states
  before 5-instruction programs. At 4 times them, the climb trips over two weak checks in s3i's climb:
  - It accepts a wrong conditional subtraction, because modx is probed only on x ≤ N.
  - With that probe extended above N, it finds every piece except the reduction. The reduction's 5-instruction program
    is there and passes the check, but a repeated subtraction that is right only for small quotients passes the
    search's probes. Its paddings fill cost 5 with tens of thousands of look-alikes, and the search tests at most
    50,000 there, one program each, so the right one was most likely never tested.

**The variant certifies.** `submission_stage3_bridge.py` is s3i with the 36-instruction set:
- subtract, add and multiply-accumulate at the pointer against X, Y, N or S, their tables learned exactly as s3i learns
  them;
- a select at the pointer, its table the compiled counting program;
- no SEL.S;
- s3i's frame (moves, pointer, shifts, BRG.X, loops).

It certifies T = 64 on both ladders, every split exact, on the official H100 run and on 8 more seeds (74 and 1 to 7,
B200). The round comes at the same training step as s3i's on 7 of the 8 seeds, and one step earlier on the eighth.
Training takes 37% longer: 1,839 s against 1,340 s on the official H100, out of 3,600 s. ABLATION_S3 found SEL.S
needed (without it no round); here a select the learner invented takes its place.

**What this says about bar 2.** Bar 2's first open question is stage 3's instruction set. This study narrows it:
- *Subtract-with-borrow and the select come out of bar 3's learner.* It practised from bar 3's own law book at one
  place and was told nothing about digits or carries.
- *Add-with-carry and multiply-accumulate are slices of laws in the same book.* The book poses them, but the learner
  does not invent them: their one missing piece is 8 instructions, past its search. The variant includes them with
  hand-written pieces. Their tables are still learned from data, as s3i's are. *Later* ([CARRY.md](CARRY.md)): the
  finished round 1 invents add-with-carry, so only multiply-accumulate's piece is still hand-written.
- *What remains designed.* The law book itself, the frame (moves, the pointer, SHL/SHR, BRG.X, the loops), and the
  choice of which slices become cells. The variant uses 4 of the 11 that lift. Each of the other 7 is one of these 4
  with its operands swapped, an operand at zero, or an output dropped (compare is subtract's borrow alone). That rule
  is ours, applied after the fact; the learner does not apply it. With all 11, s3i's climb finds no round. At s3i's
  budgets that is reach. At 4 times them it is not a missing instruction: two of the climb's checks are weak, and the
  extra instructions supply wrong programs that pass them. For the second check that is the likely cause; the run
  that would confirm it was refused by Modal's spend limit.
- *The invention happens before training.* The variant takes its instruction list from this study. It does not invent
  its cells inside its training run (§6 says why).
<!-- /PLAIN -->

## Details

### 1. A digit instruction is one place of a law

Stage 3's machine (s3i) has 32 instructions on registers of digits. Eleven apply a small table at the pointer: SBB.r
(subtract with borrow), ADC.r (add with carry) and MUL.r (multiply-accumulate), against a register r. SEL.S keeps or
discards a whole register on the borrow. The other twenty move digits or the pointer: load, store and clear, the
flags' clears, the multiplier digit's load, the pointer's step, the shifts, and BRG.X (bring X's top digit into A).
Loops repeat a body P, B or W times.

**The slice.** A digit machine computes a law on whole numbers one place at a time. At place i it sees the operands'
digits there and what the places below passed up; it writes the result's digit and passes something on. That step is
the law's *slice*, and `slices.py` reads it off the law itself, with two generic definitions:
- what place i passes on (the *state*): for a law with a value, the part of the law on the low parts that lies above
  place i, floor(L(x mod B^i, y mod B^i) / B^i) (its size, when that is a borrow); for a predicate, the predicate on
  the low parts. A constant in the law (the 1 of x + 1) becomes the state of place 0;
- what place i writes: digit i of the law's value.

A slice is posed only if it is a function of what the place sees: bar 3's own test on any problem (the same inputs
never want two answers).

**Example.** 1385 − 4189 in base 10. At place 2 the digits are 3 and 1, and the places below pass a borrow, since
85 − 89 < 0. The slice writes (3 − 1 − 1) mod 10 = 1 and passes no borrow on, since 385 − 189 ≥ 0. That is s3i's
subtraction cell, row (3, 1, borrow 1) → (1, no borrow).

**The book.** SELF_PRACTICE.md's 11 laws (lt, eq, min, max, red, div, add, sub, mul, a + b·c, a + b + c), bar 3's
succ and pred, and select, with their operands in every binding (numbers to A and R, a digit to H, a flag to C). Read
off random 3-digit numbers at bases 2 to 10:

| law | its slice | a digit instruction? |
|---|---|---|
| sub (both orders) | (a, r, borrow in) → (digit, borrow out) | yes: s3i's SBB is sub(A,R) |
| add | (a, r, carry in) → (digit, carry out) | yes: s3i's ADC |
| a + x·d (both orders) | (a, x, d, carry digit in) → (digit, carry digit out) | yes: s3i's MUL is acc.mul(A,R,H) |
| x·d (x in A or R) | (x, d, carry digit) → (digit, carry digit) | yes |
| select (both orders) | (a, r, flag) → the flag picks a digit | yes: per place, s3i's SEL.S |
| lt (both orders) | (a, r, "below so far") → "below so far" | yes: the borrow alone |
| eq | (a, r, "equal so far") → "equal so far" | yes, but its first place starts at "equal" (1) |
| succ, pred (A or R) | (a, carry) → (digit, carry), the first place's carry 1 | yes, but they start with a carry of 1 |
| a + x + d (both orders) | the digit d enters as the first place's carry | yes, but it starts with a carry of d |
| min, max | a place's digit depends on which number is smaller overall | no: not a function of the place |
| red, div | undefined on the low parts (x mod 0) | no |
| the full product x·y | a place's digit needs every lower digit pair | no |

So 18 slices are posed (eq and add are the same in both orders) and 12 laws' bindings dropped. Eleven of the 18
compose into their law when looped over the places from a cleared carry (`lift.py`, every slice checked on 3,000 random
4-digit operand tuples at base 10): sub ×2, lt ×2, add, x·d ×2, a + x·d ×2, select ×2. The other seven need a carry
*set* before the loop (eq, succ ×2, pred ×2, a + x + d ×2), and stage 3's frame can only clear one (CLC, CLK). Stage
3's four cell functions are among the eleven.

### 2. The learner on one place

**The machine** (`dcm.py`) is bar 3's bare counting machine with more registers: A (the accumulator digit: an input and
the output), R (the operand digit), C (the carry or flag, in and out), H (a digit operand), N (the base B) and T
(scratch). The instructions are bar 3's: add one, subtract one (stopping at 0), clear, copy, "repeat r times" and "if r
is 0". Nothing in it knows digits, carries or wrap-around: a wrap at B has to be built from the zero test and the
counters, as bar 3's x² mod N had to build its wrap at N. A problem's vocabulary names only its own registers (35
tokens and 10 loop kinds for a 5-register problem, as bar 3's machine; 48 and 12 with H).

**The search and the loop** (`dsearch.py`, `dlearner.py`) are bar 3's, unchanged in what they do: breadth-first search
by cost with loop bodies of cost at most 2; probes of 30,000 states then 1M for the unsolved; practice and target
searches of 3M states; MDL merges at half price; every practised problem's program a routine, the one without side
effects on its inputs (bar 3's v2); at most 5 rounds (hardest first) or 3 (the others). A problem has several inputs and
outputs (a slice writes a digit and a carry), so a program is right when every output register is right.

**The sizes.** Bar 3 searched every pair x < N ≤ 7 and accepted at N ≤ 10. Here the base plays N's part: the search
sees 32 rows sampled from bases 2 to 7, a program must be right on every row at bases 2 to 7 and at 8 to 10 (10 is
H1's base), and it is *general* when it is right at every base up to 16. For sub, add and select that is every input
of the place (278 rows at bases 2–7, 490 at 8–10); for a + x·d the rows read off the law (2,185 and 4,680).

**The sources of practice** (`dproblems.py`), bar 3's three, posed on the target's own rows:
- *law*: bar 3's law book (add, monus, absdiff, mul, div, mod, min, max, lt, le, eq, gcd, succ, pred, and select) on
  the place's values A, R, C, H and the base N, in every binding, with A built by bar 3's patterns a, a + N and
  a + qN (q a coin), values to A and predicates to C: 179 problems for a 4-input place, 280 with H;
- *slice*: the book's other slices;
- the target's *restrictions* (its rows where t1 < t2, for terms 0, 1, 2, each input, the base and the sums of two
  inputs: bar 3's grammar with 0 added), its *projections* (each output alone), and the *values* and *remainders* of
  its own near misses, with their restrictions.

That is 365 to 577 candidates a round for the arithmetic targets. Rules: hardest first (lp), the cheaper pairs signal
(ct2, BREADTH.md's: pairs with one member among the 2 hardest), and practise everything (all).

### 3. Which instructions it invents

Each of stage 3's 32 instructions, by what it does at one place:

| s3i's instructions | at one place | invented by bar 3's learner? | from which practice | search |
|---|---|---|---|---|
| SEL.S (1) | the select slice: if the flag is set, take the operand's digit | **yes**, round 0, under every rule: `L.C[MOV.A.R]` | none needed: the target probe finds it (2 instructions); under the pairs signal it came as the law R·C with C a flag, the same program | 0.9, 1.5 and 1.0 search-hours (hardest first, pairs, everything) |
| SBB.X, .Y, .N, .S (4) | the sub slice: (a, r, borrow in) → (digit, borrow out) | **yes**, round 0, with the pairs signal and with practise-everything: `L.R[INC.C] L.C[T.res]` (4). **Not** with hardest first in 2 rounds | its own restriction of the target to the rows where R < C, "a borrow comes in, nothing is subtracted": `Z.A[INC.C MOV.A.N] DEC.C DEC.A` (5). Hardest first picks the costliest problems instead (absdiff, lt, eq; then gcd, succ, a square) | 3.46 and 2.46 search-hours; hardest first 5.58 over 2 rounds |
| ADC.X, .Y, .N, .S (4) | the add slice | **no**, in round 0 under any rule (round 1 was cut short, §6) | it needs the mirror piece, "add where R < C", which no program of up to 6 instructions solves | 3.02, 4.33 and 3.57 search-hours |
| MUL.X, .Y, .N (3) | the a + x·d slice, its carry a digit | **no**, in round 0 under any rule (round 1 was cut short, §6) | it needs the same add piece | 9.73, 12.14 and 10.72 search-hours |
| LDA ×4, STA ×3, CLA, CLC, CLK, LDH ×2 | a copy or a zero | trivial: the counting machine's own copy and clear | – | – |
| SHL ×3 | a slice of x·B (its carry is the digit moving up) | not posed: the book has no law on the base | – | – |
| SHR ×3, BRG.X | digits moving down, or from X's top to A's bottom | no: not a function of one place | – | – |
| INC, the loops by P, B, W | the pointer and repetition by the frame's sizes | the frame's control | – | – |

Both invented programs compile to exactly stage 3's tables (§4).

*Later* ([CARRY.md](CARRY.md) §4): in the finished round 1, the pairs signal and practise-everything invent ADC as
"add the carry and r, compare with the base, subtract it once" (the sum stays below 2B), exactly stage 3's table. MUL,
whose sum reaches B² − 1, is still not invented.

**Why add and multiply-accumulate are harder than subtract.** Each needs one piece: the step at one place when nothing
is added but a carry comes in. For subtraction that step is "take one off, wrapping to the base below zero": the
learner's own restriction of the target to the rows where R < C, solved by a 5-instruction program
(`Z.A[INC.C MOV.A.N] DEC.C DEC.A`) at the 1M-state probe. Its program leaves the borrow as "C − 1, plus 1 if it
wrapped", so C can count: subtract-with-borrow is then `L.R[INC.C] L.C[piece]` (C counts the subtractions, then holds
the borrow), 4 instructions, found in 3 seconds. For addition the step is "add one, wrapping to 0 at the base":
reaching the base is tested by counting B − A into T first, three instructions more than testing for zero. No program
of up to 6 instructions solves "add where R < C" (round 0's 1M-state probe stored every program up to 5 and checked
every one of 6 as it streamed); the shortest we wrote takes 8
(`INC.A MOV.T.N L.A[DEC.T] Z.T[CLR.A INC.C] DEC.C`), and with it the search finds add-with-carry as
`L.R[INC.C] L.C[piece]` (4) and multiply-accumulate as `L.H[L.R[INC.C]] L.C[piece]` (5), both right at every base up
to 16. With the digit link as a routine (A mod N for A < 2N, which the learner learns in round 0 at 6 instructions),
the piece's shortest programs drop to 4, but every one of them clears the carry instead of counting with it
(`INC.A T.mod L.A[DEC.C]`): on its own rows, where a carry of 1 always comes in, the two behave the same, and the
learner keeps a problem's shortest programs only. With those as routines no add-with-carry of up to 5 instructions
exists (3M states), nor with the invented subtract added.

Searching deeper does not close the gap. Each level of these searches holds 25 to 45 times the one before, so checking
every program of 8 instructions takes about 600 times the states of checking every program of 6. bar3-scale found
that no bound on bar 3's search space shrinks it more than 1.7 times, and its compiled search is 4 to 5 times faster on
searches of a million states (`bar3/scale/SCALE.md` §8 and §1).

**What a place cannot do.** A slice sees one place's digits and what the places below passed up. Of s3i's other
instructions:
- the moves and clears (LDA, STA, CLA, CLC, CLK, LDH) are a copy or a zero at each place: the counting machine's own
  MOV and CLR, one token each; the frame applies them to whole registers;
- SHL (a register times the base) is a slice too, of x·B: each place writes the digit that came up from below and
  passes its own up, a carry that is a digit (checked on 2,000 random numbers: a function of the place). The book poses
  laws on numbers, digits and flags, not on the base, so it is not posed here;
- SHR (divided by the base) and BRG.X (X's top digit into A's lowest place) move digits down or across registers:
  place i would need a digit from a place above it, which a step that runs upward from place 0 never has. BRG.X is
  SHL through the carry applied to X and then to A, the form the generic-ops session gives it;
- INC (the pointer) and the loops (P, B or W times) are the frame's control: the counting machine has a loop of its own
  ("repeat r times"), with the count in a register.

### 4. Compiled to stage 3's tensor form

A slice's program, run on every input of the place at base 10, is a table: the form of s3i's cells (`libcore.Cells`:
rows over what the cell reads, columns the digit and the flag or carry). `lift.py` compiles each invented program and
checks two things: the table equals s3i's own cell (rows equal / rows), and the place loop of the table computes the
law on random 4-digit numbers.

| program | from | rows equal to s3i's cell | place loop = the law |
|---|---|---|---|
| select `L.C[MOV.A.R]` | invented (§3) | 200 / 200 (SEL.S's select at one place) | 3,000 / 3,000 |
| subtract `L.R[INC.C] L.C[T.res83]`, T.res83 = `Z.A[INC.C MOV.A.N] DEC.C DEC.A` | invented (§3) | 200 / 200 (SBB) | 3,000 / 3,000 |
| add `L.R[INC.C] L.C[T.ast]`, T.ast = `INC.A MOV.T.N L.A[DEC.T] Z.T[CLR.A INC.C] DEC.C` | the piece written by hand (§3) | 200 / 200 (ADC) | 3,000 / 3,000 |
| multiply-accumulate `L.H[L.R[INC.C]] L.C[T.ast]` | the same piece | 10,000 / 10,000 (MUL) | 3,000 / 3,000 |

The counting programs and s3i's learned cells are the same functions. A compiled table is what stage 3's machine runs
in one step; at base 10 the counting programs take up to 2 steps (select), 50 (subtract), 170 (add) and 1,539
(multiply-accumulate) machine steps.

### 5. Stage 3 on the invented instructions

**The machine** (`libcore_b.py`, built by `make_libcore_b.py` from the libcore body s3i inlines, md5 asserted): the
cells come from a list (`CellSpec`: what a cell reads, in order, what it writes, its table), and `set_instructions`
makes the instruction set the frame (s3i's moves, pointer, shifts, BRG.X, optionally SEL.S) plus every cell against
every register it takes. With the defaults it is libcore exactly: `test_libcore_b.py` runs every primitive, a 2,047-item
vocabulary, the round machine and the one-hot machine against libcore's, on a CPU and on an H100 (where the round
machine runs from CUDA graphs).

**The climbs** (`climb_b.py` on one H100 each: libcore.Climb with s3i's constants, the cells exact, 4,096 of H1's rows,
the round searched against 400 T = 1 answers whenever new routines appear; records `records/climb/*_h100_c1`):

| instruction set | instructions | the conditional subtraction (level 1) | multiply (level 5) | round (level 6) after |
|---|---|---|---|---|
| s3i's own (control) | 32 | `LDA.X STA.S T.lt SEL.S` | `CLA W[T.mac SHL.X SHR.Y]` | 1,030 s |
| the invented subtract, add, multiply-accumulate against every register, and s3i's SEL.S | 33 | `LDA.X STA.S T.lt SEL.S` | the same | 1,385 s |
| the same with SEL.S replaced by the invented select at the pointer (SELC.r) | 36 | `T.lt P[SELC.X INC]` | the same | 1,606 s |
| every slice that lifts (§1), each against every register it takes (B200) | 61 | never: level 0 solves nothing | – | none (stops after 35 s) |
| the same, at 4 times the climb's budgets (B200) | 61 | `T.lt P[BRG.X RSELC.X]`: passes modx's probes, wrong for N < x < 2N | never | none (stops after 329 s) |
| *diagnostic, modx also probed on N < x < 2N:* s3i's own (H100) | 32 | `LDA.X STA.S T.lt SEL.S` | `CLA W[T.mac~ SHL.Y SHR.X]` | 1,305 s |
| *the same diagnostic:* every slice that lifts, 4 times the budgets (B200) | 61 | `LDA.X CLC P[SUB.N INC] P[SELC.X INC]` | `CLA W[T.mac~ SHL.Y SHR.X]` | none: every piece but the reduction (stops after 6,684 s) |

Every climb finds the same pieces at the same levels: the comparison (level 0), the conditional subtraction (1), the
addition (2), the chain and the multiply's row (3), the accumulating row, the long-division step and the reduction
(4), the multiply (5), and the round `LDA.X STA.Y T.mul STA.X T.reduce` (6). The only change in the programs is the
conditional subtraction without SEL.S: compare x with N (A holds x − N and the borrow is set when x < N), then at every
place, if the borrow is set, take x's digit back. It costs 4 instructions, as s3i's does, and needs no copy of x in S.
The two bigger sets cost time, not reach: each level's searches hold more states (level 4 took 834 s against
533 s), and hosts differ by about 20% (ABLATION_S3.md §4).

**Every slice that lifts.** The variant's four cells are a choice: of the eleven slices that lift, the other seven
are one of the four with its operands swapped (sub(R,A), a + x·d with x and a swapped, select with its two operands
swapped), an operand at zero (x·d in both bindings: a + x·d with a = 0, after a move), or an output dropped (lt, both
orders: the subtraction's borrow alone). With all eleven against every register they take, the set has 61 instructions. At s3i's
budgets the climb's first search, the comparison, holds 9.3M programs of cost 4 against a budget of 5.8M, so it stops
there and never reaches the 5-instruction comparison, `CLC LDA.X P[SUB.N INC]` (s3i's level 4 holds 0.55M, and its
comparison is found in the streamed level 5). Nothing is solved at level 0 and the climb ends after 35 s
(`records/climb/lifted_b200_c1`). The run at 4 times the budgets (state budgets and seconds per search) gives the
61-instruction set s3i's reach: level 4 stored, level 5 streamed (`records/climb/lifted_b200_x4`). It finds the
comparison at level 0 and stops at level 2 after 329 s, with no round, for a different reason: **a weak check in s3i's
climb**. The conditional subtraction (modx) is posed on probes with x < N (keep x) and x = N (give 0), never
N < x < 2N. With 61 instructions the climb's comparison leaves A = x (`CLC LDA.X P[LT.N INC]`), and the first program
to pass modx's probes, `T.lt P[BRG.X RSELC.X]`, zeroes x = N and keeps everything else: on N < x < 2N it is right on
0 of 12 numbers, where the variant's `T.lt P[SELC.X INC]` is right on 12 of 12 (`checks/modx_overfit.py`). The
addition is checked by undoing it with modx, so with this modx no addition passes, at cost 5 or by the primitives at
108M states, and the climb has nothing left to do. In s3i's 32 instructions and the variant's 36 the shortest program
that passes is the true one, so neither is affected; the s3i ablation found the same kind of weakness in the
reduction's check (`integration/ablation_s3/reduce_check.py`).

With modx's probes extended to N < x < 2N (a third of them x + N, the target x; `climb_b.py --modx-above`, a diagnostic
that computes x + N by hand, as no training run may), the control with s3i's own 32 instructions finds the same pieces
and the same round, after 1,305 s (`records/climb/s3i_h100_c1m`). The 61-instruction set at 4 times the budgets
(`records/climb/lifted_b200_x4m`) then finds every piece but one: the comparison, the true conditional subtraction
(`LDA.X CLC P[SUB.N INC] P[SELC.X INC]`, cost 5), the addition, the chain, the multiply's row (by the x·d slice:
`LDH.Y B[INC RMULD.X]`), the accumulating row, the long-division step and the multiply. It misses the reduction at
levels 4, 5 and 6, and stops after 6,684 s with no round. The cause is again a weak check, this time reduce's own *search* probes:
- the reduction's program at cost 5, `CLA P[BRG.X STA.Y T.modB~]`, is in that machine and library: it reaches the
  goal on the search's own probes, passes the climb's acceptor, and its loop is an item of the search's vocabulary
  (908,724 items; `checks/lifted_reduce.py`, `checks/reduce_vocab.py`);
- reduce's 8 search probes have small quotients only, so a repeated subtraction, `B[T.modB]` (cost 2, right only for
  small quotients: the program the s3i ablation found slipping through the acceptor in about 3% of runs), passes them;
  its harmless paddings with the 61 instructions fill cost 5 with goal states. Rerun as the climb ran it
  (`checks/reduce_search.py`, one H100, 643 s), the search tested 52,240 goal programs at costs 2 to 5, one spelling
  per goal state, and the acceptor rejected all of them;
- the search lists at most 50,000 goal states at a cost (bfs's `max_goals`), taken in the order of their hash, so the
  true reduction's state was most likely cut before it was tested. The rerun without that cap, which would confirm
  this, was refused: Modal disabled the workspace at its spend limit (§6).

So the 4-of-11 choice is load-bearing for s3i's climb as it stands, but not because the other slices are missing or
unreachable: every extra instruction either finds the same piece by another program or adds look-alikes that pass weak
probes. With the two weak checks repaired (modx's probes above N, as here; reduce's search probes, or the goal cap),
every piece the round needs exists at cost 5 in the 61-instruction machine: the reduction checked by hand, the others
found by the climb. Whether the climb then finds the round was not run.

**The variant** (`make_s3b.py` → `submission_stage3_bridge.py`, s3i's md5 asserted; 223 changed lines, 179,720 bytes,
the same 207,374 learned numbers as s3i): s3i with
- its cell instructions from the list: SUB, ADD and MAC at the pointer against X, Y, N or S (MAC.S is new), their
  tables the ones s3i learns (the subtraction from the order of the numbers, the addition by sub.inverse, the multiply by
  mul.zero and mul.ladder, unchanged); and SELC at the pointer against X, Y, N or S, its table the compiled counting
  program (`L.C[MOV.A.R]`);
- no SEL.S; the frame (moves, the pointer, shifts, BRG.X, loops) as in s3i;
- the prediction path's one-hot machine reading one straight-through table per cell.

At the truth (`checks/designed_bridge.py`: every table exact, the library study's library rewritten in the variant's
names, SEL.S as `P[SELC.S INC]`) its prediction path is exact on depth_t_1, depth_t_8, depth_ood_n_t_2 and
depth_ood_n_t_64 (8 rows each), as s3i's is.

**A bug, first.** The first launch (md5 `25223232`) crashed on all seven B200 seeds just after its round was found: the
round machine, which a GPU replays from a CUDA graph, built an index tensor from a Python list inside the graph's
capture, which CUDA forbids. The CPU checks could not see it. Fixed (the index made on the device once), checked on an
H100 with every machine test (`test_libcore_b.py`), relaunched as md5 `85ff74c5`. The failed records are kept in
`records/runner_bug1/`.

**The certification.** The relaunched file (md5 `85ff74c5`) certifies on the official manifest, one H100, and on
seeds 74 and 1 to 7, one B200 each (`tools/modal_runner.py`, `--mode official` and `--mode seeds`; records
`records/runner/submission_stage3_bridge_official2.json` and `records/runner/submission_stage3_bridge2_b200_seed*.json`,
every one checked against the file by `tools/verify_records.py`). Every run: T = 64 on both ladders, all 14 rungs at
768/768, test 9,999/9,999, ood_t and ood_n_t 10,002/10,002.

| run | hardware | round: step, s | frozen: step, s | training s / evaluation s | s3i on the same seed (H100): round step, s |
|---|---|---|---|---|---|
| official (seed 74) | H100 | 671, 1,603 | 870, 1,690 | 1,839 / 281 | 671, 1,095 |
| seed 74 | B200 | 671, 1,825 | 870, 1,913 | 2,115 / 380 | 671, 1,095 |
| seed 1 | B200 | 613, 1,685 | 812, 1,743 | 1,954 / 325 | 613, 1,041 |
| seed 2 | B200 | 663, 1,570 | 862, 1,642 | 1,853 / 311 | 663, 948 |
| seed 3 | B200 | 584, 1,257 | 783, 1,332 | 1,460 / 178 | 584, 959 |
| seed 4 | B200 | 679, 1,299 | 878, 1,374 | 1,488 / 166 | 679, 957 |
| seed 5 | B200 | 619, 1,753 | 818, 1,816 | 2,091 / 317 | 619, 1,062 |
| seed 6 | B200 | 580, 1,702 | 779, 2,009 | 2,285 / 342 | 580, 1,054 |
| seed 7 | B200 | 664, 1,299 | 863, 1,362 | 1,488 / 172 | 665, 1,099 |

(The seconds are the evaluator's elapsed time at its first progress line after the event; `report.py --certified`.)
The round comes at the same training step as s3i's on 7 of the 8 seeds and one step earlier on seed 7 (664 against
665): the tables learn exactly as s3i's do, and the climb finds the same pieces in the same order, so mostly only its
seconds change. On the official H100 the round comes 508 s later
(1,603 s against 1,095 s) and training ends 498 s later (1,839 s against 1,340 s), within the official budget: the
climb's searches hold more states with 36 instructions than with 32, as in the exact-table climbs above. The B200 seeds
are not faster than the H100 here; the hosts differ (the climb's searches mix GPU and host work).

### 6. What was not done, and why

**Inventing inside the training run.** The variant takes its instruction list from this study; it does not invent its
cells while it trains. The obstacle is not speed but the targets. Every practice problem here has answers computed by
the law book's own arithmetic (Python's `+`, `-`, `//` on the place's values), as in bar 3. Inside a training run
those answers would be hand-written arithmetic producing training targets, which stage 3 has none of: its tables
learn from the order of the numbers and from laws that relate them, and no number is computed from a value. A learner
that took its slices' answers from the model's own learned tables instead would only re-derive tables the model
already has. So the variant keeps s3i's way of learning the tables' contents and changes which instructions exist.

**A GPU port of the counting search** (`tdcm.py`) was written toward that end. It is exact against the numpy search
(`test_tdcm.py`: the same registers and dead flags on random machines for 3,706 items, and the same levels and programs for three
searches), but a direct port is slower than numpy on a CPU core (23 s against 3 s for a 30,000-state search); its H100
benchmark was stopped before it finished, to stay within the session's 8 GPUs. It is not used. The better route is
bar3-scale's compiled engine (`bar3/scale/engine.py`, numba, exact against bar 3's search). On Modal's hosts it is 4
to 5 times faster than bar 3's search on 1M-state searches and 24 to 45 times faster with carried routine libraries,
the case of this learner's later rounds (`bar3/scale/SCALE.md` §1 and §2). Carrying it to this six-register machine was
not done here.

**The learner's later rounds.** At 07:27 PT Modal disabled the workspace at its spend limit and stopped every running
container, the learner's included. Round 0 had finished for all 12 runs and round 1 for hardest-first subtraction;
round 1 of add and multiply-accumulate (started 04:08 to 06:52) and round 2 of hardest-first subtraction were lost.
Planned were 5 rounds for hardest first and 3 for the other rules. The shared search cache keeps every finished search,
so a relaunch (`modal_bridge.py --multi --configs ... --tag r1`) replays the finished ones at no cost. CARRY.md §4
did this for add and multiply-accumulate: round 1 invents add, not multiply-accumulate.

**The one run left to do: the reduce search without its goal cap**, which would confirm why the 61-instruction climb
misses the reduction (§5). It was refused for the same reason. One H100; the capped search took 643 s, and the
uncapped one lists and tests every goal state at cost 5. The function's own limit is 2 hours:

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/bridge/modal_climb_b.py --script bar3.bridge.checks.reduce_search --tag reduce_search_nocap --args 100000000
```

It prints its last 3,000 characters and leaves the whole log on the Volume `bar3-bridge` as
`climb/script_reduce_search_nocap.log`, which `modal_climb_b.py --fetch` copies to `records/climb/`. The cap is
confirmed as the cause if a cost-5 reduction is accepted, such as `accepted: ['CLA P[BRG.X STA.Y T.modB~]', ...]`. If
none is accepted, the cause is elsewhere, for instance one spelling per goal state.

**The untargeted book mode** (`dbook.py`: practise every slice of the book with no target, so that the learner, not
us, chooses which slices to invent) ran only as a local smoke test at tiny budgets, where it invented both selects.

### 7. How hard each was tried

- **The learner**: 12 runs, the four targets (subtract, add, multiply-accumulate, select) each under three of bar 3's
  rules (hardest first: 5 rounds; the cheaper pairs signal and practise everything: 3), bar 3's budgets (probes of
  30,000 then 1M states, practice and target searches of 3M, lookaheads of 30,000), on Modal CPU containers (2 cores,
  32 GiB hard limit), identical searches shared through a cache. Round 0 finished for all 12 and round 1 for
  hardest-first subtraction before Modal's spend limit stopped them (§6): 58.4 search-hours in the finished rounds,
  over 41 billion states. Per target: select 0.9 to 1.5 search-hours a run, subtract 2.5 to 5.6, add 3.0 to 4.3,
  multiply-accumulate 9.7 to 12.1.
- **Offline checks of the pieces** (local, seconds each): the target searched with one routine given, at 3M states:
  subtract with the learner's own piece; add and multiply-accumulate with the 8-instruction piece; add with the 4-
  instruction pieces the link allows, with and without the invented subtract.
- **The climbs**: one H100 each for the control and the two invented sets (17 to 27 minutes each); the lifted set on a
  B200 at s3i's budgets (35 s), at 4 times them (329 s), and at 4 times them with modx probed above N (6,684 s), with
  its control on an H100 (1,305 s); the reduce search rerun alone on an H100 (643 s), and its vocabulary built once
  (158 s).
- **The variant**: the official run on one H100 (35 minutes) and 8 seeds on one B200 each (27 to 44 minutes), every
  one certified. Before them, the first launch crashed on all 7 of its seeds (§5); it was stopped, and its records are
  kept in `records/runner_bug1/`.
- **Money** (list prices from run times): about $51 for the evaluator runs (both launches), $22 for the climbs and GPU
  checks, $15 for the learner's CPU containers: about $90 against the $300 soft budget.

### 8. Files and how to rerun

Everything is in `cpu_first/bar3/bridge/`. It imports `library/` (dsl, merge, search's hashing), and, for the
climbs, `integration/lib/inclimb.py` (H1's rows); it changes nothing outside this folder.

| file | what |
|---|---|
| `slices.py` | the digit law book: every law cut to one place, posed when a function of the place |
| `dcm.py`, `dsearch.py` | the counting machine with a place's registers, and bar 3's search on it |
| `dproblems.py` | the candidate problems: bar 3's law book on a place's values, slices, restrictions, projections, near misses |
| `djobs.py`, `dlearner.py`, `dbook.py` | one search job; bar 3's loop per target (rules lp, ct2, all, ct1d); the loop with no target |
| `modal_bridge.py` | the learner on Modal (drivers fan searches out to CPU containers, identical jobs shared by a cache) |
| `lift.py` | compile a program to a table at base 10, check its place loop, compare with s3i's cells |
| `make_libcore_b.py` → `libcore_b.py` | libcore (s3i's library core) with cells from a list |
| `test_libcore_b.py` | libcore_b against libcore, and its machines against each other (CPU, or GPU with CUDA graphs) |
| `climb_b.py`, `modal_climb_b.py` | s3i's climb on an instruction set, exact tables, H1's rows; on a Modal GPU (`--scale` multiplies its budgets, `--modx-above` probes modx on N < x < 2N) |
| `checks/modx_overfit.py` | the lifted climb's modx against the variant's, on x < N, x = N and N < x < 2N |
| `checks/lifted_reduce.py`, `checks/reduce_vocab.py`, `checks/reduce_search.py` | why the lifted climb misses the reduction: the program's check, the search's vocabulary, the search rerun with its rejected goals |
| `make_s3b.py` → `submission_stage3_bridge.py` | the variant: s3i with the invented instruction set |
| `checks/designed_bridge.py` | the variant's prediction path at the truth, on H1's splits |
| `tdcm.py`, `test_tdcm.py`, `bench_tdcm.py` | the counting search in torch (not used) |
| `report.py` | the tables here |
| `records/` | `<rule>_<target>_r1.jsonl` (a round a line; `calls_r1.json` the driver's call), `tiny_*_local.jsonl` (local smoke tests at tiny budgets; `tiny_book_local` is the book mode's, §6), `climb/*.json, *.log` (the climbs; `script_gputest1.log` and `script_gputest2.log` are `test_libcore_b.py` on an H100, the first stopped by the test's own device bug, the second passing), `runner/` (the evaluator's records), `runner_bug1/` (the first launch, which crashed) |

```bash
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.slices
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.dlearner --config tiny_sbb
cd cpu_first && BRIDGE_MAX=18 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_bridge.py --multi --configs lp_sbb,all_sbb --tag r1
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/bridge/modal_bridge.py --fetch
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.report --runs
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.lift
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.test_libcore_b
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_climb_b.py --sets s3i,inv4 --tag c1
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_climb_b.py --sets lifted --gpu b200 --tag x4m --seconds 16200 --scale 4 --modx-above
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/bridge/modal_climb_b.py --script bar3.bridge.checks.reduce_search --tag reduce_search --args 100000000
cd cpu_first && ~/github/sotaku/venv/bin/python bar3/bridge/make_s3b.py
cd cpu_first && ~/github/sotaku/venv/bin/python bar3/bridge/checks/designed_bridge.py N=8
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode official --file bar3/bridge/submission_stage3_bridge.py --out bar3/bridge/records/runner --name submission_stage3_bridge_official2
```
