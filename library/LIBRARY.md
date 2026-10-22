# Library learning: can merging operations discover the hand-written skeleton?

*Written 2026-10-21. Everything here is in `cpu_first/library/`; records in `library/records/`. Nothing committed,
nothing submitted, no certified file touched.*

## In plain words

<!-- PLAIN -->
**The question.** Our best H1 model ([WIDE_ANCHORED.md](../wider_menus/WIDE_ANCHORED.md)) learns its digit tables and
its wiring. But the wiring plugs into a skeleton I wrote by hand:
- multi-digit subtraction as a ripple of the subtraction table;
- a *link*: subtract N, and keep the result only if nothing was borrowed;
- a *chain* of links;
- a *division step*: bring down the next digit, run the chain;
- a loop of steps;
- rows of digit products for the multiply.

A program search over single digit operations cannot find that skeleton directly: x² mod N written out in digit
operations is 22 instructions long, and the search here runs out at 5 or 6. The user's idea: do it like BPE does for
text. Solve small problems first, notice which groups of operations keep appearing in the programs that work, make
each group one new instruction, and search again. Programs get shorter, so longer ones come into reach. That is
library learning (DreamCoder, Stitch; BPE is its simplest, greedy form).

**A tiny example.** "Subtract N from x, digit by digit" is 5 primitive instructions: `LDA.X CLC P[SBB.N INC]`
(load x, clear the borrow, then at each of the P places subtract N's digit with borrow and step to the next place). The
searches for subtraction, comparison and negation all found programs like it. BPE noticed that `SBB.N INC` ("subtract
here, move on") recurs and made it one instruction, then `P[…]` of it another: the ripple. With the ripple, "x mod N
for x < 2N" (load x, keep a copy, subtract N, put the copy back if it went negative) is 4 instructions in the new
vocabulary instead of 7 primitives. The search found it in 5 seconds. Primitive search never gets to 7.

**What the merges discovered.** Every row below was found by search, from the same 32 primitives, level by level.
The first seven were at the budget where primitive search alone stops at 5 instructions. The multiply needed the
bigger budget (where it stops at 6) and loops over bodies of cost 3. The last two rows needed solved tasks reused as
routines (§5):

| new instruction | in primitives | the hand-written skeleton's part | found by |
|---|---|---|---|
| `SBB.N INC`, then `P[SBB.N INC]` | at every place: subtract N's digit with borrow, step | the ripple: multi-digit subtraction | BPE, level 0 |
| `CLC P[SBB.N INC]` | the same, with its borrow cleared first | the ripple as a self-contained instruction | MDL, level 0 |
| `STA.S CLC P[SBB.N INC] SEL.S` | keep a copy, subtract N, restore the copy if it borrowed | the restoring link, exactly | MDL (half-price library), level 1 |
| `STA.S P[SBB.N INC] SEL.S` | the same without the borrow reset | the link, with the gate's job done by the borrow (below) | BPE, level 1 |
| `B[link]` | B links in a row | the chain of links (the skeleton's 9) | both, level 2 |
| `BRG.X B[link]` | bring the next digit in, run the chain | one long-division step | both, level 2 |
| (a program, not merged) `CLA P[BRG.X B[STA.S CLC P[SBB.N INC] SEL.S]]` | P steps from an empty register | the reduction loop (the skeleton's 8 steps); token for token the hand-written reference | MDL (half price), level 3; BPE's has the reset moved |
| `H2[x] = CLA CLC P[x INC]` | a new loop-like construct: clear, then run cell x at every place | none: a "scan" the skeleton does not have | MDL with holes, level 0 |
| `LDH.Y CLK P[MUL.X INC]` | load a multiplier digit, clear the carry, multiply-accumulate at every place | a row of digit products | BPE at a bigger budget, level 1; MDL with holes found the rows through H2 at the small one |
| (a program) `CLK CLA W[LDH.Y P[MUL.X INC] INC SHL.X]` | one row per multiplier digit, X shifted up each row | the multiply loop: schoolbook, but walking the multiplier with the pointer instead of shifting it | BPE with loop bodies up to cost 3, level 4 |
| `T.mul STA.X T.reduce` (routines) | multiply, move the product to X, reduce | x·y mod N, the shape of H1's step (20 primitives) | solved tasks as routines, level 5, 11 s |
| `LDA.X STA.Y T.mul STA.X T.reduce` | copy x to Y, multiply, reduce: 22 primitives | x² mod N, the whole H1 step; it gets T = 64 on both released ladders | routines, level 5, 23 min |

**Does it match the skeleton?** For the reduction, yes: it is long division by repeated subtraction, the algorithm
the certified model chose. The differences are small, and they say something:
- **BPE's link has no borrow reset.** A link that goes below zero leaves the borrow set, so every later link in the
  same chain also goes below zero and restores its copy: the chain stops by itself. The skeleton does that job with
  its commit gate. MDL's link keeps the reset, as the skeleton does.
- **It runs B links a step where B − 1 suffice** (the last never fires), **and P steps from an empty register**, not W
  steps from the dividend's top W digits.
- **For the multiply,** the row is the skeleton's row of digit products, with a multiply-accumulate cell (§1) where the
  skeleton places products and sums them with the subtraction table.

**How much further search reaches.**
- **Primitive-only search** at 3 million stored machine states per level (20–40 s a task on 2 CPUs) finds every program
  up to 5 instructions: 12 of the 26 tasks.
- **At 16 million states**, with cost 6 streamed (6–13 minutes a task), it finds the three multiply rows too: 15 tasks.
- Each extra instruction costs about 22–25 times more states. A 7-instruction program would need about 10^8 stored
  states (≈ 100 GB), the 10-instruction reduction about 10^12, and x² mod N (22) about 10^28.
- **With the merged vocabulary, at the small budget,** each level's search looks at programs of only 3–5
  instructions. But those expand to 7–11 primitives: the link (7) in 3–5 s, the chain (8) in 1–4 s, the step (9) in
  4–9 s, the whole reduction loop (10 or 11) in 3–17 s. That is 18 tasks with BPE and 21 with MDL with holes,
  against 12 for primitives alone.
- **With solved tasks reused as routines** (§5.5), the modular multiplication, 20 primitives, is a 3-token search,
  found in 11 s. Run at H1's size on the released ladders, it gets every rung right up to T = 64, on seen and unseen
  moduli. x² mod N itself (22 primitives, about 10^28 stored states for primitive search) is a 5-token search over 80
  instructions, 23 minutes, and gets T = 64 on both ladders too.

**What else it finds.**
- **BPE merges a lot of junk:**
  - both orders of commuting setup instructions (`LDA.X CLC` and `CLC LDA.X`, `CLA CLC` and `CLC CLA`);
  - fragments that fit one task only ("keep a copy, clear, reload x, subtract").

  They widen every later search, from 32 instructions to 54.
- **MDL never merges junk, but at full price it stalls a level earlier.** With each concept used by two tasks, a
  3-instruction merge only breaks even, so the link is never adopted. With definitions at half price, MDL's whole
  library is 8 entries, and they are the skeleton's: ripple, link, chain, step.
- **MDL with holes invents a construct the skeleton does not have:** a "scan" that runs any cell at every place,
  learned from subtraction and addition programs. It then works with the multiply cell, which no program in its
  corpus had used. That was the only merge that generalized beyond what it was learned from.
- **A merge can also block what comes after.** MDL with holes chose a link that writes its result back into X. That is
  fine for the link tasks, but no division step can use it, because a step brings its digits in from X. So that run
  never found the reduction loop.
- **A bigger vocabulary also hides programs that do not use it.** Each merge widens every search. After level 0 a
  cost-6 program built only from primitives, like the multiply row, is out of reach unless the search also tries the
  primitives alone. The in-run prototype does that.

**What it took.** The climb works only when:
1. **The instruction set has generic loops.** Without `P[…]`, a ripple is a different program at every width.
2. **Each concept is used by at least two tasks.** With one task per concept, every method stalled after one level
   (§5.1).
3. **The search reaches each half's first stepping stone.** The subtraction half's is 5 primitives; the multiply's row
   is 6, past the small budget unless an abstraction (the scan) shortens it.
4. **The search can build the loops the next level needs.** The multiply's loop body is a row and two shifts, one item
   more than the search first allowed. With bodies up to cost 3 it found a multiply (one the skeleton does not have:
   it walks the multiplier with the pointer instead of shifting it, §5.3).
5. **Whole solved programs can be reused as one instruction.** At the top, composition reuses each solved program
   once ("square, then reduce"). A merge rule that needs two uses never makes those instructions, so with BPE or MDL
   alone the climb stops below x² mod N. Routines (one per solved task) do it. They need a calling convention: a
   routine restores the pointer. Without it, a multiply that leaves the pointer moved broke the reduction after it on
   8-digit moduli.

**Hand-written vs. discovered.** The skeleton's pieces can be discovered from digit cells at a pointer, moves,
shifts and loops, by searching and compressing working programs:
- the ripple, the link, the chain, the step and the reduction loop;
- the multiply row and a multiply;
- their composition into x² mod N, which is exact at H1's size.

What has to be written instead:
- the curriculum: which sub-problems to solve, with each concept used twice;
- a few rules of the climb: routines for solved tasks, the pointer-restoring call, loop bodies big enough.

The skeleton said how to compute; the curriculum says which smaller problems to solve first. Run offline and then
built into a submission, the merges would count as hand-written: they would be the skeleton again. Run in one
training run (§6), the curriculum is a set of laws on the rows' own numbers, the same laws the best model already
uses as lessons. A prototype does that, with no answers. It finds the ripple, the link, the adder, the chain, the
division step, the multiply row and the row added onto a number (§6). It is harder than the offline climb: the laws
give one task per concept, so nothing is used twice, and which pieces get merged comes down to ties among single
programs' spellings. It did not reach the multiply or x² mod N in the time given (§6.2).

**Learned cells.** With the subtraction table learned by the anchors' recipe (the order of the numbers, no answers),
the climb is the same as with the exact table, once the table is exact. A table that still has wrong entries stops
the climb almost at once:
- one wrong digit of 200, or six, and no subtraction task passes, so there is no link to find;
- a table with 176 of 200 right, every borrow correct but two digit rows swapped, passes every comparison and fails
  everything else.

So library learning has to come after the cells are finished, which the anchors' recipe does in 100–275 steps (§7).
<!-- /PLAIN -->

# Details

## 1. The machine

`dsl.py`. A setting is (B, W): base B, operands of W digits, registers of P = 2W places, digits lowest first.

- **State**: five registers A (accumulator, where answers are read), X, Y, N (inputs), S (scratch); a pointer p
  shared by every probe (control never depends on data); a borrow/carry bit c; a multiplier digit h; a multiply carry
  digit k.
- **A fresh machine starts dirty**: p = 0 and the inputs are set, but A, S, c, h and k are random (and so is any input
  a task does not use). Every program must set up what it reads, so a piece cut out of one program works wherever it
  is pasted. With a clean start the merges come out context-dependent (a ripple without its borrow reset).
- **32 primitive instructions**, one token each:

| group | tokens | what they do |
|---|---|---|
| cells (at the pointer) | `SBB.r` (r = X, Y, N, S) | (A[p], c) ← A[p] − r[p] − c: the subtraction table |
| | `ADC.r` | (A[p], c) ← A[p] + r[p] + c: the addition table |
| | `MUL.r` (r = X, Y, N) | (A[p], k) ← A[p] + r[p]·h + k: the multiply table, low digit and high digit |
| gate | `SEL.S` | A ← S on every probe whose borrow is 1: keep or discard |
| moves | `LDA.r`, `STA.r`, `CLA` | A ← r, r ← A, A ← 0 |
| flags | `CLC`, `CLK`, `LDH.r` | c ← 0, k ← 0, h ← r[p] |
| pointer | `INC` | p ← p + 1 (mod P) |
| shifts | `SHL.r`, `SHR.r` (r = A, X, Y), `BRG.X` | one place up or down; BRG.X brings X's top digit into A's bottom (a two-register shift) |
| loops | `P[…]`, `B[…]`, `W[…]` | repeat the body P, B or W times |

- **Cost** is the token count in prefix form with the loop's arity in its header: a token costs 1, a loop 1 plus its
  body. `P[SBB.N INC]` costs 3.
- The cells are tables (`Cells`): exact by default; §7 swaps in learned ones.

## 2. The tasks

`tasks.py`. 26 tiny sub-problems of x² mod N, each defined on any (B, W):

| concept | tasks (what A or c must hold) | shortest program found, in primitives |
|---|---|---|
| copy, shifts | copy (X), shl (X·B), shr (X div B) | 1–2 |
| ripple subtraction | sub (X − N), subY (X − Y), lt (c = [X < N]), ltY, neg (0 − X), negN | 5 |
| ripple addition | add (X + N), addY (X + Y), dbl (2X) | 5 |
| multiply row | mrow (X·Y₀), mrowN (N·X₀), mac (N + X·Y₀) | 6 |
| restoring link | csub (X mod N, X < 2N), csubY (Y mod N, Y < 2N) | 7 |
| chain of links | modB (X mod N, X < B·N), modBY | 8 |
| division step | step0 (X's top digit mod N), step ((Y·B + X's top digit) mod N, Y < N) | 9 |
| reduction | reduce (X mod N, any X of P digits) | 10 |
| multiply | mul (X·Y), sq (X²) | 9, 12 |
| the H1 step | mulmod (X·Y mod N), sqmod (X² mod N) | 20, 22 |

Each concept is used by at least two tasks (the same operation on another register, or inside a bigger task). This
matters: a merge has to be used more than once to pay for itself (§5.1). The first task set (record r1) had one task
per concept, and every method stalled after one level.

## 3. The search

`search.py`, `jobs.py`.

- **Probes.** Each search runs on 8 probes in each of two settings, (B, W) = (2, 3) and (3, 2): binary with 6-place
  registers and ternary with 4-place ones. Loop counts differ between them, so a program must loop by P, B or W, not
  by a fixed count, and it must not depend on the base.
- **Breadth-first by cost.** Level d holds every distinct machine state, over all 16 probes at once, that some program
  of cost d reaches and no cheaper program does. Level d is built by applying every item of cost c to every state of
  level d − c.
- **Items.** The vocabulary's tokens, plus loops over bodies of cost ≤ 2 (of at most 3 items; cost ≤ 3 in §5.3's
  continuation). Items are first merged
  by what they do to 24 random machines per setting; only the cheapest spelling of each function is searched.
- **Every minimal program is kept.** Every minimal-cost parent of every state is recorded, so the corpus gets all of a
  task's minimal programs, not one (up to 200).
- **Verification.** A program that fits the probes is run on 200 fresh inputs at eight more settings, (2, 2) to
  (10, 3). A cost whose fitting programs all fail is noted as aliases, and the search goes on.
- **The budget.** 3 million stored states per level (16 million for the "big" runs) and 40 minutes. When a level
  would pass the budget, it is finished without storing (goal checks only) and the search stops after it.
  Primitive levels grow 22–31× per cost (1, 32, 789, 18,807, 430,318, 9.5 million distinct states), so this budget
  reaches cost 5 with primitives, and cost 6 with the big one.

## 4. The merges

`merge.py`. The corpus is every verified minimal program of every solved task, in the vocabulary of the moment.

- **BPE** (`bpe`): the most frequent adjacent pair of tokens becomes one new token, again and again, up to 12 per
  level, while the best pair is used by at least two tasks. A pair is two neighbouring tokens in one item sequence (top
  level or a loop's body), or a loop whose body is one token. A task counts a pair as often as its best program uses
  it (any minimal program will do).
- **MDL** (`mdl`): any contiguous run of items (tokens and whole loops) can become one token. Each round adopts the run
  that most shortens the description: the shortest program of every task plus the library's definitions, at
  log₂(vocabulary size) bits a token. It stops when nothing shortens it. A run of length ℓ used n times saves
  n(ℓ − 1) tokens and costs ℓ + 1 to define, plus a little for the larger vocabulary.
- **MDL with holes** (`mdl(holes=True)`): a run may leave one token out as an argument. That makes a new loop-like
  construct, a header that takes one token, such as H[x] = `CLC P[x INC]`. A use costs 2, and the search can fill the
  hole with any token.
- **MDL, half-price library** (`lib_weight = 0.5`): definitions count half. This is the kind of knob DreamCoder's
  structure penalty is.

## 5. Results

Records: `records/<run>.jsonl`, one line per level, holding every search's record (the programs found, the
verification, the level sizes), the merges and the library. `python -m library.summarize records/<run>.jsonl` prints
one; `python -m library.report` and `python -m library.compare` make the tables below.

### 5.0 Control: the merges on a complete corpus

First, what the merges do when the corpus already holds every task's program: the hand-written reference programs of
`selftest.py`, one per task (`records/reference_corpus_merges.json`).
- **BPE rebuilds the whole skeleton in 25 merges**, in this order:
  - the ripple step, the ripple, the ripple with its borrow reset;
  - the link, the chain;
  - the multiply row, the division step;
  - the multiply step (`LDH.Y CLK P[MUL.X INC] SHL.X SHR.Y`), the multiply loop, the multiplier;
  - the reduction loop, the reducer.

  x² mod N becomes 2 tokens.
- **MDL does it in 8 merges**: the link, the multiplier, the step, the ripple, the reducer, `LDA.X CLC`, the multiply
  row and `LDA.X CLC P[SBB.Y INC]`. x² mod N becomes `LDA.X STA.Y M2 STA.X M5`.
- **MDL with holes** makes the same pieces, plus `H[x] = LDA.X CLC P[x INC]`.

So the merges are not the bottleneck: given the programs, compression finds the skeleton's parts. Everything below is
about getting those programs by search, one level at a time.

### 5.1 Where it stalls: single-use patterns, and MDL's break-even

**r1: one task per concept** (17 tasks: copy, shl, shr, sub, add, lt, neg, dbl, mrow, mac, csub, modB, step, reduce,
mul, sq, sqmod). Level 0, primitives alone at the 3M budget, solved the 8 tasks of cost ≤ 5. Then every method
stalled:
- **BPE** merged only `SBB.N INC`. Each task's minimal programs come in several spellings, and averaging a pair's
  count over them left every other pair below two tasks.
- **MDL** merged `CLC LDA.X P[SBB.N INC]` and `CLC P[ADC.X INC]`. That found the link `LDA.X STA.S M SEL.S` at level 1,
  and then nothing more paid for itself.
- **MDL with holes** made `H[x] = CLC LDA.X P[x INC]`, found the link at level 1, and stalled the same way.

A merge pays only if it is used more than once, and here each concept had one task.

**r2 and r3: 26 tasks, each concept in at least two.** With the variants of §2, level 0 solves 12 tasks.
- **MDL** merges four self-contained pieces: `CLC LDA.X`, the ripple `CLC P[SBB.N INC]`, `CLC P[ADC.X INC]` and
  `CLC LDA.X P[SBB.Y INC]`. Level 1 finds both links (csub and csubY: `LDA.X STA.S M SEL.S`, 3 s, 0.7M states).
  Then it stalls: the link `STA.S M SEL.S` saves 2 tokens in each of 2 programs and costs 3 + 1 to define. That is
  break-even, and the bigger vocabulary tips it negative.
- **MDL with holes** picks `H[x] = CLC LDA.X P[x INC]` again. That beats the general `CLC P[x INC]` by one token on this
  corpus: 6 tasks load x first, 3 do not. Greedy compression cannot see that the general one is what the multiply rows
  will need. It finds one link and stalls.
- **BPE** keeps climbing (§5.2): its threshold is "two tasks use it", not "it pays for itself".

### 5.2 BPE climbs the whole reduction (r3, the 3M budget)

`records/bpe_r3.jsonl`, at the same budget as primitive-only search, level by level:

| level | solved (program in the vocabulary of the moment = in primitives) | primitives | states | s |
|---|---|---|---|---|
| 0 | 12 tasks, e.g. sub `CLC LDA.X P[SBB.N INC]`, neg `CLC CLA P[SBB.X INC]` | 1–5 | 33 – 14.8M | 0–27 |
| 1 | csub `B0.1 STA.S B0.6 SEL.S` = `LDA.X CLC STA.S P[SBB.N INC] SEL.S`; csubY | 7 | 1.2M, 48.8M | 5, 68 |
| 2 | modB `B0.1 B[B1.5]` = `LDA.X CLC B[STA.S P[SBB.N INC] SEL.S]`; modBY | 8 | 60,056 | 4 |
| 2 | step `B1.3 BRG.X B[B1.5]` = `CLC LDA.Y BRG.X B[STA.S P[SBB.N INC] SEL.S]`; step0 | 9 | 2.0M | 9 |
| 3 | reduce `CLA P[CLC B2.2]` = `CLA P[CLC BRG.X B[STA.S P[SBB.N INC] SEL.S]]` | 10 | 2.4M | 17 |

The merges that made it possible, in order:

| level | merge | = in primitives | what it is |
|---|---|---|---|
| 0 | B0.3 `SBB.N INC`, B0.6 `P[B0.3]` | `P[SBB.N INC]` | the ripple subtraction of N |
| 0 | B0.4, B0.5, B0.9 | `CLC P[ADC.X INC]` | the ripple addition of X, with its borrow reset |
| 0 | B0.1 `LDA.X CLC`, B0.2 `CLC LDA.X`, B0.8 `CLC CLA`, B0.10 `CLA CLC` | | junk: both orders of commuting setups |
| 0 | B0.11, B0.12 | `CLC LDA.X P[SBB.N INC]`, `LDA.X CLC P[SBB.N INC]` | x − N, twice (the two spellings) |
| 1 | B1.1, **B1.5** `B1.1 SEL.S` | `STA.S P[SBB.N INC] SEL.S` | **the restoring link**, without a borrow reset |
| 1 | B1.2, B1.4 | `STA.S CLC LDA.X P[SBB.N INC] SEL.S` | a link that reloads x: only right as a first link |
| 1 | B1.3, B1.6–B1.8 | `CLC LDA.Y`, `P[SBB.Y INC]`, … | the same for Y |
| 2 | **B2.1** `B[B1.5]` | `B[STA.S P[SBB.N INC] SEL.S]` | **the chain** |
| 2 | **B2.2** `BRG.X B2.1` | `BRG.X B[STA.S P[SBB.N INC] SEL.S]` | **one long-division step** |

- **The reduction loop is found as a program but not merged.** One task uses it, so it never passes "two tasks".
- **The link without a reset works because of the borrow.** In `B[STA.S P[SBB.N INC] SEL.S]`, a link that goes below
  zero restores its copy but leaves the borrow set. The next link then subtracts N plus 1 from a number below N, borrows
  again, and restores again: once one link fails, all later links fail, and the chain has stopped. The reduction loop
  clears the borrow once per step (`P[CLC BRG.X …]`). That is the commit gate's job in the skeleton, done by the
  subtraction's own borrow.
- **Verification caught the near-misses.**
  - `CLA P[BRG.X B[…] CLC]` clears the borrow at the end of each step instead of the start, so a stray borrow at the
    start breaks the first step: 98% right.
  - `LDA.N P[CLC BRG.X B[…]]` starts the register at N instead of 0: 92.5%.
  - `LDA.X CLC P[P[B[…]]]` never brings a digit in: 22%.
- **Unsolved at 3M:** mrow, mrowN, mac, and so mul, sq, mulmod, sqmod. Their first stepping stone, a multiply row
  (`CLA LDH.Y CLK P[MUL.X INC]`, 6 primitives), is past cost 5, and nothing in the subtraction corpus shortens it.
  Level 4 repeated level 3's searches with the same vocabulary and found nothing. Since then the loop stops when a
  level adds no merge.

### 5.3 The 16M budget: the multiply row, and a limit of the search (r5, then r8)

`records/bpe_big_r5.jsonl`: the same climb with 16 million stored states a level (32 GB containers), BPE.

**Level 0, primitives alone, now reaches cost 6.**
- Every distinct state up to cost 5 is stored (9.5 million at level 5), and cost 6 is streamed (about 300 million
  candidates, 6–13 minutes a task).
- That adds the three multiply rows: mrow `LDH.Y CLK CLA P[MUL.X INC]`, mrowN, and mac `LDH.Y CLK LDA.N P[MUL.X INC]`.
  So 15 of the 26 tasks are solved.
- Nothing of cost 7 or more is in reach. The link costs 7: it would need about 2·10^8 stored states, roughly 100 GB.

**The levels, with the library:**

| level | solved | key merges |
|---|---|---|
| 0 | the 12 of r3, plus mrow, mrowN, mac | as r3, plus `MUL.X INC`, `LDH.Y CLK`, `CLK LDH.Y` |
| 1 | csub, csubY (7 primitives, 6 and 124 s) | **the multiply-accumulate row** B1.11 `LDH.Y CLK P[MUL.X INC]` (and B1.12, its other spelling); the link `STA.S P[SBB.N INC] SEL.S`; the ripple with its reset `CLC P[SBB.N INC]` |
| 2 | modB, modBY (8, 3–4 s), step0, step (9, 11–15 s) | the chain, the step |
| 3 | reduce `CLC CLA P[BRG.X B[…] CLC]` (11, 30 s) | none |

- **The multiply was not found, because of the search, not the merges.** With the row merged, the multiply is
  `CLA W[B1.11 SHL.X SHR.Y]`: 5 tokens, well within the budget. But its loop body (row, shift, shift) costs 3, and the
  search builds loops only over bodies of cost ≤ 2 (§3). That limit was set for the primitive level, where cost-3 bodies
  are about 100,000 items.
- **r8 continues r5 from level 4 with bodies up to cost 3.** With 61 tokens that is 245,000 more loop items, enumerated
  in about 3 minutes. They only apply to the shallowest levels, so the search costs a few minutes more per task.

**r8, level 4: the multiply** (`records/bpe_big_mb3_r8.jsonl`). It is found at cost 5, after 6 minutes of
enumerating items and 14 of search:

    mul = B0.11 W[B1.10 INC SHL.X] = CLK CLA W[LDH.Y P[MUL.X INC] INC SHL.X]

It is a schoolbook multiply, but not the reference one (`CLA W[LDH.Y CLK P[MUL.X INC] SHL.X SHR.Y]`).
- **It walks the multiplier with the pointer.** Instead of shifting Y down so the next multiplier digit is at place 0,
  it moves the pointer up one place per row (`INC`), so `LDH.Y` reads Y's next digit where it is. Each row's scan then
  starts at that place and wraps around the register. That is harmless: X's low places are zeros after the shifts,
  and the carry out of the top is 0, so the wrapped part adds nothing.
- **It clears the multiply carry once, not per row**, for the same reason: every row ends with a carry of 0.
- **Verified** exact on 300 inputs at (2, 2), (2, 4), (3, 3), (4, 2), (5, 2), (7, 2), (10, 2), (10, 3), (10, 4) and H1's
  (10, 8).
- **It is one of 27 minimal multiplies**, all verified. Among them:
  - the hand-written reference `CLA W[LDH.Y CLK P[MUL.X INC] SHL.X SHR.Y]`;
  - versions that loop P times instead of W (the extra rows multiply by Y's zero digits);
  - `CLA W[LDH.Y CLK P[MUL.X INC] SHL.X MUL.X INC]`, whose stray `MUL.X` lands on a place where X is 0;
  - `LDA.Y CLC P[LDH.Y CLK P[MUL.X INC] SHL.X SBB.Y INC]`, which starts the accumulator at Y and cancels it by
    subtracting one digit of Y per row: a ripple subtraction spread across the multiply's loop.

  All of these were checked exact up to (10, 8) as well.

sq, mulmod and sqmod are not found at level 4. Each needs the multiply as one piece: sq is `LDA.X STA.Y` and then the
multiply, 7 tokens here. And BPE does not merge the multiply, because one task uses it (§5.5).


### 5.4 MDL with a half-price library, with and without holes (r6, the 3M budget)

`records/mdl_half_r6.jsonl`, `records/mdlh_half_r6.jsonl`. Definitions count half (`lib_weight = 0.5`), so a
3-token chunk used twice now pays.

- **MDL, half price.**
  - Level 0 merges the self-contained ripples `CLC P[SBB.N INC]` and `CLC P[ADC.X INC]`, the whole subtractions
    `CLC LDA.X P[SBB.Y INC]` and `LDA.X CLC P[SBB.N INC]`, and negation `CLA CLC P[SBB.X INC]`.
  - Level 1 finds both links and adopts **`STA.S CLC P[SBB.N INC] SEL.S`: the skeleton's link exactly**, borrow
    reset included. Unlike BPE's, it never adopted a junk pair.
  - Level 2 finds modB `LDA.X B[link]`, modBY, step0 `CLA BRG.X B[link]` and step `LDA.Y BRG.X B[link]` in 1–4 s
    (the vocabulary is only 38 instructions). It adopts the chain `B[link]` and the step `BRG.X B[link]`.
  - **Level 3 finds reduce = `CLA P[M2.2]` = `CLA P[BRG.X B[STA.S CLC P[SBB.N INC] SEL.S]]`: the hand-written
    reference reduction (`selftest.py`), token for token.** It took 2.7 s and 38,099 states, from a library of 8
    entries.
  - Then it stops: the multiply loop needs a cost-3 body (§5.3), and nothing else is left in reach.
- **MDL with holes, half price.**
  - Level 0 makes two abstractions: `H1[x] = CLC LDA.X P[x INC]` and **`H2[x] = CLA CLC P[x INC]`** ("clear the
    accumulator and the borrow, then run cell x at every place").
  - Level 1 uses H2 with the multiply cell, which no program in the corpus had used: mrow is
    `LDH.Y CLK H2[MUL.X]` (4 tokens, 7 primitives), found in 2.3 s over 548,000 states, and mrowN likewise.
  - At this budget primitive search cannot reach a multiply row (6 primitives), and BPE and plain MDL never can,
    because nothing in their corpus mentions MUL. The scan abstraction transferred to a new cell. This is the one place
    where an abstraction generalized beyond what it was learned from.
  - It also merges the rows as `CLK LDH.Y H2[MUL.X]` and `CLK LDH.X H2[MUL.N]`, and finds csubY by a detour: copy Y
    into X, then use the "load X" ripple (`LDA.Y STA.S STA.X M0.1 SEL.S`).
  - Level 2 finds mac (the row on top of N), both chains and both steps. It adopts a chain and a step built on its
    first link, `STA.S CLC LDA.X P[SBB.N INC] SEL.S`, which reloads X and so needs X to hold the running value; its
    chain writes back with `STA.X`.
  - Level 3 finds nothing. The multiply loop needs a cost-3 body (§5.3), and its reduction is blocked by its own link,
    which writes the result back into X: a step brings its digit in from X, so a chain that overwrites X cannot sit in
    a step. It stops with 21 tasks, the most at the small budget.

### 5.5 Composition: solved tasks as routines (r11)

**The last steps need something BPE does not do.** x² mod N is "square, then reduce", and the square is "copy x to Y,
then multiply". Each of those is a whole solved program used again as one piece. Until the bigger task is solved,
each piece is used by one task only, so a merge rule that needs two uses never makes it an instruction. And without
the instruction, the bigger task is out of reach: sqmod is 22 primitives.

**The variant.** Every newly solved task's simplest program becomes one routine, `T.<task>`, and every program
holding it is rewritten. That is one entry per task, a bounded library growth, and what a person does with a solved
sub-problem. r11 continues r8's last level this way: all the solved tasks become routines first, then the climb goes
on.

**Routines need a calling convention.** Composing found pieces by hand first showed why:
- `LDA.X STA.Y mul STA.X reduce`, with r8's multiply and MDL's reduction, is exact on every seen-moduli rung up to
  T = 64. But it gets 708/768 at T = 1 on the unseen moduli.
- The multiply walks the multiplier with the pointer and leaves it at place W. The reduction's ripple then starts at
  place W and wraps around the register.
- With up to 7-digit moduli the remainder never reaches place W, so the misplaced ripple still gives the right
  borrow. With 8-digit moduli it does not.

A routine call therefore restores the pointer it was called with (callee-saved, as real instruction sets do). No
task's answer depends on the pointer, so every routine stays right. With that rule the same composition is exact on
the unseen moduli too (768/768 at T = 1, 2 and 64). Without it, the search would have to find glue (`W[INC]`, which
brings the pointer back) and pay 2 more tokens.

**r11** (`records/bpe_big_routines_r11.jsonl`: r8's levels 0–4 copied, the routines, then the new levels):
- **Routines.** At the resume, the solved tasks' simplest programs become routines, among them `T.mul` (r8's
  pointer-walking multiply) and `T.reduce` (`CLA P[CLC BRG.X B[STA.S P[SBB.N INC] SEL.S]]`, the simplest of BPE's reductions
  from level 3). 19 tasks become routines (the others' programs were already one token), so the vocabulary is
  61 + 19 = 80 instructions.
- **Level 5** finds:
  - **mulmod = `T.mul STA.X T.reduce`** in 11 s: multiply, move the product to X, reduce it. That is 3 tokens, 20
    primitives: the whole modular multiplication.
  - **It passes the end-to-end gate.** Run in base 10 at H1's size (W = 8, 16-place registers), with x in both X and
    Y, T times, it gets every rung of both released ladders right, 768/768 from T = 1 to T = 64, on seen and unseen
    moduli (`records/bpe_big_routines_r11_mulmod_certify.json`). This checks the found algorithm with exact cells; it
    is not a submission (no learned reader, the T loop in Python).
  - **sq = `LDA.X CLC STA.Y T.mul`**: copy x to Y, multiply.

  - **sqmod = `LDA.X STA.Y T.mul STA.X T.reduce`**: copy x to Y, multiply, move the product to X, reduce. That is 5
    tokens, 22 primitives:

        LDA.X STA.Y CLK CLA W[LDH.Y P[MUL.X INC] INC SHL.X] STA.X CLA P[CLC BRG.X B[STA.S P[SBB.N INC] SEL.S]]

    It is the longest search of the study: 80 instructions, 5.65 million states stored at level 4, 304 million streamed
    at level 5, 23 minutes. Four programs pass verification.
  - BPE then merges the composition itself: `T.mul STA.X` and `T.mul STA.X T.reduce` are now used by two tasks
    (mulmod and sqmod).
- **The end-to-end gate.** `python -m library.certify records/bpe_big_routines_r11.jsonl` runs that x² mod N in base 10
  at H1's size on every prompt of the released ladders, T times. **T = 64 on both ladders, every rung 768/768**, on
  the training pool's moduli and on the unseen ones, 8-digit moduli included
  (`records/bpe_big_routines_r11_certify.json`). As with mulmod, this checks the found algorithm with exact cells. It
  is not a submission.

**So, from 32 primitives to x² mod N.** The chain r5 → r8 → r11 solves all 26 tasks, and every program was found by
search at a cost of at most 5 in its own vocabulary:
- level 0: primitives alone (≤ 6, the 16M budget);
- levels 1–3: BPE's merges (ripple, link, chain, step; the multiply row);
- level 4: loop bodies of cost 3 (the multiply);
- level 5: routines (sq, mulmod, sqmod).

What was not discovered: the instruction set, the curriculum of 26 tasks, the routine rule and its calling
convention, the search's budget and body size, and the loop over T.

### 5.6 Primitive search against search with a library

**Per run** (`python -m library.compare --summary …`; `bpe_big_routines_r11` holds the whole chain r5 → r8 → r11):

| run | levels | tasks solved (of 26; 17 in r1) | longest program found (primitives) | library entries | distinct states searched | search hours (2-CPU containers) |
|---|---|---|---|---|---|---|
| bpe_r1 | 2 | 8 | 5 (sub) | 1 | 0.35G | 0.2 |
| mdl_r1 | 3 | 9 | 8 (csub) | 2 | 0.49G | 0.3 |
| mdlh_r1 | 3 | 9 | 8 (csub) | 1 | 0.44G | 0.2 |
| bpe_r3 | 5 | 19 | 11 (reduce) | 22 | 2.47G | 1.4 |
| mdl_r3 | 3 | 14 | 7 (csubY) | 4 | 0.87G | 0.4 |
| mdlh_r3 | 3 | 13 | 8 (csub) | 1 | 0.72G | 0.3 |
| mdl_half_r6 | 4 | 19 | 10 (reduce) | 8 | 1.12G | 0.7 |
| mdlh_half_r6 | 4 | 21 | 11 (step0) | 10 | 0.83G | 0.6 |
| bpe_big_r5 | 4 | 22 | 11 (reduce) | 29 | 5.49G | 2.6 |
| bpe_big_routines_r11 | 7 | 26 | 22 (sqmod) | 54 | 6.56G | 4.5 |

"Longest program found" is the expansion, in primitives, of the longest program any of the run's searches found.
Each search looked at programs of at most 5 tokens in its own vocabulary.

**Per task** (`python -m library.table_tasks`). For primitives alone: the cost found or "no", with the distinct
states searched and the seconds. For the library: the cheapest search (fewest states) of any run that found it.

| task | shortest program found (primitives) | primitives alone, 3M | primitives alone, 16M | cheapest search with a library: run, level, program, states, s |
|---|---|---|---|---|
| copy | 1 | cost 1, 33, 0 s | cost 1, 33, 0 s | (level 0) |
| shl | 2 | cost 2, 822, 0 s | cost 2, 822, 0 s | (level 0) |
| shr | 2 | cost 2, 822, 0 s | cost 2, 822, 0 s | (level 0) |
| sub | 5 | cost 5, 14.0M, 26 s | cost 5, 9.9M, 28 s | (level 0) |
| subY | 5 | cost 5, 14.0M, 18 s | cost 5, 10.0M, 39 s | (level 0) |
| add | 5 | cost 5, 14.0M, 20 s | cost 5, 9.9M, 34 s | (level 0) |
| addY | 5 | cost 5, 14.0M, 21 s | cost 5, 10.0M, 34 s | (level 0) |
| lt | 5 | cost 5, 14.0M, 15 s | cost 5, 9.9M, 28 s | (level 0) |
| ltY | 5 | cost 5, 14.0M, 23 s | cost 5, 10.0M, 39 s | (level 0) |
| neg | 5 | cost 5, 14.1M, 17 s | cost 5, 10.1M, 41 s | (level 0) |
| negN | 5 | cost 5, 14.8M, 18 s | cost 5, 10.6M, 59 s | (level 0) |
| dbl | 5 | cost 5, 14.1M, 27 s | cost 5, 10.1M, 29 s | (level 0) |
| mrow | 6 | no (14.0M, 39 s) | cost 6, 304.6M, 549 s | mdlh_half_r6 L1: `LDH.Y CLK H2[MUL.X]` (cost 4), 0.5M, 2 s |
| mrowN | 6 | no (14.0M, 26 s) | cost 6, 303.3M, 364 s | mdlh_half_r6 L1: `LDH.X CLK H2[MUL.N]` (cost 4), 0.6M, 4 s |
| mac | 6 | no (13.8M, 23 s) | cost 6, 299.7M, 790 s | mdlh_half_r6 L2: `M1.3 P[ADC.N INC]` (cost 4), 0.7M, 5 s |
| csub | 7 | no (14.0M, 17 s) | no (302.2M, 459 s) | mdlh_half_r6 L1: `LDA.X STA.S M0.1 SEL.S` (cost 4), 0.6M, 3 s |
| csubY | 7 | no (14.5M, 25 s) | no (318.4M, 338 s) | mdl_half_r6 L1: `LDA.Y STA.S M0.2 SEL.S` (cost 4), 0.8M, 5 s |
| modB | 8 | no (14.0M, 32 s) | no (302.1M, 376 s) | mdl_half_r6 L2: `LDA.X B[M1.1]` (cost 3), 30,949, 1 s |
| modBY | 8 | no (14.6M, 17 s) | no (321.2M, 373 s) | mdl_half_r6 L2: `LDA.Y B[M1.1]` (cost 3), 31,735, 1 s |
| step0 | 9 | no (14.5M, 17 s) | no (316.5M, 452 s) | mdl_half_r6 L2: `CLA BRG.X B[M1.1]` (cost 4), 0.8M, 4 s |
| step | 9 | no (14.1M, 16 s) | no (308.6M, 446 s) | mdl_half_r6 L2: `LDA.Y BRG.X B[M1.1]` (cost 4), 0.8M, 4 s |
| reduce | 10 | no (14.5M, 17 s) | no (316.5M, 587 s) | mdl_half_r6 L3: `CLA P[M2.2]` (cost 3), 38,099, 3 s |
| mul | 9 | no (14.0M, 17 s) | no (304.6M, 490 s) | bpe_big_mb3_r8 L4: `B0.11 W[B1.10 INC SHL.X]` (cost 5), 196.3M, 855 s |
| sq | 12 | no (14.1M, 16 s) | no (308.1M, 531 s) | bpe_big_routines_r11 L5: `B0.1 STA.Y T.mul` (cost 3), 0.2M, 11 s |
| mulmod | 20 | no (13.3M, 17 s) | no (284.3M, 480 s) | bpe_big_routines_r11 L5: `T.mul STA.X T.reduce` (cost 3), 0.1M, 11 s |
| sqmod | 22 | no (13.4M, 23 s) | no (285.6M, 351 s) | bpe_big_routines_r11 L5: `LDA.X STA.Y T.mul STA.X T.reduce` (cost 5), 309.4M, 1361 s |

- **The raw space.** Primitive programs of cost ≤ 5 number 2^26.1, ≤ 7 2^36.7, ≤ 10 2^52.8, ≤ 22 2^117
  (`dsl.raw_count`: 32 tokens, 3 loop kinds, at most 3 items a body).
- **The distinct states** a primitive search stores grow 22–31× per cost: 1, 32, 789, 18,807, 430,318, 9.5 million.
- **A library search's raw space** at its own cost 3–4 is 2^16–2^24. The reduction loop that MDL found at cost 3
  (3 tokens) stands for a 10-primitive program in a 2^52.8 space.
- **The budget gap.** Every library search up to the reduction took under a minute. The multiply (14 minutes, with
  245,000 cost-3-body loop items run one at a time in Python) and x² mod N (23 minutes over 80 instructions) are the
  exceptions. Primitive search would need about 10^8 stored states for the 7-primitive link, about 10^12 for the
  10-primitive reduction and about 10^28 for the 22-primitive x² mod N.


## 6. Offline or in one training run (rules 7, 8, 14)

**Offline.** Run the climb before training, on any computer, then write the library into the submission as built-in
instructions. Under rule 7 that is hand-written. The ripple, the link, the chain, the step and the multiply loop would
sit in the forward pass because I put them there. That they were found by compression is history, not something the
evaluator can see. It is the current skeleton again, chosen by a different process. Offline merges buy nothing for
rule 7.

**In the run.** The submission carries:
- the primitive instruction set;
- the interpreter and the search;
- BPE, and the routine rule with its calling convention;
- the laws that define the tasks;
- Occam's choice of helper, the simplest accepted program.

During training it builds every probe from the training rows' own numbers, runs the climb, and uses the program it
ends with as the forward pass, with the learned cells. That is what `inrun.py` prototypes (§6.1). Where each rule
stands:

| rule | in-run library learning | compared with the current best model |
|---|---|---|
| 7. no hard-coded algorithm in the forward pass | the forward pass is a program found in the run; its ripple, link, chain, step and rows are found, not written | less gray: no skeleton and no menus. Still gray: the instruction set (cells at a pointer, one register to subtract from, a bring-in shift, loops by P, B and W) leans toward digit arithmetic. The laws' dependency order (subtraction, then the link, the adder, the chain, the rows, the multiply, the reduction) is a curriculum that walks the search up the same skeleton. The curriculum is where "how" moved. The routine rule and the pointer-restoring call are written too |
| 8. logits inside autograd, unbroken gradient path | the found program would run on one-hot digits with straight-through tables, as the current model's machine does; the choice of program is discrete, like a menu switch, and would need the same treatment (a whole-option mixture over the found programs) | the same as now |
| 9. everything on the GPU | the search, the dedup and the merges must run on GPU tensors; the prototype is numpy on CPU | needs a port |
| 14. task-specific solvers, data inspection | the in-run search is a program search inside the loss, the same kind of thing as the menus' whole-option scoring, but over a much bigger space. The laws are the current model's lessons (x mod N = x, N mod N = 0, x + Q·N mod N = x, the times-table ladder) plus "x + N ≥ N". The ladder for the multiply and the inverse check for the adder are definitions, like the step laws | as gray, in the same places |

### 6.1 The prototype: the corpus from the rows' own numbers

`inrun.py`. Every probe is made from H1 training rows, in base 10:
- At width W, a row gives its modulus's top W digits N′, and x's digits at the same places x′. This only reads digit
  strings, and x < N makes x′ ≤ N′.
- The settings are (10, 1) and (10, 2), with 8 probes each (16 for lt), as before.
- A program is accepted when it passes its laws on fresh rows at widths 1–3, 60 probes each. For x² mod N it must
  also match the full-size T = 1 rows' labels.

The targets come from three places only: facts about the rows, programs the run has already found, and, at the end,
the labels.

| task | what it states (the law) | built with | found offline as |
|---|---|---|---|
| lt | (x′, N′) borrows exactly when the two digit strings differ; (N′, x′) and (x′, x′) never | — | lt |
| modx | x′ mod N′ = x′ (0 if equal); N′ mod N′ = 0 | — | csub |
| add | (x′ + N′) mod N′ = x′, checked by running the found modx on the candidate's output; x′ + N′ ≥ N′, checked with the found lt; x′ + 0 = x′ | lt, modx | add |
| modB | (x′ + q·N′) mod N′ = x′, q = 0…9, the inputs built by q additions | add | modB |
| mrow | the ladder: x′·0 = 0, x′·(d + 1) = x′·d + x′ | add | mrow |
| mac | x′·d + N′ = add(mrow(x′, d), N′) (from r14 on) | mrow, add | mac |
| mul | x′·y′ = y′ additions of x′ (y′ another row's x′) | add | mul |
| sq | x′·x′ with the found mul | mul | sq |
| step0, step | d mod N′ and (r·B + d) mod N′, with the found modB (r = x′ < N′, d a digit) | modB | step0, step |
| reduce | (x′ + Q·N′) mod N′ = x′, Q another row's x′, the inputs built by Q additions | add | reduce |
| sqmod | (x′·x′) mod N′ with the found mul and reduce (r12 and earlier: sq and reduce), then the T = 1 labels at full size | mul, reduce | sqmod |

A task is tried once the programs its probes need have been found. The merges are BPE's, with the threshold lowered
to one task (`min_count = 1`): the laws give one task per concept.

**A local run** (`records/inrun_mini_local.json`): subtraction-only instructions, 350,000 states a level, 33 s on one
CPU. It found, level by level:
- **lt**: `CLC LDA.X P[SBB.N INC]`, the ripple, from the order facts alone;
- **modx**: `LDA.X CLC STA.S P[SBB.N INC] SEL.S`, the link, from "x mod N = x";
- **add**: `LDA.X CLC P[ADC.N INC]`, through the inverse check;
- **modB**: `LDA.X CLC B[STA.S P[SBB.N INC] SEL.S]`, the chain, from inputs the found adder built;
- **step**, from targets the found chain made.

The merges included the link `STA.S P[SBB.N INC] SEL.S`, the chain `B[…]` and the step `BRG.X B[…]`. Every
accepted program is the right one offline, except one: step0 was accepted as `LDA.X CLC BRG.X B[…]`. That is right
when X's lower digits are zero, which they always are in step0's probes, and wrong otherwise (0% on the offline
task). A law only pins a program down on the inputs it draws.

### 6.2 On Modal, with every instruction

**The 3M budget, all 32 primitives** (`records/inrun_bpe_small_r6.jsonl`; this run's reduce still waited for the multiply):

| level | found | = in primitives | right offline? |
|---|---|---|---|
| 0 | lt (from the order facts) | `CLC LDA.X P[SBB.N INC]` | yes |
| 1 | modx (x mod N = x, N mod N = 0) | `LDA.X CLC STA.S P[SBB.N INC] SEL.S`: the link | yes (it is csub on every x < 2N) |
| 2 | add (through the inverse check) | `LDA.X CLC P[ADC.N INC]` | yes |
| 3 | modB (inputs built by the found adder) | `B[LDA.X STA.S CLC LDA.X P[SBB.N INC] SEL.S STA.X]`: a chain whose links write back to X | yes |
| 3 | mrow, mul: not found | the multiply row costs 6, and after level 0 the vocabulary is too big to store level 5 | |
| 4 | step0 | `LDA.X CLC BRG.X B[STA.S P[SBB.N INC] SEL.S]` | no (0%): right only when X's low digits are 0, as in step0's probes |
| 4 | step | `LDA.Y BRG.X CLC B[STA.S P[SBB.N INC] SEL.S]`: one long-division step | yes |
| 5 | nothing: only mrow and mul were left, out of reach at this budget. I stopped the run there; BPE at a one-task threshold never runs out of merges, so it would have gone on. | | |

**The 16M budget**, with loop bodies up to cost 3 and a primitive-only search whenever the library search fails.
§5.3 says why bodies of cost 3. Why the fallback: after level 0 the vocabulary is too big to store level 5, so a
cost-6 program that uses no merge, like the multiply row, would be out of reach, and the library would make search
worse.

**r10, and a law that pins too little.**
- Level 0 found lt; modx (7 primitives) was out of reach after 455 million streamed candidates (14 minutes).
- Level 1 accepted six modx programs. Four of them are `CLC P[LDA.X SBB.N INC] LDA.X STA.S CLA SEL.S`: "x if
  x < N, else 0". That passes "x mod N = x" and "N mod N = 0" on every x ≤ N, which are the only inputs those laws
  draw. It is not x mod N above N. The other two are true csub.
- The run used the first accepted program as the helper for the next law, the adder's (x + N) mod N = x. With the
  alias, that law would reject even the true adder, so I stopped the run at level 2.
- **The fix is Occam's choice.** The helper (and the routine, §5.5) is now the accepted program with the shortest
  expansion into primitives. That is csub here: 7 primitives against the alias's 9.
- r12 is r10 again with that rule (`records/inrun_bpe_fb_r12.jsonl`).

**r12** (`records/inrun_bpe_fb_r12.jsonl`):

| level | found (in primitives) | how | right offline? |
|---|---|---|---|
| 0 | lt: `CLC P[LDA.X SBB.N INC]` (it reloads x at every place; the borrow is still right, and only the borrow is used) | library | yes |
| 1 | modx: csub `LDA.X CLC STA.S P[SBB.N INC] SEL.S` as the helper (Occam), among 6 accepted | library | yes |
| 2 | add: `LDA.X CLC P[ADC.N INC]` (inverse check with csub and lt) | library | yes |
| 3 | modB: `LDA.X CLC B[STA.S P[SBB.N INC] SEL.S]` | library | yes |
| 3 | **mrow: `LDH.Y CLK CLA P[MUL.X INC]`** | **the primitive fallback**: the library search failed; primitives alone reached cost 6 (13 minutes) | yes |
| 4 | step: `LDA.Y BRG.X CLC B[STA.S P[SBB.N INC] SEL.S]`; step0 (again the narrow-domain alias) | library | step yes, step0 no |
| 4 | mul: not found, by the library search or the primitive fallback; reduce: its search was cut when I stopped the run | the pieces they need were not single instructions yet: one-task BPE ties every pair at weight 1, and my tie-break (the pair's spelling) spent the merges on setup orders and on the modx aliases' fragments | |

**r13** (`records/inrun_routines_r13.jsonl`) is r12's state after level 3, continued with solved tasks as routines
(§5.5).
- Level 4 builds on them: step is `LDA.Y BRG.X STA.X T.modB` (bring the digit in, then reduce with the found chain),
  and step0 is `P[T.modB SHR.X]`. Both are right offline.
- reduce was not found. The chain's only routine, `T.modB = LDA.X CLC B[link]`, has "load x" baked in, and a
  division step must keep X as the dividend. The search found only aliases (`B[T.modB STA.X]`, right only on its
  probes). It is the context-baking of MDL with holes again: a routine is a whole task's program, setup included.

**Why the in-run climb is harder than the offline one.** The laws give one task per concept: one comparison, one link
law, one adder, one ladder. So nothing is used twice, and merges are decided by ties among single programs' spellings
(three orders of the row's setup, the aliases' fragments). Offline, the variants (csubY, modBY, mrowN, mac) settled
those ties. r14 continues r12's level 4 with the in-run versions of the same fixes:
- `mac`, the row on top of a number, so the accumulating row has two uses;
- one program per task in the corpus (the simplest), so aliases and spelling variants do not vote;
- ties broken toward the pair that stands for the most primitives;
- 12 merges a level, the merges run once on the resumed corpus, routines.

**r14** (`records/inrun_v2_r14.jsonl`):

- **The merges at the resume** (one program per task, largest pair first) make, among others:
  - the chain with its reset, `CLC B[STA.S P[SBB.N INC] SEL.S]`, which puts the reduction loop at 4 tokens;
  - the row's scan `P[MUL.X INC]`;
  - the adder's ripple.
- **Level 5 finds mac = `LDH.Y CLK LDA.N P[MUL.X INC]`** (right offline) in 9.5 s of search.
- **reduce and mul were still searching after 86 minutes, and I stopped the run.** At this vocabulary, about 80
  instructions with routines, loop bodies up to cost 3 make about a million cost-4 loop items. Each search spends most
  of an hour enumerating them and running them on the shallow levels, op by op in numpy, before it reaches the
  primitive fallback.
- **The multiply would also have met a spelling problem.** mac's simplest program puts `LDA.N` between the row's
  setup and its scan, so the accumulating row `LDH.Y CLK P[MUL.X INC]` is still not one instruction. The multiply's
  loop body (row, shift, shift) would then cost 4.

**Where the in-run version stands.** From the rows' own numbers and the laws, with no answer, it found:
- the ripple (from x < N);
- the restoring link (x mod N = x, N mod N = 0);
- the adder (through (x + N) mod N = x);
- the chain ((x + q·N) mod N = x);
- the division step;
- the multiply row (the ladder);
- the row on top of a number.

It did not get to the multiply, the reduction loop or x² mod N in the time I gave it. The obstacles were not the laws.
They were the thin curriculum (one task per concept, so every merge decision is a tie) and the cost of loops over
bodies of cost 3 in this implementation. The fixes that worked offline (variants of each task, cost-3 bodies,
routines) carry over in principle, but the run needs either more tasks per concept or a faster search.


## 7. Learned cells

`learned.py`. The key step (the climb from primitives to the restoring link and beyond), repeated with a learned
subtraction table instead of the exact one.

**The table.** The anchors' certified recipe (ANCHORS.md §1.3, R2): x < N on every top part, then the column and row
order laws and both bijectivities. No answers, 400 steps of Adam at batch 64 on random H1 training rows, base 10
(`anchors/core.py`, run as is). Snapshots every 25 steps: `records/learned_tables.json`.
- **Seed 2** is exact (200/200) from step 150. On the way it has 93 entries right at step 75, 194 at step 100 and
  199 at step 125. The one wrong entry at step 125 is 1 − 1 − 1, which gives (0, borrow) instead of (9, borrow).
- **Seed 1** stalls at 176/200 by step 400. Every borrow is right, but the digits of minuends 1 and 2 are swapped
  whenever they borrow (1 − 3 − 1 gives 8 instead of 7, 2 − 3 − 1 gives 7 instead of 8, and so on).

**The climb.** The subtraction half of the task set, in base 10 (settings (10, 1) and (10, 2)), with only the
subtraction cell: no ADC, MUL or LDH instructions, so every digit of arithmetic is a lookup in the learned table.
BPE, the 3M budget.
- **Verification** still checks against the true answers. A program "works" only if the machine with this table
  computes the right numbers.
- **`exact_worst`** in each record is the same program with the true table: whether it is the true algorithm.

| table | entries right | level 0 solves | then |
|---|---|---|---|
| exact | 200/200 | copy, shl, shr, sub, subY, lt, ltY, neg, negN | level 1: both links; level 2: both chains and both steps (`BRG.X B[STA.S P[SBB.N INC] SEL.S]` merged); level 3: the reduction loop. All 16 tasks of the subtraction half |
| seed 2, step 150 | 200/200 | the same, with the same programs and the same merges | identical to exact (r1 records) |
| seed 2, step 125 | 199/200 | copy, shl, shr, lt, ltY, neg, negN: not sub or subY | level 1: nothing (the links need difference digits); stops |
| seed 2, step 100 | 194/200 | copy, shl, shr only | no subtraction task passes, so nothing to merge: the climb never starts |
| seed 1, step 400 | 176/200 | copy, shl, shr, lt, ltY, neg, negN | level 1: nothing; stops |
| seed 2, step 75 | 93/200 | copy, shl, shr only | nothing to merge: it never starts |

Per run (`python -m library.compare --summary records/lc_*_r6.jsonl`):

| table | levels | tasks solved (of 16) | longest program found | distinct states searched |
|---|---|---|---|---|
| exact | 4 | 16 | 11 (reduce) | 0.23G |
| seed 2, step 125 (199/200) | 2 | 7 | 5 (negN) | 0.67G |
| seed 2, step 100 (194/200) | 1 | 3 | 2 (shr) | 0.31G |
| seed 1, step 400 (176/200) | 2 | 7 | 5 (negN) | 0.73G |
| seed 2, step 75 (93/200) | 1 | 3 | 2 (shr) | 0.32G |

With a wrong table the failing searches run to the end of their budget, so a run searches more states and finds less.

- **What survives a wrong table** is exactly the tasks that never read a wrong entry:
  - the comparisons need only the borrow column, which seed 1 has right;
  - negation only uses minuend 0.

  Every task that reads a difference digit fails. So does the link, which needs the difference whenever x ≥ N.
- **Answers do not let the search work around a wrong entry.** No alias that avoids the wrong entries showed up: the
  digit cell is the only way to subtract, so a wrong entry the probes reach rules out every subtraction.
- **So library learning has to follow the cells, not run with them.** The anchors' recipe finishes the table in
  100–275 steps without answers (ANCHORS.md), and then the climb is the exact one. That is the order an H1 version
  would use (§8). The interesting failure is seed 1: a table that is wrong in a structured way (two rows swapped)
  passes every comparison. Of the in-run laws (§6.1), the first two would not catch it either. N − N never borrows,
  and "x mod N = x" puts x back whenever the subtraction borrows, so both read only borrows. The first law that keeps
  a borrowing difference's digits is the adder's (x + N) mod N = x. Catching it is the anchors' digit-order laws' job,
  and for seed 1 they had not done it in 400 steps.

## 8. An H1 version: a sketch

What an H1 submission doing this in its own training run would hold, and what it would cost. It was not built
(§8.3 says why).

### 8.1 The pieces

1. **Cells, first.** The subtraction table comes from the anchors' recipe (x < N on every top part, the order laws,
   bijectivity). The certified anchors files teach it exactly in 100–275 steps on the rows' own numbers, with no
   answer. The times table comes from the ladder lesson, as in SQUARE.md. Addition is a − (0 − b) with the
   subtraction table, or a third table taught by the inverse law with the finished subtraction. §7 shows the climb
   needs the cells finished before it starts: a table with even one wrong entry that the probes reach stops it.
2. **The climb, on the GPU.** `inrun.py`'s tasks, from the batch's own rows: top parts in base 10 at widths 1 and 2,
   the facts, the found programs as generators and checks. The search runs on uint8 tensors: one state per row of a
   [states, probes × registers × places] tensor, an instruction as a gather/scatter at the pointer, the dedup as a 64-bit
   hash and `torch.unique`, the merges on the host, as the schedule already is. BPE needs no gradient.
3. **The answer path.** The accepted x² mod N program, checked on the T = 1 rows at full width against their labels,
   runs at W = 8 on one-hot digits with the cells as straight-through one-hot tables. That is the current model's
   prediction path (COMBINED.md §3), with the found program in place of the skeleton. For rule 8 the choice among the
   accepted programs would be a whole-option mixture, as the menus' switches are.
4. **Reading and writing.** Reuse the learned reader and writer (IO.md, WIRING.md stage 3). The instruction set reads
   numbers from registers, not tokens.
5. **The loop over T.** A loop kind that runs its body T times, with T read like the other numbers. Or keep the learned
   countdown (CONTROL.md).

### 8.2 Cost

- **The CPU prototype's searches.** The in-run run at the 3M budget (§6.2) spent 16.4 minutes of search on 2-CPU
  containers over 5 levels. The searches that found their task took 1.9 of them; the rest was searches that ran out
  of budget (the multiply rows at every level).
- **On a GPU.** The work is batched gathers, scatters and hashes on uint8 arrays, about 2 µs per state and item in
  numpy. On an H100 the same batches should run 100–1,000 times faster. That is seconds to a minute for that climb,
  and minutes for the 16M budget. That is small against 3,600 s.
- **The real cost is the port.** About 1,500 lines of numpy and Python (search, merges, laws) would go to torch, plus
  the reader, the writer and the T loop around it, plus the rule-8 mixture over found programs.

### 8.3 Why not built here

- **Frequency-based merges alone did not reach x² mod N.** The climb got there with two additions: loops over cost-3
  bodies (§5.3), and solved tasks as routines with a calling convention (§5.5). An H1 file would inherit every such
  choice, and each is one more hand-written part to justify under rule 7.
- **The in-run version has not reached x² mod N** (§6.2). It found the reduction's pieces and the multiply rows from
  the laws alone, but not the multiply. It needs more tasks per concept, or a search fast enough for loop bodies of
  cost 3 at about 80 instructions.
- **The in-run version is a prototype on CPU.** Rule 9 needs it on the GPU, and a submission must also read and
  write tokens.
- **Each step of the removal ladder took hours of certification.** An H1 attempt is a new design (no skeleton, no
  menus), not a variant of the certified file. It belongs in its own session, with this study's records as its
  starting point.

## 9. Files and how to rerun

| file | what |
|---|---|
| `dsl.py` | the machine: registers, the 32 primitives, loops, macros and holes, batches of machines |
| `tasks.py` | the 26 tasks (and r1's 17) and their exact answers |
| `search.py` | breadth-first search by cost with merged states, all minimal programs, verification, alias skipping |
| `merge.py` | tree-BPE, MDL, MDL with holes, the description length |
| `jobs.py` | one search job → one record (Modal-friendly) |
| `wake_sleep.py` | the levels: search every unsolved task, merge, repeat; every configuration (`CONFIGS`) |
| `modal_library.py` | Modal: one driver per run, searches fanned out (≤ 16 containers), a Volume cache for big searches |
| `inrun.py` | the in-run tasks: probes from H1 rows, laws, found programs as generators and checks |
| `learned.py` | snapshots of the anchors-learned subtraction table; the subtraction-half configurations |
| `certify.py` | a found x² mod N program at H1's size on the released ladders, T = 1…64 |
| `selftest.py` | hand-written reference programs for every task (plumbing only; the search never sees them) |
| `summarize.py`, `report.py`, `compare.py`, `table_tasks.py` | printing records; the tables of §5 |
| `records/` | one `.jsonl` per run, one line per level (below), `learned_tables.json`, `reference_corpus_merges.json`, `*_certify.json`, `inrun_mini_local.json`, the call ids of each launch |

Run from `cpu_first/`:

```bash
~/github/sotaku/venv/bin/python -m library.selftest
```

```bash
~/github/sotaku/venv/bin/python -m library.wake_sleep --config tiny
```

```bash
LIB_MAX_SMALL=16 ~/github/sotaku/venv/bin/python -m modal run --detach library/modal_library.py --configs bpe,mdl,mdlh --tag r3
```

```bash
~/github/sotaku/venv/bin/python -m modal run library/modal_library.py --wait library/records/calls_r3.json
```

```bash
~/github/sotaku/venv/bin/python -m library.certify library/records/bpe_big_r5.jsonl
```

The record names:
- r1: the first task set (one task per concept);
- r2: the 26 tasks, with BPE's old non-overlapping pair counts;
- r3: standard counts and alias skipping (the small-budget reference);
- r4: the half-price MDL cut short by the container cap (level 0 only);
- r5: the 16M budget;
- r6: the half-price MDL runs, the learned cells (`lc_*`) and the small in-run run;
- r7, r9, r10: in-run runs stopped early (superseded: no cost-3 bodies; no primitive fallback; the first-accepted
  helper);
- r8: r5 continued with loop bodies up to cost 3;
- r11: r8 continued with routines;
- r12: in-run, 16M budget, cost-3 bodies, primitive fallback, Occam helpers;
- r13: r12 at level 3, continued with routines;
- r14: r12 at level 4, continued with mac, one program per task, the largest-pair tie-break and routines.

Every resumed run's record starts with copies of the levels it continues.
