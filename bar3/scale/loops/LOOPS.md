# Composition round 5: each cut's loop, the operation itself

<!-- PLAIN -->
## In plain words

**The question.**
- Round 4 (bar3/scale/slice/SLICE.md) cut the target's laws to one digit place. The multiply's row and the
  reduction's step came, and for x² the round.
- But a cut is one step of a loop. The multiply (the row over the multiplier's digits) still came from the book, and
  the whole reduction (the step over the dividend's digits) only as bar 3's rest of the way from x². x³ and 10·x² + 1
  stopped there, and without the book's multiply nothing posed it.
- This round poses each cut's loop as well, as the operation itself on whole numbers. Does the multiply come from its
  loop, and then x²'s round? Does the whole reduction come as a routine for x³ and 10·x² + 1, and their rounds with it?

**The source** (`hooks.py`).
- **Three loops:**
  - the row's loop is the multiply, x·o;
  - Horner's step's loop is the reduction, (x + oN) mod N;
  - x + d's loop is the addition.
- **How they are posed:** each once its cut is posed, on the book's own pattern for that operation, answered by
  Python's integers and accepted at the verify widths. A loop the book poses itself is left to the book.

**What came** (one B200 each, round 3 and 4's settings; every round right on both of H1's ladders, T = 1 to 64,
768/768 on all 14 rungs):

| arm | what | the round |
|---|---|---|
| `l5_x2` | x², the book lt;red;add (book_min less its reduction, its row and its multiply) | **yes, level 2** (478 s) |
| `l5_x2_bare` | round 2's search fixes off | **yes, level 2** (485 s) |
| `l5_x2_all_off` | every one of round 2's flags off | **yes, level 2** (770 s), from the round search by calls |
| `l5_x2_less_add`, `l5_x2_less_link` | the book lt;red, and lt alone | **yes, level 2** (453 s and 421 s) |
| `l5_x2_no_book` | no book | no: the multiply came, but nothing posed the comparison, so no step and no reduction |
| `l5_x2_bare_no_redloop` | the search fixes off, and the reduction's loop off | no: nothing posed the reduction |
| `l5_x3`, `l5_x3_less_mul` (and 7,200 s each) | x³, R1's book and the book less its multiply | no: the reduction and the multiply, then no route |
| `l5_10x2p1`, `l5_10x2p1_less_mul` (and 7,200 s) | 10·x² + 1 | no: the reduction, and with less_mul the multiply and x² mod N; no + 1 |
| `l5_x2_lessons` (and 7,200 s) | anchored pairs | no: every look null; at 7,200 s the row (level 3) and the multiply (level 4) by the rule's tie order, but no step or reduction |
| `replay_s4_x2` | round 4's s4_x2, the loops off | round 4's record again: the round at level 5 |
| `s4_x2_no_subgoal`, `_no_lawnear`, `_no_spell` | round 4's skipped arms: s4_x2 with one search fix off | **yes**: levels 5, 6 and 5 |

**How the round came** (`l5_x2`).
1. Level 0: the row, `LDH.Y CLK P[MUL.X INC]`, from the multiply's cut; the comparison from the book.
2. Level 1: **the multiply from its loop**, `CLA W[row INC SHL.Y]` (5 tokens over the row); Horner's step from the
   reduction's cut.
3. Level 2: **the reduction from its loop**, `LDA.N P[BRG.X step]` (4 over the step). Then the round: x², and the
   reduction (cost 3).

That is three levels before round 4. The reduction was posed and solved as a problem of its own, at the regular 1M
probe, not found as the rest of the way.

**How far down the book: to the comparison.**
- With lt alone the round still came at level 2.
- With no book, the multiply still came from its loop. But no law of x²'s holds a comparison, so nothing posed it,
  and without it there is no step, no reduction and no round.

**Round 2's search fixes: the reduction's loop replaces them.**
- With the loops, the round came with every search fix off, and with every one of round 2's flags off. With the flags
  off it came by calls: the multiply on (X, X), then the reduction.
- Without the reduction's loop, and with the fixes off, nothing posed the reduction.
- **Round 4's skipped arms** (the cuts, no loops): each of SUBGOAL, LAWNEAR and SPELL can go alone. x² has two
  posers (LAWNEAR, and a restriction of the target), and the rest of the way has two finders (SPELL's spellings, and
  REST_DEEP).

**Generality: the cube and 10·x² + 1.**
- Both now get the whole reduction as a routine at level 2. With the book less its multiply they also get the
  multiply from its loop at level 1 (with R1's book, x³'s came at level 4 and 10·x² + 1's never). No round, even at
  7,200 s.
- **The cube** needs x kept beside x² mod N: two multiplies and two reductions, 4 calls or 11 tokens over the
  routines, past the round search's 2 calls. Its values went off the route.
- **10·x² + 1** reached x² mod N as a value, but its round adds 1, and this machine has no constant.

**The consolidation's lesson: posing the loop does not fix anchored pairs' myopia.**
- Every look at every level found nothing within its 300k states. So all of anchored pairs' gains were tied, and its
  picks fell to the rule's order.
- At 2,400 s that order took the comparison's, the link's and the addition's bindings; the deadline came in level 3.
- At 7,200 s it reached the row at level 3 and the multiply at level 4, but not the step or the reduction. No round by
  the deadline, in level 6.
- So the row was not picked for its value. A look at the target with the row still gains nothing until the multiply
  exists, as in round 4.

**What it means.**
- A cut and its loop together build the operation from one digit place up. For x², the book shrinks to the comparison,
  and round 2's search fixes all go.
- What stops the other targets now:
  - compositions longer than the round search (the cube's x kept beside x² mod N);
  - constants (10·x² + 1's + 1);
  - and, for anchored pairs, a look that cannot see past one level.
- **Cost:** about $93 of the soft $100 (§9).
<!-- /PLAIN -->

## Details

### 1. Pre-registered

PREREG.md (frozen 2026-10-26 19:17:33 PDT, read-only, sha256
`9fc1ca7ba7b091819ba7d66674e1a612102f3f08ab687fd29626496e03ff03d8`, `logs/prereg_sha256.txt`), before any arm ran,
unchanged:

<!-- PREREG -->
# Composition round 5: each cut's loop, pre-registered

Written 2026-10-26, before any run of this study. LOOPS.md §1 repeats this text unchanged and records the time and hash
it was frozen with.

## The questions

**What round 4 found** (bar3/scale/slice/SLICE.md; 05810dc):
- Cutting the target's laws to one digit place brought the row and Horner's step, and for x² the round.
- x³ and 10·x² + 1 stopped. Their rounds need the whole reduction as a routine first, and it came only as the rest of
  the way from x².
- Without the book's multiply nothing posed the multiply: it is the row's loop, not a cut.

**Here each posed cut's loop is posed too, as the operation itself:**
1. **x² mod N** with the book less its reduction, its row and its multiply: does the multiply come from its loop, and
   then the round?
2. **x³ mod N and 10·x² + 1 mod N** with the book less its reduction and its row: does the whole reduction come as a
   routine, and the round with it?
3. **If those work,** the book with less again.
4. **Anchored pairs** never practised the row in round 4: a shallow look shows no gain until its loop exists. Does
   posing the loop fix that?

## Theory, in brief

1. **A cut is one step of a loop over digit places; its loop is the operation itself.**
   - The row's loop over the multiplier's digits is the multiply, x·o = Σⱼ (x·Bʲ)·oⱼ: `CLA W[T.row INC SHL.X]`, 5
     tokens over the row (9 without it).
   - Horner's step's loop over the dividend's digits, from the top, is the reduction: `STA.Y P[BRG.X T.step]` or
     `CLA P[BRG.X T.step]`, 4 tokens over the step.
   - x + d's loop is the addition, `CLC P[ADC.Y INC]`, the same program.
2. **What the source poses.** Each loop is posed on the book's own pattern for that operation:
   - the multiply on (x, o);
   - the reduction on (x + oN, n), x + oN built by Python (the book builds it by the learner's own addition);
   - the addition on the pair.

   Answers come from Python, in every binding, accepted at the verify widths. A loop the book poses itself is left to
   the book.
3. **The order is fixed.** A loop is learnable only once its cut is a routine: the row (cost 5), then the multiply; the
   comparison, then the step (4 over it), then the reduction.
4. **x².** The multiply and the reduction then exist as routines.
   - The round is two calls of them (the multiply on (X, X), the reduction on (A, N)), inside the round search by calls,
     or 6 tokens of composition.
   - It needs neither LAWNEAR's x², nor the subgoal score, nor the rest of the way, nor REST_DEEP, nor SPELL. Round 2's
     four necessary fixes were all for finding the reduction as the rest of the way.
5. **x³.** The reduction comes as a routine, but the round should not.
   - The round needs two multiplies and two reductions with x kept: `LDA.X STA.Y T.mul STA.X LDA.N T.red STA.X T.mul
     STA.X LDA.N T.red`, 11 tokens over the routines, or 4 calls. That is past the round search (cost 5, or 2 calls)
     and past the look.
   - Its values could build it in steps (x², then x² mod N, then times x, then mod N, each 3–4 tokens over the last),
     but only if each is posed and picked in turn.
   - LAWNEAR's depth-1 view sees no value one law from the cube.
   - So: the reduction, and no round.
6. **10·x² + 1.** The reduction comes as a routine, but the round adds 1, and no register holds 1 (the machine has no
   constant). No round.
7. **The book with less** (if x² works):
   - the loops' answers need no routine, so the book's addition is not needed: less add should still get the round;
   - the step can be learned over the comparison alone (`B[STA.S T.lt SEL.S]`, as in round 4), so the link is not
     needed either: less the link should still get the round;
   - the comparison is the last. None of these targets' laws holds lt, so nothing would pose it, and the book less lt
     is not run.
8. **Anchored pairs.** Posing the loop does not change the row's lookahead: the lookahead searches the target with the
   row, not the loop, and there the row gains nothing until the multiply exists. So under ct2a2 the row should still
   not be picked first.

## Design (`make_copies5.py`, `hooks.py`)

- **The copies:** round 4's (make_copies4.py's lists), paths moved to this folder, with LOOPS's changes:
  - a draw's x + oN is built by Python when the reduction's loop is posed;
  - the loops are in the cuts' table (LOOP_OF), added where their cuts are posed and the book does not pose them;
  - their answers come from Python;
  - the flag is in the config.

  With LOOPS off the copies are round 4's.
- **LOOPS=1** poses every posed cut's loop. A list (mul, red, add) poses a subset.

## Runs

Each on one B200, with:
- round 3 and 4's settings: TARGET small, ROUND_TOO, NP 8, deep probes over whole concepts, SECONDS 2,400, LEVELS 10;
- round 2's flags and TIEFOLLOW;
- SLICE=1 and LOOPS=1.

**Phase 1, in this order, at most 2 B200s at once:**
- `l5_x2`: x², the book lt;red;add on pair (book_min less its reduction, its row and its multiply);
- `replay_s4_x2`: round 4's s4_x2, LOOPS off;
- `l5_x3` and `l5_10x2p1`: R1's book (lt;red;add;mul on pair;x,o), TARGET_FN x3 and 10x2p1.

**Phase 2, only if l5_x2 gets its round, in this order:**
1. `l5_x2_bare`: round 2's search fixes off (SUBGOAL, LAWNEAR, REST_DEEP, SPELL, and TIEBREAK and TIEFOLLOW, which
   follow SUBGOAL);
2. `l5_x2_less_add`: the book lt;red;
3. `l5_x2_less_link`: the book lt;
4. `l5_x2_lessons`: ct2a2, SUBGOAL and REST_DEEP off (round 4's lessons arm, with the loops).

**Any round** is checked on both of H1's ladders to T = 64.

**The tests before the arms,** locally under a memory cap:
- the loops against known programs (`loops_check.py`);
- LOOPS off against round 4's tiny record;
- a tiny run with LOOPS on.

**Money:**
- soft $30, cap $40, `spend.py 4750` before each launch;
- one watcher at the cap less one hour of 2 B200s: stop all at $27;
- phase 2 only while the watcher's estimate leaves room, in its order.

## Predictions

| what | prediction | confidence |
|---|---|---|
| replay_s4_x2 | round 4's record again, apart from times: its round at level 5 | high |
| l5_x2: the multiply | from its loop, the level after the row, or the next | medium-high |
| l5_x2: the reduction | from its loop, the level after the step, or the next | medium-high |
| l5_x2: the round | by level 5, right on both ladders to T = 64 | medium-high |
| l5_x3 | the reduction and the multiply practised; no round | medium (no round) |
| l5_10x2p1 | the reduction practised; no round | high (no round) |
| l5_x2_bare | the round | medium |
| l5_x2_less_add | the round | medium-high |
| l5_x2_less_link | the round | medium |
| l5_x2_lessons | the row not practised in its first 3 levels; no round | medium |
| the conclusion | for x², the cuts and their loops replace the book down to the comparison, and round 2's four search fixes. The cube and 10·x² + 1 get the reduction but not their rounds, which need longer compositions, and for 10·x² + 1 a constant | medium |
<!-- /PREREG -->

PREREG_ADDENDUM1.md (frozen 2026-10-26 20:20:22 PDT, read-only, sha256
`bf54aefc506f7b537af03ce2cac1864759c8f1293d2f6e269a5b621586d8d94d`, `logs/addendum1_sha256.txt`), after the
pre-registration's eight arms and before its own, unchanged:

<!-- ADDENDUM1 -->
# Composition round 5, addendum 1: the book with less again, pre-registered

Written 2026-10-26, after the pre-registration's eight arms and before any run below. LOOPS.md §1 records its time and
hash.

## What the eight arms showed

**x² got the round in every arm that kept the comparison and the loops,** each right on both ladders to T = 64:
- `l5_x2`, the book lt;red;add: the multiply came from its loop at level 1 and the reduction from its loop at level 2;
  the round came at level 2;
- `l5_x2_bare`, round 2's search fixes off;
- `l5_x2_less_add`, the book lt;red;
- `l5_x2_less_link`, the book lt alone.

**The other arms:**
- `l5_x3`: the reduction came as a routine from its loop (level 2), and the multiply came from the book's mul at level
  4. No round by the deadline (level 6).
- `l5_10x2p1`: the reduction came from its loop (level 2). The book's mul was never posed, since its answers need the
  book's addition on the pair as a routine, and none was practised. No multiply, no round.
- `l5_x2_lessons` (anchored pairs): the row was not picked in its 3 levels, and there was no round.
- `replay_s4_x2`: round 4's record again.

## Arms

Each on one B200, the pre-registration's settings, all launched together:
- `l5_x2_no_book`: no book at all (SOURCES bar 3's four only), with the cuts and their loops;
- `l5_x2_all_off`: `l5_x2_bare` with round 2's other three flags off too (CLEAN_VALUES, ROWCHECK, RESTRICT_ROWS),
  so every one of round 2's flags is off;
- `l5_x2_bare_no_redloop`: `l5_x2_bare` with the reduction's loop off (LOOPS=mul);
- `l5_x3_less_mul` and `l5_10x2p1_less_mul`: the book lt;red;add (as `l5_x2`), x³ and 10·x² + 1, so that the
  multiply too comes from its loop.

Any round is checked on both ladders to T = 64.

## Predictions

| what | prediction | confidence |
|---|---|---|
| l5_x2_no_book | no round: nothing poses the comparison, so neither the step (7 tokens over the primitives) nor the reduction is learnable | high |
| l5_x2_all_off | the round | medium-high |
| l5_x2_bare_no_redloop | no round: without the reduction's loop and without round 2's search fixes, nothing finds the reduction | medium |
| l5_x3_less_mul | the multiply from its loop by level 2, the reduction by level 3; no round | medium (no round) |
| l5_10x2p1_less_mul | the multiply and the reduction; no round | high (no round) |

## Money

Five B200s at once. The watcher's total is the cap ($80) less one hour of five: stop all at $48.
<!-- /ADDENDUM1 -->

PREREG_ADDENDUM2.md (frozen 2026-10-26 20:41:16 PDT, read-only, sha256
`08ab0cb8fa0114cfdc10b785f0282f4e09c0fcbfcf083c0b368533cd306c6936`, `logs/addendum2_sha256.txt`), before its arms,
unchanged:

<!-- ADDENDUM2 -->
# Composition round 5, addendum 2: round 4's skipped arms, and the arms the deadline cut, pre-registered

Written 2026-10-26, after the coordinator raised the budget to soft $100, cap $140 ("Use the room to let arms finish
rather than cutting them, and to run the flag-off arms you skipped for money before"), and before any run below.
LOOPS.md §1 records its time and hash.

## Arms

Each on one B200, launched together.

**Round 4's three skipped flag-off arms** (bar3/scale/slice/PREREG.md, phase 2). They run on this study's copies with
LOOPS off, which give round 4's records (`replay_s4_x2`). Each is round 4's s4_x2 with one flag off:
- `s4_x2_no_subgoal`;
- `s4_x2_no_lawnear`;
- `s4_x2_no_spell`.

**The arms the 2,400 s deadline cut, given 7,200 s.** Everything else is unchanged, LEVELS 10 included:
- `l5_x3_long`: `l5_x3`, R1's book;
- `l5_x3_less_mul_long`: `l5_x3_less_mul`, the book lt;red;add;
- `l5_10x2p1_less_mul_long`: `l5_10x2p1_less_mul`;
- `l5_x2_lessons_long`: `l5_x2_lessons`, anchored pairs.

Any round is checked on both ladders to T = 64.

## Predictions

Round 4's three, quoted from its pre-registration unchanged:

| what | prediction | confidence |
|---|---|---|
| phase 2: SUBGOAL off | the round | low-medium |
| phase 2: LAWNEAR off | no round: x² is not posed as a value | medium |
| phase 2: SPELL off | the round | low-medium |

This study's four:

| what | prediction | confidence |
|---|---|---|
| l5_x3_long | no round by level 10: more levels of values, none on the cube's route, which needs x kept beside x² mod N | medium |
| l5_x3_less_mul_long | no round by level 10, for the same reason | medium |
| l5_10x2p1_less_mul_long | no round: no register holds 1 | high |
| l5_x2_lessons_long | the row picked by level 9 (anchored pairs work through the comparison's, the link's and the addition's bindings first); no round by level 10 | low |

## Money

- Seven B200s now, beside addendum 1's four still running: eleven at most.
- The watcher's total is the cap ($140) less one hour of the containers running. It is set at $70 now, and raised as
  arms end (to $95 at seven, $114 at four).
<!-- /ADDENDUM2 -->

### 2. Design

#### 2.1 The loops (`hooks.py`: LOOP_OF, slice_read, slice_target, xon_term)

| cut (round 4) | its loop over the digit places | posed as | its program, known |
|---|---|---|---|
| the row, a + x·d (and x·d) | the multiply, over the multiplier's digits | mul on (x, o): x·o | `CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]` (XY) |
| Horner's step, (B·r + d) mod N | the reduction, over the dividend's digits from the top | red on (x + oN, n): (x + oN) mod N | `STA.Y P[BRG.X T.red.Bx+d.AY]` (XA); `CLA P[BRG.X T.red.Bx+d.AN]` (XN) |
| x + d | the addition, over the other operand's digits | add on the pair | `CLC P[ADC.Y INC]` (AY) |

- **Which loops are posed:** a loop is posed when its cut is (once the target's law is read, at level 0) and LOOPS
  names it, unless the book poses that law on that pattern itself.
  - In `l5_x2` the book has neither the multiply nor the reduction, so both loops are this source's.
  - In `l5_x3` and `l5_10x2p1` the book's own multiply stays (R1's book), and the reduction is this source's.
- **The draws** are the book's. A draw's x + oN, which the book builds by the learner's own addition (`repeat_add`),
  is built by Python when the reduction's loop is posed: x < N or x = N, o another row's x′, all at half the
  registers' places (`double`), so x + oN < N·B^W fits.
- **The answers** are Python's integers: x·o, (x + oN) mod N, and the sum of the pair's two operands.
- **Acceptance** is the book's: 60 fresh rows at each of the verify widths 1, 2 and 3.
- **The names** are the book's, as in round 4: `mul/x,o@XY`, `red/x+on@XA`, and their routines `T.mul.x,o.XY`,
  `T.red.x+on.XA`.

#### 2.2 The copies (`make_copies5.py`)

Round 4's copies, from make_copies4.py's lists (round 3's, then round 4's), the module paths moved to this folder,
with this study's asserted replacements:
- in `spcore_l.py`, a draw's x + oN by `hooks.xon_term` when the reduction's loop is posed;
- in `core_l.py`, the run's config records LOOPS;
- in `hooks.py`, round 4's slicing source with three replacements: the loops' table and flag, the loops added in
  `slice_read`, and their answers in `slice_target`. `xon_term` is appended.

Round 4's law reader (`bar3/scale/slice/lawread.py`) and round 2's subgoal score are imported from their folders.

### 3. The tests before the arms (local, under a memory cap)

| test | what it showed |
|---|---|
| `loops_check.py` (`logs/loops_check.txt`) | 10 of 10 against known programs. The multiply over the row, the reduction over Horner's step (two bindings, three programs) and the addition pass their loops' probes and acceptors at every verify width. The row's low digit alone, a multiply from a dirty accumulator, one conditional subtraction a digit where the step is wanted, and the like, fail |
| LOOPS off (`logs/tiny_loops.txt`, `logs/tiny_slice_x2.*`) | round 4's tiny run (SLICE on, R1's book, 3 levels) gives round 4's record again, apart from times |
| LOOPS on, x² (`logs/tiny_loops_x2.*`) | the book lt;red;add, 3 levels at tiny budgets: the reading and the cuts as in round 4, and both loops posed (`mul/x,o`, `red/x+on`). 106 candidates on 92 roots at level 0. No errors |
| LOOPS on, x³ (`logs/smoke_loops_x3.*`) | R1's book, 1 level: the cuts and the reduction's loop posed; the multiply's loop left to the book's own mul. No errors |

### 4. The runs (`modal_loops.py`; `records/*.json` and `.log`; `logs/launch_*.txt`)

| launch | PDT | app | arms | B200s at once |
|---|---|---|---|---|
| phase 1 | 19:17 | `ap-fKOc27yonAqzLDGy6DBghb` | l5_x2, replay_s4_x2, l5_x3, l5_10x2p1 | 2 (max_containers 2) |
| phase 2 | 19:35 | `ap-E8vaCE0oHLtHmkkNmXdKQc` | l5_x2_bare, l5_x2_less_add, l5_x2_less_link, l5_x2_lessons | 4, beside phase 1's last 2 |
| addendum 1 | 20:20 | `ap-Rgq3V3rAbOhHqELXv4s5yT` | l5_x2_no_book, l5_x2_all_off, l5_x2_bare_no_redloop, l5_x3_less_mul, l5_10x2p1_less_mul | 5 |
| addendum 2 | 20:41 | `ap-ReeN6ffw3Jx5sCoKcjCcC8` | s4_x2_no_subgoal, s4_x2_no_lawnear, s4_x2_no_spell, l5_x3_long, l5_x3_less_mul_long, l5_10x2p1_less_mul_long, l5_x2_lessons_long | 7, beside addendum 1's 4 |

- **Each launch** came after `spend.py 4750` (LAUNCH OK). Each pre-registration was frozen before its arms:
  - PREREG.md at 19:17:33;
  - addendum 1 at 20:20:22, sha256 `bf54aefc506f7b537af03ce2cac1864759c8f1293d2f6e269a5b621586d8d94d`;
  - addendum 2 at 20:41:16, sha256 `08ab0cb8fa0114cfdc10b785f0282f4e09c0fcbfcf083c0b368533cd306c6936`.
- **The settings:** every arm used round 3 and 4's settings. The four long arms of addendum 2 had SECONDS 7,200 in
  place of 2,400, and the runner's container timeout was raised to 9,000 s for them.
- **Changes from the pre-registration, at the coordinator's word, during the study:**
  - **The budget** rose from soft $30, cap $40 to soft $60, cap $80 at about 19:20 PDT ("Everything can be on the GPU
    since we still have budget left"), then to soft $100, cap $140 at about 20:38 PDT ("Use the room to let arms
    finish rather than cutting them, and to run the flag-off arms you skipped for money before").
  - **Concurrency:** the pre-registration's "at most 2 B200s at once" became "up to about 10 at once where arms are
    independent". Each arm runs on its own B200 with its own time limits and its own records, so running them side by
    side changes no outcome.
  - **The watcher** was relaunched at each change, without stopping any arm. Its total was the cap less one hour of
    every container that could run at once: $27, then $41, $48, $70, $95 and $114 (`logs/watch_l5_*.txt`).

### 5. Outcomes (`records/*.json`; bar3/scale/slice/summary.py)

#### 5.1 l5_x2: the multiply and the reduction from their loops, and the round at level 2 (478 s)

The book was lt;red;add on pair: book_min less its reduction, its row and its multiply.

| level | practised (lpca) | from |
|---|---|---|
| 0 | the row in four bindings (`LDH.Y CLK P[MUL.X INC]`); the comparison in three; a value | the mul cut; the book |
| 1 | **the multiply** in two bindings (`CLA W[T.acc.mul.o,x,d.AYX INC SHL.Y]`, and with X and Y swapped; cost 5 over the row); Horner's step in three (`B[STA.S T.lt.pair.AX SEL.S]`); the link in three; a value | the multiply's loop; the mod cut; the book |
| 2 | **the reduction** in two bindings (`LDA.N P[BRG.X T.red.Bx+d.AN]`, `LDA.Y P[BRG.X T.red.Bx+d.AY]`; cost 4 over the step); the addition in three; x², `LDA.X STA.Y T.mul.x,o.XY`; a restriction | the reduction's loop; the book; bar 3's values |

- **The round** came in level 2's target search: `T.val.63.tgt.XN STA.X T.red.x+on.XN` (cost 3), x² and then the
  reduction, the routine learned from its loop. That is three levels before round 4's level 5.
- **The ladders:** 768/768 on all 14 rungs of both, T = 1 to 64 (`logs/ladder_l5_x2.txt`).
- **No search fix was needed to find the reduction.** It was posed and solved as a problem of its own, at the regular
  1M probe. LAWNEAR did find x² at level 2, and the subgoal score singled it out (mod(g, N)), but the round's
  reduction is the loop's routine, not a rest of the way.

#### 5.2 The replay: round 4's record again

`replay_s4_x2` (round 4's s4_x2 on this study's copies, LOOPS off): the same record, apart from times (`same.py`;
`logs/replays_same.txt`), its round at level 5 in 965 s.

#### 5.3 l5_x3: the reduction as a routine, the multiply late, no round (the deadline in level 6)

- **Level 0:** the row and the comparison. **Level 1:** the step, the link, x·d. **Level 2:** the addition and **the
  reduction from its loop**, `LDA.N P[BRG.X T.red.Bx+d.AN]` and `LDA.Y P[BRG.X T.red.Bx+d.AY]`.
- **Level 4:** the multiply, from the book's own mul. R1's book keeps it, so its loop is left to the book, and the
  book's mul is posed only once the addition is a routine.
- **Levels 3–5:** values built on the reduction (`INC INC T.red.x+on.XN`, and on those), none on the cube's route.
  LAWNEAR found no state and the subgoal score singled out nothing.
- **The end:** the 2,400 s deadline, in level 6's probes. No round.

#### 5.4 l5_10x2p1: the reduction as a routine, no multiply, no round (the deadline in level 5)

- **Level 0:** the row, the comparison, and add's cut (the addition, three bindings). **Level 1:** the step, the link,
  x·d, values. **Level 2:** **the reduction from its loop**, and values.
- **No multiply.** The book's mul is R1's, and its answers need the book's addition on the pair as a routine. That
  addition was learnable from level 2 at cost 2 (over add's cut), but it was never among lpca's four costliest
  concepts: values of cost 3 outranked it at every level. It was never practised, so the book's mul was never posed.
- **Levels 3–5:** values; two subgoals by table alone, whose rests REST_DEEP could not solve at 10M.
- **The end:** the deadline, in level 5's target search. No round.

#### 5.5 Phase 2: three more rounds, and anchored pairs still blind to the row

| arm | change from l5_x2 | the round | x² came as | routine path |
|---|---|---|---|---|
| `l5_x2_bare` | round 2's search fixes off (SUBGOAL, LAWNEAR, REST_DEEP, SPELL, TIEBREAK, TIEFOLLOW) | **level 2** (485 s) | a restriction: the target on the rows where x² < N (`target/rst4@XN`) | the multiply from its loop (1), the reduction from its loop (2) |
| `l5_x2_less_add` | the book lt;red | **level 2** (453 s) | a value | the same |
| `l5_x2_less_link` | the book lt alone | **level 2** (421 s) | a value | the same, the step over the comparison (`B[STA.S T.lt.pair.AX SEL.S]`) |
| `l5_x2_lessons` | ct2a2, SUBGOAL and REST_DEEP off | no (the deadline in level 3) | – | the comparison, the link and the addition only; the row never picked |

- **Every round is right on both ladders,** 768/768 on all 14 rungs (`logs/ladder_l5_x2_*.txt`).
- **The rounds:**
  - `l5_x2_bare`: `T.target.rst4.XN STA.X LDA.N T.red.x+on.XA` (cost 4);
  - the two less-book arms: x² and then the reduction (cost 3), as in `l5_x2`.
- **Anchored pairs.** Posing the loop did not change what anchored pairs saw. Its first three levels practised the
  comparison, the link and the addition, in several bindings each, at 270 to 880 s a level. The row was learnable from
  level 0 and never picked.

#### 5.6 Addendum 1: the floor of the book, and what replaced round 2's search fixes

| arm | change | the round | what came |
|---|---|---|---|
| `l5_x2_all_off` | every one of round 2's flags off | **level 2** (770 s), right on both ladders | the row (0), the multiply and the step (1), the reduction (2); the round from the round search **by calls**: the multiply on (X, X), then the reduction (`LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA`) |
| `l5_x2_no_book` | no book at all | no (the deadline in level 7) | the row (0) and the multiply from its loop (1), but no comparison, so no step and no reduction. LAWNEAR found x² at level 2 and REST_DEEP probed its rest at 10M for nothing |
| `l5_x2_bare_no_redloop` | round 2's search fixes off, and the reduction's loop off | no (the deadline in level 5) | the row, the multiply, the step, and x² from a restriction. With no near-miss value of x², no rest of the way was posed, and nothing posed the reduction |
| `l5_x3_less_mul` | x³, the book lt;red;add | no (the deadline in level 5) | the multiply from its loop at level 1 (three levels earlier than with R1's book), the reduction at level 2; then values off the cube's route (`INC INC T.red.x+on.XN` and on those) |
| `l5_10x2p1_less_mul` | 10·x² + 1, the book lt;red;add | no (the deadline in level 5) | the multiply (1) and the reduction (2), then x² mod N as a value (`T.mul.x,o.YX T.red.x+on.XN`, level 3); the round's + 1 never came |

- **The book's floor is the comparison.** With lt alone the round came (`l5_x2_less_link`). With nothing, the
  multiply still came from its loop, but the step, the reduction and the round did not.
- **The reduction's loop is what replaced round 2's search fixes:**
  - with the loop, the round came with every search fix off (`l5_x2_bare`), and with every flag off (`l5_x2_all_off`);
  - without it, and without the fixes, nothing posed the reduction (`l5_x2_bare_no_redloop`).

#### 5.7 Addendum 2: round 4's skipped arms

Round 4's s4_x2 (the cuts, no loops, R1's book), each with one of round 2's search fixes off, on this study's copies
with LOOPS off (which give round 4's records: `replay_s4_x2`):

| arm | the round | how the reduction came | right on both ladders |
|---|---|---|---|
| `s4_x2_no_subgoal` | **level 5** (952 s) | as in s4_x2: LAWNEAR's x², then its rest of the way, `LDA.N P[BRG.X T.red.Bx+d.AN]`, at the 1M probe | 768/768 on all 14 rungs |
| `s4_x2_no_lawnear` | **level 6** (1,650 s) | x² from a restriction (the target on the rows where x² < N), and the rest of the way from a near-miss value, at the 1M probe | 768/768 |
| `s4_x2_no_spell` | **level 5** (1,241 s) | the 1M probe missed the 4-token rest, and REST_DEEP's 10M probe found 5-token ones; the practised rest was `CLA P[BRG.X T.add.pair.AN T.red.Bx+d.AN]` | 768/768 |

So with the cuts, each of round 2's four search fixes can go alone: x² has two posers (LAWNEAR and the restriction),
and the rest of the way has two finders (SPELL's spellings at the 1M probe, and REST_DEEP's 10M).

#### 5.8 Addendum 2's long arms: 7,200 s, and still no rounds

| arm | how far | after the 2,400 s arm's end |
|---|---|---|
| `l5_x3_long` | 10 levels, the level cap (5,861 s) | levels 5–6: more values built on the reduction (`T.val.… T.red.Bx+d.AN LDA.S` and the like). Levels 7 and 9: nothing new learnable. Level 8: the practised concepts' other bindings (the row in four more). LAWNEAR found no state and the subgoal score singled out nothing at any level |
| `l5_x3_less_mul_long` | 10 levels, the level cap (5,823 s) | the same, level for level |
| `l5_10x2p1_less_mul_long` | 9 levels, the deadline in level 9 (7,209 s) | values built on x² mod N and the reduction (`T.mul.x,o.YX T.red.x+on.XN` itself at level 6, then `… INC ADC.S`, `… SBB.S SEL.S` and the like), none the target; one subgoal by table (level 3), whose rest REST_DEEP's 10M probe could not solve |
| `l5_x2_lessons_long` | 6 levels, the deadline in level 6 (7,210 s) | **the row at level 3** (two bindings), **the multiply from its loop at level 4**, two more of the row's bindings at level 5. LAWNEAR found x² at level 5 (`LDA.X STA.Y T.mul.x,o.XY`), but the step and the reduction never came |

- **The x³ arms' rounds:** at every level the round search found nothing. By composition it exhausted cost 5 (from
  level 8, its 4M-state budget), and by calls its 2 calls. The route needs 4 calls (§1, the pre-registration's theory,
  item 5).
- **Anchored pairs saw nothing at any level.** Every look returned no program within its 300k states, so every single
  scored the same (the cost of its one extra symbol) and every pair the same (two symbols).
  - With every gain tied, the picks fell to the rule's order: the pairs anchored on the 2 costliest learnable
    concepts, their partners in pool order.
  - That order took the comparison's bindings at level 0, the link's at 1 and the addition's at 2, and reached the row
    at level 3, once its bindings were the costliest left; the multiply came at level 4 the same way.
  - Round 4's lessons arm (R1's book, no loops) took the same picks through level 2. At level 3 it took the step in one
    binding, and it never took the row in its 4 levels.
- **So posing the loop did not fix the myopia.** The looks see no more than before. With the loop in the pool, the
  tie order reached the row and then the multiply, but not the step and the reduction before the deadline.

### 6. Predictions against outcomes

**The pre-registration's:**

| what | prediction | confidence | outcome |
|---|---|---|---|
| replay_s4_x2 | round 4's record again, apart from times: its round at level 5 | high | **right**: the same record (`logs/replays_same.txt`), its round at level 5 |
| l5_x2: the multiply | from its loop, the level after the row, or the next | medium-high | **right**: level 1, the level after the row |
| l5_x2: the reduction | from its loop, the level after the step, or the next | medium-high | **right**: level 2, the level after the step |
| l5_x2: the round | by level 5, right on both ladders to T = 64 | medium-high | **right**: level 2; 768/768 on all 14 rungs |
| l5_x3 | the reduction and the multiply practised; no round | medium (no round) | **right**: the reduction at level 2, the book's multiply at level 4; no round by the deadline (level 6) |
| l5_10x2p1 | the reduction practised; no round | high (no round) | **right**: the reduction at level 2; no multiply, since the book's mul was never posed; no round |
| l5_x2_bare | the round | medium | **right**: level 2, x² from a restriction |
| l5_x2_less_add | the round | medium-high | **right**: level 2 |
| l5_x2_less_link | the round | medium | **right**: level 2, the step over the comparison |
| l5_x2_lessons | the row not practised in its first 3 levels; no round | medium | **right**: the comparison, the link and the addition in levels 0–2; the deadline in level 3 |
| the conclusion | for x², the cuts and their loops replace the book down to the comparison, and round 2's four search fixes. The cube and 10·x² + 1 get the reduction but not their rounds, which need longer compositions, and for 10·x² + 1 a constant | medium | **right** in each part (§5.1–5.6), and at 7,200 s too (§5.8) |

**Addendum 1's:**

| what | prediction | confidence | outcome |
|---|---|---|---|
| l5_x2_no_book | no round: nothing poses the comparison, so neither the step nor the reduction is learnable | high | **right**: the row and the multiply came, but no comparison, step or reduction; the deadline in level 7 |
| l5_x2_all_off | the round | medium-high | **right**: level 2, from the round search by calls |
| l5_x2_bare_no_redloop | no round: without the reduction's loop and without round 2's search fixes, nothing finds the reduction | medium | **right**: the deadline in level 5, no reduction |
| l5_x3_less_mul | the multiply from its loop by level 2, the reduction by level 3; no round | medium (no round) | **right**: the multiply at level 1, the reduction at level 2; no round |
| l5_10x2p1_less_mul | the multiply and the reduction; no round | high (no round) | **right**: both, and x² mod N as a value; no round |

**Addendum 2's:**

| what | prediction | confidence | outcome |
|---|---|---|---|
| round 4's phase 2: SUBGOAL off | the round | low-medium | **right**: level 5 |
| round 4's phase 2: LAWNEAR off | no round: x² is not posed as a value | medium | **wrong**: the round at level 6. x² came from a restriction (the target on the rows where x² < N), which round 4's prediction left out |
| round 4's phase 2: SPELL off | the round | low-medium | **right**: level 5, the rest of the way found by REST_DEEP's 10M probe |
| l5_x3_long | no round by level 10 | medium | **right**: 10 levels, the level cap; values off the route, then nothing new learnable |
| l5_x3_less_mul_long | no round by level 10 | medium | **right**: the same |
| l5_10x2p1_less_mul_long | no round: no register holds 1 | high | **right**: 9 levels to the deadline; no + 1 |
| l5_x2_lessons_long | the row picked by level 9; no round by level 10 | low | **right** as far as it ran: the row at level 3, after the comparison's, the link's and the addition's bindings, as reasoned; no round in its 6 levels (the deadline in level 6). But every look was null: the row came by the rule's tie order, not by a seen gain (§5.8) |

### 7. Dead ends, and how hard each was tried

| what | how hard | where it stopped |
|---|---|---|
| the cube with R1's book (`l5_x3`, `l5_x3_long`) | 6 levels to the 2,400 s deadline; then 10 levels, the level cap, at 7,200 s | the reduction from its loop (level 2) and the book's multiply (level 4); then values built on the reduction, none on the cube's route; at 7,200 s, levels 7 and 9 had nothing new to learn. LAWNEAR saw no state one law from the cube |
| the cube with the book less its multiply (`l5_x3_less_mul`, `l5_x3_less_mul_long`) | 5 levels to the deadline; then 10 levels, the level cap, at 7,200 s | the multiply (level 1) and the reduction (level 2) from their loops; the same values off the route; levels 7 and 9 had nothing new to learn |
| 10·x² + 1 with R1's book (`l5_10x2p1`) | 5 levels, to the deadline | the reduction from its loop; no multiply: the book's mul waits on the book's addition, learnable from level 2 but never among lpca's picks |
| 10·x² + 1 with the book less its multiply (`l5_10x2p1_less_mul`, `l5_10x2p1_less_mul_long`) | 5 levels to the deadline; then 9 levels to the 7,200 s deadline | the multiply, the reduction, and x² mod N as a value; then values built on it, none the target: no + 1, which no register holds |
| anchored pairs (`l5_x2_lessons`, `l5_x2_lessons_long`) | 3 levels to the deadline; then 6 levels to the 7,200 s deadline | every look null at every level, so the picks went by the rule's tie order: the comparison's, the link's and the addition's bindings, then (at 7,200 s) the row at level 3 and the multiply at level 4; never the step or the reduction |
| no book (`l5_x2_no_book`) | 7 levels, to the deadline | the row and the multiply, but no comparison: nothing poses it, so no step and no reduction. REST_DEEP's 10M probe of x²'s rest found nothing |
| the reduction's loop off, with round 2's search fixes off (`l5_x2_bare_no_redloop`) | 5 levels, to the deadline | the row, the multiply, the step, and x² from a restriction; nothing posed the reduction |

### 8. What is left

- **The cube's round needs x kept beside x² mod N.**
  - At 7,200 s both cube arms ran out of new concepts (levels 7 and 9 had nothing new to learn) and reached the level
    cap with no round.
  - Its reading names the route: mod(mul(mod(mul(X,X),N),X),N) holds the modular multiply, mod(mul(·,·), N), twice.
  - A source that posed a reading's repeated sub-law as a law of its own (here x·o mod N, the multiply and the
    reduction in one routine) would make the cube two calls of it, as x² is now two calls of the multiply and the
    reduction. That is inside the round search.
  - Otherwise it needs a rest of the way that keeps x (COMPOSE3.md §8).
- **10·x² + 1 needs a 1.** It reached x² mod N as a value, and at 7,200 s many values built on it, but its round adds
  1, and no register or routine holds a constant.
- **The comparison is the book's last law.** It sits two levels inside the step:
  - Horner's step is B conditional subtractions, `B[link]` (the machine's B loop runs B times);
  - each conditional subtraction, the link, is a comparison and a select, `STA.S T.lt SEL.S`.

  A source that cut the step in turn, to one conditional subtraction and then to its comparison, would take the book
  to nothing. Not tried. The book less the comparison but with the link (lt off, red and add on) was not run either.
- **Anchored pairs need a look that sees the row's worth.** Here every look was null, so its picks were the rule's tie
  order. The row's worth to the target shows only once the multiply over it and the reduction both exist. A deeper
  look, or a prior that a null look cannot erase (CONSOLIDATE.md), is what it would need.

### 9. Cost (`cost.py`; `logs/cost.txt`, `logs/watch_l5_*.txt`)

- **Billed to 05:00 UTC: $79.64.**
  - Phase 1 $11.65: 02:00 UTC $9.07, 03:00 UTC $2.57.
  - Phase 2 $7.31: 02:00 UTC $5.64, 03:00 UTC $1.67.
  - Addendum 1 $19.16: 03:00 UTC $18.84, 04:00 UTC $0.33.
  - Addendum 2 $41.52: 03:00 UTC $13.85, 04:00 UTC $27.67.
- **Not yet billed:** the hour from 05:00 UTC, the four long arms' last 20 to 43 minutes, about 2.1 B200-hours or
  about $13.
- **About $93 in all,** under the soft $100.
- **B200 time:**
  - the 20 runs' 50,392 s: phase 1 6,307 s, phase 2 3,768 s, addendum 1 10,372 s, addendum 2 29,945 s;
  - about 14.0 B200-hours, plus each container's startup, at round 2's $6.37 an hour.
- **The watcher** stopped all at the cap less one hour of every container that could run at once. It was relaunched
  at each change, without stopping any arm: $27, $41, $48, $70, $95 and $114 (`logs/watch_l5_1.txt` to `_6.txt`).
  - It never stopped anything.
  - Its last estimate was $105.98 at 05:41 UTC. That counts an hour under way for apps already stopped, so it runs
    high.
  - It was stopped by hand at 05:43 UTC, once every app had stopped.
- **Every app of this study is stopped.** The fetches ran no containers and cost nothing. The local tests ran under
  8–10 GB memory caps.

### 10. Files and how to rerun

**The study:**
- `PREREG.md`, `PREREG_ADDENDUM1.md`, `PREREG_ADDENDUM2.md`: read-only, their hashes in §1 and §4.
- `make_copies5.py`: writes `spcore_l.py`, `oracle_l.py`, `core_l.py`, `climb_l.py` and `clean_l.py`, and `hooks.py`
  (round 4's, with the loops). Round 4's law reader (bar3/scale/slice/lawread.py) and round 2's subgoal score are
  imported from their folders.
- `loops_check.py`: the loops against known programs.
- `modal_loops.py`: every arm; app and Volume `bar3-loops`.
- `cost.py`: the billing report's hours for this study's apps.

**Records and logs:**
- `records/<tag>.json` and `.log`: each arm's record and log.
- `logs/`:
  - the tests: `loops_check.txt`, `tiny_loops.txt` (with `tiny_slice_x2.*`, `tiny_loops_x2.*`, `smoke_loops_x3.*`);
  - the hashes: `prereg_sha256.txt`, `addendum1_sha256.txt`, `addendum2_sha256.txt`;
  - the launches and watchers: `launch_*.txt`, `watch_l5_*.txt`;
  - the checks: `ladder_*.txt`, `replays_same.txt`;
  - the cost: `cost.txt`.

**To rerun** (from `cpu_first`):
- **The copies:** `~/github/sotaku/venv/bin/python -m bar3.scale.loops.make_copies5`
- **The loops' check:** `H1_DATA=~/github/one-layer/one-layer-deeper/data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2 PYTHONPATH=$PWD bar3/scale/rowcheck/memcap.sh 8 ~/github/sotaku/venv/bin/python -W ignore -m bar3.scale.loops.loops_check`
- **An arm:**
  1. `~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750`
  2. `~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/loops/modal_loops.py --arms l5_x2`
  3. `~/github/sotaku/venv/bin/python -m modal run bar3/scale/loops/modal_loops.py --mode fetch`
- **A round on the ladders:** `~/github/sotaku/venv/bin/python -m bar3.scale.compose3.verify3 bar3/scale/loops/records/l5_x2.json "T.val.63.tgt.XN STA.X T.red.x+on.XN"`
- **A record in short:** `~/github/sotaku/venv/bin/python -m bar3.scale.slice.summary bar3/scale/loops/records/l5_x2.json`
