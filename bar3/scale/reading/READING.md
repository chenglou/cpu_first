# Composition round 6: everything from the target's reading

<!-- PLAIN -->
## In plain words

**The question.**
- Round 5 (bar3/scale/loops/LOOPS.md) posed each cut's loop. x²'s round came with the book down to one law, the
  comparison. x³ and 10·x² + 1 got the multiply and the reduction as routines, but no round.
- This round takes four steps, each from the target's own law reading and behind its own flag:
  1. **The book to nothing:** cut Horner's step down to its comparison, so that the reading poses the comparison
     itself (DOWN).
  2. **The cube as two calls:** pose the reading's repeated sub-law, x·o mod N, as a law of its own (SUBLAW).
  3. **10·x² + 1 with its constant:** supply the + 1 from the data's constants, as the law reader reads them (CONST).
  4. **Anchored pairs with a prior** that a null look cannot erase (PRIOR).

**The source** (`hooks.py`: reading_pose, reading_target). Once the target's law is read, beyond round 5's cuts and
loops:
- **DOWN:**
  - the step's own cut, the link (one conditional subtraction, `STA.S T.lt SEL.S`);
  - the link's piece, the comparison (`CLC P[SBB.X INC]`).
- **SUBLAW:** a two-operation sub-law the reading holds twice, mod(mul(a, b), N), posed as the law (x·o) mod N.
- **CONST:**
  - each constant of the reading, made from N. This machine has none and starts dirty: a 1 is the borrow of 0 − N,
    added to 0;
  - the reading's operations with that constant (x + 1, x·10).
- **PRIOR:** under anchored pairs, ties among equal gains go first to the reading's own concepts.

**What came** (one B200 each, round 5's settings; every round right on both of H1's ladders, T = 1 to 64, 768/768 on
all 14 rungs):

| arm | what | the round |
|---|---|---|
| `r6_x2_nobook` | x², **no book**, the link and the comparison | **yes, level 2** (548 s) |
| `r6_x2_nobook_cmp` | the comparison alone | **yes, level 2** (423 s) |
| `r6_x2_nobook_link` | the link alone | no: the link was never learned, so there was no step |
| `r6_x2_nobook_bare` | no book, every one of round 2's flags off | **yes, level 2** (587 s), by calls |
| `r6_x3` | x³, the book lt;red;add, the sub-law | **yes, level 3** (2,346 s): two calls of the sub-law |
| `r6_x3_nobook` | x³, no book, the sub-law | **yes, level 3** (2,283 s), the same |
| `r6_10x2p1`, `r6_10x2p1_nobook` (and 7,200 s) | 10·x² + 1, the constant | no: make 1 at level 0 and x + 1 at level 1, then no route |
| `r6_x2_lessons_prior`, `r6_x2_lessons_nobook` (and 7,200 s) | anchored pairs, the prior or no book | no: the row at level 0, the multiply at level 2, never the step |
| `replay_l5_x2` | round 5's l5_x2, every new flag off | round 5's record again |

**How x²'s round came with no book** (`r6_x2_nobook`).
1. Level 0: the row from the multiply's cut; **the comparison from the step's cuts**.
2. Level 1: the multiply from its loop; the step over the comparison; the link.
3. Level 2: the reduction from its loop. Then the round: x², and the reduction.

These are round 5's levels and round with the book. Nothing of the book is left. Every law practised came from the
target: its reading's cuts, their loops and the step's cuts, and bar 3's values from its rows. The comparison has to be posed itself. With the link alone, the link (6 tokens over the
primitives) was never learned.

**The cube as two calls.**
- The sub-law (x·o) mod N came at level 3, 3 tokens over the multiply and the reduction. The round followed:
  `LDA.X STA.Y T.mod.mul STA.X T.mod.mul`, the sub-law on (x, x), then on (x² mod N, x), with x kept in Y.
- With the book and without it. That is round 5's missing step: the cube needed x kept beside x² mod N, 4 calls, and
  the sub-law makes it 2.
- **Slower:** the sub-law's 24 bindings make each level heavier, and the rounds came 54 and 117 s before the
  deadline.

**10·x² + 1 with its constant: the constant comes, the round does not.**
- Make 1 came at level 0, straight from the primitives, and x + 1 at level 1.
- But x² mod N never came as a value, even in 10 levels at 7,200 s. The values went around x + 1 instead.
- The round stays long: x² mod N, ×10, + 1, mod N is 4 calls. x + 1 was practised only in bindings that make the
  route 8 tokens or more, past the round search.
- **A caveat:** one learned make 1 relies on N's low digit or on the starting carry, and is wrong on 4–6% of the
  rows at widths 1 and 2. It is right at full size on this data.

**Anchored pairs: the row first, then stuck on its bindings.**
- No look found a program, so every gain tied. With the prior, or with no book, the row now comes at level 0 (round
  5: level 3).
- But anchored pairs anchor on the costliest learnable bindings, and those were the row's own others. They took
  about three picks in four.
- The multiply came at level 2. The comparison came late, in bindings the step does not use, and the step never came
  (7,200 s, 5 levels).

**What it means.**
- **For x², the book of laws is gone:** the target's reading poses every law on the way, down to the comparison.
- **For the cube,** the reading's repeated sub-law turns a 4-call composition into 2.
- **What remains hand-given** is the source's own tables: each operation's cut, each cut's loop, the step's cuts, and
  which sub-law to pose.
- **10·x² + 1** needs a composition past 2 calls, or a split of its reading that makes it 2 (the inner sub-law and the
  rest of the way). Supplying the constant is not enough.
- **Anchored pairs here** need anchors taken one binding per concept, not just a tie-break.
- **Cost:** about $72 of the soft $100 (§9).
<!-- /PLAIN -->

## Details

### 1. Pre-registered

PREREG.md (frozen 2026-10-26 23:18:53 PDT, read-only, sha256
`62469fa7cdc21195e0568b8aac70d3d8968b96cf8952f713c5c9db626131fe84`, `logs/prereg_sha256.txt`), before any arm ran,
unchanged:

<!-- PREREG -->
# Composition round 6: everything from the target's reading, pre-registered

Written 2026-10-26, before any run of this study. READING.md §1 repeats this text unchanged and records the time and
hash it was frozen with.

## The questions

**What round 5 found** (bar3/scale/loops/LOOPS.md; 355f62f):
- With each cut's loop posed, x²'s round came at level 2 with the book down to the comparison (lt) alone.
- With no book at all, the multiply came, but nothing posed the comparison, so there was no step, no reduction and no
  round.
- x³ and 10·x² + 1 got the reduction and the multiply as routines, but no round, even at 7,200 s:
  - the cube needs x kept beside x² mod N: 4 calls, past the round search's 2;
  - 10·x² + 1 reached x² mod N as a value, but this machine has no constant for its + 1.
- Anchored pairs saw nothing at any level (every look null), so their picks went by the rule's tie order.

**This round, in the coordinator's order** (each change behind its own flag):
1. **The book to nothing (DOWN).** Horner's step is cut down to its comparison, so a source poses the comparison
   itself. With no book, x² mod N: does the comparison come, then the step, the reduction and the round?
2. **The cube as two calls (SUBLAW).** The reading's repeated sub-law, x·o mod N, posed as a law of its own.
3. **10·x² + 1 with its constant (CONST).** The + 1 supplied from the data's own constants, which the law reader
   already takes from the data.
4. **A prior for anchored pairs (PRIOR)** that a null look cannot erase: lower priority, since it is cheap.

## Theory, in brief

1. **The step's own cuts (DOWN).**
   - Horner's step, (B·r + d) mod N, is B conditional subtractions: `B[link]` (the machine's B loop runs B times).
   - The link, one conditional subtraction, is a comparison and a select: `STA.S T.lt SEL.S`.
   - The comparison is the borrow of a − b: `CLC P[SBB.X INC]`.
   - **Posed as cuts, answered by Python in every binding:**
     - the link, red on pair: a − b if a ≥ b, else a, on the rows where that is one conditional subtraction (b > 0
       and a < 2b). The book's red on pair is its rows with a ≤ b;
     - the comparison, lt on pair: the carry a < b, the book's lt on pair's answers.
   - **Costs:** the comparison is 4 over the primitives. The link is 3 over the comparison, or 6 over the primitives.
     The step is 2 over the link, or 4 over the comparison.
   - **With both posed, the book is empty.** The comparison should come at level 0, the step and the multiply at
     level 1, the reduction from its loop at level 2, and the round at level 2 or 3, as round 5's book lt alone
     (`l5_x2_less_link`).
   - **The link alone is past the probes.** Its 6 tokens over the primitives lie beyond what the probes reach (the
     1M probe ends inside cost 4 or 5). So there is no link, no step and no round: the comparison has to be posed
     itself.
2. **The repeated sub-law (SUBLAW).**
   - The cube's reading, mod(mul(mod(mul(X,X),N),X),N), holds mod(mul(a, b), N) twice. It is posed as a law of its
     own: mod.mul on (x, o, n), (x·o) mod N, answered by Python.
   - Over the multiply and the reduction it is 3 tokens (`T.mul.x,o.XY STA.X T.red.x+on.XN`), so it comes the level
     after the reduction.
   - **The cube is then 5 tokens, or 2 calls,** inside the round search: `LDA.X STA.Y T.mod.mul.x,o,n.XYN STA.X
     T.mod.mul.x,o,n.XYN`. With known routines this program is right on both of H1's ladders (768/768 on all 14
     rungs).
   - x² and 10·x² + 1 hold the sub-law once each, so SUBLAW poses nothing for them.
3. **The constant (CONST).**
   - 10·x² + 1's reading, mod(add(mul(mod(mul(X,X),N),10),1),N), holds the constants 10 and 1.
   - This machine has no constant, and its registers start dirty. A 1 is made from N: the borrow of 0 − N (N ≥ 1),
     added to 0, `CLA STA.S T.lt.pair.AN CLA ADC.S`. That is 5 tokens over the comparison, 8 over the primitives.
   - **So the source poses, for each constant c of the reading:**
     - c made from N (const.c on n);
     - the reading's operations with c as an operand (add.c and mul.c on x,n: x + 1 and x·10). x·10 is `SHL.A` at
       base 10, trivial.
   - x + 1 is then 3 tokens: `STA.X T.const.1.n.N T.add.pair.AX`.
   - **The round stays long.** x² mod N, then ×10, then + 1, then mod N is 4 calls, or 9 tokens and more, past the
     round search.
   - It comes only if x² mod N comes as a value V, as in round 5 at level 3: then `V SHL.A T.add.1.x,n.AN STA.X
     T.red.x+on.XN`, 5 tokens. With mod.mul in V's place, the program is right on both ladders.
4. **Anchored pairs' prior (PRIOR).**
   - In round 5 every look was null, so every gain tied and the pool's order picked.
   - PRIOR breaks ties by putting the reading's own concepts first: its cuts and their loops, the step's cuts, its
     sub-laws and constants. Then the row goes with the first anchor at level 0.
   - **But the anchors stay the 2 costliest in pool order,** and the comparison's bindings, a book law, rank below the
     reading's own. So the step may wait for the comparison's bindings.
   - **With no book, every law is the reading's.** The pool starts with the row, so the row should come at level 0
     without the prior.

## Design (`make_copies6.py`, `hooks.py`)

- **The copies:** round 5's (make_copies5.py's lists), their paths moved to this folder, with this study's changes,
  each asserted to match once:
  - hooks.py: `reading_pose` after the loops, adding to the cuts what the reading poses. Each new law also goes in
    the law book, so that a concept practised in one binding counts as done. `reading_target` gives their answers;
  - spcore_r.py: the new patterns, with SUBLAW or CONST on: x,o,n (double width), n (full width) and x,n (carry).
    Also PRIOR's tie-break in anchored pairs' choice;
  - core_r.py: the flags in the run's config.

  With every flag off the copies are round 5's.
- **The flags:**
  - DOWN=1 poses the link and the comparison; DOWN=link or DOWN=cmp poses one;
  - SUBLAW=1 poses mod.mul where the reading holds it twice or more;
  - CONST=1 poses the reading's constants and their operations;
  - PRIOR=1 turns on the tie-break.

  A law the book poses itself is left to the book.
- **Checked before this was frozen** (`reading_check.py`, `logs/reading_check.txt`, `logs/known_routes.txt`):
  - the reading's parts on the three readings;
  - 20 programs against the new laws' probes and acceptors at every verify width, right ones passing and wrong ones
    failing;
  - the three routes above on both ladders, with known routines.

## Runs

Each on one B200, with round 5's settings:
- TARGET small, ROUND_TOO, NP 8, deep probes over whole concepts, SECONDS 2,400, LEVELS 10;
- round 2's flags and TIEFOLLOW;
- SLICE=1 and LOOPS=1.

No book means no law source: SOURCES bar 3's four only.

**Arms, launched together, at most 8 B200s at once (the first 8 in this order start, the rest as containers free):**
1. `replay_l5_x2`: round 5's l5_x2, every new flag off;
2. `r6_x2_nobook`: x², no book, DOWN=1;
3. `r6_x3`: x³, the book lt;red;add (as `l5_x3_less_mul`), SUBLAW=1;
4. `r6_10x2p1`: 10·x² + 1, the book lt;red;add (as `l5_10x2p1_less_mul`), CONST=1;
5. `r6_x2_nobook_cmp`: x², no book, DOWN=cmp;
6. `r6_x2_nobook_link`: x², no book, DOWN=link;
7. `r6_x3_nobook`: x³, no book, DOWN=1, SUBLAW=1;
8. `r6_10x2p1_nobook`: 10·x² + 1, no book, DOWN=1, CONST=1;
9. `r6_x2_nobook_bare`: `r6_x2_nobook` with every one of round 2's flags off;
10. `r6_x2_lessons_prior`: anchored pairs (ct2a2), SUBGOAL and REST_DEEP off (round 5's lessons arm), PRIOR=1;
11. `r6_x2_lessons_nobook`: anchored pairs, SUBGOAL and REST_DEEP off, no book, DOWN=1.

**Any round** is checked on both of H1's ladders to T = 64.

**The tiny tests, after this is frozen and before the arms,** locally under a memory cap:
- every flag off: round 5's tiny run again (`tiny_loops_x2`);
- DOWN, x² with no book, 3 levels;
- SUBLAW, x³, and CONST, 10·x² + 1, 1 level each: what is posed;
- PRIOR under ct2a2, 1 level.

A fix that changes no design is recorded. A change of design goes in an addendum, frozen before the arms.

**Money:**
- soft $100, cap $140; `spend.py 4750` before each launch;
- one watcher at the cap less one hour of 8 B200s: stop all at $89, raised as arms end;
- expected cost about $45.

## Predictions

| what | prediction | confidence |
|---|---|---|
| replay_l5_x2 | round 5's record again, apart from times: its round at level 2 | high |
| r6_x2_nobook: the comparison | practised at level 0, from the step's cuts | high |
| r6_x2_nobook: the step and the reduction | the step by level 1, the reduction by level 2 | medium-high |
| r6_x2_nobook: the round | by level 3, right on both ladders to T = 64 | medium-high |
| r6_x2_nobook_cmp | the round by level 3 | medium-high |
| r6_x2_nobook_link | no link, no step, no round | medium-high |
| r6_x2_nobook_bare | the round by level 3 | medium |
| r6_x3: the sub-law | practised by level 3 | medium-high |
| r6_x3: the round | by level 4, two calls of the sub-law, right on both ladders | medium |
| r6_x3_nobook | the round by level 5 | medium |
| r6_10x2p1: the constant | make 1 by level 2, x + 1 by level 3 | medium |
| r6_10x2p1: the round | by level 10 | low |
| r6_10x2p1_nobook | the round by level 10 | low |
| r6_x2_lessons_prior: the row | practised at level 0 | medium |
| r6_x2_lessons_prior: the round | by level 10 | low |
| r6_x2_lessons_nobook: the row | practised at level 0 | medium |
| r6_x2_lessons_nobook: the round | by level 10 | low-medium |
| the conclusion | the reading poses the whole route for x² with no book. For the cube it poses the sub-law that makes the round two calls. For 10·x² + 1 it supplies the constant, but the round stays long | medium |
<!-- /PREREG -->

PREREG_ADDENDUM1.md (frozen 2026-10-27 00:31:22 PDT, read-only, sha256
`4923c642e6e8cb9b1577e54113a8de584e8ce9dadd46584fa7332f0d8b741bbd`, `logs/addendum1_sha256.txt`), after the
pre-registration's eleven arms and before its own, unchanged:

<!-- ADDENDUM1 -->
# Composition round 6, addendum 1: the arms the deadline cut, given 7,200 s, pre-registered

Written 2026-10-27, after the pre-registration's eleven arms and before any run below. READING.md §1 records its time
and hash.

## What the eleven arms showed

**x² with no book got the round,** right on both of H1's ladders to T = 64:
- `r6_x2_nobook`: the comparison from the step's cuts at level 0; the step, the link and the multiply at level 1;
  the reduction from its loop at level 2; the round at level 2 (548 s);
- `r6_x2_nobook_cmp`, the comparison alone: the round at level 2 (423 s);
- `r6_x2_nobook_bare`, every one of round 2's flags off: the round at level 2, by calls (587 s);
- `r6_x2_nobook_link`, the link alone: the link was never learned, so there was no step and no round (the deadline in
  level 7).

**x³ with the sub-law got the round:** mod.mul at level 3, then the round, two calls of it (`LDA.X STA.Y
T.mod.mul.x,o,n.XYN STA.X T.mod.mul.x,o,n.XYN`). With the book lt;red;add it came at 2,346 s, and with no book at
2,283 s, each right on both ladders.

**10·x² + 1 with its constant** (`r6_10x2p1`, `r6_10x2p1_nobook`, the same picks through level 4):
- make 1 at level 0, from the primitives;
- x + 1 at level 1, in two bindings (x in X or in N, N in Y);
- the reduction at level 2;
- then values built on x + 1; x² mod N never came as a value;
- the deadline in level 5, no round.

**Anchored pairs** (`r6_x2_lessons_prior`, `r6_x2_lessons_nobook`):
- every look was null again, so every gain tied;
- the row came at level 0, with the prior and with no book;
- at level 1 the tie order took more of the row's bindings;
- at about 1,100 s a level, the deadline came in level 2.

**The replay** gave round 5's record again, apart from times.

## Arms

Each on one B200, launched together. The pre-registration's settings, with SECONDS 7,200 (the runner's container
timeout raised to 9,000 s):
- `r6_10x2p1_nobook_long`: `r6_10x2p1_nobook`. The book arm took the same picks through level 4, so one long arm
  stands for both;
- `r6_x2_lessons_prior_long`: `r6_x2_lessons_prior`;
- `r6_x2_lessons_nobook_long`: `r6_x2_lessons_nobook`.

Any round is checked on both ladders to T = 64.

## Predictions

| what | prediction | confidence |
|---|---|---|
| r6_10x2p1_nobook_long | no round by level 10. x + 1 is practised only in bindings that make the round 8 tokens or more, and lpca poses no practised concept's other bindings | medium |
| r6_x2_lessons_prior_long | the multiply by level 4; the step later, since its comparison's bindings are book laws, which the prior ranks last; no round by the deadline | low-medium |
| r6_x2_lessons_nobook_long | the multiply by level 4; no round by the deadline: the anchors stay on the costliest concepts' bindings, and the comparison comes last in pool order | low-medium |

## Money

- Three B200s. The watcher's total is the cap ($140) less one hour of three: stop all at $120.
- This study so far: about $35. With these, about $75, under the soft $100.
<!-- /ADDENDUM1 -->

### 2. Design

#### 2.1 What the reading poses (`hooks.py`: STEP_CUTS, SUBLAWS, reading_pose, reading_target)

| flag | from the reading | posed as | answered by Python | a known program |
|---|---|---|---|---|
| DOWN | the posed step, B conditional subtractions | the link: red on pair | a − b if a ≥ b, else a, where b > 0 and a < 2b | `STA.S T.lt.pair.AX SEL.S` |
| DOWN | the link, a comparison and a select | the comparison: lt on pair | the carry a < b | `CLC P[SBB.X INC]` |
| SUBLAW | mod(mul(a, b), N), twice in the cube's reading | mod.mul on x,o,n | (x·o) mod N | `T.mul.x,o.XY STA.X T.red.x+on.XN` |
| CONST | a constant c of the reading | const.c on n | c | `CLA STA.S T.lt.pair.AN CLA ADC.S` (c = 1) |
| CONST | an operation of the reading with c as an operand | add.c and mul.c on x,n | x + c, x·c | `STA.X T.const.1.n.N T.add.pair.AX` (x + 1) |

- **Where:** `reading_pose` runs once the target's law is read (level 0), after the cuts and their loops, and adds to
  the posed cuts. Each new law also goes in the law book, so that a concept practised in one binding counts as done.
  A law the book poses itself is left to the book.
- **DOWN** walks the step's cuts from every posed cut: the step's is the link, the link's is the comparison. It poses
  the ones it names (DOWN=1: both).
- **SUBLAW** counts the reading's two-operation sub-laws outer(inner(a, b), N) and poses the ones it holds twice or
  more that it knows (mod(mul): mod.mul on x,o,n).
- **CONST** poses each constant leaf c of the reading, made from N, and each of the reading's add and mul with c as an
  operand.
- **The new patterns** (with SUBLAW or CONST on):
  - x,o,n at double width, so x·o fits the registers before its reduction;
  - n at full width: N alone, the input a constant is made from (0 − N borrows);
  - x,n at carry width: x + c and x·c, with N beside them.
- **The answers** must vary, except a constant's. A row outside a law's range (the link where one subtraction is not
  enough) is masked.
- **PRIOR:** under anchored pairs (ct2a2), ties among equal gains go first to the pairs and singles with fewer members
  that are not the reading's own. With PRIOR off the tie order is round 5's.

#### 2.2 The copies (`make_copies6.py`)

Round 5's copies (make_copies5.py's lists, on round 4's and round 3's), their module paths moved to this folder,
with this study's asserted replacements (`logs/diff_*.txt` against round 5's copies):
- **spcore_r.py:** the new patterns, with SUBLAW or CONST on, and PRIOR's tie-break in ct2a2's choice (3);
- **core_r.py:** the flags in the run's config (1);
- **hooks.py:** `reading_pose` after the loops, and `reading_target` for the new laws' answers (2), with the reading
  source appended.

The law reader (`bar3/scale/slice/lawread.py`) and round 2's subgoal score are imported from their folders.

### 3. The checks and tests before the arms (local, under a memory cap)

**Before the freeze** (they set the design):

| check | what it showed |
|---|---|
| the reading's parts (`reading_check.py`, `logs/reading_check.txt`) | x²'s reading holds mod(mul) once, the cube's twice, 10·x² + 1's mod(add) and mod(mul) once each. 10·x² + 1's constants are 10 (an operand of mul) and 1 (of add) |
| the new laws (`reading_check.py`) | 20 of 20 programs right: the comparison, the link (over the comparison or the primitives), the step over the link, the sub-law over the multiply and the reduction, make 1 (over the comparison or the primitives), make 10 over make 1, x + 1 over make 1 and the addition, and x·10 pass their laws' probes and acceptors at every verify width. The unselected subtraction, the unreduced product, one subtraction a digit, a dirty S and carry, and the like, fail |
| the routes (`logs/known_routes.txt`, verify3 with the known routines in `logs/known_lib.json`) | 768/768 on all 14 rungs of both ladders, T = 1 to 64, for the cube as two calls of the sub-law, for 10·x² + 1 as the sub-law, ×10, + 1 and the reduction, and for x² as one call of the sub-law |

**After the freeze** (`logs/tiny_reading.txt`):

At tiny budgets, one at a time; no fix was needed:

| test | what it showed |
|---|---|
| every flag off (`tiny_loops_x2.*`) | round 5's tiny run again, apart from times |
| DOWN, x² with no book, 3 levels (`tiny_down_x2.*`) | the cuts with the link and the comparison posed; the comparison at level 0, the link at 1, the step at 2. No errors |
| SUBLAW, x³, 1 level (`smoke_sublaw_x3.*`) | mod.mul on x,o,n posed. No errors |
| CONST, 10·x² + 1, 1 level (`smoke_const_10x2p1.*`) | make 10, x·10, make 1 and x + 1 posed. x·10 was practised in three bindings at 2 tokens (`LDA.X SHL.A` and the like); it is trivial only with x in A. No errors |
| PRIOR under ct2a2, 1 level (`smoke_prior_x2.*`) | the comparison and a value, at tiny budgets. No errors |

### 4. The runs (`modal_reading.py`; `records/*.json` and `.log`; `logs/launch_*.txt`)

| launch | PDT | app | arms | B200s at once |
|---|---|---|---|---|
| phase 1 | 23:31 | `ap-oIB3YFGtYq3qnsnzwCl4Mu` | the pre-registration's eleven, in its order | 8 (max_containers 8) |
| addendum 1 | 00:31 | `ap-N1VXE7hldSJfUB3cYRT25H` | r6_10x2p1_nobook_long, r6_x2_lessons_prior_long, r6_x2_lessons_nobook_long | 3 |

- **The launch** came after `spend.py 4750` (LAUNCH OK), 13 minutes after the pre-registration was frozen (23:18:53),
  once the tiny tests had passed. The first eight arms started at once, the other three as containers freed.
- **The watcher** (bar3/scale/digits/watch_spend.py) stopped all at $89: the cap less one hour of 8 B200s
  (`logs/watch_r6_1.txt`).
- **Addendum 1** was frozen at 00:31:22 PDT (sha256 `4923c642e6e8cb9b1577e54113a8de584e8ce9dadd46584fa7332f0d8b741bbd`),
  14 seconds before its launch, which came after `spend.py 4750` (LAUNCH OK). Its arms had SECONDS 7,200, and the
  runner's container timeout was raised to 9,000 s. A new watcher, over both apps, stopped all at $120: the cap less
  one hour of 3 B200s (`logs/watch_r6_2.txt`).
- **No change from the pre-registration** but the addendum's arms.

### 5. Outcomes (`records/*.json`; bar3/scale/slice/summary.py)

#### 5.1 x² with no book: the reading poses the whole route, and the round comes at level 2 (548 s)

`r6_x2_nobook`: no law source (SOURCES bar 3's four only), the cuts and their loops, and DOWN=1.

| level | practised (lpca) | from |
|---|---|---|
| 0 | the row in four bindings; **the comparison** in three (`CLC P[SBB.X INC]`, AX, AY, AN; cost 4); a value | the mul cut; **the step's cuts** |
| 1 | the multiply in two bindings (5 over the row); **the step** over the comparison in three (`B[STA.S T.lt.pair.AX SEL.S]`); **the link** in three (`STA.S T.lt.pair.AX SEL.S`); a value | the multiply's loop; the mod cut; the step's cuts |
| 2 | **the reduction** in two bindings (`LDA.N P[BRG.X T.red.Bx+d.AN]`, `LDA.Y P[BRG.X T.red.Bx+d.AY]`); x² (`LDA.X STA.Y T.mul.x,o.XY`); a restriction; x·d | the reduction's loop; bar 3's values |

- **The round** came in level 2's target search: `T.val.63.tgt.XN STA.X T.red.x+on.XN` (cost 3), x² and then the
  reduction. These are `l5_x2`'s levels and its round, with nothing of the book's.
- **The ladders:** 768/768 on all 14 rungs of both, T = 1 to 64 (`logs/ladder_r6_x2_nobook.txt`).

#### 5.2 The step's cuts one at a time, and every flag off

| arm | change from `r6_x2_nobook` | the round | right on both ladders |
|---|---|---|---|
| `r6_x2_nobook_cmp` | the comparison alone (DOWN=cmp) | **level 2** (423 s): the step over the comparison at level 1, as in `l5_x2_less_link` | 768/768 on all 14 rungs |
| `r6_x2_nobook_link` | the link alone (DOWN=link) | no (the deadline in level 7) | – |
| `r6_x2_nobook_bare` | every one of round 2's flags off | **level 2** (587 s), from the round search by calls: `LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA` | 768/768 |

- **The comparison has to be posed itself.** With the link alone, the link was never learned: its 6 tokens over the
  primitives are past the probes. So there was no step and no reduction. The row, the multiply, x·d and values came,
  levels 3 to 6 only values (`logs` and `records/r6_x2_nobook_link.*`).
- **The link is not needed:** with the comparison alone, the step came over the comparison directly.

#### 5.3 The cube with its sub-law: two calls, at level 3

| arm | the book | the sub-law | the round | right on both ladders |
|---|---|---|---|---|
| `r6_x3` | lt;red;add (as `l5_x3_less_mul`) | level 3: `T.mul.x,o.YX STA.X T.red.x+on.XN` (XYN, cost 3) | **level 3** (2,346 s), from the round search | 768/768 on all 14 rungs |
| `r6_x3_nobook` | none (DOWN=1) | level 3, the same program | **level 3** (2,283 s), from the small target's search | 768/768 |

- **The round** in both: `LDA.X STA.Y T.mod.mul.x,o,n.XYN STA.X T.mod.mul.x,o,n.XYN`. That is the sub-law on (x, x),
  then on (x² mod N, x), with x kept in Y.
- **The levels before it** were `l5_x3_less_mul`'s: the row and the comparison (0); the multiply, the step, the link
  and x·d (1); the reduction (2, and the book's addition with the book).
- **Slower levels:** the sub-law is posed in all 24 bindings of x,o,n from level 0. Levels 2 and 3 took about 1,000 and
  750 s, so the rounds came 54 and 117 s before the 2,400 s deadline.

#### 5.4 10·x² + 1 with its constant: the constant comes, the round does not (the deadline in level 5)

`r6_10x2p1` (the book lt;red;add) and `r6_10x2p1_nobook` (no book, DOWN=1) took the same picks at every level.

- **Level 0:** the row, the comparison, add's cut, and **make 1** (const.1 on n) from the primitives in two
  bindings:
  - Y: `P[SBB.Y CLA SHR.Y] ADC.Y`. It carries a borrow up through N's digits;
  - N: `CLA STA.S SBB.N SEL.S ADC.S`. One place's borrow, with the carry the machine starts with.
- **Level 1:** the multiply, the step, the link, and **x + 1** (add.1 on x,n) over make 1 and add's cut, in two
  bindings:
  - XY: `CLA T.const.1.n.Y T.add.x,d.AX`, with x in X and N in Y;
  - NY: with x in N.
- **Levels 2–4:**
  - the reduction (2);
  - then values built on x + 1, such as `SHL.X T.add.1.x,n.XY T.red.Bx+d.AN` and `T.mul.x,o.YX T.add.1.x,n.XY
    T.red.Bx+d.AN LDA.S`, and on those;
  - one subgoal by table (level 2);
  - no LAWNEAR state.
- **No round:**
  - x² mod N never came as a value. In round 5 it came at level 3; here the values went around x + 1.
  - With x + 1 in the bindings practised, the round from such a value is 8 tokens or more: it moves N into Y. lpca
    poses no practised concept's other bindings, so x + 1 with x in A and N in N was never practised.
- **One routine is wrong on a few rows** (`logs/const_robust.txt`). Make 1 in binding N relies on N's low digit or
  the starting carry. It fails on about half the rows where N's low digit is 0: 6% of rows at width 1 and 4% at
  width 2, none at width 3 or at full size, where this data's moduli never end in 0. It was practised all the same
  (how it passed acceptance at widths 1 and 2 is not looked into here).
  - The routine in binding Y is right on every row.
  - No round used either one.

#### 5.5 Anchored pairs: the row at level 0, then the row's other bindings (the deadline in level 2)

| arm | level 0 | level 1 |
|---|---|---|
| `r6_x2_lessons_prior` (the book lt;red;add, PRIOR=1) | the comparison in one binding (XY, the first anchor), **the row** in three | the link over that comparison, and three more of the row's bindings |
| `r6_x2_lessons_nobook` (no book, DOWN=1) | **the row** in four bindings | four more of the row's bindings |

- **No look found a program,** in either arm, at either level, and every pair's gain tied (the cost of two extra
  symbols).
  - At the prior arm's level 1, three single looks did improve the target's best partial fit, by 12 bits: the row in
    binding AYN, x·d in XA, and a value.
  - But anchored pairs fill their picks from the pairs first, so the tie order among the pairs picked.
- **The prior did what it was made for:** among the tied gains, the pairs with the reading's own concepts went first,
  so the row came at level 0, three levels earlier than in round 5's long arm.
- **With no book, pool order alone did it:** the reading's cuts come first in the pool, the row's bindings among them.
- **Then the anchors stayed on the row.** Anchored pairs pose a practised concept's other bindings. The row's cost-5
  bindings were the costliest learnable concepts, so they anchored the pairs again, and the tie order took them.
- **Each level took about 1,100 s** (about 100 lookaheads at about 9 s each), so the deadline came in level 2.

#### 5.6 The replay: round 5's record again

`replay_l5_x2` (round 5's l5_x2 on this study's copies, every new flag off): the same record, apart from times
(`same.py`; `logs/replays_same.txt`), its round at level 2 in 448 s.

#### 5.7 Addendum 1's long arms: 7,200 s, and still no rounds

| arm | how far | what came |
|---|---|---|
| `r6_10x2p1_nobook_long` | 10 levels, the level cap (6,714 s) | through level 4, the 2,400 s arm's picks; then values built on x + 1 and on the step (`T.val.… T.red.Bx+d.AN LDA.S`, `T.val.… SHR.A SEL.S` and the like) to level 9, and make 10 (`T.const.1.n.Y SHL.A`) at level 7. x² mod N never came as a value; no LAWNEAR state; the one subgoal at level 2 |
| `r6_x2_lessons_prior_long` | 5 levels, the deadline in level 5 (7,203 s) | the row (0); more of its bindings and the link over the comparison in binding XY (1); **the multiply** and two more of the row's bindings (2); the comparison in binding AX and three more of the row's (3); the link in binding YA and three more of the row's (4). LAWNEAR found x² from level 3 (`LDA.X STA.Y T.mul.x,o.XY`). No step, no reduction |
| `r6_x2_lessons_nobook_long` | 5 levels, the deadline in level 5 (7,204 s) | the row (0); four more of its bindings (1); **the multiply** and two more of the row's (2); the comparison in bindings XY and XN and two more of the row's (3); four more of the row's (4). No step, no reduction |

- **Anchored pairs spent their picks on the row's bindings:** about three picks of every four, 14 and 16 of the
  row's 24 bindings by level 4.
- **The step never came.** It is posed in the bindings AX, AY and AN, over the comparison in the same binding. The
  comparison came late and mostly in other bindings (XY, XN). When the prior arm had it in AX (level 3), the next
  level's picks went to the row and the link instead.
- **Every pair's gain tied at every level,** in both: no look found a program.

### 6. Predictions against outcomes

**The pre-registration's:**

| what | prediction | confidence | outcome |
|---|---|---|---|
| replay_l5_x2 | round 5's record again, apart from times: its round at level 2 | high | **right** (`logs/replays_same.txt`) |
| r6_x2_nobook: the comparison | practised at level 0, from the step's cuts | high | **right**: three bindings at level 0 |
| r6_x2_nobook: the step and the reduction | the step by level 1, the reduction by level 2 | medium-high | **right**: level 1 and level 2 |
| r6_x2_nobook: the round | by level 3, right on both ladders to T = 64 | medium-high | **right**: level 2; 768/768 on all 14 rungs |
| r6_x2_nobook_cmp | the round by level 3 | medium-high | **right**: level 2 |
| r6_x2_nobook_link | no link, no step, no round | medium-high | **right**: the deadline in level 7 |
| r6_x2_nobook_bare | the round by level 3 | medium | **right**: level 2, by calls |
| r6_x3: the sub-law | practised by level 3 | medium-high | **right**: level 3 |
| r6_x3: the round | by level 4, two calls of the sub-law, right on both ladders | medium | **right**: level 3; 768/768 on all 14 rungs |
| r6_x3_nobook | the round by level 5 | medium | **right**: level 3 |
| r6_10x2p1: the constant | make 1 by level 2, x + 1 by level 3 | medium | **right**, and sooner: make 1 at level 0, x + 1 at level 1 |
| r6_10x2p1: the round | by level 10 | low | **not borne out**: no round in its 5 levels, nor in the 10 levels of the long arm (§5.7) |
| r6_10x2p1_nobook | the round by level 10 | low | **not borne out**: no round in 5 levels, and none in 10 at 7,200 s |
| r6_x2_lessons_prior: the row | practised at level 0 | medium | **right** |
| r6_x2_lessons_prior: the round | by level 10 | low | **not borne out**: no round in its 2 levels, nor in 5 at 7,200 s |
| r6_x2_lessons_nobook: the row | practised at level 0 | medium | **right** |
| r6_x2_lessons_nobook: the round | by level 10 | low-medium | **not borne out**: no round in 2 levels, nor in 5 at 7,200 s |
| the conclusion | the reading poses the whole route for x² with no book; for the cube the sub-law that makes the round two calls; for 10·x² + 1 it supplies the constant, but the round stays long | medium | **right** in each part |

The round predictions for 10·x² + 1 and anchored pairs had low confidence. The runs never reached level 10 within
2,400 s, and the long arms show no round either.

**Addendum 1's:**

| what | prediction | confidence | outcome |
|---|---|---|---|
| r6_10x2p1_nobook_long | no round by level 10: x + 1 is practised only in bindings that make the round 8 tokens or more | medium | **right**: 10 levels, no round |
| r6_x2_lessons_prior_long | the multiply by level 4; the step later; no round by the deadline | low-medium | **right**: the multiply at level 2; no step; no round (the deadline in level 5) |
| r6_x2_lessons_nobook_long | the multiply by level 4; no round by the deadline: the anchors stay on the costliest concepts' bindings, and the comparison comes last in pool order | low-medium | **right**: the multiply at level 2, the comparison at level 3 in two bindings the step does not use; no round |

### 7. Dead ends, and how hard each was tried

| what | how hard | where it stopped |
|---|---|---|
| the link alone, no comparison (`r6_x2_nobook_link`) | 7 levels, to the 2,400 s deadline | the link's 6 tokens over the primitives are past the probes: no link, so no step and no reduction; levels 3 to 6 only values |
| 10·x² + 1 with its constant (`r6_10x2p1`, `r6_10x2p1_nobook`, `r6_10x2p1_nobook_long`) | 5 levels to the deadline in each 2,400 s arm; then 10 levels, the level cap, at 7,200 s | make 1 and x + 1 came at levels 0 and 1; then values built on x + 1 and on the step, none the target. x² mod N never came as a value, and x + 1 was practised only in bindings that make the round 8 tokens or more |
| anchored pairs with the prior, or with no book (`r6_x2_lessons_prior`, `r6_x2_lessons_nobook` and their long arms) | 2 levels to the deadline at 2,400 s; then 5 levels to the deadline at 7,200 s | no look found a program, so ties decided; the anchors, the costliest learnable bindings, stayed on the row's other bindings. The multiply came at level 2, the comparison in the step's bindings late or not at all, the step never |

### 8. What is left

- **The source still holds tables.** The book of laws is gone for x² and the cube. What the source knows instead is
  how each operation breaks down on this machine:
  - CUTS: each operation's cut at one digit place;
  - LOOP_OF: each cut's loop;
  - STEP_CUTS: the step's own cuts, down to the comparison;
  - SUBLAWS: which repeated sub-law to pose, on which pattern.

  A source that derived the cuts from an operation's own definition would remove these too.
- **10·x² + 1's round needs more than 2 calls,** or a split that makes it 2. Three routes, none tried:
  - a round search past 2 calls: the multiply, the reduction, x + 1 and the reduction again;
  - the reading's split: pose the rest of the way from its inner sub-law, (10·g + 1) mod N on (g, n), as a law of its
    own, and the inner sub-law even when it is held once. Then the round is 2 calls, the sub-law and the rest;
  - x + 1 in the binding the route uses (x in A, N in N): lpca never poses a practised concept's other bindings.
- **Anchored pairs anchor on the costliest bindings.** With every look null, the anchors were the costliest learnable
  bindings, and after the row those were the row's own other bindings. Anchors taken one binding per concept, as lpca
  takes its picks, would move on to the multiply and the step.
- **A practised routine wrong on a few rows** (§5.4): make 1 in binding N fails on 4–6% of the rows at widths 1 and 2.
  How it passed acceptance is not looked into.

### 9. Cost (`cost.py`; `logs/cost.txt`, `logs/watch_r6_*.txt`)

- **Billed to 09:00 UTC: $63.08.**
  - Phase 1 $34.56: 06:00 UTC $23.51, 07:00 UTC $11.05.
  - Addendum 1 $28.52: 07:00 UTC $9.11, 08:00 UTC $19.41.
- **Not yet billed:** the hour from 09:00 UTC, the long arms' last 23 to 32 minutes on three B200s: about 1.45
  B200-hours, about $9.
- **About $72 in all,** under the soft $100.
- **B200 time:**
  - phase 1's eleven runs 18,648 s (448, 548, 2,346, 2,403, 423, 2,400, 2,283, 2,401, 587, 2,400 and 2,407);
  - addendum 1's three 21,121 s (6,714, 7,203 and 7,204);
  - about 11.0 B200-hours, plus each container's startup, at round 2's $6.37 an hour.
- **The watchers:**
  - phase 1's stopped all at $89, the cap less one hour of 8 B200s (`logs/watch_r6_1.txt`). It was stopped by hand
    after its app stopped;
  - addendum 1's, over both apps, stopped all at $120, the cap less one hour of 3 (`logs/watch_r6_2.txt`). It
    exited when every app had stopped.

  Neither stopped anything.
- **Every app of this study is stopped.** The fetches ran no containers. The local checks and tests ran under 8–10 GB
  memory caps.

### 10. Files and how to rerun

**The study:**
- `PREREG.md`, `PREREG_ADDENDUM1.md`: read-only, their hashes in §1.
- `make_copies6.py`: writes `spcore_r.py`, `oracle_r.py`, `core_r.py`, `climb_r.py` and `clean_r.py`, and `hooks.py`
  (round 5's, with the reading source). The law reader (bar3/scale/slice/lawread.py) and round 2's subgoal score are
  imported from their folders.
- `reading_check.py`: the reading's parts, and the new laws against known programs; it writes `logs/known_lib.json`.
- `const_robust.py`: a record's make-1 routines on many dirty machines.
- `modal_reading.py`: every arm; app and Volume `bar3-reading`.
- `cost.py`: the billing report's hours for this study's apps.

**Records and logs:**
- `records/<tag>.json` and `.log`: each arm's record and log.
- `logs/`:
  - the checks: `reading_check.txt`, `known_routes.txt` (with `known_lib.json`), `const_robust.txt`;
  - the tests: `tiny_reading.txt`, with `tiny_loops_x2.*`, `tiny_down_x2.*`, `smoke_sublaw_x3.*`,
    `smoke_const_10x2p1.*` and `smoke_prior_x2.*`;
  - the copies' diffs against round 5's: `diff_*.txt`;
  - the hashes: `prereg_sha256.txt`, `addendum1_sha256.txt`;
  - the launches and watchers: `launch_*.txt`, `watch_r6_*.txt`;
  - the rounds on the ladders: `ladder_*.txt`; the replay: `replays_same.txt`;
  - the cost: `cost.txt`.

**To rerun** (from `cpu_first`; `H1=~/github/one-layer/one-layer-deeper/data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2`):
- **The copies:** `~/github/sotaku/venv/bin/python -m bar3.scale.reading.make_copies6`
- **The checks:** `H1_DATA=$H1 PYTHONPATH=$PWD SUBLAW=1 CONST=1 bar3/scale/rowcheck/memcap.sh 8 ~/github/sotaku/venv/bin/python -W ignore -m bar3.scale.reading.reading_check`
- **An arm:**
  1. `~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750`
  2. `~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/reading/modal_reading.py --arms r6_x2_nobook`
  3. `~/github/sotaku/venv/bin/python -m modal run bar3/scale/reading/modal_reading.py --mode fetch`
- **A round on the ladders:** `TARGET_FN=x3 ~/github/sotaku/venv/bin/python -m bar3.scale.compose3.verify3 bar3/scale/reading/records/r6_x3.json "LDA.X STA.Y T.mod.mul.x,o,n.XYN STA.X T.mod.mul.x,o,n.XYN"`
- **A record in short:** `~/github/sotaku/venv/bin/python -m bar3.scale.slice.summary bar3/scale/reading/records/r6_x2_nobook.json`
