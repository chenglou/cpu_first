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
