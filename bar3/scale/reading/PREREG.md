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
