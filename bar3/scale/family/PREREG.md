# Composition round 7: the family, with a rule for the sub-laws, pre-registered

Written 2026-10-27, before any run of this study. FAMILY.md §1 repeats this text unchanged and records the time and
hash it was frozen with.

## The questions

**What round 6 found** (bar3/scale/reading/READING.md; 6be8a0e):
- With no book, x²'s round came at level 2: the comparison from the step's cuts, then the step, the reduction and
  the round.
- The cube's round came at level 3, as two calls of x·o mod N. But that sub-law was chosen by a hand-written table
  (SUBLAWS: mod(mul) on x,o,n, where the reading holds it twice).
- 10·x² + 1 learned its constant (make 1 at level 0, x + 1 at level 1), but no round.

**This round, the generality side, offline** (the coordinator's aim: tell the legal file's full-size run which tasks
will come, and which general changes it needs first):
1. **A rule for the sub-laws that names no task,** in place of the table. Does it still give x² with no book and the
   cube as two calls?
2. **The family, unchanged:** round 6's no-book setup with that rule, tuned to no task, on SP_BREADTH's family
   (integration/sp_breadth/SP_BREADTH.md §1) at the core's size. The tasks are ten (10·y² mod N), rho (y² + 1), plus3
   (y² + 3), minus1 (y² − 1), tenrho (10·y² + 1) and the cube, with H1 as the control. Constants come from the
   data, as in round 6. Which tasks come, which do not, and why?
3. **General fixes for the failures only,** each behind a flag, with replays of x² with no book and of the cube
   (an addendum, frozen before its arms).

## Theory, in brief

1. **The rule: the reading's windows (SUBTERMS).**
   - A window is a connected run of 2 to 3 of the reading's operations. Each operation's children off the run are cut
     off as holes, but N (the modulus) and the constants stay. At most two holes.
   - Each window is posed as a law of its own, smallest first:
     - its holes are its operands, and its modulus is bound to N (patterns w1 and w2);
     - its answers are the law reader's own arithmetic, on Python's integers.
   - The rule names no task, no operation and no pattern. Round 6's table is its special case: the cube's reading
     holds mod(mul(a, b), N) twice.
2. **The readings** (the law reader, on each task's rows; `logs/readings.txt`) **and their windows:**

   | task | reading | windows (two or three operations) |
   |---|---|---|
   | x² (H1) | mod(mul(X,X),N) | mod(mul(a,b),N) |
   | cube | mod(mul(mod(mul(X,X),N),X),N) | mod(mul(a,b),N), mul(mod(a,N),b), mod(mul(mod(a,N),b),N) |
   | ten | mod(mul(mod(mul(X,X),N),10),N) | mod(mul(a,10),N), mul(mod(a,N),10), mod(mul(a,b),N), and two of three operations |
   | rho | mod(add(mul(X,X),1),N) | mod(add(a,1),N), add(mul(a,b),1), mod(add(mul(a,b),1),N) |
   | plus3 | mod(add(mul(X,X),3),N) | mod(add(a,3),N), add(mul(a,b),3), mod(add(mul(a,b),3),N) |
   | minus1 | mod(pred(add(mul(X,X),N)),N) | mod(pred(a),N), pred(add(a,N)), add(mul(a,b),N), and two of three |
   | tenrho | mod(add(mul(mod(mul(X,X),N),10),1),N) | four of two operations, including mod(mul(a,b),N); three of three, including mod(add(mul(a,10),1),N) |

3. **Every task then has a route of at most 2 window calls,** checked right on both of H1's ladders (98 of 98 rungs
   at 768/768, `logs/known_routes.txt`) with known routines for the windows and the constants:
   - x²: the window mod(mul(a,b),N) on (x, x);
   - the cube: that window twice;
   - ten: that window, then mod(mul(a,10),N);
   - rho and plus3: the multiply, then mod(add(a,c),N);
   - minus1: the multiply, then mod(pred(add(a,N)),N);
   - tenrho: mod(mul(a,b),N), then mod(add(mul(a,10),1),N).
4. **What each route needs learned:**
   - **x², the cube and ten** need only the multiply, the reduction and the step's cuts (×10 is `SHL.A` at base 10).
     Their windows are 3 tokens over those. Their rounds should come.
   - **rho and tenrho** need x + 1 in the binding a window uses (x in A or X, N in N).
     - Round 6 practised x + 1 only with N in Y. lpca practises a concept in its cheapest bindings, and make 1 with
       N in N is wrong on some rows (READING.md §5.4), which made the binding the windows want cost one more.
     - A window can also be built over make 1 and the addition directly (5 tokens, at the edge of the probes).
     - So their rounds are uncertain.
   - **plus3** needs make 3. The source poses the reading's constant leaves only (3), not 1. Make 3 over the
     comparison is 8 tokens, past the probes. No round.
   - **minus1** has no constant leaf: pred's borrow needs a 1, and nothing makes one. pred is 8 tokens over the
     comparison. No round.
5. **The fixes for the failures** (the addendum, after these runs, only for what fails), candidates as general rules:
   - **constants:** pose every constant the data shows, and the 1 of the reader's succ and pred;
   - **bindings:** practise every learnable binding of a picked concept;
   - **calls:** a round search past two calls.

## Design (`make_copies7.py`, `hooks.py`)

- **The copies:** round 6's (make_copies6.py's lists), paths moved to this folder, with this study's changes, each
  asserted to match once:
  - hooks.py: the family's targets (ten, rho, plus3, minus1 beside x2, x3 and 10x2p1); the windows
    (`pose_subterms`, from `reading_pose`) and their answers (`subterm_target`);
  - spcore_f.py: the windows' patterns, and their modulus kept in N;
  - core_f.py: the flag in the run's config.

  With SUBTERMS off the copies are round 6's.
- **SUBTERMS=1** poses the windows (SUBTERM_OPS 3 operations at most, 2 holes at most). SUBLAW stays off.
- **Checked before this was frozen** (`family_check.py`, `logs/family_check.txt`, `logs/known_routes.txt`,
  `logs/readings.txt`):
  - the rule on three readings;
  - 12 programs against the windows' probes and acceptors at every verify width, right ones passing and wrong ones
    failing;
  - the seven routes above on both ladders;
  - each task's reading and posing, in a one-level tiny run with no errors.

## Runs

Each on one B200, with round 6's no-book settings:
- TARGET small, ROUND_TOO, NP 8, deep probes over whole concepts, LEVELS 10;
- round 2's flags and TIEFOLLOW;
- SLICE, LOOPS and DOWN on, no law source.

This study's setup adds CONST=1 and SUBTERMS=1, with SECONDS 3,600. The windows add work to every level, and round
6's cube came 54 s before its 2,400 s deadline.

**Arms, launched together, at most 8 B200s at once:**
1. `replay_r6_x2_nobook`: round 6's r6_x2_nobook, every new flag off, SECONDS 2,400;
2. `r7_x2`: H1, the control;
3. `r7_x3`: the cube;
4. `r7_ten`;
5. `r7_rho`;
6. `r7_plus3`;
7. `r7_minus1`;
8. `r7_tenrho`.

**Any round** is checked on both ladders to T = 64 (`verify7.py`).

**The tiny tests, after this is frozen and before the arms,** locally under a memory cap:
- every flag off: round 6's tiny DOWN run again (`tiny_down_x2`);
- SUBTERMS on: the cube, 3 levels.

A fix that changes no design is recorded. A change of design goes in an addendum, frozen before the arms.

**Money:**
- soft $90, cap $130; `spend.py 4750` before each launch;
- one watcher at the cap less one hour of 8 B200s: stop all at $79, raised as arms end;
- expected cost for these eight about $40.

## Predictions

| what | prediction | confidence |
|---|---|---|
| replay_r6_x2_nobook | round 6's record again, apart from times: its round at level 2 | high |
| r7_x2 | the round by level 3, right on both ladders to T = 64 | medium-high |
| r7_x3: the window | mod(mul(a,b),N) practised by level 3 | medium-high |
| r7_x3: the round | by level 4, as two calls of that window, right on both ladders | medium-high |
| r7_ten | the round by level 4, right on both ladders | medium-high |
| r7_rho | the round by level 10 | low-medium |
| r7_plus3 | no round: no source makes a 1 to build 3 from | medium-high |
| r7_minus1 | no round: pred's borrow needs a 1, and the reading has no constant | medium-high |
| r7_tenrho | the round by level 10 | low-medium |
| the conclusion | the rule gives the cube as two calls with no table, and the routes of every task. The tasks that need only the loops come (x², the cube, ten). The constant tasks hang on making 1 and on the bindings of x + c | medium |
