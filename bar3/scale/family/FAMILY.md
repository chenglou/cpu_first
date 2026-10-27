# Composition round 7: the family, with a rule for the sub-laws

<!-- PLAIN -->
## In plain words

**The question.**
- Round 6 (bar3/scale/reading/READING.md) took the book to nothing for x². It got the cube as two calls of x·o
  mod N, but a hand-written table chose that sub-law.
- This round is the generality side, offline, for the legal file's full-size run: which of SP_BREADTH's family
  will come, and which general changes come first?
  1. **A rule for the sub-laws that names no task,** in place of the table. Does it still give x² and the cube?
  2. **The family, unchanged:** ten (10·y²), rho (y² + 1), plus3 (y² + 3), minus1 (y² − 1), tenrho (10·y² + 1)
     and the cube, all mod N, with H1 as the control.
  3. **General fixes for the failures only.**

**The rule: the reading's windows** (SUBTERMS, `hooks.py`).
- A window is a connected run of two or three of the target's reading's operations. Everything off the run becomes
  a hole (at most two), but N and the constants stay.
- Each window is posed as a law of its own, smallest first:
  - its holes are its operands, and its modulus stays in N;
  - its answers are the law reader's own arithmetic.
- The cube's reading holds mod(mul(a,b),N) twice, so the rule poses round 6's sub-law with no table.
- tenrho's holds mod(add(mul(a,10),1),N), the split round 6 proposed.

**What came** (round 6's no-book setup with the rule and round 6's constants; one B200 each, 3,600 s; every round
right on both of H1's ladders, T = 1 to 64, 768/768 on all 14 rungs, and on every one of the ladders' 747 moduli
at x = 0, at a random x and at x = N − 1):

| task | the round | how |
|---|---|---|
| x² (H1) | **level 2** (569 s) | as in round 6 |
| the cube | **level 3** (1,576 s) | two calls of the window mod(mul(a,b),N) |
| ten | **level 2** (1,192 s) | x·10, then the multiply and the reduction |
| rho | **level 3** (1,728 s) | the window x·o + 1, then the reduction |
| plus3 | **level 4** (2,259 s) | make 3 (over the link, with no 1), x + 3, the window x·o + 3, then the reduction |
| tenrho | **level 3** (1,712 s) | the window 10·(x·o mod N), then + 1, the borrow a finished reduction leaves, then one Horner step |
| minus1 | **no** | no 1 to subtract, and a window learned on holes below N |

**minus1, and two general fixes that bring it** (addenda 1 and 2):
- **CONST1:** the reader's pred is a subtraction of 1, a constant with no leaf in the reading, so its 1 is made from
  N and pred is posed. Then the round search found rounds at level 3 or 4.
- **But the round kept was wrong.** The climb kept the first one found, and it is wrong exactly when an input is 0:
  5 of 14 rungs right. The training data shows it: minus1's rounds pass through 0 in 24 rows of T = 4 and 61 of
  T = 8. The climb's round search checks only T = 1.
- **ROUNDT:** a round must also give the data's own answers at T = 2, 4 and 8. Later programs of the same search are
  right everywhere, and ROUNDT keeps the first of those.
- **With CONST1 and ROUNDT, minus1 comes:** level 3 or 4, right on both ladders and at x = 0.
- **HOLES** (a window's holes from other rows, above N as often as not) was not needed and changed no outcome. The
  check shows the risk it covers: on holes below N, one conditional subtraction passes for a reduction.

**All seven come with the fixes.** rho, plus3 and tenrho came again with all three fixes on. x², the cube and ten
came again with CONST1 and HOLES, and x² and the cube with ROUNDT alone. Each came at the same level or sooner, and
ROUNDT kept every first round that was right.

**What the legal file needs first,** in order:
1. round 6's no-book posing;
2. the window rule, with a window's modulus in N;
3. round 6's constants;
4. CONST1;
5. ROUNDT.

With these, all seven tasks here come at levels 2 to 4, in 570 to 2,450 s on a B200.

**What it means.**
- Once the reading is cut into windows, every task's round is at most five tokens over its routines. It is two
  window calls, a window and the reduction, or (for ten) x·10, the multiply and the reduction. The family's
  "something of y², plus a constant, then reduce" becomes a short composition. No round search past two calls was
  needed.
- **What remains hand-given** is round 6's tables (each operation's cut, each cut's loop, the step's cuts), and the
  rule's limits (3 operations, 2 holes).
- **Not covered:** SP_BREADTH's mont, M4, xfirst and xy were not in this round's list. These are offline climbs, not
  the legal file's training run.
- **Cost:** about $85 of the soft $90 (§9).
<!-- /PLAIN -->

## Details

### 1. Pre-registered

PREREG.md (frozen 2026-10-27 03:19:24 PDT, read-only, sha256
`f832b64363dc9f287b972b854e450fea3ff8c027a873689561816767b574c576`, `logs/prereg_sha256.txt`), before any arm ran,
unchanged:

<!-- PREREG -->
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
<!-- /PREREG -->

PREREG_ADDENDUM1.md (frozen 2026-10-27 04:41:01 PDT, read-only, sha256
`d912bb5a1e3981cbe413753bb6330db9bf6c4d030018409bd09f0928ce35a968`, `logs/addendum1_sha256.txt`), after the
pre-registration's eight arms and before its own, unchanged:

<!-- ADDENDUM1 -->
# Composition round 7, addendum 1: general fixes for the one failure, pre-registered

Written 2026-10-27, after the pre-registration's eight arms and before any run below. FAMILY.md §1 records its time
and hash.

## What the eight arms showed

**Six of the seven tasks came,** each right on both of H1's ladders (14 of 14 rungs at 768/768), and on every one of
the ladders' 747 moduli at x = 0, at a random x and at x = N − 1:

| task | the round | how |
|---|---|---|
| x² (H1) | level 2 (569 s) | x², then the reduction, as in round 6 |
| the cube | level 3 (1,576 s) | two calls of the window mod(mul(a,b),N), itself built over the window mul(mod(a,N),b) |
| ten | level 2 (1,192 s) | x·10 (the constant's operation, `LDA.X SHL.A`), then the multiply and the reduction |
| rho | level 3 (1,728 s) | the window add(mul(a,b),1), then the reduction |
| plus3 | level 4 (2,259 s) | the window add(mul(a,b),3), then the reduction. Make 3 came at level 2, 4 tokens over the link (`CLA B[T.red.pair.AX ADC.S]`) |
| tenrho | level 3 (1,712 s) | the window mul(mod(mul(a,b),N),10), then `ADC.X`, then Horner's step. The `ADC.X` adds the 1 that a finished reduction's last comparison leaves as its borrow |

The replay gave round 6's record again, apart from times.

**minus1 did not** (the deadline in level 6). Its reading, mod(pred(add(mul(X,X),N)),N), has two problems:
- **No 1.** It has no constant leaf, so nothing makes a 1, and pred's subtraction needs one.
- **Holes below N.** A window's hole is drawn below N, but the route puts x² + N in it. The window mod(pred(a),N),
  learned at level 5 on holes below N, has no full reduction, so the two-call route was wrong at full size. The check
  shows the cause (`logs/family_check.txt`): on holes below N, one conditional subtraction passes for the reduction of
  a + N − 1.

## The fixes, each behind its own flag

1. **CONST1:** succ and pred are an addition and a subtraction of 1, a constant the reading holds without a leaf.
   When the reading has one, its 1 is made from N (const.1 on n), and succ or pred is posed on x,n. Their answers are
   the law reader's own: x + 1, and max(x − 1, 0). It needs CONST's patterns.
2. **HOLES:** a window's holes are drawn from other rows than its modulus's (w1o: another row's x; w2o: two other
   rows'). A hole is then above N as often as not, as in the routes. The windows' bindings stay as before.

Both name no task. CONST1 poses nothing for a reading without succ or pred, so it cannot change x², the cube, ten,
rho, plus3 or tenrho.

## Arms

Each on one B200, the pre-registration's setup (SECONDS 3,600), launched together, at most 8 at once:
- `r7_minus1_both`: minus1 with both fixes;
- `r7_minus1_const1` and `r7_minus1_holes`: minus1 with each alone;
- the family with both fixes: `r7_x2_both`, `r7_x3_both`, `r7_ten_both`, `r7_rho_both`, `r7_plus3_both` and
  `r7_tenrho_both`. Since CONST1 poses nothing without succ or pred, the x² and cube arms here are HOLES's replays;
- `r7_x2_const1` and `r7_x3_const1`: CONST1's replays.

Any round is checked on both ladders to T = 64.

## Predictions

| what | prediction | confidence |
|---|---|---|
| r7_minus1_both | the round by level 8, right on both ladders: make 1, pred, the windows over them, then the multiply and the window | low-medium |
| r7_minus1_const1 | no round: on holes below N the windows reduce with one conditional subtraction, wrong at the route's values | medium |
| r7_minus1_holes | no round: nothing makes a 1 | medium-high |
| r7_x2_both | the round by level 3 | high |
| r7_x3_both | the round by level 4, two calls of the window | medium-high |
| r7_ten_both | the round by level 3 | high |
| r7_rho_both, r7_plus3_both, r7_tenrho_both | each round again, by level 6 | medium |
| r7_x2_const1, r7_x3_const1 | r7_x2's and r7_x3's records again, apart from times | high |

## Money

- Eleven arms, at most 8 at once. The watcher, over both apps, stops all at $79 (the cap less one hour of 8 B200s).
- This study so far: about $31. With these, about $73, under the soft $90.
<!-- /ADDENDUM1 -->

PREREG_ADDENDUM2.md (frozen 2026-10-27 06:13:29 PDT, read-only, sha256
`7b220b20c176489081050dcf5fb5032a3afab107e61fa32f25c2795a37e6812b`, `logs/addendum2_sha256.txt`), after addendum
1's eleven arms and before its own, unchanged:

<!-- ADDENDUM2 -->
# Composition round 7, addendum 2: a round must give the answers of every T, pre-registered

Written 2026-10-27, after addendum 1's eleven arms and before any run below. FAMILY.md §1 records its time and hash.

## What addendum 1's arms showed

**The six tasks that came in phase 1 came again with both fixes** (CONST1 and HOLES): x², the cube, ten, rho, plus3
and tenrho. Each is right on both ladders and on every modulus at x = 0, at a random x and at x = N − 1. CONST1's
replays gave r7_x2's and r7_x3's records again, apart from times.

**minus1 found rounds, but not right ones:**

| arm | the round search | its first program, kept as the round |
|---|---|---|
| `r7_minus1_both` | 7 programs at level 3 (1,194 s) | right on 5 of 14 rungs, 0 of 747 moduli at x = 0 |
| `r7_minus1_const1` | 6 at level 4 (2,299 s) | the same |
| `r7_minus1_holes` | none (the deadline in level 6): nothing made a 1 | – |

- **The first program ends with a reduction of X and a subtraction** (`... T.red.x+on.XN T.lt.pair.AY`). It is wrong
  exactly when a round's input is 0, as SP_BREADTH's constants change was.
- **The training data shows the failure.** H1's training rows have T = 1, 2, 4, 8 and 16, and minus1's rounds pass
  through 0 in 24 of the 33,333 rows of T = 4 and 61 of T = 8. The climb's round search checks only the answers of
  T = 1.
- **Later programs of the same search are right everywhere,** for example `LDA.X STA.Y T.w.pred.add.mul.w2.XYN STA.X
  T.red.x+on.XN`: the window pred(add(mul(a,b),N)), then the whole reduction.
- **ROUNDT's filter, applied offline to the found rounds** (`roundt_offline.py`, `logs/roundt_offline.txt`):
  - it keeps the last two (CONST1) and three (both fixes) of minus1's;
  - it keeps all of the cube's, rho's and tenrho's.

## The fix, behind its own flag

**ROUNDT:** a round must also give the data's answers at every one-digit T past 1. Run T times at full size, it must
match every training row with T = 2, 4 or 8. For x², those are the data's own answers; for another target, its
function applied T times, as its T = 1 answers are. Of the programs found, the first that does is kept. If none
does, the climb goes on. It names no task: it asks a round to be the round on all the data's rows, not only on those
of T = 1. ROUNDT also asks the small target's program to be right at full size on the T = 1 answers.

## Arms

Each on one B200, addendum 1's setup (SECONDS 3,600), launched together, at most 8 at once:
- `r7_minus1_all`: minus1 with all three fixes (CONST1, HOLES, ROUNDT);
- `r7_minus1_const1_roundt`: minus1 with CONST1 and ROUNDT;
- `r7_x2_roundt` and `r7_x3_roundt`: ROUNDT's replays, on phase 1's setup;
- the family with all three fixes: `r7_x2_all`, `r7_x3_all`, `r7_ten_all`, `r7_rho_all`, `r7_plus3_all` and
  `r7_tenrho_all`.

Any round is checked on both ladders to T = 64, and on every modulus at x = 0, at a random x and at x = N − 1.

## Predictions

| what | prediction | confidence |
|---|---|---|
| r7_minus1_all | the round at level 3, the fifth of its search's seven programs, right on both ladders and at x = 0 | high |
| r7_minus1_const1_roundt | the round at level 4, the fifth of six, right on both ladders and at x = 0 | high |
| r7_x2_roundt, r7_x3_roundt | r7_x2's and r7_x3's records again, apart from times and ROUNDT's own records | high |
| r7_*_all, the six others | each round again, at its r7_*_both level, right on both ladders | high |

## Money

- Ten arms, at most 8 at once. A new watcher, over the three apps, stops all at $79 (the cap less one hour of 8
  B200s).
- This study so far: about $45. With these, about $70, under the soft $90.
<!-- /ADDENDUM2 -->

### 2. Design

#### 2.1 The rule: the reading's windows (`hooks.py`: windows, pose_subterms, subterm_target)

- **A window** is a connected run of 2 to 3 of the reading's operations (SUBTERM_OPS). Each operation's children off
  the run are cut off as holes: a cut-off operation, or a leaf X. N (the modulus) and the constants stay. Holes are
  named a, b in the order they appear; at most two (SUBTERM_HOLES).
- **Each window is posed once** (the same shape anywhere in the reading is one law), smallest first, then top-down. It
  is named by its operations and their constants (`w.mod.mul`, `w.mod.add1.mul10`). Its holes are its operands, on
  w1 (a and N) or w2 (a, b and N), at double width, with its modulus bound to N. So a window has 3 bindings (w1) or 6
  (w2), not 12 or 24.
- **Its answers** are the law reader's own arithmetic (subgoal_c2's operations) on Python's integers. Rows where a
  value on the way leaves [0, B^P) are masked.
- **It names no task, no operation and no pattern.** Round 6's table (SUBLAWS: mod(mul) on x,o,n where the reading
  holds it twice) is the cube's case of it. With the rule on, SUBLAW stays off.

#### 2.2 The family (`hooks.py`: TARGET_F)

ten 10·x² mod N, rho (x² + 1) mod N, plus3 (x² + 3) mod N and minus1 (x² − 1) mod N, beside x² (H1), the cube and
10·x² + 1 (tenrho). Every task uses H1's rows (the same x and N), its answers its own function. `verify7.py` checks a
round on both of H1's ladders against the task's function applied T times.

#### 2.3 The fixes (addenda 1 and 2), each behind its own flag

- **CONST1** (`pose_const1`; answers in `reading_target`): when the reading has succ or pred, their 1 is made from N
  (const.1 on n), and succ or pred is posed on x,n with the law reader's answers (x + 1, max(x − 1, 0)). It needs
  CONST's patterns.
- **HOLES** (patterns w1o, w2o; a draw's third row, `p`): a window's holes are drawn from other rows than its
  modulus's, so a hole is above N as often as not. The windows' bindings stay as before.
- **ROUNDT** (`more_answers`, `round_filter`):
  - it reads every training row of T = 2, 4 and 8, with the data's answers (x²) or the target's function applied T
    times;
  - where the climb keeps a round, from the small target's search or the round's, it keeps only the programs that are
    right at full size on the T = 1 answers and, run T times, on those;
  - if none is, the climb goes on.

#### 2.4 The copies (`make_copies7.py`)

Round 6's copies (make_copies6.py's lists, on rounds 3 to 5's), their module paths moved to this folder, with this
study's asserted replacements (`logs/diff_*.txt` against round 6's copies):
- **spcore_f.py** (4):
  - the windows' patterns, with SUBTERMS on;
  - a window's modulus kept in N;
  - a draw's third row and its term (HOLES).
- **core_f.py** (2): ROUNDT's answers, and the flags in the run's config.
- **climb_f.py** (3): ROUNDT's filter where the climb keeps a round.
- **hooks.py** (5):
  - the family's targets;
  - CONST1's and the windows' posing after round 6's constants;
  - succ's and pred's answers, and the windows' answers;
  - the windows in the reading's record;
  - with the rule and the fixes appended.

With every new flag off, the copies run as round 6's (§3).

### 3. The checks and tests (local, under a memory cap)

**Before each freeze** (they set the design):

| check | what it showed |
|---|---|
| the readings (`logs/readings.txt`, one-level tiny runs) | each task's reading and what it poses: the seven readings of §1, their windows, and their constants (ten 10, rho 1, plus3 3, tenrho 10 and 1; x², the cube and minus1 none). No errors |
| the windows (`family_check.py`, `logs/family_check.txt`) | the rule on three readings; 12 programs against the windows' probes and acceptors at every verify width, right ones passing and wrong ones (unreduced, or missing one of the window's operations) failing |
| the routes (`logs/known_routes.txt`) | all seven tasks' routes of §1, with known routines for the windows and the constants: 98 of 98 rungs at 768/768 on both ladders |
| addendum 1's fixes (`family_check.py`) | 7 more programs. pred over make 1 and the link passes; the plain subtraction fails at 0. On holes below N (w1), one conditional subtraction passes for the reduction of a + N − 1; on holes from another row (w1o) it fails, and only the whole reduction passes |
| addendum 2's filter (`roundt_offline.py`, `logs/roundt_offline.txt`) | ROUNDT on the rounds already found: it drops minus1's first four (wrong on 24 to 61 rows at T = 4 and 8) and keeps the rest; it keeps every round of the cube, rho and tenrho |

**After each freeze, before its launch** (tiny budgets, one at a time; no fix was needed):

| test | what it showed |
|---|---|
| every flag off (`tiny_down_x2.*`, then `tiny_down_x2_a1.*`, `tiny_down_x2_a2.*`) | round 6's tiny DOWN record again, apart from times, each time the copies changed |
| SUBTERMS on, the cube, 3 levels (`tiny_subterms_x3.*`, then `tiny_subterms_x3_a1.*`) | the three windows posed, no errors; after addendum 1's changes, the same record again |
| both of addendum 1's fixes, minus1, 2 levels (`tiny_fixed_minus1.*`) | make 1, pred, and the windows on w1o and w2o posed. No errors |
| ROUNDT, x², 1 level (`smoke_roundt_x2.*`) | its answers load and the flag is recorded. No errors |

### 4. The runs (`modal_family.py`; `records/*.json` and `.log`; `logs/launch_*.txt`)

| launch | PDT | app | arms | B200s at once |
|---|---|---|---|---|
| phase 1 | 03:27 | `ap-1D1Y8tz0CDZ3CTVtnwy776` | the pre-registration's eight | 8 |
| addendum 1 | 04:53 | `ap-zFQQAYrrPuIR0mUlMStYpA` | its eleven | 8 (max_containers 8), the rest as containers freed |
| addendum 2 | 06:18 | `ap-zKxsiEpvkIa8fXHBFlrAIX` | seven of its ten | 7 |
| addendum 2, relaunched | 07:08 | `ap-wP8PzLLDuSJgGmfl8AyY98` | the three the watcher cut | 3 |

- **Each launch** came after `spend.py 4750` (LAUNCH OK). Each pre-registration was frozen before its arms:
  - PREREG.md at 03:19:24;
  - addendum 1 at 04:41:01, sha256 `d912bb5a1e3981cbe413753bb6330db9bf6c4d030018409bd09f0928ce35a968`;
  - addendum 2 at 06:13:29, sha256 `7b220b20c176489081050dcf5fb5032a3afab107e61fa32f25c2795a37e6812b`.

  The tiny tests ran after each freeze and before its launch (§3).
- **The copies changed twice during the study,** each time behind new flags, and each time checked with every new
  flag off against round 6's tiny record, and with SUBTERMS on against this study's own tiny cube record (§3):
  - for addendum 1: CONST1 and HOLES;
  - for addendum 2: ROUNDT.
- **Changes from the pre-registrations:**
  - **Addendum 2 ran seven of its ten arms.** `r7_x2_all`, `r7_x3_all` and `r7_ten_all` were not run, for money:
    the study had billed $56.30, and all ten would have passed the watcher's $79. They are covered elsewhere: x²'s and
    the cube's rounds by ROUNDT's replays and the both-fix arms; ten's round uses no window and ROUNDT keeps it.
  - **The watcher cut three of the seven at 06:48 PDT,** 30 minutes in: `r7_minus1_const1_roundt`, `r7_plus3_all` and
    `r7_tenrho_all`.
    - Its estimate over the three apps read $110 against $73.12 billed, because it counts an hour under way for apps
      already stopped (READING.md §9 had noted that it runs high).
    - The three were relaunched unchanged at 07:08 PDT, under a watcher over their app alone: stop at $36, the cap less
      the $73.12 billed less one hour of 3 B200s.
    - Their cut records are kept in `records/cut/`.

### 5. Outcomes (`records/*.json`; bar3/scale/slice/summary.py)

#### 5.1 The family with the rule: six of seven tasks come (phase 1)

Each in round 6's no-book setup with CONST and SUBTERMS (SECONDS 3,600). Every round is right on both of H1's ladders
(14 of 14 rungs at 768/768, `logs/ladder_r7_*.txt`). It is also right on every one of the ladders' 747 moduli at
x = 0, at a random x below N and at x = N − 1 (`general_check.py`, `logs/general_check.txt`).

| task | the round | its program | how it came |
|---|---|---|---|
| x² (H1) | **level 2** (569 s) | `T.val.63.tgt.XN STA.X T.red.x+on.XN` | x² as a value, then the reduction, as in round 6 |
| the cube | **level 3** (1,576 s) | `LDA.X STA.Y T.w.mod.mul.w2.XYN T.w.mod.mul.w2.AYN` | two calls of the window mod(mul(a,b),N), practised at level 3 in two bindings, one built over the window mul(mod(a,N),b) of level 2 |
| ten | **level 2** (1,192 s) | `T.mul.10.x,n.XA STA.Y T.mul.x,o.XY STA.X T.red.x+on.XN` | x·10 (the constant's operation, `LDA.X SHL.A`, level 0), then the multiply and the reduction; no window |
| rho | **level 3** (1,728 s) | `LDA.X STA.Y T.w.add1.mul.w2.XYN STA.X T.red.x+on.XN` | the window add(mul(a,b),1) (level 3, over the multiply and x + 1), then the reduction |
| plus3 | **level 4** (2,259 s) | `LDA.X STA.Y T.w.add3.mul.w2.XYN STA.X T.red.x+on.XN` | make 3 at level 2, x + 3 at level 3, the window add(mul(a,b),3) at level 4, then the reduction |
| tenrho | **level 3** (1,712 s) | `LDA.X STA.Y T.w.mul10.mod.mul.w2.XYN ADC.X T.red.Bx+d.AN` | the window mul(mod(mul(a,b),N),10) (level 3), then `ADC.X`, then Horner's step |
| minus1 | no (the deadline in level 6) | – | §5.2 |

- **The cube with no table:** the window rule poses mod(mul(a,b),N), the law round 6's table chose by hand, and the
  cube's round is two calls of it, at level 3 as in round 6 but 770 s sooner.
- **Make 3 over the link:** `CLA B[T.red.pair.AX ADC.S]`, the link repeated B times with an addition of its borrow, is
  4 tokens. The pre-registration counted 8 over the comparison and predicted no round.
- **tenrho's + 1 is a borrow.** The window ends with a full reduction, whose last comparison of a remainder below N
  always borrows. So `ADC.X`, on an X the reduction has shifted to 0, adds exactly 1. Then one Horner step reduces
  10·r + 1 < 10·N. Neither make 1 nor x + 1 is used.
- **ten used the constant's operation, not a window:** x·10 is 2 tokens with x in X. Then 10x·x is reduced, 10x² < B^P
  at the full size.

#### 5.2 minus1: no 1, and holes below N

- **What came:**
  - the row, add's cut and the comparison (0);
  - the multiply, the step, the link and the window a·b + N (1, `LDA.N W[T.acc.mul... INC SHL.X]`: the multiply
    accumulated onto N);
  - the reduction (2);
  - nothing new at level 3, practised concepts' other bindings at 4;
  - the window mod(pred(a),N) at level 5 (`T.red.x+on.NA LDA.Y W[SBB.X INC]`);
  - the deadline in level 6.
- **No 1:** the reading has no constant leaf, so nothing made a 1, and pred's subtraction needs one.
- **Holes below N:** the route x·x + N, then mod(pred(a),N), was 2 calls, but wrong at full size. The window was
  learned on holes below N, where pred(a) is already below N. In the route its hole is x² + N.
- **The check shows the cause:** on holes below N, one conditional subtraction passes for the reduction of
  a + N − 1. With holes drawn from other rows, only the whole reduction passes (`logs/family_check.txt`).

#### 5.3 Addendum 1: the six again with both fixes; minus1 finds rounds, but keeps a wrong one

| arm | fixes | the round | right on both ladders | at x = 0 / random x / x = N − 1, every modulus |
|---|---|---|---|---|
| `r7_x2_both` | CONST1, HOLES | level 2 (586 s) | 14/14 | 747 / 747 / 747 |
| `r7_x3_both` | CONST1, HOLES | level 3 (1,153 s), two calls of the window (on w2o) | 14/14 | 747 / 747 / 747 |
| `r7_ten_both` | CONST1, HOLES | level 2 (953 s) | 14/14 | 747 / 747 / 747 |
| `r7_rho_both` | CONST1, HOLES | level 2 (1,031 s) | 14/14 | 747 / 747 / 747 |
| `r7_plus3_both` | CONST1, HOLES | level 4 (2,150 s) | 14/14 | 747 / 747 / 747 |
| `r7_tenrho_both` | CONST1, HOLES | level 3 (1,816 s) | 14/14 | 747 / 747 / 747 |
| `r7_minus1_both` | CONST1, HOLES | level 3 (1,194 s), its first of 7 found programs | **5/14** (763–767 of 768 from T = 4 or 8) | **0** / 747 / 747 |
| `r7_minus1_const1` | CONST1 | level 4 (2,299 s), its first of 6 | **5/14** | **0** / 747 / 747 |
| `r7_minus1_holes` | HOLES | no (the deadline in level 6): nothing made a 1 | – | – |
| `r7_x2_const1`, `r7_x3_const1` | CONST1 | r7_x2's and r7_x3's records again, apart from times | – | – |

- **With CONST1, minus1's search finds rounds.** Make 1 and pred came, then the window pred(add(mul(a,b),N)), x·x + N
  − 1.
- **The first program kept is wrong at 0.** It reduces X and then subtracts (`T.red.x+on.XN T.lt.pair.AY`), which
  is wrong exactly when a round's input is 0. That is the failure SP_BREADTH's constants change met (SP_BREADTH.md
  §4.4). The ladders hold rows whose rounds pass through 0 (y² = 1 mod N): from T = 8 on the seen moduli and T = 4 on
  the unseen. So it fails on 9 of 14 rungs, 763 to 767 rows of 768 right.
- **Later programs of the same search are right everywhere:** for minus1_const1 its fifth and sixth, among them
  `LDA.X STA.Y T.w.pred.add.mul.w2.XYN STA.X T.red.x+on.XN` (the window, then the whole reduction); for minus1_both
  its fifth to seventh. The climb keeps the first. Its round search checks only the T = 1 answers, whose x are
  never 0.
- **HOLES was not needed for minus1's rounds.** The window that won holds no reduction (x·x + N − 1), and the whole
  reduction comes after it, as its own routine. With HOLES alone nothing made a 1.
- **HOLES changed no outcome among the six:** the same rounds, at the same levels or sooner (rho at level 2, not 3).

#### 5.4 Addendum 2: ROUNDT, and all seven tasks come

| arm | fixes | the round | ROUNDT kept | right on both ladders | at x = 0 / random x / x = N − 1, every modulus |
|---|---|---|---|---|---|
| `r7_minus1_all` | CONST1, HOLES, ROUNDT | **level 3** (1,212 s): `T.w.mod.pred.w1o.XN T.mul.x,o.XY T.add.x,d.AY STA.X T.red.x+on.XN` | 3 of 7 | **14/14** | 747 / 747 / 747 |
| `r7_minus1_const1_roundt` | CONST1, ROUNDT | **level 4** (2,228 s): `LDA.X STA.Y T.w.pred.add.mul.w2.XYN STA.X T.red.x+on.XN` | 2 of 6 | **14/14** | 747 / 747 / 747 |
| `r7_rho_all` | all three | level 2 (849 s) | 1 of 1 | 14/14 | 747 / 747 / 747 |
| `r7_plus3_all` | all three | level 4 (2,435 s) | 1 of 1 | 14/14 | 747 / 747 / 747 |
| `r7_tenrho_all` | all three | level 3 (1,629 s) | 2 of 2 | 14/14 | 747 / 747 / 747 |
| `r7_x2_roundt`, `r7_x3_roundt` | ROUNDT | r7_x2's and r7_x3's records again, apart from times and ROUNDT's records (4 of 4, 5 of 5 kept) | – | – | – |

- **minus1 is learned with CONST1 and ROUNDT,** with HOLES or without it.
  - ROUNDT dropped the programs of the round search that are wrong at 0 and kept the first right one.
  - The climb, the level and the round search were the same as without ROUNDT (§5.3). Only the round kept differs.
- **ROUNDT changed nothing where the first round was right:** x², the cube, rho, plus3 and tenrho kept their first
  programs.
- **The three runs the watcher cut** (§4) had reached levels 3 or 4 without their rounds. Run again unchanged, each
  came where its addendum-1 arm had.

### 6. Predictions against outcomes

**The pre-registration's:**

| what | prediction | confidence | outcome |
|---|---|---|---|
| replay_r6_x2_nobook | round 6's record again, apart from times: its round at level 2 | high | **right** (`logs/replays_same.txt`) |
| r7_x2 | the round by level 3, right on both ladders to T = 64 | medium-high | **right**: level 2 |
| r7_x3: the window | mod(mul(a,b),N) practised by level 3 | medium-high | **right**: level 3, in two bindings |
| r7_x3: the round | by level 4, as two calls of that window, right on both ladders | medium-high | **right**: level 3 |
| r7_ten | the round by level 4, right on both ladders | medium-high | **right**: level 2, with no window (x·10, then the multiply and the reduction) |
| r7_rho | the round by level 10 | low-medium | **right**: level 3 |
| r7_plus3 | no round: no source makes a 1 to build 3 from | medium-high | **wrong**: level 4. Make 3 came at level 2, 4 tokens over the link, with no 1 |
| r7_minus1 | no round: pred's borrow needs a 1, and the reading has no constant | medium-high | **right** |
| r7_tenrho | the round by level 10 | low-medium | **right**: level 3. Its + 1 is a finished reduction's borrow |
| the conclusion | the rule gives the cube as two calls with no table, and the routes of every task. The tasks that need only the loops come (x², the cube, ten). The constant tasks hang on making 1 and on the bindings of x + c | medium | **partly right**: the cube, the routes, and the loop tasks, yes. But three of the four constant tasks came too; only minus1 failed |

**Addendum 1's:**

| what | prediction | confidence | outcome |
|---|---|---|---|
| r7_minus1_both | the round by level 8, right on both ladders | low-medium | **wrong**: a round at level 3, but the one kept is wrong at 0 (5 of 14 rungs); right ones were found too |
| r7_minus1_const1 | no round: on holes below N the windows reduce with one conditional subtraction | medium | **wrong**: rounds at level 4, through a window with no reduction in it; the one kept is wrong at 0 |
| r7_minus1_holes | no round: nothing makes a 1 | medium-high | **right** |
| r7_x2_both | the round by level 3 | high | **right**: level 2 |
| r7_x3_both | the round by level 4, two calls of the window | medium-high | **right**: level 3 |
| r7_ten_both | the round by level 3 | high | **right**: level 2 |
| r7_rho_both, r7_plus3_both, r7_tenrho_both | each round again, by level 6 | medium | **right**: levels 2, 4 and 3 |
| r7_x2_const1, r7_x3_const1 | r7_x2's and r7_x3's records again, apart from times | high | **right** |

**Addendum 2's:**

| what | prediction | confidence | outcome |
|---|---|---|---|
| r7_minus1_all | the round at level 3, the fifth of its search's seven programs, right on both ladders and at x = 0 | high | **right** |
| r7_minus1_const1_roundt | the round at level 4, the fifth of six, right on both ladders and at x = 0 | high | **right** (after its relaunch, §4) |
| r7_x2_roundt, r7_x3_roundt | r7_x2's and r7_x3's records again, apart from times and ROUNDT's own records | high | **right** |
| r7_*_all, the six others | each round again, at its r7_*_both level, right on both ladders | high | **right** for rho (level 2), plus3 (4) and tenrho (3); x², the cube and ten not run (money, §4) |

### 7. Dead ends, and how hard each was tried

| what | how hard | where it stopped |
|---|---|---|
| minus1 with the rule and round 6's constants (`r7_minus1`) | 6 levels, to the 3,600 s deadline | no constant leaf, so no 1; the window mod(pred(a),N), learned at level 5 on holes below N, had no full reduction. Fixed by CONST1 and ROUNDT (§5.4) |
| minus1 with HOLES alone (`r7_minus1_holes`) | 6 levels, to the deadline | still no 1 |
| minus1 with CONST1, without ROUNDT (`r7_minus1_const1`, `r7_minus1_both`) | to their rounds at levels 4 and 3 | the round kept was the first found, wrong when an input is 0. The right ones of the same search were kept only with ROUNDT |

### 8. What is left, and what the legal file needs first

**For the legal file's full-size run,** in the order the evidence gives:
1. **Round 6's no-book posing:** the cuts, their loops and the step's cuts. With it, x² comes at level 2.
2. **The window rule (SUBTERMS), with a window's modulus kept in N.** It gives the cube as two calls (level 3), and
   the routes of rho, plus3 and tenrho (levels 2 to 4). It names no task and replaces round 6's table.
3. **Round 6's constants (CONST):** make c and the reading's operations on c. ten's round uses x·10; rho's, plus3's
   and tenrho's windows build on make 1, make 3 and x + c.
4. **CONST1:** the 1 of the reader's succ and pred. Without it minus1 has no 1.
5. **ROUNDT:** a round must give the data's answers at T = 2, 4 and 8 as well as T = 1. Without it, minus1 keeps a
   round that is wrong exactly when an input is 0, though its search found right ones too. It costs a few seconds a
   round, and it changes nothing where the first round is right.

HOLES was not needed by any task here (§5.3, §5.4), but changed no outcome either. The check shows the risk it
covers: a window learned on holes below N can pass with one conditional subtraction for a reduction.

**Which tasks will come:** all seven, with these five, at levels 2 to 4 and 570 to 2,300 s on a B200. Every round
is right on both of H1's ladders and on every modulus at x = 0, at a random x and at x = N − 1.

**What this does not show:**
- **The other prompts:** SP_BREADTH's mont, M4, xfirst and xy were not run. Mont needs a reduction no source
  poses here. M4 has answers only at T = 8, with no T = 1 answers to search against. xfirst and xy change the
  prompt, which this harness does not read.
- **The legal file itself:** these runs are the offline climb (TARGET small, the round's search at full size), not
  the certified file's training run.
- **Robustness:** one seed per arm; the arms' records are deterministic (every replay gave the same record).

**Left in the source:**
- The tables of round 6 remain: CUTS (each operation's cut), LOOP_OF (each cut's loop) and STEP_CUTS. The window
  rule replaced only SUBLAWS.
- The window rule's limits are parameters: 3 operations and 2 holes. The family needed windows of 2 or 3.
- lpca practises a concept only in its cleanest bindings (x + 1 with N in Y, READING.md §5.4). Here the windows'
  own bindings carried the routes.

### 9. Cost (`cost.py`; `logs/cost.txt`, `logs/watch_r7_*.txt`)

- **Billed to 14:00 UTC: $73.12.**
  - Phase 1 $24.90: 10:00 UTC $21.03, 11:00 UTC $3.87.
  - Addendum 1 $31.40: 11:00 UTC $5.36, 12:00 UTC $26.04.
  - Addendum 2 $16.82: 13:00 UTC.
- **Not yet billed:** the relaunch (14:08 to 14:54 UTC), three B200s for 6,291 s, about 1.8 B200-hours with startup, or
  about $11.5.
- **About $85 in all,** under the soft $90.
- **B200 time:** 44,946 s over the 29 runs (phase 1 13,204; addendum 1 16,870; addendum 2 4,215, the cut three 4,365,
  the relaunch 6,291), about 12.5 B200-hours, plus each container's startup.
- **The watchers:**
  - phase 1's and addendum 1's stopped all at $79 and stopped nothing;
  - addendum 2's, over the three apps, stopped all at $79. It cut three arms at 06:48 PDT on an estimate of $110
    against $73.12 billed (§4);
  - the relaunch's, over its app alone, stopped at $36 and stopped nothing.
- **Every app of this study is stopped.** The fetches ran no containers. The local checks and tests ran under 8–10 GB
  memory caps.

### 10. Files and how to rerun

**The study:**
- `PREREG.md`, `PREREG_ADDENDUM1.md`, `PREREG_ADDENDUM2.md`: read-only, their hashes in §1.
- `make_copies7.py`: writes `spcore_f.py`, `oracle_f.py`, `core_f.py`, `climb_f.py` and `clean_f.py`, and `hooks.py`
  (round 6's, with the family's targets, the windows, CONST1, HOLES and ROUNDT).
- The checks:
  - `family_check.py`: the window rule on three readings, and 19 programs against the windows', pred's and the
    holes' probes and acceptors; it writes `logs/known_lib.json`;
  - `verify7.py`: a round on both of H1's ladders, for any of the family's targets;
  - `general_check.py`: a round off the ladders (x = 0, random x, x = N − 1, every modulus);
  - `roundt_offline.py`: ROUNDT's filter on a record's found rounds;
  - `same_but_roundt.py`: two records the same apart from times and ROUNDT's records.
- `modal_family.py`: every arm; app and Volume `bar3-family`.
- `cost.py`: the billing report's hours for this study's apps.

**Records and logs:**
- `records/<tag>.json` and `.log`: each arm's record and log; `records/cut/`, the three the watcher cut.
- `logs/`:
  - the readings: `readings.txt`, with `read_*`;
  - the checks: `family_check.txt`, `known_routes.txt`, `general_check.txt`, `roundt_offline.txt`,
    `addendum1_checks.txt`, `addendum2_checks.txt`;
  - the tests: `tiny_family.txt`, `tiny_family2.txt`, `tiny_family3.txt`, with their `tiny_*` and `smoke_*` records;
  - the copies' diffs against round 6's: `diff_*.txt`;
  - the hashes: `prereg_sha256.txt`, `addendum1_sha256.txt`, `addendum2_sha256.txt`;
  - the launches and watchers: `launch_*.txt`, `watch_r7_*.txt`;
  - the rounds on the ladders: `ladder_*.txt`; the replays: `replays_same.txt`;
  - the cost: `cost.txt`.

**To rerun** (from `cpu_first`; `H1=~/github/one-layer/one-layer-deeper/data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2`):
- **The copies:** `~/github/sotaku/venv/bin/python -m bar3.scale.family.make_copies7`
- **The checks:** `H1_DATA=$H1 PYTHONPATH=$PWD SUBTERMS=1 CONST=1 bar3/scale/rowcheck/memcap.sh 8 ~/github/sotaku/venv/bin/python -W ignore -m bar3.scale.family.family_check`
- **An arm:**
  1. `~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750`
  2. `~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/family/modal_family.py --arms r7_minus1_all`
  3. `~/github/sotaku/venv/bin/python -m modal run bar3/scale/family/modal_family.py --mode fetch`
- **A round on the ladders:** `TARGET_FN=minus1 ~/github/sotaku/venv/bin/python -m bar3.scale.family.verify7 bar3/scale/family/records/r7_minus1_all.json "T.w.mod.pred.w1o.XN T.mul.x,o.XY T.add.x,d.AY STA.X T.red.x+on.XN"`
- **Off the ladders:** the same with `bar3.scale.family.general_check`.
