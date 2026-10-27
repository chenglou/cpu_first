# The family file (phase C): round 7's four pieces in the no-book file, on H1 and the family

<!-- PLAIN -->
## In plain words

**The question.**
- Phase B's file (legal_nobook/NOBOOK.md) has no law in its book: it reads H1's law from the answers, and it certifies.
- Round 7 (bar3/scale/family/FAMILY.md) found, offline, four general pieces that bring SP_BREADTH's family:
  1. the reading's windows as laws of their own;
  2. its constants;
  3. the 1 of the reader's pred and succ;
  4. a round kept only if it also gives the answers at T = 2, 4 and 8.
- This phase ports the four into the no-book file, legally. Then it recertifies H1, and runs the family at full size
  with the same file byte for byte, plus two held-out tasks chosen before any run.

**What changed** (`make_family.py`, from the no-book file's committed bytes):
- **Windows.** Every run of 2 or 3 of the reading's operations becomes a law of its own. Its holes (at most two) are
  its operands; N and the constants stay. For H1 that is (x·o) mod N.
- **Constants.** Each constant c of the reading is made from N, and x + c, x − c and x·c are posed.
- **CONST1** needs no code of its own. Round 7's reader wrote x − 1 as pred(x); the legal reader writes the 1 as a
  leaf, so the constants already pose it.
- **Their answers** are the law reader's own arithmetic, that is the learner's routines (the addition, the comparison
  and the link). No answer is made by Python's arithmetic.
- **ROUNDT.** A round must also give the newest 4,096 kept answers of every count T > 1, run T times. In H1's data
  (and the family's) the counts kept are 1, 2, 4 and 8, so T = 2, 4 and 8.
- **Bytes.** The port's 6.8 KB came from:
  - docstrings that a design doc already explains, cut to what they say plus a pointer to the doc;
  - shorter rules between a class's methods;
  - phase B's STEP_ADD switch.

  None of these changes any code; the file ends 493 bytes under the limit.

**H1: unchanged where it should be, and it certifies.**
- H1's tiny core is phase B's: the first 17 units identical in line and machine work, the reading poses one more law
  (the window), and every level's picks, the library, the 12 rounds and the answer programs are the same. The window's
  probes add 8% work.
- The official H100 run certifies: T = 64 on both ladders, every rung 768/768, test 9,999/9,999, unseen T and unseen
  moduli 10,002/10,002; training 783 s. So do seeds 1–7 ("8 of 8 records check out"); training takes 562–986 s
  against phase B's 472–634 s, almost all of it ROUNDT's checks in the round search.

**The family**, one H100 each, seed 74, the file byte for byte, each task's data as SP_BREADTH made it:

| task | learned? | T, both ladders | test | training | why |
|---|---|---|---|---|---|
| plusy, y² + y (held out) | **yes** | 64 / 64 | 9,999/9,999 | 631 s | H1's round with x in place of 0 |
| rho, y² + 1 | **yes** | 64 / 64 | 9,999/9,999 | 695 s | its window x·o + 1, with a 1 it made from N |
| the cube, y³ | **yes** | 64 / 64 | 9,999/9,999 | 825 s | two calls of its window (x·o) mod N |
| minus1, y² − 1 | **yes** | 64 / 64 | 9,999/9,999 | 1,534 s | ROUNDT threw out the rounds wrong at 0; the round kept adds N to x first |
| plus3, y² + 3 | no | – | 6/9,999 | (3,183 s) | it never makes 3; the work budget runs out |
| ten, 10y² | no | – | 6/9,999 | (the hour) | the reader reads nothing: 10 is a factor, not a gap |
| tenrho, 10y² + 1 | no | – | 12/9,999 | (the hour) | the reader reads nothing: depth 4 |
| three2, 3y² + 2 (held out) | no | – | 4/9,999 | (the hour) | the reader reads nothing: depth 4 |

Four of the eight are learned, one of the two held out among them. Each learned task certifies as H1 does: every
rung 768/768 on both ladders, and every split exact.

**What it means.**
- **The port works as round 7 said.** The windows carried the cube and rho, and the constants gave rho its 1.
  ROUNDT turned minus1 from wrong-at-0 into right, as in round 7, where the round kept also adds N before the
  subtraction. All four were used, legally: every new law's answers come from the learner's own routines.
- **The failures are the legal file's own:**
  - **ten, tenrho and three2 are lost at the reading.** Round 7's offline climbs read ten and tenrho with round 6's
    numpy reading; the legal reader's grammar (depth 3, constants only as gaps) does not reach them. three2 fails
    the same way, so the held-out pair splits along this line.
  - **plus3 is lost to the budget.** Round 7 made its 3 "over the link, with no 1"; here nothing in the probes' reach
    makes 3, and probing the 3-laws spends the climb's work.
- **What would move them** (not tried here): a reader that takes a constant as a factor and goes to depth 4; a
  constant c posed with the smaller constants that build it; a cheaper ROUNDT (it accounts for most of the extra
  training time: H1's 562–986 s against phase B's 472–634 s).
- **What stays designed** (§9): phase B's list, plus the window rule (3 operations, 2 holes), the constants' operations
  (add, sub, mul), ROUNDT's rows (4,096 of each count) and the reading's own patterns.
- **Not covered:** SP_BREADTH's mont, M4, xfirst and xy were not in this phase's list.

**Cost:** about $40 billed of the soft $150 (nine B200 tiny cores $6.43, H1's eight H100 runs $10.52, the
family's eight $22.14); every app stopped.
<!-- /PLAIN -->

## Details

### 1. Pre-registered

`PREREG.md`, frozen 2026-10-27 11:21:04 PDT, read-only, before any GPU run of this phase, its tiny cores included;
sha256 `ddde7d5cf3a89fcb78c0f52f722f12ec10e7483f053dd8c61b96f9a971cffeb1` (`checks/logs/prereg_sha256.txt`). Its
text is not repeated here; §8 sets each prediction against the outcome. One change after the freeze, outside the
submission: the runner copy's path line (§7).

### 2. The changes (`make_family.py`)

From the no-book file's committed bytes at f7dc5f8 (md5 `b42ec304` asserted), in three groups of exact edits, each
asserted to match once:

| group | edits | what |
|---|---|---|
| doc | 1 | the module's docstring names the port |
| port | 15 | READ_WINDOW, CONST_OPS and ROUNDT; the reading's own patterns (`n`, `x,n`, `w1`, `w2`), defined after the book's grammar so the law book is phase B's; `law_target` sends a reading's own law to `read_target`; LawReader takes the routines directly and gains `value` (a tree's value on rows); `tree`, `law_text`, `windows`, `constants`, `read_more`, `read_target`; `read_law` poses `read_more`'s laws with the cuts and loops and records them; a reading's own patterns bind their last operand to N; ROUNDT in the round search's acceptance (`round_t`); `keep_counts` keeps ROUNDT's rows (the newest 4,096 of each count) and `climb_unit` hands them to the climb |
| room | 36 + 1 | phase B's STEP_ADD switch (3 edits); 32 docstrings and one comment that a design doc explains, cut to what they say plus a pointer; the 14 rules between a class's methods shortened (combined_sp2's book check reads the 11 section rules, which stay) |

The result: `submission_stage3_family.py`, md5 `457216d746d7a21e854a2d2d3673c9d6`, 261,651 bytes (493 under the
limit); the contest's source check passes. An AST comparison with the docstrings left out: 13 functions changed and 12
new (four of them nested in others), all the port's; the room changes no code.

**Choices the port made, and why:**
- **ROUNDT's rows.** Round 7 checked every training row of T = 2, 4 and 8. A run sees the rows through its batches, so
  the file keeps the newest 4,096 of each count for ROUNDT, apart from the answer program's 64. With 64, ROUNDT would
  miss round 7's failure: minus1's wrong rounds fail only on rows that pass through 0, 24 of 33,333 at T = 4 and 61 at
  T = 8 (`checks/roundt_power.py`); 64 rows hold one with probability 0.05 and 0.11, 4,096 rows with 0.96 and 0.9997.
- **ROUNDT in the acceptance.** It sits next to the full-size check of the T = 1 answers, so a cost level whose
  programs all fail it counts as aliases and the search goes on to the next cost. Round 7 filtered the rounds after
  the search.
- **sub in CONST_OPS.** Round 6 posed x + c and x·c. The legal reader writes x − 1 with sub and the leaf 1 (phase A's
  gap constant, "the other way round"), so CONST1's pred is the constants' `sub.1`.
- **A window's answers** are masked where a value on the way is not computed: a sum must leave the top place free, a
  product's factors half the places, a difference must not go below 0, and a power's count is at most 8.

### 3. Checks (local, CPU)

| check | result |
|---|---|
| combined_sp2's `checks/local_combined.py`, every part (`checks/logs/local_family.log`: 278 s, peak 4.6 GiB) | the no-book file's results part for part (`checks/compare_local.py`), but the size; the book region's diff against book_gen_fix 105 lines against 92 (the port's patterns and dispatch, the Ops docstring), check_books part 2 the same (17 concepts, 6,204 targets). One full run of an earlier build gave the grad part a gradient on `prog` (8.3e-6); eight reruns of that part on each file gave none, as phase B's log (the tournament's starting weights are random), and the final build's run gives none |
| `checks/family_local.py` MODE=pose: each task's reading and what it poses, on the legal tiny core's state at level 3 (`logs/pose.log`) | the readings below; every posed law's answers equal Python's integers on every draw (60 law-pattern pairs, 0 disagreements) |
| `checks/family_local.py` MODE=roundt, `checks/roundt_rows.py` (`logs/roundt.log`, `logs/roundt_rows.log`) | phase B's four H1 rounds pass ROUNDT on 1,000, 4,096 and 4,096 rows; with one answer of T = 4 changed each fails, as does x² without its reduction. Rows stacked as `keep_counts` stacks them and read by `climb_unit`'s own expression give the counts 1, 2, 4, 8 and ROUNDT's T = 2, 4, 8 |
| `checks/roundt_power.py` (`logs/roundt_power.log`) | minus1's rows through 0: 24 of 33,333 (T = 4), 61 (T = 8); the tiny core's draw holds 1 and 5 |
| `core_f.py` on the CPU, minus1, 282 s | the fact read on minus1's own answers; ROUNDT's rows 1,000, 4,096, 4,096 |
| `tasks_family.py --small` with the competition's generator | 644 rows of plusy and of three2, every answer its formula's |
| `make_runner.py`, `--dry` | the eight family jobs on H100s with H1's manifest and `litmus_<task>`; H1's official job untouched |
| the watcher on a synthetic billing report | a stopped app's estimate ends at its stop ($17.83 where the old rule gave $27.83) |

What each reading poses (H1's 400 answer rows of T = 1, each task's answers by Python's integers, outside any run):

| task | read (rows of 400; laws weighed) | its own laws | candidates |
|---|---|---|---|
| H1 | `mod(mul(x,x),N)` (400; 200) | `w.mod.mul` on w2 | 102 |
| plusy | `mod(add(mul(x,x),x),N)` (400; 1,140) | `w.mod.add` on w2 | 150 |
| the cube | `mod(mul(mul(x,x),x),N)` (158; 1,140) | `w.mod.mul` | 102 |
| rho | `mod(add(mul(x,x),1),N)` (400; 3,825) | `const.1`, `add.1`, `w.mod.add1` (w1), `w.add1.mul`, `w.mod.add1.mul` | 163 |
| plus3 | `mod(add(mul(x,x),3),N)` (400; 8,025) | the same with 3 | 163 |
| minus1 | `mod(sub(mul(x,x),1),N)` (400; 3,825) | `const.1`, `sub.1`, `w.mod.sub1` (w1), `w.sub1.mul`, `w.mod.sub1.mul` | 115 |
| ten, tenrho, three2 | none (1,140) | none | 0 |

### 4. The tiny cores

`core_f.py` (combined_fast's `core_c.py` with the task's answers and ROUNDT's rows) on one B200 each through
`modal_family.py` (app and Volume `legal-family`): the file's own climb with its own budgets, offline, on 4,096 read
rows and 400 answer rows of T = 1, exact cells; then the answer program's search on 64 rows of each T = 1, 2, 4, 8.
Every answer program found was then run on both ladders, T = 1 to 64, seen and unseen moduli, with the task's answers
(`checks/ladder_family.py`, `checks/logs/ladder_t_family_<task>.log`).

| task | the round | its answer programs | both ladders | climb + answer search | 
|---|---|---|---|---|
| H1 | after level 4: phase B's 12 two-call rounds (the row's loop x·o, then the reduction) | phase B's two, `C.S[T.round1 STA.X]`, `C.S[T.round3 STA.X]` | 768/768 on all 14 rungs, both | 274 s; 1.21e9 + 0.11e9 work |
| plusy | after level 4: H1's with A = x in place of 0, `LDA.X STA.Y T.add.mul.o,x,o.XYA STA.Y T.red.x+on.YN` (x + x·x, then the reduction) | four (two end in the window `w.mod.add`) | all four 768/768 on all 14 rungs | 240 s; 1.23e9 + 0.11e9 |
| the cube | after level 5: two calls of the window, `LDA.X STA.Y T.w.mod.mul.w2.XYN STA.Y T.w.mod.mul.w2.XYN` | one | 768/768 on all 14 rungs | 326 s; 2.22e9 + 1.47e9 |
| rho | after level 5: the window x·o + 1, then the reduction (`T.w.add1.mul.w2.XYN`, `T.red.x+on`) | two | both 768/768 on all 14 rungs | 332 s; 2.78e9 + 0.11e9 |
| minus1 | after level 6 (level 5's search found none): x + N or N − x, then the window (x·x − 1) mod N | three | two 768/768 on all 14 rungs; the third (round 4: N − x reached through Y) wrong on 151 of 10,752 rows | 910 s; 3.47e9 + 0.79e9 |
| plus3 | none: the climb's work budget ran out at level 6 (4.42e9) | – | – | 453 s |
| ten, tenrho, three2 | none: nothing read, so nothing posed past the fact's three laws | – | – | 88–113 s; 0.10e9 |

**H1 against phase B** (`checks/compare_cores.py`, `logs/compare_cores.log`):
- the first 17 units are the same, line for line (seconds masked) and unit for unit in machine work: levels 0–2 and
  the reading;
- level 3's pose adds the window's 6 roots (102 candidates against 96);
- every level's picks are the same, the 50-entry library is the same, and so are the 12 rounds and both answer
  programs;
- the climb's work is 1.32e9 against 1.22e9 (the window's probes and deep probes), in 274 s against 223 s.

**How the windows were used:**
- **The cube's** round is two calls of `w.mod.mul`, learned at level 5 as the row's loop x·o and the reduction (`LDA.N
  T.add.mul.o,x,o.AXY STA.Y T.red.x+on.YN`).
- **rho's** window `w.add1.mul` (x·o + 1) was learned at level 5 as `CLA T.add.mul.o,x,o.XYA STA.X T.const.1.n.N
  T.add.pair.AX`. Its 1 is the constant learned at level 3, `T.const.1.n.N = CLA STA.Y T.red.pair.YN ADC.Y`: the
  link's borrow, added to zero.
- **minus1's** window `w.sub1.mul` (x·o − 1) came at level 5. The plain round over it, the window and then the
  reduction, is right on all 400 answers of T = 1 but wrong on 1 row of T = 4 and 4 of T = 8 among ROUNDT's rows (the
  rows through 0), so ROUNDT rejected it (`checks/roundt_minus1.py`), and level 5's round search found none.
  - At level 6 the climb learned `w.mod.sub1.mul` ((x·o − 1) mod N). The round search then found rounds that first
    move x to x + N or N − x and then call it. (x + N)² ≡ (N − x)² ≡ x², and the product is never 0, so they are
    right at 0.
  - One of the three answer programs (over round 4, `LDA.X STA.Y LDA.N T.lt.pair.AY STA.Y STA.X …`: N − x reached
    through Y) passed the search's 256 rows and 223 of 224 unseen rows, yet is wrong on 151 of the ladders' 10,752
    rows. Round 2 reaches N − x directly and is right everywhere; why round 4 differs was not traced. The official
    run kept two answer programs, and its ladders are all 768/768.
- **plus3** never learned to make 3 (`const.3` stayed unsolved, and with it every 3-window). Its probes spent the rest
  of the budget.

### 5. H1: the official run

`modal_runner_family.py --mode official` (H1's manifest untouched, seed 74, one H100), record
`records/legal_family_official.json`. `tools/verify_records.py`: "1 of 1 records check out". The embedded source's md5
is the file's; RESULT_JSON equals the result.

| | phase B (the no-book file) | this file |
|---|---|---|
| T on both ladders | 64 / 64 | 64 / 64, every rung 768/768 |
| test, unseen T, unseen moduli | 9,999/9,999, 10,002/10,002, 10,002/10,002 | the same |
| the answer program | step 630 | step 632 |
| frozen | step 829 | step 831 |
| training, evaluation | 574 s, 398 s | 783 s, 368 s |

### 6. Seeds 1–7

`modal_runner_family.py --mode seeds` (seeds 1–7, H1's manifest with only the seed changed, one H100 each), records
`records/legal_family_h100_seed<N>.json`. `tools/verify_records.py` on the official record and the seven: "8 of 8
records check out" (`checks/logs/verify_h1.log`).

| seed | the answer program, step (phase B) | frozen (phase B) | the climb's work at the round | training s (phase B) | evaluation s |
|---|---|---|---|---|---|
| 74 (official) | 632 (630) | 831 (829) | 1.25e9 | 783 (574) | 368 |
| 1 | 653 (651) | 852 (850) | 1.10e9 | 568 (472) | 338 |
| 2 | 649 (651) | 848 (850) | 1.26e9 | 783 (582) | 355 |
| 3 | 676 (675) | 875 (874) | 1.38e9 | 804 (594) | 394 |
| 4 | 716 (719) | 915 (918) | 1.39e9 | 724 (634) | 380 |
| 5 | 673 (671) | 872 (870) | 1.32e9 | 986 (585) | 369 |
| 6 | 687 (689) | 886 (888) | 1.28e9 | 562 (476) | 289 |
| 7 | 632 (633) | 831 (832) | 1.34e9 | 570 (616) | 304 |

**Every seed certifies**: T = 64 on both ladders, every rung 768/768, test 9,999/9,999, unseen T and unseen moduli
10,002/10,002. The answer program comes within 3 steps of phase B's on every seed, the freeze too.

**Training is slower: 562–986 s against 472–634 s.** Almost all of it is ROUNDT. The round search after level 4 takes
187 s (official) and 344 s (seed 5) against phase B's 14 s and 21 s: it accepts 12 rounds at once, and ROUNDT runs
each 14 times at full size (2, 4 and 8 times) on up to 4,096 rows of each count. The window's probes add about 20 s.
A cheaper ROUNDT would stop once the answer program's PROG_SLOTS rounds pass, or run T = 8 alone; the file was
frozen, so it is left as it is.

### 7. The family at full size

`modal_runner_family.py --mode plan` (`plans/family.json`): each task on one H100, seed 74, the file byte for byte
(md5 `457216d7` in every record), H1's manifest with only the name and `data.data_root` changed (`litmus_<task>`), as
SP_BREADTH ran them. The data is SP_BREADTH's: each record's data md5 equals its SP_BREADTH record's (ten `f02bdb5a`,
rho `b19d39e7`, plus3 `a9d9e672`, minus1 `3089dc1e`, tenrho `0b18439b`, the cube `75a99480`). plusy's (`feb26ee2`)
and three2's (`343745f1`) were made by `tasks_family.py` in the runner's image. Summary:
`checks/summarize_family.py` (`checks/logs/summarize_family.log`).

| task | learned? | T, seen / unseen moduli | test | training s | the round (from the run's log) | why not |
|---|---|---|---|---|---|---|
| H1 (official) | **yes** | 64 / 64 | 9,999/9,999 | 783 | after level 4: the row's loop x·o, then the reduction | – |
| plusy (held out) | **yes** | 64 / 64 | 9,999/9,999 | 631 | after level 4: x + x·x, then the reduction | – |
| rho | **yes** | 64 / 64 | 9,999/9,999 | 695 | after level 5: the window x·o + 1 (its 1 the learned `const.1`), then the reduction | – |
| the cube | **yes** | 64 / 64 | 9,999/9,999 | 825 | after level 5: two calls of the window (x·o) mod N | – |
| minus1 | **yes** | 64 / 64 | 9,999/9,999 | 1,534 | after level 6: x + N (or N − x), then the window (x·o − 1) mod N | – |
| plus3 | no | – / – | 6/9,999 | 3,183 | none | the constant 3 is never made (`const.3` unsolved at every level), so no 3-window is reachable; the probes on them spend the climb's work budget (1.24e9 to 3.67e9 during level 5) and it runs out at level 7 |
| ten | no | – / – | 6/9,999 | 3,603 | none | nothing read: 10·y² has no law over x, N and a gap constant at depth 3 (10 is a factor, not a gap) |
| tenrho | no | – / – | 12/9,999 | 3,601 | none | nothing read: 10·y² + 1 needs depth 4 |
| three2 (held out) | no | – / – | 4/9,999 | 3,600 | none | nothing read: 3·y² + 2 needs depth 4 |

Every learned task: every rung 768/768 on both ladders, and unseen T and unseen moduli 10,002/10,002. Each found
its answer programs (2 to 4, the simplest `C.S[T.round1 STA.X]`) one step after its round, and froze 199 steps
later.

**How each was learned** (the official runs, as their tiny cores):
- **plusy** needs nothing new. Its round is H1's with A = x in place of 0, and its window (x + o) mod N is practised
  but not used.
- **rho** uses its window and its constant. `w.add1.mul` (x·o + 1) was learned from the row's loop, the learned 1 and
  the addition, and its round is that window then the reduction.
- **The cube** uses its window twice. Its reading, `mod(mul(mul(x,x),x),N)`, was read on 71 of the 141 answer rows
  (the others' x² does not fit half the places).
- **minus1** uses ROUNDT. The plain round over `w.sub1.mul` is wrong where the square is 0, and ROUNDT rejected it on
  the data's own rows of T = 4 and 8 (§4). The climb went on to `w.mod.sub1.mul`, and the round search found rounds
  that first move x to x + N or N − x. They are right at 0. Its level-6 round search took 744 s, most of it ROUNDT.

**Why each failed:**
- **plus3, the climb's budget.** Its reading poses `const.3`, `add.3` and three windows with 3. Nothing in reach makes
  3: rho's 1 is a four-instruction program (the link's borrow added to zero), and 3 would need that and two additions.
  So none of the five is ever learnable. The probes and deep probes on them use the rest of the climb's work (2.4e9 in
  level 5 alone); at level 7 the climb stops at the budget, and the round search only retries.
- **ten, tenrho and three2, the reader's grammar.** The fact read (every answer below N) poses the step's three laws,
  and those are practised. The law reader then tries at three levels and reads nothing:
  - 10·y² mod N is no chain over x and N of depth 3 with a gap constant: 10 is a factor, not a gap;
  - 10·y² + 1 and 3·y² + 2 need depth 4.

  With nothing read, nothing more is posed. The climb runs to its level cap, and the round search retries every 10
  steps until the hour ends.

**The runner copy, fixed after the freeze.** Its first launch failed at import in the container: the copy's path line
(`HERE.parents[1]`) has no such parent at /root. No job ran and nothing was billed. The line was made to work in both
places (`make_runner.py`), and the official run went out with `--force`, the first launch having registered the
record's name. The submission file did not change.

### 8. Predictions against outcomes

| what | prediction (confidence) | outcome |
|---|---|---|
| t_family_h1: levels 0–2 | t_nobook's, unit for unit (high) | **right**: the first 17 units the same in line and work |
| t_family_h1: level 3 | the same reading; its pose adds the window on w2, 102 candidates against 96 (high) | **right** |
| t_family_h1: levels 3–4 and the round | t_nobook's picks and two-call round, or the window and a one-call round (medium) | **right, the first branch**: the same picks, library, 12 rounds and answer programs |
| t_family_h1: both ladders | 768/768 on all 14 rungs (high) | **right** |
| H1, official run | certifies (medium-high) | **right** |
| H1, official run: training | 450–900 s (medium) | **right**: 783 s |
| H1, seeds 1–7 | 7 of 7 certify (medium-high) | **right** |
| plusy | learned: H1's round with A = x, two calls (medium-high) | **right**: that round, after level 4; training 631 s |
| the cube | learned: two calls of `w.mod.mul` (medium-low) | **right**: after level 5; training 825 s |
| rho | learned: a window with its + 1 (medium-low) | **right**: `w.add1.mul` and the reduction, after level 5; training 695 s |
| plus3 | learned, as rho with 3 (low) | **wrong**: not learned. The constant 3 is never made, so no 3-window is reachable, and the climb's work budget runs out at level 7 (tiny core: level 6) |
| minus1 | not learned: a round over its windows is wrong at 0 and ROUNDT rejects it (medium) | **wrong**: ROUNDT did reject the plain rounds, but the climb went on to `w.mod.sub1.mul` and rounds that first move x to x + N or N − x, which are right at 0; learned after level 6, training 1,534 s |
| ten, tenrho, three2 | not learned: nothing read (high) | **right**: no reading at any of the three tries; test 6, 12 and 4 of 9,999 |
| ROUNDT | keeps every first round that is right (high) | **right** where it can be told: H1's 12 rounds are phase B's, found with no ROUNDT. plusy's, the cube's and rho's rounds are right on both ladders (no run without ROUNDT to compare). On minus1 it rejected the plain rounds, wrong at 0 |
| the family's runs: training | 450–1,000 s for plusy; up to the hour for the rest (medium) | **right**: plusy 631 s; rho, the cube and minus1 695–1,534 s; plus3 to its 20,000 steps (3,183 s); ten, tenrho and three2 to the hour |
| money, phase C | about $60–100 (medium) | **lower**: about $40 |

### 9. What is still designed

Phase B's list (NOBOOK.md §7) stands: the residue rule, the step's cut table, the reader's grammar, the cut and loop
rule, the catalog and fast2's search. This phase adds:
- **The window rule:** runs of 2 or 3 operations, at most 2 holes (READ_WINDOW), named a and b in order; N and the
  constants kept; posed smallest first on w1 or w2, the modulus bound to N.
- **The constants rule:** each constant made from N, and the reading's add, sub and mul with it (CONST_OPS).
- **ROUNDT's rows:** the counts T > 1 that keep_counts keeps, the newest 4,096 of each.
- **The reading's own patterns** (`n`, `x,n`, `w1`, `w2`) and their widths (same, carry, double).

### 10. Where it still touches Python

As the no-book file (NOBOOK.md §8), plus:
- **A window's tree** is parsed from the reading's text (`tree`), and its name and the constants' digits are made from
  that text. Every value of the tree is computed by the learner's routines through the law reader.
- **ROUNDT's T** is read off each kept count's digits by the file's `count_of` (the count loop's reading), and the
  round runs T times on the GPU. One bit per candidate (right on every row, or not) is read back.

### 11. Cost

`cost.py` (`checks/logs/cost.log`, the billing report's complete hours, read at 13:04 PDT):

| app | what | billed |
|---|---|---|
| ap-OaSkeF0NN94IgRrvJQG77i | the nine tiny cores (B200s) | $6.43 |
| ap-47GRV2nMoyyPlRPdJx0P4C | the official run's first launch (failed at import; no job ran) | $0.00 |
| ap-kMy7lHDCkebWeYGrk61Dng | a launch the runner refused (the record's name was taken) | $0.00 |
| ap-7MCyQ0iwCFmtLA5gD1KEeT | H1's official run (H100) | $1.46 |
| ap-Lbe9N0JDKXkjKMHkOcB3FT | H1's seeds 1–7 (H100s) | $9.06 |
| ap-SChXfyNyNUutgLz60AVZPM | the family, eight tasks (H100s) | $22.14 |
| all | | **$39.07**, with the family's last minutes (the 20:00 UTC hour) about $1 more |

Soft $150, cap $220; the month stood at $4,056 when the first app started, and `spend.py 4750` passed before every
launch. The spend watcher (`logs/watch_*.log`) watched every app: its total was the cap less one hour of the GPUs in
flight ($164 for the nine B200s, $160 once fifteen H100s ran), and $90 an app. It never came near either, and it
estimated stopped apps only up to their stop. Every app has stopped, and so has the watcher.

### 12. Files and how to rerun

Everything is in `cpu_first/integration/legal_family/`; nothing outside it was written.

| file | what |
|---|---|
| `submission_stage3_family.py` | the file, md5 `457216d746d7a21e854a2d2d3673c9d6`, 261,651 bytes (493 under the limit), written by `make_family.py` |
| `make_family.py` | the builder (§2): the no-book file's committed bytes, md5 asserted, then the doc, port and room edits |
| `PREREG.md` | frozen; `checks/logs/prereg_sha256.txt` holds its hash and the code's |
| `core_f.py`, `modal_family.py` | the tiny core (core_c.py with the task's answers and ROUNDT's rows) and its launcher (app and Volume `legal-family`) |
| `tasks_family.py` | the held-out tasks plusy and three2 (H1's generator, other answers) |
| `make_runner.py`, `modal_runner_family.py` | the shared runner with plusy and three2 in its image and task list, and its paths for this folder |
| `watch_spend.py` | round 7's spend watcher with a stopped app's hour under way ending at its stop |
| `cost.py` | §11: the billing report's hours for this phase's apps |
| `plans/family.json` | the eight family jobs |
| `checks/family_local.py` | the readings and their posing (MODE=pose), ROUNDT on phase B's rounds (MODE=roundt) |
| `checks/roundt_rows.py`, `checks/roundt_power.py`, `checks/roundt_minus1.py` | ROUNDT's rows as a run keeps them; its reach on minus1; minus1's plain rounds |
| `checks/ladder_family.py` | a tiny core's answer programs on both ladders, with the task's answers |
| `checks/compare_local.py`, `checks/compare_cores.py` | combined_sp2's local checks against the no-book file's; H1's tiny core against phase B's |
| `checks/logs/` | the checks' logs |
| `records/` | the tiny cores (`t_family_<task>.json`, `.log`), H1's official record and seeds 1–7's (`legal_family_official.json`, `legal_family_h100_seed<N>.json`), the family's (`family_<task>_h100.json`) |
| `logs/` | the spend watchers |

From `cpu_first/`, with `PY=~/github/sotaku/venv/bin/python` and `PYTHONDONTWRITEBYTECODE=1` exported:

```bash
$PY integration/legal_family/make_family.py
$PY integration/legal_family/make_runner.py
$PY integration/legal_family/checks/family_local.py
MODE=roundt $PY integration/legal_family/checks/family_local.py
$PY integration/legal_family/checks/roundt_rows.py
TASK=minus1 $PY integration/legal_family/checks/roundt_power.py
COMBINED_FILE=integration/legal_family/submission_stage3_family.py PYTHONPATH=. $PY integration/combined_sp2/checks/local_combined.py policy grad machines isa designed buffers replay climb climb_tiny fix book
python3 integration/legal_family/checks/compare_local.py
$PY integration/ablation_s3/spend.py 4750
$PY -m modal run --detach integration/legal_family/modal_family.py --envs '[{"TAG": "t_family_h1", "TASK": "h1"}, {"TAG": "t_family_plusy", "TASK": "plusy"}, {"TAG": "t_family_cube", "TASK": "cube"}, {"TAG": "t_family_rho", "TASK": "rho"}, {"TAG": "t_family_plus3", "TASK": "plus3"}, {"TAG": "t_family_minus1", "TASK": "minus1"}, {"TAG": "t_family_ten", "TASK": "ten"}, {"TAG": "t_family_tenrho", "TASK": "tenrho"}, {"TAG": "t_family_three2", "TASK": "three2"}]'
$PY -m modal run integration/legal_family/modal_family.py --mode fetch
TASK=h1 $PY integration/legal_family/checks/ladder_family.py integration/legal_family/records/t_family_h1.json
python3 integration/legal_family/checks/compare_cores.py
TASK=minus1 $PY integration/legal_family/checks/roundt_minus1.py
$PY -m modal run --detach integration/legal_family/modal_runner_family.py --mode official --file integration/legal_family/submission_stage3_family.py --name legal_family_official --out integration/legal_family/records --force
$PY -m modal run --detach integration/legal_family/modal_runner_family.py --mode seeds --file integration/legal_family/submission_stage3_family.py --seeds 1,2,3,4,5,6,7 --hardware h100 --name legal_family --out integration/legal_family/records
$PY -m modal run --detach integration/legal_family/modal_runner_family.py --mode plan --plan integration/legal_family/plans/family.json --out integration/legal_family/records
$PY integration/legal_family/modal_runner_family.py --mode fetch --out integration/legal_family/records
python3 tools/verify_records.py integration/legal_family/records/legal_family_official.json=integration/legal_family/submission_stage3_family.py integration/legal_family/records/legal_family_h100_seed{1,2,3,4,5,6,7}.json=integration/legal_family/submission_stage3_family.py
$PY integration/legal_family/cost.py
```

(The ladder check takes TASK for each task's tiny core; `--force` on the official launch because the runner copy's
first launch, which failed at import, had registered the record's name.)
