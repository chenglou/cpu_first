# The countdown over T, learned instead of written (bar 2, OVERVIEW §7 question 4)

*Written 2026-10-24. Code in `cpu_first/integration/learned_loop/`, records in `integration/learned_loop/records/`.
Nothing committed, nothing submitted, no certified file touched (the builder asserts their md5s).*

## In plain words

<!-- PLAIN -->
**The question.** Stage 3 learns the round of x^(2^T) mod N, "copy x, multiply, reduce", inside training. The
repetition around it was written by hand: subtract 1 from T, and while that does not go below zero, run the round
again on its own answer. Can the learner find that loop itself?

**What changed.** Three hand-written things are gone from sp2's file: the countdown, the gate that stopped it (with
the "loop law" that taught the gate), and the round search's assumption that the answer at T = t is the round run t
times. In their place:
- the digit machine gets one generic instruction, the count loop `C.r[body]`: run the body as many times as register
  r holds, counting a copy down with the learned subtraction table;
- the prompt's third number goes into a register at the start, as x and N do;
- once the round is found, a second search looks for the whole answer's program, composing the round, the other
  routines, register moves and count loops, against answers at several values of T.

**Does the learner find the loop? Yes.** In every run where the round came, the next search found
`C.S[T.round1 STA.X]`: count the third number down; each time run the round and copy its answer back into x's
register. That is the countdown, found rather than written. The search also had to work out which number counts the
rounds (T, not x or N), that the round runs exactly T times, and that each answer feeds the next round. It takes one
search of 29–96 s over about 14,000 candidate items, and it needs answers at two different values of T (from T = 1
alone it returns the round itself, with no loop).

**Does it certify?** On the untouched official manifest, on an H100, yes: T = 64 on both ladders, every rung
768/768, every split exact, frozen by 1,002 s of the 3,600 s budget. On 8 B200 seeds, 7 certify. Seed 2 runs out of
time before any loop is searched: sp2's practice climb, unchanged here, took 10 levels on that seed's path to build the
round's last piece, the long division, and finished it just as the climb's time ran out. With the budget doubled (a
research run), the same seed finds the round, then the loop, and certifies, frozen by 3,741 s. And the slow path is
not the loop's: with sp2's own random start (a diagnostic copy that keeps sp2's unused gate), seed 2 replays sp2's
climb exactly, finds the round at level 4, and the learned loop certifies it, frozen by 1,062 s.

**How T is read.** That is designed too, and part of it carries weight. The reader takes T as the prompt's third
number with exactly 2 digits, and that width is what tells T apart from x: both are below N, so the order facts the
reader learns from cannot separate them. Read T as wide as x (8 digits), and the reader takes x from T's digits on 2 of
5 local seeds. Through the evaluator, 2 of 3 seeds take that wrong reading and never get past learning the
subtraction table; the third certifies.

**Still designed:** the count loop itself (generic, but its stop rule is the old countdown's), T's place in the
machine, the answer program's search space and sizes, T's 2-digit field, and everything sp2 still designs (the law
book, the choosing rule, the instruction set, the round's search). A strict judge can call the count loop the
countdown made into an instruction. What is no longer written anywhere is the decision to repeat the round, how many
times, and on what.
<!-- /PLAIN -->

## 1. What was written by hand, and what replaces it

A prompt gives N, x and T; the answer is x squared T times, the remainder by N kept each time. Stage 3 (s3i, sp2) learns
the round, "copy x, multiply, reduce", inside training. The repetition around it was written by hand, in three places:

| hand-written in sp2 | where | what it said | here |
|---|---|---|---|
| the countdown | the forward pass (`predict_ste`) and training's checks (`run_rounds`) | "subtract 1 from T with the subtraction table; while that does not borrow, run the round on x and feed its answer back as x" | **gone**. The forward pass runs a program found in training. The only loop with a data count is one generic instruction (§2) |
| the gate and its lesson | a 2 x 2 parameter `g` and the loop law (`loop.count`: "a count of at least 1 runs its round; a count of 0 stops the loop") | the countdown's stopping rule, as a law the gate was taught | **gone**: no gate parameter, no loop law |
| what the round is | the round's search (`spcore.round_ok_t`) | "the answer at T = t is the round run t times": a candidate was run t times against the answers of the smallest T | **gone**. The round's search takes the smallest T's answers as they are (on H1, T = 1). How T enters the answer is found by a second search (§3) |

What replaces them:

- **One instruction kind, the count loop `C.r[body]`** (§2): the body runs as a routine as many times as register r
  holds. A hidden copy of r is counted down by the learned subtraction table, and the loop ends when that count down
  borrows. It works on any register (A, X, Y, N, S), and any body can go inside.
- **T in the machine.** The prompt's third number goes into register S at the start, as x goes into X and N into N.
- **A second search** (§3): once the round is found, the answer's program is searched over the round (as a routine),
  the library's other routines, the whole-register instructions and count loops. Its goal is the answers of rows of
  every T whose count reads the same both ways, so T's reading direction is not needed yet.

**An example** (the training row of OVERVIEW §1, with T = 4): `N 9 8 1 4 X 1 3 8 5 T 4`. The machine starts with
X = 1385, N = 4189 and S = 4. The answer's program found in training is `C.S[T.round1 STA.X]`: T.round1 is the round
(x² mod N, into A) and STA.X copies its answer back into X. The count loop takes S's 4, counts it down 4 → 3 → 2 → 1
→ 0 with the subtraction table and runs the body each time: 1385 → 3852 → 466 → 3517 → 3361. The fifth count down
(0 − 1) borrows, and the loop ends. A holds 3361 = 1385^16 mod 4189. Nothing in the file says "repeat", "T times" or
"feed the answer back": the count loop is a generic instruction, and that program is what the search found to
reproduce the answers.

## 2. The count loop

**What it does.** `C.r[body]`, for any register r of the five (A, X, Y, N, S):
1. As the loop begins, it takes a hidden copy of r's digits: its count. The body can then use and overwrite r freely,
   like the hidden counters of the P, B and W loops (which repeat by the register's length, the base and the operand
   width).
2. Each round it counts the copy down by one with the machine's own subtraction cell: the digit 1 at place 0, the
   borrow rippling up every place, exactly as the old countdown subtracted 1 from T. If that count down borrows, the
   count is used up and the loop ends.
3. Otherwise the body runs as a routine: it starts at the loop's pointer, and the pointer comes back there after it.
   So the pointer never depends on the data, as everywhere else in the machine.
4. Each probe (row) runs its own number of rounds. A count above `LOOP_CAP` = 100 is an error: that row's registers
   and flags are all cleared, and the loop ends for it.

The cap is a size, like the registers' 16 places. It is the old countdown's own cap (10^K rounds, K = 2), and it keeps
the search cheap: a loop on a register that holds N (four to eight digits), or almost any x, finds its count above
100 at its 101st count down, before any round runs, and costs only the count downs.

**Where it lives.** It is the machine's only loop whose count is data, so every machine that runs programs needed it:
- `run_items`, the reference machine (`libcore._run_count`);
- `apply_items`, the search's machine (`_apply_counts`): every count loop's rows run their bodies together, one batched
  call per round, each probe keeping the result while its own count lasts;
- `OuterMachine`, phase D's (training's checks): each stretch of plain items and each loop body is s3i's
  `RoundMachine` on one shared state, replayed from a CUDA graph;
- `SteMachine`, the prediction path's, on one-hot digits: the count down through the straight-through subtraction
  table, and each round a blend of the body's state and the old one by the count down's borrow (with one-hot digits
  the blend is exact; it is how stage 1's countdown blended its rounds).

It is not in the practice problems' searches (the climb): those problems have no count field, and a count loop there
would offer, for instance, "multiply by repeated addition", correct only below the cap. Count loops appear only at the
top of the answer's program.

**Checked** (`checks/test_count.py`, local, 12 s): the four machines give the same registers, flags and pointer on 46
programs with count loops (random bodies of 1–4 primitives, plain items before and after, loops on S and on other
registers, counts 0–12 and one of 150 to hit the error). And at full size (16 places) with sp2's recorded library, the
program `C.S[T.round1 STA.X]` gives x^(2^T) mod N on real rows for T = 1, 2, 4, 8 and 16.

## 3. The answer program's search

**When.** In phase C, once the climb's round search (sp2's, unchanged) has found the round: a program for the answers
of the smallest T the batches showed, taken as they are. On H1 that is T = 1, and the round is x² mod N. The next
step's climb unit is the answer program's search (`spcore.SelfClimb.search_outer`), one unit.

**The vocabulary.**
- The round programs, each a routine `T.round<i>` (the 4 simplest, as sp2's tournament took).
- Every routine the climb learned, and the 18 instructions that act on whole registers (load, store, clear, shift,
  select, bring in), each one item.
- A count loop on each of the five registers over every distinct body of one or two of those items. Bodies are
  deduplicated by what they do on random machines; the loops are not, since on random machines every count is above
  the cap.

On sp2's official library (62 routines) that is 15,421 items: 63 single items and 3,071 bodies × 5 registers. The
search goes breadth-first by cost up to cost 3, at most 10^6 states a level (`OUTER_COST`, `OUTER_BODY`,
`OUTER_STATES`).

**The goal.** The machine starts with x in X, N in N and the prompt's third number in S; the rest is random (the
dirty start every search uses, so a program must set up what it reads). The goal: A holds the answer.
- The rows are the training batches' answer rows whose count reads the same both ways: T's two readings agree. On
  H1 those are T = 1, 2, 4 and 8, never 16, so T's direction (16 or 61?) is not needed yet. They are kept per count,
  the newest 64 of each (`COUNT_EACH`, `COUNT_POOL`).
- The 8 probes are taken in turn from each count. A program is accepted when it reproduces every kept row.
- The search waits until the rows show two counts (`COUNT_MIN`). With one count nothing tells a loop from no loop:
  on T = 1 rows alone the search returns the round itself (below).

**What it finds.** Offline first (`outer_offline.py`: sp2's official run's recorded library and rounds, exact cells,
H1 training rows; records in `records/offline/`), then checked on 224 rows it never saw (32 each of T = 1, 2, 4, 8,
16, 32, 64; T = 32 and 64 from the depth splits):

| rows given | vocabulary | found | cost | levels (states) | seconds | unseen rows right |
|---|---|---|---|---|---|---|
| T = 1, 2, 4, 8 | every routine (15,421 items) | `C.S[T.round1 STA.X]` | 3 | 1, 63, 2,293, 100,571 | 60.5 (B200) | 224 / 224 |
| T = 1, 2, 4, 8 | every routine, one round | the same | 3 | the same | 72.4 (B200) | 224 / 224 |
| T = 1, 2 | every routine | the same | 3 | 1, 63, 2,292, 100,187 | 57.3 (B200) | 224 / 224 |
| T = 4, 8 | every routine | the same | 3 | 1, 63, 2,283, 99,407 | 35.2 (B200) | 224 / 224 |
| T = 1 | every routine | `T.round1` (no loop) | 1 | 1, 63 | 6.8 (B200) | 32 / 224 (T = 1 only) |
| T = 1, 2, 4, 8 | the rounds and the 18 instructions (1,099 items) | `C.S[T.round1 STA.X]` | 3 | 1, 19, 194, 2,707 | 33.6 (laptop CPU) | 224 / 224 |

On every arm with two counts or more it is the only program the search accepts, at cost 3. Read as a program, it says
what the hand-written countdown said: count S (the prompt's third number) down to zero; each time run the round on X
and N, then store the answer in X for the next round. Two smaller facts fall out of it:
- the second round program (sp2 found two, the same function) is merged with the first, since both do the same on
  random machines, so the tournament holds one program instead of two;
- the loop counts S, not X or N: a loop on N ends in the cap's error on every probe (every N is above 100), and one on
  X on almost every probe, so neither reaches the goal.

**Why not one search for the loop and its body together** (the brief's first way). The body is the round plus the
store back into X, 7 items: `LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA STA.X`. As a loop body over sp2's 94
tokens (32 primitives and 62 routines) that is about 94^7 ≈ 6·10^13 candidate bodies. Over sp2's round search "by
calls" (routines with their inputs routed in, 451–477 items) it is 3 items, about 10^8 bodies for each of 5 registers.
Both are far beyond a search that stores 10^6–10^7 states a level. The rows whose count is 1 are where a count loop's
body shows itself exactly once, so finding the body there first (the round's search), then the loop around it, is
what makes it affordable.

## 4. The file

[`submission_stage3_ll.py`](submission_stage3_ll.py), md5 `419c15e70ccf6483ff68b04b08f52bdc`, 240,201 bytes (sp2:
250,643), 207,370 learned numbers in 15 tensors (sp2: 207,374 in 16; the gate's 4 are gone). It passes the
competition's static source policy. It is built by [`make_ll.py`](make_ll.py), and rebuilding it from these sources
gives the same bytes (`make_ll.build()`'s md5 is the file's):
- [`stage3_ll_template.py`](stage3_ll_template.py), sp2's template with these changes:
  - out: `predict_ste`'s countdown loop, `round_ste`, `run_rounds`, the gate parameter `g` and its lesson in
    `cell_lessons`, the loop law, and the gate in phase B's switch;
  - in: `outer_ste` (the prediction path: the selected answer program on the one-hot machine, x, N and T's digits in
    X, N and S), `run_outer` (training's checks, on `OuterMachine`), `keep_counts` (the rows of every count), and
    `outer_unit` (the answer program's search, one climb unit, again after `OUTER_RETRY` = 50 steps or a new count
    while none is found);
  - changed: `t_readings` factors out T's two readings, which `t_lesson` and `keep_counts` both use; the tournament
    and T's lesson run the answer program instead of the countdown;
- [`libcore.py`](libcore.py), integration/lib's with the count loop (§2), `OuterMachine`, and s3i's practice tasks and
  its `Climb` left out (sp2 never runs them: 23 KB of dead code, and room under the 256 KiB limit);
- [`spcore.py`](spcore.py), self_practice's with `round_ok_t` and `_round_many` out (the round as "run t times") and
  `search_outer` and `outer_classes` in;
- stage 1's reader and writer, extracted by name as for sp2.

Everything else is sp2's: the reader, the cells and their lessons, the law book and the choosing rule, the climb and
its searches, the round's search, the tournament, phase D's freeze. The training does not replay sp2's step for step,
though. Without the gate's parameter every later tensor starts from different random draws, and the evaluator clips
the gradient's norm at 1 over all parameters, which the gate's lesson used to share.

The first build, `submission_stage3_ll_v1.py` (md5 `ba07d656`, saved from its record), had one bug: the rows kept per
count were trimmed a whole batch at a time, and a batch of T = 4 or T = 8 rows (about 167 of each) is bigger than the
64 kept, so those counts were always emptied. Its answer program was searched on T = 1 and 2 rows only, which still
identify the loop (the offline T = 1, 2 arm above), and it certified (§5). The final file keeps each count's newest
rows, with the comments that describe that, and fixes three stale comments (`K`'s, the one-hot machine's and the
library field's); the two builds differ in nothing else.

**Plumbing checks** (local, CPU):
- `checks/designed_ll.py`: the reader and every cell at the truth, sp2's library, the answer program given. The
  prediction path is exact on depth_t_1, 8 and 16 and depth_ood_n_t_2 and 64, and at T = 64 its count downs meet 24
  rows of the subtraction table, as many as sp2's countdown met in its official run;
- `checks/smoke_replay.py`: a finished climb holding sp2's rounds, then real training steps. The count pool fills, the
  answer program is found inside the file at the next step (84 rows of 3 counts, 68 s on the laptop), T's direction
  is resolved by it, and the model freezes;
- `integration/checks/stage3_train_smoke.py` with this file: phases A to C from step 1, no errors.

**Rule 8's gradient test** (`checks/gradtest_ll.py`, local; s3i's test, `integration/checks/stage3_gradtest.py`, run
on this file and on sp2's). Setup: the reader and every cell at the truth, two contestants (the count loop over each
of sp2's two rounds), and the right tokens' logits as the objective, on 2 rows each of T = 1 and T = 2. The gradient
from the final logits goes through the count loop to the subtraction table (both outputs) and on to every reader
tensor, and to the multiply table, as small as in sp2 (1e-15 to 1e-18). Three things look missing; the first two hold
for sp2's file in the same test:
- **the addition table (`aa`, `ab`) is not in the graph.** sp2's round never adds: its multiply accumulates in the
  multiply-accumulate cell, and its reduction subtracts. s3i's rounds had `T.add` among their contestants, so s3i's
  test reached it;
- **the tournament's logits get exactly 0.** Both contestants compute the same function, and the derivatives of a
  softmax sum to zero, so equal outputs cancel;
- **at T = 2, the reader's two unused pointer shapes (`off`, `absp`) get exactly 0.** sp2 gets 1e-13 there, and 1e-6
  for the tournament. The count register is 16 places wide and its count down damps every place's cell output (×0.1,
  `GRAD_DAMP`), so the path is deeper than sp2's 2-place countdown. With the damping off the tournament gets 7.6e-6
  and those two overflow to NaN, so the path is unbroken and the zeros are underflow: rule 8's known gray edge (b),
  slightly deeper.

## 5. Results on H1

**The official run.** `records/ll2_official.json`: the file byte for byte, the untouched official manifest, one H100,
through the shared runner. **It certifies**: T = 64 on both ladders, every rung 768/768, test 9,999/9,999, ood_t and
ood_n_t 10,002/10,002 (`tools/verify_records.py`: the embedded source is the file's md5, the manifest the official
one). In that run:
- the climb (sp2's) found the round at step 610, `LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA` (copy x,
  multiply, reduce), by calls;
- at step 611 the answer program's search found `C.S[T.round1 STA.X]`: 256 rows of 4 counts (T = 1, 2, 4, 8),
  14,212 items, levels of 1, 61, 2,242 and 98,226 states, 95.8 s;
- at step 631 T's direction was resolved, each reading of the T = 16 rows run through that program;
- at step 810 the model froze, by 1,002 s of the 3,600;
- training took 1,191.6 s (to the file's 20,000 steps, at learning rate 0 after the freeze), evaluation 195.7 s.

sp2's official run, for comparison: the round at step 688 by 1,029 s, frozen at 887 by 1,077 s, 1,276.7 s / 225.5 s.
The two runs do not follow each other step for step (§4).

**Seeds** (B200s, the official manifest with only the seed changed; research, not certification):

| record | seed | T, seen / unseen moduli | certified | round (step) | answer program (step, search s) | T resolved | frozen (step, by s) | training / evaluation s |
|---|---|---|---|---|---|---|---|---|
| `ll2_official` (H100) | 74 | 64 / 64 | yes | 610 | 611, 95.8 | 631 | 810, 1,002 | 1,191.6 / 195.7 |
| `ll2_b200_seed74` | 74 | 64 / 64 | yes | 610 | 611, 78.1 | 631 | 810, 724 | 845.4 / 94.0 |
| `ll2_b200_seed1` | 1 | 64 / 64 | yes | 634 | 635, 32.9 | 655 | 834, 927 | 1,095.8 / 199.6 |
| `ll2_b200_seed2` | 2 | none / none | no (out of time in the climb) | none | none | none | none | 3,600.0 / 4.9 |
| `ll2_b200_seed3` | 3 | 64 / 64 | yes | 662 | 663, 67.2 | 738 | 862, 796 | 918.7 / 95.3 |
| `ll2_b200_seed4` | 4 | 64 / 64 | yes | 703 | 704, 51.6 | 724 | 903, 922 | 1,132.4 / 162.7 |
| `ll2_b200_seed5` | 5 | 64 / 64 | yes | 659 | 660, 44.9 | 736 | 859, 818 | 937.6 / 94.4 |
| `ll2_b200_seed6` | 6 | 64 / 64 | yes | 673 | 674, 77.4 | 694 | 873, 830 | 961.1 / 95.1 |
| `ll2_b200_seed7` | 7 | 64 / 64 | yes | 612 | 613, 28.8 | 633 | 812, 811 | 1,016.4 / 157.8 |
| `ll2_b200_seed2_relaxed` (7,200 s budget) | 2 | 64 / 64 | yes, on the doubled budget | 698 | 699, 54.2 | 772 | 898, 3,741 | 3,874.1 / 96.0 |

Every certified run's splits are exact, and its tournament holds the one program `C.S[T.round1 STA.X]` (the second
round, where there was one, is merged into the first, §3). In every run where the round came, the answer program came at
the next step, from one search, and that run certified. That is 16 runs: the 8 above (searches of 29–96 s over
13,336–15,561 items), the first build's 4 below (11–90 s, on the T = 1 and 2 rows alone), the 8-digit T run of §6 (27 s,
11,581 items), seed 2 on the doubled budget (54 s, 16,983 items) and the diagnostic copy's 2 runs below (29 s and 52 s).
No other program was ever accepted. sp2's 8 B200 seeds froze by 666–1,804 s; these 7 by 724–927 s.

The first build (`ba07d656`, §4) ran the official manifest and seeds 74, 1, 2 and 3. On each seed it follows the final
file step for step up to the round (the same steps for the round, the answer program, T and the freeze), with its
answer program searched on the T = 1 and 2 rows alone. Its official run certified (`records/ll_official.json`: frozen
at step 810 by 1,204 s, 1,412.7 s / 332.6 s), as did seeds 74, 1 and 3. Seed 2 failed the same way as below.

**Seed 2 fails in sp2's climb, before the round: it runs out of time.** Every certified run found its round at level
4, after 563–770 s of climbing. Seed 2's climb needed 10 levels to build the round's last piece:
- by level 4 it had the comparisons, the additions, the multiply-accumulate row and the multiply, and later the chain
  and the long-division step (`T.red.Bx+d.*`);
- the full long division, `red` on the pattern `x+on` (solved at cost 4 in a level-4 probe on seed 74), never came
  from an ordinary probe. The deep probes, the climb's larger searches for the problems closest to solved, went to
  the quotient law (`div` on `x+dn`) at levels 5 to 9, and reached `x+on` only at level 10;
- on the final file's run, level 10 solved it, `T.red.x+on.XN = LDA.N STA.Y P[BRG.X T.red.x+dn.AN]` (long division
  with the modulus in N), at step 697, at about 3,480 s: just as the climb's deadline came (120 s before the budget's
  end). The round's search at the next step stopped at that deadline while building its vocabulary, and so did every
  retry after it. On the first build's run, on another host, the deadline cut level 10's deep probes after their first
  group, before the group that solved `x+on` on the final file's run.

Offline, on the first build's final library (48 routines, rebuilt from its log; `round_offline.py`, B200, 7 s), the
round's search finds nothing up to cost 5 or two routine calls (283,526 states), as expected without the long
division. With `T.red.x+on.XN`, the round is `LDA.X STA.Y T.mul.x,o.XY STA.X T.red.x+on.XN`, the form sp2's seed 3
found.

The climb is sp2's, unchanged. Seed 2's path differs from sp2's seed 2 (which found its round at level 4 by 744 s)
only because this training starts from different random draws (§4).

**With the budget doubled** (7,200 s of training, a research run: `records/b200/ll2_b200_seed2_relaxed.json`), seed 2
follows the same path. This time level 10's deep probes all finish (320, 310 and 318 s), the second one solves the
long division (`red/x+on@XN`, cost 5), and:
- at step 698 the round's search finds `LDA.X STA.Y T.mul.x,o.XY STA.X T.red.x+on.XN` by calls (9.8 s);
- at step 699 the answer program's search finds `C.S[T.round1 STA.X]` (54.2 s, 16,983 items, 256 rows of 4 counts);
- T is resolved at step 772, and the model freezes at step 898, by 3,741 s;
- it certifies: T = 64 on both ladders, every rung 768/768, every split exact (training 3,874.1 s, evaluation 96.0 s).

So on the official budget, seed 2 is short by the climb's last level: its round came at 3,673 s, past the climb's
deadline of 3,480 s, and its freeze 141 s past the end of the 3,600 s budget.

**Is the time-out sp2's or the loop's? The random start's.** sp2's own seed 2 found its round at level 4 by 744 s, so
the slow path is not sp2's on that seed. The two climbs part at level 0. The two trainings reach the climb at
different steps (step 631 for sp2, 583 here) with different pools of read rows, so different bindings tie for the
cleanest practice: sp2 practises `red/pair@AY` at level 1 and the chain `red/x+dn@AY` at level 3, and that gives it
the long division `red/x+on@XA` at level 4. This file got `red/pair@AN` only, the chain one level later in binding
AN, and the long division in that binding (`red/x+on@XN`) waited until level 10.

To test it, `diag/submission_stage3_ll_gate.py` (md5 `2afacc78`, built by `diag/make_gate_replay.py`; not a
submission) is this file with sp2's gate put back as dead weight: the parameter created where sp2 creates it, its
loop law, and the gate in phase B's switch, with nothing reading it. Every tensor then starts from sp2's draws and
every lesson until the round is sp2's. On two B200 seeds (records `records/b200/llgate_b200_seed*`, both verified;
`diag/compare_climb.py` compares them with sp2's B200 records):

| seed | against sp2's own run, up to the round | the round | answer program | T resolved | frozen | certified |
|---|---|---|---|---|---|---|
| 74 | identical: all 18 logged losses, all 48 climb lines | step 688, level 4 (sp2: 688) | step 689, 28.9 s, 15,421 items (sp2's official library's count, §3) | 709 (sp2: 708) | 888, by 735 s | yes |
| 2 | identical: all 18 logged losses, all 52 climb lines | step 683, level 4 (sp2: 683) | step 684, 52.2 s, 13,746 items | 704 (sp2: 703) | 883, by 1,062 s | yes |

With sp2's start, seed 2 takes sp2's fast path, and the learned loop then finds `C.S[T.round1 STA.X]` at the next step
and certifies. So nothing the loop does slows the climb: its own code runs in phase C too (keeping the rows of every
count), and the climb's lines still match sp2's. The learned-loop file loses seed 2 because dropping the gate's 4
parameters moves every later tensor's random draws, and dropping its lesson changes the gradient's clipped norm at
every step from phase B on. The climb that follows is a new random sample of sp2's climb, and on this seed a slow one. (Locally,
`diag/replay_check.py` runs sp2's file and the diagnostic copy side by side on a CPU: the same schedule event for
event, and the same losses until step 158, as close as sp2's file repeats itself on that CPU, step 166.)

## 6. How T is read

The brief asked to question T's reading too. Three things about it are designed, and one of them carries weight.

**What is designed.**
- **T is a field of its own, read into K = 2 digits.** The reader (stage 1's, unchanged) finds the prompt's three digit
  runs, and its hypotheses say which run is N and which is x, each with a reading direction (24 hypotheses). T is the
  run left over, read by its own pointer into `K` = 2 slots, chosen knowing T ≤ 64.
- **T's direction** (is "16" sixteen or sixty-one?) is not designed: it is chosen by speculation from the answers
  (IO.md). Here each reading is run through the answer's program instead of the countdown, from step 631 on in both
  official runs.
- **Where T goes in the machine:** register S, as x goes into X and N into N (§1).

**The width carries weight: it is what tells T from x.** While the reader chooses its reading, the evidence it has is
the mined order fact x < N, checked under every hypothesis (`passes`, the exact verdicts). But T < N holds on every
row too, and more easily: a one- or two-digit number is below N by its length alone, before the digit code is good.
What rules out "x is the third run" is that the run it leaves for T would be longer than T's 2 slots (`tfits`). With T
read as wide as N and x (K = 8), nothing does. `checks/reader_k.py` runs the file's phase A (the reader and the
subtraction table from random, local, CPU) until the reading is chosen, on 5 seeds each:

| T's field | first-ranked reading | reading chosen | x taken from |
|---|---|---|---|
| K = 2 (the file) | right on 5 of 5 (steps 120–152) | steps 139–174 | x's run on 5 of 5 |
| K = 8 (as wide as x) | wrong on 4 of 5 (x from T's run, step 24–26) | steps 125–847 | **T's run on 2 of 5** (seeds 0 and 2) |

**End to end** (the file with `K = 8` and nothing else changed, the official manifest, 3 B200 seeds, records
`records/b200/llk8_b200_seed*`):

| seed | the reading chosen (N run, N dir, x run, x dir) | what followed | certified |
|---|---|---|---|
| 74 | (0, 1, 2, 0): x from T's run | phase B never ends: the subtraction table's checks stay between 32% and 66% from step 1,000 to the last, 20,000 (they must reach 100%; its lessons are built from x's digits, which are now T's one or two) | no |
| 1 | (0, 0, 1, 1): right, after ranking x-from-T first at step 25 | the climb, the round at step 637, `C.S[T.round1 STA.X]` at step 638 with T's 8 digits in S, T resolved at 706, frozen at 837 | **yes**, T = 64 on both ladders |
| 2 | (0, 0, 2, 0): x from T's run | as seed 74 (between 18% and 46%) | no |

So an 8-digit T in the count register works when the reading is right (seed 1). But the reader gets the reading right
only some of the time: 3 of 5 local seeds and 1 of 3 evaluator seeds. Reading T without being told it is short would
take another source of evidence, such as letting the answers choose between the two readings, as general lessons v3
lets them choose whole digit codes. That would be a change to the reader, not tried here.

## 7. What is still designed

The countdown's three hand-written parts are gone (§1). Against SELF_PRACTICE.md §6's list for sp2, the changes are:

| designed in sp2 | here |
|---|---|
| the countdown over T, its gate's shape and the loop law | **gone**: found as `C.S[T.round1 STA.X]` by the answer program's search |
| the round's search looks for "the answer is the round run t times" | **gone**: the round is the smallest T's answers as they are |
| the instruction set (32 instructions, loops by P, B, W) | the same, **plus the count loop**: a generic instruction (any register, any body), but a designed one, and its stopping rule ("count down by one; stop at the borrow") is the old countdown's |
| where the prompt's numbers go | x in X, N in N, as before, **and T in S** |
| where the round is looked for | the same (sp2's round search); **plus the answer program's search space**: the round and the library as routines, the 18 whole-register instructions, count loops over bodies of at most 2 items, cost at most 3, 10^6 states a level; its rows are those whose count reads the same both ways, and it waits for two counts |
| the sizes | the same, **plus** `LOOP_CAP` = 100 rounds (the old countdown's 10^K cap) and the pool sizes (64 rows a count, 512 in all) |
| T's field: 2 digits, the third run | the same; §6 shows that the 2 digits are what tell T from x |
| the reader, the cells, the law book, the choosing rule, the climb, BPE, routines, Occam, the tournament, the schedule | the same |

So the claim is narrow: no countdown is written, the answer program that repeats the round is found by search from the
answers, and the search also finds which number is the count, how many rounds (exactly T, not T ± 1) and how each
round's answer feeds the next. A strict reader can still say the count loop is the countdown made into an instruction.
It is generic (it computes many things the task never needs, on any register), but it was added knowing the task
family, like SEL.S and BRG.X before it.

## 8. Cost, and how hard it was tried

**The search's size.** The answer program's search is small next to the climb's: 13,336–15,561 items and about
100,000 states at its last level, 29–96 s once per run, against 563–770 s for the climb to its round. The count loops
are most of the items (5 registers × 2,655–3,099 bodies), and most of them cost almost nothing, since a loop on a
register holding x or N ends in the cap's error after its count downs, before any body runs.

**What was run.**
- **The count loop**: one design, checked on the four machines with 46 random programs and the H1 program (local).
- **The answer program's search offline**: 6 arms, five on B200s and one on the laptop (§3); `records/offline/`.
- **The evaluator**: the first build on the official manifest and 4 B200 seeds; the final file on the official
  manifest and 8 B200 seeds; T as wide as x on 3 B200 seeds; seed 2 on a doubled budget; the diagnostic copy with
  sp2's gate on seeds 74 and 2. 20 records.
- **T's reading**: 10 local reader runs (5 seeds at each width) and the 3 evaluator runs above.
- **Rule 8**: the gradient test at T = 1 and T = 2 on this file and on sp2's, and once with the damping off (local).

**Money.** About $58 of Modal at list price, against the $400 soft budget. The evaluator runs come to $54.9: the
first build's official run $2.0 and 4 seeds $12.0; the final file's official run $1.6 and 8 seeds $20.2; T as wide as
x $7.8; seed 2 on the doubled budget $7.0; the diagnostic copy on 2 seeds $4.3. The offline searches, 6 short B200
jobs, came to about $3. That is about 8.5 B200-hours and 1 H100-hour. Everything else ran locally on a laptop
CPU, each check in seconds to a few minutes.

**Dead ends, and how hard each was tried.**
- One search for the loop and its body together (the brief's first way): not run. Counting its candidates (§3) puts
  it far past the searches this file can afford: about 6·10^13 loop bodies flat, or 5·10^8 loop items by calls,
  against 10^6–10^7 states a level.
- The first build's count pool dropped the T = 4 and 8 rows (a trimming bug). It still found the loop from T = 1 and
  2 and certified; fixed in the final file.
- T read as wide as x: tried locally on 5 seeds and in the evaluator on 3. It fails whenever the reader takes x from
  T's run (§6).
- Seed 2: a failure of sp2's climb on this file's trajectory, not of the loop. Diagnosed from its log and offline,
  certified on a doubled budget, and certified on the official budget once the training starts from sp2's own draws
  (the diagnostic copy, §5).

## 9. Files and how to rerun

Everything is in `cpu_first/integration/learned_loop/`; nothing outside it was changed.

| file | what |
|---|---|
| `submission_stage3_ll.py` | the file (md5 `419c15e7`), built by `make_ll.py` |
| `submission_stage3_ll_v1.py` | the first build (md5 `ba07d656`), saved from its official record |
| `stage3_ll_template.py`, `libcore.py`, `spcore.py`, `make_ll.py` | its sources and builder (§4) |
| `outer_offline.py` | the answer program's search offline on sp2's recorded library (§3) |
| `round_offline.py` | the round's search offline on a library a run left (§5, seed 2) |
| `modal_ll.py` | the offline modules on a Modal GPU (app and Volume `learned-loop`) |
| `rows.py`, `summarize.py` | H1 rows as digits; one line per evaluator record |
| `checks/` | `test_count.py` (the four machines), `designed_ll.py` (the prediction path at the truth), `smoke_replay.py` (training after the round), `gradtest_ll.py` (rule 8), `reader_k.py` (T's width, §6); `sp2_official_lib.json` and `ll_seed2_lib.json`, the recorded libraries |
| `diag/` | seed 2's question (§5): `make_gate_replay.py` builds `submission_stage3_ll_gate.py` (this file with sp2's gate as dead weight); `replay_check.py` (sp2, it and this file side by side on a CPU); `compare_climb.py` (two records' training up to the round) |
| `records/` | the evaluator's records: `ll2_official.json` (the file), `ll_official.json` (the first build); `b200/` the seeds (`ll2_b200_seed*`, `ll_b200_seed*`), T as wide as x (`llk8_b200_seed*`) and the diagnostic copy (`llgate_b200_seed*`); `offline/` the offline searches |

From `cpu_first/`, with the sotaku venv's Python:

```bash
~/github/sotaku/venv/bin/python integration/learned_loop/make_ll.py
```

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/learned_loop/checks/test_count.py
```

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/learned_loop/checks/designed_ll.py N=4
```

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/learned_loop/checks/smoke_replay.py STEPS=60 B=128
```

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/learned_loop/checks/gradtest_ll.py N=2 SPLIT=depth_t_2
```

```bash
~/github/sotaku/venv/bin/python integration/learned_loop/checks/reader_k.py STEPS=1500 B=256 K=8 SEEDS=0,1,2,3,4
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach integration/learned_loop/modal_ll.py --envs '[{"TAG": "outer_all", "VOCAB": "all"}]'
```

The evaluator's runs, through the shared runner (records to `records/` and `records/b200/`):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode official --file integration/learned_loop/submission_stage3_ll.py --name ll2_official --out integration/learned_loop/records --run ll2-official
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/learned_loop/submission_stage3_ll.py --seeds 8 --hardware b200 --cap 5 --name ll2 --out integration/learned_loop/records/b200 --run ll2-b200-s8
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/learned_loop/submission_stage3_ll.py --seeds 74,1,2 --hardware b200 --set '{"K": "8"}' --timeout 2400 --name llk8 --out integration/learned_loop/records/b200 --run llk8-b200
```

```bash
python3 tools/verify_records.py integration/learned_loop/records/ll2_official.json=integration/learned_loop/submission_stage3_ll.py
```
