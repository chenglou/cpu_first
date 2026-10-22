# One Layer Deeper H1, from the CPU up: where everything stands

*The entry point for `cpu_first/`. First written 2026-10-21 at 03:45 Pacific and refreshed three times: at about 10:45
and 15:15 that day, as the ablation, digit-anchor, learned-wiring and wide-menu work was committed, and on 2026-10-22 at
about 12:45, after the bridge, litmus, library, tournament, general-lessons, inventing and handwired-bridge studies
were committed.*
- *What was checked against the run records (`runs/`, `ablation/runs/`, `anchors/records/`, `bridge/h1/records/`,
  `tournament/h1_records/`, `general_lessons/records/`, `handwired_bridge/h1/records/`, `litmus/records/`,
  `library/records/`): every certification, run time, freeze step, parameter count and CPU-seed count quoted here. The
  source embedded in each record was hashed and compared with the file's md5.*
- *The data counts in §1–2 were computed from the released H1 files, and the commits come from `git log`.*
- *The one-line findings of the timeline (§3) and the summaries of how each piece learns summarize their write-ups,
  which are linked; they were not re-run.*
- *The older write-ups are left as they were. §9 lists where a write-up and a record disagree.*
- *All the work described is committed (the last commit is `22df2de`, Oct 22 at 09:32). One new session started as
  this was refreshed: the bar-2 integration (§7), which had written nothing but a first working file yet.*

**In one minute.**
- The task is exact repeated modular squaring. Almost every entry on the leaderboard scored near 0%, because the
  remainder erases partial credit: an almost-right model gets random-looking answers, so gradients have nothing to
  follow (§1–2).
- Designs built like a CPU certify: discrete digits, one small cell reused everywhere, and labels that choose between
  whole machines. Four lines of work certified through the official evaluator: SpecBit, UnfoldB, UnfoldS and the
  learned digit machine (§3).
- The project is measured against three bars (§4):
  1. **Certify H1 with learned parts: done.** The certified models are paper-arithmetic calculators in which every part
     that computes is learned, the wiring is chosen in training from menus, and no hand-written arithmetic runs
     anywhere.
  2. **Nothing a strict judge would call the hand-written algorithm: nearly there.** Every hand-written piece has now
     been replaced somewhere, in separate files:
     - generic MLP digit cells;
     - the division's wiring chosen inside training at free attention's width;
     - lessons general enough that one file also learns seven other formulas;
     - and, outside a submission, a search-and-compress loop that rediscovers the skeleton itself.

     What remains is putting these in one file, plus the instruction set and alphabet that the loop starts from.
  3. **A general algorithm learner: open.** Across task families the method still needs a designer for some tasks.
     The inventing test points at two missing ingredients: choosing its own practice problems, and the wrap-around
     concept.
- It got there by removing hand-written parts one at a time, with certification as the gate at each step (§5).

## 1. The task, in plain words

A prompt gives three numbers: a modulus **N**, a start **x** and a count **T**. The model must square x, keep the
remainder after dividing by N, and repeat that T times, then write the result's digits. In short: x^(2^T) mod N.

**A real training row.** The prompt reads `N 9 8 1 4 X 1 3 8 5 T 1`. N's digits are written lowest first, so:
- N = 4189, x = 1385, T = 1;
- 1385² = 1,918,225 = 457 × 4189 + 3852;
- so the answer is `3852`.

A T = 2 row squares twice. With x = 2605 and the same N, 2605² leaves 4034, and 4034² leaves 3080.

**What training shows** (counted from the released data):
- 101,999 rows from only 430 different moduli of 13, 16 and 20 bits;
- every modulus is a Blum integer: N = p × q, with p and q each 3 more than a multiple of 4 (4189 = 59 × 71);
- T = 4, 8 and 16 have 33,333 rows each, and T = 1 and T = 2 only 1,000 each;
- no training number has more than 7 digits.

**What certification asks.**
- There are seven rungs, T = 1, 2, 4, 8, 16, 32, 64, of 768 fresh prompts each.
- A rung counts only if all 768 answers are exactly right, and only as part of an unbroken run from T = 1.
- There are two ladders. One uses moduli from the training pool. The other uses 505 moduli never seen in training, at
  14, 18, 22 and 24 bits; 110 prompts of each of its rungs have an 8-digit N, a size training never shows.
- The evaluator also scores three splits: test (9,999 rows, held-out moduli, T = 4, 8 and 16), and ood_t and
  ood_n_t (10,002 rows each, T = 32).
- Budgets: 3,600 s of training on one H100, and 1,800 s of evaluation.

The rules are [one-layer-deeper/README.md §Rules](../one-layer-deeper/README.md#rules). The Hard tier ranks entries by
the certified T on the two ladders.

## 2. Why it is hard

**The remainder erases partial credit.** Suppose a model squares 1385 but gets one high digit wrong: 2,918,225 instead
of 1,918,225.
- Without the remainder, the answer would be off in one digit.
- With it, the answer is 2681 instead of 3852: no digit in common, and nothing that says "almost".

So every wrong digit fact before the remainder turns into a random-looking answer. Gradient descent from answers gets
no sense of direction until the arithmetic is already right, and one more squaring scrambles any error left.

**There are few easy rows, and the test goes further than training.**
- Only 5 of the 1,000 T = 1 rows have x² < N, the only rows where no remainder is taken at all.
- T = 32 and 64, the tens digits 2–6 of T, and 8-digit moduli never appear in training.
- Certification is all or nothing: one wrong prompt out of 768 loses that rung and every rung above it.

**What happened on the leaderboard.** [REPORT.md](REPORT.md) §0 counted 3 of 146 entries certified to T = 64. The
other 143 sat at T = 1 with at most 12.8% (most below 1%). That count comes from the public leaderboard at the time;
it is not in a run record and was not re-checked here.

**The diagnosis** ([REPORT.md](REPORT.md) §2–3): labels can *check* the arithmetic but not *teach* it. What works is a
CPU's own structure: digits snapped to exact values, one small cell reused at every place, and lessons that either
check the machine against itself or choose among complete machines that the labels then verify.

## 3. Timeline: what failed, what certified, and why

Commit times are Pacific. Everything before the first commit is in the snapshot `36cac5a` (Oct 19, 23:50).

### Oct 19: the diagnosis, then the first certifications

| approach | result | write-up |
|---|---|---|
| generic networks (MLP, transformer, looped transformer) on small versions of the task | addition is learned as a rule; multiplication, squaring and the remainder only as lookup tables. On moduli the network never trained on, the remainder is at chance | [microlab/LADDER.md](microlab/LADDER.md), [LADDER_MATCHED.md](microlab/LADDER_MATCHED.md), [FINDINGS_pre_remainder.md](microlab/FINDINGS_pre_remainder.md) |
| learned digit tables composed into arithmetic, from answers | one table works; deep compositions (schoolbook multiplication, about 72 lookups in a row) stall even with no remainder | [FINDINGS_substrate.md](FINDINGS_substrate.md) (first part) |
| looping to large T | not the obstacle, once the state is snapped to digits and the step is exact | [microlab/FINDINGS_looping.md](microlab/FINDINGS_looping.md) |
| relaxing one rule at a time | traces that show the work certify at 8–24 bits, but only when every trace token is a local operation, and traces are extra supervision. Generic looped networks stop at 6–8 bits | [RELAXED.md](RELAXED.md) |
| designed machines with self-calibrated cells (`cadet.py`, `menu.py`) | labels choose among complete machines; certify T = 64 in a local evaluator | [REPORT.md](REPORT.md) §4 |
| **SpecBit**: a designed binary netlist with one learned 3-in/2-out gate, run as a mixture over all 65,536 possible gates at once | **certified** through the official evaluator. The labels pick the full adder | [REPORT.md](REPORT.md) §9 |
| search over whole machines in bit lanes on a GPU | a grammar of 2^63.9 machines was searched in 20 minutes on one H100 (576 of its 5,184 survivors certify). Later, laws that need no labels left only machines that certify | [REPORT.md](REPORT.md) §9–14 |
| exact arithmetic blocks with learned wiring | with exact multiply and remainder blocks the step is a 2-instruction program, and one label picks it by enumeration. Soft (gradient) selection of the same program fails | [microlab/FINDINGS_exact_blocks.md](microlab/FINDINGS_exact_blocks.md), [FINDINGS_loop_csub.md](microlab/FINDINGS_loop_csub.md) |
| **UnfoldB**: 4 learned numbers (a digit base B and V = w · (1, s, s²)), fitted with a loss that unfolds the remainder by itself | **certified**, but it needs a batch of 100,000 rows | [UNFOLD.md](UNFOLD.md), [unfold_sub/NOTE.md](unfold_sub/NOTE.md) |
| an unknown recurrence | lattice reduction recovers a hidden polynomial step from a few rows | [UNFOLD_FAMILY.md](UNFOLD_FAMILY.md) |

### Night of Oct 19–20: a small-batch fix, and the quotient-digit puzzle

| approach | result | write-up |
|---|---|---|
| **UnfoldS**: UnfoldB at the manifest's batch of 512, carrying the fit's information in the optimizer state | **certified** | [unfold_small/NOTE.md](unfold_small/NOTE.md) |
| long division with a learned subtraction and a learned choice of quotient digit, from answers | each half learns when the other is given; together they stall (about half right at 6 bits, near 0 at 7). Hard EM, a "clock" loss and curricula do not fix it | [microlab/FINDINGS_quotient.md](microlab/FINDINGS_quotient.md) and its `_em`, `_clock`, `_shared` siblings |
| the same puzzle, shrunk until it can be computed exactly | explained: identical trial slots sit on a symmetric saddle and split the wrong way. Fix: give every slot a fixed meaning (one shared subtractor fed exact multiples of N), teach it with a chooser that starts undecided, then let the subtraction's own borrow choose | [tinycore/TINYCORE.md](tinycore/TINYCORE.md) |

### Oct 20: learned parts in the real machine, one at a time

| commit | what became learned | write-up |
|---|---|---|
| 05:38 `824d073` | the subtraction table, inside a designed long division | [LEARNED_REMAINDER.md](learned_remainder/LEARNED_REMAINDER.md) |
| 14:51 `e1dd000` | the multiples of N, now built by repeated subtraction | [CHAIN.md](learned_remainder/CHAIN.md) |
| 22:16 `c0c86a2`, `c6c9597` | the times table (squaring); the countdown of T and the division's stop rule | [SQUARE.md](learned_remainder/SQUARE.md), [CONTROL.md](learned_remainder/CONTROL.md) |
| 22:55 `550f60e`, `f342a91` | the adder in the self-check; the prompt reader and answer writer | [ADDER.md](learned_remainder/ADDER.md), [IO.md](learned_remainder/IO.md) |

### Oct 21: one model, then simpler lessons, learned wiring, and both together

| commit | what | write-up |
|---|---|---|
| 00:55 `9959997` | **every learned piece in one model**, with a prediction path inside autograd | [COMBINED.md](learned_remainder/COMBINED.md) |
| 00:55 `9332360` | wiring and tables learned together in tiny machines | [tinyrouting/TINYROUTING.md](tinyrouting/TINYROUTING.md) |
| 05:19 `f498805` | an ablation of the training recipe; a simplified recipe certified | [ablation/ABLATION.md](ablation/ABLATION.md) |
| 05:35 `5d2b917` | **digit meanings from the order of the numbers**: the subtraction table learned with no hand-written adder at any step | [anchors/ANCHORS.md](anchors/ANCHORS.md) |
| 10:25 `36fc156` | **the machine's wiring learned**, in three certified stages | [learned_wiring/WIRING.md](learned_wiring/WIRING.md) |
| 13:02 `b76f2cd` | the learned wiring with no hand-written arithmetic anywhere | [learned_wiring/ANCHORED.md](learned_wiring/ANCHORED.md) |
| 13:10 `c166f8d` | **wide menus**: the wiring chosen from about 4·10^17 whole machines | [wider_menus/MENUS.md](wider_menus/MENUS.md) |
| 14:29 `d8856a8` | **the wide menus with no hand-written arithmetic anywhere** | [wider_menus/WIDE_ANCHORED.md](wider_menus/WIDE_ANCHORED.md) |

### Oct 21 evening to Oct 22: generic parts, other tasks, and discovering the skeleton

| commit | what | write-up |
|---|---|---|
| Oct 21, 20:42 `16b66dd` | **the machine is one setting of a generic looped transformer**; loosened notch by notch. Generic MLP digit cells certify at H1; soft wiring learns only from short menus | [bridge/BRIDGE.md](bridge/BRIDGE.md) |
| Oct 22, 02:04 `ec35027` | **the litmus test**: the best model run unchanged on 26 other tasks | [litmus/LITMUS.md](litmus/LITMUS.md) |
| Oct 22, 02:04 `25d6404` | **library learning**: merging the operations of solved programs rediscovers the hand-written skeleton | [library/LIBRARY.md](library/LIBRARY.md) |
| Oct 22, 06:15 `90f245f` | **the tournament**: the division's wiring chosen inside training, at free attention's width | [tournament/TOURNAMENT.md](tournament/TOURNAMENT.md) |
| Oct 22, 06:15 `1613c28` | **general lessons**: one unchanged file learns H1 and seven other formulas | [general_lessons/LESSONS.md](general_lessons/LESSONS.md) |
| Oct 22, 06:15 `63f1613` | **inventing**: which of our inputs a search-and-compress learner cannot do without | [library/INVENTING.md](library/INVENTING.md) |
| Oct 22, 09:32 `22df2de` | the machine compiled by handwired into affine + ReGLU layers, loosened the same way | [handwired_bridge/HANDWIRED_BRIDGE.md](handwired_bridge/HANDWIRED_BRIDGE.md) |

### The lessons that came out of the failures

- **Generic networks memorize instead of computing the remainder**, and on a new modulus they are at chance.
- **Soft mixtures of choices learn nothing here.** After the remainder, the loss stays flat until the wrong options
  hold almost no weight.
- **Answers alone, pushed through a learned machine, do not teach it.** Untrained parts get bypassed, or trial slots
  co-adapt into a lookup arrangement. Even with every training row answered, answers did not start the subtraction
  table from random digits within that study's budget ([ANCHORS.md](anchors/ANCHORS.md) §4.3).
- **A check made only of the learned table can be passed by whole families of wrong tables**: digits relabeled,
  borrows swapped, "copy the first number". Meaning has to come from outside the table. First that was a hand-written
  adder for a short while ([ABLATION.md](ablation/ABLATION.md) §7). Now the order of the numbers does it alone:
  x < N teaches every borrow, and two laws of subtraction, applied one digit at a time, leave only the true table
  ([ANCHORS.md](anchors/ANCHORS.md)).
- **Wiring is learned by trying whole options and scoring each pass or fail**, not by blending options' digits on the
  wire. Blended switches settle on a machine that does nothing ([WIRING.md](learned_wiring/WIRING.md),
  [TINYROUTING.md](tinyrouting/TINYROUTING.md)).
- **The traps of loosened wiring** ([BRIDGE.md](bridge/BRIDGE.md), [HANDWIRED_BRIDGE.md](handwired_bridge/HANDWIRED_BRIDGE.md)):
  - **the bypass**: free soft wiring reads the prompt's clean digits instead of the machine's computed ones. The pull
    comes from the data, not from attention's softmax: a handwired network with no attention felt the same pull;
  - **the near miss**: a 7-option menu taught by answers lands on one nearby wrong wiring on every seed, though the
    answers alone single out the right one;
  - **the constant source**: wiring that cannot drift to the prompt settles on reading a constant zero;
  - **ties**: an MLP digit cell settles on two entries giving the same digit, until "every digit once per row and
    column" breaks the tie.
- **With wide menus, the checks still identify; the search is what can fail.** In every menu counted exactly, every
  wiring that passes all the checks is a correct one. Learning failed only where the search could not see the right
  combination: shown only a few sampled combinations instead of every one, a switch was caught by impostors in 14 of
  16 runs ([MENUS.md](wider_menus/MENUS.md)). The tournament found the identified wiring among 5.3·10^10 inside
  training; where the checks let impostors through, it crowned one ([TOURNAMENT.md](tournament/TOURNAMENT.md)).
- **What worked every time:** digits snapped to exact values, one shared cell, checks on each training row's own
  numbers that need no answer, and pass/fail scoring of whole choices.

## 4. Where it stands: three bars, and the models at the frontier

**Bar 1: certify H1 with learned parts. Done.** The first learned table certified on Oct 20 (§5, step 1). From step 11
on, every certified model learns every part that computes, chooses its wiring in training from menus, and runs no
hand-written arithmetic anywhere, in training or in the answer.

**Bar 2: nothing a strict judge would call the hand-written algorithm. Nearly there.** Every piece that used to be
hand-written is now learned or discovered somewhere, but not all in one file:

| the piece | how it is learned or discovered now | in a certified H1 file? |
|---|---|---|
| the digit tables | learned tables (step 1 on), or generic ReLU MLPs (step 12) | yes |
| what the digits mean | from the order of the numbers, with no hand-written adder (steps 6b, 8, 9, 11 on) | yes |
| the division chain's wiring | chosen inside training among 7.9·10^9 wirings in which every connection reads any register at any nearby place (step 13) | yes |
| the rest of the wiring | chosen from wide menus that hold several correct algorithms (step 10 on) | yes; the menus are hand-written |
| the lessons | laws stated once, facts the model finds in the data, and lesson sets the answers choose (step 14) | yes, in a different file from step 13 |
| the skeleton: one register, steps of links, loops, rows of products | rediscovered by search and compression from 32 digit instructions and 26 practice tasks ([LIBRARY.md](library/LIBRARY.md)) | no: done offline. A version inside one training run found the division's step and a multiply row, but not the whole multiply or x² mod N |
| the instruction set and the alphabet | hand-written | no |

So two kinds of work remain:
- **integration**: one file with the tournament, the generic cells and the general lessons, whose skeleton comes from
  the library's loop run inside training;
- **the starting point**: the instruction set and the alphabet that the loop builds from.

**Bar 3: a general algorithm learner. Open.**
- **The litmus test** ([LITMUS.md](litmus/LITMUS.md)) ran the best H1 model unchanged on 26 other tasks.
  - It learned H1 with a reordered prompt, and the practice datasets whose moduli vary (5 of those 6 certified).
  - It failed on other formulas, on the 14 datasets with one fixed modulus, and on a four-number prompt.

  Its verdict: "a method within its family; an engineered solution across families".
- **General lessons** ([LESSONS.md](general_lessons/LESSONS.md)) fixed the formula failures with one unchanged file:
  y² + 1, 10·y², cubing, Montgomery's squaring, and two formulas nobody designed for. Still failing: y² − 1, whose
  step is not in its menu; the 14 one-modulus datasets, where one modulus cannot say what the digits mean; and M4,
  whose rows all have T = 8.
- **The inventing test** ([INVENTING.md](library/INVENTING.md)) took away our inputs from the search-and-compress loop
  one at a time:
  - without our instruction set, on bare counters, it still invented x² mod N, but only when given practice tasks
    whose short solutions are the pieces;
  - without the practice tasks it never did;
  - given the single step "add one, wrap around at N", the search found x² mod N in about a second.

  Choosing its own practice problems, and the wrap-around concept, are what it lacks.

### 4.1 The models at the frontier

Each certified on the untouched official manifest on an H100, the file run byte for byte: T = 64 on both ladders,
every rung 768/768, test 9,999/9,999, ood_t and ood_n_t 10,002/10,002. Each also certified on 8 of 8 CPU seeds
through the same evaluator.

| file (md5) | builds on | what it adds | learned numbers | official record: training s (frozen at step) / evaluation s | what it chose |
|---|---|---|---|---|---|
| [`wider_menus/submission_wide_anchored.py`](wider_menus/submission_wide_anchored.py) (`c82b11fa`) | the combined model (§4.2) | the wiring chosen from about 4·10^17 whole machines; no hand-written arithmetic (§4.3, §4.4) | 11,187 | `wide_anchored_official`: 558.0 (588) / 41.4 | repeated subtraction |
| [`bridge/h1/submission_wide_mlp.py`](bridge/h1/submission_wide_mlp.py) (`22bd34d0`) | the wide model | both digit tables as generic ReLU MLPs (§4.5) | 26,275 | `wide_mlp_official`: 1,047.7 (743) / 98.0 | trial multiples |
| [`tournament/submission_tournament.py`](tournament/submission_tournament.py) (`7ae547d9`) | the MLP model | the division chain's wiring at free attention's width, 7.9·10^9 chains and about 3·10^23 whole machines, chosen by tournaments inside training (§4.3) | 123,993 | `tourney2_official`: 2,114.0 (705) / 89.8 | repeated subtraction |
| [`general_lessons/submission_general.py`](general_lessons/submission_general.py) (`ddfdc065`) | the wide model | task-general lessons; the same file learns seven other formulas (§4.6) | 11,589 | `gen_h1_official`: 1,855.6 (645) / 97.2 | long division, chosen by the answers over a Montgomery machine it also learned |

For H1 alone the tournament file is the most learned, and the general file is the most general. No file combines
them yet (§7).

In every one of these official runs the learned tables came out exact, and the audits found no table entry used at
evaluation that training never used. What every one of them still writes by hand:
- **the skeleton** the switches plug into: one register of 9 places, 8 steps of 9 links each, in the order bring a
  digit in → run the chain → move, then one final correction and one readout; the multiply as one product per digit
  pair, placed and summed;
- **the menus**: every option was written by hand, and the right wirings are among them. Some methods are not in them
  at all, such as non-restoring division and Barrett reduction. The tournament's chain is chosen from relative
  positions instead, but its window of offsets and its grouping into blocks are designed;
- **the sizes**: 8, 8 and 2 digits read and 8 written, and a 2-digit countdown;
- **the loop over T**: count down by subtracting the number 1, and run rounds until no prompt wants another, at most
  100;
- **a + b = a − (0 − b)**, both done with the subtraction table or cell;
- **the reader's menus and grammar**: its menu of 24 readings (which digit run is N and which is x, and which way
  each runs), its 3 shapes, and small rules such as "a slot reads 0 once a slot before it has met a non-digit";
- **the lessons and their schedule.** Among them are the anchors' three statements ("x < N on every top part",
  "subtraction keeps, reverses or wraps the order", "every row and column of the table uses each digit once"), and two
  step laws that are definitions of long division's step ("one step from x + q·N gives back x", and how a step takes
  its digit in). The general file states its laws once and lets the answers choose between two lesson sets; the
  tournament file adds three lessons of its own (§6, rule 14).

### 4.2 The calculator underneath: the combined model

**[`learned_remainder/submission_combined.py`](learned_remainder/submission_combined.py)**: md5
`4f5cecc780894499a462499c3cd7d67f`, committed in `9959997`. Write-up: [COMBINED.md](learned_remainder/COMBINED.md).
It was the first model in which every computing part is learned. Its wiring and its warm start were still written by
hand. Every model at the frontier is this calculator with more of it learned.

**What it is.** A small calculator, wired the way a person does this arithmetic on paper. All 5,079 learned numbers
start random and are learned together, in one training run:

| learned part | numbers | what it does |
|---|---|---|
| subtraction table: (digit, digit, borrow in) → (digit, borrow out), 200 rows | 2,400 | every subtraction, and every addition too, as a + b = a − (0 − b) |
| times table: (digit, digit) → (low digit, high digit), 100 cells | 2,000 | schoolbook squaring |
| commit gate: borrow → keep or discard | 4 | every keep-or-discard decision, in the division and in the countdown |
| reader and writer | 675 | which tokens are digits and which digit each one is; where each number starts and which way to read it; where and how to write the answer |

**The example row, run through it.**
1. **Read.** The reader learned to start at the N marker and step right, since N's digits come lowest first, giving
   N = 4189. It reads x by starting at the T marker and stepping left (1385), and T from the end of the prompt (1).
2. **Square.** It writes one row of digit products per digit of 1385, using the times table, and adds the rows with
   the subtraction table: 1,918,225.
3. **Divide.** It brings down one digit at a time and subtracts 4189 again and again, with the subtraction table.
   The gate discards the subtraction that goes below zero, which it sees as a final borrow. In the early steps 4189
   does not fit at all. Then it fits into 19182 four times (2426 left over), into 24262 five times (3317), and into
   33175 seven times (3852).
4. **Count down.** It subtracts 1 from T's digits with the same table. 1 − 1 = 0 leaves no borrow, so the round runs.
   In the next round 0 − 1 borrows, so the gate holds. Once no prompt in the batch wants another round, the loop ends.
5. **Write.** The writer turns 3852 back into tokens.

**How it learns.** Mostly not from answers. Most lessons are checks on each training row's own numbers that need no
answer:
- subtract N, add it back, and the start must come back;
- x < N, so x − N must borrow and N − x must not;
- x × (b + 1) = x × b + x;
- round trips through the reader and writer.

The answers of the T ≤ 2 rows join late, and only along the machine's own path.

**Its official runs** (Modal H100, the official manifest untouched, seed 74; records in `runs/learned_remainder/`):

| record | what ran | max T, seen / unseen moduli | rungs | test, ood_t, ood_n_t | training s (frozen at step) | evaluation s |
|---|---|---|---|---|---|---|
| `combined_official` | the file byte for byte | **64 / 64** | 768/768 at every rung of both ladders | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 666.6 (545) | 48.2 |
| `combined_index_official` | the file with `PATH = "index"` (the older integer prediction path), the only line changed | 64 / 64 | all 768/768 | all exact | 566.5 (545) | 23.9 |
| `combined_s0_official` | `MODE = "designed"`: every table set to the truth, a plumbing check only | 64 / 64 | all 768/768 | all exact | 143.0 (frozen from the start) | 34.3 |

- **CPU seeds:** the file byte for byte, through the same evaluator with `device = cpu`, certified on 8 of 8 seeds (74
  and 1–7), each on both ladders, every rung 768/768.
- **Inside autograd (rule 8).** At evaluation every digit is a one-hot vector, and every table acts as a
  straight-through one-hot matrix. So the logits are exactly the calculator's, while a gradient from the final logits
  reaches all 10 learned tensors (tested in [COMBINED.md](learned_remainder/COMBINED.md) §3).

### 4.3 The wiring, learned: short menus, wide menus, then a tournament

**Short menus** ([WIRING.md](learned_wiring/WIRING.md), steps 7a–7c in §5). Every connection the combined model wired
by hand became a switch: a small menu of options, one of which is right. The division got 7 switches (864 wirings),
the squaring's layout 2 placement rules of 24 each (576 layouts), and the reader and writer one shape switch of 3.
Each switch is taught by a check that only the right wiring passes, run on the rows' own numbers: one division step
started from x + q·N must give back x; dividing x by N must leave x, and N by N must leave 0; the times-table ladder;
for the reader's shape, the reading under which x < N holds on every question. For the division, every combination of
options is tried on the snapped machine, and the combinations that pass gain weight.

**Wide menus** ([MENUS.md](wider_menus/MENUS.md), step 10). Each connection got many options: 4.1 billion division
wirings instead of 3,456, and 33 million multiply layouts instead of 576, about 4·10^17 whole machines with the
reader's 3 shapes. The menus now hold genuinely different algorithms, and the machine picks among those that work:
- **two long divisions**: subtract N again and again until the result would go negative, or try the step's number
  minus s·N for every s and keep the last that stays positive (trial multiples, as on paper);
- **six groupings of the products**: rows by either digit, or diagonals, each in either order;
- **a Montgomery reduction**, which runs exactly but computes a different number (the dividend times 10^−W, mod N),
  so the checks reject it.

A new law, x + Q·N mod N = x with Q another row's x, removes impostor wirings that never read the dividend's top half.

**The tournament** ([TOURNAMENT.md](tournament/TOURNAMENT.md), step 13). The chain's connections stop being a menu of
meaningful parts: each operand may be any of 7 registers moved up or down by up to 3 places, the borrow may come from
1, 2 or 3 places below, the gate may read the borrow out of any place, and a link may write back any of 7 values at
any offset. That is 7.9·10^9 chain wirings. Inside ordinary training the model holds one weight per whole combination
of a block of connections. Each step it runs every combination still in the running, scores each pass or fail, and
takes one gradient step on −log (the total weight of those that passed). Combinations far behind the leader stop
being run.
- **In tiny machines** it found the one right wiring among 5.3·10^10 in 8 of 8 runs, and among up to 3·10^12 with wider
  windows, where soft attention never found it. Where the checks did not single out the wiring, it crowned an
  impostor: with no answers, all 8 runs ended on the bypass.
- **At H1** the wider menu exposed two gaps that the training data cannot see, and two lessons closed them:
  - the register's top place is never used by 7-digit training moduli, so the chain's checks also run on each row's
    numbers moved up until N fills 8 places;
  - the gate's meaning could flip, so a loop law says every count's first round must run and a count of 0 must stop.

  Without the first lesson 4 of 8 CPU seeds certified. A first version without the second certified on 3 of 8.

### 4.4 Digit meanings from the order of the numbers

Write-up: [ANCHORS.md](anchors/ANCHORS.md). The subtraction table used to learn what its digits mean from a
hand-written adder, used for its first steps; the ablation showed that this warm start was load-bearing. The anchors
replace it:
1. **The borrows, from x < N on every "top part".** Take x's digits from some place upward, and N's digits from the
   same place. The first number is at most the second, and strictly smaller exactly when the two digit strings differ.
   So taking N's part from x's part must borrow exactly then, and taking x's part from N's never borrows.
2. **The digits, from two laws of subtraction**, applied to one digit at a time once the borrows are right: taking the
   same digit away from two digits keeps their order; taking more away from the same digit leaves less; except when
   exactly one of the two goes below zero, which wraps round and comes out on top. In each column the laws put the ten
   digits in one strict order, and ten digits in a strict order can only be 0 to 9 (ANCHORS.md §2.3).

This was certified on the single-piece machine (step 6b) and the combined model (step 8), then put into the wiring
model (step 9) and the wide-menu model (step 11), and every later file inherits it. The files from step 9 on have no
hand-written adder in their source at all.

### 4.5 Generic cells: MLPs certify, ReGLU only sometimes

**The bridge** ([BRIDGE.md](bridge/BRIDGE.md)). The calculator is one weight setting of a generic looped transformer:
attention heads copy digits between the tokens of a worksheet (the wiring), one small MLP does the digit arithmetic
(the tables), and the block runs until every token settles (the loop). Compiled that way, it is exact at 14 sizes.
Then it was loosened one notch at a time and trained from random:
- **the digit tables can be generic MLPs**, if the cell gets the anchors' lessons: 31 of 32 tiny runs certified;
- **the plumbing is harder**: one MLP learning the subtraction, the copy and the keep-or-discard rules together
  certified in 4 of 16 runs;
- **the wiring learns only from short menus**: up to 10 options per connection with the tables given, 7 with the
  tables learned, none at 13 or more, and never with free attention, which falls for the bypass.

At H1, the wide model with both tables replaced by generic ReLU MLPs ([`bridge/h1/submission_wide_mlp.py`](bridge/h1/submission_wide_mlp.py),
step 12) certified, and both MLPs came out computing exactly the true tables.

**The handwired bridge** ([HANDWIRED_BRIDGE.md](handwired_bridge/HANDWIRED_BRIDGE.md)). The same machine compiled by
the user's handwired project into weighted sums and ReGLU products, with no attention at all:
- **its table rows train, its switches do not**;
- **the wiring learns only from short menus**: 7, 10 or 13 options, depending on the kind of switchboard;
- **the bypass is just as strong**, so its pull comes from the data, not from attention.

At H1 the wide model with ReGLU cells did not certify on the official H100 run: seed 74 fails, deterministically.
Through the same evaluator on CPU it certified on 1 of 4 seeds (seed 1).

### 4.6 General lessons, and the litmus test that asked for them

**The litmus test** ([LITMUS.md](litmus/LITMUS.md)). The best H1 model, unchanged, on 26 other tasks:
- **learned**: H1 with its prompt's fields reordered, and 5 of the 6 practice datasets whose moduli vary. The sixth,
  M5, read a two-digit T backwards: its T always has one digit, so nothing tells 16 from 61;
- **failed**: the 14 practice datasets with one fixed modulus (at most 2.4%); Montgomery's squaring, 10·y², y² + 1
  and cubing, where it built a perfect x² mod N machine and ignored the answers; and a prompt with four numbers.

Each failure needed a designer's change, and the one-modulus datasets had no small fix at all.

**General lessons** ([LESSONS.md](general_lessons/LESSONS.md), step 14). One file, unchanged on every task:
- **laws stated once**, about an operation, and applied to whatever numbers a row offers;
- **facts the model finds**: relations between the numbers that hold on every row it has seen (which is smaller, how
  many digits, divisibility). "x < N" is found, not told. Facts that can hold by accident ("N has at most 7 digits")
  are never used;
- **lesson sets the answers choose**: one template for a division step in two versions, long division and
  Montgomery's reduction. Each trains its own machine, and the training answers pick which one answers;
- **one menu of extra round steps** for every task: ×10, + a one-digit constant, × the round's input.

It learned, with zero per-task changes, y² + 1, 10·y², cubing and Montgomery's squaring, plus 10·y² + 1 and y² + 3,
which nobody designed for, each on every seed run. It still certified H1, more slowly (1,856 s against 558 s).

### 4.7 Discovering the skeleton: library learning and the inventing test

Neither of these is a submission.

**Library learning** ([LIBRARY.md](library/LIBRARY.md)). Search for short programs over 32 digit instructions on 26
practice tasks, notice which groups of instructions keep appearing in the programs that work, make each group one new
instruction (as BPE does for text, or by minimum description length), and search again. Level by level it rediscovered
the skeleton's parts: the ripple subtraction, the restoring link, the chain, the division step, the reduction loop
(token for token the hand-written reference), and the multiply rows. With solved tasks reused as routines, it then
found x² mod N itself, 22 primitives long, in a 5-token search of 23 minutes. That program is exact on every rung of
both H1 ladders up to T = 64. A version run inside one training run found the division's step and a multiply row, but
not the whole multiply or x² mod N.

**The inventing test** ([INVENTING.md](library/INVENTING.md)). It removed the three inputs the loop had from us, one at
a time, on the tiniest x² mod N:

| | with our practice tasks | with the target only, at growing sizes |
|---|---|---|
| our digit instructions | found | not found |
| bare counters (add one, subtract one, clear, copy, repeat, if zero) | found, if solved tasks become routines | not found: shortcuts fit the smallest sizes, then the search runs out |

The practice tasks did the inventing; the instruction set only decided which algorithm came out. Given the one step
"add one, and wrap around to 0 at N", the same search found x² mod N in about a second.

## 5. The removal ladder

Each step removed one hand-written piece. To count, it had to certify on the untouched official manifest on an H100,
with the file run byte for byte: T = 64 on both ladders, every rung 768/768. Then it was repeated on CPU seeds
through the same evaluator.

The ladder branches:
- steps 4a–4d each started from step 3's machine, and step 5 put them together;
- steps 6a and 6b went back to 4c's single-piece machine, whose reading, squaring and loop are still designed;
- steps 7a–7c stack on step 5, and so does step 8;
- step 9 is 7c with step 8's lessons, step 10 widens 7c's menus, and step 11 is 10 with step 9's change;
- steps 12–13 stack on 11, and step 14 starts again from 11.

| step | builds on | what became learned (the hand-written piece removed) | file (md5) | official record | training s (frozen at step) / evaluation s | CPU seeds certified | commit |
|---|---|---|---|---|---|---|---|
| 0 | — | nothing: the designed subtraction table, a plumbing check | an earlier `learned_remainder/submission.py` (`9732a11f`) | `s0_official` | 3.3 (100 steps) / 5.5 | 1 of 1 | `824d073` |
| 1 | 0 | the 200-entry subtraction table, taught with made-up practice divisions (gray under rule 14) | `submission.py` with `LESSON = "madeup"` | `final_s1_official` | 154.5 (484) / 4.8 | 8 of 8 (`s1v2` recipe) | `824d073` |
| 2 | 1 | no made-up numbers: the table checks itself on the rows' own numbers | [`submission.py`](learned_remainder/submission.py) (`bd477b9a`) | `final_s2_official` | 181.9 (499) / 4.9 | 8 of 8 (`s2v2` recipe), and 4 of 4 that never see an answer | `824d073` |
| 3 | 2 | the multiples of N: repeated subtraction with the learned table | [`submission_chain.py`](learned_remainder/submission_chain.py) (`ed360cb1`) | `chain_v2_official` | 271.8 (417) / 15.4 | 8 of 8, and 4 of 4 | `e1dd000` |
| 4a | 3 | integer squaring: a learned 100-entry times table, schoolbook | [`submission_square.py`](learned_remainder/submission_square.py) (`37765051`) | `square_official` | 270.5 (499) / 20.5 | 8 of 8, and 4 of 4 | `c0c86a2` |
| 4b | 3 | the integer loop over T and the stop rule: a countdown on the learned table, and a learned 2 × 2 gate | [`submission_control.py`](learned_remainder/submission_control.py) (`ec79a312`) | `control_official` | 239.7 (469) / 12.8 | 8 of 8, and 4 of 4 | `c6c9597` |
| 4c | 3 | the hand-written adder in the self-check, after a 70-step warm start | [`submission_adder.py`](learned_remainder/submission_adder.py) (`44e658ab`) | `adder_warm_official` | 77.8 (343) / 12.8 | 8 of 8, and 4 of 4 | `550f60e` |
| 4d | 3 | hand-written prompt parsing and answer writing: a learned reader and writer | [`submission_io.py`](learned_remainder/submission_io.py) (`45cc6e1b`) | `io_official` | 220.0 (440) / 12.8 | 8 of 8 | `f342a91` |
| 5 | 4a–4d | all of 4a–4d in one model, predicting inside autograd | [`submission_combined.py`](learned_remainder/submission_combined.py) (`4f5cecc7`) | `combined_official` | 666.6 (545) / 48.2 | 8 of 8 | `9959997` |
| 6a | 4c | most of the training recipe: three facts and the check remain; no answers, phases or freeze | [`ablation/submission_simple.py`](ablation/submission_simple.py) (`f374ec70`) | `simple_official` | 300.7 (3,000 steps, no freeze) / 23.1 | 8 of 8 (a bit-identical twin), and the file itself 4 of 4 | `f498805` |
| 6b | 6a | the warm start: digit meanings from the order of the numbers; no hand-written adder, no answers | [`anchors/submission_anchors.py`](anchors/submission_anchors.py) (`a9e20f66`) | `anchors_official` | 207.7 (3,000 steps, no freeze) / 16.0 | 8 of 8 | `5d2b917` |
| 7a | 5 | the division's wiring: 7 switches, 864 wirings | [`learned_wiring/submission_div.py`](learned_wiring/submission_div.py) (`11cccf13`); first version `submission_div_v1.py` (`a0050a2f`) | `div2_official`; first version: `div_official` | 313.6 (615) / 36.8; first version: 393.4 (640) / 48.5 | 8 of 8; first version: 7 of 8 | `36fc156` |
| 7b | 7a | the squaring's layout: 576 layouts | [`learned_wiring/submission_sq.py`](learned_wiring/submission_sq.py) (`781928ea`) | `sq_official` | 377.7 (479) / 49.0 | 8 of 8 | `36fc156` |
| 7c | 7b | the reader's and writer's shape | [`learned_wiring/submission_io3.py`](learned_wiring/submission_io3.py) (`c2331362`); first version `submission_io3_v1.py` (`070d2ee5`) | `io3v2_official`; first version: `io3_official` | 728.6 (532) / 51.8; first version: 729.4 (532) / 48.9 | 8 of 8; first version: 6 of 8 | `36fc156` |
| 8 | 5 | the warm start in the combined model, with step 6b's lessons | [`anchors/submission_combined_anchors.py`](anchors/submission_combined_anchors.py) (`8887fdae`) | `cmbanchors_official` | 454.0 (553) / 40.6 | 8 of 8, and 4 of 4 with the old table lessons off | `5d2b917` |
| 9 | 7c, with step 8's lessons | the warm start in the wiring model: no hand-written arithmetic anywhere | [`learned_wiring/submission_anchored.py`](learned_wiring/submission_anchored.py) (`6328a7d2`) | `anchored_official` | 466.1 (522) / 50.3 | 8 of 8 | `b76f2cd` |
| 10 | 7c | the menus' narrowness: the wiring chosen from about 4·10^17 whole machines instead of about 6 million | [`wider_menus/submission_wide.py`](wider_menus/submission_wide.py) (`43db5606`) | `wide_official` | 591.8 (566) / 42.9 | 8 of 8 | `c166f8d` |
| 11 | 10, with step 9's change | the wide menus with no hand-written arithmetic anywhere | [`wider_menus/submission_wide_anchored.py`](wider_menus/submission_wide_anchored.py) (`c82b11fa`) | `wide_anchored_official` | 558.0 (588) / 41.4 | 8 of 8 | `d8856a8` |
| 12 | 11 | the tables' form: both digit tables as generic ReLU MLPs | [`bridge/h1/submission_wide_mlp.py`](bridge/h1/submission_wide_mlp.py) (`22bd34d0`) | `wide_mlp_official` | 1,047.7 (743) / 98.0 | 8 of 8 | `16b66dd` |
| 13 | 12 | the chain's menu: its wiring chosen inside training among 7.9·10^9 relative-position wirings | [`tournament/submission_tournament.py`](tournament/submission_tournament.py) (`7ae547d9`); first version `312d9422` (not kept as a file) | `tourney2_official` | 2,114.0 (705) / 89.8 | 8 of 8; first version: 3 of 8 | `90f245f` |
| 14 | 11 | task-specific lessons: laws stated once, mined facts, lesson sets chosen by the answers | [`general_lessons/submission_general.py`](general_lessons/submission_general.py) (`ddfdc065`) | `gen_h1_official` | 1,855.6 (645) / 97.2 | 8 of 8 | `1613c28` |

Notes:
- Training time includes the steps after the freeze, when the learning rate is 0. "Never see an answer" means the same
  file with one constant changed (`REAL_MAX_T = 0`).
- Records live in `runs/learned_remainder/`, `runs/learned_wiring/`, `runs/wider_menus/`, `ablation/runs/`,
  `anchors/records/`, `bridge/h1/records/`, `tournament/h1_records/` and `general_lessons/records/`, one
  `<name>.json` each. Each holds the full log, the result and the exact source.
- MENUS.md's four variant H100 runs certified too: `wide_try2_h100` and `wide_d4m3_try1_h100` (earlier builds at
  narrower and full width), `wide_enum_d3m2_h100` (without one of the step laws) and `wide_joint_d1m2_h100` (the
  general laws alone, at the narrowest level, in 1,002.6 s).
- The general file's other tasks (`general_lessons/records/gen_*`): every seed run certified on y² + 1, 10·y², cubing,
  Montgomery's squaring (4 of 4), 10·y² + 1 and y² + 3 (2 of 2 each). It also certified on H1 with a reordered prompt
  and on the practice set M5.
- Records of failures are kept too:
  - `s1_official`: Stage 1's first recipe, with 8 of 200 rows right and no rung certified;
  - `adder_official`: a learned adder from a random start with no warm start, with no rung certified;
  - the first versions' failed CPU seeds: `div_cpu_seed7`, `io3_cpu_seed2`, `io3_cpu_seed5`, and five
    `tourney_cpu_seed*` (the chain's gate flipped, so no round ever ran);
  - `tourney2_nonorm_cpu_seed*`: the tournament without the moved-up rows, 4 of 8 certified;
  - `wide_try1_h100`: a first wide-menu build that crashed on a bug when its chain lesson joined (MENUS.md §4.2);
  - `handwired_bridge/h1/records/wide_reglu_official` and `wide_reglu_official2`: the wide model with ReGLU cells,
    nothing right on either official run, and 1 of 4 CPU seeds certified.

  `chain_official`, the chain's first recipe, certified, but one of its CPU seeds that never sees an answer did not.

## 6. Rules 1–16 for the models at the frontier

The rules are [one-layer-deeper/README.md §Rules](../one-layer-deeper/README.md#rules). "Gray" means a reasonable
reader could object; the point is flagged, not argued away. This covers the two newest frontier files, the tournament
file ([`tournament/submission_tournament.py`](tournament/submission_tournament.py)) and the general file
([`general_lessons/submission_general.py`](general_lessons/submission_general.py)). Both build on
[`wider_menus/submission_wide_anchored.py`](wider_menus/submission_wide_anchored.py), whose status they keep except
where noted. It follows COMBINED.md's own table, the rules tables of WIRING.md, MENUS.md, TOURNAMENT.md and LESSONS.md,
and ANCHORS.md §7. The imports, the initialization and the absence of backward calls were re-checked in both sources.

| rule | status | why |
|---|---|---|
| 1. one `submission.py` exporting one `Submission` | yes | each file is sent as `submission.py`; each has one `SUBMISSION` with model, optimizer, loss, batch sizes and a step cap |
| 2. self-contained | yes | each imports only `torch`, `itertools`, `collections` and the public `benchmark` API; no files, no network |
| 3. the participant defines model, optimizer, loss, batch sizes, steps | yes | Adam; the rounds run until no prompt wants another (adaptive computation, capped at 100) |
| 4. the evaluator owns the loop, data order, backward, clipping, cadence, seeds, deadline | **gray** | the evaluator is untouched: one pass per step, no batch reuse. But the schedule lives in the model as Python state: the reader's stages, the phases, when each table, lesson and switch joins, the Adam resets, and a freeze that sets the learning rate to 0 once every check has passed for 200 steps. That freeze is early stopping on the model's own checks. The general file's fact miner and lesson-set selector add more of that state |
| 5. at most 500M parameters | yes | 123,993 (tournament: mostly one weight per wiring combination) and 11,589 (general) |
| 6. no hard-coded weights | yes | every parameter starts random: 0.1 × random normal, and PyTorch's default random init for the tournament file's MLP layers, except those MLPs' output biases, which start at 0 and are trained. The "designed" mode, with every table and switch at the truth, is only a plumbing check, never the submission |
| 7. no hard-coded algorithm in the forward pass | **gray, less than before** | every computation is a learned table lookup or, in the tournament file, a generic ReLU MLP, and every connection is a learned switch. **Tournament file:** the chain's connections are chosen from relative positions (any register at any offset), not from a menu of meaningful parts. **General file:** the wide model's menus, plus a menu of extra round steps and a second, Montgomery lesson set; the answers choose among them, not a designer per task. **Still gray in both:** the skeleton is designed (one register, 8 steps of 9 links, bring-in → chain → move; one product per digit pair, placed and summed; the T loop); the I/O and multiply menus are written by hand and contain the true wiring; so do the tournament's offset window and blocks. A strict reader can say the algorithm moved into the skeleton and the menus |
| 8. final logits from the model inside autograd, with an unbroken gradient path | yes, with gray edges | at evaluation every digit is a one-hot vector and every table, cell and switch a straight-through one-hot, so the answer is exactly the machine's, and a gradient from the final logits reaches the learned tensors. Gray: (a) in training the forward returns zero logits, so the tables and switches learn through the lessons, not through the final logits; (b) the gradient is damped 10× per subtraction link, addition and round so it does not overflow, and on deep paths it underflows to 0; (c) some switches or combinations get exactly 0 gradient on a given batch: the wide menu's final correction on some trained models (aliases whose gradients cancel, MENUS.md §4.5), up to a few dozen of the tournament's 98,104 combinations, and in the general file the lesson set that was not selected; (d) which reading hypothesis a prompt uses is decided by designed run-detection logic, with a soft surrogate on the gradient side |
| 9. everything on the GPU | yes | state and computation stay on the device. The schedule reads a few scalars to Python per step, and evaluation syncs one boolean per round to end the loop |
| 10. OOM or timeout fails | ok | neither happened on the official runs |
| 11. 3,600 s of training, compilation included | yes | 2,114.0 s (tournament) and 1,855.6 s (general), no compilation |
| 12. a custom loss returns one finite differentiable scalar | yes | `token_training_loss` |
| 13. evaluation within 1,800 s; certification | yes | 89.8 s and 97.2 s; T = 64 on both ladders |
| 14. no data inspection, augmentation, task-specific solvers, custom loops or participant backward passes | **gray** | neither file calls `backward` or `autograd.grad`, defines a custom autograd function, or reads the data files, and no hand-written arithmetic runs at any step of training. But the lessons carry designed knowledge: the anchors' facts and laws; the two step laws, which are definitions of long division's step; "x mod N = x", "N mod N = 0" and "x + Q·N mod N = x" through the whole division; the times-table ladder; the old table lessons, which join late and include "a wrapped x − N is large", an assumption about the register's width; the reading and T's direction, chosen inside the loss by trying every complete hypothesis; and only answers of T ≤ 2 rows whose own checks pass. The switches' pass/fail verdicts come from running the snapped machine on candidate wirings inside the loss. **Tournament file, four new items:** the link law (what one link does); the moved-up rows, new numbers made from each row's own x and N, close to augmentation; the loop law (the loop's stop rule); and the pruning of combinations more than 12 nats behind. **General file, better and new:** no lesson states a fact about H1's data, and the step lessons are candidates the answers can reject; but the model computes statistics over its training batches to mine facts, and the template and the round menu are designed |
| 15. repeated rule-breaking | n/a | the write-ups flag their own gray areas, and nothing has been submitted to the hosted evaluator |
| 16. do not exploit the Hard metric recorder | yes | not touched |

**The earlier certified models, in one line each.**
- **SpecBit**: only one gate's truth table is learned, and the netlist around it is the algorithm. Under a strict
  reading of rule 7, the design panel's critics rated every one of the panel's designs fatal
  ([REPORT.md](REPORT.md) §6).
- **UnfoldB and UnfoldS**: 4 learned numbers, with the snapping, the remainder and the loop built into the model.
  Their notes discuss rules 7, 8 and 14 ([unfold_sub/NOTE.md](unfold_sub/NOTE.md),
  [unfold_small/NOTE.md](unfold_small/NOTE.md)).

## 7. Status and open questions

**What is running.** Every study described here has finished and is committed. One new session, "Bar 2 integration:
tournament + library + general lessons", started at 12:38 on Oct 22 and takes up open question 1 below. When this was
refreshed, its folder [`integration/`](integration/) held a single working file, `work/merged_raw.py`, and nothing was
committed.

**Open questions for bar 2** (nothing a strict judge would call the hand-written algorithm):
1. **One file.** The tournament, the generic cells and the general lessons each certify H1 on their own. Combining
   them has not been tried. The general file's lesson-set selector already chooses among whole candidates by their
   answers, and LESSONS.md proposes the tournament's pruning to make the losing lesson set cheap.
2. **The skeleton inside training.** Library learning rediscovers the skeleton offline, from curated practice tasks.
   Inside one training run, with laws on the rows' own numbers as its practice, it found the division's step and a
   multiply row, but not the whole multiply or x² mod N (LIBRARY.md §6). Which pieces it merges also comes down to
   ties between programs' spellings.
3. **The instruction set and the alphabet.** The loop starts from 32 hand-written digit instructions, and every
   model's cells work on a hand-chosen alphabet of 10 decimal digits. The inventing test shows that bare counters
   suffice in principle, at sizes far below H1.
4. **The late lessons.** Every frontier file still runs the old table lessons 100 steps after the digit laws start,
   and the width-dependent wrap-around fact is among them. Switching them off was tested only in the combined model,
   where 4 of 4 seeds certified without them.
5. **Rules 8 and 14.** Training never sends the loss through the final logits, and the prediction path's gradient is
   damped and can miss some switches. The lessons state facts and laws, and the tournament's moved-up rows are close to
   augmentation. A strict reviewer should decide these before any real submission.

**Open questions for bar 3** (a general algorithm learner):

6. **Practice problems.** The inventing test found that the loop cannot invent without practice tasks whose short
   solutions are the pieces. A learner would have to pose its own, for example from its near misses ("what does my
   almost-right program get wrong?"). Untested.
7. **The wrap-around concept.** Given the one step "add one, wrap around at N", the search finds x² mod N at once.
   Where that concept should come from is open.
8. **What the data cannot say.** With one fixed modulus, facts cannot tell the model what its digits mean; in 12 of
   the 14 such datasets only the answers single out the true digit code, among 8,100 to 453,600, and nothing here
   searches whole codes. With T always one digit, nothing says which way a two-digit T runs.
9. **A step outside the menu.** y² − 1 fails because no menu option computes "subtract 1". The answers choose among
   what the designer wrote; they cannot invent a missing step.

**And:** submitting to the hosted evaluator after the deadline, just to see a response, has not been done.

## 8. Where everything is

| path | what |
|---|---|
| [`OVERVIEW.md`](OVERVIEW.md) | this file |
| [`README.md`](README.md) | one line for every file and folder |
| [`REPORT.md`](REPORT.md) | the first-day report: the analysis (§0–8), SpecBit and the grammar search (§9–14) |
| [`learned_remainder/`](learned_remainder/) | the removal ladder, steps 0–5: every `submission*.py`, its write-up (LEARNED_REMAINDER, CHAIN, SQUARE, CONTROL, ADDER, IO, COMBINED), the Modal launchers, plans and local checks |
| [`ablation/`](ablation/) | which training ingredients matter ([ABLATION.md](ablation/ABLATION.md)), and `submission_simple.py` (step 6a) |
| [`anchors/`](anchors/) | digit meanings from the order of the numbers ([ANCHORS.md](anchors/ANCHORS.md)): `submission_anchors.py` (step 6b), `submission_combined_anchors.py` (step 8), records in `anchors/records/` |
| [`learned_wiring/`](learned_wiring/) | the wiring learned in three stages ([WIRING.md](learned_wiring/WIRING.md)): `submission_div.py`, `submission_sq.py`, `submission_io3.py` (steps 7a–7c) and the first versions `*_v1.py`; then with no warm start ([ANCHORED.md](learned_wiring/ANCHORED.md)): `submission_anchored.py` (step 9) |
| [`wider_menus/`](wider_menus/) | the wide menus ([MENUS.md](wider_menus/MENUS.md)): `submission_wide.py` (step 10); with no warm start ([WIDE_ANCHORED.md](wider_menus/WIDE_ANCHORED.md)): `submission_wide_anchored.py` (step 11) |
| [`bridge/`](bridge/) | the machine inside a generic looped transformer, loosened notch by notch ([BRIDGE.md](bridge/BRIDGE.md)); `h1/submission_wide_mlp.py` (step 12) |
| [`handwired_bridge/`](handwired_bridge/) | the machine compiled by handwired into affine + ReGLU layers, loosened the same way ([HANDWIRED_BRIDGE.md](handwired_bridge/HANDWIRED_BRIDGE.md)); `h1/submission_wide_reglu.py`, which did not certify officially |
| [`tournament/`](tournament/) | the wiring chosen inside training by tournaments ([TOURNAMENT.md](tournament/TOURNAMENT.md)): `submission_tournament.py` (step 13), the tiny machine `tiny.py`, records in `tournament/h1_records/` |
| [`general_lessons/`](general_lessons/) | a law library, mined facts and answer-chosen lessons ([LESSONS.md](general_lessons/LESSONS.md)): `submission_general.py` (step 14), run on every task, records in `general_lessons/records/` |
| [`litmus/`](litmus/) | the best model run unchanged on 26 other tasks ([LITMUS.md](litmus/LITMUS.md)), with the per-task variants that fixed some |
| [`library/`](library/) | search and compression that rediscovers the skeleton ([LIBRARY.md](library/LIBRARY.md)), and the inventing test ([INVENTING.md](library/INVENTING.md), code in `library/inventing/`) |
| [`integration/`](integration/) | the bar-2 integration session's folder: just started, not committed |
| [`tinyrouting/`](tinyrouting/) | wiring and tables learned together in tiny machines |
| [`tinycore/`](tinycore/) | the quotient-digit chicken-and-egg, computed exactly |
| [`microlab/`](microlab/) | a small harness for one-minute experiments, and its findings ([microlab/README.md](microlab/README.md)) |
| [`unfold_sub/`](unfold_sub/), [`unfold_small/`](unfold_small/) | the UnfoldB and UnfoldS submissions |
| [`submission.py`](submission.py) | SpecBit |
| [`panel/`](panel/), [`panel_synthesis.md`](panel_synthesis.md) | the five-design panel and its synthesis |
| [`FINDINGS_substrate.md`](FINDINGS_substrate.md), [`RELAXED.md`](RELAXED.md), [`UNFOLD.md`](UNFOLD.md), [`UNFOLD_FAMILY.md`](UNFOLD_FAMILY.md) | the day-1 topic write-ups |
| `runs/` | run records. Official ones: `runs/learned_remainder/*_official.json`, `runs/learned_wiring/*_official.json`, `runs/wider_menus/*_official.json`, `runs/unfold_sub/official_h100.json`, `runs/unfold_small/official_h100.json`, `runs/h100_h1.json`. Later folders keep their own: `ablation/runs/`, `anchors/records/`, `bridge/h1/records/`, `tournament/h1_records/`, `general_lessons/records/`, `handwired_bridge/h1/records/`, `litmus/records/`, `library/records/` |
| `../one-layer-deeper/` | the competition clone at `87e5c26` ("Releasing H1"), with the H1 data under `data/generated/` |
| `~/github/handwired-lab/montgomery-search` | outside this repo: search over a Montgomery squaring program in the handwired project (branch `lab/montgomery-search`) |

**The certified submissions, with their records.** Steps 0–14 are in the §5 table. The earlier three:

| model | file (md5) | official record | training s / evaluation s | learned numbers |
|---|---|---|---|---|
| SpecBit | [`submission.py`](submission.py) (`a432bb5e`; the record embeds no source) | `runs/h100_h1.json` | 16.5 (20 steps) / 26.9 | 65,536 (one logit per possible gate) |
| UnfoldB | [`unfold_sub/submission.py`](unfold_sub/submission.py) (`17299657`) | `runs/unfold_sub/official_h100.json` | 384.3 (1,000 steps, batch 100,000) / 3.7 | 4 |
| UnfoldS | [`unfold_small/submission.py`](unfold_small/submission.py) (`eb6aedfc`) | `runs/unfold_small/official_h100.json` | 1,624.9 (200,000 steps, batch 512) / 3.7 | 4 |

All three reached T = 64 on both ladders, with every rung 768/768 and every split exact.

**How to rerun.** Each write-up ends with its commands. The ladder's official runs go through a Modal launcher
started from the competition clone: `learned_remainder/modal_chain.py --file <submission>` for steps 0–5, and each
later folder's own copy (`ablation/modal_ablate.py`, `anchors/modal_anchors.py`, `learned_wiring/modal_wiring.py`,
`wider_menus/modal_wide.py`, `bridge/modal_bridge.py` and `bridge/h1/`, `tournament/modal_h1.py`,
`general_lessons/modal_general.py`). Each builds the competition's image, regenerates H1 with the official generator,
and runs the untouched manifest.

## 9. Where docs and records disagree

What was checked, for every number above:
- each official record's certified T, rung counts, split counts, training and evaluation seconds, parameter count and
  freeze step, against the write-ups that cite them;
- the md5 of the source each record embeds, against the files. Where they differ, the difference is the one `--set`
  line the write-up names, or a first version the write-up keeps by its md5;
- the CPU-seed claims, by tallying the records;
- for the models at the frontier, the final tables printed in the official logs, against true subtraction and
  multiplication;
- for the litmus and general-lessons tasks, the certified T and score of each record the write-ups' tables cite.

The numbers checked this way all match their records in the write-ups that cite them, in LEARNED_REMAINDER, CHAIN,
SQUARE, CONTROL, ADDER, IO, COMBINED, ABLATION, ANCHORS, WIRING, ANCHORED, MENUS, WIDE_ANCHORED, BRIDGE, TOURNAMENT,
LESSONS, LITMUS, LIBRARY and HANDWIRED_BRIDGE, and in the two Unfold notes and REPORT.md §9. The exceptions:
- **COMBINED.md calls `combined_official` and `combined_index_official` "the same training run".** They are two runs
  with identical training: the same file apart from the `PATH` line, the same seed and data order, and both froze at
  step 545 with the same tables. They were launched together, and their training times differ (666.6 s and 566.5 s).
- **The leaderboard count** (3 of 146 certified, REPORT.md §0) has no record behind it and was not re-checked.

Two counts need a word, though neither is wrong:
- **The tournament's machine count.** The official log prints a total of 2.8·10^22 machines. TOURNAMENT.md's
  "about 3·10^23" also counts the gate table's 4 settings and the reader's 3 shapes; with them, the log's numbers
  give 3.4·10^23.
- **The practice tiers.** Their rungs have anywhere from 14 to 768 prompts, depending on the dataset, so their records
  were checked by their certified T and score rather than against H1's 768 per rung.

The first version of this file also flagged WIRING.md for contradicting itself on stage 1's CPU seeds. The finished
WIRING.md says 7 of 8 throughout, and the record of the failed seed (`div_cpu_seed7`) is in `runs/learned_wiring/`.
