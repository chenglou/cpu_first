# One Layer Deeper H1, from the CPU up: where everything stands

*The entry point for `cpu_first/`. First written 2026-10-21 at 03:45 Pacific and refreshed twice: at about 10:45, after
the ablation, digit-anchor and learned-wiring work was committed, and at about 15:15, after the wide menus and the
anchored wiring model were committed.*
- *What was checked against the run records (`runs/`, `ablation/runs/`, `anchors/records/`): every certification, run
  time, freeze step, parameter count and CPU-seed count quoted here. The source embedded in each record was hashed and
  compared with the file's md5.*
- *The data counts in §1–2 were computed from the released H1 files, and the commits come from `git log`.*
- *The one-line findings of the timeline (§3) and the summaries of how each piece learns summarize their write-ups,
  which are linked; they were not re-run.*
- *The older write-ups are left as they were. §9 lists where a write-up and a record disagree.*
- *Nothing was running when this was refreshed, and all the work described is committed (the last commit is
  `d8856a8`, at 14:29).*

**In one minute.**
- The task is exact repeated modular squaring. Almost every entry on the leaderboard scored near 0%, because the
  remainder erases partial credit: an almost-right model gets random-looking answers, so gradients have nothing to
  follow (§1–2).
- Designs built like a CPU certify: discrete digits, one small cell reused everywhere, and labels that choose between
  whole machines. Four lines of work certified through the official evaluator: SpecBit, UnfoldB, UnfoldS and the
  learned digit machine (§3).
- **The current best model** is a paper-arithmetic calculator in which everything that computes is learned (§4):
  - its arithmetic is small learned tables;
  - its wiring is chosen in training from menus of about 4·10^17 whole machines. The menus hold genuinely different
    algorithms: repeated subtraction and trial multiples for the division, six ways to group the products, and a
    Montgomery reduction that the checks reject;
  - no hand-written arithmetic runs anywhere, in training or in the answer: the digits' meanings come from the order
    of the numbers.

  It certifies T = 64 on both ladders on the official H100 manifest, and on 8 of 8 CPU seeds.
- It got there by removing hand-written parts one at a time, with certification as the gate at each step (§5).
- Still hand-written: the skeleton the wiring plugs into, the menus, the sizes, the loop over T and the lessons. Two of
  the lessons are definitions of long division's step (§4, §7).

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
| 14:29 `d8856a8` | **the wide menus with no hand-written arithmetic anywhere**: the current best model | [wider_menus/WIDE_ANCHORED.md](wider_menus/WIDE_ANCHORED.md) |

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
- **With wide menus, the checks still identify; the search is what can fail.** In every menu counted exactly, every
  wiring that passes all the checks is a correct one. Learning failed only where the search could not see the right
  combination: shown only a few sampled combinations instead of every one, a switch was caught by impostors in 14 of
  16 runs ([MENUS.md](wider_menus/MENUS.md)).
- **What worked every time:** digits snapped to exact values, one shared cell, checks on each training row's own
  numbers that need no answer, and pass/fail scoring of whole choices.

## 4. The current best model

**[`wider_menus/submission_wide_anchored.py`](wider_menus/submission_wide_anchored.py)**: md5
`c82b11fa8a93b09c0724c3931cf8de98`, committed in `d8856a8`. Write-ups: [WIDE_ANCHORED.md](wider_menus/WIDE_ANCHORED.md)
for this file, [MENUS.md](wider_menus/MENUS.md) for its menus, [ANCHORED.md](learned_wiring/ANCHORED.md) for its
digit lessons.

It is the combined model's calculator (§4.1), with two things learned that were hand-written there:
- **its wiring**, chosen from wide menus (§4.2);
- **its digit meanings**, taught by the order of the numbers instead of a hand-written adder (§4.3).

No hand-written arithmetic runs anywhere, in training or in the answer.

| | |
|---|---|
| learned numbers | 11,187, all starting at 0.1 × random normal |
| official run (H100, untouched manifest, the file byte for byte) | `wide_anchored_official`: T = 64 on both ladders, every rung 768/768, test 9,999/9,999, ood_t and ood_n_t 10,002/10,002 each |
| training s (frozen at step) / evaluation s | 558.0 (588) / 41.4 |
| CPU seeds (same evaluator, `device = cpu`) | 8 of 8 certified |
| tables at the end of the official run | subtraction 200/200 and times 100/100, both equal to the truth; the gate the true one; the audit found no entry used at evaluation that training never used |
| what it chose | long division by **repeated subtraction**, a correct wiring with two harmless aliases (it is not literally the reference wiring); the schoolbook rows summed in reverse order; the reader's step shape. The CPU seeds chose repeated subtraction 7 times and **trial multiples** once (seed 1) |

**What is still hand-written** (MENUS.md §1 and §5, WIDE_ANCHORED.md, ANCHORED.md):
- **the skeleton** the switches plug into: one register of 9 places, 8 steps of 9 links each, in the order bring a
  digit in → run the chain → move, then one final correction and one readout; the multiply as one product per digit
  pair, placed and summed;
- **the menus**: every option was written by hand, and the right wirings are among them. Some methods are not in them
  at all, such as non-restoring division and Barrett reduction;
- **the sizes**: 8, 8 and 2 digits read and 8 written, and a 2-digit countdown;
- **the loop over T**: count down by subtracting the number 1, and run rounds until no prompt wants another, at most
  100;
- **a + b = a − (0 − b)**, both done with the subtraction table;
- **the reader's menus and grammar**: its menu of 24 readings (which digit run is N and which is x, and which way
  each runs), its 3 shapes, and small rules such as "a slot reads 0 once a slot before it has met a non-digit";
- **the lessons and their schedule.** Among them are the anchors' three statements ("x < N on every top part",
  "subtraction keeps, reverses or wraps the order", "every row and column of the table uses each digit once"), and two
  step laws that are definitions of long division's step ("one step from x + q·N gives back x", and how a step takes
  its digit in).

### 4.1 The calculator underneath: the combined model

**[`learned_remainder/submission_combined.py`](learned_remainder/submission_combined.py)**: md5
`4f5cecc780894499a462499c3cd7d67f`, committed in `9959997`. Write-up: [COMBINED.md](learned_remainder/COMBINED.md).
It was the first model in which every computing part is learned. Its wiring and its warm start were still written by
hand.

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

**What was hand-written in it** ([COMBINED.md](learned_remainder/COMBINED.md) "In plain words"):
- **The wiring**, which is the paper layout:
  - schoolbook squaring in 8 rows;
  - long division in 8 steps, each with up to 9 subtractions of N, on a 9-digit register;
  - a 2-digit countdown register and the number 1;
  - the keep-or-discard plumbing;
  - "run rounds until no prompt wants another", capped at 100.
- **The reader's and writer's shape**: start next to some symbol, step one way, and stop at the first non-digit.
  There are 8, 8 and 2 slots for reading and 8 for writing, and 24 hypotheses about which digit run is which number.
- **A warm start.** At first the subtraction table checks itself against a hand-written adder, until the reader has
  worked out the prompt and for at least 70 steps. That ended at step 90 in the official run, and at steps 90–273
  across all runs. After it, no hand-written arithmetic runs anywhere, in training or in the answer.
- **The lessons and their schedule.**

**How it learns.** Mostly not from answers. Most lessons are checks on each training row's own numbers that need no
answer:
- subtract N, add it back, and the start must come back;
- x < N, so x − N must borrow and N − x must not;
- x × (b + 1) = x × b + x;
- round trips through the reader and writer.

The answers of the T ≤ 2 rows join late, and only along the machine's own path.

**Official results** (Modal H100, the official manifest untouched, seed 74; records in `runs/learned_remainder/`):

| record | what ran | max T, seen / unseen moduli | rungs | test, ood_t, ood_n_t | training s (frozen at step) | evaluation s |
|---|---|---|---|---|---|---|
| `combined_official` | the file byte for byte | **64 / 64** | 768/768 at every rung of both ladders | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 666.6 (545) | 48.2 |
| `combined_index_official` | the file with `PATH = "index"` (the older integer prediction path), the only line changed | 64 / 64 | all 768/768 | all exact | 566.5 (545) | 23.9 |
| `combined_s0_official` | `MODE = "designed"`: every table set to the truth, a plumbing check only | 64 / 64 | all 768/768 | all exact | 143.0 (frozen from the start) | 34.3 |

- **CPU seeds:** the file byte for byte, through the same evaluator with `device = cpu`, certified on 8 of 8 seeds (74
  and 1–7), each on both ladders, every rung 768/768.
- **The audit lines** of the official run say that every table entry used at evaluation was trained: 200/200
  subtraction rows, 100/100 times-table cells, 24 countdown rows and 2 gate rows.
- **Inside autograd (rule 8).** At evaluation every digit is a one-hot vector, and every table acts as a
  straight-through one-hot matrix. So the logits are exactly the calculator's, while a gradient from the final logits
  reaches all 10 learned tensors (tested in [COMBINED.md](learned_remainder/COMBINED.md) §3).

### 4.2 The wiring, learned: from short menus to wide ones

**Short menus first** ([WIRING.md](learned_wiring/WIRING.md), steps 7a–7c in §5). Every connection the combined model
wired by hand became a switch: a small menu of options, one of which is right. Each switch starts at random and is
learned alongside the tables, in three cumulative stages:
1. **The long division**, 7 switches, 864 wirings in all: which end of the square the division starts from; how the
   register makes room for the next digit; what each subtraction subtracts from, and what it subtracts; what its
   result does (always replace, replace when the gate says so, or never); whether N's digits are reversed; which
   places are the answer.
2. **The squaring's layout**: one placement rule for the products' low digits and one for their high digits, each
   chosen from 24 rules, 576 layouts in all.
3. **The reader's and writer's shape**, one switch for all four fields: an anchor and a shared step, an anchor and a
   separate offset per digit, or a fixed position per digit.

Each switch is taught by a check that only the right wiring passes, run on the rows' own numbers:
- one division step started from x + q·N must give back x;
- dividing x by N must leave x, and N by N must leave 0;
- the times-table ladder, row by row;
- for the reader's shape, the reading under which x < N holds on every question.

For the division, every combination of options is tried on the snapped machine, and the combinations that pass gain
weight. The squaring's rules and the reader's shapes are weighed as a mixture of whole options, and training raises
the mixture's chance of a right row or a right reading.

Two first versions failed CPU seeds and were fixed, each kept with its records:
- the division's first version (`submission_div_v1.py`) failed 1 of 8, because its untrained reader read every N as 0;
- stage 3's first version (`submission_io3_v1.py`) failed 2 of 8, because T's lesson pulled the shared shape switch
  off the right shape.

**Then wide menus** ([MENUS.md](wider_menus/MENUS.md), step 10). Each connection got many options:
- **the division**: a link can take any of several registers on either input, including zero and the multiples of N;
  it can add or subtract and stop on several kinds of signal; the register can move any number of places, take the
  incoming digit at any place, start empty or full, and give its answer from several windows. That is 4.1 billion
  wirings, against stage 3's 3,456 (its 864 times the gate's 4 tables);
- **the multiply**: each digit product can go to any column given by a simple formula of its digit positions, and the
  products can be grouped into the numbers to be added in nine ways. That is 33 million layouts, against 576;
- **together with the reader's 3 shapes**, about 4·10^17 whole machines, against about 6 million.

The menus now hold genuinely different algorithms, and the machine picks among those that work:
- **two long divisions**: subtract N again and again until the result would go negative (stage 3's way), or try the
  step's number minus s·N for every s and keep the last that stays positive (trial multiples, as on paper, using the
  machine's own multiplier). Both pass every check;
- **six groupings of the products**: rows by either digit, or diagonals, each in either order;
- **a Montgomery reduction**, the right-to-left method of cryptographic hardware. It runs exactly but computes a
  different number (the dividend times 10^−W, mod N), so the checks reject it.

One new general law matters: x + Q·N mod N = x, with Q another row's x. It fills the dividend's top half, where
"x mod N = x" and "N mod N = 0" never look, and so removes impostor wirings that never read that half.

### 4.3 No hand-written adder: digit meanings from the order of the numbers

Write-up: [ANCHORS.md](anchors/ANCHORS.md). Until then the subtraction table learned what its digits mean from a
hand-written adder, used for its first steps. The ablation showed that this warm start was load-bearing: without it,
no run got past 39 of 200 table entries. The anchors replace it with the order of the numbers:
1. **The borrows, from x < N on every "top part".** Take x's digits from some place upward, and N's digits from the
   same place. The first number is at most the second, and strictly smaller exactly when the two digit strings differ.
   So taking N's part from x's part must borrow exactly then, and taking x's part from N's never borrows. That teaches
   every borrow.
2. **The digits, from two laws of subtraction**, applied to one digit at a time once the borrows are right:
   - taking the same digit away from two digits keeps their order;
   - taking more away from the same digit leaves less;
   - except when exactly one of the two goes below zero: that one wraps round and comes out on top.

   In each column the borrows say which entries go below zero, and the laws put the column's ten digits in one
   strict order. Ten digits in a strict order can only be 0 to 9, so only the true table passes (ANCHORS.md §2.3).
   The file also asks every row and column of the table to use each digit once.

The table gets no lesson until the reader has worked out the prompt. Before that, the untrained reader reads nonsense,
such as every N as 0. The old table lessons join 100 steps after the digit laws, when the table is already exact.

This was certified first on the single-piece machine (`anchors/submission_anchors.py`, step 6b) and on the combined
model (step 8). It was then put into the stage-3 wiring model (step 9) and into the wide-menu model (step 11). The
single-piece, step-9 and step-11 files have no hand-written adder in their source at all. The combined-anchors file
(step 8) still contains it behind a switch (`TABLE_TEACH = "order"`), where it never runs.

## 5. The removal ladder

Each step removed one hand-written piece. To count, it had to certify on the untouched official manifest on an H100,
with the file run byte for byte: T = 64 on both ladders, every rung 768/768. Then it was repeated on CPU seeds
through the same evaluator.

The ladder branches:
- steps 4a–4d each started from step 3's machine, and step 5 put them together;
- steps 6a and 6b went back to 4c's single-piece machine, whose reading, squaring and loop are still designed;
- steps 7a–7c stack on step 5, and so does step 8;
- step 9 is 7c with step 8's lessons, step 10 widens 7c's menus, and step 11 is 10 with step 9's change.

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
| **11** | 10, with step 9's change | **the wide menus with no hand-written arithmetic anywhere: the current best model** | [`wider_menus/submission_wide_anchored.py`](wider_menus/submission_wide_anchored.py) (`c82b11fa`) | `wide_anchored_official` | 558.0 (588) / 41.4 | 8 of 8 | `d8856a8` |

Notes:
- Training time includes the steps after the freeze, when the learning rate is 0. "Never see an answer" means the same
  file with one constant changed (`REAL_MAX_T = 0`).
- Records live in `runs/learned_remainder/`, `runs/learned_wiring/`, `runs/wider_menus/`, `ablation/runs/` and
  `anchors/records/`, one `<name>.json` each. Each holds the full log, the result and the exact source.
- MENUS.md's four variant H100 runs certified too: `wide_try2_h100` and `wide_d4m3_try1_h100` (earlier builds at
  narrower and full width), `wide_enum_d3m2_h100` (without one of the step laws) and `wide_joint_d1m2_h100` (the
  general laws alone, at the narrowest level, in 1,002.6 s).
- Records of failures are kept too:
  - `s1_official`: Stage 1's first recipe, with 8 of 200 rows right and no rung certified;
  - `adder_official`: a learned adder from a random start with no warm start, with no rung certified;
  - the first versions' failed CPU seeds: `div_cpu_seed7`, and `io3_cpu_seed2` and `io3_cpu_seed5`;
  - `wide_try1_h100`: a first wide-menu build that crashed on a bug when its chain lesson joined (MENUS.md §4.2).

  `chain_official`, the chain's first recipe, certified, but one of its CPU seeds that never sees an answer did not.

## 6. Rules 1–16 for the best model

The rules are [one-layer-deeper/README.md §Rules](../one-layer-deeper/README.md#rules). "Gray" means a reasonable
reader could object; the point is flagged, not argued away. This covers
[`wider_menus/submission_wide_anchored.py`](wider_menus/submission_wide_anchored.py). It follows COMBINED.md's own
table, the rules 7, 8 and 14 tables of WIRING.md and MENUS.md, and ANCHORS.md §7. The imports, the initialization and
the absence of backward calls were re-checked in the source.

| rule | status | why |
|---|---|---|
| 1. one `submission.py` exporting one `Submission` | yes | the file is sent as `submission.py`; it has one `SUBMISSION` with model, optimizer, loss, batch sizes and a step cap |
| 2. self-contained | yes | it imports only `torch`, `itertools`, `collections` and the public `benchmark` API; no files, no network |
| 3. the participant defines model, optimizer, loss, batch sizes, steps | yes | Adam; the rounds run until no prompt wants another (adaptive computation, capped at 100) |
| 4. the evaluator owns the loop, data order, backward, clipping, cadence, seeds, deadline | **gray** | the evaluator is untouched: one pass per step, no batch reuse. But the schedule lives in the model as Python state: the reader's stages, the phases, when each table, lesson and switch joins, the Adam resets, and a freeze that sets the learning rate to 0 once every check has passed for 200 steps. That freeze is early stopping on the model's own checks |
| 5. at most 500M parameters | yes | 11,187 |
| 6. no hard-coded weights | yes | every parameter starts at 0.1 × random normal. The "designed" mode, with every table and switch at the truth, is only a plumbing check, never the submission |
| 7. no hard-coded algorithm in the forward pass | **gray, but much less than before** | every product, difference, sum, borrow, count and keep-or-discard decision, and every symbol read or written, is a lookup in a learned table. Every connection is a learned switch, and the menus no longer single out one algorithm: they hold two long divisions, a Montgomery reduction, six groupings of the products and about 4·10^17 machines in all, and different seeds pick different correct algorithms. Still gray: the skeleton is designed (one register, W steps of 9 links, a digit brought in, a chain, a move; one product per digit pair, placed and summed); the menus are written by hand and contain the true wiring; the sizes, the T loop and a + b = a − (0 − b) are fixed. A strict reader can say the algorithm moved into the skeleton and the menus |
| 8. final logits from the model inside autograd, with an unbroken gradient path | yes, with gray edges | at evaluation every digit is a one-hot vector and every table and switch a straight-through one-hot, so the answer is exactly the machine's, and a gradient from the final logits reaches the learned tensors. Gray: (a) in training the forward returns zero logits, so the tables and switches learn through the lessons, not through the final logits; (b) the gradient is damped 10× per subtraction link, addition and round so it does not overflow, and on deep paths it underflows to 0; (c) on the deeper of two gradient tests (T = 8 and OOD-N T = 4 rows), one switch, the final correction, gets exactly 0 on 3 of 4 trained diagnostic models: its menu's never-firing final subtractions are aliases whose gradients cancel (MENUS.md §4.5); (d) which reading hypothesis a prompt uses is decided by designed run-detection logic, with a soft surrogate on the gradient side |
| 9. everything on the GPU | yes | state and computation stay on the device. The schedule reads a few scalars to Python per step, and evaluation syncs one boolean per round to end the loop |
| 10. OOM or timeout fails | ok | neither happened |
| 11. 3,600 s of training, compilation included | yes | 558.0 s, no compilation |
| 12. a custom loss returns one finite differentiable scalar | yes | `token_training_loss` |
| 13. evaluation within 1,800 s; certification | yes | 41.4 s; T = 64 on both ladders |
| 14. no data inspection, augmentation, task-specific solvers, custom loops or participant backward passes | **gray, less than before** | the file calls no `backward` or `autograd.grad`, defines no custom autograd function, and makes up no numbers. No hand-written arithmetic runs at any step of training: the hand-written adder is deleted, and every comparison a lesson makes reads the learned borrow table. But the lessons carry designed knowledge: the anchors' facts and laws (x < N on every top part, the order laws, each digit once per row and column); the two step laws, which are definitions of long division's step; "x mod N = x", "N mod N = 0" and "x + Q·N mod N = x" through the whole division; the times-table ladder (the definition of multiplication) and distributivity for the number rule; the old table lessons, which join late and include "a wrapped x − N is large", an assumption about the register's width; the reading and T's direction, chosen inside the loss by trying every complete hypothesis; and only answers of T ≤ 2 rows whose own checks pass. The switches' pass/fail verdicts come from running the snapped machine on candidate wirings inside the loss: a search over the menu, done by gradient on mixture weights, not by an outer loop. The sizes come from the task's spec |
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

**Nothing is running.** Every study described here has finished and is committed. Since the last refresh of this
file:
- **The anchored wiring model** ([ANCHORED.md](learned_wiring/ANCHORED.md), `b76f2cd`) put the anchors' lessons into
  stage 3 in place of the warm start (step 9).
- **The wide menus** ([MENUS.md](wider_menus/MENUS.md), `c166f8d`) widened stage 3's menus from about 6 million whole
  machines to about 4·10^17 (step 10).
- **The two together** ([WIDE_ANCHORED.md](wider_menus/WIDE_ANCHORED.md), `d8856a8`) made the current best model
  (step 11). It chose exactly the wiring and layout that the wide model's own official run chose.

**Open questions.**
1. **The skeleton and the step laws.** What still singles out long division is the skeleton the switches plug into,
   plus two lessons about a single step. Without the step laws, the general laws alone still pick out the right
   wirings, but only by trying every whole combination of the menu's parts at once. That certified at the narrowest
   menu level on an H100, and stops being affordable beyond about 10^5 combinations (MENUS.md §2.3, §4). Can the
   step laws go at full width, or the skeleton itself be learned?
2. **The late lessons.** The best file still runs the old table lessons 100 steps after the digit laws start, and the
   width-dependent wrap-around fact is among them. Switching them off was tested only in the combined model, where
   4 of 4 seeds certified without them.
3. **Rule 8.** Training never sends the loss through the final logits, and the prediction path's gradient is damped,
   underflows deep in long runs, and misses the final-correction switch on some trained models. Is that "an unbroken
   gradient path"? A menu without the never-firing aliases would remove the last of these.
4. **Rule 14.** Are facts like "x < N on every top part", laws of subtraction and the step laws allowed lessons, or are
   they a "task-specific solver"? A strict reviewer should decide before any real submission.
5. **The hosted evaluator.** Submitting after the deadline, just to see a response, has not been done.

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
| [`wider_menus/`](wider_menus/) | the wide menus ([MENUS.md](wider_menus/MENUS.md)): `submission_wide.py` (step 10); with no warm start ([WIDE_ANCHORED.md](wider_menus/WIDE_ANCHORED.md)): `submission_wide_anchored.py` (step 11, the best model) |
| [`tinyrouting/`](tinyrouting/) | wiring and tables learned together in tiny machines |
| [`tinycore/`](tinycore/) | the quotient-digit chicken-and-egg, computed exactly |
| [`microlab/`](microlab/) | a small harness for one-minute experiments, and its findings ([microlab/README.md](microlab/README.md)) |
| [`unfold_sub/`](unfold_sub/), [`unfold_small/`](unfold_small/) | the UnfoldB and UnfoldS submissions |
| [`submission.py`](submission.py) | SpecBit |
| [`panel/`](panel/), [`panel_synthesis.md`](panel_synthesis.md) | the five-design panel and its synthesis |
| [`FINDINGS_substrate.md`](FINDINGS_substrate.md), [`RELAXED.md`](RELAXED.md), [`UNFOLD.md`](UNFOLD.md), [`UNFOLD_FAMILY.md`](UNFOLD_FAMILY.md) | the day-1 topic write-ups |
| `runs/` | the run records. Official ones: `runs/learned_remainder/*_official.json`, `runs/learned_wiring/*_official.json`, `runs/wider_menus/*_official.json`, `runs/unfold_sub/official_h100.json`, `runs/unfold_small/official_h100.json`, `runs/h100_h1.json`. The ablation's are in `ablation/runs/`, the anchors' in `anchors/records/` |
| `../one-layer-deeper/` | the competition clone at `87e5c26` ("Releasing H1"), with the H1 data under `data/generated/` |
| `~/github/handwired-lab/montgomery-search` | outside this repo: search over a Montgomery squaring program in the handwired project (branch `lab/montgomery-search`) |

**The certified submissions, with their records.** Steps 0–11 are in the §5 table. The earlier three:

| model | file (md5) | official record | training s / evaluation s | learned numbers |
|---|---|---|---|---|
| SpecBit | [`submission.py`](submission.py) (`a432bb5e`; the record embeds no source) | `runs/h100_h1.json` | 16.5 (20 steps) / 26.9 | 65,536 (one logit per possible gate) |
| UnfoldB | [`unfold_sub/submission.py`](unfold_sub/submission.py) (`17299657`) | `runs/unfold_sub/official_h100.json` | 384.3 (1,000 steps, batch 100,000) / 3.7 | 4 |
| UnfoldS | [`unfold_small/submission.py`](unfold_small/submission.py) (`eb6aedfc`) | `runs/unfold_small/official_h100.json` | 1,624.9 (200,000 steps, batch 512) / 3.7 | 4 |

All three reached T = 64 on both ladders, with every rung 768/768 and every split exact.

**How to rerun.** Each write-up ends with its commands. The ladder's official runs go through a Modal launcher
started from the competition clone: `learned_remainder/modal_chain.py --file <submission>` for steps 0–5, and each
later folder's own copy (`ablation/modal_ablate.py`, `anchors/modal_anchors.py`, `learned_wiring/modal_wiring.py`,
`wider_menus/modal_wide.py`). Each builds the competition's image, regenerates H1 with the official generator, and
runs the untouched manifest.

## 9. Where docs and records disagree

What was checked, for every number above:
- each official record's certified T, rung counts, split counts, training and evaluation seconds, parameter count and
  freeze step, against the write-ups that cite them;
- the md5 of the source each record embeds, against the files. Where they differ, the difference is the one `--set`
  line the write-up names;
- the CPU-seed claims, by tallying the records;
- for the best models, the final tables printed in the official logs, against true subtraction and multiplication.

The numbers checked this way all match their records in the write-ups that cite them: each piece's official training
and evaluation times, freeze steps, parameter counts and CPU-seed counts in LEARNED_REMAINDER, CHAIN, SQUARE, CONTROL,
ADDER, IO, COMBINED, ABLATION, ANCHORS, WIRING, ANCHORED, MENUS and WIDE_ANCHORED. The same goes for the two Unfold
notes and REPORT.md §9. The exceptions:
- **COMBINED.md calls `combined_official` and `combined_index_official` "the same training run".** They are two runs
  with identical training: the same file apart from the `PATH` line, the same seed and data order, and both froze at
  step 545 with the same tables. They were launched together, and their training times differ (666.6 s and 566.5 s).
- **The leaderboard count** (3 of 146 certified, REPORT.md §0) has no record behind it and was not re-checked.

The first version of this file also flagged WIRING.md for contradicting itself on stage 1's CPU seeds. The finished
WIRING.md says 7 of 8 throughout, and the record of the failed seed (`div_cpu_seed7`) is in `runs/learned_wiring/`.
