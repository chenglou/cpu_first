# The H1 machine learns its wiring with no hand-written adder

## In plain words

**The task.** The model reads three numbers written as digits in a prompt: a modulus N, a starting number x and a
count T. It must square x, keep only the remainder after dividing by N, repeat that T times, and write the result's
digits. To be certified it must get every one of 768 fresh questions exactly right at T = 1, 2, 4, …, 64, for moduli
it saw in training and for larger moduli it never saw (up to 8 digits). It has 3,600 s of training on one H100 and
1,800 s of evaluation.

**Where this started.** Two certified models each removed a different hand-written piece from the same small
calculator ([COMBINED.md](../../learned_remainder/COMBINED.md)):
- [WIRING.md](WIRING.md)'s `submission_io3.py` learns the calculator's **wiring**: which part feeds which, the layout
  of the squaring and the shape of the reader. But its subtraction table got its start from a **hand-written adder**:
  from the first step until 70 steps after the reader had worked out the prompt, the table was checked against it.
- [ANCHORS.md](../../anchors/ANCHORS.md)'s `submission_combined_anchors.py` needs no hand-written adder at any step: the
  digits' meanings come from the order of the numbers. But its wiring is fixed by hand.

**This file does both.** [`submission_anchored.py`](submission_anchored.py) is the wiring model with the anchors'
lessons in place of the warm start. The hand-written adder is deleted from the file. No hand-written arithmetic runs
anywhere, in training or in the answer.

**What the machine learns.** All 7,037 numbers start random and are learned in one training run:

| part | what it learns |
|---|---|
| subtraction table (200 rows) | every subtraction, and every addition too, as a + b = a − (0 − b) |
| times table (100 cells) | the digit products of the squaring |
| keep-or-discard gate (2 rows) | whether a subtraction's result is kept, in the division and in the count-down |
| the division's 7 switches | which end of the square the division starts from, how the remainder makes room for the next digit, what each subtraction subtracts from what, what it writes back, whether N's digits are reversed, which places hold the answer |
| the squaring's layout (two switches of 24 rules) | in which column each digit product goes |
| the reader's and writer's shape (one switch of 3 shapes) and their pointers | where each number's digits sit in the prompt and which way they run; which symbols are digits and what each is worth; where and how to write the answer |

**How the subtraction table now learns what the digits mean.**
1. **It waits for real numbers.** Nothing teaches the table until the reader has worked out the prompt. Before that
   the reader can read nonsense. On one seed below it read every N as 0 for its first 90 steps; a table taught from
   that learns only "minus 0", which is how stage 1's first version failed a seed in WIRING.md.
2. **Borrows from "x is smaller than N".** In every question x < N. So x − N goes below zero (it borrows) and N − x
   does not. The same holds for the numbers formed by x's and N's digits from any place upward. That teaches the
   table's borrows.
3. **Digits from two laws of subtraction.** Once those borrow facts have held on every question of 10 batches in a
   row, the learned borrows referee two laws. Taking the same digit away from two digits keeps their order. Taking
   more away from the same digit leaves less. The exception is when exactly one of the two goes below zero: that one
   wraps round to the top. Ten different digits in a known order can only be 0 to 9.
4. **The old self-checks join late.** 100 steps after that, the old checks start: subtract N, add it back with the
   same table, and see whether the number comes back. By then the table is already exact.

Everything else is the wiring model's training, unchanged. Each switch waits until the tables it uses are exact, then
is found in a few steps.

**What is still written by hand.** The complete list:
- **The menus.** Each switch chooses among options written by hand, and the right wiring is one of them:
  - the division: 7 switches, 864 wirings (times the gate's 4 tables);
  - the squaring: a placement rule for the low digits and one for the high digits, each one of 24 (576 layouts);
  - the reader and writer: one shape of 3, shared by the four fields;
  - the reading: which of the prompt's three digit runs is N, which is x and which way each runs (24 hypotheses).
- **The sizes.** 8 division steps of up to 9 subtractions each, on a 9-place register; 8 squaring rows of 18 columns;
  8, 8 and 2 digits read and 8 written; a 2-digit count-down.
- **The loop.** Count T down by subtracting the number 1 with the table. A round squares and divides when the gate
  says to. Rounds run until no question wants another, at most 100.
- **The adder's wiring**: a + b is computed as a − (0 − b), both with the subtraction table.
- **Small fixed plumbing**:
  - a subtraction runs from the lowest place up, and the gate reads its final borrow;
  - the squaring's rows are added in order;
  - a slot reads 0 once a slot before it has met a non-digit;
  - a digit's value is its rank among the symbols believed to be digits.
- **The lessons and their schedule**: what each table and switch is checked against, which switches are weighed
  together, what waits for what, and when training stops. These include:
  - the anchors' three statements: "x < N on every top part", "subtraction keeps, reverses or wraps the order", and
    "every column and row of the table uses each digit once";
  - every training answer's position and symbols, which teach the writer where to write and with which symbols (a
    round trip through the reader's and the writer's codes; the machine is not involved);
  - the answers of the T ≤ 2 questions (2% of training; picked by comparing T's read digit with 1 and 2), which the
    machine must reproduce: they help pick the division's wiring;
  - the answers of the T = 16 questions, which settle which way T is read.

The true tables and the true wiring still appear in the file in two places only: the plumbing check
(`MODE = "designed"`, every part set to the truth) and the reports printed at evaluation. Neither is used by training
or by the answer.

**Did it certify?** Yes. The run used the untouched official H1 manifest on an H100 (`anchored_official`, the file
byte for byte):
- it got all 768 questions right at every T up to 64, on seen and unseen moduli, and every split exactly right;
- every learned part came out exact: the subtraction and times tables, the gate, the division's wiring, the squaring's
  layout and the reader's shape;
- the coverage audit found no table entry used at evaluation that training never used.

The same file certified on 8 of 8 CPU seeds through the same evaluator, every learned part exact on each.

**Times.** On the H100, training took 466 s of its 3,600 s budget (frozen at step 522) and evaluation 50 s of 1,800 s.
Stage 3's official run took 729 s and 52 s. Most of that difference is which way T's reading happened to start, not
the anchors (§2.4). On the CPU seeds training took 600–1,111 s and evaluation 387–596 s; those are CPU times, not
the budget's H100.

**What it took.** No change to the wiring model's schedule; the anchors' lessons dropped in as they are.
- **The table waits for real numbers, and gets them.** Before the reading, the reader does read nonsense: on every
  seed checked it read N as 0 on most questions at some step, and on one seed for 90 steps. The table learned nothing
  from it, because it had no lesson yet. By the time its lessons start, the reader reads every question right; on all
  8 seeds checked this held from 8–14 steps before.
- **The switches wait for exact tables, as before.** The table was exact 66–122 steps after the reading. Everything
  downstream (the times table, the division's switches, the lesson that settles which way T is read) then started
  within a few dozen steps of when it started in stage 3.
- **The wait for the reading is a margin.** In an ablation the table's lessons start at step 1, on nonsense. The
  borrows idle until the reader is right, the digit lessons' gate stays shut, and both seeds tried still train to the
  end with every rung exact.
- **One side effect costs time.** When T's reading started backwards (4 of the 8 CPU seeds), T's own lesson,
  unchanged from stage 3, took longer to turn it round: 108–239 steps against 69–93 in stage 3. Every run still
  certified; the slowest CPU seed trained for 1,111 s on CPU. Part of the cause: a reset of the optimizer's running
  averages, which stage 3 made at or soon after the reading, now comes after the table is exact, in the middle of
  T's drift, and lets T sink deeper into the backwards reading. The rest is not settled (§3.4).

## Details

### 1. What changed in the file

`submission_anchored.py` is `submission_io3.py` (WIRING.md's stage 3, md5 `c2331362…`) with these changes and no
others (about 200 changed lines of 1,800):
- **Removed**: the designed adder (`add_hard`, `add_soft`) and `ADDER_WARM`. Every number computed in training and in
  the answer now comes from the learned tables.
- **Added**, from `anchors/submission_combined_anchors.py` without change: `GATE_WIN = 10`, `LATE_AFTER = 100`,
  `order_pairs`, the top-part facts, the order laws and bijectivity, and the gate between them (`Control.facts_streak`,
  `Control.gated_on`).
- **The table's lessons** (`lessons()`), against stage 3's:

  | from | stage 3 (`submission_io3.py`) | this file |
  |---|---|---|
  | step 1 | the hand-written adder's check on every link (x and the square's windows minus N again and again), R − 0 = R, R − R = 0 | nothing |
  | the reading is chosen (I/O stage 4) | + x − N borrows, N − x does not, (x − N) − N does not | x < N on every top part |
  | 70 steps after the reading | the check with the learned adder, + 0 + N = N, R + N = N + R, 0 − N borrows | – |
  | the borrow facts hold on every row of 10 batches in a row | – | + the order laws and bijectivity |
  | 100 steps after that | – | + all of stage 3's table lessons after its warm start |

- **The table's hard checks**, which the phases, the ladder and the switches wait for, are stage 3's checks after its
  warm start, from step 1: the learned-adder check on every link, R − 0 = R, R − R = 0, 0 + N = N, R + N = N + R, and
  from the reading the facts. A random table fails them. Phase B waits for them to pass on 80% of the links, the
  times table's ladder for 20 exact steps in a row, and the switches for 20 more with the ladder exact too. Phase B
  and the ladder also have stage 3's 300-step fallbacks, which no run here needed (§3.3).
- The anchors' combined file computes the old lessons with weight 0 until they join; here they are skipped until then
  (the same gradient, less work).
- The log: `order_digits` and `pieces` join the schedule's history, and the facts' streak joins the log line.

Everything else is stage 3's, line for line: the datapath and the prediction path, the reader's and writer's lessons,
the times table's ladder, the switches' lessons, T's lesson, the rules of the phases and the freeze.

### 2. Results

#### 2.1 The official run

The official manifest `benchmark/manifests/h100_hard_h1.json`, untouched (seed 74, 3,600 s training and 1,800 s
evaluation budgets), on a Modal H100, with the file sent byte for byte through WIRING.md's runner `modal_wiring.py`.
The record holds RESULT_JSON, the full log, the exact source and the manifest.

| run | what ran | max T (seen / unseen moduli) | rungs | test, ood_t, ood_n_t | learned parts | training s (frozen at step) | evaluation s |
|---|---|---|---|---|---|---|---|
| **`anchored_official`** | **`submission_anchored.py` byte for byte** (md5 `6328a7d239ec2fe08de2e78ad8f28ed9`) | **64 / 64** | 768/768 at every rung of both ladders | 1.0, 1.0, 1.0 | subtraction 200/200, times 100/100, the gate commits with no borrow (p 0.9976) and holds on one (p 0.0020); the true division wiring; the true layout (288/288 placements); the step shape; the true reading, code and writer | 466.1 (522) | 50.3 |
| for comparison: `io3v2_official` | `submission_io3.py` (WIRING.md: the hand-written adder for 70 steps after the reading) | 64 / 64 | all 768/768 | all exact | all exact and true | 728.6 (532) | 51.8 |
| for comparison: `cmbanchors_official` | `anchors/submission_combined_anchors.py` (ANCHORS.md: no hand-written adder, wiring fixed by hand) | 64 / 64 | all 768/768 | all exact | all exact | 454.0 (553) | 40.6 |

Its schedule (the `LR` lines): the reading chosen at step 123 (the top-part facts start), the digit lessons at 160,
phase B at 203, the times table's ladder at 223, the old table lessons at 260, the chain's switches at 298, the
division's I/O switches at 320, T's direction settled at 343, frozen at 522. It has 7,037 learned numbers, the same as
stage 3.

#### 2.2 Every run through the evaluator

The official run and 8 CPU seeds: the same evaluator with the manifest's device set to cpu and the seed changed
(`anchored/plans/anchored_cpu.json`), the file byte for byte. Stage 3's nine runs below them for comparison. Event
columns give the step; "never trained" is the largest count of table rows, cells or gate rows that evaluation used but
training never did (§2.3).

<!-- RUNS -->
| run | seed | max T seen / unseen | rungs (fewest right of 768, seen / unseen) | test, ood_t, ood_n_t | subtraction, product, gate | division wiring | layout | shape | reading chosen | digit lessons | phase B | ladder | pieces' lessons | chain lesson | I/O lesson | T settled | frozen | never trained | training s | evaluation s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anchored_official (H100) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 123 | 160 | 203 | 223 | 260 | 298 | 320 | 343 | 522 | 0 | 466.1 | 50.3 |
| anchored_cpu_seed74 (CPU) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 123 | 160 | 203 | 223 | 260 | 298 | 320 | 343 | 522 | 0 | 600.0 | 447.4 |
| anchored_cpu_seed1 (CPU) | 1 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 109 | 149 | 193 | 213 | 249 | 292 | 315 | 423 | 517 | 0 | 722.1 | 401.0 |
| anchored_cpu_seed2 (CPU) | 2 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 108 | 150 | 193 | 213 | 250 | 283 | 306 | 426 | 508 | 0 | 802.7 | 490.6 |
| anchored_cpu_seed3 (CPU) | 3 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 115 | 156 | 199 | 219 | 256 | 290 | 312 | 335 | 514 | 0 | 871.3 | 596.0 |
| anchored_cpu_seed4 (CPU) | 4 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 114 | 200 | 241 | 261 | 300 | 334 | 356 | 379 | 558 | 0 | 649.5 | 491.6 |
| anchored_cpu_seed5 (CPU) | 5 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 118 | 167 | 209 | 229 | 267 | 301 | 322 | 561 | 740 | 0 | 1110.9 | 386.6 |
| anchored_cpu_seed6 (CPU) | 6 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 175 | 212 | 253 | 273 | 312 | 351 | 373 | 496 | 576 | 0 | 796.9 | 408.2 |
| anchored_cpu_seed7 (CPU) | 7 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 96 | 162 | 205 | 225 | 262 | 296 | 319 | 342 | 521 | 0 | 611.0 | 410.5 |
| io3v2_official (H100) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 123 | - | 158 | 226 | - | 306 | 329 | 420 | 532 | 0 | 728.6 | 51.8 |
| io3v2_cpu_seed74 (CPU) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 123 | - | 158 | 226 | - | 306 | 329 | 420 | 532 | 0 | 904.9 | 472.3 |
| io3v2_cpu_seed1 (CPU) | 1 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 109 | - | 109 | 227 | - | 312 | 335 | 404 | 537 | 0 | 708.4 | 379.9 |
| io3v2_cpu_seed2 (CPU) | 2 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 108 | - | 147 | 235 | - | 310 | 334 | 427 | 536 | 0 | 1060.0 | 543.3 |
| io3v2_cpu_seed3 (CPU) | 3 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 115 | - | 115 | 224 | - | 298 | 321 | 345 | 524 | 0 | 770.4 | 444.7 |
| io3v2_cpu_seed4 (CPU) | 4 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 114 | - | 168 | 188 | - | 267 | 289 | 312 | 491 | 0 | 579.5 | 415.1 |
| io3v2_cpu_seed5 (CPU) | 5 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 118 | - | 118 | 203 | - | 280 | 301 | 386 | 503 | 0 | 805.2 | 459.0 |
| io3v2_cpu_seed6 (CPU) | 6 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 175 | - | 175 | 232 | - | 317 | 339 | 420 | 543 | 0 | 647.9 | 398.3 |
| io3v2_cpu_seed7 (CPU) | 7 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 96 | - | 96 | 184 | - | 262 | 286 | 310 | 489 | 0 | 898.7 | 558.6 |
<!-- /RUNS -->

#### 2.3 The coverage audit

At its first evaluation batch the file prints how often training's hard checks used each table row and cell (the
`TABLE ... train_visits` lines), and at every rung which rows evaluation used and how many of them training never used
(the `AUDIT` lines, CHAIN.md's coverage audit). As in every earlier file, the training counts are the hard checks'
lookups over the whole of training. Here those checks run from step 1, before the table's first lesson. With every
table entry equal to the truth this is a formality, but it is the same audit as before.

<!-- AUDIT -->
| run | subtraction rows used at evaluation | fewest training visits of a row | product cells used at evaluation | fewest training visits of a cell | count-down rows used | gate rows used | used at evaluation but never trained (rows, cells, count-down, gate) |
|---|---|---|---|---|---|---|---|
| anchored_official | 200/200 | 963413 | 100/100 | 184162 | 24 | 2/2 | 0, 0, 0, 0 |
| anchored_cpu_seed74 | 200/200 | 963413 | 100/100 | 184162 | 24 | 2/2 | 0, 0, 0, 0 |
| anchored_cpu_seed1 | 200/200 | 954251 | 100/100 | 185539 | 24 | 2/2 | 0, 0, 0, 0 |
| anchored_cpu_seed2 | 200/200 | 1195344 | 100/100 | 184902 | 24 | 2/2 | 0, 0, 0, 0 |
| anchored_cpu_seed3 | 200/200 | 991967 | 100/100 | 184302 | 24 | 2/2 | 0, 0, 0, 0 |
| anchored_cpu_seed4 | 200/200 | 1398515 | 100/100 | 184270 | 24 | 2/2 | 0, 0, 0, 0 |
| anchored_cpu_seed5 | 200/200 | 1212024 | 100/100 | 346506 | 24 | 2/2 | 0, 0, 0, 0 |
| anchored_cpu_seed6 | 200/200 | 819466 | 100/100 | 185262 | 24 | 2/2 | 0, 0, 0, 0 |
| anchored_cpu_seed7 | 200/200 | 871181 | 100/100 | 185522 | 24 | 2/2 | 0, 0, 0, 0 |
<!-- /AUDIT -->

#### 2.4 Time

Seconds to a training step, from the evaluator's progress lines of the three official runs:

<!-- TIMES -->
| run | frozen at step | T settled at step | s to step 100 | s to step 200 | s to step 300 | s to step 400 | s to step 500 | s to step 600 | s to step 1000 | training s | evaluation s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anchored_official | 522 | 343 | 17.0 | 35.7 | 62.0 | 124.9 | 198.1 | 217.8 | 224.9 | 466.1 | 50.3 |
| io3v2_official | 532 | 420 | 25.8 | 46.1 | 76.6 | 334.9 | 484.6 | 507.6 | 512.3 | 728.6 | 51.8 |
| cmbanchors_official | 553 | 449 | 10.3 | 21.3 | 36.7 | 125.7 | 271.9 | 294.1 | 297.4 | 454.0 | 40.6 |
<!-- /TIMES -->

- **Most of the 262 s saved against stage 3 is T's direction, not the anchors.** In `io3v2_official` T's reading
  started backwards (a two-digit T read as 61) and settled 91 steps after the I/O lesson joined. T's lesson runs whole
  count-downs, about 4.5 s a step on the H100 (WIRING.md §6), so steps 300–400 took 258 s. Here T started the right
  way and settled 23 steps after the I/O lesson joined, 20 of them the lesson's wait. Which way T starts is settled by
  chance before T's lesson (WIRING.md §4.4); it started backwards in 5 of stage 3's 8 CPU seeds.
- **The anchors are a little cheaper before the table is exact**: step 300 at 62 s against 77 s, since nothing
  checks the table against an adder before the reading, and the old lessons are skipped until they join.
- **After the freeze** the remaining ~19,500 steps take 160–250 s in each run.
- **Over the CPU seeds** the two files train in about the same time: 600–1,111 s here and 580–1,060 s for stage 3.
  Here T started backwards on seeds 1, 2, 5 and 6, and in stage 3 on 74, 1, 2, 5 and 6. When it did, its lesson took
  longer here, and the slowest run here (seed 5, 1,111 s) is that (§3.4).

### 3. The interactions

Two interactions needed watching:
- the switches learn only once the tables they use are exact;
- an untrained reader can feed the table's lessons nonsense, as when it read every N as 0.

Both were watched with training diagnostics: `anchored/local_anchored.py`, WIRING.md's local harness with a few
additions, all outside the submission:
- it checks the snapped reader against each row's own N, x and T, which the data files give, on the batch the step's
  lessons see, before they run;
- it counts the table's right borrows and right digits against true subtraction;
- from plan `a3` on, it logs gradient norms (§3.4).

It logs a line every 5 steps and, at the end, runs WIRING.md's checks: every depth rung on both prediction paths and
the gradient test. The harness draws its own batches, so its runs follow their own path; they are not the evaluator's
runs with the same seed. Seeds 74, 1, 2 and 4 are WIRING.md's diagnostic seeds (its `d13` is stage 3 on them in the
same harness); 3, 5, 6 and 7 complete the eight.

<!-- DIAG -->
| run | most rows read as N = 0 before the reading (logged steps with 90% or more) | N and x read right on every row from | reading chosen: the table's first lesson | most rows misread after it | borrows 200/200 | digit lessons | digits 200/200 | the table's checks all pass from (logged steps failing after) | phase B | ladder | pieces' lessons | chain lesson | I/O lesson | T settled | frozen | rungs exact on both paths, identical logits | gradient test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a1_s74 | 1.00 (1) | 115 | 125 | 0.0000 | 155 | 161 | 190 | 195 (0) | 204 | 224 | 261 | 298 | 320 | 436 | 522 | 14/14 | True |
| a1_s1 | 1.00 (1) | 130 | 138 | 0.0000 | 170 | 178 | 210 | 210 (0) | 221 | 241 | 278 | 320 | 343 | 456 | 545 | 14/14 | True |
| a1_s2 | 1.00 (1) | 95 | 109 | 0.0000 | 150 | 156 | 185 | 185 (0) | 198 | 218 | 256 | 288 | 311 | 424 | 513 | 14/14 | True |
| a1_s4 | 1.00 (19) | 100 | 108 | 0.0000 | 190 | 197 | 225 | 230 (0) | 238 | 258 | 297 | 332 | 354 | 377 | 556 | 14/14 | True |
| a2_s3 | 1.00 (1) | 95 | 108 | 0.0000 | 140 | 149 | 180 | 185 (0) | 193 | 213 | 249 | 285 | 307 | 330 | 509 | 14/14 | True |
| a2_s5 | 1.00 (2) | 110 | 120 | 0.0000 | 150 | 158 | 190 | 190 (0) | 201 | 221 | 258 | 293 | 314 | 533 | 712 | 14/14 | True |
| a2_s6 | 1.00 (1) | 135 | 149 | 0.0000 | 175 | 183 | 215 | 215 (0) | 225 | 245 | 283 | 324 | 346 | 465 | 549 | 14/14 | True |
| a2_s7 | 1.00 (3) | 65 | 76 | 0.0000 | 135 | 139 | 170 | 170 (0) | 183 | 203 | 239 | 274 | 297 | 320 | 499 | 14/14 | True |
<!-- /DIAG -->

#### 3.1 The reader before the reading

The untrained reader does read nonsense:
- on seed 4 it read every N as 0 (93% of rows at step 1, then 100%) on every logged step up to step 90. That is the
  fault that broke stage 1's first version on CPU seed 7 (WIRING.md §3.4);
- on every other seed it read N as 0 on 90% of rows or more for a logged step or two:
  - seeds 74, 1, 2, 3, 5 and 7 just after the reading lesson joined (I/O stage 3), while the pointers reorganized;
  - seeds 6 and 7 also at step 1, from the random start;
- between those episodes the share of rows read as N = 0 was typically 6–38%, the median before the reading lesson.

In stage 3 the hand-written adder taught the table from step 1 on whatever the reader read. On seed 4 that table sat
at 27/200 right from step 25 to step 75 (`d13_s4`): its check only ever subtracted 0. Stage 3 survived because its
warm start ran 70 steps past the reading.

Here the table has no lesson before the reading, so the nonsense reaches nothing. By the time the reading is chosen
the reader is right. On all 8 seeds the snapped reader read N and x right on every row of every logged step from 8–14
steps before the reading was chosen, and it never misread a row after that. (Columns: "N and x read right on every
row from" and "most rows misread after it".)

#### 3.2 Borrows, then digits

- **The borrows** were all right 26–82 steps after the reading. Seed 4 was the slowest: its borrows first fell from
  110 right to about 40, sat there for about 40 steps, then rose to 185 within about 15 steps.
- **The digit lessons** opened once the borrow facts had held on every row of 10 batches. The digits were all right
  28–32 steps later on every seed.
- **The table's hard checks** all passed from steps 170–230 and never failed again on any logged step.
- **Against stage 3**: on seeds 74, 1, 2 and 4 the whole table was right at steps 190, 210, 185 and 225; stage 3's
  was at 225, 200, 225 and 175 on the same seeds in the same harness (logged every 25 steps there, every 5 here).

#### 3.3 The switches still wait for exact tables

Nothing downstream changed its rule:
- phase B (a fresh optimizer) waits for the table's checks to pass on 80% of the links;
- the times table's ladder waits for 20 steps with every check exact;
- the division's chain switches wait for 20 more with the ladder exact too, and its I/O switches for 20 steps of the
  chain passing the division ladder.

In every run the ladder joined on its 20-step streak and phase B on the 80% mark; neither 300-step fallback fired.
Once the table is exact, a few dozen steps later or earlier than in stage 3, everything after it shifts by about as
much:
- on 7 of the evaluator's 8 CPU seeds the freeze came 28 steps earlier to 67 steps later than stage 3's on the same
  seed; on seed 5 it came 237 steps later, all of it T's lesson (§3.4);
- in the diagnostics, 25 steps earlier to 72 later than WIRING.md's `d13`.

The switches were as quick as in stage 3. The I/O lesson joined 21–23 steps after the chain lesson in every run, 20 of
them its waiting window, so the chain's switches passed the division ladder within 1–3 steps (WIRING.md: 1–4). The
I/O switches were right by the next logged step.

Two things now happen in a different order:
- **phase B's fresh optimizer** comes after the table is exact: steps 183–238 in the diagnostics and 203 in the
  official run. In stage 3 it came at or soon after the reading;
- **the pieces' old table lessons** (the learned-adder check and the rest) join on a table whose checks have passed for
  about 70 steps.

The second changed nothing visible: in no diagnostic run did a check fail on a logged step after the table first
passed them all. The first has a side effect on T's reading direction (§3.4).

#### 3.4 T's direction and the moved fresh optimizer

T's lesson is stage 3's, unchanged (WIRING.md §4.4). It starts once the whole machine has been exact for 20 steps
after the I/O lesson joined, and settles which way T is read. When T's reading had started the right way it settled
23–24 steps after the I/O lesson in every run of both files. When it had started backwards it took longer here than
in stage 3. On the evaluator's CPU seeds:

<!-- TCPU -->
| seed | this file: reading, phase B, I/O lesson, T settled | steps from the I/O lesson to T settled | stage 3: reading, phase B, I/O lesson, T settled | steps |
|---|---|---|---|---|
| 74 | 123, 203, 320, 343 | 23 (T started the right way) | 123, 158, 329, 420 | 91 |
| 1 | 109, 193, 315, 423 | 108 | 109, 109, 335, 404 | 69 |
| 2 | 108, 193, 306, 426 | 120 | 108, 147, 334, 427 | 93 |
| 3 | 115, 199, 312, 335 | 23 (T started the right way) | 115, 115, 321, 345 | 24 (T started the right way) |
| 4 | 114, 241, 356, 379 | 23 (T started the right way) | 114, 168, 289, 312 | 23 (T started the right way) |
| 5 | 118, 209, 322, 561 | 239 | 118, 118, 301, 386 | 85 |
| 6 | 175, 253, 373, 496 | 123 | 175, 175, 339, 420 | 81 |
| 7 | 96, 205, 319, 342 | 23 (T started the right way) | 96, 96, 286, 310 | 24 (T started the right way) |
<!-- /TCPU -->

T started backwards on 4 of 8 seeds here (1, 2, 5 and 6) and on 5 of 8 in stage 3 (74, 1, 2, 5 and 6). Where it
did, its lesson ran 108–239 steps here against 69–93 in stage 3. The diagnostics agree: where T started backwards it
settled 113–219 steps after the I/O lesson here (seeds 74, 1, 2, 5 and 6) and 76–95 in stage 3 (`d13` seeds 74, 1
and 2, and `d3_s5`).

It costs time, not correctness. Every run certified. On an H100, T's lesson costs about 4.5 s a step (WIRING.md §6),
so the slowest case, 239 steps, is about 1,100 s of the 3,600 s budget; CPU seed 5 trained 1,111 s on CPU.

**What is known about why.** Until T's lesson starts, the reading lessons count T's two directions together, so T's
pointer drifts toward whichever direction it already favours, to near certainty (WIRING.md §4.4). The drift starts
when the reading is chosen. Phase B clears Adam's state. In stage 3 phase B came at the reading, just before the
drift's first, large gradients. Here it comes once the table is exact, about 80 steps into the drift. From plan `a3`
on, the diagnostic harness also logs the gradient on T's pointer every 5 steps (the norm over T's anchor and step
logits; the logs also hold the whole gradient norm before clipping). The runs: `d3_s5` and `d13_s74` are stage 3;
`a3_s5` and `a1_s74` are this file; `h1_s5` and `k1_*` are the two variants below. `a1_s74` and `d13_s74` predate the
gradient logging.

<!-- TLESSON -->
| run | reading chosen | phase B: Adam state cleared | T's pointer gradient 5 steps before | 20 steps after | when T's lesson can start | I/O lesson | T settled | steps from the I/O lesson to T settled |
|---|---|---|---|---|---|---|---|---|
| d3_s5 | 120 | 120 | 6.6e-07 | 9.4e-01 | 1.9e-02 | 298 | 393 | 95 |
| a3_s5 | 120 | 201 | 2.2e-02 | 4.2e-03 | 3.8e-04 | 314 | 533 | 219 |
| h1_s5 | 120 | 201 | 2.2e-02 | 4.2e-03 | 3.8e-04 | 314 | 533 | 219 |
| k1_s5 | 120 | 201 | 2.2e-02 | 1.7e-02 | 8.7e-03 | 315 | 585 | 270 |
| d13_s74 | 125 | 159 | – | – | – | 331 | 422 | 91 |
| a1_s74 | 125 | 204 | – | – | – | 320 | 436 | 116 |
| k1_s74 | 125 | 204 | 1.7e-02 | 1.3e-02 | 6.7e-03 | 313 | 379 | 66 |
<!-- /TLESSON -->

- **Not the anchors' lessons.** `variants/submission_anchored_handover.py` switches the anchors' own lessons (the
  top-part facts, the order laws, bijectivity) off once the old lessons join. `h1_s5` then matches `a3_s5` event for
  event.
- **The moved reset lets T's pointer saturate deeper.** On seed 5 the gradient on T's pointer fell from 2.2e-2 to
  4.2e-3 in the 20 steps after the reset, and was 3.8e-4 by the time T's lesson could start, against 1.9e-2 in stage
  3. `variants/submission_anchored_keepadam.py` clears only the subtraction table's Adam state at phase B. It keeps
  the gradient about 20 times larger (8.7e-3 on seed 5, 6.7e-3 on seed 74, at the same point).
- **But that is not the whole story.** With the keep-Adam variant, seed 74's T settled 66 steps after the I/O lesson
  (the file: 116; stage 3: 91), and seed 5's 270 (the file: 219).
- **What the slow turns share.** For most of each slow turn T's lesson barely moved the step shape's pointer. Its
  gradient stayed small and kept falling (seed 5: from 1.0e-2 to 3.5e-3 over 240 steps with keep-Adam), then jumped
  by two orders of magnitude or more when the reading flipped. A guess, not tested here: the lesson's verdict is
  normalized over T's two directions. The two untrained shapes' own T pointers, which have not chosen a direction, can
  raise it instead of the step shape's pointer, WIRING.md §4.4's "third option that has not decided" in a new place.

The cause is not settled. It changes training time, not what is learned, and the file is certified as it stands.

#### 3.5 Does the table need to wait for the reading?

The ablation `variants/submission_anchored_facts1.py` changes one line, `facts = True`: the top-part facts start at
step 1, on whatever the reader reads. It ran on seed 4 (every N read as 0 up to step 90) and on seed 74:

<!-- ABLATION -->
| run | most rows read as N = 0 before the reading (logged steps with 90% or more) | N and x read right on every row from | reading chosen: the table's first lesson | most rows misread after it | borrows 200/200 | digit lessons | digits 200/200 | the table's checks all pass from (logged steps failing after) | phase B | ladder | pieces' lessons | chain lesson | I/O lesson | T settled | frozen | rungs exact on both paths, identical logits | gradient test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f1_s4 | 1.00 (19) | 100 | 108 | 0.0000 | 140 | 146 | 180 | 180 (0) | 190 | 210 | 246 | 283 | 305 | 328 | 507 | 14/14 | True |
| f1_s74 | 1.00 (1) | 115 | 125 | 0.0000 | 185 | 188 | 215 | 220 (0) | 230 | 250 | 288 | 325 | 347 | 464 | 549 | 14/14 | True |
<!-- /ABLATION -->

In this table the column "reading chosen" is only the reading: the ablation's table lessons start at step 1.

- While the reader read nonsense the borrows stayed near their random start: 89–109 of 200 right on seed 4, 94–108 on
  seed 74.
- They were all right 40–70 steps after the reader read every row right.
- The digit lessons did not start earlier. Their gate needs the borrow facts on every row of 10 batches, and on
  these two seeds the nonsense numbers never satisfied them.
- Both runs froze (507 and 549, against 556 and 522 with the wait), with every rung exact on both paths and the
  gradient test passing.

So in this model the wait for the reading is a margin, not a necessity: the anchors' gate already keeps the nonsense
out of the digits. The old warm start had no such gate; it ended at a fixed step whether or not the table had learned
anything. The file keeps the wait, as the certified anchors' combined model does.

#### 3.6 Schedule changes, and why

None beyond the anchors' own:
- the table's first lesson waits for the reading, as in `submission_combined_anchors.py`;
- the digits wait for the borrows (`GATE_WIN`);
- the old lessons wait `LATE_AFTER` steps more.

The wiring's own schedule needed no change, for three reasons:
- the reader is right on every row before the table's first lesson (§3.1);
- the gate keeps the digit lessons off nonsense (§3.5);
- every switch already waits for the tables it uses to be exact, so a table that is exact a few dozen steps later or
  earlier only shifts the schedule (§3.3).

No run failed, so no small reproduction was needed. Two surprises resolved themselves every time they appeared:
seed 4's 40-step plateau in the borrows (§3.2), and T's slower turn when it started backwards (§3.4).

### 4. The prediction path and the gradient test

The datapath and the prediction path are stage 3's, line for line (WIRING.md §5). On the prediction path every table
and switch is a straight-through one-hot, so the forward pass is exactly the snapped machine, and the final logits'
gradient reaches every learned tensor.
- **At the truth** (`MODE = "designed"`, local, WIRING.md's harness):
  - both prediction paths got all 16 prompts of each of the 14 depth rungs right, with identical logits;
  - the gradient test passes: all 21 parameter tensors get a finite gradient that is not all zero.
- **Trained**: all 15 diagnostic runs of §3 froze: 9 of this file, 5 of its variants and 1 of stage 3. On every
  one:
  - both paths got all 128 prompts of each of the 14 rungs right, with identical logits;
  - the gradient test passes on the trained model.

### 5. Rules 7, 8 and 14

The rules are [one-layer-deeper/README.md](../../../one-layer-deeper/README.md) §Rules. WIRING.md's table covers the
three the wiring touches and ANCHORS.md §7 the anchors' changes. Here, against WIRING.md's stage 3:

| rule | stage 3 (WIRING.md) | this file | still gray |
|---|---|---|---|
| **7. no hard-coded algorithm in the forward pass** | less gray than the combined model: the wiring is chosen by learned switches from designed menus | **the same**: the forward pass is unchanged | the menus contain the true wiring; the sizes, the T loop, a + b = a − (0 − b) and the order of the squaring's summation are fixed |
| **8. logits inside autograd, unbroken gradient path** | yes, with gray edges (straight-through tables and switches, a discounted gradient) | **the same**: the prediction path is unchanged, and the gradient test passes on the trained models (§4) | the same edges |
| **14. no data inspection, task-specific solvers, custom loops, participant backward** | gray: the lessons carry designed knowledge, and for the first 70 steps after the reading a hand-written adder checked the table | **less gray**: no hand-written adder at any step. Every comparison a table lesson makes is refereed by the learned borrow table. The new statements are laws of subtraction and the fact x < N, which the prompts guarantee and the earlier files already used | the lessons are still designed knowledge: "x < N on every top part" (the top parts compare digit strings, not values), "subtraction keeps, reverses or wraps the order", "every column and row uses each digit once", and stage 3's division ladder, I/O laws and routed ladder |

### 6. Reproduce

Everything heavy runs on Modal. Launch from the one-layer-deeper repository root. The official run and the CPU seeds
go through WIRING.md's unchanged runner `modal_wiring.py` (app and Volume `learned-wiring`, folder `records/`), with
the file passed by name; `wait_fetch.py` downloads the records to `cpu_first/history/runs/learned_wiring/<name>.json`.

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/history/learned_wiring/modal_wiring.py --mode official --name anchored_official --file submission_anchored.py
```

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/history/learned_wiring/modal_wiring.py --mode plan --plan ../cpu_first/history/learned_wiring/anchored/plans/anchored_cpu.json --file submission_anchored.py
```

```bash
cd ~/github/one-layer/cpu_first/history/learned_wiring && ~/github/sotaku/venv/bin/python wait_fetch.py anchored_official anchored_cpu_seed74 anchored_cpu_seed1 anchored_cpu_seed2 anchored_cpu_seed3 anchored_cpu_seed4 anchored_cpu_seed5 anchored_cpu_seed6 anchored_cpu_seed7
```

```bash
cd ~/github/one-layer/cpu_first/history/learned_wiring/anchored && ~/github/sotaku/venv/bin/python summarize_anchored.py runs anchored io3v2
```

The training diagnostics run on Modal CPU containers through `anchored/modal_diag_anchored.py` (app and Volume
`learned-wiring-anchored`, at most 8 containers; logs to `anchored/logs/`). They use `anchored/local_anchored.py`:
local_wiring.py's training loop, plus a line every 5 steps with the reader checked against the rows' own numbers, the
table's borrows and digits, and from plan `a3` on the gradient norms. The plans:
- `a1`, `a2`: the file on seeds 74, 1, 2, 4 and 3, 5, 6, 7, and the ablation `facts1` on seeds 4 and 74;
- `a3`: seed 5 for the file, the `handover` variant and stage 3, with gradient norms;
- `a4`: the `keepadam` variant on seeds 5 and 74.

The commands below run plan `a1`, then refill this write-up's generated tables:

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/history/learned_wiring/anchored/modal_diag_anchored.py --plan ../cpu_first/history/learned_wiring/anchored/plans/a1.json
```

```bash
cd ~/github/one-layer/cpu_first/history/learned_wiring/anchored && ~/github/sotaku/venv/bin/python summarize_anchored.py fill ../ANCHORED.md
```

Local checks, each under a minute and well under 1 GB (WIRING.md's harness; the evaluator stub stands in for the API):

```bash
cd ~/github/one-layer/cpu_first/history/learned_wiring/local && ~/github/sotaku/venv/bin/python local_wiring.py submission_anchored.py designed N=16
```

```bash
cd ~/github/one-layer/cpu_first/history/learned_wiring/local && ~/github/sotaku/venv/bin/python local_wiring.py submission_anchored.py grad N=32
```

Do not run `local_wiring.py ... train` or `local_anchored.py` locally beyond a few steps: their final checks run an
untrained machine's 100-round count-down inside autograd, which took 5 GB in a 5-step test here.

### 7. Files

In `cpu_first/history/learned_wiring/`:
- **`submission_anchored.py`**: the file, certified byte for byte (md5 `6328a7d239ec2fe08de2e78ad8f28ed9`);
- `ANCHORED.md`: this write-up;
- `anchored/`: `local_anchored.py` (the diagnostic harness), `modal_diag_anchored.py`, `plans/` (`a1`–`a4`: the
  diagnostics; `anchored_cpu.json`: the CPU seeds), `logs/`, `summarize_anchored.py` (this write-up's tables),
  `variants/`: the ablation and diagnostics of §3.4–3.5, not submissions (`submission_anchored_facts1.py`,
  `submission_anchored_handover.py`, `submission_anchored_keepadam.py`);
- records: `cpu_first/history/runs/learned_wiring/anchored_*.json` (RESULT_JSON, the full log, the exact source, the
  manifest).

The certified files are untouched: `submission_io3.py` (`c2331362…`) and the other stages' files, and the anchors'
`submission_anchors.py` (`a9e20f66…`) and `submission_combined_anchors.py` (`8887fdae…`).
