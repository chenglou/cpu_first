# Digit meanings from the order of the numbers: the H1 subtraction table learned with no hand-written adder

## In plain words

**The task.** The model reads a modulus N, a starting number x and a count T, all written as digits. It must square x,
keep the remainder after dividing by N, and repeat that T times. To be certified it must get every one of 768 fresh
questions exactly right at T = 1, 2, 4, …, 64, on moduli it saw in training and on larger ones it never saw.

**The calculator.** [CHAIN.md](../learned_remainder/CHAIN.md)'s calculator does the long division with one learned
rule, a table of 200 entries: "this digit minus that digit, minus a borrow, gives this digit and maybe a new borrow".
The table starts random. Nothing tells it what the digit symbols mean.

**Why the meanings are hard to pin.** Training never shows the table a correct subtraction. The earlier pieces checked
the table against itself instead: subtract N from a number, add N back, and see whether the number comes back. The
adding back was done with the same table (a + b = a − (0 − b), [ADDER.md](../learned_remainder/ADDER.md)), so the check
only asks the table to agree with itself, and many wrong tables do:
- subtraction with the digit symbols consistently renamed (call 3 "7" and 7 "3");
- the arithmetic of another number system of the same size, where carries run only through "the remainder mod 5" of
  each digit;
- "copy the first number", and tables that ignore the borrow coming in.

From a random start training lands on one of these and stays there. So far the meaning came from a hand-written adder,
used in the check for the first 70 training steps. It knows what the digits mean; once the table was mostly right,
the self-check could finish the job.

**What pins them: order.** A digit's meaning is its place in the order 0 < 1 < … < 9. The questions carry that order:
in every question x is smaller than N. Two kinds of lessons turn it into a teacher.
1. **Borrows from x < N.** A borrow says "this went below zero". Since x < N, x − N goes below zero and N − x does not.
   On its own that teaches the borrows on only 1 seed in 10. The signal has to travel through all nine digit places,
   and at the start it fades on the way. The same fact holds for the top digits of the two numbers: the number formed
   by x's digits from any place upward is at most N's, and strictly smaller exactly when those digits differ. Taught on
   every such top part, it gets every borrow right on 10 of 10 seeds. It teaches nothing else.
2. **Digits from two laws of subtraction.** Once the borrows are right, the table can say which of two digits is
   smaller (does one minus the other borrow?) and whether a one-digit subtraction goes below zero. Two laws of
   subtraction then fix every digit:
   - taking the same digit away from two digits keeps their order;
   - taking more away from the same digit leaves less;
   - except when exactly one of the two goes below zero: that one wraps round and comes out on top.

   In each column of the table, ten entries must then come out in one known order. Ten different digits in a known
   order can only be 0, 1, …, 9, so only the true table passes. A renamed or mod-5 table breaks these laws.

**Is the warm start gone? Yes.** [`submission_anchors.py`](submission_anchors.py) has no hand-written adder at any
step. It uses no training answer, and does without the learned-adder check and the wrap-around fact as well. Its whole
training is the facts of point 1, then the two laws of point 2. On the untouched official H1 manifest on an H100
(`anchors_official`, the file byte for byte) it got all 768 questions right at every T up to 64, on seen and unseen
moduli. The table came out 200/200, and every entry used at evaluation was trained. Training took 208 s and evaluation
16 s. The same file certified on 8 of 8 further seeds on CPU, and none of them ever sees an answer.

**Why this trains when the self-check did not.** The self-check is a puzzle in which every entry depends on the others,
so gradient descent settles on whichever consistent table it meets first. The order laws split the table into 40 small
sorting problems, one per column and one per row. The learned borrows referee every comparison, so each entry is told
directly to be smaller than these and larger than those. Gradient descent sorts well. The same law stated on whole
numbers (two results of the machine compared, also on their lowest k digits) stalled at 80 to 117 of 200 entries. A
comparison of two long numbers is settled by their first differing digit, so most entries were never tested, and a
tie there was excused by the digits below. Applied one digit at a time, the law tests every entry.

**The combined model.** The same lessons replace the warm start in
[`submission_combined_anchors.py`](submission_combined_anchors.py), the model in which every part is learned: reader,
writer, times table, subtraction table and gate. There the subtraction table waits untouched until the reader has
worked out the prompt (step 90 in the official run). Then the top-part facts teach its borrows and the order laws its
digits, and about 80 steps later the table is exact; the rest of the certified recipe carries on as before. On the
official manifest on an H100 (`cmbanchors_official`) it certified with every learned part exact, training in 454 s
where the certified combined took 667 s. The same file certified on 8 of 8 CPU seeds with every learned part exact
on each. On none of them did the table settle on a wrong number system, as the simplified recipe of ABLATION.md §11
did on 1 of 4 (mod 5). It also certifies with the old table lessons switched off altogether (4 of 4 seeds, the
learned-adder check among them), so there too the order of the numbers does the whole job.

**What is still hand-written** in the core file (as in [`submission_simple.py`](../ablation/submission_simple.py)):
- reading and writing the digits, squaring, the long-division layout, the chain of subtractions, the stop rule and the
  loop over T;
- the lessons themselves: the statements "x < N on every top part", "order is kept, reversed or wrapped", "a column
  and a row of the cell use every digit once";
- the rule that the digit lessons start once the borrow facts hold on every row of 10 batches.

There is no adder, no table and no answer among them.

## Contents

1. Results
2. Which wrong tables each signal admits (exact search; base-10 landscape)
3. The tiny core: the failure is in training, not in what the lessons allow
4. What makes it train: the borrows, then the digits
5. The recipe and the submission file
6. The combined model
7. Gray areas against the rules
8. Reproduce

## 1. Results

### 1.1 Official runs
The official manifest `benchmark/manifests/h100_hard_h1.json`, untouched (cuda:0, bf16 autocast, the submission's
batch, 3,600 s training and 1,800 s evaluation budgets, seed 74), on a Modal H100 80GB, through the same runner as the
earlier pieces (`modal_anchors.py` is ablation's `modal_ablate.py` with its own app and Volume: the competition image,
H1 regenerated with the official generator arguments, `benchmark.runner`). Every rung is 768 prompts. Every record
holds the exact source that ran.

| run | what ran | max T (seen / unseen moduli) | rungs | test, ood_t, ood_n_t | tables | training s (steps) | evaluation s |
|---|---|---|---|---|---|---|---|
| **`anchors_official`** | **`submission_anchors.py` byte for byte** (md5 `a9e20f66cceba504fea1b0e389461313`) | **64 / 64** | 768/768 at every rung, both profiles | 1.0, 1.0, 1.0 | 200/200; AUDIT `eval_rows_used=200/200 used_but_never_trained=0` | 208 (3,000; digit lessons from step 50) | 16 |
| `anchors_s0_official` | the same file with the exact table (`MODE = "designed"`, `MAX_STEPS = 100`): the plumbing check | 64 / 64 | 768/768 everywhere | 1.0, 1.0, 1.0 | designed | 4 (100) | 20 |
| **`cmbanchors_official`** | **`submission_combined_anchors.py` byte for byte** (md5 `8887fdae6348c034f1d83f9b13e95213`; §6) | **64 / 64** | 768/768 at every rung, both profiles | 1.0, 1.0, 1.0 | subtraction 200/200, product 100/100, gate true, true reading and writer; AUDIT every row used at evaluation trained | 454 (frozen at step 553) | 41 |
| for comparison: `adder_warm_official` | `submission_adder.py` (hand-written adder for 70 steps) | 64 / 64 | 768/768 | all exact | 200/200 | 78 (frozen at 343) | 13 |
| for comparison: `simple_official` | `submission_simple.py` (hand-written adder for 70 steps) | 64 / 64 | 768/768 | all exact | 200/200 | 301 (3,000) | 23 |
| for comparison: `combined_official` | `submission_combined.py` (hand-written adder until the reading is chosen) | 64 / 64 | 768/768 | all exact | all exact | 667 (frozen at 545) | 48 |

### 1.2 CPU seeds
The same files through the same unmodified evaluator on Modal CPU containers (8 CPUs), the manifest's device set to
cpu and the seed changed, everything else official. One seed per run (`plans/cpu_anchors.json`,
`plans/cpu_combined_anchors*.json`).

| file | seeds | certified T = 64 on both profiles | tables | schedule | training s / evaluation s (CPU) |
|---|---|---|---|---|---|
| `submission_anchors.py` | 74, 1, 2, 3, 4, 5, 6, 7 | **8/8** (every rung 768/768; test, ood_t, ood_n_t 1.0) | 200/200 on every seed; every audit `used_but_never_trained=0` | digit lessons from step 50, 43, 55, 58, 94, 76, 49, 69 | 222–430 / 16–30 |
| `submission_combined_anchors.py` | 74, 1, 2, 3, 4, 5, 6, 7 | **8/8** (every rung 768/768; test, ood_t, ood_n_t 1.0) | every learned part exact on every seed (subtraction 200/200, times 100/100, the gate, the reading, code and writer); every audit line `used_but_never_trained=0` | reading at 90–274, order lessons from 127–407, frozen at 553–823 | 693–1002 / 410–558 |

No `submission_anchors.py` run uses an answer: the file has no lesson that reads one. Its seed 74 on CPU took the H100
run's path (digit lessons from step 50 on both): no lesson draws a random number. The combined file reads answers in
its phase C, as the certified one does (§6).

### 1.3 Training runs on the H1 machine
`core.py` trains the table with any set of lessons the way the evaluator would (Adam at 0.05, initial logits of
std 0.1, gradient clipping 1, the rows' own numbers, the machine's own link inputs), on random H1 training rows, and
prints the snapped table against the truth. On Modal CPU containers (`modal_lab.py`; plans p2–p5). "First 200/200" is
the first logged step (every 25) with the whole table right. R1 is the first working recipe, R2 the certified one.

| run | seeds 200/200 | entries right at the end (each seed) | digit lessons start (step) | first 200/200 (step) |
|---|---|---|---|---|
| **R2: top-part facts, then both order laws + both bijectivities (the file's recipe), batch 64, 400 steps** | **12/12** | 200 × 12 | 44–96 | 125–275 |
| **R2, batch 512 (the official batch)** | **4/4** | 200 × 4 | 43–96 | 100–150 |
| R2 + the learned-adder check and the wrap fact 100 steps after the gate, batch 64 | 12/12 | 200 × 12 | 44–96 | 125–275 |
| R2 without the row order law | 4/6 | 190, 200, 200, 200, 186, 200 | 59–96 | 125–200 |
| R2 without the column order law | 4/6 | 190, 200, 190, 200, 200, 200 | 59–96 | 125–275 |
| R2 without row bijectivity | 6/6 | 200 × 6 | 59–96 | 125–275 |
| R2 without column bijectivity | 6/6 | 200 × 6 | 59–96 | 125–275 |
| R2 without either bijectivity (order laws only) | 6/6 | 200 × 6 | 59–96 | 150–275 |
| R2 without the gate (every lesson from step 1) | 6/6 | 200 × 6 | – | 150–400 |
| R2 without the gate and without bijectivity, batch 64 / 512 | 2/12 / 2/4 | 196, 200, 152, 156, 195, 185, 177, 182, 200, 156, 163, 196 / 200, 200, 191, 160 | – | 350–400 / 225–350 |
| R2 with bijectivity only (no order law) | 0/6 | 17, 25, 21, 17, 29, 22 | 59–96 | – |
| R2 with x − N borrows and N − x does not instead of the top-part facts | 2/6 | 200, 5, 200, 2, 4, 3 (the gate never opens on 4) | 82–149 | 150–200 |
| R1: top-part facts, then the column order law on pairs of link places + column bijectivity, then the learned check and wrap, batch 64 | 6/12 | 182, 200, 200, 196, 194, 200, 200, 194, 200, 156, 192, 200 | 44–96 | 175–250 |
| R1, batch 512 | 4/4 | 200 × 4 | 43–97 | 175–225 |
| R1 without bijectivity / without the order law / without the late check | 0/4 / 0/4 / 1/4 | 113–150 / 17–23 / 188–200 | 60–96 | – / – / 225 |
| R1 with the learned check joining at the gate (not 100 steps later) | 0/4 | 55, 54, 70, 72 | 60–96 | – |
| the answers as the only digit teacher (top-part facts for the borrows, then the rows' T ≤ 2 answers along the own hard path), H1's 2% answer rows | 0/4 | 23, 21, 18, 18 | 58–96 | – |
| the same with 10%, 30%, 100% of the rows answered (§4.3) | 0/4, 0/4, 0/4 | 15–23, 17–26, 15–20 | 43–100 | – |

## 2. Which wrong tables each signal admits

### 2.1 Exact search on tiny machines
[`exact.py`](exact.py) finds every table that passes a set of lessons, by complete search. A table is 2B² digit entries
and 2B² borrow entries. Every lesson becomes a list of instances on fixed inputs: every modulus N coprime to B with
up to W digits, every x < N, every link input D < B·N, so every number the machine's chain could meet. An instance is
run on a partial table and passes, fails, or stops at the first entry it needs that is not yet set. A depth-first
search sets one entry at a time, always one that some instance is waiting on, and backtracks when an instance fails.
When nothing waits on the remaining entries they are free: they are counted, not enumerated. The search is exact: it
returns every passing table (or stops at 400 groups or 600 s, marked ≥).

Base 3 with up to 3 digits is the smallest machine whose training fails like base 10's (§3). Base 4 with 2 digits has
the extra wrong systems a base that is not prime allows. "Borrows held true" fixes the 2B² borrow entries at the truth,
which is what the order facts teach (their rows below show it); the count is then over digit tables. A borrow-only
lesson does not constrain digits, so its count includes all B^(2B²) digit tables (3¹⁸ ≈ 3.87·10⁸ in base 3, 4³² ≈
1.84·10¹⁹ in base 4): a count of exactly that size means one borrow table.

The rows are the signals the task listed, in the lessons' short names:
- the learned self-check, (D − N) − (0 − N) = D: `lc` (digits), `lcc` (complete: the final borrows too); for
  comparison `dc`, the hand-written adder's check;
- "x − N borrows, N − x does not": `less`, `more`; "x mod N = x": `xmod`. In the core model the reader is designed, so
  IO.md's "x < N" fact is these two facts; `pref` is the same fact on every top part. In the combined model IO.md's
  ranking pins the reader's digit order from the same fact;
- the identity facts on the rows' own numbers: `zero` (R − 0 = R), `self` (R − R = 0), both containing 0 − 0 = 0;
  and the learned adder's own rules `neg2` (0 + N = N) and `comm` (R + N = N + R);
- the wrap fact `wrap`, and its version on the machine's chains `chain`; `dec`: a link that does not borrow lowers its
  number, one that borrows raises it;
- the T = 1 and T = 2 answers along the machine's own hard path, `ans` and `ans2` (with the borrows held true, as the
  task asked: once the borrows are exact); and the label-free `nn0`, N² mod N = 0 along the same path;
- this study's lessons: `mon`, the order law on whole numbers; `colord` and `roword`, the order laws one digit at a
  time; `bijc`, column bijectivity.

Cells: **1 (the truth)**: exactly one complete table passes, the true one. **the truth** (k outputs left free): every
output the lessons constrain must be the true one, and k outputs are not constrained at all (a borrow-only lesson
leaves every digit free; a 2-digit machine never uses some entries). "N: the truth + W wrong": N complete tables pass.
"N, in G groups": N tables in G distinct solutions with free outputs. "≥ N": the search stopped at 400 groups, or
(600 s) at its time limit; "unfinished": no complete table found within 600 s (a free-borrow search for a lesson that
says little); "–": not run (a borrow-only lesson with the borrows held true constrains nothing). `runs/x1.jsonl` holds
every search's solutions.

| lessons | base 3, borrows free | base 3, borrows held true | base 4, borrows free | base 4, borrows held true |
|---|---|---|---|---|
| lc | ≥ 1.01e+07 | 22: the truth + 21 wrong | unfinished (600 s) | unfinished (600 s) |
| lcc | ≥ 2.02e+07 | 2: the truth + 1 wrong | unfinished (600 s) | unfinished (600 s) |
| dc | ≥ 1 (600 s) | **1 (the truth)** | unfinished (600 s) | **the truth** (1 output left free) |
| less | ≥ 5.02e+11 | – | ≥ 1.37e+26 | – |
| more | ≥ 6.78e+11 | – | ≥ 1.46e+26 | – |
| less, more | **the truth** (18 outputs left free) | – | **the truth** (32 outputs left free) | – |
| xmod | ≥ 1.11e+07 | **the truth** (18 outputs left free) | ≥ 1.15e+18 | **the truth** (32 outputs left free) |
| zero | 4.72e+11 | **the truth** (15 outputs left free) | **the truth** (56 outputs left free) | **the truth** (28 outputs left free) |
| self | 4.72e+11 | **the truth** (15 outputs left free) | **the truth** (56 outputs left free) | **the truth** (28 outputs left free) |
| zero, self | 1.31e+10 | **the truth** (13 outputs left free) | **the truth** (50 outputs left free) | **the truth** (25 outputs left free) |
| neg2 | 2.71e+12 | 1.38e+07 | ≥ 1.58e+26 | 1.14e+17 |
| comm | ≥ 1.42e+10 | **1 (the truth)** | ≥ 3.81e+22 | ≥ 6.87e+10 (600 s) |
| wrap | ≥ 9.48e+07 | ≥ 400 | ≥ 6.37e+19 | ≥ 6.71e+09 |
| chain | ≥ 9.48e+07 | ≥ 400 | ≥ 7.59e+19 | ≥ 6.71e+09 |
| dec | unfinished (600 s) | ≥ 400 | unfinished (600 s) | ≥ 1600 |
| mon | unfinished (600 s) | **1 (the truth)** | unfinished (600 s) | **the truth** (1 output left free) |
| pref | **the truth** (18 outputs left free) | – | **the truth** (32 outputs left free) | – |
| colord | ≥ 400 | **1 (the truth)** | ≥ 80 (600 s) | **1 (the truth)** |
| roword | ≥ 400 | **1 (the truth)** | ≥ 64 (600 s) | **1 (the truth)** |
| bijc | ≥ 1.05e+08 | ≥ 400 | ≥ 1.72e+12 | ≥ 400 |
| ans | unfinished (600 s) | **1 (the truth)** | unfinished (600 s) | **the truth** (6 outputs left free) |
| ans2 | unfinished (600 s) | **1 (the truth)** | unfinished (600 s) | ≥ 4096 (600 s) |
| nn0 | ≥ 1.72e+09 | **the truth** (1 output left free) | ≥ 5.74e+16 | ≥ 7.05e+07 |
| lcc, zero, self, wrap | ≥ 1.01e+07 | **1 (the truth)** | ≥ 5.83e+14 | 4: the truth + 3 wrong |
| lcc, zero, self, wrap, neg2, comm | 1.01e+07, in 123 groups (one the truth) | **1 (the truth)** | ≥ 2.81e+14 | **1 (the truth)** |
| less, more, lcc, zero, self, wrap, neg2, comm | **1 (the truth)** | **1 (the truth)** | unfinished (600 s) | **1 (the truth)** |
| colord, roword | 360: the truth + 359 wrong | **1 (the truth)** | unfinished (600 s) | **1 (the truth)** |
| colord, bijc | ≥ 400 | **1 (the truth)** | ≥ 400 | **1 (the truth)** |
| pref, colord | **1 (the truth)** | **1 (the truth)** | **1 (the truth)** | **1 (the truth)** |
| pref, colord, roword | **1 (the truth)** | **1 (the truth)** | **1 (the truth)** | **1 (the truth)** |
| pref, colord, roword, bijc | **1 (the truth)** | **1 (the truth)** | **1 (the truth)** | **1 (the truth)** |
| less, more, colord, roword | **1 (the truth)** | **1 (the truth)** | **1 (the truth)** | **1 (the truth)** |

What the map says:
- **The order facts teach the borrows and nothing else.** "x − N borrows, N − x does not" leaves exactly one borrow
  table, the true one (base 3 and base 4), and every digit table. So does "x < N on every top part". "x mod N = x"
  says the same once the borrows are true (it reads only borrows then), and alone it leaves many borrow tables.
- **The learned-adder check alone admits a zoo.** With the borrows held true it still leaves the truth and 21 wrong
  tables in base 3 (its complete form: the truth and 1 wrong); with free borrows over 10⁷ tables, the renamed digits
  and other number systems among them. The identity facts R − 0 = R and R − R = 0 pin only their own entries, and the
  wrap fact and the chain rule leave hundreds of tables or more even with the borrows true.
- **Some single lessons do pin base 3 once the borrows are true:** R + N = N + R, the whole-number order law, the
  T = 1 and T = 2 answers through the machine, and the hand-written adder's check. In base 4 the answers and the
  order law leave only outputs the 2-digit machine never uses; N² mod N = 0 does not pin base 4.
- **Label-free identities can pin the table on every input,** but only in combination. With the borrows held true,
  in base 3 the complete check, R − 0 = R, R − R = 0 and the wrap fact leave only the truth; base 4 needs 0 + N = N
  and R + N = N + R as well. With free borrows, in base 3, the core recipe's lessons with the two facts leave only the
  truth. So the old recipe's failure from a random start was never that its lessons allowed a wrong table. It is
  training (§3).
- **The order laws pin the digits given the borrows, alone.** One column law (the order of r kept, or wrapped) leaves
  only the truth once the borrows are true: ten different values in a known order are 0…9. With free borrows, the
  top-part facts plus one order law leave only the true table.
- **The smallest combination that leaves only the true table is two lessons:** x < N as a borrow fact (the top-part
  form, or x − N borrows with N − x does not) and the column order law. The smallest that also trains from a random
  start on every seed tried, on the H1 machine, adds the row order law and either the gate or bijectivity (§1.3);
  the submission keeps both.

### 2.2 Base 10 on the H1 rows' own numbers
The exact search runs on base 10 too when the instances come from H1 rows (`exact.py 10 8 LESSONS --data N`): x and N
of N random training rows, and the link inputs of the true machine's division of each x². On 8,192 rows, "x < N on
every top part" leaves exactly one borrow table, the true one (on 2,048 rows a few entries are still undecided: the
rows never make them the deciding digit).

[`landscape.py`](landscape.py) runs every lesson on the consistent wrong tables of ADDER.md §3 and on their
true-borrow cousins, on each table's own numbers as training would (the link inputs of its own division of 1,024 H1
rows' x², the rows' x and N, and 512 rows with T ≤ 2 for the answers). Share of instances passed:

| table | learned check | complete | hand-written adder's check | x − N borrows | N − x no borrow | x mod N = x | R − 0 = R | R − R = 0 | 0 + N = N | R + N = N + R | wrap | chain | E < D iff no borrow | **x < N on every top part** | **column order** | **row order** | column bijective | row bijective | answers T = 1 | T = 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| true | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| copy the minuend | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 0.063 | 0 | 0 | 1 | – | 0 | 0 | 0 | 0 | 1 | 0 | 0.006 | 0 |
| lazy (ADDER.md §4) | 0.99 | 0.99 | 0 | 1 | 0.003 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0.99 | 0.003 | 0.99 | 0.09 | 0.55 | 0 | 0.006 | 0 |
| no borrows at all | 1 | 1 | 0.05 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | – | 0 | 0 | 0.367 | 0.367 | 1 | 1 | 0 | 0 |
| swapped borrow code | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| digits renamed (d → −d) | 1 | 1 | 0 | 0.219 | 0.219 | 0.219 | 1 | 1 | 1 | 1 | 0.336 | 1 | 1 | 0.219 | 0.2 | 0.2 | 1 | 1 | 0.006 | 0 |
| borrows in the mod-2 part | 1 | 1 | 0.018 | 0.451 | 0.499 | 0.451 | 1 | 1 | 1 | 1 | 0.76 | 1 | 1 | 0.251 | 0.511 | 0.511 | 1 | 1 | 0 | 0 |
| borrows in the mod-5 part | 1 | 1 | 0.017 | 0.677 | 0.679 | 0.677 | 1 | 1 | 1 | 1 | 0.872 | 1 | 1 | 0.615 | 0.722 | 0.722 | 1 | 1 | 0.006 | 0 |
| true borrows, digits ignore the borrow in | 1 | 0.448 | 0.049 | 1 | 1 | 1 | 1 | 1 | 1 | 0.192 | 0.699 | 0.595 | 0.713 | 1 | 0.9 | 0.9 | 1 | 1 | 0.006 | 0 |
| true borrows, digit r − m − [m even] | 1 | 0.984 | 0.007 | 1 | 1 | 1 | 0 | 0.001 | 0 | 0.038 | 1 | 1 | 0.946 | 1 | 0.9 | 0.8 | 1 | 0 | 0.006 | 0 |
| true borrows, digit r − m − [m = 0] | 1 | 0.983 | 0.002 | 1 | 1 | 1 | 0 | 0.174 | 0 | 0.066 | 1 | 1 | 0.946 | 1 | 0.9 | 0.88 | 1 | 0 | 0.006 | 0 |
| true except (r, 0, 1) → r (run-of-9s family) | 0.669 | 0.388 | 0.364 | 1 | 1 | 1 | 1 | 1 | 0.646 | 0.107 | 0.749 | 0.704 | 0.732 | 1 | 0.99 | 0.98 | 1 | 0.5 | 0.026 | 0 |
| true except (0, 0, 1) → 0 | 0.707 | 0.707 | 0.503 | 1 | 1 | 1 | 1 | 1 | 0.646 | 0.159 | 0.749 | 0.684 | 1 | 1 | 0.99 | 0.99 | 0.95 | 0.95 | 0.917 | 0.88 |

The order laws read the table's own borrows as their referee, so a table with wrong borrows is judged by its own
wrong order. "Chain" (the link after a borrowing one does not borrow) has no instance (–) for tables that never
borrow. Three facts in this table matter for §4:
- the learned check passes every consistent wrong system on its own numbers (1.0), and the lazy table on 99%;
- the top-part facts pass exactly the tables with the true borrows, as the exact search says;
- among the tables with the true borrows only the truth passes the two order laws; the others fail 1–20% of the
  comparisons. Those comparisons are the gradient that moves them (§4.2). The answers fail every wrong system almost
  everywhere, but at 2% of the rows they cannot move a random table (§4.3).

### 2.3 Why the order laws leave only the truth
Take one column of the table, the ten entries (r, m, c) for r = 0…9, with m and c fixed. Their borrows (true) say
which entries go below zero (r < m + c) and which of two digits is smaller. The column law then orders the ten
entries completely: the ones that stay at or above zero in the order of r, then the ones that go below zero in the
order of r, all above the first group. Ten different digits in that order must be 0, 1, …, 9, so (r, m, c) ↦
(r − m − c) mod 10: the true column. Every column is pinned the same way, and so is every row by the row law. The
argument needs only the borrow table; nothing in it names a digit's value. The value of a digit is its rank, and the
rank comes from the learned borrows, which come from x < N.

## 3. The tiny core: the failure is in training, not in what the lessons allow

[`tiny.py`](tiny.py) trains the table with the full batch of every input of a tiny machine (every N, every x, every
link input D < B·N), so the gradient is exact and the data cannot be the problem. With the borrows held true (the
digit table is all that is learned), base 3 with up to 3 digits:

| core (every input, borrows held true, 400 full-batch steps) | lessons | seeds reaching the true table | entries right at the end (each seed) |
|---|---|---|---|
| base 3, up to 3 digits | the core recipe: learned-adder check (complete, lowest place ×5), R − 0 = R, R − R = 0, x − N borrows, N − x does not, wrap, 0 − N borrows, 0 + N = N, R + N = N + R | **0/8** | 10, 10, 10, 12, 12, 10, 10, 10 of 18 |
| | the hand-written adder's check, R − 0 = R, R − R = 0, the facts, wrap | **8/8** | 18 × 8 |
| | the core recipe without R − 0 = R, R − R = 0 | 1/8 | 13, 10, 10, 10, 18, 10, 11, 10 |
| | the core recipe, R − 0 = R and R − R = 0 rippled with the hard borrows (ABLATION.md's Rule 3 fix) | 0/8 | 10, 10, 10, 12, 12, 10, 10, 10 |
| | + "E < D iff the link does not borrow" | 0/8 | 12, 10, 10, 10, 12, 10, 10, 10 |
| | + the chain rule (the link after a borrowing one does not borrow) | 1/8 | 10, 10, 10, 10, 18, 10, 6, 10 |
| | + N² mod N = 0 along the own hard path | 0/8 | 9, 11, 9, 11, 11, 11, 9, 9 |
| | + the answers (T = 1) along the own hard path, on every row | 8/8 | 18 × 8 |
| | + the same answers on 10% of the rows | 3/8 | 8, 12, 18, 10, 18, 18, 12, 12 |
| | + the order law on whole numbers | 2/8 | 11, 11, 11, 18, 18, 11, 11, 11 |
| | **the order law on whole numbers alone** (pairs of link inputs with the same N; coupled comparator) | **8/8** | 18 × 8 |
| | the order law + R − 0 = R, R − R = 0 | 8/8 | 18 × 8 |
| | the order law with the plain (independent) soft comparator | 7/8 | 18, 18, 15, 18, 18, 18, 18, 18 |
| base 4, up to 2 digits | the order law alone | 0/8 | 27, 26, 27, 26, 31, 31, 27, 26 of 32 |
| | the order law + R − 0 = R | 8/8 | 32 × 8 |
| | the order law + R − 0 = R, R − R = 0 | 8/8 | 32 × 8 |
| base 5, up to 2 digits | the order law + R − 0 = R, R − R = 0 | 6/8 | 50, 46, 46, 50, 50, 50, 50, 50 of 50 |
| | the order law at every width + R − 0 = R, R − R = 0 | 8/8 | 50 × 8 |
| base 6, up to 2 digits | the order law + R − 0 = R, R − R = 0 | 0/8 | 59, 59, 61, 58, 59, 65, 59, 59 of 72 |
| | the order law at every width + R − 0 = R, R − R = 0 | 8/8 | 72 × 8 |

What it says:
- **The smallest failing core is base 3.** The core recipe never reaches the truth from random digits, though the
  exact search (§2.1) says its lessons leave no other table. Every seed stops at 10–12 of 18 entries with its loss
  above zero: a local minimum of the soft lessons, not a wrong solution. The hand-written adder's check from the same
  start reaches the truth on every seed. None of the label-free additions tried rescues the core recipe (0–2 of 8),
  the order law included: the core's own lessons hold the wrong basin.
- **Answers work only when dense.** The rows' answers along the machine's own hard path, on every row, get every seed
  to the truth; on 10% of the rows, 3 of 8. H1 has them on 2% (and at base 10 even 100% does not start, §4.3).
- **The order law on whole numbers, on its own, trains in base 3 but needs help beyond.** "Subtracting N keeps two
  numbers' order, unless exactly one wraps" alone reaches the truth on 8 of 8 seeds in base 3. In base 4 it stalls
  with the "subtract 0" column one step too high (r − 0 = r + 1, the top value tied); R − 0 = R fixes that (8/8). In
  bases 5 and 6 it stalls (6/8, 0/8) with a block of entries in one column a step too low; the same law at every width
  (the lowest k places of a subtraction are a subtraction mod Bᵏ) trains both (8/8, 8/8), and base 10 with 2-digit
  moduli (2 of 3 seeds exact, the third at 195 of 200; plan p1).
- Two details of the lesson matter. A comparator applied to two soft numbers treats two uses of the same entry as two
  independent draws, which rewards blurry entries; the comparator here couples them (same entry, same digit). And
  each pair must be asked which of the two must come out smaller, strictly: the "not smaller" target is passed by a
  tie.

## 4. What makes it train on the H1 machine

### 4.1 The borrows: x < N on every top part
The borrows alone ([`borrows.py`](borrows.py): only the 200 borrow outputs are trained, base 10, the H1 rows, batch
64, 300 steps, 10 seeds; `runs/borrows_census.txt`):

| facts | seeds with all 200 borrows right | borrows right at the end |
|---|---|---|
| x − N borrows, N − x does not | 1/10 | 40, 46, 38, 42, 200, 44, 44, 46, 40, 44 |
| + 0 − N borrows | 0/10 | 43–50 |
| the order rule of ABLATION.md §8 (every pair among 0, x, N, and a − a) | 0/10 | 36–41 |
| the order rule's strict pairs only | 1/10 | 39–46, one 200 |
| only "borrows" facts (x − N, 0 − N, 0 − x) | 0/10 | 100–150 |
| x − N, N − x with the hard machine's own borrows into each place | 0/10 | 89–110 |
| **x < N on every top part** | **10/10** | 200 × 10 |
| x < N on every top part + x − N, N − x | 10/10 | 200 × 10 |

ADDER.md found that the order facts teach all 200 borrows "in every run"; they did there with the self-check and the
anchors beside them. On their own they almost never do. A whole-number fact reaches a place only through every place
above it. At the start the padding places above N's width (0 − 0 with a borrow in, and without) do not pass a borrow
on, so the gradient below them is near zero. The two facts also push those top entries in opposite directions,
x − N toward "borrow" and N − x toward "no borrow", a saddle that a lucky seed leaves the right way. The top parts give
every place its own target. The number formed by the digits from place j upward is a smaller instance of the same
fact, with no borrow coming in. The shortest top parts decide the top entries, and each longer one adds one place
below. "The two digit strings differ there" is a comparison of tokens, not of values.

One limit: on tiny machines with made-up rows the top-part facts do not always teach the borrows (the R2 recipe in
`core.py` on such rows, plan p3: the gate never opened on 2 of 4 runs in base 3, 2 of 4 in base 4, 1 of 4 in base 6;
every run whose gate opened ended exact). On the H1 rows they never failed: 10 of 10 borrow-only runs, and the gate
opened on all 124 training runs of plans p2–p5 that gate on them, and on every evaluator run (all 22 that train a table: the two files' official runs, their 16 CPU seeds and the 4 of §6's ablation).

### 4.2 The digits: the order laws, one digit at a time
Once the borrow facts hold on every row of 10 batches, the learned borrows can referee (the gate). What was tried, on
the H1 machine's own numbers, in order (seed 1 or 2, batch 64, `core.py`):

| digit lessons after the gate | entries right | why it stops |
|---|---|---|
| the order law on pairs of the machine's link inputs (one row, same N) | 84 at step 250, then ~80 | a comparison of two subtraction results is decided at their first differing digit, so the low entries are almost never the deciding one; 99.8% of the comparisons pass |
| the same at every width k (each comparison on the lowest k places) | 105–117 | every place can now decide, but a tie at the deciding place is excused whenever the places below happen to be in the right order |
| the column law on single entries: two places of a row's links reading the same column (m, c) | 140 at step 300 | ties now cost, but a block of consecutive entries one step off (a tie at one end, a gap at the other) can only be fixed by moving the whole block, and nothing pushes the entry next to the gap |
| + column bijectivity (each column uses every digit once) | 191–193 | the gap pulls; what is left is an adjacent swap or a block in one column, held by the learned-adder check if it joins (it passes mutually inverse swaps, ADDER.md §6) |
| the column law on every pair of entries of each column the batch uses, + bijectivity | 192 / 200 (seed 1 / 2) | one column with its whole no-wrap run one step high: all its comparisons pass but one |
| **+ the row law (subtracting more gives less) and row bijectivity** | **200 on 12 of 12 seeds** | each entry now has two neighbours that anchor it, in its column and in its row: a shifted block in a column is out of order in every row it crosses |

The row law adds no new kind of fact; it is the same law with the roles of the two digits exchanged. "a − b keeps the
order of a and reverses the order of b, unless the wrap": every entry is sorted against its column and against its
row. Bijectivity ("different digits minus the same digit give different digits", and the same with the roles
exchanged) is implied by strict order, but as a lesson it pulls toward unused values; either it or the gate is needed,
not both (§1.3).

In terms of the gradient, each comparison tells two entries "this one below, that one above", and the referee (which
entries wrap, which of two digits is smaller) is a lookup in the learned borrow table, with no gradient through it. No
lesson composes one entry's output with another's lookup, so there is no copy-the-input fixed point and no mutually
inverse pair to fall into.

### 4.3 How much answer signal would be needed
The task asked how much answer signal the digits would need if nothing label-free pinned them. With the borrows
already right (the top-part facts, gated as in R2), the rows' answers along the machine's own hard path as the only
digit lesson (`core.py`, batch 64, 600 steps, 4 seeds each; `answer_share` makes that share of the training rows
T ≤ 2 rows, which carry the answers H1 trains on):

| rows with an answer (T ≤ 2) | seeds 200/200 | entries right at the end |
|---|---|---|
| 2% (H1's own share) | 0/4 | 23, 21, 18, 18 |
| 10% | 0/4 | 18, 15, 23, 17 |
| 30% | 0/4 | 20, 17, 26, 21 |
| 100% | 0/4 | 15, 20, 17, 16 |
| 100%, from a table with 180 of 200 entries started at the truth (borrows held true; one seed, 150 steps) | 199 at step 150, rising | |

The answer lesson works: from a table that is 90% right it climbs to 199 in 150 steps, and the answers that pass rise
from 45% to 80%. From random digits it does not start, even with every row answered. The machine's own path through a
random table is the wrong division of the right numbers, so the answer of the last link is taught to the wrong
inputs. So within this study's budgets (600 steps at batch 64) no amount of answer signal replaces the warm start.
Answers can only finish a table that something else has nearly made. That is the role the earlier recipes gave them.

## 5. The recipe and the file
[`submission_anchors.py`](submission_anchors.py) is [`submission_simple.py`](../ablation/submission_simple.py)'s machine
(datapath unchanged: reading, squaring, the chain, the stop rule, the loop over T, the writer) with a new training
recipe:
1. **Every step: x < N on every top part.** For every place j, x[j:] − N[j:] (the numbers formed by the digits from
   place j up, P − j places, no borrow in) borrows exactly when the two digit strings differ there, and
   N[j:] − x[j:] never borrows. Soft ripples of the cell's borrow outputs; only the borrows read them.
2. **From the step after the borrow facts hold on every row of `GATE_WIN` = 10 consecutive batches** (x − N borrows,
   N − x does not; steps 43–94 over the 9 certified runs):
   - the column law on every pair (r, m, c), (r′, m, c) and the row law on every pair (r, m, c), (r, m′, c) of the
     entries the batch's links read: −log P(the lower one's digit < the upper one's), with independent digit
     distributions of the two entries; "lower" from the learned hard borrows: the one that does not wrap, else (same
     wrap) the smaller r in a column, the larger m in a row (`order_pairs`);
   - bijectivity: −mean log(10 · q) for the mean digit distribution q of every column and of every row of the cell.
3. Nothing else: no answer, no learned-adder check (`LATE_AFTER = None`; with a number it adds ADDER.md's complete
   check and the wrap fact that many steps after the gate, as R2 + late in §1.3), no freeze, no phases. 3,000 steps of
   Adam at 0.05.

Everything is fp32 with autocast off; no lesson draws a random number. Its log line every 250 steps reads the facts,
the learned-adder check and the wrap fact on the hard machine (they are 1.0 from step 250 on in the official run),
though neither of the last two is trained.

## 6. The combined model
[`submission_combined.py`](../learned_remainder/submission_combined.py) learns every part at once: the reader and the
writer, the times table, the subtraction table, the commit gate. Its subtraction table had the hand-written adder's
check until the reader chose a reading (steps 90–273 across its runs), because every fact about x and N needs the
reading. ABLATION.md §11 found that its simplified table recipe settled on the mod-5 number system on 1 of 4 seeds,
because the reader delays the facts and the bare check meets the wrong systems first.

[`submission_combined_anchors.py`](submission_combined_anchors.py) is that file with one switch, `TABLE_TEACH =
"order"`, written as a patch of the certified file: the switch, the order lessons, and a weight that holds the
pieces' table lessons back; every other line is the certified one.
- **No hand-written adder at any step,** and until the reading is chosen no lesson trains the subtraction table. It
  waits at its random start, so nothing can pull it toward a wrong system.
- **From the step the reading is chosen:** x < N on every top part, on the reader's x and N (§5, point 1).
- **Once x − N borrows and N − x does not on every row of 10 batches:** the order laws and bijectivity (§5, point 2).
- **100 steps later:** the pieces' own table lessons join (the learned-adder check, R − 0 = R, R − R = 0, 0 + N = N,
  R + N = N + R, the facts and the wrap fact), with the learned adder from their first step. The phases, the times
  table's ladder, the gate's facts, the answers, T's direction and the freeze are the certified file's.

The official run, step by step (`cmbanchors_official`, H100; the log every 250 steps and the schedule's events):

| step | event |
|---|---|
| 90 | the reader chooses the true reading; the top-part facts start |
| 127 | the borrow facts have held on every row for 10 batches: the order laws and bijectivity start |
| 171 | phase B (the table's checks pass on 80% of the links); the table is exact from here on |
| 191 | the table has been exact for 20 steps: the times table's ladder joins; phase C (the gate's facts, the answers) |
| 227 | the pieces' own table lessons join (the table is already exact) |
| 449 | T's direction settled by the answers |
| 553 | frozen (the certified combined: 545) |

Training took 454 s and evaluation 41 s; the certified combined took 667 s and 48 s. Every table came out exact: the
subtraction table 200/200, the times table 100/100, the gate (commit with probability 0.9999 with no borrow, 0.0007 with
one), the true reading (N after its marker reading rightwards, x before T's marker reading leftwards, T and the answer
at the end reading leftwards), the identity code and the identity writer. The audits: every subtraction row, product
cell, countdown row and gate row used at evaluation was trained.

Every run of the file, step by step (`combined/schedule_table.py`; the step of each schedule event; "order lessons" is when the borrow facts had held on every row of 10 batches):

| run | where | max T (seen / unseen) | reading | order lessons | phase B | ladder | phase C | T's direction | frozen | training / evaluation s |
|---|---|---|---|---|---|---|---|---|---|---|
| cmbanchors_official | H100 | 64 / 64 (14/14 rungs) | 90 | 127 | 171 | 191 | 191 | 449 | 553 | 454 / 41 |
| cmbanchors_cpu_s74 | CPU seed 74 | 64 / 64 (14/14 rungs) | 90 | 127 | 171 | 191 | 191 | 449 | 553 | 817 / 454 |
| cmbanchors_cpu_s1 | CPU seed 1 | 64 / 64 (14/14 rungs) | 120 | 157 | 201 | 221 | 221 | 485 | 585 | 911 / 536 |
| cmbanchors_cpu_s2 | CPU seed 2 | 64 / 64 (14/14 rungs) | 93 | 139 | 182 | 202 | 202 | 398 | 577 | 756 / 478 |
| cmbanchors_cpu_s3 | CPU seed 3 | 64 / 64 (14/14 rungs) | 115 | 156 | 200 | 220 | 220 | 480 | 577 | 1002 / 558 |
| cmbanchors_cpu_s4 | CPU seed 4 | 64 / 64 (14/14 rungs) | 274 | 407 | 300 | 461 | 377 | 692 | 823 | 983 / 410 |
| cmbanchors_cpu_s5 | CPU seed 5 | 64 / 64 (14/14 rungs) | 108 | 153 | 196 | 216 | 216 | 451 | 565 | 907 / 552 |
| cmbanchors_cpu_s6 | CPU seed 6 | 64 / 64 (14/14 rungs) | 93 | 131 | 176 | 196 | 196 | 466 | 567 | 693 / 419 |
| cmbanchors_cpu_s7 | CPU seed 7 | 64 / 64 (14/14 rungs) | 144 | 208 | 248 | 268 | 268 | 550 | 653 | 917 / 450 |

All 9 certified, every learned part exact on every one (`combined/check_tables.py`). Seed 74 on CPU took the H100 run's path step for step. On every seed the table was exact within 34–45 steps of the order lessons' start (the ladder joins after 20 exact steps in a row). Seed 4 is the slow reader, as in COMBINED.md: its reading came at step 274, phase B by the 300-step fallback, and phase C at 377, before the digit lessons; its table was exact by step 441 all the same, and it froze at 823. The certified combined's own 8 CPU seeds froze at 535–714 (COMBINED.md); these at 553–823, the later ones the seeds with late readings. With the pieces' own table lessons never joining (`LATE_AFTER = 10 ** 9`,
`plans/cpu_combined_nolate.json`, seeds 74, 1, 2, 3), the file certified on 4 of 4 seeds with every learned part exact
and clean audits, and every schedule event fell on the same step as with them (they join only after the table is
exact). So in the combined model too the order lessons do the subtraction table's whole job: the learned-adder check,
which ABLATION.md §11 found load-bearing there, is no longer needed.

The combined model still reads answers, as the certified one does: T's direction (16 or 61) is settled by the
answers of the T = 16 rows (IO.md), and phase C trains on the rows' T ≤ 2 answers. Neither can reach the subtraction
table before it is exact. T's direction waits for the whole machine to be exact. The phase-C answers run only once the
times table passes its ladder everywhere, and the ladder joins only after the subtraction table has passed every check
for 20 steps. That holds even when phase C opens before the digit lessons do: on the slow-reader seed 4 the reading
came at step 274, phase C at 377 (its switch reads only "x − N borrows"), the order lessons at 407, and the ladder at
461. The gate's own facts run on a detached copy of the table (CONTROL.md) and teach only the gate.

## 7. Gray areas against the rules
Same as ABLATION.md §10 / ADDER.md §9 for the datapath, with these changes:
- **Rule 14, "no data augmentation, task-specific solvers".**
  - No hand-written adder at any step (ADDER.md: the first 70; COMBINED.md: until the reading is chosen). No
    hand-written arithmetic checks the table any more: every comparison a lesson makes is the learned borrow table.
    (The core's datapath, its designed squaring included, is the earlier pieces' own.)
  - The top parts x[j:], N[j:] are numbers formed by the row's own digits from place j up, like the register windows
    the long division already reads. "The two digit strings differ from place j up" compares tokens, not values. The
    fact used is x < N, which the prompts guarantee and the earlier pieces used ("x mod N = x" since
    LEARNED_REMAINDER.md, "x − N borrows, N − x does not" since ADDER.md, the reader's code since IO.md).
  - The order laws and bijectivity are laws of subtraction, like R − 0 = R and R − R = 0 (which the earlier recipes
    used, and which state outputs outright). They state relations between entries, never an entry's digit. They are
    applied to the cell's entries that the batch's links read. Which pairs to compare is bookkeeping in the loss; which
    of a pair must come out smaller is read from the learned borrows.
- **Rule 7, the schedule.** One gate (the digit lessons wait for the borrow facts) instead of the earlier warm start,
  phases, Adam reset and freeze.
- The combined file keeps every gray area of COMBINED.md except the warm start.

## 8. Reproduce
Everything heavy runs on Modal. Local commands are the small ones (each under a minute, well under 1 GB); they need
torch and the H1 jsonl files (`H1_DATA`, as in `cpu_first/h1.py`).

The official runs and the CPU seeds go through the deployed app `anchors` (`modal_anchors.py`: the competition image,
H1 regenerated with the official generator arguments, the untouched manifest for GPU jobs, `device: cpu` and the seed
for CPU jobs; at most 20 CPU containers and 2 H100s; records on the Volume `anchors`). Deploy once from the
one-layer-deeper root, then launch a plan and wait for its records (`records/<name>.json`):

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal deploy ../cpu_first/anchors/modal_anchors.py
```

```bash
cd ~/github/one-layer/cpu_first/anchors && ~/github/sotaku/venv/bin/python launch.py plans/h1_anchors.json
```

```bash
cd ~/github/one-layer/cpu_first/anchors && ~/github/sotaku/venv/bin/python launch.py plans/cpu_anchors.json
```

```bash
cd ~/github/one-layer/cpu_first/anchors && ~/github/sotaku/venv/bin/python fetch.py plans/h1_anchors.json plans/cpu_anchors.json
```

```bash
cd ~/github/one-layer/cpu_first/anchors && ~/github/sotaku/venv/bin/python summarize.py
```

The combined file the same way: `plans/h1_combined_anchors.json`, `plans/cpu_combined_anchors.json`,
`plans/cpu_combined_anchors2.json`, and `plans/cpu_combined_nolate.json` for the ablation of §6.

The lab runs (§1.3, §2.1, §3) go through `modal_lab.py` (app `anchors-lab`, CPU containers, its own light image with
the H1 training rows; the client waits and appends one JSON line per job to `runs/<plan>.jsonl`). `make_plans.py`
writes the plans: p1–p5 (H1-machine runs and tiny cores), x1 (the exact map), t1 (the tiny-core censuses; p5 is p4
and t1 together):

```bash
cd ~/github/one-layer/cpu_first/anchors && ~/github/sotaku/venv/bin/python make_plans.py
```

```bash
cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run anchors/modal_lab.py --plan anchors/plans/p3.json
```

```bash
cd ~/github/one-layer/cpu_first/anchors && ~/github/sotaku/venv/bin/python tables.py train
```

Local checks:

```bash
cd ~/github/one-layer/cpu_first/anchors && ~/github/sotaku/venv/bin/python local_anchors.py submission_anchors.py 300 64 1
```

```bash
cd ~/github/one-layer/cpu_first/anchors && ~/github/sotaku/venv/bin/python exact.py 3 3 pref,colord
```

```bash
cd ~/github/one-layer/cpu_first/anchors && ~/github/sotaku/venv/bin/python exact.py 3 3 lcc,zero,self,wrap --borrows-true
```

```bash
cd ~/github/one-layer/cpu_first/anchors && ~/github/sotaku/venv/bin/python borrows.py 300 10 prefix
```

```bash
cd ~/github/one-layer/cpu_first/anchors && ~/github/sotaku/venv/bin/python landscape.py 1024
```

Files in `cpu_first/anchors/`:
- **`submission_anchors.py`**: the core machine with the order recipe, certified byte for byte (`anchors_official`).
- **`submission_combined_anchors.py`**: `submission_combined.py` with `TABLE_TEACH = "order"` (§6), certified byte for
  byte (`cmbanchors_official`). `TABLE_TEACH = "adder"` runs the certified recipe's lessons and schedule, but not the
  certified run bit for bit (`combined/local_equiv_combined.py`: identical through step 7, then the parameters drift,
  most likely through the zero-valued loss term the order branch adds).
- `core.py`: the chain machine in any base, every lesson of the pieces and of this study, training on the machine's
  own numbers (`core.run`); `tiny.py`: the full-batch tiny-core trainer on every input; `exact.py`: the exact search;
  `borrows.py`: the borrow-only trainer; `landscape.py`: §2.2's table; `local_anchors.py`: a submission file trained
  locally as the evaluator would; `combined/local_combined_anchors.py`: COMBINED.md's local harness pointed at the
  combined file; `combined/local_equiv_combined.py`: the "adder" mode against the certified combined file.
- `modal_anchors.py`, `launch.py`, `fetch.py`, `summarize.py`: the evaluator runs (`records/`); `modal_lab.py`,
  `make_plans.py`, `plans/`, `tables.py`: the lab runs (`runs/`).

The certified files of the earlier pieces are untouched: `submission_adder.py` (md5 `44e658ab…`),
`submission_simple.py` (`f374ec70…`), `submission_combined.py` (`4f5cecc7…`), `submission_chain.py` (`ed360cb1…`),
`submission.py` (`bd477b9a…`), `modal_chain.py` (`ad5fbd9c…`).
