# H1 with learned squaring: schoolbook multiplication from a learned times table

## In plain words

**The task.** The model reads three numbers written as digits: a modulus N, a starting number x and a count T. It must
square x, keep only the remainder after dividing by N, and repeat that T times. To be certified it must get every one
of 768 fresh questions exactly right at T = 1, 2, 4, …, 64, both for moduli it saw in training and for larger moduli
it never saw (up to 8 digits).

**Where this starts.** [CHAIN.md](CHAIN.md)'s calculator does the remainder by long division, and every subtraction
in it is one learned rule: a 200-entry table, "this digit minus that digit, minus a borrow, gives this digit and maybe
a new borrow". But it squared the number with the computer's built-in multiplication.

**What changed.** The calculator now squares the way a schoolchild does:
- A learned **times table** of 100 entries gives the two digits of every one-digit product: 7 × 8 = 56 is "low digit
  6, high digit 5".
- For each digit of the number it writes one row of such products, shifted one place per digit, and adds the rows up.
  An 8-digit number gives a 16-digit square.
- The adding uses the learned subtraction table too. In ten's complement, "0 minus b" is "minus b", so
  a + b = a − (0 − b) (the trick of [ADDER.md](ADDER.md)).

So every arithmetic step the model takes, multiplying, adding and subtracting, now comes from one of its two learned
tables. No built-in arithmetic touches the numbers. Only the loop counter T is still read as an ordinary integer.

**What is still hand-written:**
- reading and writing the digits, and the loop over T;
- the layouts: the schoolbook rows, the long division, the chain of subtractions and its stop rule;
- for training only, an adder inside the checks: the subtraction table's check (as in CHAIN.md) and the times
  table's check. [ADDER.md](ADDER.md)'s attempt to learn without it did not learn.

**Did it certify?** Yes, on the untouched official H1 manifest on an H100 (`square_official`, the final file byte for
byte):
- all 768 questions right at every T up to 64, on seen and on unseen moduli. The unseen moduli have up to 8 digits,
  so the squares have up to 16;
- both learned tables are exactly right (100 of 100 times-table entries, 200 of 200 subtraction entries), and every
  entry used at evaluation was trained;
- training: 270.5 s. Both tables are done at step 499, about 113 s in, and then hold still;
- evaluation: 20.5 s, against 15.4 s for CHAIN.md's chain. The budget is 1,800 s.

The same file certified on 8 of 8 further seeds on CPU, and on 4 of 4 seeds that never see an answer.

The version with a hand-written adder in the squaring (the stand-in used until the learned adder was available) also
certified, officially (17.7 s evaluation) and on 12 of 12 CPU seeds.

**How the times table learns: not from the answers.** The only labels come after the remainder. One wrong digit in the
square shifts it by an amount that the remainder turns into a pseudorandom answer ([REPORT.md](../REPORT.md) §2). On a
small version of the problem (3-digit numbers, an exact remainder), the answers taught nothing from scratch: 0 to 4 of
the 100 entries came out right. Starting from a nearly right table, they broke right entries more often than they
repaired wrong ones.

What teaches it is the definition of multiplication, checked on each training row's own numbers x and N, with no
label:
- x × 0 = 0;
- x × (b + 1) = x × b + x, for b = 0 to 8: the ladder of x's multiples.

**What broke on the way, and the fix.**
- **Checking both sides at once** ("x × (b+1) minus x × b should give x", both sides still being learned) freezes the
  ladder into blocks. Inside a block each step is consistent but off by a constant; between blocks one check fails.
  Repairing it means moving a whole block, and most first moves make the score worse. It stalled with 28–32 of 100
  entries right.
- **Learning one way only** fixes that: x × (b+1) learns from the calculator's own x × b plus x. But early on x × b is
  itself wrong, and the entries that learn from it can get stuck in a blur. A wrong low digit makes a carry that covers
  for a wrong high digit. 3 of 6 small runs took 800, 2,200 and more than 2,500 steps (the others 400–600).
- **The fix, "the verified frontier":** x × (b+1) learns from x × b only for the numbers x whose rungs x × 0, …,
  x × b all check out. Then everything the table learns from is exact. 8 of 8 small runs finished in 200 steps. In the
  full model the times table is exact by step 250 in 22 of the 25 learned runs, and by step 300 in the other three.
- **One more in the full model.** With the learned adder, a single wrong subtraction entry also spoils some squares. I
  first used the answers only in steps where every square in the batch checked out. That shut them off exactly when
  they were needed, and one local seed stalled at 199 of 200 entries. Using the answers of the rows whose own squares
  check out fixed it.

## 1. Results

### 1.1 The official runs
Official manifest `benchmark/manifests/h100_hard_h1.json`, untouched:
- cuda:0, bf16 autocast (the model turns it off inside), the batch from the submission;
- 3,600 s training budget, 1,800 s evaluation budget;
- Modal H100 80GB, with the image and H1 data regenerated with the official generator arguments (`modal_chain.py`);
- seed 74. Every rung is 768 prompts.

| run | what ran | max T (seen moduli) | OOD-N max T | rungs T=1…64, seen / OOD-N | test, ood_t, ood_n_t | subtractor rows right | product cells right | training s (frozen at step) | evaluation s |
|---|---|---|---|---|---|---|---|---|---|
| `square_s0_official` | the exact tables (`MODE = "designed"`, 100 steps, learned adder): a plumbing check | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 (designed) | 100/100 (designed) | 3.2 (–) | 19.7 |
| `square_dadder_official` | the final file with `ADDER = "designed"` (the hand-written adder in the squaring) | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 | 100/100 | 259.8 (420) | 17.7 |
| **`square_official`** | **`submission_square.py` byte for byte**, md5 `37765051a8e73685f33330d43f151b76` (learned adder) | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 | 100/100 | 270.5 (499) | 20.5 |
| for comparison: `chain_v2_official` | `submission_chain.py` (designed integer square) | 64 | 64 | 768 everywhere | all exact | 200/200 | – | 271.8 (417) | 15.4 |

Every learned run trains all 4,400 logits: the subtractor's 2,400 (200 rows × (10 digit + 2 borrow)) and the product
cell's 2,000 (100 cells × (10 lo + 10 hi)).

The audit of the certified run: `AUDIT eval_rows_used=200/200 used_but_never_trained=0 eval_product_cells_used=100/100
product_used_but_never_trained=0`. Training's hard checks used every subtractor row at least 202,647 times (median
457,650) and every product cell at least 103,397 times (median 208,673). `summarize_square.py` compares both snapped
tables with true subtraction and multiplication outside the submission: no wrong entry. The plumbing run's audit
counts nothing as trained, because its tables are designed and nothing trains.

Most of each training run is the frozen remainder of the 20,000 steps:
- `square_official` reached step 400 at 81 s and froze at 499 (step 500 at 113 s);
- `square_dadder_official` froze at 420 (step 500 at 103 s).

### 1.2 Robustness on CPU
The same evaluator, unmodified, on Modal CPU containers: the manifest's device set to cpu, everything else official,
one seed per run. 768 prompts per rung.

The final file (learned adder; `plans/sq2_cpu.json`). The 8 runs with the answers ran it byte for byte (md5
`37765051…`, as `square_official`); the identity-only runs differ only in `REAL_MAX_T = 0`:

| run | seed | max T | OOD-N max T | rungs (seen / OOD) | test, ood_t, ood_n_t | subtractor rows right | product cells right | frozen at step | train s | eval s |
|---|---|---|---|---|---|---|---|---|---|---|
| square_c2_cpu_seed74 | 74 | 64 | 64 | 768 / 768 at every rung | 9999/9999, 10002/10002, 10002/10002 | 200/200 | 100/100 | 499 | 350.1 | 25.3 |
| square_c2_cpu_seed1 | 1 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 447 | 465.2 | 33.6 |
| square_c2_cpu_seed2 | 2 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 477 | 490.2 | 32.1 |
| square_c2_cpu_seed3 | 3 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 483 | 498.7 | 37.1 |
| square_c2_cpu_seed4 | 4 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 414 | 434.1 | 32.4 |
| square_c2_cpu_seed5 | 5 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 425 | 476.7 | 35.2 |
| square_c2_cpu_seed6 | 6 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 404 | 439.9 | 36.3 |
| square_c2_cpu_seed7 | 7 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 442 | 507.7 | 36.2 |
| square_c2_idonly_cpu_seed74 | 74 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 529 | 357.1 | 24.3 |
| square_c2_idonly_cpu_seed1 | 1 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 447 | 485.8 | 36.2 |
| square_c2_idonly_cpu_seed2 | 2 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 546 | 486.9 | 36.8 |
| square_c2_idonly_cpu_seed3 | 3 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 538 | 499.2 | 37.6 |

**12 of 12 certified**: 8 of 8 with the rows' answers (frozen at 404–499) and 4 of 4 identity-only (447–546). Seed
74 froze at 499 on CPU as on the H100: the same trajectory. Seed 1 froze at 447 with and without the answers. The
answers join only once every ladder check passes (about step 250), and in seed 1 every check was exact by then, so
both runs started the same 200-step freeze streak.

The stand-in version (hand-written adder in the squaring; `plans/sq1_cpu.json`). These ran the earlier file (md5
`66f84339d90fa150e1e8831db27d63d9`; its exact source is in each record), which
learns exactly as the final file with `ADDER = "designed"`: the same lessons and random numbers, and with the
hand-written adder every square checks out, so the per-row and the per-step answers gates agree. The official
`square_dadder_official` and the CPU seed 74 below froze at the same step, 420:

| run | seed | max T | OOD-N max T | rungs (seen / OOD) | test, ood_t, ood_n_t | subtractor rows right | product cells right | frozen at step | train s | eval s |
|---|---|---|---|---|---|---|---|---|---|---|
| square_c1_cpu_seed74 | 74 | 64 | 64 | 768 / 768 at every rung | 9999/9999, 10002/10002, 10002/10002 | 200/200 | 100/100 | 420 | 345.7 | 22.7 |
| square_c1_cpu_seed1 | 1 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 465 | 351.8 | 21.4 |
| square_c1_cpu_seed2 | 2 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 437 | 317.1 | 19.8 |
| square_c1_cpu_seed3 | 3 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 492 | 324.4 | 19.9 |
| square_c1_cpu_seed4 | 4 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 433 | 326.8 | 20.8 |
| square_c1_cpu_seed5 | 5 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 473 | 351.5 | 20.0 |
| square_c1_cpu_seed6 | 6 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 462 | 366.3 | 23.8 |
| square_c1_cpu_seed7 | 7 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 430 | 431.7 | 26.9 |
| square_c1_idonly_cpu_seed74 | 74 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 421 | 436.8 | 25.9 |
| square_c1_idonly_cpu_seed1 | 1 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 464 | 447.6 | 25.9 |
| square_c1_idonly_cpu_seed2 | 2 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 438 | 393.3 | 26.7 |
| square_c1_idonly_cpu_seed3 | 3 | 64 | 64 | 768 / 768 | all exact | 200/200 | 100/100 | 555 | 368.7 | 22.5 |

`idonly` is `REAL_MAX_T = 0`: training never sees a label. Every record's audit reads
`AUDIT eval_rows_used=200/200 used_but_never_trained=0 eval_product_cells_used=100/100 product_used_but_never_trained=0`.
In `square_c1_cpu_seed74` training's hard checks used every subtractor row at least 168,807 times and every product
cell at least 95,171 times. For comparison, `submission_chain.py`'s final recipe froze at 417–481 on its 8 CPU seeds
and at 408–544 identity-only (CHAIN.md §1).

Training time is mostly the frozen rest of the 20,000 steps. The model has 4,400 numbers: 2,400 subtractor logits and
2,000 product logits.

## 2. What is designed and what is learned

| part | `submission_chain.py` | `submission_square.py` |
|---|---|---|
| read N (LSB first), x and T; write the answer's digits right-aligned | designed | designed |
| loop T times | designed | designed |
| the state between steps | an integer | **digits** (8 places); no integer arithmetic touches the numbers |
| square the state | designed (int64 product) | **schoolbook**: designed layout (row j = s × s_j: cell (s_i, s_j)'s lo digit at place i, hi digit at place i + 1; rows shifted by j) |
| one-digit products | (inside the int64 product) | **learned product cell** (a, b) → (lo digit, hi digit), 100 rows × (10 + 10) logits |
| additions in the squaring (lo + shifted hi; the rows' sum) | (inside the int64 product) | **the learned subtraction table**, A + B = A − (0 − B) (`ADDER = "learned"`); or a hand-written ripple adder (`ADDER = "designed"`, the stand-in) |
| long division, chain of subtractions of N's prompt digits, stop rule | designed wiring | designed wiring (unchanged) |
| the subtractor cell (r, m, borrow in) → (digit, borrow out) | learned (2,400 logits) | learned (2,400 logits), now also the squaring's adder |
| what teaches the subtractor | label-free link checks (designed adder), "x mod N = x", the rows' answers on the own hard path | the same; link inputs come from the machine's own division of its own learned square of x; the answers only on rows whose squares check out |
| what teaches the product cell | — | label-free: x × 0 = 0 and the ladder x × (b + 1) = x × b + x along the verified frontier, on x and N (designed adder in the check) |

Evaluation snaps both cells to their argmax tables and runs the whole machine on integer digits.

## 3. The datapath

**Squaring** (`square_hard`). The state s is 8 digits, LSB first.
- Row j is s × s_j. Its place i gets the lo digit of cell (s_i, s_j) and its place i + 1 the hi digit, and the adder
  sums the two digit vectors in 10 places (`row_hard`). All 8 rows are formed side by side.
- The adder then sums the rows, row j shifted by j places, into an 18-place accumulator, one row at a time. Its low
  16 places are s². The two spare places mean no table, however wrong, can wrap around.

With `ADDER = "learned"` each addition A + B is two ripples of the learned subtraction table: C = 0 − B, then A − C
(`add_fwd`, `sub_hard`; ADDER.md's `add_learned`). In ten's complement C = 10^L − B, so A − C = A + B mod 10^L.

**Reduction.** CHAIN.md's restoring long division, unchanged, fed with the 16 digits instead of an integer: the
register starts at the top 8 digits (below N because s < N) and takes 8 steps, each shifting in the next digit and
running the chain of up to 9 subtractions of N, stopped at the first borrow.

**What it costs.** Per squaring, with the learned adder: 2 × 10 lookups in sequence for the rows (side by side) and
8 × 2 × 18 for the sum, 308 in all, against the reduction's 648. Evaluation took 20.5 s on the H100 (CHAIN.md's
chain: 15.4 s); with the hand-written adder, 17.7 s. On CPU: 24–38 s.

**The plumbing check.** With the exact tables (`MODE = "designed"`), `square_hard` equals v × v on 20,006 numbers below
10^8 (both adders), and the whole machine reproduces every answer of all 14 depth rungs locally
(`square_local/local_square.py designed`). Officially, `square_s0_official` certified T = 64 on both profiles.

## 4. How the tables learn

**The product cell** (`ladder`), on each training row's own numbers X = x and X = N (8 digits each, leading zeros
included), every step:
- **X × 0 = 0**: the soft row X × 0 (the cell applied to digit distributions, summed by a hand-written soft adder)
  against all-zero digits.
- **The ladder**: for b = 0, …, 8, the soft row X × (b + 1) against the digits of the hard machine's own X × b plus
  X (hand-written adder), but only for the numbers X whose rungs 0, …, b all check out on the hard machine
  (`LADDER_GATE`). Rung 0 checks out when the hard X × 0 is 0. Rung c checks out when the hard X × c equals the hard
  X × (c − 1) plus X. When rungs 0..b check out, the hard X × b is exactly b·X (induction), so the target is exact.
- The ladder covers every cell (a, b): a from X's digits (all ten digits occur in x and N), b = 0..9. The cells used
  at evaluation, (s_i, s_j), are the same kind: first index from the multiplicand, second from the multiplier digit.

No label and no made-up number: the inputs are the rows' x and N and what the machine computes from them. The rows'
answers never reach the product cell: their dividend is the machine's hard square.

**The subtractor:** CHAIN.md §4, unchanged, except where its numbers come from:
- Phase A's link checks run on the machine's own division of its own learned square of x. Until both tables are
  right that square is wrong, but the check (D − N) + N = D holds for any D.
- Phase B adds "x mod N = x" and the rows' answers along the own hard path, from the switch on (80% of the link
  checks passing, as in the chain).
- The answers are used in steps in which every ladder check passes. With the learned adder they are also limited to
  the rows whose squarings equal those made with the check's hand-written adder: the square of x, and that of the state
  before the last step (§6).

**Freeze** after 200 consecutive steps in which every check is exact on the hard machine: the link checks, "x mod N =
x", every ladder rung, and (learned adder) every square of x against the check's adder.

## 5. The tiny core

`square_local/tiny.py`: base 10, the product cell the only learned part. The rows' own numbers are 2,048 3-digit
moduli N (N mod 10 in {1, 3, 7, 9}, as in H1) and x < N, in 3-digit registers. Adam (lr 0.05), logits initialised
with std 0.1, batches of 256 rows (512 numbers), gradient clip 1. A run of 1,500 steps takes 20–60 s on the laptop.
Products are soft (the cell on digit distributions, places independent, summed by a hand-written soft adder); checks
are hard (argmax cell, integers). "Right" counts cells whose lo and hi digits are both right.

### 5.1 Answers through the reduction scramble
The lesson: the T = 1 answer x² mod N, with the reduction exact (the remainder taken as solved). The division runs
along the own hard path, as in the chain: the hard machine picks each quotient digit, and the soft square runs that
many exact subtractions.
- **From random init:** 0, 1 and 4 cells of 100 right (3 seeds, 600–1,500 steps); hard exact answers ≈ 0.
- **Near the truth** (every cell's logits ±2 around a chosen digit, so p ≈ 0.86 on it; k cells set wrong), after 300
  steps:

  | wrong at the start | answers alone | the ladder (§5.5) | the ladder + answers |
  |---|---|---|---|
  | 1 | 99, 100, 98, 98 | 100, 100, 100, 100 | 100, 100, 100, 100 |
  | 2 | 97, 98, 97, 98 | 100 ×4 | 100 ×4 |
  | 4 | 97, 95, 97, 96 | 100 ×4 | 100, 100, 100, 99 |
  | 8 | 93, 90, 95, 95 | 100 ×4 | 100, 100, 100, 98 |

  The answers repair little and break more. Mixed with the ladder they occasionally cost a cell. So the full model
  keeps them away from the product cell.
- **Why** (`tiny.py scramble`, 200 random single-cell errors): one wrong cell changes the square of 7.3% of the rows.
  Of those, 0.01% still get the right answer, and 26% of the answer digits agree (chance: 10%; the extra comes from
  errors at high places, which leave the low digits of the remainder alone).

Along the own hard path, the answer does say something exact: with the quotient digits fixed, the remainder is
v − qN, so the lesson pulls the square toward y + q·N. That is the true square exactly when the hard quotient is
right, that is, when the square is already nearly right.

### 5.2 Which identities can pin the table: exact null spaces
`tiny.py null` writes each identity, on 400 rows' numbers, as linear equations in the 100 cell values
f(a, b) = lo + 10·hi (the adder is exact, so a product is Σ f(x_i, y_j)·10^(i+j)), and computes the rank exactly
(modular arithmetic) and the free directions:

| identity on the rows' own numbers | cells it reaches | free directions | what they are |
|---|---|---|---|
| X × 1 = X (unit) | 10 | 0 among them | pins only the (a, 1) cells |
| X × 0 = 0 (zero) | 10 | 0 among them | pins only the (a, 0) cells |
| x × N = N × x (commutativity) | 90 | 45 | pins only the antisymmetric part; the zero table passes |
| X × (b+1) − X × b = X, b = 0..8 (the ladder) | 100 | 10 | f(a, b) + c(a): a constant per multiplicand digit |
| the ladder + zero, or + unit | 100 | **0** | unique: the truth |
| s(s+1) − s² = s (distributivity on the squares) | 100 | 10 | the same c(a) |
| (s+1)² − s² = 2s + 1 | 100 | 46 | the antisymmetric part (45) and a global constant |
| s² − (s−1)(s+1) = 1 ("the square computed two ways") | 100 | 3 | a global constant, digit-linear offsets λ(a + b), … |
| the two above together | 100 | 1 | the global constant |
| exact square labels s² (not in H1; for reference) | 100 | 45 | the antisymmetric part |

Consequences:
- **Identities on squares alone can never pin the antisymmetric part** f(a, b) − f(b, a). A square only sees
  f(a, b) + f(b, a). This one is harmless: those tables square every number exactly.
- **Difference identities cannot see a constant added to every cell.** Every product of two 3-digit registers gains
  the same amount, and it cancels in the difference.
- So there are **consistent impostors**. f(a, b) = ab + c (any c ≤ 18) and f(a, b) = (a+1)(b+1) − 1 both pass
  s² = (s−1)(s+1) + 1 on every row, and square no number right (checked on 2,048 rows).
- **Homogeneous identities** (commutativity, or the ladder written as X × (b+1) − X × b = X × 1) are passed by the
  zero table.
- **Anchors fix the ladder.** An exact anchor, X × 0 = 0 or X × 1 = X, makes the ladder pin all 100 cells.

### 5.3 Checks with both sides soft freeze
From random init, 1,500 steps (600 for seed 1 of the first three rows):

| lessons | cells right (seeds 1, 2, 3) | hard checks passing |
|---|---|---|
| unit + zero + the ladder as X × (b+1) − X × b = X, both rows soft | 32, 32, 28 | 34–40% of the ladder |
| unit + s(s+1) − s² = s | 23, 23, 14 | 18–25% |
| unit + (s+1)² − s² = 2s + 1 | 11, 11, 10 | 7–22% |
| s² − (s−1)(s+1) = 1 | 1, 3 (seeds 1, 2) | 20–29% |
| unit + zero + s² − (s−1)(s+1) = 1 | 23, 23 (seeds 1, 2) | 22–24% |

The ladder's stall is a set of **domain walls** (`probe_wall.py`). The stalled table's offsets f(a, b) − ab are
constant in blocks of b. Row a = 1, for b = 0…9, reads 0, 0, 0 | 21, 21 | 61, 61, 61 | 71, 71: the anchored block,
then blocks each consistent with itself. Each wall is one failing check. Moving the first block above a wall to
continue the block below removes that wall (the hard ladder checks rise from 34.1% to 34.5–36.8%), but in 7 of the
10 rows of a it raises the soft loss. Against a control that sharpens the same cells without moving them: 0.508 →
0.592 for row 1, 0.484 → 0.559 for row 3. Repairs need collective moves, and the first steps of those go uphill.

### 5.4 One-way checks, and the carry-compensated blur
The ladder in one direction ("bootstrap"): the soft X × (b + 1) learns the digits of the hard machine's own X × b plus
X, and X × b gets no gradient from it.

| lessons | seeds | all 100 cells right at step |
|---|---|---|
| zero + bootstrap | 1–6 | 800, 500, 600, **never** (98 at 2,500), 400, **2,200** |
| unit + zero + bootstrap | 1 | 500 |
| zero + the ladder with the lower rung as a soft target (stop-gradient, X × (b+1) − sg(X × b) = X) | 1–4 | never: 51, 25, 35, 39 cells at 1,500 |

The direction alone is not enough. The target must also be snapped to the hard machine's value: a soft target passes
its blur up the ladder.

The stalls of the bootstrap all have the same shape: a cell whose step crosses a ten. In seed 4, 4 × 8 = 28 + 4 = 32
should carry into the hi digit.
- The stuck cell (4, 8) has hi = 2 (p = 0.9998; truth 3) and lo blurred between 2 (p = 0.58, the truth) and 9
  (p = 0.39).
- With lo = 9, the place sum 9 + (the neighbour's hi) produces a carry that makes the next place right despite the
  missing hi unit. The wrong lo costs one digit, at its own place.
- The cell above copies the error: (4, 9) = 26, consistent with its predecessor 22.
- Moving hi's mass from 2 to 3 (`probe_carry.py`): the loss is 0.0165 at 0%, 0.0113 at 25%, 0.062 at 50%, 0.060 at
  100%. Only a joint move of lo and hi goes downhill.
- The trap is made by early wrong targets, not by the row's structure. Rows learned from exact targets from the start
  (a diagnostic with a designed multiplier in the loss, not usable for real) reach 100/100 on 8 of 8 seeds by step
  100.

### 5.5 The fix: the verified frontier
The bootstrap, with rung b + 1 trained only on the numbers whose rungs 0..b all check out on the hard machine: **8 of 8
seeds, all 100 cells right by step 200** (logged every 100 steps). It also repairs 1–8 wrong cells on 4 of 4 seeds
each (§5.1's table).

No symmetric saddle shows up here. The cell's outputs have fixed places (lo at i, hi at i + 1), and each rung's
cells are different table rows, tied to the anchor by the order of the ladder. What remains is a staging problem: learn
a rung only once the rung below is verified. Labels play no part.

## 6. What broke in the full model
- **The answers gate.** The first learned-adder file used the rows' answers only in steps in which the machine's own
  square of *every* x in the batch matched the check adder's square. Local seed 2 (batch 64) then stalled at 199/200
  subtractor rows: (8, 8, borrow 0) had borrow 1. That one entry also spoils the additions that pass through it, so
  about 20% of the squares failed, and the gate was closed in almost every step. The answers are the lesson that
  repairs such (r, r, ·) entries (LEARNED_REMAINDER.md §4.2, CHAIN.md §5). Gating per row instead (use the answer rows
  whose squarings check out) fixed it: frozen at step 719. Seeds 1, 3, 6 and 7 froze at 419–463 with the per-step
  gate; seeds 1 and 7, rerun with the per-row gate, froze at the same steps (423, 463).
- **Nothing else needed changing in the subtractor's recipe.** Its link inputs now come from a learned square that is
  wrong for the first ~200 steps, and the chain's schedule (switch at 80% of the link checks, about step 70–100)
  still works.
- **When each table is done.** At step 250 (the logs' first point after step 1) the ladder passes every check in 22
  of the 25 learned runs. The other three are seed 74 with the learned adder (official, CPU, identity-only), at 96.1%
  then and exact by step 300 at the latest. The link checks pass on 67–100% at step 250. The two tables' lessons are
  independent, but they share the evaluator's gradient clip (norm 1, over all parameters) and the Adam reset at the
  switch, so the subtractor's trajectory moves the ladder's timing a little.

## 7. Gray areas against the rules
Same as CHAIN.md §6, except:
- **Rule 7, "no hard-coded algorithm in the forward pass."**
  - The forward pass has no designed arithmetic left: every product, sum and difference is a lookup in one of the
    two learned tables. The lookups compute their row index from the digits (a·10 + b, or r·20 + m·2 + c), which is
    addressing, as in CHAIN.md.
  - Still designed: the digit I/O, the schoolbook layout, the long-division and chain wiring, the stop rule and the T
    loop (which reads T as an integer).
  - It is still a designed CPU, now with two learned gate tables.
- **Rule 14, "no data augmentation, task-specific solvers."**
  - The training checks use a hand-written adder: the subtractor's (as in CHAIN.md) and the ladder's.
  - The ladder is the definition of multiplication (x × 0 = 0, x × (b+1) = x × b + x) applied to the rows' own x and
    N. It computes no product by itself: every target is the machine's own hard row plus x.
  - Through the verified frontier, those targets are exact, because the hand-written adder is. So the ladder is
    induction with a designed adder: the adder plays the part it plays in the chain's (D − N) + N = D. REPORT.md §3's
    caution about exact solvers hidden in a loss applies to that adder, which is why it is the next piece to remove
    (§8).
  - The gate reads the machine's own hard checks, as the chain's freeze does.
- **The ten multiples X × 0..9 of the rows' numbers** are computed for training only. The squaring itself only forms
  s × s_j for the state's own digits s_j. (The chain likewise checks links whose results the machine throws away.)
- **Labels never reach the product cell.** The answers train the subtractor only, as in identity-only runs. The rule's
  "end-to-end gradient path" holds for the subtractor, and for the product cell only through the label-free ladder.
  §5.1 is the reason.
- **Schedule, freeze, Adam reset, argmax snapping, rule 8.** Unchanged.

## 8. Next pieces to remove
1. **The adder inside the training checks** (the ladder's targets and soft rows, the subtractor's link check).
   ADDER.md found that the subtraction table does not learn when its check's adder is made of itself.
2. **The schoolbook layout and the ladder's order.** Here the layout is given. Learning which cell goes to which place
   is wiring, which gradients do not find (REPORT.md §3).
3. **The stop rule, the long-division and chain wiring, the loop counter and the I/O**, as in CHAIN.md §7.

## 9. Reproduce
Everything heavy runs on Modal. The launcher is the existing [`modal_chain.py`](modal_chain.py), unchanged, with
`--file submission_square.py`, started from the one-layer-deeper repository root. Records land on the Modal Volume
`learned-remainder`; [`wait_fetch.py`](wait_fetch.py) downloads them to `cpu_first/runs/learned_remainder/<name>.json`
(full log, RESULT_JSON, the exact submission source).

The final file as it is (learned adder):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --name square_official --file submission_square.py
```

The hand-written adder in the squaring (the stand-in):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --name square_dadder_official --file submission_square.py --set '{"ADDER": "\"designed\""}'
```

The plumbing check with the exact tables:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --name square_s0_official --file submission_square.py --set '{"MODE": "\"designed\"", "MAX_STEPS": "100"}'
```

The CPU seed plans (one container each; `plans/sq2_cpu.json` the final file, `plans/sq1_cpu.json` ran the stand-in
file of §1.2):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode plan --plan cpu_first/learned_remainder/plans/sq2_cpu.json --file submission_square.py
```

Fetch, then tabulate with both table audits:

```bash
~/github/sotaku/venv/bin/python cpu_first/learned_remainder/wait_fetch.py square_official square_dadder_official square_s0_official
```

```bash
~/github/sotaku/venv/bin/python cpu_first/learned_remainder/summarize_square.py --table square_official square_dadder_official square_s0_official
```

Files:
- **`submission_square.py`**: the submission, certified byte for byte as `square_official` (md5
  `37765051a8e73685f33330d43f151b76`).
  `ADDER = "designed"` gives the hand-written adder. `MODE = "designed"` gives the plumbing check.
- **`summarize_square.py`**: summarize.py plus the product cell's audit against true multiplication.
- **`plans/sq1_cpu.json`, `plans/sq2_cpu.json`**: the CPU seed plans.
- **`square_local/`**: the tiny core and the small local checks, each under a minute and well under 1 GB:
  - `tiny.py null`: §5.2's exact null spaces;
  - `tiny.py train LESSONS SEED STEPS`: §5.1, §5.3–5.5 (e.g. `zero+peanogate`, `zero+peanoboot`, `unit+zero+peano`,
    `answers`, `twoway`);
  - `tiny.py align K`: the near-truth table;
  - `tiny.py scramble`: one wrong cell's effect on the answers;
  - `probe_carry.py SEED STEPS`: the carry-compensated blur;
  - `probe_wall.py SEED STEPS`: the domain walls;
  - `local_square.py designed`: the plumbing check, with `ADDER=designed` in the environment for the stand-in;
  - `local_square.py train STEPS B SEED`: a short training run that prints both tables' errors at the end.

  They need torch and, for `local_square.py`, the H1 jsonl files (`H1_DATA`, as in `cpu_first/h1.py`). `stub/` stands
  in for the evaluator's `benchmark` API.
- **Untouched:** `submission_chain.py` (md5 `ed360cb107e644615e30c910ba6f6a2d`), `submission.py` (md5
  `bd477b9aadc9fcf34f428f728b5bea20`), `modal_chain.py`, `summarize.py`, `wait_fetch.py`.
