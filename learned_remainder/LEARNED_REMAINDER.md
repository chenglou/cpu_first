# H1 with the remainder learned

## In plain words

**The task.** The model reads three numbers written as digits: a modulus N, a starting number x and a count T. It must
square x, keep only the remainder after dividing by N, and repeat that T times. To be certified it must get every one
of 768 fresh questions exactly right at T = 1, 2, 4, …, 64, both for moduli it saw in training and for larger moduli
it never saw (up to 8 digits).

**The machine.** It is a small calculator wired the way a schoolchild does long division:
- square the number;
- divide by N one digit at a time, trying all ten possible quotient digits at once (subtract 0·N, 1·N, …, 9·N);
- keep the largest one that does not make the running remainder go negative;
- repeat T times, then write out the digits.

**What it learns.** The heart of the remainder is one tiny rule, used at every digit of every subtraction: "this digit
minus that digit, minus a borrow, gives this digit and maybe a new borrow". Here that rule is not written by hand. It is
a 200-entry table that starts out random and is learned by gradient descent. Whether a subtraction "went negative" is
read from the table's own final borrow, so the learned table also decides which quotient digit is kept. Everything else
is wired by hand: reading the digits, squaring, the long-division layout, the multiples of N, the "largest
non-negative" rule, the loop over T.

**What certified.** On the untouched official H1 manifest on an H100, all exact on every rung up to T = 64, on seen
and unseen moduli:
- **Stage 0**, the hand-written subtraction table: a check of the plumbing. 3 s of training, 5 s of evaluation.
- **Stage 1**, the learned table: 155 s of training (it is done after 484 steps and then holds still), 5 s of
  evaluation. The same recipe certified on 8 of 8 further seeds through the same evaluator on CPU.
- **Stage 2**, the learned table with the made-up lessons removed: 182 s of training (done after 499 steps), 5 s of
  evaluation. The same recipe certified on 8 of 8 seeds on CPU, and on 4 of 4 even without ever seeing an answer.
  This is the final `submission.py`, run byte for byte.

**The fine print.** The algorithm itself, long division, is still designed; only the subtraction table is learned.
Stage 1 teaches the table with practice divisions that the model makes up from each training row's own N and x (it
knows their answers by construction). That is a gray area under the competition's "no data augmentation" rule. Stage 2
replaces them: the calculator checks its own subtractions on the training rows' own numbers ("subtract, then add
back, and you must get the original"; "a number smaller than N is its own remainder"), and then learns from the
training answers. That check needs a hand-written adder.

## 1. Results per stage

Official manifest `benchmark/manifests/h100_hard_h1.json`, untouched (cuda:0, bf16 autocast, batch from the submission,
3,600 s training budget, 1,800 s evaluation budget), Modal H100 80GB, image and H1 data regenerated with the official
generator arguments (`modal_learned_remainder.py`). Seed 74. Every "rung" is 768 prompts.

| stage | record | max T (seen moduli) | OOD-N max T | rungs T=1…64, seen / OOD-N | test, ood_t, ood_n_t | subtractor rows right | training s (steps) | evaluation s |
|---|---|---|---|---|---|---|---|---|
| 0: designed table | `s0_official` | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 (designed) | 3.3 (100) | 5.5 |
| 1, first recipe | `s1_official` | none | none | 0, 1, 0, 2, 0, 0, 0 / 0 at every rung | 1/9,999, 1/10,002, 8/10,002 | 8/200 | 583 (20,000) | 4.0 |
| **1: learned table** | `s1v2_official` | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 | 154.9 (20,000; frozen at 484) | 4.9 |
| 2, first recipe | `s2_idans_official_v1_cancelled.log` | – | – | cancelled at step ~6,000 (swapped code: identity check 0.06, real rows 0%) | – | – | – | – |
| **2: learned table, no made-up numbers** | `s2v2_idans_official` | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 | 184.2 (20,000; frozen at 499) | 5.4 |
| **final file, Stage 2 (its default)** | `final_s2_official` | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 | 181.9 (20,000; frozen at 499) | 4.9 |
| **final file, Stage 1** (`LESSON = "madeup"`) | `final_s1_official` | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 | 154.5 (20,000; frozen at 484) | 4.8 |

`final_s2_official` ran the final `submission.py` byte for byte (md5 `bd477b9aadc9fcf34f428f728b5bea20`).
`final_s1_official` ran it with only the `LESSON` line changed.

Model state (all trainable):
- **The subtractor.** 2,400 logits: 200 rows × (10 digit + 2 borrow).
- **Stage 1** adds the estimator's 1,000,000, used only while learning: 1,002,400 elements. Optimizer state after the
  first step: 2,004,803.
- **Stage 2.** Its official run `s2v2_idans_official` still carried the estimator's table, unused, with no gradient
  (4,802 optimizer elements). The final file drops it, so every weight is trained (rule 6): 2,400 elements
  (`final_s2_official`).

The first recipes of Stages 1 and 2 failed on their official runs, both by settling on the "swapped borrow code" (§3.4,
§4.2).

The same submissions through the same unmodified evaluator on Modal CPU containers (manifest with `device: cpu`, all
else official), one seed per run. Runs of superseded recipes still going at the end were stopped to save compute, and are
marked as such:

| recipe | seeds | certified T = 64 on both profiles | table rows right | frozen at step | training s / evaluation s (CPU) |
|---|---|---|---|---|---|
| Stage 0 (designed table) | 74 | 1/1 | 200/200 | – | 4 / 16 |
| Stage 1, first recipe (uniform hold) | 2, 4, 5, 74, 1 (3 stopped early) | 3/5: 74 and 1 failed (swapped code: 3/200 and 6/200 rows right) | 200/200 where certified | 401–437 | 264–294 / 13–16 |
| **Stage 1** | 74, 1, 2, 3, 4, 5, 6, 7 | **8/8** | 200/200 | 477–492 | 196–243 / 9–12 |
| Stage 2, first recipe (no lowest-place weight, no 300-step cap) | 1, 2, 5; 3 and 4 stopped early, stuck in the swapped code (identity check 0.06) | 3/5 | 200/200 where certified | 478–580 | 260–370 / 11–12 |
| Stage 2, first recipe, identity only | 1 | 0/1: stalled at 199/200 rows (row (9, 9, 0)) for 19,500 steps; test 41.6% | 199/200 | – | 2,849 / 11 |
| **Stage 2** (identity + answers) | 74, 1, 2, 3, 4, 5, 6, 7 | **8/8** | 200/200 | 499–815 | 240–379 / 9–14 |
| **Stage 2, identity only (no labels at all)** | 74, 1, 2, 3 | **4/4** | 200/200 | 661–715 | 297–364 / 10–14 |
| identity without "x mod N = x" (4,000 steps, first recipe) | 74, 1; 74 with answers | 0/3 (seed 1: 199/200, row (0, 0, 1) wrong; seed 74: swapped code) | 0–199/200 | – | 487–618 / 9–11 |
| the rows' answers only, batch 512 | 74, 1 | 0/2 (hard exact 0.000 throughout) | 6, 10/200 | – | 665–736 / 12 |
| the rows' answers only, batch 4096 | 1 (live log; both runs stopped at step ~19,000) | 0/1 seen: hard exact 0.000 throughout | – | – | – |

## 2. What is designed and what is learned

| part | Stage 0 | Stage 1 | Stage 2 |
|---|---|---|---|
| read N (LSB first), x and T from the prompt; write the answer's digits right-aligned | designed | designed | designed |
| loop T times | designed | designed | designed |
| square the state | designed (integer product) | designed | designed |
| long-division wiring: start from the dividend's top digits v // 10^8 (below N), shift in one digit per step, 8 steps per reduction, 9 register places, all 10 trials at once | designed | designed | designed |
| the multiples s·N that trial s subtracts | designed | designed | designed |
| **the subtractor cell** (r, m, borrow in) → (digit, borrow out): one 200-row table shared by every place, trial, step and width | designed (exact table) | **learned** (2,400 logits: 200 rows × (10 digit + 2 borrow)) | **learned** |
| quotient digit: the largest trial whose final borrow is 0 | designed rule | designed rule reading the learned borrows | same |
| estimator: a table over the top 3 places of 10R + v_t and the top 2 digits of N → trial (training only) | – | learned, used only in the bootstrap | – |
| what teaches the cell | – | made-up divisions q·N + x from each row's N and x (GRAY) | label-free checks on the rows' own numbers, then the rows' answers |
| adder in the training check | – | – | designed |

Evaluation snaps the learned table to its argmax and runs the whole machine on integers.

## 3. Stage 1: how the table learns, and what broke on the way

### 3.1 The recipe that certified
The starting point was TINYCORE's recipe ([TINYCORE.md](../history/tinycore/TINYCORE.md)), adjusted by microlab's test
of it ([FINDINGS_quotient_shared.md](../history/microlab/FINDINGS_quotient_shared.md)):
- **Fixed slot meanings.** One shared subtractor; trial s subtracts the exact digits of s·N.
- **Lessons (gray).** For each training row, a made-up dividend v = q·N + x with q a random k-digit number. Since x < N
  its remainder is x, which the prompt contains, so no label is used. The datapath starts from v // 10^k and divides in
  k steps.
- **Phase A, bootstrap.** Stage k = 1. The choice is the estimator's (soft), from the first step. The remainder's lowest
  place weighs 5× the others in the loss (§3.4).
- **Hand-off.** As soon as the hard machine (argmax table, own-borrow choice) is ≥ 0.99 exact on the stage's lessons over
  20 steps. Adam's state is reset.
- **Phase B, the own path.** Each step's trial is chosen by the hard machine. The soft table is trained along the
  chosen trials only. Stages k = 2…8 (each ends at ≥ 0.99 exact over 20 steps). At k = 8 the batch's real rows with
  T ≤ 2 join: the dividend is the machine's own s_{T−1}², the answer is the row's label.
- **Freeze.** Learning rate 0 after 200 consecutive steps in which every full-size lesson (512 per step) is exact.
- **Loss.** Cross-entropy of the soft remainder's digits below N's width. Adam at 0.05 on logits initialized N(0, 0.1²).

On the official run (seed 74): hand-off at step 164, stages 2–7 at 20 steps each, k = 8 from step 284, frozen at step
484. The remaining 19,516 steps are frozen and cheap.

### 3.2 Scoring only the remainder's digits
The first version scored all 9 register places. In the uniform-choice phase, the over-subtracting trials always write
9s above N's width, where the answer has 0s. That is a consistent wrong lesson, and it pushed row (0, 0, borrow 1) to
digit 0 (truth 9). Fixed by scoring only the digits below N's width, as TINYCORE does.

### 3.3 Soft choices teach biased lessons: train along the own hard path
With the soft own-borrow choice in phase B (TINYCORE's recipe), row (0, 0, 1) stayed at digit 0. A gradient
decomposition after 1,100 local steps (stage k = 6, 1,024 lessons from rows of every width), on the gradient of row (0, 0, 1)'s logits for digits 0 and 9 (a negative entry is pushed up):

| N's width | true quotient digits forced | soft own borrow |
|---|---|---|
| 4 | toward 9 (d0 +1.1e-2, d9 −1.1e-2) | toward 0 (−2.0e-2, +1.9e-2) |
| 5 | toward 9 | toward 0 |
| 6 | toward 9 | toward 9 |

A soft choice puts some weight on wrong trials. Their lessons are correlated with the answer, not random: where 10R +
v_t has a 0, the answer's digit there is often 0. Hence the fix of §3.1: phase B follows the hard machine's choices,
so when they are right the lesson is the true-digits lesson. It is also ten times cheaper (one trial per step).

Two more things broke here:
- **Jumping to full size.** Jumping straight to k = 8 after the hand-off, with a soft own borrow and a table that is right
  but not yet sharp, gave a soft loss of 3.98 while the hard machine was exact, and its gradients wrecked the table
  (137/200 rows).
- **Adam's memory.** Adam's second-moment memory of phase A's large opposite gradients slowed the repair. Hence the reset
  at the hand-off.

### 3.4 The swapped borrow code: why the first official run failed
The first recipe used microlab's uniform hold (400 steps of a uniform choice before the estimator). It certified on CPU
seeds 2, 4 and 5 and failed the official run: hard exact 0 at every stage, 8/200 rows right.

The failing table is a perfectly consistent alternative subtractor: digit r − m − (1 − c), with the two borrow symbols'
meanings exchanged. 200/200 digit rows and 198/200 borrow rows match it. It computes Rp − s·N − 1. It differs from true
subtraction only at place 0, whose borrow-in is fixed. The own-borrow reading then picks the wrong trial every time.

Loss of the phase-A objective (nats per scored digit) under exact tables held at 0.99 confidence, 2,000 lessons:

| choice | k | true table | swapped table |
|---|---|---|---|
| uniform | 1 | 1.914 | 1.954 |
| uniform | 2 | 2.280 | 2.279 |
| own borrow | 1 | 0.117 | 5.381 |

A uniform hold cannot tell the two codes apart, so the seed decides. The repairs, tried locally (local harness on H1
training rows, batches of 256, 500–700 steps, phase A at k = 1). In every row that failed, the failing seeds settled on
the swapped code:

| phase-A variant | seeds tried | swapped code (failed) |
|---|---|---|
| uniform hold 400 steps, then the estimator (the first recipe) | 1, 2, 3 | 1, 3 |
| half of the choice weight on the own borrow | 1, 2, 3 | 1, 3 |
| every row's borrow output starting biased toward "no borrow" (logit +2) | 1, 2, 3, 4 | 1, 3, 4 |
| the own borrow from step 1 | 1–6 | 1, 3, 6 |
| the estimator from step 1 | 1, 3, 6 | 3, 6 |
| scoring places one at a time from the lowest | 1, 2, 3 | none, but no hand-off within 600 steps |
| **the estimator from step 1 + weight 5 on the remainder's lowest place** | **0–14** | **none** (hand-off at steps 195–213, 200/200 rows) |

That last row is the recipe of §3.1. It then certified on 8/8 CPU seeds through the evaluator and on the official run.

### 3.5 Audit
At its first evaluation batch the submission prints its snapped table, how often training used each row, and which
rows evaluation uses (`TABLE` and `AUDIT` lines). `summarize.py` compares the table with true subtraction outside the
submission. Official Stage 1 run:
- **The snapped table** equals true subtraction on all 200 rows, digits and borrows.
- **Training's hard checks** (every trial of every lesson, both phases) used every row at least 201,374 times (median
  330,249).
- **Evaluation**, at T up to 64 with 8-digit N included, used all 200 rows: `AUDIT eval_rows_used=200/200
  used_but_never_trained=0`.

The table is position-generic, so the 8-digit moduli never seen in training use the same 200 rows as 4–7-digit ones.

## 4. Stage 2: removing the made-up dividends

The made-up dividends q·N + x are Stage 1's main gray area. H1's own rows rarely contain easy dividends. x²/N has a
quotient of 4.4 digits on average over the training rows (at most 6). Of the 1,000 T = 1 rows, the only ones whose
division is labeled, 20 have at most one quotient digit, 66 at most two and 198 at most three. Two routes were tried:
learning from the rows' answers only, and learning from label-free checks on the rows' own numbers.

### 4.1 The rows' answers only: the bootstrap starves
`LESSON = "rows"`:
- **Lessons.** Stage k trains on the batch's T = 1 rows whose dividend x² has at most k quotient digits, from x² // 10^k
  in k steps, against the label.
- **Schedule.** Otherwise Stage 1's.
- **Side channel.** Here the schedule reads the labels' exactness through the loss.

At batch 512 (2 seeds) the hard machine was 0.000 exact at every stage for 20,000 steps; the table ended with 6 and
10 of 200 rows right. With about 5 T = 1 rows per batch, and 20 rows in the whole split with a one-digit quotient,
stage 1 sees about 1 lesson every 10 steps. Stage 1's bootstrap needed tens of thousands of one-digit lessons. Each stage
timed out at 1,500 steps, so the model reached k = 8 untrained. At batch 4096 (about 40 T = 1 rows per step) it is no
different. Seed 1's live log at step 18,750: hard exact 0.000 at every stage, phase B forced at step 12,000, loss 2.25.
The labeled divisions are simply too few and too long to start the bootstrap.

### 4.2 Label-free checks on the rows' own numbers (`LESSON = "identity"`)
- **Phase A: the identity.** Every trial of the machine's own division of each row's x² checks itself: (Rp − s·N) + s·N
  = Rp, digit by digit. The subtraction uses the learned cell, the addition a designed adder, and Rp is the machine's own
  register at each of the 8 steps (512 rows × 8 steps × 10 trials = 40,960 checks per step). Label-free, and nothing is
  made up: the inputs are the rows' x and N and what the machine does with them.
- **Phase B.** Two lessons join once the identity checks out on 80% of the trials, or after 300 steps:
  - **"x mod N = x".** x < N, so dividing x by N must keep trial 0. One division step from x // 10 with the soft
    own-borrow choice. Label-free.
  - **The rows' answers along the own hard path.** Rows with T ≤ 2, as in Stage 1.
  - Adam's state is reset at the hand-off.
- **Freeze** after 200 consecutive steps in which every identity check and every x mod N = x is exact.

What broke on the way:
- **The identity alone stalls on a saddle.** Rows (r, r, borrow 1), true digit 9 and borrow 1, sit at digit 0 or 9
  50/50 with borrow 0. The designed adder's carry makes up for a wrong digit. For row (0, 0, 1), counting wrong places on
  a typical check:

  | (digit, borrow) | wrong places |
  |---|---|
  | (9, 1), the truth | 0 |
  | (0, 0) | 1 |
  | (9, 0) | 1 |
  | (0, 1) | 2 |

  So neither half can move alone. On Modal CPU, the identity alone for 4,000 steps ended at 199/200 rows (only (0, 0, 1)
  wrong). That certifies nothing: with that borrow wrong, every over-subtracting trial looks valid.
- **"x mod N = x" fixes the saddle, if it comes in at the right time.**
  - From the start it biases every borrow toward 1 and breaks many rows.
  - As an 8-step division its soft own-borrow chain wrecks the table (69/200 rows).
  - As one step after the identity's plateau it works: 200/200 rows.
- **The swapped borrow code traps the identity too.** It fails the check only at place 0. Three of four local seeds, and
  the first official Stage 2 run, landed there (identity check at 0.06, real rows 0% exact). That run was cancelled at
  step ~6,000; its log is `history/runs/learned_remainder/s2_idans_official_v1_cancelled.log`. Weighting the lowest
  place 5× in the check moves those seeds to the true code.
- **The identity's plateau varies by seed** (0.79–0.86), so phase A ends at 80% or after 300 steps, whichever comes
  first.
- **There are more such stalls than (r, r, 1).** Under the first recipe, the identity-only run on seed 1 (no answers)
  ended with only row (9, 9, borrow 0) wrong, having checked out on 97.7% of the trials for 19,500 steps. The same seed
  certifies with the lowest place weighted. The rows' answers along the own path add a second, direct lesson on such rows,
  which is one reason the default keeps them (REAL_MAX_T = 2).

**Result.** The official run `s2v2_idans_official` certified T = 64 on both profiles:
- every rung 768/768;
- training: phase A ended at step 103 (82% of the checks passing), frozen at step 499, 184 s;
- evaluation: 5.4 s;
- the snapped table equals true subtraction on all 200 rows; training's hard checks used every row at least 281,631
  times, and evaluation used all 200.

On CPU through the same evaluator:
- **Identity + answers.** 8/8 seeds certified (frozen at steps 499–815).
- **Identity only, no labels.** `REAL_MAX_T = 0`, a model that never sees an answer, certified 4/4 (frozen at
  661–715).

So the remainder can be learned here from label-free checks on the rows' own numbers alone. The answers mostly confirm
it.

Identity-mode training has no random numbers in it (every lesson comes from the rows), so the official H100 run and
the CPU run of the same seed took the same path: hand-off at step 103, frozen at 499.

## 5. Gray areas against the rules

The rules are the repository README's §Rules. In the spirit of the brief, flagged rather than argued away.

- **Rule 7, "no hard-coded algorithm in the forward pass".** Bent heavily, as in SpecBit and UnfoldS. The forward pass
  is long division:
  - all designed: the digit I/O, the integer square, the shift-in wiring, the multiples s·N, the "largest trial without
    a borrow" rule and the T loop;
  - learned: only the 200-row subtraction table.

  It is a designed CPU with one learned gate table, the BRIEF's spectrum point (0).
- **Rule 8, "final logits … inside the autograd graph, an unbroken gradient path from the loss to the parameters
  responsible for the prediction".**
  - Evaluation runs the argmax-snapped table on integers, so the final logits carry no gradient.
  - During training the forward returns zero logits. The custom loss reaches the table through the soft datapath (phase
    A) and through the soft cell along the own hard path (phase B).
  - SpecBit and UnfoldS bridge this with a probability or straight-through term in the readout. Here no such term was
    added; it would be cosmetic.
- **Rule 14, "no data augmentation, task-specific solvers".**
  - *Stage 1:* the lessons are dividends made up inside the forward pass (q·N + x, with q random), whose remainders are
    known by construction. That is data augmentation with a designed multiplier, the main gray area.
  - *Stage 2:* removes the made-up numbers, but its check uses a designed exact adder, a piece of solver arithmetic
    inside the loss, and the fact "x < N, so x is its own remainder".
  - *Both stages:* the multiples s·N are computed by designed arithmetic.
- **Rule 4, the evaluator owns the loop.** The data order and outer loop are untouched, but the schedule lives inside
  the model as Python state:
  - stages;
  - the hand-off, triggered by the hard machine's exactness on label-free lessons;
  - the freeze (learning rate 0) after 200 perfect steps, which amounts to early stopping on a label-free self-check;
  - the Adam reset at the hand-off.

  The loss only logs; it does not feed the schedule, except in the rows-only experiment (§4.1).
- **Weighting the remainder's lowest place 5× (both stages)** is a loss choice made after seeing a failure. It encodes
  the knowledge that place 0's borrow-in is the one fixed input.
- **Snapping to argmax at evaluation** (also SpecBit's choice).

## 6. Next pieces to remove

In rough order of how cheap a certification-gated attempt looks:
1. **The estimator.** Stage 2 already does without it. Stage 1 still needs it: a uniform choice instead lets half the
   seeds settle on the swapped code.
2. **The multiples s·N.** Trial s could be s successive subtractions of N with the same learned cell (TINYCORE's
   "repeat"), so that every arithmetic step of the reduction is the learned table. It costs 9 sequential subtractions
   per step at evaluation, and soft chains of them blurred in TINYCORE's tests. Stage 2's identity on single
   subtractions plus the own path would sidestep that.
3. **The designed adder in Stage 2's check.** A learned adder pinned by its own identities: TINYCORE warns that
   mutually inverse wrong pairs then satisfy (a − b) + b = a unless something anchors them.
4. **The "largest trial without a borrow" rule.**
5. **The squaring.** microlab's column-sum cells learn multiplication as a rule (LADDER_MATCHED.md), but
   squaring an 8-digit state for unseen 24-bit moduli is wider than anything trained, and §2 of REPORT.md says the
   labels cannot teach arithmetic upstream of a reduction.
6. **The long-division wiring, the loop counter and the I/O.**

## 7. Reproduce

Everything heavy runs on Modal; the runner must be launched from the one-layer-deeper repository root (it builds the
competition image from it and regenerates H1 inside with the official generator arguments). Results land on the Modal
Volume `learned-remainder`; `wait_fetch.py` downloads them to `cpu_first/history/runs/learned_remainder/<name>.json`
(full log, RESULT_JSON, the exact submission source).

Stage 2, the final file as it is:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_learned_remainder.py --mode official --name final_s2_official
```

Stage 1:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_learned_remainder.py --mode official --name final_s1_official --set '{"LESSON": "\"madeup\""}'
```

Stage 0:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_learned_remainder.py --mode official --name s0_official --set '{"MODE": "\"designed\"", "MAX_STEPS": "100"}'
```

A plan of CPU runs (one container each):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_learned_remainder.py --mode plan --plan cpu_first/learned_remainder/plans/p7_cpu.json
```

Fetch records, then tabulate them with the table audit:

```bash
~/github/sotaku/venv/bin/python cpu_first/learned_remainder/wait_fetch.py final_s2_official final_s1_official
```

```bash
~/github/sotaku/venv/bin/python cpu_first/learned_remainder/summarize.py --table s0_official s1v2_official s2v2_idans_official final_s2_official final_s1_official
```

Plans (`plans/*.json`), each set on top of the file as it was at launch:
- p1: the first Stage 1 recipe, and Stage 0, on CPU;
- p2 and p6: the first Stage 2 recipe;
- p3: Stage 1 on 8 seeds;
- p4: rows only;
- p5: identity without "x mod N = x";
- p7: Stage 2 on 8 + 4 seeds.

Every record keeps the exact source it ran, so a rerun against the final file must set `LESSON` where the plan relied
on the default of its day (p1 and p3 ran Stage 1 by default).

Files:
- `submission.py`: the submission, certified byte for byte as `final_s2_official`. Its default is Stage 2. Switch with
  `LESSON = "madeup"` for Stage 1 (`final_s1_official`) or `MODE = "designed"` for Stage 0.
- `modal_learned_remainder.py`: the Modal runner.
- `wait_fetch.py`: downloads records from the Volume.
- `summarize.py`: the tables and the table audit.
