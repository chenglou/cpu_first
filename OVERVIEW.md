# One Layer Deeper H1, from the CPU up: where everything stands

*The entry point for `cpu_first/`, written 2026-10-21 at about 03:45 Pacific.*
- *What was checked against the run records (`runs/`, `ablation/runs/`): every certification, run time, freeze step,
  parameter count and CPU-seed count quoted here. The source embedded in each record was hashed and compared with the
  file's md5.*
- *The data counts in §1–2 were computed from the released H1 files, and the commits come from `git log`.*
- *The one-line findings of the timeline (§3) and of the ablation (§7) summarize their write-ups, which are linked;
  they were not re-run.*
- *The older write-ups are left as they were. §9 lists where a write-up and a record disagree.*
- *Three sessions were still working when this was written (§7), so their folders may have moved on.*

**In one minute.**
- The task is exact repeated modular squaring. Almost every entry on the leaderboard scored near 0%, because the
  remainder erases partial credit: an almost-right model gets random-looking answers, so gradients have nothing to
  follow (§1–2).
- Designs built like a CPU certify: discrete digits, one small cell reused everywhere, and labels that choose between
  whole machines. Four lines of work certified through the official evaluator: SpecBit, UnfoldB, UnfoldS and the
  learned digit machine (§3).
- The current best model is a paper-arithmetic calculator. Every part of it that computes is a small learned table,
  5,079 numbers in all, and it certifies T = 64 on both ladders on the official H100 manifest and on 8 of 8 CPU seeds
  (§4).
- It got there by removing hand-written parts one at a time, with certification as the gate at each step (§5).
- Still hand-written: the wiring, a short warm start, and the lessons. Work on the wiring and the warm start is under
  way (§7).

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

### Oct 21: one model, then simplification and learned wiring

| when | what | write-up |
|---|---|---|
| 00:55 `9959997` | **every learned piece in one model**, with a prediction path inside autograd | [COMBINED.md](learned_remainder/COMBINED.md) |
| 00:55 `9332360` | wiring and tables learned together in tiny machines | [tinyrouting/TINYROUTING.md](tinyrouting/TINYROUTING.md) |
| overnight, not committed | an ablation of the training recipe; a simplified recipe certified | [ablation/ABLATION.md](ablation/ABLATION.md) |
| overnight, not committed, still running | the division's wiring, then the squaring's layout, learned and certified | [learned_wiring/WIRING.md](learned_wiring/WIRING.md) |

### The lessons that came out of the failures

- **Generic networks memorize instead of computing the remainder**, and on a new modulus they are at chance.
- **Soft mixtures of choices learn nothing here.** After the remainder, the loss stays flat until the wrong options
  hold almost no weight.
- **Answers alone, pushed through a learned machine, do not teach it.** Untrained parts get bypassed, or trial slots
  co-adapt into a lookup arrangement.
- **A check made only of the learned table can be passed by whole families of wrong tables**: digits relabeled,
  borrows swapped, "copy the first number". Meaning has to come from outside the table: a trusted adder for a short
  while, and true facts about the row's numbers such as x < N ([ABLATION.md](ablation/ABLATION.md) §7).
- **What worked every time:** digits snapped to exact values, one shared cell, checks on each training row's own
  numbers that need no answer, and pass/fail scoring of whole choices.

## 4. The current best model

**[`learned_remainder/submission_combined.py`](learned_remainder/submission_combined.py)**: md5 `4f5cecc780894499a462499c3cd7d67f`,
committed in `9959997`. Write-up: [COMBINED.md](learned_remainder/COMBINED.md).

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

**What is still hand-written** ([COMBINED.md](learned_remainder/COMBINED.md) "In plain words"):
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

**Further along, but not committed.** The learned-wiring stages (§5, 6b–6c) learn the division's switches and the
squaring's layout on top of this model. Both certified officially, and they may still change. The ablation's
[`submission_simple.py`](ablation/submission_simple.py) certifies with a much shorter recipe, for the single-piece
machine of ADDER.md, not for this one.

## 5. The removal ladder

Each step removed one hand-written piece. To count, it had to certify on the untouched official manifest on an H100,
with the file run byte for byte: T = 64 on both ladders, every rung 768/768. Then it was repeated on CPU seeds
through the same evaluator.

Steps 4a–4d each started from step 3's machine, separately. Step 5 put them together.

| step | what became learned (the hand-written piece removed) | file (md5) | official record | training s (frozen at step) / evaluation s | CPU seeds certified | commit |
|---|---|---|---|---|---|---|
| 0 | nothing: the designed subtraction table, a plumbing check | an earlier `learned_remainder/submission.py` (`9732a11f`) | `s0_official` | 3.3 (100 steps) / 5.5 | 1 of 1 | `824d073` |
| 1 | the 200-entry subtraction table, taught with made-up practice divisions (gray under rule 14) | `submission.py` with `LESSON = "madeup"` | `final_s1_official` | 154.5 (484) / 4.8 | 8 of 8 (`s1v2` recipe) | `824d073` |
| 2 | no made-up numbers: the table checks itself on the rows' own numbers | [`submission.py`](learned_remainder/submission.py) (`bd477b9a`) | `final_s2_official` | 181.9 (499) / 4.9 | 8 of 8 (`s2v2` recipe), and 4 of 4 that never see an answer | `824d073` |
| 3 | the multiples of N: repeated subtraction with the learned table | [`submission_chain.py`](learned_remainder/submission_chain.py) (`ed360cb1`) | `chain_v2_official` | 271.8 (417) / 15.4 | 8 of 8, and 4 of 4 | `e1dd000` |
| 4a | integer squaring: a learned 100-entry times table, schoolbook | [`submission_square.py`](learned_remainder/submission_square.py) (`37765051`) | `square_official` | 270.5 (499) / 20.5 | 8 of 8, and 4 of 4 | `c0c86a2` |
| 4b | the integer loop over T and the stop rule: a countdown on the learned table, and a learned 2 × 2 gate | [`submission_control.py`](learned_remainder/submission_control.py) (`ec79a312`) | `control_official` | 239.7 (469) / 12.8 | 8 of 8, and 4 of 4 | `c6c9597` |
| 4c | the hand-written adder in the self-check, after a 70-step warm start | [`submission_adder.py`](learned_remainder/submission_adder.py) (`44e658ab`) | `adder_warm_official` | 77.8 (343) / 12.8 | 8 of 8, and 4 of 4 | `550f60e` |
| 4d | hand-written prompt parsing and answer writing: a learned reader and writer | [`submission_io.py`](learned_remainder/submission_io.py) (`45cc6e1b`) | `io_official` | 220.0 (440) / 12.8 | 8 of 8 | `f342a91` |
| 5 | all of 4a–4d in one model, predicting inside autograd | [`submission_combined.py`](learned_remainder/submission_combined.py) (`4f5cecc7`) | `combined_official` | 666.6 (545) / 48.2 | 8 of 8 | `9959997` |
| 6a | most of 4c's training recipe: three facts and the check remain; no answers, phases or freeze | [`ablation/submission_simple.py`](ablation/submission_simple.py) (`f374ec70`) | `simple_official` | 300.7 (3,000 steps, no freeze) / 23.1 | 8 of 8 (a bit-identical twin), and the file itself 4 of 4 | not committed |
| 6b | on top of 5: the division's wiring, 7 switches (864 wirings) | `learned_wiring/submission_div_v1.py` (`a0050a2f`); fixed version `submission_div.py` (`11cccf13`) | `div_official`; fixed: `div2_official` | 393.4 (640) / 48.5; fixed: 313.6 (615) / 36.8 | first version: 7 records, all certified (§9); fixed version: running | not committed |
| 6c | plus the squaring's layout (576 layouts) | `learned_wiring/submission_sq.py` (`781928ea`) | `sq_official` | 377.7 (479) / 49.0 | 3 records so far (74, 3, 6), all certified | not committed |

Notes:
- Training time includes the steps after the freeze, when the learning rate is 0. "Never see an answer" means the same
  file with one constant changed (`REAL_MAX_T = 0`).
- Official records live in `runs/learned_remainder/<name>.json`, `ablation/runs/` and `runs/learned_wiring/`. Each
  holds the full log, the result and the exact source.
- Records of failures are kept too:
  - `s1_official`: Stage 1's first recipe, with 8 of 200 rows right and no rung certified;
  - `adder_official`: a learned adder from a random start with no warm start, with no rung certified.
  `chain_official`, the chain's first recipe, certified, but one of its CPU seeds that never sees an answer did not.

## 6. Rules 1–16 for the best model

The rules are [one-layer-deeper/README.md §Rules](../one-layer-deeper/README.md#rules). "Gray" means a reasonable
reader could object; the point is flagged, not argued away. This follows COMBINED.md's own table, with the imports,
the initialization and the absence of backward calls re-checked in the source.

| rule | status | why |
|---|---|---|
| 1. one `submission.py` exporting one `Submission` | yes | the runner sends the file as `submission.py`; it has one `SUBMISSION` with model, optimizer, loss, batch sizes and a step cap |
| 2. self-contained | yes | it imports only `torch`, `itertools`, `collections` and the public `benchmark` API; no files, no network |
| 3. the participant defines model, optimizer, loss, batch sizes, steps | yes | Adam; the rounds run until no prompt wants another (adaptive computation, capped at 100) |
| 4. the evaluator owns the loop, data order, backward, clipping, cadence, seeds, deadline | **gray** | the evaluator is untouched: one pass per step, no batch reuse. But the schedule lives in the model as Python state: the reader's stages, the phases, when each table joins, the Adam resets, and a freeze that sets the learning rate to 0 once every check has passed for 200 steps. That freeze is early stopping on the model's own checks |
| 5. at most 500M parameters | yes | 5,079 |
| 6. no hard-coded weights | yes | all 10 tensors start at 0.1 × random normal and are trained. The "designed" mode, with the tables at the truth, is only a plumbing check, never the submission |
| 7. no hard-coded algorithm in the forward pass | **gray, the main one** | every product, difference, sum, borrow, count and keep-or-discard decision, and every symbol read or written, is a lookup in a learned table. But the forward pass is a designed CPU: the paper layouts, the countdown register and the number 1, the keep-or-discard select, "rounds until none commits", and the reader's grammar. The learned-wiring work moves the layouts into learned switches, but their menus are designed and contain the true wiring |
| 8. final logits from the model inside autograd, with an unbroken gradient path | yes, with gray edges | the prediction path above. Gray: (a) in training the forward returns zero logits, and the tables learn through the lessons, not through the final logits; (b) the gradient is damped 10× per subtraction link, addition and round so it does not overflow, and on T = 64 prompts it underflows to 0 for T's anchor and step and for x's step; (c) which reading hypothesis a prompt uses is decided by designed run-detection logic, with a soft surrogate on the gradient side |
| 9. everything on the GPU | yes | state and computation stay on the device. The schedule reads a few scalars to Python per step, and evaluation syncs one boolean per round to end the loop |
| 10. OOM or timeout fails | ok | neither happened |
| 11. 3,600 s of training, compilation included | yes | 666.6 s, no compilation |
| 12. a custom loss returns one finite differentiable scalar | yes | `token_training_loss` |
| 13. evaluation within 1,800 s; certification | yes | 48.2 s; T = 64 on both ladders |
| 14. no data inspection, augmentation, task-specific solvers, custom loops or participant backward passes | **gray, the other main one** | there is no `backward` or `autograd.grad` call in the file, no custom autograd function, and no made-up numbers. But the lessons carry designed knowledge: the hand-written adder during the warm start; facts about the inputs (x < N, N > 0, and "a wrapped x − N is large", which assumes the register's width); the times-table ladder, which is the definition of multiplication; how decimal comparison works (the deciding digit), used to learn the digit order; the reading and T's direction, chosen inside the loss by trying every complete hypothesis; and only answers of T ≤ 2 rows whose own checks pass. The sizes (8, 8 and 2 digits in, 8 out) come from the task's spec |
| 15. repeated rule-breaking | n/a | the write-ups flag their own gray areas, and nothing has been submitted to the hosted evaluator |
| 16. do not exploit the Hard metric recorder | yes | not touched |

**The earlier certified models, in one line each.**
- **SpecBit**: only one gate's truth table is learned, and the netlist around it is the algorithm. Under a strict
  reading of rule 7, the design panel's critics rated every one of the panel's designs fatal
  ([REPORT.md](REPORT.md) §6).
- **UnfoldB and UnfoldS**: 4 learned numbers, with the snapping, the remainder and the loop built into the model.
  Their notes discuss rules 7, 8 and 14 ([unfold_sub/NOTE.md](unfold_sub/NOTE.md),
  [unfold_small/NOTE.md](unfold_small/NOTE.md)).

## 7. What is running now, and the open questions

As of about 03:45 on Oct 21, none of this work is committed.

**Ablation of the training recipe** ([ablation/ABLATION.md](ablation/ABLATION.md), complete as written at 03:24;
262 run records in `ablation/runs/`).
- In the single-piece recipe (step 4c), only 2 of 21 ingredients break training when removed alone:
  - the 70-step warm start with the hand-written adder (0 of 4 seeds without it);
  - the wrap-around fact, "(x − N) − N does not borrow" (0 of 4).
- Two harmless-sounding lessons, R − 0 = R and R − R = 0, planted the very traps that later lessons had been added to
  repair.
- The simplified recipe certified (step 6a).
- In the combined model the picture shifts:
  - the wrap-around fact can go (4 of 4), because the warm start runs longer there;
  - the learned-adder check becomes essential (0 of 2 without it).

**Learned wiring** ([learned_wiring/WIRING.md](learned_wiring/WIRING.md), being written, with placeholders in ⟨ ⟩).
- Stage 1 (the division's 7 switches) certified officially. The session reports that the first version failed one
  CPU seed (seed 7, no record yet): its untrained reader read every N as 0. A fixed version certified officially too
  (`div2_official`), and its CPU seeds were running.
- Stage 2 (plus the squaring's layout) certified officially, and its CPU seeds were coming in.
- Stage 3 (one shared shape switch for the reader and writer) had its official run going. WIRING.md lists two fixes
  found on the way: a one-digit x has no reading direction, and a rewritten writer lesson had lost a side effect.

**Digit anchors** (the session "Anchor digit meanings without the hand-written warm start"). It was running. At 03:41
its folder [`anchors/`](anchors/) held only `core.py`, a tiny core of the digit-meaning problem (the chain machine in
any base, with its lessons), and no write-up yet. Its target is the warm start: the ablation found it load-bearing,
and it is the last hand-written arithmetic in training.

**Open questions.**
1. **The warm start.** Can digit meanings come from somewhere other than a hand-written adder? Checks made only of the
   table cannot supply them (ABLATION.md §7, Rule 1).
2. **The wrap-around fact.** Can it be replaced by something that does not assume the register's width? One
   replacement, rippling two lessons with the machine's own borrows, did its job on 3 of 4 seeds inside the full
   recipe, but not alone (ABLATION.md §8).
3. **The wiring menus.** How large can they grow before the machine discovers the algorithm instead of confirming
   it? Today they contain the true wiring, and the sizes, the T loop and the adder's composition a − (0 − b) are fixed.
4. **Rule 8.** Training never sends the loss through the final logits, and the prediction path's gradient is damped
   and underflows deep in long runs. Is that "an unbroken gradient path"?
5. **Rule 14.** Is a self-check on the rows' own numbers "data augmentation" or a "task-specific solver"? A strict
   reviewer should decide before any real submission.
6. **The hosted evaluator.** Submitting after the deadline, just to see a response, has not been done.

## 8. Where everything is

| path | what |
|---|---|
| [`OVERVIEW.md`](OVERVIEW.md) | this file |
| [`README.md`](README.md) | one line for every file and folder |
| [`REPORT.md`](REPORT.md) | the first-day report: the analysis (§0–8), SpecBit and the grammar search (§9–14) |
| [`learned_remainder/`](learned_remainder/) | the removal ladder, steps 0–5: every `submission*.py`, its write-up (LEARNED_REMAINDER, CHAIN, SQUARE, CONTROL, ADDER, IO, COMBINED), the Modal launchers, plans and local checks |
| [`ablation/`](ablation/) | which training ingredients matter, and `submission_simple.py` (another session's folder, not committed) |
| [`learned_wiring/`](learned_wiring/) | the wiring learned in stages (another session's folder, not committed, still changing) |
| [`anchors/`](anchors/) | digit meanings without the hand-written warm start (another session's folder, just started, not committed) |
| [`tinyrouting/`](tinyrouting/) | wiring and tables learned together in tiny machines |
| [`tinycore/`](tinycore/) | the quotient-digit chicken-and-egg, computed exactly |
| [`microlab/`](microlab/) | a small harness for one-minute experiments, and its findings ([microlab/README.md](microlab/README.md)) |
| [`unfold_sub/`](unfold_sub/), [`unfold_small/`](unfold_small/) | the UnfoldB and UnfoldS submissions |
| [`submission.py`](submission.py) | SpecBit |
| [`panel/`](panel/), [`panel_synthesis.md`](panel_synthesis.md) | the five-design panel and its synthesis |
| [`FINDINGS_substrate.md`](FINDINGS_substrate.md), [`RELAXED.md`](RELAXED.md), [`UNFOLD.md`](UNFOLD.md), [`UNFOLD_FAMILY.md`](UNFOLD_FAMILY.md) | the day-1 topic write-ups |
| `runs/` | the run records. Official ones: `runs/learned_remainder/*_official.json`, `runs/learned_wiring/`, `runs/unfold_sub/official_h100.json`, `runs/unfold_small/official_h100.json`, `runs/h100_h1.json`; the ablation's are in `ablation/runs/` |
| `../one-layer-deeper/` | the competition clone at `87e5c26` ("Releasing H1"), with the H1 data under `data/generated/` |
| `~/github/handwired-lab/montgomery-search` | outside this repo: search over a Montgomery squaring program in the handwired project (branch `lab/montgomery-search`) |

**The certified submissions, with their records.** Steps 0–6c are in the §5 table. The earlier three:

| model | file (md5) | official record | training s / evaluation s | learned numbers |
|---|---|---|---|---|
| SpecBit | [`submission.py`](submission.py) (`a432bb5e`; the record embeds no source) | `runs/h100_h1.json` | 16.5 (20 steps) / 26.9 | 65,536 (one logit per possible gate) |
| UnfoldB | [`unfold_sub/submission.py`](unfold_sub/submission.py) (`17299657`) | `runs/unfold_sub/official_h100.json` | 384.3 (1,000 steps, batch 100,000) / 3.7 | 4 |
| UnfoldS | [`unfold_small/submission.py`](unfold_small/submission.py) (`eb6aedfc`) | `runs/unfold_small/official_h100.json` | 1,624.9 (200,000 steps, batch 512) / 3.7 | 4 |

All three reached T = 64 on both ladders, with every rung 768/768 and every split exact.

**How to rerun.** Each write-up ends with its commands. The official runs of the ladder go through
`learned_remainder/modal_chain.py --file <submission>`, launched from the competition clone. It builds the
competition's image, regenerates H1 with the official generator, and runs the untouched manifest.

## 9. Where docs and records disagree

What was checked, for every number above:
- each official record's certified T, rung counts, split counts, training and evaluation seconds, parameter count and
  freeze step, against the write-ups that cite them;
- the md5 of the source each record embeds, against the files. Where they differ, the difference is the one `--set`
  line the write-up names;
- the CPU-seed claims, by tallying the records.

The numbers checked this way all match their records in the committed write-ups. Those numbers are each piece's
official training and evaluation times, freeze steps, parameter counts and CPU-seed counts, in LEARNED_REMAINDER,
CHAIN, SQUARE, CONTROL, ADDER, IO and COMBINED. The same goes for the two Unfold notes and REPORT.md §9. The
exceptions:
- **WIRING.md contradicts itself on stage 1's CPU seeds.** Its prose says the first version failed one of 8 seeds
  (seed 7), but its generated table says "7/7 (1 running)". `runs/learned_wiring/` holds 7 certified records (74 and
  1–6) and none for seed 7, so the failure is the session's report, not yet a record. The write-up is in progress.
- **COMBINED.md calls `combined_official` and `combined_index_official` "the same training run".** They are two runs
  with identical training: the same file apart from the `PATH` line, the same seed and data order, and both froze at
  step 545 with the same tables. They were launched together, and their training times differ (666.6 s and 566.5 s).
- **The leaderboard count** (3 of 146 certified, REPORT.md §0) has no record behind it and was not re-checked.
