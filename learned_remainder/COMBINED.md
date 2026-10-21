# H1 with every working part learned, in one model, predicting inside autograd

## In plain words

**The task.** The model reads three numbers written as digits in a prompt: a modulus N, a starting number x and a count
T. It must square x, keep only the remainder after dividing by N, repeat that T times, and write the result's digits.
To be certified it must get every one of 768 fresh questions exactly right at T = 1, 2, 4, …, 64, both for moduli it
saw in training and for larger moduli it never saw (up to 8 digits). Training only ever shows T = 1, 2, 4, 8 and 16.

**What the one model is.** A small calculator, wired the way a person does this arithmetic on paper. Every part of it
that computes something is a small learned table:
- a **reader** that finds the three numbers in the prompt, decides which way to read each one, and decides which
  symbol is which digit;
- a **times table** (each digit times each digit gives a low and a high digit) for schoolbook squaring;
- a **subtraction table** ("this digit minus that digit, minus a borrow, gives this digit and maybe a new borrow").
  It does every subtraction, and every addition too, through a trick: a + b = a − (0 − b);
- a **two-entry switch** that looks at whether a subtraction went below zero and decides whether to keep its result.
  It runs the long division and the count-down of T;
- a **writer** that puts the result's digits back into the prompt as symbols.

That is about 5,000 learned numbers. All five parts start from random numbers and are learned together, in one
training run, from the training prompts and their answers.

The five parts were first learned one at a time, each in its own file, with the other parts written by hand
([IO.md](IO.md), [SQUARE.md](SQUARE.md), [CHAIN.md](CHAIN.md), [ADDER.md](ADDER.md), [CONTROL.md](CONTROL.md)). This
file puts them together.

**What is learned.** Everything that computes:
- which symbol is which digit, where each number starts and which way to read it;
- every digit product, every digit difference and borrow, and through them every sum;
- the count-down of T;
- every keep-or-discard decision;
- where to write the answer, and with which symbols.

**What is still written by hand.**
- **The wiring.** This is the paper layout:
  - schoolbook squaring: 8 rows, shifted and added;
  - long division: 8 steps, each of up to 9 subtractions of N, on a 9-digit register;
  - the count-down: a 2-digit register, and the number 1 to subtract;
  - the plumbing that keeps or discards a result;
  - "run rounds until no question wants another", with a cap of 100 rounds.
- **The reader's and writer's shape.** Each number is read by "start next to some symbol, step left or right, stop at
  the first symbol that is not a digit". N and x get 8 slots, T gets 2, and the writer writes 8 slots the same way. The
  reader also knows that a prompt holds three runs of digits; it does not know which run is which number.
- **A warm start.** For its first steps the subtraction table checks itself with a hand-written adder. It stops as
  soon as the reader has worked out the prompt, and never before step 70. In the official run that was step 90;
  across all runs it was step 90 to 273. From then on no hand-written adder runs: every sum, difference and product,
  in training as in the answer, is done by the learned tables.
- **The lessons and their schedule.** This covers what each table is checked against, in which order the tables
  join, and when training stops.

**Did it certify?** Yes, on the untouched official H1 manifest on an H100, with the file run byte for byte:
- all 768 questions right at every T up to 64, on seen and on unseen moduli, and every question of the test,
  ood_t and ood_n_t splits right;
- every table exactly right: the subtraction table 200/200, the times table 100/100, the switch the true one, and the
  true reading, codes and writer;
- every table entry used at evaluation was trained;
- the same file certified too when it computes the answer the way the separate pieces did (whole numbers looking up
  each table's best entry, outside automatic differentiation), from the same training run. So did the plumbing check,
  with every table set to the truth;
- the same file certified on 8 of 8 seeds on CPU through the same evaluator (seed 74 and seven more).

**Times.**
- Training: 667 s of the 3,600 s budget. The model is done at step 545, about 450 s in, and then holds still.
  Settling which way to read T (the count-down machine run from both readings) takes most of that.
- Evaluation: 48 s of the 1,800 s budget with the answer computed inside autograd; 24 s the pieces' way.

**Rule 8.** At evaluation the answer is now computed inside PyTorch's automatic differentiation:
- every digit is a vector of ten numbers;
- every table is used as a matrix;
- a standard trick, the straight-through estimator, makes every table act exactly as its best entries, so the
  answers are exactly the calculator's, while a gradient can still flow from the answer back to every table.

A test pushes a loss on the final answer back through the whole calculator. Every one of the ten learned tensors gets
a gradient, and the answer stays exact. To make this work, the gradient had to be scaled down at every step. The
exact calculator's gradient grows about five-fold per subtraction, so unscaled it overflows within one division.

## Rule by rule

The rules are [one-layer-deeper/README.md](../../one-layer-deeper/README.md) §Rules. "Gray" means a reasonable reader
could object; the point is flagged, not argued away.

| rule | status | why |
|---|---|---|
| 1. one `submission.py` exporting a `Submission` | yes | `submission_combined.py` is sent as `submission.py` by the runner; one `SUBMISSION` with model, optimizer, custom loss, batch sizes, step cap |
| 2. self-contained | yes | imports `torch`, `itertools`, `collections` and the public `benchmark` API only; no files, no network |
| 3. participant defines model, optimizer, loss, batch sizes, steps; recurrence and adaptive computation allowed | yes | Adam on all logits; the rounds run until no row commits (adaptive computation, capped at 100) |
| 4. the evaluator owns the loop, data order, backward, clipping, optimizer cadence, seeds, deadline | **gray** | untouched, one pass per step, no batch reuse. But the schedule lives in the model as Python state: the reader's 5 stages, phases A, B, C, when the product cell joins, the Adam resets, and the freeze (learning rate 0 once every check has passed for 200 steps). The freeze is early stopping on label-free self-checks plus the reader's round trips |
| 5. at most 500M parameters | yes | 5,079 |
| 6. no hard-coded weights; random init, updated in training | yes | every tensor starts at 0.1 · N(0, 1) and is trained. `MODE = "designed"` (every table at the truth) is the plumbing check only, never the submission |
| 7. no hard-coded algorithm in the forward pass | **gray, the main one** | every product, difference, sum, borrow, count and keep-or-discard decision is a lookup in a learned table, and so is every symbol read and written. But the forward pass is still a designed CPU: the schoolbook and long-division wiring, the count-down register and the number 1, the keep-or-discard select, "rounds until no row commits", and the reader's grammar (anchor + one step per field, stop at a non-digit, 24 reading hypotheses over digit runs, rank codes) are written by hand |
| 8. final logits produced by the model inside autograd, unbroken gradient path | **yes, with gray edges** | §3. The prediction path is one-hot digits through straight-through tables: tensor operations only, no number held as an integer, and a gradient from the final logits reaches every one of the 10 learned tensors (tested, also on a trained model). Gray: (a) training's forward returns zero logits, so the training loss reaches the tables through the lessons (soft relaxations of the same machine) and not through the final logits; (b) the gradient is the straight-through surrogate, discounted 10× per link, addition and round (otherwise it overflows), so deep into a long run (x's and T's fields at T = 64) it underflows to 0; (c) the hard halves of the straight-through pairs are argmax decisions, and the reader's choice of reading hypothesis is made by designed run-detection logic, with a soft surrogate on the gradient side |
| 9. everything on the GPU | yes | model state and computation stay on the device. The schedule reads a few scalars per step to Python, and evaluation syncs one boolean per round to end the loop: control flow, not offloading |
| 10. OOM or timeout fails | ok | no OOM; training 667 s, evaluation 48 s |
| 11. Hard: 3,600 s of training, compilation included | yes | 666.6 s (no compile), frozen at step 545 |
| 12. custom loss returns one differentiable finite scalar | yes | `token_training_loss` |
| 13. evaluation budget 1,800 s; certification | yes | 48.2 s; T = 64 certified on both profiles, every rung 768/768 |
| 14. no data inspection, augmentation, task-specific solvers, custom loops, participant backward passes | **gray, the other main one** | no `backward`/`grad` call anywhere in the submission, no custom autograd function, no made-up numbers. But the lessons carry designed knowledge: (a) the hand-written adder in the check until the reading is chosen; (b) facts about the inputs: x < N, N > 0, and "x − N wraps to at least N" (the register width); (c) the ladder is the definition of multiplication; (d) Bradley-Terry's decisive place is how decimal comparison works; (e) the reading and T's direction are picked by speculation over complete hypotheses; (f) the answers used are only those of rows with T ≤ 2 whose own count-down, squarings and ladder check out (a filter chosen by T) |
| 15. repeated rule-breaking | n/a | this file flags its own gray areas |
| 16. do not exploit the Hard metric recorder | yes | not touched |

## Details

### 1. Results

**Official runs** (Modal H100, the untouched official manifest `h100_hard_h1.json`, seed 74; records in
`cpu_first/runs/learned_remainder/<name>.json` with the full log and the exact source):

| run | what | max T seen / unseen moduli | rungs | test, ood_t, ood_n_t | tables at evaluation | frozen at step | train s | eval s |
|---|---|---|---|---|---|---|---|---|
| **combined_official** | `submission_combined.py` byte for byte (md5 `4f5cecc7…`), `PATH = "ste"` | **64 / 64** | every rung 768/768 | 9,999/9,999, 10,002/10,002, 10,002/10,002 | subtractor 200/200, product 100/100, gate true, reading true | 545 | 666.6 | 48.2 |
| combined_index_official | the same file with `PATH = "index"` (the pieces' integer prediction path) | 64 / 64 | every rung 768/768 | all exact | the same | 545 | 566.5 | 23.9 |
| combined_s0_official | `MODE = "designed"`: every table at the truth, never trained (plumbing), `PATH = "ste"` | 64 / 64 | every rung 768/768 | all exact | true by construction | (frozen from the start) | 143.0 | 34.3 |

The first two are the same training run: the same file, seed and data order, and training never uses the prediction
path. Both froze at step 545 with the same tables. The task asked for the combination to certify first and for the
rule-8 path second. Both paths were built before the first official run, and the two runs were launched together. The
integer path's record came in a few minutes earlier. The straight-through path, the file as it is, certified too.

**The official run's schedule** (`combined_official`):

| step | what happened |
|---|---|
| 5 | the reader's digit-ness splits every prompt into three digit runs (IO stage 2) |
| 63 | a reading hypothesis passes x < N on 99.9% of rows for 20 steps (stage 3): N from its marker rightwards, x from T's marker leftwards |
| 90 | the reading is chosen (stage 4). From this step the hand-written adder is gone, the learned adder checks the table, and the facts about x and N join |
| 172 | phase B: 80% of the table's checks pass; Adam reset |
| 192 | every check of the table exact for 20 steps: the product cell's ladder joins. "x − N borrows" on 99%: phase C, the gate's lessons and the answers join |
| 448 | T's direction settled (it started on "61") |
| 545 | frozen: 200 steps with every check exact |

Wall time: step 300 at 52 s, step 400 at 214 s, step 500 at 435 s, frozen at about 450 s. The rest of the 666.6 s
is the 19,455 frozen steps. From about step 360 to 448 (inferred from those step times) training runs T's speculation,
about 4 s a step on the H100: the count-down machine runs from both readings of T (up to 61 rounds), and the loop is
launch-bound.

**The audit** (the evaluation's AUDIT lines in `combined_official`; table rows against the truth by
`summarize_combined.py`, outside the submission):
- subtraction table: 200/200 right, 200/200 rows used at evaluation, **0 never trained**. Each row was used by
  training's checks at least 638,616 times (median 1,296,833);
- product cell: 100/100 right, 100/100 cells used at evaluation, **0 never trained** (each at least 79,710 ladder
  lessons);
- count-down: the 24 rows CONTROL.md lists, (0…9, 1, 0) and (0…6, 0, 0/1), **0 never trained**. Each was used by
  training's checks at least 1,512,669 times. At evaluation, 10 of them, the tens digits 2 to 6, are met only by T = 32
  and 64. In training the count-down meets them only in T's speculative run of the wrong reading ("61"), which teaches
  nothing; their lessons come from the division, which shares the table;
- gate: 2/2 rows used and trained (322,713 and 3,071,007 uses), P(commit | no borrow) = 0.9999, P(commit | borrow) =
  0.0008;
- reader: N anchored at token 2 (the N marker) stepping +1; x at token 4 (the T marker) stepping −1; T and the writer
  at the prompt's end stepping −1; digit tokens 7…16 with code and writer both the identity.

**Robustness on CPU** (plan `plans/cb1_cpu.json`: the same file through the same evaluator, manifest with
`device = cpu` and the seed changed):

| seed | max T seen / unseen | rungs | test, ood_t, ood_n_t | subtractor / product / gate / reader | reading chosen | ladder joins | T settled | frozen | train s | eval s | rows used at evaluation never trained |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 74 | 64 / 64 | all 768/768 | all exact | 200/200, 100/100, true, true | 90 | 192 | 448 | 545 | 640.5 | 487.9 | 0 |
| 1 | 64 / 64 | all 768/768 | all exact | 200/200, 100/100, true, true | 120 | 216 | 465 | 568 | 803.4 | 433.4 | 0 |
| 2 | 64 / 64 | all 768/768 | all exact | 200/200, 100/100, true, true | 93 | 193 | 388 | 567 | 669.5 | 493.5 | 0 |
| 3 | 64 / 64 | all 768/768 | all exact | 200/200, 100/100, true, true | 115 | 201 | 463 | 567 | 807.6 | 434.9 | 0 |
| 4 | 64 / 64 | all 768/768 | all exact | 200/200, 100/100, true, true | 273 | 352 | 595 | 714 | 922.7 | 443.0 | 0 |
| 5 | 64 / 64 | all 768/768 | all exact | 200/200, 100/100, true, true | 108 | 188 | 420 | 535 | 616.2 | 466.8 | 0 |
| 6 | 64 / 64 | all 768/768 | all exact | 200/200, 100/100, true, true | 93 | 235 | 515 | 610 | 878.1 | 553.5 | 0 |
| 7 | 64 / 64 | all 768/768 | all exact | 200/200, 100/100, true, true | 144 | 245 | 494 | 627 | 799.9 | 426.6 | 0 |

**8 of 8 certified**, each running the file byte for byte (md5 `4f5cecc7…`). Every record has 9,999/9,999 test,
10,002/10,002 ood_t and 10,002/10,002 ood_n_t. "Never trained" is 0 for the subtraction table, the product cell, the
count-down's rows and the gate on every seed. Seed 74 on CPU retraces the H100 run step for step: no lesson draws a
random number, so the same seed and data order give the same trajectory. Modal lost seed 6's host once ("Worker
disappeared") and reran it from the start. The training times include the frozen steps, 9–15 ms each on 8 CPU
cores. Seed 74 reached step 600 at 466 s on CPU against 463 s on the H100: the machine is thousands of small lookups
in sequence, which leaves the GPU waiting on kernel launches.

**Harness runs on the final file** (`combined_local/logs/e3_*.log`: the evaluator's step outside the evaluator, the
tables against the truth every 25 steps, then 128 rows of every rung on both paths and the gradient test on the
trained model):

| seed | reading chosen | phase B | ladder joins | gate (phase C) | T settled | frozen | 14 rungs, both paths, identical logits | gradient test |
|---|---|---|---|---|---|---|---|---|
| 74 | 92 | 173 | 193 | 193 | 455 (started wrong) | 552 | 14/14 | passes |
| 1 | 133 | 133 | 232 | 187 | 457 (started wrong) | 588 | 14/14 | passes |
| 2 | 105 | 178 | 198 | 198 | 403 | 582 | 14/14 | passes |
| 4 | 225 | 272 | 310 | 292 | 559 | 672 | 14/14 | passes |

### 2. What changed where the pieces meet

Each piece's lessons and fixes are kept: the complete identity with the lowest place weighted 5× (LEARNED_REMAINDER,
CHAIN), the order facts and the wrap-around fact (ADDER), the verified frontier (SQUARE), the gate's facts on a detached
table and the trial mixture (CONTROL), the reader's five stages (IO). What had to change is where they meet.

| where | in the piece | here | why |
|---|---|---|---|
| **how long the hand-written adder checks the table** | ADDER: steps 1–70, with the facts about x and N on from step 1 | until the reading is chosen (IO's stage 4), and at least 70 steps: steps 90–273 across all runs | the learned adder needs the order facts ("x − N borrows", "N − x does not", "(x − N) − N does not": ADDER §5), the facts need x and N, and x and N need the reader. With 70 steps from step 1 the learned adder ran 22–155 steps before the reading was even chosen, and in e1 the facts came only at step 300: the table drifted (§4.1) |
| **when the facts about x and N join** | ADDER: from step 1. CONTROL: "x − N borrows" at phase B | the step the reading is chosen, independent of the phase | they are true only of the true reading. Waiting for phase B (80% of the checks) waited for a check the learned adder could not pass without them |
| **when the product cell learns** | SQUARE: from step 1 | once every check of the table has been exact 20 steps in a row in phase B or C (steps 188–352 across all runs), or 300 steps into phase B | its targets X·b + X now come from the learned adder. A wrong adder gives wrong targets, and wrong early targets leave the carry-compensated blur (SQUARE §5.4) |
| **the ladder's soft rows** | SQUARE: a hand-written soft adder (`add_dd`) | the learned adder on the snapped table, its two ripples carried as one chain over their joint borrow state (`add_soft_joint`) | the hand-written adder had to go. The two separate ripples of a − (0 − b) stall the ladder at 29–59 of 100 cells on every seed (§4.2) |
| **which answer rows train** | SQUARE: rows whose squarings equal the hand-written adder's | rows whose count-down committed then held, and whose squarings check out with learned parts only: the ladder verifies every rung up to the number's largest digit, and every addition passes its own subtraction check (a + b) − b = a, borrow = carry | no hand-written adder to compare with. The point of SQUARE's per-row filter stays (a wrong square makes its answer noise for the subtractor) |
| **"x mod N = x" for the table** | CHAIN, ADDER: through the hand-written stop rule (soft own-borrow choice) | gone. The table gets "x − N borrows" directly (ADDER's anchor and CONTROL's comparison lesson are one lesson here), and "x mod N = x" and "N mod N = 0" teach only the gate, on a detached table (CONTROL) | the stop rule is the gate now (CONTROL) |
| **T** | CONTROL: a count-down of T's width in the batch; IO: T read as digits, then made an integer | the count-down is T's two read slots, always 2 places; T's direction is settled by running the count-down machine from both readings | no integer T anywhere |
| **the writer's code round trip** | IO: the digits of the integer x² | the low 8 digits of the machine's own learned square of x | any digits do; no integer square |
| **the phases** | IO: stages 1–5 + phases A, B; CONTROL: A, B, C | IO's stages and A → B → C, with the ladder as its own switch. B needs the reading chosen and 80% of the checks; C needs "x − N borrows" on 99%; the answers need the ladder passing; T's direction needs 20 exact steps in C; the freeze needs 200 | each lesson waits for what it relies on |

**Which check trusts which table, at the end.** No check uses a hand-written adder after the reading is chosen:
- the link identity (D − N) − (0 − N) = D: the soft table, all three subtractions;
- the ladder: the snapped table, for its targets, its checks and its soft rows;
- the gate's facts: a detached copy of the soft table, so they move only the gate;
- the answers along the own path: the soft table and the soft gate along the hard machine's choices;
- the freeze: the hard machine on every check.

### 3. The prediction path (rule 8)

**Before.** Every piece evaluated by snapping its tables to their argmax and running the machine on integers: digit
indices into the tables, and in most pieces int64 values too (`from_digits`, `x * x`, an integer T). The final logits
were a scatter of 0s into −20s at those tokens. Nothing on that path had a gradient, so rule 8's "inside the autograd
graph, an unbroken gradient path" did not hold for the prediction (LEARNED_REMAINDER.md §5 flagged it).

**Now (`PATH = "ste"`, `predict_ste`).** The same machine on one-hot digits, every table applied as a matrix:
- **straight-through tables.** Each table is `ste(one-hot(argmax), softmax)` = one-hot + (softmax − softmax.detach()),
  computed once per forward: the subtractor [20 (r, c), 120 (m, digit + borrow)], the product cell [100, 20], the gate
  [2, 2]. The bracket is exactly 0 in floating point, so every table's value is exactly 0 or 1;
- **lookups are contractions.** A subtraction place is (digit ⊗ borrow) @ table, contracted with the subtrahend's
  one-hot digit (`sub_soft`). A product cell is (s_i ⊗ s_j) @ table. A gate decision is borrow @ gate. A keep-or-discard
  is k · new + (1 − k) · old. With 0/1 operands every product and sum is exact, so the forward pass is the snapped
  machine bit for bit;
- **the reader.** Anchors are straight-through one-hots over positions, steps straight-through over {+1, −1}, and
  digit-ness straight-through over tokens. A slot is the anchor shifted by (j + 1) · step, its token the slot
  contracted with the prompt's one-hot tokens, its liveness a product of digit-nesses. The code is the selected
  reading's straight-through rank code. The selection is one-hot of the snapped hypothesis forward, and the soft
  weights of the reading mixture backward;
- **the writer.** Its pointer shifted the same way, the state's digit contracted with a straight-through value → token
  table, the logits 20 · (Σ_j position_j ⊗ token_j) − 20.

No number is ever held as an integer on the path: no int64 value of N, x, T or a square. Digits stay one-hot vectors
from the prompt's tokens to the answer's. Integer tensors remain only as the evaluator's token ids, the argmax
indices of the straight-through pairs' hard halves, and the reader's run detection (positions). Locally, with every table at the truth, `predict_ste` and the integer path `predict_index`
give **identical logits** (`torch.equal`) on all 14 depth rungs (`local_combined.py designed`). On the trained models of
the harness runs they do too, on 128 rows of every rung (e1 seed 74, and all four e3 seeds).

**The gradient explodes; it is discounted.** The first backward pass gave NaN. The exact machine's multilinear
gradient grows about 5× per link: each gate decision k = gate(final borrow) multiplies (new − old), which differs in
most of the 9 places, into the borrow chain, which feeds every place's input. So it passes fp32's maximum within one
division. `damp` multiplies the state's gradient by `GRAD_DAMP` after every link, addition and round; the value is
untouched (x.detach() + γ(x − x.detach())). Largest gradient on the subtraction table's logits, true tables, 8 rows,
plain cross-entropy:

| GRAD_DAMP | T = 1 | T = 2 | T = 4 | T = 8 |
|---|---|---|---|---|
| 1 (none) | NaN | NaN | NaN | NaN |
| 0.5 | 1.6e22 | NaN | NaN | NaN |
| 0.3 | 2.7e5 | 1.3e20 | NaN | NaN |
| 0.2 | 1.5e−8 | 4.0e−7 | 4.2e−2 | 1.1e23 |
| **0.1** | 4.1e−14 | 2.2e−14 | 4.5e−14 | 2.2e−14 |

At 0.1 the gradient decays toward the start of the computation, by about half per link. It is finite at every depth,
and dominated by the lookups nearest the answer. Damping only the gate's input (0.01) still overflows by T = 16. So the
gradient is the straight-through surrogate, discounted. It is a path, not the derivative of anything exact: the
multilinear extension's true derivative at a vertex is astronomically large.

**The test (`local_combined.py grad`, and at the end of every `train` run).**
- The model in eval mode with autograd on.
- A cross-entropy on the final logits at the answer's target positions, as the evaluator gathers them, divided by 20
  first.
- One backward pass.
- Every parameter tensor must get a finite gradient that is not all zero, and the rows must still be exact.

The division by 20 is needed because the logits are 0 and −20. The plain cross-entropy of a right answer is then exactly
0 in fp32, and so is its gradient on the right token's logit. The writer's pointer reaches the loss only through that
logit, so its step got no gradient.

| model | rows | exact | every tensor gets a gradient | rows of each table with a nonzero gradient |
|---|---|---|---|---|
| every table at the truth | 32 (T = 1, 2) | 1.0 | **yes** | subtractor digit 185/200, borrow 169/200; product lo 80/100, hi 78/100; gate 2/2; digit-ness 17/17; the selected reading's code 1/24 (the others are not used); writer 10/10; anchors 4/4; steps 4/4 |
| trained (e1, seed 74, froze at step 1,129) | 32 (T = 1, 2) | 1.0 | **yes** | 185/200, 177/200; 84/100, 81/100; 2/2; 17/17; 1/24; 10/10; 4/4; 4/4 |
| trained (e3 seeds 74, 1, 2, 4) | 32 (T = 1, 2) | 1.0 | **yes, 4/4** | 183–188/200, 173–179/200; 79–82/100, 82–84/100; 2/2; 17/17; 1/24; 10/10; 4/4; 4/4 |
| every table at the truth, T = 64 rows (informational) | 32 | 1.0 | yes | the T field's anchor and step, and x's step, get 0: they act only through 64 rounds of discounted gradient and underflow |

Rows a batch does not use get no gradient, which is right: 32 prompts do not meet every table row.

**What the path does not cover.**
- In training the forward returns zero logits, as every piece's did. The training loss reaches every table through its
  lessons: soft relaxations of the same machine, on the machine's own path. It does not reach them through the final
  logits; running the full machine every training step would cost about as much as evaluating.
- At evaluation the evaluator runs under `torch.no_grad()`, so no graph is built. The path is a property of the
  computation, checked by the test.
- The hard halves of the straight-through pairs are argmax decisions. One of them, which reading hypothesis a row's
  pointers make, is designed run-detection logic (`selected`), with the reading mixture's soft weights as its gradient
  side.

**Time.** Evaluation on the H100: 48.2 s with `PATH = "ste"` (official run), 23.9 s with `PATH = "index"`, and
34.3 s for the plumbing run (true tables, "ste"). The budget is 1,800 s. The path is launch-bound: several small
kernels per subtraction place, about 650 places per round, and some 330 rounds over all splits. torch.compile or CUDA
graphs would cut that, but at under 3% of the budget they do not matter, so neither is used. On CPU the "ste" path
costs about 0.14 ms per row per round at 768 rows, against 0.014 ms for "index". A whole CPU evaluation took
427–554 s of the 1,800 s budget across the 8 seeds.

### 4. What broke and why

Diagnostic sweeps ran `combined_local/local_combined.py train` (the evaluator's step: zero_grad, backward, clip 1,
Adam, scheduler) on Modal CPU containers at batch 512, logging the snapped tables against the truth every 25 steps.
Logs in `combined_local/logs/` (e1, e2: the first two recipes; e3: the final file).

#### 4.1 The 70-step warm start, as in ADDER.md

With ADDER's 70 steps counted from step 1, the learned adder took over at step 71. The reading was chosen only at step
92–225, and the facts waited for phase B, which only came at step 300 (the 300-step fallback): the learned adder's
identity check passed on 15–60% of the links, below the 80% switch. In between, the table drifted toward a
self-consistent wrong table, as ADDER.md §4 describes for a learned adder without order facts.

| recipe (sweep) | seeds | reading chosen at step | table right at step 150 / 300 | table first 200/200 at step |
|---|---|---|---|---|
| ADDER's 70 steps from step 1, facts at phase B (e1) | 74, 1–5 | 92–225 | 129–183 / 103–171 | 475–950 on 5 seeds; seed 4 never (130/200 at step 1,725) |
| hand-written adder to step 150 (e1) | 74, 1 | 92, 133 | 190, 191 / 174, 200 | 425, 200 |
| hand-written adder until phase B (e2 "B") | 74, 1–5 | 92–225 | 176–191 / 190–200 | 225–400 |
| 70 steps, facts as soon as the reading is chosen (e2 "C") | 74, 4 | 92, 225 | 175, 129 / 200, 97 | 225; seed 4 never (99/200 at 1,275) |
| **hand-written adder until the reading is chosen, facts from then (e2 "BC", the final schedule)** | 74, 1, 2, 4 | 92–225 | 176–195 / **200 on all** | **175–300** |

Seed 4 is the slow reader (chosen at step 225). Its table drops from 168 to 103 rows right while it waits, and it does
not recover with the facts at step 300 (e1) or at step 225 (e2 "C"). With the hand-written adder kept until its
reading is chosen, it reaches 200/200 at step 300.

The rows that went wrong in e1 are the known trap families: (0, 0, 1) (the run of 9s), (r, 0, 1) and (r, r, 1). A
parallel ablation of `submission_adder.py`'s recipe (`cpu_first/ablation/`, its `probe_trap.py`; its finding, not
re-derived here) explains why waiting is what hurts:
- R − 0 = R and R − R = 0 create those families through soft borrows;
- with (0, 0, 1) wrong, the learned-adder check (D − N) − (0 − N) = D and 0 + N = N have a barrier that holds it;
- the hand-written check, "(x − N) − N does not borrow" and R + N = N + R slope back down to the truth.

In e1 the wrap-around fact waited for the reading, so from step 71 the anchors that make the traps and the checks that
hold them ran without it.

#### 4.2 The ladder stalls on the learned adder's soft form

Once the table was right, the product cell stopped at 29–59 of its 100 cells on every seed of e1 and e2. Only one e1
run escaped, and froze at step 1,129. A few percent of the rung checks passed, and the cells left wrong came in whole
families such as (2, b): the cell 2 × 0 sat at lo 0, hi 9.

The same ladder alone, on the training rows' own x and N digits with the true subtraction table, reproduces it
(`combined_local/ladder_adders.py`, seed 0):

| soft adder in the ladder's soft rows | cells right at step 50 / 100 / 150 / 200 |
|---|---|
| SQUARE's hand-written soft adder | 21 / 49 / 82 / 100 |
| the learned adder as two soft ripples, a − (0 − b) | 22 / 40 / 48 / 57 (stalled) |
| one operand soft at a time (lo soft with hi snapped, and the reverse) | 22 / 23 / 23 / 23 (frozen; 11–23 on 3 seeds) |
| **the learned adder, both ripples as one chain over their joint borrows** | **21 / 49 / 82 / 100** |

On uniform random digits the two-ripple form learns too (90/100 by step 150). The rows' digits carry many zeros, and
that brings out the flaw.

**Why.** a − (0 − b) runs two ripples, and the second treats the digits of c = 0 − b as independent. They are not:
c's borrow chain ties them. Take a hi digit that should be 0, with probability p on 9. Then c is 1 at that place and
9 at every place above with probability p, and 0 otherwise. The second ripple then mixes "9 above, borrow coming in"
with "0 above, no borrow" as if they were independent. Each place above gets a spurious error near 2p(1 − p), whose
loss is lowest at p = 0 *or* p = 1. Against the true push toward p = 0 at the digit's own place, with about seven places
above, the two balance near p ≈ 0.93: a wrong argmax that the gradient holds in place.

**The fix** carries both ripples as one chain over their joint borrow state (4 states). The sum's digit marginals are
then exact for independent input digits. With the true table it equals the hand-written soft adder to 7·10⁻⁷, and on
exact digits it equals the learned adder.

#### 4.3 T's direction can start saturated the wrong way

On some seeds the tiling stage settles T's pointer on the wrong direction ("61" for 16) with all of the weight before
T's lesson starts. This happened on the official run and on e3's seeds 74 and 1; e3's seed 2 started right. IO.md saw it
once, on its seed 6. The lesson then reverses a saturated choice at about one Adam step per training step, and each of
its steps runs the count-down machine from both readings on 64 rows (up to 61 rounds). It costs time, not
correctness. In the official run the lesson ran from about step 360 and settled at step 448, about 4 s a step on the
H100. The harness runs settled 30–80 steps after the lesson began.

#### 4.4 The gradient through the exact machine overflows

See §3.

### 5. Reproduce

Everything heavy runs on Modal. Launch from the one-layer-deeper repository root; the existing `modal_chain.py` (app
`learned-chain`, Volume `learned-remainder`, unchanged) takes `--file`. It builds the competition image, regenerates H1
inside with the official generator arguments, and writes each record to the Volume. `wait_fetch.py` downloads the
records to `cpu_first/runs/learned_remainder/<name>.json`.

```bash
cd ~/github/one-layer/one-layer-deeper
M=~/github/one-layer/cpu_first/learned_remainder/modal_chain.py
~/github/sotaku/venv/bin/python -m modal run --detach $M --mode official --name combined_official --file submission_combined.py
~/github/sotaku/venv/bin/python -m modal run --detach $M --mode official --name combined_index_official --file submission_combined.py --set '{"PATH": "\"index\""}'
~/github/sotaku/venv/bin/python -m modal run --detach $M --mode official --name combined_s0_official --file submission_combined.py --set '{"MODE": "\"designed\""}'
~/github/sotaku/venv/bin/python -m modal run --detach $M --mode plan --plan ~/github/one-layer/cpu_first/learned_remainder/plans/cb1_cpu.json --file submission_combined.py
cd ../cpu_first/learned_remainder
~/github/sotaku/venv/bin/python wait_fetch.py combined_official combined_index_official combined_s0_official
~/github/sotaku/venv/bin/python summarize_combined.py --table combined_official combined_index_official combined_s0_official
```

Local checks, each well under a minute and 1 GB (`combined_local/`, with `chain_local/stub` standing in for the
evaluator's API):

```bash
cd ~/github/one-layer/cpu_first/learned_remainder/combined_local
~/github/sotaku/venv/bin/python local_combined.py designed N=8     # true tables: both paths exact, identical logits, 14 rungs
~/github/sotaku/venv/bin/python local_combined.py grad N=32        # the gradient test at the truth
~/github/sotaku/venv/bin/python ladder_adders.py designed ripples # §4.2 (also: split, joint)
```

The diagnostic sweeps (`combined_local/modal_diag_combined.py`, app and Volume `learned-combined-diag`, logs to
`combined_local/logs/`):

```bash
cd ~/github/one-layer/one-layer-deeper
~/github/sotaku/venv/bin/python -m modal run ~/github/one-layer/cpu_first/learned_remainder/combined_local/modal_diag_combined.py --plan ~/github/one-layer/cpu_first/learned_remainder/combined_local/plans/e3.json
```

`plans/e3.json` runs the final file. `e1.json` and `e2.json` ran it before the ladder fix, when the soft rows used
`add_soft_learned`; the logs keep their argv but not that source. e1 is `WARM_TO_FACTS=False FACTS_FROM=B` with
`ADDER_WARM` 70 or 150, and e2 is the variants of §4.1. `ladder_adders.py` reproduces the ladder's stall on its own.
