# H1 with the loop over T and the division's stop rule made of learned parts

## In plain words

**The task.** The model reads three numbers written as digits: a modulus N, a starting number x and a count T. It must
square x, keep only the remainder after dividing by N, and repeat that T times. To be certified it must get every one
of 768 fresh questions exactly right at T = 1, 2, 4, …, 64, both for moduli it saw in training and for larger moduli
it never saw (up to 8 digits). Training only ever shows T = 1, 2, 4, 8 and 16.

**Where this starts.** [CHAIN.md](CHAIN.md)'s calculator does long division with a single learned rule, a 200-entry
subtraction table ("this digit minus that digit, minus a borrow, gives this digit and maybe a new borrow"). Two pieces
of control around it were still written by hand:
- **the count:** read T as a number and square T times;
- **the stop rule of each division step:** subtract N again and again, and keep the last result before the first
  subtraction that goes below zero.

**What changed.** Both are now made of learned parts.
- **The count.** T stays the digits the prompt writes. Each round the calculator subtracts 1 from those digits with the
  same learned subtraction table. When that subtraction goes below zero (it ends in a borrow), T has reached 0 and the
  rounds stop. T = 32 and T = 64 never appear in training, and neither do the tens digits 2 to 6. The count gets them
  right because the table it uses was taught by the division, where every digit appears.
- **The stop rule.** Each subtraction of N is now either kept or thrown away. A tiny learned switch decides, looking
  only at whether that subtraction borrowed. It has 4 learned numbers, and we call it the commit gate. When a
  subtraction is thrown away, the next one starts from the same number, gets the same answer and is thrown away too. So
  nothing has to remember that the step has already stopped. The same switch decides every round of the count.

**What is still written by hand:**
- reading and writing the digits;
- the squaring (an integer multiplication);
- the long-division layout: 8 steps, 9 places, up to 9 subtractions of N per step;
- the count's layout: a register as wide as T is written, and the number 1 to subtract;
- the keep-or-throw-away wiring itself: a register takes its new value only when the switch says so;
- ending the rounds when no question wants another one (with a safety cap);
- for training only: the hand-written adder in the self-check, which questions' answers are used, and the schedule.

**What is learned:** the 200-entry subtraction table (all the arithmetic of the division and of the count) and the
2-entry switch (every keep-or-throw-away decision).

**Did it certify?** Yes, on the untouched official H1 manifest on an H100, with the file run byte for byte:
- all 768 questions right at every T up to 64, on seen and on unseen moduli;
- the subtraction table exactly right (200 of 200 entries) and the switch the true one;
- every table entry used at evaluation was trained, including the 24 the count uses, at least 471,864 times each;
- the same file certified on 8 of 8 seeds on CPU through the same evaluator (seed 74 and seven more), and on 4 of 4
  seeds that never see an answer.

The variant without the switch also certified on the official H100 run. There the table's own borrow is wired straight
to "throw away", so the rule is expressed through the table's own outputs.

**Times.**
- Training: 240 s of the 3,600 s budget. The model is done at step 469, under 2 minutes in (step 500 came at 113 s),
  and then holds still.
- Evaluation: 12.8 s of the 1,800 s budget, against 15.4 s for CHAIN.md's file. The arithmetic is the same: 648
  subtractions in a row per remainder, and 64 remainders in a row at T = 64.

**What broke, and why.** Three earlier recipes failed on every seed, in 33 diagnostic runs. The switch can only learn
what to do with a borrow once the table knows what a borrow means. CHAIN.md's recipe taught one entry of that meaning,
"0 minus 0 with a borrow coming in", through the hand-written stop rule ("a number below N is its own remainder, so
its first subtraction must stop the step"). Without that rule the entry stayed wrong. Every subtraction that went
below zero then looked as if it did not, and the switch learned to throw everything away.

Three more traps followed:
- a stronger self-check made other entries stick;
- the switch and the table bent toward each other;
- a soft version of the switch blended numbers and settled on a wrong balance.

The recipe that works:
- adds a direct lesson, "x is below N, so x − N must end in a borrow";
- gives the switch its own later phase, whose lessons cannot change the table;
- lets the switch's soft version weigh each candidate result separately (§6).

## 1. Results

Official manifest `benchmark/manifests/h100_hard_h1.json`, untouched:
- cuda:0, bf16 autocast, the batch from the submission;
- 3,600 s training budget, 1,800 s evaluation budget;
- Modal H100 80GB, with the image and H1 data regenerated with the official generator arguments (`modal_chain.py`,
  unchanged, with `--file submission_control.py`);
- seed 74. Every rung is 768 prompts.

| run | what ran | max T (seen moduli) | OOD-N max T | rungs T=1…64, seen / OOD-N | test, ood_t, ood_n_t | table rows right | gate | phase B / C / frozen at step | training s | evaluation s |
|---|---|---|---|---|---|---|---|---|---|---|
| `control_s0_official` | the exact table and gate (`MODE = "designed"`, 100 steps): a plumbing check | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 (designed) | true (designed) | – | 3.3 | 16.6 |
| **`control_official`** | **`submission_control.py` byte for byte**, md5 `ec79a312e493f1731cf770a2528f46ff` | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | **200/200** | **true**: P(commit \| no borrow) = 0.9999, P(commit \| borrow) = 0.0002 | 72 / 104 / **469** | 239.7 | **12.8** |
| `control_borrow_official` | the file with `GATE = "borrow"`: the table's final borrow wired to the hold, no gate (2,400 weights) | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 | – (wired: borrow → hold) | 72 / 104 / 396 | 269.3 | 14.8 |
| for comparison: `chain_v2_official` | `submission_chain.py` (CHAIN.md) | 64 | 64 | 768 everywhere | all exact | 200/200 | – (designed stop rule and loop) | 71 / – / 417 | 271.8 | 15.4 |

`control_borrow_official` is the variant in which the selection is expressed through the table's own output (§4). It
certified too. There the two facts have nothing to teach, so they only gate the freeze; the table learns as in the
main file.

Every weight is trained: 2,404 numbers, the table's 200 rows × (10 digit + 2 borrow) logits and the gate's 2 × 2. The
gate gets its first gradient in phase C, so Adam's state after the first step covers the table only (4,802 elements).
Most of the training time is the frozen rest of the 20,000 steps: step 500 came at 112.9 s, step 5,000 at 141.6 s and
step 20,000 at 239.7 s.
Evaluation took 12.8 s here and 15.4 s for `submission_chain.py`; it does the same 648 lookups per reduction, plus a
2-place subtraction and one GPU-to-host sync per round of the count. The difference was not profiled and may be noise
between containers. Either is far inside the 1,800 s budget.

**The audit** (printed by the submission at its first evaluation batch, compared outside it by `summarize_control.py`):
- the snapped table equals true subtraction on all 200 rows, and the gate is the true one;
- `AUDIT eval_rows_used=200/200 used_but_never_trained=0`: training's hard checks used every row at least 194,233 times;
- `AUDIT countdown_rows_used=24 used_but_never_trained=0 min_train_visits=471864`: the count used the 24 rows of §3,
  and the lessons used each at least 471,864 times;
- `AUDIT gate_rows_used=2/2 used_but_never_trained=0`: the lessons' hard machine read the gate 423,258 times on
  "no borrow" and 3,204,822 times on "borrow".

The count's audit line also prints `not_met_by_the_countdown_in_training=0`. That counter is not the clean statement it
looks like. It adds up every round of a count run on the training batches' T before the freeze, including the first
steps, when the table was still random and the count wandered through arbitrary digits. Which rows a *correct* count
meets on the training T values is a property of subtraction; with the exact table it is 14 of the 24 (§3).

The same submission through the same unmodified evaluator on Modal CPU containers, with the manifest's device set to
cpu and everything else official, one seed per run (`plans/k1_cpu.json`):

| recipe | seeds | certified T = 64 on both profiles | table rows right, gate | phase B / C / frozen at step | training s / evaluation s (CPU) |
|---|---|---|---|---|---|
| **the file as it is** (with the rows' answers), byte for byte | 74, 1, 2, 3, 4, 5, 6, 7 | **8/8** | 200/200, true gate | 74: 72 / 104 / 469; 1: 91 / 155 / 550; 2: 85 / 147 / 566; 3: 300 / 363 / 577; 4: 300 / 365 / 844; 5: 74 / 125 / 630; 6: 90 / 150 / 545; 7: 300 / 363 / 862 | 306–559 / 18–26 |
| **identity only** (`REAL_MAX_T = 0`: never sees a label) | 74, 1, 2, 3 | **4/4** | 200/200, true gate | 74: 72 / 104 / 582; 1: 91 / 155 / 550; 2: 85 / 147 / 566; 3: 300 / 363 / 654 | 286–396 / 17–25 |

Every run is exact on every rung and on test, ood_t and ood_n_t. Every audit reads `used_but_never_trained=0` for the
table, the count and the gate.

Seed 74 on CPU took the same path as on the H100: phase B at 72, C at 104, frozen at 469. Identity-mode training has
no random numbers in it, as in LEARNED_REMAINDER.md §4.2. Seeds 1 and 2 froze at the same step with and without the
answers. They pass phases A and B identically, since the answers only join in phase C. There, the 200-step run of exact
checks that triggers the freeze began at the same step with and without them.

Seeds 3, 4 and 7 are the ones whose phase A sat at 50–51% of the checks until its 300-step cap, on the row of §6.1.
"x − N borrows" then fixed it within 63–65 steps, and they froze at 577, 844 and 862. For comparison,
`submission_chain.py` froze at 417–481 on its 8 CPU seeds and at 408–544 identity-only (CHAIN.md §1).

## 2. What is designed and what is learned

| part | `submission_chain.py` | `submission_control.py` |
|---|---|---|
| read N (LSB first), x and T; write the answer's digits right-aligned | designed | designed |
| **the count over T** | designed: T read as an integer, squaring repeated T times | **the learned table** subtracts 1 from T's digits each round, and **the learned gate** on that subtraction's borrow decides whether the round squares or stops. Designed: the count's register (as wide as T is written), the number 1, the rounds-until-no-commit loop and its cap of 10^K rounds |
| square the state | designed (integer product) | designed (integer product) |
| long-division wiring: start from v // 10^8, shift in one digit per step, 8 steps, 9 register places | designed | designed |
| N inside the reduction | only its digits, as the prompt gives them | the same |
| a step's subtractions | a chain of up to 9 subtractions of N (designed wiring) | the same chain on one register; each link either takes its difference or keeps the register |
| **which result a step keeps** | designed stop rule: a latch ("no link has borrowed yet") and a select ("keep the last trial before the first borrow") | **the learned gate** on each link's own final borrow: commit or hold. No latch: a held register makes the next link repeat the same subtraction and the same decision. Designed: the write-enable (a register takes the new value or keeps the old one) |
| the subtractor cell (r, m, borrow in) → (digit, borrow out), one 200-row table | learned (2,400 logits) | learned (2,400 logits); also does the count's arithmetic |
| **the commit gate**, borrow → (hold, commit) | – | **learned**: 2 rows × 2 logits, shared by every link of every step and every round of the count |
| what teaches the table | the complete identity (D − N) + N = D, "x mod N = x" (through the stop rule), the rows' answers | the complete identity, **"x − N borrows"**, the rows' answers |
| what teaches the gate | – | "x mod N = x" and **"N mod N = 0"** on a detached copy of the table, and the rows' answers along the own path |
| adder in the training check | designed | designed |

Evaluation snaps the table and the gate to their argmax and runs everything on integers. `GATE = "borrow"` wires the
table's borrow straight to the hold (no gate): the variant in which the selection is expressed through the table's own
outputs, with a fixed polarity as its only extra design (§4).

## 3. The count

**Wiring.** The prompt writes T's digits. The count is a register C of K places, LSB first, holding those digits; K is
T's width (the widest in the batch: 1 for T ≤ 8, 2 for T ≥ 16). Each round:
- C − 1 with the learned table, place by place from the lowest: (c₀, 1, borrow 0) at place 0, then (c_j, 0, borrow)
  above it;
- the gate reads that subtraction's final borrow;
- commit: the state is squared, reduced mod N, and C takes C − 1;
- hold: the state and C stay as they are.

When C is 0, 0 − 1 borrows and the round holds. C stays 0, so every later round repeats the same subtraction and holds
again. The rounds end when no row of the batch commits (one GPU-to-host sync per round). The cap of 10^K rounds is only
a safety net: a working K-digit count ends sooner. With the exact table and gate, every row commits exactly T rounds and
the machine reproduces all 14 depth rungs (`control_local/local_control.py designed`).

**Which table rows the count uses.** Evaluation's count uses 24 of the 200 rows:
- (0…9, 1, 0) at place 0;
- (0…6, 0, 0) and (0…6, 0, 1) at place 1.

Counting down from the training T values {1, 2, 4, 8, 16} meets only 14 of them. The other 10, (2…6, 0, 0) and
(2…6, 0, 1), are the tens digits 2 to 6 of T = 32 and T = 64. Only the division meets them in training. There they subtract a 0 digit of N (an inner zero, or the place just
above N's top digit) from a register digit 2 to 6. This is the condition [FINDINGS_looping.md](
../microlab/FINDINGS_looping.md) found for a count that reaches unseen T. There, a decimal countdown cell of its own
failed at T = 21: "tens digit 2..9 with no borrow" never occurs in training counts. Here the count's cell is the
division's cell, and the division's lessons pin every row. Nothing trains the count itself: no lesson's gradient passes
through it.

In `control_official` the count used exactly these 24 rows at evaluation. Training's lessons used each of them at least
471,864 times (`TABLE train_visits` in the record; the median row over all 200 was used 418,565 times). That minimum is
one of the 10 rows only the division meets, (6, 0, borrow 1). The other 9 were used 569,243 to 1,741,253 times. The
count's own `not_met_by_the_countdown_in_training` counter is not meaningful (§1).

## 4. The stop rule

**Before.** `submission_chain.py`'s `step_hard` ran the chain on one value and kept two more pieces of designed state:
- a latch, alive ← alive AND (the link did not borrow);
- a select, R ← alive ? the link's difference : R.

The result was "the last trial before the first borrow".

**Now.** A step has one register D, starting at Rp. Link s computes E = D − N and its final borrow b with the table,
and the gate maps b to commit (D ← E) or hold (D unchanged). With the exact table and gate this is (non-performing)
restoring division: D takes Rp − N, Rp − 2N, … while the subtraction does not borrow; Rp < 10N, so 9 links suffice.
There is no latch. Once a link holds, the next link reads the same D, computes the same subtraction, gets the same
borrow, and the gate (a function) holds again.

**What design remains, precisely:**
- **the write-enable:** a register takes a new value or keeps its old one, as a bit says. This is a 2-way select. The
  chain's register, the count and the state all use it;
- **the wiring:** each link reads the register; the gate reads the link's final borrow; its output drives the
  write-enable.

The decision is learned: which borrow value means "keep" is the gate's, and what a borrow means is the table's. With
`GATE = "borrow"`, the decision is expressed through the table's own output: the final borrow drives the hold directly.
The design is then the select plus a fixed polarity (borrow → hold), and the table alone carries the meaning.

**Cost.** Unchanged: 9 links × 9 places per step, 648 table lookups in a row per reduction. The links after a hold
recompute the held subtraction, where `submission_chain.py`'s links after the first borrow computed wrapped numbers.

## 5. How the table and the gate learn

The table must first learn what a borrow means; only then can the gate learn what to do with it (§6 is how that order
was found). All lessons are on the training rows' own numbers; nothing is made up.
- **Phase A (the table).** As in CHAIN.md: every link of the machine's own division of each row's x² checks itself,
  (D − N) + N = D, with the soft table and a designed adder. 512 rows × 8 steps × 9 links = 36,864 checks per training
  step. The check is complete: the final borrow must equal the adder's final carry. The lowest place weighs 5×.
- **Phase B (the table)**, once 80% of the checks pass over 20 steps, or after 300 steps. Adam's state is reset. A new
  lesson joins, **"x − N borrows"**: x < N, so the soft final borrow of x − N must be 1. In `submission_chain.py`,
  "x mod N = x" taught this through the designed stop rule. Without that rule, nothing else teaches the borrow of
  (0, 0, borrow 1), the row at the zeros above N's top digit (§6.1).
- **Phase C (the gate)**, once "x − N borrows" holds on the hard machine for 99% of the rows over 20 steps, or after
  300 more steps. Three lessons join:
  - **"x mod N = x".** One division step from the register x must keep trial 0. Since x − N borrows, this teaches
    "hold on a borrow".
  - **"N mod N = 0".** One step from the register N must keep trial 1. N − N does not borrow, so commit; then 0 − N
    borrows, so hold.

    Both facts run on a detached copy of the soft table, so they teach the gate and nothing else (§6.3). Their soft
    version is CHAIN.md's soft own-borrow choice, with the gate in place of the stop rule. The trials Rp, Rp − N, … are
    computed one after another. Trial j is kept with the probability that links 1…j commit and link j + 1 holds, each
    link committing with the soft gate's probability at that trial's soft borrow (§6.4).
  - **The rows' answers along the model's own path.** These are the batch's rows with T ≤ 2 (about 10 per step). The
    hard machine runs its own count, and only rows whose count committed and then held within 3 rounds are used. Its
    division of the last squaring's input picks trials with the argmax table and gate. The soft table computes the same
    trials from the soft register and keeps them with the soft gate's probabilities at the hard borrows. So the answers
    teach the table and the gate.
- **Freeze** (learning rate 0) after 200 consecutive phase-C steps in which every check holds on the hard machine:
  the identity, "x − N borrows", "x mod N = x" and "N mod N = 0". The last two pass only with the true gate.

## 6. What broke and why

Three sweeps of recipes failed (33 runs) before the one of §5. All four sweeps ran the training loop of
`control_local/local_control.py` on Modal CPU containers at the real batch of 512 (d1 to d4, logs in
`control_local/logs/`). Each line of a log is the snapped table against true subtraction, and the gate, every 25 steps.
These sweeps ran the lessons only, not the evaluator.

| sweep | recipe | runs | ended with |
|---|---|---|---|
| d1 | CHAIN.md's lessons; the gate's two facts join at phase B; a blended soft chain; the facts also teach the table | 11 (6 seeds; live links ×2; identity only ×2; own path without gate ×1) | **0/11**: the gate always holds, table 131–135/200 |
| d2 | + "x − N borrows"; the identity checks the borrow at every place (11 runs) or only the final one (1 run) | 12 | **0/12**: the gate always holds; 188–192/200 with every-place borrows, 199/200 with the final borrow only |
| d3 | final borrow only; the gate's facts wait for phase C and run on a detached table; still the blended chain | 10 (6 seeds; identity only ×2; linearized own path ×2) | **0/10**: table 200/200 on all, but every gate at P(commit \| no borrow) ≈ 0.13–0.14, P(commit \| borrow) = 0, so it always holds |
| **d4** | **§5: + the trial mixture** | 11 | **11/11 frozen**, 200/200, true gate (§1) |

d2 and d3 were stopped once their failure was clear (d2 at steps 750–1,175, d3 at 675–900); d1 ran its 1,200 steps.

### 6.1 The table never learns the borrow of (0, 0, 1) by itself
Every run, whatever gate its random start gave it, reached phase B with 195–199/200 rows right:
- **the row that is always wrong:** (0, 0, borrow 1), at digit 9 or 0 but always with **borrow 0**;
- **sometimes a few more:** for instance (1, 1, 1), also with borrow 0 (seed 74);
- **the checks passing:** the seeds that hit the 300-step cap sat at 50–51% (seeds 3 and 7); the others switched at
  the 80% mark (steps 72–96).

CHAIN.md met this row too: "phase A ends with only (0, 0, borrow 1) wrong … 'x mod N = x' fixes that row within about
60 steps". There it worked through the designed stop rule, which turns "x is its own remainder" into "x − N must
borrow". With a learned gate, the same lesson can be satisfied by the gate instead. The row is the borrow running
through the zeros above N's top digit. While it is wrong, every subtraction that goes below zero loses its borrow
there: x − N and 0 − N look non-negative. So both facts teach "hold when there is **no** borrow":
- the gate reached P(commit) ≈ 0.04–0.09 for both borrows within 25–40 steps of phase B, and stayed;
- the hard machine then never subtracted;
- the rows' answers never ran, because their count never committed;
- in d1 the facts also pulled the table's digits toward "x − N = x": 131–135/200 rows by step 1,200.

The fix is a lesson aimed straight at the borrow, **"x − N borrows"**. It reaches the table and nothing else (phase B).
At the real batch it fixed the row in 50–100 steps on every seed that had it (d2–d4). Sometimes the fix moved the row
from (9, 0) to (0, 1), borrow right and digit wrong. Then the complete identity, which has no barrier along that row
once its borrow is right (CHAIN.md §5.2), finished it in phase C. Seed 3 in d4: 199/200 from step 100 to 300 in
phase A; (0, 0, 1) → (0, 1) at step 350; 200/200 at step 400.

### 6.2 Checking every place's borrow deepens another saddle
The identity holds for every block of low places, so the borrow out of each place must equal the adder's carry out of
it (`ID_BORROWS = "all"`). That pins every row's borrow locally, and was meant to fix 6.1. It made phase A worse:
- 188–190/200 at the switch instead of 199/200;
- rows (r, r, borrow 1) → (0, 0) for r = 0, 1, 2, and (0, 2, 0) → 7 or 8.

At (r, r, borrow 1) the wrong state (digit 0, borrow 0) fails one place. The per-place check makes each half-move cost
more: moving the digit alone to 9 now also fails the borrow check at that place, on top of the next place's digit. So
the saddle that LEARNED_REMAINDER.md §4.2 described deepens. The same seed with the final borrow only reached 199/200
(`d2_final_s3`), and the file keeps `ID_BORROWS = "final"`.

### 6.3 The gate and the borrows co-adapt
In d2 the gate's facts still reached the table. Once the gate held more on "no borrow" than on "borrow", "N mod N = 0"
had two ways to make N's first link commit:
- move the gate;
- make N − N borrow.

It took the second: rows (r, r, 0) went to borrow 1 (digit 0 or 9) for r = 4…8 (`d2_final_s3` from step 475;
`d2_def_s74`). The fix is twofold:
- the facts run on a detached copy of the soft table (`FACT_TABLE = False`), so they teach only the gate;
- they wait for phase C, once the table's borrows mean "below N".

### 6.4 A blended soft chain has a wrong resting point
d3 had both fixes: all 10 tables reached 200/200, and all 10 gates stopped at P(commit | no borrow) = 0.13–0.14 and
P(commit | borrow) = 0.0. The soft chain there took one commit probability per link and applied it to the whole soft
register. After a partial commit the register is a blend, for instance of x and the wrapped x − N, and:
- **"x mod N = x" learns from the blend.** It wants every link to hold. The wrapped part does not borrow, so the lesson
  pushes P(commit | no borrow) down through it;
- **"N mod N = 0" cancels itself.** Its later links mix "0 − N: hold" and "N − N: commit" into one probability, so
  their pushes cancel.

The fix is CHAIN.md's own soft choice, with the gate where the stop rule was. Compute the trials Rp, Rp − N, … one
after another. Keep trial j with the probability that links 1…j commit and link j + 1 holds, each link's commit
probability being the gate at that trial's own borrow (`keep_soft`). Each trial's borrow then comes from that trial
alone:
- "x mod N = x" only says "hold on a borrow";
- "N mod N = 0" says "commit on no borrow, then hold on a borrow".

With it, d4 froze 11 of 11 runs (§1).

## 7. Gray areas against the rules
Same as CHAIN.md §6, except:
- **Rule 7, "no hard-coded algorithm in the forward pass".** The forward pass is still a designed CPU. It now has two
  learned tables: the arithmetic (200 rows) and every keep-or-discard decision (2 rows). Still designed:
  - the digit I/O and the integer square;
  - the long-division layout and the chain's length;
  - the count's register and the number 1;
  - the write-enable select;
  - "run rounds until no row commits", with its cap.

  Whether a loop exists, and how it ends, is design. Whether each round squares is learned. CHAIN.md listed "the stop
  rule" and "the T loop" as designed; both decisions are now the table's borrow read by the gate.
- **Rule 14, "no data augmentation, task-specific solvers".**
  - The identity still uses a designed adder; its final carry is still what teaches the table's borrow its meaning.
  - **"x − N borrows"** uses the fact x < N, the same fact behind "x mod N = x". It now reaches the table directly,
    where CHAIN.md's recipe got it through the designed stop rule.
  - **"N mod N = 0"** divides the row's own N, a number from the prompt, as "x mod N = x" divides x.
  - The rows' answers are chosen by T's digits (T ≤ 2), and only rows whose own count stopped within 3 rounds are used:
    a training-time filter.
- **Rule 4, the schedule inside the model.** Now three phases. The switch to phase C is triggered by a label-free
  check, "x − N borrows" on the hard machine.
- **Choices made after seeing failures** (§6): the comparison lesson, the gate's late phase, the facts on a detached
  table, and the trial mixture. The first three are training-time choices; the trial mixture is the soft relaxation
  the gradient flows through. None of them is in the evaluated forward pass.
- **Rule 8, snapping to argmax.** Unchanged.

## 8. Next pieces to remove
Following CHAIN.md §7, whose item 2 (the stop rule) and the loop counter of its item 4 are done here:
1. **The designed adder in the check.** "Learn addition", a parallel session, is working on it. Here the adder's final
   carry is still what teaches the borrow its meaning, and through it the gate.
2. **The write-enable select.** It is a 2-way choice between a register's old and new value. A learned select cell,
   (enable, old digit, new digit) → digit, would take its place. It would bring its own lessons, and the identity
   cannot check it.
3. **"Run rounds until no row commits", and the chain's length of 9 links.** Both are adaptive-halting wiring around
   the learned decisions.
4. **The squaring, the long-division layout and the I/O.**

## 9. Reproduce
Everything heavy runs on Modal. Launch from the one-layer-deeper repository root. The runner is CHAIN.md's
`modal_chain.py`, unchanged, with `--file submission_control.py` (app `learned-chain`, Volume `learned-remainder`).
`wait_fetch.py` downloads the records to `cpu_first/runs/learned_remainder/<name>.json`.

The certified file as it is:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --name control_official --file submission_control.py
```

The wired-borrow variant:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --name control_borrow_official --file submission_control.py --set '{"GATE": "\"borrow\""}'
```

The plumbing check with the exact table and gate:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --name control_s0_official --file submission_control.py --set '{"MODE": "\"designed\"", "MAX_STEPS": "100"}'
```

The CPU seed plan, one container per entry:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode plan --plan /path/to/cpu_first/learned_remainder/plans/k1_cpu.json --file submission_control.py
```

Fetch the records, then tabulate them with the table, gate and count audits (from the directory above `cpu_first`):

```bash
~/github/sotaku/venv/bin/python cpu_first/learned_remainder/wait_fetch.py control_official control_s0_official control_borrow_official
```

```bash
~/github/sotaku/venv/bin/python cpu_first/learned_remainder/summarize_control.py --table control_s0_official control_official control_borrow_official
```

The diagnostic sweeps of §6 run the lessons alone, without the evaluator, and print the snapped table and gate every
25 steps. They run on Modal CPU containers (app `learned-control-diag`, Volume `learned-control-diag`). Only d4's plan
reruns against the final file as it is. d1–d3 used the blended soft chain of §6.4, which the file no longer has; the
rest of their recipes are switches that are still there (§6 table).

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/control_local/modal_diag_control.py --plan /path/to/cpu_first/learned_remainder/control_local/plans/d4.json
```

Files:
- **`submission_control.py`.** The submission, certified byte for byte as `control_official`. Its variants:
  - `MODE = "designed"`: the plumbing check;
  - `GATE = "borrow"`: the borrow wired to the hold, no gate;
  - `REAL_MAX_T = 0`: never sees an answer;
  - `ID_BORROWS = "all"`, `CMP_LESSON = False`, `GATE_PHASE = "B"`, `FACT_TABLE = True`: the failed steps of §6, for
    the parts that are still switches.
- **`summarize_control.py`.** `summarize.py`'s tables plus the gate and the count's audits.
- **`plans/k1_cpu.json`.** The CPU seed plan.
- **`control_local/`.** `local_control.py` is the local harness: `designed` replays the depth rungs with the exact table
  and gate, and `train STEPS B SEED K=V` runs a short training. Locally it fits under a minute and 1 GB only at small
  batches. `modal_diag_control.py` runs it on Modal; `plans/d1–d4.json` and `logs/` are the sweeps of §6.
- **`submission.py`, `submission_chain.py`, `modal_chain.py`.** Untouched. The md5 of `submission_chain.py` is
  `ed360cb107e644615e30c910ba6f6a2d` and of `submission.py` `bd477b9aadc9fcf34f428f728b5bea20`.
