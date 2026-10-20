# Clock scoring for long division

**Question.** [FINDINGS_quotient.md](FINDINGS_quotient.md) ends on a chicken-and-egg in the long-division datapath:
- the digit cell learns the subtraction when the quotient digits are given (modq);
- the digits learn when the subtraction is exact;
- from answers, neither starts.

Scored on the answer's exact digits (cross-entropy), a wrong guess and a wrong subtraction look equally wrong. But with
an exact subtraction and no overflow, a wrong quotient digit changes the remainder by a multiple of N, so the final
value stays congruent to the answer. The clock loss 1 − cos(2π(r − y)/N) therefore charges nothing for guesses.
Does scoring on it let the cell learn the subtraction first, whatever the guesses, and break the chicken-and-egg?

**Answer: no, at 6, 7 or 10 bits.**
- **From scratch the clock teaches the cell nothing, even when the true digits are given.**
  - With the digits given, cross-entropy trains the cell: 0.91–0.93 exact at 6 bits and 0.38–0.39 at 7 bits in 3k
    steps.
  - The clock stays at chance. Its loss sits at 1.00 for the whole run, and the register lands on the right hour
    only 1/N of the time (0.02 at 6 bits, 0.01 at 7).
  - That holds for both gradient forms tried, with or without noisy digits, and at batch 16,384 for 400 steps,
    8.5× run 1's examples.
- **Trained jointly from answers (run 2), every clock variant is at chance**, at 6, 7 and 10 bits, 3k and 12k steps.
  - That covers both choices: longdiv's borrow rule taken hard, and the estimator's top slot taught label-free.
  - The cross-entropy controls with the same hard choices reach only 0.02–0.05 at 6 and 7 bits (0.00–0.01 at 10).
    The soft-choice baselines of FINDINGS_quotient.md reach 0.44–0.59 at 6 bits and 12k.
- **Why nothing starts.** Run 3 compares the clock's gradient for the cell with the teacher-forced exact-subtraction
  lesson.
  - At initialization it is noise at every batch size up to 65,536: cosines between −0.7 and +0.7, changing sign
    from batch to batch and seed to seed, with no trend as the batch grows.
  - The answer's cross-entropy agrees from batch 256: +0.34 to +0.51 at 6 bits, +0.97 to +0.98 at 7 and 10.
  - The loss itself is sound. Given its own digit logits per example, it fits r ≡ y (mod N) in 300 Adam steps on
    0.69–0.90 of examples at 6 bits and 0.59–0.76 at 7 (straight-through form; `toy`).
  - Through one shared cell it gives no common lesson.
- **The clock is blind to more than the guesses.** It cannot tell which multiple of N a slot subtracts.
  - Any cell whose slots subtract fixed multiples of N scores zero, whatever the choices.
  - So does the copy cell, which subtracts nothing: the headroom that makes wrong guesses harmless also holds the
    unreduced dividend.
  - So even a clock that trained could not single out the subtraction.
- **Near a solution its gradient points the right way, but training on it still loses ground.**
  - After 3k cross-entropy steps, at 7 bits, the clock's gradient agrees with the exact lesson at +0.97 to +0.98
    (batch ≥ 4,096), and not with the copy cell. Cross-entropy's own gradient there is flat (+0.0 ± 0.1): it is
    stuck at its plateau.
  - Switching to the clock for 3k more steps still lowers the cell. At 7 bits it falls from 0.39 (cross-entropy
    control) to 0.29–0.30, and at 6 bits from 0.97–0.99 to 0.75–0.89.
  - The clock does train what cross-entropy ignores: the register's high places, so that the whole register lands
    on the right hour (0.09–0.17 → 0.50–0.86 at 6 bits). It pays for this with the places the answer reads.
- **In practice it is not blind to wrong digits.** With 30% of the given digits off by one after the switch:
  - the clock collapses the cell, at 6 bits from 0.94–0.97 to 0.06–0.09, and at 7 bits from 0.37–0.38 to 0.17–0.21;
  - cross-entropy with the same noise holds: 0.83–0.87 and 0.41–0.45.
  - Wrong digits send the register to values the cell never computed exactly, negative or above N. The invariance
    needs an exact subtraction there.

**Verdict.** Clock scoring does not break the chicken-and-egg at any width tested. It removes the penalty for a wrong
guess, and with it the lesson for the cell. One number per example, a position on an N-hour clock, does not say which
digit each place of the register should hold, so a shared cell gets no consistent lesson from it. It also accepts
every cell that subtracts any multiples of N, including one that subtracts nothing. Near a solution it points the right
way but still trains the cell worse than cross-entropy. The answer's exact digits remain the only signal here that
gives the cell a consistent lesson.

## Setup (`ext_quotient_clock.py`, model `longdiv_clock`)
- longdiv's digit cell and longdiv_est's estimator, with the same initialization per seed for every option.
- **Register:** W = L + w + 1 digits (L dividend digits, w digits of N), one extra digit per division step, so no
  sequence of slots 0–9 can overflow it.
  - Arithmetic is mod 10^W: the shift and the final borrow drop the top.
  - The value is read in ten's complement (top digit ≥ 5 is negative), so a remainder that went negative reads as
    negative.
- **One slot per step**, in training and evaluation, because a blend of slots is congruent to none of them. The cell
  trains through that path only. The slot comes from one of three choices:
  - the given digits (task modq), optionally with 30% of steps off by one while training;
  - longdiv's borrow rule taken hard: the largest slot whose trial does not borrow out of the register;
  - the estimator's top slot, taught label-free. Its target is the slot whose trial result, read from the cell's
    digits, is the smallest value ≥ 0, which is the true digit when the cell is exact.
- **Scores:**
  - cross-entropy on the register's low w digits, as longdiv;
  - "clock": 1 − cos(2π(r − y)/N) in fp64, with r read from the register's argmax digits. Each digit's distribution
    gets the gradient of its expected phasor, with the other digits held at their argmax (straight-through). This
    gradient is bounded, and it does not vanish when many digits are unsure;
  - "clock_mean": the expected phasor over independent digits. Its gradient shrinks with the product of the other
    places' phasors: 1e-4 at 6 bits and 1e-7 at 10 bits at initialization.
- **Checked with the exact cell** (`selftest`):
  - true digits are exact;
  - 30% of digits off by one, or random slots, give wrong answers but stay exact mod N with clock loss 0, at 6, 7 and
    10 bits;
  - the borrow rule is exact, and the label-free targets are the true digits;
  - the copy cell is exact mod N on every example, with clock loss ~1e-17.
- **Recipe of FINDINGS_quotient.md:** fresh data, batch 256, AdamW lr 3e-3, OneCycle, clipping 1.0, 2 seeds, holdout=4
  runs for held-out moduli, Modal 4-CPU containers.
- **Measures:** FINDINGS_quotient's, plus:
  - **exact mod N**: the whole register congruent to the answer, which is what the clock asks for;
  - the cell on its own:
    - **cell**: the share of single trials right on the answer's w places, slot k on Rp in [kN, (k+1)N), where long
      division uses it;
    - **oracle**: exact with the true digits given.

## Results
**Digits given (runs 1 and 3; 2 seeds):**

| score | bits | steps | exact | exact mod N | cell |
|---|---|---|---|---|---|
| cross-entropy | 6 | 3k | 0.91–0.93 | 0.08–0.16 | · |
| cross-entropy, 30% of digits off by one | 6 | 3k | 0.89–0.91 | 0.04–0.11 | · |
| clock | 6 | 3k | 0.00 | 0.02–0.03 | · |
| clock, 30% of digits off by one | 6 | 3k | 0.00–0.02 | 0.02 | · |
| clock_mean | 6 | 3k | 0.00 | 0.02 | · |
| clock, batch 16,384 | 6 | 400 | 0.00–0.02 | 0.02–0.03 | 0.01–0.02 |
| cross-entropy | 7 | 3k | 0.38–0.39 | 0.36–0.39 | · |
| cross-entropy, 30% of digits off by one | 7 | 3k | 0.36–0.42 | 0.36–0.42 | · |
| clock | 7 | 3k | 0.00 | 0.01 | · |
| clock, 30% of digits off by one | 7 | 3k | 0.00–0.01 | 0.01 | · |
| clock_mean | 7 | 3k | 0.00 | 0.01 | · |
| clock, batch 16,384 | 7 | 400 | 0.00 | 0.01 | 0.00 |

Exact mod N is low for cross-entropy at 6 bits because cross-entropy never trains the register's high places, and no
6-bit modulus has a 0 digit where the cell could learn them. Chance for exact mod N is about 1/N: 0.022 at 6 bits,
0.011 at 7. The cell column is filled only for runs whose weights were saved.

**The basin test (run 3): 3k cross-entropy steps with the digits given, then 3k steps of:**

| bits | at the switch | cross-entropy | cross-entropy, 30% noise | clock | clock, 30% noise |
|---|---|---|---|---|---|
| 6 | 0.94–0.97 | 0.97–0.99 (cell 0.97–0.99) | 0.83–0.87 (cell 0.81) | 0.75–0.89 (cell 0.77–0.89) | **0.06–0.09** (cell 0.07–0.11) |
| 7 | 0.37–0.38 | 0.39–0.40 (cell 0.39) | 0.41–0.45 (cell 0.40–0.43) | **0.29–0.30** (cell 0.33–0.34) | **0.17–0.21** (cell 0.25–0.27) |

Cells end on the identity slot map (slot k subtracts kN) in every column except clock with noise at 6 bits, whose map
breaks (e.g. [0, 0, 2, 1, 0, 5, 0, 1, 8, 1]). At 7 bits the clock without noise keeps the tens digit at 0.37 and
lowers the units (1.00 → 0.83) and hundreds (0.96 → 0.77).

**Jointly from answers (run 2):** exact, the whole register congruent mod N, held-out moduli (trained moduli in
brackets), one bit wider, one digit wider, and every choice right:

| choice, score | bits | steps | exact | exact mod N | held-out moduli (trained) | one bit wider | one digit wider | every choice right |
|---|---|---|---|---|---|---|---|---|
| borrow rule, cross-entropy | 6 | 3k | 0.04–0.05 | 0.04 | 0.02–0.03 (0.05) | 0.00–0.01 | 0.01–0.02 | 0.02 |
| borrow rule, cross-entropy | 6 | 12k | 0.04–0.05 | 0.04–0.05 | 0.02–0.03 (0.05) | 0.01–0.02 | 0.01 | 0.02 |
| borrow rule, cross-entropy | 7 | 3k | 0.02 | 0.02 | 0.02 (0.03) | 0.01 | 0.00–0.01 | 0.01 |
| borrow rule, cross-entropy | 7 | 12k | 0.03 | 0.03 | 0.03 (0.04) | 0.01 | 0.01 | 0.01 |
| borrow rule, cross-entropy | 10 | 3k | 0.00–0.01 | 0.00–0.01 | 0.00 (0.00) | 0.00 | 0.00 | 0.00 |
| borrow rule, cross-entropy | 10 | 12k | 0.00 | 0.00 | 0.00 (0.00–0.01) | 0.00 | 0.00 | 0.00 |
| borrow rule, clock | 6 | 3k | 0.00–0.03 | 0.02–0.03 | 0.01 (0.02) | 0.00 | 0.00–0.01 | 0.02 |
| borrow rule, clock | 6 | 12k | 0.00–0.02 | 0.02–0.03 | 0.01 (0.01–0.02) | 0.00 | 0.00 | 0.02 |
| borrow rule, clock | 7 | 3k | 0.00 | 0.01 | 0.00 (0.00) | 0.00 | 0.00 | 0.01 |
| borrow rule, clock | 7 | 12k | 0.00 | 0.01 | 0.00 (0.00) | 0.00 | 0.00 | 0.01 |
| borrow rule, clock | 10 | 3k | 0.00 | 0.00 | 0.00 (0.00) | 0.00 | 0.00 | 0.00 |
| borrow rule, clock | 10 | 12k | 0.00 | 0.00 | 0.00 (0.00) | 0.00 | 0.00 | 0.00 |
| estimator, cross-entropy | 6 | 3k | 0.04–0.05 | 0.04 | 0.02–0.03 (0.04) | 0.01–0.02 | 0.01 | 0.02 |
| estimator, cross-entropy | 6 | 12k | 0.04–0.05 | 0.04–0.05 | 0.02–0.03 (0.05) | 0.01 | 0.01 | 0.02 |
| estimator, cross-entropy | 7 | 3k | 0.02 | 0.02 | 0.02 (0.02) | 0.01 | 0.01 | 0.01 |
| estimator, cross-entropy | 7 | 12k | 0.03 | 0.03 | 0.02–0.03 (0.03–0.04) | 0.01 | 0.01 | 0.01 |
| estimator, cross-entropy | 10 | 3k | 0.00 | 0.00 | 0.00–0.01 (0.00) | 0.00 | 0.00 | 0.00 |
| estimator, cross-entropy | 10 | 12k | 0.00 | 0.00 | 0.00–0.01 (0.00) | 0.00 | 0.00 | 0.00 |
| estimator, clock | 6 | 3k | 0.02 | 0.02 | 0.02–0.04 (0.02) | 0.00 | 0.00 | 0.02 |
| estimator, clock | 6 | 12k | 0.02 | 0.02 | 0.00 (0.00) | 0.00–0.01 | 0.00–0.01 | 0.02 |
| estimator, clock | 7 | 3k | 0.00–0.01 | 0.01 | 0.00 (0.00) | 0.00–0.01 | 0.00–0.01 | 0.01 |
| estimator, clock | 7 | 12k | 0.00 | 0.01 | 0.00 (0.00) | 0.00–0.01 | 0.00 | 0.01 |
| estimator, clock | 10 | 3k | 0.00 | 0.00 | 0.00 (0.00) | 0.00 | 0.00 | 0.00 |
| estimator, clock | 10 | 12k | 0.00 | 0.00 | 0.00 (0.00) | 0.00 | 0.00 | 0.00 |

**Gradient agreement (run 3).** This is the cosine between each loss's gradient for the cell's parameters and the
teacher-forced lesson toward the exact subtraction. That lesson is the cross-entropy of every step's trial toward the
exact next register, from the exact previous one. Digits are given, 2 seeds, and the reference uses 65,536 other
examples:

| bits | cell | batch | clock vs exact lesson | clock vs copy-cell lesson | answer cross-entropy vs exact lesson |
|---|---|---|---|---|---|
| 6 | at init | 256 | −0.59 to −0.50 | −0.60 to −0.51 | +0.34 to +0.42 |
| 6 | at init | 65,536 | −0.40 to +0.15 | −0.40 to +0.15 | +0.43 to +0.49 |
| 7 | at init | 256 | +0.22 to +0.70 | +0.21 to +0.70 | +0.97 to +0.98 |
| 7 | at init | 65,536 | −0.23 to −0.01 | −0.23 to −0.01 | +0.97 to +0.98 |
| 10 | at init | 256 | −0.58 to −0.09 | −0.58 to −0.09 | +0.96 to +0.98 |
| 10 | at init | 65,536 | −0.58 to −0.26 | −0.58 to −0.25 | +0.97 to +0.98 |
| 6 | after 3k cross-entropy steps (oracle 0.91–0.92) | 65,536 | +0.21 to +0.34 | +0.22 to +0.29 | −0.15 to +0.21 |
| 7 | after 3k cross-entropy steps (oracle 0.38–0.39) | 256 | +0.70 to +0.97 | −0.41 to +0.06 | +0.10 |
| 7 | after 3k cross-entropy steps (oracle 0.38–0.39) | 65,536 | +0.97 to +0.98 | −0.47 to +0.04 | −0.10 to +0.03 |
| 10 | after 3k cross-entropy steps (oracle 0.12–0.13) | 65,536 | −0.04 to +0.81 | −0.36 to +0.03 | +0.00 to +0.14 |

- **At init the clock is noise.** At 1,024, 4,096 and 16,384 rows it swings the same way (e.g. 6 bits: −0.70 to
  +0.49). At init the exact and copy lessons point the same way, so they cannot be told apart there.
- **Per division step**, the answer's cross-entropy gives the first w steps exactly zero gradient. Those steps never
  reach the answer's digits. Its lesson comes from the last step (+0.99 at 7 and 10 bits).
- **After 3k cross-entropy steps at 7 bits**, the clock's agreement comes from the last two steps (+0.75 to +0.97).
- `report --grad` prints every batch size and the per-step cosines.

## Why, in plain words
- **One hand on one clock.** The clock gives each example one number: where its register lands on an N-hour clock.
  - Every digit place moves that hand by its own amount, 10^j mod N per unit. So for a wrong result, many different
    digit changes would put the hand right.
  - Each example asks for a different mix. Through one cell shared by every place and step, those requests do not
    add up to a lesson. The answer's digits say which digit each place should hold.
- **It accepts any multiple of N.** A cell is scored zero if each slot subtracts some fixed multiple of N. That
  includes zero, so an untrained-looking copy cell is a perfect clock solution.
- **Its tolerance to wrong digits needs the subtraction to be already exact.** A wrong digit sends the register below
  zero or above N. A cell trained on true paths was never exact there. So the clock's lesson there is about the
  cell's mistakes on unfamiliar values, and following it costs the cell what it knew.
- **It behaves like a basin loss**, as in UNFOLD.md: no signal far from a solution, a right-pointing one near it. But
  "near" has to mean nearly exact, and here cross-entropy already gets the cell there faster.

## Caveats
- **Two gradient forms.** A readout through expected digit values (r = Σ 10^j E[d_j]) was not trained. Its gradient
  scales with 10^j and would be dominated by the register's top places.
- **No curriculum.** UnfoldB's clock-loss success used a 4-parameter model, Levenberg–Marquardt, an easiest-first
  curriculum and ~100k rows per step. Here the cell has ~60k parameters and Adam. Easiest-first on this datapath
  (dividends that need no reduction first) was not tried.
- **The wide register changes the cross-entropy controls too.** Their hard choices and the W-digit register differ
  from longdiv's soft choice and (w+1)-digit register. That is one reason they stay at 0.02–0.05, where longdiv
  reaches 0.44–0.59 at 6 bits.
- **The first cell measure was wrong.** Runs 1 and 2 recorded the cell against the canonical borrow table. That is
  meaningless, because the 10-way borrow is a code the cell passes to itself. Those values are not shown.
  - The cell column above was recomputed from saved weights (`recheck`): single trials on the answer's places, where
    long division uses each slot. It tracks the oracle exactness within 0.0–0.08.
  - Runs 1 and 2 did not save weights.
- **2 seeds per cell.** The gradient checks use 2 seeds per row.

## Reproduce
Run from `cpu_first/`. Training runs on Modal (4-CPU containers).

```bash
python -m microlab.ext_quotient_clock selftest
```

```bash
python -m microlab.ext_quotient_clock toy
```

```bash
modal run --detach microlab/ext_quotient_clock_modal.py --task modq --sizes 6,7 --steps 3000 --tag C1 --extra '[{"pick": "given", "score": "ce"}, {"pick": "given", "score": "ce", "noise": 0.3}, {"pick": "given", "score": "clock"}, {"pick": "given", "score": "clock", "noise": 0.3}, {"pick": "given", "score": "clock_mean"}]'
```

```bash
modal run --detach microlab/ext_quotient_clock_modal.py --task mod --sizes 6,7,10 --steps 3000,12000 --holdout 0,4 --tag C2 --extra '[{"pick": "borrow", "score": "clock"}, {"pick": "est", "score": "clock"}, {"pick": "borrow", "score": "ce"}, {"pick": "est", "score": "ce"}]'
```

```bash
modal run --detach microlab/ext_quotient_clock_modal.py --grad --sizes 6,7,10 --pre 0,3000 --tag C3g
```

```bash
modal run --detach microlab/ext_quotient_clock_modal.py --task modq --sizes 6,7 --steps 6000 --tag C3 --extra '[{"pick": "given", "score": "clock", "noise": 0.3, "pre_score": "ce", "pre_frac": 0.5}, {"pick": "given", "score": "ce", "noise": 0.3, "pre_score": "ce", "pre_frac": 0.5}, {"pick": "given", "score": "clock", "pre_score": "ce", "pre_frac": 0.5}, {"pick": "given", "score": "ce", "pre_score": "ce", "pre_frac": 0.5}]'
```

```bash
modal run --detach microlab/ext_quotient_clock_modal.py --task modq --sizes 6,7 --steps 400 --tag C3b --extra '{"pick": "given", "score": "clock", "batch": 16384}'
```

```bash
python -m microlab.ext_quotient_clock report
```

`report --grad` prints the gradient checks and `report --runs` each run's slot map and training curve.
`recheck <cell.pt ...>` recomputes the cell measures from weights saved on the Volume `microlab-quotient-clock`.
Rows are in `results_quotient_clock.jsonl`, `results_quotient_clock_grad.jsonl` and
`results_quotient_clock_cells.jsonl`; logs are in `../runs/quotient_clock/`.

## In plain language
No. Scoring long division on a clock was meant to let the subtraction learn without the quotient digits, because a
wrong digit only moves the result by whole turns of an N-hour clock. That part is true, but the clock also stops
saying which digit each place of the result should hold. With one cell shared everywhere, nothing learns from it at
6, 7 or 10 bits, even when the right digits are handed over. It is also satisfied by a cell that subtracts nothing.
Close to a solution it does point the right way, but training on it still makes the cell worse than plain
cross-entropy. With some wrong digits mixed in it wrecks the cell. The chicken-and-egg stays: the cell learns from
exact answer digits, and only along the right digits.
