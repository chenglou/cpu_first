# Hard EM for the quotient digit: the best slot as the lesson

**Question.** [FINDINGS_quotient.md](FINDINGS_quotient.md): in the long-division datapath each half learns when the
other is given.
- The leading-digit estimator learns the choice from answers when the subtraction is exact.
- The digit cell learns the subtraction when the quotient digits are given (modq).
- Trained jointly from answers, both stall from a symmetric start, ending with permuted and merged slots.

The idea tested here is a best-slot lesson from answers alone (hard EM):
- at the last division step, run every slot's trial subtraction and take the slot whose result best matches the
  answer digits;
- at earlier steps, once the later ones are right, do the same with the rest of the division run by the model;
- use that slot as a cross-entropy target for the estimator, and as the slot the cell is trained through.

Does that let the datapath learn the remainder from answers alone?

**Answer: no, at 6 or 7 bits.**
- **Exact:**
  - 6 bits: 0.04–0.05 at 3k steps and 0.04 at 12k.
  - Trained on modmix: 0.07 and 0.12.
  - Starting the estimator at q = 0: 0.02–0.03 and 0.04–0.05.
  - The soft estimator of FINDINGS_quotient.md, trained the ordinary way from answers, gets 0.53–0.54 at 12k
    (0.35–0.45 on modmix).
  - 7 bits: 0.00–0.01 for every variant (soft estimator: 0.03–0.05; on modmix 0.18–0.26).
  - Held-out moduli 0.00–0.03; one bit or one digit wider ≤ 0.01.
- **The search finds the answers, not the cell.**
  - With the answer's help, the E-step finds a path that gives the exact answer on 0.86–1.00 of test examples by step
    1k.
  - The model's own path is right on 0.01–0.13.
  - The cell does not compute the subtraction. On the inputs long division uses (Rp in [mN, (m+1)N)), no slot
    computes the remainder for any digit: the best digit per slot is right on 0.03–0.21 of trials. The cell trained
    with the digits given (FINDINGS_quotient_clock.md's modq runs) gets 0.97–0.99 on the same test.
  - The E-step compares about 310 candidate paths per example at 6 bits (510 at 7) against about 100 possible
    answers (about 250). A cell that writes plausible remainders that change with the slot is hit by chance:
    1 − 0.99^310 ≈ 0.96.
- **So the estimator is taught noise.**
  - For each true digit, its most common taught slot covers 14–29% of the examples. A slot that stood for a digit
    would make this 1.0.
  - The estimator's choices stay at chance on every step whose digit varies: 0.22–0.28 and 0.11–0.12 at 6 bits,
    0.13–0.15 and 0.10–0.12 at 7.
  - It gets right only the leading steps whose digit is always 0.
- **Three repairs, none of which help:**
  - **A weaker search** (one-step changes off, last step only, 10 candidates). The E-step then finds an exact path on
    only 0.33–0.36, the model's own path stays at 0.03–0.05 (6 bits, 3k and 12k), and the taught last slot is still
    unrelated to the digit (0.17–0.19).
  - **Tie-aware lessons**, as pre-remainder's best-fix: keep the estimator's own slot among equally good ones, and
    teach the set of tied slots. 6 bits: 0.03–0.04 at 3k and 0.04–0.05 at 12k; 7 bits: 0.02 at both.
  - **Removing the steps the answer cannot see**, by starting the division with the dividend's top w digits (below).
    6 bits/12k: 0.13–0.16, with held-out moduli 0.01–0.02. 7 bits: 0.01.

**Verdict.** The best-slot lesson does not make the remainder learnable from answers. The lesson assumes what is not
yet learned, namely that a slot stands for a digit, and the answer alone does not supply it. Among ten
interchangeable slots, some path almost always reproduces the answer, so the lesson selects paths, not digits. The cell
fits those paths entry by entry and never becomes the subtraction table it becomes when the digits are given. In EM
terms, the latent assignment is never forced to be consistent: the model can explain each example separately. Hard
EM ends below the plain soft-choice training it was meant to fix.

## What the answer can and cannot teach
Both facts were checked with the exact cell (`python -m microlab.ext_quotient_prefix reach`).
- **The first w choices never reach the answer.**
  - The datapath keeps w+1 digits and drops the top one at every shift. So its output is (v mod 10^w − q·N) mod 10^w,
    with q made of the last w choices.
  - Random slots at the first w steps leave the answer exact (1.000 at 6, 7, 8 and 10 bits).
  - Those steps' true digit is always 0, since q < N < 10^w. The answer reaches them only through the remainders they
    hand the estimator.
- **For a fifth of the moduli the top visible digit is fixed only up to parity.**
  - For N divisible by 5, changing the digit at step w−1 by an even amount changes q·N by a multiple of 10^w.
  - Changing one of the last w digits still gives an exact answer on 18–22% of examples.
- **Starting the division with the dividend's top w digits** (`ext_quotient_prefix.py`, schoolbook style) removes the
  first fact.
  - It adds no information about any example: the skipped digits are 0 for every input.
  - It does not rescue hard EM: 0.06–0.07 at 6 bits/3k, 0.13–0.16 at 12k, 0.01 at 7 bits.
  - The soft estimator with the same start gets 0.40–0.70 at 6 bits/12k (full datapath: 0.53–0.54), but 0.03 on
    held-out moduli, and 0.05–0.06 at 7 bits.

## Method (`ext_quotient_em.py`, model `longdiv_em`)
- **Model:** longdiv_est's parameters and initialization: the estimator sees the top 3 digits of 10R + v_t and the top
  2 of N; one shared digit cell; ten trial subtractions per step. Only the training differs. Evaluation is
  longdiv_est's soft forward, which never sees the answer, so every measure is FINDINGS_quotient's.
- **E-step**, per training example, no gradient:
  - the model's own path: the estimator's top slot at every step;
  - at the last step, all ten trial results are scored against the answer digits (most digits right, then the
    answer's log-likelihood), and the best slot is taken;
  - one-step changes: for every earlier step t and slot k, step t takes slot k, the later steps take the estimator's
    top slot on the changed remainders, and the last step takes its best slot;
  - the path that explains the answer best wins, and ties keep the model's path.
- **M-step:**
  - the cell gets the answer's cross-entropy through the winning path, one slot per step, as modq trains it through
    the given digits;
  - the estimator gets cross-entropy toward the winning path's slot: at the last step always, and at every step when
    the winning path is exactly right. Its inputs are detached.
- **Variants:**
  - plain (symmetric start);
  - trained on modmix (easy dividends mixed in);
  - starts at q = 0 (est_bias0 = 4), FINDINGS_quotient's names.
- **Diagnostics** (`ext_quotient_em_set.py`): tie-aware lessons (`longdiv_em_set`) and the last-step-only E-step
  (`em_depth=0`), with weights saved and three measures:
  - which digit each slot's trial computes the remainder for, on Rp in [mN, (m+1)N);
  - whether the taught slot is a function of the true digit;
  - how many last-step slots give the answer.
- **Recipe of FINDINGS_quotient.md:** fresh data, batch 256, AdamW lr 3e-3, OneCycle, clipping 1.0, 2 seeds, holdout=4
  runs for held-out moduli, Modal 4-CPU containers. A training step costs 3.2× (6 bits) to 3.3× (7 bits) a soft
  longdiv_est step. A 7-bit 12k cell takes about 1.7 h.

## Results
Exact on 2,000 fresh examples, the range over 2 seeds. Columns are as in FINDINGS_quotient.md; "exact, hard choice"
takes the estimator's top slot at every step instead of its soft blend.

| variant | bits | steps | exact | held-out moduli (trained) | one bit wider | one digit wider | every choice right | exact, hard choice |
|---|---|---|---|---|---|---|---|---|
| hard EM | 6 | 3k | 0.04–0.05 | 0.02 (0.04–0.05) | 0.00 | 0.00 | 0.03–0.04 | 0.04–0.06 |
| hard EM | 6 | 12k | 0.04 | 0.02 (0.04–0.05) | 0.01 | 0.00–0.01 | 0.02–0.03 | 0.04–0.05 |
| hard EM | 7 | 3k | 0.00 | 0.00–0.01 (0.00–0.01) | 0.00–0.01 | 0.01 | 0.01 | 0.01 |
| hard EM | 7 | 12k | 0.00 | 0.00–0.01 (0.00) | 0.00–0.01 | 0.01 | 0.01 | 0.01 |
| hard EM, starts at q = 0 | 6 | 3k | 0.02–0.03 | 0.02–0.03 (0.03–0.04) | 0.00–0.01 | 0.00–0.01 | 0.02–0.04 | 0.02–0.04 |
| hard EM, starts at q = 0 | 6 | 12k | 0.04–0.05 | 0.01–0.02 (0.04–0.05) | 0.00 | 0.00–0.01 | 0.03 | 0.05 |
| hard EM, starts at q = 0 | 7 | 3k | 0.00–0.01 | 0.00 (0.00–0.01) | 0.00–0.01 | 0.00–0.01 | 0.01 | 0.01 |
| hard EM, starts at q = 0 | 7 | 12k | 0.00–0.01 | 0.00–0.01 (0.01) | 0.00 | 0.00–0.01 | 0.01 | 0.01 |
| hard EM, trained on modmix | 6 | 3k | 0.07 | 0.02–0.03 (0.07–0.09) | 0.01 | 0.01 | 0.02 | 0.06–0.07 |
| hard EM, trained on modmix | 6 | 12k | 0.12 | 0.02–0.03 (0.11–0.15) | 0.01 | 0.01 | 0.02–0.03 | 0.11–0.13 |
| hard EM, trained on modmix | 7 | 3k | 0.00–0.01 | 0.00 (0.00) | 0.00–0.01 | 0.00–0.01 | 0.01–0.02 | 0.01 |
| hard EM, trained on modmix | 7 | 12k | 0.00 | 0.00–0.01 (0.00–0.01) | 0.00–0.01 | 0.01 | 0.01 | 0.01 |
| *estimator (soft, FINDINGS_quotient)* | 6 | 3k | 0.04–0.05 | 0.02 (0.04–0.05) | 0.00–0.01 | 0.00–0.02 | 0.02 | |
| *estimator (soft)* | 6 | 12k | 0.53–0.54 | 0.02–0.03 (0.05) | 0.01 | 0.01 | 0.23–0.27 | |
| *estimator (soft)* | 7 | 3k | 0.03 | 0.02–0.03 (0.05–0.06) | 0.01 | 0.01 | 0.01 | |
| *estimator (soft)* | 7 | 12k | 0.05 | 0.02 (0.06) | 0.01 | 0.00–0.01 | 0.01 | |
| *estimator (soft), starts at q = 0* | 6 | 12k | 0.05–0.18 | | 0.01 | 0.00–0.01 | 0.02–0.08 | |
| *estimator (soft), starts at q = 0* | 7 | 12k | 0.05–0.06 | | 0.01–0.02 | 0.00 | 0.01 | |
| *estimator (soft), trained on modmix* | 6 | 12k | 0.35–0.45 | | 0.01 | 0.00–0.01 | 0.28–0.40 | |
| *estimator (soft), trained on modmix* | 7 | 12k | 0.18–0.26 | | 0.02 | 0.00–0.01 | 0.16–0.24 | |
| *longdiv, borrow choice (soft)* | 6 | 12k | 0.44–0.59 | 0.02–0.03 (0.04–0.06) | 0.01 | 0.00–0.01 | 0.14–0.49 | |
| *quotient digits given (modq)* | 6 | 12k | 0.99–1.00 | 0.47–0.49 (1.00) | 0.07–0.09 | 0.07–0.08 | | |
| *quotient digits given (modq)* | 7 | 12k | 0.39–0.40 | 0.21–0.23 (0.42–0.43) | 0.15 | 0.04 | | |

**What the E-step sees** (on the test examples, with the answer; the model's own choices per division step, first step
first, read through each slot's most common digit; the estimator's agreement with its lessons at the end of training):

| variant | bits | steps | model's own path right | best one-step-changed path right | choices right per step | estimator agrees with lessons |
|---|---|---|---|---|---|---|
| hard EM | 6 | 3k | 0.02–0.06 | 0.89–0.93 | 1.00 1.00 0.23–0.24 0.12 | 0.61–0.62 |
| hard EM | 6 | 12k | 0.02–0.05 | 0.92–0.99 | 1.00 1.00 0.24 0.12 | 0.62 |
| hard EM | 7 | 3k | 0.01 | 0.89–0.94 | 1.00 1.00 1.00 0.95 0.13 0.10–0.12 | 0.69–0.70 |
| hard EM, trained on modmix | 6 | 12k | 0.10–0.13 | 0.93–0.98 | 1.00 0.96–1.00 0.22–0.25 0.11–0.12 | 0.79–0.80 |
| hard EM, starts at q = 0 | 7 | 12k | 0.01 | 0.98–1.00 | 1.00 1.00 1.00 0.95 0.14–0.15 0.11 | 0.70 |

The steps at 1.00 are the ones whose digit is always 0, and 0.95 is step 2 at 7 bits, whose digit is 0 unless q ≥ 100.
The last two are at the share of the most common digit, which is chance. On modmix's own distribution, where most
dividends need no reduction, the model's path is right on 0.42.

**Diagnostics from saved weights** (2 seeds each). The columns:
- slot right for its best digit: per slot, the best share over digits of trials right on Rp in [mN, (m+1)N);
- taught slot = digit's modal slot: steps 1 and 0;
- slots giving the answer at the last step: shares of 1 / 2 / 3+ on the winning path.

| E-step | bits | steps | model's own path | best path right | slot right for its best digit | taught slot = digit's modal slot | slots giving the answer at the last step |
|---|---|---|---|---|---|---|---|
| one-step changes (as above) | 6 | 3k | 0.04–0.06 | 0.89 | 0.05–0.21 (no digit covered) | 0.23 / 0.16–0.17 | 0.72–0.81 / 0.06–0.14 / 0.01–0.04 |
| tie-aware lessons | 6 | 3k | 0.04–0.05 | 0.99–1.00 | 0.06–0.17 (none) | 0.21–0.29 / 0.14 | 0.94–0.96 / 0.01–0.02 / 0.01–0.04 |
| last step only | 6 | 3k | 0.03–0.04 | 0.33–0.36 | 0.03–0.09 (none) | 1.00 / 0.17 | 0.31–0.35 / 0.01–0.02 / 0.00 |
| last step only | 6 | 12k | 0.04–0.05 | 0.35 | 0.04–0.13 (none) | 0.95–1.00 / 0.17–0.19 | 0.34 / 0.01–0.02 / 0.00–0.01 |
| one-step changes (as above) | 7 | 3k | 0.01 | 0.89 | 0.04–0.16 (none) | 0.16–0.17 / 0.14–0.15 | 0.88–0.89 / 0.00–0.01 / 0.00 |
| tie-aware lessons | 7 | 3k | 0.02–0.03 | 0.89–0.92 | 0.02–0.16 (none) | 0.19–0.20 / 0.14 | 0.84–0.88 / 0.02–0.03 / 0.02–0.03 |
| *exact cell, true digits* | | | 1.00 | 1.00 | 1.00 for slot k = digit k | 1.00 / 1.00 | 1.00 / 0 / 0 |

At 7 bits the two entries are steps 1 and 0. Step 2's digit is almost always 0, and its taught slot follows it on
0.94–0.96. With the last-step-only E-step, the estimator settles on one constant slot at step 1, which makes the step-1
entry 1.00.

**Schoolbook start** (`ext_quotient_prefix.py`): the division starts with the dividend's top w digits, so only the w
steps the answer can see remain.

| variant | bits | steps | exact | held-out moduli (trained) | one bit wider | one digit wider | every choice right |
|---|---|---|---|---|---|---|---|
| hard EM | 6 | 3k | 0.06–0.07 | 0.03 (0.07–0.11) | 0.00 | 0.00 | 0.03 |
| hard EM | 6 | 12k | 0.13–0.16 | 0.01–0.02 (0.18) | 0.00 | 0.00 | 0.02–0.03 |
| hard EM | 7 | 3k | 0.01 | 0.01–0.02 (0.01–0.02) | 0.00–0.01 | 0.00 | 0.02 |
| hard EM | 7 | 12k | 0.01 | 0.01–0.02 (0.01) | 0.00–0.01 | 0.00–0.01 | 0.03 |
| soft estimator | 6 | 3k | 0.04–0.05 | 0.02–0.03 (0.14–0.23) | 0.00–0.01 | 0.00–0.02 | 0.03 |
| soft estimator | 6 | 12k | 0.40–0.70 | 0.03 (0.29–0.30) | 0.00–0.01 | 0.00–0.01 | 0.08–0.40 |
| soft estimator | 7 | 3k | 0.03 | 0.02 (0.04) | 0.01 | 0.00–0.01 | 0.01 |
| soft estimator | 7 | 12k | 0.05–0.06 | 0.02–0.03 (0.06–0.08) | 0.01 | 0.00–0.01 | 0.02 |

## Why, in plain words
- **The lesson presumes the part that is missing.**
  - "Teach the slot whose result matches the answer" makes sense once each slot subtracts a fixed multiple of N.
    Then exactly one slot matches, and it names the digit.
  - Early on no slot means anything, and with ten slots and a few steps there are hundreds of paths. One of them
    usually lands on the right answer by luck, about as often as there are answers to hit.
  - The lesson then rewards that path, whatever its slots. The cell learns to make those particular paths come out
    right (a lookup over the examples it has seen), and the estimator is told a different slot for the same digit
    from one example to the next.
- **A weaker search doesn't fix it.** With only the last step searched, the lucky hits drop to a third, but the slots
  still never come to mean digits. At 6 bits every test input also occurs in training, so fitting example by example
  always pays.
- **The plain soft training gets further at 6 bits** (0.53–0.54 at 12k), but FINDINGS_quotient.md found that it too
  ends in a lookup, with permuted and merged slots. Its gradient reaches every slot on every example, so it can fit
  the seen examples. Under hard EM, one winner takes each example's whole lesson, and noise decides the winner.
- **The answer cannot see the first w choices,** and for moduli divisible by 5 it cannot see the parity of the top
  visible digit. Taking those steps out (the schoolbook start) shortens the chain but does not change the outcome.

## Caveats
- **The answer is used only to rank candidate paths during training.** Every measure runs the model as longdiv_est
  does. "Best path right" uses the answer and is a diagnostic only.
- **The E-step's rules matter, and only three were tried:**
  - the one-step-change search with ties going to the model's path;
  - the tie-aware set lessons;
  - the last step only.
  - A beam over whole paths, or a soft (probabilistic) E-step, were not tried. Neither changes the core problem: the
    answer allows many paths.
- **The diagnostics ran at 6 bits (3k/12k steps) and 7 bits (3k)**, 2 seeds each. The 7-bit 12k rows of the main grid
  show the same signature (best path 0.89–1.00, own path 0.01, choices at chance) without the slot table.
- **Measure history.** The first slot table (`slot_table` in the S1/S2 rows) sampled Rp from [0, 10N) for every slot
  and compared all w+1 places. That is out of range for most slots, and it includes a place the answer never trains.
  The table here is the in-range one, recomputed from saved weights (`ext_quotient_em_set.py recheck`). It reads the
  exact cell as 1.00 for slot k = digit k.
- **2 seeds per cell; 6 and 7 bits only.** 10 bits was not run: 7 bits already fails.

## Reproduce
Run from `cpu_first/`. Training runs on Modal (4-CPU containers).

```bash
python -m microlab.ext_quotient_em selftest
```

```bash
python -m microlab.ext_quotient_prefix reach
```

```bash
modal run --detach microlab/ext_quotient_em_modal.py --sizes 6,7 --steps 3000,12000 --holdout 0,4 --tag E1 --extra '[{}, {"train_on": "modmix", "name": "_mix"}, {"est_bias0": 4.0, "name": "_b4"}]'
```

```bash
modal run --detach microlab/ext_quotient_prefix_modal.py --models longdiv_em_prefix,longdiv_est_prefix --sizes 6,7 --steps 3000,12000 --holdout 0,4 --tag P1
```

```bash
modal run --detach microlab/ext_quotient_em_set_modal.py --models longdiv_em,longdiv_em_set --sizes 6,7 --steps 3000 --tag D1
```

```bash
modal run --detach microlab/ext_quotient_em_set_modal.py --models longdiv_em --sizes 6 --steps 3000,12000 --tag D2 --extra '{"em_depth": 0, "name": "_d0"}'
```

```bash
python -m microlab.ext_quotient_em report
```

`report --steps` prints each run's choices per step and its E-step curve.
`python -m microlab.ext_quotient_em_set recheck <cell.pt ...>` recomputes the diagnostics from weights saved on the
Volume `microlab-quotient-em-set`. Rows: `results_quotient_em.jsonl`, `results_quotient_prefix.jsonl`,
`results_quotient_em_set.jsonl`, `results_quotient_em_set_cells.jsonl`. Logs: `../runs/quotient_em/`,
`../runs/quotient_prefix/`, `../runs/quotient_em_set/`.
