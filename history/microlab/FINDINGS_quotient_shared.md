# Fixed slot meanings in the long-division datapath: TINYCORE's fix, with microlab's MLP cell

**Question.** [FINDINGS_quotient.md](FINDINGS_quotient.md): trained from answers, the long-division datapath's two
learned parts, the digit cell and the quotient-digit choice, each learn when the other is given and stall together
(6 bits 0.44–0.59, 7 bits 0.05). [TINYCORE.md](../tinycore/TINYCORE.md) traces that to the learned slot labels and
predicts a fix (§6):
1. replace longdiv's learned slot embedding with one shared subtractor cell fed the exact digits of q·N;
2. teach it with longdiv_est's soft estimator choice, from the dividend's top digits, one quotient digit at a time;
3. then switch to longdiv's own borrow choice.

In the tabular tiny core at base 10 that route was exact in 11/11, 11/11 and 7/8 seeds at 6, 7 and 10 bits. The same
route with learned slot labels failed (0/4). Does it hold with microlab's MLP cell?

**Answer: the mechanism holds, but the route as specified is far less reliable than with tables. Its two weak points
are specific to the MLP. With one change it is exact at 6 and 10 bits, and a rule when trained at 10 bits; 7 bits
stays unreliable.**
- **Fixed slot meanings are what make the subtractor learnable, as predicted.**
  - All 84 runs with the hold (below) ended the estimator phase with the identity slot map and a subtractor that is
    right on the true path: 0.96–1.00 exact given the true quotient digits (0.83–1.00 when the borrow is a bit).
    Without the hold, so did all 13 runs whose estimator phase got started.
  - The same route with longdiv's own cell (a learned label per slot) never got there: 0/9 runs. Exact given the
    true digits was 0.00–0.06, the slot maps were permuted and merged (e.g. [2, 5, 4, 1, 9, 0, 2, 4, 3, 3]), and
    final exact was 0.45–0.71 at 6 bits, 0.05–0.06 at 7 and 0.00–0.01 at 10.
- **The route exactly as specified: 10 of 24 runs exact** (exact ≥ 0.99 with the model's own borrow; 4 seeds, plain
  and held-out-moduli runs pooled). That is 5/8 at 6 bits, 2/8 at 7 and 3/8 at 10, against the tiny core's 11/11,
  11/11 and 7/8. The 14 misses split into two failure modes the tables never showed:
  - **The estimator does not stay undecided (8 runs).** microlab's MLP estimator, trained with AdamW, puts every
    example on a single slot within about 100 steps (entropy 2.30 → 0.00 nats). That happens before the subtractor
    has learned anything, so the subtractor trains only through that one slot's trial and never becomes subtraction.
  - **The switch to the own borrow wrecks a subtractor that is already right (6 runs, at 7 and 10 bits).** The
    estimator phase never uses the final borrow, so it leaves it uncalibrated. The own borrow keeps the largest slot
    whose trial does not borrow, so it multiplies the trials' P(no borrow), and any miscalibration sends weight to
    wrong slots. Right after the switch the own borrow is poor at every width (0.002–0.56 exact in the hold runs
    below), and phase B has to calibrate the final borrow while choosing with it. When it chooses wrong, the wrong slots' lessons corrupt the
    digits.
- **Holding the choice uniform for the first 400 steps removes the first failure.** This makes TINYCORE's "starts
  undecided" explicit. None of the 84 runs with the hold failed to start.
  - **6 bits: 8/8. 10 bits: 8/8.** Every run was exact 100–200 steps after the switch, with the hard datapath too
    (argmax choices, snapped digits), and exact on held-out moduli (1.00 at both widths).
  - **Trained at 10 bits, the result is a rule.** All 200 subtractor rows are right, and the same weights are exact
    at 6, 7, 10 and 20 bits (0.97–1.00 at 14).
  - **Trained at 6 bits, it is a table of the rows the data visits.** It is right on the 162 rows the true path uses
    and wrong on (all but at most one of) the 38 it never visits. So the same weights give about 0.8 at 7 bits, 0.5
    at 10, 0.15 at 14 and 0.02 at 20. The MLP did not fill the gaps by sharing structure, which TINYCORE §7 left
    open.
  - **7 bits: 1/8.** A rerun of the four plain runs again solved the same seed (1/4). The estimator phase is as good
    as at the other widths (0.97–1.00 given the true digits, identity maps), but the own borrow ends anywhere in
    0.01–0.99, mostly 0.83–0.98.
- **Why 7 bits fails at the switch: the MLP's borrow comes out as an analog code.**
  - After the estimator phase, the true trial's P(borrow out) is about 0.5 (0.48–0.59) when there is no borrow and
    0.82–0.99 when there is (later steps, places 1–3; two seeds). The next place reads that code correctly, so the digits are
    right on the true path.
  - On one-hot inputs, though, only 46–63% of the 200 rows are right. The own borrow reads the code as a
    probability, so valid trials show P(no borrow) ≈ 0.4, and the choice falls apart: 0.19–0.24 exact right after
    the switch, measured on saved weights. Phase B then has to rebuild the code while choosing with it.
  - At 10 bits all 200 rows came out right on one-hot inputs, so phase B only had to calibrate the final borrow,
    which took 100–200 steps. At 6 bits 0.68–0.80 of all rows came out right, and phase B fixed the rest of the
    visited ones.
- **Four repairs at 7 bits, none reliable:**

  | repair | 7 bits |
  |---|---|
  | hand the choice over gradually (500 steps) | 0/8 |
  | lower learning rate after the switch | 0/8 |
  | the borrow a bit between places, during both phases | 2/8 (and 0/8 at 10 bits) |
  | also read the final borrow as a bit after the switch | 2/8 (0/8 at 6 bits, 1/8 at 10) |

  - A bit-valued borrow does make the estimator phase's table right on one-hot inputs: all 200 rows in one seed, 199
    in another.
    Read as a bit, one such table is exact at the switch with no further training: 1.000, 0.997, 0.988 and 0.907
    at 6, 7, 10 and 14 bits.
  - But training on bit-valued borrows after the switch is unstable. Any example where no trial reads as valid
    leaves the remainder at nothing, and the run dies.
  - The five 7-bit runs that reached 1.000 are rules: all 200 rows right and 0.998–1.00 at every width from 6 to
    20 bits. The three that stopped at 0.99 are not: 0.74–0.98 of their rows are right.
- **For 7 bits, the working answer is to train at 10 bits.** All 16 hold runs trained at 10 bits are exact at 7 bits
  with the same weights, as TINYCORE §6 step 5 advises ("train at the widest size wanted").
- **Ablations (2–3 seeds each, the route as specified otherwise):**
  - Staging is necessary: without it, 0/6.
  - So is starting each stage from v // 10^k: always starting from the top w digits gives 0/6.
  - The coarse 2/1 window gives 3/3 at 6 bits and 0/3 at 7 and at 10.
  - The own borrow from the start, with no estimator, solves 6 bits (2/2, exact by the end of staging), unlike the
    tiny core at microlab's moduli. It gives 0.94 and 0.99 at 7 bits and fails at 10 (0/2).

**Verdict.** TINYCORE's diagnosis carries over. With fixed slot meanings the estimator teaches the subtractor, to the
identity slot map, at every width, and with learned slot labels it never does. Its recipe does not carry over as
written. The MLP estimator must be held undecided explicitly. The handover to the own borrow then works only where
the estimator phase leaves the subtractor right on one-hot inputs, or nearly so: true at 10 bits, mostly true at 6,
false at 7, where the MLP's borrow came out as an analog code. With the hold, microlab's datapath learns x mod N from
answers alone at 6 and 10 bits in 8/8 runs each. Trained at 10 bits it is a rule: exact at 6–20 bits and on moduli
never trained. It is the first route in microlab's long-division datapath to learn the remainder from answers; the
earlier best there was 0.44–0.59 at 6 bits and 0.18–0.26 at 7, with easy dividends mixed in
([FINDINGS_quotient.md](FINDINGS_quotient.md)).

## Against the tiny-core predictions (TINYCORE.md §6)

| prediction | microlab, MLP cell |
|---|---|
| fixed slot meanings let the estimator teach the subtractor | **yes**: identity map and 0.96–1.00 given the true digits in every run that started (all 84 with the hold; 0.83–1.00 when the borrow is a bit) |
| the same route with learned slot labels fails (tiny core 0/4 at 6 bits) | **yes**: 0/9 at 6, 7 and 10 bits; permuted and merged maps; 0.00–0.06 given the true digits |
| exact at 6, 7 and 10 bits (tiny core 11/11, 11/11, 7/8) | **as specified, no**: 5/8, 2/8, 3/8. **With a 400-step uniform hold**: 8/8, 1/8, 8/8 |
| the estimator only bootstraps; the own borrow then makes the datapath exact within a few hundred steps | yes at 6 and 10 bits (100–200 steps); no at 7 bits (the switch fails) |
| staging is needed | yes: 0/6 without |
| keep microlab's window (3/2) at 10 bits; 2/1 may not start | yes: 2/1 gives 0/3 at 10 bits (and 0/3 at 7), 3/3 at 6 |
| train at the widest size wanted; a table is right only where its data goes | yes: trained at 10 bits, exact at 6–20 bits; trained at 6 bits, wrong on (nearly) exactly the unvisited rows |
| the own borrow alone cannot start at microlab's moduli (tiny core 0/3 at 6 bits) | **no at 6 bits** (2/2); yes at 10 (0/2); 7 bits 0/2 at 0.94 and 0.99 |
| (not predicted) the undecided start | the MLP estimator commits within ~100 steps; it has to be held uniform |
| (not predicted) the switch | needs a subtractor right on one-hot inputs; the MLP's borrow came out as an analog code at 7 bits |

## Setup (`ext_quotient_shared.py`, model `longdiv_shared`)
- **Datapath.** ext_quotient's: an MSB-first scan, all ten trials at once, a soft choice, digit distributions. It
  starts the schoolbook way, as ext_quotient_prefix does: from the dividend's top digits, running only the steps
  whose quotient digit can be nonzero.
- **Cell.** One shared subtractor (r_j, m_j, borrow) → (digit, borrow): longdiv's MLP (linear embeddings of the
  digit distributions, d = 32, two GELU layers of 128) with a binary borrow and no slot embedding. Slot q is fed the
  exact digits m of q·N (computed: multiplying N by one known digit). 55,510 parameters with the estimator.
- **Phase A, estimator.** longdiv_est's MLP estimator on the top 3 places of 10R + v_t and the top 2 digits of N,
  soft.
  - Staged: stage k = 1..w trains on v < N·10^k and runs the k steps whose digit can be nonzero, from R = v // 10^k
    (below N). Stage w is microlab's own mod distribution, v < N², from the dividend's top w digits.
  - At most 1,000 steps per stage, advancing early once 0.99 exact on the stage's own validation sample; 1,500 at
    the last stage.
  - So at most 2,500 / 3,500 / 4,500 steps at 6 / 7 / 10 bits. No stage before the last ever reached 0.99 at 7 or
    10 bits. The window caps a hard estimator at 0.89 / 0.85 for one step and at 0.78 / 0.64 for all steps
    (`window`); the soft one reached at most 0.88 / 0.86 at stage 1.
- **Phase B, own borrow.** longdiv's rule, the largest q whose trial does not borrow, softly. 2,000 steps on
  v < N², with a fresh AdamW on the cell alone.
- **Training.** microlab's: batch 256 of fresh examples, AdamW lr 3e-3, weight decay 0.01, gradient clip 1.0. Phase
  A uses a 100-step linear warmup and then a constant rate; phase B uses OneCycle.
- **Changes found here (options):**
  - `hold=400`: for the first 400 steps every slot gets equal odds, and the estimator is not consulted.
  - `handover=500`: phase B's choice moves from the estimator to the own borrow over 500 steps.
  - `lr_b`: phase B's learning rate.
  - `hard_borrow`: the borrow is a bit (argmax, straight-through gradient): `True` between places, `"all"` the final
    one too. `hard_b` sets phase B's value. `from_a` starts phase B from saved phase-A weights.
- **Control.** The same route with longdiv's own cell, `slots="learned"`: a learned label per slot and carries 0..9.
- **Data.** Odd N in [2^(b−1), 2^b). Holdout runs (`holdout=4`) never train on N with (N // 2) % 4 = 0, a quarter of
  the moduli.
- **Measures, after each phase.** All on microlab's 2,000 test examples (seed 10,000), with the soft datapath as
  everywhere in microlab:
  - exact with the model's own choices, and exact given the true quotient digits;
  - the hard datapath (argmax choices, snapped digits);
  - held-out moduli;
  - the slot maps (ext_quotient.choices: slot i stands mostly for digit map[i]);
  - the subtractor on its 200 one-hot rows: all of them, the rows the true path visits, and the borrow alone on the
    rows the over-subtracting trials visit;
  - the same weights at 6, 7, 10, 14 and 20 bits.

## Results
"Solved" means exact ≥ 0.99 with the own borrow after the switch. Ranges are min–max over runs. Each cell pools 4
seeds × {all moduli, holdout 4} unless noted. Phase A is the estimator, phase B the own borrow. The full tables are
in [runs/quotient_shared/report.md](../runs/quotient_shared/report.md) (`python -m microlab.ext_quotient_shared_report`).

### 1. The route as specified, the hold, and the learned-slot control

| variant | bits | solved | phase A: exact / given the true digits | phase B: exact / given | held-out moduli (phase B) |
|---|---|---|---|---|---|
| as specified (S1) | 6 | 5/8 | 0.04–0.86 / 0.05–1.00 | 0.07–1.00 / 0.04–1.00 | 0.02–1.00 |
| as specified | 7 | 2/8 | 0.02–0.85 / 0.01–0.98 | 0.02–1.00 / 0.02–1.00 | 0.03–1.00 |
| as specified | 10 | 3/8 | 0.00–0.78 / 0.00–1.00 | 0.00–1.00 / 0.00–1.00 | 0.00–1.00 |
| **choice held uniform 400 steps (S2)** | 6 | **8/8** | 0.81–0.97 / 0.99–1.00 | **1.00** / 1.00 | **1.00** |
| **choice held uniform 400 steps** | 7 | 1/8 | 0.80–0.85 / 0.97–1.00 | 0.01–0.99 / 0.10–1.00 | 0.02–0.99 |
| **choice held uniform 400 steps** | 10 | **8/8** | 0.73–0.76 / 1.00 | **1.00** / 1.00 | **1.00** |
| learned slots, same route (S1L; 3 seeds, all moduli) | 6 | 0/3 | 0.42–0.48 / 0.02–0.03 | 0.45–0.71 / 0.02 | · |
| learned slots | 7 | 0/3 | 0.25–0.41 / 0.01–0.06 | 0.05–0.06 / 0.02–0.04 | · |
| learned slots | 10 | 0/3 | 0.09–0.17 / 0.00 | 0.00–0.01 / 0.00 | · |

- **For reference, microlab's earlier from-answers results.**
  - longdiv: 0.44–0.59 at 6 bits and 0.05–0.06 at 7 after 12k steps.
  - longdiv_est: 0.53–0.54 at 6 bits and 0.05 at 7.
  - longdiv_est_prefix, the same start with learned slots and no staging: 0.40–0.70 at 6 bits and 0.05–0.06 at 7.
- **The hold's runs at 6 and 10 bits** were exact 100–200 steps after the switch. They took 2.5 and 11–15 minutes on
  a 4-CPU container.
- **The 7-bit hold runs** ended at 0.50, 0.83, 0.95 and 0.99 (all moduli) and 0.01, 0.95, 0.97 and 0.98 (holdout 4).
  The rerun of the four plain runs ended at 0.83, 0.92, 0.95 and 0.99.

**How the 14 misses of the route as specified break down:**
- **Never started (8):** phase A ended with exact given the true digits below 0.10 and a collapsed slot map, such as
  [0, 0, 2, 0, 0, 0, 0, 0, 0, 0]. That was 3 runs at 6 bits, 2 at 7 and 3 at 10. In one 6-bit run the own borrow
  still reached 0.985.
- **Lost at the switch (6):** phase A was right (0.96–1.00 given, identity map), but exact was 0.003–0.17 right
  after the switch. That was 4 runs at 7 bits and 2 at 10. One recovered to 0.988; the others ended at 0.003–0.034.
- Two more 6-bit runs started only partly (0.38 and 0.60 given), and the own borrow then finished them (0.998 and
  1.000).

### 2. The first failure: the estimator commits before the subtractor learns
Stage 1 at 6 bits, seed 1, which never started in the route as specified (`python -m microlab.ext_quotient_shared
start --seeds 1`; seeds 0 and 1 ran locally in seconds):

| step | estimator entropy (nats) | P(true digit) | slots its argmax uses | subtractor rows right | exact given the true digits |
|---|---|---|---|---|---|
| 0 | 2.30 | 0.100 | 8 (1,984 of 2,000), 5 | 0.04 | 0.004 |
| 50 | 1.65 | 0.099 | 2 (all) | 0.01 | 0.019 |
| 100–400 | 0.00 | 0.100 | 2 (all) | 0.03–0.06 | 0.04–0.08 |

- Seed 0 also collapsed (onto slot 5 at step 50, slot 2 at step 100), then recovered: P(true) 0.46 by step 400.
- **With the choice held uniform for 400 steps,** seed 1 reaches P(true) 0.57 at step 500, 0.92 at step 800, and
  0.997 exact given the true digits.
- **Under a uniform choice every trial gets an equal share of the lesson.** Only the right trial is consistently
  right, so the shared subtractor drifts toward subtraction. That is TINYCORE's argument, which needs the choice to
  wait.
- **A 10× lower learning rate on the estimator also rescues seed 1** (a local check: 0.95 given the true digits by
  step 700). It is less clean.

### 3. The second failure: the switch to the own borrow at 7 bits
On saved phase-A weights, using exact long division with one-hot inputs on the test examples (`python -m
microlab.ext_quotient_shared probe <weights> --bits 7`):

| 7-bit phase-A weights | true trial's P(borrow out), true borrow 0 / 1 (later steps, places 1–3) | final P(no borrow): trials that don't borrow / that do | own borrow right after the switch: as is / final borrow read as a bit |
|---|---|---|---|
| soft borrow (hold run, seed 0) | 0.51–0.59 / 0.82–0.92 | 0.37–0.40 / 0.03 | 0.19 / 0.22 |
| soft borrow (hold run, seed 3) | 0.48–0.57 / 0.96–0.99 | 0.43 / 0.002 | 0.24 / 0.26 |
| bit between places (seed 0) | 0.00–0.02 / 1.00 | 0.99 / 0.28 (0.94 of them above 0.1) | 0.013 / **0.997** |
| bit between places (seed 1) | 0.00–0.08 / 0.99–1.00 | 0.97–1.00 / 0.04 (none above 0.1) | 0.66 / 0.92 |

- **With soft borrows the code is analog.** "No borrow" reads about 0.5. The next place decodes it, so the digits
  are right (0.98 given the true digits). On one-hot inputs, though, 55–84 of the 200 rows have a wrong digit and
  39–73 a wrong borrow, most of those with borrow-in 1.
- **The own borrow keeps the largest valid slot**, P(q) = valid_q · Π_{q' > q} (1 − valid_q'), so it needs P(no
  borrow) near 1 on valid trials and near 0 on the others.
- **With the borrow a bit between places,** the table is nearly right on one-hot inputs: 0–15 rows with a wrong
  digit and 0–30 with a wrong borrow (none at all in seed 0, one borrow in seed 1). But the top place's final borrow
  stays soft on over-subtracting trials (0.28 in seed 0). The product then puts most of the weight on the top slots,
  and that run collapsed in phase B from a perfect table.
- **Read as a bit,** the same seed-0 table is exact at the switch at 6, 7, 10 and 14 bits: 1.000, 0.997, 0.988 and
  0.907.

The repairs, on the same 7-bit configuration with the hold (4 seeds; S3 and the bit variants also with holdout 4):

| phase B variant | 6 bits | 7 bits | 10 bits | 7-bit final exact |
|---|---|---|---|---|
| own borrow (S2 above) | 8/8 | 1/8 | 8/8 | 0.01–0.99 |
| handover over 500 steps (S3) | 7/8 | 0/8 | 8/8 | 0.01–0.99 |
| lr 1e-3 / 3e-4 after the switch (D7, all moduli) | · | 0/4 / 0/4 | · | 0.32–0.98 / 0.05–0.95 |
| borrow a bit between places, both phases (S4) | 7/8 | 2/8 | 0/8 | 0.03–1.00 |
| S4's phase A, then every borrow a bit (S5) | 0/8 | 2/8 | 1/8 | 0.01–1.00 |

- **The bit readout during training is unstable.** If no trial reads as valid, every weight is 0, the remainder
  underflows, and the loss sits at ln 10 with no gradient. That is where 13 of the 21 failed S5 runs ended, with
  exact at the share of remainder-0 examples (0.012 at 7 bits). The soft readout fell into the same state twice
  (S2 and S3, both 7-bit seed 1 with holdout 4).
- **At 6 bits the bit readout trips over the 38 never-visited rows.** Those runs stall at loss 1.8 and exact 0.04.
  The soft readout's gradient fixes those rows; the bit readout cannot.
- **At 10 bits a bit-valued borrow between places hurt** (0/8). With soft borrows every row had come out right there
  anyway.
- **Holding the digits out of the lesson for 200 steps** (`freeze_digits`, a local check on the perfect seed-0 table)
  did not help either. Exact stayed at 0.02 through 300 steps, and the table fell from 200 to 115 right rows.

### 4. Held-out moduli (holdout 4: a quarter of the moduli never trained)

| variant | 6 bits: held-out (trained) | 7 bits | 10 bits |
|---|---|---|---|
| as specified | 0.02–1.00 (0.11–1.00) | 0.03–1.00 (0.03–1.00) | 0.00–1.00 (0.00–1.00) |
| choice held uniform 400 steps | **1.00 (1.00)**, 4/4 | 0.02–0.99 (0.01–0.98), 0/4 | **1.00 (1.00)**, 4/4 |

Every holdout run that was solved (24 runs across all variants) was also exact on the held-out moduli (≥ 0.998).
Before this route, held-out moduli stayed at 0.00–0.08 for every from-answers variant of this datapath in microlab.

### 5. The same weights at other widths
Own borrow, with exact given the true digits in brackets:

| trained at | 6 bits | 7 bits | 10 bits | 14 bits | 20 bits |
|---|---|---|---|---|---|
| 6 bits, hold (8 runs) | 1.00 (1.00) | 0.78–0.81 (0.79–0.81) | 0.46–0.51 (0.52–0.55) | 0.13–0.16 (0.23–0.26) | 0.01–0.02 (0.05–0.06) |
| 10 bits, hold (8 runs) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.97–1.00 (1.00) | 1.00 (1.00) |
| 10 bits, hold + handover (8 runs) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 7 bits, the 5 runs that reached 1.000 (one as specified, two with the bit between places, two with every borrow a bit) | 0.998–1.00 | 1.00 | 1.00 | 1.00 | 0.998–1.00 |
| 7 bits, the 3 runs that stopped at 0.99 (as specified; hold, twice) | 0.85–0.94 | 0.99 | 0.74–0.93 | 0.24–0.77 | 0.01–0.44 |

- **Trained at 6 bits,** the subtractor is right on 161–162 of the 162 rows the true path visits and on at most one
  of the 38 it never visits. Its exact at other widths is whatever those wrong rows leave.
- **The tiny core's tables trained at 6 bits did worse:** 0.71–0.77 at 7 bits and 0.07–0.29 at 10.
- **At 7 and 10 bits the true path visits all 200 rows.**

### 6. Ablations (the route as specified otherwise; all moduli)

| variant | 6 bits | 7 bits | 10 bits |
|---|---|---|---|
| not staged (v < N² from the start, 4,500 steps) | 0/2 (0.10–0.16) | 0/2 (0.03) | 0/2 (0.00) |
| staged data, but always from the top w digits (the fixed w-step datapath) | 0/2 (0.07–0.27) | 0/2 (0.03–0.06) | 0/2 (0.00) |
| own borrow from the start (staged, no estimator) | 2/2 (1.00) | 0/2 (0.94, 0.99) | 0/2 (0.00–0.01) |
| window 2/1 (3 seeds) | 3/3 (1.00) | 0/3 (0.53–0.98) | 0/3 (0.00–0.01) |

- **Starting each stage from v // 10^k matters.** Always starting from the top w digits leaves the w − k leading
  steps, whose quotient digit is always 0, to the undecided estimator. Presumably their soft output blurs the
  register before the steps that count; that was not measured.
- **The own borrow alone starts at 6 bits.** That is unlike the tiny core at microlab's moduli (0/3 there), but it
  does not scale to 10 bits.

## Why, in plain words
- **Fixed slot meanings remove the trap TINYCORE found.** With a learned label per slot, every slot starts out the
  same, and the slots end up permuted and merged (0/9 here, as in every earlier microlab run). With slot q fed the
  digits of q·N, the slots differ from the start and in the right way. Whenever the subtractor learned, slot q meant
  digit q.
- **But the chooser must wait, and an MLP under Adam does not.**
  - TINYCORE's argument: under an undecided choice every trial gets an equal share of the lesson, and only the right
    one is consistent, so the shared subtractor drifts toward subtraction.
  - microlab's estimator is an MLP trained with AdamW at 3e-3. Its first gradients are noise, but Adam turns them
    into full-size steps, and within 100 steps it has decided on one slot for every example. The mixture-of-experts
    literature calls this routing collapse.
  - After that, the subtractor learns only through the chosen slot's trial, whose lesson (make 10R + v_t − s·N look
    like the answer) contradicts subtraction.
  - Holding the choice uniform for 400 steps gives the subtractor its unbiased lesson first.
- **The handover needs a subtractor that is right as a table, not just as a code.**
  - The estimator phase trains the borrow only as a message to the next place. An MLP can pass that message as an
    analog code, and the next place decodes it.
  - The own borrow reads the same signal as a probability and multiplies ten of them. A code of "0.5 means no" then
    looks like "maybe borrows" to the rule. Even a right table has a final borrow that phase A never used, so it is
    uncalibrated at the switch at every width.
  - At 10 bits the table came out right on one-hot inputs, so phase B only had to calibrate that final borrow, and
    the result was a rule. At 6 bits it came out mostly right. At 7 bits the code was analog, and phase B had to
    rebuild it while choosing with it. Why 7 bits differs is not established here.
  - Forcing the code to be a bit fixes the table but not the reading: the final borrow stays uncalibrated. Reading it
    as a bit fixes that, but then a single wrong row can leave no valid slot, and training dies.
- **A table is right only where its training data goes, even an MLP's.** The 6-bit runs leave all 38 unvisited rows
  wrong. At 10 bits the data visits every row, so the weights work at every width.

## Caveats
- **4 seeds per cell (2–3 for ablations), one learning rate (3e-3), one width of MLP.** The 7-bit verdict rests on 52
  runs across seven variants, but each variant has only 4–8.
- **Runs are not bit-reproducible across Modal containers.** The same configuration and seed gave different phase-A
  results in two otherwise identical runs, presumably from different CPU kernels. Paired comparisons (D7 vs S2, S5
  vs S4) are therefore approximate, except for S5, which starts from S4's saved weights.
- **The hold and the bit-valued borrow are design choices made after seeing failures.** The hold was tested on all
  24 configurations after diagnosing one run. The 7-bit repairs were each tried once, on the same seeds.
- **"Solved" is exact ≥ 0.99 on 2,000 test examples.** Several 7-bit runs ended at 0.95–0.99 and count as misses.
- **The 7-bit explanation is measured on saved weights** (D7 and S4 phase-A weights, on the Volume). Why the MLP's
  borrow comes out analog at 7 bits and crisp at 10 is not established. Possible causes are the narrower range of
  N's leading digits at 7 bits (N in 65–127), or the estimator's soft mixtures feeding later steps.
- **The exact multiples q·N are built in**, as in the tiny core: multiplying N by one known digit, like the
  long-division wiring itself.
- **Not tested:** mixing widths in training (TINYCORE's other suggestion), soft or annealed versions of the hold,
  and a calibrated (temperature-scaled) final borrow. A local check with temperature 0.25 on every borrow broke
  phase B at 6 bits (code not kept).

## Reproduce
Run from `cpu_first/`. Training runs on Modal: one 4-CPU container per run, app and Volume
`microlab-quotient-shared`. The weights after each phase are on the Volume as `<tag>/<run>_a.pt` and `_b.pt` for
runs tagged D7, S4 and S5.

```bash
~/github/sotaku/venv/bin/python -m microlab.ext_quotient_shared selftest
```

```bash
~/github/sotaku/venv/bin/python -m microlab.ext_quotient_shared window --bits 6,7,10
```

```bash
~/github/sotaku/venv/bin/python -m microlab.ext_quotient_shared start --seeds 0,1 --hold 0
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach microlab/ext_quotient_shared_modal.py --sizes 6,7,10 --seeds 4 --holdout 0,4 --tag S1
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach microlab/ext_quotient_shared_modal.py --sizes 6,7,10 --seeds 3 --tag S1L --extra '{"slots": "learned"}'
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach microlab/ext_quotient_shared_modal.py --sizes 6,7,10 --seeds 2 --tag S1A --extra '[{"staged": false, "final": 4500, "name": "_unstaged"}, {"first": "borrow", "name": "_borrowfirst"}, {"start": "top", "name": "_top"}]'
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach microlab/ext_quotient_shared_modal.py --sizes 6,7,10 --seeds 3 --tag S1W --extra '{"kr": 2, "kn": 1, "name": "_kr2kn1"}'
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach microlab/ext_quotient_shared_modal.py --sizes 6,7,10 --seeds 4 --holdout 0,4 --tag S2 --extra '{"hold": 400, "name": "_hold400"}'
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach microlab/ext_quotient_shared_modal.py --sizes 6,7,10 --seeds 4 --holdout 0,4 --tag S3 --extra '{"hold": 400, "handover": 500, "name": "_hold400_hand500"}'
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach microlab/ext_quotient_shared_modal.py --sizes 7 --seeds 4 --tag D7 --extra '[{"hold": 400, "name": "_hold400"}, {"hold": 400, "lr_b": 1e-3, "name": "_hold400_lrb1e-3"}, {"hold": 400, "lr_b": 3e-4, "name": "_hold400_lrb3e-4"}]'
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach microlab/ext_quotient_shared_modal.py --sizes 6,7,10 --seeds 4 --holdout 0,4 --tag S4 --extra '{"hold": 400, "hard_borrow": true, "name": "_hold400_hardb"}'
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach microlab/ext_quotient_shared_modal.py --sizes 6,7,10 --seeds 4 --holdout 0,4 --tag S5 --extra '{"hold": 400, "hard_borrow": true, "hard_b": "all", "from_a": "/out/S4/shared_{bits}_s{seed}_ho{holdout}_hold400_hardb_a.pt", "name": "_fromS4_bits"}'
```

```bash
~/github/sotaku/venv/bin/python -m modal run microlab/ext_quotient_shared_modal.py --fetch --tag S2
```

```bash
~/github/sotaku/venv/bin/python -m modal volume get microlab-quotient-shared S4/shared_7_s0_ho0_hold400_hardb_a.pt .
```

```bash
~/github/sotaku/venv/bin/python -m microlab.ext_quotient_shared probe shared_7_s0_ho0_hold400_hardb_a.pt --bits 7 --hard_borrow
```

```bash
~/github/sotaku/venv/bin/python -m microlab.ext_quotient_shared_report
```

Code:
- `ext_quotient_shared.py`: model `longdiv_shared` (slots `shared` or `learned`), the staged route (`run`) and the
  measures. Also `selftest`, `window`, `start` and `probe`.
- `ext_quotient_shared_modal.py`: one container per run.
- `ext_quotient_shared_report.py`: the tables.

Rows: `results_quotient_shared.jsonl`, with tags S1, S1L, S1A, S1W, S2, S3, D7, S4 and S5. Launch logs are in
`../runs/quotient_shared/`.
