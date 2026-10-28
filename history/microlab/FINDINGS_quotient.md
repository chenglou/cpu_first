# Quotient digits from the leading digits

**Question.** In [LADDER_MATCHED.md](LADDER_MATCHED.md) §3's long-division datapath, the subtraction is learned once
the quotient digits are given, but choosing each digit from the digit cell's own borrow learns slowly from answers.
Does a separate small learned estimator help? It picks each quotient digit from the leading digits of the running
remainder 10R + v_t and of N, the way SRT division and Knuth's Algorithm D estimate it, optionally checked by the
cell's borrow. Does that make the choice learnable from answers, and how does it scale from 6 to 7+ bits?

**Answer.**
- **Inside the datapath, trained from answers together with the digit cell: no.** The estimator does what longdiv's
  borrow choice does.
  - 3k steps: 0.02–0.05 for every variant, longdiv included, at 6 and 7 bits.
  - 12k steps, 6 bits: estimator 0.53–0.54, estimator checked by the borrow 0.45–0.48, longdiv 0.44–0.59. The
    estimator lifts off sooner (0.30–0.37 at 6k steps, when longdiv is at 0.04), then stops at the same level.
  - 12k steps, 7 bits: 0.03–0.06 for every variant, including the one that sees every digit (longdiv 0.05–0.06).
  - Nothing works one bit or one digit wider (≤ 0.02). The held-out-moduli runs never take off for any variant
    (0.04–0.08), so held-out moduli cannot tell rule from table here.
  - A wrap curriculum is the one thing that helps, and only a little. Trained with easy dividends mixed in (modmix,
    v < N·10^k), the estimator gets 0.18–0.26 at 7 bits after 12k steps (plain: 0.05), with seen and unseen inputs
    equal, so a weak rule. longdiv with the same curriculum got 0.11–0.18. At 6 bits the curriculum costs a little:
    0.35–0.45 (plain 0.53–0.54; longdiv with it 0.13–0.34). Starting the estimator at q = 0 makes things worse.
- **Given an exact subtraction, the same estimator learns the choice from answers quickly, and as a rule.** This is
  a diagnostic: the digit cell is replaced by its truth table, applied to digit distributions, and only the
  estimator learns.
  - 6 bits: exact in 3k steps (0.94 after 750). On moduli never trained it gets 0.78–0.79, and 0.99 with the borrow
    check. Every from-answers run gets ~0.02 there; the quotient-given cell gets 0.47–0.49.
  - 7 bits: it learns the best table its window allows. Seeing the top 3 digits of 10R + v_t and the top 2 of N, it
    gets 0.77–0.78. An estimator that commits to one digit per window cannot beat 0.78 on those digits, and every
    miss is one digit off. Seeing every digit, it gets 1.00, and 0.93–0.94 on unseen moduli. Trained at 7 bits it
    gets 0.49–0.71 one digit wider (10 bits).
  - 10 bits: it needs longer. From plain data it is at 0.00 after 3k steps (also when it starts at q = 0), but a
    12k run lifts off between 3k and 6k steps and reaches 0.70–0.72, almost all of it on inputs it never saw.
    Trained on easy dividends (modmix: v < N·10^k, so only the last k quotient digits can be nonzero) it gets 0.65
    in 3k steps, its window's best (0.64), on inputs it never saw.
- **Checking the estimate with the borrow backfires once choices are soft.** Blur a right estimate 10% toward
  uniform. The estimator alone stays exact at 6, 7 and 10 bits; checked by the borrow it gets 0.53, 0.00 and 0.00.
  A blurred remainder's borrow is unreliable, and it blurs the next choice. Even with the exact cell, the checked
  variant learns only at 6 bits.

**Verdict.** A learned leading-digit estimator does not make the remainder learnable from answers in this datapath.
It is a good way to choose the digit: small, quickly learned, and a rule that carries over to new moduli. But that
holds only once the subtraction is right. Together with a wrap curriculum it beats the borrow choice at 7 bits
(0.18–0.26 against 0.11–0.18): a step, not a fix. What stays hard is learning both halves at once. The subtraction
is learned when the digits are given (modq), and the digits are learned when the subtraction is given (this note).
From answers alone, neither has the other to learn from. The 6-bit runs that do lift off end in a lookup
arrangement, with longdiv's borrow rule as well as with the estimator: their quotient slots are permuted and merged,
one slot serving two digits and some digits none.

**Scaling.** From answers, every variant is at 0.03–0.06 at 7 bits, as longdiv is, or 0.18–0.26 with the wrap
curriculum. In isolation, going from 6 to 7 bits costs only what the window cannot see, and 10 bits takes longer to
lift off unless the easy dividends come first. A choice gets a clean lesson only when every later choice is already
right, and the number of real quotient digits grows with the width: two per example at 6 bits, three at 7 and four
at 10 (at 7 and 10 bits the first is only 0 or 1).

## Results
Exact match on 2,000 fresh examples, the range over 2 seeds; fresh data, batch 256, lr 3e-3 (longdiv's).
- **Held-out moduli:** separate runs that never train on a quarter of the moduli (holdout=4); in brackets, the same
  runs on their trained moduli.
- **One bit wider:** 6 → 7 bits and 7 → 8 bits. **One digit wider:** 6 → 7 and 7 → 10 bits, on inputs with a digit
  past the trained width.
- **Every choice right:** the share of test examples whose quotient digits are all chosen right, reading each slot
  as the digit it most often stands for. With the exact cell the slots are the digits themselves.

**Trained from answers** (longdiv's rows from ladder_matched_results.jsonl; everything else is new):

| variant | bits | steps | exact | held-out moduli (trained) | one bit wider | one digit wider | every choice right |
|---|---|---|---|---|---|---|---|
| longdiv, borrow choice (LADDER_MATCHED) | 6 | 3k | 0.04–0.05 | · | 0.00–0.01 | 0.01 | · |
| longdiv, borrow choice (LADDER_MATCHED) | 6 | 12k | 0.44–0.55 | 0.02–0.03 (0.04–0.06) | 0.01 | 0.01 | · |
| longdiv, borrow choice (LADDER_MATCHED) | 7 | 3k | 0.02–0.03 | · | 0.01 | 0.01 | · |
| longdiv, borrow choice (LADDER_MATCHED) | 7 | 12k | 0.05–0.06 | · | 0.01 | 0.00 | · |
| longdiv, borrow choice (LADDER_MATCHED), trained on modmix | 6 | 3k | 0.11–0.16 | · | · | 0.01 | · |
| longdiv, borrow choice (LADDER_MATCHED), trained on modmix | 6 | 12k | 0.13–0.34 | · | · | 0.00 | · |
| longdiv, borrow choice (LADDER_MATCHED), trained on modmix | 7 | 3k | 0.04–0.05 | · | · | 0.01 | · |
| longdiv, borrow choice (LADDER_MATCHED), trained on modmix | 7 | 12k | 0.11–0.18 | · | · | 0.01 | · |
| longdiv, borrow choice (rerun) | 6 | 3k | 0.04–0.05 | 0.02–0.03 (0.04–0.05) | 0.00–0.01 | 0.01 | 0.02 |
| longdiv, borrow choice (rerun) | 6 | 12k | 0.44–0.59 | 0.02–0.03 (0.04–0.06) | 0.01 | 0.00–0.01 | 0.14–0.49 |
| longdiv, borrow choice (rerun) | 7 | 3k | 0.02–0.03 | 0.02–0.03 (0.02–0.04) | 0.01 | 0.01 | 0.01 |
| longdiv, borrow choice (rerun) | 7 | 12k | 0.05–0.06 | 0.03 (0.06–0.07) | 0.01 | 0.00 | 0.01 |
| estimator | 6 | 3k | 0.04–0.05 | 0.02 (0.04–0.05) | 0.00–0.01 | 0.00–0.02 | 0.02 |
| estimator | 6 | 12k | 0.53–0.54 | 0.02–0.03 (0.05) | 0.01 | 0.01 | 0.23–0.27 |
| estimator | 7 | 3k | 0.03 | 0.02–0.03 (0.05–0.06) | 0.01 | 0.01 | 0.01 |
| estimator | 7 | 12k | 0.05 | 0.02 (0.06) | 0.01 | 0.00–0.01 | 0.01 |
| estimator, window 4/3 | 7 | 3k | 0.03 | 0.03–0.04 (0.04–0.05) | 0.01–0.02 | 0.00–0.01 | 0.01 |
| estimator, window 4/3 | 7 | 12k | 0.05 | 0.02–0.03 (0.07–0.09) | 0.01 | 0.00–0.01 | 0.01 |
| estimator, starts at q = 0 | 6 | 12k | 0.05–0.18 | · | 0.01 | 0.00–0.01 | 0.02–0.08 |
| estimator, starts at q = 0 | 7 | 12k | 0.05–0.06 | · | 0.01–0.02 | 0.00 | 0.01 |
| estimator, trained on modmix | 6 | 12k | 0.35–0.45 | · | 0.01 | 0.00–0.01 | 0.28–0.40 |
| estimator, trained on modmix | 7 | 12k | 0.18–0.26 | · | 0.02 | 0.00–0.01 | 0.16–0.24 |
| estimator + borrow check | 6 | 3k | 0.04–0.05 | 0.01–0.02 (0.04–0.05) | 0.01 | 0.01 | 0.02 |
| estimator + borrow check | 6 | 12k | 0.45–0.48 | 0.03–0.04 (0.05–0.09) | 0.01 | 0.01 | 0.13–0.14 |
| estimator + borrow check | 7 | 3k | 0.03 | 0.02–0.03 (0.04–0.05) | 0.01 | 0.01 | 0.01 |
| estimator + borrow check | 7 | 12k | 0.05 | 0.02–0.03 (0.06–0.08) | 0.01–0.02 | 0.00 | 0.01 |
| estimator + borrow check, window 4/3 | 7 | 3k | 0.03–0.04 | 0.02–0.03 (0.04–0.05) | 0.01–0.02 | 0.00 | 0.01 |
| estimator + borrow check, window 4/3 | 7 | 12k | 0.05–0.06 | 0.02–0.03 (0.06) | 0.01 | 0.01 | 0.01 |
| estimator + borrow check, starts at q = 0 | 6 | 12k | 0.05–0.17 | · | 0.00–0.01 | 0.00–0.01 | 0.02–0.07 |
| estimator + borrow check, starts at q = 0 | 7 | 12k | 0.03–0.04 | · | 0.01 | 0.00 | 0.01 |
| quotient digits given (modq) | 6 | 3k | 0.95–0.98 | · | 0.07–0.08 | 0.05–0.07 | · |
| quotient digits given (modq) | 6 | 12k | 0.99–1.00 | 0.47–0.49 (1.00) | 0.07–0.09 | 0.07–0.08 | · |
| quotient digits given (modq) | 7 | 3k | 0.38–0.39 | · | 0.15 | 0.03 | 1.00 |
| quotient digits given (modq) | 7 | 12k | 0.39–0.40 | 0.21–0.23 (0.42–0.43) | 0.15 | 0.04 | · |

**With the exact digit cell** (its truth table, applied to digit distributions; only the estimator learns; 3k steps):

| variant | bits | steps | exact | held-out moduli (trained) | one bit wider | one digit wider | every choice right |
|---|---|---|---|---|---|---|---|
| estimator | 6 | 3k | 1.00 | 0.78–0.79 (1.00) | 0.01–0.02 | 0.02 | 1.00 |
| estimator | 7 | 3k | 0.77–0.78 | 0.73 (0.79–0.80) | 0.04 | 0.49–0.53 | 0.78 |
| estimator | 10 | 3k | 0.00 | 0.00 (0.00) | 0.00 | 0.00 | 0.00 |
| estimator | 10 | 12k | 0.70–0.72 | · | 0.04 | 0.08 | 0.71–0.72 |
| estimator, window 4/3 | 7 | 3k | 1.00 | 0.93–0.94 (1.00) | 0.04 | 0.66–0.71 | 0.99–1.00 |
| estimator, starts at q = 0 | 7 | 3k | 0.77–0.91 | · | 0.04 | 0.30–0.51 | 0.78–0.91 |
| estimator, starts at q = 0 | 10 | 3k | 0.00 | · | 0.00 | 0.00 | 0.00 |
| estimator, trained on modmix | 10 | 3k | 0.65 | · | 0.04 | 0.07 | 0.66 |
| estimator + borrow check | 6 | 3k | 1.00 | 0.99 (1.00) | 0.12 | 0.13 | 1.00 |
| estimator + borrow check | 7 | 3k | 0.06–0.14 | 0.00 (0.09–0.10) | 0.00 | 0.00 | 0.02 |
| estimator + borrow check | 10 | 3k | 0.00 | 0.00 (0.00) | 0.00 | 0.00 | 0.00 |
| estimator + borrow check, starts at q = 0 | 7 | 3k | 0.04 | · | 0.00 | 0.00 | 0.01 |
| estimator + borrow check, starts at q = 0 | 10 | 3k | 0.00 | · | 0.00 | 0.00 | 0.00 |

**What a window of leading digits can know.** Exact long division on the training distribution, scoring the best
estimator that sees only the window (the most common digit for each window):

| bits | digits seen (remainder / N) | distinct windows | steps right | examples with every step right |
|---|---|---|---|---|
| 6 | 3 / 2 (all of them) | 7,680 | 1.00 | 1.00 |
| 7 | 3 / 2 | 691 | 0.96 | 0.78 |
| 7 | 4 / 3 (all of them) | 30,571 | 1.00 | 1.00 |
| 8 | 3 / 2 | 2,712 | 0.98 | 0.87 |
| 10 | 3 / 2 | 503 | 0.95 | 0.64 |
| 10 | 4 / 3 | 40,023 | 0.995 | 0.96 |

Every miss is off by exactly one, so a right borrow check could repair all of them.

## Why, in plain words
- **Each half learns when the other is given, not together.** From answers, the estimator and the cell start
  symmetric. Every slot computes the same blend, so the estimator has nothing to prefer, and the cell gets the same
  lesson for every slot. After 3k steps every variant still reads as "always 0": right on the division steps
  whose digit is (almost) always 0, at chance (0.10–0.22) on the others.
- **When a run lifts off, it builds a lookup arrangement, not the digit rule.** At 6 bits after 12k steps the slot
  maps are, for example, [3, 4, 6, 7, 9, 1, 5, 0, 5, 2] for longdiv's borrow rule and [0, 9, 4, 4, 5, 3, 0, 2, 1, 0]
  for the estimator (slot k stands mostly for that digit). The cell then makes whichever slot is chosen produce the
  right difference, entry by entry. Hence the slow climb and the zero transfer.
- **The borrow is a fragile check for soft choices.** The borrow of a blurred remainder is a product over its whole
  carry chain. One soft choice blurs the next borrow, and the error compounds over the steps. longdiv's own choice
  reads all ten borrows, so this is probably part of why it is slow too.
- **The chain of choices is long.** A choice is only taught when the later choices are right. An undecided start
  blurs every step, so the lesson fades with every extra quotient digit. That is why the estimator alone takes longer
  at 10 bits from plain data and is quick with easy dividends: the last digit is taught first, and the estimator is
  shared across steps, so it then chooses the earlier digits right too.
- **The window sits at fixed digit positions.** It transfers between widths where N's leading digit has the same
  range: 7 → 10 bits (top digit 0 or 1 in both), not 6 → 7 or 7 → 8.

## Caveats
- **The quotient-given ceiling at 7 bits is the cell, not the choice.** With the digits given, the cell stops at
  0.38–0.40 at 7 bits. Its errors sit in the tens digit of the remainder (units 1.00, tens 0.40–0.41, hundreds
  0.96), across every value of N's digits. So even a perfect choice would cap from-answers training near 0.4 at
  7 bits with this cell.
- **The exact cell is a diagnostic.** It removes the subtraction from what is learned, as modq removes the choice.
  A frozen MLP cell fitted to the truth table does not work as a stand-in: the soft mixtures training produces are
  inputs it never saw, and the estimator stays at 0.03–0.04.
- **2 seeds per cell, 3k–12k steps.** The 6-bit lift-off is seed- and data-stream-dependent: every held-out-moduli run
  (a different sample stream) stayed at 0.04–0.08 for all variants, as longdiv's did in LADDER_MATCHED.md.
- **Soft execution can beat the window's bound for a hard choice:** 0.91 at 7 bits (one seed that started at
  q = 0; bound 0.78) and 0.70–0.72 at 10 bits (bound 0.64). When unsure, the estimator keeps both candidates,
  and the next step's window sees which one wrapped below zero. A snapped (hard) datapath could not do that.
- **"Every choice right" undercounts permuted slots.** Reading each slot as one digit misses slots that stand for two;
  from-answers runs are exact on more examples than this measure credits.

## Reproduce
Run from `cpu_first/`. Training runs on Modal (4-CPU containers; a 7-bit 12k cell takes about 40 minutes).

```bash
python -m microlab.ext_quotient selftest
```

```bash
python -m microlab.ext_quotient window --bits 6,7,8,10
```

```bash
python -m microlab.ext_quotient blurtest
```

```bash
modal run --detach microlab/ext_quotient_modal.py --models longdiv_tr,longdiv_est,longdiv_estc --sizes 6,7 --steps 3000,12000 --holdout 0,4 --tag Q12k
```

```bash
modal run --detach microlab/ext_quotient_modal.py --models longdiv_est,longdiv_estc --sizes 7 --steps 3000,12000 --holdout 0,4 --tag Q12k --extra '{"kr": 4, "kn": 3, "name": "_kr4kn3"}'
```

```bash
modal run --detach microlab/ext_quotient_modal.py --models longdiv_est,longdiv_estc --sizes 6,7,10 --steps 3000 --holdout 0,4 --tag Qsoft --extra '{"soft_cell": true, "name": "_soft"}'
```

```bash
modal run --detach microlab/ext_quotient_modal.py --models longdiv_est --sizes 10 --steps 3000 --tag Qsoft --extra '{"soft_cell": true, "train_on": "modmix", "name": "_soft_mix"}'
```

```bash
modal run --detach microlab/ext_quotient_modal.py --models longdiv_est --sizes 10 --steps 12000 --tag Qsoft --extra '{"soft_cell": true, "name": "_soft"}'
```

```bash
modal run --detach microlab/ext_quotient_modal.py --models longdiv_est --sizes 6,7 --steps 12000 --tag Q12k --extra '{"train_on": "modmix", "name": "_mix"}'
```

```bash
modal run --detach microlab/ext_quotient_modal.py --models longdiv_est,longdiv_estc --sizes 6,7 --steps 12000 --tag Q12k --extra '{"est_bias0": 4.0, "name": "_b4"}'
```

```bash
python -m microlab.ext_quotient_report --findings
```

Code: `ext_quotient.py` (models longdiv_tr, longdiv_est, longdiv_estc; options kr/kn for the window, soft_cell,
est_bias0, train_on), `ext_quotient_modal.py`, `ext_quotient_report.py`. Rows: `results_quotient.jsonl`; logs in
`../runs/quotient/`.
