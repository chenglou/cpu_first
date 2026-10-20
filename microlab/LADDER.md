# Ladder: where generic networks stop learning each piece of x^(2^T) mod N

**Answer: the remainder.** Generic networks learn addition as a rule. They learn every other piece as a lookup table
of answers: multiplication, squaring, the remainder, and their compositions. Each table piece breaks where the table
outgrows the training run, and moves with budget only as fast as memory grows. The remainder is the worst of them:

- It breaks earliest and moves least.
  - No model passes even 6-bit moduli at 3k steps.
  - At 12k steps only the MLP does, while every model passes 2-digit multiplication.
- Nothing about it carries over to a modulus the network was not trained on: held-out moduli are at chance.
- On a new modulus, and on new inputs at the competition's 1,000-example budget, a network is right only where no
  remainder has to be taken.
  - Trained on 1,000 sqmod@7 examples, it scores 78% on unseen inputs with x² < N.
  - With x² ≥ N it scores 1.5–2%.

## What was run
- Tasks, models and data are microlab's (`tasks.py`, `models.py`), unchanged.
  - Tasks: `add`, `mul`, `sq` at 2–8 digits; `mod`, `sqmod`, `powmod2`, `powmod4` at 6–16-bit moduli.
  - Models: `mlp`, `xf`, `loop` at their defaults.
  - Data: fresh examples every step, or a fixed pool of 1,000 (the competition's T=1 budget).
- Each cell trains 2 seeds for 3,000 steps of batch 256, which is 768k examples when fresh.
  - A size **passes** if the better seed reaches ≥ 99% exact on microlab's 2,000 test examples.
  - Each ladder climbs from the smallest size and stops after two failing sizes in a row.
- Every ladder that broke was re-run from its breaking point at 12,000 steps (4×).
- Then come probes on the remainder (Section 4), held-out moduli (2b) and a split by quotient (2c).
  - That is 508 cells in all.
  - On Modal it took about 5.4 L4-hours and 2.2 CPU-container-hours.
- Every cell also records:
  - the share of test inputs that also occurred in training;
  - exact match on the test inputs that did not;
  - per-digit accuracy against a majority-class baseline (always guessing that digit's most common value).

  Exact match on unseen inputs is the line between a rule and a table.
- Code: `ext_ladder.py` (cells, ladders, held-out moduli, quotient split), `ext_ladder_modal.py` (one Modal container
  per ladder), `ext_ladder_report.py` (these tables). Raw cells: `ladder_results.jsonl` and `ladder_quotient.jsonl`.
- The relaxed-rules write-up (`../RELAXED.md`, in progress) covers single-step sqmod with its own models: a looped
  transformer on fresh data at 6–10 bits, and periodic units. This ladder does not repeat that. The pre-remainder
  session (`ext_preremainder.py`) separates squaring from the remainder on sqmod's own values.

## 1. Breaking points
The largest size where the better of 2 seeds is ≥ 99% exact. The first number is at 3k steps. After the arrow is the
result at 12k steps, re-run from the 3k breaking point. `<2` and `<6` mean not even the smallest size passed; `≥8`
means every size passed.

| task (size) | mlp fresh | mlp pool 1k | xf fresh | xf pool 1k | loop fresh | loop pool 1k |
|---|---|---|---|---|---|---|
| add (digits) | 5 → ≥8 | <2 → <2 | ≥8 | <2 → <2 | ≥8 | <2 → <2 |
| mul (digits) | <2 → 2 | <2 → <2 | <2 → 2 | <2 → <2 | <2 → 2 | <2 → <2 |
| sq (digits) | 3 → 4 | 2 → 2 | 3 → 4 | 2 → 2 | 3 → 4 | 2 → 2 |
| mod (bits) | <6 → 6 | <6 → <6 | <6 → <6 | <6 → <6 | <6 → <6 | <6 → <6 |
| sqmod (bits) | 7 → 8 | <6 → <6 | 7 → 8 | <6 → <6 | 7 → 8 | <6 → <6 |
| powmod2 (bits) | 7 → 8 | <6 → <6 | 7 → 8 | <6 → <6 | 7 → 8 | <6 → <6 |
| powmod4 (bits) | 7 → 8 | <6 → <6 | 7 → 8 | <6 → <6 | 7 → 8 | <6 → <6 |

How to read it:
- **Every boundary except addition's is a memory boundary.**
  - At 3k steps with fresh data (768k examples), every model passes exactly the sizes whose whole set of possible
    inputs has ≤ 3,072 members, and fails from about 10k.
    - sqmod@7 (3,072 (N, x) pairs) passes.
    - These fail: sqmod@8 (12,288), sq@4 (10,000 values), mul@2 (10,000 pairs), mod@6 (38,224 (N, v) pairs).
  - At 12k steps every model moves one step: sqmod, powmod2 and powmod4 from 7 to 8 bits, sq from 3 to 4 digits,
    and mul from <2 to 2.
  - The MLP also completes mod@6. sqmod@9 (49,152 pairs) fails for every model; the best is 0.62.
  - One more bit of modulus is 4× the sqmod table, so 4× the steps bought exactly one bit.
  - The remainder moves least: only the MLP passes 6 bits, at 12k.
- **Composition is free.** x², x⁴ and x¹⁶ mod N (`sqmod`, `powmod2`, `powmod4`) break at the same size in every
  column, at both budgets.
- **With 1,000 examples only tiny tables pass, and 4× the steps changes nothing.**
  - sq@2 passes; it has 100 inputs.
  - The pool is memorized within the first quarter of training: train exact 1.00, flat test curves.
  - Test exact then equals the share of test inputs that are in the pool. For sqmod@7, 30% of test inputs are seen,
    and exact is 33–38%.
- **The two kinds of model differ in strength.**
  - The transformers memorize more slowly than the MLP. At sqmod@8 and 3k steps: xf 0.32, loop 0.19, mlp 0.88.
  - They are better at rules. They pass addition at every size, and the next section shows they learn most of it
    from 1,000 examples.

## 2. Rule or table? Inputs never seen in training
**2a. Unseen inputs (xf, 1,000 training examples, 3k steps).** This uses the smallest size whose test inputs are
mostly unseen. For sq that is the largest size tested, where 42% are unseen. Digits run lo→hi and are the mean of
2 seeds.

| task@size | test inputs unseen | exact on unseen | unseen digits lo→hi | majority baseline |
|---|---|---|---|---|
| add@3 | 0.87 | **0.878** | 1.00 0.94 0.94 0.99 | 0.11 0.14 0.44 0.94 |
| mul@3 | 0.87 | 0.155 | **1.00** 0.34 0.35 0.59 0.83 **0.97** | 0.26 0.19 0.28 0.49 0.76 0.93 |
| sq@4 | 0.42 | 0.032 | **1.00 0.99** 0.47 0.17 0.29 0.51 0.77 **0.96** | 0.21 0.26 0.36 0.37 0.56 0.60 0.80 0.83 |
| mod@6 | 0.97 | 0.037 | 0.18 0.24 | 0.11 0.22 |
| sqmod@7 | 0.72 | 0.101 | 0.25 0.20 0.94 | 0.14 0.18 0.95 |
| powmod2@7 | 0.72 | 0.043 | 0.22 0.15 0.92 | 0.20 0.21 0.95 |
| powmod4@7 | 0.72 | 0.066 | 0.24 0.18 0.94 | 0.21 0.23 0.96 |

mlp and loop give the same picture. Exact on unseen inputs, mlp / loop:

| task | mlp | loop |
|---|---|---|
| add | 0.37 | 0.90 |
| mul | 0.08 | 0.18 |
| mod | 0.04 | 0.04 |
| sqmod | 0.09 | 0.10 |

- **add:** every digit is a rule.
- **mul and sq:** a few digits are rules.
  - The units digit (and sq's tens digit) is right, at 1.00.
  - The top digit, set mostly by the sizes, is right, at about 0.95.
  - The middle digits sit near the baseline or below it.
- **mod, sqmod, powmod:** every digit is at or near the baseline, except sqmod's units digit (0.25 vs 0.14). 2c
  shows where that exception comes from.
- **With much more data, xf starts on multiplication's middle.** At 12k fresh steps it gets 60% of the unseen 3-digit
  products (mlp 7%, loop 16%). That is about 120 unseen test inputs, because the skewed sampler repeats short
  numbers. Nothing like that happens for the remainder (2b).

**2b. Moduli never seen in training (12k steps).** A quarter of the moduli, N with (N // 2) % 4 = 0, never enter
training. They are scored separately; 2 seeds averaged.

| task@size | model | data | trained moduli: exact | held-out moduli: exact |
|---|---|---|---|---|
| mod@6 | mlp | fresh | 1.00 | 0.03 |
| mod@6 | mlp | pool 100k | 0.99 (unseen inputs: 0.78) | 0.03 |
| mod@6 | xf | fresh | 0.83 | 0.03 |
| mod@6 | xf | pool 100k | 0.96 (unseen inputs: 0.67) | 0.03 |
| mod@7 | mlp | fresh | 0.85 | 0.04 |
| mod@7 | mlp | pool 100k | 0.68 (unseen inputs: 0.49) | 0.04 |
| mod@7 | xf | fresh | 0.25 | 0.03 |
| mod@7 | xf | pool 100k | 0.27 (unseen inputs: 0.25) | 0.02 |
| sqmod@7 | mlp / xf | fresh | 1.00 / 1.00 | 0.15 / 0.13 |
| sqmod@8 | mlp / xf | fresh | 1.00 / 1.00 | 0.09 / 0.09 |

- **Chance is about 0.02 for mod** (the product of the per-digit baselines).
- **Every remainder model is at chance on held-out moduli.** That includes the ones that fill most gaps inside the
  moduli they were trained on, from 100k examples.
- **sqmod's held-out score is the no-remainder share.** The share of held-out inputs with x² < N is 0.12 at 7 bits
  and 0.07 at 8.

**2c. Unseen inputs by quotient q = ⌊x²/N⌋ (sqmod) or ⌊v/N⌋ (mod).** q = 0 means the remainder changes nothing.
These come from 20,000 fresh test examples, scored on a seed-0 model.

| run | q = 0 | q = 1 | q ≥ 2 |
|---|---|---|---|
| sqmod@7, mlp, 1,000 examples | **0.78** | 0.015 | 0.022 |
| sqmod@7, xf, 1,000 examples | **0.88** | 0.045 | 0.017 |
| sqmod@8, mlp, fresh 12k, held-out moduli | **0.94** | 0.18 | 0.025 |
| mod@6, xf, 1,000 examples | 0.20 | 0.04 | 0.03 |
| mod@7, mlp, 100k examples (unseen v, trained moduli) | 0.88 | 0.60 | 0.36 |

**Summary.** Away from the inputs it stored, a network gets x² mod N right only when x² < N: there, it is just
squaring. Inside a modulus it has seen densely, it can interpolate: r rises by 1 as v rises by 1. Across moduli
nothing carries over.

This is the wall RELAXED.md meets from the other side. Its periodic nets learn as soon as the wraps are taken away,
either by a curriculum that starts with x² < N or by giving x² as a label.

## 3. Which digits fail first
At each fresh-data breaking point except add's, every test input was also in training. So these digits show what
the network has not finished memorizing. Runs are xf at 3k steps, mean of 2 seeds; add comes from the mlp, the only model that broke
on it.

| task@size | model | exact | digits lo→hi | majority baseline |
|---|---|---|---|---|
| add@6 | mlp | 0.988 | 1.00 1.00 1.00 1.00 **0.99 0.99** 1.00 | 0.11 0.12 0.18 0.30 0.49 0.71 0.99 |
| mul@2 | xf | 0.974 | 1.00 **0.96** 0.99 1.00 | 0.25 0.22 0.46 0.84 |
| sq@4 | xf | 0.807 | 1.00 1.00 1.00 **0.89 0.83** 0.97 1.00 1.00 | 0.21 0.26 0.36 0.37 0.56 0.60 0.80 0.83 |
| mod@6 | xf | 0.179 | **0.21** 0.78 | 0.11 0.22 |
| sqmod@8 | xf | 0.320 | **0.58 0.50** 0.76 | 0.13 0.16 0.56 |
| powmod2@8 | xf | 0.251 | **0.47 0.49** 0.73 | 0.18 0.16 0.56 |
| powmod4@8 | xf | 0.238 | **0.52 0.50** 0.70 | 0.18 0.19 0.57 |

- **add:** the high digits fail first, where the longest carry chains end.
- **mul and sq:** the middle digits fail first. The lowest digits and the top digit are already right.
- **mod, sqmod, powmod:** the low and middle digits fail first. The top digit is easiest, because N bounds it.
- mlp and loop fail in the same order.

## 4. Probes on the hardest boundary: the remainder (`mod`)
Each lever is tried at the remainder's breaking point. Entries are best-seed exact; after a slash is exact on unseen
test inputs, where at least a quarter are unseen.

| lever | run | mod@6 (38k inputs) | mod@7 (306k) | mod@8 (2.4M) |
|---|---|---|---|---|
| base, 3k steps | mlp / xf, fresh | 0.84 / 0.18 | 0.10 / 0.04 | |
| more steps | mlp / xf, fresh, 12k | **1.00** / 0.91 | 0.75 / 0.10 | 0.07 / — |
| more steps | mlp / xf, fresh, 48k | **1.00 / 1.00** | 0.96 / 0.96 | 0.25 / 0.07 |
| bigger model (14× / 8× params) | mlp hidden 2048 / xf d=256 depth 6, fresh, 12k | **1.00 / 1.00** | 0.68 / 0.86 | 0.06 / 0.02 |
| more data | mlp, pool 1k / 10k / 100k, 12k | 0.07 / 0.32 / 0.98 | 0.02 / 0.07 / 0.53 (unseen 0.38) | |
| more data | xf, pool 1k / 10k / 100k, 12k | 0.08 / 0.36 / 0.97 | 0.02 / 0.07 / 0.26 (unseen 0.21) | |
| new moduli | mlp / xf, fresh and pool 100k, 12k, a quarter of the moduli held out | held-out moduli: 0.03 | 0.02–0.04 | |

- **Every lever moves the boundary, and only as far as it adds memory.**
  - **Steps.** 16× the steps took mod@7 from 0.10 to 0.96 (mlp) and from 0.04 to 0.96 (xf). Its table is 8× mod@6's.
    mod@8, 8× bigger again, is at 0.25 (mlp) and 0.07 (xf).
  - **Model size.** A bigger transformer memorizes faster: mod@7 goes from 0.10 to 0.86 at 12k. A bigger MLP does
    not, at the same learning rate.
  - **Data.** Exact follows the share of test inputs in the pool. It goes further only when the pool covers most of
    a table: 100k examples are 2.6× mod@6's 38k inputs. Then the network also fills in the rest (0.75 on the 7–9% of
    test inputs not in the pool), by interpolating inside each modulus.
- **No lever reaches a modulus outside training.** Held-out moduli stay at chance, and so does one more bit (ood@8
  after training at 7 bits: 0.01–0.02).
- **A bit costs more for the remainder than for x² mod N.**
  - 4× the steps moved sqmod one bit.
  - After mod@6 passed at 12k, another 4× (48k) still left mod@7 below 99%.

  That fits the tables: mod's grows 8× per bit, sqmod's 4×.

## 5. What makes the problem hard, in plain words
**Addition is local.** Each digit of a sum depends on two input digits and a carry from the right. The same small
rule applies at every position, so a few examples already show it, and the networks find it. The transformers get
88–90% of unseen 3-digit sums right from 1,000 examples.

**Multiplication is local only at its edges.**
- The units digit of a product depends only on the two units digits.
- The top digit mostly depends on the sizes of the numbers.

Networks learn both edges as rules. The middle digits add up many digit products at once, and those are only
memorized.

**The remainder is not local anywhere.** Every digit of v mod N depends on every input digit at once, through a
quotient that never appears in the data. Change a high digit of v, or any digit of N, and the answer jumps to an
unrelated residue. There is no small piece of the rule to pick up first, so gradient descent has nothing to climb.
All a network can do is store answers. Stored answers:
- are exact on inputs the network has seen;
- fill in gaps inside a modulus it has seen densely, because r rises by 1 as v rises by 1;
- are useless on a new modulus.

**So the size limit is a memory limit.** The number of answers to store grows with every bit of modulus: 4× per bit
for x² mod N and 8× for a general remainder. These networks complete tables like this:
- up to about 3k answers from 768k examples, every model;
- about 10k (transformers) to 40k (MLP) answers from 3M examples.

Each 4× of compute buys about one more bit of x² mod N.

**Composition adds nothing on top.**
- A table of x¹⁶ mod N is no bigger than a table of x² mod N.
- So powmod4 breaks exactly where sqmod does, in every column.

That is what a lookup table predicts. A network building on a partly learned squaring step would do worse at T = 4
than at T = 1.

**At the competition's size the table is out of reach.** The setting is 24-bit moduli, moduli at test time the
network has never seen, and 1,000 rows at T=1. Over all 24-bit moduli the x² mod N table has about 5·10¹³ entries,
and stored answers do not transfer between moduli. So a generic network scores about 0. That matches the leaderboard,
and REPORT.md's "labels can check the arithmetic but can't teach it".

## 6. Caveats
- **Plain baselines.** The models use learned absolute positions, predict all digits at once, and have no scratchpad.
  Transformers with better position schemes, reversed outputs or chain-of-thought are known to learn longer
  multiplication. So "multiplication is only memorized" describes these baselines and budgets. The remainder result
  is that these models store answers per modulus. A model allowed to write out the long division is a different
  experiment.
- **Short budgets.** The budgets are at most 48k steps. Table boundaries move with budget by design, so both 3k and
  12k are reported, and probes reach 48k.
- **Skewed numbers.** microlab draws numbers with a uniformly random digit length, so short numbers are common. At
  k digits many test operands are much shorter, and many fresh-data test inputs were seen in training even at sizes
  whose full input space is huge (seen ≈ 0.85 at mul@3).
- **`mod` draws v below N².** That is a far larger set than sqmod's squares, so mod's table is about N× bigger at
  the same modulus size. The remainder restricted to squares is the pre-remainder session's `modsq`.
- **Few seeds, finite test set.** There are 2 seeds per cell.
  - They agree within 0.05 in 97% of cells (median gap 0.004).
  - The exceptions are two transformer cells midway through memorizing mod@6 (0.24 and 0.31 apart).
  - A pass is ≥ 99% on 2,000 examples (±0.5%).

## 7. Reproduce
Run from `cpu_first/`. Locally, cap threads:

```bash
python -m microlab.ext_ladder cell sqmod --size 8 --model xf --steps 3000 --threads 2
```

```bash
python -m microlab.ext_ladder climb mod --model mlp --data pool1000 --seeds 2 --workers 2 --threads 2
```

The sweep on Modal: 42 ladders take about 15 minutes, limited by the account's 10-GPU cap.

```bash
modal run --detach microlab/ext_ladder_modal.py --tasks add,mul,sq,mod,sqmod,powmod2,powmod4 --models mlp,xf,loop --data fresh,pool1000 --steps 3000 --tag S3k
```

The 4× re-test from each breaking point (start sizes as JSON, e.g. `{"sqmod,xf,fresh": 8}`):

```bash
modal run --detach microlab/ext_ladder_modal.py --tasks sqmod --models xf --data fresh --steps 12000 --tag S12k --start '{"sqmod,xf,fresh": 8}'
```

Held-out moduli: a quarter of the moduli never enter training, and they are scored separately.

```bash
modal run --detach microlab/ext_ladder_modal.py --tasks mod --models mlp,xf --data fresh,pool100000 --steps 12000 --tag P_holdout --extra '{"sizes": [6, 7], "holdout": 4}'
```

Every table above:

```bash
python -m microlab.ext_ladder_report --tags S3k,S12k
```
