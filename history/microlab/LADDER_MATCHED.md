# Ladder, matched: does an architecture shaped like the arithmetic turn the tables into rules?

**Answer: for multiplication, yes. The mismatch was real. For the remainder it helps only part of the way: the
quotient-digit choice stays slow.**

- **Multiplication becomes a rule once its columns are built in.** The column-sum model has a learned digit-pair
  table, summed along the schoolbook columns, then one shared carry cell.
  - It becomes exact past where the plain models stop: 3 digits at 12k steps, 4 at 48k. The best plain model
    reaches 2 digits, by memorizing.
  - At equal operand lengths it is as right on products it never saw as on ones it did: 0.98 vs 0.98–0.99 at
    5 digits.
  - It multiplies numbers wider than anything it trained on: 0.80 one digit wider after 4 digits (12k steps), 0.86
    after 6 digits (48k).
  - It is not exact beyond 4 digits yet: 0.98 at 6 digits after 48k steps.
- **Position coupling alone is not enough.** Abacus-style transformers, looped or not, still store multiplication
  as a table: 0.01–0.06 at equal lengths, 0.04–0.05 one digit wider.
- **Addition was never the problem.** Every model learns it.
  - The column scan does it at any width: 1.00 at 9–12 digits after training on 8, and also from only 1,000
    examples.
  - Abacus transformers carry it a digit or two past training: 0.95 one digit wider, 0.99 when looped.
- **Squaring follows multiplication, but only once its table is too big to store.** At ≤ 10⁴ possible inputs even
  the matched model stores answers: exact, but 0.03–0.09 one digit wider. At 6 digits it squares by rule.
- **The remainder: long division gives the right wiring, and the hard part is choosing the quotient digit.**
  - With its one digit cell fitted to that cell's 10,000-entry truth table, the long-division datapath is exact at
    6, 7, 10, 14 and 20 bits, for every modulus.
  - Trained from answers alone, it learns slowly and unevenly.
    - At 6 bits: 0.04 after 3k steps; 0.44–0.55 after 12k (0.04 in another data stream); 0.45 and 0.90 after 48k.
    - At 7 bits: 0.05–0.06 after 12k, and 0.06 after 24k.
    - Nothing carries one bit wider.
  - Handed the quotient digits, the same cell learns the subtraction.
    - It is exact at 6 bits by 12k steps, and gets 0.47–0.49 on moduli it never saw, against 1.00 on trained ones.
    - At 7 bits it stops at 0.39–0.40.
  - So multiplying back and subtracting is local, and learnable once the choice is given. Choosing each quotient
    digit, i.e. the wrap count, is what makes it slow, and it gets slower fast as N grows.
    [RELAXED.md](../RELAXED.md)'s periodic units also need a curriculum over wraps for exactly that.

## What was run
- **Same setup as [LADDER.md](LADDER.md).**
  - Tasks and sizes: add, mul and sq at 2–8 digits; mod and sqmod at 6–16-bit moduli.
  - Data: fresh, or a pool of 1,000 examples.
  - Budget: 3,000 steps of batch 256, 2 seeds; a size passes if the better seed is ≥ 99% exact.
  - Each ladder climbs and stops after two failing sizes in a row. Every broken ladder is re-run from its breaking
    point at 12,000 steps.
  - Longer probes: 48k steps for colsum mul/sq and longdiv mod@6, 24k for longdiv mod@7.
- **Models, in `ext_matched.py`.** Each reads its widths from the input, so it also runs on wider numbers than it
  trained on.
  - **colsum, candidate (a).** A learned table P[a_i, b_j] holds one vector per digit pair (a bilinear form on one-hot
    digits). Its entries are summed along anti-diagonals into the schoolbook columns c_k = Σ_{i+j=k} P[a_i, b_j]. One
    shared GRU carry cell then scans the columns from the lowest digit up, writing one answer digit per column.
    - Addition uses the aligned pairs (i = j); squaring uses x·x.
    - Nothing says what a product or a carry is. The structure only makes every answer digit depend on its own column
      plus a carry, the way addition already is.
    - Learning rate 2e-3 (better than 5e-4 in calibration).
  - **abacus and abacus_loop, candidate (b).** microlab's transformer, except that each digit's position embedding is
    its significance, shared by every operand and the answer (position coupling). Each training example gets a random
    offset of 0–32 positions (Abacus embeddings), so wider numbers reuse trained positions. abacus has 3 layers;
    abacus_loop is one shared block applied 6 times.
  - **longdiv, candidate (c).** A scan over the dividend's digits from the top, keeping a running remainder R as w+1
    digit distributions.
    - Each step forms 10R + v_t and tries all ten quotient digits q at once, with one shared learned digit cell
      (r_j, n_j, q, carry) → (digit, carry) scanned from the lowest digit up.
    - It keeps the largest q whose subtraction does not borrow: softly, as P(q is the largest non-borrowing digit),
      from the cells' final-carry probabilities.
    - Only the digit cell is learned. Variants:
      - longdiv_st: straight-through choice and digits snapped every step;
      - trained on modmix: easy dividends mixed in, v < N·10^k with k uniform, a wrap curriculum as data;
      - modq: the quotient's digits given as an input, used instead of the choice.
  - **colsum_longdiv.** x² mod N end to end: colsum's x² columns, as digit distributions, are longdiv's dividend.
  - **Periodic units tied to N.** These are [RELAXED.md](../RELAXED.md) §3 and are not repeated here.
- **Rule-vs-table measures, added to LADDER.md's.**
  - In fresh runs, LADDER.md's "unseen" number mixes in operand length: the unseen test inputs are mostly the longest
    ones.
  - **Seen vs unseen at equal lengths.** Within each bucket of operand lengths, compare exact on test inputs that
    occurred in training with exact on those that did not, over 10,000 test examples.
    - Equal means a rule; unseen far below seen means a table.
    - Both near 0 means neither: those lengths are simply not learned yet.
  - **One step wider.** Exact one size up, on the test inputs that have a digit past the trained width. The
    sampler's many short operands are left out.
  - **Held-out moduli.** As in LADDER.md 2b: a quarter of the moduli never enter training.
- **Compute.** Everything ran on Modal. The account's GPU quota was taken by another session, so the transformers
  ran on 16-CPU containers and the scan models on 8-CPU ones.
- **Code and data.** `ext_matched.py` has the models, tasks and measures; `ext_matched_modal.py` runs them;
  `ext_matched_report.py` makes the tables. Raw cells are in `ladder_matched_results.jsonl`, with tags
  - M3k and M12k: the ladder;
  - Mrem: remainder variants;
  - Mlong: long probes;
  - Mho: held-out moduli;
  - Mcal: calibration.

## 1. Breaking points
The largest size where the better of 2 seeds is ≥ 99% exact. The first number is at 3k steps. After the arrow is the
result at 12k steps, re-run from the 3k breaking point. `<2` and `<6` mean not even the smallest size passed; `≥8`
means every size passed; `·` means not applicable. The longdiv* column holds longdiv for mod and colsum_longdiv for
sqmod. The last column is the best of LADDER.md's mlp, xf and loop.

| task (size) | colsum fresh | colsum pool 1k | longdiv* fresh | longdiv* pool 1k | abacus fresh | abacus pool 1k | abacus_loop fresh | abacus_loop pool 1k | best plain model, fresh (LADDER.md) |
|---|---|---|---|---|---|---|---|---|---|
| add (digits) | ≥8 | ≥8 | · | · | 6 | <2 | ≥8 | <2 | ≥8 → ≥8 |
| mul (digits) | 2 → 3 | <2 → <2 | · | · | <2 → 2 | <2 | <2 → 2 | <2 | <2 → 2 |
| sq (digits) | 3 → 4 | 2 → 2 | · | · | 3 | 2 | 2 | 2 | 3 → 4 |
| mod (bits) | · | · | <6 → <6 | <6 | <6 | <6 | <6 | <6 | <6 → 6 |
| sqmod (bits) | · | · | <6 → <6 | <6 | 6 | <6 | 6 | <6 | 7 → 8 |

Read with the longer probes (48k steps, fresh data):

| model | task@size | exact | seen / unseen, equal lengths | one digit wider |
|---|---|---|---|---|
| colsum | mul@3 | **1.000** | 1.00 / 1.00 | 0.53 |
| colsum | mul@4 | **0.996** | 1.00 / 0.99 | 0.92 |
| colsum | mul@6 | 0.976 | 1.00 / 1.00 | 0.86 |
| colsum | mul@8 | 0.937 | 1.00 / 1.00 | 0.73 |
| colsum | sq@4 | **1.000** | (all inputs seen) | 0.07 |
| colsum | sq@6 | **0.994** | 0.96 / 0.98 | 0.89 |

- **The breaking points move less than the rule-vs-table line does.**
  - colsum gains one step at 12k, as LADDER.md's MLP did.
  - Its errors are in rare long carry chains, not in unseen inputs: at 8 digits, unseen inputs are as right as seen
    ones (1.00 / 1.00 at matched lengths) while the overall exact is 0.94.
- **abacus is slightly worse than plain xf in distribution.** It reaches add@6 against xf's ≥ 8: the random offsets
  cost some accuracy in exchange for wider numbers.
- **sqmod:** abacus and abacus_loop memorize sqmod@6 (768 inputs). Every model fails the remainder at 6 bits.

## 2. Rule or table
- **Unseen inputs, pool 1k:** exact on the test inputs of a 1,000-example run that never occurred in training, at the
  first size where most are unseen.
- **Seen / unseen at equal lengths:** fresh data, at the largest size that has unseen test inputs.
- **One step wider:** the best over trained sizes of exact one size up, counting only inputs with a digit past the
  trained width.
- **Held-out moduli:** at 12k steps.

| task | model | unseen inputs, pool 1k (size) | seen / unseen at equal lengths, fresh (size) | one step wider than trained, fresh (best) | held-out moduli (size) |
|---|---|---|---|---|---|
| add | colsum | 0.99 (3) | 1.00 / 1.00 (8) | 3 → 4: 1.00 | · |
| add | abacus | 0.93 (3) | 0.98 / 1.00 (8) | 7 → 8: 0.95 | · |
| add | abacus_loop | 0.98 (3) | 1.00 / 1.00 (8) | 8 → 9: 0.99 | · |
| mul | colsum | 0.30 (3) | 0.98 / 0.99 (5) | 4 → 5: 0.80 | · |
| mul | abacus | 0.25 (3) | 0.06 / 0.05 (3) | 2 → 3: 0.04 | · |
| mul | abacus_loop | 0.26 (3) | 0.02 / 0.01 (3) | 2 → 3: 0.05 | · |
| sq | colsum | · | 0.75 / 0.76 (6) | 6 → 7: 0.50 | · |
| sq | abacus | · | 0.00 / 0.00 (5) | 2 → 3: 0.00 | · |
| sq | abacus_loop | · | · | 3 → 4: 0.00 | · |
| mod | longdiv | 0.04 (6) | · | 6 → 7: 0.01 | 0.02 (6) |
| mod | abacus | 0.03 (6) | 0.02 / 0.03 (7) | 6 → 7: 0.01 | · |
| mod | abacus_loop | 0.04 (6) | 0.02 / 0.03 (7) | 6 → 7: 0.01 | · |
| sqmod | colsum_longdiv | 0.03 (7) | · | 6 → 7: 0.04 | · |
| sqmod | abacus | 0.06 (7) | · | 6 → 7: 0.02 | · |
| sqmod | abacus_loop | 0.10 (7) | · | 6 → 7: 0.02 | · |

- **Addition is a rule for every model.** Even from 1,000 examples it is a rule for colsum, abacus and abacus_loop
  (0.93–0.99 on unseen inputs; plain xf managed 0.88). Trained on 3 or more digits, colsum is also exact on every
  wider size tested.
- **Multiplication is a rule only with the columns built in.** Squaring follows once its table is too big to store.
  - From 1,000 examples even colsum mostly stores answers: 0.30 on unseen inputs, against plain xf's 0.155.
  - With fresh data it computes. Seen equals unseen at every size where there are unseen inputs, for example
    0.98 / 0.99 at 5 digits.
- **The abacus transformers do not get multiplication.** Values like "0.06 / 0.05" mean they get neither seen nor
  unseen 3-digit products of full length. The exact match they do have (0.74–0.79 at mul@3 after 12k) comes from
  short operands.
- **Every remainder model is at chance on unseen inputs, wider moduli and held-out moduli.** The exception is the
  quotient-given diagnostic below. longdiv's held-out-moduli entry comes from a run that never took off; see §3.

## 3. The remainder inside a long-division datapath
Exact match (best seed; ranges are the 2 seeds), fresh data. Steps are 3k unless marked.

| long-division variant | 6 bits | 7 bits |
|---|---|---|
| digit cell fitted to its truth table (no answers used) | **1.00** | **1.00** (and 1.00 at 10, 14, 20 bits) |
| from answers, 3k | 0.04–0.05 | 0.02–0.03 |
| from answers, 12k | 0.44–0.55; 0.04 in another data stream | 0.05–0.06 |
| from answers, 48k (6 bits) / 24k (7 bits) | 0.45, **0.90** | 0.06 |
| from answers, pool of 1,000 | 0.05–0.06 | 0.02 |
| straight-through choice, 3k | 0.01 | 0.01 |
| easy dividends mixed in, 3k / 12k | 0.11–0.16 / 0.07–0.34 (two attempts) | 0.04–0.05 / 0.11–0.18 |
| quotient digits given, 3k / 12k | 0.95–0.98 / **0.99–1.00** | 0.38–0.39 / 0.39–0.40 |
| quotient given: held-out moduli (trained moduli), 12k | **0.47–0.49** (1.00) | 0.21–0.23 (0.42–0.43) |
| x² mod N end to end (colsum_longdiv), 3k / 12k | 0.59–0.70 / 0.67–0.68 | 0.16–0.18 / 0.42–0.47 |
| the same with small x mixed in, 3k / 12k | 0.41–0.51 / 0.44–0.60 | 0.18 / 0.26–0.28 |
| every trained row above, one bit wider | ≤ 0.08 | ≤ 0.05 |

- **The wiring is right.** Fitted to its truth table, the digit cell makes the datapath exact at every width and for
  every modulus. A learned cell is wrong only where its own entries are wrong.
- **From answers alone the cell learns, but slowly and unevenly.**
  - At 6 bits, two seeds reached 0.44 and 0.55 by 12k. Three identical reruns gave the same numbers.
  - The held-out-moduli run, with the same settings but a different data stream, stayed at 0.04.
  - At 48k the two seeds reached 0.45 and 0.90. So the choice can be learned from answers, given time.
  - At 7 bits it is 0.05–0.06 by 12k, and still 0.06 at 24k.
  - The straight-through choice is worse (0.01).
  - Mixing in easy dividends, the wrap curriculum as data, helps a little.
    - At 6 bits: 0.11–0.16 at 3k, then 0.07–0.34 at 12k across two attempts, which is below the plain run, probably
      because the full-length divisions it needs are rarer.
    - At 7 bits: 0.11–0.18 at 12k against the plain 0.05–0.06, with seen and unseen equal (0.18 / 0.20). That is a
      rule, but a weak one.
- **Given the quotient digits, the same cell learns the subtraction.**
  - It is exact at 6 bits by 12k.
  - Trained on 12 of the 16 six-bit moduli, it gets 0.47–0.49 on the other four, against chance (~0.02) for every
    other remainder model in either ladder.
  - The digit cell's entries are shared by every input that uses them. So it is right on new moduli wherever their
    digit combinations were trained, and wrong elsewhere.
  - At 7 bits it stops at 0.39–0.40 from 3k to 12k, equal on seen and unseen inputs (0.35 / 0.34).
    - Likely cause: the cell's rarer entries. An example takes 24 cell calls, so 96% right per call already caps
      the example at about 0.4.
- **So the remainder's hard part is choosing each quotient digit.** In this datapath the choice is a comparison read
  off the cell's own borrow. Until the cell is right the choice is wrong, and while the choice is wrong the cell gets
  inconsistent lessons.
  - Compare 3 digits of multiplication, exact in 12k steps, with 6-bit long division at 0.45–0.90 after 48k, and
    almost nothing at 7 bits.
  - That is REPORT.md's wraparound again, now isolated to one decision per step.
  - RELAXED.md's periodic units tied to N face the same quantity, the wrap count. They learn it only with a curriculum
    over wraps and a scale-free optimizer (5/5 exact to 24 bits there, 0/5 without the curriculum).

## 4. What makes the difference, in plain words
**Matching buys locality.** Every answer digit then depends on its own column plus a carry, so one shared cell gets
the same lesson at every position of every example, and gradient descent finds the rule.
- That is why addition was always easy.
- It is also what multiplication was missing. A plain network has to discover, from answers alone, which digit
  pairs to multiply and add for each column.
- Once that wiring is given, the products and carries are learnable, and the column-sum model multiplies numbers it
  never saw and numbers wider than it trained on.
- Position coupling (the abacus embeddings) lines the digits up but does not do the summing, so it helps addition
  carry past its trained width and leaves multiplication a table.

**A small table still wins when it is cheap.**
- At up to 10⁴ possible inputs (sq@4, mul@2), even the matched model stores answers. It is exact there, but it fails
  one digit wider.
- It switches to the rule once there are too many inputs to store (sq@6, mul@3 and up).

So the architecture makes the rule available; the data decides whether it is used.

**The remainder has no such locality, even inside long division.**
- Each quotient digit is a comparison of the whole running remainder with N, and every later digit depends on it.
- Long division makes that decision explicit, once per step. The rest of the step, multiplying N by the chosen digit
  and subtracting with borrows, is local and is learned once the decision is given.
- The decision itself learns from answers only slowly. It is about 0.9 after 48k steps at 6 bits, and almost
  nothing at 7.
- It is the wrap count again: the quantity REPORT.md's scrambling hides, and the one RELAXED.md's periodic units only
  get with a wrap curriculum.

**Which pieces become rules, and which stay tables:**

| piece | plain models (LADDER.md) | matched architecture |
|---|---|---|
| addition | rule | rule, at any width (column scan) |
| multiplication | table | **rule** with the column structure; still a table with position coupling only |
| squaring | table | **rule** once its input set is too big to store; a table below that |
| multiply back and subtract (given the quotient) | table | **rule-shaped**: exact at 6 bits, half right on unseen moduli; 0.4 at 7 bits |
| choosing the quotient digit (the wrap count) | table | **learned only slowly**: 0.45–0.90 at 6 bits after 48k steps, 0.06 at 7 bits after 24k |
| x² mod N | table | table (colsum_longdiv: 0.67–0.68 at 6 bits, 0.42–0.47 at 7, ≤ 0.05 one bit wider) |

## 5. Caveats
- **colsum hard-codes the schoolbook wiring**, which pairs feed which column. That is the "impedance match" under test.
  Its learned parts are a 100-entry pair table and a carry cell.
- **colsum's extrapolation is partial.** Its carry is a GRU state, and wider numbers bring column sums larger than any
  it trained on.
  - One digit wider is 0.73–0.92 from 4 digits up, and less two digits wider.
  - Training longer does not always help: mul@3 at 48k gets 0.53 one digit wider, against 0.74–0.80 at 12k.
  - A cell that saw only bounded values (schoolbook partial products, then additions) would be width-free. REPORT.md
    found that version does not learn from answers: its credit path is too long.
- **The abacus models are this harness's encoder with parallel outputs.** The Abacus paper used a causal decoder
  over reversed digits; that variant is untested here.
- **longdiv's choice rule is one relaxation:** a soft "largest non-borrowing q". The natural next variant is a learned
  quotient-digit estimator from the leading digits (SRT-style), checked by the borrow.
- **Small budgets and few seeds.** There are 2 seeds per cell. longdiv from answers is the one place where runs
  disagree in kind:
  - 0.44–0.55 in one data stream and 0.04 in another at 12k;
  - 0.45 vs 0.90 across seeds at 48k.
- **The transformers ran on CPU** because the GPU quota was in use. The numbers are unaffected, but each abacus cell
  took 6–16 minutes.

## 6. Reproduce
Run from `cpu_first/`. Everything trains on Modal.

The 3k ladder (add `--cpu-only` when the GPU quota is taken):

```bash
modal run --detach microlab/ext_matched_modal.py --tasks add,mul,sq --models colsum --data fresh,pool1000 --tag M3k --extra '{"lr": 2e-3}'
```

```bash
modal run --detach microlab/ext_matched_modal.py --tasks add,mul,sq,mod,sqmod --models abacus,abacus_loop --data fresh,pool1000 --tag M3k
```

```bash
modal run --detach microlab/ext_matched_modal.py --tasks mod --models longdiv --data fresh,pool1000 --tag M3k --extra '{"lr": 3e-3}'
```

The 12k re-test from a breaking point:

```bash
modal run --detach microlab/ext_matched_modal.py --tasks mul --models colsum --data fresh --steps 12000 --tag M12k --extra '{"lr": 2e-3}' --start '{"mul,colsum,fresh": 3}'
```

Remainder variants: the quotient given; easy dividends mixed in; held-out moduli.

```bash
modal run --detach microlab/ext_matched_modal.py --tasks modq --models longdiv --data fresh --steps 12000 --tag Mrem --extra '{"lr": 3e-3, "sizes": [6, 7], "patience": 9}'
```

```bash
modal run --detach microlab/ext_matched_modal.py --tasks mod --models longdiv --data fresh --steps 12000 --tag Mrem --extra '{"lr": 3e-3, "train_on": "modmix", "sizes": [6, 7]}'
```

```bash
modal run --detach microlab/ext_matched_modal.py --tasks modq --models longdiv --data fresh --steps 12000 --tag Mho --extra '{"lr": 3e-3, "sizes": [6, 7], "holdout": 4}'
```

Tables:

```bash
python -m microlab.ext_matched_report --tags M3k,M12k --holdout Mho
```

```bash
python -m microlab.ext_matched_report --tags Mrem,Mlong,Mho --ledger
```
