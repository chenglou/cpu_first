# The quotient-digit chicken-and-egg, shrunk until it can be computed exactly

## The answer in plain words

**What is being learned.** microlab's long-division datapath ([LADDER_MATCHED.md](../microlab/LADDER_MATCHED.md) §3) divides
digit by digit. At each step it tries every possible quotient digit at once, one "trial slot" per digit, and keeps
one. Two things are learned from final remainders only: a small digit table that does the subtracting (with
borrows), and the rule that picks the quotient digit. Each one learns fast when the other is given. Together they
stall ([FINDINGS_quotient.md](../microlab/FINDINGS_quotient.md)).

**The smallest machine that still gets stuck.** Binary (base 2), moduli N = 4 to 7, dividends below 4N, so every
example has two quotient bits. That is 88 examples in all. The machine has two trial slots per step. Each slot has its
own little subtraction table, and a chooser picks a slot. Nothing says in advance that one slot should subtract
nothing and the other should subtract N.
- Given the right quotient bits, the tables learn to subtract in 100 steps.
- Given exact subtraction, the chooser learns in 50 steps.
- Trained together from remainders, 0 of 6 runs get there. They stop at about half the answers right.

**Why.** At the start both slots are the same blank table.
- The chooser learns by comparing slot A's answer with slot B's. They are identical, so its lesson is exactly zero.
- Each slot learns "make the answer right on the examples routed to you". A 50/50 chooser routes both slots the same
  mix of "subtract" and "don't subtract" examples, so both learn the same compromise and stay identical.
- Training therefore drifts into this "all slots alike" state. It is a saddle: the only ways down are directions in
  which the slots differ. Those differences start as random noise and must grow on their own.
- When they grow, the slots split along whatever split the noise and the loss happen to favor. With two slots and one
  quotient digit that split is the right one. With a chain of quotient digits, or three or more slots, it usually is
  not: each lesson then passes through other undecided steps, and many wrong splits grow about as fast as the right
  one.
- The run then settles into a "lookup arrangement": slots permuted, one slot doing two digits' jobs, some doing none,
  and the tables patched entry by entry to make that work. It is a dead end: the straight walk from there to the true
  tables goes uphill.

A manager and two identical new hires are in the same bind. The manager can route work only once the hires differ,
and the hires come to differ only once they get different work. They end up specialized by accident.

**What fixes it.** The root is that each half's lesson passes through the other. Fixes that keep both halves
learning from the same answers at the same time leave the trap in place, or trade it for another dead end:
- alternating which half trains, or a curriculum alone, leaves it in place;
- hard EM locks in the first, random split;
- scoring the answer only modulo N (a "clock") is satisfied perfectly by a table that never subtracts.

What works is giving the subtraction table a fair lesson that does not depend on the choice being right:
1. **Fix what each slot means.** All slots share one subtraction table, and slot s subtracts the exact multiple s·N.
   The slots now differ from the start, and in the right way, so there is no symmetric trap. Within 25 training steps
   the chooser's lesson points at the true digit for 93–99% of choices.
2. **Teach that table with a chooser that starts undecided and cannot cover for it.** Use an estimator that sees only
   the leading digits of the running remainder and of N (microlab's default window), starting with equal odds on
   every slot.
   - Every trial gets a fair share of lessons. Only the right slot's trial is consistently right, so the shared table
     drifts toward true subtraction.
   - A window that misses the low digits cannot memorize which wrong slot happens to fix which wrong table entry.
   - Teach one quotient digit first, then add one at a time.
3. **Then hand the choice to the subtraction's own borrow** (the largest slot whose result does not go negative). Once
   the table subtracts correctly, that rule is exact at any width, with no window limit.

**How well it works.**
- **Tiny core** (bases 2–5, moduli using every leading digit): fixed slot meanings, the subtraction's own borrow and one
  quotient digit at a time solve every size, 6 of 6 seeds, in about 100 steps. Each piece alone fails somewhere.
- **microlab's moduli** (odd N between 2^(b−1) and 2^b) use only a few leading digits. There the own borrow cannot start
  the learning, so the estimator has to bootstrap the table first (step 2).
- **microlab's own sizes in base 10**, answers only:
  - 6 bits: 11 of 11 seeds exact;
  - 7 bits: 11 of 11;
  - 10 bits: 7 of 8. All 4 with microlab's estimator window (top 3 / top 2); 3 of 4 with a coarser one (top 2 /
    top 1), where in the fourth the table never got started;
  - each in a few thousand steps. Tables trained at 7 bits also work unchanged at 10 bits: 0.92–0.99 exact.
- **A second route** teaches the table from identities with a trusted adder, (a − b) + b = a on random numbers. It is
  exact at 6, 7 and 10 bits.

**Prediction for microlab at 7–10 bits.**
- **What should work:**
  - replace longdiv's learned slot embedding with one shared subtractor cell fed the exact digits of q·N;
  - train it with longdiv_est's estimator choice, one quotient digit at a time from the dividend's top digits;
  - then switch to longdiv's own borrow choice.

  That should be exact at 7 and 10 bits.
- **What should not:** hard EM and the clock loss, both running in microlab now.
  - EM locks in the first random split: 0/6 at every tiny size.
  - The clock is satisfied by a cell that copies, so expect exact near the share of examples whose quotient is 0.

Details in §6.

## 1. The tiny core
`core.py` is microlab's `LongDiv` with every learned part a table, so that every example can be enumerated and every
gradient computed exactly (float64).

**Datapath.** Same as microlab's:
- The running remainder R is w+1 digit distributions. It starts at the dividend's top w digits, which are below N.
- Each step forms Rp = B·R + v_t, the next dividend digit shifted in.
- It then tries all B slots. A trial is a ripple of a digit cell from the lowest digit up, ending in a carry.
- A choice weights the slots, and R becomes the mixture of the trials' digit distributions.
- The loss is the cross-entropy of R's low w digits against v mod N.

Tables are applied to digit distributions multilinearly (the expected output over independent inputs), the way
microlab's `soft_cell` diagnostic applies the truth table.

**A size** is (B, w, k): base B, so B slots, moduli with w base-B digits, and dividends v < N·B^k, so k quotient
digits. Every (N, v) of the size is used. The base-10 runs (§5) instead use microlab's own distribution, odd N in
[2^(b-1), 2^b) and v < N², with fresh batches.

| part | variants |
|---|---|
| cell | **slot**: one table (r, n, slot, carry) → (digit, carry), carries 0..B−1. What a slot subtracts is learned, as with microlab's learned slot embedding. **mult**: one shared subtractor (r, m, borrow) → (digit, borrow); slot s subtracts the exact digits of s·N. **repeat**: the same subtractor run s times for slot s. |
| choice | **borrow**: microlab longdiv, the largest slot whose final carry is 0, softly. **est**: microlab longdiv_est, a table over the top digits of Rp and of N (all of them in the tiny sizes, 3 and 2 at base 10). **given**: the true digits (microlab modq). |

Training is Adam at lr 0.05 on the logits, batches of 256 drawn from all examples, and initial logits N(0, 0.1²). A run
stops when exact match reaches 0.99 and gets at most 4,000 steps. There are 6 seeds per cell. Every variant is exact
with the true tables (`python -m tinycore.selftest`).

## 2. Where it goes from easy to never
Joint training from remainders (`phase.py`, rows in `runs/phase.jsonl`). Each entry gives solved / flat-then-take-off
/ never, out of 6 seeds. A run counts as flat-then-take-off when it sat still for 300+ steps above the start before
solving.

| B (slots) | w | k | examples | estimator choice | own-borrow choice | median final exact (est, borrow) |
|---|---|---|---|---|---|---|
| 2 | 3 | 1 | 44 | 4/6 (0 / 2) | 5/6 (0 / 1) | 1.00, 1.00 |
| 2 | 3 | 2 | 88 | **0/6** (0 / 6) | 3/6 (0 / 3) | 0.59, 1.00 |
| 2 | 3 | 3 | 176 | 0/6 (0 / 6) | 0/6 (0 / 6) | 0.51, 0.43 |
| 2 | 4 | 1 | 184 | 5/6 (0 / 1) | 6/6 (0 / 0) | 1.00, 1.00 |
| 2 | 4 | 2 | 368 | 4/6 (0 / 2) | 3/6 (0 / 3) | 1.00, 1.00 |
| 2 | 4 | 3 | 736 | 0/6 (0 / 6) | 3/6 (1 / 3) | 0.33, 1.00 |
| 2 | 4 | 4 | 1,472 | 0/6 (0 / 6) | 0/6 (0 / 6) | 0.31, 0.22 |
| 2 | 5 | 1 | 752 | 4/6 (0 / 2) | 6/6 (0 / 0) | 1.00, 1.00 |
| 2 | 5 | 2 | 1,504 | 5/6 (1 / 1) | 6/6 (0 / 0) | 1.00, 1.00 |
| 2 | 5 | 3 | 3,008 | 1/6 (1 / 5) | 4/6 (0 / 2) | 0.35, 1.00 |
| 2 | 5 | 4 | 6,016 | 0/6 (0 / 6) | 1/6 (0 / 5) | 0.19, 0.13 |
| 3 | 2 | 1 | 99 | **0/6** (0 / 6) | 4/6 (2 / 2) | 0.93, 1.00 |
| 3 | 2 | 2 | 297 | 0/6 (0 / 6) | 0/6 (0 / 6) | 0.76, 0.65 |
| 3 | 3 | 1 | 945 | 0/6 (0 / 6) | 6/6 (1 / 0) | 0.89, 1.00 |
| 3 | 3 | 2 | 2,835 | 0/6 (0 / 6) | 0/6 (0 / 6) | 0.39, 0.78 |
| 4 | 2 | 1 | 456 | 0/6 (0 / 6) | 1/6 (1 / 5) | 0.91, 0.96 |
| 4 | 2 | 2 | 1,824 | 0/6 (0 / 6) | 0/6 (0 / 6) | 0.67, 0.52 |
| 5 | 2 | 1 | 1,450 | 0/6 (0 / 6) | 0/6 (0 / 6) | 0.83, 0.93 |
| 5 | 2 | 2 | 7,250 | 0/6 (0 / 6) | 0/6 (0 / 6) | 0.55, 0.43 |

(Also run, all 0/6 except where shown: (3, 2, 3), (3, 3, 3), (4, 2, 3).)

**The microlab symptoms reproduce.**
- **Each half learns alone.** At (2, 3, 2), given the true digits the cell is exact in 100 steps. With the exact cell
  the estimator is exact in 50.
- **A flat stretch, then a sudden take-off.** Full batch at (2, 4, 3) with the own-borrow choice, seed 1:
  - exact sits at 0.22 from step 100 to step 1,000, with every choice at about 50/50 whatever the true digit;
  - meanwhile the cell's table sharpens: its digit entropy falls from 0.69 to 0.12 nats;
  - exact then goes 0.25 → 0.73 → 1.00 within 100 steps. The first division step (the one with clean inputs) commits
    first, and the others follow.
- **Permuted and merged slots** at the end of failed runs: slot maps [4, 1, 0, 3, 3] and [1, 3, 4, 3, 2] at (5, 2, 2),
  and [2, 0, 3, 3] at (4, 2, 1). Slot i stands mostly for digit map[i], so two slots serve one digit and some digits
  have none.
- **Failure one size up.**
  - In base 2, one quotient digit is easy at every width, two are seed-dependent, three or four mostly never work.
  - In base 3, the own-borrow choice solves one quotient digit (4–6/6) and none of two.
  - Three or more learned slots fail with the estimator even at one quotient digit.
- **Size scaling.** More slots and more quotient digits hurt most. More digits of N help a little in base 2: with more
  moduli, a lookup arrangement fits fewer examples.

**The smallest failing cores** are (B, w, k) = (2, 3, 2) with the estimator, 88 examples and 2 slots, and (3, 2, 1)
with the estimator, 99 examples with one quotient digit and 3 slots.

## 3. Why: exact computations

### 3.1 The first gradient
At the exact uniform start (every logit 0), on every example (`exact.py grad0`; base 10 in `suites.py grad0`, on
Modal):
- **The choice's gradient is exactly 0 with learned slots**, at every size: every slot computes the same output, and
  the choice's lesson is the difference between slots' outputs.
- **Every slot gets the identical lesson** (the spread across slots is exactly 0), so the slots stay identical.
- **Every carry entry's gradient is exactly 0** with learned slots and with the shared subtractor, at every size,
  base 10 included. A carry matters only through the next cell, which at the start ignores its carry input. So the
  own-borrow choice, which reads the carries, starts with no lesson either. (With repeated subtraction the carries do
  get a first lesson: slot 0 is then the exact input, so slot 1's borrow decides something from the start.)
- **The digit entries get the slot-averaged lesson.** It points toward the truth for only 53–63% of the table rows:
  - with learned slots, 10 toward / 6 away at (2, 4, 3), 36 / 27 at (3, 2, 2), 132 / 76 at (4, 2, 2), and 4,812 /
    4,288 at base 10 (2 digits, 2 quotient digits);
  - with the shared subtractor, 57–64% toward at base 10 (113 / 87);
  - the "away" rows are carry-blind: rows with carry 0 and carry 1 get the same lesson while the carry is still
    uniform.
- **With repeated subtraction the first estimator gradient is not zero** (slot 0 is then the exact input). With the
  shared subtractor it is zero at the exact start only because every trial outputs the same uniform digits. That is
  an accident of the starting point, not a symmetry; §3.6 shows the lesson appearing within 25 steps.

### 3.2 The undecided plateau is a symmetric saddle
`saddle.py`: start exactly slot-symmetric, with every slot the same random table and the estimator equal across
slots. Gradient descent then stays symmetric forever: the antisymmetric norm stays 0.0 to machine precision.
- Along this trajectory the loss settles on a plateau. At (2, 3, 1) that is exact 0.545 at loss 0.35.
- The Hessian's most negative eigenvalue lies in the slot-antisymmetric block: −1.81·10⁻² for the full Hessian and
  for that block alike. So the plateau is a saddle, and its only exits are the slots becoming different.
- **The plateau's length follows the saddle law.** Add an antisymmetric perturbation of size ε, then run plain
  gradient descent at lr 1, (2, 3, 1). The predicted step count is t = t₀ + ln(1/ε)/λ.
  - Steps to exact ≥ 0.99, for ε = 10⁻¹, 10⁻², 10⁻³, 10⁻⁴, 10⁻⁶, 10⁻⁸, 10⁻¹⁰:
    - seeds 0 and 1: 490, 630, 760, 890, 1,140, 1,390, 1,650 (the seed-1 run takes 10 steps more at 10⁻⁶ and 10⁻⁸);
    - seed 2: 760, 850, 970, 1,100, 1,360, 1,610, 1,860.
  - That is 129 (seeds 0 and 1) and 122 (seed 2) steps per factor of 10 in ε. The Hessian predicts ln 10 / 0.0181 =
    127.

### 3.3 The escape slows down with every extra digit or slot
The escape rate λ is the most negative antisymmetric eigenvalue, along the symmetric trajectory after 1,000 steps
(`escape.py`, Lanczos on Hessian-vector products). 1/λ is the number of plain-gradient steps (lr 1) per e-fold of
asymmetry.

| B | w | k | λ at step 0 | λ at step 1,000 | 1/λ (steps) |
|---|---|---|---|---|---|
| 2 | 3 | 1 | 1.81·10⁻² | 1.80·10⁻² | 56 |
| 2 | 4 | 1 | 1.41·10⁻² | 8.08·10⁻³ | 124 |
| 2 | 5 | 1 | 1.40·10⁻² | 3.85·10⁻³ | 260 |
| 2 | 3 | 2 | 1.04·10⁻² | 6.44·10⁻³ | 155 |
| 2 | 3 | 3 | 1.04·10⁻² | 3.76·10⁻³ | 266 |
| 2 | 4 | 3 | 6.02·10⁻³ | 3.54·10⁻³ | 282 |
| 2 | 5 | 3 | 4.73·10⁻³ | 3.01·10⁻³ | 332 |
| 3 | 2 | 1 | 9.17·10⁻³ | 4.82·10⁻³ | 207 |
| 3 | 3 | 1 | 6.30·10⁻³ | 1.88·10⁻³ | 532 |
| 4 | 2 | 1 | 4.35·10⁻³ | 1.46·10⁻³ | 685 |

- Each extra quotient digit, digit of N or slot roughly halves λ, so the flat stretch roughly doubles.
- λ also keeps falling while the run waits: an undecided cell sharpens toward the slot-averaged table, and a sharp
  table passes on smaller lessons. At (2, 3, 3) λ falls from 1.04·10⁻² to 3.76·10⁻³ in 1,000 steps.

### 3.4 Where the escape leads: the right arrangement only in the smallest cases
For each of the six fastest escape directions, take the slot it favors in every estimator window. Score that against
the true quotient digit under the best slot relabeling, weighting windows by how often the first division step uses
them. 1.00 means the direction is the true arrangement; 0.5 is chance with 2 slots.

| B | w | k | agreement of the fastest escape directions, at step 1,000 | spectrum near the top |
|---|---|---|---|---|
| 2 | 3 | 1 | **1.00**, 0.64, 0.57 | gap: 1.80, 1.16, 0.42 ·10⁻² |
| 2 | 4 | 1 | **1.00**, 0.62, 0.62 | gap: 8.1, 5.0, 2.6 ·10⁻³ |
| 2 | 5 | 1 | **1.00**, 0.61, 0.59 | gap: 3.9, 2.3, 1.4 ·10⁻³ |
| 2 | 3 | 2 | 0.50, 0.64, 0.50 | 6.4, 5.0, 1.5 ·10⁻³ |
| 2 | 3 | 3 | 0.52, 0.59, 0.50 | 3.8, 2.0, 1.3 ·10⁻³ |
| 2 | 4 | 2 | 0.52, 0.85, 0.55 | 4.2, 2.9, 2.3 ·10⁻³ |
| 2 | 4 | 3 | 0.51, 0.57, 0.51 | 3.5, 1.7, 1.3 ·10⁻³ |
| 2 | 5 | 2 | 0.90, 0.51, 0.82 | 2.6, 2.1, 1.2 ·10⁻³ |
| 3 | 2 | 1 | 0.63, 0.63, 0.63 | degenerate pairs: 4.82, 4.82, 4.55, 4.55, 4.24 ·10⁻³ |
| 4 | 2 | 1 | 0.43, 0.43, 0.34 | near-degenerate: 1.46, 1.46, 1.45, 1.45, 1.39 ·10⁻³ |

This table explains the phase map:
- **One quotient digit, two slots:** the fastest escape direction is exactly the true arrangement, with a clear gap
  to the next. These sizes solve.
- **More quotient digits:** the fastest directions are at chance. The exceptions match the phase map: (2, 5, 2) at
  0.90 solves 5–6/6, and (2, 4, 2), whose second direction is at 0.85, solves 3–4/6.
- **Three or more slots:** with interchangeable slots the escape directions come in equal-rate pairs (the
  representation theory of permuting 3 or 4 slots), with many more just behind them. No single arrangement wins.
  - The base-3 table has 100–176 unstable directions.
  - The run leaves along a random mixture and commits to whatever arrangement that mixture favors.

### 3.5 Lookup arrangements are real local minima; the start is not
Loss along straight lines on every example (`paths.py`):
- **From the uniform start to the true tables**, the loss falls monotonically at every size tested: (2, 3, 1),
  (2, 3, 2), (2, 4, 3), (3, 2, 1), (3, 2, 2) and (4, 2, 1), with either choice. Nothing blocks the direct route.
  Gradient descent just does not take it, because the first gradient points to the slot-averaged table.
- **From a stuck run to the truth** under the best slot relabeling, the loss first rises:

| size and choice | stuck run's exact | barrier (nats per digit) |
|---|---|---|
| (2, 3, 2) est | 0.59 | +0.24 |
| (2, 4, 3) est | 0.33 | +0.20 |
| (2, 4, 3) borrow | 0.29 | +0.32 |
| (3, 2, 1) est | 0.90 | +0.25 |
| (3, 2, 2) est | 0.65 | +0.40 |
| (4, 2, 1) est | 0.87 | +0.30 |

So "never" means the run reached a separate basin, not that it was slow.

### 3.6 Optimizer, and where the flat stretch comes from
- **Under Adam** (microlab's optimizer), the symmetric saddle is left quickly whatever ε is. At (2, 3, 1), lr 0.05:
  - seed 0 solves in 40–50 steps for every ε from 10⁻¹ to 10⁻¹²;
  - seed 1 solves for ε = 10⁻¹ and 10⁻² (90 and 130 steps) and lands in a 0.84 lookup state for every ε ≤ 10⁻³.
  - Adam divides each update by the gradient's recent size, so a plateau's small gradients become full-size steps.
  - Under Adam, then, the danger is less the waiting than the direction: the arrangement is decided at the exit,
    and §3.4 says it is usually wrong.
- **The flat stretches that do occur come with the own-borrow choice**, in the tiny core as in microlab's longdiv. That
  choice reads the carries, and §3.1 shows their lesson is exactly zero at the start. It appears only once the digit
  entries depend on the incoming carry *and* the slots differ: a product of two quantities that both start at zero.
- **With 10 slots the own-borrow choice also starts biased.** Keep-the-largest-valid gives slot s the weight
  valid_s · Π_{s' > s} (1 − valid_s'). With a binary borrow every validity starts near 1/2, so slots 9, 8, 7, …
  start with weights of about 1/2, 1/4, 1/8, ….
  - At 7 bits with the shared subtractor, the mean chosen slot starts at 7.9 against a true mean of 4.5. It is still
    8.2 after 300 steps.
  - So the subtraction table learns mostly from over-subtracting trials, a biased lesson that does not average out.
  - An estimator starts with equal odds on every slot. With constructed slots that is a fair lesson: only the right
    slot's trial is consistently right across examples, so the shared table drifts toward the truth. At 7 bits, with
    one quotient digit, it is right on 0.98 of examples given the true digits by step 200 (§5).
- **Does the choice get a usable lesson?** `lesson.py` starts from the exact uniform point and trains jointly. Every 25
  steps it measures the share of choices whose probability of the true digit rises under a small step on the choice's
  own parameters:

| size | learned slots, estimator | constructed slots (mult), estimator | learned slots, own borrow | constructed slots, own borrow |
|---|---|---|---|---|
| (2, 4, 1), step 25 | 0.00 (no lesson at all) | **0.99** | 0.89 | **0.99** |
| (3, 2, 1), step 25 | 0.58 (from rounding noise) | **0.98** | 0.19 (mostly away) | 0.76, then 1.00 at step 75 |
| (4, 2, 1), step 25 | 0.00 | **0.93** | 0.53 | **0.94** |
| (2, 4, 3), step 100 | 0.00 | 0.78 | 0.50 | 0.72 |
| (3, 2, 2), step 100 | 0.46 | 0.53 | 0.41 | 0.60 |

  - With learned slots and the estimator, the choice's lesson stays exactly zero at every step: the symmetric subspace
    is invariant. At (3, 2, 1) floating-point rounding broke the tie, and the lesson then is chance.
  - With constructed slots the lesson appears within 25 steps and points at the true digit for 93–99% of choices with
    one quotient digit.
  - With two or three quotient digits even the constructed lesson is only 53–78% right, because it passes through
    undecided later steps. That is what staging removes.

### 3.7 The statistical-mechanics reading
Saad & Solla's soft committee machines fit the first half of this picture:
- interchangeable hidden units give a symmetric, unspecialized saddle (the "symmetric phase");
- the time to leave it grows like ln(1/ε) over the escape rate (measured exactly here, §3.2);
- the plateau lengthens with the number of interchangeable units: their K, our slots (§3.3).

([Saad & Solla, NeurIPS 1995](https://papers.neurips.cc/paper_files/paper/1995/file/a1519de5b5d44b31a01de013b9b51a80-Paper.pdf);
on plateau length growing with K, [arXiv:2104.14546](https://arxiv.org/pdf/2104.14546).)

The second half does not fit the realizable committee machine. There, leaving the plateau leads to the right
specialization. Here it usually does not (§3.4): the lessons that should tell slots apart pass through other undecided
steps, and many wrong splits grow nearly as fast as the right one.

That second half is the known pathology of mixture-of-experts training. The slots are experts and the choice is the
gate. EM and gradient descent are known to get stuck in local optima there.
[Makkuva et al. (ICML 2019)](https://proceedings.mlr.press/v97/makkuva19a.html) break the deadlock by learning the
experts first from a signal that does not involve the gate, then the gate, which is then easy. The fixes that work here
do the same: they give the slots their meaning without the choice.

## 4. The candidate fixes on the core
Each fix runs on the sizes where joint training fails (`fixes.py`, `fixsweep.py`, `constructs.py`; rows in `runs/`).
Solved seeds out of 6; in brackets the median final exact of the runs that did not solve.

| fix | what it changes in the analysis | (2, 4, 3) est | (2, 4, 3) borrow | (3, 2, 2) est | (3, 2, 2) borrow |
|---|---|---|---|---|---|
| joint (baseline) | – | 0/6 (0.33) | 3/6 | 0/6 (0.76) | 0/6 (0.62) |
| alternate (freeze one half, swap every 200 steps) | nothing: from a symmetric start the frozen half leaves the other a zero or slot-identical lesson; it only slows the coupling | 0/6 (0.34) | 5/6, 4× slower (median 2,300 steps) | 0/6 (0.72) | 0/6 (0.63) |
| hard EM, every path searched | no saddle (hard assignment breaks the tie at once), but the first E-steps pick paths from the random start, k-means-style, and lock in | 0/6 (0.31) | 0/6 (0.24) | 0/6 (0.36) | 0/6 (0.30) |
| hard EM, microlab's E-step (own path and one-step changes) | the same. With an exact cell and an untrained estimator its candidates miss the true path on 14–29% of examples | 0/6 (0.39) | · | 0/6 (0.36) | · |
| clock scoring, hard estimator choice, label-free estimator lessons | removes the choice from the cell's lesson, but a cell that copies (every slot subtracts 0·N) scores a perfect clock | 0/6: exact mod N 1.00, exact 0.125 = the share with quotient 0 | · | 0/6 (exact 0.05; mod N 0.61) | · |
| clock, true digits with 30% off by one | the same collapse | 1/6 (mod N 0.49) | · | 0/6 (0.00) | · |
| mixed easy dividends (every batch k = 1..K; microlab modmix) | cleans part of the chain; the slot symmetry stays | 5/6 | 4/6 | 0/4 (0.87) | 2/6 |
| staged: k = 1 until exact, then 2, ... | each lesson direct; the slot symmetry stays, so it works only where one quotient digit works | 5/6 | 5/6 | 0/6 (0.84) | 4/6 |
| label-free checks, then joint | the cell is learned without the choice (from a − 0·b = a and (a − s·b) + s·b = a, with an exact adder, on random numbers), so the choice then faces the exact-cell problem | **6/6** | **6/6** | 5/6 | 4/6 |
| repeat (slot s = s subtractions) | no slot symmetry; the first estimator gradient is not zero; chained soft subtractions blur as B grows | 5/6 | **6/6** | 0/6 (0.70) | 2/6 |
| mult (slot s subtracts exact s·N, one shared subtractor) | no slot symmetry, no extra depth; the choice's lesson points at the truth within 25 steps | · | · | 0/6 (0.72) | 2/6 |
| **mult + staged** | all three parts of the trap removed | · | · | 2/6 (0.81) | **6/6** |

The construction sweep extends the last rows to more slots (`constructs.py`):

| size | mult + borrow, joint | **mult + borrow, staged** | mult + est, staged | repeat + borrow, staged |
|---|---|---|---|---|
| (3, 2, 1) | 6/6 | (k = 1: same as joint) | 1/6 joint (0.90) | 4/6 joint |
| (3, 3, 2) | 6/6 | **6/6** | 6/6 | 4/6 |
| (4, 2, 1) | 6/6 | (same as joint) | 6/6 joint | 5/6 joint |
| (4, 2, 2) | 0/6 (0.32) | **6/6** | 3/6 (0.99) | 5/6 |
| (5, 2, 1) | 6/6 | (same as joint) | 3/6 joint (0.91) | 2/6 joint (0.46) |
| (5, 2, 2) | 1/6 (0.23) | **6/6** | 3/6 (0.89) | 2/6 (0.30) |

At bases 4 and 5, (4, 2, 1), (4, 2, 2) and (5, 2, 1), the same picture holds:
- Label-free checks: 4–6/6.
- Repeated subtraction: 2–5/6 with the own borrow.
- Staging with learned slots: 0–1/6.
- Hard EM, either E-step: 0/6, ending at 0.22–0.85.
- Alternating: 0/6.
- Clock: 0/6. Every slot that settles on a consistent multiple of N settles on 0, and exact is exactly the share of
  examples whose quotient is 0 (1/B^k: 0.25, 0.20, 0.0625) while exact mod N is 0.9–1.0.

What the tables say:
- **Construction, own-borrow choice and staging together are the one combination that never failed**: 6/6 at every
  size, in a median of 100 steps.
- **Each alone fails somewhere.**
  - Construction with joint training fails at two quotient digits once there are 4–5 slots.
  - Staging with learned slots fails with 3 slots and the estimator.
  - The estimator choice with constructed slots stalls at 0.81–0.99 in half the runs. In those runs the table choice
    picks a "wrong" slot on a few windows to cover for a wrong subtraction entry: at (3, 2, 1) one stuck run's
    subtractor has 17 of 18 rows literally different from the truth and is exact on only 79% of examples given the
    true digits.
  - Repeated subtraction degrades by base 5 (0.30–0.46 even with the right slot map): four chained soft
    subtractions blur.
- **Label-free checks also break the deadlock** when the identities pin the cell: 6/6 at (2, 4, 3), 4–5/6 at (3, 2, 2).
  They rely on an exact adder.

## 5. microlab's own sizes (base 10)
`b10.py`, on Modal:
- base 10, with microlab's moduli: odd N in [2^(b−1), 2^b), so 33–63 at 6 bits, 65–127 at 7 and 513–1023 at 10;
- dividends v < N², fresh batches of 256 every step;
- exact match on a fixed sample of 4,000;
- the datapath starts from the dividend's top w digits unless noted;
- 2–4 seeds per row: solved seeds, then the median final exact.

**The microlab analogs fail the way microlab does.**

| learned slots (microlab's cell) | 6 bits | 7 bits |
|---|---|---|
| own borrow, joint (longdiv) | 0/3, 0.29 | 0/3, 0.07 |
| estimator, joint (longdiv_est) | 0/3, 0.51; slots [0, 2, 2, 1, 0, 2, 1, 2, 1, 1] | 0/2, 0.11 |
| own borrow, staged | 0/3, 0.62; slots [4, 4, 8, 7, 5, 3, 2, −1, 1, 9] | 0/3, 0.22 |
| own borrow, joint, every 2-digit modulus | 0/3, 0.16 | · |

For comparison, microlab measured longdiv at 0.44–0.55 (6 bits, 12k steps) and 0.05–0.06 (7 bits), and longdiv_est at
0.53–0.54 (6 bits). The slot maps are microlab's permuted, merged arrangements.

**Construction with staging works on full-range moduli, not on microlab's.**

| shared subtractor, own borrow | 6 bits | 7 bits | 10 bits |
|---|---|---|---|
| joint | 0/3, 0.07 | 0/3, 0.02 | · |
| staged | 0/3, 0.16 | 0/1, 0.04 | · |
| staged, starting from R = 0 (w always-0 steps first, as microlab runs) | 0/2, 0.06 | 0/3, 0.02 | · |
| staged, the register's top digit also scored (target 0) | 2/4 | · | · |
| staged, **every w-digit modulus** | **3/3** (200 steps) | **3/3** (200) | **3/3** (300) |

Why microlab's moduli break it:
- The own-borrow choice starts with most of its weight on the top slots (§3.6), so the table learns mostly from
  over-subtracting trials.
- microlab's moduli span only a few leading digits, so a wrong table survives that wider moduli would contradict. One
  stuck 6-bit run has mostly-right digits but inverted borrows on many rows: 115 of the 161 rows its true paths visit
  are wrong.
- One quotient digit at microlab's 6-bit moduli: 1/3 seeds solve with only the answer's digits scored, 3/3 when the
  register's top digit is scored too. Scoring that digit did not carry over to base 5 (§7).

**Two routes that do work at microlab's moduli.**

*Answers only.* Shared subtractor; estimator choice, starting at equal odds; staged; then the subtractor's own
borrow. `b10d`, `b10e`, `b10f`:

| trained at | cell | estimator window | seeds | estimator alone: exact / exact given the true digits | after switching to the own borrow: solved (median steps) | the same tables at 6 / 7 / 10 bits |
|---|---|---|---|---|---|---|
| 6 bits | shared subtractor | top 2 / top 1 | 4 | 0.68–0.70 / 1.00 | 4/4 (200) | 0.99 / 0.73–0.75 / 0.07–0.08 |
| 6 bits | shared subtractor | top 3 / top 2 | 7 | 0.58–0.75 / 1.00 | 7/7 (300) | 0.99–1.00 / 0.71–0.77 / 0.07–0.29 |
| 6 bits | learned slots | top 3 / top 2 | 4 | 0.73–0.77 / 0.03–0.07 | 0/4 (0.74–0.77 after 2,000 steps) | · |
| 7 bits | shared subtractor | top 2 / top 1 | 4 | 0.29–0.30 / 1.00 | 4/4 (300) | 1.00 / 0.99–1.00 / 0.96–0.99 |
| 7 bits | shared subtractor | top 3 / top 2 | 7 | 0.76–0.77 / 1.00 | 7/7 (exact at the switch) | 1.00 / 1.00 / 0.92 |
| 7 bits | learned slots | top 3 / top 2 | 1 | 0.71 / 0.01 | 0/1 (0.88 after 2,000 steps) | · |
| 10 bits | shared subtractor | top 2 / top 1 | 4 | 0.02 / 1.00 in 3 seeds; 0.00 in 1 | 3/4 (800) | 1.00 / 0.99–1.00 / 0.99–1.00 for the 3 that learned; 0.02 / 0.02 / 0.01 for the fourth |
| 10 bits | shared subtractor | top 3 / top 2 | 4 | 0.62 / 1.00 | 4/4 (100) | · |

- **At 10 bits a too-coarse window can fail to start.** With top 2 / top 1, one seed in four never started:
  - the estimator alone is nearly blind there (0.02), so the table's only lesson is the right slot's 1-in-10 share;
  - after 3,000 steps the table is right on 0.3% of examples given the true digits;
  - 2,000 steps of the own borrow then leave it at 0.001.

  With microlab's window (top 3 / top 2) all four 10-bit seeds start, and are exact 100 steps after the switch.
- **The estimator only bootstraps.** It stays at its window's ceiling, while the subtractor under it becomes exact on
  the true digits. Handing the choice to the subtractor's own borrow then makes the whole datapath exact within a few
  hundred steps.
- **The same route with learned slots fails** (microlab's cell, a learned label per slot; 6 bits, 0 of 4 seeds).
  - The estimator phase ends in a lookup arrangement, with slot maps [0, 0, 3, 1, 3, 1, 1, 0, 4, 4] and
    [0, 2, 2, 9, 2, 3, 4, 1, 1, 0]. It is right on 3–7% of examples even given the true digits.
  - Switching to the borrow leaves it at 0.74–0.77 after 2,000 more steps.
  - Fixing what the slots mean is what lets the estimator teach the table.
- **A coarse window is faster.** At 6 bits, in one 800-step run, the top-3/top-2 window (which sees every digit) left
  the subtractor right on 62% of examples, against 100% with top 2 / top 1. With 3,000 steps both got there.
- **A table is right only where its training data goes.**
  - Trained at 6 bits, 39 of its 200 rows are never visited, and they come out wrong. So the 6-bit tables work at
    7 bits only partly (0.71–0.77) and at 10 bits hardly at all (0.07–0.30).
  - Trained at 7 bits the table is nearly complete: the same tables are 1.00 at 6 bits and 0.92–0.99 at 10 bits.

*With a trusted adder.* The shared subtractor learns first from (a − b) + b = a on random numbers: 500 steps, no
remainders. Then answers:
- **3/3 at 6, 7 and 10 bits**, exact after 0–100 answer steps.
- With learned slots instead (10,000 rows to pin), 6 bits then 7 bits:
  - own borrow: 2/3, then 0/2;
  - estimator: 3/3 (in 2,200 steps), then 0/2 (0.68);
  - checked estimator: 1/3, then 0/2 (0.93).

  The construction is what lets the label-free lesson pin the table.

## 6. Predictions for microlab at 7–10 bits
These follow from the tiny core and its base-10 runs, so the running microlab experiments can be checked against them.
- **Hard EM (`ext_quotient_em`, `ext_quotient_em_set`) will not reach the rule from a fresh start.**
  - In the tiny core it is 0/6 at every size, with an exhaustive E-step or microlab's (own path plus one-step
    changes), ending in permuted and merged slot maps.
  - Expect at most a faster lift-off at 6 bits that ends in a lookup arrangement, with held-out moduli near chance,
    and no better than the soft baseline at 7 bits.
  - It does finish the job once the cell subtracts: 0.97–0.98 from an exact cell. So EM after a cell-first phase is
    reasonable; EM as the starter is not.
- **Clock scoring (`ext_quotient_clock`) will collapse to copying.**
  - The slots will not learn the multiples 0..9. Slots that settle on a consistent multiple will settle on 0.
  - Exact will sit near the share of examples whose quotient is 0. With v < N² that is about 1/N, roughly 2% at
    6 bits and 1% at 7, while exact mod N stays high.
  - The same holds with given digits plus noise.
  - The label-free estimator lesson cannot help: when every trial copies, every trial is the same.
- **The prefix start alone (`ext_quotient_prefix`) will not fix 7 bits.** Every tiny-core run already used it, and
  learned slots still reach only 0.07 at 7 bits.
- **The fix that should survive at 7–10 bits:**
  1. Replace longdiv's learned slot embedding with one shared subtractor cell (r_j, m_j, borrow) → (digit, borrow),
     fed the exact digits m of q·N for each slot q.
  2. Choose with longdiv_est's estimator, softly, starting undecided. Its default window (top 3 of 10R + v_t, top 2 of
     N) worked at 6, 7 and 10 bits here. Top 2 / top 1 is faster at 6 bits, but at 10 bits it is too coarse to start
     reliably.
  3. Start from the dividend's top digits and add one quotient digit per stage.
  4. Then switch to longdiv's own borrow choice.
  5. Train at the widest size wanted, or mix widths, so every digit combination of the subtractor is visited.
- **Expected outcome:** exact from answers at 6, 7 and 10 bits within a few thousand steps
  (tiny core at base 10: 11/11, 11/11 and 7/8 seeds),
  with the trained tables working unchanged at other widths. Keep microlab's window at 10 bits: with a coarser one,
  one tiny-core seed in four never started. If a trusted adder is allowed, teaching the subtractor
  from (a − b) + b = a first gets there faster and transfers to every width.
- **Checked so far.** [FINDINGS_quotient_clock.md](../microlab/FINDINGS_quotient_clock.md) was written after the
  prediction above.
  - It agrees on the verdict: no fix at 6, 7 or 10 bits, and exact at 1/N (0.02 at 6 bits, 0.01 at 7).
  - It names the same degeneracy: any cell whose slots subtract fixed multiples of N scores zero, including the copy
    cell.
  - One difference: microlab's MLP cell never reaches the copy cell. Its clock loss stays at 1.00 and exact mod N
    stays at chance, because the clock's gradient through one shared cell is noise. Every tiny-core clock run did
    collapse to copying (exact mod N 0.9–1.0). So the tiny core understates how little the clock teaches.
- **Checked: the fix in microlab's MLP datapath.** [FINDINGS_quotient_shared.md](../microlab/FINDINGS_quotient_shared.md)
  ran the follow-up, 168 runs on Modal.
  - The mechanism holds: with fixed slot meanings every run that started ended the estimator phase with the identity
    slot map and a subtractor right on the true path. With learned slot labels, 0/9.
  - The route exactly as written: 10 of 24 runs. There are two failures the tables never showed:
    - microlab's MLP estimator commits to one slot within about 100 steps, before the subtractor learns anything;
    - at 7 and 10 bits, switching to the own borrow can wreck a subtractor that was already right.
  - Holding the choice uniform for the first 400 steps (making "starts undecided" explicit):
    - 6 and 10 bits: 8/8 each, also on held-out moduli;
    - trained at 10 bits, the same weights are exact from 6 to 20 bits;
    - 7 bits: 1/8. There the MLP's borrow comes out as an analog code that the own-borrow rule misreads.
  - Training at 10 bits, as step 5 above advises, gives exact 7-bit results with the same weights: 16/16.
- **What would contradict this picture:**
  - microlab's longdiv with learned slot embeddings learning the 7-bit rule from answers through EM, clock scoring or
    the prefix start;
  - the shared-subtractor route failing at 7 bits with an MLP cell.

## 7. Caveats
- **Tables, not MLPs.** microlab's cell is an MLP over digit embeddings, so its entries share structure, and it treats
  a mixture of digits differently from a multilinear table. The symmetry argument (§3.1–3.4) holds for any cell whose
  slot input is a learned label. The numbers do not carry over one to one.
- **The tiny estimator sees every digit** below base 10, so it can always store a lookup arrangement. That makes the
  estimator's failures there a worst case. At base 10 the window is microlab's (3/2) or coarser (2/1).
- **Hard EM was tested as implemented in `fixes.py`.**
  - E-step: exhaustive, or microlab-style. M-step: Adam. Ties are broken toward the model's own path.
  - Soft EM, annealed EM and restarts were not tested.
  - "Try-and-keep" by restarts is only as good as the per-seed success rate, which is 0/6 for every hard-EM cell here.
- **The label-free checks need a trusted carrying adder.**
  - Without it, a − a = 0, a − 0 = a and (a − b) − c = (a − c) − b on random numbers do not pin the table (`identities.py`).
    Training settles at loss 0.12 on a wrong table: 11.5% of the shared subtractor's rows are right, and 4.5% of
    7-bit examples are right even given the true digits.
  - Subtraction without borrows (digit by digit, mod 10) satisfies all three identities exactly, so no amount of
    training on them alone can single out the real one. `b10.label_free` adds the exact adder, through
    (a − b) + b = a, and that does pin it.
  - With a learned adder, (a − b) + b = a also holds for mutually inverse wrong pairs unless anchored; untested.
- **The shared subtractor needs the exact multiples s·N.** That is built-in arithmetic (multiplying N by one known
  digit), like the long-division wiring itself. If those multiples were learned too, their slot labels could again be
  interchangeable.
- **A table is right only where the data goes.** The answers-only route leaves wrong every subtractor row that its
  training data never visits. At 6 bits that is 39 of 200 rows, so tables trained there transfer to 7 bits only partly
  and to 10 bits not at all (§5). An MLP cell might fill such gaps by sharing structure, or might not.
- **microlab's moduli use few leading digits.** Odd N in [2^(b−1), 2^b) has a narrow top-digit range. That is what
  makes construction plus staging with the own borrow fail at microlab's sizes while it works on every w-digit
  modulus (§5).
- **Small budgets**:
  - Tiny core: 6 seeds, 4,000 steps, one learning rate (0.05), init scale 0.1.
  - Base 10: 2–4 seeds per row, 3,000–6,000 steps.
  - The ε and λ measurements use plain gradient descent at lr 1 from an exactly symmetric start.

## 8. Reproduce
Run from `cpu_first/` with the sotaku venv's Python (it has torch). Only seconds-long checks run locally. The sweeps
and everything at base 10 run on Modal (app and Volume `tinycore`; suites in `suites.py`).

```bash
~/github/sotaku/venv/bin/python -m tinycore.exact grad0
```

```bash
~/github/sotaku/venv/bin/python -m tinycore.saddle eps --B 2 --w 3 --k 1 --steps 4000 --lr 1.0 --seeds 3 --eps 1e-1,1e-2,1e-3,1e-4,1e-6,1e-8,1e-10
```

```bash
~/github/sotaku/venv/bin/python -m tinycore.escape "2,3,1;2,4,1;2,5,1;3,2,1;4,2,1" 0,1000
```

```bash
~/github/sotaku/venv/bin/python -m tinycore.paths
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tinycore/modal_tinycore.py --suite b10f
```

```bash
~/github/sotaku/venv/bin/python -m modal run tinycore/modal_tinycore.py --suite b10f --fetch
```

Suites on Modal:
- `fixes_rest`: the fix comparison's last sizes;
- `grad0` and `lesson`: §3.1 and §3.6;
- `b10`: microlab's analogs and the construction variants;
- `b10b`: label-free cell, wide-moduli controls;
- `b10c`: top digit scored; stopped early;
- `b10d`: estimator 3/2 then the own borrow;
- `b10e`: width transfer;
- `b10f`: estimator 2/1 then the own borrow, with transfer.

The phase map, fix and construction sweeps (`phase.py`, `fixsweep.py`, `constructs.py`) ran locally before the switch
to Modal.

Tables:
- `python -m tinycore.summarize runs/phase.jsonl`
- `python -m tinycore.fixtable runs/fixes.jsonl runs/fixes_rest.jsonl runs/constructs.jsonl`
- `python -m tinycore.b10table runs/b10.jsonl runs/b10b.jsonl runs/b10c.jsonl`
- `python -m tinycore.switchtable runs/b10d.jsonl runs/b10e.jsonl runs/b10f.jsonl`

Code:
- `core.py`: datapath, cells, choices, training;
- `exact.py`: gradients, Hessians, symmetric starts;
- `saddle.py`, `escape.py`, `paths.py`, `lesson.py`: the exact analyses;
- `fixes.py`: every fix;
- `b10.py`: microlab's sizes;
- `suites.py` and `modal_tinycore.py`: Modal.
