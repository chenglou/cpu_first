# Relaxing the rules: which single relaxation makes x^(2^T) mod N learnable?

**Question.** REPORT.md found that labels can check the arithmetic but cannot teach it: one wrong digit fact inside
a squaring scrambles the residue, so gradient descent gets no warmer/colder signal. If one rule is dropped, does a
mostly generic network trained from scratch learn y = x^(2^T) mod N exactly, and loop to T = 64?

**Short answer.**
- **Show your work is the one relaxation that makes a generic network learn the step, but only when every trace
  token is a local operation.**
  - With the quotient digit found by comparing against the table of multiples (formats v1–v3), a 4.75M-parameter GPT
    plateaus at 97–98% exact at every size from 8 to 24 bits. This holds for 3 seeds and a 3× bigger model; 5× more
    steps creeps up to 99.2%.
  - Every residual error is a comparison between E and k·N that share their leading 3–5 digits. Those near-ties are
    rare in the data and the model never learns to look that deep.
  - Writing the nine trial subtractions instead (v4), each ending in its borrow digit, removes the comparison. The
    same GPT is then **100% exact on 20k–50k fresh cases at every size from 8 to 24 bits**, on training-pool and
    unseen moduli alike.
  - Looped with hard snapping, it **certifies T=64 at 8, 12, 16, 20 and 24 bits** on both ladders, every rung
    768/768. 12, 16 and 24 bits pass in one 8k-step run. 8 and 20 bits need more steps: after 8k, 8 bits has 4
    misses in 20k and 20 bits has one ladder row short. A 24-bit run trains in 24 minutes on one H100.
  - Trained on H1's own rows with their traces, the model is 99.998% exact (1 miss in 50k). It certifies H1's
    depth_t profile only to T=4: one row of 768 fails from T=8 on, the same count as before 8k more steps.
  - The OOD profile fails, 32% at T=1 down to 21% at T=64, because H1 never shows an 8-digit number in training.
  - It does not survive the real rules: the trace is the algorithm, handed over as labels.
- **Periodic units tied to N learn the step at 24 bits in under a second and certify T=64 on both real H1 profiles.**
  - They work because the unit is the modular reduction and the bilinear layer is the multiplier.
  - They need either fresh small-s data with a curriculum, or the pre-reduction value s² as a label. With H1's own
    1,000 T=1 rows they fail.
- **Per-step labels and unlimited data alone stop at 6–8 bits for a generic looped transformer.**
  - Unlimited data certifies T=64 at 6 and 8 bits.
  - Per-step labels on H1-shaped data certify at 6 bits, and at 8 bits miss one row at T=2.
  - At 10 bits, unlimited data reaches 24% after 40k steps (20M examples), creeping up. With H1's real
    13–20-bit per-step data the step is 0.05% exact.
  - Supervising s² as well teaches the multiplication (99.9% exact) but not the reduction (8%, the same as without).
- **The pattern.** The multiplication is learnable everywhere. Finding the quotient, i.e. the mod-N reduction, is
  what a generic network cannot do by itself. The relaxations that work either spell the division out as local steps
  or build the reduction into the unit.

## The table
| relaxation | architecture (all trained from scratch) | largest modulus with 100% exact single step | loop certifies T=64? | wall time | survives the real rules? |
|---|---|---|---|---|---|
| 1. per-step labels (every intermediate squaring), H1-shaped data | generic looped transformer, 0.8M params | 6 bits on the states the loop visits; never on uniform s (81–82% at 6 and 8 bits: the data's trajectories contain no non-units and few non-squares) | yes at 6 bits (both ladders); 8 bits: one ladder certifies, the other misses 1 row of 768 at T=2 (24k steps) | 3 min (6 bits), 9 min (8 bits, 24k steps) on an H100 | no: H1 gives final answers only |
| 1. per-step labels, H1's real data (13/16/20-bit) | same, digits padded to 8 | none: 0.05% | no | 8 min (H100) | no |
| 2. show your work, quotient by comparison (v1–v3) | GPT 4.75M, 6 layers, generates its own trace | none: 97–98% at 8, 12, 16 and 24 bits (24: 98.5% / 97.9% (20k steps)); 98.6–99.2% after 40k steps at 12 bits | no | 2–5 min per run (H100) | no: the trace is the algorithm |
| 2. show your work, trial subtractions (v4), H1-shaped data | same GPT | **24 bits**: 100% at every size from 8 to 24 bits on 20k–50k cases, pool and unseen moduli (8 and 20 bits after more steps) | **yes at 8, 12, 16, 20 and 24 bits** (both ladders, every rung 768/768; 8 and 20 bits after more steps) | training 6.4 / 9.3 / 24 min at 12 / 16 / 24 bits; with both ladders 19 / 32 / 112 min (one H100) | no: the trace is the algorithm |
| 2. show your work (v4) on H1's own 101,999 rows + their traces | same GPT, digits padded to 8 | none: 99.985% (3 misses in 20k); 99.998% (1 miss in 50k) after 8k more steps | depth_t to T=4 only (1 row of 768 fails from T=8, before and after the extra steps); depth_ood_n_t 32% → 21%: no 8-digit number in training | 22 + 41 min training, 102 + 134 min with both profiles | no |
| 2. + normal curriculum (first lines first) | same, v1 | none: 95–96% at 12 bits (v1 without: 97%) | no | 2.7 min | no |
| 2. + reverse curriculum (most lines given, help removed) | same, v1 | none: 25–32% (88% in a third run) at 12 bits | no | 2.7 min | no |
| 3. periodic units tied to N, fp64, snapping; fresh data + small-s curriculum | 6 weights on (1, s, N, s², sN, N²), or 4 bilinear units, into cos/sin(2πv/N) | 24 bits (fp64 holds the exact weights to 26) | yes: both real H1 profiles, 6/6 seeds | < 1 s train, 0.2 s certify (CPU) | no: the reduction and the multiplier are designed in, and H1 has no small-s rows |
| 3. same, H1's own 1,000 T=1 rows only | same | none | no | < 1 s | — (this is the real-rules version) |
| 3. + s² labels (value before the remainder), no curriculum | same, also with digit inputs | 24 bits | yes: both H1 profiles, 5/5 | 0.1–9 s | no: s² is an extra label |
| 4. more data: unlimited fresh (N, s) | generic looped transformer, 0.8M params | 8 bits (10 bits: 24% after 40k steps) | yes at 6 and 8 bits (own ladder) | 14 / 114 min (laptop GPU, shared); 17 min for 40k steps at 10 bits (H100) | no: the training data is fixed |
| 4. + an s² head | same + one linear head | none: 10 bits stays at 8% (s² itself 99.9%) | no | stopped at 5k steps | no |

## Setup
- **Moduli.** Small-modulus stand-ins for H1: a b-bit modulus is any odd N in [2^(b−1), 2^b). Balanced Blum
  semiprimes, H1's family, barely exist below 13 bits (none at 6 or 8 bits, 2 at 10 and 12). The arithmetic is the
  same for any N.
- **H1-shaped finite data at b bits** (`relax_common.h1like`): a pool of up to 430 moduli, 1,000 rows at T=1, 1,000 at
  T=2, and 33,333 each at T=4/8/16, x a unit. Every intermediate state along every row is known, which gives 936k
  single-step examples. This is the data for relaxations 1 and 2. At 6–10 bits the pool is every odd modulus; from 12
  bits, held-out moduli also exist ("unseen").
- **Exactness.** Single-step exact match on 20k–200k fresh cases. Then the learned step is looped with hard snapping
  on a T-ladder: T = 1, 2, 4, ..., 64, with 768 fresh rows per rung (H1's rung size). "Certifies" means every rung is
  100%. Where the size allows, the real H1 profiles too (`h1.certify`, depth_t and depth_ood_n_t).
- **Compute.** The first runs were on the laptop: MPS fp32 for transformers, CPU fp64 for the periodic nets. They
  shared the GPU with up to 6 other jobs and ran the laptop out of memory, so everything from the trace runs on moved
  to Modal H100s (`modal_relaxed.py`). There a 12-bit trace run takes 3 minutes instead of 2.4 hours. A 24-bit v4
  run (1,366-token traces) trains in 24 minutes, and its two T=64 ladders take another 90.

## 1. Per-step labels — `relax_step.py --data h1like / h1`
**Model.** A generic looped transformer, 0.8M parameters:
- Tokens: the digits of N and s, plus 2W blank scratch slots.
- One pre-LN block applied 12 times with input injection.
- Rotary positions on each digit's place value. Cross-entropy on y's digits.
- Argmax snapping between steps.

**Results.**
- **6 bits.** Trained on the 936k step pairs of an H1-shaped set, the step is 81.6% exact on uniform s. Both T=64
  ladders certify, because they only visit units and the data covers those.
- **8 bits.**
  - 8k steps: 82% on uniform s and 99.6–100% per ladder rung, which certifies nothing.
  - 24k steps: 82.5% on uniform s. One ladder certifies T=64; the other misses 1 row of 768 at T=2.
- **Why uniform s stays at ~82%.** Single-step accuracy on uniform s is capped by the data: H1-shaped trajectories
  never visit a non-unit and rarely a non-square.
- The same model on H1's real per-step data (13/16/20-bit moduli, digits padded to 8) gets 0.05% exact.
- Unlimited fresh data (relaxation 4, below) dominates this data at every size and stops at 8 bits, so per-step labels
  cannot do better at H1's sizes.

## 2. Show your work — `relax_trace.py`
A GPT (pre-LN, 6 layers, d=256, 4.75M parameters) is supervised on the full worked trace of one squaring:
- Rotary positions over the sequence, plus a learned "place within its number" embedding.
- At test time it generates the trace greedily with a KV cache, and the last remainder line is the answer.
- It is looped T times; the parsed answer is the next s.
- Data: per-step traces along the H1-shaped set at each size, 8k steps of batch 64, AdamW.

The trace of s² mod N, every number LSB-first:

```
[N] n  [S] s
[P] s·s_j  [A] running sum            for each digit j of s     (long multiplication; the last A line is s²)
[M] k  k·N                            for k = 1..9              (the divisor's multiples table)
then for each long-division step:
  v1/v2: [E] 10R + next digit  [Q] q  ([C] q·N, v2)  [R] E − q·N
  v3:    v2 with [G] g_1..g_9 (g_k = 1 if k·N <= E) written before q
  v4:    [E] e  then [D] k (E − k·N mod 10^(W+1)) borrow, for k = 1..9, then [Q] q  [R] remainder
```

- **v1**: the chosen multiple is found in the table by position. Early in training most errors are in the
  subtraction lines; by the end they are in the quotient digit.
- **v2** gives every table line its multiplier and copies the chosen multiple next to the subtraction. Subtraction
  errors disappear: at 12 bits, step 1k, 71% exact vs v1's 4%.
- **v3** writes the nine comparisons out explicitly.
- **v4** never compares at all: k·N <= E is exactly "the subtraction E − k·N does not borrow out", and subtraction is
  the one thing the model already does without error.

**Exact single step (pool moduli / unseen moduli), 8k steps:**

| format | 8 bits | 12 bits | 16 bits | 20 and 24 bits | where the errors are |
|---|---|---|---|---|---|
| v1 | — | 97.1% / 96.5%, 97.1% / 96.8% (2 seeds) | — | — | quotient digit |
| v2 | 96.9% | 97.8% / 97.3%, 97.7% / 97.4% (2 seeds) | 98.1% / 97.6% | 24 bits: 98.5% / 97.9% (20k steps) | quotient digit |
| v2, d=384, 8 layers (14M params) | — | 98.0% / 97.2% | — | — | quotient digit |
| v2, 40k steps, batch 128 | — | 99.2% / 98.6% (50k-case test) | — | — | quotient digit |
| v3 | — | 97.7% / 96.4% | 98.2% / 97.8% | — | comparison bits |
| v4 (trial subtractions) | 99.98% (4 misses in 20k, all partial-product lines); **100% of 50k after 24k steps × batch 128, ladder certifies T=64** | **100% / 100%**, both ladders certify; again 100% / 100% of 50k after 24k steps | **100% / 100%**, both ladders certify | 20 bits: 99.995% / 100% (1 miss in 20k), unseen ladder 7 of 7 rungs ≥ 99.87% (1 of 768 rows fails at T=64); after 8k more steps (batch 128, lr 3e-4) **100% / 100% of 50k, both ladders certify**; **24 bits: 100% / 100%, both ladders certify** | none at 12, 16 and 24 bits; at 8 and 20, rare partial-product / running-sum digits |

**Why v1–v3 stall.** The errors are the rare near-ties. Over 20k traces of the v3 12-bit model, all 450 failures are
comparison bits. By the number of leading digits E and k·N share in the comparison that failed (`relax_trace_errors.py`):

| leading digits shared | share of all comparisons | failure rate |
|---|---|---|
| 1 | 73% | ~0 |
| 2 | 14% | 0.5% |
| 3 | 1.7% | 12% |
| 4 | 0.15% | 33% |
| 5 | rare | ~100% |

Three seeds, a 3× bigger model and the explicit comparison line all land at 97–98%. 5× more steps creeps to
99.2% / 98.6%. The model learns to decide from the top two or three digits, and the ties that need deeper digits are
too rare to teach it.

**Curricula** (v1, 12 bits, 8k steps; curricula are loss masks over trace lines, all teacher-forced):
- **None:** 97.1% / 96.5% and 97.1% / 96.8% (2 seeds).
- **Normal** (supervise the first lines first, add later lines over the first 60%): 95.7% / 95.0% and
  94.9% / 94.3%, plus 96.4% / 95.7% from the laptop run. Slightly worse than none.
- **Reverse** (start with all but the last 2–3 lines given in context; shrink the given prefix to zero by 60%):
  32.1% / 27.2% and 24.8% / 21.0%, plus 87.9% / 83.5% from the laptop run.
  - Much worse and much more variable.
  - Learning the division lines first, before the lines they read from, is slow: training loss 0.5 → 0.08 over the
    curriculum phase, versus 0.01 → 0.002 for the normal curriculum.
  - Its failures stay in the subtraction lines.
- Neither curriculum fixes the comparison.

## 3. Periodic units tied to N (fp64, integer snapping) — `relax_periodic.py`, `relax_periodic_cert.py`, `relax_square.py`
The step net reads s and N as numbers. Its pre-activation v is a learned quadratic of (1, s, N): either 6 monomial
weights, or 4 bilinear (GLU-style) units with 27 weights. The output unit is (cos, sin)(2πv/N). Snapping decodes the
phase to the nearest residue and feeds it back as the next s. The exact solutions are a lattice: v = s² plus integer
multiples of N, sN and N².
- Designed: the period N in the unit, the phase decoder, rounding, and s given as a number.
- Learned: the quadratic, i.e. which products of the inputs feed the phase, with which weights.

**How exact the weights must be.** The decoded residue is off by δ·s² when the s² weight is off by δ, and s² ~ 2^(2b).

| bits | exact weights, fp64 | exact weights, fp32 | largest rel. error on the s² weight (log2) | local-descent basin, rel. (log2) |
|---|---|---|---|---|
| 6 | 100% | 100% | −13 | −4 |
| 8 | 100% | 100% | −17 | −6.5 |
| 12 | 100% | 100% | −25 | −9.5 |
| 14 | 100% | 60% | −29 | −13 |
| 16 | 100% | 24% | −33 | −14.5 |
| 20 | 100% | 2.5% | −41 | −21 |
| 24 | 100% | 0.17% | −49 | −24.5 |
| 26 | 100% | 0.01% | −53 | −27.5 |
| 28 | 72% | 0.01% | — | — |

- The tolerance is 2^−(2b+1) relative: 6e-8 at 12 bits, 2e-15 at 24 bits.
- fp32 cannot hold even the exact solution past 12 bits: s² overflows the 24-bit mantissa. fp64 holds to 26 bits
  and breaks at 28 (s² > 2^53).
- Local optimization finds its way back from about 2^−b relative error. That basin is ~2^−b wide around a target of
  size 1, so from a random start the target is a needle. With no curriculum, 0/5 seeds reach it at every size.

**What makes it learnable: a curriculum over how often s² wraps, plus a scale-free optimizer.**
- Curriculum: start with s < N^(1/4), where s² < N never wraps and v = s² is plain regression; grow to s < N over 48
  stages.
- Optimizer: Levenberg-Marquardt on the circular loss. The current model says how many times each example wraps
  (wrapped residual), the Jacobian is closed-form, and columns are normalized.
- Result: 5/5 seeds exact at 6, 8, 12, 16, 20 and 24 bits, for both the 6-weight and the 27-weight form, in 0.1–0.2 s
  (monomial) and 2–6 s (bilinear) per seed.
- The 24-bit nets loop with snapping to **T = 64 on both real H1 profiles** (6/6 seeds, every rung 768/768). Each one
  trains in under 0.5 s and certifies in 0.2 s. The learned weights are w(s²) = 1.0 exactly and the other five at
  1e-11 to 1e-24.

**What breaks it.**
| change | result |
|---|---|
| no curriculum (s uniform from the start) | 0/5 at every size, 6–24 bits |
| curriculum over modulus size instead (4 → b bits) | 0/5 at every size: 4-bit moduli already wrap s² up to 15 times |
| Adam instead of LM (lr 1e-2, small-s curriculum) | 5/5 at 6 bits, then 0/5 from 8 bits (monomial) or 12 bits (bilinear): each step moves the N² weight by ~lr, which turns the phase by ~lr·N |
| digits of s as inputs instead of its value (place values must be learned) | 0/3 at every size |
| only H1's 1,000 T=1 rows, easiest first by x²/N | 0/3 (monomial and bilinear; 24, 48 or 200 stages) |
| no snapping between steps (24 bits) | fp64 drift grows ~2N-fold per squaring: 100% at T=2, 50% at T=4, 0% at T=64 |

- **Why digits fail.** A digit at place 10^i first shows up when s ≥ 10^i. For the top digits that only happens once
  s² already wraps past N. The wrap count can't be inferred before that digit's place value is known, and the place
  value can't be learned without the wrap count.
- **Why H1's rows fail.** Only 5 of the 1,000 T=1 rows have x² < N, and the 6th already wraps twice. That's too
  little to pin down even 6 weights before wrap counts have to be guessed.

**Supervising the value before the remainder (s², up to 14 digits)** — `relax_square.py`. Regress v on s² directly,
with no wrap ambiguity, and let the periodic unit and snapping do the reduction. No curriculum, small random init,
Levenberg-Marquardt.

| model | fresh data, 6–24 bits | H1's own 1,000 T=1 rows + x² labels |
|---|---|---|
| 6 weights on (1, s, N, s², sN, N²) | 5/5 exact at every size; 24-bit certifies H1 5/5 | 5/5 certify both H1 profiles (0.1 s) |
| 4 bilinear units on (1, s, N) | 5/5 at 6–16 bits, 3/5 at 20, 4/5 at 24 (certify 4/5) | 5/5 certify both profiles (0.2 s) |
| 4 bilinear units on (1, 8 digits of s, N): place values learned | **5/5 exact at every size; 24-bit certifies H1 5/5** | 0/5: certifies depth_t only to T=4, 8% single-step at 24 bits |

- So yes: once the pre-reduction value is a label, the gradient is usable everywhere. The needle came entirely from
  the wraparound.
- A first-order optimizer also gets there. Adam on the same regression is then finished by an fp64 LM polish: 3/3 at
  6, 12 and 24 bits (digits: 2/3 at 24), with H1 certified.
- Supervising s² also fixes the digit-input failure above: the place values 10^i become plain regression.
- The one failure is the width alias from REPORT.md. H1's training x never has more than 6 digits, so the weights for
  the 10^6 and 10^7 places are never constrained. They then fail on 22- and 24-bit moduli.

## 4. More data: unlimited fresh (N, s) — `relax_step.py --data fresh`
Same generic looped transformer as in relaxation 1, with fresh examples every step (batch 512, AdamW, cosine
schedule).

| bits | exact single step | T-ladder (768 rows/rung) | notes |
|---|---|---|---|
| 6 | 100% of 100k (from step 1k) | certifies T=64 | 2.5k steps, 14 min (laptop) |
| 8 | 100% of 100k (99.96% at step 8k, 100% at 11k) | certifies T=64 | 12k steps, 114 min (laptop GPU shared with up to 6 jobs) |
| 10 | 24.3% of 100k after 40k steps (H100; 4% at 2k, 12% at 10k, 17% at 20k, 22% at 30k) | 1.2% at T=64 | 17 min on an H100; still creeping up |
| 10, plus a head that predicts s²'s digits | s² 99.9% exact, y 8.0% at step 5k (answer-only: 8.1%) | — | stopped |

The extra s² head is the generic-track answer to the user's question. The network learns the 7-digit square almost
perfectly within 1k steps, yet y does not move: the answer-only run and the s²-supervised run follow the same curve.
Supervising the value before the remainder does make the multiplication learnable, but the mod-N reduction is still
the wall.

## What survives the real rules, and what does not
The real rules give 1,000 T=1 rows, 1,000 T=2 rows and pseudorandom-looking T=4/8/16 rows, labelled by final answers
only, and a fixed test-time interface.
- **Does not survive:**
  - Per-step labels. H1's T≥2 rows carry final answers only.
  - Show-your-work traces. They are the long-multiplication and long-division algorithm itself, written as
    supervision.
  - Unlimited fresh data, and s² labels.
  - The periodic net's training: on H1's own rows it fails for lack of small-s examples.
- **Would survive, as findings rather than recipes:**
  - The precision analysis. A residue representation needs weights exact to 2^−(2b+1); fp32 cannot represent even
    the exact solution past 12 bits, fp64 to 26.
  - Exactness needs snapping between steps: without it, fp64 error grows ~2N-fold per squaring.
  - The generic multiplier is learnable (s² 99.9% exact at 10 bits from answers on s²). The quotient is the part no
    generic model found on its own.
  - The v1–v3 failure mode: comparisons between near-tied numbers are rare in natural data, and a model that only
    reads the top digits looks 97% right.
  - The v4 lesson: a generic network learns exact arithmetic when every step it must take is local (a digit, a
    carry, a borrow). One global decision (compare two numbers) is enough to cap it at 97–99%. A comparison written
    as a subtraction's borrow-out is local, and goes to 100% at 24 bits.
  - The width alias, again. The v4 model trained on H1's own rows never sees an 8-digit number, and H1's OOD profile
    (22/24-bit moduli) falls to 21–32%. The same model trained on 24-bit data certifies both of its own ladders, so
    the gap is data, not capacity.
- **Designed versus learned, per the user's note that exact arithmetic is allowed:**
  - The periodic net designs the reduction (period N, phase decoder, rounding) and learns the polynomial.
  - The trace models design nothing inside the network; the algorithm lives in the trace format they are trained to
    write. v4's format is the designed part: long multiplication, a multiples table, and trial subtraction.
  - The looped transformer designs nothing, and learns the step only up to 8 bits.

## Files
| file | what |
|---|---|
| `relax_common.py` | moduli, H1-shaped data with every intermediate state, fresh single steps, the T-ladder |
| `relax_step.py` | relaxations 1 and 4: the generic looped transformer (`--data fresh/h1like/h1`, `--aux_square`) |
| `relax_trace.py` | relaxation 2: trace formats v1–v4, the GPT with KV-cache decoding, curricula, `--resume` |
| `relax_trace_errors.py` | where a trace model fails, and how close the failing comparisons were |
| `relax_periodic.py` | relaxation 3: precision table, training (LM/L-BFGS/Adam; curricula), H1-rows mode |
| `relax_periodic_cert.py` | 24-bit periodic nets → own ladder + both real H1 profiles |
| `relax_square.py` | s² supervision for the periodic nets (fresh data or H1's rows) |
| `relax_table.py` | prints every result in `runs/` and `runs/modal/` |
| `modal_relaxed.py` | runs any of the above on a Modal H100; results on the `relaxed-runs` Volume, copies in `runs/modal/` |
