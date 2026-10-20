# Exact blocks, learned control: how little is left to learn for x² mod N?

**Question.** Hand the model exact arithmetic as fixed, non-learned blocks (multiply, remainder, add, compare, …).
What is the least it must still learn to compute one step, x² mod N? Can it learn that from final answers alone?

**Answer.** It must learn the wiring of two instructions: "multiply x by x, then take the remainder by N". That is one
choice among 2^11–2^17 programs, depending on how many blocks are on offer.
- Speculation learns it from **one** labelled row. Enumerate every program and keep those that reproduce the answers.
  A single T=1 row leaves exactly the right program (6 of 6 randomly chosen rows), and looped T times it certifies
  T=64 on both H1 profiles.
- Soft selection that hardens does **not** learn it: 0 of 20 seeds, even at this size. A mod N after the choice
  makes the loss flat until the wrong choices hold less than about 1/N² of the weight.
- fp64 fixes the arithmetic, not the learning. Learned digit→value place weights must start within about 0.01–0.03
  of the true place values, in absolute terms, on every place. Snapping the encoder output to integers does not widen
  that; it makes the inside exact and the outside flat.

## Plain-language summary
- **What is left to learn is tiny.** With exact multiply and remainder blocks, x² mod N is a 2-instruction program.
  - With a 2-block kit {mul, mod}: 2,280 possible programs, 136 of them non-redundant.
  - With 5 blocks: 15,750. With 13 blocks: 92,565 (16.5 bits).
  - Also learning which way the prompt's digit strings run (N is written LSB-first, x MSB-first) adds 1.8 bits.
- **Answers can pick it, but only by trying candidates.** The labels act as test vectors: a wrong program almost never
  reproduces a residue.
  - In every x² mod N search (2 or 4 instructions, 2 to 13 blocks), one of the 1,000 single-step rows alone left
    exactly one program: the right one. This held for 6 of 6 randomly chosen rows.
  - Without a remainder block, the 4-instruction x² − ⌊x²/N⌋·N was found among 2^34 programs in 8 s.
  - Exhaustive search runs out at about 6–7 instructions. 6 instructions (x⁸ mod N, 2^46.7 programs) took 83 s on a
    32-core box. Each extra instruction multiplies the space by about 2^8.5 and the time by 30–100.
- **Gradients can't pick it.** I tried three soft learners:
  - Mixing values, DARTS-style: 0/20 seeds. On tiny 4- and 6-bit moduli, where the trap is widest: 0/6 each.
  - Straight-through Gumbel, which runs one hard program forward and the soft relaxation backward: 0/4.
  - REINFORCE: 1/5, worse than sampling programs blindly.

  The reason is measurable. Walk the mix in a straight line from uniform to the right program. The loss stays at
  chance until the wrong choices hold less than about 1/N² of the weight: 1e-9 at 16 bits, 1e-14 at 24 bits. Without
  the final mod, the same walk slopes downhill from the start.

  The one soft method that works is a mixture over complete programs, which hardens in 50 steps. But it has to run
  every program, so it is speculation with soft weights.
- **A loop removes the size limit.** Suppose the only reduction block is one conditional subtract (no whole-number
  remainder). Then a 3-instruction body applied once per bit of x is Blakley's binary modular multiplication.
  - Speculation found it from 1 label among 2^22 bodies in 5.5 s, and it certifies T=64 on both H1 profiles.
  - The same body needs 6 instructions in base 4 (2^47) and 14 in base 10 (2^123).
  - With small blocks, binary makes the learned control smallest by far.
- **fp64 is necessary but not sufficient.**
  - fp32 cannot even run the true weights exactly beyond 12-bit moduli. fp64 can, up to 26-bit moduli.
  - From a random start, learned place weights never become exact: 0/5 runs in each of 4 settings (with or without
    snapping, with or without a small-to-large curriculum).
  - Parameters that are themselves discrete do work. Integer bases (b_x, b_N) ∈ {2..16}² are a 225-way choice, and
    one label picks (10, 10).
- **Every alias was a register-width alias.** Some programs fit every training row but fail on bigger moduli. Each one
  is a true congruence that only breaks because a 64-bit register overflows at 20–24 bits; run on unbounded integers,
  each is exact. Example: x²·(x² mod N) mod N for x⁴. This is the report's "structure that only matters
  off-distribution", in program form.

## 1. Exact blocks as layers; the program is learned (idea 1)
**Setup** (`ext_exact.py`, `ext_exact_soft.py`).
- Registers start as [N, x, 0, 1, 2].
- A program is L instructions (op, a, b), each over the registers and the earlier results; the answer is the last
  result.
- Blocks are exact int64 operations: wraparound like a 64-bit ALU, floor division, and x/0 = 0, x%0 = x as in RISC-V.
- Block kits:
  - tiny: {mul, mod}
  - basic: {add, sub, mul, div, mod}
  - nomod: basic without mod
  - full: basic plus {min, max, lt, and, or, xor, shl, shr}, 13 ops
  - full_nomod: full without mod
- Labels: the H1 training rows. T=1 is 1,000 rows of 13/16/20-bit moduli; T=2 is used for x⁴.
  - Survivors are re-checked on all 1,000 rows, then on 2,000 fresh rows each at 13/16/20/24 bits.
  - x⁸ uses freely generated 16-bit rows.

### 1a. Speculation
- Every program of length L is enumerated bottom-up.
- Partial programs that agree on 8 probe rows are merged (observational equivalence). Instructions that only
  recompute an existing register are dropped, and so are programs with unused results.
- The last instruction is scored on the first probe, and on the other probes only where it passes.

| kit (ops) | target | L | raw programs | non-redundant | time | left after 1 label | after 8 | on all 1,000 rows |
|---|---|---|---|---|---|---|---|---|
| tiny (2) | x² mod N | 2 | 2,280 (2^11.2) | 136 | <0.1 s | 1 | 1 | 1, correct |
| basic (5) | x² mod N | 2 | 15,750 (2^13.9) | 1,580 | <0.1 s | 1 | 1 | 1, correct |
| basic + digit order | x² mod N | 2 | 53,592 (2^15.7) | 6,040 | 0.1 s | 1 | 1 | 1, correct |
| full (13) | x² mod N | 2 | 92,565 (2^16.5) | 7,560 | <0.1 s | 1 | 1 | 1, correct |
| nomod (4) | x² mod N | 4 | 2.8e8 (2^28.1) | 1.4e6 | 0.4 s | 1 | 1 | 1: x²−⌊x²/N⌋N |
| full_nomod (12) | x² mod N | 4 | 1.9e10 (2^34.1) | 4.2e7 | 7.8 s | 1 | 1 | 1: x²−⌊x²/N⌋N |
| basic (5) | x⁴ mod N | 4 | 8.4e8 (2^29.7) | 2.6e6 | 2.7 s | 28 | 2 | 2: correct + 1 width alias |
| full (13) | x⁴ mod N | 4 | 2.9e10 (2^34.8) | 6.4e7 | 35 s | 57 | 2 | 2: correct + 1 width alias |
| basic (5) | x⁸ mod N | 5 | 2.8e11 (2^38.0) | 1.9e8 | 51 s (2.6 s on 32 cores) | 1 | 1 | 0: x⁸ needs 6 |
| basic (5) | x⁸ mod N | 6 | 1.2e14 (2^46.7) | 1.8e10 | 83 s on 32 cores | 336 | 64 | 7: 1 correct + 6 in 4 width-alias behaviours |

**Notes on the table.**
- Laptop times are on 3 workers on a shared machine (load average 40–56). The 32-core runs are on Modal
  (`ext_exact_modal.py`).
- Each extra instruction multiplies the raw space by about 2^8.5 and the non-redundant programs by 70–95. On
  32 cores, L=5 → L=6 took 32× longer.
  - Extrapolated, L=7 would take 1–2 hours on 32 cores, and L=8 days.
  - So exhaustive straight-line speculation tops out around 6–7 instructions, roughly 2^47–2^55 programs.
- After 8 labels, everything left is either correct or a width alias:
  - The x⁴ alias is x²·(x² mod N) mod N.
  - The x⁸ aliases use x·(x−N) ≡ x², t·(t−N) ≡ t² for t = x² mod N, or the x⁴ alias's trick of multiplying an
    unreduced square by a reduced one.
  - All overflow int64 at 20–24 bits (24-bit accuracy 0–14%). Run on unbounded integers, each is exact.
- The x² mod N searches had no aliases at all.

### 1b. Soft selection that hardens
The program space is the same; blocks are exact on reals (fp64; floor for div/mod). L=2, basic kit, H1 rows unless
noted. "Correct" means the argmax program is exact on all held-out sets. Blind guessing would hit 1 in 22,500
(4.4e-5).

| method | setting | result |
|---|---|---|
| value mixing (DARTS): softmax over ops and operands, values mixed, temperature 1→0.05, Adam | circular loss | **0/20** |
| | robust MSE loss | 0/20 |
| | 2-block kit {mul, mod} (3,600 programs) | 0/20 |
| | tiny moduli, 4 bits / 6 bits | 0/6 / 0/6 |
| straight-through Gumbel (one hard program forward, soft backward) | x² mod N | 0/4 |
| output-level mixture over all 22,500 complete programs, π factorised per slot | x² mod N | correct in 50 Adam steps (slot max prob 1.000); the enumeration took 0.8 s |
| REINFORCE, 64 hard programs sampled per step, 1,500 steps (96k samples) | x² mod N | **1/5** (hit at sample 320, converged by 2,624). Uniform sampling would hit within 96k samples 98.6% of the time, but the policy collapses onto wrong programs |

**What the mod does to the landscape.** Set every slot to π = (1−ε)·onehot(right choice) + ε·uniform and read the
loss (`ext_exact_soft interp`). Chance is about 1.0.

| loss first below 0.6 at ε = | 4 bits | 8 | 13 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|
| x² mod N (circular loss) | 1e-1 | 1e-4 | 1e-7 | 1e-9 | 1e-12 | 1e-14 |
| 1/N² for a typical N | 7e-3 | 3e-5 | 3e-8 | 4e-10 | 2e-12 | 6e-15 |

- With the reduction, the loss sits at chance until ε is within about 4× of 1/N² (on this decade grid). It is 1/N²
  rather than 1/N because the modulus operand also mixes in t0 = x² ≈ N², and its error is then multiplied by the
  quotient (≈ N).
- Control without a reduction, (N−x)·x with log loss: the loss falls steadily, by 2–8× per decade of ε, from
  ε = 0.1 on (24 bits: 457 → 421 → 195 → 70 → 34 → 14 → 3.8 → 0.65 → 0.08 …).
- Add one mod N to that control, ((N−x)·x) mod N: it is flat again until ε = 1e-5 / 1e-10 / 1e-14 at 8 / 16 / 24
  bits.

**What gradients do near the answer** (`ext_exact_soft basin`, 16 bits, 10 seeds). Each slot starts with mass p on
the right choice and 1−p on one wrong choice.

| p | 0.1 | 0.3 | 0.45 | 0.55 | 0.7 | 0.9 | 0.99 |
|---|---|---|---|---|---|---|---|
| argmax right at the start | 0 | 0 | 0 | 10 | 10 | 10 | 10 |
| with mod: right after training | 0 | 0 | 2 | **1** | 3 | 6 | 10 |
| no mod, (N−x)·x: right after training | 1 | 3 | 5 | 7 | 7 | 7 | 9 |

- Without the reduction, training turns a wrong start into the right program in up to half the seeds.
- With it, training mostly destroys a right start: 10/10 becomes 1/10 at p = 0.55.

**Caveat.** Value mixing also failed the no-mod control from a uniform start: 0/4, landing on programs such as 2N
or N·x. So the 0/20 is not *only* the reduction's doing. The landscape and the near-answer table isolate the reduction's
contribution.

### Answer to 1
- **The least to learn** is a 2-instruction wiring, about 11–17 bits, plus 1.8 bits for the input digit orders.
- **From answers alone:**
  - Yes, by speculation. One labelled row suffices, and the looped step certifies T=64 on both profiles.
  - No, by soft selection, once a reduction sits between the learned choice and the answer. This was tested at L=2,
    the smallest such program.
- **Where each method fails:**
  - Soft value mixing and straight-through fail at the first choice (L=2, about 2^14 programs).
  - REINFORCE is at the edge at L=2.
  - Speculation (and the complete-program mixture, which is the same enumeration) reaches about L=6–7, i.e. 2^47–2^55
    programs.
  - Beyond that, the structure must be shared across steps (a loop), as in section 3.

## 2. fp64 "analog" arithmetic with learned encoders (idea 2)
**Setup** (`ext_exact_analog.py`).
- The learned place weights encode v_x = Σ w_i x_i and v_N = Σ u_i N_i over the LSB-first digits. The true weights are
  10^i.
- Optionally, each value is snapped to the nearest integer right after the encoder, with a straight-through gradient.
- Then come exact fp64 blocks: p = v_x², r = p − v_N⌊p/v_N⌋, corrected once. The answer is round(r), wrapped into
  [0, round(v_N)).
- Loss: the circular loss 1 − cos(2π(r−y)/N), with Adam and a decaying step size.
- Data: 1,000 training rows of microlab `sqmod` at each size, and 2,000 fresh rows to score exactness.

**Is the exact forward even representable?** These are the true weights, no learning. Share of rows exact:

| modulus bits | ≤12 | 13 | 16 | 20 | 24 | 26 | 27 | 28 |
|---|---|---|---|---|---|---|---|---|
| fp32 | 1.0 | 0.79 | 0.14 | 0.01 | 0 | 0 | 0 | 0 |
| fp64 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.93 | 0.62 |

So fp64 does fix representability up to 26-bit moduli, because x² < 2^53. (The official H1 manifest runs bf16 autocast;
no analog arithmetic survives that.)

**The basin.** Every place weight of both encoders gets Gaussian noise of absolute size σ, and Adam starts with a
step size proportional to σ. The budget is 1,000 steps with 2 step sizes at 8–16 bits, and 4,000 steps with 3 step
sizes at 20–24 bits; a seed counts if any step size works. Count seeds (of 6) that end exact on the 2,000 fresh
rows:

| σ (absolute, per weight) | 1e-6 | 1e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 | 3e-1 |
|---|---|---|---|---|---|---|---|---|
| 8 bits, no snap | 6 | 6 | 6 | 6 | 6 | 6 | 4 | 0 |
| 13 bits, no snap | 6 | 6 | 6 | 6 | 6 | 6 | 1 | 0 |
| 16 bits, no snap | 6 | 6 | 6 | 4 | 6 | 4 | 3 | 0 |
| 20 bits, no snap | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 24 bits, no snap | 3 | 1 | 1 | 1 | 3 | 4 | 1 | 0 |
| 8 bits, snapped | 6 | 6 | 6 | 6 | 6 | 4 | 0 | 0 |
| 13 / 16 bits, snapped | 6 | 6 | 6 | 6 | 6 | 0 | 0 | 0 |
| 20 / 24 bits, snapped | 6 | 6 | 6 | 6 | 5 | 1 / 0 | 0 | 0 |

**Reading the table.**
- The rms error of the encoded value is about 9σ to 13σ. So "σ ≈ 0.03" means v_x must start within about 0.3 of the
  true integer.
- **Without snapping, the gradient basin ends between σ = 0.1 and 0.3 at every size.**
  - No seed converged from σ = 0.3, even on the training rows.
  - σ ≤ 0.03 is reliable at 8–16 bits, i.e. the encoded integer must start within about 0.3 of the truth.
- The tolerance is absolute, so the relative tolerance shrinks with size:
  - on the encoded value: 1e-3 at 8 bits, 4e-6 at 16 bits, 2e-8 at 24 bits;
  - on the 10^7 weight: 3e-9 relative at 24 bits.
- So the report's "within about 1e-7 relative" has the right order of magnitude around 20–24 bits.
- **Inside the basin, getting to exactly right is itself hard at 20–24 bits.** There are two separate reasons:
  - **24 bits: a valley.** Scaling w → c·w and u → c²·u leaves every answer almost unchanged. Adam slides into that
    valley and stalls with answers off by 1–40 on 3–90% of rows (traced from σ = 1e-4). It does so even starting
    from σ = 1e-6, and ends exact in only 1–4 of 6 seeds.
  - **20 bits: coverage.** Adam fits all 1,000 training rows (3–6 of 6 seeds for σ ≤ 0.1, in a 2,000-step run), but
    x ≥ 10^6 never occurs in those rows. So the 10^6 weight of x keeps its starting error (7e-7 from σ = 1e-6), and
    0.1% of fresh rows fail: 0/6 exact.
- **Snapping does not make it learnable.**
  - Snapped encoders are exact exactly when they start exact: the worst row's error must already be < 0.5, i.e.
    σ ≲ 0.01.
  - In all 45 snapped cells, the trained count equals the count that was exact before training. Training changed
    nothing.
- **What snapping does fix is the endgame.** Rounding absorbs any residual below 0.5, so the valley and the coverage
  gap stop mattering. At 20–24 bits, snapped encoders are exact on fresh rows whenever they start in the zone (5–6
  of 6 at σ ≤ 0.01).

**One learned base b** (v = Σ b^i d_i for both inputs; `ext_exact_analog base`). Each cell is the largest |b − 10|
that works:

| | 8 bits | 13 | 16 | 20 | 24 |
|---|---|---|---|---|---|
| forward exact without snapping | 1e-5 | 1e-8 | 1e-10 | 1e-12 | <1e-13 (estimated ≈ 2e-15, one fp64 ulp of 10) |
| forward exact with snapping | 1e-2 | 1e-4 | 1e-5 | 1e-7 | 1e-8 |
| Adam converges, no snap | 1e-2 | 1e-3 | 1e-4 | 1e-6 | 1e-10 |

- With snapping, Adam sometimes ends exact from outside the exact zone: 13 bits from 1e-1 but not from 1e-3; 24 bits
  from 1e-4 in 1 of 2 starts. These hits are not monotone in the distance. The optimizer's noise wanders into the
  flat exact zone and stays there, because the gradient is zero inside. That is a random search, not descent.
- (An earlier run of this test was wrong: the parameter b was float32, whose ~1e-6 grid near 10 faked successes.
  Every number here is from the fp64 rerun.)

**From a random start** (w_i = 10^i·U(0.5, 1.5), 24 bits, 5 seeds):
- 0/5 training directly at 24 bits, and 0/5 through a curriculum of 4→7→10→13→16→20→24 bits.
- Both hold with and without snapping. The weights stay 10–50% off.

**What does work: discrete parameters.** If the base itself is an integer, the encoder is a small discrete choice.
Over (b_x, b_N) ∈ {2..16}², the first H1 label leaves exactly (10, 10) of 225, and it is exact on all 1,000 rows.

### Answer to 2
- **How close must the weights be?**
  - Without snapping: within about 0.03 absolute on every place weight for gradients to converge reliably (encoded
    value within about 0.3). Nothing converges from 0.3.
  - With snapping: within about 0.01 for the encoder to be exact at all.
  - That is about 1e-8 relative on the encoded value at 24 bits, and about 1e-9 on the top place weight.
  - For a single shared base b, it is 1e-10 (gradient) or 1e-8 (snapped) at 24 bits.
- **Does snapping make it learnable?** No. It turns a narrow gradient basin into an equally narrow plateau that is
  exact inside and flat outside.
- **Does fp64 fix the precision cliff?** It removes the representation problem, and fp32 can't even run 13-bit
  moduli. The learning problem stays: the basin is set by N, not by the float format.
- Parameters that are themselves discrete (integer weights or bases) turn the needle into an enumeration that one
  label solves.

## 3. Which representation makes the learned control easiest? (idea 3)
**With whole-number blocks, representation doesn't change the control.** Exact multiply and remainder blocks compute
the same functions whether they hold fp64, int64 or digit vectors, so the search is identical. Representation only
changes three things:
- **The input/output conversion.**
  - Decimal digit-vector blocks need none.
  - Scalar blocks need atoi/itoa: free if fixed and exact, a needle if learned (section 2).
  - Bit blocks need decimal↔binary converters. REPORT.md found these with a label-free round trip.
- **Digit order.** In the real prompt, N's digits are LSB-first and x's MSB-first. Offering both readings of both
  strings costs 1.8 bits, and the first label resolves it (table above).
- **Register width.** This is where aliases come from: bounded registers (int64, or fp64 past 2^53) admit programs
  that are exact congruences until they overflow. Unbounded digit vectors would have none.

**With small blocks, binary wins by a wide margin** (`ext_exact_loop.py`).
- Blocks: {add, sub, mul, addm}, where addm(a, b) = a + b − N·[a + b ≥ N] is one conditional subtract.
- The loop scans x's digits from the top and applies one learned body s ← body(s, x, N, d), starting from s = 0.

| base | shortest body | bodies of that length | speculation |
|---|---|---|---|
| 2 | 3: t0=addm(s,s); t1=mul(d,x); t2=addm(t0,t1) (Blakley) | 4.9e6 (2^22.2) | 1 label leaves 3 bodies, all correct orderings; 5.5 s; certifies T=64 on both profiles |
| 4 | 6: 4s by two addm, then d·x < 3N reduced twice | 2^47.2 | out of reach; the hand-built body certifies T=64 |
| 10 | 14: 10s by four addm, then d·x < 9N reduced 8 times | 2^122.6 | out of reach; the hand-built body certifies T=64 |

- No body of length ≤ 3 exists in base 4 or 10 (searched).
- Follow-up, in `FINDINGS_loop_csub.md`: without the fused addm (blocks {add, sub, mul, csub}), the shortest binary
  body is 5 instructions (2^37.3 raw bodies, 3.4e6 distinct functions).
  - Speculation still finds it, in 8–10 s on 32 cores, and it certifies T=64.
  - But isolating it takes a median of 5 labels, not 1. Partial-reduction near-misses (bodies that leave s in
    [N, 2N)) fit the minimal body.
  - The fused grammar with one spare instruction needs the same: a median of 5 labels.
- A digit d·x reaches (base−1)·N. One conditional subtract removes one N, so the body grows linearly with the base
  while the number of loop iterations only shrinks with its logarithm.
- Whole-number fp64/int blocks give the shortest control of all (2 instructions, no loop), but only because the
  remainder block does the hard part.
- If a remainder block is allowed, scalars are easiest. If only the small, CPU-like blocks are allowed, bits are
  easiest, and decimal is hopeless for search.

## 4. Certification (looping the learned step T times, `h1.certify`)
| learned step | depth_t | depth_ood_n_t |
|---|---|---|
| mod(mul(s,s),N): found by speculation from 1 label | T=64 | T=64 |
| s² − ⌊s²/N⌋N (no remainder block): found from 1 label | T=64 | T=64 |
| Blakley loop body over the bits of s (3 instructions): found from 1 label | T=64 | T=64 |
| (hand-built base-4 and base-10 bodies, for comparison) | T=64 | T=64 |

## Caveats
- **The blocks carry the algorithm.** A remainder block, or a conditional subtract plus a loop over bits, is most of
  what x² mod N is. The question was how little is left once those exist, and the answer is: only wiring.
- **Some data is freely generated.** The main searches use H1 rows, but x⁸, the analog tests and the tiny-moduli runs
  use freely generated data, as microlab allows.
- **The DARTS result is partly about DARTS.** The landscape and near-answer measurements are the
  implementation-independent part (see the caveat in 1b).
- **Timings are noisy.** The laptop was shared, with load averages of 40–56. Treat laptop times as upper bounds.
- **Some soft-selection counts are small** (4–6 seeds). Every 0/n there sits next to a chance rate of about 1e-4 or
  below.

## Files
Everything lives in `cpu_first/microlab/`.
- Code:
  - `ext_exact.py`: blocks, speculation, the `certify` command.
  - `ext_exact_soft.py`: DARTS / straight-through / mixture / REINFORCE, plus the `interp` and `basin` probes.
  - `ext_exact_analog.py`: fp64 encoders.
  - `ext_exact_loop.py`: loop bodies.
  - `ext_exact_modal.py`: the L=5/6 searches on 32 cores.
- Data: every run is appended to `results_exact.jsonl`, and logs are in `runs_exact/`.
  - The file is append-only. For `analog_base`, use the last record: the first two had the float32 parameter.
  - Logs prefixed `SUPERSEDED_` are from runs with bugs that were later fixed: the float32 base, and a read-out that
    wrapped residue 0 to N.

```bash
python -m microlab.ext_exact spec --isa nomod --target sq --L 4
```

```bash
python -m microlab.ext_exact_soft interp --isa basic --target sq --loss circ
```

```bash
python -m microlab.ext_exact_analog basin --bits 16 --snaps 0 1
```

```bash
python -m microlab.ext_exact_loop search --base 2 --L 3
```

```bash
python -m microlab.ext_exact_loop certify --base 2 --body "mul,1,3;addm,0,0;addm,7,8"
```
