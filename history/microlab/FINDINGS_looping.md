# Looping x^(2^T) mod N: does sotaku's "train on later iterations" trick help?

**Question.** Sotaku keeps a looped transformer improving for 1,000+ iterations by training on later iterations:
on 20% of batches it runs 32–512 iterations without gradients, detaches the state, and trains a 16-iteration
window with the final answer as the target at every iteration. Does that make a learned loop of modular squaring
stay exact for many iterations, and let it work for more squarings T than it was trained on?

**Answer.** It helps the loop **rest**, not **count**.
- **Late windows make a reached answer stay put.** Share of answers still right at iteration 512:

  | | trained T ≤ 4, answer at T=4 (3 seeds) | trained T ≤ 16, answer at T=16 (1 seed) |
  |---|---|---|
  | continuous state, plain | 0.67 | 0.49 |
  | continuous state, late windows | 0.90 | 1.00 |
  | sotaku's recurrence (predictions fed back), plain | 0.98 | 0.84 |
  | sotaku's recurrence, late windows | 1.00 | 1.00 |

  Past the 528 iterations the late windows train, at iteration 2048 (T=4, seed 2): 0.70 for the continuous state
  with late windows (0.52 without), and 1.00 for sotaku's recurrence with late windows (0.91 without).
- **They do nothing for T-extrapolation.** Every continuous-state loop breaks within a few squarings past the
  trained T, with or without late windows: at the 5th after T ≤ 4 training, between the 24th and 32nd after T ≤ 16.
  At T=64 all of them sit at the level of a loop that simply stops squaring (0.15–0.25 right at every iteration;
  stopping gets 0.25).
- **Snapping the state to digits after every iteration** fixes extrapolation by construction. With per-step labels,
  one step is exact on every input (all 768 (N, r) pairs at 6 bits, all 12,288 at 8 bits, checked exhaustively).
  The loop is then exact at every T with any number of extra iterations: T=64 run for 512 iterations is 1.00,
  trained on T ≤ 4. Final-answer labels alone get there too at T ≤ 4, with 4× the steps.
- **The countdown is the other half.** A learned decimal countdown fails at the first unseen count (T=5 after
  T ≤ 4; T=17, or 21 with a shared digit cell, after T ≤ 16). One learned cell shared by every *binary* digit
  position counts exactly to 127 after seeing counts up to 4. With that counter driving the snapped step, the only
  hand-made parts are the wiring, and every T ≤ 64 is 1.00 (both seeds, trained on T ≤ 4 or T ≤ 16).

## Plain-language summary
- **The trick only ever trains resting.** Training T is at most 16, so by iteration 32 (where late windows start)
  every example has already reached its answer. The late window teaches "keep the answer". It never shows the
  model a squaring from a state it has not seen.
- **So it helps exactly where a continuous state decays: holding an answer for hundreds of iterations.**
  - Continuous state, trained on T ≤ 4: answers at T=1..4 still right after 512 iterations: 0.41–0.70 → 0.59–0.94
    with late windows (3 seeds).
  - Sotaku's recurrence (state plus fed-back softmax predictions) already holds well: 0.91–0.99 → 0.98–1.00.
  - Trained on T ≤ 16, a T=16 answer is never held during plain training (the window ends at iteration 16), so it
    is lost by iteration 512 (0.49; 0.84 with feedback). With late windows: 1.00 for both.
  - Past the trained horizon the benefit shrinks but stays. Sotaku's full recipe still holds at 2048 iterations
    (0.95–1.00), like sotaku's own small decay from 1024 to 4096.
  - The state keeps moving. Every continuous state is still moving at iteration 512: each iteration shifts it by
    0.15–0.2% of its length, so its norm grows roughly linearly. By iteration 512 the plain state is 16–45× its norm
    at T, the fed-back state 36–68×. The readout's LayerNorm hides much of that.
  - Late windows slow the plain state's growth: 2× instead of 16× after T ≤ 16 training, 28× instead of 45× after
    T ≤ 4. They barely change the fed-back state's, which adds a vector every iteration.
- **Continuous states break soon after the trained count.**
  - Trained on T ≤ 4, the readout is right after 4 squarings for every example and after 8 for about 0.4. T=64
    chain: 1.00 at 4, 0.35–0.44 at 8, 0.23–0.31 at 16, over 3 seeds and all four variants.
  - The same network squares a *digit embedding* correctly for 98–100% of inputs. The knowledge is there; the 5th
    squaring just starts from a latent the network never saw.
  - Trained on T ≤ 16 (with 4× the steps), they stay exact to about 20 squarings, are at half by 32, and at the
    baseline by 64. Late windows change none of that.
- **Snapping makes every iteration the same problem.** If the state is re-embedded from digits after each step, the
  step sees the same kind of input at iteration 60 as at iteration 1. Accuracy becomes flat in T: whatever the step
  gets right at T=1 it gets right at T=64.
  - With per-step labels the step is exact, so the loop is exact. Checked on every reachable input, so this is a
    proof for these moduli with an exact counter, not a sample.
  - With final-answer labels only, the snapped loop is still flat in T, but the step learns slowly: the loss only
    trains the last step, from the model's own state.
    - T ≤ 4: 0.87 in range and 0.83 strict at T=64 after 3000 steps. After 12,000 steps the step is exact on every
      input and the loop is exact at every T.
    - T ≤ 16: 0.19 after 3000 steps. After 12,000 steps, 0.85 at T=1 and 0.66 strict at T=64.
  - Straight-through gradients made the final-answer version worse (0.50 at T ≤ 4), not better.
  - A soft digit state (re-embedded softmax) learned poorly here: 0.38 in range at T ≤ 4, 0.14 at T ≤ 16.
- **Counters don't extrapolate unless every case of the counting rule was seen.** Decimal countdown digits fail
  as soon as the count is new. A decimal cell shared by both digit positions gets a little further (exact to T=20
  after T ≤ 16). Then it needs "tens digit 2 with no borrow", which never occurs in training. In binary, all four
  cases of the shared bit cell occur in the low bits of counts 0..4, so it counts exactly at any width.
- **For One Layer Deeper:** once a step is exact on digits, looping and T-extrapolation are free, *if* the state is
  snapped to digits each iteration and the countdown is exact (or built from a shared binary cell). Sotaku's trick is
  a stabiliser for resting continuous states. It is worth having if the state stays continuous, but it will not
  produce T=32 or T=64 from T ≤ 16 training. The hard part stays the step itself (the remainder at 24 bits).

## Setup
- **Task.** b-bit odd moduli N ∈ [2^(b−1), 2^b), x uniform in [0, N), answer x^(2^T) mod N. Main runs at 6 bits:
  16 moduli, 768 (N, r) pairs, so one squaring step is a lookup table a small MLP can learn (as `sqmod --size 6`).
  One confirmation at 8 bits. Training T uniform in 1..4 or 1..16, fresh data every batch.
- **Loop.** State = residue digits; each iteration squares mod N while the countdown is above 0, and holds once it
  is 0. Part 1 keeps the countdown exact, outside the network; the network sees go = [countdown > 0] each iteration.
  Part 2 learns the countdown.
- **Step network** (shared by every variant): `z = h + MLP(LN(h) + emb(N) + emb(go))`, residue logits `R(LN(z))`.
  MLP 64→256→256→64 (about 103K parameters; hidden 512 at 8 bits). h0 = digit embedding of x. The state kinds
  differ only in what the next iteration gets:

  | state | next iteration gets | gradient across iterations |
  |---|---|---|
  | latent | z | yes |
  | fb (sotaku's recurrence) | z + P(softmax(logits)) | yes |
  | soft | E(softmax(logits)), the expected digit embedding | yes |
  | hard | E(onehot(argmax logits)), snapped to digits | no |
  | ste | hard forward, soft backward | straight-through |

- **Training.** 3000 steps unless noted, batch 256, AdamW lr 3e-3 one-cycle, wd 0.01, grad clip 1.
  - **plain:** iterations 1–16 from the start, backprop through all 16. Target = final answer at every iteration
    ≥ T. Iterations before T have no correct "final" target, so they are masked.
  - **late:** sotaku's recipe, as `BASE_REPLACEMENT` in `looping/exp_stay_solved.py`. 80% of batches as plain. On
    20%, run 32/64/128/256/512 iterations without gradients from the start, detach, then train iterations
    k+1..k+16 with the final answer at every iteration. The late loss replaces the normal one on those batches
    (mix 1.0).
  - **steps:** per-step labels: the true state is the target at every iteration. For hard the inputs are the true
    states too (teacher forcing), so each step is trained on its own.
- **Eval.** 256 fresh (N, x) per T ∈ {1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 32, 48, 64}. Every example runs 512
  iterations (≥ 8T), and the later runs keep going to 2048. The readout is decoded every iteration.
  - **exact@T:** answer right at iteration T.
  - **chain:** right after each of the T squarings.
  - **strict:** right at every iteration 1..8T: the chain, then 7T extra iterations.
  - **at512 / at2048:** answer right at that iteration.
  - **step check:** one iteration from the digit embedding of every input: squaring for all (N, r), holding for all
    squares (every answer with T ≥ 1 is a square).
- **Why strict, not exact.** Squaring mod a 6-bit N falls into short cycles: 196 of 768 pairs end at a fixed point,
  and the other cycles have length 2, 3, 4, 6, 11, 12 or 28. A loop that is exact for 4 squarings and then stops gets
  **0.89** exact at T=64, because 60 extra squarings usually come back to the same residue. Under strict it gets
  0.25 (only trajectories already at a fixed point). Both baselines are in the tables.

## Part 1: the loop, exact countdown (6 bits)

### Trained on T ≤ 4
3 seeds for latent and fb, 2 for hard, 1 for soft and ste. Seed means; seeds differ by ≤ 0.04 on the strict
columns.

| state / training | step: square | exact, T ≤ 4 | strict T=8 | strict T=16 | strict T=64 | exact T=64 | right at iteration 512, T=4 |
|---|---|---|---|---|---|---|---|
| latent / plain | 1.00 | 1.00 | 0.31 | 0.20 | 0.15 | 0.22 | 0.67 (0.64–0.70) |
| latent / late | 0.99 | 1.00 | 0.33 | 0.23 | 0.17 | 0.22 | 0.90 (0.88–0.94) |
| fb / plain | 1.00 | 1.00 | 0.33 | 0.23 | 0.17 | 0.22 | 0.98 |
| fb / late | 0.99 | 1.00 | 0.35 | 0.25 | 0.19 | 0.25 | 1.00 |
| soft / plain | 0.38 | 0.38 | 0.16 | 0.16 | 0.14 | 0.28 | 0.36 |
| soft / late | 0.30 | 0.23 | 0.08 | 0.07 | 0.07 | 0.11 | 0.17 |
| ste / plain | 0.55 | 0.50 | 0.38 | 0.41 | 0.37 | 0.46 | 0.46 |
| ste / late | 0.42 | 0.38 | 0.20 | 0.16 | 0.16 | 0.27 | 0.34 |
| hard / plain | 0.85 | 0.87 (0.83–0.90) | 0.78 | 0.83 | 0.83 (0.80–0.86) | 0.84 | 0.87 |
| hard / plain, 12,000 steps (1 seed) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **hard / steps** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| stop after 4 squarings (baseline) | — | 1.00 | 0.29 | 0.27 | 0.25 | 0.89 | — |

"step: square" is one squaring from the digit embedding, over all 768 (N, r).
- For latent and fb that is only the first iteration's input, which is why it can be 1.00 while the loop fails.
  For hard it is every iteration's input.
- hard/steps (and hard/plain at 12,000 steps) also hold every reachable input exactly. Together with the exact
  counter, that proves the 6-bit loop exact for every T and every number of extra iterations.

Right at iteration 512, for each trained T (seeds 0, 1, 2):

| state / training | T=1 | T=2 | T=3 | T=4 |
|---|---|---|---|---|
| latent / plain | 0.41, 0.41, 0.52 | 0.54, 0.54, 0.58 | 0.61, 0.64, 0.69 | 0.64, 0.68, 0.70 |
| latent / late | 0.63, 0.59, 0.66 | 0.77, 0.80, 0.88 | 0.88, 0.83, 0.93 | 0.88, 0.88, 0.94 |
| fb / plain | 0.94, 0.95, 0.91 | 0.96, 0.95, 0.96 | 0.95, 0.99, 0.93 | 0.98, 0.98, 0.98 |
| fb / late | 1.00, 1.00, 0.98 | 1.00, 1.00, 1.00 | 1.00, 1.00, 1.00 | 1.00, 1.00, 1.00 |

**Rest past 528.** Seed 2 also ran on to iteration 2048, four times past the deepest late window. Answer right at
T=1, 2, 3, 4:

| state / training | iteration 512 | iteration 2048 | state norm at 512 / at T=4 |
|---|---|---|---|
| latent / plain | 0.52, 0.58, 0.69, 0.70 | 0.37, 0.40, 0.43, 0.52 | 45× |
| latent / late | 0.66, 0.88, 0.93, 0.94 | 0.45, 0.54, 0.68, 0.70 | 28× |
| fb / plain | 0.91, 0.96, 0.93, 0.98 | 0.85, 0.88, 0.90, 0.91 | 67× |
| fb / late | 0.98, 1.00, 1.00, 1.00 | 0.95, 0.98, 0.99, 1.00 | 68× |
| hard / plain, 12,000 steps | 1.00 | 1.00 | 1× |

Where the continuous loops break. T=64, share whose readout is right after i squarings (3-seed mean):

| state / training | i=1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| latent / plain | 1.00 | 1.00 | 1.00 | 0.39 | 0.26 | 0.24 | 0.22 |
| latent / late | 0.98 | 1.00 | 1.00 | 0.39 | 0.27 | 0.24 | 0.22 |
| fb / plain | 1.00 | 1.00 | 1.00 | 0.37 | 0.27 | 0.24 | 0.22 |
| fb / late | 0.99 | 1.00 | 1.00 | 0.42 | 0.31 | 0.26 | 0.25 |
| hard / steps | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Trained on T ≤ 16
At 3000 steps the continuous loops do not even fit the training range (0.54–0.68 exact for T ≤ 16; soft, ste and
hard/plain ≤ 0.19), while hard/steps is 1.00 everywhere on 2 seeds. So the continuous rows below use 12,000 steps
(4× the budget, seed 0). There every T ≤ 16 is exact, so extrapolation can be judged. hard/plain (snapped,
final-answer labels) is still learning at 12,000 steps, but is roughly flat in T.

| state / training | exact, T ≤ 16 | strict T=16 | right at 512, T=16 | chain T=20 | chain T=24 | chain T=32 | strict T=32 | strict T=64 | exact T=64 | state norm at 512 / at T=16 |
|---|---|---|---|---|---|---|---|---|---|---|
| latent / plain | 1.00 | 0.92 | 0.49 | 1.00 | 0.98 | 0.50 | 0.33 | 0.24 | 0.29 | 16× |
| latent / late | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 0.48 | 0.42 | 0.25 | 0.34 | 2.0× |
| fb / plain | 1.00 | 0.96 | 0.84 | 1.00 | 0.96 | 0.53 | 0.45 | 0.25 | 0.29 | 46× |
| fb / late | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.58 | 0.57 | 0.25 | 0.30 | 36× |
| hard / plain (12,000 steps) | 0.78 | 0.71 | 0.74 | 0.71 | 0.67 | 0.65 | 0.65 | 0.66 | 0.69 | 1× |
| hard / steps (3000 steps, 2 seeds) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1× |
| stop after 16 squarings (baseline) | 1.00 | 1.00 | — | 0.26 | 0.23 | 0.25 | 0.25 | 0.25 | 0.89 | — |

"chain T=k" = right after every one of the k squarings, with no extra iterations, so it separates counting from
resting.
- **Late windows fix resting.** A T=16 answer is never held during plain training (the window ends at iteration
  16), so plain loops lose it by iteration 512 (0.49, 0.84). With late windows: 1.00 for both. They also stop the
  plain state's growth (2× instead of 16× by iteration 512).
- **They do not extend counting.** Squarings past 16 are about as good with late windows as without (chain at 32:
  0.48–0.58 for all four). The strict@32 gain (0.33 → 0.42, 0.45 → 0.57) is resting after a correct chain.
- **Training on longer chains buys a few extra squarings, not a rule.** After 16 consecutive training squarings the
  loops stay exact to about 20 squarings and mostly right at 24, against breaking at the 5th after T ≤ 4. By 32 they
  are at half, and by 64 at the baseline.

### 8 bits (snapped step)
hard/steps with hidden 512, T ≤ 16, 6000 steps, seed 0. The step is exact on all 12,288 8-bit (N, r) inputs
(square) and on every reachable hold. Strict 1.00 at every T up to 64, and the answer is still right at iteration
2048. Cycles are longer at 8 bits, so the stop-after-16 baselines are lower: exact@64 0.59, strict@64 0.09. The
continuous variants were not run at 8 bits.

## Part 2: the countdown

### Counter alone
Each counter is trained on the transitions of counts 0..tmax (count → count − 1, 0 → 0). It is tested from every
start count up to its width's limit: it must read T − i at every iteration, then stay at 0 through 2T + 4
iterations.
- **Shared cell:** one small MLP, (digit, borrow in) → (digit, borrow out), applied to every position with a fixed
  ripple. A borrow out of the top digit means the count was 0, so it holds.
- **One MLP:** sees all digits at once.

2 seeds each; both seeds gave identical results.

| countdown | trained on counts ≤ 4: first wrong start | trained on counts ≤ 16: first wrong start | right starts ≤ 64 (≤ 16 training) |
|---|---|---|---|
| decimal, 2 digits, one MLP | 5 | 17 | 16/64 |
| decimal, 2 digits, shared cell | 5 | 21 | 20/64 |
| binary, 7 bits, one MLP | 5 | 17 | 16/64 |
| **binary, 7 bits, shared cell** | **none (all 127 right)** | **none (all 127 right)** | **64/64** |
| numeric (exact integer, Part 1) | none | none | 64/64 |

Why the decimal shared cell stops at 20:
- Units digits pass through every (digit, borrow 1) case, so 20 → 19 works.
- "Tens digit 2..9 with no borrow" (21 → 20, 31 → 30, …) only happens at counts ≥ 21, never in training.
- After counts ≤ 4 even the units digits 5..9 are new, so every learned counter fails at 5 except the binary cell.
- In binary, counts 0..4 already contain all four (bit, borrow) cases.

### Residue and decimal countdown in one snapped state (the formulation taken literally)
One MLP, state = residue digits + 2 decimal countdown digits, snapped every iteration, per-step labels. The network
decides from the countdown digits whether to square.

| trained on | T in range | strict, first T past range | strict T=32 | strict T=64 | countdown right, any T past range |
|---|---|---|---|---|---|
| T ≤ 4 (seeds 0, 1) | 1.00 | T=5: 0.16, 0.08 | 0.07, 0.06 | 0.09, 0.10 | 0.00 |
| T ≤ 16 (seeds 0, 1) | 1.00 | T=20: 0.09, 0.08 | 0.11, 0.12 | 0.14, 0.15 | 0.00 |

### Everything learned
The hard/steps residue step, driven by the learned binary shared-cell countdown (its go = no borrow out of the top
bit). Both are trained on T ≤ tmax only. The only hand-made parts are the ripple wiring and "hold when the count is
0".

| trained on | strict, every T in {1, …, 64} (8T iterations each) | countdown go right at every iteration |
|---|---|---|
| T ≤ 4 (2 seeds) | 1.00 | 1.00 |
| T ≤ 16 (2 seeds) | 1.00 | 1.00 |

## Caveats
- **Tiny moduli.** At 6–8 bits the step is a lookup table (768 or 12,288 entries) learned by memorisation. That is
  the point, since it separates the loop from the remainder, but it says nothing about learning the step at 24 bits.
- **MLP, not a transformer.** The state is one vector, not sotaku's per-cell states with attention. The main
  negative result does not depend on that: with T ≤ 16 training, late windows (iteration ≥ 33) can only supervise
  resting.
- **Part 1's countdown is exact.** Part 2 shows what a learned one costs.
- **Few seeds:** 3 for the latent/fb T ≤ 4 comparison, 1–2 elsewhere. The effects reported are large compared with
  the seed spread. The T ≤ 16 continuous comparison is one seed at 12,000 steps.
- **The 2048-iteration check** exists only for the runs started after it was added (seed 2 of latent/fb at T ≤ 4,
  hard/plain at 12,000 steps, 8 bits).
- Fresh data every batch (microlab's exploration convention), not the competition's fixed pool.

## Reproduce
From `cpu_first/`. Use 1 thread each, since the machine is shared.

```bash
python -m microlab.ext_looping train --state fb --method late --tmax 4 --steps 3000 --lr 3e-3 --seeds 0,1,2
```

```bash
python -m microlab.ext_looping train --state latent --method late --tmax 16 --steps 12000 --lr 3e-3
```

```bash
python -m microlab.ext_looping train --state hard --method steps --tmax 4 --steps 3000 --lr 3e-3
```

```bash
python -m microlab.ext_looping train --state hard --method steps --bits 8 --tmax 16 --steps 6000 --lr 3e-3 --hidden 512
```

```bash
python -m microlab.ext_looping counter --tmax 16
```

```bash
python -m microlab.ext_looping joint --tmax 16 --seeds 0,1
```

```bash
python -m microlab.ext_looping full --tmax 4 --seeds 0,1
```

```bash
python -m microlab.ext_looping report
```

Every run appends one JSON line to `history/microlab/results_looping.jsonl`. `report` prints the tables above from it.
