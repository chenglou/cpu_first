# Unfolding the remainder by itself

**Question.** The answer y = s² mod N is s² folded by N. Without being given s², can a model "unfold" the folded labels
by itself, or can we find a loss whose gradient points the right way despite the wraparound?

RELAXED.md showed that once s² is a label, periodic nets learn exactly at 6–24 bits. Here the label is only y. The setup
is RELAXED.md's (fp64, a model predicts the pre-remainder value V, exact snapping round(V) mod N, Levenberg-Marquardt);
the code is `unfold.py`.

**TL;DR**
- **The clock loss already is self-unfolding, and it works only in a basin.**
  - 1 − cos(2π(V − y)/N) is plain regression toward V* = y + N·round((V − y)/N), the model's own wrap count.
  - Plain LM recovers from a median row error of 0.1–0.3 N, i.e. while most rows (66–100%) are within N/2 of s². That
    confirms the prediction.
  - **A sensitivity-weighted clock loss** (each row's gradient divided by how fast its phase moves) widens that a lot for
    one shared parameter: the radix model recovers from 30–100 N, with only 4–7% of rows in the basin. It barely helps
    models with many free parameters.
- **The wrap count does not come from the top digits.** At 24 bits, the top 5 digits of s give the right wrap count for
  8.5% of rows, 6 digits for 20%, 7 for 83%, all 8 for 100%. H1's rows need all 6 digits. Coarse-to-fine from the top
  cannot bootstrap the unfolding.
- **A rung head (log k = α log s + β log N + γ) trained toward its own nearest rung has no restoring force.**
  - The nearest rung always agrees with the head's estimate to within half a rung, so every residual is ≤ 0.5/k, right
    or aliased. From a random start, failing runs end exactly at their initial parameters.
  - It learns only from rows it already places below rung 1. Started there and fed easiest-first, it is exact: 5/5 at
    8–20 bits, and on H1 5/5 (s as a number) or 4/5 (shared radix), certified T=64 in distribution.
  - Ordered by its own rung estimate (no oracle), it bootstraps at 8–16 bits (3–4/5) but not at 20 bits or on H1 (0/5).
- **A fold curriculum carries learning across folds exactly when every parameter is pinned before its rows wrap.**
  - Value inputs: 5/5 at 8–24 bits.
  - Digit inputs with **one shared radix** (place value i = B^i): 5/5 at 12–24 bits; it learns B = 10.000000.
  - Digit inputs with **free place values** (8 weights, or RELAXED's 70-weight bilinear model): 0/5 at every size.
    Place value 10^h is first used by rows that already wrap about 10^(2h)/N times: 600 times for 10^5 at 24 bits. The
    loss weighting doesn't change this.
  - So the needle is the wraparound times unshared parameters.
- **On H1's own 1,000 answer-only T=1 rows, self-unfolding certifies T=64 on both profiles.**
  - Setup: fit the 5 non-wrapping rows by plain regression, then add rows easiest first with the clock loss.
  - Certifies: value inputs (6 or 3 weights) 5/5, one shared radix 5/5, radix + 3 weights 4/5.
  - Fails: radix + 6 weights 0/5, free place values 0/5, bilinear 0/5.
  - With the clock loss from the start, only the shared radix survives (3/5, or 5/5 weighted). From V ≈ 0, non-wrapping
    rows with y > N/2 sit nearest the wrong wrap. RELAXED.md's H1 run also started its clock loss from V ≈ 0, which
    fits its failure.
- **Analytically, yes: solve every wrap count at once by lattice reduction**, as GPS carrier-phase ambiguity resolution
  does.
  - Value features (1, s, s²): **s² from 2 random H1 rows** (12/12 trials with 2–10 rows), exact on all 1,000 T=1 rows.
  - Digit features with free place values, a general quadratic form in the digits (28 integer coefficients up to
    10^10): **exact s², with every place-value product 10^(i+j), from 60 random rows of H1's T=1 set** (3/3 trials,
    2.4 s), exact on all 1,000 T=1 rows.
  - That is the model every gradient method above fails to learn.
  - Rows needed per size: 30 at 13 bits and 40 at 16 bits (20 and 30 rows failed); 60 suffices at 20 bits, the
    smallest count tried.

## 1. Test 1 — the clock loss's basin
Start at an exact solution and perturb the parameters in a random direction. The size of the push is set so the median
row's |V − s²| is A·N. Then run 20 LM iterations on 4,096 fresh rows. Success means 100% exact on 20,000 fresh rows;
there are 5 directions per cell. The table gives the largest A recovered in ≥ 4/5 runs, with the share of rows inside
|V − s²| < N/2 at that A:

| model | parameters | plain clock loss | sensitivity-weighted |
|---|---|---|---|
| value: w · (1, s, N, s², sN, N²) | 6 | A = 0.1–1 (3–100% rows in basin) | A = 0.3–1 |
| value3: w · (1, s, s²) | 3 | A = 0.3 (80–94%) | A = 0.3 |
| tied: (Σ B^i d_i)² | 1 | A = 0.1–0.3 (66–100%) | **A = 30–100 (4–7%)** |
| tied3 / tied6: radix + 3 / 6 weights | 4 / 7 | A ≤ 0.1 | A = 0.1–0.3 |
| untied: (Σ w_i d_i)² | 8 | A = 0.1–0.3 (66–97%) | A = 0.1–1 (29–97%) |
| bidig (RELAXED.md's bilinear digits) | 70 | A = 0.1–0.3 (81–100%) | A = 0.1–0.3 |

Ranges are over 8, 12, 16, 20 and 24 bits; per-size rows are in `runs/unfold/basin2.json` and `basin_sens.json`.

- **Why plain LM's basin is narrow.** LM weights rows by their Jacobian, and the rows whose phase moves fastest (large s,
  many digits) are the first to alias onto a wrong wrap count. At A = 1 only 0.2–1.6% of LM's weight sits on in-basin
  rows, for every model. For the radix model this already happens at A = 0.3 (1–9%).
- **The weighted loss** divides each row's clock loss by its sensitivity |∂V/∂θ|/N, so every row gets one unit-length
  vote.
  - An aliased row's vote is then small and bounded rather than large and random.
  - With one shared parameter the in-basin votes all agree. Each step moves the share of in-basin rows of the way toward
    the truth, so the share keeps growing.
  - With many free parameters the aliased rows still steer the unconstrained directions.

**Usable on H1?** Only to polish a model that is already close. The H1 rows are mostly far from any start: 50 of 1,000
wrap ≤ 64 times.

## 2. Test 2 — does the wrap count depend mostly on the top digits?
s is known only to its top j significant digits; the rest are replaced by their midpoint. The table gives the share of
rows whose unfolded target y + N·round(ŝ²/N − y/N) is still right, i.e. |ŝ² − s²| < N/2:

| rows | j=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 8 bits (s ≤ 3 digits) | 0.17 | 0.59 | 1 | | | | | |
| 12 bits | 0.05 | 0.15 | 0.41 | 1 | | | | |
| 16 bits | 0.01 | 0.03 | 0.13 | 0.30 | 1 | | | |
| 20 bits | 0.003 | 0.011 | 0.026 | 0.11 | 0.24 | 1.00 | 1 | |
| 24 bits | 0.001 | 0.002 | 0.009 | 0.021 | 0.085 | 0.20 | 0.83 | 1 |
| H1 T=1 rows | 0.013 | 0.058 | 0.14 | 0.47 | 0.76 | 1 | | |

- The wrap count is ⌊s²/N⌋, and it is right for most rows only once **every** digit of s is right.
- The top digits fix it only for rows with a small wrap count, which are the same rows a fold curriculum starts from.
- A "learn the top digits first" schedule therefore has no correct targets to learn from. **Not usable**, on fresh data
  or on H1.

### 2b. The rung estimate as a model
A head predicts the rung (wrap count) in log space: log k̂ = α·log s′ + β·log N + γ, with V = N·k̂. The exact head is
α = 2, β = −1, γ = 0. s′ is s read as a number (value), Σ B^i d_i (one shared radix) or Σ w_i d_i (free place values).

Each of 48 rounds:
- gives every row its nearest unfolded rung, k* = max(0, round(k̂ − y/N));
- then runs LM on the log-space residual log k̂ − log(k* + y/N), with those targets held fixed.

Schedules:
- **none:** all rows every round.
- **oracle:** easiest first by the true s²/N.
- **self:** the rows the head itself rates lowest-rung first. This uses no oracle at all.

Fresh rows at 8–20 bits (fp64 can't hold exp(log k) to ±0.5 beyond about 20 bits) and H1's 1,000 T=1 rows; 5 seeds;
run on Modal. Fresh-data cells list 8 / 12 / 16 / 20 bits; H1 counts runs exact on all 1,000 rows, all of which
certify T=64 on depth_t.

| head | start | none | oracle | self | H1, oracle | H1, self |
|---|---|---|---|---|---|---|
| value | random | 0/5 | 2/5 at each size | 1, 1, 1, 0 | 1/5 | 0/5 |
| value | below rung 1 (k̂ = e^−10) | 0/5 | **5/5 at each size** | 3, 4, 4, 0 | **5/5** | 0/5 |
| tied (radix) | random | 0/5 | 2, 2, 1, 1 | 1, 1, 1, 0 | 1/5 | 0/5 |
| tied (radix) | below rung 1 | 0/5 | **5/5 at each size** | 4, 4, 1, 0 | **4/5** | 0/5 |
| untied | either | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |

- **No restoring force above rung 1.**
  - The folded label fixes only the fractional part of the rung. The nearest-rung assignment is always within half a
    rung of the head's estimate, so every log residual is at most 0.5/k̂, whether the row is right or aliased.
  - With a random start (typical k̂ in the thousands), the failing runs end at exactly their initial parameters. The
    random-start successes are the seeds whose start happened to place the easy rows below rung 1.
- **Below rung 1 the target is the true value.** There the assigned rung is 0, so the head sees y = s² itself. Started
  there with easy rows first, it learns α = 2 and β = −1 exactly and unfolds every later row.
  - The OOD profile (22- and 24-bit moduli) fails already at T=1. That is fp64, not the fit: exp(log k) cannot land
    within 0.5 of s² there. The value-space clock models of §4 certify that profile.
- **Self-paced works only where non-wrapping rows are common.** Ordering by the head's own estimate needs no oracle and
  works at 8–16 bits. At 20 bits only 0.1% of rows don't wrap, on H1 0.5%, and it gets 0/5.
- **Usable on H1?** Only with the easiest-first order (or an equivalent first stage that fits the non-wrapping rows).
  That is the same requirement as the clock loss in §4: the rung head adds a parameterization, not new information.

## 3. Test 3 — a fold curriculum
s < N^e, with e growing from 1/4 (no wraps at all) to 1 over 48 stages of 4,096 fresh rows. Each stage runs 3 LM
iterations on the clock loss. Each cell is 5 seeds from a small random start; success means 100% exact on 20,000 fresh
full-range rows.

| model | 8 bits | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|
| value (6 weights) | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| value3 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| tied (one radix) | 0/5 | **5/5** | **5/5** | **5/5** | **5/5** |
| tied3 (radix + 3 weights) | 0/5 | 5/5 | 3/5 | 4/5 | 3/5 |
| tied6 (radix + 6 weights) | 0/5 | 5/5 | 4/5 | 3/5 | 4/5 |
| untied (8 free place values) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| bidig (70 weights) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |

The sensitivity-weighted loss gives the same table within one seed per cell: it does not rescue untied or bidig.

- **Learning carries across folds only through parameters that the unwrapped rows already pinned.**
  - At 20 bits, untied learns 1.85 / 10.7 / 96.7 / 1001 for the place values 1 to 1,000: roughly right.
  - The 10^4 place value comes out as −33.6. It is first used by rows that wrap about 100 times, so nothing ever
    told the model its value inside the basin.
- **A shared radix fixes it.** The place values of the digits that appear only in wrapping rows are then B^i, already
  known from the small digits.
- **The radix models fail at 8 bits for the same reason.** B is first exercised by 2-digit s, and at 8 bits those
  already wrap, so B is never pinned inside the basin.

**Usable on H1?** Yes, as an easiest-first order over H1's own rows (§4), for models whose parameters are all pinned by
the few non-wrapping rows.

## 4. H1's own rows, answer-only
The 1,000 T=1 training rows (13/16/20-bit moduli) are sorted by x²/N. 5 rows don't wrap; the 6th wraps twice. The
schedule:
- **plain:** fit those 5 rows by plain regression first (the sort already says they don't wrap);
- then 48 stages growing geometrically from 5 to 1,000 rows, each running LM on the clock loss.

Certified means the snapped step, looped, is 100% on every rung T = 1 … 64 of both H1 profiles (depth_t and
depth_ood_n_t). Certified seeds out of 5:

| model | first stage plain | plain + weighted | first stage clock | clock + weighted |
|---|---|---|---|---|
| value (6) | **5/5** | 5/5 | 0/5 | 0/5 |
| value3 | **5/5** | 5/5 | 0/5 | 0/5 |
| tied (1) | **5/5** | 5/5 | 3/5 | **5/5** |
| tied3 (4) | 4/5 | 3/5 | 0/5 | 0/5 |
| tied6 (7) | 0/5 | 0/5 | 0/5 | 0/5 |
| untied (8) | 0/5 | 0/5 | 0/5 | 0/5 |
| bidig (70) | 0/5 | 0/5 | 0/5 | 0/5 |

- Every certifying run trains in under 0.1 s. The radix runs print B = 10.0000000000.
- **Why the first stage matters.** From V ≈ 0, a non-wrapping row with y > N/2 has its nearest clock minimum at y − N.
  H1's five easy rows have y/N up to 0.68. Free weights fit the wrong target; the one-parameter radix is too rigid to.
  RELAXED.md's H1 run also started its clock loss from V ≈ 0, on the 16 easiest rows. My clock-first runs reproduce
  that failure, and fitting the 5 non-wrapping rows by plain regression first fixes it.
- **The limit is parameters vs non-wrapping rows.** H1 gives 5 rows of at most 3 digits.
  - They pin up to about 6 weights on values, or a radix plus 3 weights on digits.
  - They never pin a free place value above 100.

## 5. Unfolding analytically: integer ambiguity resolution
For a hypothesis that is linear in integer coefficients, V = c · φ(s), each row says c · φ(s_i) − y_i = k_i N_i for an
unknown integer wrap count k_i. All the k_i and c can be found at once: take the shortest vector of Kannan's embedding
lattice and reduce with LLL (`lattice_solve`). This is how GPS receivers resolve carrier-phase cycle counts. It needs no
curriculum and no easy rows.

Rows are random T=1 rows from H1's training set. Each count is the number of trials (out of 3) that found exactly s²;
every success is also exact on all T=1 rows of that set. Times use python-flint's LLL; sympy's pure-Python LLL gives the
same answers about 250× slower.

| features | rows: trials found exactly s² | time per trial |
|---|---|---|
| value: (1, s, s²), 3 coefficients | 2: 3/3, 3: 3/3, 5: 3/3, 10: 3/3 | < 0.5 s |
| digits, free place values (1, d_i, d_i d_j); 13-bit rows (4-digit x, 15 coefficients) | 20: 0/3, **30: 3/3**, 40: 3/3, 60: 3/3 | < 2 s |
| same; 16-bit rows (5 digits, 21 coefficients) | 30: 0/3, **40: 3/3**, 60: 3/3 | < 5 s |
| same; 20-bit rows (6 digits, 28 coefficients up to 10^10) | **60: 3/3**, 80: 3/3, 100: 3/3, 150: 1/1 | 5–190 s |
| same; all 1,000 T=1 rows mixed (28 coefficients) | **60: 3/3**, 100: 3/3, 150: 3/3 | 2.4 s / 15–20 s / 110–130 s |

- In the value-feature trials, even the least-wrapped row of a trial wraps up to 8,179 times. In the digit trials,
  almost every row wraps; the least-wrapped row is usually at 2–113.

- The rows needed: the wrong solutions sit about (∏ N_i)^(1/p) away in a p-coefficient lattice, and they must be
  farther than the true coefficients. That means m ≳ p · log₂(c_max) / log₂ N rows. At 13 bits that is 15 · 20 / 13 ≈
  23 rows: 20 failed and 30 worked. At 16 bits it is ≈ 35 (30 failed, 40 worked); at 20 bits ≈ 46 (60 worked).
- The width alias remains. H1's training x have at most 6 digits, so these 28 coefficients never include the 10^6
  and 10^7 places that 22- and 24-bit moduli need. That is the same limit as RELAXED.md's s²-supervised digit model.
- The free-place-value model is exactly the one gradient methods cannot learn from answers (§3, §4). Exact integer
  search learns it from a few dozen arbitrary rows.
- This is REPORT.md's mechanism (labels select among exact hypotheses) on a hypothesis space where the search is
  polynomial (LLL), not enumerative.
- **Usable on H1?** Yes: any rows, no x² labels. The catch is that the hypothesis must be linear in integer
  coefficients over designed features: monomials in s or in the digits of s.

## 6. Takeaways
1. **"Unfold to suit the gradient" and "a loss that points the right way" are the same thing here.**
   - The clock loss regresses toward the model's own unfolding, so it points the right way exactly where the model's
     wrap count is right.
   - The wrap count needs s to every digit (test 2), so no coarse signal widens the basin for free.
2. **What extends the basin is shared structure, not a cleverer loss.**
   - If every parameter is pinned by rows that don't wrap, unfolding carries fold to fold: value inputs, or digits read
     through one radix. That works on H1's own rows and certifies T=64.
   - Free place values are never pinned inside the basin, so no gradient method learns them from answers.
3. **The sensitivity-weighted clock loss is the one loss-level gain.** For a single tied parameter, the recovery range
   goes from 0.3 N to 100 N. It does not help where information is missing.
4. **Analytically, the wraparound is a solved problem.**
   - Integer ambiguity resolution unfolds all rows at once, from arbitrary rows: 2 for (1, s, s²), 60 for a general
     quadratic form in the digits.
   - It even learns free place values that gradient methods can't, in seconds.
   - It is exact search, not gradient descent, and it needs a hypothesis linear in integer coefficients.

## 7. Plain-language summary
The label only tells the network where the square lands on a clock face with N hours. It never says how many full turns
the square made. The natural loss, distance on the clock, quietly assumes the network already knows the number of
turns. So it works when the network's guess is within half a turn of right, and points somewhere random otherwise.

Knowing the first few digits of the number doesn't help: the number of turns depends on every digit. What does help is
building up from small numbers that never go all the way around, if the network has no knob that only big numbers turn.
A network that reads digits through one shared "base ten" knob can learn everything from H1's five tiny examples and
then handle every size. A network with a separate knob per digit position can't: the knobs for the big positions are
only ever turned by examples that wrap hundreds of times.

There is also an exact, non-gradient way. Treat each example's number of turns as an unknown whole number and solve for
all of them at once with lattice reduction, the trick GPS receivers use to count carrier-wave cycles. It recovers "x
squared" from two random examples, and even the full digit-by-digit place values from about sixty.

## 8. Reproduce
From `cpu_first/` (fp64, CPU; each line takes seconds to minutes):

```bash
python unfold.py basin --iters 20 --lm_weight sens
```

```bash
python unfold.py topdigits
```

```bash
python unfold.py fold
```

```bash
python unfold.py h1 --seeds 5 --stage0 plain
```

```bash
python unfold.py rung --rung_init low --data h1 --rung_kinds value tied --schedules oracle self --seeds 5
```

```bash
python unfold_lattice.py --feats digits --rows 60 --seeds 3
```

On the shared laptop, run these on Modal instead. `unfold_modal.py` runs one CPU container per line of a command list
(`<name> <script> <args>`) and writes `runs/unfold/<name>.log` and `.json`:

```bash
modal run unfold_modal.py --commands runs/unfold/modal_rung_low.txt
```

`unfold_lattice.py` needs only numpy, plus python-flint for fast LLL (`pip install python-flint`; without it, sympy is
used). `unfold.py lattice` calls the same code.

Raw results: `runs/unfold/*.json` and `*.log`.
