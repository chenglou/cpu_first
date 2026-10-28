# Learning which formula: self-unfolding for an unknown recurrence

**Question.** UNFOLD.md learned the step s → s² mod N from answer-only rows: y only, never s². The competition page
warns that a "Generalised Hard" version may change the recurrence itself. Can the same two methods find *which*
polynomial step made the labels, without a hand-written menu like menu.py's 24 programs? The two methods are:
- (a) the clock loss with easiest-first ordering;
- (b) lattice reduction that solves every wrap count at once.

**Setup.**
- **Hidden steps.** menu.py's s³, s² + 1, s²·x0, s² + x0 and s³ + 2 (mod N), plus H1's own s² as a control.
  - Each one relabels H1's (N, x, T) rows: y = f^T(x mod N).
  - Training rows are H1's 1,000 T=1 and 1,000 T=2 rows. T=4 rows are used only where stated.
- **Model.** f(s, x0, N) = Σ c·s^a·x0^b·N^c over all 20 monomials with a + b + c ≤ 3, with integer coefficients.
  - The 10 monomials that contain N are multiples of N, so they never change a label or the recurrence. That leaves
    10 monomials in (s, x0).
  - The family contains every hidden step and 20 of menu.py's 24 programs. The 4 left out are s³·x0 + a, which have
    degree 4.
  - With |c| ≤ C it has (2C+1)^10 members: 59,049 at C = 1, about 10^33 at C = 1,000.
- **Digit variant.** s and x0 enter as Σ B^i d_i over their decimal digits, with one shared base B (UNFOLD.md's radix
  model).
- **Success.** Run the learned step with exact integers. It must be 100% right on every rung T = 1 … 64 of both H1
  profiles, relabeled under the hidden step:
  - depth_t: 13/16/20-bit moduli;
  - depth_ood_n_t: 14/18/22/24-bit moduli.
  - Gradient models are read out by rounding their weights and radix to integers.
- **Code.** `unfold_family.py` (numpy + python-flint; it reuses `unfold_lattice.lattice_solve`).
  - Every run was on Modal through `unfold_family_modal.py`: about 80 CPU containers, each finishing in 2–100 s.
  - Raw results are in `runs/unfold_family/`. `python unfold_family.py report` prints every table below.

**TL;DR**
- **Lattice reduction learns which formula from 2–3 answer-only rows.**
  - **One random T=1 row gives the diagonal f(x, x) exactly:** 714/720 trials, and 2,880/2,880 with 2 or more rows.
  - **T=1 rows can't tell s from x0**, because s = x0 there. For example s², s·x0 and x0² give the same labels.
    - The lattice returns one of the equally short candidates. 174/600 certify, which is chance level.
    - An s-only prior (assume the step never reads x0) certifies the 4 steps that don't read x0 (399/400). It learns
      s³ for s²·x0 and s² + s for s² + x0 (0/200).
  - **One T=2 row fixes it.**
    - The stage-1 model's own T=1 answer is that row's middle state, where s ≠ x0. One more lattice then pins every
      coefficient.
    - 2 T=1 + 1 T=2 rows certify all six hidden steps: 2,400/2,400 trials over every cell with ≥ 2 T=1 and ≥ 1 T=2
      rows. 1 + 1 rows: 120/120.
    - A trial takes 1.7 ms.
  - **No menu needed.** Random hidden steps with coefficients up to 1, 3, 10, 100 and 1,000 need 2, 2, 3, 4 and 6 rows
    of each kind (50/50).
  - **Digits through one shared base.** Scanning B = 2..16 picks B = 10 alone from 2 T=1 rows (720/720). Stage 2 then
    proceeds as with value features: 540/540 certified with T=2 rows.
- **T=2 rows leave one blind spot.**
  - When g(x) = f(x, x) is quadratic (square, square + 1, square + x0), f + k·(s − x0)(s − g(x0)) matches f on every
    T=1 and T=2 row (1,000/1,000 each). It matches ≤ 0.1% of T=4 rows and certifies only T=2.
  - The lattice returns the shortest member, and for all three menu steps that is the hidden one.
  - If the hidden step is the longer member, the lattice returns the shorter one (0/20). One T=4 row fixes it (20/20).
- **The gradient (clock loss + easiest-first) does not learn which formula.**
  - **H1's own rows, full 20-monomial model:** 0/5 for every hidden step, order and weighting.
  - **Without the N-monomials:** 5/5 for square and square + 1, but only with an oracle order (the true wrap counts).
    The label-free order gets 0/5 everywhere.
  - **One shared radix:** 2–3/5 on the same two steps, oracle order only.
  - **Calibration:** UNFOLD.md's (1, s, s²) model certifies 5/5 here with every order.
  - **Fresh rows with a fold curriculum** (plenty of non-wrapping rows): 0/5 in all 72 cells. The diagonal is learned
    at 12–16 bits; the rest of the formula never is.
  - **Freezing the learned diagonal and training only the rest on T=2 rows:** 1/120 fold runs. On H1 it does the same
    as without freezing.
- **Why the lattice succeeds where the gradient fails.**
  - T=1 rows carry no signal that separates s from x0.
  - The blind spot is exact mod N, so no T≤2 label can move it.
  - The clock loss measures error in turns of the clock, and exactness needs an error under 1/(2N) of a turn. LM
    happily gives up exactness on many rows to fit a few.
  - The oracle's two successes come from fitting T=2 rows whose last step does not wrap. "This row did not wrap" is a
    fact about the integer, which the mod-N labels don't carry. That fact even settles the blind spot.

## 1. Lattice, stage 1: T=1 rows give the diagonal, not the formula
- **Stage 1:** solve c·φ(x0, x0, N) ≡ y (mod N) on m1 random T=1 rows.
  - This is UNFOLD.md §5's Kannan embedding and LLL, now with 20 unknown coefficients.
  - The columns are put in a random order each trial, because LLL breaks ties by basis order.
- **At T=1 the state is s = x0.** Monomials s^a·x0^b with the same a + b give identical columns. So the rows only see the
  diagonal g(x) = f(x, x) = d0 + d1·x + d2·x² + d3·x³: 4 integers.
- **The rows are random.** They wrap thousands to millions of times: the median wrap count at T=1 is 8,250 for the
  squares and 1.9·10^8 for the cubes. There is no curriculum.

Certified out of 20 trials, T=1 rows only:

| hidden step | 1 row | 2 rows | 3 rows | 4 rows | 6 rows | formulas returned (all 100 trials) |
|---|---|---|---|---|---|---|
| s² | 10 | 7 | 7 | 6 | 8 | s² 38, x0² 31, s·x0 31 |
| s³ | 5 | 7 | 2 | 4 | 5 | s·x0² 28, s²·x0 27, s³ 23, x0³ 22 |
| s² + 1 | 8 | 7 | 6 | 9 | 10 | s² + 1 40, x0² + 1 31, s·x0 + 1 29 |
| s²·x0 | 4 | 6 | 8 | 3 | 5 | x0³ 33, s²·x0 26, s³ 22, s·x0² 19 |
| s² + x0 | 5 | 3 | 1 | 5 | 2 | s·x0 + x0 23, x0² + x0 21, s² + x0 16, s² + s 16, s·x0 + s 13, x0² + s 11 |
| s³ + 2 | 7 | 6 | 8 | 6 | 4 | s³ + 2 31, x0³ + 2 25, s²·x0 + 2 23, s·x0² + 2 21 |

- **Every returned formula has the right diagonal.**
  - The diagonal is exact on all 1,000 T=1 rows in 120/120 trials that used a single row and no T=2 rows.
  - Across every cell with one T=1 row it is 714/720; with ≥ 2 T=1 rows it is 2,880/2,880.
- **Which of the equally short formulas comes back is chance.**
  - Chance is 1/3 for s² and s² + 1, 1/4 for the three cubic diagonals, and 1/6 for s² + x0. That predicts 158/600;
    the observed count is 174/600.
  - More T=1 rows don't help. They all have s = x0, so they all see the same diagonal.
- **The 6 wrong diagonals all came from a single T=1 row,** 5 of them for s³ + 2.
  - A different polynomial of the same length fits one 13-bit row. One example is −s²·x0 − x0³ + x0² − s − x0, of length
    √5 like s³ + 2.
  - From one b-bit row, the shortest wrong diagonal is about (2^b)^(1/4)/2 ≈ 2–5 long. Two rows push that to 20–1,000.

**An s-only prior.** Suppose the model drops x0, keeping only the monomials in s and N. Then T=1 rows have no ties.
Certified out of 20, T=1 rows only:

| hidden step | 1 row | 2 | 3 | 4 | 6 | formula |
|---|---|---|---|---|---|---|
| s², s³, s² + 1 | 20 each | 20 | 20 | 20 | 20 | the hidden one |
| s³ + 2 | 19 | 20 | 20 | 20 | 20 | s³ + 2 |
| s²·x0 | 0 | 0 | 0 | 0 | 0 | s³ |
| s² + x0 | 0 | 0 | 0 | 0 | 0 | s² + s |

That is the "s·x0 ≡ s² at T=1" problem from menu.py, seen from the learner's side. T=1 labels can't say whether the
step reads x0.

## 2. Lattice, stage 2: the model's own answer is the T=2 row's middle state
- **Stage 2:**
  - For each of m2 random T=2 rows, run the stage-1 model once on (x0, x0). That gives s1 = g(x0) mod N, which is exact
    whenever the diagonal is.
  - Then solve the lattice again on the T=1 rows (features at (x0, x0)) plus the T=2 rows (features at (s1, x0)).
    Here s1 ≠ x0.
- **No label for s1 is used.** This is self-unfolding in time: the model's own T=1 answer unrolls the T=2 rows.

Certified out of 20 per cell (m1 T=1 rows, m2 T=2 rows):

| hidden step | m1 = 1, m2 = 1 | m1 = 1, m2 = 2, 3, 4, 6 | m1 = 2, 3, 4, 6 × m2 = 1, 2, 3, 4, 6 |
|---|---|---|---|
| s² | 20 | 80/80 | 400/400 |
| s³ | 20 | 80/80 | 400/400 |
| s² + 1 | 20 | 80/80 | 400/400 |
| s²·x0 | 20 | 80/80 | 400/400 |
| s² + x0 | 20 | 79/80 | 400/400 |
| s³ + 2 | 20 | 75/80 | 400/400 |

- **Every success is the hidden formula,** after dropping N-monomials: 5,436 certified trials, 5,436 hidden formulas
  over §1, §2 and §4. About 10% of solutions carry ±1 on an N-monomial. That is an exact multiple of N, which changes nothing.
- **The 6 misses at m1 = 1 are wrong stage-1 diagonals** (§1). The middle states are then wrong, and the
  constraints contradict each other:
  - in 4 trials stage 2 returns a long junk polynomial, for example 3s³ − 11s²·x0 + … with squared length 480;
  - in 2 it returns no solution at all.
  - Every miss fits under 0.2% of the T=1 rows, so a held-out row would flag it.
- **Time** (python-flint LLL, both stages): median 1.7 ms per trial, 99th percentile 7 ms, maximum 144 ms over 3,600
  trials.
- **Comparison with menu.py.** menu.py selected among 24 designed programs using 400 T≤2 rows and certified 6/6. The
  lattice uses 2–3 rows and no list: any degree-≤3 polynomial with modest coefficients.

## 3. The T≤2 blind spot, and T=4 rows
- **When two steps agree on T=1 rows.** f and f′ agree on every T=1 row exactly when f − f′ = (s − x0)·q, i.e. it
  vanishes at s = x0.
- **When they also agree on T=2 rows.** They agree on every T=2 row too only if q also vanishes at s = g(x0) mod N.
  - Within total degree 3, that forces q = k·(s − g(x0)), and deg g ≤ 2.
  - So the one change no T≤2 label can see is k·h, with h = (s − x0)(s − g(x0)).
  - It exists exactly when the diagonal is at most quadratic.
  - For s³, s²·x0 and s³ + 2, h has degree 4, so there is no blind spot inside the family.

| hidden step | neighbour f + k·h | k | squared length (hidden) | T=1 rows | T=2 rows | T=4 rows | certified max T |
|---|---|---|---|---|---|---|---|
| s² | s·x0² − x0³ + s·x0 | −1 | 3 (1) | 1.000 | 1.000 | 0.0008 | 2 / 2 |
| s² | −s·x0² + x0³ + 2s² − s·x0 | +1 | 7 (1) | 1.000 | 1.000 | 0.0010 | 2 / 2 |
| s² + 1 | s·x0² − x0³ + s·x0 + s − x0 + 1 | −1 | 6 (2) | 1.000 | 1.000 | 0.0006 | 2 / 2 |
| s² + 1 | −s·x0² + x0³ + 2s² − s·x0 − s + x0 + 1 | +1 | 10 (2) | 1.000 | 1.000 | 0.0004 | 2 / 2 |
| s² + x0 | s·x0² − x0³ + 2s·x0 − x0² + x0 | −1 | 8 (2) | 1.000 | 1.000 | 0.0007 | 2 / 2 |
| s² + x0 | −s·x0² + x0³ + 2s² − 2s·x0 + x0² + x0 | +1 | 12 (2) | 1.000 | 1.000 | 0.0006 | 2 / 2 |

- **The lattice chooses by length** (Occam). For the three menu steps the hidden one is the shortest.
- **Adversarial hidden steps.** Here the hidden step is the longer neighbour: 2s² − s·x0² − s·x0 + x0³, and
  s·x0² − x0³ + 2s·x0 − x0² + x0.
  - With 2 T=1 + 2 T=2 rows the lattice returns s² and s² + x0: 0/20 certified each.
- **Stage 3.**
  - Try c + k·h over |k| ≤ 20, shortest first, and keep the first that matches m4 random T=4 rows.
  - This is exact evaluation, no lattice.
  - With one T=4 row it finds k = +1 and −1: 20/20 each. Two T=4 rows: 20/20.
- **Is this real?** The extra T=2 rows are not wasted, since they fix everything but k. For a recurrence family
  that really is unknown, the safe pipeline is T=1 → T=2 → one T=4 row.

## 4. Beyond the menu: random hidden steps
- **The hidden steps.** For each coefficient range [−C, C]: 10 dense random polynomials over the 10 (s, x0) monomials,
  keeping only steps that read s.
- **The trials.** 5 trials each, with m T=1 + m T=2 rows.

Certified out of 50 trials:

| coefficients in | m = 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | time at m = 24 |
|---|---|---|---|---|---|---|---|---|---|---|
| [−1, 1] | 49 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 1.15 s |
| [−3, 3] | 21 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 1.15 s |
| [−10, 10] | 0 | 44 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 1.27 s |
| [−100, 100] | 0 | 0 | 34 | 50 | 50 | 50 | 50 | 50 | 50 | 1.20 s |
| [−1000, 1000] | 0 | 0 | 0 | 21 | 50 | 50 | 50 | 50 | 50 | 0.98 s |

- **Every success is the hidden polynomial.**
- **Rows needed grow with log C.** UNFOLD.md §5's rule, applied per stage:
  - the diagonal (4 coefficients of size up to about 4C) needs m1·b/4 ≫ log₂(4C) bits;
  - the rest (6 coefficients of size about C) needs m2·b/6 ≫ log₂ C bits.
  - At b ≈ 16 bits and C = 1,000 that is about 3 and 4 rows, plus a margin. 6 rows was the first count that worked
    every time.
- **Time.** Under 10 ms up to m = 6. LLL in dimension 69 (m = 24) takes about 1 s.
- **No blind-spot failures.** None appeared: a dense random step has a cubic diagonal almost surely.

## 5. Digits through one shared base
- **The model.** s and x0 are read from their decimal digits as Σ B^i d_i.
- **The scan.** B is not linear in the unknowns, so it is scanned.
  - Run one stage-1 lattice per B = 2..16 on the same T=1 rows, and keep the B whose solution is shortest.
  - Stage 2 then runs with that B.

Base 10 picked alone, and certified, out of 10 (m1 T=1 rows, m2 T=2 rows):

| hidden step | m1 = 1, m2 = 0 | 1, 1 | 1, 2 | 1, 3 | 2, 0 | 2, 1 | 2, 2 | 2, 3 | 3–4, any m2 ≥ 1 |
|---|---|---|---|---|---|---|---|---|---|
| s² | 10, 1 | 10, 10 | 10, 10 | 10, 10 | 10, 4 | 10, 10 | 10, 10 | 10, 10 | all 10, 10 |
| s³ | 10, 0 | 9, 9 | 10, 10 | 10, 10 | 10, 0 | 10, 10 | 10, 10 | 10, 10 | all 10, 10 |
| s² + 1 | 9, 1 | 8, 10 | 9, 9 | 10, 10 | 10, 4 | 10, 10 | 10, 10 | 10, 10 | all 10, 10 |
| s²·x0 | 10, 3 | 10, 10 | 10, 10 | 10, 10 | 10, 1 | 10, 10 | 10, 10 | 10, 10 | all 10, 10 |
| s² + x0 | 10, 3 | 10, 10 | 9, 9 | 9, 9 | 10, 1 | 10, 10 | 10, 10 | 10, 10 | all 10, 10 |
| s³ + 2 | 8, 2 | 9, 9 | 9, 9 | 7, 8 | 10, 3 | 10, 10 | 10, 10 | 10, 10 | all 10, 10 |

- **Choosing B.**
  - From one T=1 row, B = 10 is picked alone in 226/240 trials. The wrong picks were 2, 3, 5, 7 and 13. With one row
    the wrong bases also find short solutions: lengths 1.7–12, against 1.0–2.2 at B = 10.
  - From 2–4 rows: 720/720.
  - With m2 ≥ 1, certified 172/180 at m1 = 1 and 540/540 at m1 ≥ 2.
  - The T=1 tie (s vs x0) is the same as in §1.
- **No width alias.** The place values for 10^6 and 10^7 are B^6 and B^7, so the 22–24-bit profile certifies.
  - UNFOLD.md's free place values had that alias.
  - A free-place-value version of this degree-3 family would need 455 coefficients, the monomials of degree ≤ 3 in 12
    digits. Not attempted.
- **Time.** About 10 ms per trial (15 lattices).

## 6. Gradient: clock loss + easiest-first
**Protocol.** UNFOLD.md §4, extended to two phases.
- **Models** (fp64 weights on the same monomials):
  - poly20: all 20 monomials;
  - poly10: without the N-monomials;
  - tied10 and tied20: s and x0 read through one learned radix B, starting in [1.5, 4] as in UNFOLD.md;
  - value3: UNFOLD.md's (1, s, s²), as a calibration.
- **Phase A, T=1 rows easiest first.**
  - Stage 0 fits the rows the order calls non-wrapping, by plain regression.
  - Then 48 clock-loss LM stages grow the prefix from 5 to 1,000 rows.
- **Phase B, all T=1 rows plus a growing prefix of T=2 rows.**
  - Each T=2 row's middle state is the model's own snapped T=1 answer, computed exactly and recomputed every stage.
  - Stage 0 again fits the non-wrapping rows.
- **Orders.**
  - Oracle: sorted by the true wrap count under the hidden step. Stage 0 then uses the rows that truly don't wrap:
    1–5 T=1 rows and 1–11 T=2 rows.
  - Label-free: x0³/N for T=1 rows and max(ŝ1, x0)³/N for T=2 rows. Its stage 0 uses the rows with x0³ < N, which
    is a single row.
- **Weighting.** None, or UNFOLD.md's sensitivity weighting.
- **Readout.** Round every weight and the radix to the nearest integer, then certify with exact integers.

Certified out of 5 seeds, as "oracle order (no weighting, sensitivity) / label-free order (no weighting, sensitivity)":

| hidden step | poly20 | poly10 | tied10 | tied20 | value3 |
|---|---|---|---|---|---|
| s² | 0, 0 / 0, 0 | **5, 5** / 0, 0 | 2, 2 / 0, 0 | 0, 0 / 0, 0 | **5, 5 / 5, 5** |
| s³ | 0, 0 / 0, 0 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | |
| s² + 1 | 0, 0 / 0, 0 | **5, 5** / 0, 0 | 3, 2 / 0, 0 | 0, 0 / 0, 0 | |
| s²·x0 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | |
| s² + x0 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | |
| s³ + 2 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | |

- **The fp64 model after phase A** (oracle order) is exact on all 1,000 T=1 rows:
  - for s² and s² + 1: poly10 5/5, tied10 2–3/5;
  - for s² + x0: poly10 5/5, tied10 3/5, but phase B then breaks it: 0/5 certified;
  - for the cubic diagonals: 0/5 everywhere, with 1 non-wrapping T=1 row for 4 diagonal coefficients;
  - poly20: 0/5 everywhere, even on s². With real weights the N-monomials are not invisible, so T=1 rows see 10
    distinct functions (1, x, x², x³, N, x·N, …, N³) against at most 5 non-wrapping rows.
- **Time.** 1.1 s per seed (median), 3.2 s at most.

**Fresh rows, fold curriculum** (UNFOLD.md test 3):
- 2,048 fresh rows per stage, at 12, 16, 20 and 24 bits.
- Phase A uses T=1 rows with x0 < N^e, e from 0.1 to 1, over 32 stages.
- Phase B uses full-range T=1 rows plus T=2 rows with x0 < N^e, e from 0.05 to 1, over 32 stages.
- **Result:** 0/5 in all 72 cells (6 steps × poly20, poly10, tied10 × 4 sizes).
- **Phase A at 12 and 16 bits** learns the diagonal exactly: 5/5 in 33 of 36 cells. tied10 misses the cubic diagonals
  at 12 bits.
- **Phase A at 20 and 24 bits** never learns it.
  - The low-order weights drift once rows wrap about 10^3 times. In a 20-bit s² run the constant goes
    0 → −0.004 → 2.4 → −22, while every row stays within N/2 of the truth.
  - This is an fp64 LM limit with x³ columns, like the one UNFOLD.md's rung head hit at 22–24 bits.
- **Phase B breaks within 3 stages,** before any T=2 row wraps. In a 16-bit s² run:
  - the first stages see x0 ∈ {1, 2, 3};
  - LM fits them by moving the s and x0 weights from ±8.4 to ±23.5 (a direction T=1 rows don't see) and the constant
    from 0 to −6.5;
  - T=1 exactness falls from 100% to 27%, while the clock loss on those rows rises by only (6.5/N)².
- **Failing runs don't reproduce exactly across machines.** A 12-bit run repeated on the laptop matches Modal's
  per-stage trace until phase B breaks (1.0, 1.0, 1.0, 0.943, 0.985, then 0). After that it ends at different
  junk. The certifying H1 runs reproduce: same formula, weights within 4·10^-9.

**Frozen diagonal.** Write f = g(x0) + (s − x0)·q(s, x0). Every f with diagonal g has this form.
- Round g after phase A and freeze it. Train only q's 6 weights on T=2 rows with the clock loss.
- On H1 the label-free order sorts by |s1 − x0|·max(s1, x0)²/N, the size of the q part.
- **H1, oracle order:** s² 5/5, s² + 1 5/5. s² + x0 gets 0/5 even though its diagonal is right 5/5. The cubic steps get
  0/5 because their diagonal is wrong.
- **H1, label-free order:** 0/30.
- **Fold:**
  - the diagonal is right in 60/60 runs at 12–16 bits, 4/30 at 20 bits and 0/30 at 24 bits;
  - certified 1/120 (s², 12 bits);
  - quadratic diagonals mostly land on the blind-spot neighbour s·x0² − x0³ + s·x0, plus a small multiple of (s − x0);
  - cubic diagonals end with coefficients in the hundreds.

**Why the gradient fails.**
1. **T=1 rows give no signal** that splits s from x0. The split the gradient ends with is wherever the initialization
   and LM's minimum-norm steps leave it: non-integer, so rounding breaks it.
   - Example: after phase A of a 12-bit s² fold run, the weights are s² 0.579, s·x0 0.283, x0² 0.138, s −0.078,
     x0 +0.078.
   - The diagonal is exact (the weights sum to 1 and 0), but each one rounds to 0.
2. **The blind spot (§3) is exact mod N.** No T≤2 label prefers k = 0. The lattice prefers it by length; the gradient
   has no such preference that lands on integers.
3. **The clock loss measures error in turns.** A row is exact only if its error is under 1/(2N) of a turn.
   - Moving the constant by a few units costs each T=1 row about 10^-4 of a turn, which the loss barely registers.
   - That is enough to make every row wrong.
4. **The oracle order carries extra information.**
   - Every T=2 row the oracle calls "non-wrapping" has a wrapped middle state that happens to be small: 11/11 for
     s² and s² + 1, 4/4 for s² + x0. So regression on them sees f(s1, x0) as an integer, not a residue.
   - That breaks the blind spot (h(s1, x0) = −k1·N·(s1 − x0) ≠ 0 as an integer), which no label can do.
   - Without the oracle, the label-free order gets 0/5 for every step and every model except UNFOLD.md's 3-weight
     square.

## 7. Takeaways
1. **For the lattice, "which formula" is as easy as "the formula".**
   - An unknown degree-≤3 step adds 10 integer unknowns, and 2–3 random answer-only rows determine them in
     milliseconds.
   - There is no curriculum, no easy rows and no menu. Random steps with coefficients up to 1,000 need 12 rows.
2. **The kind of row matters in a new way.**
   - T=1 rows see only the diagonal f(x, x).
   - T=2 rows, unrolled through the model's own T=1 answer, see the rest.
   - T=4 rows settle the one blind spot, which only quadratic diagonals have. The shortest-vector rule picks the hidden
     step in every menu case.
   - So a learner restricted to T=1 rows can't answer menu.py's s·x0 vs s² question, and one T=2 row is enough.
3. **The gradient methods hit UNFOLD.md's limit, plus two new ones.**
   - UNFOLD.md's limit: they need non-wrapping rows for every parameter. The family has 10–20 parameters, while H1
     offers 1–5 non-wrapping T=1 rows.
   - Even with plenty of easy rows, the T=2 phase has a direction no label sees.
   - And it has an exactness the clock loss doesn't see.
4. **Limits.**
   - The hypothesis must be linear in integer coefficients over designed features, here monomials of degree ≤ 3.
   - Degree 4 (menu.py's s³·x0 + a) would mean 15 monomials in (s, x0), still a small lattice.
   - A step with a branch, a different modulus or a non-polynomial part is outside the family.
   - Like menu.py, this is a hypothesis space chosen by hand, but it is a space of about 10^33 steps rather than a list
     of 24.

## 8. Plain-language summary
The labels say where a hidden rule lands on a clock with N hours after T steps. In UNFOLD.md the rule was known to be
"square it", and the job was to recover the missing full turns. Here the rule itself is hidden. It could be "cube it",
"square it and add one", "square it and multiply by the starting number", and so on.

Write the rule as a short recipe with whole-number amounts of a few ingredients: s³, s²·x0, …, 1. Then solve for all
the amounts and all the turn counts at once with lattice reduction, the same trick as before.
- **One example after one step** pins down what the rule does when the current number equals the starting number.
- **It can't tell "s times s" from "s times the starting number",** because after one step those are the same number.
- **One example after two steps fixes that.** Run the recipe found so far once to get the middle number, which now
  differs from the starting number, and solve again.
- **Three examples in total** find every hidden rule, in about two milliseconds, with no list of candidate rules. Random
  rules with amounts up to a thousand need about twelve examples.

There is one blind spot. When the rule's one-step behavior is a square, a second, longer recipe agrees with it on every
one-step and two-step example and only parts ways at step three. The solver picks the shortest recipe, which was the
hidden one every time. When the hidden rule is the longer one, a single four-step example corrects it.

Gradient training on the clock loss doesn't get there. It needs one example that never goes around the clock for each
adjustable amount, and the H1 rows have only one to five. Even with plenty of such examples it can't tell s from the
starting number, and it can't see the blind spot. And the clock loss hardly notices the small errors that turn a right
answer into a wrong one.

## 9. Reproduce
From `cpu_first/`. Each line runs in seconds to two minutes on one CPU. numpy and python-flint only; without
python-flint, sympy's LLL is used and is much slower.

```bash
python unfold_family.py check
```

```bash
python unfold_family.py lattice --trials 20
```

```bash
python unfold_family.py lattice --family sn --m2 0 --trials 20
```

```bash
python unfold_family.py ambiguity --trials 20 --m4 0 1 2
```

```bash
python unfold_family.py random --trials 5
```

```bash
python unfold_family.py digits --m1 1 2 3 4 --m2 0 1 2 3 --trials 10
```

```bash
python unfold_family.py grad --models poly20 poly10 tied10 tied20 --seeds 5
```

```bash
python unfold_family.py fold --models poly20 poly10 tied10 --stages 32 --weights none --seeds 5
```

```bash
python unfold_family.py qgrad --seeds 5
```

On the shared laptop, run them on Modal instead: one CPU container per line of a command list, with results in
`runs/unfold_family/<name>.log` and `.json`. The lists used are `modal_lattice.txt`, `modal_grad.txt`,
`modal_fold.txt`, `modal_tied_rerun.txt` (the radix models, rerun after a fix), `modal_extra.txt` and
`modal_qgrad.txt`.

```bash
modal run unfold_family_modal.py --commands runs/unfold_family/modal_lattice.txt
```

The tables:

```bash
python unfold_family.py report > runs/unfold_family/report.md
```
