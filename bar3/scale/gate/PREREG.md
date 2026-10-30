# A gate that reads the numbers: pre-registration

Written 2026-10-29, before any Modal launch of this study. GATE.md records this file's hash and time. Development so far
was local: the engine and its exactness check (`check.py`: 14 jobs, the gated engine with no gate and with a gate that
delays nothing gives the original record field for field), and a signal check on a tiny corpus of training-fold targets
only (h1, rho, cube, affine at 1k/5k-state budgets: held-out AUC 0.62–0.72, toy b_eff 6.4–8.8 against 10). No
held-out formula's problem was looked at.

## The question

Bar 3's search grows 9–10 times per instruction over the primitives and about 40 times over a carried library
(SCALE.md §3). A fixed prior over tokens (§4.4's guided search) cut a search by a constant factor and changed no
outcome. Does a gate conditioned on the problem's rows and on the values each partial program produces (BUSTLE's
execution guidance; theory/GUIDED_SEARCH.md) cut the effective branching factor, and does that change what the learner
invents?

## The gate (code in `bar3/scale/gate/`)

- **Engine** (`gengine.py`): bar3/scale/engine.bfs with one change: each kept state gets a delay t from the gate and is
  expanded t levels later (its children cost t more). Delays are finite: reordered, never dropped. With no gate it is
  the original search.
- **Features** (`features.py`): property signatures (AllTrue, AnyTrue over the probes) of each register against the
  rows (= answer, = x, = N, = 0, < N, ≤ answer, ≥ answer, same parity as the answer, unchanged, constant in x at each
  N, depends on x only), the share right, whether the answer is still a function of (register, N), of the register, of
  the whole state; and the rows' own signature. Comparisons only: equality, order, the lowest bit.
- **Model** (`model.py`): a one-hidden-layer network (64 units), positives and negatives weighted equally. Delay =
  clip(round(κ·(ref − logit)), 0, D).
- **Training data** (`capture.py`, `gjobs.harvest_states`): the learner's own solved practice only. Round 0 of the
  consolidation arm `alloc_deep_ct2a2_probe` (the learner copy and config that kept 13 of the 16 earlier wins), its
  target probe and both probes (30k, then 1M states for the unsolved) searched for real, the 10M deep probe not run.
  Positives: the states each accepted program passes through. Negatives: up to 64 other states of each kept level below
  the solution's.
- **Two corpora.** C0: round 0 from scratch (35 primitives), the 20 formulas: the ten of SCALE.md (x² mod N, x² ± 1,
  x² + 3, 2x + 1, 10·x², 10·x² + 1, x³, Montgomery's, Fibonacci) and the ten held-out ones (x² + x, (x + 1)², 3·x², x⁴,
  x³ + x, 2^x, Lucas, Tribonacci, floor(x²/N), the triangular numbers). C1: round 0 over the 25-routine library the
  anchored-pairs sequence carried after 2x + 1 (`seq_ct2a2_clean_new_targets_L1.state.affine.json`), for the 15
  formulas not yet in it (10·x², 10·x² + 1, x³, Montgomery's, Fibonacci and the ten held-out ones).
- **Folds.** Problems posed the same way for every target (the law book, its depth-2 laws, dreams) are always
  training. The others belong to their target. Fold A = the SCALE formulas, fold B = the held-out ones. A gate trained
  on one fold is judged on the other fold's problems only. Validation targets inside each training fold choose the
  delay map (κ, ref, D) by the toy model's b_eff (`train.tune`): fold A Fibonacci and Montgomery's; fold B Lucas and
  floor(x²/N). One gate per corpus and fold (C0: b = 10, C1: b = 40), plus the ablations.
- **Ablations** (features masked): `noprob` (no rows' signature), `fitonly` (only the answer-equality features: a
  partial-fit heuristic, Probe's signal), `noanswer` (nothing that reads the answer: only x, N, 0, unchanged, constant),
  `nodet` (no determination features).

## Arms and measures

Every arm runs the same job (the learner's own posed problem, its instances and vocabulary) with a hard cap on states
(gengine "hard": the search stops when the running total passes the cap). "States" are candidates generated, counted
the same way in every arm; "states to solution" is the count when the first state with an accepted program appeared.
- `plain`: the search as the learner runs it (from the capture).
- `gate`: the gate of the other fold; `gate_prune`: with dead states never expanded (two probes with the same registers
  but different wanted answers: an exact test, nothing after such a state can fit); `prune`: that test alone.
- `guided` (C1 only; over the primitives it is the plain search): SCALE.md §4.4's learned costs.
- the ablations' gates.

Measures:
1. **Separation** on the test fold: AUC, the delays given to right and wrong states, the toy b_eff.
2. **States to solution** on test problems the plain search solves at cost ≥ 3: the ratio plain / arm, its geometric
   mean and the share of problems the arm solves in fewer states.
3. **Reach**: the slope of log10(states to solution) against the plain search's cost, per arm (1 / slope = instructions
   per tenfold of states; bits per tenfold for guided); and, of the test problems the plain search leaves unsolved, the
   share an arm solves within the states the plain search spent on them (capped at 10M).
4. **Outcomes**: whole learner runs, the consolidation arm (`alloc_deep_ct2a2_probe`; `alloc_ns_ct2a2_probe` for the
   triangular numbers and 10·x² + 1) with the gate as a second search wherever a probe (1M), the deep probe (10M) or the
   round's target search (3M) finds nothing, given as many states as the plain search spent; the gate trained on the
   other fold (C0's on an empty library, C1's over one). On the 16 earlier wins less binary x mod N (the digits
   machine has another engine; not covered): this arm's own records brought 13 of 16 (x² mod N, 2x + 1, x² ± 1,
   x² + x in round 0; x³, Fibonacci, x² + 3, 2^x, 3·x², (x + 1)², the triangular numbers in round 1; binary x mod N in
   round 1) and not 10·x², x⁴, 10·x² + 1 (stopped after 3–4 rounds). Plus the five held-out formulas no rule has brought
   (x³ + x, Lucas, Tribonacci, floor(x²/N), the triangular numbers with the held-out config). At most 5 rounds, stopping
   when the target comes.

## Predictions

| measure | prediction | confidence |
|---|---|---|
| separation, C0 test folds | AUC 0.70–0.85 for the full gate; `fitonly` ≤ 0.65; `noprob` within 0.03 of full | medium |
| toy b_eff, C0 | 6–9 (plain 10): +0.05 to +0.3 instructions per tenfold | medium |
| states to solution, C0, gate | geometric mean plain/gate between 1 and 3; the gate faster on 50–75% of problems, much slower (over 10×) on some | medium |
| states to solution, C1 | gate and guided each 1–3× on average; neither changes the slope | medium-low |
| prune | 1.2–1.7× fewer states at cost 5–6, the slope a few percent lower; never slower | high |
| reach (slope) | instructions per tenfold: plain 1.0 (C0), 0.6 (C1); gate within 1.0–1.3 (C0), 0.6–0.8 (C1) | medium |
| reach (plain-unsolved) | the gate solves at most 10% of the test problems the plain search leaves unsolved, at the plain search's states | medium |
| outcomes, the 13 wins | all 13 still come, none in a later round; at most 2 come a round earlier | medium-high |
| outcomes, 10·x², x⁴, 10·x² + 1 | none comes in 5 rounds | medium |
| outcomes, the five held-out non-wins | none comes | high |
| overall | the gate is a constant factor, like §4.4's prior, not a change of slope: no outcome changes | medium |

What would refute the overall prediction: a toy b_eff below 5 on the test fold with matching search measurements (more
than 1.5 instructions per tenfold), or any of the 8 non-wins invented with the gate.

## Money and limits

Soft $40, cap $60 (asked before passing it). `integration/ablation_s3/spend.py 4750` before each launch; at most 8
containers at once, each with a timeout and a hard memory limit; local runs only for checks (under a minute, under
1 GB). If money runs short, the outcomes come before the ablations, and the 8 non-wins before the 13 wins.
