# A gate that reads the numbers: can it order bar 3's searches well enough to reach longer programs?

*Written 2026-10-29/30. Code in `cpu_first/bar3/scale/gate/`, records in `bar3/scale/gate/records/`, theory in
[theory/GUIDED_SEARCH.md](../../../theory/GUIDED_SEARCH.md). Nothing outside those changed.*

<!-- PLAIN -->
## In plain words

**The question.**
- Bar 3's learner finds programs by brute search: every program of one instruction, then of two, and so on. Each
  instruction multiplies the states to search (9–10 times over the 35 primitives, SCALE.md §3), so programs much past
  7 instructions are out of its reach.
- BUSTLE (Odena et al. 2021) orders such a search with a model that reads what each partial program computes on the
  examples and puts the unlikely ones later. The coordinator asked whether a gate like it, trained only on the
  learner's own solved practice, orders bar 3's search well enough to reach longer programs, and whether that changes
  what the learner invents.

**1. On practice problems it was not trained on, the gate changes the search's slope.**
- The gate is a small network (64 hidden units). It reads comparisons only: does a register equal the answer, x, N or
  0, is it below N, is the answer still a function of the registers. It delays a doubtful state by a few levels.
  Nothing is dropped, so every program is still reached.
- It finds the same programs with 4.5 times fewer states from scratch, and 6.8 times fewer over the learner's library.
  The saving grows with the program's length: from 1.9× at 2 instructions to 7.7× at 5 from scratch, and from 2.8× at
  2 to 15.5× at 4 over the library.
- So each added instruction costs fewer states. From scratch it multiplies them by 5.2 instead of 6.7, and over the
  library by 14 instead of 29. That is 1.40 instructions per tenfold of states instead of 1.21, and 0.87 instead of
  0.68. With 300k states, the gate reaches programs two instructions longer: 7 against 5 from scratch, 5 against 3
  over the library.
- The learned prior over tokens (SCALE.md §4.4) gives a similar factor over the library (5.2×), but no growth with
  length. The literature expects that of a prior ([theory/GUIDED_SEARCH.md](../../../theory/GUIDED_SEARCH.md)).
- Over the library, the gate also solved 20% of the practice problems the plain search could not, within the same
  states. And it alone found a program for 2^x within 3M states.
- This is more than pre-registered, but less than the theory note's toy model gives for the gate's separation. The toy
  model gives 1.3–2.2 times the plain search's instructions per tenfold; the search gave 1.1–1.4 times. A gated state
  also costs about 5 times the wall time of a plain one, so on wall time the gate is clearly ahead only on longer
  problems: 5–6 instructions from scratch, 3–4 over the library.

**2. In whole learner runs, the gate changes what the learner invents, both ways. Most of its losses came through one
rule of the learner.**
- These are 20 runs of the consolidation's arm: its 12 wins, and 8 formulas it never brought. The gate runs as a
  second search wherever a plain search comes back empty, with at most the same states (and at most 4M). At most 5
  rounds, as registered. The 20 ran twice: as registered (§6), and with one rule of the learner changed (§9).
- *As registered.* One formula that no rule had brought came: the triangular numbers under the main setup
  (alloc_deep), in round 3. The same arm without the gate, rerun here, had nothing in 5 rounds. The other seven came
  neither with the gate nor without it.
- Of the 12 wins, one came a round early: (x + 1)², with a routine only the gate had solved. Three came in the same
  round: 2x + 1, Fibonacci and x² + 3.
- Four came later: x² − 1 (round 1 instead of 0), x² + x (2 instead of 0), 2^x (2 instead of 1), and the triangular
  numbers under the setup that won them (alloc_ns; 3 instead of 1).
- Four did not come in the 5 rounds: x² mod N and x² + 1 (round 0 without the gate), 3·x² and x³ (round 1).
- *With the anchors kept plain.* The arm practises pairs anchored on its two costliest solved practice problems, and as
  registered the gate's finds took those places. With those places kept for the plain search's finds, seven of the
  eight late or lost wins came back in their twin's round. 3·x² still did not come, and (x + 1)² again came a round
  early.
- In those runs x⁴ came, in round 1, which this arm had never brought (the held-out study did, with a subgoal). The
  triangular numbers under alloc_deep did not come.

**3. It changes them through what the learner practises, not by finding the targets.**
- In no run did the gate's own search find a target (0 of the 68 target searches it ran as registered, 0 of 42 with
  the anchors kept plain). Every target came from the plain search, over a library shaped by what the gate had solved.
- After an empty 1M probe of a practice problem, the gate's search ran 9,311 times over the 20 registered runs and
  found a program 356 times (3.8%), mostly at cost 6–7 (437 of 6,404 with the anchors kept plain).
- The arm practises pairs anchored on the two costliest learnable problems (anchored pairs). As registered, the gate's
  new cost-7 problems took those places.
- Without the gate, 11 of the 12 wins' programs call one cost-6 practice problem as a routine: x mod N
  (`law/mod(X,N)/x+qN`), picked in round 0 in all 11 runs. With the gate it was picked in round 0 in one of them
  (x² + 3), in round 2 in two (x² + x, and the triangular numbers under alloc_ns), and never in the other eight. Four
  of those eight came anyway, by other programs. The other four are the four wins lost. With the gate's finds kept out
  of the anchor places, it was picked in round 0 in all 11 again.
- The invention went the other way. Two practice problems only the gate had solved, (x·x) mod N and
  N mod (x + 1), gave the learner two routines in round 3: "add x into A" and "reduce A mod N". With them, the target
  was four tokens long (17 primitive instructions). It came through the anchor places: with the gate's finds kept out
  of them, the two were never picked, and the triangular numbers did not come.
- The gate's finds still change outcomes as partners of plain anchors. x⁴ came through (x·x) mod N, solved only by the
  gate in round 1, over the mod law picked in round 0. And keeping the anchors plain lost 3·x²: after a gate find
  changed its round-0 library, the two problems whose pair led its twin's round 1 were solved only by the gate, so
  that pair could not be scored.
- CONSOLIDATE.md saw the same interaction with its 10M probes: anchored pairs give the anchor places to whatever newly
  becomes solvable at the top cost.

**4. What this answers.**
- **Reach: yes, a measured change of slope, but a small one.** A gate trained only on the learner's own practice buys
  about 0.2 instructions per tenfold of states. β is 0.87 from scratch and 0.78 over the library, both with 90%
  intervals below 1. Each further instruction still multiplies the states by about 5 from scratch and 14 over the
  library.
- **Outcomes: changed, through the picks, and mostly through the anchor places.** As registered: one invention and one
  early win, against four wins late and four lost in the 5 rounds. With the anchors kept plain: one invention (x⁴)
  and one early win, against one win lost (3·x²). The pre-registered prediction ("a constant factor, no outcome
  changes") is refuted on both counts.
- **Next:** anchor the pairs by measured gain instead of by cost, the other way to separate the gate from the anchor
  rule. Keeping the anchors plain cost 3·x² its win; a gain rule would not ask which search solved a problem.
- **Cost:** $102.93 of the study's $1,000 cap, $23.07 of it for the runs with the anchors kept plain (Modal's billing
  for the app `bar3-gate`).
<!-- /PLAIN -->

## Details

### 1. Pre-registered

- [PREREG.md](PREREG.md): frozen 2026-10-29 21:03 PDT, read-only, sha256
  9f8613d5bf828d8e4d21b0f5b9853f2dedeb9380ecd48936dd44771753cb3ed1, before any Modal launch of this study.
- [PREREG_ADDENDUM1.md](PREREG_ADDENDUM1.md): frozen 21:17 PDT, read-only, sha256
  97fe057a5ef6aca7a0df0c7f40a4bacdcaf27baeb12f3612ebb3afc7d7171089, while the corpus captures ran, before any gate was
  trained: the whole runs' second search is capped at 4M states (the gated engine keeps every level it expands).
- [PREREG_ADDENDUM2.md](PREREG_ADDENDUM2.md): frozen 2026-10-30 11:08 PDT, read-only, sha256
  dd95f0a0fc320ac7099fe0fb39df176d7a07b5b56e474b84d763a9b9de480ec3, after the fifth launch's outcomes and before any
  run of §9: the 20 whole runs again, with anchored pairs anchored only on the plain search's finds.
- Deviations:
  - **The matched tables (§4).** Their slope axis and their cap rule were changed after the arms' first results were
    seen. Both corrections count against the gate. On C1 its β moved in three steps:
    - 0.67: a log-log slope on the first 231 problems, misses counted at the cap;
    - 0.74: on the cost axis, all 660;
    - 0.78: within the cap, 630.

    Measure 2 as registered (cost ≥ 3, misses at the cap) is in `records/report_c*.txt`.
  - **Four whole-run launches stopped and relaunched (§7),** each before any of its outcomes was seen. The runs reported
    in §6 are the fifth launch, at PREREG's "at most 5 rounds".
  - **Plain reruns of the eight non-wins (§6), added after round 0's outcomes were seen** (`pt1_a/b`, `learnrun` with
    no gate, 5 rounds). The five held-out formulas have no record of this arm, and the misses' records stop after 3–4
    rounds (the consolidation's spend stop). So a non-win that comes with the gate could not have been credited to it.
    This adds a comparison and changes no registered measure.

**Predictions against outcomes** (PREREG.md's table, in its order):

| measure | predicted (confidence) | came | right? |
|---|---|---|---|
| separation, C0 test folds | AUC 0.70–0.85 for the full gate; `fitonly` ≤ 0.65; `noprob` within 0.03 of full (medium) | AUC 0.937 and 0.950; `fitonly` 0.670 and 0.748; `noprob` within 0.01 | `noprob` yes; the rest no, separating better than predicted |
| toy b_eff, C0 | 6–9 (plain 10): +0.05 to +0.3 instructions per tenfold (medium) | b_eff 2.86 and 5.06; in search +0.19 (1.21 → 1.40) | b_eff no (lower); the search's gain yes |
| states to solution, C0, gate | geometric mean plain/gate 1–3; faster on 50–75%; over 10× slower on some (medium) | 4.46×; faster on 99%; slower on 2 of 191, by at most 1.5× | no: better on all three |
| states to solution, C1 | gate and guided each 1–3×; neither changes the slope (medium-low) | gate 6.82×, β 0.78 [0.74, 0.83]; guided 5.22×, β 0.97 [0.93, 1.01] | guided's slope yes; the factors and the gate's slope no |
| prune | 1.2–1.7× at cost 5–6, the slope a few percent lower; never slower (high) | C0 1.25× and 1.36× at cost 5 and 6, β 0.991; C1 2.07× (costs 2–4), β 0.920; never slower (0 of 821, 16 ties) | yes |
| reach (slope) | plain 1.0 (C0), 0.6 (C1); gate 1.0–1.3 (C0), 0.6–0.8 (C1) (medium) | plain 1.21 and 0.68; gate 1.40 and 0.87 | the gain over plain (+16%, +28%) inside the predicted (up to +30%, +33%); the values above |
| reach (plain-unsolved) | the gate at most 10% (medium) | C0 2.3%; C1 20.1% | C0 yes; C1 no |
| outcomes, the 13 wins (12 here: binary x mod N not covered) | all still come, none in a later round; at most 2 a round earlier (medium-high) | 3 the same round, 1 earlier, 4 later, 4 not in 5 rounds (x² mod N, x² + 1, 3·x², x³) | no |
| outcomes, 10·x², x⁴, 10·x² + 1 | none comes in 5 rounds (medium) | none in 5 rounds, nor in the plain reruns' 5 | yes |
| outcomes, the five held-out non-wins | none comes (high) | the triangular numbers (held-out config) in round 3, against none in the plain rerun's 5; Lucas, Tribonacci, floor(x²/N) and x³ + x none, nor in the reruns | no |
| overall | a constant factor like §4.4's prior, not a change of slope: no outcome changes (medium) | β 0.87 and 0.78, both intervals below 1; outcomes changed both ways | no |

The registered refutation was "a toy b_eff below 5 on the test fold with matching search measurements (more than 1.5
instructions per tenfold), or any of the 8 non-wins invented with the gate." Its first half did not happen. On the fold
where fold A's gate has b_eff 2.86, the search reached 1.37 instructions per tenfold (plain 1.06). Its second half did:
the triangular numbers under alloc_deep.

### 2. The gate

**The engine** (`gengine.py`). bar3/scale/engine.bfs line for line, with one change: every state a level keeps gets a
delay t from the gate, and it sources later levels as if it were t levels deeper (its children cost t more). The level
structure, the read-back of every cheapest program, the acceptance test and the near misses are the original's. Delays
are finite, so every state is still expanded: the search is reordered, never pruned (BUSTLE's reweighting, w' = w +
5 − δ). Two more budget rules beside the original's ("level"): "total" and "hard" (stop as soon as the running total
passes the cap, inside a level if need be), and three counts every arm shares: candidates generated, and the count when
the first state fitting every probe, and the first with an accepted program, appeared ("states to solution").

**Exact.** `check.py` runs 14 jobs of every kind the learner poses: law-book problems, the target's restrictions, the
target probe, and a target search over a carried 25-routine library. On all of them, the gated engine with no gate, and
with a gate that delays nothing, gives the original's record field for field. The only exceptions are the timings and
the gated engine's own counts.

**The price.** A gated record gives, as a plain one does, the price of its cheapest accepted program as `found_cost`,
with only the programs at that price. The gated level, which counts the delays, is kept as `gated_level`. The learner
reads `found_cost` as a problem's price (§7).

**The features** (`features.py`): property signatures (Odena and Sutton 2020) of a state against the problem's rows.
For each register (X, N, A, T, U), AllTrue and AnyTrue over the probes of: = the answer, = x, = N, = 0, < N, ≤ the
answer, ≥ the answer, the answer's parity, unchanged since the start, constant in x at each N, a function of x alone;
the share of probes right and the best share over the registers; whether the answer is still a function of (register,
N), of the register alone, and of the whole state. The rows' own signature: the answer < N, = x, = 0, ≤ x, ≥ x, x's
parity, constant in x at each N, a function of x alone; three shape flags. Comparisons only (equality, order, the lowest
bit): no number is made, and nothing names a target.

**The model** (`model.py`): one hidden layer of 64 units, positives and negatives weighted equally (BUSTLE pairs one
of each). Delay = clip(round(κ·(ref − logit)), 0, D): none for a state as likely as ref, one level per 1/κ nats of
evidence against it, at most D.

### 3. The corpus: the learner's own solved practice

`capture.py` runs round 0 of the consolidation arm (`alloc_deep_ct2a2_probe`, its learner copy unchanged) and writes
every search the round makes: the target probe, both probes of every posed problem (30k states, then 1M for the ones
the first leaves unsolved). The 10M deep probe is not run, and the round stops at its first look. From each search that
found an accepted program, `gjobs.harvest_states` keeps the training states:
- **positives:** the states its accepted programs pass through;
- **negatives:** up to 64 other states of each kept level below the solution's.

No target is made by Python arithmetic. A problem is the learner's own (its rows, vocabulary and acceptance test), and
its label is whether the learner's own search reached an accepted program through that state.

| corpus | runs | searches (distinct jobs) | solved problems: shared, formula-specific | costs | positives, negatives |
|---|---|---|---|---|---|
| C0: round 0 from scratch (35 primitives) | 20 | 8,945 (4,609) | 526: 252, 274 | 1–7 | 2,004, 67,688 |
| C1: round 0 over the 25-routine library | 15 | 6,629 (3,459) | 1,175: 279, 896 | 1–6 (5–6: 37) | 4,451, 95,159 |

C1's formulas are the 15 not in its library: 10·x², 10·x² + 1, x³, Montgomery's, Fibonacci and the ten held-out ones.
The library makes their problems shorter (most solved at cost 1–4).

**Folds.** Problems posed the same way for every target (the law book, its depth-2 laws, dreams) are always training.
The rest belong to their target:
- fold A: the ten formulas of SCALE.md;
- fold B: the ten held-out formulas.

A gate trained on one fold is judged only on the other fold's problems, and a whole run on a fold-A formula uses the
fold-B gate. Inside each training fold, two validation formulas pick the delay map by the toy model's b_eff, with no
retraining: fold A Fibonacci and Montgomery's, fold B Lucas and floor(x²/N).

### 4. Separation, states to solution and reach (measures 1–3)

**Separation** (measure 1), judged on the test fold before any search (`train.judge`, `records/models/judge_c*.json`).
"Right" states are the ones an accepted program passes through; the rest of each kept level is "wrong". Each ablation
cell gives AUC / toy b_eff.

| corpus, gate (judged on the other fold) | test problems | AUC | toy b_eff (plain b) | mean delay: right, wrong | noprob | fitonly | noanswer | nodet |
|---|---|---|---|---|---|---|---|---|
| C0, fold A's | 128 | 0.937 | 2.86 (10) | 0.68, 4.12 | 0.928 / 3.07 | 0.670 / 10.1 | 0.885 / 4.82 | 0.911 / 3.66 |
| C0, fold B's | 146 | 0.950 | 5.06 (10) | 0.02, 0.95 | 0.945 / 4.04 | 0.748 / 8.91 | 0.853 / 6.14 | 0.927 / 6.01 |
| C1, fold A's | 624 | 0.912 | 11.0 (40) | 1.03, 4.24 | 0.910 / 11.9 | 0.775 / 32.8 | 0.744 / 37.8 | 0.855 / 19.1 |
| C1, fold B's | 272 | 0.881 | 15.9 (40) | 0.69, 2.79 | 0.889 / 15.2 | 0.711 / 43.0 | 0.644 / 79.9 | 0.812 / 29.7 |

No single feature separates: the best alone, `live` (whether the answer is still a function of the whole state), has
AUC 0.63 on C0. The network's combination of them reaches 0.88–0.95 (§7). Fold B's C0 gate was tuned timid on its 16
validation problems (κ = 0.25, almost no delays), and it shows in the search below.

**States to solution and reach** (measures 2 and 3), with every arm on the same problems. These are the test problems
of both folds that the plain search solved at cost ≥ 2 within the arms' 5M cap (`report.matched`):
- C0: 191 problems;
- C1: 630 problems.

"Slope" is the least-squares slope of log10(states to solution) on the plain program's cost; 1/slope is the
instructions reached per tenfold of states. β is the arm's slope over the plain search's, with a 90% bootstrap interval
over problems. β < 1 means the arm's advantage grows by a constant factor per instruction; β = 1 means a constant
factor.

| corpus | arm | geo-mean plain/arm | median | faster on | > 10× slower | slope | instructions per tenfold | β [90%] | plain/arm by cost |
|---|---|---|---|---|---|---|---|---|---|
| C0 | plain | 1 | 1 | – | – | 0.825 | 1.21 | 1 | – |
| C0 | gate | 4.46 | 3.70 | 99% | 0 | 0.716 | 1.40 | 0.868 [0.826, 0.912] | 2: 1.9, 3: 3.0, 4: 4.7, 5: 7.7, 6: 7.4, 7: 2.6 |
| C0 | gate_prune | 4.55 | | 99% | 0 | 0.713 | 1.40 | 0.865 [0.824, 0.909] | |
| C0 | prune | 1.24 | 1.24 | 96% | 0 | 0.818 | 1.22 | 0.991 [0.985, 0.998] | 2: 1.1, 3: 1.2, 4: 1.3, 5: 1.3, 6: 1.4, 7: 1.0 |
| C0 | noprob | 5.59 | | 99% | 0 | 0.674 | 1.48 | 0.817 [0.781, 0.855] | |
| C0 | noanswer | 3.46 | | 99.5% | 0 | 0.711 | 1.41 | 0.862 [0.846, 0.879] | |
| C0 | nodet | 3.68 | | 98% | 0 | 0.735 | 1.36 | 0.891 [0.851, 0.931] | |
| C0 | fitonly | 1.61 | 1.29 | 90% | 0 | 0.798 | 1.25 | 0.968 [0.936, 1.000] | |
| C1 | plain | 1 | 1 | – | – | 1.467 | 0.68 | 1 | – |
| C1 | gate | 6.82 | 6.90 | 89% | 1.6% | 1.145 | 0.87 | 0.781 [0.740, 0.825] | 2: 2.8, 3: 7.7, 4: 15.5 |
| C1 | gate_prune | 6.82 | | 89% | | 1.145 | 0.87 | 0.781 [0.740, 0.824] | |
| C1 | guided (§4.4) | 5.22 | 6.38 | 90% | 1.4% | 1.418 | 0.71 | 0.967 [0.926, 1.007] | 2: 5.1, 3: 4.3, 4: 7.1 |
| C1 | prune | 2.07 | 1.97 | 99% | 0 | 1.350 | 0.74 | 0.920 [0.912, 0.929] | 2: 1.5, 3: 2.2, 4: 2.7 |
| C1 | noprob | 6.81 | | 90% | | 1.199 | 0.83 | 0.817 [0.779, 0.860] | |
| C1 | nodet | 3.99 | | 82% | | 1.237 | 0.81 | 0.844 [0.800, 0.888] | |
| C1 | fitonly | 1.70 | 1.92 | 77% | 3.8% | 1.269 | 0.79 | 0.865 [0.834, 0.894] | |
| C1 | noanswer | 1.20 | | 83% | | 1.382 | 0.72 | 0.942 [0.926, 0.957] | |

By fold, the gate's β is:
- C0: 0.896 [0.865, 0.930] on fold A's problems (fold B's gate), 0.776 [0.730, 0.823] on fold B's (fold A's gate);
- C1: 0.902 [0.829, 0.984] and 0.717 [0.672, 0.769].

The gate trained on SCALE.md's formulas carries to the held-out ones better than the reverse.

**Beyond the cap.** Some problems the plain search solved needed more than 5M states (a 1M probe streams its last
level). They are left out above because an arm's miss there says nothing:
- C0: 4 problems (cost 7, median 8.5M). Every gate arm solved all 4 within 5M; prune solved 1.
- C1: 30 problems (cost 5, median 26M). Solved within 5M:
  - the gate 28 and gate_prune 28;
  - nodet 27, noprob 27;
  - guided 21, fitonly 19;
  - noanswer 7, prune 7.

**Reach at a fixed budget** (`report.reach_table`, both folds): the largest cost up to which at least half of each
cost's problems are solved within B states.

| B | C0 plain | C0 gate | C1 plain | C1 guided | C1 gate |
|---|---|---|---|---|---|
| 30k | 3 | 5 (cost 5: 17/34; plain 0/34) | 3 | 3 | 3 (cost 4: 53/186; plain 1/186) |
| 300k | 5 | 7 (cost 6: 17/29; plain 2/29) | 3 | 4 | 5 (cost 5: 19/34; plain 1/34) |
| 3M | 7 | 7 (cost 6: 29/29; plain 21/29) | 4 | 5 | 5 (cost 5: 30/34; plain 2/34) |

**The plain search's leftovers** (measure 3). These are the test problems the plain search's 1M probe left unsolved,
given each arm the states the plain search spent on them (at most 5M):

| corpus | gate | gate_prune | prune | guided | programs the gate found |
|---|---|---|---|---|---|
| C0 | 44 of 1,911 (2.3%) | 47 | 0 | – | 5–10 tokens (7–8: 35) |
| C1 | 192 of 954 (20.1%) | 192 | 6 | 71 (7.4%) | 3–7 tokens (5–6: 170) |

**The targets at 3M states** (the target probe's job, every arm with a 3M cap):
- **C0, from scratch:** no arm finds any of the 20.
- **C1, over the library:** plain, prune and guided find the same 6: (x + 1)², x² + x, x⁴, 3·x², x³ and x³ + x. The gate
  finds 5 of them in 4× to 52× fewer states (x³ at 123k against plain's 2.18M; 3·x² at 54k against 2.09M; x³ + x at
  44k against 2.18M; x⁴ at 819 against 42k). The sixth, x² + x, it finds later (2,888 states against 1,195), with a
  first accepted program of price 3 against plain's 2.
- **2^x mod N:** the gate alone also finds it, `L.X[MOV.X.A T.res17] L.N[M0.6 DEC.X]` (price 6), at 1.17M states. It is
  right on every pair up to N = 40 (`library.inventing.jobs.general`; against x³ + x the same program scores 0.09).
  Plain, prune and guided find nothing in 3M.

**Pruning on top of the gate adds nothing** (gate_prune = gate within 2%): the gate already delays dead states, through
its `live` feature.

**Wall time.** This study counts states, and a gated state costs more: the features and the network run for every
kept state (numba, unoptimized). Take the jobs where the plain and gated arms both ran, on the same containers, with
over 50k states each (20 in C0, 12 in C1). A gated state took a median 5.1× (C0) and 4.5× (C1) the wall time of a plain
one. Against the matched geometric means (4.5× and 6.8× fewer states), that leaves the gate about even on wall time
from scratch (0.9×) and ahead over the library (1.5×). It is clearly ahead where its state ratio passes that overhead:
from scratch at costs 5 and 6 (7.7× and 7.4×), over the library at costs 3 and 4 (7.7× and 15.5×).

### 5. Dead states: an exact test the numbers allow

Each probe runs as its own machine, and its future depends on its five registers alone. So a state in which two probes
hold the same registers but want different answers can never be fixed by any later instruction: nothing after it fits
every probe. `gengine.dead_states` finds them (probes grouped by equal registers); with `prune` they stay in their level
(for the read-back) but are never expanded. This is a bound in SCALE.md §8's sense, and it was not among §8's.
Measured on the target search over the 35 primitives (N ≤ 7, 28 probes, local): 29% of level 1's states are dead, 37%,
39%, 41% and 42% of levels 2–5 (x² mod N; 2x + 1 up to 47%, Fibonacci 18–29%). Never expanding them:

| target | level 5, plain → pruned | level 6 (streamed), plain → pruned |
|---|---|---|
| x² mod N | 154,464 → 120,324 | 4,158,426 → 2,803,308 (1.48×) |
| 2x + 1 | 154,464 → 116,313 | 4,158,426 → 2,659,254 (1.56×) |
| Fibonacci | 154,464 → 132,174 | 4,158,426 → 3,268,214 (1.27×) |

So about 1.3–1.6× at level 6, and the growth per level only a little lower: most dead states die in the first
instruction or two (a CLR or MOV that overwrites the only copy of x), and their subtrees are what the pruning removes.
Like §8's other bounds, it is far from the 9–10× each instruction costs.

### 6. Outcomes: whole learner runs with the gate (measure 4)

Twenty runs of the consolidation arm, unchanged, with one change in its mapper (`learnrun.py`): wherever a 1M probe,
the 10M deep probe or the round's target search comes back with nothing, the gate searches the same job again with a
hard cap of the plain search's own spend or 4M states, whichever is smaller, and a program it finds (verified, as any)
replaces the empty record. The gates are cross-fitted (a fold-A formula's run uses fold B's gates; C0's gate on an
empty library, C1's over one). At most 5 rounds, stopping when the target comes. The twins are the arm's own records in
`consolidate/records` (tags ab1/ab2). The five held-out formulas have none: no rule brought them in the held-out study,
and this arm was never run on them. Launch `gt1s_a/b/c`, 3 containers (16 CPUs, 64 GB), 00:14 PDT Oct 30.

The eight non-wins were also run without the gate (`pt1_a/b`, 2 containers from 02:13 PDT; §1's deviation): the same
`learnrun` with no gates, 5 rounds. Where the consolidation has a record, the rerun repeats it exactly: the same picks,
library and target search in every round both have (10·x² rounds 0–3, x⁴ 0–3, 10·x² + 1 0–2). So a twin is this arm
without the gate, and every difference below comes from the gate.

Rounds are numbered from 0, and "no (k rounds)" means the target had not come after k rounds. "Kind" is a win or a
miss of this arm in the consolidation, or held out (no record of this arm). The last two columns are the gated run's.

| formula | config | kind | plain twin | with the gate | its picks only the gate solved, by round | second searches: found / run |
|---|---|---|---|---|---|---|
| 2x + 1 | alloc_deep | win | round 0 | round 0 | 4 | 18 / 136 |
| x² + 1 | alloc_deep | win | round 0 | no (5 rounds) | 2, 3, 0, 1, 0 | 15 / 942; target 0 / 5 |
| x² + x | alloc_deep | win | round 0 | round 2 | 3, 2, 0 | 15 / 355; target 0 / 2 |
| x² mod N | alloc_deep | win | round 0 | no (5 rounds) | 2, 1, 1, 0, 1 | 10 / 537; target 0 / 5 |
| x² − 1 | alloc_deep | win | round 0 | round 1 | 2, 0 | 12 / 298; target 0 / 1 |
| (x + 1)² | alloc_deep | win | round 1 | round 0 | 3 | 13 / 149 |
| 2^x | alloc_deep | win | round 1 | round 2 | 1, 4, 1 | 34 / 363; target 0 / 2 |
| 3·x² | alloc_deep | win | round 1 | no (5 rounds) | 1, 2, 3, 0, 1 | 24 / 466; target 0 / 5 |
| Fibonacci | alloc_deep | win | round 1 | round 1 | 2, 0 | 8 / 221; target 0 / 1 |
| triangular | alloc_ns | win | round 1 | round 3 | 4, 3, 0, 2 | 21 / 411; target 0 / 3 |
| x² + 3 | alloc_deep | win | round 1 | round 1 | 2, 1 | 14 / 349; target 0 / 1 |
| x³ | alloc_deep | win | round 1 | no (5 rounds) | 2, 2, 1, 2, 0 | 15 / 556; target 0 / 5 |
| 10·x² | alloc_deep | miss | no (4 rounds); rerun no (5 rounds) | no (5 rounds) | 2, 1, 2, 0, 0 | 9 / 546; target 0 / 5 |
| 10·x² + 1 | alloc_ns | miss | no (3 rounds); rerun no (5 rounds) | no (5 rounds) | 2, 1, 2, 1, 0 | 14 / 749; target 0 / 5 |
| x⁴ | alloc_deep | miss | no (4 rounds); rerun no (5 rounds) | no (5 rounds) | 3, 2, 0, 0, 1 | 13 / 629; target 0 / 5 |
| Lucas | alloc_deep | held-out | –; rerun no (5 rounds) | no (5 rounds) | 3, 2, 2, 0, 2 | 32 / 698; target 0 / 5 |
| Tribonacci | alloc_deep | held-out | –; rerun no (5 rounds) | no (5 rounds) | 2, 3, 2, 0, 1 | 26 / 716; target 0 / 5 |
| floor(x²/N) | alloc_deep | held-out | –; rerun no (5 rounds) | no (5 rounds) | 2, 1, 2, 0, 1 | 22 / 574; target 0 / 5 |
| triangular | alloc_deep | held-out | –; rerun no (5 rounds) | round 3 | 4, 3, 0, 2 | 18 / 385; target 0 / 3 |
| x³ + x | alloc_deep | held-out | –; rerun no (5 rounds) | no (5 rounds) | 1, 2, 2, 2, 1 | 23 / 554; target 0 / 5 |

**What came.**
- *The 12 wins.* 3 came in the same round (2x + 1, Fibonacci, x² + 3), 1 a round earlier ((x + 1)²), and 4 later
  (x² − 1, x² + x, 2^x, and the triangular numbers under alloc_ns). 4 did not come in 5 rounds: x² mod N and x² + 1
  (round 0 in their twins), 3·x² and x³ (round 1).
- *The 8 non-wins.* The triangular numbers under alloc_deep came in round 3; the plain rerun had none in 5 rounds.
  The other seven came in neither: none in 5 rounds with the gate, and none in the plain reruns' 5.
- *The programs.* Where a win came with the gate, its program was mostly as long as the twin's or shorter. In
  primitive instructions, gate against twin: 2x + 1 9 and 10, (x + 1)² 12 and 12, x² − 1 14 and 14, Fibonacci 13 and
  12, x² + 3 14 and 23, 2^x 14 and 20. It was longer for x² + x (17 against 12) and for the triangular numbers under
  alloc_ns (17 against 11).

**How the gate changes a run: through the picks.** In round 0, every gated run picked 1 to 4 problems that only the
gate had solved. The arm scores pairs of candidate routines by how much they compress the target. Every scored pair
must include one of the two costliest learnable problems (`anchor_cost: 2`, a hardest-first rule). The gate solves
cost-7 law-book problems that the plain 1M probe leaves unsolved, and two of them take the anchor places
(`law/absdiff(X,N)/x+qN` and `law2/max(add(X,X),N)` in most runs).
- *Four wins lost round 0 this way.* In the twins of x² mod N, x² − 1, x² + 1 and x² + x, the cost-6 law
  `law/mod(X,N)/x+qN` was an anchor. Its pair with the formula's other piece (`law/mul(X,X)/x`, `restrict/target/X·X<N`,
  `law2/succ(mul(X,X))`, `law2/mul(succ(X),X)`) found the target in its look and gained 66–91 bits. The round's program
  ends in `L.N[MOV.X.A T.mod(X,N)@x+qN]`. With the gate, neither member of that pair is an anchor, so the pair is never
  scored. The mod law's pairs with the new anchors gain at most 28 bits, and none finds the target in its look. The mod
  law is not picked, and the round's target search finds nothing.
- *Across the rounds.* Without the gate, the mod law was picked in round 0 for 11 of the 12 wins: all but 2x + 1,
  whose program needs no mod. Each of those 11 programs calls it as a routine, directly or through another. With the
  gate, of those 11 runs, it was picked in round 0 only for x² + 3, in round 2 for x² + x and the triangular numbers
  under alloc_ns, and never for the other eight. Four of the eight came anyway, by programs without it: (x + 1)² in
  round 0, x² − 1 and Fibonacci in round 1, 2^x in round 2. The other four, x² mod N, x² + 1, 3·x² and x³, did not
  come in 5 rounds. (Among the non-wins, the gated runs picked it in round 1 for floor(x²/N), in round 2 for the
  triangular numbers under alloc_deep, and in round 4 for x³ + x. The plain reruns picked it in round 0 for four of
  the eight, in round 2 or 3 for three, and never for x⁴.)
- *One win came a round early.* For (x + 1)², the gate solved `restrict/remainder[MOV.A.X INC.A -> A]/X·X<X+N` at
  price 9. Its first accepted program came at 1.2M states, within the 2.0M the plain search spent finding nothing. The
  learner picked it, and the round's target search found `INC.X L.X[T.res2 CLR.X]` with that routine as `T.res2`. The
  twin's came in round 1.
- *One non-win came: the triangular numbers under alloc_deep, in round 3.* Its program is `L.X[M3.4 DEC.X] M3.1`
  (17 primitive instructions). It adds x, x − 1, …, 1 into A with `M3.4 = L.X[INC.A]`, then reduces A mod N with
  `M3.1 = L.A[MOV.X.A T.mod(X,N)@x+qN]`.
  - Both routines were made in round 3, when the description-length merge compressed the programs of every problem
    picked so far.
  - Two of that round's four picks only the gate had solved: (x·x) mod N (`law2/mod(mul(X,X),N)`, solved as
    `L.X[M3.4] M3.1`) and N mod (x + 1) (`law2/mod(N,succ(X))`). Their programs are the only ones that call M3.1,
    directly or through M3.3.
  - The plain rerun picked the mod law in round 0 and made routines around it: its round-1 `M1.1` ends in M3.1's body.
    But it never made either piece a routine of its own, and its target search found nothing in 5 rounds.

**The gate found no target itself.** Its second searches ran 9,311 times after a 1M probe and found 356 programs
(3.8%, at cost 4–11, 277 of them at cost 6–7). They ran 255 times after the 10M deep probe and 68 times after the
round's target search, and found nothing in either. Every target in the table came from the plain target search.

### 7. Dead ends, and how hard each was tried

**One feature at a time.** No single feature makes a gate. Each feature's own AUC on C0's held-out fold is at most
0.63 (`live`). The network's 0.94–0.95 comes from combining them. Its AUC by level for fold A's gate is 0.86 at level 1
and 0.94–0.95 at levels 2–4. No state on a solution's path is dead, against 26% of the wrong ones.

**Probe's partial-fit signal alone** (`fitonly`: the answer-equality features). It gives a small constant factor:
- AUC 0.67–0.78;
- toy b_eff 9–10 of 10 (C0) and 33–43 of 40 (C1);
- in search 1.6–1.7×, β 0.97 (C0) and 0.87 (C1).

This was tried as one ablation per corpus and fold (4 gates), searched on every solved test problem and on 50 unsolved
ones per fold.

**The rows' own signature** (`noprob`). It adds nothing. Without it the AUC stays within 0.01 and the search is as
good or better (C0 5.59× against 4.46×; C1 6.81× against 6.82×). Same scope as fitonly.

**A gate that never reads the answer** (`noanswer`: only x, N, 0, unchanged, constant). On C0 it keeps most of the
gain (3.46×, β 0.86). Over the library it loses almost all of it (1.20×, β 0.94, AUC 0.64–0.74), so there the signal
is in comparing registers with the wanted answer. Same scope.

**Dead-state pruning** (§5). It is exact but a constant factor:
- C0: 1.24×, β 0.99;
- C1: 2.07×, β 0.92.

On top of the gate it adds nothing.

**Two measurement traps, caught and corrected:**
- *A log-log slope for reach* (log arm states on log plain states). It reads a constant factor as a change of slope,
  because the plain count also carries where the solution sits inside its level. It was replaced by slopes on the
  plain program's cost (`report.matched`'s docstring).
- *The arms' cap below the plain search's own count.* A 1M probe streams its last level, so the plain count reached
  43M against the arms' 5M cap. Counting a miss at the cap credited every arm with up to 8× on those problems. The
  matched tables keep only problems the plain search solved within 5M and count the rest apart.

**Four whole-run launches stopped and relaunched.** They cost about $6 in all, $5.44 of it the last:
- *gt1a–d* (2 min): the held-out target names were not registered in the main process (`KeyError`).
- *gt1wa–wd* (20 min): I had capped each run's rounds at its plain twin's, against PREREG's "at most 5 rounds". The
  launch also made 9 containers at once, against PREREG's 8.
- *w5a–c* (6 min): a gated record gave its gated level (price plus delays) as `found_cost`. The learner reads that field
  as the problem's price (hardest-first picks, the prior, the gain threshold), so a gate-found problem would have
  looked harder than it is. `gjobs` now gives the cheapest accepted program's price and only those programs, as a plain
  record does; the level is kept as `gated_level`.
- *gt1_a–c* (about 1 h): two of the three containers were preempted, and a preempted run restarted from round 0. Unlike
  the consolidation's groups, `learnrun` had no store. It now appends every plain and gated search to the container's
  stores and replays them after a restart (local check: replay in 3 s instead of 76, identical records).

**A silent check failure.** `check.py` had been failing quietly since the `dead` field was added: a key None in one
record and absent in the other compares unequal, but showed an empty diff. It is fixed (14 jobs, 0 differ).

**Preemptions, a memory kill and a lost call in the fifth launch.** None of them changed a record:
- *Preemptions.* Modal preempted the containers often until about 06:20 PDT, and not after (each record carries the
  start time of the container that wrote it). `gt1s_a` and `gt1s_b` were preempted twice each, `pt1_b` twice before
  its call was lost, `pt1_a`'s relaunch once, and `gt1s_c` never. Each restart replayed its runs from the stores and
  went on.
- *A failed job at a preemption.* A job that fails as the container is stopped is logged as `job failed`, and its run
  goes on without it. That run dies with the container, a failed job is never stored, and the replay searches it again.
- *A memory kill.* `pt1_a`'s first container gave all 16 workers to the plain search. In round 1 it ran 16 of the 10M
  subgoal searches at once, passed the 96 GB limit and was killed (about 03:20 PDT). It was relaunched at 03:56 with 8
  workers (the plain searcher of every gated container) and replayed its store.
- *A lost call.* Modal lost `pt1_b`'s call after a preemption ("Server has lost track of input"; no progress after
  04:42). It was relaunched the same way at 06:22.
- *The replays are exact.* A replay's rounds match the first run's, and the plain reruns match the consolidation's own
  records pick for pick and routine for routine in every round both have.

**Two restarts in §9.** `gt2_a`'s container restarted twice, between 12:01 and 12:11 PDT and at 13:05, and each restart
replayed its runs from the stores and went on; its records come from the third container. Before each restart, the
targets that had come had come in the rounds of the final records (x² mod N and 2x + 1 in round 0, x³ in round 1).
`gt2_b` and `gt2_c` ran in one container each. The logs Modal keeps for a stopped app do not say why `gt2_a` restarted.

### 8. Files and how to rerun

Everything is in `cpu_first/bar3/scale/gate/`; it imports `bar3/`, `bar3/breadth/`, `bar3/scale/` (the engine, the jobs,
the consolidation learner copy) and `library/`, and changes none of them.
- `gengine.py`: the gated engine (`bfs`: delays, the three budget rules, dead states); `gjobs.py`: a search job on it,
  and the harvest of a solved search's training states.
- `features.py`, `model.py`: the gate's inputs and network; `train.py`: the dataset, the folds, training, the delay
  map's tuning by the toy model, the judgement before search; `toy.py`: the toy model (theory/GUIDED_SEARCH.md §3).
- `capture.py`: round 0 of the consolidation learner with every search written out (the corpus); `tasks.py`: the arms'
  search jobs; `learnrun.py`: whole runs with the gate as a second search; `report.py`: the arms' tables;
  `wcompare.py`: the whole runs against their twins (§6, §9).
- `make_learner.py`: writes `learn_anchor.py`, the consolidation learner with §9's one change (anchors among the plain
  search's finds, under `cfg["anchor_plain"]`).
- `modal_gate.py`: Modal (app and Volume `bar3-gate`): `capture`, `train`, `evaluate`, `learn`.
- `check.py`: exactness (about 10 s locally).
- `records/`: the corpus captures (`capture_*.jsonl.gz`), the gates (`models/`), the arms' results, the whole runs.

The compiled engine needs numba 0.68 with numpy 2.5.4 (the Modal image pins both). `M` below is a Python with Modal
(`~/github/sotaku/venv/bin/python` here); `P` is one with numba (a uv venv with the Modal image's pins). Run each Modal
launch behind `$M integration/ablation_s3/spend.py 4750`, the monthly guard (5750 from 03:45 PDT Oct 30, after the
workspace limit was raised).

The exactness check (local, about 25 s):

```bash
cd cpu_first && $P -m bar3.scale.gate.check
```

The corpora, one container per label (C0 from scratch; C1 over the library: `@STATE` is under `bar3/scale/records/`):

```bash
$M -m modal run --detach bar3/scale/gate/modal_gate.py --capture-label c0b --specs alloc_deep_ct2a2_probe:mont,alloc_deep_ct2a2_probe:ten,alloc_deep_ct2a2_probe:cube,alloc_deep_ct2a2_probe:fib,alloc_deep_ct2a2_probe:tenrho
```

```bash
$M -m modal run --detach bar3/scale/gate/modal_gate.py --capture-label c1a --specs alloc_deep_ct2a2_probe:mont@seq_ct2a2_clean_new_targets_L1.state.affine.json,alloc_deep_ct2a2_probe:tenrho@seq_ct2a2_clean_new_targets_L1.state.affine.json,alloc_deep_ct2a2_probe:ten@seq_ct2a2_clean_new_targets_L1.state.affine.json,alloc_deep_ct2a2_probe:fib@seq_ct2a2_clean_new_targets_L1.state.affine.json,alloc_deep_ct2a2_probe:cube@seq_ct2a2_clean_new_targets_L1.state.affine.json
```

The labels and their formulas:
- c0a: h1;
- c0b: mont, ten, cube, fib, tenrho;
- c0c: x2x, three, x3x, x1sq, x4;
- c0d: pow2, floorsq, trib, tri, lucas;
- c0e: rho, minus1, plus3, affine;
- c1a: mont, tenrho, ten, fib, cube;
- c1b: x1sq, x2x, three, x3x, x4;
- c1c: trib, tri, floorsq, lucas, pow2.

The gates (with the judgement before search, `v1/models/judge_<corpus>.json`):

```bash
$M -m modal run --detach bar3/scale/gate/modal_gate.py --train-corpus c0 --labels c0a,c0b,c0c,c0d,c0e --b 10
```

```bash
$M -m modal run --detach bar3/scale/gate/modal_gate.py --train-corpus c1 --labels c1a,c1b,c1c --b 40
```

The arms: build the task file, put it on the Volume, and run it in shards (a rerun skips finished tasks):

```bash
$P -m bar3.scale.gate.tasks c1 /out/v1/models bar3/scale/gate/records/tasks/c1.jsonl.gz bar3/scale/gate/records/capture_c1a.jsonl.gz bar3/scale/gate/records/capture_c1b.jsonl.gz bar3/scale/gate/records/capture_c1c.jsonl.gz
```

```bash
$M -m modal volume put bar3-gate bar3/scale/gate/records/tasks/c1.jsonl.gz v1/tasks/c1.jsonl.gz
```

```bash
$M -m modal run --detach bar3/scale/gate/modal_gate.py --evaluate-name c1 --shards 4
```

The whole runs, one container per group (records to `v1/learn/<label>/`). The three groups as launched:

```bash
D=alloc_deep_ct2a2_probe; NS=alloc_ns_ct2a2_probe; r() { echo "${1}_${2}_gt1:${1}:${2}"; }
G='{"A":{"empty":"/out/v1/models/c0_A_full.json","lib":"/out/v1/models/c1_A_full.json"},"B":{"empty":"/out/v1/models/c0_B_full.json","lib":"/out/v1/models/c1_B_full.json"}}'
A="$(r $D ten),$(r $D x3x),$(r $D lucas),$(r $D h1),$(r $D cube),$(r $D three),$(r $D affine)"
B="$(r $D x4),$(r $D trib),$(r $D floorsq),$(r $D fib),$(r $D x1sq),$(r $D rho),$(r $D pow2)"
C="$(r $NS tenrho),$(r $D tri),$(r $NS tri),$(r $D minus1),$(r $D plus3),$(r $D x2x)"
for x in a:"$A" b:"$B" c:"$C"; do $M -m modal run --detach bar3/scale/gate/modal_gate.py --learn-label gt1s_${x%%:*} --runs "${x#*:}" --gates "$G" --workers 16; done
```

The plain reruns of the eight non-wins: the same arm with no gate (`--gates '{}'`), every worker on the plain search.
They were first launched with 16 workers at 02:13 PDT, which runs out of memory (§7), so use 8. A relaunch with the
same label replays the label's store.

```bash
D=alloc_deep_ct2a2_probe; NS=alloc_ns_ct2a2_probe; r() { echo "${1}_${2}_pt1:${1}:${2}"; }
A="$(r $D x3x),$(r $D lucas),$(r $D ten),$(r $D x4)"
B="$(r $D trib),$(r $D floorsq),$(r $D tri),$(r $NS tenrho)"
for x in a:"$A" b:"$B"; do $M -m modal run --detach bar3/scale/gate/modal_gate.py --learn-label pt1_${x%%:*} --runs "${x#*:}" --gates '{}' --workers 8; done
```

§9's runs: write the learner copy, seed each group's stores with the fifth launch's (a search already done replays), and
launch the same groups with `--anchor-plain 1` (as launched 11:09 PDT Oct 30):

```bash
python3 -m bar3.scale.gate.make_learner
```

```bash
for g in a b c; do for s in store_plain.jsonl store_gate.jsonl; do $M -m modal volume cp bar3-gate v1/learn/gt1s_$g/$s v1/learn/gt2_$g/$s; done; done
```

```bash
D=alloc_deep_ct2a2_probe; NS=alloc_ns_ct2a2_probe; r() { echo "${1}_${2}_gt2:${1}:${2}"; }
G='{"A":{"empty":"/out/v1/models/c0_A_full.json","lib":"/out/v1/models/c1_A_full.json"},"B":{"empty":"/out/v1/models/c0_B_full.json","lib":"/out/v1/models/c1_B_full.json"}}'
A="$(r $D ten),$(r $D x3x),$(r $D lucas),$(r $D h1),$(r $D cube),$(r $D three),$(r $D affine)"
B="$(r $D x4),$(r $D trib),$(r $D floorsq),$(r $D fib),$(r $D x1sq),$(r $D rho),$(r $D pow2)"
C="$(r $NS tenrho),$(r $D tri),$(r $NS tri),$(r $D minus1),$(r $D plus3),$(r $D x2x)"
for x in a:"$A" b:"$B" c:"$C"; do $M -m modal run --detach bar3/scale/gate/modal_gate.py --learn-label gt2_${x%%:*} --runs "${x#*:}" --gates "$G" --workers 16 --anchor-plain 1; done
```

Download each run's records to `records/learn/` (its rounds `<run>.jsonl` and its second searches
`<run>.gate.jsonl`; the stores stay on the Volume):

```bash
for l in gt1s_a gt1s_b gt1s_c pt1_a pt1_b gt2_a gt2_b gt2_c; do for f in $($M -m modal volume ls bar3-gate v1/learn/$l | grep -E "_(gt1|gt2|pt1)(\.gate)?\.jsonl$"); do $M -m modal volume get --force bar3-gate $f bar3/scale/gate/records/learn/$(basename $f); done; done
```

The tables (from the downloaded results and captures):

```bash
$P -m bar3.scale.gate.report c1 bar3/scale/gate/records/results/c1 bar3/scale/gate/records/capture_c1a.jsonl.gz bar3/scale/gate/records/capture_c1b.jsonl.gz bar3/scale/gate/records/capture_c1c.jsonl.gz
```

§6 from the downloaded whole runs: each run round by round against its twin, the table, the rounds that picked the mod
law and whether each target's program calls it, and the second searches' totals:

```bash
python3 -m bar3.scale.gate.wcompare bar3/scale/gate/records/learn
```

§9 the same way, with the fifth launch's outcome beside each run, and the anchor rule checked in every round:

```bash
python3 -m bar3.scale.gate.wcompare bar3/scale/gate/records/learn gt2
```

### 9. The whole runs again, with the anchors kept to the plain search's finds

[PREREG_ADDENDUM2.md](PREREG_ADDENDUM2.md), frozen before the launch (§1). The same 20 runs as §6, with one change in
the learner: anchored pairs anchor only on the two costliest learnable problems that the plain search solved
(`learn_anchor.py`, written by `make_learner.py` from the consolidation's learner, with that one line changed). A
problem only the gate's second search solved stays learnable, so it can still be a pair's other member, a single or a
pick. Everything else is §6's: the gates, the second search and its cap, at most 5 rounds, the same three groups
(`gt2_a/b/c`, 3 containers, 11:09 PDT Oct 30). Each container started from copies of its fifth-launch group's stores,
so round 0 up to the pair looks is the fifth launch's exactly, and every difference comes from the anchor rule. The
twins and the plain reruns are §6's.

The rule held. In all 58 rounds of the 20 runs, every scored pair holds one of the round's two costliest plain finds,
with up to 17 learnable gate finds in a round kept out of the anchor places (`wcompare.anchor_check`; on the fifth
launch's records the same check finds 54 rounds that break it).

| formula | config | kind | plain twin | fifth launch | anchors kept plain | its picks only the gate solved, by round | second searches: found / run |
|---|---|---|---|---|---|---|---|
| 2x + 1 | alloc_deep | win | round 0 | round 0 | round 0 | 1 | 18 / 136 |
| x² + 1 | alloc_deep | win | round 0 | no (5 rounds) | round 0 | 0 | 6 / 219 |
| x² + x | alloc_deep | win | round 0 | round 2 | round 0 | 0 | 11 / 96 |
| x² mod N | alloc_deep | win | round 0 | no (5 rounds) | round 0 | 0 | 6 / 148 |
| x² − 1 | alloc_deep | win | round 0 | round 1 | round 0 | 1 | 9 / 163 |
| (x + 1)² | alloc_deep | win | round 1 | round 0 | round 0 | 1 | 13 / 149 |
| 2^x | alloc_deep | win | round 1 | round 2 | round 1 | 2, 2 | 22 / 269; target 0 / 1 |
| 3·x² | alloc_deep | win | round 1 | no (5 rounds) | no (5 rounds) | 1, 3, 1, 0, 0 | 48 / 359; target 0 / 1 |
| Fibonacci | alloc_deep | win | round 1 | round 1 | round 1 | 0, 0 | 17 / 252; target 0 / 1 |
| triangular | alloc_ns | win | round 1 | round 3 | round 1 | 1, 1 | 18 / 230; target 0 / 1 |
| x² + 3 | alloc_deep | win | round 1 | round 1 | round 1 | 0, 1 | 17 / 303; target 0 / 1 |
| x³ | alloc_deep | win | round 1 | no (5 rounds) | round 1 | 0, 1 | 12 / 253; target 0 / 1 |
| 10·x² | alloc_deep | miss | no (4 rounds); rerun no (5 rounds) | no (5 rounds) | no (5 rounds) | 0, 0, 1, 0, 0 | 28 / 509; target 0 / 5 |
| 10·x² + 1 | alloc_ns | miss | no (3 rounds); rerun no (5 rounds) | no (5 rounds) | no (5 rounds) | 0, 0, 0, 1, 2 | 29 / 602; target 0 / 5 |
| x⁴ | alloc_deep | miss | no (4 rounds); rerun no (5 rounds) | no (5 rounds) | round 1 | 1, 1 | 14 / 212; target 0 / 1 |
| Lucas | alloc_deep | held-out | –; rerun no (5 rounds) | no (5 rounds) | no (5 rounds) | 1, 1, 0, 0, 1 | 49 / 707; target 0 / 5 |
| Tribonacci | alloc_deep | held-out | –; rerun no (5 rounds) | no (5 rounds) | no (5 rounds) | 1, 1, 0, 1, 1 | 33 / 677; target 0 / 5 |
| floor(x²/N) | alloc_deep | held-out | –; rerun no (5 rounds) | no (5 rounds) | no (5 rounds) | 0, 0, 0, 1, 1 | 21 / 418; target 0 / 5 |
| triangular | alloc_deep | held-out | –; rerun no (5 rounds) | round 3 | no (5 rounds) | 1, 0, 1, 0, 0 | 27 / 468; target 0 / 5 |
| x³ + x | alloc_deep | held-out | –; rerun no (5 rounds) | no (5 rounds) | no (5 rounds) | 1, 1, 1, 0, 0 | 40 / 444; target 0 / 5 |

**What came.**
- *The 12 wins.* Ten came in their twin's round. (x + 1)² came a round early again, in round 0 with the fifth launch's
  program. 3·x² did not come in 5 rounds.
- *The fifth launch's eight late or lost wins.* Seven came back in their twin's round: x² mod N, x² − 1, x² + 1 and
  x² + x in round 0; 2^x, x³ and the triangular numbers under alloc_ns in round 1. Only 3·x² is still missing.
- *The 8 non-wins.* x⁴ came in round 1. This arm had not brought it: not in the consolidation's 4 rounds, nor in the
  plain rerun's 5 or the fifth launch's 5. (The held-out study brought it in round 1 with its subgoal, HELDOUT.md.) The
  triangular numbers under alloc_deep did not come in 5 rounds; the fifth launch had them in round 3. The other six
  came in none of their 5 rounds.
- *The programs.* x³, Fibonacci, x² mod N, x² + 1 and x² + x found the twin's program exactly. x² − 1 and 2^x found
  programs as long as the twin's (14 and 20 primitive instructions) through other routines, and (x + 1)² the fifth
  launch's (12, as long as the twin's). Three are longer: 2x + 1 (15 against 10), x² + 3 (33 against 23) and the
  triangular numbers under alloc_ns (12 against 11). x⁴'s is 24.

**Predictions against outcomes** (PREREG_ADDENDUM2.md's table, in its order):

| measure | predicted (confidence) | came | right? |
|---|---|---|---|
| x mod N in round 0 | in at least 10 of the 11 runs whose twin picked it (medium) | in all 11 | yes |
| x² mod N, x² − 1, x² + 1, x² + x | all four in round 0 (medium-high) | all four in round 0 | yes |
| 3·x², x³, 2^x, triangular (alloc_ns) | all four within 5 rounds, at least two in round 1 (medium-low) | x³, 2^x and the triangular numbers in round 1; 3·x² not in 5 rounds | no: three of the four |
| 2x + 1, Fibonacci, x² + 3 | their twins' rounds, 0, 1 and 1 (medium-high) | 0, 1 and 1 | yes |
| (x + 1)² | round 0 (medium) | round 0, with the fifth launch's program | yes |
| triangular (alloc_deep) | does not come in 5 rounds (medium-low) | not in 5 rounds | yes |
| the other seven non-wins | none comes in 5 rounds (high) | x⁴ in round 1; the other six none | no |
| overall | at most two of the eight late or lost wins still late or lost (medium) | one: 3·x² | yes |

**What the refutations say.**
- *The round-0 account (§6) stands.* x² mod N, x² − 1, x² + 1 and x² + x all came in round 0 again. In each, the twin's
  best round-0 pair was scored again with the same gain (66–91 bits), and the program ends in
  `L.N[MOV.X.A T.mod(X,N)@x+qN]`, as the twin's does.
- *The anchor places were the main route of the losses.* One of the eight late or lost wins is still missing (3·x²);
  three would have refuted it.
- *The invention came through the anchor places.* The triangular numbers under alloc_deep did not come. In the fifth
  launch, round 3's two anchors were the gate's finds (x·x) mod N and N mod (x + 1), both picked, and their routines
  made the program (§6). Here the gate found (x·x) mod N from round 1 and N mod (x + 1) in round 3, both at cost 6, but
  they were kept out of the anchor places, and neither was picked in 5 rounds.

**How the outcomes moved.**
- *The mod law is back in round 0.* It was picked in round 0 in all 11 runs whose twin picked it (the fifth launch
  picked it there in one of them), and in 17 of the 20 runs.
- *The gate's finds are still picked, now beside plain anchors.* In round 0, 11 of the 20 runs picked a problem only
  the gate had solved (all 20 in the fifth launch). x³, Fibonacci, x² + 1 and x² + x had their twin's round-0 picks
  and library exactly.
- *x⁴ came through a gate find picked beside the mod law.*
  - Round 0. The twin's costliest anchor, `law2/min(monus(N,X),X)` (cost 7, found there by the 10M deep probe), is the
    gate's find here, so the anchors are the mod law and `law2/lt(succ(X),N)`. The best pairs put the gate's find
    beside each (14.3 bits), and the mod law was picked. In the twin the mod law was an anchor too, but the cost-7
    anchor's pairs with the add laws scored as high and filled the four picks. It was never picked there in 4 rounds,
    nor in the plain rerun's 5.
  - Round 1. Over a library with the mod routine, the plain 1M probe of (x·x) mod N spent 4.45M states without a
    program. The gate found `L.X[L.X[INC.A]] L.X[MOV.X.A T.mod(X,N)@x+qN]` at price 6, first accepted at 480k states.
    It gained 39.8 bits with either anchor and was picked. Its routine `T.law6` heads the round's program,
    `T.law6 T.law7 M1.1` (24 primitive instructions; `M1.1` reduces A mod N with the mod routine).
  - The held-out study's hardest-first program for x⁴ has the same shape and length (`T.law5 T.law8 M1.1`, 24).
- *3·x² was lost through the rule itself.*
  - Round 0 differs from the twin's in two picks. The best pairs put the gate's find `restrict/target/X·X<X+X`
    (price 11) beside each anchor (44.7 bits), so it and the second anchor, `law2/lt(succ(X),N)`, were picked in place
    of two of the twin's three constant-3 picks. The mod law was picked in both.
  - Over that library (8 routines against the twin's 5), round 1's plain 1M probes left (x·x) mod N and
    `restrict/target/X<3` unsolved (4.3M and 4.1M states). The gate found both at price 6. In the twin both were plain
    finds, and their pair gained the most that round (77.8 bits, with (x·x) mod N an anchor); its target came in that
    round.
  - Here neither can anchor, so that pair was never scored. N mod (x + 1), the twin's other anchor, was not solved at
    all. From round 1 the target search found programs that fit the target's examples (in rounds 2–4 a single
    restriction routine, `T.res10`), but none was general, and the target did not come in 5 rounds.
  - So keeping the gate's finds out of the anchor places has a cost of its own. Where the gate's earlier picks change
    the library, a problem the twin solved plainly can become a gate find, and lose the anchor place it had.

**The gate found no target itself.** Its second searches ran 6,404 times after a 1M probe and found 437 programs (6.8%,
351 of them at cost 6–7). They ran 168 times after the 10M deep probe, with one find (`law2/mod(add(X,X),N)` at
price 8, for 10·x² in round 2, picked), and 42 times after the round's target search, with none. Every target came
from a plain search.

**Cost.** $23.07 for this rerun (Modal's billing for `bar3-gate` from 18:00 UTC Oct 30 through the hour that began at
01:00 UTC Oct 31; its last container stopped a minute after it), $102.93 for the study in all.
