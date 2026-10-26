# Three changes to the picks: ties by the subgoal score, the subgoal's twin, and a score that sees big x

<!-- PLAIN -->
## In plain words

**The question.** With the duplicate check fixed (ROWCHECK.md), three targets were still blocked by how the learner
picks its practice:
- **10·x² + 1 under hardest first.** Its piece, 10X mod N, is learnable but loses hardest first's ties to pool order.
- **x² mod N in base 2 under hardest first.** Its hidden state, x mod N on long x, is a true copy of a cheap problem
  that hardest first never picks.
- **x² mod N in base 2 under anchored pairs.** Its pick score sees only the rows, and counting programs fit those.

Each fix went behind its own flag, in copies of the learners, with predictions written down first (`PREREG.md`):
- **ties by the subgoal score:** once a subgoal is practised, score the pool against the rest of the way from its
  routine, and give hardest first's ties to the candidate the score singles out;
- **the subgoal's twin:** when the subgoal is a copy of a learnable problem, that problem inherits its priority and is
  practised as soon as it is learnable;
- **a score that sees big x:** anchored pairs' description length also charges for the check at big x, so a look
  that finds a program the target accepts gains that charge.

**What came.**

| change | arm | before | with the change |
|---|---|---|---|
| ties by the subgoal score (with the row check) | 10·x² + 1, hardest first | no, 5 rounds | **round 2** |
| ties by the subgoal score | the triangular numbers, hardest first | no | no: the same run (it has no subgoal) |
| the subgoal's twin | x² mod N, base 2, hardest first | no | no, 5 rounds |
| a score that sees big x | x² mod N, base 2, anchored pairs | no | no: the same run |

- **10·x² + 1, hardest first, round 2.**
  - In round 1 the learner practised its subgoal, x² mod N.
  - In round 2 the score, run against the rest of the way from that routine, singled out 10X mod N. It is one law
    from the rest (add 1, reduce), with key (1, 11): the piece SUBGOAL.md §8 found by hand.
  - It won hardest first's ties at cost 5 and was practised.
  - A merge of its program (`M2.2`) made the target four tokens: `T.law5 M2.2 INC.X T.mod`, x² mod N, then 10X mod N,
    then add 1 and reduce.
  - The program is right on all 7,260 pairs x < N ≤ 120, given steps. The subgoal, two-source and row-check studies
    did not get this target under hardest first.
- **The big-x score changed nothing it could have changed.** In the anchored-pairs x² mod N run no look ever found a
  program the target accepts. So the new charge cancels in every score, and the run is the earlier one. The
  blocker there is the looks' reach. The program that passes at big x is `D.2[step] MOV.X.A CLR.A T.law10`, five
  tokens. A look's 30,000 states end with level 4, one short.
- **The twin did what it was built to do, and that was not enough.**
  - Under hardest first on x² mod N, x mod N on long x's twin was practised in round 1.
  - From round 2 the source posed its step, learnable at cost 4–5. But hardest first left it behind the cost-5
    problems ahead of it in pool order, and in round 4 it was out of the probe's reach.
- **The twin's pick cost an earlier win.**
  - In hardest first's binary x mod N, round 1's subgoal was a copy of law2/div(succ(X),N). Practising that copy
    took one of the four picks.
  - The problem it displaced took a slot in round 2, and the queue of cost-5 problems moved back by one.
  - Before, the steps were picked in round 3. Now their turn came in round 4, when the probe no longer reached them,
    and x mod N did not come.

**The earlier wins.** With each change on, the 11 earlier wins (nine held-out, and binary x mod N in the digits study
under both rules) were replayed.
- **Ties by the subgoal score: all 11 unchanged.** The tie-break acts only from the round after a subgoal is
  practised, and every win had come by then.
- **The big-x score: the two digits wins unchanged.** It changes a score only through a look that finds an accepted
  program, and the one run with such looks already picked them.
- **The twin: 10 unchanged, one lost.** Hardest first's binary x mod N came in round 3 before and does not come
  now.

**Against the predictions.**
- Held: every "certain" item (where the change cannot act, it did not), and the twin's mechanism (practised in round
  1, its step posed and learnable, the step not picked).
- Failed, both medium:
  - "10·x² + 1 does not come": it came in round 2. The prediction's own caveat named the way through, a merge.
  - "the binary x mod N win still comes under the twin": it was lost.

**Cost.** About $5.3 on Modal, for the twin's two runs (one app, 4.2 hours, 2,072 new searches, 10.5 search-hours).
Everything else ran here from the earlier stores, with one new search.

**What it says.** Breaking hardest first's ties by the score on the rest of the way gets 10·x² + 1. The piece the
two-source study's offset posed, the row check let through, and the score singles out is the one the target needed,
and a merge did the rest. Passing a subgoal's priority to its twin works as built. But practising the twin costs a
pick, and in hardest first's queue a pick lost in round 1 delays everything behind it: here, past the round where the
probe still reaches the step. The big-x score is right in principle and changes nothing until a look can reach
a program that passes at big x: the looks' budget is the limit there.
<!-- /PLAIN -->

## Details

### 1. Pre-registered

`PREREG.md`, read-only since it was written, sha256 `68b1cb9a844bdd4a58f79bd8d5a5d7b760a7525a19b9570b5d0f83f2dc92dc7e`.
- Its written time first read 20:49 PDT. It was frozen at 20:48, corrected to that true time, then frozen and hashed
  again within the minute, before anything ran with a change on.
- Before it: the all-off replays (§2.3), and the facts settled from the records (its "Settled" section).

PREREG.md's text, unchanged:

````markdown
# Three changes to the picks: pre-registration

Written 2026-10-25 20:48 PDT, before any run with one of the changes on. PICKS.md §1 repeats this text unchanged.

## The question

ROWCHECK.md §7 named what still blocks two targets once the duplicate check is right:
- 10·x² + 1 under hardest first: its piece, 10X mod N, is learnable but loses hardest first's ties to pool order;
- x² mod N in base 2 under hardest first: the hidden state, x mod N on long x, is a true copy of a cheap problem
  hardest first never picks;
- x² mod N in base 2 under anchored pairs: the pick score sees only the rows, which counting programs already fit.

Each fix goes behind its own flag. Does it unlock its target, and does it leave the earlier wins as they were?

## The changes (`picks.py`, `make_learners.py`)

- **`tiescore`, with the row check on.** Once a subgoal has been practised, the rest of the way from its routine is
  the problem left. The pool is scored against that rest with score.py's rule, as SUBGOAL.md §8's own-subgoal rule
  scores it. The candidate the score singles out wins hardest first's ties at its cost, as the subgoal does. Escher
  chooses what to grow by a heuristic function and names better ones an open problem (Albarghouthi, Gulwani and
  Kincaid, CAV 2013, §3.3); here pool order is the heuristic being replaced.
- **`twin`.** A subgoal that is a duplicate of a learnable problem passes its priority to that problem. The twin
  wins ties as the subgoal would, and is practised as soon as it is learnable, in the last pick's place. Ties alone
  would do nothing under hardest first, where the twin is cheap. This is priority inheritance: "when a job J blocks one
  or more higher priority jobs, it ... executes its critical section at the highest priority level of all the jobs it
  blocks" (Sha, Rajkumar and Lehoczky, IEEE Trans. Computers 39(9), 1990, §III).
- **`bigx_score`.** For a target with a check at big x, the description length that anchored pairs scores looks by
  (bar3/learner.mdl) also charges every level without an accepted program for the 200 big-x answers it does not
  give: 200·(log₂ 200 + log₂ 10) ≈ 2,193 bits. A level whose program the target accepts is not charged. So a look
  that finds an accepted program gains that much, and when no record has an accepted level every score is the frozen
  one. In MDL terms the data then include the check the target is judged on (Rissanen, Automatica 14(5), 1978).

Learner copies: `learn_newton.py` and `learn_digits.py`, each written by `make_learners.py` from the learner frozen at
97f30b9 with that study's asserted replacements, then the row check's 3, then this study's 7. With every flag off,
15 earlier records replay exactly from their stores (`check.py off`; `logs/check_off_*.txt`):
- the nine held-out wins;
- the two-source study's hardest-first runs on 10·x² + 1 and the triangular numbers;
- the digits study's four base-2 runs with the source.

## Settled from the records before this file (no run)

- **The tie-break acts only once a subgoal has been practised.** In each of the 11 earlier wins, that happens in the
  round its target came, or never, so the tie-break cannot change them. The triangular numbers have no subgoal in any
  round of the earlier run, so it cannot change that run either. On 10·x² + 1, x² mod N was practised in round 1, so
  the rest of the way from it is scored from round 2.
- **The twin acts only where the subgoal is a duplicate of a learnable problem:**
  - x² mod N in base 2, hardest first: in rounds 1–4, x mod N on long x is a duplicate of the law book's x mod N on
    short pairs;
  - hardest first's binary x mod N (an earlier win): in round 1, value/T.law2 → X is a duplicate of
    law2/div(succ(X),N);
  - in no other earlier win.
- **The big-x score changes a score only through a look or pair that finds a program the target accepts:**
  - none did in any of the 5 rounds of the anchored-pairs x² mod N run, so that run cannot change. It replays the
    same stored searches, and every score is the frozen one;
  - 15 looks did in round 1 of anchored pairs' binary x mod N (an earlier win), so its scores change there;
  - the held-out targets have no check at big x, and hardest first does not pick by this score: the flag does not
    apply to them.

## Runs

Each arm from the earlier runs' stores, 5 rounds, seed 0, the frozen budgets, with its change on:
- **tiescore:** 10·x² + 1 under hardest first, both sources and the row check (`lp_clean_deep_sub_ns_rc_tie`). The
  triangular numbers under hardest first (same config) are replayed only to confirm they cannot change.
- **twin:** x² mod N in base 2, hardest first (`lp_clean_b_src_twin`).
- **bigx_score:** x² mod N in base 2, anchored pairs (`ct2a2_clean_b_src_bigx`), replayed only to confirm it cannot
  change.
- **The 11 earlier wins** with each change on, up to the round each came. The bigx score is replayed only on the two
  digits wins, to which it applies.

Searches the stores lack run locally under a memory cap when few. A run that diverges early runs on Modal, its store
uploaded. Money: soft $20, cap $25. Every launch is guarded by `spend.py 4750`, with one spend watcher for all apps.

## Predictions

| arm | prediction | confidence |
|---|---|---|
| tiescore, 10·x² + 1 | the piece is picked in round 2; the target does not come in 5 rounds | medium-high; medium |
| tiescore, the triangular numbers | the earlier run, round for round | certain (no subgoal) |
| twin, x² mod N base 2, hardest first | x mod N on long x's twin practised in round 1; the source poses its step from round 2, learnable; the step not picked; the target does not come | high; medium-high; medium; medium-high |
| bigx_score, x² mod N base 2, anchored pairs | the earlier run, round for round | certain (no accepted look) |
| the earlier wins, tiescore | all 11 the same | certain |
| the earlier wins, twin | 10 the same; hardest first's binary x mod N changes from round 1 and still comes | certain; medium |
| the earlier wins, bigx_score | anchored pairs' binary x mod N still comes in round 1; with the same program | high; medium |

Why:
- **tiescore, 10·x² + 1.** The rest from x² mod N's routine is posed in round 2, and with it its offset, 10X mod N,
  learnable at cost 5 with the row check. Against that rest the offset is one law away (add 1, reduce), so the
  score should single it out (SUBGOAL.md §8's local test: first and alone), and cost 5 is round 2's top cost. Then
  the target needs 10x² + 1 from x² mod N and 10X mod N: `T.law5 MOV.X.A T.off MOV.X.A INC.X T.mod`, 6 tokens,
  past the 3M-state target search. Hardest first does not pick the rest itself (cost 4 once 10X mod N is a routine)
  while 12 problems cost 5. A merge or the guided search would have to shorten it.
- **twin, x² mod N.** The twin is learnable at cost 2–3, so it is practised in round 1. In the anchored-pairs run,
  where the twin was practised, its step was learnable from round 2 at cost 4–5. Under hardest first the digits
  source comes last in pool order, and many problems cost 5. So the step is not picked, and without it no program
  passes at big x.
- **The wins under twin.** In hardest first's binary x mod N, round 1's fourth pick gives way to law2/div(succ(X),N).
  The rounds after differ. The step was learnable from round 1 at cost 4–5 and picked in round 3; whether it still is
  is open.
- **The wins under bigx_score.** In round 1 the steps' looks find the target. They then gain about 2,193 bits more
  than every other candidate's, so the steps are picked as before or more. The fourth pick, and so the step that
  comes first, may change.

If an arm does not diverge from the stored searches, its replay is its run and costs nothing. What is reported for
each run: whether and in which round its target came, the program and its check; what the change did each round
(its `picks` field); and the first round whose picks, library or target search differ from the earlier run's.
````

### 2. The theory and the changes

#### 2.1 What each change borrows

- **Ties by a better heuristic.** Escher (Albarghouthi, Gulwani and Kincaid, CAV 2013, §3.3, "Heuristic search")
  chooses what to grow by a heuristic function h. Program size is its simple example, and it calls the design of
  better ones "an important and interesting problem". Hardest first is cost, then pool order. The change replaces
  pool order in its ties by the subgoal score on the rest of the way: SUBGOAL.md §8's own-subgoal rule, used for ties
  only.
- **Priority inheritance.** Sha, Rajkumar and Lehoczky (IEEE Trans. Computers 39(9), 1990, §III): "when a job J blocks
  one or more higher priority jobs, it ignores its original priority assignment and executes its critical section at
  the highest priority level of all the jobs it blocks". A subgoal that is a duplicate cannot be practised under its
  own name; its twin holds what it needs. So the twin inherits the subgoal's priority.
- **The description length of the whole check.** MDL scores a model by the bits it takes to describe the data
  (Rissanen, "Modeling by shortest data description", Automatica 14(5), 1978). The looks' description covered only
  the rows sampled, so a program right on the rows and wrong at big x scored as if it described everything. The
  change adds the big-x answers to the data a level must describe.

#### 2.2 The changes (`picks.py`)

- **`tiescore`, with the row check on.**
  - Which rest: the rest of the way from the latest practised subgoal's routine. The learner already records these
    routines (`chained`), whether or not the chain config is on, as the rrem problem, under that name or another
    candidate's that posed the same problem.
  - How it is scored: with score.py's rule, as SUBGOAL.md §8 did.
  - What it gives: the candidate the score singles out (a positive key no other candidate shares) joins the
    subgoal's ties, under hardest first only.
- **`twin`.**
  - When it fires: a subgoal whose status is "duplicate" of a learnable problem.
  - What the twin gets: it joins the subgoal set and its ties, and it is practised as soon as it is learnable, in the
    last pick's place (the frozen `pick` rule, applied to the twin).
- **`bigx_score`.**
  - What it charges: every level without an accepted program, 200·(log₂ 200 + log₂ 10) bits more, in the
    description length the target probe, the looks and the pairs are scored by.
  - Where it applies: digits targets only. An accepted level is not charged, so with no accepted level anywhere
    every score is the frozen one.

#### 2.3 The copies (`make_learners.py`, `check.py off`)

`make_learners.py` writes two copies from the learner frozen at 97f30b9, each by asserted exact replacements:
- `learn_newton.py`: the two-source study's 3, the row check's 3, then this study's 7;
- `learn_digits.py`: the digits study's 11 (and its DigitsTap), the row check's 3, then this study's 7.

A round's record gets a `picks` field (what the changes did) only when a change is on. With every flag off, 15
earlier records replay exactly from their stores (`logs/check_off_newton.txt`, `logs/check_off_digits.txt`):
- the nine held-out wins;
- the two-source study's hardest-first runs on 10·x² + 1 and the triangular numbers;
- the digits study's four base-2 runs.

### 3. The tiny test: replays with each change on (`check.py on`; `logs/check_on_*.txt`)

Every arm and every win was replayed with each change on. Cheap searches the stores lacked were run here, and each
replay stopped at the first other search.
- **ties by the subgoal score:**
  - 10·x² + 1: rounds 0–1 the earlier run's, round 2 different, stopping at round 2's target search. Replayed
    again with that search run here, the target came in round 2 (`logs/check_on_tie_tenrho_r2.txt`).
  - The triangular numbers: all 5 rounds the earlier run's.
  - The 11 wins: unchanged.
- **the twin:**
  - The 9 held-out wins and anchored pairs' binary x mod N: unchanged.
  - x² mod N under hardest first (the arm) and hardest first's binary x mod N (a win): different from round 1 on.
  - x² mod N under anchored pairs: also different from round 1 on. No arm and no win; not run.
- **the big-x score:** every digits run is the earlier one, round for round, with no new search.

### 4. The runs (`run.py` here, `modal_picks.py` on Modal; `records/*_pk1.jsonl`; `logs/run_*`, `logs/app/`)

Here, from the earlier stores, each under `memcap.sh` at 10 GB:

| run | wall | read back | run here |
|---|---|---|---|
| 10·x² + 1, hardest first, `lp_clean_deep_sub_ns_rc_tie` | 15 s | 1,259 | 1 (the target search of round 2) |
| the triangular numbers, the same config | 13 s | 1,917 | 0 |
| x² mod N, base 2, anchored pairs, `ct2a2_clean_b_src_bigx` | 40 s | 4,126 | 0 |

On Modal, the two twin runs that leave the stores in round 1: x² mod N in base 2 and binary x mod N, both under
hardest first (`lp_clean_b_src_twin`; app `ap-8EDvv8aQEQfF7oZNrXQTO7`). They ran in one container (16 CPUs, 64 GiB),
from the digits study's two base-2 stores joined (Volume `bar3-picks`, `v1/store_pk1_digits.jsonl`). Before the
launch, `spend.py 4750` printed LAUNCH OK at $3,514.05 billed this month. One spend watcher would stop the app at $12
(`logs/watch_pk1.log`).

The app ran from 03:55 to about 08:05 UTC. It was preempted once, at 04:53:47 UTC, and replayed from the store; its
group then finished in 10,951 s. It made 2,072 new searches, 10.52 search-hours, counted from the store's new lines.
Billed: $5.18 for 03:00–07:00 UTC, with the 08:00 hour not yet in the report (about $0.1 more). The watcher never had
to stop it.

### 5. Outcomes (`verify.py`; `logs/verify_pk1*.txt`)

**10·x² + 1, hardest first, ties by the subgoal score: round 2.**
- **Rounds 0–1** are the earlier run's. The subgoal, x² mod N, was practised in round 1 as before (`T.law5`).
- **Round 2: the score.** The rest of the way from `T.law5` was posed as remainder/M1.4 → A, the same problem as
  rrem/T.law5. Scored against it, offset/remainder/M1.4 → A (10X mod N) was singled out with key (1, 11): one law
  away, mod(succ(g), N), exact on all rows, saving 11.
- **Round 2: the pick.** That piece costs 5, round 2's top cost. With the tie it was picked first: offset/remainder/
  M1.4 → A, law2/gcd(succ(X),N), law2/gcd(pred(X),N), law2/mul(add(X,N),N). The earlier run had picked
  law2/mul(add(X,N),X) instead.
- **Round 2: the merge.** Its program, `T.con2 MOV.A.X L.X[MOV.X.U M1.4]` (U = 3, A = X, then X times add 3² and
  reduce), and the corpus gave the merge M2.2 = `MOV.A.X T.con2 L.X[MOV.X.U M1.4]`.
- **Round 2: the target search** found the target at cost 4: `T.law5 M2.2 INC.X T.mod`. That is 39 primitive
  instructions.
- **The check:** right on all 7,260 pairs x < N ≤ 120 with 10⁹ steps. On machine_for(120) (721,000 steps), 5,061
  right and 2,199 out of steps: the 10X loop costs about 10·X·N steps.

**The triangular numbers, hardest first, the same change: no.** It is the earlier run in all 5 rounds. That run has no
subgoal in any round, so nothing is chained and the change never acts. Its step loses to pool order as before
(NEWTON.md §5).

**x² mod N, base 2, hardest first, the twin: no.**
- **Round 0** is the earlier run's.
- **Round 1.** The subgoal, x mod N on long x (law/mod(X,N)/rows), is a duplicate of law/mod(X,N)/x+QN. That
  twin was practised in the fourth pick's place: law/absdiff(X,N)/x+qN, law/mod(N,X)/x, law2/mul(pred(X),X) and
  law/mod(X,N)/x+QN. The earlier run had law2/div(N,pred(X)) fourth.
- **From round 2** the source posed the twin routine's step, digit/T.mod(X,N)@x+QN/*:
  - rounds 2–3: learnable at cost 4 (A.X) and 5 (A.U, A.T, and in round 2 also U.X and T.X);
  - round 4: only A.X, at cost 4.
- **The picks.** Rounds 2–3 picked laws at cost 5, which stood ahead in pool order. In round 4 the top cost was 5,
  then 4s.
- **The target.** No program that passes at big x was found. The subgoal is None from round 2, once x mod N on long
  x is trivial over the routine.

**x² mod N, base 2, anchored pairs, the big-x score: no.** It is the earlier run in all 5 rounds, with no new search.
No look or pair in any round found a program the target accepts, so the added charge is the same on both sides of
every score.

### 6. The earlier wins with each change on

| change | the nine held-out wins | binary x mod N, hardest first | binary x mod N, anchored pairs |
|---|---|---|---|
| ties by the subgoal score | same round, same program | same (round 3) | same (round 1) |
| the twin | same round, same program | **lost**: no, 5 rounds (round 3 before) | same (round 1) |
| the big-x score | not applicable (no check at big x) | same (round 3) | same (round 1, the same picks) |

Under the big-x score, anchored pairs' binary x mod N had 15 looks in round 1 that find the target, and they gained
the new charge. The round's four picks were already the steps' placements and law2/mod(N,pred(X)), so the picks
are the same.

### 7. Predictions against outcomes

| prediction (PREREG.md) | confidence | outcome |
|---|---|---|
| tiescore: the piece picked in round 2 | medium-high | **held** |
| tiescore: 10·x² + 1 does not come in 5 rounds | medium | **failed**: it came in round 2, by a merge, as the caveat allowed |
| tiescore: the triangular numbers are the earlier run | certain | **held** |
| twin: x mod N on long x's twin practised in round 1 | high | **held** |
| twin: the source poses its step from round 2, learnable | medium-high | **held** |
| twin: the step not picked | medium | **held** |
| twin: x² mod N does not come | medium-high | **held** |
| bigx_score: the anchored-pairs run is the earlier one | certain | **held** |
| the wins, tiescore: all 11 the same | certain | **held** |
| the wins, twin: 10 the same | certain | **held** |
| the wins, twin: hardest first's binary x mod N changes from round 1 | certain | **held** |
| the wins, twin: and still comes | medium | **failed**: it does not come in 5 rounds |
| the wins, bigx_score: anchored pairs' binary x mod N in round 1 | high | **held** |
| the wins, bigx_score: with the same program | medium | **held** (the same picks) |

### 8. Dead ends, loose ends, and what is left

- **The looks' reach.** Under anchored pairs on x² mod N, from round 3 the program that passes at big x is
  `D.2[step] MOV.X.A CLR.A T.law10`, over the twin's step (`L.A[INC.A] T.law10`). It is right on all 490 and 1,000
  rows and at big x, and costs 5: the digit loop over the step costs 2. Checked locally on round 3's library:
  - without `CLR.A` (cost 4) it is wrong, 183 of 490 rows, since T.law10 adds onto A;
  - the step's look (30,000 states) streams level 4 (784,720 states) and finds no accepted program, with the
    search's limit of 200 programs or with 20,000;
  - the item `D.2[T.look0]` is its own class in the search, not merged with another.

  A score that sees big x needs a look that reaches cost 5: a deeper look for the steps, as the deeper probe is for
  the subgoal.
- **The twin under anchored pairs.** It also changes x² mod N under anchored pairs from round 1, where it was no arm.
  Not run.
- **The twin's slot.** Practising the twin in the last pick's place delays the queue behind it. Two ways to test:
  - practise the twin as a fifth pick, outside the four;
  - pass it only the subgoal's ties, under the subgoal study's `_subpick` rule instead.
  Not tried.
- **The probe's reach, again.** In both twin runs the steps' cost-5 placements are unsolved in round 4, as in the
  two-source study. Hardest first's timing decides whether a cost-5 step is picked before the library outgrows the
  1M-state probe.

### 9. Files and how to rerun

Python with numba: the scratchpad venv that ran the digits study (`nvenv`). The plain one:
`~/github/sotaku/venv/bin/python`.

- **The changes:** `picks.py`; `make_learners.py` writes `learn_newton.py` and `learn_digits.py`; `configs.py`.
- **Replays and runs:** `check.py` (`off`, `on --variant=...`); `run.py` (here, from the stores);
  `modal_picks.py` (on Modal).
- **Results:** `verify.py`.
- **Records:** `PREREG.md` (read-only); `records/*_pk1.jsonl`; `logs/`: every output above, the runs' own stores, the
  Modal app's captured logs and the watcher's log.

The earlier stores are not in the repository:
- the held-out test's: Modal Volume `bar3-heldout`, v1/, `store_ctl_<rule>_ho1.jsonl`;
- the subgoal study's: Volume `bar3-subgoal`, v1/;
- the digits study's base-2 ones: Volume `bar3-digits`, v1/, joined on `bar3-picks` as `v1/store_pk1_digits.jsonl`;
- the two-source study's: `bar3/scale/newton/logs/`.

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/picks/make_learners.py
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.picks.check off [--digits] STORES -- RECORDS
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.picks.check on --variant=_rc_tie|_twin|_bigx [--digits] STORES -- RECORDS
```

```bash
cd cpu_first && bar3/scale/rowcheck/memcap.sh 10 <python with numba> -m bar3.scale.picks.run lp_clean_deep_sub_ns_rc_tie tenrho pk1 STORES
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/picks/modal_picks.py --fresh lp_clean_b_src_twin --targets dxsq,dxmod --tag pk1 --store store_pk1_digits.jsonl
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.picks.verify
```
