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
