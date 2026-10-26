# Two sources from Newton's differences: pre-registration

Written 2026-10-25 13:28 PDT, before the runs, and before reading the output of the local replay of 10·x² + 1 with
the sources on (`check.py on`, started 13:24, still running). NEWTON.md §1 repeats this text unchanged.

## The question

The held-out test (HELDOUT.md §7) and the subgoal study (SUBGOAL.md §8) each named a practice problem no source
poses:
- the triangular numbers' step, (Y + X + 1) mod N: y(x + 1) from y(x) and x, a running sum's step;
- for 10·x² + 1, 10X mod N: the rest of the way from x² mod N, (10X + 1) mod N, less its value at X = 0.

These are the first two moves of Newton's forward differences: take away the constant term, and difference, whose
inverse is the running sum. Does each source pose its piece in the learner, and do the targets then come?

## The learner (`learn.py`, written by `make_learner.py`)

The learner frozen at 97f30b9 (`bar3/scale/subgoal/learn.py`, configs `lp_clean_deep_sub` and
`ct2a2_clean_deep_sub`), with two sources, on in the configs `_ns` (`configs.py`) and nothing else changed:
1. `rsum` (`sources.running_sum`), right after the recurrences: y(x + 1) from y(x) and x on the data's pairs
   x < N, with y(x) in A, U or T (updated in place) and x in X, U or T: 7 placements, every round.
2. `offset` (`sources.offsets`), right after the routine remainders: for each rest of the way in the pool (the
   routine remainders and the near misses' remainders), its answer less its answer at X = 0, mod N; not posed when
   that answer is 0 for every N. Not for the target itself: on 10·x² + 1 the target's own offset, 10·x² mod N, is one
   law from the target (add 1, reduce), so the subgoal's score ranks it (1, 35), above x² mod N (0, 24), and it would
   take the deeper probe that solved x² mod N in the earlier run (`logs/tiny_pose.txt`; NEWTON.md §3).

With neither flag the copy is the frozen learner: the four earlier records below replay exactly from their stores
(`check.py off`).

## What ran before this file

- `tiny.py pose` (no search, `logs/tiny_pose.txt`): the running sum's step is posed from round 0 on both formulas
  (21 rows at k = 7, 45 at k = 10); on the triangular numbers no offset (their answer at X = 0 is 0); on
  10·x² + 1, 10X mod N is posed once x² mod N is a routine (hardest first's library at round 2).
- `check.py on`, the triangular numbers, hardest first, rounds 0–2, the searches the stores lack run locally when
  cheap (`logs/check_on_tri.txt`): the steps are learnable from round 1 (rsum/A.X at cost 4; A.U, A.T, U.X and T.X
  at 5) and not picked; every round's picks, library and target search are the earlier run's.
- `check.py on`, the triangular numbers, anchored pairs, rounds 0–1, every missing search run locally
  (`logs/check_on_tri_ct2a2_wide.txt`): **rsum/A.U is picked in round 1 and the target comes in round 1**,
  `L.X[M1.4]`. That is an outcome seen before this file, not a prediction.
- A count from the earlier records (no search): under hardest first, in rounds 0–3 of both formulas, at least four
  problems learnable in the earlier run cost more than 5 or cost 5 and stand ahead of the new sources in pool
  order; in round 4 the costliest of them cost 4. So a new problem still learnable at cost 5 in round 4 is among
  that round's four picks, and none is before.

## Runs

The triangular numbers (x·(x + 1)/2 mod N, held-out #10, `tri`) and 10·x² + 1 (`tenrho`), under both rules, with
both sources, 5 rounds, seed 0, the frozen budgets: 4 runs. They start from the earlier runs' stores, so a search
those runs made is read back, not repeated (their records reproduce exactly from them). The earlier records are the
runs without the sources (`bar3/scale/heldout/records/*_tri_ho1.jsonl`, `bar3/scale/subgoal/records/
*_deep_sub_tenrho_s1.jsonl`); no run without them is needed.

The check on the earlier wins: the 9 held-out runs whose formula came, all in rounds 0–1: (x² + x), (x + 1)², 3·x²
and x⁴ mod N under both rules (`x2x`, `x1sq`, `three`, `x4`), and 2^x mod N under anchored pairs (`pow2`). Each is
replayed with both sources up to the round its formula came, the same way.

Every program found is checked on every pair x < N up to N = 120. Money: soft $20, cap $25, launches guarded by
`spend.py 4750` and a watcher that stops the apps at the cap.

## Predictions

| formula | hardest first | anchored pairs |
|---|---|---|
| triangular numbers | comes in round 4 (low-medium) | round 1 (seen before this file; the run should give it again: high) |
| 10·x² + 1 | no (medium) | no (high) |

- **Triangular numbers, hardest first.** The cost-5 placements of the step take round 4's picks (the count above),
  if their cost is still 5 then. The target is then `L.X[step INC.U]` (3 tokens) over rsum/A.U, within the target
  search. Against: a merge by round 4 can make the step cost 4, and then 24 problems at cost 4 stand ahead of it.
- **10·x² + 1, hardest first.** x² mod N is practised in round 1 as before (the subgoal's deeper probe), its rest of
  the way is posed from round 2, and its offset, 10X mod N, with it. That offset is learnable at cost 5 (SUBGOAL.md
  §8: at the 1M-state probe) and is picked in round 4 (medium), not before. The target does not come in round 4: the
  shortest program over those routines is 6 tokens (`T.law5 MOV.X.A T.off MOV.X.A INC.X T.mod`), past the target
  search (medium). Risk: a near miss's remainder equal to the target posed again has the target's own offset as its
  offset, which would take the subgoal from x² mod N in round 0 or 1 (low).
- **10·x² + 1, anchored pairs.** The earlier run never practised x² mod N, so its rest of the way and that rest's
  offset are never posed: the run is the earlier one, with the new problems' probes and looks added (high).
- **The earlier wins.** All 9 come in the same round with the same program (medium-high): in rounds 0–1 hardest
  first has no free slot at cost 5 for the new problems, and under anchored pairs their looks do not shorten
  these formulas.

What is reported for each run: the round, or "no, 5 rounds"; the program and its check; for each new problem, the
rounds it was posed, learnable (with its cost) and picked; for the earlier wins, whether each comes in the same
round with the same program, and if not, where the run first differs.
