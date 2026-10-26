# A deeper look at the posed steps, and the twin as a fifth pick: pre-registration

Written 2026-10-26 01:24 PDT, before any run with one of the changes on. REACH.md §1 repeats this text unchanged.

## The question

PICKS.md §8 named two changes aimed at x² mod N at nine-digit x:
- **Under anchored pairs**, the five-token program that passes at big x, `D.2[step] MOV.X.A CLR.A T.law10`, is one
  level past every look, so the big-x score never sees it.
- **Under hardest first**, practising the subgoal's twin in one of the four slots delayed the queue and lost binary
  x mod N.

Does a deeper look at the posed steps let anchored pairs pick the step and get x² mod N at big x? Does the twin as
a fifth pick keep binary x mod N, and does it help x² mod N?

## The changes (`reach.py`, `make_learners.py`)

- **`deeplook`.** Anchored pairs' single look at a step the digits source posed gets a deeper budget, as the
  subgoal gets a deeper probe:
  - 1M states, so that it stores level 4 and streams level 5;
  - 3,600 s;
  - 20,000 programs checked per level instead of 200.

  The extra checks are there because at level 4 alone, 602 counting programs fit every row and fail at big x
  (PICKS.md §8). Pair looks keep their budget. The arm runs it with the picks study's big-x score on (`_bigx_deep`),
  since the deeper look is there for that score to act on.
- **`twin5`.** The subgoal's twin (picks.twins: the learnable problem the subgoal is a duplicate of) is practised as a
  fifth pick as soon as it is learnable. It neither replaces one of the four nor joins the subgoal's ties, so the four
  picks are the frozen learner's. It joins the subgoal set, so its routine counts as a practised subgoal's.

The flags of the earlier studies stay available. The arms run only the flags named for them below.

Learner copies: `learn_newton.py` and `learn_digits.py`, each written by `make_learners.py` from the learner frozen at
97f30b9 with that study's asserted replacements, then the row check's 3, the picks study's 7 and this study's 4. With
every flag off, 14 earlier records replay exactly from their stores (`check.py off`; `logs/check_off_*.txt`):
- the nine held-out wins;
- the picks study's 10·x² + 1 (its config `lp_clean_deep_sub_ns_rc_tie`);
- the digits study's four base-2 runs with the source.

## Settled from the records before this file (no run)

- **`deeplook` acts only on looks at the digits source's steps, which only anchored pairs runs, and only in the
  digits runs.** So it cannot change:
  - the nine held-out wins, 10·x² + 1, or hardest first's binary x mod N.

  Where it does act:
  - **anchored pairs' binary x mod N (an earlier win):** in round 1, 9 steps are learnable: the target's 5 and
    T.add(X,N)'s 4;
  - **the arm, anchored pairs' x² mod N:** no step is learnable in rounds 0–1, then 5, 10 and 12 in rounds 2–4.
- **`twin5` acts only where the subgoal is a duplicate of a learnable problem:**
  - x² mod N, hardest first, rounds 1–4;
  - binary x mod N, hardest first, round 1;
  - x² mod N, anchored pairs, round 1.

  No subgoal in the nine held-out wins, 10·x² + 1, or anchored pairs' binary x mod N is a duplicate before its target
  came.

## Runs

Each arm from the earlier runs' stores, 5 rounds, seed 0, the frozen budgets, with its flags:
- **`ct2a2_clean_b_src_bigx_deep`:** x² mod N, base 2, anchored pairs. Flags: big-x score, deeper look.
- **`lp_clean_b_src_twin5`:** x² mod N, base 2, hardest first; and binary x mod N, hardest first, which must still
  come in round 3. Flag: twin5.
- **The 12 earlier wins,** replayed with each change on, up to the round each came:
  - the nine held-out wins (`<rule>_deep_sub_deep`, `_twin5`);
  - the two binary x mod N wins (`<rule>_b_src_bigx_deep`, `_twin5`);
  - 10·x² + 1 (`lp_clean_deep_sub_ns_rc_tie_deep`, `_twin5`).

The tiny test, first: the deeper look at round 3's twin step, on the earlier anchored-pairs run's library, locally.
Searches the stores lack run locally under a memory cap when few, else on Modal from the stores. If x² mod N comes,
its program is checked as the digits study checked binary x mod N:
- on the rows;
- at big x;
- on 1,000 x below 10⁹ at N ≤ 10;
- on 1,000 x at N ≤ 40.

Money: soft $20, cap $25, launches guarded by `spend.py 4750`, one spend watcher.

## Predictions

| arm | prediction | confidence |
|---|---|---|
| deeper look, x² mod N base 2, anchored pairs | rounds 0–1 the earlier run's | certain |
| | round 2: no deeper look finds an accepted program, and the picks are the earlier run's | high |
| | round 3: the twin step's deeper look finds `D.2[step] MOV.X.A CLR.A T.law10` | medium-high |
| | then that step is picked | high, given the look |
| | x² mod N comes, in round 3 or 4 | low-medium |
| | if it comes, its program is right on the 1,000 and 1,000 x | high |
| twin5, x² mod N base 2, hardest first | the twin practised as a fifth pick in round 1; its step learnable from round 2, not picked; no | high; medium-high; medium-high |
| twin5, binary x mod N, hardest first | comes in round 3 | medium |
| the earlier wins, deeper look | 11 unchanged | certain |
| | anchored pairs' binary x mod N comes in round 1 | high |
| | with the same program | medium |
| the earlier wins, twin5 | 11 unchanged | certain |
| | hardest first's binary x mod N: the arm above | |

Why:
- **The deeper look, round 2.** x² mod N on the pairs (T.law10) is practised in round 2 and is not yet a routine,
  so the program is longer than five tokens.
- **The deeper look, round 3.** The program is right on all rows and at big x on round 3's library (PICKS.md §8). A
  1M-state look stores level 4 (about 0.8M states) and streams level 5. Checking 20,000 programs per level should
  reach it among the look-alikes.
- **The deeper look, the pick.** Found, it gains the big-x charge, about 2,193 bits, which no other candidate gets.
- **The target.** The target search (3M states, 200 programs per level) must then find the same five tokens with
  the step as a routine. At level 5 the look-alikes may fill its 200 programs, unless a merge shortens the program.
  The guided search is the other way.
- **The fifth pick, x² mod N.** The twin is practised as before (PICKS.md §5), and its step, posed from round 2,
  waits behind the cost-5 problems ahead of it in pool order. In round 4 it is out of the probe's reach.
- **The fifth pick, binary x mod N.** The four picks of round 1 are the earlier run's, plus the twin. The queue of
  cost-5 problems is the earlier run's, so the steps' turn comes in round 3 again. But one more routine widens every
  level, and round 3 was the last round in which the steps' cost-5 placements were in the probe's reach.
- **The wins under the deeper look.** The target's steps' looks find the target at cost 2–3, and a deeper look stops
  at the same level, so their gains are the same. T.add(X,N)'s steps get deeper looks too. One of those could find an
  accepted program the shallow look missed and change the picks.

What is reported for each run: whether and in which round its target came, the program and its checks; what the
changes did each round (the deeper looks' results, the fifth picks); and the first round that differs from the
earlier run.
