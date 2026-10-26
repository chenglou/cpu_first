# The duplicate check on a problem's own rows: pre-registration

Written 2026-10-25 20:04 PDT, before any run with the new check on. ROWCHECK.md §1 repeats this text unchanged.

## The question

The frozen learner calls a solved problem a duplicate when its best program has the same signature as an earlier
learnable problem's, and an alias when it has a routine's or a primitive's. A signature is what a program does to 24
random machine states (bar3/learner.Signatures). Every program that dies on all 24 shares one signature. That is how
10X mod N, the piece the offset posed for 10·x² + 1, was called a duplicate of 3X (NEWTON.md §3.4).

Enumerative synthesis tells programs apart by their outputs on the task's examples: TRANSIT (Udupa et al., PLDI
2013, §1) prunes "by considering two expressions equivalent if they evaluate to the same values for the given concrete
examples". Escher (Albarghouthi, Gulwani and Kincaid, CAV 2013, §3.3) names it observational equivalence reduction,
and proves its search stays complete under it. The new check applies that to each problem on its own rows. Does it
unlock what the earlier studies named?

## The change (`rowcheck.py`, `make_learners.py`)

With cfg["rowcheck"], a shared signature only proposes. A problem is an alias when a primitive or routine with its
signature is right on all its own rows, and a duplicate when an earlier learnable problem's best program with its
signature is. Its own rows are the instances it is accepted on and its check instances, its registers set as its job
sets them. Otherwise it stays learnable. Nothing else changes.

The flag is added to copies of the two learners the reruns need, each written by `make_learners.py` from the learner
frozen at 97f30b9 with that study's asserted replacements, then this study's three:
- `learn_newton.py`, the two-source study's learner: for 10·x² + 1, and with no source on, the held-out test's;
- `learn_digits.py`, the digits study's.

Run before this file (`check.py off`; `logs/check_off_*.txt`): with the flag off, 14 earlier records replay exactly
from their stores:
- the nine held-out wins and the two-source study's hardest-first run on 10·x² + 1, on `learn_newton.py`;
- the digits study's four base-2 runs with the source, x² mod N and x mod N under both rules, on `learn_digits.py`.

## The two pieces, settled from the records (no run)

- **10·x² + 1: the fix separates them.** On its own rows 10X mod N differs from 3X (at N = 7, x = 1..6: 3, 6, 2, 5, 1, 4
  against 3, 6, 9, 12, 15, 18; NEWTON.md §3.4).
- **x² mod N: the fix does not.** The problem the digits study's subgoal found, x mod N on long x
  (`law/mod(X,N)/rows`), is a duplicate of the law book's x mod N on short pairs (`law/mod(X,N)/x+QN`). In every
  round of the records the two problems' best programs are the same program (`L.X[T.mod MOV.X.A]`, later
  `MOV.A.X M1.1`), so that program is right on the twin's own rows too. The check at big x is the target's acceptance,
  not part of a practice problem's rows.

## Runs

The arms, from scratch, 5 rounds, seed 0, the frozen budgets, with the check on:
- 10·x² + 1 under hardest first, both sources (`lp_clean_deep_sub_ns_rc`);
- x² mod N in base 2 with the digits source, under both rules (`lp_clean_b_src_rc`, `ct2a2_clean_b_src_rc`).

The earlier wins, replayed with the check on up to the round each came:
- the nine held-out wins (`<rule>_deep_sub_rc`, no source on);
- binary x mod N in the digits study, under both rules (`<rule>_b_src_rc`).

Each run starts from the earlier runs' stores, and reads a stored search back instead of repeating it. Searches the
stores lack run locally under a memory cap (`memcap.sh`). Modal is used only for what is too big to run locally.
Money: soft $20, cap $25.

## Predictions

| arm | outcome | confidence |
|---|---|---|
| 10·x² + 1, hardest first | no, 5 rounds | medium |
| x² mod N, base 2, hardest first | no, 5 rounds | high |
| x² mod N, base 2, anchored pairs | no, 5 rounds | high |
| the 11 earlier wins | each in the same round with the same program | medium |

Why:
- **10·x² + 1, hardest first.** If rounds 0–1 keep the earlier picks, the piece is posed in round 2 and is learnable
  at cost 5 in rounds 2 and 3 (medium). It is still not picked. Hardest first takes the four costliest problems, ties
  in pool order. The offsets come last in the pool, and 24 (round 2) and 12 (round 3) problems learnable at cost 5
  stood ahead of them in the earlier run. In round 4 the piece was out of the 1M-state probe's reach.
- **x² mod N, base 2, hardest first.** x mod N on long x stays a duplicate of its twin (high), and that twin, at cost
  2–3, is never among hardest first's four costliest. So no routine holds x mod N on long x, and the source poses no
  step for x² mod N, as before.
- **x² mod N, base 2, anchored pairs.** The twin was practised in round 1 and the step was learnable from round 2.
  The blocker was the pick score: a look with the step gained −0.1 bits, since counting programs already fit every
  row. The check does not touch that score.
- **Elsewhere.** The check also reaches every other problem whose best program shares a signature with a program
  that is wrong on its rows. Loop-heavy programs die on the random states (3,450 steps at k = 7), so in at least one
  of the 14 runs some status changes from duplicate or alias to learnable (high). Picks change in at least one round
  of at least one run (medium). A changed pick in rounds 0–1 can change an earlier win. That is why the wins are
  rechecked.

So I expect the fix to change none of the three arms' outcomes. The local replays settle that at no cost. If they do
not change the arms, nothing runs on Modal, and that is the report.

What is reported for each run: whether and in which round its target came, the program and its check; every status
the check changed, by round; and the first round whose picks, library or target search differ from the earlier
run's.
