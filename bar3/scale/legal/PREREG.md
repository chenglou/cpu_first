# The composition route's last two steps, legally: pre-registration

Written 2026-10-26 19:20 PDT, before any run on a GPU. LEGAL_TEST.md §1 repeats this text unchanged and records the
time and hash it was frozen with.

## The question

bar3/LEGAL_SOURCES.md §5 named this test 2. In the composition harness, round 4's s4_x2 got H1's round, and its last
two steps read the small target, x'² mod N' computed by Python's integers. LAWNEAR's law test picked x² as a near miss.
The rest of the way from x² was posed with the small target's answers. In a legal run the only target is H1's own
answer rows at full size. So, from s4_x2's library at the start of its level 5, at full size, with the learner's own
routines in every test:
- **Q1.** Does the law view by routines pick x² among the round search's held states?
- **Q2.** Does the rest of the way from the value it picks, posed on the data's rows, find the reduction?
- **Q3.** Is the round they make right?

## What is legal here, and what is not

- **Legal.** The answers are the data's: H1's T = 1 rows, read as digit strings. Values are what the learner's
  programs leave on its own machine. The law test runs two of the learner's routines and compares digit strings:
  - the chain `T.red.Bx+d.AN` = `B[link]`, practised at level 1;
  - the comparison `T.lt.pair.AN`, whose borrow says "below N".

  The rest's inputs are a value the learner's program leaves. No test computes a number from a value; Python's
  integers appear only in the record's diagnostics.
- **Not legal, not tested.** The library itself. s4_x2's cuts were answered by Python's integers, and its value
  routines were posed and chosen against the small target. The "pieces" arm drops the value routines. The cuts'
  routines stay: their legal counterparts are the book's own constructions (LEGAL_SOURCES.md §3.13). Whether a legal
  climb reaches this library is test 3.
- **Not tested.** The climb's own steps at full size: LAWNEAR's four-state limit, the subgoal score, the rest posed
  in a level, REST_DEEP's trigger. This test runs each step directly.

## Design (`rowtarget.py`, `offline.py`)

- **Rows.** The composition harness's 400 answer rows: `core_c3.load_rows(400, seed=1, T=1)`, the same permutation.
  The rows780 arm takes the first 780 of that permutation, the 400 among them. 780 is the most rows a state can hold
  under libcore's state hash, 8,192 words.
- **Q1.**
  - The held states: the round search's vocabulary (the primitives, then the routines in the library's order,
    straight-line items), on all 400 rows. Every state of cost ≤ 3 is kept.
  - The values: registers A, X, Y and S of every held state. Distinct values in LAWNEAR's order: by cost, then by
    register (A, X, Y, S), then by the state's index.
  - The law test (`law_mod`): the chain, then the comparison. A value passes when the chain's result equals the answer
    on every row where it is below N, on at least 4 such rows, and the value is not the answer itself.
  - The pick: the first value that passes.
  - Diagnostics, by Python's integers: is it x²? Does numpy's law view (all rows) agree?
- **Q2.**
  - The value: the pick's program. If nothing passes, x² is handed instead (`LDA.X STA.Y T.mul.x,o.XY`, marked
    "handed").
  - The problem: X = that value, N = N, the answer the row's own. The probes are the first 8 answer rows, the round's.
    A program is accepted when it leaves the answer in A on all 400 rows.
  - The vocabulary: the composition vocabulary at full size, the primitives and every routine, with loops of bodies up
    to 3.
  - The search: `spcore_s.bfs_roots`, the climb's probe, to cost 7: at 1M states and 900 s; if unsolved, at
    REST_DEEP's 10M and 900 s. SPELL 16.
- **Q3.** The round: the value's program, STA.X, then the rest's first accepted program. It is checked on the 400
  rows and on all 1,000 T = 1 rows; then `ladder.py`, locally, on both ladders, T = 1 to 64.
- **Arms, in order:** full (the library as recorded, 44 entries); pieces (less the ten value routines and the merge
  that calls one: 33 entries); rows780 (Q1 alone). One B200, at most 2,700 s.

## Seen before this was written

Local, on the CPU: `smoke.py`, and dry runs of `offline.py` at tiny sizes.
- x² by the learner's multiply is x² on all 1,000 T = 1 rows, and clean.
- The law test by routines, on handed values:
  - x² and x² + N pass, with 8 exact rows of the 400 and 20 of the 1,000;
  - 2x (400 exact rows, none equal) and x² + 1 fail.
- The rest's acceptor accepts s4_x2's own reduction from x², `LDA.N P[BRG.X T.red.Bx+d.AN]`, on the 400 rows, and
  rejects the chain alone. That round is right on all 1,000 T = 1 rows and on both ladders (ladder.py).
- Q1 on 16 rows to cost 2: 1,406 distinct values, none passes. Q2 at 20,000 states with loops of 1-item bodies: no
  program (level 3 streamed).
- The first 780 rows hold 16 rows with x² < 11·N, 3 with x² < N.

So the pieces work when handed. The questions are whether the searches find them, and whether the law test by
routines, which sees 2% of the rows, picks only x².

## Predictions

| what | prediction | confidence |
|---|---|---|
| Q1 full: the held states | about 100k–150k of cost ≤ 3 (level 2 about 2,700) | medium |
| Q1 full: x² | passes on 8 exact rows: `LDA.X STA.Y T.mul.x,o.XY`, register A, cost 3 | high |
| Q1 full: the pick | x², the first value that passes (none at cost 1 or 2) | high |
| Q1 full: anything else | nothing else passes; numpy's view passes the same one value | medium |
| Q2 full | the rest at cost 4 by the 1M probe, `LDA.N P[BRG.X T.red.Bx+d.AN]` or a spelling; 400/400 | medium |
| Q2 full, if not at 1M | found at cost 4 by the 10M probe | medium |
| Q3 full | the round right on 400/400 and 1,000/1,000; both ladders 768/768 on all 14 rungs | high, given Q2 |
| pieces | the same pick and the same rest, with fewer held states and a smaller vocabulary | medium-high |
| rows780 | x² passes on 16 exact rows; nothing else passes | medium-high |
| time and money | 10–25 minutes on the B200, about $1–3 | medium |
| the conclusion | given this library, both steps work legally at full size (see below) | medium |

The conclusion, in full: given this library, the route's last two steps work legally at full size. The law view by
routines picks x² from 8 rows, and the rest of the way posed on the data's rows finds the reduction.

## Money

Soft $5, cap $8. `integration/ablation_s3/spend.py 4750` runs before the launch. One B200 for at most 2,700 s is at
most $4.8 at $6.37 an hour. The app is stopped when the run ends.
