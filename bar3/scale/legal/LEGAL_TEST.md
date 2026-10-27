# The composition route's last two steps, legally

*Written 2026-10-26. One B200 run (Modal app `bar3-legal`, `ap-Z9QkxEXcTDN3cy04JySmvU`), about $3. The rest ran
locally on the CPU. Code is in this folder. The record is `records/lt1.json` (log `records/lt1.log`); the local checks
are in `logs/`. Nothing was committed.*

<!-- PLAIN -->
## In plain words

**The question.**
- Round 4's s4_x2 got H1's round with bar 3's sources in place of the book's row and its last law.
- Its last two steps read a target computed with Python's integers, the small target x'² mod N' at 1–2 digits:
  - LAWNEAR's law test picked x² against it;
  - the rest of the way from x² took its answers from it.
- A legal training run has no such target. bar3/LEGAL_SOURCES.md named the cheapest decisive test: rebuild those two
  steps on H1's own answer rows at full size, with every test made of the learner's own routines, and run them from
  s4_x2's library. Do they still work?

**The full-size target mode** (`rowtarget.py`). Nothing in it computes a number from a value.
- The target is H1's T = 1 rows, read as digit strings.
- The law view tests "the answer is g mod N" with two of the learner's routines and a comparison of digit strings:
  - its chain (ten conditional subtractions, exact below 11·N);
  - its comparison, which finds the rows where the chain is exact.
- The rest of the way from a value is posed on the same rows, at full size.

**What came** (one B200, 26 minutes, about $3; predictions written and frozen first):
- **The law view by routines picks x², and only x².** The round search held 41,007 distinct values at cost ≤ 3.
  - Exactly one passes: x², `LDA.X STA.Y T.mul.x,o.XY`, on its 8 exact rows of the 400.
  - numpy's test on all 400 rows agrees value for value.
  - Without the small target's value routines (31,586 values), and on 780 rows (16 exact rows): the same, x² alone.
- **The rest of the way from x², posed on the data's rows, finds the reduction.**
  - Cost 4, by the regular 1M-state probe, in 84 s.
  - Four spellings were accepted on the 400 rows, s4_x2's own `LDA.N P[BRG.X T.red.Bx+d.AN]` among them.
- **The round is right on the 400 rows and on all 1,000 T = 1 rows.** But the spelling it took, the first one listed,
  `P[INC BRG.X T.red.Bx+d.AN]`, fails on unseen 24-bit moduli: 757–764 right of 768 on that ladder's rungs.

**So both steps work legally at full size, but no training row can reject one of the look-alike spellings.**
- The law test by routines sees only 2% of the rows, those where the chain is exact. Still it singles out x², as
  numpy's test on every row does.
- The rest of the way finds the reduction within the regular probe, as s4_x2 did at the practice widths.
- H1's training moduli are 13, 16 and 20 bits. The first spelling is right on all of them and wrong on some 24-bit
  ones, so no training row can reject it.
- A second look-alike, `P[BRG.X T.red.Bx+d.AN INC]`, depends on what A holds when it starts, a register it is not
  given. It passed the acceptor's one dirty start, and fails in the round, where A starts as x².

**What picks the right spelling** (checked after the run, not pre-registered):
- **The climb's own choice** among accepted programs, fewest steps executed, keeps `LDA.N P[...]` (445 steps against
  448). It is right on all 14 rungs of both ladders.
- **A legal identity check** rejects both look-alikes: reduce r + q·N' (whose answer is r by construction), with N'
  a training modulus moved up one digit (a digit move). The right spellings get 1,000 of 1,000 right; the look-alikes
  997 and 964.

**What it does not show.**
- **The library.** It is s4_x2's own, from a run whose cuts were answered by Python's integers and whose value
  routines were chosen against the small target. Whether a legal climb reaches it is test 3.
- **The climb's own steps at full size:** LAWNEAR's state limit, the subgoal score, posing the rest in a level, the
  routine choice. This test ran each step directly.
- **The cost of full size** is the vocabulary: built on full-size machines it has 286k item classes and took 10–13
  minutes, most of the run. The searches took seconds to 84 s.

**Next: test 3, the tiny core with this mode in its level loop.**
- Full-size roots probed apart from the practice roots, with the climb's routine choice.
- For a reduction found from the data, the identity check at one more digit.
- Its cost turns on building the full-size vocabulary once and caching it (§8).
<!-- /PLAIN -->

## Details

### 1. Pre-registered

PREREG.md was first frozen (read-only) at 19:21:13 PDT, sha256 `4811785a…7017`. Its header gave the time it was
written as 19:25, a slip: it was written by 19:20. Before any run the header was corrected to 19:20 and the file
re-frozen at 19:21:25 PDT, sha256 `d9119076d93ec855315c42cdac0e9af98047d7f7287fbc5e77407311db89d8ff`
(`logs/prereg_sha256.txt`). The run started at 19:21:35. The code that ran is the code hashed with it, unchanged:
`rowtarget.py` `00c87df8…`, `offline.py` `1d9bde9f…`, `modal_legal.py` `74ece44e…`, `ladder.py` `2fc31e3e…`,
`smoke.py` `be766630…`. The text, unchanged:

<!-- PREREG -->
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
<!-- /PREREG -->

### 2. The full-size target mode (`rowtarget.py`)

The small target's roles in the harness (`oracle_s.SmallTarget`), taken over by H1's own rows:

| role | the small target (oracle_s.py) | the full-size mode (rowtarget.py) |
|---|---|---|
| the target's rows | the rows' top parts; x'² mod N' by Python | H1's T = 1 rows, read (`answer_rows`) |
| the look's held states | its own search at the practice widths | the round's vocabulary on every row (`held`) |
| a value | a near miss's register, practice widths | a held state's or a program's register (`value`) |
| the law view | numpy or torch arithmetic on g, X and N | the learner's chain and comparison (`law_mod`) |
| the rest of the way | X = the value; the small target's answers | X = the value; the row's own answer (`rest_root`) |
| its acceptance | 60 rows a verify width, by Python | the answer rows at full size (`rest_acceptor`) |

Two details of the search:
- **Every state of cost ≤ 3 kept.** bfs_roots keeps only goal states at its last level. The held-state search uses a
  goal that every state but the root reaches (`Kept`) and an acceptor that accepts nothing, so it keeps its last
  level whole and never stops early.
- **780 rows at most.** libcore's state hash (`hash_rows`) multiplies a state's 64-bit words by a table of 8,192. A
  state over 1,000 rows has 10,376 words, so a search can hold at most 780 rows. The first dry run on 1,000 rows
  failed there, before the pre-registration, and the wide arm became rows780.

What the mode does not have yet is the climb's level loop. A practice probe searches every root of a level in one
bfs_roots call, at the practice settings. Full-size roots have another setting and another vocabulary, so the climb
would probe them apart, as it already searches the round apart (§8).

### 3. Local checks before the run (CPU, $0)

`smoke.py` (7–8 s, peak 314 MB) and two dry runs of `offline.py` at tiny sizes:
- x² by the learner's multiply, `LDA.X STA.Y T.mul.x,o.XY`, is x² on all 1,000 T = 1 rows, and clean;
- the law test by routines on handed values:
  - x² and x² + N pass, with 8 exact rows of the 400 answer rows and 20 of the 1,000;
  - 2x (exact on every row, never equal) and x² + 1 (8 and 20 exact rows, never equal) fail;
- the rest's acceptor accepts s4_x2's reduction from x² on the 400 rows and rejects the chain alone. The round from it
  is right on all 1,000 T = 1 rows, and `ladder.py` gives 768/768 on all 14 rungs (s4_x2's own result, again);
- Q1 on 16 rows to cost 2: 1,406 distinct values, none passes. Q2 at 20,000 states with loops of 1-item bodies: no
  program.

### 4. The run (`records/lt1.json`, one B200, 1,542 s)

| | full (44 entries) | pieces (33) | rows780 (44) |
|---|---|---|---|
| rows | 400 | 400 | 780 |
| held states, costs 0–3 | 1, 63, 2,666, 99,745 | 1, 53, 2,110, 76,424 | 1, 63, 2,666, 99,793 |
| distinct values (A, X, Y, S) | 41,007 | 31,586 | 41,014 |
| held search, law test | 2.6 s, 1.8 s | 0.9 s, 1.3 s | 3.3 s, 3.3 s |
| values that pass | 1: x², 8 exact rows | 1: x², 8 exact rows | 1: x², 16 exact rows |
| numpy's view (all rows) | the same 1; none differs | the same 1; none differs | the same 1; none differs |
| the vocabulary at full size | 285,819 classes, 758 s | 234,149 classes, 609 s | — |
| the rest at 1M states | cost 4, 83.5 s | cost 4, 74.6 s | — |
| its levels, costs 0–4 | 1, 63, 2,713, 105,564, 5.39M streamed | 1, 53, 2,174, 83,530, 4.19M streamed | — |
| the round, 400 and 1,000 rows | right, right | right, right | — |

- **The passing value** is x² in A, `LDA.X STA.Y T.mul.x,o.XY`, at cost 3: clean, and x² on every row (Python's check).
- **Most values have no exact row.** 27,449 of the 41,007 are at least 11·N on every row, or dirty. Those that have
  exact rows have up to 41 of the 400, and none but x² matches the answer on all of them.
- **The rest's accepted spellings,** in the order bfs_roots lists them (fewest top-level items first), the same in
  both arms: `P[INC BRG.X T.red.Bx+d.AN]`, `P[BRG.X T.red.Bx+d.AN INC]`, `LDA.N P[BRG.X T.red.Bx+d.AN]`,
  `CLA P[BRG.X T.red.Bx+d.AN]`.
- **The round** as pre-registered takes the first: `LDA.X STA.Y T.mul.x,o.XY STA.X P[INC BRG.X T.red.Bx+d.AN]`.
  `ladder.py` (`logs/ladder_lt1_full.txt`, `ladder_lt1_pieces.txt`):

| ladder | T = 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| seen moduli (13, 16, 20 bits) | 768 | 768 | 768 | 768 | 768 | 768 | 768 |
| unseen moduli (14, 18, 22, 24 bits) | 764 | 764 | 759 | 758 | 758 | 758 | 757 |

### 5. After the run (not pre-registered; `posthoc.py`, `logs/posthoc_lt1.txt`, local, 57 s)

**Each accepted spelling as the round's rest**, on both ladders (of 768) and on the training rows:

| spelling | seen moduli, T = 1 … 64 | unseen moduli | the 400; all 1,000 T = 1 |
|---|---|---|---|
| `P[INC BRG.X T.red.Bx+d.AN]` | 768 on every rung | 764, 764, 759, 758, 758, 758, 757 | right; right |
| `P[BRG.X T.red.Bx+d.AN INC]` | 765, 765, 761, 751, 752, 751, 749 | 720, 705, 685, 667, 650, 654, 648 | wrong; wrong |
| `LDA.N P[BRG.X T.red.Bx+d.AN]` | 768 on every rung | 768 on every rung | right; right |
| `CLA P[BRG.X T.red.Bx+d.AN]` | 768 on every rung | 768 on every rung | right; right |

- **Where they fail.**
  - The first look-alike fails only on 24-bit moduli (8-digit N; 4 rows at T = 1). Its INC moves the pointer every
    step, and the chain's subtractions start at the pointer. So most likely their borrow runs from mid-register and
    wraps round, which on these rows goes wrong only when N has 8 digits (not traced further).
  - The second fails on 13- and 20-bit moduli too. Neither INC spelling sets A first. With its INC after the chain,
    this one is wrong on some of the 400 rows when A starts as x², as in the round. The acceptor's one dirty start
    passed it.
- **The climb's own choice** (`SelfClimb.choose_routine`: keeps its inputs, registers touched, size, steps executed).
  The four tie on the first three: none keeps X, which BRG.X shifts out; two registers are touched; size 10. Steps:
  448 for the two INC spellings, 445 for `LDA.N` and `CLA`. So the climb would keep `LDA.N P[BRG.X T.red.Bx+d.AN]`,
  the first of the two at 445, which is right on every rung.
- **A legal identity check.** Each spelling reduces 1,000 instances r + q·N', whose answer r is known by construction.
  N' is a training modulus as it is, or moved up one digit (N' = 10·N + d). The instances were built here with Python's
  integers, to ask whether such a check tells the spellings apart. In a run, the book's own moves would build them:
  the digit move (stage 3's Bx+d) and the learner's addition.

| spelling | N' as in the data | N' one digit longer |
|---|---|---|
| `P[INC BRG.X T.red.Bx+d.AN]` | 1,000 | 997 |
| `P[BRG.X T.red.Bx+d.AN INC]` | 999 | 964 |
| `LDA.N P[BRG.X T.red.Bx+d.AN]` | 1,000 | 1,000 |
| `CLA P[BRG.X T.red.Bx+d.AN]` | 1,000 | 1,000 |

So at the data's own sizes only the routine choice tells the first look-alike apart. One digit more of modulus, which
a digit move gives legally, exposes both.

### 6. Predictions against outcomes

| prediction | outcome | |
|---|---|---|
| Q1 full: 100k–150k held states of cost ≤ 3, level 2 about 2,700 | 102,475; level 2 2,666 | held |
| Q1 full: x² passes on 8 exact rows, register A, cost 3 | so | held |
| Q1 full: x² is the first value that passes, none at cost 1 or 2 | so | held |
| Q1 full: nothing else passes; numpy's view the same one value | so: 1 passes, both 1, either alone 0 | held |
| Q2 full: cost 4 at 1M, s4_x2's reduction or a spelling; 400/400 | cost 4 at 1M; 4 spellings, s4_x2's too | held |
| Q2 full, if not at 1M: cost 4 at 10M | not needed | — |
| Q3 full: right on 400 and 1,000 rows | so | held |
| Q3 full: both ladders 768/768 on all 14 rungs | seen 768 on all 7; unseen 757–764 (24-bit N) | **missed** |
| pieces: the same pick and rest, fewer states, smaller vocabulary | so (76,424 states, 234,149 classes) | held |
| rows780: x² passes on 16 exact rows, nothing else | so | held |
| time and money: 10–25 minutes, about $1–3 | 25.7 minutes; about $3 (§7) | at the edge |
| the conclusion: both steps work legally at full size | they do; the spelling needs the climb's rule | held, see Q3 |

Eight of nine pre-registered outcomes held; the ninth, the ladders, missed. The time and money estimate landed at its
upper edge. The miss is in what the test chose: Q3 took the first accepted spelling. The pre-registration did not
foresee that acceptance on H1's training rows would admit look-alikes.

### 7. Cost

- One B200: the container ran from 19:21:35 to about 19:49 PDT, when the app was stopped, of which the module took
  1,542 s. At $6.37 a B200-hour that is about $3.
- `cost.py` reads the billing report's hours for the two apps (the run's and the fetch's). The report covers complete
  hours only; at 02:52 UTC it showed $0.00 for the 02:00 UTC hour, so the billed figure was not in yet.
- Spend before the launch: $3,746.02 this month (`integration/ablation_s3/spend.py 4750`: LAUNCH OK).
- Under the soft $5 and the cap $8. Both apps are stopped.

### 8. What it does not show, and test 3

- **The library is not legally reached.** s4_x2's cuts were answered by Python's integers, and its value routines were
  posed and chosen against the small target. The pieces arm drops the value routines and gives the same result, so
  they did not matter here. The cuts' routines did: the chain is the step's cut, and the multiply is built over the
  row's cut. Their legal counterparts are the book's constructions (LEGAL_SOURCES.md §3.13).
- **The climb's own steps at full size were not run.** That includes:
  - LAWNEAR's four-state limit;
  - the subgoal score (whose law view would be this one);
  - posing the rest in a level, and REST_DEEP's trigger;
  - the routine choice, which §5 shows matters.
- **Test 3**, the tiny core with this mode in its level loop:
  - full-size roots probed apart from the practice roots;
  - the held states taken from the round's own search;
  - the climb's routine choice;
  - for a reduction found from the data, the identity check at one more digit as part of acceptance.
- **Its cost turns on the vocabulary.** On full-size machines the composition vocabulary has 234k–286k item classes
  and took 10–13 minutes to build; the searches took seconds to 84 s. libcore's enumeration can carry a signature
  cache across calls, so a climb would pay that once and then only for new routines. Bodies of up to 2 items would
  also do here: s4_x2's reduction has a 2-item body, and both look-alikes have 3. That would be far cheaper, though
  untested.
- **The acceptance's blind spots, as found:** one dirty start, and only the training moduli's sizes.

### 9. Files and how to rerun

- `rowtarget.py`: the full-size target mode.
- `offline.py`: the test (Q1–Q3, three arms).
- `modal_legal.py`: the launcher (one B200, 2,700 s timeout; Volume `bar3-legal`).
- `smoke.py`: the local checks of §3.
- `ladder.py`: both ladders for a record's round.
- `posthoc.py`: §5.
- `cost.py`: §7.
- `PREREG.md`: frozen. `logs/prereg_sha256.txt`: its hashes.
- `records/lt1.json`, `records/lt1.log`: the run. `logs/lt1_app.log`: the app's log, captured live.
- `logs/ladder_lt1_*.txt`, `logs/posthoc_lt1.txt`: the local checks after the run.

From `cpu_first`, with `PY=~/github/sotaku/venv/bin/python` and `PYTHONDONTWRITEBYTECODE=1` exported:

```bash
$PY -m bar3.scale.legal.smoke
$PY integration/ablation_s3/spend.py 4750
$PY -m modal run --detach bar3/scale/legal/modal_legal.py --tag lt1
$PY -m modal run bar3/scale/legal/modal_legal.py --mode fetch
$PY -m bar3.scale.legal.ladder bar3/scale/legal/records/lt1.json full
$PY -m bar3.scale.legal.posthoc bar3/scale/legal/records/lt1.json
$PY bar3/scale/legal/cost.py
```

The exported `PYTHONDONTWRITEBYTECODE=1` keeps Python from writing caches into `bar3/scale/slice/`, whose code this
study imports and does not change.
